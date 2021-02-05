#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

using tapa::detach;
using tapa::istream;
using tapa::mmap;
using tapa::ostream;
using tapa::packet;
using tapa::seq;
using tapa::stream;
using tapa::streams;
using tapa::task;

constexpr int kQueueCount = 4;

static_assert(kQueueCount % kShardCount == 0,
              "current implementation requires that queue count is a multiple "
              "of shard count");

// Verbosity definitions:
//   v=5: O(1)
//   v=8: O(#vertex)
//   v=9: O(#edge)

void HeapIndexMem(istream<HeapIndexReq>& req_q, ostream<Vid>& resp_q,
                  tapa::async_mmap<Vid> mem) {
  CLEAN_UP(clean_up, [&]() {
    // Check that heap_index is restored to the initial state.
    for (int i = 0; i < mem.size(); ++i) {
      CHECK_EQ(mem.get()[i], kNullVid) << "i = " << i;
    }
  });

  DECL_BUF(Vid, read_addr);
  DECL_BUF(Vid, read_data);
  DECL_BUF(Vid, write_addr);
  DECL_BUF(Vid, write_data);

spin:
  for (uint8_t lock = 0;; lock = lock > 0 ? lock - 1 : 0) {
#pragma HLS pipeline II = 1
    if (!req_q.empty()) {
      const auto req = req_q.peek(nullptr);
      switch (req.op) {
        case GET: {
          if (!read_addr_valid && lock == 0) {
            read_addr = req.vid;
            read_addr_valid = true;
            req_q.read(nullptr);
          }
        } break;
        case SET: {
          if (!write_addr_valid && !write_data_valid) {
            write_addr = req.vid;
            write_data = req.index;
            write_addr_valid = write_data_valid = true;
            req_q.read(nullptr);
            lock = 30;
          }
        } break;
        default: {
          LOG(FATAL) << "invalid usage";
        } break;
      }
    }
    UNUSED RESET(read_addr_valid, mem.read_addr_try_write(read_addr));
    UNUSED RESET(write_addr_valid, mem.write_addr_try_write(write_addr));
    UNUSED RESET(write_data_valid, mem.write_data_try_write(write_data));
    UNUSED UPDATE(read_data, mem.read_data_try_read(read_data),
                  resp_q.try_write(read_data));
  }
}

void HeapIndexCache(istream<HeapIndexReq>& req_in_q, ostream<Vid>& resp_out_q,
                    ostream<HeapIndexReq>& req_out_q, istream<Vid>& resp_in_q) {
  constexpr int kIndexCacheSize = 4096 * 4;
  tapa::packet<Vid, Vid> heap_index_cache[kIndexCacheSize];
#pragma HLS resource variable = heap_index_cache core = RAM_2P_URAM latency = 4
#pragma HLS data_pack variable = heap_index_cache
  int32_t read_hit = 0;
  int32_t read_miss = 0;
  int32_t write_hit = 0;
  int32_t write_miss = 0;
heap_index_cache_init:
  for (Vid i = 0; i < kIndexCacheSize; ++i) {
#pragma HLS pipeline II = 1
    heap_index_cache[i].addr = kNullVid;
  }

  CLEAN_UP(clean_up, [&]() {
    VLOG(3) << "read hit rate: " << read_hit * 100. / (read_hit + read_miss)
            << "%";
    VLOG(3) << "write hit rate: " << write_hit * 100. / (write_hit + write_miss)
            << "%";

    for (int i = 0; i < kIndexCacheSize; ++i) {
      CHECK_EQ(heap_index_cache[i].addr, kNullVid) << "i = " << i;
    }
  });

spin:
  for (;;) {
    if (!req_in_q.empty()) {
      const auto req = req_in_q.read(nullptr);
      const auto old_entry =
          heap_index_cache[req.vid / kQueueCount % kIndexCacheSize];
      auto new_entry = old_entry;
      bool write_enable = false;
      switch (req.op) {
        case GET: {
          if (old_entry.addr != req.vid) {
            req_out_q.write({GET, req.vid});
            if (old_entry.addr != kNullVid) {
              ++write_miss;
              req_out_q.write({SET, old_entry.addr, old_entry.payload});
            }
            write_enable = true, new_entry = {req.vid, resp_in_q.read()};
            ++read_miss;
          } else {
            ++read_hit;
          }
          resp_out_q.write(new_entry.payload);
        } break;
        case SET: {
          CHECK_NE(req.vid, kNullVid);
#ifdef NOALLOC
          if (old_entry.addr == req.vid) {
            ++write_hit;
            write_enable = true, new_entry = {req.vid, req.index};
          } else {
            ++write_miss;
            req_out_q.write({SET, req.vid, req.index});
          }
#else   // NOALLOC
          if (old_entry.addr != req.vid && old_entry.addr != kNullVid) {
            req_out_q.write({SET, old_entry.addr, old_entry.payload});
            ++write_miss;
          } else {
            ++write_hit;
          }
          write_enable = true, new_entry = {req.vid, req.index};
#endif  // NOALLOC
        } break;
        case CLEAR: {
          ++write_miss;
          if (old_entry.addr == req.vid) {
            write_enable = true, new_entry.addr = kNullVid;
          }
          req_out_q.write({SET, req.vid, kNullVid});
        } break;
        default: {
          LOG(FATAL) << "invalid heap index request";
        } break;
      }
      if (write_enable) {
        heap_index_cache[req.vid / kQueueCount % kIndexCacheSize] = new_entry;
      }
    }
  }
}

void HeapArrayMem(Vid qid, istream<HeapArrayReq>& req_q,
                  ostream<TaskOnChip>& resp_q, tapa::async_mmap<Task> mem) {
  DECL_BUF(Vid, read_addr);
  DECL_BUF(Task, read_data);
  DECL_BUF(Vid, write_addr);
  DECL_BUF(Task, write_data);

spin:
  for (uint8_t lock = 0;; lock = lock > 0 ? lock - 1 : 0) {
#pragma HLS pipeline II = 1
    if (!req_q.empty()) {
      const auto req = req_q.peek(nullptr);
      CHECK_GE(req.i, kHeapOnChipSize);
      const auto i = (req.i - kHeapOnChipSize) * kQueueCount + qid;
      switch (req.op) {
        case GET:
        case SYNC: {
          if (!read_addr_valid && (req.op == GET || lock == 0)) {
            read_addr = i;
            read_addr_valid = true;
            req_q.read(nullptr);
          }
        } break;
        case SET: {
          if (!write_addr_valid && !write_data_valid) {
            write_addr = i;
            write_data = req.task;
            write_addr_valid = write_data_valid = true;
            req_q.read(nullptr);
            lock = 30;
          }
        } break;
        default: {
          LOG(FATAL) << "invalid heap array request";
        } break;
      }
    }
    UNUSED RESET(read_addr_valid, mem.read_addr_try_write(read_addr));
    UNUSED RESET(write_addr_valid, mem.write_addr_try_write(write_addr));
    UNUSED RESET(write_data_valid, mem.write_data_try_write(write_data));
    UNUSED UPDATE(read_data, mem.read_data_try_read(read_data),
                  resp_q.try_write(read_data));
  }
}

// Each push request puts the task in the queue if there isn't a task for the
// same vertex in the queue, or decreases the priority of the existing task
// using the new value. Whether a new task is created is returned in the
// response.
//
// Each pop removes a task if the queue is not empty, otherwise the response
// indicates that the queue is empty.
void TaskQueue(
    // Scalar
    Vid qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    tapa::ostream<QueueOpResp>& queue_resp_q,
    // Heap array.
    ostream<HeapArrayReq>& heap_array_req_q,
    istream<TaskOnChip>& heap_array_resp_q,
    // Heap index.
    ostream<HeapIndexReq>& heap_index_req_q, istream<Vid>& heap_index_resp_q) {
#pragma HLS inline recursive
  // Heap rule: child <= parent
  Vid heap_size = 0;

  TaskOnChip heap_array_cache[kHeapOnChipSize];
#pragma HLS resource variable = heap_array_cache core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_array_cache

  CLEAN_UP(clean_up, [&]() {
    // Check that heap_index is restored to the initial state.
    CHECK_EQ(heap_size, 0);
  });

spin:
  for (;;) {
    bool do_push = false;
    bool do_pop = false;
    const auto push_req = push_req_q.read(do_push);
    pop_req_q.read(do_pop);

    QueueOp req;
    if (do_push) {
      req.task = push_req;
      req.op = do_pop ? QueueOp::PUSHPOP : QueueOp::PUSH;
    } else if (do_pop) {
      req.task.set_vid(qid);
      req.op = QueueOp::POP;
    } else {
      continue;
    }

    QueueOpResp resp{
        .queue_op = req.op,
        .task_op = TaskOp::NOOP,
        .task = req.task,
    };
    switch (req.op) {
      case QueueOp::PUSH: {
        const auto new_task = req.task;
        CHECK_EQ(new_task.vid() % kQueueCount, qid);
        heap_index_req_q.write({GET, new_task.vid()});
        ap_wait();
        const Vid task_index = heap_index_resp_q.read();
        bool heapify = true;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const auto old_task =
              task_index < kHeapOnChipSize
                  ? heap_array_cache[task_index]
                  : (heap_array_req_q.write({SYNC, task_index}), ap_wait(),
                     heap_array_resp_q.read());
          CHECK_EQ(old_task.vid(), new_task.vid());
          if (new_task <= old_task) {
            heapify = false;
            heap_index_req_q.write({CLEAR, new_task.vid()});
          }
        } else {
          heapify_index = heap_size;
          ++heap_size;
          resp.task_op = TaskOp::NEW;
        }

        if (heapify) {
          // Prefetch heap array elements.
          uint8_t pending_request_count = 0;
        heapify_up_prefetch:
          for (Vid i = heapify_index; !(i < kHeapOffChipBound);) {
#pragma HLS pipeline II = 1
            const auto parent =
                (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                kHeapOffChipWidth;
            heap_array_req_q.write({GET, parent});
            ++pending_request_count;
            i = parent;
          }

          // Increase the priority of heap_array[i] if necessary.
          Vid i = heapify_index;
          const auto task_i = new_task;

        heapify_up:
          for (; i != 0;) {
#pragma HLS pipeline
            const auto parent =
                i < kHeapOnChipSize
                    ? (i - 1) / kHeapOnChipWidth
                    : (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                          kHeapOffChipWidth;
            const auto task_parent =
                i < kHeapOffChipBound
                    ? heap_array_cache[parent]
                    : (--pending_request_count, heap_array_resp_q.read());
            if (task_i <= task_parent) break;

            if (i < kHeapOnChipSize) {
              heap_array_cache[i] = task_parent;
            } else {
              heap_array_req_q.write({SET, i, task_parent});
            }
            heap_index_req_q.write({SET, task_parent.vid(), i});
            i = parent;
          }

          if (i < kHeapOnChipSize) {
            heap_array_cache[i] = task_i;
          } else {
            heap_array_req_q.write({SET, i, task_i});
          }
          heap_index_req_q.write({SET, task_i.vid(), i});

        heapify_up_discard:
          while (pending_request_count > 0) {
#pragma HLS pipeline II = 1
            if (!heap_array_resp_q.empty()) {
              heap_array_resp_q.read(nullptr);
              --pending_request_count;
            }
          }
        }
        break;
      }
      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        CHECK_EQ(resp.task.vid() % kQueueCount, qid);
        const bool is_pushpop = req.op == QueueOp::PUSHPOP;
        const auto front = heap_array_cache[0];
        if (heap_size != 0 && !(is_pushpop && front <= req.task)) {
          heap_index_req_q.write({CLEAR, front.vid()});

          if (!is_pushpop) {
            --heap_size;
          }

          resp.task_op = TaskOp::NEW;
          resp.task = front;

          if (heap_size != 0) {
            // Find proper index `i` for `task_i`.
            const auto task_i =
                is_pushpop ? req.task
                           : (heap_size < kHeapOnChipSize
                                  ? heap_array_cache[heap_size]
                                  : (heap_array_req_q.write({SYNC, heap_size}),
                                     ap_wait(), heap_array_resp_q.read()));
            Vid i = 0;

          heapify_down_on_chip:
            for (; i < kHeapOnChipBound;) {
#pragma HLS pipeline

              Vid max = -1;
              auto task_max = task_i;
              for (int j = 1; j <= kHeapOnChipWidth; ++j) {
#pragma HLS unroll
                const Vid child = i * kHeapOnChipWidth + j;
                if (child < heap_size) {
                  const auto task_child = heap_array_cache[child];
                  if (!(task_child <= task_max)) {
                    max = child;
                    task_max = task_child;
                  }
                }
              }
              if (max == -1) break;

              heap_array_cache[i] = task_max;
              heap_index_req_q.write({SET, task_max.vid(), i});
              i = max;
            }

          heapify_down_off_chip:
            for (; !(i < kHeapOnChipBound);) {
              Vid max = -1;
              auto task_max = task_i;
              const auto child_begin = i * kHeapOffChipWidth -
                                       kHeapDiff * (kHeapOffChipWidth - 1) + 1;
              const auto child_end =
                  std::min(heap_size, child_begin + kHeapOffChipWidth);
            heapify_down_cmp:
              for (Vid child_req = child_begin, child_resp = child_begin;
                   child_resp < child_end;) {
#pragma HLS pipeline II = 1
                if (child_req < child_end &&
                    heap_array_req_q.try_write({GET, child_req})) {
                  ++child_req;
                }

                if (!heap_array_resp_q.empty()) {
                  const auto task_child = heap_array_resp_q.read();
                  if (!(task_child <= task_max)) {
                    max = child_resp;
                    task_max = task_child;
                  }
                  ++child_resp;
                }
              }
              if (max == -1) break;

              if (i < kHeapOnChipSize) {
                heap_array_cache[i] = task_max;
              } else {
                heap_array_req_q.write({SET, i, task_max});
              }
              heap_index_req_q.write({SET, task_max.vid(), i});
              i = max;
            }

            if (i < kHeapOnChipSize) {
              heap_array_cache[i] = task_i;
            } else {
              heap_array_req_q.write({SET, i, task_i});
            }
            heap_index_req_q.write({SET, task_i.vid(), i});
          }
        } else if (is_pushpop) {
          resp.task_op = TaskOp::NEW;
        }
        break;
      }
    }
    queue_resp_q.write(resp);
  }
}

using SpqElem = TaskOnChip;

const SpqElem kSpqElemMin(Task{.vertex = {.distance = kInfDistance}});
const SpqElem kSpqElemMax(Task{.vertex = {.distance = 0}});

constexpr int kSpqSize = 7;

SpqElem SpqUpdate(SpqElem& a, SpqElem& b, SpqElem& c) {
  // Make c <= b <= a by updating a and b, invalidating the old c, and returning
  // the new c.

  const bool x = b <= a;
  const bool y = c <= a;
  const bool z = c <= b;
  // a <= b <= c : !x && !y && !z
  // a <= c <= b : !x && !y &&  z
  // b <= a <= c :  x && !y && !z
  // c <= a <= b : !x &&  y &&  z
  // b <= c <= a :  x &&  y && !z
  // c <= b <= a :  x &&  y &&  z
  const auto new_a = x && y ? a : z ? b : c;
  const auto new_b = x == y ? y == z ? b : c : a;
  const auto new_c = y && z ? c : x ? b : a;

  a = new_a;
  b = new_b;
  c = kSpqElemMin;
  return new_c;
}

void SystolicQueue(
    // Queue ID.
    int id,
    // Request-response interfaces.
    istream<QueueOp>& req_q, ostream<QueueOpResp>& resp_q,
    // Connection to overflow queue.
    ostream<SpqElem>& overflow_q) {
  DECL_ARRAY(SpqElem, a, kSpqSize + 1, kSpqElemMin);
  DECL_ARRAY(SpqElem, b, kSpqSize + 1, kSpqElemMin);
  const auto& front = a[0];
  auto& overflow = b[kSpqSize];

spin:
  for (;;) {
    // Do not pipeline to avoid Vivado HLS bug.
    if (!req_q.empty()) {
      const auto spq_op = req_q.read();
      switch (spq_op.op) {
        case QueueOp::PUSH:
          a[0] = spq_op.task;
          b[0] = kSpqElemMax;
          break;
        case QueueOp::PUSHPOP:
          a[0] = spq_op.task;
          b[0] = kSpqElemMin;
          break;
        case QueueOp::POP:
          a[0] = b[0] = kSpqElemMin;
          break;
      }

      for (int i = 1; i <= kSpqSize; i += 2) {
#pragma HLS unroll
        b[i] = SpqUpdate(a[i - 1], a[i], b[i - 1]);
      }

      for (int i = 2; i <= kSpqSize; i += 2) {
#pragma HLS unroll
        b[i] = SpqUpdate(a[i - 1], a[i], b[i - 1]);
      }

      QueueOpResp resp{
          .queue_op = spq_op.op,
          .task_op = front.vertex().is_inf() ? TaskOp::NOOP : TaskOp::NEW,
          .task = spq_op.is_push() ? spq_op.task : front,
      };
      if (resp.is_pop_noop()) {
        resp.task.set_vid(id);
      }
      CHECK_GT(resp.task.vertex().distance, 0);
      CHECK_EQ(resp.task.vid() % kQueueCount, id);
      resp_q.write(resp);

      if (!overflow.vertex().is_inf()) {
        overflow_q.write(overflow);
      }
    }
  }
}

void CascadedQueue(
    // Queue ID.
    int id,
    // Request-response interfaces.
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    ostream<QueueOpResp>& resp_q,
    // Connection to systolic queue.
    ostream<QueueOp>& spq_req_q, istream<QueueOpResp>& spq_resp_q,
    // Connection to overflow queue.
    ostream<bool>& ofq_pop_req_q, istream<QueueOpResp>& ofq_resp_q) {
  DECL_BUF(QueueOpResp, spq_resp);
  DECL_BUF(QueueOpResp, ofq_resp);
  bool both_pending = false;
  bool ofq_pending = false;
  bool pushpop_pending = false;
  bool spq_was_empty = false;

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(spq_resp, spq_resp_q.try_read(spq_resp), spq_resp.is_push());
    UPDATE(ofq_resp, ofq_resp_q.try_read(ofq_resp), ofq_resp.is_push());

    // Buffers for output streams.
    DECL_BUF(QueueOpResp, resp);
    DECL_BUF(QueueOp, spq_req);
    DECL_BUF(bool, ofq_pop_req);

    // Send POP response, step 1.
    if (both_pending && spq_resp_valid) {
      spq_was_empty = spq_resp.is_pop_noop();
      if (!spq_was_empty) {
        resp = spq_resp, resp_valid = true;  // resp_q.write(spq_resp);
      }

      both_pending = spq_resp_valid = false;
      ofq_pending = true;
    } else if (pushpop_pending && spq_resp_valid) {
      // Or, send PUSHPOP response.
      CHECK_EQ(spq_resp.queue_op, QueueOp::PUSHPOP);
      CHECK_EQ(spq_resp.task_op, TaskOp::NEW);

      resp = spq_resp, resp_valid = true;  // resp_q.write(spq_resp);

      pushpop_pending = spq_resp_valid = false;
    }

    // Send POP response, step 2.
    if (ofq_pending &&     // Systolic queue response has been processed.
        ofq_resp_valid &&  // Overflow queue response is valid.
        !(spq_was_empty && resp_valid)  // Response can be sent.
    ) {
      if (spq_was_empty) {
        resp = ofq_resp, resp_valid = true;  // resp_q.write(ofq_resp);
      } else if (ofq_resp.is_pop_new()) {
        spq_req.op = QueueOp::PUSH;    // spq_req_q.write(...);
        spq_req.task = ofq_resp.task;  // spq_req_q.write(...);
        spq_req_valid = true;          // spq_req_q.write(...);
      }

      ofq_pending = ofq_resp_valid = false;
    }

    // Handle new requests.
    if (!resp_valid &&                    // Response can be sent.
        !spq_req_valid &&                 // Systolic queue request can be sent.
        !ofq_pop_req_valid &&             // Overflow queue request can be sent.
        !both_pending && !ofq_pending &&  // Not waiting for POP response.
        !pushpop_pending &&               // Not waiting for PUSHPOP response.
        (!push_req_q.empty() || !pop_req_q.empty())  // Request is valid.
    ) {
      spq_req_valid = true;  // spq_req_q.write(req);

      bool do_push = false;
      bool do_pop = false;
      const auto push_req = push_req_q.read(do_push);
      pop_req_q.read(do_pop);
      if (do_push) {
        spq_req.task = push_req;
        if (do_pop) {
          spq_req.op = QueueOp::PUSHPOP;
          pushpop_pending = true;
        } else {
          spq_req.op = QueueOp::PUSH;
          resp = {
              .queue_op = QueueOp::PUSH,  // resp_q.write(req);
              .task_op = {},              // resp_q.write(req);
              .task = push_req,           // resp_q.write(req);
          };                              // resp_q.write(req);
          resp_valid = true;              // resp_q.write(req);
        }
      } else if (do_pop) {
        spq_req.task.set_vid(id);
        spq_req.op = QueueOp::POP;
        ofq_pop_req_valid = true;  // ofq_req_q.write(req);
        both_pending = true;
      }
    }

    if (resp_valid) {
      resp_q.write(resp);
      CHECK_EQ(resp.task.vid() % kQueueCount, id);
    }
    if (spq_req_valid) {
      spq_req_q.write(spq_req);
      CHECK_EQ(spq_req.task.vid() % kQueueCount, id);
    }
    if (ofq_pop_req_valid) {
      ofq_pop_req_q.write(ofq_pop_req);
    }
  }
}

// A VidMux merges two input streams into one.
void VidMux(istream<TaskOnChip>& in0, istream<TaskOnChip>& in1,
            ostream<TaskOnChip>& out) {
spin:
  for (bool flag = false;; flag = !flag) {
#pragma HLS pipeline II = 1
    TaskOnChip data;
    if (flag ? in0.try_read(data) : in1.try_read(data)) out.write(data);
  }
}

// A VidDemux routes input streams based on the specified bit in Vid.
void VidDemux(int b, istream<TaskOnChip>& in, ostream<TaskOnChip>& out0,
              ostream<TaskOnChip>& out1) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    TaskOnChip data;
    if (in.try_read(data)) {
      const auto addr = data.vid();
      const bool select = ap_uint<sizeof(addr) * CHAR_BIT>(addr).test(b);
      select ? out1.write(data) : out0.write(data);
    }
  }
}

void EdgeReqArbiter(tapa::istreams<EdgeReq, kPeCount>& req_q,
                    tapa::ostreams<SourceVertex, kShardCount>& src_q,
                    tapa::ostreams<Vid, kShardCount>& addr_q) {
  DECL_ARRAY(EdgeReq, req, kShardCount, EdgeReq());
  DECL_ARRAY(bool, req_valid, kShardCount, false);
  DECL_ARRAY(SourceVertex, src, kShardCount, SourceVertex());
  DECL_ARRAY(bool, src_valid, kShardCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kShardCount) + 2> pe_sid = 0;; ++pe_sid) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      const auto pe = pe_sid / 8 * kShardCount + sid;
      if (!src_valid[sid] &&
          SET(req_valid[sid], req_q[pe].try_read(req[sid]))) {
        src[sid] = req[sid].payload;
        src_valid[sid] = true;
      }
    });

    RANGE(sid, kShardCount, {
      UNUSED RESET(src_valid[sid], src_q[sid].try_write(src[sid]));
      UNUSED RESET(req_valid[sid], addr_q[sid].try_write(req[sid].addr));
    });
  }
}

void UpdateReqArbiter(tapa::istreams<TaskOnChip, kShardCount>& in_q,
                      tapa::ostreams<TaskOnChip, kIntervalCount>& out_q) {
  static_assert(kIntervalCount % kShardCount == 0,
                "current implementation requires that queue count is a "
                "multiple of interval count");
  DECL_ARRAY(TaskOnChip, update, kShardCount, TaskOnChip());
  DECL_ARRAY(bool, update_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      UNUSED SET(update_valid[sid], in_q[sid].try_read(update[sid]));
      const auto iid =
          update[sid].vid() % kIntervalCount / kShardCount * kShardCount + sid;
      UNUSED RESET(update_valid[sid], out_q[iid].try_write(update[sid]));
    });
  }
}

void EdgeMem(tapa::istream<Vid>& read_addr_q, tapa::ostream<Edge>& read_data_q,
             tapa::async_mmap<Edge> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void VertexMem(tapa::istream<Vid>& read_addr_q,
               tapa::ostream<Vertex>& read_data_q,
               tapa::async_mmap<Vertex> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void ProcElemS0(istream<TaskOnChip>& task_in_q, ostream<Vid>& task_resp_q,
                ostream<EdgeReq>& edge_req_q) {
  EdgeReq req;

spin:
  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      req = {task.vertex().offset, {task.vid(), task.vertex().distance}};
      i = task.vertex().degree;
    }

    if (i > 0) {
      edge_req_q.write(req);
      if (i == 1) {
        task_resp_q.write(req.payload.parent);
      }

      ++req.addr;
      --i;
    }
  }
}

void ProcElemS1(istream<SourceVertex>& src_in_q,
                istream<Edge>& edges_read_data_q,
                ostream<TaskOnChip>& update_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!src_in_q.empty() && !edges_read_data_q.empty()) {
      const auto src = src_in_q.read(nullptr);
      const auto edge = edges_read_data_q.read(nullptr);
      update_out_q.write(Task{
          .vid = edge.dst,
          .vertex = {src.parent, src.distance + edge.weight},
      });
    }
  }
}

void VertexReaderS0(
    // Input.
    istream<TaskOnChip>& update_in_q,
    // Outputs.
    ostream<TaskOnChip>& update_out_q, ostream<Vid>& vertex_read_addr_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!update_in_q.empty()) {
      const auto req = update_in_q.read(nullptr);
      update_out_q.write(req);
      vertex_read_addr_q.write(req.vid() / kIntervalCount);
    }
  }
}

void VertexReaderS1(
    // Inputs.
    istream<TaskOnChip>& update_in_q, istream<Vertex>& vertex_read_data_q,
    // Outputs.
    ostream<TaskOnChip>& new_q, ostream<bool>& noop_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!update_in_q.empty() && !vertex_read_data_q.empty()) {
      auto req = update_in_q.read(nullptr);
      const auto vertex = vertex_read_data_q.read(nullptr);
      if (vertex <= req.vertex()) {
        noop_q.write(false);
      } else {
        new_q.write(req);
      }
    }
  }
}

void VertexUpdater(istream<TaskOnChip>& task_in_q,
                   ostream<TaskOnChip>& task_out_q, ostream<bool>& noop_q,
                   mmap<Vertex> vertices) {
spin:
  for (;;) {
    if (!task_in_q.empty()) {
      auto task = task_in_q.read(nullptr);
      const auto addr = task.vid() / kIntervalCount;
      const auto vertex = vertices[addr];
      if (vertex <= task.vertex()) {
        noop_q.write(false);
      } else {
        task.set_offset(vertex.offset);
        task.set_degree(vertex.degree);
        vertices[addr] = task.vertex();
        task_out_q.write(task);
      }
    }
  }
}

using uint_noop_t = ap_uint<bit_length(kIntervalCount * 2)>;

void NoopMerger(tapa::istreams<bool, kIntervalCount>& pkt_in_q1,
                tapa::istreams<bool, kIntervalCount>& pkt_in_q2,
                ostream<uint_noop_t>& pkt_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    uint_noop_t count = 0;
    bool buf;
    RANGE(iid, kIntervalCount, pkt_in_q1[iid].try_read(buf) && ++count);
    RANGE(iid, kIntervalCount, pkt_in_q2[iid].try_read(buf) && ++count);
    if (count) {
      pkt_out_q.write(count);
    }
  }
}

void PushReqArbiter(tapa::istreams<TaskOnChip, kIntervalCount>& in_q,
                    tapa::ostreams<TaskOnChip, kQueueCount>& out_q) {
  static_assert(kQueueCount % kIntervalCount == 0,
                "current implementation requires that queue count is a "
                "multiple of interval count");
  DECL_ARRAY(TaskOnChip, task, kIntervalCount, TaskOnChip());
  DECL_ARRAY(bool, task_valid, kIntervalCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(iid, kIntervalCount, {
      UNUSED SET(task_valid[iid], in_q[iid].try_read(task[iid]));
      const auto qid =
          task[iid].vid() % kQueueCount / kIntervalCount * kIntervalCount + iid;
      UNUSED RESET(task_valid[iid], out_q[qid].try_write(task[iid]));
    });
  }
}

using uint_qid_t = ap_uint<bit_length(kQueueCount - 1)>;

void PopReqArbiter(istream<uint_qid_t>& req_in_q,
                   tapa::ostreams<bool, kQueueCount>& req_out_q) {
  DECL_BUF(uint_qid_t, qid);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(qid, req_in_q.try_read(qid), req_out_q[qid].try_write(false));
  }
}

void QueueRespArbiter(tapa::istreams<QueueOpResp, kQueueCount>& resp_in_q,
                      tapa::ostream<QueueOpResp>& resp_out_q) {
  DECL_BUF(QueueOpResp, resp);
spin:
  for (uint8_t q = 0;; ++q) {
#pragma HLS pipeline II = 1
    UPDATE(resp, resp_in_q[q % kQueueCount].try_read(resp),
           resp_out_q.try_write(resp));
  }
}

using uint_pe_t = ap_uint<bit_length(kPeCount - 1)>;

using TaskReq = packet<uint_pe_t, TaskOnChip>;

void TaskReqArbiter(istream<TaskReq>& req_in_q,
                    tapa::ostreams<TaskOnChip, kPeCount>& req_out_q) {
  DECL_BUF(TaskReq, req);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(req, req_in_q.try_read(req),
           req_out_q[req.addr].try_write(req.payload));
  }
}

using TaskResp = packet<uint_pe_t, Vid>;

void TaskRespArbiter(tapa::istreams<Vid, kPeCount>& resp_in_q,
                     ostream<TaskResp>& resp_out_q) {
  DECL_BUF(Vid, vid);
spin:
  for (uint_pe_t pe = 0;; ++pe) {
#pragma HLS pipeline II = 1
    static_assert(is_power_of(kPeCount, 2),
                  "pe needs rounding if kPeCount is not a power of 2");
    UPDATE(vid, resp_in_q[pe].try_read(vid), resp_out_q.try_write({pe, vid}));
  }
}

void Dispatcher(
    // Scalar.
    const Task root,
    // Metadata.
    tapa::mmap<int64_t> metadata,
    // Task and queue requests.
    ostream<TaskReq>& task_req_q, istream<TaskResp>& task_resp_q,
    istream<uint_noop_t>& update_noop_q, ostream<uint_qid_t>& queue_req_q,
    tapa::istream<QueueOpResp>& queue_resp_q) {
  // Process finished tasks.
  bool queue_buf_valid = false;
  QueueOpResp queue_buf;

  // Number of tasks whose parent task is sent to the PEs but not yet
  // acknowledged.
  int32_t active_task_count = root.vertex.degree;

  // Number of tasks generated by the PEs but not yet sent to the PEs.
  int32_t pending_task_count = 0;

  // Number of POP requests sent but not acknowledged.
  DECL_ARRAY(ap_uint<bit_length(kPeCount / kQueueCount)>, task_count_per_queue,
             kQueueCount, 0);
  DECL_ARRAY(ap_uint<bit_length(kPeCount / kShardCount - 1)>, pe_base_per_shard,
             kShardCount, 0);

  ap_uint<kQueueCount> queue_empty = -1;
  ap_uint<kQueueCount> queue_active = 0;
  ap_uint<kPeCount> pe_active = 0;

  task_req_q.write({root.vid % kShardCount, root});
  ++task_count_per_queue[root.vid % kQueueCount];
  pe_active.set(root.vid % kShardCount);

  // Statistics.
  int32_t visited_vertex_count = 1;
  int32_t visited_edge_count = 0;
  int32_t queue_count = 0;
  int64_t cycle_count = 0;
  int64_t queue_full_cycle_count = 0;
  int64_t pe_fullcycle_count = 0;
  DECL_ARRAY(int64_t, pe_active_count, kPeCount, 0);

  // Format log messages.
#define STATS(tag, content)                                              \
  do {                                                                   \
    VLOG_F(9, tag) << content " | " << std::setfill(' ') << std::setw(1) \
                   << active_task_count << " active + " << std::setw(2)  \
                   << pending_task_count << " pending tasks";            \
    CHECK_GE(active_task_count, 0);                                      \
    CHECK_GE(pending_task_count, 0);                                     \
  } while (0)

  auto shard_is_done = [&](int sid) {
    bool result = true;
    RANGE(i, kQueueCount / kShardCount,
          result &= queue_empty.bit(i * kShardCount + sid));
    return result;
  };

spin:
  for (; active_task_count || queue_empty.nand_reduce() ||
         any_of(task_count_per_queue);
       ++cycle_count) {
#pragma HLS pipeline II = 1
    RANGE(pe, kPeCount, pe_active.bit(pe) && ++pe_active_count[pe]);
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      const auto qid = queue_buf.task.vid() % kQueueCount;
      queue_empty.clear(qid);
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          --active_task_count;
          ++pending_task_count;

          // TaskOp statistics.
          ++queue_count;
          break;
        case QueueOp::PUSHPOP:
          CHECK_EQ(queue_buf.task_op, TaskOp::NEW);
          queue_active.clear(qid);
          --active_task_count;
          ++pending_task_count;

          // TaskOp statistics.
          ++queue_count;
          break;
        case QueueOp::POP:
          queue_active.clear(qid);
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            STATS(recv, "QUEUE: NEW ");
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            queue_empty.set(qid);
            STATS(recv, "QUEUE: NOOP");
          }
          break;
      }
    }

    {
      const auto qid = cycle_count % kQueueCount;
      if (task_count_per_queue[qid] < kPeCount / kQueueCount &&
          !(queue_buf_valid && queue_buf.task.vid() % kQueueCount == qid) &&
          !queue_active.bit(qid) && !queue_empty.bit(qid)) {
        // Dequeue tasks from the queue.
        if (queue_req_q.try_write(qid)) {
          queue_active.set(qid);
          STATS(send, "QUEUE: POP ");
        } else {
          ++queue_full_cycle_count;
        }
      }
    }

    // Assign tasks to PEs.
    {
      const auto sid = queue_buf.task.vid() % kShardCount;
      const auto pe = pe_base_per_shard[sid] * kShardCount + sid;
      if (queue_buf_valid) {
        if (!pe_active.bit(pe) && task_req_q.try_write({pe, queue_buf.task})) {
          active_task_count += queue_buf.task.vertex().degree;
          --pending_task_count;
          queue_buf_valid = false;
          ++task_count_per_queue[queue_buf.task.vid() % kQueueCount];
          pe_active.set(pe);

          // Statistics.
          ++visited_vertex_count;
        } else {
          ++pe_fullcycle_count;
        }
        ++pe_base_per_shard[sid];
      }
    }

    {
      uint_noop_t count;
      if (update_noop_q.try_read(count)) {
        active_task_count -= count;
        visited_edge_count += count;
      }
    }

    if (!task_resp_q.empty()) {
      const auto resp = task_resp_q.read(nullptr);
      --task_count_per_queue[resp.payload % kQueueCount];
      pe_active.clear(resp.addr);
    }
  }

#ifndef __SYNTHESIS__
  RANGE(qid, kQueueCount, CHECK_EQ(task_count_per_queue[qid], 0));
  CHECK_EQ(queue_empty, decltype(queue_empty)(-1));
  CHECK_EQ(queue_active, 0);
  CHECK_EQ(pe_active, 0);
  CHECK_EQ(active_task_count, 0);
#endif  // __SYNTHESIS__

  metadata[0] = visited_edge_count;
  metadata[2] = queue_count;
  metadata[4] = visited_vertex_count;
  metadata[6] = cycle_count;
  metadata[7] = queue_full_cycle_count;
  metadata[8] = pe_fullcycle_count;

meta:
  for (int pe = 0; pe < kPeCount; ++pe) {
#pragma HLS pipeline II = 1
    metadata[9 + pe] = pe_active_count[pe];
  }
}

void SSSP(Vid vertex_count, Task root, tapa::mmap<int64_t> metadata,
          tapa::async_mmaps<Edge, kShardCount> edges,
          tapa::async_mmaps<Vertex, kIntervalCount> vertices,
          // For queues.
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  streams<TaskOnChip, kIntervalCount, 8> push_req_q;
  streams<TaskOnChip, kQueueCount, 512> push_req_qi;
  stream<uint_qid_t, 2> pop_req_q("pop_req");
  streams<bool, kQueueCount, 2> pop_req_qi("pop_req_i");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  streams<QueueOp, kQueueCount, 2> spq_req_q("systolic_req");
  streams<QueueOpResp, kQueueCount, 2> spq_resp_q("systolic_resp");
  streams<TaskOnChip, kQueueCount, 512> spq_overflow_q("systolic_overflow");
  streams<bool, kQueueCount, 2> ofq_pop_req_q("overflow_pop_req");
  streams<QueueOpResp, kQueueCount, 2> ofq_resp_q("overflow_resp");

  streams<HeapIndexReq, kQueueCount, 64> heap_index_req_q;
  streams<Vid, kQueueCount, 2> heap_index_resp_q;
  streams<HeapIndexReq, kQueueCount, 2> heap_index_req_qi;
  streams<Vid, kQueueCount, 2> heap_index_resp_qi;

  streams<HeapArrayReq, kQueueCount, 64> heap_array_req_q;
  streams<TaskOnChip, kQueueCount, 2> heap_array_resp_q;

  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Vid, kPeCount, 2> task_resp_qi("task_resp_i");

  stream<TaskReq, 2> task_req_q("task_req");
  stream<TaskResp, 64> task_resp_q("task_resp");

  // For edges.
  streams<Vid, kShardCount, 2> edge_read_addr_q("edge_read_addr");
  streams<Edge, kShardCount, 2> edge_read_data_q("edge_read_data");
  streams<EdgeReq, kPeCount, 512> edge_req_q("edge_req");
  streams<SourceVertex, kShardCount, 512> src_q("source_vertices");

  // For vertices.
  //   Connect PEs to the update request network.
  streams<TaskOnChip, kShardCount, 512> update_req_q;
  //   Compose the update request network.
  streams<TaskOnChip, kIntervalCount, 8> update_req_qi1;
  streams<TaskOnChip, kIntervalCount, 8> update_req_0_qi0;
  streams<TaskOnChip, kIntervalCount, 8> update_req_1_qi0;
  streams<TaskOnChip, kIntervalCount, 8> update_req_qi0;
  //   Connect the vertex readers and updaters.
  streams<TaskOnChip, kIntervalCount, 64> update_qi0;
  streams<TaskOnChip, kIntervalCount, 512> update_new_qi;
  streams<bool, kIntervalCount, 2> update_noop_qi1;
  streams<bool, kIntervalCount, 2> update_noop_qi2;
  streams<Vid, kIntervalCount, 2> vertex_read_addr_q;
  streams<Vertex, kIntervalCount, 2> vertex_read_data_q;

  stream<uint_noop_t, 2> update_noop_q;

  tapa::task()
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 update_noop_q, pop_req_q, queue_resp_q)
      .invoke<detach>(TaskReqArbiter, task_req_q, task_req_qi)
      .invoke<detach>(TaskRespArbiter, task_resp_qi, task_resp_q)
#if 0
      .invoke<detach, kQueueCount>(CascadedQueue, seq(), push_req_qi,
                                   pop_req_qi, queue_resp_qi, spq_req_q,
                                   spq_resp_q, ofq_pop_req_q, ofq_resp_q)
      .invoke<detach, kQueueCount>(SystolicQueue, seq(), spq_req_q, spq_resp_q,
                                   spq_overflow_q)
      .invoke<detach, kQueueCount>(TaskQueue, seq(), spq_overflow_q,
                                   ofq_pop_req_q, ofq_resp_q, heap_array_req_q,
                                   heap_array_resp_q, heap_index_req_q,
                                   heap_index_resp_q)
#else
      .invoke<detach, kQueueCount>(TaskQueue, seq(), push_req_qi, pop_req_qi,
                                   queue_resp_qi, heap_array_req_q,
                                   heap_array_resp_q, heap_index_req_q,
                                   heap_index_resp_q)
#endif
      .invoke<detach, kQueueCount>(HeapIndexMem, heap_index_req_qi,
                                   heap_index_resp_qi, heap_index)
      .invoke<detach, kQueueCount>(HeapIndexCache, heap_index_req_q,
                                   heap_index_resp_q, heap_index_req_qi,
                                   heap_index_resp_qi)
      .invoke<detach, kQueueCount>(HeapArrayMem, seq(), heap_array_req_q,
                                   heap_array_resp_q, heap_array)
      .invoke<detach>(PushReqArbiter, push_req_q, push_req_qi)
      .invoke<detach>(PopReqArbiter, pop_req_q, pop_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)

      // For edges.
      .invoke<detach, kShardCount>(EdgeMem, edge_read_addr_q, edge_read_data_q,
                                   edges)
      .invoke<detach>(EdgeReqArbiter, edge_req_q, src_q, edge_read_addr_q)

      // For vertices.
      .invoke<detach>(UpdateReqArbiter, update_req_q, update_req_qi1)

      // clang-format off
      .invoke<detach, kIntervalCount>(VidDemux, 0, update_req_qi1, update_req_0_qi0, update_req_1_qi0)
      .invoke<detach>(VidMux, update_req_0_qi0[0], update_req_0_qi0[1], update_req_qi0[0])
      .invoke<detach>(VidMux, update_req_1_qi0[0], update_req_1_qi0[1], update_req_qi0[1])
      // clang-format on

      .invoke<detach, kIntervalCount>(VertexMem, vertex_read_addr_q,
                                      vertex_read_data_q, vertices)
      .invoke<detach, kIntervalCount>(VertexReaderS0, update_req_qi0,
                                      update_qi0, vertex_read_addr_q)
      .invoke<detach, kIntervalCount>(VertexReaderS1, update_qi0,
                                      vertex_read_data_q, update_new_qi,
                                      update_noop_qi1)
      .invoke<detach, kIntervalCount>(VertexUpdater, update_new_qi, push_req_q,
                                      update_noop_qi2, vertices)
      .invoke<detach>(NoopMerger, update_noop_qi1, update_noop_qi2,
                      update_noop_q)

      // PEs.
      .invoke<detach, kPeCount>(ProcElemS0, task_req_qi, task_resp_qi,
                                edge_req_q)
      .invoke<detach, kShardCount>(ProcElemS1, src_q, edge_read_data_q,
                                   update_req_q);
}

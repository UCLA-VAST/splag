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

// The queue is initialized with the root vertex. Each request is either a push
// or a pop.
//
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
    tapa::ostream<QueueOpResp>& queue_resp_q, tapa::mmap<Task> heap_array,
    tapa::mmap<Vid> heap_index) {
#pragma HLS inline recursive
  // Heap rule: child <= parent
  Vid heap_size = 0;

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
  auto get_heap_index = [&](Vid vid) -> Vid {
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr != vid) {
      if (entry.addr != kNullVid) {
        ++write_miss;
        heap_index[entry.addr] = entry.payload;
      }
      entry.addr = vid;
      entry.payload = heap_index[vid];
      ++read_miss;
    } else {
      ++read_hit;
    }
    return entry.payload;
  };
  auto set_heap_index_noalloc = [&](Vid vid, Vid index) {
    CHECK_NE(vid, kNullVid);
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr == vid) {
      ++write_hit;
      entry.addr = vid;
      entry.payload = index;
    } else {
      ++write_miss;
      heap_index[vid] = index;
    }
  };
  auto set_heap_index = [&](Vid vid, Vid index) {
    CHECK_NE(vid, kNullVid);
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr != vid && entry.addr != kNullVid) {
      heap_index[entry.addr] = entry.payload;
      ++write_miss;
    } else {
      ++write_hit;
    }
    entry.addr = vid;
    entry.payload = index;
  };
  auto clear_heap_index = [&](Vid vid) {
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    ++write_miss;
    if (entry.addr == vid) {
      entry.addr = kNullVid;
    }
    heap_index[vid] = kNullVid;
  };

  // #elements shared by the on-chip heap and the off-chip heap.
  constexpr int kHeapSharedSize = 4096;
  static_assert(is_power_of(kHeapSharedSize, kHeapOnChipWidth),
                "invalid heap configuration");
  static_assert(is_power_of(kHeapSharedSize, kHeapOffChipWidth),
                "invalid heap configuration");
  // #elements in the on-chip heap.
  constexpr int kHeapOnChipSize =
      (kHeapSharedSize * kHeapOnChipWidth - 1) / (kHeapOnChipWidth - 1);
  // #elements whose children are on chip.
  constexpr int kHeapOnChipBound = (kHeapOnChipSize - 1) / kHeapOnChipWidth;
  // #elements skipped in the off-chip heap (because they are on-chip).
  constexpr int kHeapOffChipSkipped =
      (kHeapOffChipWidth * kHeapOnChipSize * (kHeapOnChipWidth - 1) +
       kHeapOffChipWidth - kHeapOnChipWidth) /
      (kHeapOffChipWidth - 1) / kHeapOnChipWidth;
  // #elements difference between off-chip indices and mixed indices.
  constexpr int kHeapDiff = kHeapOnChipSize - kHeapOffChipSkipped;
  // #elements in the mixed heap whose parent is on chip.
  constexpr int kHeapOffChipBound =
      kHeapOnChipSize +
      kHeapOffChipWidth *
          (kHeapOnChipSize - (kHeapOnChipSize - 1) / kHeapOnChipWidth);

  /*
   *  parent of i:
   *    if i < kHeapOnChipSize:                           {on-chip}
   *      (i-1)/kHeapOnChipWidth                            {on-chip}
   *    elif i < kHeapOffChipBound                        {off-chip}
   *      (i-kHeapDiff-1)/kHeapOffChipWidth+kHeapDiff       {on-chip}
   *        = (i+(kHeapDiff*(kHeapOffChipWidth-1)-1))/kHeapOffChipWidth
   *    else:                                             {off-chip}
   *      (i-kHeapDiff-1)/kHeapOffChipWidth+kHeapDiff       {off-chip}
   *        = (i+(kHeapDiff*(kHeapOffChipWidth-1)-1))/kHeapOffChipWidth
   *
   *  first child of i:
   *    if i < kHeapOnChipBound:                                    {on-chip}
   *      i*kHeapOnChipWidth+1                                        {on-chip}
   *    elif i < kHeapOnChipSize:                                   {on-chip}
   *      (i-kHeapDiff)*kHeapOffChipWidth+kHeapDiff+1                 {off-chip}
   *        = i*kHeapOffChipWidth-kHeapDiff*(kHeapOffChipWidth-1)+1
   *    else:                                                       {off-chip}
   *      (i-kHeapDiff)*kHeapOffChipWidth+kHeapDiff+1                 {off-chip}
   *        = i*kHeapOffChipWidth-kHeapDiff*(kHeapOffChipWidth-1)+1
   */

  TaskOnChip heap_array_cache[kHeapOnChipSize];
#pragma HLS resource variable = heap_array_cache core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_array_cache

  auto get_heap_array_index = [&](Vid i) {
    return (i - kHeapOnChipSize) * kQueueCount + qid;
  };
  auto get_heap_elem_on_chip = [&](Vid i) {
    CHECK_LT(i, kHeapOnChipSize);
    return Task(heap_array_cache[i]);
  };
  auto get_heap_elem_off_chip = [&](Vid i) {
    CHECK_GE(i, kHeapOnChipSize);
    return heap_array[get_heap_array_index(i)];
  };
  auto get_heap_elem = [&](Vid i) {
    return i < kHeapOnChipSize ? get_heap_elem_on_chip(i)
                               : get_heap_elem_off_chip(i);
  };
  auto set_heap_elem_on_chip = [&](Vid i, Task task) {
    CHECK_LT(i, kHeapOnChipSize);
    heap_array_cache[i] = TaskOnChip(task);
  };
  auto set_heap_elem_off_chip = [&](Vid i, Task task) {
    CHECK_GE(i, kHeapOnChipSize);
    heap_array[get_heap_array_index(i)] = task;
  };
  auto set_heap_elem = [&](Vid i, Task task) {
    i < kHeapOnChipSize ? set_heap_elem_on_chip(i, task)
                        : set_heap_elem_off_chip(i, task);
  };

  // Performance counters.
  int32_t heapify_up_count = 0;
  int32_t heapify_up_on_chip = 0;
  int32_t heapify_up_off_chip = 0;
  int32_t heapify_down_count = 0;
  int32_t heapify_down_on_chip = 0;
  int32_t heapify_down_off_chip = 0;

  CLEAN_UP(clean_up, [&]() {
    VLOG(3) << "average heapify up trip count (on-chip): "
            << 1. * heapify_up_on_chip / heapify_up_count;
    VLOG(3) << "average heapify up trip count (off-chip): "
            << 1. * heapify_up_off_chip / heapify_up_count;
    VLOG(3) << "average heapify down trip count (on-chip): "
            << 1. * heapify_down_on_chip / heapify_down_count;
    VLOG(3) << "average heapify down trip count (off-chip): "
            << 1. * heapify_down_off_chip / heapify_down_count;
    VLOG(3) << "read hit rate: " << read_hit * 100. / (read_hit + read_miss)
            << "%";
    VLOG(3) << "write hit rate: " << write_hit * 100. / (write_hit + write_miss)
            << "%";

    // Check that heap_index is restored to the initial state.
    CHECK_EQ(heap_size, 0);
    for (int i = 0; i < heap_index.size(); ++i) {
      CHECK_EQ(heap_index.get()[i], kNullVid) << "i = " << i;
    }
    for (int i = 0; i < kIndexCacheSize; ++i) {
      CHECK_EQ(heap_index_cache[i].addr, kNullVid) << "i = " << i;
    }
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
        const auto new_task = Task(req.task);
        CHECK_EQ(new_task.vid % kQueueCount, qid);
        const Vid task_index = get_heap_index(new_task.vid);
        bool heapify = true;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const Task old_task = get_heap_elem(task_index);
          CHECK_EQ(old_task.vid, new_task.vid);
          if (new_task <= old_task) {
            heapify = false;
            clear_heap_index(new_task.vid);
          }
        } else {
          heapify_index = heap_size;
          ++heap_size;
          resp.task_op = TaskOp::NEW;
        }

        if (heapify) {
          // Increase the priority of heap_array[i] if necessary.
          Vid i = heapify_index;
          const Task task_i = new_task;

          ++heapify_up_count;

        heapify_up_off_chip:
          for (; !(i < kHeapOffChipBound);) {
#pragma HLS pipeline
            ++heapify_up_off_chip;

            const auto parent =
                (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                kHeapOffChipWidth;
            const auto task_parent = get_heap_elem_off_chip(parent);
            if (task_i <= task_parent) break;

            set_heap_elem_off_chip(i, task_parent);
            set_heap_index(task_parent.vid, i);
            i = parent;
          }

          if (!(i < kHeapOnChipSize) && i < kHeapOffChipBound) {
            const auto parent =
                (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                kHeapOffChipWidth;
            const auto task_parent = get_heap_elem_on_chip(parent);
            if (!(task_i <= task_parent)) {
              ++heapify_up_off_chip;

              set_heap_elem_off_chip(i, task_parent);
              set_heap_index(task_parent.vid, i);
              i = parent;
            } else {
              ++heapify_up_on_chip;
            }
          }

        heapify_up_on_chip:
          for (; i != 0 && i < kHeapOnChipSize;) {
#pragma HLS pipeline
            ++heapify_up_on_chip;

            const auto parent = (i - 1) / kHeapOnChipWidth;
            const auto task_parent = get_heap_elem_on_chip(parent);
            if (task_i <= task_parent) break;

            set_heap_elem_on_chip(i, task_parent);
            set_heap_index(task_parent.vid, i);
            i = parent;
          }

          set_heap_elem(i, task_i);
          set_heap_index(task_i.vid, i);
        }
        break;
      }
      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        CHECK_EQ(resp.task.vid() % kQueueCount, qid);
        const bool is_pushpop = req.op == QueueOp::PUSHPOP;
        if (heap_size != 0) {
          const Task front(heap_array_cache[0]);
          clear_heap_index(front.vid);

          if (!is_pushpop) {
            --heap_size;
          }

          resp.task_op = TaskOp::NEW;
          resp.task = front;

          if (heap_size != 0) {
            ++heapify_down_count;

            // Find proper index `i` for `task_i`.
            const Task task_i =
                is_pushpop ? Task(req.task) : get_heap_elem(heap_size);
            Vid i = 0;

          heapify_down_on_chip:
            for (; i < kHeapOnChipBound;) {
#pragma HLS pipeline
              ++heapify_down_on_chip;

              Vid max = -1;
              Task task_max = task_i;
              for (int j = 1; j <= kHeapOnChipWidth; ++j) {
#pragma HLS unroll
                const Vid child = i * kHeapOnChipWidth + j;
                if (child < heap_size) {
                  const auto task_child = get_heap_elem_on_chip(child);
                  if (!(task_child <= task_max)) {
                    max = child;
                    task_max = task_child;
                  }
                }
              }
              if (max == -1) break;

              set_heap_elem_on_chip(i, task_max);
              set_heap_index(task_max.vid, i);
              i = max;
            }

            auto heapify_down_cmp = [&](Task& task_max, Vid& max) {
              heapify_down_cmp:
                for (int j = 1; j <= kHeapOffChipWidth; ++j) {
#pragma HLS pipeline
                  const Vid child = i * kHeapOffChipWidth -
                                    kHeapDiff * (kHeapOffChipWidth - 1) + j;
                  if (child < heap_size) {
                    const auto task_child = get_heap_elem_off_chip(child);
                    if (!(task_child <= task_max)) {
                      max = child;
                      task_max = task_child;
                    }
                  }
                }
            };

            if (!(i < kHeapOnChipBound) && i < kHeapOnChipSize) {
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              heapify_down_cmp(task_max, max);

              if (max != -1) {
                set_heap_elem_on_chip(i, task_max);
                set_heap_index(task_max.vid, i);
                i = max;
              }
            }

          heapify_down_off_chip:
            for (; !(i < kHeapOnChipSize);) {
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              heapify_down_cmp(task_max, max);
              if (max == -1) break;

              set_heap_elem_off_chip(i, task_max);
              set_heap_index(task_max.vid, i);
              i = max;
            }

            set_heap_elem(i, task_i);
            set_heap_index(task_i.vid, i);
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

void EdgeReadAddrArbiter(tapa::istreams<Vid, kPeCount>& req_q,
                         tapa::ostreams<PeId, kShardCount>& id_q,
                         tapa::ostreams<Vid, kShardCount>& addr_q) {
  DECL_ARRAY(PeId, id, kShardCount, PeId());
  DECL_ARRAY(bool, id_valid, kShardCount, false);
  DECL_ARRAY(Vid, addr, kShardCount, kNullVid);
  DECL_ARRAY(bool, addr_valid, kShardCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kShardCount) + 2> pe_sid = 0;; ++pe_sid) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      const auto pe = pe_sid / 8 * kShardCount + sid;
      if (!id_valid[sid] &&
          SET(addr_valid[sid], req_q[pe].try_read(addr[sid]))) {
        id[sid] = pe;
        id_valid[sid] = true;
      }
    });

    RANGE(sid, kShardCount, {
      UNUSED RESET(id_valid[sid], id_q[sid].try_write(id[sid]));
      UNUSED RESET(addr_valid[sid], addr_q[sid].try_write(addr[sid]));
    });
  }
}

void EdgeReadDataArbiter(tapa::istreams<PeId, kShardCount>& id_q,
                         tapa::istreams<Edge, kShardCount>& data_in_q,
                         tapa::ostreams<Edge, kPeCount>& data_out_q) {
  DECL_ARRAY(PeId, id, kShardCount, PeId());
  DECL_ARRAY(bool, id_valid, kShardCount, false);
  DECL_ARRAY(Edge, data, kShardCount, Edge());
  DECL_ARRAY(bool, data_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      UNUSED SET(id_valid[sid], id_q[sid].try_read(id[sid]));
      UNUSED SET(data_valid[sid], data_in_q[sid].try_read(data[sid]));
      RANGE(pe_sid, kPeCount / kShardCount, {
        const auto pe = pe_sid * kShardCount + sid;
        if (id_valid[sid] && data_valid[sid] && id[sid] == pe &&
            data_out_q[pe].try_write(data[sid])) {
          id_valid[sid] = data_valid[sid] = false;
        }
      });
    });
  }
}

void UpdateReqArbiter(tapa::istreams<TaskOnChip, kPeCount>& in_q,
                      tapa::ostreams<TaskOnChip, kIntervalCount>& out_q) {
  DECL_ARRAY(TaskOnChip, update, kIntervalCount, TaskOnChip());
  DECL_ARRAY(bool, update_valid, kIntervalCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kIntervalCount - 1)> pe_iid = 0;;
       ++pe_iid) {
#pragma HLS pipeline II = 1
    RANGE(iid, kIntervalCount, {
      const auto pe = pe_iid * kIntervalCount + iid;
      UNUSED SET(update_valid[iid], in_q[pe].try_read(update[iid]));
      UNUSED RESET(update_valid[iid], out_q[iid].try_write(update[iid]));
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

void ProcElemS0(istream<TaskOnChip>& task_in_q, ostream<TaskOnChip>& task_out_q,
                ostream<Vid>& task_resp_q, ostream<Vid>& edges_read_addr_q) {
  Vid vid;
  Eid offset;

spin:
  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      task_out_q.write(task);
      vid = task.vid();
      offset = task.vertex().offset;
      i = task.vertex().degree;
    }

    if (i > 0) {
      edges_read_addr_q.write(offset);
      if (i == 1) {
        task_resp_q.write(vid);
      }

      ++offset;
      --i;
    }
  }
}

void ProcElemS1(istream<TaskOnChip>& task_in_q,
                istream<Edge>& edges_read_data_q,
                ostream<TaskOnChip>& update_out_q) {
  Vid vid;
  float distance;

  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      vid = task.vid();
      distance = task.vertex().distance;
      i = task.vertex().degree;
    }

    if (i > 0 && !edges_read_data_q.empty()) {
      const auto edge = edges_read_data_q.read(nullptr);
      update_out_q.write(Task{
          .vid = edge.dst,
          .vertex = {.parent = vid, .distance = distance + edge.weight},
      });
      --i;
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

using uint_sid_t = ap_uint<bit_length(kShardCount - 1)>;

void PopReqArbiter(istream<uint_sid_t>& req_in_q,
                   tapa::ostreams<bool, kQueueCount>& req_out_q) {
  static_assert(is_power_of(kQueueCount, 2), "invalid queue count");
  DECL_BUF(uint_sid_t, sid);
  DECL_ARRAY(ap_uint<bit_length(kQueueCount / kShardCount)>, sid_base,
             kShardCount, 0);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UNUSED SET(sid_valid, req_in_q.try_read(sid));
    if (sid_valid) {
      const auto qid = (sid_base[sid] * kShardCount + sid) % kQueueCount;
      if (req_out_q[qid].try_write(false)) {
        sid_valid = false;
        ++sid_base[sid];
      }
    }
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

using uint_pe_t = ap_uint<bit_length(kPeCount)>;
using uint_pe_per_shart_t = ap_uint<bit_length(kPeCount / kShardCount)>;

void Dispatcher(
    // Scalar.
    const Task root,
    // Metadata.
    tapa::mmap<int64_t> metadata,
    // Task and queue requests.
    tapa::ostreams<TaskOnChip, kPeCount>& task_req_q,
    tapa::istreams<Vid, kPeCount>& task_resp_q,
    istream<uint_noop_t>& update_noop_q, ostream<uint_sid_t>& queue_req_q,
    tapa::istream<QueueOpResp>& queue_resp_q) {
  // Process finished tasks.
  bool task_buf_valid = false;
  TaskOp task_buf;
  bool queue_buf_valid = false;
  QueueOpResp queue_buf;

  // Number of tasks whose parent task is sent to the PEs but not yet
  // acknowledged.
  int32_t active_task_count = root.vertex.degree;

  // Number of tasks generated by the PEs but not yet sent to the PEs.
  int32_t pending_task_count = 0;

  // Number of POP requests sent but not acknowledged.
  DECL_ARRAY(uint_pe_per_shart_t, pop_count, kShardCount, 0);

  DECL_ARRAY(bool, queue_empty, kQueueCount, true);

  DECL_ARRAY(uint_pe_per_shart_t, task_count_per_shard, kShardCount, 0);
  DECL_ARRAY(ap_uint<1>, task_count_per_pe, kPeCount, 0);

  DECL_ARRAY(uint_pe_per_shart_t, pe_base_per_shard, kShardCount, 0);

  task_req_q[root.vid % kShardCount].write(root);
  ++task_count_per_shard[root.vid % kShardCount];
  ++task_count_per_pe[root.vid % kShardCount];

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
          result &= queue_empty[i * kShardCount + sid]);
    return result;
  };

spin:
  for (; active_task_count || !all_of(queue_empty)
#ifndef __SYNTHESIS__
         || any_of(task_count_per_shard)
#endif  // __SYNTHESIS__
           ;
       ++cycle_count) {
    // Technically the loop should also check if there are active tasks not
    // acknowledged by the PEs (`any_of(task_count_per_pe)`), because if the
    // updates arrive before the task responses (which is only very rarely
    // possible in csim), the loop may leave the responses unconsumed.
#pragma HLS pipeline II = 1
    RANGE(pe, kPeCount, task_count_per_pe[pe] && ++pe_active_count[pe]);
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      queue_empty[queue_buf.task.vid() % kQueueCount] = false;
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
          --pop_count[queue_buf.task.vid() % kShardCount];
          --active_task_count;
          ++pending_task_count;

          // TaskOp statistics.
          ++queue_count;
          break;
        case QueueOp::POP:
          --pop_count[queue_buf.task.vid() % kShardCount];
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            STATS(recv, "QUEUE: NEW ");
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            queue_empty[queue_buf.task.vid() % kQueueCount] = true;
            STATS(recv, "QUEUE: NOOP");
          }
          break;
      }
    }

    const Vid sid =
        (task_buf_valid ? task_buf.task.vid() : cycle_count) % kShardCount;
    const bool should_pop =
        task_count_per_shard[sid] + pop_count[sid] + queue_buf_valid <
            kPeCount / kShardCount &&
        !shard_is_done(sid);
    if (should_pop) {
      // Dequeue tasks from the queue.
      if (queue_req_q.try_write(sid)) {
        ++pop_count[sid];
        STATS(send, "QUEUE: POP ");
      } else {
        ++queue_full_cycle_count;
      }
    }

    // Assign tasks to PEs.
    {
      const auto sid = queue_buf.task.vid() % kShardCount;
      const auto pe = (pe_base_per_shard[sid] * kShardCount + sid) % kPeCount;
      if (queue_buf_valid) {
        if (task_count_per_pe[pe] == 0 &&
            task_req_q[pe].try_write(queue_buf.task)) {
          active_task_count += queue_buf.task.vertex().degree;
          --pending_task_count;
          queue_buf_valid = false;
          ++task_count_per_shard[sid];
          ++task_count_per_pe[pe];

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

    {
      Vid vid;
      const auto pe = cycle_count % kPeCount;
      if (task_resp_q[pe].try_read(vid)) {
        --task_count_per_shard[vid % kShardCount];
        --task_count_per_pe[pe];
      }
    }
  }

#ifndef __SYNTHESIS__
  RANGE(sid, kShardCount, {
    CHECK_EQ(pop_count[sid], 0);
    CHECK_EQ(task_count_per_shard[sid], 0);
  });
  RANGE(pe, kPeCount, CHECK_EQ(task_count_per_pe[pe], 0));
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
  stream<uint_sid_t, 2> pop_req_q("pop_req");
  streams<bool, kQueueCount, 2> pop_req_qi("pop_req_i");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  streams<QueueOp, kQueueCount, 2> spq_req_q("systolic_req");
  streams<QueueOpResp, kQueueCount, 2> spq_resp_q("systolic_resp");
  streams<TaskOnChip, kQueueCount, 512> spq_overflow_q("systolic_overflow");
  streams<bool, kQueueCount, 2> ofq_pop_req_q("overflow_pop_req");
  streams<QueueOpResp, kQueueCount, 2> ofq_resp_q("overflow_resp");

  streams<TaskOnChip, kPeCount, 2> task_req_q("task_req");
  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Vid, kPeCount, 64> task_resp_q("task_resp");

  // For edges.
  tapa::streams<Vid, kPeCount, 2> edge_read_addr_q("edge_read_addr");
  tapa::streams<Edge, kPeCount, 2> edge_read_data_q("edge_read_data");
  tapa::streams<Vid, kShardCount, 2> edge_read_addr_qi("edges.read_addr");
  tapa::streams<Edge, kShardCount, 2> edge_read_data_qi("edges.read_data");
  tapa::streams<PeId, kShardCount, 256> edge_pe_qi("edges.pe");

  // For vertices.
  //   Connect PEs to the update request network.
  streams<TaskOnChip, kPeCount, 512> update_req_q;
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
#if 1
      .invoke<detach, kQueueCount>(CascadedQueue, seq(), push_req_qi,
                                   pop_req_qi, queue_resp_qi, spq_req_q,
                                   spq_resp_q, ofq_pop_req_q, ofq_resp_q)
      .invoke<detach, kQueueCount>(SystolicQueue, seq(), spq_req_q, spq_resp_q,
                                   spq_overflow_q)
      .invoke<detach, kQueueCount>(TaskQueue, seq(), spq_overflow_q,
                                   ofq_pop_req_q, ofq_resp_q, heap_array,
                                   heap_index)
#else
      .invoke<detach, kQueueCount>(TaskQueue, seq(), push_req_qi, pop_req_qi,
                                   queue_resp_qi, heap_array, heap_index)
#endif
      .invoke<detach>(PushReqArbiter, push_req_q, push_req_qi)
      .invoke<detach>(PopReqArbiter, pop_req_q, pop_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)

      // For edges.
      .invoke<-1, kShardCount>(EdgeMem, edge_read_addr_qi, edge_read_data_qi,
                               edges)
      .invoke<-1>(EdgeReadAddrArbiter, edge_read_addr_q, edge_pe_qi,
                  edge_read_addr_qi)
      .invoke<-1>(EdgeReadDataArbiter, edge_pe_qi, edge_read_data_qi,
                  edge_read_data_q)

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
      .invoke<detach, kPeCount>(ProcElemS0, task_req_q, task_req_qi,
                                task_resp_q, edge_read_addr_q)
      .invoke<detach, kPeCount>(ProcElemS1, task_req_qi, edge_read_data_q,
                                update_req_q);
}

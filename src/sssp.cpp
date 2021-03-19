#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

using tapa::detach;
using tapa::istream;
using tapa::istreams;
using tapa::mmap;
using tapa::ostream;
using tapa::ostreams;
using tapa::packet;
using tapa::seq;
using tapa::stream;
using tapa::streams;
using tapa::task;

static_assert(kQueueCount % kShardCount == 0,
              "current implementation requires that queue count is a multiple "
              "of shard count");

constexpr int kQueueMemCount = 2;

// Verbosity definitions:
//   v=5: O(1)
//   v=8: O(#vertex)
//   v=9: O(#edge)

void HeapIndexReqArbiter(istreams<HeapIndexReq, kQueueCount>& req_in_q,
                         ostreams<HeapIndexReq, kQueueMemCount>& req_out_q) {
  static_assert(kQueueCount % kQueueMemCount == 0,
                "this implementation requires that queue count is a multiple "
                "of queue memory count per channel");
  DECL_ARRAY(HeapIndexReq, req, kQueueMemCount, HeapIndexReq());
  DECL_ARRAY(bool, req_valid, kQueueMemCount, false);

spin:
  for (ap_uint<bit_length(kQueueCount / kQueueMemCount - 1)> base = 0;;
       ++base) {
#pragma HLS pipeline II = 1
    RANGE(mid, kQueueMemCount, {
      const auto qid = base * kQueueMemCount + mid;
      UNUSED SET(req_valid[mid], req_in_q[qid].try_read(req[mid]));
      UNUSED RESET(req_valid[mid], req_out_q[mid].try_write(req[mid]));
    });
  }
}

void HeapIndexMem(istream<HeapIndexReq>& req_q, ostream<HeapIndexEntry>& resp_q,
                  tapa::async_mmap<HeapIndexEntry> mem) {
  CLEAN_UP(clean_up, [&]() {
    // Check that heap_index is restored to the initial state.
    for (int i = 0; i < mem.size(); ++i) {
      CHECK(!mem.get()[i].valid()) << "i = " << i;
      CHECK_EQ(mem.get()[i].qid(), i % kQueueCount) << "i = " << i;
    }
  });

  DECL_BUF(Vid, read_addr);
  DECL_BUF(HeapIndexEntry, read_data);
  DECL_BUF(Vid, write_addr);
  DECL_BUF(HeapIndexEntry, write_data);

  DECL_ARRAY(ap_uint<5>, lock, kQueueCount, 0);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!req_q.empty()) {
      const auto req = req_q.peek(nullptr);
      switch (req.op) {
        case GET: {
          if (!read_addr_valid && lock[req.vid % kQueueCount] == 0) {
            read_addr = req.vid;
            read_addr_valid = true;
            req_q.read(nullptr);
          }
        } break;
        case SET: {
          if (!write_addr_valid && !write_data_valid) {
            write_addr = req.vid;
            write_data.set_qid(req.vid % kQueueCount);
            if (req.index == kNullVid) {
              write_data.invalidate();
            } else {
              write_data.set_index(0, req.index);
            }
            write_addr_valid = write_data_valid = true;
            req_q.read(nullptr);
            lock[req.vid % kQueueCount] = 30;
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
    RANGE(qid, kQueueCount, lock[qid] > 0 && --lock[qid]);
  }
}

void HeapIndexRespArbiter(istreams<HeapIndexEntry, kQueueMemCount>& resp_in_q,
                          ostreams<HeapIndexEntry, kQueueCount>& resp_out_q) {
  static_assert(kQueueCount % kQueueMemCount == 0,
                "this implementation requires that queue count is a multiple "
                "of queue memory count per channel");

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(mid, kQueueMemCount, {
      if (!resp_in_q[mid].empty()) {
        const auto resp = resp_in_q[mid].peek(nullptr);
        CHECK_EQ(resp.qid() % kQueueMemCount, mid);
        const auto qid = resp.qid() / kQueueMemCount * kQueueMemCount + mid;
        if (resp_out_q[qid].try_write(resp)) {
          resp_in_q[mid].read(nullptr);
        }
      }
    });
  }
}

void HeapIndexCache(istream<HeapIndexReq>& req_in_q, ostream<Vid>& resp_out_q,
                    ostream<HeapIndexReq>& req_out_q,
                    istream<HeapIndexEntry>& resp_in_q) {
  constexpr int kIndexCacheSize = 4096 * 4;
  tapa::packet<Vid, Vid> heap_index_cache[kIndexCacheSize];
#pragma HLS resource variable = heap_index_cache core = RAM_2P_URAM latency = 3
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
            const auto resp = resp_in_q.read();
            write_enable = true,
            new_entry = {req.vid, resp.valid() ? resp.index(0) : kNullVid};
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

void HeapArrayReqArbiter(istreams<HeapArrayReq, kQueueCount>& in_q,
                         ostreams<HeapArrayReq, kQueueMemCount>& out_q) {
  static_assert(kQueueCount % kQueueMemCount == 0,
                "this implementation requires that queue count is a multiple "
                "of queue memory count per channel");

spin:
  for (ap_uint<bit_length(kQueueCount / kQueueMemCount - 1)> base = 0;;
       ++base) {
#pragma HLS pipeline II = 1
    RANGE(mid, kQueueMemCount, {
      const auto qid = base * kQueueMemCount + mid;
      if (!in_q[qid].empty()) {
        auto req = in_q[qid].peek(nullptr);
        CHECK_GE(req.i, kHeapOnChipSize);
        req.i = (req.i - kHeapOnChipSize) * kQueueCount + qid;
        if (out_q[mid].try_write(req)) {
          in_q[qid].read(nullptr);
        }
      }
    });
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
      switch (req.op) {
        case GET:
        case SYNC: {
          if (!read_addr_valid && (req.op == GET || lock == 0)) {
            read_addr = req.i;
            read_addr_valid = true;
            req_q.read(nullptr);
          }
        } break;
        case SET: {
          if (!write_addr_valid && !write_data_valid) {
            write_addr = req.i;
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

void HeapArrayRespArbiter(istreams<TaskOnChip, kQueueMemCount>& in_q,
                          ostreams<TaskOnChip, kQueueCount>& out_q) {
  static_assert(kQueueCount % kQueueMemCount == 0,
                "this implementation requires that queue count is a multiple "
                "of queue memory count per channel");

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(mid, kQueueMemCount, {
      if (!in_q[mid].empty()) {
        const auto resp = in_q[mid].peek(nullptr);
        CHECK_EQ(resp.vid() % kQueueMemCount, mid);
        const auto qid =
            resp.vid() % kQueueCount / kQueueMemCount * kQueueMemCount + mid;
        if (out_q[qid].try_write(resp)) {
          in_q[mid].read(nullptr);
        }
      }
    });
  }
}

constexpr int kLevelCount = 13;

void PheapPush(QueueOp req, int idx, HeapElem& elem,
               ostream<HeapReq>& req_out_q) {
#pragma HLS inline
  if (elem.valid) {
    CHECK(elem.cap_left > 0 || elem.cap_right > 0);

    if (!(req.task <= elem.task)) {
      std::swap(elem.task, req.task);
    }

    if (elem.cap_right < elem.cap_left) {
      --elem.cap_left;
      idx = idx * 2;
    } else {
      --elem.cap_right;
      idx = idx * 2 + 1;
    }
    req_out_q.write({.addr = idx, .payload = req});
  } else {
    elem.task = req.task;
    elem.valid = true;
  }
}

void PheapPop(QueueOp req, int idx, HeapElem& elem,
              istream<HeapResp>& resp_in_q, ostream<HeapReq>& req_out_q) {
#pragma HLS inline
  CHECK(elem.valid);
  req_out_q.write({.addr = idx, .payload = req});
  ap_wait();
  const auto resp = resp_in_q.read();
  elem.task = resp.task;
  switch (resp.op) {
    case EMPTY: {
      elem.valid = false;
    } break;
    case LEFT: {
      ++elem.cap_left;
    } break;
    case RIGHT: {
      ++elem.cap_right;
    } break;
    case NOCHANGE: {
    } break;
  }
}

HeapRespOp PheapCmp(const HeapElem& left, const HeapElem& right) {
#pragma HLS inline
  const bool left_le_right = left.task <= right.task;
  const bool left_is_ok = !left.valid || (right.valid && left_le_right);
  const bool right_is_ok = !right.valid || (left.valid && !left_le_right);
  if (left_is_ok && right_is_ok) {
    return EMPTY;
  }
  CHECK_NE(left_is_ok, right_is_ok);
  if (!left_is_ok) {
    return LEFT;
  }
  CHECK(!right_is_ok);
  return RIGHT;
}

HeapRespOp PheapCmp(const HeapElem& left, const HeapElem& right,
                    const QueueOp& elem) {
#pragma HLS inline
  const bool left_le_elem = left.task <= elem.task;
  const bool right_le_elem = right.task <= elem.task;
  const bool left_le_right = left.task <= right.task;
  const bool left_is_ok = !left.valid || left_le_elem;
  const bool right_is_ok = !right.valid || right_le_elem;
  if (left_is_ok && right_is_ok) {
    return EMPTY;
  }
  if (!left_is_ok && (right_is_ok || !left_le_right)) {
    return LEFT;
  }
  CHECK(!right_is_ok && (left_is_ok || left_le_right));
  return RIGHT;
}

void PheapHead(
    // Scalar
    Vid qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    ostream<QueueOpResp>& queue_resp_q,
    // Internal
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q) {
#pragma HLS inline recursive
  const auto cap = (1 << (kLevelCount - 1)) - 1;
  HeapElem root{.valid = false, .cap_left = cap, .cap_right = cap};

  CLEAN_UP(clean_up, [&] {
    CHECK_EQ(root.valid, false) << "q[" << qid << "]";
    CHECK_EQ(root.cap_left, cap) << "q[" << qid << "]";
    CHECK_EQ(root.cap_right, cap) << "q[" << qid << "]";
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

    switch (req.op) {
      case QueueOp::PUSH: {
        VLOG(5) << "PUSH q[" << qid << "] <-  " << req.task;
      } break;
      case QueueOp::POP: {
        if (root.valid) {
          VLOG(5) << "POP  q[" << qid << "]  -> " << root.task;
        } else {
          VLOG(5) << "POP  q[" << qid << "]  -> {}";
        }
      } break;
      case QueueOp::PUSHPOP: {
        VLOG(5) << "PUSH q[" << qid << "] <-  " << req.task;
        VLOG(5) << "POP  q[" << qid << "]  -> "
                << (root.task <= req.task ? req.task : root.task);
      } break;
    }

    QueueOpResp resp{
        .queue_op = req.op,
        .task_op = TaskOp::NEW,
        .task = req.task,
    };

    switch (req.op) {
      case QueueOp::PUSH: {
        PheapPush(req, 0, root, req_out_q);
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        if (root.valid && !(req.is_pushpop() && root.task <= req.task)) {
          resp.task = root.task;
          CHECK_EQ(resp.task.vid() % kQueueCount, qid);

          PheapPop(req, 0, root, resp_in_q, req_out_q);
        } else if (req.is_pop()) {
          resp.task_op = TaskOp::NOOP;
        }
      } break;
    }

    queue_resp_q.write(resp);
  }
}

template <int level>
void PheapBody(
    // Heap array
    HeapElem (&heap_array)[1 << level],
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q,
    // Child level
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q) {
  const auto cap = (1 << (kLevelCount - level - 1)) - 1;

init:
  for (int i = 0; i < 1 << level; ++i) {
#pragma HLS pipeline II = 1
    heap_array[i].valid = false;
    heap_array[i].cap_left = heap_array[i].cap_right = cap;
  }

  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < 1 << level; ++i) {
      CHECK_EQ(heap_array[i].valid, false);
      CHECK_EQ(heap_array[i].cap_left, cap);
      CHECK_EQ(heap_array[i].cap_right, cap);
    }
  });

spin:
  for (;;) {
#pragma HLS pipeline off
    const auto req = req_in_q.read();
    auto idx = req.addr;
    CHECK_GE(idx, 0);
    CHECK_LT(idx, 1 << level);

    const auto elem_0 = heap_array[idx / 2 * 2];
    const auto elem_1 = heap_array[idx / 2 * 2 + 1];
    auto elem = idx % 2 == 0 ? elem_0 : elem_1;
    bool elem_write_enable = false;

    switch (req.payload.op) {
      case QueueOp::PUSH: {
        PheapPush(req.payload, idx, elem, req_out_q);
        elem_write_enable = true;
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        CHECK_EQ(idx % 2, 0);
        const bool is_pushpop = req.payload.is_pushpop();
        switch (auto op = is_pushpop ? PheapCmp(elem_0, elem_1, req.payload)
                                     : PheapCmp(elem_0, elem_1)) {
          case EMPTY:
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = req.payload.task,
            });
            break;
          case RIGHT:
            ++idx;
            elem = elem_1;
          case LEFT:
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = elem.task,
            });
            PheapPop(req.payload, idx * 2, elem, resp_in_q, req_out_q);
            elem_write_enable = true;
            break;
          default:
            CHECK(false);
        }
      } break;
    }

    if (elem_write_enable) {
      heap_array[idx] = elem;
    }
  }
}

void PheapTail(
    // Scalar
    Vid qid,
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q) {
  HeapElem heap_array[1 << (kLevelCount - 1)];
#pragma HLS data_pack variable = heap_array
#pragma HLS resource variable = heap_array core = RAM_2P_URAM

init:
  for (int i = 0; i < 1 << (kLevelCount - 1); ++i) {
#pragma HLS pipeline II = 1
    heap_array[i].valid = false;
    heap_array[i].cap_left = heap_array[i].cap_right = 0;
  }

  bool used = false;
  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < 1 << (kLevelCount - 1); ++i) {
      CHECK_EQ(heap_array[i].valid, false);
      CHECK_EQ(heap_array[i].cap_left, 0);
      CHECK_EQ(heap_array[i].cap_right, 0);
    }
    LOG_IF(INFO, !used) << "tail of queue[" << qid << "] was never used";
  });

spin:
  for (;; used = true) {
    const auto req = req_in_q.read();
    auto idx = req.addr;
    CHECK_GE(idx, 0);
    CHECK_LT(idx, 1 << (kLevelCount - 1));

    const auto elem_0 = heap_array[idx / 2 * 2];
    const auto elem_1 = heap_array[idx / 2 * 2 + 1];
    auto elem = idx % 2 == 0 ? elem_0 : elem_1;
    bool elem_write_enable = false;

    switch (req.payload.op) {
      case QueueOp::PUSH: {
        CHECK(!elem.valid) << "insufficient heap capacity";
        elem.valid = true;
        elem.task = req.payload.task;
        elem_write_enable = true;
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        CHECK_EQ(idx % 2, 0);
        const bool is_pushpop = req.payload.is_pushpop();
        switch (auto op = is_pushpop ? PheapCmp(elem_0, elem_1, req.payload)
                                     : PheapCmp(elem_0, elem_1)) {
          case EMPTY:
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = req.payload.task,
            });
            break;
          case RIGHT:
            ++idx;
            elem = elem_1;
          case LEFT:
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = elem.task,
            });
            CHECK(elem.valid);
            elem.task = req.payload.task;
            elem.valid = !is_pushpop;
            elem_write_enable = true;
            break;
          default:
            CHECK(false);
        }
      } break;
    }

    if (elem_write_enable) {
      heap_array[idx] = elem;
    }
  }
}

#define HEAP_PORTS                                         \
  istream<HeapReq>&req_in_q, ostream<HeapResp>&resp_out_q, \
      ostream<HeapReq>&req_out_q, istream<HeapResp>&resp_in_q
#define HEAP_BODY(level, mem)                               \
  _Pragma("HLS inline recursive");                          \
  HeapElem heap_array[1 << level];                          \
  _Pragma("HLS data_pack variable = heap_array");           \
  DO_PRAGMA(HLS resource variable = heap_array core = mem); \
  PheapBody<level>(heap_array, req_in_q, resp_out_q, req_out_q, resp_in_q)

void PheapBodyL1(HEAP_PORTS) { HEAP_BODY(1, RAM_S2P_LUTRAM); }
void PheapBodyL2(HEAP_PORTS) { HEAP_BODY(2, RAM_S2P_LUTRAM); }
void PheapBodyL3(HEAP_PORTS) { HEAP_BODY(3, RAM_S2P_LUTRAM); }
void PheapBodyL4(HEAP_PORTS) { HEAP_BODY(4, RAM_S2P_LUTRAM); }
void PheapBodyL5(HEAP_PORTS) { HEAP_BODY(5, RAM_S2P_LUTRAM); }
void PheapBodyL6(HEAP_PORTS) { HEAP_BODY(6, RAM_S2P_LUTRAM); }
void PheapBodyL7(HEAP_PORTS) { HEAP_BODY(7, RAM_S2P_LUTRAM); }
void PheapBodyL8(HEAP_PORTS) { HEAP_BODY(8, RAM_S2P_LUTRAM); }
void PheapBodyL9(HEAP_PORTS) { HEAP_BODY(9, RAM_S2P_BRAM); }
void PheapBodyL10(HEAP_PORTS) { HEAP_BODY(10, RAM_S2P_BRAM); }
void PheapBodyL11(HEAP_PORTS) { HEAP_BODY(11, RAM_2P_URAM); }
#undef HEAP_BODY
#undef HEAP_PORTS

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
  // Heap rule: child <= parent
  streams<HeapReq, kLevelCount - 1, 2> req_q;
  streams<HeapResp, kLevelCount - 1, 2> resp_q;
  task()
      // clang-format off
      .invoke<detach>(PheapHead, qid, push_req_q, pop_req_q, queue_resp_q, req_q[0], resp_q[0])
      .invoke<detach>(PheapBodyL1,  req_q[ 0], resp_q[ 0], req_q[ 1], resp_q[ 1])
      .invoke<detach>(PheapBodyL2,  req_q[ 1], resp_q[ 1], req_q[ 2], resp_q[ 2])
      .invoke<detach>(PheapBodyL3,  req_q[ 2], resp_q[ 2], req_q[ 3], resp_q[ 3])
      .invoke<detach>(PheapBodyL4,  req_q[ 3], resp_q[ 3], req_q[ 4], resp_q[ 4])
      .invoke<detach>(PheapBodyL5,  req_q[ 4], resp_q[ 4], req_q[ 5], resp_q[ 5])
      .invoke<detach>(PheapBodyL6,  req_q[ 5], resp_q[ 5], req_q[ 6], resp_q[ 6])
      .invoke<detach>(PheapBodyL7,  req_q[ 6], resp_q[ 6], req_q[ 7], resp_q[ 7])
      .invoke<detach>(PheapBodyL8,  req_q[ 7], resp_q[ 7], req_q[ 8], resp_q[ 8])
      .invoke<detach>(PheapBodyL9,  req_q[ 8], resp_q[ 8], req_q[ 9], resp_q[ 9])
      .invoke<detach>(PheapBodyL10, req_q[ 9], resp_q[ 9], req_q[10], resp_q[10])
      .invoke<detach>(PheapBodyL11, req_q[10], resp_q[10], req_q[11], resp_q[11])
      .invoke<detach>(PheapTail, qid, req_q[11], resp_q[11])
      // clang-format on
      ;
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
  DECL_ARRAY(bool, src_valid, kShardCount, false);
  DECL_ARRAY(bool, addr_valid, kShardCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kShardCount) + 2> pe_sid = 0;; ++pe_sid) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      const auto pe = pe_sid / 8 * kShardCount + sid;
      if (!src_valid[sid] && !addr_valid[sid] && req_q[pe].try_read(req[sid])) {
        src_valid[sid] = addr_valid[sid] = true;
      }

      UNUSED RESET(addr_valid[sid], addr_q[sid].try_write(req[sid].addr));
      UNUSED RESET(src_valid[sid], src_q[sid].try_write(req[sid].payload));
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
  int32_t visited_edge_count = 0;
  int32_t push_count = 0;
  int32_t pushpop_count = 0;
  int32_t pop_valid_count = 0;
  int32_t pop_noop_count = 0;
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

          // Statistics.
          ++push_count;
          break;
        case QueueOp::PUSHPOP:
          CHECK_EQ(queue_buf.task_op, TaskOp::NEW);
          queue_active.clear(qid);
          --active_task_count;
          ++pending_task_count;

          // Statistics.
          ++pushpop_count;
          break;
        case QueueOp::POP:
          queue_active.clear(qid);
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            STATS(recv, "QUEUE: NEW ");

            // Statistics.
            ++pop_valid_count;
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            queue_empty.set(qid);
            STATS(recv, "QUEUE: NOOP");

            // Statistics.
            ++pop_noop_count;
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
  metadata[1] = push_count;
  metadata[2] = pushpop_count;
  metadata[3] = pop_valid_count;
  metadata[4] = pop_noop_count;
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
          tapa::mmaps<Edge, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices,
          // For queues.
          tapa::mmap<Task> heap_array, tapa::mmap<HeapIndexEntry> heap_index) {
  streams<TaskOnChip, kIntervalCount, 2> push_req_q;
  streams<TaskOnChip, kQueueCount, 512> push_req_qi;
  stream<uint_qid_t, 2> pop_req_q("pop_req");
  streams<bool, kQueueCount, 2> pop_req_qi("pop_req_i");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  streams<HeapIndexReq, kQueueCount, 64> heap_index_req_q;
  streams<Vid, kQueueCount, 2> heap_index_resp_q;
  streams<HeapIndexReq, kQueueCount, kQueueCount / kQueueMemCount>
      heap_index_req_qi;
  streams<HeapIndexEntry, kQueueCount, kQueueCount / kQueueMemCount>
      heap_index_resp_qi;
  streams<HeapIndexReq, kQueueMemCount, 2> heap_index_req_qii;
  streams<HeapIndexEntry, kQueueMemCount, 2> heap_index_resp_qii;

  streams<HeapArrayReq, kQueueCount, 64> heap_array_req_q;
  streams<HeapArrayReq, kQueueMemCount, 2> heap_array_req_qi;
  streams<TaskOnChip, kQueueMemCount, 2> heap_array_resp_qi;
  streams<TaskOnChip, kQueueCount, 2> heap_array_resp_q;

  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Vid, kPeCount, 2> task_resp_qi("task_resp_i");

  stream<TaskReq, 2> task_req_q("task_req");
  stream<TaskResp, kPeCount> task_resp_q("task_resp");

  // For edges.
  streams<Vid, kShardCount, 2> edge_read_addr_q("edge_read_addr");
  streams<Edge, kShardCount, 2> edge_read_data_q("edge_read_data");
  streams<EdgeReq, kPeCount, kPeCount / kShardCount * 8> edge_req_q("edge_req");
  streams<SourceVertex, kShardCount, 64> src_q("source_vertices");

  // For vertices.
  //   Connect PEs to the update request network.
  streams<TaskOnChip, kShardCount, 2> update_req_q;
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
      .invoke<detach, kQueueCount>(TaskQueue, seq(), push_req_qi, pop_req_qi,
                                   queue_resp_qi, heap_array_req_q,
                                   heap_array_resp_q, heap_index_req_q,
                                   heap_index_resp_q)
      .invoke<detach, kQueueCount>(HeapIndexCache, heap_index_req_q,
                                   heap_index_resp_q, heap_index_req_qi,
                                   heap_index_resp_qi)
      .invoke<detach>(HeapIndexReqArbiter, heap_index_req_qi,
                      heap_index_req_qii)
      .invoke<detach, kQueueMemCount>(HeapIndexMem, heap_index_req_qii,
                                      heap_index_resp_qii, heap_index)
      .invoke<detach>(HeapIndexRespArbiter, heap_index_resp_qii,
                      heap_index_resp_qi)
      .invoke<detach>(HeapArrayReqArbiter, heap_array_req_q, heap_array_req_qi)
      .invoke<detach, kQueueMemCount>(HeapArrayMem, seq(), heap_array_req_qi,
                                      heap_array_resp_qi, heap_array)
      .invoke<detach>(HeapArrayRespArbiter, heap_array_resp_qi,
                      heap_array_resp_q)
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

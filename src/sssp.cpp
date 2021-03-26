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

void PiHeapIndexReqArbiter(istreams<HeapIndexReq, kLevelCount>& req_in_q,
                           ostream<packet<LevelId, HeapIndexReq>>& req_out_q) {
spin:
  for (LevelId level = 0;;
       level = level == kLevelCount - 1 ? LevelId(0) : LevelId(level + 1)) {
#pragma HLS pipeline II = 1
    if (req_in_q[level].empty()) continue;
    req_out_q.write({level, req_in_q[level].read(nullptr)});
  }
}

void PiHeapIndexRespArbiter(istream<packet<LevelId, HeapIndexResp>>& req_in_q,
                            ostreams<HeapIndexResp, kLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (req_in_q.empty()) continue;
    const auto resp = req_in_q.read(nullptr);
    req_out_q[resp.addr].write(resp.payload);
  }
}

void PiHeapIndex(istream<packet<LevelId, HeapIndexReq>>& req_q,
                 ostream<packet<LevelId, HeapIndexResp>>& resp_q) {
#pragma HLS inline recursive

  constexpr int kFreshCacheSize = 4096 * 8;
  constexpr int kStaleCapacity = kLevelCount * 2 + 2;

  HeapIndexEntry fresh_index[kFreshCacheSize];
#pragma HLS bind_storage variable = fresh_index type = RAM_S2P impl = URAM
#pragma HLS aggregate variable = fresh_index bit
init:
  for (int i = 0; i < kFreshCacheSize; ++i) {
    fresh_index[i].invalidate();
  }
  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < kFreshCacheSize; ++i) {
      CHECK(!fresh_index[i].valid());
    }
  });
  auto GetFreshIndexLocked = [&fresh_index](Vid vid) {
    return fresh_index[vid];
  };
  auto SetFreshIndexLocked = [&fresh_index](Vid vid,
                                            const HeapIndexEntry& entry) {
    fresh_index[vid] = entry;
    VLOG(5) << "INDX q[" << (vid % kQueueCount) << "] v[" << vid << "] -> "
            << entry;
  };

  DECL_ARRAY(HeapStaleIndexEntry, stale_index, kStaleCapacity, nullptr);
#pragma HLS aggregate variable = stale_index bit
  auto GetStaleIndexLocked = [&stale_index](Vid vid,
                                            int& idx) -> HeapIndexEntry {
    // If found, idx is set to the entry, entry is returned; otherwise, idx is
    // set to an available location, which is invalid.
    bool found_match = false;
    int match_idx;
    bool found_empty = false;
    int empty_idx;
    RANGE(i, kStaleCapacity, {
      const auto entry = stale_index[i];
      if (!found_match && entry.matches(vid)) {
        found_match |= true;
        match_idx = i;
      }
      if (!found_empty && !entry.valid()) {
        found_empty |= true;
        empty_idx = i;
      }
    });

    idx = found_match ? match_idx : empty_idx;
    return stale_index[idx];
  };
  auto SetStaleIndexLocked = [&stale_index](int idx, Vid vid,
                                            const HeapIndexEntry& entry) {
    CHECK(!stale_index[idx].valid() || stale_index[idx].matches(vid));
    stale_index[idx].set(vid, entry);
    VLOG(5) << "INDX q[" << (vid % kQueueCount) << "] v[" << vid << "] -> "
            << entry << " (stale)";
  };

spin:
  for (;;) {
#pragma HLS pipeline
    if (req_q.empty()) continue;
    const auto pkt = req_q.read(nullptr);
    const auto req = pkt.payload;
    int idx;
    const auto stale_entry = GetStaleIndexLocked(req.vid, idx);
    switch (req.op) {
      case GET_STALE: {
        int idx;
        resp_q.write({pkt.addr, {.entry = stale_entry}});
      } break;
      case CLEAR_STALE: {
        CHECK(stale_entry.valid());
        SetStaleIndexLocked(idx, req.vid, nullptr);
      } break;
      case ACQUIRE_INDEX: {
        if (stale_entry.valid()) {
          resp_q.write({pkt.addr, {.entry = {}, .yield = true}});
          break;
        }
        const auto active_entry = GetFreshIndexLocked(req.vid);
        if (active_entry.valid()) {
          if (active_entry.distance_le(req.entry)) {
            resp_q.write(
                {pkt.addr, {.entry = {}, .yield = false, .enable = false}});
            break;
          }
          SetStaleIndexLocked(idx, req.vid, active_entry);
        }
        CHECK_EQ(req.entry.level(), 0);
        CHECK_EQ(req.entry.index(), 0);
        SetFreshIndexLocked(req.vid, req.entry);

        resp_q.write({pkt.addr,
                      {.entry = active_entry, .yield = false, .enable = true}});
      } break;
      case UPDATE_INDEX: {
        if (stale_entry.distance_eq(req.entry)) {
          SetStaleIndexLocked(idx, req.vid, req.entry);
        } else {
          SetFreshIndexLocked(req.vid, req.entry);
        }
        resp_q.write({pkt.addr});
      } break;
      case CLEAR_FRESH: {
        SetFreshIndexLocked(req.vid, nullptr);
      } break;
    }
  }
}

template <int level>
void PiHeapPush(int qid, HeapReq req, HeapElem<level>& elem,
                istream<HeapResp>& resp_in_q, ostream<HeapReq>& req_out_q,
                ostream<HeapIndexReq>& index_req_q,
                istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline
  auto& idx = req.index;
  if (elem.valid) {
    if (req.replace) {
      if (elem.task.vid() == req.vid) {
        // Current element is replaced by req.
        // req must have a higher priority.
        CHECK_LE(elem.task, req.task)
            << "replacing " << elem.task << " with " << elem.task;

#ifndef __SYNTHESIS__
        {
          index_req_q.write({.op = GET_STALE, .vid = req.vid});
          ap_wait();
          const auto entry = index_resp_q.read().entry;
          CHECK(entry.valid());
          CHECK_EQ(entry.level(), level)
              << "q[" << qid << "] level: " << level << " vid: " << req.vid
              << " elem: " << elem.task;
          CHECK_EQ(entry.index(), idx)
              << "q[" << qid << "] level: " << level << " vid: " << req.vid
              << " elem: " << elem.task;
          CHECK_EQ(entry.distance(), elem.task.vertex().distance);
        }
#endif

        index_req_q.write({.op = CLEAR_STALE, .vid = req.vid});
        elem.task = req.task;
        // Do nothing if new element has a lower priority.
      } else {
        // Forward the current element to correct place.
        index_req_q.write({.op = GET_STALE, .vid = req.vid});
        ap_wait();
        const auto old_idx = index_resp_q.read().entry;
        CHECK(old_idx.valid()) << "q[" << qid << "] level: " << level
                               << " vid: " << req.vid << " elem: " << req.task;
        CHECK_GE(old_idx.level(), level)
            << "q[" << qid << "] level: " << level << " vid: " << req.vid
            << " elem: " << req.task;
        CHECK(old_idx.is_descendant_of(level, idx))
            << "q[" << qid << "]; old_idx: " << old_idx << "; level: " << level
            << "; idx: " << idx;

        CHECK_GE(old_idx.level(), level + 1)
            << "q[" << qid << "] level: " << level << " vid: " << req.vid
            << " elem: " << req.task;
        if (old_idx.is_descendant_of(level + 1, idx * 2)) {
          idx = idx * 2;
        } else {
          idx = idx * 2 + 1;
        }

        if (!(req.task <= elem.task)) {
          std::swap(elem.task, req.task);
        }
        index_req_q.write({
            .op = UPDATE_INDEX,
            .vid = req.task.vid(),
            .entry = {level + 1, idx, req.task.vertex().distance},
        });
        ap_wait();
        index_resp_q.read();
        req_out_q.write(req);
      }
    } else {  // req.replace
      CHECK(elem.cap_left > 0 || elem.cap_right > 0);

      if (elem.cap_right <= elem.cap_left) {  // TODO: optimize priority?
        --elem.cap_left;
        idx = idx * 2;
      } else {
        --elem.cap_right;
        idx = idx * 2 + 1;
      }

      if (!(req.task <= elem.task)) {
        std::swap(elem.task, req.task);
      }
      index_req_q.write({
          .op = UPDATE_INDEX,
          .vid = req.task.vid(),
          .entry = {level + 1, idx, req.task.vertex().distance},
      });
      ap_wait();
      index_resp_q.read();
      req_out_q.write(req);
    }
  } else {  // elem.valid
    elem.task = req.task;
    elem.valid = true;
    CHECK_EQ(req.replace, false) << "q[" << qid
                                 << "] if the write target is empty, we must "
                                    "not be replacing existing vid";
  }
}

template <int level>
void PiHeapPop(QueueOp req, int idx, HeapElem<level>& elem,
               istream<HeapResp>& resp_in_q, ostream<HeapReq>& req_out_q) {
#pragma HLS inline
  CHECK(elem.valid);
  req_out_q.write({
      .index = idx * 2,
      .op = req.op,
      .task = req.task,
      .replace = false,
  });
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

template <int level>
HeapRespOp PiHeapCmp(const HeapElem<level>& left,
                     const HeapElem<level>& right) {
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

template <int level>
HeapRespOp PiHeapCmp(const HeapElem<level>& left, const HeapElem<level>& right,
                     const HeapReq& elem) {
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

void PiHeapHead(
    // Scalar
    Vid qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    ostream<QueueOpResp>& queue_resp_q,
    // Internal
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q,
    // Heap index
    ostream<HeapIndexReq>& index_req_q, istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline recursive
  const auto cap = (1 << (kLevelCount - 1)) - 1;
  HeapElem<0> root{.valid = false, .cap_left = cap, .cap_right = cap};

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
    if (!do_push) pop_req_q.read(do_pop);

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
        HeapIndexResp heap_index_resp;
      acquire:
        do {
#pragma HLS pipeline off
          index_req_q.write({.op = ACQUIRE_INDEX,
                             .vid = req.task.vid(),
                             .entry = {0, 0, req.task.vertex().distance}});
          ap_wait();
          heap_index_resp = index_resp_q.read();
        } while (heap_index_resp.yield);
        if (heap_index_resp.enable) {
          PiHeapPush(qid,
                     {
                         .index = 0,
                         .op = req.op,
                         .task = req.task,
                         .replace = heap_index_resp.entry.valid(),
                         .vid = req.task.vid(),
                     },
                     root, resp_in_q, req_out_q, index_req_q, index_resp_q);
        }
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        if (root.valid && !(req.is_pushpop() && root.task <= req.task)) {
          resp.task = root.task;
          CHECK_EQ(resp.task.vid() % kQueueCount, qid);

          index_req_q.write({.op = CLEAR_FRESH, .vid = root.task.vid()});

          PiHeapPop(req, 0, root, resp_in_q, req_out_q);
        } else if (req.is_pop()) {
          resp.task_op = TaskOp::NOOP;
        }
      } break;
    }

    queue_resp_q.write(resp);
  }
}

template <int level>
void PiHeapBody(
    //
    int qid,
    // Heap array
    HeapElem<level> (&heap_array)[1 << level],
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q,
    // Child level
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q,
    //
    ostream<HeapIndexReq>& index_req_q, istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline recursive
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
    auto idx = req.index;
    CHECK_GE(idx, 0);
    CHECK_LT(idx, 1 << level);

    const auto elem_0 = heap_array[idx / 2 * 2];
    const auto elem_1 = heap_array[idx / 2 * 2 + 1];
    auto elem = idx % 2 == 0 ? elem_0 : elem_1;
    bool elem_write_enable = false;

    switch (req.op) {
      case QueueOp::PUSH: {
        PiHeapPush(qid, req, elem, resp_in_q, req_out_q, index_req_q,
                   index_resp_q);
        elem_write_enable = true;
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        CHECK_EQ(idx % 2, 0);
        const bool is_pushpop = req.op == QueueOp::PUSHPOP;
        switch (auto op = is_pushpop ? PiHeapCmp(elem_0, elem_1, req)
                                     : PiHeapCmp(elem_0, elem_1)) {
          case EMPTY:
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = req.task,
            });
            break;
          case RIGHT:
            ++idx;
            elem = elem_1;
          case LEFT:
            index_req_q.write({
                .op = UPDATE_INDEX,
                .vid = elem.task.vid(),
                .entry = {level - 1, idx / 2, elem.task.vertex().distance},
            });
            ap_wait();
            index_resp_q.read();
            resp_out_q.write({
                .op = is_pushpop ? NOCHANGE : op,
                .task = elem.task,
            });
            PiHeapPop({.op = req.op, .task = req.task}, idx, elem, resp_in_q,
                      req_out_q);
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

void PiHeapDummyTail(
    // Scalar
    Vid qid,
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q) {
spin:
  for (;;) {
    const auto is_empty = req_in_q.empty();
    const auto is_full = resp_out_q.full();
    CHECK(is_empty);
    CHECK(!is_full);
  }
}

#define HEAP_PORTS                                                  \
  int qid, istream<HeapReq>&req_in_q, ostream<HeapResp>&resp_out_q, \
      ostream<HeapReq>&req_out_q, istream<HeapResp>&resp_in_q,      \
      ostream<HeapIndexReq>&index_req_out_q,                        \
      istream<HeapIndexResp>&index_resp_in_q
#define HEAP_BODY(level, mem)                                                  \
  _Pragma("HLS inline recursive");                                             \
  HeapElem<level> heap_array[1 << level];                                      \
  _Pragma("HLS aggregate variable = heap_array bit");                          \
  DO_PRAGMA(HLS bind_storage variable = heap_array type = RAM_S2P impl = mem); \
  PiHeapBody<level>(qid, heap_array, req_in_q, resp_out_q, req_out_q,          \
                    resp_in_q, index_req_out_q, index_resp_in_q)

void PiHeapBodyL1(HEAP_PORTS) { HEAP_BODY(1, LUTRAM); }
void PiHeapBodyL2(HEAP_PORTS) { HEAP_BODY(2, LUTRAM); }
void PiHeapBodyL3(HEAP_PORTS) { HEAP_BODY(3, LUTRAM); }
void PiHeapBodyL4(HEAP_PORTS) { HEAP_BODY(4, LUTRAM); }
void PiHeapBodyL5(HEAP_PORTS) { HEAP_BODY(5, LUTRAM); }
void PiHeapBodyL6(HEAP_PORTS) { HEAP_BODY(6, LUTRAM); }
void PiHeapBodyL7(HEAP_PORTS) { HEAP_BODY(7, LUTRAM); }
void PiHeapBodyL8(HEAP_PORTS) { HEAP_BODY(8, LUTRAM); }
void PiHeapBodyL9(HEAP_PORTS) { HEAP_BODY(9, BRAM); }
void PiHeapBodyL10(HEAP_PORTS) { HEAP_BODY(10, BRAM); }
void PiHeapBodyL11(HEAP_PORTS) { HEAP_BODY(11, BRAM); }
void PiHeapBodyL12(HEAP_PORTS) { HEAP_BODY(12, URAM); }
void PiHeapBodyL13(HEAP_PORTS) { HEAP_BODY(13, URAM); }
void PiHeapBodyL14(HEAP_PORTS) { HEAP_BODY(14, URAM); }
#undef HEAP_BODY
#undef HEAP_PORTS

void PiHeapArrayMem(tapa::async_mmap<Task> mem) {
spin:
  for (;;) {
    if (!mem.read_data_empty()) {
      mem.read_addr_try_write({});
      mem.write_addr_try_write({});
      mem.write_data_try_write({});
    }
  }
}

void PiHeapIndexMem(tapa::async_mmap<HeapIndexEntry> mem) {
spin:
  for (;;) {
    if (!mem.read_data_empty()) {
      mem.read_addr_try_write({});
      mem.write_addr_try_write({});
      mem.write_data_try_write({});
    }
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
    // Queue responses.
    tapa::ostream<QueueOpResp>& queue_resp_q) {
  // Heap rule: child <= parent
  streams<HeapReq, kLevelCount, 2> req_q;
  streams<HeapResp, kLevelCount, 2> resp_q;
  streams<HeapIndexReq, kLevelCount, 2> index_req_qs;
  streams<HeapIndexResp, kLevelCount, 2> index_resp_qs;
  stream<packet<LevelId, HeapIndexReq>, 2> index_req_q;
  stream<packet<LevelId, HeapIndexResp>, 2> index_resp_q;
  task()
      .invoke<detach>(PiHeapHead, qid, push_req_q, pop_req_q, queue_resp_q,
                      req_q[0], resp_q[0], index_req_qs[0], index_resp_qs[0])
      // clang-format off
      .invoke<detach>(PiHeapBodyL1,  qid, req_q[ 0], resp_q[ 0], req_q[ 1], resp_q[ 1], index_req_qs[ 1], index_resp_qs[ 1])
      .invoke<detach>(PiHeapBodyL2,  qid, req_q[ 1], resp_q[ 1], req_q[ 2], resp_q[ 2], index_req_qs[ 2], index_resp_qs[ 2])
      .invoke<detach>(PiHeapBodyL3,  qid, req_q[ 2], resp_q[ 2], req_q[ 3], resp_q[ 3], index_req_qs[ 3], index_resp_qs[ 3])
      .invoke<detach>(PiHeapBodyL4,  qid, req_q[ 3], resp_q[ 3], req_q[ 4], resp_q[ 4], index_req_qs[ 4], index_resp_qs[ 4])
      .invoke<detach>(PiHeapBodyL5,  qid, req_q[ 4], resp_q[ 4], req_q[ 5], resp_q[ 5], index_req_qs[ 5], index_resp_qs[ 5])
      .invoke<detach>(PiHeapBodyL6,  qid, req_q[ 5], resp_q[ 5], req_q[ 6], resp_q[ 6], index_req_qs[ 6], index_resp_qs[ 6])
      .invoke<detach>(PiHeapBodyL7,  qid, req_q[ 6], resp_q[ 6], req_q[ 7], resp_q[ 7], index_req_qs[ 7], index_resp_qs[ 7])
      .invoke<detach>(PiHeapBodyL8,  qid, req_q[ 7], resp_q[ 7], req_q[ 8], resp_q[ 8], index_req_qs[ 8], index_resp_qs[ 8])
      .invoke<detach>(PiHeapBodyL9,  qid, req_q[ 8], resp_q[ 8], req_q[ 9], resp_q[ 9], index_req_qs[ 9], index_resp_qs[ 9])
      .invoke<detach>(PiHeapBodyL10, qid, req_q[ 9], resp_q[ 9], req_q[10], resp_q[10], index_req_qs[10], index_resp_qs[10])
      .invoke<detach>(PiHeapBodyL11, qid, req_q[10], resp_q[10], req_q[11], resp_q[11], index_req_qs[11], index_resp_qs[11])
      .invoke<detach>(PiHeapBodyL12, qid, req_q[11], resp_q[11], req_q[12], resp_q[12], index_req_qs[12], index_resp_qs[12])
      .invoke<detach>(PiHeapBodyL13, qid, req_q[12], resp_q[12], req_q[13], resp_q[13], index_req_qs[13], index_resp_qs[13])
      .invoke<detach>(PiHeapBodyL14, qid, req_q[13], resp_q[13], req_q[14], resp_q[14], index_req_qs[14], index_resp_qs[14])
      // clang-format on
      .invoke<detach>(PiHeapDummyTail, qid, req_q[kLevelCount - 1],
                      resp_q[kLevelCount - 1])
      .invoke<detach>(PiHeapIndexReqArbiter, index_req_qs, index_req_q)
      .invoke<detach>(PiHeapIndex, index_req_q, index_resp_q)
      .invoke<detach>(PiHeapIndexRespArbiter, index_resp_q, index_resp_qs);
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
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void VertexMem(tapa::istream<Vid>& read_addr_q,
               tapa::ostream<Vertex>& read_data_q,
               tapa::async_mmap<Vertex> mem) {
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
                                   queue_resp_qi)
      .invoke<detach>(PushReqArbiter, push_req_q, push_req_qi)
      .invoke<detach>(PopReqArbiter, pop_req_q, pop_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)

      // Put mmaps are in the top level to enable flexible floorplanning.
      .invoke<detach>(PiHeapArrayMem, heap_array)
      .invoke<detach>(PiHeapIndexMem, heap_index)

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

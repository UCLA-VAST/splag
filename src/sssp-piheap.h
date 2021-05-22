#ifndef TAPA_SSSP_PIHEAP_H_
#define TAPA_SSSP_PIHEAP_H_

#include "sssp-kernel.h"

#define TAPA_SSSP_PHEAP_INDEX

using uint_qid_t = ap_uint<bit_length(kQueueCount - 1)>;

using uint_piheap_size_t = ap_uint<bit_length(kPiHeapCapacity)>;

enum HeapOp {
  GET_STALE,
  CLEAR_STALE,
  ACQUIRE_INDEX,
  UPDATE_INDEX,
  CLEAR_FRESH,
  GET = GET_STALE,
  SET = UPDATE_INDEX,
  CLEAR = CLEAR_STALE,
};

namespace queue_state {
enum QueueState {
  IDLE = 0,
  PUSH = 1,
  POP = 2,
  PUSHPOP = 3,
  INDEX = 3,
};
}  // namespace queue_state
using queue_state::QueueState;

struct QueueStateUpdate {
  QueueState state;
  uint_piheap_size_t size;
};

struct HeapIndexReq {
  HeapOp op;
  Vid vid;
  HeapIndexEntry entry;
};

struct HeapAcquireIndexContext {
  Vid vid;
  HeapIndexEntry entry;
};

struct HeapIndexResp {
  HeapIndexEntry entry;
  bool yield;   // If true, requester should try again.
  bool enable;  // If true, enable PUSH.
};

struct HeapArrayReq {
  HeapOp op;
  Vid i;
  TaskOnChip task;
};

template <int level>
struct HeapElem {
  using Capacity = ap_uint<bit_length(GetChildCapOfLevel(level))>;

  bool valid;
  TaskOnChip task;
  Capacity cap[kPiHeapWidth];
};

struct HeapReq {
  LevelIndex index;
  QueueOp::Op op;
  TaskOnChip task;
  bool replace;
  Vid vid;
};

enum HeapRespOp {
  EMPTY,     // No element returned.
  CHILD,     // New element is from a child.
  NOCHANGE,  // New element returned but children capacity not changed.
};

using uint_heap_child_t = ap_uint<log2(kPiHeapWidth)>;

struct HeapResp {
  HeapRespOp op;
  uint_heap_child_t child;
  TaskOnChip task;
};

using PiHeapStat = int32_t;

template <typename HeapElemType>
inline ap_uint<log2(kPiHeapWidth)> FindMax(HeapElemType elem) {
  ap_uint<log2(kPiHeapWidth)> pos = 0;
  auto max = elem.cap[0];
find_max:
  for (ap_uint<bit_length(kPiHeapWidth)> i = 1; i < kPiHeapWidth; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 2
    if (!(elem.cap[i] <= max)) {
      max = elem.cap[i];
      pos = i;
    }
  }
  return pos;
}

template <typename HeapElemType>
inline bool IsUpdateNeeded(const HeapElemType(&elems), const HeapReq& elem,
                           bool is_pushpop, uint_heap_child_t& max_pos) {
#pragma HLS inline
  bool is_max_pos_valid = false;
  bool is_max_task_valid = is_pushpop;
  TaskOnChip max_task = elem.task;
find_update:
  for (ap_uint<bit_length(kPiHeapWidth)> i = 0; i < kPiHeapWidth; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 2
    if (elems[i].valid &&
        (!is_max_task_valid || !(elems[i].task <= max_task))) {
      max_pos = i;
      max_task = elems[i].task;
      is_max_pos_valid |= true;
      is_max_task_valid |= true;
    }
  }
  return is_max_pos_valid;
}

template <typename HeapElemType>
inline void PiHeapPush(uint_qid_t qid, int level, HeapReq req,
                       HeapElemType& elem, tapa::istream<HeapResp>& resp_in_q,
                       tapa::ostream<HeapReq>& req_out_q,
                       tapa::ostream<HeapIndexReq>& index_req_q,
                       tapa::istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline
  auto& idx = req.index;
  if (elem.valid) {
    if (req.replace) {
#ifdef TAPA_SSSP_PHEAP_INDEX
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
        ap_wait();
        index_resp_q.read();
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
        const auto new_idx = old_idx.parent_index_at(level + 1);
        CHECK_GE(new_idx, idx * kPiHeapWidth);
        CHECK_LT(new_idx, (idx + 1) * kPiHeapWidth);
        idx = new_idx;

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
#ifdef TAPA_SSSP_PHEAP_PUSH_ACK
        ap_wait();
        resp_in_q.read();
#endif  // TAPA_SSSP_PHEAP_PUSH_ACK
      }
#endif  // TAPA_SSSP_PHEAP_INDEX

    } else {  // req.replace
      bool has_slot = false;
      RANGE(i, kPiHeapWidth, has_slot |= elem.cap[i] > 0);
      CHECK(has_slot) << "level " << level;

      const auto max = FindMax(elem);
      CHECK_GT(elem.cap[max], 0);
      --elem.cap[max];
      idx = idx * kPiHeapWidth + max;

      if (!(req.task <= elem.task)) {
        std::swap(elem.task, req.task);
      }
#ifdef TAPA_SSSP_PHEAP_INDEX
      index_req_q.write({
          .op = UPDATE_INDEX,
          .vid = req.task.vid(),
          .entry = {level + 1, idx, req.task.vertex().distance},
      });
      ap_wait();
      index_resp_q.read();
#else   // TAPA_SSSP_PHEAP_INDEX
      const auto is_full = index_req_q.full();
      const auto is_empty = index_resp_q.empty();
      CHECK(!is_full);
      CHECK(is_empty);
#endif  // TAPA_SSSP_PHEAP_INDEX
      req_out_q.write(req);
#ifdef TAPA_SSSP_PHEAP_PUSH_ACK
      ap_wait();
      resp_in_q.read();
#endif  // TAPA_SSSP_PHEAP_PUSH_ACK
    }
  } else {  // elem.valid
    elem.task = req.task;
    elem.valid = true;
    CHECK_EQ(req.replace, false) << "q[" << qid
                                 << "] if the write target is empty, we must "
                                    "not be replacing existing vid";
  }
}

template <typename HeapElemType>
inline void PiHeapPop(QueueOp req, int idx, HeapElemType& elem,
                      tapa::istream<HeapResp>& resp_in_q,
                      tapa::ostream<HeapReq>& req_out_q) {
#pragma HLS inline
  CHECK(elem.valid);
  req_out_q.write({
      .index = idx * kPiHeapWidth,
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
    case CHILD: {
      ++elem.cap[resp.child];
      CHECK_GT(elem.cap[resp.child], 0);
    } break;
    case NOCHANGE: {
    } break;
  }
}

template <typename HeapElemType>
inline bool IsPiHeapElemUpdated(  //
    uint_qid_t qid, int level, const HeapReq& req,
    //
    const HeapElemType (&elems)[kPiHeapWidth],
    LevelIndex& idx,     // In/out
    HeapElemType& elem,  // Output
                         // Parent level
    tapa::istream<HeapReq>& req_in_q, tapa::ostream<HeapResp>& resp_out_q,
    // Child level
    tapa::ostream<HeapReq>& req_out_q, tapa::istream<HeapResp>& resp_in_q,
    //
    tapa::ostream<HeapIndexReq>& index_req_q,
    tapa::istream<HeapIndexResp>& index_resp_q) {
  elem = elems[idx % kPiHeapWidth];
  bool is_elem_written = false;
  switch (req.op) {
    case QueueOp::PUSH: {
#ifdef TAPA_SSSP_PHEAP_PUSH_ACK
      resp_out_q.write({});
#endif  // TAPA_SSSP_PHEAP_PUSH_ACK
      PiHeapPush(qid, level, req, elem, resp_in_q, req_out_q, index_req_q,
                 index_resp_q);
      is_elem_written = true;
    } break;

    case QueueOp::PUSHPOP:
    case QueueOp::POP: {
      CHECK_EQ(idx % kPiHeapWidth, 0);
      const bool is_pushpop = req.op == QueueOp::PUSHPOP;
      uint_heap_child_t child;
      if (IsUpdateNeeded(elems, req, is_pushpop, child)) {
        idx += child;
        elem = elems[child];
#ifdef TAPA_SSSP_PHEAP_INDEX
        index_req_q.write({
            .op = UPDATE_INDEX,
            .vid = elem.task.vid(),
            .entry = {level - 1, idx / kPiHeapWidth,
                      elem.task.vertex().distance},
        });
        ap_wait();
        index_resp_q.read();
#endif  // TAPA_SSSP_PHEAP_INDEX
        resp_out_q.write({
            .op = is_pushpop ? NOCHANGE : CHILD,
            .child = child,
            .task = elem.task,
        });
        PiHeapPop({.op = req.op, .task = req.task}, idx, elem, resp_in_q,
                  req_out_q);
        is_elem_written = true;
      } else {
        resp_out_q.write({
            .op = is_pushpop ? NOCHANGE : EMPTY,
            .child = child,
            .task = req.task,
        });
      }
    } break;
  }
  return is_elem_written;
}

template <int level>
inline void PiHeapBody(
    //
    uint_qid_t qid,
    // Heap array
    HeapElem<level> (&heap_array)[GetCapOfLevel(level)],
    // Parent level
    tapa::istream<HeapReq>& req_in_q, tapa::ostream<HeapResp>& resp_out_q,
    // Child level
    tapa::ostream<HeapReq>& req_out_q, tapa::istream<HeapResp>& resp_in_q,
    //
    tapa::ostream<HeapIndexReq>& index_req_q,
    tapa::istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline recursive
  const auto cap = GetChildCapOfLevel(level);

init:
  for (int i = 0; i < GetCapOfLevel(level); ++i) {
#pragma HLS pipeline II = 1
    heap_array[i].valid = false;
    RANGE(j, kPiHeapWidth, heap_array[i].cap[j] = cap);
  }

  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < GetCapOfLevel(level); ++i) {
      CHECK_EQ(heap_array[i].valid, false);
      RANGE(j, kPiHeapWidth, CHECK_EQ(heap_array[i].cap[j], cap));
    }
  });

spin:
  for (;;) {
#pragma HLS pipeline off
    const auto req = req_in_q.read();
    auto idx = req.index;
    CHECK_GE(idx, 0);
    CHECK_LT(idx, GetCapOfLevel(level));

    HeapElem<level> elems[kPiHeapWidth];
#pragma HLS aggregate variable = elems bit

  read_elems:
    for (int i = 0; i < kPiHeapWidth; ++i) {
      elems[i] = heap_array[idx / kPiHeapWidth * kPiHeapWidth + i];
    }

    HeapElem<level> elem;  // Output from IsPiHeapElemUpdated.
    if (IsPiHeapElemUpdated(qid, level, req, elems, idx, elem, req_in_q,
                            resp_out_q, req_out_q, resp_in_q, index_req_q,
                            index_resp_q)) {
      heap_array[idx] = elem;
    }
  }
}

#endif  // TAPA_SSSP_PIHEAP_H_

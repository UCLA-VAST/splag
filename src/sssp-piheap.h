#ifndef TAPA_SSSP_PIHEAP_H_
#define TAPA_SSSP_PIHEAP_H_

#include "sssp-kernel.h"

#define TAPA_SSSP_PHEAP_INDEX

using uint_qid_t = ap_uint<bit_length(kQueueCount - 1)>;

using uint_piheap_size_t = ap_uint<bit_length(kPiHeapCapacity)>;

using uint_on_chip_level_index_t =
    ap_uint<bit_length(GetCapOfLevel(kOnChipLevelCount - 1) - 1)>;

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

struct IndexStateUpdate {
  LevelId level;
  HeapOp op;
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
  ap_uint<Capacity::width + log2(kPiHeapWidth)> size;  // Size of all children.
};

struct HeapReq {
  LevelIndex index;
  QueueOp::Op op;
  TaskOnChip task;
  bool replace;
  Vid vid;
};

using uint_heap_child_t = ap_uint<log2(kPiHeapWidth)>;

struct HeapResp {
  uint_heap_child_t child;
  TaskOnChip task;
};

using PiHeapStat = int32_t;

template <typename HeapElemType, int N>
inline void ReadElemPair(const HeapElemType (&elems_in)[N],
                         uint_on_chip_level_index_t pos,
                         HeapElemType (&elem_pair_out)[2]) {
  elem_pair_out[0] = elems_in[pos / 2 * 2];
  elem_pair_out[1] = elems_in[pos / 2 * 2 + 1];
}

inline void ReadElemPair(tapa::istream<HeapElemPacked>& read_data_q,
                         uint_on_chip_level_index_t,
                         HeapElemAxi (&elem_pair_out)[2]) {
  HeapElemAxi::Unpack(read_data_q.read(), elem_pair_out);
}

template <typename HeapElemType, typename HeapElemSource>
inline bool IsUpdateNeeded(HeapElemSource& elems_in, const HeapReq& req,
                           LevelIndex& idx, HeapElemType (&elem_pair_out)[2]) {
#pragma HLS inline
  CHECK_EQ(req.index % kPiHeapWidth, 0);
  bool is_max_pos_valid = false;
  HeapElemType max_elem;
  max_elem.valid = req.op == QueueOp::PUSHPOP;
  max_elem.task = req.task;
find_update:
  for (ap_uint<bit_length(kPiHeapWidth)> i = 0; i < kPiHeapWidth; i += 2) {
#pragma HLS pipeline II = 1 rewind
    DECL_ARRAY(HeapElemType, elem_pair, 2, HeapElemType());
    ReadElemPair(elems_in, req.index + i, elem_pair);
    const bool is_update_needed_0 =
        elem_pair[0].valid &&
        (!max_elem.valid || !(elem_pair[0].task <= max_elem.task));
    const bool is_update_needed_1 =
        elem_pair[1].valid &&
        (!max_elem.valid || !(elem_pair[1].task <= max_elem.task));
    if (is_update_needed_0 || is_update_needed_1) {
      elem_pair_out[0] = elem_pair[0];
      elem_pair_out[1] = elem_pair[1];
      is_max_pos_valid |= true;
      if (!elem_pair[0].valid ||
          (elem_pair[1].valid && elem_pair[0].task <= elem_pair[1].task)) {
        idx = req.index + i + 1;
        max_elem = elem_pair[1];
      } else {
        idx = req.index + i + 0;
        max_elem = elem_pair[0];
      }
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

      ap_uint<log2(kPiHeapWidth)> max = 0;
      auto max_cap = elem.cap[0];
    find_max:
      for (ap_uint<bit_length(kPiHeapWidth)> i = 1; i < kPiHeapWidth; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = 2
        if (!(elem.cap[i] <= max_cap)) {
          max_cap = elem.cap[i];
          max = i;
        }
      }

      CHECK_GT(elem.cap[max], 0);
      --elem.cap[max];
      ++elem.size;
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
  const bool is_pushpop = req.op == QueueOp::PUSHPOP;

  // If no child is valid, PUSHPOP will return the current element and POP will
  // return invalid. No need to send request to the next level.
  if (elem.size == 0) {
    elem.valid = is_pushpop;
    elem.task = req.task;
    return;
  }

  req_out_q.write({
      .index = idx * kPiHeapWidth,
      .op = req.op,
      .task = req.task,
      .replace = false,
  });
  ap_wait();
  const auto resp = resp_in_q.read();
  elem.task = resp.task;
  if (!is_pushpop) {
    ++elem.cap[resp.child];
    --elem.size;
    CHECK_GT(elem.cap[resp.child], 0);
  }
}

template <typename HeapElemType, typename HeapElemSource>
inline bool IsPiHeapElemUpdated(  //
    uint_qid_t qid, int level, const HeapReq& req,
    //
    HeapElemSource& elems_in,          // Input
    LevelIndex& idx,                   // Output
    HeapElemType (&elem_pair_out)[2],  // Output

    // Parent level
    tapa::istream<HeapReq>& req_in_q, tapa::ostream<HeapResp>& resp_out_q,
    // Child level
    tapa::ostream<HeapReq>& req_out_q, tapa::istream<HeapResp>& resp_in_q,
    //
    tapa::ostream<HeapIndexReq>& index_req_q,
    tapa::istream<HeapIndexResp>& index_resp_q) {
  CHECK_GE(req.index, 0);
  CHECK_LT(req.index, GetCapOfLevel(level));
  bool is_elem_written = false;
  switch (req.op) {
    case QueueOp::PUSH: {
#ifdef TAPA_SSSP_PHEAP_PUSH_ACK
      resp_out_q.write({});
#endif  // TAPA_SSSP_PHEAP_PUSH_ACK
      ReadElemPair(elems_in, req.index, elem_pair_out);
      idx = req.index;
      PiHeapPush(qid, level, req, elem_pair_out[idx % 2], resp_in_q, req_out_q,
                 index_req_q, index_resp_q);
      is_elem_written = true;
    } break;

    case QueueOp::PUSHPOP:
    case QueueOp::POP: {
      const bool is_updated_needed =
          IsUpdateNeeded(elems_in, req, idx, elem_pair_out);
#ifdef TAPA_SSSP_PHEAP_INDEX
      if (is_updated_needed) {
        CHECK_EQ(req.index % kPiHeapWidth, 0);
        index_req_q.write({
            .op = UPDATE_INDEX,
            .vid = elem_pair_out[idx % 2].task.vid(),
            .entry = {level - 1, req.index / kPiHeapWidth,
                      elem_pair_out[idx % 2].task.vertex().distance},
        });
        ap_wait();
        index_resp_q.read();
      }
#endif  // TAPA_SSSP_PHEAP_INDEX

      resp_out_q.write(
          {.child = idx % kPiHeapWidth, .task = elem_pair_out[idx % 2].task});

      if (is_updated_needed) {
        PiHeapPop({.op = req.op, .task = req.task}, idx, elem_pair_out[idx % 2],
                  resp_in_q, req_out_q);
        is_elem_written = true;
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
    heap_array[i].size = 0;
  }

  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < GetCapOfLevel(level); ++i) {
      CHECK_EQ(heap_array[i].valid, false);
      RANGE(j, kPiHeapWidth, CHECK_EQ(heap_array[i].cap[j], cap));
      CHECK_EQ(heap_array[i].size, 0);
    }
  });

spin:
  for (;;) {
#pragma HLS pipeline off
    const auto req = req_in_q.read();

    // Outputs from IsPiHeapElemUpdated.
    LevelIndex idx;
    HeapElem<level> elem_pair[2];
#pragma HLS array_partition variable = elem_pair complete
#pragma HLS array_partition variable = elem_pair[0].cap complete
#pragma HLS array_partition variable = elem_pair[1].cap complete
    if (IsPiHeapElemUpdated(qid, level, req, heap_array, idx, elem_pair,
                            req_in_q, resp_out_q, req_out_q, resp_in_q,
                            index_req_q, index_resp_q)) {
      heap_array[idx] = elem_pair[idx % 2];
    }
  }
}

#endif  // TAPA_SSSP_PIHEAP_H_

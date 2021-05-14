#ifndef TAPA_SSSP_PIHEAP_H_
#define TAPA_SSSP_PIHEAP_H_

#include "sssp-kernel.h"

#define TAPA_SSSP_PHEAP_INDEX

template <typename HeapElemType>
inline ap_uint<log2(kPiHeapWidth)> FindMax(HeapElemType elem) {
  ap_uint<log2(kPiHeapWidth)> pos = 0;
  auto max = elem.cap[0];
find_max:
  for (ap_uint<bit_length(kPiHeapWidth)> i = 1; i < kPiHeapWidth; ++i) {
#pragma HLS pipeline II = 1
    if (!(elem.cap[i] <= max)) {
      max = elem.cap[i];
      pos = i;
    }
  }
  return pos;
}

template <typename HeapElemType>
inline bool IsUpdateNeeded(const HeapElemType(&elems), const HeapReq& elem,
                           bool is_pushpop, uint_pi_child_t& max_pos) {
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

#endif  // TAPA_SSSP_PIHEAP_H_

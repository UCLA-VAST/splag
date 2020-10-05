#ifndef TAPA_SSSP_KERNEL_H_
#define TAPA_SSSP_KERNEL_H_

#include <algorithm>
#include <type_traits>

#include "sssp.h"

// Constants and types.
using VertexAttrVec = tapa::vec_t<VertexAttr, kVertexVecLen>;

constexpr int kVertexPartitionFactor =
    kEdgeVecLen > kVertexVecLen
        ? kEdgeVecLen > kUpdateVecLen ? kEdgeVecLen : kUpdateVecLen
        : kVertexVecLen > kUpdateVecLen ? kVertexVecLen : kUpdateVecLen;

struct TaskReq {
  enum Phase { kScatter = 0, kGather = 1 };
  Phase phase;
  Iid iid;
  Vid interval_size;
  Eid edge_count;
  Vid vid_offset;
  Eid eid_offset;
  bool scatter_done;
};

inline std::ostream& operator<<(std::ostream& os, const TaskReq::Phase& obj) {
  return os << (obj == TaskReq::kScatter ? "SCATTER" : "GATHER");
}

inline std::ostream& operator<<(std::ostream& os, const TaskReq& obj) {
  if (obj.scatter_done) {
    return os << "{scatter_done: " << obj.scatter_done << "}";
  }
  return os << "{phase: " << obj.phase << ", iid: " << obj.iid
            << ", interval_size: " << obj.interval_size
            << ", edge_count: " << obj.edge_count
            << ", vid_offset: " << obj.vid_offset
            << ", eid_offset: " << obj.eid_offset << "}";
}

struct TaskResp {
  bool active;
};

inline std::ostream& operator<<(std::ostream& os, const TaskResp& obj) {
  return os << "{active: " << obj.active << "}";
}

struct UpdateReq {
  TaskReq::Phase phase;
  Iid iid;
  Eid update_count;
};

inline std::ostream& operator<<(std::ostream& os, const UpdateReq& obj) {
  return os << "{phase: " << obj.phase << ", iid: " << obj.iid
            << ", update_count: " << obj.update_count << "}";
}

struct VertexReq {
  Vid offset;
  Vid length;
};

using UpdateCount = tapa::packet<Iid, Eid>;
using UpdateVecPacket = tapa::packet<Iid, UpdateVec>;

// Convenient functions and macros.
inline bool All(const bool (&array)[1]) {
#pragma HLS inline
  return array[0];
}

template <int N>
inline bool All(const bool (&array)[N]) {
#pragma HLS inline
  return All((const bool(&)[N / 2])(array)) &&
         All((const bool(&)[N - N / 2])(array[N / 2]));
}

template <typename T>
inline void MemSet(T (&array)[1], T value) {
#pragma HLS inline
  array[0] = value;
}

template <typename T, int N>
inline void MemSet(T (&array)[N], T value) {
#pragma HLS inline
  MemSet((T(&)[N / 2])(array), value);
  MemSet((T(&)[N - N / 2])(array[N / 2]), value);
}

// Prints logging messages with tag and function name as prefix.
#define VLOG_F(level, tag) VLOG(level) << #tag << "@" << __FUNCTION__ << ": "
#define LOG_F(level, tag) LOG(level) << #tag << "@" << __FUNCTION__ << ": "

// Creates an unrolled loop with "var" iterating over [0, bound).
#define RANGE(var, bound, body)               \
  do {                                        \
    for (int var = 0; var < (bound); ++var) { \
      _Pragma("HLS unroll") body;             \
    }                                         \
  } while (0)

// Creates a pragma with stringified arguments.
#define DO_PRAGMA(x) _Pragma(#x)

// Fully partitions array "var" and initializes all elements with value "val".
// Note that "val" is evaluated once for each element.
#define INIT_ARRAY(var, val)                              \
  static_assert(std::is_array<decltype(var)>::value,      \
                "'" #var "' is not an array");            \
  DO_PRAGMA(HLS array_partition complete variable = var); \
  RANGE(_i, sizeof(var) / sizeof(var[0]), var[_i] = (val))

// Defines fully partitioned array "var" and initializes all elements with
// "val". Note that "val" is evaluated once for each element.
#define DECL_ARRAY(type, var, size, val) \
  type var[(size)];                      \
  INIT_ARRAY(var, val);

// Updates "var" with "val" if and only "var" is not true. Returns the possibly
// updated "var". Note that "val" is evaluated if and only if "var" is not true.
#define UPDATE(var, val) (var = var || (val))

#ifndef __SYNTHESIS__
inline void ap_wait() {}
inline void ap_wait_n(int) {}
#endif  // __SYNTEHSIS__

#endif  // TAPA_SSSP_KERNEL_H_

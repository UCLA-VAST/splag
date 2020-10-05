#ifndef TAPA_SSSP_H_
#define TAPA_SSSP_H_

#include <iostream>
#include <limits>

#include <tapa.h>

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.
using Iid = int16_t;  // Large enough to index all intervals.

constexpr Vid kNullVertex = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();
constexpr int kVertexUpdateDepDist = 3 + 1;  // II + 1

struct VertexAttr {
  Vid parent;
  float distance;
};

inline std::ostream& operator<<(std::ostream& os, const VertexAttr& obj) {
  return os << "{parent: " << obj.parent << ", distance: " << obj.distance
            << "}";
}

struct Edge {
  Vid src;
  Vid dst;
  float weight;
  uint32_t padding;
};

inline std::ostream& operator<<(std::ostream& os, const Edge& obj) {
  if (obj.src == kNullVertex) return os << "{}";
  return os << "{" << obj.src << " -> " << obj.dst << " (" << obj.weight
            << ")}";
}

struct Update {
  Vid src;
  Vid dst;
  float new_distance;
  uint32_t padding;
};

inline std::ostream& operator<<(std::ostream& os, const Update& obj) {
  if (obj.src == kNullVertex) return os << "{}";
  return os << "{src: " << obj.src << ", dst: " << obj.dst
            << ", new_distance: " << obj.new_distance << "}";
}

// Platform-specific constants and types.
constexpr int kMaxIntervalCount = 2048;
constexpr int kMaxIntervalSize = 1024 * 256;
constexpr int kPeCount = 8;
constexpr int kVecLenBytes = 64;  // 512 bits
constexpr int kVertexVecLen = kVecLenBytes / sizeof(float);
constexpr int kEdgeVecLen = kVecLenBytes / sizeof(Edge);
constexpr int kUpdateVecLen = kVecLenBytes / sizeof(Update);

// The host-kernel interface requires type with power-of-2 widths.
template <int N>
inline constexpr bool IsPowerOf2() {
  return N > 0 ? N % 2 == 0 && IsPowerOf2<N / 2>() : false;
}
template <>
inline constexpr bool IsPowerOf2<1>() {
  return true;
}
static_assert(IsPowerOf2<sizeof(Edge)>(),
              "Edge is not aligned to a power of 2");
static_assert(IsPowerOf2<sizeof(Update)>(),
              "Update is not aligned to a power of 2");

using VidVec = tapa::vec_t<Vid, kVertexVecLen>;
using FloatVec = tapa::vec_t<float, kVertexVecLen>;
using EdgeVec = tapa::vec_t<Edge, kEdgeVecLen>;
using UpdateVec = tapa::vec_t<Update, kUpdateVecLen>;

// operator<< overloads that skip null elements.
inline std::ostream& operator<<(std::ostream& os, const EdgeVec& obj) {
  os << "{";
  bool first = true;
  for (int i = 0; i < EdgeVec::length; ++i) {
    if (obj[i].src != kNullVertex) {
      if (!first) os << ", ";
      os << "[" << i << "]: " << obj[i];
      first = false;
    }
  }
  return os << "}";
}
inline std::ostream& operator<<(std::ostream& os, const UpdateVec& obj) {
  os << "{";
  bool first = true;
  for (int i = 0; i < UpdateVec::length; ++i) {
    if (obj[i].src != kNullVertex) {
      if (!first) os << ", ";
      os << "[" << i << "]: " << obj[i];
      first = false;
    }
  }
  return os << "}";
}

#endif  // TAPA_SSSP_H_

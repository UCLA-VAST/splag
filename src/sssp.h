#ifndef TAPA_SSSP_H_
#define TAPA_SSSP_H_

#include <iostream>
#include <limits>

#include <tapa.h>

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.

constexpr Vid kNullVertex = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();
constexpr int kVertexUpdateDepDist = 3 + 1;  // II + 1

// Push-based.
struct Edge {
  Vid dst;
  float weight;
};

inline std::ostream& operator<<(std::ostream& os, const Edge& obj) {
  if (obj.dst == kNullVertex) return os << "{}";
  return os << "{ * -> " << obj.dst << " (" << obj.weight << ") }";
}

struct Index {
  Eid offset;
  Vid count;
};

inline std::ostream& operator<<(std::ostream& os, const Index& obj) {
  return os << "{ offset: -> " << obj.offset << ", count: " << obj.count
            << " }";
}

// Platform-specific constants and types.
constexpr int kPeCount = 8;
constexpr int kVecLenBytes = 64;  // 512 bits
constexpr int kVertexVecLen = kVecLenBytes / sizeof(float);
static_assert(sizeof(float) == sizeof(Vid), "Vid must be 32-bit");
constexpr int kEdgeVecLen = kVecLenBytes / sizeof(Edge);

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

using VidVec = tapa::vec_t<Vid, kVertexVecLen>;
using FloatVec = tapa::vec_t<float, kVertexVecLen>;
using EdgeVec = tapa::vec_t<Edge, kEdgeVecLen>;

#endif  // TAPA_SSSP_H_

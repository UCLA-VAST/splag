#ifndef TAPA_SSSP_H_
#define TAPA_SSSP_H_

#include <iostream>
#include <limits>

#include <tapa.h>

// Kernel-friendly bit_cast.
template <typename To, typename From>
inline To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "unsupported bitcast");
  union {
    To to;
    From from;
  } u;
  u.from = from;
  return u.to;
}

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.

constexpr Vid kNullVid = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();
constexpr int kVertexUpdateDepDist = 3 + 1;  // II + 1

struct Vertex {
  Vid parent;
  float distance;

  // Compares distance.
  bool operator<=(const Vertex& other) const {
    return bit_cast<uint32_t>(distance) <= bit_cast<uint32_t>(other.distance);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Vertex& obj) {
  return os << "{parent: " << obj.parent << ", distance: " << obj.distance
            << "}";
}

// Push-based.
struct Edge {
  Vid dst;
  float weight;
};

inline std::ostream& operator<<(std::ostream& os, const Edge& obj) {
  if (obj.dst == kNullVid) return os << "{}";
  if (obj.weight < bit_cast<float>(0x10000000)) {
    // Weights are evenly distributed between 0 and 1; if it is this small,
    // almost certainly it should actually be interpreted as Vid.
    return os << "{ src: " << obj.dst
              << ", count: " << bit_cast<Vid>(obj.weight) << " }";
  }
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

struct Task {
  Vid vid;
  Vertex vertex;
  uint32_t padding;

  // Compares priority.
  bool operator<=(const Task& other) const { return other.vertex <= vertex; }
};

inline std::ostream& operator<<(std::ostream& os, const Task& obj) {
  return os << "{vid: " << obj.vid << ", vertex: " << obj.vertex << "}";
}

// Platform-specific constants and types.
constexpr int kPeCount = 2;
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

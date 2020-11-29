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

// Test power-of-k.
template <int N, int K>
struct is_power_of_t {
  constexpr static bool value =
      N > 0 ? N % K == 0 && is_power_of_t<N / K, K>::value : false;
};
template <int K>
struct is_power_of_t<1, K> {
  constexpr static bool value = true;
};
template <int N, int K>
inline constexpr bool is_power_of() {
  return is_power_of_t<N, K>::value;
}

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.

constexpr Vid kNullVid = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();

struct Vertex {
  Vid parent;
  float distance;
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
  bool operator<=(const Task& other) const {
    return bit_cast<uint32_t>(vertex.distance) >=
           bit_cast<uint32_t>(other.vertex.distance);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Task& obj) {
  return os << "{vid: " << obj.vid << ", vertex: " << obj.vertex << "}";
}

// Platform-specific constants and types.
constexpr int kPeCount = 4;
constexpr int kHeapOnChipWidth = 2;   // #children per on-heap element.
constexpr int kHeapOffChipWidth = 8;  // #children per off-heap element.

#endif  // TAPA_SSSP_H_

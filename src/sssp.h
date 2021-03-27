#ifndef TAPA_SSSP_H_
#define TAPA_SSSP_H_

#include <iostream>
#include <limits>

#include <tapa.h>

#include <ap_int.h>

// Kernel-friendly bit_cast.
template <typename To, typename From>
inline To bit_cast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "unsupported bitcast");
  return reinterpret_cast<const To&>(from);
}

// Test power-of-k.
inline constexpr bool is_power_of(int64_t n, int64_t b) {
  return n == 0 ? false : n == 1 ? true : n % b == 0 && is_power_of(n / b, b);
}

// Compile-time integer logarithm.
inline constexpr int64_t log(int64_t n, int64_t b) {
  return n == 1 ? 0 : log(n / b, b) + 1;
}
inline constexpr int64_t bit_length(int64_t n) { return 1 + log(n, 2); }

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.

constexpr Vid kNullVid = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();

struct Vertex {
  Vid parent;
  float distance;

  Eid offset;
  Vid degree;

  // Compares distance.
  bool operator<=(const Vertex& other) const {
    return bit_cast<uint32_t>(distance) <= bit_cast<uint32_t>(other.distance);
  }

  bool is_inf() const {
    return bit_cast<uint32_t>(distance) == bit_cast<uint32_t>(kInfDistance);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Vertex& obj) {
  return os << "{parent: " << obj.parent << ", distance: " << obj.distance
            << ", offset: " << obj.offset << ", degree: " << obj.degree << "}";
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
  uint32_t _padding_0;
  uint32_t _padding_1;
  uint32_t _padding_2;

  // Compares priority.
  bool operator<=(const Task& other) const { return other.vertex <= vertex; }
};

inline std::ostream& operator<<(std::ostream& os, const Task& obj) {
  return os << "{vid: " << obj.vid << ", vertex: " << obj.vertex << "}";
}

constexpr int kQueueCount = 4;

class HeapIndexEntry {
 public:
  int qid() const { return data(kQidMsb, kQidLsb); }
  void set_qid(int qid) { data(kQidMsb, kQidLsb) = qid; }

  bool valid() const { return data.bit(kValidBit); }
  void invalidate() { return data.clear(kValidBit); }

  int index(int i) const {
    CHECK(valid());
    CHECK_GE(i, 0);
    CHECK_LT(i, kIndexCount);
    const auto lsb = kIndexLsb + kIndexWidth * i;
    const auto msb = lsb + kIndexWidth - 1;
    CHECK_GE(lsb, kIndexLsb);
    CHECK_LT(msb, kValidBit);
    return data(msb, lsb);
  }
  void set_index(int i, int index) {
    CHECK_GE(i, 0);
    CHECK_LT(i, kIndexCount);
    const auto lsb = kIndexLsb + kIndexWidth * i;
    const auto msb = lsb + kIndexWidth - 1;
    CHECK_GE(lsb, kIndexLsb);
    CHECK_LT(msb, kValidBit);
    data.set(kValidBit);
    data(msb, lsb) = index;
  }

  static constexpr int kWidth = 32;
  static constexpr int kIndexCount = 1;

 private:
  ap_uint<kWidth> data;
  static constexpr int kQidWidth = log(kQueueCount, 2);
  static constexpr int kIndexWidth = 29;

  static constexpr int kQidLsb = 0;
  static constexpr int kQidMsb = kQidLsb + kQidWidth - 1;
  static constexpr int kIndexLsb = kQidMsb + 1;
  static constexpr int kValidBit = kIndexLsb + kIndexWidth * kIndexCount;

  static_assert(is_power_of(kWidth, 2), "AXI requires power-of-2 width");
  static_assert(kValidBit < kWidth, "invalid configuration");
};

inline std::ostream& operator<<(std::ostream& os, const HeapIndexEntry& obj) {
  os << "{qid: " << obj.qid();
  if (obj.valid()) {
    for (int i = 0; i < obj.kIndexCount; ++i) {
      os << ", index[" << i << "]: " << obj.index(i);
    }
  }
  return os << "}";
}

// Platform-specific constants and types.
constexpr int kPeCount = 32;

constexpr int kShardCount = 2;  // #edge partitions.
static_assert(
    kPeCount % kShardCount == 0,
    "current implementation assumes PE count is a multiple of shard count");

constexpr int kIntervalCount = 2;  // #vertex partitions.

constexpr int kHeapOnChipWidth = 8;  // #children per on-heap element.

constexpr int kHeapOffChipWidth = 16;  // #children per off-heap element.

#endif  // TAPA_SSSP_H_

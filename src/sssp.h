#ifndef TAPA_SSSP_H_
#define TAPA_SSSP_H_

#include <cstddef>

#include <array>
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
inline constexpr bool is_power_of(int n, int b) {
  return n == 0 ? false : n == 1 ? true : n % b == 0 && is_power_of(n / b, b);
}

// Compile-time integer logarithm.
inline constexpr int log(int n, int b) {
  return n == 1 ? 0 : log(n / b, b) + 1;
}
inline constexpr int log2(int n) { return log(n, 2); }
inline constexpr int bit_length(int n) { return 1 + log2(n); }

// Application-specific constants and type definitions.
using Vid = int32_t;  // Large enough to index all vertices.
using Eid = int32_t;  // Large enough to index all edges.

// since all distances are positive and normalized, we can store fewer bits.
// kFloatWidth bits from kFloatMsb is used.
constexpr int kFloatMsb = 30;
constexpr int kFloatWidth = 18;
constexpr int kFloatLsb = kFloatMsb - kFloatWidth + 1;
static_assert(kFloatLsb >= 0, "invalid configuration");
static_assert(kFloatMsb <= 32, "invalid configuration");

static constexpr int kVidWidth = 23;
static constexpr int kEidWidth = 28;

constexpr Vid kNullVid = -1;
constexpr float kInfDistance = std::numeric_limits<float>::infinity();

inline bool DistLt(float a, float b) {
  return static_cast<ap_uint<kFloatWidth>>(
             bit_cast<ap_uint<32>>(a).range(kFloatMsb, kFloatLsb)) <
         static_cast<ap_uint<kFloatWidth>>(
             bit_cast<ap_uint<32>>(b).range(kFloatMsb, kFloatLsb));
}

struct Vertex {
  Vid parent;
  float distance;

  Eid offset;
  Vid degree;

  // Compares distance.
  bool operator<(const Vertex& other) const {
    return DistLt(distance, other.distance);
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

  // Compares priority.
  bool operator<(const Task& other) const { return other.vertex < vertex; }
};

inline std::ostream& operator<<(std::ostream& os, const Task& obj) {
  return os << "{vid: " << obj.vid << ", vertex: " << obj.vertex << "}";
}

constexpr int kQueueCount = 1;

constexpr int kGlobalStatCount = 5;

constexpr int kEdgeUnitStatCount = 4;

constexpr int kVertexUniStatCount = 13;

constexpr int kCgpqPushPortCount = 16;

constexpr int kCgpqPushStageCount = log2(kCgpqPushPortCount);

constexpr int kQueueStatCount = 11 * kCgpqPushPortCount;

class TaskOnChip {
 public:
  TaskOnChip() {}

  TaskOnChip(std::nullptr_t) : data(-1) {}

  TaskOnChip(const Task& task) {
    set_vid(task.vid);
    set_vertex(task.vertex);
  }

  operator Task() const { return {.vid = vid(), .vertex = vertex()}; }

  // Compares priority.
  bool operator<(const TaskOnChip& other) const {
    return other.vertex() < vertex();
  }

  Vid vid() const { return data.range(vid_msb, vid_lsb); };
  Vertex vertex() const {
    ap_uint<32> distance = 0;
    distance.range(kFloatMsb, kFloatLsb) =
        data.range(distance_msb, distance_lsb);
    return {
        .parent = Vid(data.range(parent_msb, parent_lsb)),
        .distance = bit_cast<float>(distance.to_uint()),
    };
  }

  bool is_valid() const {
    return ap_int<kFloatWidth>(data.range(distance_msb, distance_lsb)) != -1;
  }

 private:
  void set_vid(Vid vid) { data.range(vid_msb, vid_lsb) = vid; }
  void set_value(const Vertex& vertex) {
    set_parent(vertex.parent);
    set_distance(vertex.distance);
  }
  void set_vertex(const Vertex& vertex) { set_value(vertex); }
  void set_parent(Vid parent) { data.range(parent_msb, parent_lsb) = parent; }
  void set_distance(float distance) {
    data.range(distance_msb, distance_lsb) =
        ap_uint<32>(bit_cast<uint32_t>(distance)).range(kFloatMsb, kFloatLsb);
  }

  static constexpr int kDegreeWidth = 20;
  static constexpr int vid_lsb = 0;
  static constexpr int vid_msb = vid_lsb + kVidWidth - 1;
  static constexpr int parent_lsb = vid_msb + 1;
  static constexpr int parent_msb = parent_lsb + kVidWidth - 1;
  static constexpr int distance_lsb = parent_msb + 1;
  static constexpr int distance_msb = distance_lsb + kFloatWidth - 1;
  static constexpr int length = distance_msb + 1;
  ap_uint<length> data;

  friend struct HeapElemAxi;
};

// Platform-specific constants and types.

constexpr int kShardCount = 8;  // #edge partitions.

constexpr int kEdgeVecLen = 2;

static_assert(kShardCount * kEdgeVecLen == kCgpqPushPortCount * kQueueCount);

constexpr int kIntervalCount = 16;
constexpr int kSubIntervalCount = kIntervalCount * 1;

using EdgeVec = tapa::vec_t<Edge, kEdgeVecLen>;

constexpr int kSwitchMuxDegree = 2;  // Mux output from this many switches.

constexpr int kSwitchPortCount = kSubIntervalCount;

constexpr int kSwitchStageCount = log2(kSwitchPortCount);

constexpr int kSwitchCount =
    kSwitchPortCount / 2 * kSwitchStageCount * kSwitchMuxDegree;

constexpr int kSwitchStatCount = 5;

constexpr int kPeCount = kShardCount;

static_assert(
    kPeCount % kShardCount == 0,
    "current implementation assumes PE count is a multiple of shard count");

constexpr int kHeapOnChipWidth = 8;  // #children per on-heap element.

constexpr int kHeapOffChipWidth = 16;  // #children per off-heap element.

constexpr int kSpilledTaskVecLen = 16;

static_assert(kShardCount * kEdgeVecLen == kSubIntervalCount);

static_assert(kSubIntervalCount == kSpilledTaskVecLen * kQueueCount);

constexpr int kPopSwitchPortCount = kSubIntervalCount / kQueueCount;

constexpr int kPopSwitchStageCount = log2(kPopSwitchPortCount);

using SpilledTask = std::array<TaskOnChip, kSpilledTaskVecLen>;

constexpr int kCgpqPhysMemCount = 4;

constexpr int kCgpqLogicMemCount = 2;

constexpr int kCgpqLogicMemWidth = kCgpqPhysMemCount / kCgpqLogicMemCount;

constexpr int kCgpqBankCountPerMem = kCgpqPushPortCount / kCgpqLogicMemCount;

constexpr int kSpilledTaskVecLenPerMem =
    kSpilledTaskVecLen / kCgpqLogicMemWidth;

using SpilledTaskPerMem = std::array<TaskOnChip, kSpilledTaskVecLenPerMem>;

constexpr int kCgpqChunkSize = 1024;

constexpr int kCgpqLevel = 14;

constexpr int kCgpqCapacity = (1 << kCgpqLevel) - 1;

using uint_spill_addr_t =
    ap_uint<bit_length(kCgpqCapacity) +
            log2(kCgpqChunkSize / kSpilledTaskVecLen * kCgpqBankCountPerMem)>;

using uint_interval_t = ap_uint<3>;

#endif  // TAPA_SSSP_H_

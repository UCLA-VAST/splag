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
constexpr int kFloatWidth = 30;
constexpr int kFloatLsb = kFloatMsb - kFloatWidth + 1;
static_assert(kFloatLsb >= 0, "invalid configuration");
static_assert(kFloatMsb <= 32, "invalid configuration");

static constexpr int kVidWidth = 25;
static constexpr int kEidWidth = 28;

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

#ifndef TAPA_SSSP_PHEAP_WIDTH
#define TAPA_SSSP_PHEAP_WIDTH 16
#endif

constexpr int kPiHeapWidth = TAPA_SSSP_PHEAP_WIDTH;

inline constexpr int GetCapOfLevel(int level) {
  constexpr int width = log2(kPiHeapWidth);  // Makes HLS happy.
  return 1 << (width * level);
}

#if TAPA_SSSP_PHEAP_WIDTH == 2
constexpr int kOnChipLevelCount = 15;
constexpr int kOffChipLevelCount = 5;
#elif TAPA_SSSP_PHEAP_WIDTH == 4
constexpr int kOnChipLevelCount = 8;
constexpr int kOffChipLevelCount = 3;
#elif TAPA_SSSP_PHEAP_WIDTH == 8
constexpr int kOnChipLevelCount = 5;
constexpr int kOffChipLevelCount = 3;
#elif TAPA_SSSP_PHEAP_WIDTH == 16
constexpr int kOnChipLevelCount = 4;
constexpr int kOffChipLevelCount = 2;
#else
#error "invalid TAPA_SSSP_PHEAP_WIDTH"
#endif  // TAPA_SSSP_PHEAP_WIDTH

constexpr int kLevelCount = kOnChipLevelCount + kOffChipLevelCount;

inline constexpr int GetChildCapOfLevel(int level) {
  return (GetCapOfLevel(kLevelCount - level - 1) - 1) / (kPiHeapWidth - 1);
}

constexpr int kPiHeapCapacity = GetChildCapOfLevel(0) * kPiHeapWidth + 1;

using OffChipLevelId = ap_uint<bit_length(kOffChipLevelCount - 1)>;
using LevelId = ap_uint<bit_length(kLevelCount - 1)>;
using LevelIndex = ap_uint<bit_length(GetCapOfLevel(kLevelCount - 1) - 1)>;

inline int GetAddrOfOffChipHeapElem(int level, int idx, int qid) {
  CHECK_GE(level, kOnChipLevelCount) << "not an off-chip level";
  CHECK_LT(level, kLevelCount);
  CHECK_GE(idx, 0);
  CHECK_LT(idx, GetCapOfLevel(level));
  CHECK_GE(qid, 0);
  CHECK_LT(qid, kQueueCount);
  CHECK_EQ(GetCapOfLevel(level) % kPiHeapWidth, 0);

  // Raw index in queue: GetCapOfLevel(level) + idx.
  // Mapped index is concatenated by the following parts:
  //  1. index;
  //  2. qid: log2(kQueueCount) bits;
  //  3. offset: log2(kPiHeapWidth)  bits.
  constexpr int kQidWidth = log2(kQueueCount);
  constexpr int kOffsetWidth = log2(kPiHeapWidth);
  constexpr int kIndexWidth = LevelIndex::width + 1 - kOffsetWidth;
  static_assert(kIndexWidth + kQidWidth + kOffsetWidth < 32,
                "need to change return value");
  const auto raw_index =
      ap_uint<kIndexWidth + kOffsetWidth>(GetCapOfLevel(level) + idx);
  return ap_uint<kIndexWidth>(
             raw_index.range(kIndexWidth + kOffsetWidth - 1, kOffsetWidth)),
         ap_uint<kQidWidth>(qid), ap_uint<kOffsetWidth>(idx);
}

constexpr int kVertexUniStatCount = 10;

constexpr int kPiHeapStatCount[] = {
    10,  // PiHeapHead
    10,  // PiHeapIndex
};
constexpr int kPiHeapStatTotalCount = kPiHeapStatCount[0] + kPiHeapStatCount[1];
constexpr int kPiHeapStatTaskCount =
    sizeof(kPiHeapStatCount) / sizeof(kPiHeapStatCount[0]);

class TaskOnChip {
 public:
  TaskOnChip() {}

  TaskOnChip(const Task& task) {
    set_vid(task.vid);
    set_vertex(task.vertex);
  }

  operator Task() const { return {.vid = vid(), .vertex = vertex()}; }

  bool operator<=(const TaskOnChip& other) const {
    return Task(*this) <= Task(other);
  }

  Vid vid() const { return data.range(vid_msb, vid_lsb); };
  Vertex vertex() const {
    ap_uint<32> distance = 0;
    distance.range(kFloatMsb, kFloatLsb) =
        data.range(distance_msb, distance_lsb);
    return {
        .parent = Vid(data.range(parent_msb, parent_lsb)),
        .distance = bit_cast<float>(distance.to_uint()),
        .offset = Eid(data.range(offset_msb, offset_lsb)),
        .degree = Vid(data.range(degree_msb, degree_lsb)),
    };
  }

  void set_vid(Vid vid) { data.range(vid_msb, vid_lsb) = vid; }
  void set_value(const Vertex& vertex) {
    set_parent(vertex.parent);
    set_distance(vertex.distance);
  }
  void set_metadata(const Vertex& vertex) {
    set_offset(vertex.offset);
    set_degree(vertex.degree);
  }
  void set_vertex(const Vertex& vertex) {
    set_value(vertex);
    set_metadata(vertex);
  }
  void set_parent(Vid parent) { data.range(parent_msb, parent_lsb) = parent; }
  void set_distance(float distance) {
    data.range(distance_msb, distance_lsb) =
        ap_uint<32>(bit_cast<uint32_t>(distance)).range(kFloatMsb, kFloatLsb);
  }
  void set_offset(Eid offset) { data.range(offset_msb, offset_lsb) = offset; }
  void set_degree(Vid degree) { data.range(degree_msb, degree_lsb) = degree; }

 private:
  static constexpr int vid_lsb = 0;
  static constexpr int vid_msb = vid_lsb + kVidWidth - 1;
  static constexpr int parent_lsb = vid_msb + 1;
  static constexpr int parent_msb = parent_lsb + kVidWidth - 1;
  static constexpr int distance_lsb = parent_msb + 1;
  static constexpr int distance_msb = distance_lsb + kFloatWidth - 1;
  static constexpr int offset_lsb = distance_msb + 1;
  static constexpr int offset_msb = offset_lsb + kEidWidth - 1;
  static constexpr int degree_lsb = offset_msb + 1;
  static constexpr int degree_msb = degree_lsb + kVidWidth - 1;
  static constexpr int length = degree_msb + 1;
  ap_uint<length> data;

  friend struct HeapElemAxi;
};

using HeapElemPacked = ap_uint<256>;

struct HeapElemAxi {
  static constexpr int kCapWidth =
      bit_length(GetChildCapOfLevel(kOnChipLevelCount));
  using Capacity = ap_uint<kCapWidth>;

  bool valid;
  TaskOnChip task;
  Capacity cap[kPiHeapWidth];

  static HeapElemAxi Unpack(const HeapElemPacked& packed) {
    HeapElemAxi elem;
    elem.valid = packed.get_bit(kValidBit);
    elem.task.data = packed.range(kTaskMsb, kTaskLsb);
    for (int i = 0; i < kPiHeapWidth; ++i) {
#pragma HLS unroll
      elem.cap[i] = packed.range(kCapLsb + kCapWidth * (i + 1) - 1,
                                 kCapLsb + kCapWidth * i);
    }
    return elem;
  }

  HeapElemPacked Pack() const {
    HeapElemPacked packed;
    packed.set_bit(kValidBit, valid);
    packed.range(kTaskMsb, kTaskLsb) = task.data;
    for (int i = 0; i < kPiHeapWidth; ++i) {
#pragma HLS unroll
      packed.range(kCapLsb + kCapWidth * (i + 1) - 1, kCapLsb + kCapWidth * i) =
          cap[i];
    }
    return packed;
  }

 private:
  static constexpr int kValidBit = 0;
  static constexpr int kTaskLsb = kValidBit + 1;
  static constexpr int kTaskMsb = kTaskLsb + TaskOnChip::length - 1;
  static constexpr int kCapLsb = kTaskMsb + 1;

  static_assert(kCapLsb + kCapWidth * kPiHeapWidth <= HeapElemPacked::width,
                "HeapElemPacked has insufficient width");
};

class HeapIndexEntry {
 public:
  HeapIndexEntry() {}
  HeapIndexEntry(std::nullptr_t) { invalidate(); }
  HeapIndexEntry(LevelId level, LevelIndex index, float distance) {
    this->set(level, index, distance);
  }

  bool valid() const { return data_.bit(kValidBit); }
  void invalidate() { data_.clear(kValidBit); }

  LevelId level() const {
    CHECK(valid());
    return data_.range(kLevelMsb, kLevelLsb);
  }
  LevelIndex index() const {
    CHECK(valid());
    return data_.range(kIndexMsb, kIndexLsb);
  }
  float distance() const {
    CHECK(valid());
    return bit_cast<float>(
        uint32_t(data_.range(kDistanceMsb, kDistanceLsb) << kFloatLsb));
  }
  void set(LevelId level, LevelIndex index, float distance) {
    CHECK_GE(level, 0);
    CHECK_LT(level, kLevelCount);
    CHECK_GE(index, 0);
    CHECK_LT(index, GetCapOfLevel(level));
    data_.set(kValidBit);
    data_.range(kLevelMsb, kLevelLsb) = level;
    data_.range(kIndexMsb, kIndexLsb) = index;
    data_.range(kDistanceMsb, kDistanceLsb) =
        bit_cast<ap_uint<32>>(distance).range(kFloatMsb, kFloatLsb);
  }

  LevelIndex parent_index_at(LevelId level) const {
    CHECK(valid());
    CHECK_GE(this->level(), level);
    return this->index() / GetCapOfLevel(this->level() - level);
  }

  bool is_descendant_of(LevelId level, LevelIndex index) const {
    return parent_index_at(level) == index;
  };

  bool distance_eq(const HeapIndexEntry& other) const {
    return valid() && data_.range(kDistanceMsb, kDistanceLsb) ==
                          other.data_.range(kDistanceMsb, kDistanceLsb);
  }
  bool distance_le(const HeapIndexEntry& other) const {
    return valid() && other.valid() &&
           data_.range(kDistanceMsb, kDistanceLsb) <=
               other.data_.range(kDistanceMsb, kDistanceLsb);
  }

 private:
  ap_uint<256> data_;
  // bool valid;
  // LevelId level;
  // LevelIndex index;
  // ap_uint<kFloatWidth> distance;
  static constexpr int kValidBit = 0;
  static constexpr int kLevelLsb = kValidBit + 1;
  static constexpr int kLevelMsb = kLevelLsb + LevelId::width - 1;
  static constexpr int kIndexLsb = kLevelMsb + 1;
  static constexpr int kIndexMsb = kIndexLsb + LevelIndex::width - 1;
  static constexpr int kDistanceLsb = kIndexMsb + 1;
  static constexpr int kDistanceMsb = kDistanceLsb + kFloatWidth - 1;
  static_assert(kDistanceMsb < decltype(data_)::width, "invalid configuration");
};

inline std::ostream& operator<<(std::ostream& os, const HeapIndexEntry& obj) {
  if (obj.valid()) {
    return os << obj.level() << "`" << obj.index() << " (" << obj.distance()
              << ")";
  }
  return os << "invalid";
}

// Platform-specific constants and types.
constexpr int kPeCount = 32;

#ifndef TAPA_SSSP_SHARD_COUNT
#define TAPA_SSSP_SHARD_COUNT 2
#endif

constexpr int kShardCount = TAPA_SSSP_SHARD_COUNT;  // #edge partitions.
static_assert(
    kPeCount % kShardCount == 0,
    "current implementation assumes PE count is a multiple of shard count");

constexpr int kIntervalCount = 2;  // #vertex partitions.
constexpr int kSubIntervalCount = kIntervalCount * 2;

constexpr int kHeapOnChipWidth = 8;  // #children per on-heap element.

constexpr int kHeapOffChipWidth = 16;  // #children per off-heap element.

#endif  // TAPA_SSSP_H_

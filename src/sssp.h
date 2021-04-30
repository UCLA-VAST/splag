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

// since all distances are positive and normalized, we can store fewer bits.
// kFloatWidth bits from kFloatMsb is used.
constexpr int kFloatMsb = 30;
constexpr int kFloatWidth = 30;
constexpr int kFloatLsb = kFloatMsb - kFloatWidth + 1;
static_assert(kFloatLsb >= 0, "invalid configuration");
static_assert(kFloatMsb <= 32, "invalid configuration");

static constexpr int kVidWidth = 20;
static constexpr int kEidWidth = 25;

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
constexpr int kOnChipLevelCount = 15;
constexpr int kOffChipLevelCount = 5;
constexpr int kLevelCount = kOnChipLevelCount + kOffChipLevelCount;
using OffChipLevelId = ap_uint<bit_length(kOffChipLevelCount - 1)>;
using LevelId = ap_uint<bit_length(kLevelCount - 1)>;
using LevelIndex = ap_uint<kLevelCount - 1>;

inline int GetAddrOfOffChipHeapElem(int level, int idx, int qid) {
  CHECK_GE(level, 0);
  CHECK_LT(level, kLevelCount);
  CHECK_GE(idx, 0);
  CHECK_LT(idx, 1 << level);
  CHECK_GE(qid, 0);
  CHECK_LT(qid, kQueueCount);
  return ((1 << level) / 2 + idx / 2) * kQueueCount + qid;
}

constexpr int kPiHeapStatCount[] = {
    1,  // PiHeapHead
    5,  // PiHeapIndex
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

struct HeapElemPairAxi : public ap_uint<256> {
  static constexpr int length = 256;
};

struct HeapElemAxi {
  static constexpr int kCapWidth =
      (HeapElemPairAxi::length / 2 - TaskOnChip::length - 1) / 2;
  using Capacity = ap_uint<kCapWidth>;

  bool valid;
  Capacity cap_left;
  Capacity cap_right;
  TaskOnChip task;

  template <int idx>  // Makes sure HLS won't generate complex structure.
  static HeapElemAxi ExtractFromPair(const HeapElemPairAxi& pair) {
    constexpr int offset = idx * (HeapElemPairAxi::length / 2);
    HeapElemAxi elem;
    elem.valid = pair.get_bit(offset);
    elem.cap_left = pair.range(offset + kCapWidth, offset + 1);
    elem.cap_right = pair.range(offset + kCapWidth * 2, offset + kCapWidth + 1);
    elem.task.data = pair.range(offset + kCapWidth * 2 + TaskOnChip::length,
                                offset + kCapWidth * 2 + 1);
    return elem;
  }

  template <int idx>  // Makes sure HLS won't generate complex structure.
  void UpdatePair(HeapElemPairAxi& pair) const {
    constexpr int offset = idx * (HeapElemPairAxi::length / 2);
    pair.set_bit(offset, valid);
    pair.range(offset + kCapWidth, offset + 1) = cap_left;
    pair.range(offset + kCapWidth * 2, offset + kCapWidth + 1) = cap_right;
    pair.range(offset + kCapWidth * 2 + TaskOnChip::length,
               offset + kCapWidth * 2 + 1) = task.data;
  }
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
    CHECK_LT(index, (1 << level));
    data_.set(kValidBit);
    data_.range(kLevelMsb, kLevelLsb) = level;
    data_.range(kIndexMsb, kIndexLsb) = index;
    data_.range(kDistanceMsb, kDistanceLsb) =
        bit_cast<ap_uint<32>>(distance).range(kFloatMsb, kFloatLsb);
  }

  bool is_descendant_of(LevelId level, LevelIndex index) const {
    CHECK(valid());
    CHECK_GE(this->level(), level);
    return this->index() / (1 << (this->level() - level)) == index;
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
  ap_uint<64> data_;
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

constexpr int kShardCount = 2;  // #edge partitions.
static_assert(
    kPeCount % kShardCount == 0,
    "current implementation assumes PE count is a multiple of shard count");

constexpr int kIntervalCount = 2;  // #vertex partitions.

constexpr int kHeapOnChipWidth = 8;  // #children per on-heap element.

constexpr int kHeapOffChipWidth = 16;  // #children per off-heap element.

#endif  // TAPA_SSSP_H_

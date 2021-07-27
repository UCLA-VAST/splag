#ifndef TAPA_SSSP_CGPQ_H_
#define TAPA_SSSP_CGPQ_H_

#include <algorithm>
#include <limits>
#include <queue>
#include "sssp.h"

#include "hls_vector.h"

// #define TAPA_SSSP_2X_BUFFER

// #define TAPA_SSSP_CGPQ_LOG_BUCKET

namespace cgpq {

constexpr int kBucketCount = 64;
constexpr int kChunkSize = 1024;
constexpr int kCgpqLevel = 15;
constexpr int kCgpqCapacity = (1 << kCgpqLevel) - 1;

#ifdef TAPA_SSSP_2X_BUFFER
constexpr int kBufferSize = kChunkSize * 2;
#else
constexpr int kBufferSize = kChunkSize;
#endif  // TAPA_SSSP_2X_BUFFER

constexpr int kChunkPartFac = kBucketCount / (4096 / kBufferSize);

using uint_spill_addr_t = ap_uint<24>;
using int_spill_addr_t = ap_int<uint_spill_addr_t::width + 1>;

using uint_bid_t = ap_uint<bit_length(kBucketCount - 1)>;
using int_bid_t = ap_int<uint_bid_t::width + 1>;

using uint_chunk_size_t = ap_uint<bit_length(kChunkSize)>;

struct PushReq {
  uint_bid_t bid;
  TaskOnChip task;
};

struct PushBuf {
  bool is_valid = false;
  TaskOnChip task;
  uint_bid_t bid;
};

class ChunkMeta {
 public:
  using uint_pos_t = ap_uint<bit_length(kBufferSize - 1)>;
  using uint_size_t = ap_uint<bit_length(kBufferSize)>;

  auto GetSize() const {
    return is_full_ ? uint_size_t(kBufferSize)
                    : uint_size_t(uint_pos_t(write_pos_ - read_pos_));
  }

  auto GetReadPos() const {
    // Do not check for emptiness because read is not destructive.
    return read_pos_;
  }

  auto GetWritePos() const {
    // Do check for fullness because write is destructive if full.
    CHECK(!IsFull());
    return write_pos_;
  }

  bool IsFull() const {
    if (is_full_) {
      CHECK_EQ(write_pos_, read_pos_);
    }
    return is_full_;
  }

  bool IsEmpty() const { return write_pos_ == read_pos_ && !is_full_; }

  bool IsAlmostFull() const {
#ifdef TAPA_SSSP_2X_BUFFER
    return GetSize() >= kBufferSize / 4 * 3;
#else
    return IsFull();
#endif  // TAPA_SSSP_2X_BUFFER
  }
  bool IsAlmostEmpty() const {
#ifdef TAPA_SSSP_2X_BUFFER
    return GetSize() < kBufferSize / 4;
#else
    return IsEmpty();
#endif  // TAPA_SSSP_2X_BUFFER
  }

  void Push() {
    CHECK(!IsFull());
    const auto before = GetSize();
    ++write_pos_;
    is_full_ = write_pos_ == read_pos_;
    const auto after = GetSize();
    CHECK_EQ(before + 1, after);
  }

  void Pop() {
    CHECK(!IsEmpty());
    const auto before = GetSize();
    ++read_pos_;
    is_full_ = false;
    const auto after = GetSize();
    CHECK_EQ(before, after + 1);
  }

 private:
  uint_pos_t read_pos_ = 0;
  uint_pos_t write_pos_ = 0;
  bool is_full_ = false;
};

struct ChunkRef {
  uint_spill_addr_t addr;
  uint_bid_t bucket;

  // Compares priority.
  bool operator<(const ChunkRef& that) const {
    return that.bucket < this->bucket;
  }

  bool operator==(const ChunkRef& that) const {
    return this->addr == that.addr && this->bucket == that.bucket;
  }
};

using ChunkRefPair = hls::vector<ChunkRef, 2>;
inline bool ChunkRefPairEq(const ChunkRefPair& lhs, const ChunkRefPair& rhs) {
  return lhs[0] == rhs[0] && lhs[1] == rhs[1];
}

namespace internal {

template <int begin, int len>
struct Arbiter {
  static void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCount],
                        bool& is_output_valid, int_bid_t& output_bid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        int_bid_t& full_bid, ChunkMeta& full_meta) {
    bool is_output_valid_0, is_output_valid_1, is_full_valid_0, is_full_valid_1;
    int_bid_t output_bid_0, output_bid_1, full_bid_0, full_bid_1;
    ChunkMeta output_meta_0, output_meta_1, full_meta_0, full_meta_1;
    Arbiter<begin, len / 2>::FindChunk(
        chunk_meta, is_output_valid_0, output_bid_0, output_meta_0,
        is_full_valid_0, full_bid_0, full_meta_0);
    Arbiter<begin + len / 2, len - len / 2>::FindChunk(
        chunk_meta, is_output_valid_1, output_bid_1, output_meta_1,
        is_full_valid_1, full_bid_1, full_meta_1);
    is_output_valid = is_output_valid_0 || is_output_valid_1;
    output_bid = is_output_valid_0 ? output_bid_0 : output_bid_1;
    output_meta = is_output_valid_0 ? output_meta_0 : output_meta_1;
    is_full_valid = is_full_valid_0 || is_full_valid_1;
    full_bid = is_full_valid_0 ? full_bid_0 : full_bid_1;
    full_meta = is_full_valid_0 ? full_meta_0 : full_meta_1;
  }
};

template <int begin>
struct Arbiter<begin, 1> {
  static void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCount],
                        bool& is_output_valid, int_bid_t& output_bid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        int_bid_t& full_bid, ChunkMeta& full_meta) {
    output_meta = full_meta = chunk_meta[begin];
    is_output_valid = !output_meta.IsEmpty();
    is_full_valid = full_meta.IsFull();
    output_bid = is_output_valid ? begin : -1;
    full_bid = is_full_valid ? begin : -1;
  }
};

}  // namespace internal

inline void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCount],
                      int_bid_t& output_bid, ChunkMeta& output_meta,
                      int_bid_t& full_bid, ChunkMeta& full_meta) {
#pragma HLS inline recursive
  bool is_output_valid, is_full_valid;
  internal::Arbiter<0, kBucketCount>::FindChunk(
      chunk_meta, is_output_valid, output_bid, output_meta, is_full_valid,
      full_bid, full_meta);
}

}  // namespace cgpq

#endif  // TAPA_SSSP_CGPQ_H_

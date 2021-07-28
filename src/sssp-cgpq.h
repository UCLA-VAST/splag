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

constexpr int kChunkSize = kCgpqChunkSize;

#ifdef TAPA_SSSP_2X_BUFFER
constexpr int kBufferSize = kChunkSize * 2;
#else
constexpr int kBufferSize = kChunkSize;
#endif  // TAPA_SSSP_2X_BUFFER

constexpr int kChunkPartFac = kBucketCount / (4096 / kBufferSize);

using uint_spill_addr_t = ap_uint<24>;

using uint_bid_t = ap_uint<bit_length(kBucketCount - 1)>;

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

  auto GetSize() const { return size_; }

  auto GetReadPos() const { return read_pos_; }

  auto GetWritePos() const { return write_pos_; }

  bool IsFull() const {
    if (is_full_) {
      CHECK_EQ(write_pos_, read_pos_);
      CHECK(!is_empty_);
    }
    return is_full_;
  }

  bool IsEmpty() const {
    if (is_empty_) {
      CHECK_EQ(write_pos_, read_pos_);
      CHECK(!is_full_);
    }
    return is_empty_;
  }

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
    ++write_pos_;
    ++size_;
    is_empty_ = false;
    is_full_ = write_pos_ == read_pos_;
  }

  void Pop() {
    CHECK(!IsEmpty());
    ++read_pos_;
    --size_;
    is_empty_ = write_pos_ == read_pos_;
    is_full_ = false;
  }

 private:
  uint_pos_t read_pos_ = 0;
  uint_pos_t write_pos_ = 0;
  uint_size_t size_ = 0;
  bool is_empty_ = true;
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
                        bool& is_output_valid, uint_bid_t& output_bid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        uint_bid_t& full_bid, ChunkMeta& full_meta) {
    bool is_output_valid_0, is_output_valid_1, is_full_valid_0, is_full_valid_1;
    uint_bid_t output_bid_0, output_bid_1, full_bid_0, full_bid_1;
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

  static void ReadChunk(
      // Inputs.
      const TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize],
      const ap_uint<kChunkPartFac> is_spill, const uint_bid_t spill_bid,
      const ChunkMeta::uint_pos_t spill_pos,
      const ap_uint<kChunkPartFac> is_output, const uint_bid_t output_bid,
      const ChunkMeta::uint_pos_t output_pos,
      // Outputs.
      TaskOnChip& spill_task, TaskOnChip& output_task) {
    Arbiter<begin, len / 2>::ReadChunk(chunk_buf, is_spill, spill_bid,
                                       spill_pos, is_output, output_bid,
                                       output_pos, spill_task, output_task);
    Arbiter<begin + len / 2, len - len / 2>::ReadChunk(
        chunk_buf, is_spill, spill_bid, spill_pos, is_output, output_bid,
        output_pos, spill_task, output_task);
  }

  static void WriteChunk(
      // Inputs.
      const ap_uint<kChunkPartFac> is_refill, const uint_bid_t refill_bid,
      const ChunkMeta::uint_pos_t refill_pos, const TaskOnChip& refill_task,
      const ap_uint<kChunkPartFac> is_input, const uint_bid_t input_bid,
      const ChunkMeta::uint_pos_t input_pos, const TaskOnChip& input_task,
      // Outputs.
      TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize]) {
    Arbiter<begin, len / 2>::WriteChunk(is_refill, refill_bid, refill_pos,
                                        refill_task, is_input, input_bid,
                                        input_pos, input_task, chunk_buf);
    Arbiter<begin + len / 2, len - len / 2>::WriteChunk(
        is_refill, refill_bid, refill_pos, refill_task, is_input, input_bid,
        input_pos, input_task, chunk_buf);
  }
};

template <int begin>
struct Arbiter<begin, 1> {
  static void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCount],
                        bool& is_output_valid, uint_bid_t& output_bid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        uint_bid_t& full_bid, ChunkMeta& full_meta) {
    output_meta = full_meta = chunk_meta[begin];
    is_output_valid = !output_meta.IsEmpty();
    is_full_valid = full_meta.IsFull();
    output_bid = begin;
    full_bid = begin;
  }

  static void ReadChunk(
      // Inputs.
      const TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize],
      const ap_uint<kChunkPartFac> is_spill, const uint_bid_t spill_bid,
      const ChunkMeta::uint_pos_t spill_pos,
      const ap_uint<kChunkPartFac> is_output, const uint_bid_t output_bid,
      const ChunkMeta::uint_pos_t output_pos,
      // Outputs.
      TaskOnChip& spill_task, TaskOnChip& output_task) {
    const auto bid = is_spill.bit(begin) ? spill_bid : output_bid;
    const auto pos = is_spill.bit(begin) ? spill_pos : output_pos;
    const auto task =
        chunk_buf[bid / kChunkPartFac * kChunkPartFac + begin][pos];
    if (is_spill.bit(begin)) {
      spill_task = task;
    }
    if (is_output.bit(begin)) {
      output_task = task;
    }
  }

  static void WriteChunk(
      // Inputs.
      const ap_uint<kChunkPartFac> is_refill, const uint_bid_t refill_bid,
      const ChunkMeta::uint_pos_t refill_pos, const TaskOnChip& refill_task,
      const ap_uint<kChunkPartFac> is_input, const uint_bid_t input_bid,
      const ChunkMeta::uint_pos_t input_pos, const TaskOnChip& input_task,
      // Outputs.
      TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize]) {
    const auto bid = is_refill.bit(begin) ? refill_bid : input_bid;
    const auto pos = is_refill.bit(begin) ? refill_pos : input_pos;
    const auto task = is_refill.bit(begin) ? refill_task : input_task;
    if (is_refill.bit(begin) || is_input.bit(begin)) {
      chunk_buf[bid / kChunkPartFac * kChunkPartFac + begin][pos] = task;
    }
  }
};

}  // namespace internal

inline void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCount],
                      bool& is_output_valid, uint_bid_t& output_bid,
                      ChunkMeta& output_meta, bool& is_full_valid,
                      uint_bid_t& full_bid, ChunkMeta& full_meta) {
#pragma HLS inline recursive
  internal::Arbiter<0, kBucketCount>::FindChunk(
      chunk_meta, is_output_valid, output_bid, output_meta, is_full_valid,
      full_bid, full_meta);
}

inline void ReadChunk(
    // Inputs.
    const TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize],
    const bool is_spill_valid, const uint_bid_t spill_bid,
    const ChunkMeta::uint_pos_t spill_pos, const bool is_output_valid,
    const uint_bid_t output_bid, const ChunkMeta::uint_pos_t output_pos,
    // Outputs.
    TaskOnChip& spill_task, TaskOnChip& output_task) {
#pragma HLS inline recursive
  ap_uint<kChunkPartFac> is_spill = 0, is_output = 0;
  is_spill.bit(spill_bid % kChunkPartFac) = is_spill_valid;
  is_output.bit(output_bid % kChunkPartFac) = is_output_valid;
  internal::Arbiter<0, kChunkPartFac>::ReadChunk(
      chunk_buf, is_spill, spill_bid, spill_pos, is_output, output_bid,
      output_pos, spill_task, output_task);
}

inline void WriteChunk(
    // Inputs.
    const bool can_recv_refill, const uint_bid_t refill_bid,
    const ChunkMeta::uint_pos_t refill_pos, const TaskOnChip& refill_task,
    const bool can_enqueue, const uint_bid_t input_bid,
    const ChunkMeta::uint_pos_t input_pos, const TaskOnChip& input_task,
    // Outputs.
    TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize]) {
#pragma HLS inline recursive
  ap_uint<kChunkPartFac> is_refill = 0, is_input = 0;
  is_refill.bit(refill_bid % kChunkPartFac) = can_recv_refill;
  is_input.bit(input_bid % kChunkPartFac) = can_enqueue;
  internal::Arbiter<0, kChunkPartFac>::WriteChunk(
      is_refill, refill_bid, refill_pos, refill_task, is_input, input_bid,
      input_pos, input_task, chunk_buf);
}

}  // namespace cgpq

#endif  // TAPA_SSSP_CGPQ_H_

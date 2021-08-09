#ifndef TAPA_SSSP_CGPQ_H_
#define TAPA_SSSP_CGPQ_H_

#include <algorithm>
#include <array>

#include <ap_int.h>

#include "sssp-kernel.h"
#include "sssp.h"

#define TAPA_SSSP_2X_BUFFER

#define TAPA_SSSP_CGPQ_LOG_BUCKET

namespace cgpq {

constexpr int kBucketCount = 64;

constexpr int kChunkSize = kCgpqChunkSize;

constexpr int kVecCountPerChunk = kChunkSize / kSpilledTaskVecLen;

#ifdef TAPA_SSSP_2X_BUFFER
constexpr int kBufferSize = kChunkSize * 2;
#else
constexpr int kBufferSize = kChunkSize;
#endif  // TAPA_SSSP_2X_BUFFER

constexpr int kPosPartFac = kSpilledTaskVecLen;

constexpr int kBucketPartFac = 4;

using uint_bid_t = ap_uint<bit_length(kBucketCount - 1)>;

using uint_chunk_size_t = ap_uint<bit_length(kChunkSize)>;

using uint_heap_size_t = ap_uint<bit_length(kCgpqCapacity + 1)>;

using uint_heap_pos_t = ap_uint<bit_length(kCgpqCapacity)>;

using uint_heap_pair_pos_t = ap_uint<bit_length(kCgpqCapacity / 2)>;

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
  using uint_delta_t = ap_uint<bit_length(kSpilledTaskVecLen)>;

  auto GetSize() const { return size_; }

  auto GetFreeSize() const { return free_size_; }

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

  void Push(uint_delta_t n, int bid) {
    CHECK(!IsFull());
    CHECK_GE(free_size_, n);

    write_pos_ += n;
    size_ += n;
    free_size_ -= n;
    is_empty_ = false;
    is_full_ = write_pos_ == read_pos_;

    CHECK_EQ(size_ + free_size_, kBufferSize);
    VLOG(5) << std::setfill(' ') << "push[" << std::setw(2) << bid
            << "]: " << std::setw(4) << size_ - n << " -> " << std::setw(4)
            << size_;
  }

  void Pop(int bid) {
    CHECK(!IsEmpty());
    CHECK_GE(size_, kSpilledTaskVecLen);
    CHECK_EQ(read_pos_ % kSpilledTaskVecLen, 0);

    read_pos_ += kSpilledTaskVecLen;
    size_ -= kSpilledTaskVecLen;
    free_size_ += kSpilledTaskVecLen;
    is_empty_ = write_pos_ == read_pos_;
    is_full_ = false;

    CHECK_EQ(size_ + free_size_, kBufferSize);
    VLOG(5) << std::setfill(' ') << "pop [" << std::setw(2) << bid
            << "]: " << std::setw(4) << size_ + kSpilledTaskVecLen << " -> "
            << std::setw(4) << size_;
  }

 private:
  uint_pos_t read_pos_ = 0;
  uint_pos_t write_pos_ = 0;
  uint_size_t size_ = 0;
  uint_size_t free_size_ = kBufferSize;
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

using ChunkRefPair = std::array<ChunkRef, 2>;

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
      const ap_uint<kBucketPartFac> is_spill, const uint_bid_t spill_bid,
      const ChunkMeta::uint_pos_t spill_pos,
      const ap_uint<kBucketPartFac> is_output, const uint_bid_t output_bid,
      const ChunkMeta::uint_pos_t output_pos,
      // Outputs.
      bool& is_spill_written, SpilledTask& spill_task, bool& is_output_written,
      SpilledTask& output_task) {
    bool is_spill_written_0, is_spill_written_1, is_output_written_0,
        is_output_written_1;
    SpilledTask spill_task_0, spill_task_1, output_task_0, output_task_1;
    Arbiter<begin, len / 2>::ReadChunk(
        chunk_buf, is_spill, spill_bid, spill_pos, is_output, output_bid,
        output_pos, is_spill_written_0, spill_task_0, is_output_written_0,
        output_task_0);
    Arbiter<begin + len / 2, len - len / 2>::ReadChunk(
        chunk_buf, is_spill, spill_bid, spill_pos, is_output, output_bid,
        output_pos, is_spill_written_1, spill_task_1, is_output_written_1,
        output_task_1);
    is_spill_written = is_spill_written_0 || is_spill_written_1;
    spill_task = is_spill_written_0 ? spill_task_0 : spill_task_1;
    is_output_written = is_output_written_0 || is_output_written_1;
    output_task = is_output_written_0 ? output_task_0 : output_task_1;
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
    is_full_valid = full_meta.IsAlmostFull();
    output_bid = begin;
    full_bid = begin;
  }

  static void ReadChunk(
      // Inputs.
      const TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize],
      const ap_uint<kBucketPartFac> is_spill, const uint_bid_t spill_bid,
      const ChunkMeta::uint_pos_t spill_pos,
      const ap_uint<kBucketPartFac> is_output, const uint_bid_t output_bid,
      const ChunkMeta::uint_pos_t output_pos,
      // Outputs.
      bool& is_spill_written, SpilledTask& spill_task, bool& is_output_written,
      SpilledTask& output_task) {
    is_spill_written = is_spill.bit(begin);
    is_output_written = is_output.bit(begin);
    const auto bid = is_spill_written ? spill_bid : output_bid;
    const auto pos = is_spill_written ? spill_pos : output_pos;

    // Read chunk_buf[bid][pos : pos+kPosPartFac).
    RANGE(j, kPosPartFac, {
      spill_task[j] = output_task[j] =
          chunk_buf[assume_mod(bid, kBucketPartFac, begin)][assume_mod(
              ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)), kPosPartFac,
              j)];
    });
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
    SpilledTask& spill_task, SpilledTask& output_task) {
#pragma HLS inline recursive
  ap_uint<kBucketPartFac> is_spill = 0, is_output = 0;
  is_spill.bit(spill_bid % kBucketPartFac) = is_spill_valid;
  is_output.bit(output_bid % kBucketPartFac) = is_output_valid;
  bool is_spill_written, is_output_written;
  internal::Arbiter<0, kBucketPartFac>::ReadChunk(
      chunk_buf, is_spill, spill_bid, spill_pos, is_output, output_bid,
      output_pos, is_spill_written, spill_task, is_output_written, output_task);
}

inline void WriteChunk(
    // Inputs.
    const bool can_recv_refill, const uint_bid_t refill_bid,
    const ChunkMeta::uint_pos_t refill_pos, const SpilledTask& refill_task,
    const bool can_enqueue, const uint_bid_t input_bid,
    const ChunkMeta::uint_pos_t input_pos, const TaskOnChip& input_task,
    // Outputs.
    TaskOnChip (&chunk_buf)[kBucketCount][kBufferSize]) {
#pragma HLS inline recursive
  ap_uint<kBucketPartFac> is_refill = 0, is_input = 0;
  is_refill.bit(refill_bid % kBucketPartFac) = can_recv_refill;
  is_input.bit(input_bid % kBucketPartFac) = can_enqueue;
  for (int i = 0; i < kBucketPartFac; ++i) {
#pragma HLS unroll
    const auto bid = is_refill.bit(i) ? refill_bid : input_bid;
    const auto pos = is_refill.bit(i) ? refill_pos : input_pos;

    auto tasks = refill_task;
    if (is_input.bit(i)) {
      tasks[pos % kPosPartFac] = input_task;
    }

    ap_uint<kPosPartFac> is_written = is_refill.bit(i) ? -1 : 0;
    is_written.bit(pos % kPosPartFac) = is_refill.bit(i) || is_input.bit(i);

    RANGE(j, kPosPartFac, {
      if (is_written.bit(j)) {
        chunk_buf[assert_mod(bid, kBucketPartFac, i)]
                 [assume_mod(ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)),
                             kPosPartFac, j)] = tasks[j];
      }
    });
  }
}

}  // namespace cgpq

struct CgpqHeapReq {
  bool is_push;
  cgpq::ChunkRef elem;
};

#endif  // TAPA_SSSP_CGPQ_H_

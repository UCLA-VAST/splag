#ifndef TAPA_SSSP_CGPQ_H_
#define TAPA_SSSP_CGPQ_H_

#include <algorithm>
#include <array>

#include <ap_int.h>

#include "sssp-kernel.h"
#include "sssp.h"

#define TAPA_SSSP_2X_BUFFER

namespace cgpq {

constexpr int kBucketCount = 128;

constexpr int kBucketCountPerBank = kBucketCount / kCgpqPushPortCount;

constexpr int kChunkSize = kCgpqChunkSize;

constexpr int kVecCountPerChunk = kChunkSize / kSpilledTaskVecLen;

#ifdef TAPA_SSSP_2X_BUFFER
constexpr int kBufferSize = kChunkSize * 2;
#else
constexpr int kBufferSize = kChunkSize;
#endif  // TAPA_SSSP_2X_BUFFER

constexpr int kPosPartFac = kSpilledTaskVecLen;

constexpr int kBucketPartFac = kCgpqPushPortCount;

using uint_bank_t = ap_uint<bit_length(kCgpqPushPortCount - 1)>;

using uint_bid_t = ap_uint<bit_length(kBucketCount - 1)>;

using uint_chunk_size_t = ap_uint<bit_length(kChunkSize)>;

using uint_heap_size_t = ap_uint<bit_length(kCgpqCapacity + 1)>;

using uint_heap_pos_t = ap_uint<bit_length(kCgpqCapacity)>;

using uint_heap_pair_pos_t = ap_uint<bit_length(kCgpqCapacity / 2)>;

struct uint_bbid_t : public ap_uint<bit_length(kBucketCountPerBank - 1)> {
  using ap_uint<bit_length(kBucketCountPerBank - 1)>::ap_uint;
  uint_bid_t bid(uint_bank_t bank) const { return *this, bank; }
};

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
  ~ChunkMeta() {
#pragma HLS aggregate variable = this bit
  }

  using uint_pos_t = ap_uint<bit_length(kBufferSize - 1)>;
  using uint_size_t = ap_uint<bit_length(kBufferSize)>;
  using uint_delta_t = ap_uint<bit_length(kSpilledTaskVecLen)>;
  using int_delta_t = ap_int<uint_delta_t::width + 1>;

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
    return is_almost_full_;
#else
    return IsFull();
#endif  // TAPA_SSSP_2X_BUFFER
  }
  bool IsAlmostEmpty() const {
#ifdef TAPA_SSSP_2X_BUFFER
    return is_almost_empty_;
#else
    return IsEmpty();
#endif  // TAPA_SSSP_2X_BUFFER
  }

  void Update(uint_delta_t push, uint_delta_t pop, int_delta_t delta, int bid) {
    if (push) {
      CHECK(!IsFull());
      CHECK_GE(free_size_, push);
    }
    if (pop) {
      CHECK(!IsEmpty());
      CHECK_GE(size_, pop - push);
      CHECK_EQ(pop, kSpilledTaskVecLen);
    }
    CHECK_EQ(read_pos_ % kSpilledTaskVecLen, 0);

    read_pos_ += pop;
    write_pos_ += push;
    CHECK_EQ(delta, push - pop);
    size_ += delta;
    free_size_ -= delta;
    is_empty_ = size_ == 0;
    is_full_ = free_size_ == 0;
    UpdateAlmostBits();

    CHECK_EQ(size_ + free_size_, kBufferSize);
    VLOG_IF(5, push) << std::setfill(' ') << "push[" << std::setw(2) << bid
                     << "]: " << std::setw(4) << size_ - push << " -> "
                     << std::setw(4) << size_;
    VLOG_IF(5, pop) << std::setfill(' ') << "pop [" << std::setw(2) << bid
                    << "]: " << std::setw(4) << size_ + kSpilledTaskVecLen
                    << " -> " << std::setw(4) << size_;
  }

 private:
  void UpdateAlmostBits() {
    is_almost_empty_ = size_ < kBufferSize / 4;
    is_almost_full_ = free_size_ < kBufferSize / 4;
  }
  uint_pos_t read_pos_ = 0;
  uint_pos_t write_pos_ = 0;
  uint_size_t size_ = 0;
  uint_size_t free_size_ = kBufferSize;
  bool is_empty_ = true;
  bool is_full_ = false;
  bool is_almost_empty_ = true;
  bool is_almost_full_ = false;
};

struct ChunkRef {
  uint_spill_addr_t addr;
  uint_bbid_t bbid;

  // Compares priority.
  bool operator<(const ChunkRef& that) const { return that.bbid < this->bbid; }

  bool operator==(const ChunkRef& that) const {
    return this->addr == that.addr && this->bbid == that.bbid;
  }
};

using ChunkRefPair = std::array<ChunkRef, 2>;

namespace internal {

template <int begin, int len>
struct Arbiter {
  static void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCountPerBank],
                        bool& is_output_valid, uint_bbid_t& output_bbid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        uint_bbid_t& full_bbid, ChunkMeta& full_meta) {
    bool is_output_valid_0, is_output_valid_1, is_full_valid_0, is_full_valid_1;
    uint_bbid_t output_bbid_0, output_bbid_1, full_bbid_0, full_bbid_1;
    ChunkMeta output_meta_0, output_meta_1, full_meta_0, full_meta_1;
    Arbiter<begin, len / 2>::FindChunk(
        chunk_meta, is_output_valid_0, output_bbid_0, output_meta_0,
        is_full_valid_0, full_bbid_0, full_meta_0);
    Arbiter<begin + len / 2, len - len / 2>::FindChunk(
        chunk_meta, is_output_valid_1, output_bbid_1, output_meta_1,
        is_full_valid_1, full_bbid_1, full_meta_1);
    is_output_valid = is_output_valid_0 || is_output_valid_1;
    output_bbid = is_output_valid_0 ? output_bbid_0 : output_bbid_1;
    output_meta = is_output_valid_0 ? output_meta_0 : output_meta_1;
    is_full_valid = is_full_valid_0 || is_full_valid_1;
    full_bbid = is_full_valid_1 ? full_bbid_1 : full_bbid_0;
    full_meta = is_full_valid_1 ? full_meta_1 : full_meta_0;
  }
};

template <int begin>
struct Arbiter<begin, 1> {
  static void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCountPerBank],
                        bool& is_output_valid, uint_bbid_t& output_bid,
                        ChunkMeta& output_meta, bool& is_full_valid,
                        uint_bbid_t& full_bid, ChunkMeta& full_meta) {
    output_meta = full_meta = chunk_meta[begin];
    is_output_valid = !output_meta.IsEmpty();
    is_full_valid = full_meta.IsAlmostFull();
    output_bid = begin;
    full_bid = begin;
  }
};

}  // namespace internal

inline void FindChunk(const ChunkMeta (&chunk_meta)[kBucketCountPerBank],
                      bool& is_output_valid, uint_bbid_t& output_bbid,
                      ChunkMeta& output_meta, bool& is_full_valid,
                      uint_bbid_t& full_bbid, ChunkMeta& full_meta) {
#pragma HLS inline recursive
  internal::Arbiter<0, kBucketCountPerBank>::FindChunk(
      chunk_meta, is_output_valid, output_bbid, output_meta, is_full_valid,
      full_bbid, full_meta);
}

}  // namespace cgpq

struct CgpqHeapReq {
  bool is_push;
  cgpq::ChunkRef elem;
};

#endif  // TAPA_SSSP_CGPQ_H_

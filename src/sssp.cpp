#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-cgpq.h"
#include "sssp-kernel.h"
#include "sssp.h"

using tapa::async_mmap;
using tapa::detach;
using tapa::istream;
using tapa::istreams;
using tapa::join;
using tapa::mmap;
using tapa::ostream;
using tapa::ostreams;
using tapa::packet;
using tapa::seq;
using tapa::stream;
using tapa::streams;
using tapa::task;

using cgpq::PushReq;

void CgpqSpillMem(
    //
    istream<uint_spill_addr_t>& read_addr_q,
    ostream<SpilledTaskPerMem>& read_data_q,
    istream<packet<uint_spill_addr_t, SpilledTaskPerMem>>& write_req_q,
    ostream<bool>& write_resp_q,
    //
    tapa::async_mmap<SpilledTaskPerMem> mem) {
  ReadWriteMem(read_addr_q, read_data_q, write_req_q, write_resp_q, mem);
}

void CgpqBucketGen(bool is_log_bucket, float min_distance, float max_distance,
                   istream<bool>& done_q, istream<TaskOnChip>& in_q,
                   ostream<PushReq>& out_q) {
  using namespace cgpq;

  const auto norm =
      kBucketCount / (is_log_bucket
                          ? (std::log(max_distance) - std::log(min_distance))
                          : (max_distance - min_distance));

spin:
  for (ap_uint<log2(kCgpqPushPortCount)> port = 0; done_q.empty(); ++port) {
#pragma HLS pipeline II = 1
    if (!in_q.empty()) {
      const auto task = in_q.read(nullptr);
      const uint_bid_t bid = std::max(
          std::min(
              int((is_log_bucket ? (std::log(task.vertex().distance) -
                                    std::log(min_distance))
                                 : (task.vertex().distance - min_distance)) *
                  norm),
              kBucketCount - 1),
          0);

      out_q.write({.bid = bid == kBucketCount - 1
                              ? uint_bid_t(bid / kCgpqPushPortCount *
                                               kCgpqPushPortCount +
                                           port)
                              : bid,
                   .task = task});
    }
  }

  done_q.read(nullptr);
}

// On-chip priority queue.
// Using a binary heap for fewer number of memory banks.
// Latency of each request << latency of memory for each chunk.
void CgpqHeap(istream<CgpqHeapReq>& req_q, ostream<cgpq::ChunkRef>& resp_q) {
#pragma HLS inline recursive

  using namespace cgpq;

  // Heap rule: !(parent < children) (comparing priority).
  // Root is at pos = 1.
  // Parent of pos = pos / 2.
  // Children of pos = pos * 2, pos * 2 + 1.
  ChunkRefPair heap_array[(kCgpqCapacity + 1) / 2];
#pragma HLS bind_storage variable = heap_array type = RAM_S2P impl = URAM
#pragma HLS aggregate variable = heap_array bit
  // Extra copy of heap top in registers due to limited read ports.
  ChunkRef heap_root;
  // Heap position that should be accessed;
  uint_heap_pos_t heap_pos;
  // Current heap size.
  uint_heap_size_t heap_size = 0;
  // Heap element that is being send up or down.
  ChunkRef heap_elem;
  // Heap position that should be read.
  uint_heap_pair_pos_t heap_read_pos = 0;
  // Whether heap was written in the previous iteration.
  bool is_heap_written = false;
  // Heap position that was written in the previous iteration.
  uint_heap_pair_pos_t heap_write_pos;
  // Heap pair that was read in the previous iteration.
  ChunkRefPair heap_pair_prev;
#pragma HLS aggregate variable = heap_pair_prev[0] bit
#pragma HLS aggregate variable = heap_pair_prev[1] bit
  bool is_heapifying_up = false;
  bool is_heapifying_up_init = false;
  bool is_heapifying_down = false;
  bool is_heapifying_down_init = false;

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    bool is_heapifying_up_next = is_heapifying_up;
    bool is_heapifying_up_init_next = is_heapifying_up_init;
    bool is_heapifying_down_next = is_heapifying_down;
    bool is_heapifying_down_init_next = is_heapifying_down_init;

#pragma HLS dependence variable = heap_array inter false

    // If write, this should be written.
    auto heap_pair = heap_pair_prev;
#pragma HLS aggregate variable = heap_pair[0] bit
#pragma HLS aggregate variable = heap_pair[1] bit

    // Heap pair that is read from heap array.
    heap_pair_prev = heap_array[heap_read_pos];
    if ((is_heapifying_up || is_heapifying_down) && is_heap_written) {
      CHECK_NE(heap_write_pos, heap_read_pos);
    }

    // Do modify states here.
    if (is_heapifying_up) {
      if (is_heapifying_up_init) {
        is_heap_written = false;
      } else {
        const auto heap_pos_next = heap_pos / 2;
        const auto heap_elem_next = heap_pair_prev[heap_pos_next % 2];

        heap_write_pos = heap_pos / 2;
        is_heap_written = true;

#if defined(__has_feature)
#if !__has_feature(memory_sanitizer)
        CHECK(heap_array[heap_pos / 2] == heap_pair);
#endif
#endif
        if (heap_pos > 1 && heap_elem_next < heap_elem) {
          heap_pair[heap_pos % 2] = heap_elem_next;
          if (heap_pos == 1) {
            heap_root = heap_elem_next;
          }
          heap_pos = heap_pos_next;
        } else {
          heap_pair[heap_pos % 2] = heap_elem;
          if (heap_pos == 1) {
            heap_root = heap_elem;
          }
          is_heapifying_up_next = false;
          CHECK(!resp_q.full());  // 1 request at a time.
          resp_q.try_write(heap_root);
        }
      }
      is_heapifying_up_init_next = false;
      heap_read_pos = heap_pos / 4;  // Read parent of heap_pos.
    } else if (is_heapifying_down) {
      if (is_heapifying_down_init) {
        heap_elem = heap_pair_prev[(heap_size + 1) % 2];

        heap_pair_prev[1] = heap_root;
        heap_pos = 1;
        is_heap_written = false;
      } else {
        CHECK(!is_heapifying_up);
        const auto heap_pos_next_left = heap_pos * 2;
        const auto heap_pos_next_right = heap_pos * 2 + 1;
        const bool is_left_valid = heap_pos_next_left < heap_size + 1;
        const bool is_right_valid = heap_pos_next_right < heap_size + 1;

        const auto heap_elem_next_left = heap_pair_prev[0];
        const auto heap_elem_next_right = heap_pair_prev[1];
        const bool is_left_update_needed =
            is_left_valid && heap_elem < heap_elem_next_left;
        const bool is_right_update_needed =
            is_right_valid && heap_elem < heap_elem_next_right;

        CHECK(heap_array[heap_pos / 2][1] == heap_pair[1]);
        if (heap_pos / 2 != 0) {
          CHECK(heap_array[heap_pos / 2][0] == heap_pair[0]);
        }

        heap_write_pos = heap_pos / 2;
        is_heap_written = true;

        if (is_left_update_needed || is_right_update_needed) {
          if (is_right_valid && heap_elem_next_left < heap_elem_next_right) {
            heap_pair[heap_pos % 2] = heap_elem_next_right;
            if (heap_pos == 1) {
              heap_root = heap_elem_next_right;
            }
            heap_pos = heap_pos_next_right;
          } else {
            heap_pair[heap_pos % 2] = heap_elem_next_left;
            if (heap_pos == 1) {
              heap_root = heap_elem_next_left;
            }
            heap_pos = heap_pos_next_left;
          }
        } else {
          heap_pair[heap_pos % 2] = heap_elem;
          if (heap_pos == 1) {
            heap_root = heap_elem;
          }
          is_heapifying_down_next = false;
          CHECK(!resp_q.full());  // 1 request at a time.
          resp_q.try_write(heap_root);
        }
      }
      is_heapifying_down_init_next = false;
      heap_read_pos = heap_pos;  // Read children of heap_pos;
    } else {
      is_heap_written = false;
    }

    if (is_heap_written) {
      heap_array[heap_write_pos] = heap_pair;
    }

    if (!is_heapifying_up && !is_heapifying_down && !req_q.empty()) {
      const auto req = req_q.read(nullptr);
      if (req.is_push) {
        heap_elem = req.elem;
        ++heap_size;
        heap_pos = heap_size;
        heap_read_pos = heap_size / 2;  // Read last element.
        is_heapifying_up_next = is_heapifying_up_init_next = true;
      } else {
        heap_read_pos = heap_size / 2;  // Read last element.
        is_heapifying_down_next = is_heapifying_down_init_next = true;
        CHECK_GT(heap_size, 0);
        --heap_size;
      }
    }

    is_heapifying_up = is_heapifying_up_next;
    is_heapifying_up_init = is_heapifying_up_init_next;
    is_heapifying_down = is_heapifying_down_next;
    is_heapifying_down_init = is_heapifying_down_init_next;
  }
}

void CgpqMinBucketFinder(
    istreams<cgpq::uint_bid_t, kCgpqPushPortCount>& bid_in_q,
    ostreams<cgpq::uint_bid_t, kCgpqPushPortCount>& bid_out_q) {
  using namespace cgpq;

  DECL_ARRAY(uint_bid_t, bid, kCgpqPushPortCount, kBucketCount - 1);
  uint_bid_t prev_min_bid = kBucketCount - 1;

spin:
  for (;;) {
    RANGE(bank, kCgpqPushPortCount, bid_in_q[bank].try_read(bid[bank]));
    const auto min_bid = find_min(bid);
    RANGE(bank, kCgpqPushPortCount, bid_out_q[bank].try_write(min_bid));
    VLOG_IF(4, prev_min_bid != min_bid)
        << "min bid: " << prev_min_bid << " -> " << min_bid;
    prev_min_bid = min_bid;
  }
}

// Coarse-grained priority queue.
// Implements chunks of buckets.
void CgpqCore(
    //
    istream<bool>& done_q, ostream<int32_t>& stat_q,
    // Scalar
    cgpq::uint_bank_t bank,
    //
    ostream<cgpq::uint_bid_t>& bid_out_q, istream<cgpq::uint_bid_t>& bid_in_q,
    // Queue requests.
    istream<PushReq>& push_req_q,
    // Queue outputs.
    ostream<SpilledTask>& pop_q,
    //
    ostream<CgpqHeapReq>& heap_req_q, istream<cgpq::ChunkRef>& heap_resp_q,
    //
    ostream<uint_spill_addr_t>& mem_read_addr_q,
    istream<SpilledTask>& mem_read_data_q,
    ostream<packet<uint_spill_addr_t, SpilledTask>>& mem_write_req_q,
    istream<bool>& mem_write_resp_q) {
#pragma HLS inline recursive

  using namespace cgpq;

  // Whether the external priority queue is busy.
  bool is_heap_available = true;
  // The current top of heap, valid only if heap is not busy.
  ChunkRef heap_root{.addr = 0, .bbid = 0};
  // Current heap size, kept track both in the heap and here as perf counter.
  uint_heap_size_t heap_size = 0;
  // Maximum heap size in this execution (as perf counter).
  uint_heap_size_t max_heap_size = 0;

  DECL_ARRAY(ChunkMeta, chunk_meta, kBucketCountPerBank, ChunkMeta());

  TaskOnChip chunk_buf[kBucketCountPerBank][kBufferSize];
#pragma HLS bind_storage variable = chunk_buf type = RAM_S2P impl = BRAM
#pragma HLS array_partition variable = chunk_buf cyclic factor = \
    kPosPartFac dim = 2

  const uint_spill_addr_t spill_addr_base = (1 << uint_spill_addr_t::width) /
                                            kCgpqBankCountPerMem *
                                            (bank % kCgpqBankCountPerMem);

  bool is_spill_valid = false;
  uint_bbid_t spill_bbid = 0;
  uint_spill_addr_t spill_addr_req = spill_addr_base;
  uint_spill_addr_t spill_addr_resp = spill_addr_base;
  ChunkMeta::uint_size_t task_to_spill_count = kChunkSize;

  // Refill requests should read from this address.
  bool is_refill_addr_valid = false;
  uint_spill_addr_t refill_addr;
  // Refill data should write to this bucket.
  bool is_refill_valid = false;
  uint_bbid_t refill_bbid = 0;
  // Future refill data should write to this bucket.
  bool is_refill_next_valid = false;
  uint_bbid_t refill_bbid_next = 0;
  // Number of data remaining to refill.
  uint_chunk_size_t refill_remain_count;

  bool is_started = false;
  int32_t cycle_count = 0;

  // Cannot enqueue because buffer is full.
  int32_t enqueue_full_count = 0;

  // Cannot enqueue because chunk is refilling and space is insufficient.
  int32_t enqueue_current_refill_count = 0;

  // Cannot enqueue because chunk is scheduled for refilling.
  int32_t enqueue_future_refill_count = 0;

  // Cannot enqueue due to bank conflict with refilling.
  int32_t enqueue_bank_conflict_count = 0;

  // Cannot dequeue because output FIFO is full.
  int32_t dequeue_full_count = 0;

  // Cannot dequeue because chunk is spilling and element is insufficient.
  int32_t dequeue_spilling_count = 0;

  // Cannot dequeue due to bank conflict with spilling.
  int32_t dequeue_bank_conflict_count = 0;

  // Cannot dequeue due to alignment, which means the dequeue bucket is being
  // refilled or enqueued at the same piece of aligned memory space.
  int32_t dequeue_alignment_count = 0;

  uint_bid_t min_bid = 0;

spin:
  for (; done_q.empty();) {
#pragma HLS pipeline II = 1

    bool is_active = false;  // Log an empty line only if active.

    bool is_input_valid;
    const auto push_req = push_req_q.peek(is_input_valid);
    if (is_input_valid) {
      CHECK_EQ(push_req.bid % kCgpqPushPortCount, bank);
    }
    const uint_bbid_t input_bbid = div<kCgpqPushPortCount>(push_req.bid);
    const auto input_task = push_req.task;
    const auto input_meta = chunk_meta[input_bbid];

    bool is_refill_task_valid;
    const auto refill_task = mem_read_data_q.peek(is_refill_task_valid);

    const auto spill_meta = chunk_meta[spill_bbid];
    const auto refill_meta = chunk_meta[refill_bbid];

    bool is_output_valid;
    uint_bbid_t output_bbid;
    ChunkMeta output_meta;
    bool is_full_valid;
    uint_bbid_t full_bbid;
    ChunkMeta full_meta;
    FindChunk(chunk_meta, is_output_valid, output_bbid, output_meta,
              is_full_valid, full_bbid, full_meta);

    const bool is_top_valid = heap_size != 0;
    const uint_bbid_t top_bbid = heap_root.bbid;

    // Read refill data and push into chunk buffer if
    const bool can_recv_refill =
        is_refill_valid &&    // there is active refilling, and
        is_refill_task_valid  // refill data is available for read.
        ;

    const bool is_input_blocked_by_full_buffer = input_meta.IsFull();
    const bool is_input_blocked_by_current_refill =
        is_refill_valid && input_bbid == refill_bbid &&
        input_meta.GetFreeSize() <= refill_remain_count;
    const bool is_input_blocked_by_future_refill =
        is_refill_next_valid && input_bbid == refill_bbid_next &&
        input_meta.GetFreeSize() <= kChunkSize;
    const bool is_input_blocked_by_bank_conflict = can_recv_refill;

    // Read input task and push into chunk buffer if
    const bool can_enqueue =
        // there is an input task, and
        is_input_valid &&

        // chunk is not already full, and
        !is_input_blocked_by_full_buffer &&

        // if chunk is refilling, available space must be greater than #tasks
        // to refill, and
        !is_input_blocked_by_current_refill &&

        // if chunk will refill soon, available space must be greater than
        // #tasks to refill, and
        !is_input_blocked_by_future_refill &&

        // if reading refill data, the input bucket must access a different
        // bank as the refill bucket.
        !is_input_blocked_by_bank_conflict;

    if (is_input_valid) {
      if (is_input_blocked_by_full_buffer) {
        ++enqueue_full_count;
      }
      if (is_input_blocked_by_current_refill) {
        ++enqueue_current_refill_count;
      }
      if (is_input_blocked_by_future_refill) {
        ++enqueue_future_refill_count;
      }
      if (is_input_blocked_by_bank_conflict) {
        ++enqueue_bank_conflict_count;
      }
    }

    // Pop from chunk buffer and write spill data if
    const bool can_send_spill =
        is_spill_valid &&         // there is active spilling, and
        !mem_write_req_q.full();  // spill data is available for write.

    const bool is_output_blocked_by_full_fifo = pop_q.full();
    const bool is_output_blocked_by_bank_conflict = is_spill_valid;
    // Note: to avoid chunk_buf read address being dependent on FIFO fullness,
    // can_dequeue must not depend on the fullness of mem_write_req_q.

    const bool is_output_blocked_by_spilling =
        is_spill_valid && output_meta.GetSize() / kPosPartFac <=
                              task_to_spill_count / kPosPartFac;

    bool is_output_bid_same_as_input = false;
    RANGE(i, kCgpqPushPortCount, is_output_bid_same_as_input |= can_enqueue);
    const bool is_output_unaligned = output_meta.GetSize() < kSpilledTaskVecLen;
    const bool is_output_blocked_by_alignment =
        is_output_unaligned && (can_recv_refill || is_output_bid_same_as_input);

    // Pop from highest-priority chunk buffer and write output data if
    const bool can_dequeue =
        is_output_valid &&  // there is a non-empty chunk, and

        output_bbid.bid(bank) == min_bid &&  // output bid is the minimum, and

        // output is available for write, and
        !is_output_blocked_by_full_fifo &&

        // if chunk is spilling, available elements must be greater than #tasks
        // to spill, and
        !is_output_blocked_by_spilling &&

        // if there is active spilling, the output bucket and the spill bucket
        // must operate on different banks, and
        !is_output_blocked_by_bank_conflict &&

        // if available output count is less than vector length, must not refill
        // or enqueue the same bucket, and
        !is_output_blocked_by_alignment &&

        // on-chip chunk has higher priority.
        (!is_top_valid || output_bbid <= top_bbid);

    if (is_output_valid) {
      is_output_blocked_by_full_fifo && ++dequeue_full_count;
      is_output_blocked_by_spilling && ++dequeue_spilling_count;
      is_output_blocked_by_bank_conflict && ++dequeue_bank_conflict_count;
      is_output_blocked_by_alignment && ++dequeue_alignment_count;
    }

    // Start spilling a new chunk if
    const bool can_start_spill =
        !is_spill_valid &&  // there is no active spilling, and
        is_full_valid &&    // chunks need spilling, and
        !can_dequeue &&     // chunk won't pop, and
        is_heap_available   // heap is available.
        ;

    const bool is_top_bid_same_as_input = can_enqueue && top_bbid == input_bbid;

    // Start refilling a new chunk if
    const bool can_schedule_refill =
        !is_refill_addr_valid &&  // there is no active refilling, and
        !is_refill_next_valid &&  // no future refilling is scheduled, and
        is_top_valid &&           // there is a chunk for refilling, and
        (!is_refill_valid ||
         top_bbid != refill_bbid) &&  // the chunk is not refilling now, and
        (!is_output_valid ||          // either on-chip chunk is empty, or
         output_bbid >= top_bbid      // off-chip chunk has higher priority,
         ) &&                         // and
        chunk_meta[top_bbid].IsAlmostEmpty() &&  // chunk is almost empty,
        !is_top_bid_same_as_input &&             // chunk won't push, and
        is_heap_available &&                     // heap is available, and
        !can_start_spill  // heap won't be occupied by spilling.
        ;

    // Send refill data read address if
    const bool can_send_refill =
        is_refill_addr_valid &&  // there is an active refill request, and
        refill_addr <= spill_addr_resp &&  // writes have finished, and
        !mem_read_addr_q.full()            // address output is not full.
        ;

    SpilledTask spill_task, output_task;
    {
      const auto bbid = is_spill_valid ? spill_bbid : output_bbid;
      const auto pos =
          is_spill_valid ? spill_meta.GetReadPos() : output_meta.GetReadPos();

      // Read chunk_buf[bid][pos : pos+kPosPartFac).
      RANGE(j, kPosPartFac, {
        spill_task[j] = output_task[j] = chunk_buf[bbid][assume_mod(
            ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)), kPosPartFac,
            j)];

        VLOG_IF(5, is_spill_valid || is_output_valid)
            << "chunk_buf[" << bbid * kCgpqPushPortCount + bank << "]["
            << assume_mod(ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)),
                          kPosPartFac, j)
            << "] -> " << spill_task[j];
      });
    }
    {
      {
        const auto bbid = can_recv_refill ? refill_bbid : input_bbid;
        const auto pos = can_recv_refill ? refill_meta.GetWritePos()
                                         : input_meta.GetWritePos();

        ap_uint<kPosPartFac> is_written = can_recv_refill ? -1 : 0;
        is_written.bit(pos % kPosPartFac) = can_recv_refill || can_enqueue;

        RANGE(j, kPosPartFac, {
          if (is_written.bit(j)) {
            chunk_buf[bbid][assume_mod(
                ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)), kPosPartFac,
                j)] = can_enqueue ? input_task : refill_task[j];

            VLOG(5) << "chunk_buf[" << bbid.bid(bank) << "]["
                    << assume_mod(
                           ChunkMeta::uint_pos_t(pos + (kPosPartFac - 1 - j)),
                           kPosPartFac, j)
                    << "] <- " << (can_enqueue ? input_task : refill_task[j]);
          }
        });
      }
    }

    if (can_recv_refill) {
      CHECK_GE(chunk_meta[refill_bbid].GetFreeSize(), refill_remain_count)
          << "refill_bid: " << refill_bbid.bid(bank);
      mem_read_data_q.read(nullptr);

      CHECK_EQ(refill_remain_count % kPosPartFac, 0);
      refill_remain_count -= kPosPartFac;
      VLOG(5) << "refilling bucket " << refill_bbid.bid(bank) << " ("
              << refill_remain_count << " to go)";
      if (refill_remain_count == 0) {
        is_refill_valid = false;
      }

      is_active = true;
    }

    {
      if (can_enqueue) {
        push_req_q.read(nullptr);

        VLOG(5) << "enqueue " << input_task << " to bucket "
                << input_bbid.bid(bank) << " (which has "
                << input_meta.GetSize() << " tasks before)";

        is_active = true;
      }
    }

    if (can_send_spill) {
      CHECK_GE(chunk_meta[spill_bbid].GetSize(),  // Available tasks.
               task_to_spill_count                // Remaining to spill.
      );
      mem_write_req_q.try_write({spill_addr_req, spill_task});

      ++spill_addr_req;
      CHECK_EQ(task_to_spill_count % kPosPartFac, 0);
      task_to_spill_count -= kPosPartFac;
      if (task_to_spill_count == 0) {
        is_spill_valid = false;
        task_to_spill_count = kChunkSize;
      }

      is_active = true;
    }

    if (can_dequeue) {
      RANGE(i, kSpilledTaskVecLen, {
        if (i >= output_meta.GetSize()) {
          output_task[i] = nullptr;
        }
      });
      pop_q.try_write(output_task);

      for (int i = 0; i < kSpilledTaskVecLen; ++i) {
        VLOG_IF(5, i < output_meta.GetSize())
            << "dequeue " << output_task[i] << " from bucket "
            << output_bbid.bid(bank) << " (which has " << output_meta.GetSize()
            << " tasks before)";
      }

      is_active = true;
    }

    bid_in_q.try_read(min_bid);
    if (!is_output_valid ||
        (can_dequeue && output_meta.GetSize() <= kCgpqPushPortCount)) {
      bid_out_q.write(kBucketCount - 1);  // Invalidate this bank's vote.
    } else {
      bid_out_q.try_write(output_bbid.bid(bank));  // Vote for min bid.
    }

    {
      ap_uint<kBucketCountPerBank> is_refill = 0;
      is_refill.bit(refill_bbid) = can_recv_refill;

      ap_uint<kBucketCountPerBank> is_input = 0;
      is_input.bit(input_bbid) = can_enqueue;

      ap_uint<kBucketCountPerBank> is_align = 0;
      is_align.bit(output_bbid) = can_dequeue && is_output_unaligned;

      ap_uint<kBucketCountPerBank> is_spill = 0;
      is_spill.bit(spill_bbid) = can_send_spill;

      ap_uint<kBucketCountPerBank> is_output = 0;
      is_output.bit(output_bbid) = can_dequeue;

      RANGE(i, kBucketCountPerBank, {
        const bool is_pop = is_spill.bit(i) || is_output.bit(i);
        ChunkMeta::uint_delta_t n_push = 0;
        const ChunkMeta::uint_delta_t n_pop = is_pop ? kSpilledTaskVecLen : 0;
        ChunkMeta::int_delta_t n_delta = 0;
        if (is_align.bit(i)) {
          CHECK(is_pop);
          n_push = kSpilledTaskVecLen - output_meta.GetSize();
          n_delta = -output_meta.GetSize();
        } else if (is_refill.bit(i)) {
          n_push = kSpilledTaskVecLen;
          n_delta = is_pop ? 0 : kSpilledTaskVecLen;
        } else if (is_input.bit(i)) {
          n_push = 1;
          n_delta = is_pop ? 1 - kSpilledTaskVecLen : 1;
        } else if (is_pop) {
          n_delta = -kSpilledTaskVecLen;
        }

        chunk_meta[i].Update(n_push, n_pop, n_delta, i);
      });
    }

    bool is_heap_requested = false;
    CgpqHeapReq heap_req;

    if (can_start_spill) {
      spill_bbid = full_bbid;
      is_spill_valid = true;

      is_heap_requested = true;
      heap_req = {
          .is_push = true,
          .elem = {.addr = spill_addr_req, .bbid = spill_bbid},
      };

      ++heap_size;
      max_heap_size = std::max(max_heap_size, heap_size);
      CHECK_LT(heap_size, kCgpqCapacity);

      VLOG(5) << "start spilling bucket " << spill_bbid.bid(bank) << " to ["
              << spill_addr_req
              << "], current buffer size: " << full_meta.GetSize();

      is_active = true;
    }

    if (can_schedule_refill) {
      refill_bbid_next = top_bbid;
      is_refill_next_valid = true;
      refill_addr = heap_root.addr;
      is_refill_addr_valid = true;
      CHECK_EQ(refill_addr % kVecCountPerChunk, 0);

      is_heap_requested = true;
      heap_req.is_push = false;

      CHECK_GT(heap_size, 0);
      --heap_size;

      VLOG(5) << "schedule refilling bucket " << refill_bbid_next.bid(bank)
              << " from " << refill_addr
              << ", current buffer size: " << chunk_meta[top_bbid].GetSize();

      is_active = true;
    }

    if (is_heap_requested) {
      CHECK(is_heap_available);
      CHECK(!heap_req_q.full());
      heap_req_q.try_write(heap_req);
      is_heap_available = false;

      is_active = true;
    }

    if (can_send_refill) {
      VLOG(5) << "refilling from [" << refill_addr << "]";

      mem_read_addr_q.try_write(refill_addr);
      ++refill_addr;
      if (refill_addr % kVecCountPerChunk == 0) {
        is_refill_addr_valid = false;
      }

      is_active = true;
    }

    if (!mem_write_resp_q.empty()) {
      mem_write_resp_q.read(nullptr);
      ++spill_addr_resp;

      is_active = true;
    }

    if (!is_refill_valid) {
      refill_bbid = refill_bbid_next;
      is_refill_valid = is_refill_next_valid;
      is_refill_next_valid = false;
      refill_remain_count = kChunkSize;
    }

    if (!heap_resp_q.empty()) {
      CHECK(!is_heap_available);
      heap_root = heap_resp_q.read(nullptr);
      is_heap_available = true;

      is_active = true;
    }

    if (is_started) {
      ++cycle_count;
    }

    if (can_enqueue) {
      is_started = true;
    }

    VLOG_IF(5, is_active);
  }

  done_q.read(nullptr);

  stat_q.write((spill_addr_req - spill_addr_base) / kVecCountPerChunk);
  stat_q.write(max_heap_size);
  stat_q.write(cycle_count);
  stat_q.write(enqueue_full_count);
  stat_q.write(enqueue_current_refill_count);
  stat_q.write(enqueue_future_refill_count);
  stat_q.write(enqueue_bank_conflict_count);
  stat_q.write(dequeue_full_count);
  stat_q.write(dequeue_spilling_count);
  stat_q.write(dequeue_bank_conflict_count);
  stat_q.write(dequeue_alignment_count);

  CHECK_EQ(heap_size, 0);

  for (int bbid = 0; bbid < kBucketCountPerBank; ++bbid) {
    CHECK(chunk_meta[bbid].IsEmpty());
  }
}

void CgpqReadAddrArbiter(  //
    istreams<uint_spill_addr_t, kCgpqBankCountPerMem>& req_in_q,
    ostream<cgpq::uint_bank_t>& req_id_q,
    ostreams<uint_spill_addr_t, kCgpqLogicMemWidth>& req_out_q) {
spin:
  for (ap_uint<kCgpqBankCountPerMem> priority = 1;;) {
#pragma HLS pipeline II = 1
    if (int bank; find_non_empty(req_in_q, priority, bank)) {
      req_id_q.write(bank);
      const auto req = req_in_q[bank].read(nullptr);
      RANGE(i, kCgpqLogicMemWidth, req_out_q[i].write(req));
      priority = 0;
      priority.set(bank);  // Make long burst.
    } else {
      priority.lrotate(1);
    }
  }
}

void CgpqReadDataArbiter(  //
    istream<cgpq::uint_bank_t>& req_id_q,
    istreams<SpilledTaskPerMem, kCgpqLogicMemWidth>& req_in_q,
    ostreams<SpilledTask, kCgpqBankCountPerMem>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    DECL_ARRAY(bool, is_req_valid, kCgpqLogicMemWidth, !req_in_q[_i].empty());
    if (!req_id_q.empty() && all_of(is_req_valid)) {
      SpilledTask task;
      RANGE(i, kCgpqLogicMemWidth, {
        const auto part = req_in_q[i].read(nullptr);
        RANGE(j, kSpilledTaskVecLenPerMem,
              task[j * kCgpqLogicMemWidth + i] = part[j]);
      });
      const auto bank = req_id_q.read(nullptr);
      req_out_q[bank].write(task);
    }
  }
}

void CgpqWriteReqArbiter(  //
    istreams<packet<uint_spill_addr_t, SpilledTask>, kCgpqBankCountPerMem>&
        req_in_q,
    ostream<cgpq::uint_bank_t>& req_id_q,
    ostreams<packet<uint_spill_addr_t, SpilledTaskPerMem>, kCgpqLogicMemWidth>&
        req_out_q) {
spin:
  for (ap_uint<kCgpqBankCountPerMem> priority = 1;;) {
#pragma HLS pipeline II = 1
    if (int bank; find_non_empty(req_in_q, priority, bank)) {
      req_id_q.write(bank);
      const auto req_in = req_in_q[bank].read(nullptr);
      RANGE(i, kCgpqLogicMemWidth, {
        packet<uint_spill_addr_t, SpilledTaskPerMem> req_out;
        req_out.addr = req_in.addr;
        RANGE(j, kSpilledTaskVecLenPerMem,
              req_out.payload[j] = req_in.payload[j * kCgpqLogicMemWidth + i]);
        req_out_q[i].write(req_out);
      });
      priority = 0;
      priority.set(bank);  // Make long burst.
    } else {
      priority.lrotate(1);
    }
  }
}

void CgpqWriteRespArbiter(  //
    istream<cgpq::uint_bank_t>& req_id_q,
    istreams<bool, kCgpqLogicMemWidth>& req_in_q,
    ostreams<bool, kCgpqBankCountPerMem>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    DECL_ARRAY(bool, is_req_valid, kCgpqLogicMemWidth, !req_in_q[_i].empty());
    if (!req_id_q.empty() && all_of(is_req_valid)) {
      RANGE(i, kCgpqLogicMemWidth, req_in_q[i].read(nullptr));
      const auto bank = req_id_q.read(nullptr);
      req_out_q[bank].write(false);
    }
  }
}

void CgpqDuplicateDone(istream<bool>& in_q,
                       ostreams<bool, kCgpqPushPortCount * 2 + 1>& out_q) {
  Duplicate(in_q, out_q);
}

void CgpqStatArbiter(istreams<int32_t, kCgpqPushPortCount>& in_q,
                     ostream<int32_t>& out_q) {
  using uint_i_t = ap_uint<bit_length(kCgpqPushPortCount)>;
  using uint_j_t = ap_uint<bit_length(kQueueStatCount / kCgpqPushPortCount)>;
spin:
  for (;;) {
    for (uint_i_t i = 0; i < kCgpqPushPortCount; ++i) {
      for (uint_j_t j = 0; j < kQueueStatCount / kCgpqPushPortCount;) {
#pragma HLS pipeline II = 1
        if (!in_q[i].empty()) {
          out_q.write(in_q[i].read(nullptr));
          ++j;
        }
      }
    }
  }
}

void CgpqOutputArbiter(istream<bool>& done_q,
                       istreams<SpilledTask, kCgpqPushPortCount>& in_q,
                       ostreams<TaskOnChip, kSpilledTaskVecLen>& out_q) {
spin:
  for (ap_uint<kCgpqPushPortCount> priority = 1; done_q.empty();
       priority.lrotate(1)) {
#pragma HLS pipeline II = 1
    if (int bank; find_non_empty(in_q, priority, bank)) {
      const auto vec = in_q[bank].read(nullptr);
      RANGE(i, kSpilledTaskVecLen, {
        if (vec[i].is_valid()) {
          out_q[i].write(vec[i]);
        };
      });
    }
  }
  done_q.read(nullptr);
}

void CgpqSwitch(
    //
    int b,
    //
    istream<PushReq>& in_q0, istream<PushReq>& in_q1,
    //
    ostreams<PushReq, 2>& out_q) {
  b = kCgpqPushStageCount - 1 - b / kSwitchMuxDegree;

  bool should_prioritize_1 = false;
spin:
  for (bool is_pkt_0_valid, is_pkt_1_valid;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    const auto pkt_0 = in_q0.peek(is_pkt_0_valid);
    const auto pkt_1 = in_q1.peek(is_pkt_1_valid);

    const auto addr_0 = pkt_0.bid;
    const auto addr_1 = pkt_1.bid;

    const bool should_fwd_0_0 = is_pkt_0_valid && !addr_0.get_bit(b);
    const bool should_fwd_0_1 = is_pkt_0_valid && addr_0.get_bit(b);
    const bool should_fwd_1_0 = is_pkt_1_valid && !addr_1.get_bit(b);
    const bool should_fwd_1_1 = is_pkt_1_valid && addr_1.get_bit(b);

    const bool has_conflict = is_pkt_0_valid && is_pkt_1_valid &&
                              should_fwd_0_0 == should_fwd_1_0 &&
                              should_fwd_0_1 == should_fwd_1_1;

    const bool should_read_0 = !((!should_fwd_0_0 && !should_fwd_0_1) ||
                                 (should_prioritize_1 && has_conflict));
    const bool should_read_1 = !((!should_fwd_1_0 && !should_fwd_1_1) ||
                                 (!should_prioritize_1 && has_conflict));
    const bool should_write_0 = should_fwd_0_0 || should_fwd_1_0;
    const bool should_write_1 = should_fwd_1_1 || should_fwd_0_1;
    const bool shoud_write_0_0 =
        should_fwd_0_0 && (!should_fwd_1_0 || !should_prioritize_1);
    const bool shoud_write_1_1 =
        should_fwd_1_1 && (!should_fwd_0_1 || should_prioritize_1);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, check for conflict
    const bool is_0_written =
        should_write_0 && out_q[0].try_write(shoud_write_0_0 ? pkt_0 : pkt_1);
    const bool is_1_written =
        should_write_1 && out_q[1].try_write(shoud_write_1_1 ? pkt_1 : pkt_0);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, round robin priority of both ins
    if (should_read_0 && (shoud_write_0_0 ? is_0_written : is_1_written)) {
      in_q0.read(nullptr);
    }
    if (should_read_1 && (shoud_write_1_1 ? is_1_written : is_0_written)) {
      in_q1.read(nullptr);
    }

    if (has_conflict) {
      should_prioritize_1 = !should_prioritize_1;
    }
  }
}

void CgpqSwitchInnerStage(int b,
                          istreams<PushReq, kCgpqPushPortCount / 2>& in_q0,
                          istreams<PushReq, kCgpqPushPortCount / 2>& in_q1,
                          ostreams<PushReq, kCgpqPushPortCount>& out_q) {
  task().invoke<detach, kCgpqPushPortCount / 2>(CgpqSwitch, b, in_q0, in_q1,
                                                out_q);
}

void CgpqSwitchStage(int b, istreams<PushReq, kCgpqPushPortCount>& in_q,
                     ostreams<PushReq, kCgpqPushPortCount>& out_q) {
  task().invoke<detach>(CgpqSwitchInnerStage, b, in_q, in_q, out_q);
}

void CgpqSwitchDemux(istream<PushReq>& in_q,
                     ostreams<PushReq, kSwitchMuxDegree>& out_q) {
spin:
  for (ap_uint<kSwitchMuxDegree> priority = 1;; priority.lrotate(1)) {
    if (int pos; find_non_full(out_q, priority, pos) && !in_q.empty()) {
      out_q[pos].try_write(in_q.read(nullptr));
    }
  }
}

void CgpqSwitchMux(istreams<PushReq, kSwitchMuxDegree>& in_q,
                   ostream<PushReq>& out_q) {
spin:
  for (ap_uint<kSwitchMuxDegree> priority = 1;; priority.lrotate(1)) {
#pragma HLS pipeline II = 1
    if (int pos; find_non_empty(in_q, priority, pos) && !out_q.full()) {
      out_q.try_write(in_q[pos].read(nullptr));
    }
  }
}

void CgpqPushAdapter(
    istreams<PushReq, kCgpqPushPortCount * kSwitchMuxDegree>& in_q,
    ostreams<PushReq, kCgpqPushPortCount * kSwitchMuxDegree>& out_q) {
  Transpose<kSwitchMuxDegree, kCgpqPushPortCount>(
      in_q, out_q, [](const auto& req, int old_pos, int new_pos) {
        CHECK_EQ(req.bid % kCgpqPushPortCount, old_pos % kCgpqPushPortCount);
      });
}

void SwitchMux(istreams<TaskOnChip, kSwitchMuxDegree>& in_q,
               ostream<TaskOnChip>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    TaskOnChip task;
    int pos;
    if (find_max_non_empty(in_q, task, pos)) {
      in_q[pos].read(nullptr);
      out_q.write(task);
    }
  }
}

void CgpqPopAdapter(istream<TaskOnChip>& in_q,
                    ostreams<TaskOnChip, kSubIntervalPerSwPort>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (TaskOnChip task; in_q.try_read(task)) {
      out_q[task.vid() % kSubIntervalPerSwPort].write(task);
    }
  }
}

void Switch2x2(
    //
    int b,
    //
    istream<TaskOnChip>& in_q0, istream<TaskOnChip>& in_q1,
    //
    ostreams<TaskOnChip, 2>& out_q) {
  b = kSwitchStageCount - 1 - b / kSwitchMuxDegree;

  bool should_prioritize_1 = false;
  int64_t total_cycle_count = 0;
  int64_t full_0_cycle_count = 0;
  int64_t full_1_cycle_count = 0;
  int64_t conflict_0_cycle_count = 0;
  int64_t conflict_1_cycle_count = 0;

spin:
  for (bool is_pkt_0_valid, is_pkt_1_valid;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    const auto pkt_0 = in_q0.peek(is_pkt_0_valid);
    const auto pkt_1 = in_q1.peek(is_pkt_1_valid);

    const uint_vid_t addr_0 = pkt_0.vid();
    const uint_vid_t addr_1 = pkt_1.vid();

    const bool should_fwd_0_0 = is_pkt_0_valid && !addr_0.get_bit(b);
    const bool should_fwd_0_1 = is_pkt_0_valid && addr_0.get_bit(b);
    const bool should_fwd_1_0 = is_pkt_1_valid && !addr_1.get_bit(b);
    const bool should_fwd_1_1 = is_pkt_1_valid && addr_1.get_bit(b);

    const bool has_conflict = is_pkt_0_valid && is_pkt_1_valid &&
                              should_fwd_0_0 == should_fwd_1_0 &&
                              should_fwd_0_1 == should_fwd_1_1;

    const bool should_read_0 = !((!should_fwd_0_0 && !should_fwd_0_1) ||
                                 (should_prioritize_1 && has_conflict));
    const bool should_read_1 = !((!should_fwd_1_0 && !should_fwd_1_1) ||
                                 (!should_prioritize_1 && has_conflict));
    const bool should_write_0 = should_fwd_0_0 || should_fwd_1_0;
    const bool should_write_1 = should_fwd_1_1 || should_fwd_0_1;
    const bool shoud_write_0_0 =
        should_fwd_0_0 && (!should_fwd_1_0 || !should_prioritize_1);
    const bool shoud_write_1_1 =
        should_fwd_1_1 && (!should_fwd_0_1 || should_prioritize_1);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, check for conflict
    const bool is_0_written =
        should_write_0 && out_q[0].try_write(shoud_write_0_0 ? pkt_0 : pkt_1);
    const bool is_1_written =
        should_write_1 && out_q[1].try_write(shoud_write_1_1 ? pkt_1 : pkt_0);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, round robin priority of both ins
    if (should_read_0 && (shoud_write_0_0 ? is_0_written : is_1_written)) {
      in_q0.read(nullptr);
    }
    if (should_read_1 && (shoud_write_1_1 ? is_1_written : is_0_written)) {
      in_q1.read(nullptr);
    }

    if (has_conflict) {
      should_prioritize_1 = !should_prioritize_1;
    }

    if (should_write_0) {
      if (!is_0_written) {
        ++full_0_cycle_count;
      }
      if (has_conflict) {
        ++conflict_0_cycle_count;
      }
    }
    if (should_write_1) {
      if (!is_1_written) {
        ++full_1_cycle_count;
      }
      if (has_conflict) {
        ++conflict_1_cycle_count;
      }
    }
    ++total_cycle_count;
  }
}

void SwitchInnerStage(int b, istreams<TaskOnChip, kSwitchPortCount / 2>& in_q0,
                      istreams<TaskOnChip, kSwitchPortCount / 2>& in_q1,
                      ostreams<TaskOnChip, kSwitchPortCount>& out_q) {
  task().invoke<detach, kSwitchPortCount / 2>(Switch2x2, b, in_q0, in_q1,
                                              out_q);
}

void SwitchStage(int b, istreams<TaskOnChip, kSwitchPortCount>& in_q,
                 ostreams<TaskOnChip, kSwitchPortCount>& out_q) {
  task().invoke<detach>(SwitchInnerStage, b, in_q, in_q, out_q);
}

void PopSwitch2x2(
    //
    int b,
    //
    istream<TaskOnChip>& in_q0, istream<TaskOnChip>& in_q1,
    //
    ostreams<TaskOnChip, 2>& out_q) {
  constexpr int kIgnoredWidth = log2(kSubIntervalPerSwPort);
  b = kPopSwitchStageCount - 1 - b + kIgnoredWidth;

  bool should_prioritize_1 = false;

spin:
  for (bool is_pkt_0_valid, is_pkt_1_valid;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    const auto pkt_0 = in_q0.peek(is_pkt_0_valid);
    const auto pkt_1 = in_q1.peek(is_pkt_1_valid);

    const uint_vid_t addr_0 = pkt_0.vid();
    const uint_vid_t addr_1 = pkt_1.vid();

    const bool should_fwd_0_0 = is_pkt_0_valid && !addr_0.get_bit(b);
    const bool should_fwd_0_1 = is_pkt_0_valid && addr_0.get_bit(b);
    const bool should_fwd_1_0 = is_pkt_1_valid && !addr_1.get_bit(b);
    const bool should_fwd_1_1 = is_pkt_1_valid && addr_1.get_bit(b);

    const bool has_conflict = is_pkt_0_valid && is_pkt_1_valid &&
                              should_fwd_0_0 == should_fwd_1_0 &&
                              should_fwd_0_1 == should_fwd_1_1;

    const bool should_read_0 = !((!should_fwd_0_0 && !should_fwd_0_1) ||
                                 (should_prioritize_1 && has_conflict));
    const bool should_read_1 = !((!should_fwd_1_0 && !should_fwd_1_1) ||
                                 (!should_prioritize_1 && has_conflict));
    const bool should_write_0 = should_fwd_0_0 || should_fwd_1_0;
    const bool should_write_1 = should_fwd_1_1 || should_fwd_0_1;
    const bool shoud_write_0_0 =
        should_fwd_0_0 && (!should_fwd_1_0 || !should_prioritize_1);
    const bool shoud_write_1_1 =
        should_fwd_1_1 && (!should_fwd_0_1 || should_prioritize_1);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, check for conflict
    const bool is_0_written =
        should_write_0 && out_q[0].try_write(shoud_write_0_0 ? pkt_0 : pkt_1);
    const bool is_1_written =
        should_write_1 && out_q[1].try_write(shoud_write_1_1 ? pkt_1 : pkt_0);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, round robin priority of both ins
    if (should_read_0 && (shoud_write_0_0 ? is_0_written : is_1_written)) {
      in_q0.read(nullptr);
    }
    if (should_read_1 && (shoud_write_1_1 ? is_1_written : is_0_written)) {
      in_q1.read(nullptr);
    }

    if (has_conflict) {
      should_prioritize_1 = !should_prioritize_1;
    }
  }
}

void PopSwitchInnerStage(int b,
                         istreams<TaskOnChip, kPopSwitchPortCount / 2>& in_q0,
                         istreams<TaskOnChip, kPopSwitchPortCount / 2>& in_q1,
                         ostreams<TaskOnChip, kPopSwitchPortCount>& out_q) {
  task().invoke<detach, kPopSwitchPortCount / 2>(PopSwitch2x2, b, in_q0, in_q1,
                                                 out_q);
}

void PopSwitchStage(int b, istreams<TaskOnChip, kPopSwitchPortCount>& in_q,
                    ostreams<TaskOnChip, kPopSwitchPortCount>& out_q) {
  task().invoke<detach>(PopSwitchInnerStage, b, in_q, in_q, out_q);
}

void SwitchDemux(istream<TaskOnChip>& in_q,
                 ostreams<TaskOnChip, kSwitchMuxDegree>& out_q) {
spin:
  for (ap_uint<kSwitchMuxDegree> priority = 1;; priority.lrotate(1)) {
    if (int pos; find_non_full(out_q, priority, pos) && !in_q.empty()) {
      out_q[pos].try_write(in_q.read(nullptr));
    }
  }
}

void PushAdapter(
    istreams<TaskOnChip, kSwitchPortCount * kSwitchMuxDegree>& in_q,
    ostreams<TaskOnChip, kSwitchPortCount * kSwitchMuxDegree>& out_q) {
  Transpose<kSwitchMuxDegree, kSwitchPortCount>(
      in_q, out_q, [](const auto& task, int old_pos, int new_pos) {
        CHECK_EQ(task.vid() % kSwitchPortCount, old_pos % kSwitchPortCount);
        CHECK_EQ(task.vid() % kSwitchPortCount,
                 new_pos / kSwitchMuxDegree % kSwitchPortCount);
      });
}

void SwitchOutputArbiter(
    tapa::istreams<TaskOnChip, kShardCount * kEdgeVecLen>& in_q,
    tapa::ostreams<TaskOnChip, kCgpqPushPortCount>& out_q) {
  TaskArbiterTemplate(in_q, out_q);
}

void EdgeMem(istream<bool>& done_q, ostream<int64_t>& stat_q,
             istream<Vid>& read_addr_q, ostream<EdgeVec>& read_data_q,
             async_mmap<EdgeVec> mem) {
  int64_t total_cycle_count = 0;
  int64_t active_cycle_count = 0;
  int64_t mem_stall_cycle_count = 0;
  int64_t pe_stall_cycle_count = 0;

spin:
  for (; done_q.empty();) {
#pragma HLS pipeline II = 1

    ++total_cycle_count;

    if (!read_addr_q.empty()) {
      if (mem.read_addr.full()) {
        ++mem_stall_cycle_count;
      } else {
        ++active_cycle_count;
        mem.read_addr.try_write(read_addr_q.read(nullptr));
      }
    }

    if (!mem.read_data.empty()) {
      if (read_data_q.full()) {
        ++pe_stall_cycle_count;
      } else {
        read_data_q.try_write(mem.read_data.read(nullptr));
      }
    }
  }

  done_q.read(nullptr);
  stat_q.write(total_cycle_count);
  stat_q.write(active_cycle_count);
  stat_q.write(mem_stall_cycle_count);
  stat_q.write(pe_stall_cycle_count);
}

void EdgeReqGen(istream<Task>& task_in_q, ostream<uint_vid_t>& task_count_q,
                ostream<SourceVertex>& src_q, ostream<Vid>& edge_addr_q) {
  EdgeReq req;

spin:
  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      req = {task.vertex.offset,
             {
                 .vid = task.vid,
                 .parent = task.vertex.parent,
                 .distance = task.vertex.distance,
             }};
      task_count_q.write(task.vertex.degree - 1);  // Don't visit parent.
      i = tapa::round_up_div<kEdgeVecLen>(task.vertex.degree);
    }

    if (i > 0) {
      src_q.write(req.payload);
      edge_addr_q.write(req.addr);
      ++req.addr;
      --i;
    }
  }
}

void DistGen(istream<SourceVertex>& src_in_q,
             istream<EdgeVec>& edges_read_data_q,
             ostreams<TaskOnChip, kEdgeVecLen>& update_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!src_in_q.empty() && !edges_read_data_q.empty()) {
      const auto src = src_in_q.read(nullptr);
      const auto edge_v = edges_read_data_q.read(nullptr);
      TaskVec task_v;
      RANGE(i, kEdgeVecLen, {
        if (!std::isinf(edge_v[i].weight) &&
            uint_vid_t(edge_v[i].dst) != uint_vid_t(src.parent)) {
          update_out_q[i].write(Task{
              .vid = edge_v[i].dst,
              .vertex = {src.vid, src.distance + edge_v[i].weight},
          });
        }
      });
    }
  }
}

void VertexMem(
    //
    istream<Vid>& read_addr_q, ostream<Vertex>& read_data_q,
    istream<packet<Vid, Vertex>>& write_req_q, ostream<bool>& write_resp_q,
    //
    tapa::async_mmap<Vertex> mem) {
  ReadWriteMem(read_addr_q, read_data_q, write_req_q, write_resp_q, mem);
}

void VertexCache(
    //
    istream<bool>& done_q, ostream<int32_t>& stat_q,
    //
    istream<TaskOnChip>& push_in_q, ostream<TaskOnChip>& push_out_q,
    //
    istream<TaskOnChip>& pop_in_q, ostream<Task>& pop_out_q,
    //
    ostream<bool>& noop_q,
    //
    ostream<Vid>& read_vid_out_q, istream<Vid>& read_vid_in_q,
    ostream<Vid>& write_vid_out_q, istream<Vid>& write_vid_in_q,
    //
    ostream<Vid>& read_addr_q, istream<Vertex>& read_data_q,
    ostream<packet<Vid, Vertex>>& write_req_q, istream<bool>& write_resp_q) {
  constexpr int kLogLevel = 5;

  constexpr int kVertexCacheSize = 4096 * 16;
  VertexCacheEntry cache[kVertexCacheSize];
#pragma HLS bind_storage variable = cache type = RAM_S2P impl = URAM
#pragma HLS aggregate variable = cache bit

init:
  for (int i = 0; i < kVertexCacheSize; ++i) {
#pragma HLS dependence variable = cache inter false
    cache[i] = nullptr;
  }

  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < kVertexCacheSize; ++i) {
      CHECK(!cache[i].is_valid);
    }
  });

exec:
  for (;;) {
#pragma HLS pipeline off

    DECL_ARRAY(int32_t, perf_counters, kVertexUniStatCount, 0);

    auto& read_hit = perf_counters[0];
    auto& read_miss = perf_counters[1];
    auto& write_hit = perf_counters[2];
    auto& write_miss = perf_counters[3];

    auto& write_resp_count = perf_counters[4];
    auto& read_resp_count = perf_counters[5];
    auto& push_busy_count = perf_counters[6];
    auto& pop_busy_count = perf_counters[7];
    auto& noop_busy_count = perf_counters[8];
    auto& entry_busy_count = perf_counters[9];
    auto& read_busy_count = perf_counters[10];
    auto& write_busy_count = perf_counters[11];
    auto& idle_count = perf_counters[12];

    const int kMaxActiveWriteCount = 63;
    int8_t active_write_count = 0;

    bool is_started = false;

    Vertex vertex;
    bool is_read_data_valid = false;

    Vid read_vid;
    bool is_read_vid_valid = false;

    TaskOnChip push_task = nullptr;
    bool is_push_task_valid = false;

    TaskOnChip pop_task = nullptr;
    bool is_pop_task_valid = false;

    using uint_cache_index_t = ap_uint<log2(kVertexCacheSize)>;
    uint_cache_index_t prev_index = 0;
    VertexCacheEntry prev_entry = nullptr;

#define GEN_NOOP(is_push)                  \
  do {                                     \
    noop_q.try_write(is_push);             \
    VLOG(kLogLevel) << "task     -> NOOP"; \
  } while (0)

#define MARK_DIRTY()                                                       \
  do {                                                                     \
    entry.is_dirty = true;                                                 \
    VLOG(kLogLevel) << "v$$$[" << entry.GetTask().vid << "] marked dirty"; \
  } while (0)

#define WRITE_MISS()                                             \
  do {                                                           \
    CHECK_LT(active_write_count, kMaxActiveWriteCount);          \
    ++active_write_count;                                        \
    ++write_miss;                                                \
    VLOG(kLogLevel) << "vmem[" << entry.GetTask().vid << "] <- " \
                    << entry.GetTask();                          \
  } while (0)

  spin:
    for (; done_q.empty();) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = cache inter true distance = 2

      const bool is_write_resp_valid = !write_resp_q.empty();
      bool is_write_vid_valid;
      const auto write_vid = write_vid_in_q.peek(is_write_vid_valid);
      const bool is_write_ack_valid = is_write_resp_valid && is_write_vid_valid;

      const bool is_read_ack_valid = is_read_data_valid && is_read_vid_valid;

      const auto vid = is_write_ack_valid   ? write_vid
                       : is_read_ack_valid  ? read_vid
                       : is_push_task_valid ? push_task.vid()
                                            : pop_task.vid();

      // curr_index = vid / kSubIntervalCount but HLS seems to have trouble
      // optimizing this, thus the following.
      constexpr int kSubIntervalWidth = log2(kSubIntervalCount);
      const uint_cache_index_t curr_index = uint_vid_t(vid).range(
          uint_cache_index_t::width + kSubIntervalWidth - 1, kSubIntervalWidth);

      // Forward when there is a read-after-write dependence from preivous 1
      // iteration.
      auto entry = curr_index == prev_index ? prev_entry : cache[curr_index];
      bool is_entry_updated = false;

      const bool is_hit = entry.is_valid && entry.GetTask().vid == vid;
      const bool is_entry_busy =
          entry.is_valid && (entry.is_reading || entry.is_writing);
      const bool is_read_busy = read_addr_q.full() || read_vid_out_q.full();
      const bool is_write_busy = entry.is_valid && entry.is_dirty &&
                                 (write_req_q.full() || write_vid_out_q.full());
      const bool is_push_out_busy = push_out_q.full();
      const bool is_pop_out_busy = pop_out_q.full();
      const bool is_noop_busy = noop_q.full();

      if (is_write_ack_valid) {
        ++write_resp_count;

        write_resp_q.read(nullptr);
        write_vid_in_q.read(nullptr);
        CHECK(entry.is_valid);
        CHECK_NE(entry.GetTask().vid, vid);
        CHECK(entry.is_writing);
        entry.is_writing = false;
        is_entry_updated = true;
        CHECK_GT(active_write_count, 0);
        --active_write_count;
      } else if (is_read_ack_valid) {
        if (!is_push_out_busy && !is_noop_busy) {
          ++read_resp_count;

          is_read_data_valid = is_read_vid_valid = false;
          VLOG(kLogLevel) << "vmem[" << vid << "] -> " << vertex;
          CHECK(entry.is_valid);
          CHECK_EQ(entry.GetTask().vid, vid);
          CHECK(entry.is_reading);
          entry.is_reading = false;
          // Vertex updates do not have metadata of the destination vertex, so
          // update cache using metadata from DRAM.
          entry.SetMetadata(vertex);
          if (entry.is_push) {  // Reading for PUSH.
            if (!(entry.GetTask().vertex < vertex)) {
              // Distance in DRAM is closer; generate NOOP and update cache.
              GEN_NOOP(/*is_push=*/true);
              entry.SetValue(vertex);
            } else {
              // Distance in cache is closer; generate PUSH.
              if (entry.GetTask().vertex.degree > 1) {
                push_out_q.try_write(entry.GetTask());
                VLOG(kLogLevel) << "task     -> PUSH " << entry.GetTask();
              } else {
                GEN_NOOP(/*is_push=*/true);
              }
              MARK_DIRTY();
              ++write_hit;
            }
          } else {  // Reading for POP.
            if (vertex < entry.GetTask().vertex) {
              // Distance in DRAM is closer; update cache.
              entry.SetValue(vertex);
            } else if (entry.GetTask().vertex < vertex) {
              // Distance in cache is closer; mark cache dirty.
              MARK_DIRTY();
              ++write_hit;
            }
          }
          is_entry_updated = true;
        } else if (is_push_out_busy) {
          ++push_busy_count;
        } else {
          CHECK(is_noop_busy);
          ++noop_busy_count;
        }
      } else if (is_push_task_valid) {
        is_started = true;
        CHECK_EQ(vid, push_task.vid());

        if (is_hit) {
          if (!is_push_out_busy && !is_noop_busy) {
            ++read_hit;

            is_push_task_valid = false;
            VLOG(kLogLevel) << "task     <- " << push_task;

            if (entry.is_reading && !entry.is_push) {  // Reading for POP.

              if ((is_entry_updated = entry.GetTask() < push_task)) {
                // New PUSH task has higher priority.
                // Update cache and read for PUSH.
                entry.SetValue(push_task.vertex());
                entry.is_push = true;
                ++write_hit;
              } else {
                // New PUSH task does not have higher priority.
                // Discard PUSH task.
                GEN_NOOP(/*is_push=*/true);
              }
            } else {  // Reading for PUSH or not reading.

              // Update cache if new task has higher priority.
              if ((is_entry_updated = entry.GetTask() < push_task)) {
                entry.SetValue(push_task.vertex());
                ++write_hit;
              }

              // Generate PUSH if and only if cache is updated and not reading.
              // If reading, PUSH will be generated when read finishes, if
              // necessary.
              if (is_entry_updated && !entry.is_reading) {
                if (entry.GetTask().vertex.degree > 1) {
                  push_out_q.try_write(entry.GetTask());
                  VLOG(kLogLevel) << "task     -> PUSH " << entry.GetTask();
                } else {
                  GEN_NOOP(/*is_push=*/true);
                }

                MARK_DIRTY();
                ++write_hit;
              } else {
                GEN_NOOP(/*is_push=*/true);
              }
            }
          } else if (is_push_out_busy) {
            ++push_busy_count;
          } else {
            CHECK(is_noop_busy);
            ++noop_busy_count;
          }

        } else if (!is_entry_busy && !is_read_busy && !is_write_busy) {
          ++read_miss;

          is_push_task_valid = false;
          VLOG(kLogLevel) << "task     <- " << push_task;

          // Issue DRAM read request.
          read_addr_q.try_write(vid / kIntervalCount);
          VLOG(kLogLevel) << "vmem[" << vid << "] ?";
          read_vid_out_q.try_write(vid);

          // Issue DRAM write request.
          if (entry.is_valid && entry.is_dirty) {
            entry.is_writing = true;
            write_req_q.try_write(
                {entry.GetTask().vid / kIntervalCount, entry.GetTask().vertex});
            write_vid_out_q.try_write(entry.GetTask().vid);
            WRITE_MISS();
          } else {
            entry.is_writing = false;
          }

          // Replace cache with new task.
          entry.is_valid = true;
          entry.is_reading = true;
          entry.is_dirty = false;
          entry.is_push = true;
          entry.SetVid(vid);
          entry.SetValue(push_task.vertex());
          is_entry_updated = true;
        } else if (is_entry_busy) {
          ++entry_busy_count;
        } else if (is_read_busy) {
          ++read_busy_count;
        } else {
          CHECK(is_write_busy);
          ++write_busy_count;
        }

      } else if (is_pop_task_valid) {
        CHECK_EQ(vid, pop_task.vid());

        if (is_hit) {
          if (!is_pop_out_busy && !is_noop_busy) {
            ++read_hit;

            if (entry.is_reading) {
              if (entry.is_push) {  // Reading for PUSH.
                if (pop_task < entry.GetTask()) {
                  // PUSH task has higher priority.
                  // Discard POP task.
                  is_pop_task_valid = false;
                  GEN_NOOP(/*is_push=*/false);
                } else {
                  // PUSH task does not have higher priority.
                  // Discard PUSH task.
                  // Read for POP.
                  GEN_NOOP(/*is_push=*/true);
                  entry.is_push = false;
                  is_entry_updated = true;
                }
              } else {  // Reading for POP.
                // Just wait for read data.
              }
            } else {  // Not reading.
              is_pop_task_valid = false;
              VLOG(kLogLevel) << "task     <- " << pop_task;

              if (pop_task < entry.GetTask()) {
                // POP task is stale.
                GEN_NOOP(/*is_push=*/false);
              } else {
                // POP task is not stale.
                // It must be exactly the same as in the cache.
                CHECK_EQ(entry.GetTask().vertex.distance,
                         pop_task.vertex().distance)
                    << vid;
                CHECK_EQ(entry.GetTask().vertex.parent,
                         pop_task.vertex().parent)
                    << vid;
                CHECK_GT(entry.GetTask().vertex.degree, 1) << vid;

                pop_out_q.try_write(entry.GetTask());
                VLOG(kLogLevel) << "task     -> POP " << entry.GetTask();
              }
            }
          } else if (is_pop_out_busy) {
            ++pop_busy_count;
          } else {
            CHECK(is_noop_busy);
            ++noop_busy_count;
          }

        } else if (!is_entry_busy && !is_read_busy && !is_write_busy) {
          ++read_miss;

          // Issue DRAM read request.
          read_addr_q.try_write(vid / kIntervalCount);
          VLOG(kLogLevel) << "vmem[" << vid << "] ?";
          read_vid_out_q.try_write(vid);

          // Issue DRAM write request.
          if (entry.is_valid && entry.is_dirty) {
            entry.is_writing = true;
            write_req_q.try_write(
                {entry.GetTask().vid / kIntervalCount, entry.GetTask().vertex});
            write_vid_out_q.try_write(entry.GetTask().vid);
            WRITE_MISS();
          } else {
            entry.is_writing = false;
          }

          // Replace cache with new task.
          entry.is_valid = true;
          entry.is_reading = true;
          entry.is_dirty = false;
          entry.is_push = false;
          entry.SetVid(vid);
          entry.SetValue(pop_task.vertex());
          is_entry_updated = true;
        } else if (is_entry_busy) {
          ++entry_busy_count;
        } else if (is_read_busy) {
          ++read_busy_count;
        } else {
          CHECK(is_write_busy);
          ++write_busy_count;
        }
      } else if (is_started) {
        ++idle_count;
      }

      if (is_entry_updated) {
        cache[curr_index] = entry;
        VLOG(kLogLevel) << "v$$$[" << vid << "] <- " << entry.GetTask();
      }
      prev_index = curr_index;
      prev_entry = entry;

      if (!is_read_data_valid) {
        is_read_data_valid = read_data_q.try_read(vertex);
      }
      if (!is_read_vid_valid) {
        is_read_vid_valid = read_vid_in_q.try_read(read_vid);
      }
      if (!is_pop_task_valid) {
        is_pop_task_valid = pop_in_q.try_read(pop_task);
      }
      if (!is_push_task_valid) {
        is_push_task_valid = push_in_q.try_read(push_task);
      }
    }

  reset:
    for (int i = 0; i < kVertexCacheSize;) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = cache inter false
      // Limit number of outstanding write requests to kMaxActiveWriteCount.
      const bool is_ready = active_write_count < kMaxActiveWriteCount;
      if (!write_resp_q.empty()) {
        write_resp_q.read(nullptr);
        --active_write_count;
      }
      if (is_ready) {
        auto entry = cache[i];
        CHECK(!entry.is_reading);

        if (entry.is_valid && entry.is_dirty) {
          write_req_q.write(
              {entry.GetTask().vid / kIntervalCount, entry.GetTask().vertex});
          WRITE_MISS();
        }

        entry.is_valid = false;
        entry.is_writing = false;
        cache[i] = entry;
        ++i;
      }
    }

  clean:
    while (active_write_count > 0) {
      if (!write_resp_q.empty()) {
        write_resp_q.read(nullptr);
        --active_write_count;
      }
    }

    done_q.read(nullptr);
  stat:
    for (ap_uint<bit_length(kVertexUniStatCount)> i = 0;
         i < kVertexUniStatCount; ++i) {
      stat_q.write(perf_counters[i]);
    }
  }

#undef WRITE_MISS
#undef MARK_DIRTY
#undef GEN_NOOP
}

void VertexNoopMerger(istreams<bool, kSubIntervalCount>& noop_in_q,
                      ostream<VertexNoop>& noop_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    VertexNoop count = {0, 0};
    RANGE(iid, kSubIntervalCount, {
      if (bool is_push; noop_in_q[iid].try_read(is_push)) {
        if (is_push) {
          ++count.push_count;
        } else {
          ++count.pop_count;
        }
      }
    });
    if (count.push_count || count.pop_count) {
      noop_out_q.write(count);
    }
  }
}

void TaskCountMerger(istreams<uint_vid_t, kPeCount>& in_q,
                     ostream<TaskCount>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    TaskCount count = {0, 0};
    RANGE(peid, kPeCount, {
      if (!in_q[peid].empty()) {
        ++count.old_task_count;
        count.new_task_count += in_q[peid].read(nullptr);
      }
    });
    if (count.old_task_count) {
      out_q.write(count);
    }
  }
}

void VertexOutputAdapter(istreams<Task, kSubIntervalCount>& in_q,
                         ostreams<Task, kSubIntervalCount>& out_q) {
  Transpose<kSubIntervalCount / kShardCount, kShardCount>(
      in_q, out_q, [](const auto& task, int old_pos, int new_pos) {
        CHECK_EQ(task.vid % kSubIntervalCount, old_pos);
        CHECK_EQ(task.vid % kShardCount,
                 new_pos / (kSubIntervalCount / kShardCount) % kShardCount);
      });
}

void VertexOutputArbiter(istreams<Task, kSubIntervalCount / kShardCount>& in_q,
                         ostream<Task>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    Task task;
    int pos;
    if (find_max_non_empty(in_q, task, pos)) {
      in_q[pos].read(nullptr);
      out_q.write(task);
    }
  }
}

const int kTaskInputCount = kShardCount;

void TaskArbiter(  //
    istream<Task>& task_init_q, istreams<Task, kTaskInputCount>& task_in_q,
    ostreams<Task, kPeCount>& task_req_q) {
exec:
  for (;;) {
    static_assert(kPeCount % kTaskInputCount == 0, "");

    const auto task_init = task_init_q.read();
    const auto tid_init = task_init.vid % kTaskInputCount;
    task_req_q[tid_init].write(task_init);

  spin:
    for (; !task_init_q.eot(nullptr);) {
#pragma HLS pipeline II = 1
      // Issue task requests.
      RANGE(tid, kTaskInputCount, {
        if (!task_in_q[tid].empty() && !task_req_q[tid].full()) {
          const auto task = task_in_q[tid].read(nullptr);
          CHECK_EQ(task.vid % kTaskInputCount, tid);
          task_req_q[tid].try_write(task);
        }
      });
    }

    task_init_q.try_open();
  }
}

void Dispatcher(
    // Scalar.
    const Task root,
    // Metadata.
    tapa::mmap<int64_t> metadata,
    // Vertex cache control.
    ostreams<bool, kSubIntervalCount>& vertex_cache_done_q,
    istreams<int32_t, kSubIntervalCount>& vertex_cache_stat_q,
    ostreams<bool, kShardCount>& edge_done_q,
    istreams<int64_t, kShardCount>& edge_stat_q, ostream<bool>& queue_done_q,
    istream<int32_t>& queue_stat_q,
    // Task initialization.
    ostream<Task>& task_init_q,
    // Task count.
    istream<VertexNoop>& vertex_noop_q, istream<TaskCount>& task_count_q) {
  task_init_q.write(root);

  // Statistics.
  int32_t visited_edge_count = 1;
  int32_t visited_vertex_count = -1;
  int32_t push_noop_count = 0;
  int32_t pop_noop_count = 0;
  int64_t cycle_count = 0;

  constexpr int kTerminationHold = 500;
  ap_uint<bit_length(kTerminationHold)> termination = 0;

spin:
  for (int32_t active_task_count = 2;  // Because root doesn't have a parent.
       active_task_count || termination < kTerminationHold; ++cycle_count) {
#pragma HLS pipeline II = 1
    if (active_task_count == 0) {
      ++termination;
    } else {
      termination = 0;
    }

    if (!vertex_noop_q.empty()) {
      const auto previous_task_count = active_task_count;
      const auto count = vertex_noop_q.read(nullptr);
      active_task_count -= count.push_count;
      active_task_count -= count.pop_count;
      push_noop_count += count.push_count;
      pop_noop_count += count.pop_count;
      VLOG(4) << "#task " << previous_task_count << " -> " << active_task_count;
    }

    if (!task_count_q.empty()) {
      const auto previous_task_count = active_task_count;
      const auto count = task_count_q.read(nullptr);
      active_task_count += count.new_task_count - count.old_task_count;
      visited_edge_count += count.new_task_count;
      visited_vertex_count += count.old_task_count;
      VLOG(4) << "#task " << previous_task_count << " -> " << active_task_count;
    }
  }

  RANGE(iid, kSubIntervalCount, vertex_cache_done_q[iid].write(false));
  RANGE(sid, kShardCount, edge_done_q[sid].write(false));
  queue_done_q.write(false);
  task_init_q.close();

  metadata[0] = visited_edge_count;
  metadata[1] = visited_vertex_count;
  metadata[2] = push_noop_count;
  metadata[3] = pop_noop_count;
  metadata[4] = cycle_count;

vertex_cache_stat:
  for (int i = 0; i < kSubIntervalCount; ++i) {
    for (int j = 0; j < kVertexUniStatCount; ++j) {
#pragma HLS pipeline II = 1
      metadata[kGlobalStatCount + i * kVertexUniStatCount + j] =
          vertex_cache_stat_q[i].read();
    }
  }

edge_stat:
  for (int i = 0; i < kShardCount; ++i) {
    for (int j = 0; j < kEdgeUnitStatCount; ++j) {
#pragma HLS pipeline II = 1
      metadata[kGlobalStatCount + kSubIntervalCount * kVertexUniStatCount +
               i * kEdgeUnitStatCount + j] = edge_stat_q[i].read();
    }
  }

queue_stat:
  for (int i = 0; i < kQueueStatCount; ++i) {
#pragma HLS pipeline II = 1
    metadata[kGlobalStatCount + kShardCount * kEdgeUnitStatCount +
             kSubIntervalCount * kVertexUniStatCount + i] = queue_stat_q.read();
  }
}

void SSSP(Task root, tapa::mmap<int64_t> metadata,
          tapa::mmaps<EdgeVec, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices, bool is_log_bucket,
          float min_distance, float max_distance,
          tapa::mmaps<SpilledTaskPerMem, kCgpqPhysMemCount> cgpq_spill) {
  streams<Task, kSubIntervalCount, 2> vertex_out_q;
  streams<Task, kSubIntervalCount, 2> VAR(vertex_out_qx);
  streams<Task, kShardCount, 2> VAR(vertex_out_qi);
  stream<bool, 2> VAR(queue_done_q);
  stream<int32_t, 2> VAR(queue_stat_q);

  streams<uint_spill_addr_t, kCgpqPhysMemCount, 2> VAR(cgpq_spill_read_addr_q);
  streams<SpilledTaskPerMem, kCgpqPhysMemCount, 2> VAR(cgpq_spill_read_data_q);
  streams<packet<uint_spill_addr_t, SpilledTaskPerMem>, kCgpqPhysMemCount, 2>
      VAR(cgpq_spill_write_req_q);
  streams<bool, kCgpqPhysMemCount, 2> VAR(cgpq_spill_write_resp_q);

  streams<Task, kPeCount, 2> task_req_qi("task_req_i");

  stream<Task, 2> task_init_q;
  streams<TaskOnChip, kPopSwitchPortCount*(kPopSwitchStageCount + 1), 32> VAR(
      pop_xbar_q);
  streams<TaskOnChip, kSpilledTaskVecLen, 32> VAR(queue_pop_q);

  // For edges.
  streams<Vid, kShardCount, 2> edge_read_addr_q("edge_read_addr");
  streams<EdgeVec, kShardCount, 2> VAR(edge_read_data_q);
  streams<SourceVertex, kShardCount, 64> src_q("source_vertices");

  streams<TaskOnChip, kSwitchPortCount, 2> VAR(xbar_in_q);
  streams<TaskOnChip,
          kSwitchPortCount * kSwitchMuxDegree*(kSwitchStageCount + 1), 32>
      VAR(xbar_q);
  streams<TaskOnChip, kSwitchPortCount * kSwitchMuxDegree, 2> VAR(xbar_out_q);
  streams<TaskOnChip, kSwitchPortCount, 2> VAR(xbar_out_qx);

  streams<TaskOnChip, kSubIntervalCount, 32> VAR(vertex_in_q);
  //   Connect the vertex readers and updaters.
  streams<Vid, kSubIntervalCount, 2> vertex_read_addr_q;
  streams<Vertex, kSubIntervalCount, 2> vertex_read_data_q;
  streams<packet<Vid, Vertex>, kSubIntervalCount, 2> vertex_write_req_q;
  streams<bool, kSubIntervalCount, 2> vertex_write_resp_q;
  streams<Vid, kSubIntervalCount, 32> read_vid_q;
  streams<Vid, kSubIntervalCount, 32> write_vid_q;
  streams<bool, kSubIntervalCount, 2> vertex_cache_done_q;
  streams<int32_t, kSubIntervalCount, 2> vertex_cache_stat_q;

  streams<bool, kShardCount, 2> edge_done_q;
  streams<int64_t, kShardCount, 2> edge_stat_q;

  stream<VertexNoop, 2> VAR(vertex_noop_q);
  streams<bool, kSubIntervalCount, 2> VAR(vertex_noop_qi);

  streams<TaskOnChip, kSubIntervalCount, 2> VAR(vertex_filter_out_q);

  streams<uint_vid_t, kPeCount, 2> task_count_qi;
  stream<TaskCount, 2> task_count_q;

  streams<PushReq, kCgpqPushPortCount, 2> VAR(cgpq_push_req_q);
  streams<PushReq,
          kCgpqPushPortCount * kSwitchMuxDegree*(kCgpqPushStageCount + 1), 32>
      VAR(cgpq_xbar_q);
  streams<PushReq, kCgpqPushPortCount * kSwitchMuxDegree, 32> VAR(
      cgpq_xbar_out_qi);
  streams<PushReq, kCgpqPushPortCount, 32> VAR(cgpq_xbar_out_q);

  streams<SpilledTask, kCgpqPushPortCount, 2> VAR(cgpq_pop_qi);

  streams<CgpqHeapReq, kCgpqPushPortCount, 2> VAR(cgpq_heap_req_q);
  streams<cgpq::ChunkRef, kCgpqPushPortCount, 2> VAR(cgpq_heap_resp_q);

  streams<uint_spill_addr_t, kCgpqPushPortCount, 2> VAR(cgpq_read_addr_qi);
  streams<SpilledTask, kCgpqPushPortCount, 2> VAR(cgpq_read_data_qi);
  streams<packet<uint_spill_addr_t, SpilledTask>, kCgpqPushPortCount, 2> VAR(
      cgpq_write_req_qi);
  streams<bool, kCgpqPushPortCount, 2> VAR(cgpq_write_resp_qi);
  streams<cgpq::uint_bank_t, kCgpqLogicMemCount, 64> VAR(cgpq_read_id_q);
  streams<cgpq::uint_bank_t, kCgpqLogicMemCount, 64> VAR(cgpq_write_id_q);

  streams<cgpq::uint_bid_t, kCgpqPushPortCount, 2> VAR(cgpq_min_bid_req_q);
  streams<cgpq::uint_bid_t, kCgpqPushPortCount, 2> VAR(cgpq_min_bid_resp_q);

  streams<bool, kCgpqPushPortCount * 2 + 1, 2> VAR(cgpq_done_qi);
  streams<int32_t, kCgpqPushPortCount, 2> VAR(cgpq_stat_qi);

  tapa::task()
      .invoke(  //
          Dispatcher, root, metadata, vertex_cache_done_q, vertex_cache_stat_q,
          edge_done_q, edge_stat_q, queue_done_q, queue_stat_q, task_init_q,
          vertex_noop_q, task_count_q)
      .invoke<detach>(TaskArbiter, task_init_q, vertex_out_qi, task_req_qi)
      .invoke<join, kCgpqPushPortCount>(
          CgpqBucketGen, is_log_bucket, min_distance, max_distance,
          cgpq_done_qi, vertex_filter_out_q, cgpq_push_req_q)
      .invoke<detach, kCgpqPushPortCount>(CgpqSwitchDemux, cgpq_push_req_q,
                                          cgpq_xbar_q)
      .invoke<detach, kSwitchMuxDegree * kCgpqPushStageCount>(
          CgpqSwitchStage, seq(), cgpq_xbar_q, cgpq_xbar_q)
      .invoke<detach>(CgpqPushAdapter, cgpq_xbar_q, cgpq_xbar_out_qi)
      .invoke<detach, kCgpqPushPortCount>(  //
          CgpqSwitchMux, cgpq_xbar_out_qi, cgpq_xbar_out_q)
      .invoke<detach, kCgpqPushPortCount>(
          CgpqCore, cgpq_done_qi, cgpq_stat_qi, seq(),  //
          cgpq_min_bid_req_q, cgpq_min_bid_resp_q,      //
          cgpq_xbar_out_q, cgpq_pop_qi,                 //
          cgpq_heap_req_q, cgpq_heap_resp_q,            //
          cgpq_read_addr_qi, cgpq_read_data_qi,         //
          cgpq_write_req_qi, cgpq_write_resp_qi)
      .invoke(  //
          CgpqOutputArbiter, cgpq_done_qi, cgpq_pop_qi, queue_pop_q)
      .invoke<detach, kCgpqPushPortCount>(  //
          CgpqHeap, cgpq_heap_req_q, cgpq_heap_resp_q)
      .invoke<detach>(  //
          CgpqMinBucketFinder, cgpq_min_bid_req_q, cgpq_min_bid_resp_q)
      .invoke<detach, kCgpqLogicMemCount>(  //
          CgpqReadAddrArbiter, cgpq_read_addr_qi, cgpq_read_id_q,
          cgpq_spill_read_addr_q)
      .invoke<detach, kCgpqLogicMemCount>(  //
          CgpqReadDataArbiter, cgpq_read_id_q, cgpq_spill_read_data_q,
          cgpq_read_data_qi)
      .invoke<detach, kCgpqLogicMemCount>(  //
          CgpqWriteReqArbiter, cgpq_write_req_qi, cgpq_write_id_q,
          cgpq_spill_write_req_q)
      .invoke<detach, kCgpqLogicMemCount>(  //
          CgpqWriteRespArbiter, cgpq_write_id_q, cgpq_spill_write_resp_q,
          cgpq_write_resp_qi)
      .invoke<detach>(CgpqDuplicateDone, queue_done_q, cgpq_done_qi)
      .invoke<detach>(CgpqStatArbiter, cgpq_stat_qi, queue_stat_q)
      .invoke<detach, kSpilledTaskVecLen>(SwitchDemux, queue_pop_q, pop_xbar_q)
      .invoke<detach, kPopSwitchStageCount>(  //
          PopSwitchStage, seq(), pop_xbar_q, pop_xbar_q)
      .invoke<detach, kPopSwitchPortCount>(  //
          CgpqPopAdapter, pop_xbar_q, vertex_in_q)
      .invoke<detach>(VertexOutputAdapter, vertex_out_q, vertex_out_qx)
      .invoke<detach, kShardCount>(VertexOutputArbiter, vertex_out_qx,
                                   vertex_out_qi)

      // Put mmaps are in the top level to enable flexible floorplanning.
      .invoke<detach, kCgpqPhysMemCount>(
          CgpqSpillMem, cgpq_spill_read_addr_q, cgpq_spill_read_data_q,
          cgpq_spill_write_req_q, cgpq_spill_write_resp_q, cgpq_spill)

      // For edges.
      .invoke<join, kShardCount>(EdgeMem, edge_done_q, edge_stat_q,
                                 edge_read_addr_q, edge_read_data_q, edges)

      // For vertices.
      // Route updates via a kShardCount x kShardCount network.
      .invoke<detach, kShardCount * kEdgeVecLen>(SwitchDemux, xbar_in_q, xbar_q)
      .invoke<detach, kSwitchMuxDegree * kSwitchStageCount>(  //
          SwitchStage, seq(), xbar_q, xbar_q)
      .invoke<detach>(PushAdapter, xbar_q, xbar_out_q)
      .invoke<detach, kSubIntervalCount>(SwitchMux, xbar_out_q, xbar_out_qx)
      .invoke<detach>(VertexNoopMerger, vertex_noop_qi, vertex_noop_q)

      .invoke<detach, kSubIntervalCount>(VertexMem, vertex_read_addr_q,
                                         vertex_read_data_q, vertex_write_req_q,
                                         vertex_write_resp_q, vertices)
      .invoke<detach, kSubIntervalCount>(
          VertexCache, vertex_cache_done_q, vertex_cache_stat_q,
          //
          xbar_out_qx, vertex_filter_out_q,
          //
          vertex_in_q, vertex_out_q,
          //
          vertex_noop_qi,
          //
          read_vid_q, read_vid_q, write_vid_q, write_vid_q, vertex_read_addr_q,
          vertex_read_data_q, vertex_write_req_q, vertex_write_resp_q)

      // PEs.
      .invoke<detach, kPeCount>(EdgeReqGen, task_req_qi, task_count_qi, src_q,
                                edge_read_addr_q)
      .invoke<detach>(TaskCountMerger, task_count_qi, task_count_q)
      .invoke<detach, kShardCount>(DistGen, src_q, edge_read_data_q, xbar_in_q);
}

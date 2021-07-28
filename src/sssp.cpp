#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-cgpq.h"
#include "sssp-kernel.h"
#include "sssp-piheap-index.h"
#include "sssp-piheap.h"
#include "sssp.h"

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
using cgpq::uint_spill_addr_t;

static_assert(kQueueCount % kShardCount == 0,
              "current implementation requires that queue count is a multiple "
              "of shard count");

// Verbosity definitions:
//   v=5: O(1)
//   v=8: O(#vertex)
//   v=9: O(#edge)

void PiHeapIndexReqArbiter(istreams<HeapIndexReq, kLevelCount>& req_in_q,
                           ostream<IndexStateUpdate>& state_q,
                           ostream<packet<LevelId, HeapIndexReq>>& req_out_q) {
spin:
  for (ap_uint<kLevelCount> priority = 1;; priority.lrotate(1)) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    LevelId level;
    if (find_non_empty(req_in_q, priority, level)) {
      const auto req = req_in_q[level].read(nullptr);
      req_out_q.write({level, req});
      state_q.write({.level = level, .op = req.op});
    }
  }
}

void PiHeapIndexRespArbiter(istream<packet<LevelId, HeapIndexResp>>& req_in_q,
                            ostreams<HeapIndexResp, kLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    if (req_in_q.empty()) continue;
    const auto resp = req_in_q.read(nullptr);
    req_out_q[resp.addr].write(resp.payload);
  }
}

void PiHeapIndex(
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    //
    ostream<IndexStateUpdate>& state_q,
    //
    uint_qid_t qid,
    //
    istream<AcquireIndexReq>& acquire_index_req_q,
    istream<packet<LevelId, HeapIndexReq>>& req_q,
    ostream<packet<LevelId, HeapIndexResp>>& resp_q,
    //
    ostream<Vid>& read_addr_q, istream<HeapIndexEntry>& read_data_q,
    ostream<packet<Vid, HeapIndexEntry>>& write_req_q,
    istream<bool>& write_resp_q,
    //
    ostream<HeapAcquireIndexContext>& acquire_index_ctx_out_q,
    istream<HeapAcquireIndexContext>& acquire_index_ctx_in_q) {
#pragma HLS inline recursive

  HeapIndexCacheEntry fresh_index[kPifcSize];
#pragma HLS bind_storage variable = fresh_index type = RAM_S2P impl = URAM
#pragma HLS aggregate variable = fresh_index bit
init:
  for (int i = 0; i < kPifcSize; ++i) {
    fresh_index[i] = {
        .is_dirty = false,
        .tag = 0,
        .index = nullptr,
    };
  }

  DECL_ARRAY(HeapStaleIndexEntry, stale_index, kPiscSize, nullptr);
#pragma HLS aggregate variable = stale_index bit
  auto SetStaleIndexLocked = [&stale_index](uint_pisc_pos_t idx, Vid vid,
                                            const HeapIndexEntry& entry) {
    CHECK(!stale_index[idx].valid() || stale_index[idx].matches(vid));
    stale_index[idx].set(vid, entry);
    VLOG(5) << "INDX q[" << (vid % kQueueCount) << "] v[" << vid << "] -> "
            << entry << " (stale)";
  };

  DECL_ARRAY(PiHeapStat, op_stats, kPiHeapStatCount[1], 0);
  auto& read_hit = op_stats[0];
  auto& read_miss = op_stats[1];
  auto& write_hit = op_stats[2];
  auto& write_miss = op_stats[3];
  auto& idle_count = op_stats[4];
  auto& busy_count = op_stats[5];

  ap_uint<bit_length(kPiHeapStatCount[1])> stat_idx = 0;

  shiftreg<uint_pifc_index_t, 16> reading_fresh_pos;
  shiftreg<uint_pifc_index_t, 16> writing_fresh_pos;

  bool is_started = false;

spin:
  for (;;) {
#pragma HLS pipeline II = 2

    // Determine what to do.
    const bool can_send_stat = !done_q.empty();

    const bool can_collect_write = !write_resp_q.empty();

    bool is_ctx_valid;
    const auto ctx = acquire_index_ctx_in_q.peek(is_ctx_valid);
    const bool can_acquire_index =
        // Context must be valid.
        is_ctx_valid &&
        // Read data must be valid.
        !read_data_q.empty();

    const bool is_write_req_full =
        writing_fresh_pos.full() || write_req_q.full();

    bool is_req_valid;
    const auto pkt = req_q.peek(is_req_valid);
    const auto req = pkt.payload;
    const auto req_pos = GetPifcIndex(req.vid);
    const bool can_process_req =
        // Request must be valid.
        is_req_valid &&
        // The request must not violate fresh index dependency.
        (
            // Get stale and clear stale won't create any violation.
            req.op == GET_STALE || req.op == CLEAR_STALE ||
            !(
                // Outstanding reads must not be accessed.
                reading_fresh_pos.contains(req_pos) ||
                // Outstanding writes must not be accessed.
                writing_fresh_pos.contains(req_pos) ||
                // Memory write must not block.
                is_write_req_full));

    bool is_acquire_index_req_valid;
    const auto acquire_index_req =
        acquire_index_req_q.peek(is_acquire_index_req_valid);

    // Determine which vertex to work on.
    const auto vid = can_acquire_index ? ctx.vid
                     : can_process_req ? req.vid
                                       : acquire_index_req.vid;
    const auto pifc_index = GetPifcIndex(vid);

    // Read fresh entry.
    auto fresh_entry = fresh_index[pifc_index];
    const bool is_fresh_entry_hit = fresh_entry.IsHit(vid);

    // Read stale entry.
    bool is_stale_entry_pos_valid;
    uint_pisc_pos_t stale_entry_pos;
    auto stale_entry = GetStaleIndexLocked(
        stale_index, vid, is_stale_entry_pos_valid, stale_entry_pos);
    bool is_stale_entry_updated = false;

    const bool can_process_acquire_index_req =
        is_acquire_index_req_valid && is_stale_entry_pos_valid &&
        !stale_entry.valid() &&
        !(
            // Outstanding reads must not be accessed.
            reading_fresh_pos.contains(pifc_index) ||
            // Outstanding writes must not be accessed.
            writing_fresh_pos.contains(pifc_index) ||
            // Memory write must not block.
            is_write_req_full ||
            // memory read must not block; and
            reading_fresh_pos.full() || read_addr_q.full() ||
            // context write must not block.
            acquire_index_ctx_out_q.full());

    // Prepare response.
    HeapIndexResp resp;
    bool is_resp_written = false;

    if (can_send_stat) {
      stat_q.write(op_stats[stat_idx]);
      op_stats[stat_idx] = 0;
      if (stat_idx == kPiHeapStatCount[1] - 1) {
        stat_idx = 0;
        done_q.read(nullptr);
      } else {
        ++stat_idx;
      }
      is_started = false;
    } else if (can_collect_write) {
      write_resp_q.read(nullptr);
      writing_fresh_pos.pop();
    } else if (can_acquire_index) {
      CHECK(is_stale_entry_pos_valid);
      is_resp_written = true;

      acquire_index_ctx_in_q.read(nullptr);

      CHECK(!stale_entry.valid());
      CHECK(!read_data_q.empty());
      const auto reading_pos = reading_fresh_pos.pop();
      CHECK_EQ(reading_pos, GetPifcIndex(ctx.vid));

      const auto read_data = read_data_q.read(nullptr);

      if (read_data.distance_le(ctx.entry)) {  // Implies read_data.valid().
        resp = {.entry = {}, .enable = false};

        fresh_entry.is_dirty = false;
        fresh_entry.UpdateTag(ctx.vid);
        fresh_entry.index = read_data;
      } else {
        if (read_data.valid()) {
          stale_entry = read_data;
          is_stale_entry_updated = true;
        }

        resp = {.entry = read_data, .enable = true};

        fresh_entry.is_dirty = true;
        fresh_entry.UpdateTag(ctx.vid);
        fresh_entry.index = ctx.entry;
      }
    } else if (can_process_req) {
      is_started = true;
      is_resp_written = true;

      req_q.read(nullptr);

      switch (req.op) {
        case GET_STALE: {
          CHECK(is_stale_entry_pos_valid);
          resp.entry = stale_entry;
        } break;
        case CLEAR_STALE: {
          CHECK(is_stale_entry_pos_valid);
          CHECK(stale_entry.valid());
          stale_entry.invalidate();
          is_stale_entry_updated = true;
        } break;
        case UPDATE_INDEX: {
          if (stale_entry.distance_eq(req.entry)) {
            CHECK(is_stale_entry_pos_valid);
            stale_entry = req.entry;
            is_stale_entry_updated = true;
          } else {
            // Write cache entry to memory if necessary.
            if (!is_fresh_entry_hit && fresh_entry.is_dirty) {
              write_req_q.try_write(
                  {fresh_entry.GetVid(pifc_index, qid), fresh_entry.index});
              writing_fresh_pos.push(pifc_index);

              ++write_miss;
            } else {
              ++write_hit;
            }

            fresh_entry.is_dirty = true;
            fresh_entry.UpdateTag(req.vid);
            fresh_entry.index = req.entry;
          }
        } break;
        case CLEAR_FRESH: {
          if (is_fresh_entry_hit) {
            fresh_entry.is_dirty = true;
            fresh_entry.index.invalidate();

            ++write_hit;
          } else {
            write_req_q.try_write({req.vid, nullptr});
            writing_fresh_pos.push(pifc_index);

            ++write_miss;
          }
        } break;
      }
      state_q.write({.level = pkt.addr, .op = req.op});
    } else if (can_process_acquire_index_req) {
      acquire_index_req_q.read(nullptr);
      const HeapIndexEntry entry(/*level=*/0, /*index=*/0,
                                 acquire_index_req.distance);

      // Fetch entry from memory on miss.
      if (!is_fresh_entry_hit) {
        is_resp_written = false;

        // Write cache entry to memory if necessary.
        if (fresh_entry.is_dirty) {
          write_req_q.try_write(
              {fresh_entry.GetVid(pifc_index, qid), fresh_entry.index});
          writing_fresh_pos.push(pifc_index);

          ++write_miss;
        } else {
          ++write_hit;
        }

        // Issue read request and save context.
        read_addr_q.try_write(vid);
        reading_fresh_pos.push(pifc_index);
        acquire_index_ctx_out_q.try_write({.vid = vid, .entry = entry});

        ++read_miss;
      } else {  // Read hit.
        is_resp_written = true;
        if (fresh_entry.index.distance_le(entry)) {
          // Implies fresh_entry.index.valid().
          resp = {.entry = {}, .enable = false};
        } else {
          if (fresh_entry.index.valid()) {
            stale_entry = fresh_entry.index;
            is_stale_entry_updated = true;
          }

          resp = {.entry = fresh_entry.index, .enable = true};

          fresh_entry.is_dirty = true;
          fresh_entry.index = entry;
        }

        ++read_hit;
      }
    } else if (is_req_valid || is_acquire_index_req_valid) {
      ++busy_count;
    } else if (is_started) {
      ++idle_count;
    }

    if (is_resp_written) {
      resp_q.write({
          .addr = !can_acquire_index && can_process_req
                      ? pkt.addr
                      : LevelId(0),  // Only the head can acquire index.
          .payload = resp,
      });
    }

    if (!can_send_stat && !can_collect_write &&
        (can_acquire_index || can_process_req ||
         can_process_acquire_index_req)) {
      fresh_index[pifc_index] = fresh_entry;
    }

    if (is_stale_entry_updated) {
      SetStaleIndexLocked(stale_entry_pos, vid, stale_entry);
    }
  }
}

void PiHeapPerf(
    //
    uint_qid_t qid,
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    //
    istream<QueueStateUpdate>& queue_state_q,
    istream<IndexStateUpdate>& index_req_state_q,
    istream<IndexStateUpdate>& index_resp_state_q) {
  /*
    States w/ index
      + IDLE: queue is idle
        - push_started -> PUSH
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE
      + PUSH: queue is acquring index for INDEX
        - push_started -> ERROR
        - pop_started -> ERROR
        - index_acquired -> INDEX
        - idle -> ERROR
      + POP: queue is processing POP
        - push_started -> PUSH
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE
      + INDEX: queue is processing PUSH with index already acquired
        - push_started -> PUSH
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE

    States w/o index
      + IDLE: queue is idle
        - push_started -> PUSH
        - pop_started -> POP
        - pushpop_started -> PUSHPOP
        - idle -> IDLE
      + PUSH: queue is processing PUSH with index already acquired
        - push_started -> PUSH
        - pop_started -> POP
        - pushpop_started -> PUSHPOP
        - idle -> IDLE
      + POP: queue is processing POP
        - push_started -> PUSH
        - pop_started -> POP
        - pushpop_started -> PUSHPOP
        - idle -> IDLE
      + PUSHPOP: queue is acquring index for PUSH
        - push_started -> PUSH
        - pop_started -> POP
        - pushpop_started -> PUSHPOP
        - idle -> IDLE
  */

  /*
    Each level sends an IndexStateUpdate whenever a request is started.
    The index sends an IndexStateUpdate when the response is sent.
    The state of index is maintained per-level, since different levels may have
    outstanding requests at the same time.
   */

  constexpr int kQueueStateCountSize = 4;
  constexpr int kQueueOpCountSize = kQueueStateCountSize - 1;

exec:
  for (;;) {
    DECL_ARRAY(PiHeapStat, queue_state_count, kQueueStateCountSize, 0);
    DECL_ARRAY(PiHeapStat, queue_op_count, kQueueOpCountSize, 0);

    PiHeapStat index_state_count[kLevelCount][kPiHeapIndexOpTypeCount];
    PiHeapStat index_op_count[kLevelCount][kPiHeapIndexOpTypeCount];
#pragma HLS array_partition complete variable = index_state_count dim = 0
#pragma HLS array_partition complete variable = index_op_count dim = 0
    RANGE(i, kLevelCount, RANGE(j, kPiHeapIndexOpTypeCount, {
            index_state_count[i][j] = 0;
            index_op_count[i][j] = 0;
          }));

    bool is_started = false;

    using namespace queue_state;
    QueueState queue_state = IDLE;

    uint_piheap_size_t current_size = 0;
    uint_piheap_size_t max_size = 0;
    ap_uint<64> total_size = 0;

    DECL_ARRAY(bool, is_index_busy, kLevelCount, false);
    DECL_ARRAY(HeapOp, index_state, kLevelCount, HeapOp());

  spin:
    for (int64_t cycle_count = 0; done_q.empty(); ++cycle_count) {
#pragma HLS pipeline II = 1
      if (is_started) {
        ++queue_state_count[queue_state];
        RANGE(level, kLevelCount, {
          if (is_index_busy[level]) {
            ++index_state_count[level][index_state[level]];
          }
        });
        total_size += current_size;
        max_size = std::max(max_size, current_size);
      }
      if (!queue_state_q.empty()) {
        const auto update = queue_state_q.read(nullptr);
        if (update.state != IDLE) {
          is_started = true;
          ++queue_op_count[update.state - 1];
        }
        current_size = update.size;
        queue_state = update.state;
      }
      if (!index_req_state_q.empty() &&
          !is_index_busy[index_req_state_q.peek(nullptr).level]) {
        const auto update = index_req_state_q.read(nullptr);
        is_index_busy[update.level] = true;
        index_state[update.level] = update.op;
        ++index_op_count[update.level][update.op];
      }
      if (!index_resp_state_q.empty() &&
          is_index_busy[index_resp_state_q.peek(nullptr).level]) {
        const auto update = index_resp_state_q.read(nullptr);
        CHECK_EQ(index_state[update.level], update.op);
        is_index_busy[update.level] = false;
      }
    }

    done_q.read(nullptr);
    for (int i = 0; i < kQueueStateCountSize; ++i) {
#pragma HLS unroll
      stat_q.write(queue_state_count[i]);
    }
    for (int i = 0; i < kQueueOpCountSize; ++i) {
#pragma HLS unroll
      stat_q.write(queue_op_count[i]);
    }
    stat_q.write(max_size);
    stat_q.write(total_size.range(63, 32));
    stat_q.write(total_size.range(31, 0));

    for (int level = 0; level < kLevelCount; ++level) {
#pragma HLS unroll
      for (int op = 0; op < kPiHeapIndexOpTypeCount; ++op) {
#pragma HLS unroll
        stat_q.write(index_state_count[level][op]);
        stat_q.write(index_op_count[level][op]);
      }
    }
  }
}

void PiHeapHead(
    // Scalar
    uint_qid_t qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q,
    // NOOP acknowledgements
    ostream<bool>& noop_q,
    // Queue outputs.
    ostream<TaskOnChip>& pop_q,
    // Internal
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q,
    // Statistics.
    ostream<QueueStateUpdate>& queue_state_q,
    // Heap index
    ostream<AcquireIndexReq>& acquire_index_req_q,
    ostream<HeapIndexReq>& index_req_q, istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline recursive
  const auto cap = GetChildCapOfLevel(0);
  HeapElem<0> root{.valid = false};
#pragma HLS array_partition variable = root.cap complete
  RANGE(i, kPiHeapWidth, root.cap[i] = cap);
  root.size = 0;

  CLEAN_UP(clean_up, [&] {
    CHECK_EQ(root.size, 0);
    CHECK_EQ(root.valid, false) << "q[" << qid << "]";
    RANGE(i, kPiHeapWidth, CHECK_EQ(root.cap[i], cap) << "q[" << qid << "]");
  });

spin:
  for (;;) {
#pragma HLS pipeline off

#ifdef TAPA_SSSP_PHEAP_INDEX
#ifdef TAPA_SSSP_PRIORITIZE_PUSH
    const bool do_push = !push_req_q.empty();
    const bool do_pop = !do_push && !pop_q.full() && root.valid;
#else   // TAPA_SSSP_PRIORITIZE_PUSH
    const bool do_pop = !pop_q.full() && root.valid;
    const bool do_push = !do_pop && !push_req_q.empty();
#endif  // TAPA_SSSP_PRIORITIZE_PUSH
#else   // TAPA_SSSP_PHEAP_INDEX
    const bool do_push = !push_req_q.empty();
    const bool do_pop = !pop_q.full() && (do_push || root.valid);
#endif  // TAPA_SSSP_PHEAP_INDEX

    const auto push_req = do_push ? push_req_q.read(nullptr) : TaskOnChip();

    queue_state_q.write({
        .state = do_push ? (do_pop ? QueueState::PUSHPOP : QueueState::PUSH)
                         : (do_pop ? QueueState::POP : QueueState::IDLE),
        .size = root.size,
    });

    VLOG_IF(5, do_push) << "PUSH q[" << qid << "] <-  " << push_req;
    VLOG_IF(5, do_pop) << "POP  q[" << qid << "]  -> "
                       << (!root.valid || (do_push && root.task <= push_req)
                               ? push_req
                               : root.task);

    {
      if (do_push && !do_pop) {
        HeapIndexResp heap_index_resp;
#ifdef TAPA_SSSP_PHEAP_INDEX
        acquire_index_req_q.write({
            .vid = push_req.vid(),
            .distance = push_req.vertex().distance,
        });
        ap_wait();
        heap_index_resp = index_resp_q.read();
        queue_state_q.write({QueueState::INDEX, root.size});
#else   // TAPA_SSSP_PHEAP_INDEX
        heap_index_resp.entry.invalidate();
        heap_index_resp.enable = true;
        {
          const bool is_full = acquire_index_req_q.full();
          CHECK(!is_full);
        }
        {
          const bool is_full = noop_q.full();
          CHECK(!is_full);
        }
#endif  // TAPA_SSSP_PHEAP_INDEX
        if (!heap_index_resp.enable || heap_index_resp.entry.valid()) {
          noop_q.write(false);
        }
        if (heap_index_resp.enable) {
          PiHeapPush(qid, /*level=*/0,
                     {
                         .index = 0,
                         .op = QueueOp::PUSH,
                         .task = push_req,
                         .replace = heap_index_resp.entry.valid(),
                         .vid = push_req.vid(),
                     },
                     root, resp_in_q, req_out_q, index_req_q, index_resp_q);
        }
      } else if (do_pop) {
        const bool is_update_needed =
            root.valid && !(do_push && root.task <= push_req);

        const auto resp_task = is_update_needed ? root.task : push_req;
        CHECK_EQ(resp_task.vid() % kQueueCount, qid);
        pop_q.try_write(resp_task);

        if (is_update_needed) {
#ifdef TAPA_SSSP_PHEAP_INDEX
          index_req_q.write({.op = CLEAR_FRESH, .vid = root.task.vid()});
          ap_wait();
          index_resp_q.read();
#endif  // TAPA_SSSP_PHEAP_INDEX

          PiHeapPop(
              {
                  .op = do_push ? QueueOp::PUSHPOP : QueueOp::POP,
                  .task = push_req,
              },
              0, root, resp_in_q, req_out_q);
        }
      }
    }
  }
}

void PiHeapBodyOffChip(
    //
    uint_qid_t qid,
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q,
    // Child level
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q,
    //
    ostream<HeapIndexReq>& index_req_q, istream<HeapIndexResp>& index_resp_q,
    //
    LevelId level,
    //
    ostream<Vid>& read_addr_q, istream<HeapElemPacked>& read_data_q,
    ostream<packet<Vid, HeapElemPacked>>& write_req_q,
    istream<bool>& write_resp_q) {
#pragma HLS inline recursive
  const auto cap = GetChildCapOfLevel(level);

  // Store outstanding indices (divided by kPiHeapWidth).
  shiftreg<LevelIndex, 16> writing_index;

spin:
  for (bool is_first_time = true;;) {
#pragma HLS pipeline off
    if (!write_resp_q.empty()) {
      write_resp_q.read(nullptr);
      writing_index.pop();
    }

    bool is_req_valid = false;
    const auto req = req_in_q.peek(is_req_valid);
    if (is_req_valid && !(writing_index.contains(req.index / kPiHeapWidth) ||
                          writing_index.full() || write_req_q.full())) {
      req_in_q.read(nullptr);
      LOG_IF(INFO, is_first_time)
          << "q[" << qid << "]: off-chip level " << level << " accessed";
      is_first_time = false;

      const ap_uint<bit_length(kPiHeapWidth)> elem_count =
          req.op == QueueOp::PUSH ? 1 : kPiHeapWidth / 2;
    read_elems:
      for (ap_uint<bit_length(kPiHeapWidth)> i = 0; i < elem_count; ++i) {
#pragma HLS pipeline II = 1 rewind
        read_addr_q.write(
            GetAddrOfOffChipHeapElem(level, req.index + i * 2, qid));
      }

      // Outputs from IsPiHeapElemUpdated.
      LevelIndex idx;
      HeapElemAxi elem_pair[2];
#pragma HLS array_partition variable = elem_pair complete
#pragma HLS array_partition variable = elem_pair[0].cap complete
#pragma HLS array_partition variable = elem_pair[1].cap complete
      if (IsPiHeapElemUpdated(qid, level, req, read_data_q, idx, elem_pair,
                              req_in_q, resp_out_q, req_out_q, resp_in_q,
                              index_req_q, index_resp_q)) {
        write_req_q.try_write({GetAddrOfOffChipHeapElem(level, idx, qid),
                               HeapElemAxi::Pack(elem_pair)});
        writing_index.push(idx / kPiHeapWidth);
      }
    }
  }
}

void PiHeapDummyTail(
    // Scalar
    uint_qid_t qid,
    // Parent level
    istream<HeapReq>& req_in_q, ostream<HeapResp>& resp_out_q) {
spin:
  for (;;) {
    const bool is_empty = req_in_q.empty();
    const bool is_full = resp_out_q.full();
    CHECK(is_empty);
    CHECK(!is_full);
  }
}

#define HEAP_PORTS                                                         \
  uint_qid_t qid, istream<HeapReq>&req_in_q, ostream<HeapResp>&resp_out_q, \
      ostream<HeapReq>&req_out_q, istream<HeapResp>&resp_in_q,             \
      ostream<HeapIndexReq>&index_req_out_q,                               \
      istream<HeapIndexResp>&index_resp_in_q
#define HEAP_BODY(level, mem)                                                 \
  _Pragma("HLS inline recursive");                                            \
  HeapElem<level> heap_array[GetCapOfLevel(level)];                           \
  _Pragma("HLS aggregate variable = heap_array bit");                         \
  DO_PRAGMA(HLS bind_storage variable = heap_array type = RAM_2P impl = mem); \
  PiHeapBody<level>(qid, heap_array, req_in_q, resp_out_q, req_out_q,         \
                    resp_in_q, index_req_out_q, index_resp_in_q)

#if TAPA_SSSP_PHEAP_WIDTH == 2
void PiHeapBodyL1(HEAP_PORTS) { HEAP_BODY(1, LUTRAM); }
void PiHeapBodyL2(HEAP_PORTS) { HEAP_BODY(2, LUTRAM); }
void PiHeapBodyL3(HEAP_PORTS) { HEAP_BODY(3, LUTRAM); }
void PiHeapBodyL4(HEAP_PORTS) { HEAP_BODY(4, LUTRAM); }
void PiHeapBodyL5(HEAP_PORTS) { HEAP_BODY(5, LUTRAM); }
void PiHeapBodyL6(HEAP_PORTS) { HEAP_BODY(6, LUTRAM); }
void PiHeapBodyL7(HEAP_PORTS) { HEAP_BODY(7, LUTRAM); }
void PiHeapBodyL8(HEAP_PORTS) { HEAP_BODY(8, LUTRAM); }
void PiHeapBodyL9(HEAP_PORTS) { HEAP_BODY(9, BRAM); }
void PiHeapBodyL10(HEAP_PORTS) { HEAP_BODY(10, BRAM); }
void PiHeapBodyL11(HEAP_PORTS) { HEAP_BODY(11, BRAM); }
void PiHeapBodyL12(HEAP_PORTS) { HEAP_BODY(12, URAM); }
void PiHeapBodyL13(HEAP_PORTS) { HEAP_BODY(13, URAM); }
void PiHeapBodyL14(HEAP_PORTS) { HEAP_BODY(14, URAM); }
#elif TAPA_SSSP_PHEAP_WIDTH == 4
void PiHeapBodyL1(HEAP_PORTS) { HEAP_BODY(1, LUTRAM); }
void PiHeapBodyL2(HEAP_PORTS) { HEAP_BODY(2, LUTRAM); }
void PiHeapBodyL3(HEAP_PORTS) { HEAP_BODY(3, LUTRAM); }
void PiHeapBodyL4(HEAP_PORTS) { HEAP_BODY(4, LUTRAM); }
void PiHeapBodyL5(HEAP_PORTS) { HEAP_BODY(5, BRAM); }
void PiHeapBodyL6(HEAP_PORTS) { HEAP_BODY(6, URAM); }
void PiHeapBodyL7(HEAP_PORTS) { HEAP_BODY(7, URAM); }
#elif TAPA_SSSP_PHEAP_WIDTH == 8
void PiHeapBodyL1(HEAP_PORTS) { HEAP_BODY(1, LUTRAM); }
void PiHeapBodyL2(HEAP_PORTS) { HEAP_BODY(2, LUTRAM); }
void PiHeapBodyL3(HEAP_PORTS) { HEAP_BODY(3, BRAM); }
void PiHeapBodyL4(HEAP_PORTS) { HEAP_BODY(4, URAM); }
#elif TAPA_SSSP_PHEAP_WIDTH == 16
void PiHeapBodyL1(HEAP_PORTS) { HEAP_BODY(1, LUTRAM); }
void PiHeapBodyL2(HEAP_PORTS) { HEAP_BODY(2, LUTRAM); }
void PiHeapBodyL3(HEAP_PORTS) { HEAP_BODY(3, URAM); }
#else
#error "invalid TAPA_SSSP_PHEAP_WIDTH"
#endif  // TAPA_SSSP_PHEAP_WIDTH
#undef HEAP_BODY
#undef HEAP_PORTS

void PiHeapArrayMem(
    //
    istream<Vid>& read_addr_q, ostream<HeapElemPacked>& read_data_q,
    istream<packet<Vid, HeapElemPacked>>& write_req_q,
    ostream<bool>& write_resp_q,
    //
    tapa::async_mmap<HeapElemPacked> mem) {
  ReadWriteMem(read_addr_q, read_data_q, write_req_q, write_resp_q, mem);
}

void PiHeapArrayReadAddrArbiter(  //
    istreams<Vid, kOffChipLevelCount>& req_in_q,
    ostream<OffChipLevelId>& req_id_q, ostream<Vid>& req_out_q) {
spin:
  for (ap_uint<kLevelCount> priority = 1;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    LevelId level;
    if (find_non_empty(req_in_q, priority, level) && !req_id_q.full() &&
        !req_out_q.full()) {
      req_id_q.try_write(level);
      req_out_q.try_write(req_in_q[level].read(nullptr));
      priority = 0;
      priority.set(level);  // Make long burst.
    } else {
      priority.lrotate(1);
    }
  }
}

void PiHeapArrayReadDataArbiter(  //
    istream<OffChipLevelId>& req_id_q, istream<HeapElemPacked>& req_in_q,
    ostreams<HeapElemPacked, kOffChipLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    if (!req_id_q.empty() && !req_in_q.empty()) {
      req_out_q[req_id_q.read(nullptr)].write(req_in_q.read(nullptr));
    }
  }
}

void PiHeapArrayWriteReqArbiter(  //
    istreams<packet<Vid, HeapElemPacked>, kOffChipLevelCount>& req_in_q,
    ostream<OffChipLevelId>& req_id_q,
    ostream<packet<Vid, HeapElemPacked>>& req_out_q) {
spin:
  for (OffChipLevelId level = 0;; level = level == kOffChipLevelCount - 1
                                              ? OffChipLevelId(0)
                                              : OffChipLevelId(level + 1)) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    if (!req_in_q[level].empty() && !req_id_q.full() && !req_out_q.full()) {
      req_id_q.try_write(level);
      req_out_q.try_write(req_in_q[level].read(nullptr));
    }
  }
}

void PiHeapArrayWriteRespArbiter(  //
    istream<OffChipLevelId>& req_id_q, istream<bool>& req_in_q,
    ostreams<bool, kOffChipLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    if (!req_id_q.empty() && !req_in_q.empty()) {
      req_out_q[req_id_q.read(nullptr)].write(req_in_q.read(nullptr));
    }
  }
}

void PiHeapIndexMem(
    //
    istream<Vid>& read_addr_q, ostream<HeapIndexEntry>& read_data_q,
    istream<packet<Vid, HeapIndexEntry>>& write_req_q,
    ostream<bool>& write_resp_q,
    //
    tapa::async_mmap<HeapIndexEntry> mem) {
  ReadWriteMem(read_addr_q, read_data_q, write_req_q, write_resp_q, mem);
}

void PiHeapStatArbiter(
    //
    istream<bool>& done_in_q, ostream<PiHeapStat>& stat_out_q,
    //
    ostreams<bool, kPiHeapStatTaskCount>& done_out_q,
    istreams<PiHeapStat, kPiHeapStatTaskCount>& stat_in_q) {
spin:
  for (;;) {
#pragma HLS pipeline off
    const auto done = done_in_q.read();
    RANGE(i, kPiHeapStatTaskCount, done_out_q[i].write(done));
    ap_wait();
    RANGE(i, kPiHeapStatTaskCount, {
      RANGE(j, kPiHeapStatCount[i], stat_out_q.write(stat_in_q[i].read()));
    });
  }
}

void CgpqSpillMem(
    //
    istream<uint_spill_addr_t>& read_addr_q, ostream<SpilledTask>& read_data_q,
    istream<packet<uint_spill_addr_t, SpilledTask>>& write_req_q,
    ostream<bool>& write_resp_q,
    //
    tapa::async_mmap<SpilledTask> mem) {
  ReadWriteMem(read_addr_q, read_data_q, write_req_q, write_resp_q, mem);
}

void CgpqBucketGen(istream<TaskOnChip>& in_q, ostream<PushReq>& out_q) {
  using namespace cgpq;

  const float min_distance = 0.01;
  const float max_distance = 1;

spin:
  for (;;) {
    if (!in_q.empty()) {
      const auto task = in_q.read(nullptr);
      const uint_bid_t bid = std::max(
          std::min(
              int(
#ifdef TAPA_SSSP_CGPQ_LOG_BUCKET
                  (log(task.vertex().distance + 1) - log(min_distance + 1)) /
                  (log(max_distance + 1) - log(min_distance + 1))
#else
                  (task.vertex().distance - min_distance) /
                  (max_distance - min_distance)
#endif
                  * kBucketCount),
              kBucketCount - 1),
          0);

      out_q.write({.bid = bid, .task = task});
    }
  }
}

// Coarse-grained priority queue.
// Implements chunks of buckets and an on-chip heap that manages the chunks.
void CgpqCore(
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    // Scalar
    uint_qid_t qid,
    // Queue requests.
    istream<PushReq>& push_req_q,
    // NOOP acknowledgements
    ostream<bool>& noop_q,
    // Queue outputs.
    ostream<TaskOnChip>& pop_q,
    //
    ostream<uint_spill_addr_t>& mem_read_addr_q,
    istream<TaskOnChip>& mem_read_data_q,
    ostream<packet<uint_spill_addr_t, TaskOnChip>>& mem_write_req_q,
    istream<bool>& mem_write_resp_q) {
#pragma HLS inline recursive

  using namespace cgpq;

  // Heap rule: !(parent < children) (comparing priority).
  // Root is at pos = 1.
  // Parent of pos = pos / 2.
  // Children of pos = pos * 2, pos * 2 + 1.
  ChunkRefPair heap_array[(kCgpqCapacity + 1) / 2];
#pragma HLS bind_storage variable = heap_array type = RAM_T2P impl = URAM
  // Extra copy of heap top in registers due to limited read ports.
  ChunkRef heap_root;
  ap_uint<bit_length(kCgpqCapacity + 1)> heap_pos, heap_size = 0,
                                                   max_heap_size = 0;
  ap_int<bit_length(kCgpqCapacity + 1) + 1> heap_pos_prev = -1;
  ChunkRef heap_elem;
  ChunkRefPair heap_pair_prev;
  bool is_heapifying_up = false;
  bool is_heapifying_up_init = false;
  bool is_heapifying_down = false;
  bool is_heapifying_down_init = false;

  DECL_ARRAY(ChunkMeta, chunk_meta, kBucketCount, ChunkMeta());

  TaskOnChip chunk_buf[kBucketCount][kBufferSize];
#pragma HLS bind_storage variable = chunk_buf type = RAM_S2P impl = URAM
#pragma HLS array_partition variable = chunk_buf cyclic factor = \
    kChunkPartFac dim = 1

  bool is_spill_valid = false;
  uint_bid_t spill_bid;
  uint_spill_addr_t spill_addr_req = 0;
  uint_spill_addr_t spill_addr_resp = 0;

  // Refill requests should read from this address.
  bool is_refill_addr_valid = false;
  uint_spill_addr_t refill_addr;
  // Refill data should write to this bucket.
  bool is_refill_valid = false;
  uint_bid_t refill_bid;
  // Future refill data should write to this bucket.
  bool is_refill_next_valid = false;
  uint_bid_t refill_bid_next;
  // Number of data remaining to refill.
  uint_chunk_size_t refill_remain_count;

spin:
  for (; done_q.empty();) {
#pragma HLS pipeline II = 1
    bool is_heapifying_up_next = is_heapifying_up;
    bool is_heapifying_up_init_next = is_heapifying_up_init;
    bool is_heapifying_down_next = is_heapifying_down;
    bool is_heapifying_down_init_next = is_heapifying_down_init;

#pragma HLS dependence variable = heap_array inter false

    if (is_heapifying_up_init) {
      CHECK(is_heapifying_up);
      CHECK_NE(heap_pos_prev, heap_pos / 2);
      heap_pair_prev = heap_array[heap_pos / 2];
      is_heapifying_up_init_next = false;
      heap_pos_prev = -1;
    } else if (is_heapifying_up) {
      CHECK(!is_heapifying_down);
      const auto heap_pos_next = heap_pos / 2;
      CHECK_NE(heap_pos_prev, heap_pos_next / 2);
      auto heap_pair = heap_pair_prev;
      heap_pair_prev = heap_array[heap_pos_next / 2];
      const auto heap_elem_next = heap_pair_prev[heap_pos_next % 2];
      heap_pos_prev = heap_pos / 2;
      CHECK(ChunkRefPairEq(heap_array[heap_pos / 2], heap_pair));
      if (heap_pos > 1 && heap_elem_next < heap_elem) {
        heap_pair[heap_pos % 2] = heap_elem_next;
        heap_array[heap_pos / 2] = heap_pair;
        if (heap_pos == 1) {
          heap_root = heap_elem_next;
        }
        heap_pos = heap_pos_next;
      } else {
        heap_pair[heap_pos % 2] = heap_elem;
        heap_array[heap_pos / 2] = heap_pair;
        if (heap_pos == 1) {
          heap_root = heap_elem;
        }
        is_heapifying_up_next = false;
      }
    } else if (is_heapifying_down_init) {
      CHECK(is_heapifying_down);
      CHECK_NE(heap_pos_prev, (heap_size + 1) / 2);
      heap_elem = heap_array[(heap_size + 1) / 2][(heap_size + 1) % 2];
      is_heapifying_down_init_next = false;
      heap_pair_prev[1] = heap_root;
      heap_pos = 1;
      heap_pos_prev = -1;
    } else if (is_heapifying_down) {
      CHECK(!is_heapifying_up);
      const auto heap_pos_next_left = heap_pos * 2;
      const auto heap_pos_next_right = heap_pos * 2 + 1;
      const bool is_left_valid = heap_pos_next_left < heap_size + 1;
      const bool is_right_valid = heap_pos_next_right < heap_size + 1;
      CHECK_NE(heap_pos_prev, heap_pos);
      auto heap_pair = heap_pair_prev;
      heap_pair_prev = is_left_valid ? heap_array[heap_pos] : ChunkRefPair();
      const auto heap_elem_next_left = heap_pair_prev[0];
      const auto heap_elem_next_right = heap_pair_prev[1];
      const bool is_left_update_needed =
          is_left_valid && heap_elem < heap_elem_next_left;
      const bool is_right_update_needed =
          is_right_valid && heap_elem < heap_elem_next_right;
      heap_pos_prev = heap_pos / 2;
      CHECK(heap_array[heap_pos / 2][1] == heap_pair[1]);
      if (heap_pos / 2 != 0) {
        CHECK(heap_array[heap_pos / 2][0] == heap_pair[0]);
      }
      const auto heap_pos_curr = heap_pos;
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
      }
      heap_array[heap_pos_curr / 2] = heap_pair;
    } else {
      heap_pos_prev = -1;
    }

    bool is_input_valid;
    const auto push_req = push_req_q.peek(is_input_valid);
    const auto input_task = push_req.task;

    bool is_refill_task_valid;
    const auto refill_task = mem_read_data_q.peek(is_refill_task_valid);

    const auto input_bid = push_req.bid;
    const auto input_meta = chunk_meta[input_bid];

    const auto spill_meta = chunk_meta[spill_bid];
    const auto refill_meta = chunk_meta[refill_bid];

    bool is_output_valid;
    uint_bid_t output_bid;
    ChunkMeta output_meta;
    bool is_full_valid;
    uint_bid_t full_bid;
    ChunkMeta full_meta;
    FindChunk(chunk_meta, is_output_valid, output_bid, output_meta,
              is_full_valid, full_bid, full_meta);

    const bool is_top_valid = heap_size != 0;
    const uint_bid_t top_bid = heap_root.bucket;

    // Read refill data and push into chunk buffer if
    const bool can_recv_refill =
        is_refill_valid &&    // there is active refilling, and
        is_refill_task_valid  // refill data is available for read.
        ;

    // Read input task and push into chunk buffer if
    const bool can_enqueue =
        is_input_valid &&        // there is an input task, and
        !input_meta.IsFull() &&  // chunk is not already full, and
        (!is_refill_valid ||
         input_bid != refill_bid ||          // if chunk is refilling,
         kBufferSize - input_meta.GetSize()  // available space
             >                               // must be greater than
             refill_remain_count             // #tasks to refill,
         ) &&                                // and
        (!is_refill_next_valid ||
         input_bid != refill_bid_next ||     // if chunk will refill soon,
         kBufferSize - input_meta.GetSize()  // available space
             >                               // must be greater than
             kChunkSize                      // #tasks to refill,
         ) &&                                // and
        (!can_recv_refill ||                 // if reading refill data,
         input_bid % kChunkPartFac           // the input bucket
             !=                              // must access a different bank
             refill_bid % kChunkPartFac      // as the refill bucket.
        );

    // Pop from chunk buffer and write spill data if
    const bool can_send_spill =
        is_spill_valid &&         // there is active spilling, and
        !mem_write_req_q.full();  // spill data is available for write.

    // Pop from highest-priority chunk buffer and write output data if
    const bool can_dequeue =
        is_output_valid &&  // there is a non-empty chunk, and
        !pop_q.full() &&    // output is available for write, and
        (!is_spill_valid || output_bid != spill_bid ||  // if chunk is spilling,
         output_meta.GetSize()                          // available tasks
             >                                          // must be greater than
             kChunkSize - spill_addr_req % kChunkSize   // #tasks to spill,
         ) &&                                           // and
        (!is_spill_valid ||             // if there is active spilling,
         output_bid % kChunkPartFac     // the output bucket
             !=                         // must access a different bank
             spill_bid % kChunkPartFac  // as the spill bucket.
        );
    // Note: to avoid chunk_buf read address being dependent on FIFO fullness,
    // can_dequeue must not depend on the fullness of mem_write_req_q.

    // Start spilling a new chunk if
    const bool can_start_spill =
        !is_spill_valid &&           // there is no active spilling, and
        is_full_valid &&             // chunks need spilling, and
        full_meta.IsAlmostFull() &&  // chunk is almost full, and
        !(can_dequeue && full_bid == output_bid) &&  // chunk won't pop, and
        !is_heapifying_up && !is_heapifying_down     // heap is available.
        ;

    // Start refilling a new chunk if
    const bool can_schedule_refill =
        !is_refill_addr_valid &&  // there is no active refilling, and
        !is_refill_next_valid &&  // no future refilling is scheduled, and
        is_top_valid &&           // there is a chunk for refilling, and
        (!is_refill_valid ||
         top_bid != refill_bid) &&  // the chunk is not refilling now, and
        (!is_output_valid ||        // either on-chip chunk is empty, or
         output_bid >= top_bid      // off-chip chunk has higher priority,
         ) &&                       // and
        chunk_meta[top_bid].IsAlmostEmpty() &&     // chunk is almost empty,
        !(can_enqueue && top_bid == input_bid) &&  // chunk won't push, and
        !can_start_spill &&
        !is_heapifying_up && !is_heapifying_down  // heap is available.
        ;

    // Send refill data read address if
    const bool can_send_refill =
        is_refill_addr_valid &&  // there is an active refill request, and
        refill_addr <= spill_addr_resp &&  // writes have finished, and
        !mem_read_addr_q.full()            // address output is not full.
        ;

    TaskOnChip spill_task, output_task;
    ReadChunk(chunk_buf, is_spill_valid, spill_bid, spill_meta.GetReadPos(),
              is_output_valid, output_bid, output_meta.GetReadPos(), spill_task,
              output_task);
    WriteChunk(can_recv_refill, refill_bid, refill_meta.GetWritePos(),
               refill_task, can_enqueue, input_bid, input_meta.GetWritePos(),
               input_task, chunk_buf);

    ap_uint<kBucketCount> is_push = 0;

    if (can_recv_refill) {
      CHECK_GE(
          kBufferSize - chunk_meta[refill_bid].GetSize(),  // Available slots.
          refill_remain_count  // Remaining to refill.
          )
          << "refill_bid: " << refill_bid;
      mem_read_data_q.read(nullptr);
      CHECK(!is_push.bit(refill_bid));
      is_push.bit(refill_bid) = true;

      --refill_remain_count;
      VLOG(5) << "refilling bucket " << refill_bid << " ("
              << refill_remain_count << " to go)";
      if (refill_remain_count == 0) {
        is_refill_valid = false;
      }
    }

    if (can_enqueue) {
      push_req_q.read(nullptr);
      CHECK(!is_push.bit(input_bid));
      is_push.bit(input_bid) = true;

      VLOG(5) << "enqueue " << push_req.task << " to bucket " << input_bid
              << " (which already has " << input_meta.GetSize() << " tasks)";
    }

    ap_uint<kBucketCount> is_pop = 0;

    if (can_send_spill) {
      CHECK_GE(chunk_meta[spill_bid].GetSize(),          // Available tasks.
               kChunkSize - spill_addr_req % kChunkSize  // Remaining to spill.
      );
      mem_write_req_q.try_write({spill_addr_req, spill_task});

      CHECK(!is_pop.bit(spill_bid));
      is_pop.bit(spill_bid) = true;
      ++spill_addr_req;
      if (spill_addr_req % kChunkSize == 0) {
        is_spill_valid = false;
      }
    }

    if (can_dequeue) {
      pop_q.try_write(output_task);
      CHECK(!is_pop.bit(output_bid));
      is_pop.bit(output_bid) = true;
    }

    RANGE(i, kBucketCount, {
      if (is_push.bit(i)) {
        chunk_meta[i].Push();
      }
      if (is_pop.bit(i)) {
        chunk_meta[i].Pop();
      }
    });

    if (can_start_spill) {
      spill_bid = full_bid;
      is_spill_valid = true;

      heap_elem = {
          .addr = spill_addr_req,
          .bucket = spill_bid,
      };
      ++heap_size;
      max_heap_size = std::max(max_heap_size, heap_size);
      CHECK_LT(heap_size, kCgpqCapacity);

      // Heapify up.
      heap_pos = heap_size;
      is_heapifying_up_next = is_heapifying_up_init_next = true;

      VLOG(5) << "start spilling bucket " << spill_bid << " to ["
              << spill_addr_req
              << "], current buffer size: " << full_meta.GetSize();
    }

    if (can_schedule_refill) {
      refill_bid_next = top_bid;
      is_refill_next_valid = true;
      refill_addr = heap_root.addr;
      is_refill_addr_valid = true;
      CHECK_EQ(refill_addr % kChunkSize, 0);

      CHECK_GT(heap_size, 0);
      --heap_size;

      // Heapify down.
      CHECK(!is_heapifying_down_init);
      is_heapifying_down_next = is_heapifying_down_init_next = true;

      VLOG(5) << "schedule refilling bucket " << refill_bid_next << " from "
              << refill_addr
              << ", current buffer size: " << chunk_meta[top_bid].GetSize();
    }

    if (can_send_refill) {
      VLOG(5) << "refilling from [" << refill_addr << "]";

      mem_read_addr_q.try_write(refill_addr);
      ++refill_addr;
      if (refill_addr % kChunkSize == 0) {
        is_refill_addr_valid = false;
      }
    }

    if (!mem_write_resp_q.empty()) {
      mem_write_resp_q.read(nullptr);
      ++spill_addr_resp;
    }

    if (!is_refill_valid) {
      refill_bid = refill_bid_next;
      is_refill_valid = is_refill_next_valid;
      is_refill_next_valid = false;
      refill_remain_count = kChunkSize;
    }

    is_heapifying_up = is_heapifying_up_next;
    is_heapifying_up_init = is_heapifying_up_init_next;
    is_heapifying_down = is_heapifying_down_next;
    is_heapifying_down_init = is_heapifying_down_init_next;
  }

  done_q.read(nullptr);

  stat_q.write(spill_addr_req / kChunkSize);
  stat_q.write(max_heap_size);

  CHECK_EQ(heap_size, 0);

  for (int bid = 0; bid < kBucketCount; ++bid) {
    CHECK(chunk_meta[bid].IsEmpty());
  }
}

// Each push request puts the task in the queue if there isn't a task for the
// same vertex in the queue, or decreases the priority of the existing task
// using the new value. Whether a new task is created is returned in the
// response.
//
// Each pop removes a task if the queue is not empty, otherwise the response
// indicates that the queue is empty.
void TaskQueue(
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    // Scalar
    uint_qid_t qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q,
    // NOOP acknowledgements
    ostream<bool>& noop_q,
    // Queue outputs.
    ostream<TaskOnChip>& pop_q,
#ifdef TAPA_SSSP_COARSE_PRIORITY
    //
    ostream<uint_spill_addr_t>& cgpq_spill_read_addr_q,
    istream<TaskOnChip>& cgpq_spill_read_data_q,
    ostream<packet<uint_spill_addr_t, TaskOnChip>>& cgpq_spill_write_req_q,
    istream<bool>& cgpq_spill_write_resp_q
#else   // TAPA_SSSP_COARSE_PRIORITY
    //
    ostream<Vid>& piheap_array_read_addr_q,
    istream<HeapElemPacked>& piheap_array_read_data_q,
    ostream<packet<Vid, HeapElemPacked>>& piheap_array_write_req_q,
    istream<bool>& piheap_array_write_resp_q,
    //
    ostream<Vid>& piheap_index_read_addr_q,
    istream<HeapIndexEntry>& piheap_index_read_data_q,
    ostream<packet<Vid, HeapIndexEntry>>& piheap_index_write_req_q,
    istream<bool>& piheap_index_write_resp_q
#endif  // TAPA_SSSP_COARSE_PRIORITY
) {
#ifdef TAPA_SSSP_COARSE_PRIORITY
  stream<PushReq> push_req_qi;

  task()
      .invoke<detach>(CgpqBucketGen, push_req_q, push_req_qi)
      .invoke<detach>(CgpqCore, done_q, stat_q, qid, push_req_qi, noop_q, pop_q,
                      cgpq_spill_read_addr_q, cgpq_spill_read_data_q,
                      cgpq_spill_write_req_q, cgpq_spill_write_resp_q);
#else  // TAPA_SSSP_COARSE_PRIORITY
  // Heap rule: child <= parent
  streams<HeapReq, kLevelCount, 1> req_q;
  streams<HeapResp, kLevelCount, 1> resp_q;
  streams<HeapIndexReq, kLevelCount, 1> index_req_qs;
  streams<HeapIndexResp, kLevelCount, 1> index_resp_qs;
  stream<packet<LevelId, HeapIndexReq>, 1> index_req_q;
  stream<packet<LevelId, HeapIndexResp>, 1> index_resp_q;

  stream<OffChipLevelId, 64> array_read_id_q;
  streams<Vid, kOffChipLevelCount, kPiHeapWidth> array_read_addr_q;
  streams<HeapElemPacked, kOffChipLevelCount, 1> array_read_data_q;

  stream<OffChipLevelId, 64> array_write_id_q;
  streams<packet<Vid, HeapElemPacked>, kOffChipLevelCount, 1> array_write_req_q;
  streams<bool, kOffChipLevelCount, 1> array_write_resp_q;

  streams<bool, kPiHeapStatTaskCount, 2> done_qi;
  streams<PiHeapStat, kPiHeapStatTaskCount, 2> stat_qi;

  stream<AcquireIndexReq, 2> acquire_index_req_q;

  stream<HeapAcquireIndexContext, 64> acquire_index_ctx_q;

  stream<QueueStateUpdate, 2> queue_state_q;

  stream<IndexStateUpdate, 2> index_req_state_q;
  stream<IndexStateUpdate, 2> index_resp_state_q;

  task()
      .invoke<detach>(PiHeapStatArbiter, done_q, stat_q, done_qi, stat_qi)
      .invoke<detach>(PiHeapPerf, qid, done_qi[0], stat_qi[0], queue_state_q,
                      index_req_state_q, index_resp_state_q)
      .invoke<detach>(PiHeapHead, qid, push_req_q, noop_q, pop_q, req_q[0],
                      resp_q[0], queue_state_q, acquire_index_req_q,
                      index_req_qs[0], index_resp_qs[0])
#if TAPA_SSSP_PHEAP_WIDTH == 2
      // clang-format off
      .invoke<detach>(PiHeapBodyL1,  qid, req_q[ 0], resp_q[ 0], req_q[ 1], resp_q[ 1], index_req_qs[ 1], index_resp_qs[ 1])
      .invoke<detach>(PiHeapBodyL2,  qid, req_q[ 1], resp_q[ 1], req_q[ 2], resp_q[ 2], index_req_qs[ 2], index_resp_qs[ 2])
      .invoke<detach>(PiHeapBodyL3,  qid, req_q[ 2], resp_q[ 2], req_q[ 3], resp_q[ 3], index_req_qs[ 3], index_resp_qs[ 3])
      .invoke<detach>(PiHeapBodyL4,  qid, req_q[ 3], resp_q[ 3], req_q[ 4], resp_q[ 4], index_req_qs[ 4], index_resp_qs[ 4])
      .invoke<detach>(PiHeapBodyL5,  qid, req_q[ 4], resp_q[ 4], req_q[ 5], resp_q[ 5], index_req_qs[ 5], index_resp_qs[ 5])
      .invoke<detach>(PiHeapBodyL6,  qid, req_q[ 5], resp_q[ 5], req_q[ 6], resp_q[ 6], index_req_qs[ 6], index_resp_qs[ 6])
      .invoke<detach>(PiHeapBodyL7,  qid, req_q[ 6], resp_q[ 6], req_q[ 7], resp_q[ 7], index_req_qs[ 7], index_resp_qs[ 7])
      .invoke<detach>(PiHeapBodyL8,  qid, req_q[ 7], resp_q[ 7], req_q[ 8], resp_q[ 8], index_req_qs[ 8], index_resp_qs[ 8])
      .invoke<detach>(PiHeapBodyL9,  qid, req_q[ 8], resp_q[ 8], req_q[ 9], resp_q[ 9], index_req_qs[ 9], index_resp_qs[ 9])
      .invoke<detach>(PiHeapBodyL10, qid, req_q[ 9], resp_q[ 9], req_q[10], resp_q[10], index_req_qs[10], index_resp_qs[10])
      .invoke<detach>(PiHeapBodyL11, qid, req_q[10], resp_q[10], req_q[11], resp_q[11], index_req_qs[11], index_resp_qs[11])
      .invoke<detach>(PiHeapBodyL12, qid, req_q[11], resp_q[11], req_q[12], resp_q[12], index_req_qs[12], index_resp_qs[12])
      .invoke<detach>(PiHeapBodyL13, qid, req_q[12], resp_q[12], req_q[13], resp_q[13], index_req_qs[13], index_resp_qs[13])
      .invoke<detach>(PiHeapBodyL14, qid, req_q[13], resp_q[13], req_q[14], resp_q[14], index_req_qs[14], index_resp_qs[14])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[14], resp_q[14], req_q[15], resp_q[15], index_req_qs[15], index_resp_qs[15], 15, array_read_addr_q[0], array_read_data_q[0], array_write_req_q[0], array_write_resp_q[0])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[15], resp_q[15], req_q[16], resp_q[16], index_req_qs[16], index_resp_qs[16], 16, array_read_addr_q[1], array_read_data_q[1], array_write_req_q[1], array_write_resp_q[1])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[16], resp_q[16], req_q[17], resp_q[17], index_req_qs[17], index_resp_qs[17], 17, array_read_addr_q[2], array_read_data_q[2], array_write_req_q[2], array_write_resp_q[2])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[17], resp_q[17], req_q[18], resp_q[18], index_req_qs[18], index_resp_qs[18], 18, array_read_addr_q[3], array_read_data_q[3], array_write_req_q[3], array_write_resp_q[3])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[18], resp_q[18], req_q[19], resp_q[19], index_req_qs[19], index_resp_qs[19], 19, array_read_addr_q[4], array_read_data_q[4], array_write_req_q[4], array_write_resp_q[4])
  // clang-format on
#elif TAPA_SSSP_PHEAP_WIDTH == 4
      // clang-format off
      .invoke<detach>(PiHeapBodyL1,  qid, req_q[0], resp_q[0], req_q[1], resp_q[1], index_req_qs[1], index_resp_qs[1])
      .invoke<detach>(PiHeapBodyL2,  qid, req_q[1], resp_q[1], req_q[2], resp_q[2], index_req_qs[2], index_resp_qs[2])
      .invoke<detach>(PiHeapBodyL3,  qid, req_q[2], resp_q[2], req_q[3], resp_q[3], index_req_qs[3], index_resp_qs[3])
      .invoke<detach>(PiHeapBodyL4,  qid, req_q[3], resp_q[3], req_q[4], resp_q[4], index_req_qs[4], index_resp_qs[4])
      .invoke<detach>(PiHeapBodyL5,  qid, req_q[4], resp_q[4], req_q[5], resp_q[5], index_req_qs[5], index_resp_qs[5])
      .invoke<detach>(PiHeapBodyL6,  qid, req_q[5], resp_q[5], req_q[6], resp_q[6], index_req_qs[6], index_resp_qs[6])
      .invoke<detach>(PiHeapBodyL7,  qid, req_q[6], resp_q[6], req_q[7], resp_q[7], index_req_qs[7], index_resp_qs[7])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[ 7], resp_q[ 7], req_q[ 8], resp_q[ 8], index_req_qs[ 8], index_resp_qs[ 8],  8, array_read_addr_q[0], array_read_data_q[0], array_write_req_q[0], array_write_resp_q[0])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[ 8], resp_q[ 8], req_q[ 9], resp_q[ 9], index_req_qs[ 9], index_resp_qs[ 9],  9, array_read_addr_q[1], array_read_data_q[1], array_write_req_q[1], array_write_resp_q[1])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[ 9], resp_q[ 9], req_q[10], resp_q[10], index_req_qs[10], index_resp_qs[10], 10, array_read_addr_q[2], array_read_data_q[2], array_write_req_q[2], array_write_resp_q[2])
  // clang-format on
#elif TAPA_SSSP_PHEAP_WIDTH == 8
      // clang-format off
      .invoke<detach>(PiHeapBodyL1,  qid, req_q[0], resp_q[0], req_q[1], resp_q[1], index_req_qs[1], index_resp_qs[1])
      .invoke<detach>(PiHeapBodyL2,  qid, req_q[1], resp_q[1], req_q[2], resp_q[2], index_req_qs[2], index_resp_qs[2])
      .invoke<detach>(PiHeapBodyL3,  qid, req_q[2], resp_q[2], req_q[3], resp_q[3], index_req_qs[3], index_resp_qs[3])
      .invoke<detach>(PiHeapBodyL4,  qid, req_q[3], resp_q[3], req_q[4], resp_q[4], index_req_qs[4], index_resp_qs[4])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[4], resp_q[4], req_q[5], resp_q[5], index_req_qs[5], index_resp_qs[5], 5, array_read_addr_q[0], array_read_data_q[0], array_write_req_q[0], array_write_resp_q[0])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[5], resp_q[5], req_q[6], resp_q[6], index_req_qs[6], index_resp_qs[6], 6, array_read_addr_q[1], array_read_data_q[1], array_write_req_q[1], array_write_resp_q[1])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[6], resp_q[6], req_q[7], resp_q[7], index_req_qs[7], index_resp_qs[7], 7, array_read_addr_q[2], array_read_data_q[2], array_write_req_q[2], array_write_resp_q[2])
  // clang-format on
#elif TAPA_SSSP_PHEAP_WIDTH == 16
      // clang-format off
      .invoke<detach>(PiHeapBodyL1,  qid, req_q[0], resp_q[0], req_q[1], resp_q[1], index_req_qs[1], index_resp_qs[1])
      .invoke<detach>(PiHeapBodyL2,  qid, req_q[1], resp_q[1], req_q[2], resp_q[2], index_req_qs[2], index_resp_qs[2])
      .invoke<detach>(PiHeapBodyL3,  qid, req_q[2], resp_q[2], req_q[3], resp_q[3], index_req_qs[3], index_resp_qs[3])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[3], resp_q[3], req_q[4], resp_q[4], index_req_qs[4], index_resp_qs[4], 4, array_read_addr_q[0], array_read_data_q[0], array_write_req_q[0], array_write_resp_q[0])
      .invoke<detach>(PiHeapBodyOffChip, qid, req_q[4], resp_q[4], req_q[5], resp_q[5], index_req_qs[5], index_resp_qs[5], 5, array_read_addr_q[1], array_read_data_q[1], array_write_req_q[1], array_write_resp_q[1])
  // clang-format on
#else
#error "invalid TAPA_SSSP_PHEAP_WIDTH"
#endif  // TAPA_SSSP_PHEAP_WIDTH
      .invoke<detach>(PiHeapDummyTail, qid, req_q[kLevelCount - 1],
                      resp_q[kLevelCount - 1])

      .invoke<detach>(PiHeapArrayReadAddrArbiter, array_read_addr_q,
                      array_read_id_q, piheap_array_read_addr_q)
      .invoke<detach>(PiHeapArrayReadDataArbiter, array_read_id_q,
                      piheap_array_read_data_q, array_read_data_q)
      .invoke<detach>(PiHeapArrayWriteReqArbiter, array_write_req_q,
                      array_write_id_q, piheap_array_write_req_q)
      .invoke<detach>(PiHeapArrayWriteRespArbiter, array_write_id_q,
                      piheap_array_write_resp_q, array_write_resp_q)

      .invoke<detach>(PiHeapIndexReqArbiter, index_req_qs, index_req_state_q,
                      index_req_q)
      .invoke<detach>(PiHeapIndex, done_qi[1], stat_qi[1], index_resp_state_q,
                      qid, acquire_index_req_q, index_req_q, index_resp_q,
                      piheap_index_read_addr_q, piheap_index_read_data_q,
                      piheap_index_write_req_q, piheap_index_write_resp_q,
                      acquire_index_ctx_q, acquire_index_ctx_q)
      .invoke<detach>(PiHeapIndexRespArbiter, index_resp_q, index_resp_qs);
#endif  // TAPA_SSSP_COARSE_PRIORITY
}

#if TAPA_SSSP_SHARD_COUNT >= 2
void Switch(
    //
    ap_uint<log2(kShardCount)> b,
    //
    istream<bool>& done_q, ostream<int64_t>& stat_q,
    //
    istream<TaskOnChip>& in_q0, istream<TaskOnChip>& in_q1,
    //
    ostream<TaskOnChip>& out_q0, ostream<TaskOnChip>& out_q1) {
  bool should_prioritize_1 = false;
  int64_t total_cycle_count = 0;
  int64_t full_0_cycle_count = 0;
  int64_t full_1_cycle_count = 0;
  int64_t conflict_0_cycle_count = 0;
  int64_t conflict_1_cycle_count = 0;

spin:
  for (bool is_pkt_0_valid, is_pkt_1_valid; done_q.empty();) {
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
        should_write_0 && out_q0.try_write(shoud_write_0_0 ? pkt_0 : pkt_1);
    const bool is_1_written =
        should_write_1 && out_q1.try_write(shoud_write_1_1 ? pkt_1 : pkt_0);

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

  done_q.read(nullptr);
  stat_q.write(total_cycle_count);
  stat_q.write(full_0_cycle_count);
  stat_q.write(full_1_cycle_count);
  stat_q.write(conflict_0_cycle_count);
  stat_q.write(conflict_1_cycle_count);
}
#endif  // TAPA_SSSP_SHARD_COUNT

void EdgeReqArbiter(tapa::istreams<EdgeReq, kPeCount>& req_q,
                    tapa::ostreams<SourceVertex, kShardCount>& src_q,
                    tapa::ostreams<Vid, kShardCount>& addr_q) {
  DECL_ARRAY(EdgeReq, req, kShardCount, EdgeReq());
  DECL_ARRAY(bool, src_valid, kShardCount, false);
  DECL_ARRAY(bool, addr_valid, kShardCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kShardCount) + 2> pe_sid = 0;; ++pe_sid) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      const auto pe = pe_sid / 8 * kShardCount + sid;
      if (!src_valid[sid] && !addr_valid[sid] && req_q[pe].try_read(req[sid])) {
        src_valid[sid] = addr_valid[sid] = true;
      }

      UNUSED RESET(addr_valid[sid], addr_q[sid].try_write(req[sid].addr));
      UNUSED RESET(src_valid[sid], src_q[sid].try_write(req[sid].payload));
    });
  }
}

#ifdef TAPA_SSSP_IMMEDIATE_RELAX
const int kSwitchOutputCount = kQueueCount;
#else   // TAPA_SSSP_IMMEDIATE_RELAX
const int kSwitchOutputCount = kSubIntervalCount;
#endif  // TAPA_SSSP_IMMEDIATE_RELAX

void SwitchOutputArbiter(
    tapa::istreams<TaskOnChip, kShardCount>& in_q,
    tapa::ostreams<TaskOnChip, kSwitchOutputCount>& out_q) {
  static_assert(kSwitchOutputCount % kShardCount == 0,
                "current implementation requires that sub-interval count is a "
                "multiple of shard count");
  DECL_ARRAY(TaskOnChip, update, kShardCount, TaskOnChip());
  DECL_ARRAY(bool, update_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      UNUSED SET(update_valid[sid], in_q[sid].try_read(update[sid]));
      const auto soid =
          update[sid].vid() % kSwitchOutputCount / kShardCount * kShardCount +
          sid;
      UNUSED RESET(update_valid[sid], out_q[soid].try_write(update[sid]));
    });
  }
}

void EdgeMem(istream<bool>& done_q, ostream<int64_t>& stat_q,
             tapa::istream<Vid>& read_addr_q, tapa::ostream<Edge>& read_data_q,
             tapa::async_mmap<Edge> mem) {
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

void ProcElemS0(istream<TaskOnChip>& task_in_q, ostream<Vid>& task_resp_q,
                ostream<uint_vid_t>& task_count_q,
                ostream<EdgeReq>& edge_req_q) {
  EdgeReq req;

spin:
  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      req = {task.vertex().offset, {task.vid(), task.vertex().distance}};
      i = task.vertex().degree;
      task_count_q.write(i);
    }

    if (i > 0) {
      edge_req_q.write(req);
      if (i == 1) {
        task_resp_q.write(req.payload.parent);
      }

      ++req.addr;
      --i;
    }
  }
}

void ProcElemS1(istream<SourceVertex>& src_in_q,
                istream<Edge>& edges_read_data_q,
                ostream<TaskOnChip>& update_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!src_in_q.empty() && !edges_read_data_q.empty()) {
      const auto src = src_in_q.read(nullptr);
      const auto edge = edges_read_data_q.read(nullptr);
      update_out_q.write(Task{
          .vid = edge.dst,
          .vertex = {src.parent, src.distance + edge.weight},
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
    istream<TaskOnChip>& req_q, ostream<TaskOnChip>& push_q,
    ostream<bool>& noop_q,
    //
    ostream<Vid>& read_vid_out_q, istream<Vid>& read_vid_in_q,
    ostream<Vid>& write_vid_out_q, istream<Vid>& write_vid_in_q,
    //
    ostream<Vid>& read_addr_q, istream<Vertex>& read_data_q,
    ostream<packet<Vid, Vertex>>& write_req_q, istream<bool>& write_resp_q) {
  constexpr int kVertexCacheSize = 4096;
  VertexCacheEntry cache[kVertexCacheSize];
#pragma HLS bind_storage variable = cache type = RAM_S2P impl = URAM
#pragma HLS aggregate variable = cache bit

init:
  for (int i = 0; i < kVertexCacheSize; ++i) {
#pragma HLS dependence variable = cache inter false
    cache[i] = {
        .is_valid = false,
        .is_reading = false,
        .is_writing = false,
        .is_dirty = false,
    };
  }

  CLEAN_UP(clean_up, [&] {
    for (int i = 0; i < kVertexCacheSize; ++i) {
      CHECK(!cache[i].is_valid);
    }
  });

  auto GenPush = [&push_q](VertexCacheEntry& entry) {
    push_q.write(entry.task);
    VLOG(5) << "task     -> " << entry.task;
    entry.is_dirty = true;
    VLOG(5) << "v$$$[" << entry.task.vid() << "] marked dirty";
  };

  auto GenNoop = [&noop_q] {
    noop_q.write(false);
    VLOG(5) << "task     -> NOOP";
  };

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
    auto& req_hit_count = perf_counters[6];
    auto& req_miss_count = perf_counters[7];
    auto& req_busy_count = perf_counters[8];
    auto& idle_count = perf_counters[9];

    const int kMaxActiveWriteCount = 63;
    int8_t active_write_count = 0;

    bool is_started = false;

  spin:
    for (; done_q.empty();) {
#pragma HLS pipeline II = 2
      if (!write_resp_q.empty() && !write_vid_in_q.empty()) {
        ++write_resp_count;

        write_resp_q.read(nullptr);
        const auto vid = write_vid_in_q.read(nullptr);
        const auto entry = cache[vid / kSubIntervalCount % kVertexCacheSize];
        CHECK(entry.is_valid);
        CHECK_NE(entry.task.vid(), vid);
        CHECK(entry.is_writing);
        cache[vid / kSubIntervalCount % kVertexCacheSize].is_writing = false;
        CHECK_GT(active_write_count, 0);
        --active_write_count;
      } else if (!read_data_q.empty() && !read_vid_in_q.empty()) {
        ++read_resp_count;

        const auto vertex = read_data_q.read(nullptr);
        const auto vid = read_vid_in_q.read(nullptr);
        VLOG(5) << "vmem[" << vid << "] -> " << vertex;
        auto entry = cache[vid / kSubIntervalCount % kVertexCacheSize];
        CHECK(entry.is_valid);
        CHECK_EQ(entry.task.vid(), vid);
        CHECK(entry.is_reading);
        entry.is_reading = false;
        // Vertex updates do not have metadata of the destination vertex, so
        // update cache using metadata from DRAM.
        entry.task.set_metadata(vertex);
        if (vertex <= entry.task.vertex()) {
          // Distance in DRAM is closer; generate NOOP and update cache.
          GenNoop();
          entry.task.set_value(vertex);
        } else {
          // Distance in cache is closer; generate PUSH.
          GenPush(entry);
          ++write_hit;
        }
        cache[vid / kSubIntervalCount % kVertexCacheSize] = entry;
      } else if (!req_q.empty()) {
        is_started = true;

        const auto task = req_q.peek(nullptr);
        const auto vid = task.vid();
        auto entry = cache[vid / kSubIntervalCount % kVertexCacheSize];
        bool is_entry_updated = false;
        if (entry.is_valid && entry.task.vid() == task.vid()) {  // Hit.
          ++req_hit_count;

          req_q.read(nullptr);
          VLOG(5) << "task     <- " << task;

          // Update cache if new task has higher priority.
          if ((is_entry_updated = !(task <= entry.task))) {
            entry.task.set_value(task.vertex());
          }

          // Generate PUSH if and only if cache is updated and not reading.
          // If reading, PUSH will be generated when read finishes, if
          // necessary.
          if (is_entry_updated && !entry.is_reading) {
            GenPush(entry);
            ++write_hit;
          } else {
            GenNoop();
          }

          ++read_hit;
        } else if (!(entry.is_valid &&
                     (entry.is_reading || entry.is_writing)) &&
                   !read_addr_q.full() && !read_vid_out_q.full() &&
                   !(entry.is_valid && entry.is_dirty &&
                     (write_req_q.full() ||
                      write_vid_out_q.full()))) {  // Miss.
          ++req_miss_count;

          req_q.read(nullptr);
          VLOG(5) << "task     <- " << task;

          // Issue DRAM read request.
          read_addr_q.try_write(vid / kIntervalCount);
          VLOG(5) << "vmem[" << vid << "] ?";
          read_vid_out_q.try_write(vid);

          // Issue DRAM write request.
          if (entry.is_valid && entry.is_dirty) {
            entry.is_writing = true;
            write_req_q.try_write(
                {entry.task.vid() / kIntervalCount, entry.task.vertex()});
            write_vid_out_q.try_write(entry.task.vid());
            CHECK_LT(active_write_count, kMaxActiveWriteCount);
            ++active_write_count;
            ++write_miss;
            VLOG(5) << "vmem[" << entry.task.vid() << "] <- " << entry.task;
          } else {
            entry.is_writing = false;
          }

          // Replace cache with new task.
          entry.is_valid = true;
          entry.is_reading = true;
          entry.is_dirty = false;
          entry.task.set_vid(vid);
          entry.task.set_value(task.vertex());
          is_entry_updated = true;

          ++read_miss;
        } else {  // Otherwise, wait until entry is not reading.
          ++req_busy_count;
        }

        if (is_entry_updated) {
          cache[vid / kSubIntervalCount % kVertexCacheSize] = entry;
          VLOG(5) << "v$$$[" << vid << "] <- " << entry.task;
        }
      } else if (is_started) {
        ++idle_count;
      }
    }

  reset:
    for (int i = 0; i < kVertexCacheSize;) {
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
              {entry.task.vid() / kIntervalCount, entry.task.vertex()});
          ++active_write_count;
          ++write_miss;
          VLOG(5) << "vmem[" << entry.task.vid() << "] <- " << entry.task;
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
}

void VertexNoopMerger(istreams<bool, kSubIntervalCount>& pkt_in_q,
                      ostream<uint_vertex_noop_t>& pkt_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    uint_vertex_noop_t count = 0;
    bool buf;
    RANGE(iid, kSubIntervalCount, pkt_in_q[iid].try_read(buf) && ++count);
    if (count) {
      pkt_out_q.write(count);
    }
  }
}

void QueueNoopMerger(istreams<bool, kQueueCount>& in_q,
                     ostream<uint_queue_noop_t>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    uint_queue_noop_t count = 0;
    bool buf;
    RANGE(qid, kQueueCount, in_q[qid].try_read(buf) && ++count);
    if (count) {
      out_q.write(count);
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

void QueueOutputArbiter(tapa::istreams<TaskOnChip, kQueueCount>& in_q,
                        tapa::ostreams<TaskOnChip, kSubIntervalCount>& out_q) {
  static_assert(kQueueCount % kSubIntervalCount == 0,
                "current implementation requires that queue count is a "
                "multiple of interval count");
spin:
  for (uint_qid_t qid_base = 0;; ++qid_base) {
#pragma HLS pipeline II = 1
    RANGE(iid, kSubIntervalCount, {
      const auto qid =
          qid_base % kQueueCount / kSubIntervalCount * kSubIntervalCount + iid;
      if (!in_q[qid].empty() && !out_q[iid].full()) {
        out_q[iid].try_write(in_q[qid].read(nullptr));
      }
    });
  }
}

void VertexOutputArbiter(tapa::istreams<TaskOnChip, kSubIntervalCount>& in_q,
                         tapa::ostreams<TaskOnChip, kQueueCount>& out_q) {
  static_assert(kQueueCount % kSubIntervalCount == 0,
                "current implementation requires that queue count is a "
                "multiple of interval count");
  DECL_ARRAY(TaskOnChip, task, kSubIntervalCount, TaskOnChip());
  DECL_ARRAY(bool, task_valid, kSubIntervalCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(iid, kSubIntervalCount, {
      UNUSED SET(task_valid[iid], in_q[iid].try_read(task[iid]));
      const auto vid = task[iid].vid();
      if (task_valid[iid]) {
        CHECK_EQ(vid % kSubIntervalCount, iid);
      }
      const auto qid =
          vid % kQueueCount / kSubIntervalCount * kSubIntervalCount + iid;
      UNUSED RESET(task_valid[iid], out_q[qid].try_write(task[iid]));
    });
  }
}

#ifdef TAPA_SSSP_IMMEDIATE_RELAX
const int kTaskInputCount = kSubIntervalCount;
#else   // TAPA_SSSP_IMMEDIATE_RELAX
const int kTaskInputCount = kQueueCount;
#endif  // TAPA_SSSP_IMMEDIATE_RELAX

void TaskArbiter(  //
    istream<TaskOnChip>& task_init_q,
    istreams<TaskOnChip, kTaskInputCount>& task_in_q,
    ostreams<TaskOnChip, kPeCount>& task_req_q,
    istreams<Vid, kPeCount>& task_resp_q) {
exec:
  for (;;) {
    static_assert(kPeCount % kTaskInputCount == 0, "");
    bool is_pe_active[kTaskInputCount][kPeCount / kTaskInputCount];
#pragma HLS array_partition variable = is_pe_active complete dim = 0
    RANGE(peid, kPeCount,
          is_pe_active[peid % kTaskInputCount][peid / kTaskInputCount] = false);

    CLEAN_UP(clean_up, [&] {
      RANGE(
          peid, kPeCount,
          CHECK(!is_pe_active[peid % kTaskInputCount][peid / kTaskInputCount]));
    });

    const auto task_init = task_init_q.read();
    const auto tid_init = task_init.vid() % kTaskInputCount;
    task_req_q[tid_init].write(task_init);
    is_pe_active[tid_init][0] = true;

  spin:
    for (; task_init_q.empty();) {
#pragma HLS pipeline II = 1

      // Issue task requests.
      RANGE(tid, kTaskInputCount, {
        ap_uint<bit_length(kPeCount / kTaskInputCount - 1)> pe_tid;
        if (!task_in_q[tid].empty() && find_false(is_pe_active[tid], pe_tid)) {
          const auto peid = pe_tid * kTaskInputCount + tid;
          const auto task = task_in_q[tid].read(nullptr);
          CHECK_EQ(task.vid() % kTaskInputCount, tid);
          CHECK(!task_req_q[peid].full()) << "PE rate limit needed";
          task_req_q[peid].try_write(task);
          CHECK(!is_pe_active[tid][pe_tid]);
          is_pe_active[tid][pe_tid] = true;
        }
      });

      // Collect task responses.
      RANGE(peid, kPeCount, {
        if (!task_resp_q[peid].empty()) {
          const auto vid = task_resp_q[peid].read(nullptr);
          CHECK_EQ(vid % kTaskInputCount, peid % kTaskInputCount);
          CHECK(is_pe_active[peid % kTaskInputCount][peid / kTaskInputCount]);
          is_pe_active[peid % kTaskInputCount][peid / kTaskInputCount] = false;
        }
      });
    }
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
    istreams<int64_t, kShardCount>& edge_stat_q,
    ostreams<bool, kQueueCount>& queue_done_q,
    istreams<PiHeapStat, kQueueCount>& queue_stat_q,
#if TAPA_SSSP_SHARD_COUNT > 1
    ostreams<bool, kSwitchCount>& switch_done_q,
    istreams<int64_t, kSwitchCount>& switch_stat_q,
#endif  // TAPA_SSSP_SHARD_COUNT
    // Task initialization.
    ostream<TaskOnChip>& task_init_q,
    // Task count.
    istream<uint_vertex_noop_t>& vertex_noop_q,
    istream<uint_queue_noop_t>& queue_noop_q,
    istream<TaskCount>& task_count_q) {
  task_init_q.write(root);

  // Statistics.
  int32_t visited_edge_count = 0;
  int32_t push_count = 0;
  int32_t pushpop_count = 0;
  int32_t pop_valid_count = 0;
  int64_t cycle_count = 0;

  constexpr int kTerminationHold = 500;
  ap_uint<bit_length(kTerminationHold)> termination = 0;

spin:
  for (int32_t active_task_count = 1;
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
      active_task_count -= count;
      VLOG(5) << "#task " << previous_task_count << " -> " << active_task_count;
    }

    if (!queue_noop_q.empty()) {
      const auto previous_task_count = active_task_count;
      const auto count = queue_noop_q.read(nullptr);
      active_task_count -= count;
      push_count += count;
      VLOG(5) << "#task " << previous_task_count << " -> " << active_task_count;
    }

    if (!task_count_q.empty()) {
      const auto previous_task_count = active_task_count;
      const auto count = task_count_q.read(nullptr);
      active_task_count += count.new_task_count - count.old_task_count;
      visited_edge_count += count.new_task_count;
      pop_valid_count += count.old_task_count;
      VLOG(5) << "#task " << previous_task_count << " -> " << active_task_count;
    }
  }

  push_count += pop_valid_count;

  RANGE(iid, kSubIntervalCount, vertex_cache_done_q[iid].write(false));
  RANGE(sid, kShardCount, edge_done_q[sid].write(false));
  RANGE(qid, kQueueCount, queue_done_q[qid].write(false));
#if TAPA_SSSP_SHARD_COUNT > 1
  RANGE(swid, kSwitchCount, switch_done_q[swid].write(false));
#endif  // TAPA_SSSP_SHARD_COUNT

  metadata[0] = visited_edge_count;
  metadata[1] = push_count;
  metadata[2] = pushpop_count;
  metadata[3] = pop_valid_count;
  metadata[6] = cycle_count;

vertex_cache_stat:
  for (int i = 0; i < kSubIntervalCount; ++i) {
    for (int j = 0; j < kVertexUniStatCount; ++j) {
      metadata[9 + i * kVertexUniStatCount + j] = vertex_cache_stat_q[i].read();
    }
  }

edge_stat:
  for (int i = 0; i < kShardCount; ++i) {
    for (int j = 0; j < kEdgeUnitStatCount; ++j) {
      metadata[9 + kSubIntervalCount * kVertexUniStatCount +
               i * kEdgeUnitStatCount + j] = edge_stat_q[i].read();
    }
  }

queue_stat:
  for (int i = 0; i < kQueueCount; ++i) {
    for (int j = 0; j < kQueueStatCount; ++j) {
      metadata[9 + kShardCount * kEdgeUnitStatCount +
               kSubIntervalCount * kVertexUniStatCount + i * kQueueStatCount +
               j] = queue_stat_q[i].read();
    }
  }

#if TAPA_SSSP_SHARD_COUNT > 1
switch_stat:
  for (int i = 0; i < kSwitchCount; ++i) {
    for (int j = 0; j < kSwitchStatCount; ++j) {
      metadata[9 + kShardCount * kEdgeUnitStatCount +
               kSubIntervalCount * kVertexUniStatCount +
               kQueueCount * kQueueStatCount + i * kSwitchStatCount + j] =
          switch_stat_q[i].read();
    }
  }
#endif  // TAPA_SSSP_SHARD_COUNT
}

void SSSP(Vid vertex_count, Task root, tapa::mmap<int64_t> metadata,
          tapa::mmaps<Edge, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices,
// For queues.
#ifdef TAPA_SSSP_COARSE_PRIORITY
          tapa::mmap<SpilledTask> cgpq_spill
#else   // TAPA_SSSP_COARSE_PRIORITY
          tapa::mmap<HeapElemPacked> heap_array,
          tapa::mmap<HeapIndexEntry> heap_index
#endif  // TAPA_SSSP_COARSE_PRIORITY
) {
  streams<TaskOnChip, kSubIntervalCount, 2> vertex_out_q;
  streams<TaskOnChip, kQueueCount, 512> queue_push_q;
  streams<bool, kQueueCount, 2> queue_done_q;
  streams<PiHeapStat, kQueueCount, 2> queue_stat_q;

#ifdef TAPA_SSSP_COARSE_PRIORITY
  stream<uint_spill_addr_t> cgpq_spill_read_addr_q;
  stream<TaskOnChip> cgpq_spill_read_data_q;
  stream<packet<uint_spill_addr_t, TaskOnChip>> cgpq_spill_write_req_q;
  stream<bool> cgpq_spill_write_resp_q;
#else   // TAPA_SSSP_COARSE_PRIORITY
  streams<Vid, kQueueCount, 2> piheap_array_read_addr_q;
  streams<HeapElemPacked, kQueueCount, 2> piheap_array_read_data_q;
  streams<packet<Vid, HeapElemPacked>, kQueueCount, 2> piheap_array_write_req_q;
  streams<bool, kQueueCount, 2> piheap_array_write_resp_q;

  streams<Vid, kQueueCount, 2> piheap_index_read_addr_q;
  streams<HeapIndexEntry, kQueueCount, 2> piheap_index_read_data_q;
  streams<packet<Vid, HeapIndexEntry>, kQueueCount, 2> piheap_index_write_req_q;
  streams<bool, kQueueCount, 2> piheap_index_write_resp_q;
#endif  // TAPA_SSSP_COARSE_PRIORITY

  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Vid, kPeCount, 2> task_resp_qi("task_resp_i");

  stream<TaskOnChip, 2> task_init_q;
  streams<TaskOnChip, kQueueCount, 2> queue_pop_q("task_resp");

  // For edges.
  streams<Vid, kShardCount, 2> edge_read_addr_q("edge_read_addr");
  streams<Edge, kShardCount, 2> edge_read_data_q("edge_read_data");
  streams<EdgeReq, kPeCount, kPeCount / kShardCount * 8> edge_req_q("edge_req");
  streams<SourceVertex, kShardCount, 64> src_q("source_vertices");

  // For vertices.
  // Route updates via a kShardCount x kShardCount network.
  streams<TaskOnChip, kShardCount, 32> xbar_q0;
#if TAPA_SSSP_SHARD_COUNT >= 2
  streams<TaskOnChip, kShardCount, 32> xbar_q1;
#endif  // TAPA_SSSP_SHARD_COUNT
#if TAPA_SSSP_SHARD_COUNT >= 4
  streams<TaskOnChip, kShardCount, 32> xbar_q2;
#endif  // TAPA_SSSP_SHARD_COUNT

  streams<TaskOnChip, kSubIntervalCount, 32> vertex_in_q;
  //   Connect the vertex readers and updaters.
  streams<bool, kSubIntervalCount, 2> update_noop_qi1;
  streams<Vid, kSubIntervalCount, 2> vertex_read_addr_q;
  streams<Vertex, kSubIntervalCount, 2> vertex_read_data_q;
  streams<packet<Vid, Vertex>, kSubIntervalCount, 2> vertex_write_req_q;
  streams<bool, kSubIntervalCount, 2> vertex_write_resp_q;
  streams<Vid, kSubIntervalCount, 16> read_vid_q;
  streams<Vid, kSubIntervalCount, 16> write_vid_q;
  streams<bool, kSubIntervalCount, 2> vertex_cache_done_q;
  streams<int32_t, kSubIntervalCount, 2> vertex_cache_stat_q;

  streams<bool, kShardCount, 2> edge_done_q;
  streams<int64_t, kShardCount, 2> edge_stat_q;

  streams<bool, kSwitchCount, 2> switch_done_q;
  streams<int64_t, kSwitchCount, 2> switch_stat_q;

  stream<uint_vertex_noop_t, 2> vertex_noop_q;

  streams<bool, kQueueCount, 2> queue_noop_qi;
  stream<uint_queue_noop_t, 2> queue_noop_q;

  streams<uint_vid_t, kPeCount, 2> task_count_qi;
  stream<TaskCount, 2> task_count_q;

  tapa::task()
      .invoke<join>(Dispatcher, root, metadata, vertex_cache_done_q,
                    vertex_cache_stat_q, edge_done_q, edge_stat_q, queue_done_q,
                    queue_stat_q,
#if TAPA_SSSP_SHARD_COUNT > 1
                    switch_done_q, switch_stat_q,
#endif  // TAPA_SSSP_SHARD_COUNT
                    task_init_q, vertex_noop_q, queue_noop_q, task_count_q)
#ifdef TAPA_SSSP_IMMEDIATE_RELAX
      .invoke<detach>(TaskArbiter, task_init_q, vertex_out_q, task_req_qi,
#else   // TAPA_SSSP_IMMEDIATE_RELAX
      .invoke<detach>(TaskArbiter, task_init_q, queue_pop_q, task_req_qi,
#endif  // TAPA_SSSP_IMMEDIATE_RELAX
                      task_resp_qi)
      .invoke<detach, kQueueCount>(
          TaskQueue, queue_done_q, queue_stat_q, seq(), queue_push_q,
          queue_noop_qi, queue_pop_q,
#ifdef TAPA_SSSP_COARSE_PRIORITY
          cgpq_spill_read_addr_q, cgpq_spill_read_data_q,
          cgpq_spill_write_req_q, cgpq_spill_write_resp_q
#else   // TAPA_SSSP_COARSE_PRIORITY
          piheap_array_read_addr_q, piheap_array_read_data_q,
          piheap_array_write_req_q, piheap_array_write_resp_q,
          piheap_index_read_addr_q, piheap_index_read_data_q,
          piheap_index_write_req_q, piheap_index_write_resp_q
#endif  // TAPA_SSSP_COARSE_PRIORITY
          )
      .invoke<detach>(QueueNoopMerger, queue_noop_qi, queue_noop_q)
#ifdef TAPA_SSSP_IMMEDIATE_RELAX
      .invoke<detach>(QueueOutputArbiter, queue_pop_q, vertex_in_q)
#else   // TAPA_SSSP_IMMEDIATE_RELAX
      .invoke<detach>(VertexOutputArbiter, vertex_out_q, queue_push_q)
#endif  // TAPA_SSSP_IMMEDIATE_RELAX

  // Put mmaps are in the top level to enable flexible floorplanning.
#ifdef TAPA_SSSP_COARSE_PRIORITY
      .invoke<detach, kQueueCount>(
          CgpqSpillMem, cgpq_spill_read_addr_q, cgpq_spill_read_data_q,
          cgpq_spill_write_req_q, cgpq_spill_write_resp_q, cgpq_spill)
#else   // TAPA_SSSP_COARSE_PRIORITY
      .invoke<detach, kQueueCount>(
          PiHeapArrayMem, piheap_array_read_addr_q, piheap_array_read_data_q,
          piheap_array_write_req_q, piheap_array_write_resp_q, heap_array)
      .invoke<detach, kQueueCount>(
          PiHeapIndexMem, piheap_index_read_addr_q, piheap_index_read_data_q,
          piheap_index_write_req_q, piheap_index_write_resp_q, heap_index)
#endif  // TAPA_SSSP_COARSE_PRIORITY

      // For edges.
      .invoke<join, kShardCount>(EdgeMem, edge_done_q, edge_stat_q,
                                 edge_read_addr_q, edge_read_data_q, edges)
      .invoke<detach>(EdgeReqArbiter, edge_req_q, src_q, edge_read_addr_q)

  // For vertices.
  // Route updates via a kShardCount x kShardCount network.
  // clang-format off
#if TAPA_SSSP_SHARD_COUNT >= 2
      .invoke<join>(Switch, 0, switch_done_q[0], switch_stat_q[0], xbar_q0[0], xbar_q0[1], xbar_q1[0], xbar_q1[1])
#endif // TAPA_SSSP_SHARD_COUNT
#if TAPA_SSSP_SHARD_COUNT >= 4
      .invoke<join>(Switch, 0, switch_done_q[1], switch_stat_q[1], xbar_q0[2], xbar_q0[3], xbar_q1[2], xbar_q1[3])
      .invoke<join>(Switch, 1, switch_done_q[2], switch_stat_q[2], xbar_q1[0], xbar_q1[2], xbar_q2[0], xbar_q2[2])
      .invoke<join>(Switch, 1, switch_done_q[3], switch_stat_q[3], xbar_q1[1], xbar_q1[3], xbar_q2[1], xbar_q2[3])
#endif // TAPA_SSSP_SHARD_COUNT
      // clang-format on

      // Distribute updates amount sub-intervals.
      .invoke<detach>(SwitchOutputArbiter,
#if TAPA_SSSP_SHARD_COUNT == 1
                      xbar_q0
#elif TAPA_SSSP_SHARD_COUNT == 2
                      xbar_q1
#elif TAPA_SSSP_SHARD_COUNT == 4
                      xbar_q2
#else  // TAPA_SSSP_SHARD_COUNT
#error "invalid TAPA_SSSP_SHARD_COUNT"
#endif  // TAPA_SSSP_SHARD_COUNT
                      ,
#ifdef TAPA_SSSP_IMMEDIATE_RELAX
                      queue_push_q)
#else   // TAPA_SSSP_IMMEDIATE_RELAX
                      vertex_in_q)
#endif  // TAPA_SSSP_IMMEDIATE_RELAX

      .invoke<detach, kSubIntervalCount>(VertexMem, vertex_read_addr_q,
                                         vertex_read_data_q, vertex_write_req_q,
                                         vertex_write_resp_q, vertices)
      .invoke<detach, kSubIntervalCount>(
          VertexCache, vertex_cache_done_q, vertex_cache_stat_q, vertex_in_q,
          vertex_out_q, update_noop_qi1, read_vid_q, read_vid_q, write_vid_q,
          write_vid_q, vertex_read_addr_q, vertex_read_data_q,
          vertex_write_req_q, vertex_write_resp_q)
      .invoke<detach>(VertexNoopMerger, update_noop_qi1, vertex_noop_q)

      // PEs.
      .invoke<detach, kPeCount>(ProcElemS0, task_req_qi, task_resp_qi,
                                task_count_qi, edge_req_q)
      .invoke<detach>(TaskCountMerger, task_count_qi, task_count_q)
      .invoke<detach, kShardCount>(ProcElemS1, src_q, edge_read_data_q,
                                   xbar_q0);
}

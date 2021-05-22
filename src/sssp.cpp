#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"
#include "sssp-piheap-index.h"
#include "sssp-piheap.h"

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

static_assert(kQueueCount % kShardCount == 0,
              "current implementation requires that queue count is a multiple "
              "of shard count");

// Verbosity definitions:
//   v=5: O(1)
//   v=8: O(#vertex)
//   v=9: O(#edge)

void PiHeapIndexReqArbiter(istreams<HeapIndexReq, kLevelCount>& req_in_q,
                           ostream<packet<LevelId, HeapIndexReq>>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    LevelId level;
    if (find_non_empty(req_in_q, level)) {
      req_out_q.write({level, req_in_q[level].read(nullptr)});
    }
  }
}

void PiHeapIndexRespArbiter(istream<packet<LevelId, HeapIndexResp>>& req_in_q,
                            ostreams<HeapIndexResp, kLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (req_in_q.empty()) continue;
    const auto resp = req_in_q.read(nullptr);
    req_out_q[resp.addr].write(resp.payload);
  }
}

void PiHeapIndex(
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    //
    uint_qid_t qid,
    //
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
  auto& read_hit = op_stats[5];
  auto& read_miss = op_stats[6];
  auto& write_hit = op_stats[7];
  auto& write_miss = op_stats[8];

  ap_uint<bit_length(kPiHeapStatCount[1])> stat_idx = 0;

  shiftreg<uint_pifc_index_t, 16> reading_fresh_pos;
  shiftreg<uint_pifc_index_t, 16> writing_fresh_pos;

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
                writing_fresh_pos.full() ||
                // If acquire index is requested...
                (req.op == ACQUIRE_INDEX &&
                 // memory read must not block.
                 reading_fresh_pos.full())));

    // Determine which vertex to work on.
    const auto vid = can_acquire_index ? ctx.vid : req.vid;
    const auto pifc_index = GetPifcIndex(vid);

    // Read fresh entry.
    auto fresh_entry = fresh_index[pifc_index];

    // Read stale entry.
    bool is_stale_entry_pos_valid;
    uint_pisc_pos_t stale_entry_pos;
    auto stale_entry = GetStaleIndexLocked(
        stale_index, vid, is_stale_entry_pos_valid, stale_entry_pos);
    bool is_stale_entry_updated = false;

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
        resp = {.entry = {}, .yield = false, .enable = false};

        fresh_entry.is_dirty = false;
        fresh_entry.UpdateTag(ctx.vid);
        fresh_entry.index = read_data;
      } else {
        if (read_data.valid()) {
          stale_entry = read_data;
          is_stale_entry_updated = true;
        }

        resp = {.entry = read_data, .yield = false, .enable = true};

        fresh_entry.is_dirty = true;
        fresh_entry.UpdateTag(ctx.vid);
        fresh_entry.index = ctx.entry;
      }
    } else if (can_process_req) {
      is_resp_written = true;

      req_q.read(nullptr);

      const bool is_fresh_entry_hit = fresh_entry.IsHit(req.vid);

      ++op_stats[req.op];

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
        case ACQUIRE_INDEX: {
          CHECK_EQ(pkt.addr, 0);
          CHECK_EQ(req.entry.level(), 0);
          CHECK_EQ(req.entry.index(), 0);

          if (!is_stale_entry_pos_valid || stale_entry.valid()) {
            resp.yield = true;
            break;
          }

          CHECK(is_stale_entry_pos_valid);

          // Fetch entry from memory on miss.
          if (!is_fresh_entry_hit) {
            is_resp_written = false;

            // Write cache entry to memory if necessary.
            if (fresh_entry.is_dirty) {
              write_req_q.write(
                  {fresh_entry.GetVid(pifc_index, qid), fresh_entry.index});
              writing_fresh_pos.push(pifc_index);

              ++write_miss;
            } else {
              ++write_hit;
            }

            // Issue read request and save context.
            read_addr_q.write(req.vid);
            reading_fresh_pos.push(pifc_index);
            acquire_index_ctx_out_q.write({.vid = req.vid, .entry = req.entry});

            ++read_miss;
          } else {  // Read hit.
            if (fresh_entry.index.distance_le(req.entry)) {
              // Implies read_data.valid().
              resp = {.entry = {}, .yield = false, .enable = false};
            } else {
              if (fresh_entry.index.valid()) {
                stale_entry = fresh_entry.index;
                is_stale_entry_updated = true;
              }

              resp = {
                  .entry = fresh_entry.index, .yield = false, .enable = true};

              fresh_entry.is_dirty = true;
              fresh_entry.index = req.entry;
            }

            ++read_hit;
          }
        } break;
        case UPDATE_INDEX: {
          if (stale_entry.distance_eq(req.entry)) {
            CHECK(is_stale_entry_pos_valid);
            stale_entry = req.entry;
            is_stale_entry_updated = true;
          } else {
            // Write cache entry to memory if necessary.
            if (!is_fresh_entry_hit && fresh_entry.is_dirty) {
              write_req_q.write(
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
            write_req_q.write({req.vid, nullptr});
            writing_fresh_pos.push(pifc_index);

            ++write_miss;
          }
        } break;
      }
    }

    if (is_resp_written) {
      resp_q.write({
          .addr = can_acquire_index
                      ? LevelId(0)  // Only the head can acquire index.
                      : pkt.addr,
          .payload = resp,
      });
    }

    if (!can_send_stat && !can_collect_write &&
        (can_acquire_index || can_process_req)) {
      fresh_index[pifc_index] = fresh_entry;
    }

    if (is_stale_entry_updated) {
      SetStaleIndexLocked(stale_entry_pos, vid, stale_entry);
    }
  }
}

void PiHeapPerf(
    //
    istream<bool>& done_q, ostream<PiHeapStat>& stat_q,
    //
    istream<QueueStateUpdate>& queue_state_q) {
  /*
    States w/ index
      + IDLE: queue is idle
        - push_started -> INDEX
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE
      + PUSH: queue is processing PUSH with index already acquired
        - push_started -> INDEX
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE
      + POP: queue is processing POP
        - push_started -> INDEX
        - pop_started -> POP
        - index_acquired -> ERROR
        - idle -> IDLE
      + INDEX: queue is acquring index for PUSH
        - push_started -> ERROR
        - pop_started -> ERROR
        - index_acquired -> PUSH
        - idle -> ERROR

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

exec:
  for (;;) {
    DECL_ARRAY(PiHeapStat, stats, kPiHeapStatCount[0], 0);

    bool is_started = false;

    using namespace queue_state;
    QueueState state = IDLE;

    uint_piheap_size_t current_size = 0;
    uint_piheap_size_t max_size = 0;
    ap_uint<64> total_size = 0;

  spin:
    for (int64_t cycle_count = 0; done_q.empty(); ++cycle_count) {
#pragma HLS pipeline II = 1
      if (is_started) {
        ++stats[state];
        total_size += current_size;
        max_size = std::max(max_size, current_size);
      }
      if (!queue_state_q.empty()) {
        const auto update = queue_state_q.read(nullptr);
        if (update.state != IDLE) {
          is_started = true;
          ++stats[update.state + 3];
        }
        current_size = update.size;

#ifdef TAPA_SSSP_PHEAP_INDEX
        switch (update.state) {
          case IDLE:
          case POP: {
            CHECK_NE(state, INDEX);
            state = update.state;
          } break;
          case PUSH: {
            CHECK_NE(state, INDEX);
            state = INDEX;
          } break;
          case INDEX: {
            CHECK_EQ(state, INDEX);
            state = PUSH;
          } break;
        }
#else   // TAPA_SSSP_PHEAP_INDEX
        state = update.state;
#endif  // TAPA_SSSP_PHEAP_INDEX
      }
    }

    stats[7] = max_size;
    stats[8] = total_size.range(63, 32);
    stats[9] = total_size.range(31, 0);

    done_q.read(nullptr);
  stat:
    for (ap_uint<bit_length(kPiHeapStatCount[0])> i = 0;
         i < kPiHeapStatCount[0]; ++i) {
      stat_q.write(stats[i]);
    }
  }
}

void PiHeapHead(
    // Scalar
    uint_qid_t qid,
    // Queue requests.
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    ostream<QueueOpResp>& queue_resp_q,
    // Internal
    ostream<HeapReq>& req_out_q, istream<HeapResp>& resp_in_q,
    // Statistics.
    ostream<QueueStateUpdate>& queue_state_q,
    // Heap index
    ostream<HeapIndexReq>& index_req_q, istream<HeapIndexResp>& index_resp_q) {
#pragma HLS inline recursive
  const auto cap = GetChildCapOfLevel(0);
  HeapElem<0> root{.valid = false};
  RANGE(i, kPiHeapWidth, root.cap[i] = cap);
  uint_piheap_size_t size = 0;

  CLEAN_UP(clean_up, [&] {
    CHECK_EQ(size, 0);
    CHECK_EQ(root.valid, false) << "q[" << qid << "]";
    RANGE(i, kPiHeapWidth, CHECK_EQ(root.cap[i], cap) << "q[" << qid << "]");
  });

spin:
  for (;;) {
#pragma HLS pipeline off
    bool do_push = false;
    bool do_pop = false;
    const auto push_req = push_req_q.read(do_push);
#ifdef TAPA_SSSP_PHEAP_INDEX
    if (!do_push)
#endif  // TAPA_SSSP_PHEAP_INDEX
      pop_req_q.read(do_pop);

    QueueOp req;
    if (do_push) {
      req.task = push_req;
      req.op = do_pop ? QueueOp::PUSHPOP : QueueOp::PUSH;
      queue_state_q.write(
          {do_pop ? QueueState::PUSHPOP : QueueState::PUSH, size});
    } else if (do_pop) {
      req.task.set_vid(qid);
      req.op = QueueOp::POP;
      queue_state_q.write({QueueState::POP, size});
    } else {
      queue_state_q.write({QueueState::IDLE, size});
      continue;
    }

    switch (req.op) {
      case QueueOp::PUSH: {
        VLOG(5) << "PUSH q[" << qid << "] <-  " << req.task;
      } break;
      case QueueOp::POP: {
        if (root.valid) {
          VLOG(5) << "POP  q[" << qid << "]  -> " << root.task;
        } else {
          VLOG(5) << "POP  q[" << qid << "]  -> {}";
        }
      } break;
      case QueueOp::PUSHPOP: {
        VLOG(5) << "PUSH q[" << qid << "] <-  " << req.task;
        VLOG(5) << "POP  q[" << qid << "]  -> "
                << (root.task <= req.task ? req.task : root.task);
      } break;
    }

    QueueOpResp resp{
        .queue_op = req.op,
        .task_op = TaskOp::NEW,
        .task = req.task,
    };

    switch (req.op) {
      case QueueOp::PUSH: {
        HeapIndexResp heap_index_resp;
#ifdef TAPA_SSSP_PHEAP_INDEX
      acquire:
        do {
#pragma HLS pipeline off
          index_req_q.write({.op = ACQUIRE_INDEX,
                             .vid = req.task.vid(),
                             .entry = {0, 0, req.task.vertex().distance}});
          ap_wait();
          heap_index_resp = index_resp_q.read();
        } while (heap_index_resp.yield);
        queue_state_q.write({QueueState::INDEX, size});
#else   // TAPA_SSSP_PHEAP_INDEX
        heap_index_resp.entry.invalidate();
        heap_index_resp.enable = true;
#endif  // TAPA_SSSP_PHEAP_INDEX
        if (heap_index_resp.enable) {
          PiHeapPush(qid, /*level=*/0,
                     {
                         .index = 0,
                         .op = req.op,
                         .task = req.task,
                         .replace = heap_index_resp.entry.valid(),
                         .vid = req.task.vid(),
                     },
                     root, resp_in_q, req_out_q, index_req_q, index_resp_q);
        }
        if (heap_index_resp.enable && !heap_index_resp.entry.valid()) {
          ++size;
          CHECK_LT(size, kPiHeapCapacity);
        }
      } break;

      case QueueOp::PUSHPOP:
      case QueueOp::POP: {
        if (root.valid && !(req.is_pushpop() && root.task <= req.task)) {
          resp.task = root.task;
          CHECK_EQ(resp.task.vid() % kQueueCount, qid);

#ifdef TAPA_SSSP_PHEAP_INDEX
          index_req_q.write({.op = CLEAR_FRESH, .vid = root.task.vid()});
          ap_wait();
          index_resp_q.read();
#endif  // TAPA_SSSP_PHEAP_INDEX

          PiHeapPop(req, 0, root, resp_in_q, req_out_q);
          if (req.is_pop()) {
            CHECK_GT(size, 0);
            --size;
          }
        } else if (req.is_pop()) {
          resp.task_op = TaskOp::NOOP;
        }
      } break;
    }

    queue_resp_q.write(resp);
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

spin:
  for (bool is_first_time = true;; is_first_time = false) {
#pragma HLS pipeline off
    const auto req = req_in_q.read();
    LOG_IF(INFO, is_first_time)
        << "q[" << qid << "]: off-chip level " << level << " accessed";
    auto idx = req.index;
    CHECK_GE(idx, 0);
    CHECK_LT(idx, GetCapOfLevel(level));

    HeapElemAxi elems[kPiHeapWidth];
#pragma HLS aggregate variable = elems bit

    // Read [idx_base + begin : idx_base + end] to elems[begin : end].
    const LevelIndex idx_base = idx / kPiHeapWidth * kPiHeapWidth;
    const ap_uint<bit_length(kPiHeapWidth)> begin = idx % kPiHeapWidth;
    ap_uint<bit_length(kPiHeapWidth)> end = kPiHeapWidth;
    if (req.op == QueueOp::PUSH) {  // Only need to read one element.
      end = begin + 1;
    } else {
      CHECK_EQ(begin, 0);
    }
  read_elems:
    for (auto i_req = begin, i_resp = begin; i_resp < end;) {
#pragma HLS pipeline II = 1
      if (i_req < end && read_addr_q.try_write(GetAddrOfOffChipHeapElem(
                             level, idx_base + i_req, qid))) {
        if (req.op == QueueOp::PUSH) {
          CHECK_EQ(idx, idx_base + i_req);
          CHECK_EQ(i_req + 1, end);
        }
        ++i_req;
      }
      if (!read_data_q.empty()) {
        elems[i_resp] = HeapElemAxi::Unpack(read_data_q.read(nullptr));
        ++i_resp;
      }
    }

    HeapElemAxi elem;  // Output from IsPiHeapElemUpdated.
    if (IsPiHeapElemUpdated(qid, level, req, elems, idx, elem, req_in_q,
                            resp_out_q, req_out_q, resp_in_q, index_req_q,
                            index_resp_q)) {
      write_req_q.write(
          {GetAddrOfOffChipHeapElem(level, idx, qid), elem.Pack()});
      ap_wait();
      write_resp_q.read();
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
    if (!req_in_q.empty()) {
      const auto req = req_in_q.read(nullptr);
      CHECK_NE(req.op, QueueOp::PUSH);
      const bool is_pushpop = req.op == QueueOp::PUSHPOP;
      resp_out_q.write({
          .op = req.op == is_pushpop ? NOCHANGE : EMPTY,
          .child = 0,
          .task = req.task,
      });
    }
  }
}

#define HEAP_PORTS                                                         \
  uint_qid_t qid, istream<HeapReq>&req_in_q, ostream<HeapResp>&resp_out_q, \
      ostream<HeapReq>&req_out_q, istream<HeapResp>&resp_in_q,             \
      ostream<HeapIndexReq>&index_req_out_q,                               \
      istream<HeapIndexResp>&index_resp_in_q
#define HEAP_BODY(level, mem)                                                  \
  _Pragma("HLS inline recursive");                                             \
  HeapElem<level> heap_array[GetCapOfLevel(level)];                            \
  _Pragma("HLS aggregate variable = heap_array bit");                          \
  DO_PRAGMA(HLS bind_storage variable = heap_array type = RAM_S2P impl = mem); \
  PiHeapBody<level>(qid, heap_array, req_in_q, resp_out_q, req_out_q,          \
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
  for (OffChipLevelId level = 0;; level = level == kOffChipLevelCount - 1
                                              ? OffChipLevelId(0)
                                              : OffChipLevelId(level + 1)) {
#pragma HLS pipeline II = 1
    if (!req_in_q[level].empty() && !req_id_q.full() && !req_out_q.full()) {
      req_id_q.try_write(level);
      req_out_q.try_write(req_in_q[level].read(nullptr));
    }
  }
}

void PiHeapArrayReadDataArbiter(  //
    istream<OffChipLevelId>& req_id_q, istream<HeapElemPacked>& req_in_q,
    ostreams<HeapElemPacked, kOffChipLevelCount>& req_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
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
    istream<TaskOnChip>& push_req_q, istream<bool>& pop_req_q,
    // Queue responses.
    ostream<QueueOpResp>& queue_resp_q,
    //
    ostream<Vid>& piheap_array_read_addr_q,
    istream<HeapElemPacked>& piheap_array_read_data_q,
    ostream<packet<Vid, HeapElemPacked>>& piheap_array_write_req_q,
    istream<bool>& piheap_array_write_resp_q,
    //
    ostream<Vid>& piheap_index_read_addr_q,
    istream<HeapIndexEntry>& piheap_index_read_data_q,
    ostream<packet<Vid, HeapIndexEntry>>& piheap_index_write_req_q,
    istream<bool>& piheap_index_write_resp_q) {
  // Heap rule: child <= parent
  streams<HeapReq, kLevelCount, 2> req_q;
  streams<HeapResp, kLevelCount, 2> resp_q;
  streams<HeapIndexReq, kLevelCount, 2> index_req_qs;
  streams<HeapIndexResp, kLevelCount, 2> index_resp_qs;
  stream<packet<LevelId, HeapIndexReq>, 2> index_req_q;
  stream<packet<LevelId, HeapIndexResp>, 2> index_resp_q;

  stream<OffChipLevelId, 64> array_read_id_q;
  streams<Vid, kOffChipLevelCount, 2> array_read_addr_q;
  streams<HeapElemPacked, kOffChipLevelCount, 2> array_read_data_q;

  stream<OffChipLevelId, 64> array_write_id_q;
  streams<packet<Vid, HeapElemPacked>, kOffChipLevelCount, 2> array_write_req_q;
  streams<bool, kOffChipLevelCount, 2> array_write_resp_q;

  streams<bool, kPiHeapStatTaskCount, 2> done_qi;
  streams<PiHeapStat, kPiHeapStatTaskCount, 2> stat_qi;

  stream<HeapAcquireIndexContext, 64> acquire_index_ctx_q;

  stream<QueueStateUpdate, 2> queue_state_q;

  task()
      .invoke<detach>(PiHeapStatArbiter, done_q, stat_q, done_qi, stat_qi)
      .invoke<detach>(PiHeapPerf, done_qi[0], stat_qi[0], queue_state_q)
      .invoke<detach>(PiHeapHead, qid, push_req_q, pop_req_q, queue_resp_q,
                      req_q[0], resp_q[0], queue_state_q, index_req_qs[0],
                      index_resp_qs[0])
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

      .invoke<detach>(PiHeapIndexReqArbiter, index_req_qs, index_req_q)
      .invoke<detach>(PiHeapIndex, done_qi[1], stat_qi[1], qid, index_req_q,
                      index_resp_q, piheap_index_read_addr_q,
                      piheap_index_read_data_q, piheap_index_write_req_q,
                      piheap_index_write_resp_q, acquire_index_ctx_q,
                      acquire_index_ctx_q)
      .invoke<detach>(PiHeapIndexRespArbiter, index_resp_q, index_resp_qs);
}

void Switch(
    //
    ap_uint<log2(kShardCount)> b,
    //
    istream<TaskOnChip>& in_q0, istream<TaskOnChip>& in_q1,
    //
    ostream<TaskOnChip>& out_q0, ostream<TaskOnChip>& out_q1) {
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
  }
}

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

void UpdateReqArbiter(tapa::istreams<TaskOnChip, kShardCount>& in_q,
                      tapa::ostreams<TaskOnChip, kSubIntervalCount>& out_q) {
  static_assert(kSubIntervalCount % kShardCount == 0,
                "current implementation requires that sub-interval count is a "
                "multiple of shard count");
  DECL_ARRAY(TaskOnChip, update, kShardCount, TaskOnChip());
  DECL_ARRAY(bool, update_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      UNUSED SET(update_valid[sid], in_q[sid].try_read(update[sid]));
      const auto iid =
          update[sid].vid() % kSubIntervalCount / kShardCount * kShardCount +
          sid;
      UNUSED RESET(update_valid[sid], out_q[iid].try_write(update[sid]));
    });
  }
}

void EdgeMem(tapa::istream<Vid>& read_addr_q, tapa::ostream<Edge>& read_data_q,
             tapa::async_mmap<Edge> mem) {
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void ProcElemS0(istream<TaskOnChip>& task_in_q, ostream<Vid>& task_resp_q,
                ostream<EdgeReq>& edge_req_q) {
  EdgeReq req;

spin:
  for (Eid i = 0;;) {
#pragma HLS pipeline II = 1
    if (i == 0 && !task_in_q.empty()) {
      const auto task = task_in_q.read(nullptr);
      req = {task.vertex().offset, {task.vid(), task.vertex().distance}};
      i = task.vertex().degree;
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
#pragma HLS pipeline
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

void NoopMerger(tapa::istreams<bool, kSubIntervalCount>& pkt_in_q,
                ostream<uint_noop_t>& pkt_out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    uint_noop_t count = 0;
    bool buf;
    RANGE(iid, kSubIntervalCount, pkt_in_q[iid].try_read(buf) && ++count);
    if (count) {
      pkt_out_q.write(count);
    }
  }
}

void PushReqArbiter(tapa::istreams<TaskOnChip, kSubIntervalCount>& in_q,
                    tapa::ostreams<TaskOnChip, kQueueCount>& out_q) {
  static_assert(kSubIntervalCount % kQueueCount == 0,
                "current implementation requires that interval count is a "
                "multiple of queue count");
  DECL_ARRAY(TaskOnChip, task, kSubIntervalCount, TaskOnChip());
  DECL_ARRAY(bool, task_valid, kSubIntervalCount, false);

spin:
  for (ap_uint<log2(kSubIntervalCount / kQueueCount)> qid_iid;; ++qid_iid) {
#pragma HLS pipeline II = 1
    RANGE(qid, kQueueCount, {
      const auto iid = qid_iid * kQueueCount + qid;
      UNUSED SET(task_valid[iid], in_q[iid].try_read(task[iid]));
      UNUSED RESET(task_valid[iid], out_q[qid].try_write(task[iid]));
    });
  }
}

void PopReqArbiter(istream<uint_qid_t>& req_in_q,
                   tapa::ostreams<bool, kQueueCount>& req_out_q) {
  DECL_BUF(uint_qid_t, qid);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(qid, req_in_q.try_read(qid), req_out_q[qid].try_write(false));
  }
}

void QueueRespArbiter(tapa::istreams<QueueOpResp, kQueueCount>& resp_in_q,
                      tapa::ostream<QueueOpResp>& resp_out_q) {
  DECL_BUF(QueueOpResp, resp);
spin:
  for (uint8_t q = 0;; ++q) {
#pragma HLS pipeline II = 1
    UPDATE(resp, resp_in_q[q % kQueueCount].try_read(resp),
           resp_out_q.try_write(resp));
  }
}

void TaskReqArbiter(istream<TaskReq>& req_in_q,
                    tapa::ostreams<TaskOnChip, kPeCount>& req_out_q) {
  DECL_BUF(TaskReq, req);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(req, req_in_q.try_read(req),
           req_out_q[req.addr].try_write(req.payload));
  }
}

void TaskRespArbiter(tapa::istreams<Vid, kPeCount>& resp_in_q,
                     ostream<TaskResp>& resp_out_q) {
  DECL_BUF(Vid, vid);
spin:
  for (uint_pe_t pe = 0;; ++pe) {
#pragma HLS pipeline II = 1
    static_assert(is_power_of(kPeCount, 2),
                  "pe needs rounding if kPeCount is not a power of 2");
    UPDATE(vid, resp_in_q[pe].try_read(vid), resp_out_q.try_write({pe, vid}));
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
    ostreams<bool, kQueueCount>& queue_done_q,
    istreams<PiHeapStat, kQueueCount>& queue_stat_q,
    // Task and queue requests.
    ostream<TaskReq>& task_req_q, istream<TaskResp>& task_resp_q,
    istream<uint_noop_t>& update_noop_q, ostream<uint_qid_t>& queue_req_q,
    tapa::istream<QueueOpResp>& queue_resp_q) {
  // Process finished tasks.
  bool queue_buf_valid = false;
  QueueOpResp queue_buf;

  // Number of tasks whose parent task is sent to the PEs but not yet
  // acknowledged.
  int32_t active_task_count = root.vertex.degree;

  // Number of tasks generated by the PEs but not yet sent to the PEs.
  int32_t pending_task_count = 0;

  // Number of POP requests sent but not acknowledged.
  DECL_ARRAY(ap_uint<bit_length(kPeCount / kQueueCount)>, task_count_per_queue,
             kQueueCount, 0);
  DECL_ARRAY(ap_uint<bit_length(kPeCount / kShardCount - 1)>, pe_base_per_shard,
             kShardCount, 0);

  ap_uint<kQueueCount> queue_empty = -1;
  ap_uint<kQueueCount> queue_active = 0;
  ap_uint<kPeCount> pe_active = 0;

  task_req_q.write({root.vid % kShardCount, root});
  ++task_count_per_queue[root.vid % kQueueCount];
  pe_active.set(root.vid % kShardCount);

  // Statistics.
  int32_t visited_edge_count = 0;
  int32_t push_count = 0;
  int32_t pushpop_count = 0;
  int32_t pop_valid_count = 0;
  int32_t pop_noop_count = 0;
  int64_t cycle_count = 0;
  int64_t queue_full_cycle_count = 0;
  int64_t pe_fullcycle_count = 0;
  DECL_ARRAY(int64_t, pe_active_count, kPeCount, 0);

  // Format log messages.
#define STATS(tag, content)                                              \
  do {                                                                   \
    VLOG_F(9, tag) << content " | " << std::setfill(' ') << std::setw(1) \
                   << active_task_count << " active + " << std::setw(2)  \
                   << pending_task_count << " pending tasks";            \
    CHECK_GE(active_task_count, 0);                                      \
    CHECK_GE(pending_task_count, 0);                                     \
  } while (0)

  auto shard_is_done = [&](int sid) {
    bool result = true;
    RANGE(i, kQueueCount / kShardCount,
          result &= queue_empty.bit(i * kShardCount + sid));
    return result;
  };

spin:
  for (; active_task_count || queue_empty.nand_reduce() ||
         any_of(task_count_per_queue);
       ++cycle_count) {
#pragma HLS pipeline II = 1
    RANGE(pe, kPeCount, pe_active.bit(pe) && ++pe_active_count[pe]);
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      const auto qid = queue_buf.task.vid() % kQueueCount;
      queue_empty.clear(qid);
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          --active_task_count;
          ++pending_task_count;

          // Statistics.
          ++push_count;
          break;
        case QueueOp::PUSHPOP:
          CHECK_EQ(queue_buf.task_op, TaskOp::NEW);
          queue_active.clear(qid);
          --active_task_count;
          ++pending_task_count;

          // Statistics.
          ++pushpop_count;
          break;
        case QueueOp::POP:
          queue_active.clear(qid);
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            STATS(recv, "QUEUE: NEW ");

            // Statistics.
            ++pop_valid_count;
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            queue_empty.set(qid);
            STATS(recv, "QUEUE: NOOP");

            // Statistics.
            ++pop_noop_count;
          }
          break;
      }
    }

    {
      const auto qid = cycle_count % kQueueCount;
      if (task_count_per_queue[qid] < kPeCount / kQueueCount &&
          !(queue_buf_valid && queue_buf.task.vid() % kQueueCount == qid) &&
          !queue_active.bit(qid) && !queue_empty.bit(qid)) {
        // Dequeue tasks from the queue.
        if (queue_req_q.try_write(qid)) {
          queue_active.set(qid);
          STATS(send, "QUEUE: POP ");
        } else {
          ++queue_full_cycle_count;
        }
      }
    }

    // Assign tasks to PEs.
    {
      const auto sid = queue_buf.task.vid() % kShardCount;
      const auto pe = pe_base_per_shard[sid] * kShardCount + sid;
      if (queue_buf_valid) {
        if (!pe_active.bit(pe) && task_req_q.try_write({pe, queue_buf.task})) {
          active_task_count += queue_buf.task.vertex().degree;
          --pending_task_count;
          queue_buf_valid = false;
          ++task_count_per_queue[queue_buf.task.vid() % kQueueCount];
          pe_active.set(pe);
        } else {
          ++pe_fullcycle_count;
        }
        ++pe_base_per_shard[sid];
      }
    }

    {
      uint_noop_t count;
      if (update_noop_q.try_read(count)) {
        active_task_count -= count;
        visited_edge_count += count;
      }
    }

    if (!task_resp_q.empty()) {
      const auto resp = task_resp_q.read(nullptr);
      --task_count_per_queue[resp.payload % kQueueCount];
      pe_active.clear(resp.addr);
    }
  }

#ifndef __SYNTHESIS__
  RANGE(qid, kQueueCount, CHECK_EQ(task_count_per_queue[qid], 0));
  CHECK_EQ(queue_empty, decltype(queue_empty)(-1));
  CHECK_EQ(queue_active, 0);
  CHECK_EQ(pe_active, 0);
  CHECK_EQ(active_task_count, 0);
#endif  // __SYNTHESIS__

  RANGE(iid, kSubIntervalCount, vertex_cache_done_q[iid].write(false));
  RANGE(qid, kQueueCount, queue_done_q[qid].write(false));

  metadata[0] = visited_edge_count;
  metadata[1] = push_count;
  metadata[2] = pushpop_count;
  metadata[3] = pop_valid_count;
  metadata[4] = pop_noop_count;
  metadata[6] = cycle_count;
  metadata[7] = queue_full_cycle_count;
  metadata[8] = pe_fullcycle_count;

meta:
  for (int pe = 0; pe < kPeCount; ++pe) {
#pragma HLS pipeline II = 1
    metadata[9 + pe] = pe_active_count[pe];
  }

vertex_cache_stat:
  for (int i = 0; i < kSubIntervalCount; ++i) {
    for (int j = 0; j < kVertexUniStatCount; ++j) {
      metadata[9 + kPeCount + i * kVertexUniStatCount + j] =
          vertex_cache_stat_q[i].read();
    }
  }

queue_stat:
  for (int i = 0; i < kQueueCount; ++i) {
    for (int j = 0; j < kPiHeapStatTotalCount; ++j) {
      metadata[9 + kPeCount + kSubIntervalCount * kVertexUniStatCount +
               i * kPiHeapStatTotalCount + j] = queue_stat_q[i].read();
    }
  }
}

void SSSP(Vid vertex_count, Task root, tapa::mmap<int64_t> metadata,
          tapa::mmaps<Edge, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices,
          // For queues.
          tapa::mmap<HeapElemPacked> heap_array,
          tapa::mmap<HeapIndexEntry> heap_index) {
  streams<TaskOnChip, kSubIntervalCount, 2> push_req_q;
  streams<TaskOnChip, kQueueCount, 512> push_req_qi;
  stream<uint_qid_t, 2> pop_req_q("pop_req");
  streams<bool, kQueueCount, 2> pop_req_qi("pop_req_i");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");
  streams<bool, kQueueCount, 2> queue_done_q;
  streams<PiHeapStat, kQueueCount, 2> queue_stat_q;

  streams<Vid, kQueueCount, 2> piheap_array_read_addr_q;
  streams<HeapElemPacked, kQueueCount, 2> piheap_array_read_data_q;
  streams<packet<Vid, HeapElemPacked>, kQueueCount, 2> piheap_array_write_req_q;
  streams<bool, kQueueCount, 2> piheap_array_write_resp_q;

  streams<Vid, kQueueCount, 2> piheap_index_read_addr_q;
  streams<HeapIndexEntry, kQueueCount, 2> piheap_index_read_data_q;
  streams<packet<Vid, HeapIndexEntry>, kQueueCount, 2> piheap_index_write_req_q;
  streams<bool, kQueueCount, 2> piheap_index_write_resp_q;

  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Vid, kPeCount, 2> task_resp_qi("task_resp_i");

  stream<TaskReq, 2> task_req_q("task_req");
  stream<TaskResp, kPeCount> task_resp_q("task_resp");

  // For edges.
  streams<Vid, kShardCount, 2> edge_read_addr_q("edge_read_addr");
  streams<Edge, kShardCount, 2> edge_read_data_q("edge_read_data");
  streams<EdgeReq, kPeCount, kPeCount / kShardCount * 8> edge_req_q("edge_req");
  streams<SourceVertex, kShardCount, 64> src_q("source_vertices");

  // For vertices.
  //   Connect PEs to the update request network.
  streams<TaskOnChip, kShardCount, 2> update_req_q;
  //   Compose the update request network.
  streams<TaskOnChip, kShardCount, 8> update_req_qi1;
  streams<TaskOnChip, kSubIntervalCount, 8> update_req_qi0;
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

  stream<uint_noop_t, 2> update_noop_q;

  tapa::task()
      .invoke<join>(Dispatcher, root, metadata, vertex_cache_done_q,
                    vertex_cache_stat_q, queue_done_q, queue_stat_q, task_req_q,
                    task_resp_q, update_noop_q, pop_req_q, queue_resp_q)
      .invoke<detach>(TaskReqArbiter, task_req_q, task_req_qi)
      .invoke<detach>(TaskRespArbiter, task_resp_qi, task_resp_q)
      .invoke<detach, kQueueCount>(
          TaskQueue, queue_done_q, queue_stat_q, seq(), push_req_qi, pop_req_qi,
          queue_resp_qi, piheap_array_read_addr_q, piheap_array_read_data_q,
          piheap_array_write_req_q, piheap_array_write_resp_q,
          piheap_index_read_addr_q, piheap_index_read_data_q,
          piheap_index_write_req_q, piheap_index_write_resp_q)
      .invoke<detach>(PushReqArbiter, push_req_q, push_req_qi)
      .invoke<detach>(PopReqArbiter, pop_req_q, pop_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)

      // Put mmaps are in the top level to enable flexible floorplanning.
      .invoke<detach, kQueueCount>(
          PiHeapArrayMem, piheap_array_read_addr_q, piheap_array_read_data_q,
          piheap_array_write_req_q, piheap_array_write_resp_q, heap_array)
      .invoke<detach, kQueueCount>(
          PiHeapIndexMem, piheap_index_read_addr_q, piheap_index_read_data_q,
          piheap_index_write_req_q, piheap_index_write_resp_q, heap_index)

      // For edges.
      .invoke<detach, kShardCount>(EdgeMem, edge_read_addr_q, edge_read_data_q,
                                   edges)
      .invoke<detach>(EdgeReqArbiter, edge_req_q, src_q, edge_read_addr_q)

      // For vertices.
      // Route updates via a kShardCount x kShardCount network.
      // clang-format off
      .invoke<detach>(Switch, 0, update_req_q[0], update_req_q[1], update_req_qi1[0], update_req_qi1[1])
      // clang-format on

      // Distribute updates amount sub-intervals.
      .invoke<detach>(UpdateReqArbiter, update_req_qi1, update_req_qi0)

      .invoke<detach, kSubIntervalCount>(VertexMem, vertex_read_addr_q,
                                         vertex_read_data_q, vertex_write_req_q,
                                         vertex_write_resp_q, vertices)
      .invoke<detach, kSubIntervalCount>(
          VertexCache, vertex_cache_done_q, vertex_cache_stat_q, update_req_qi0,
          push_req_q, update_noop_qi1, read_vid_q, read_vid_q, write_vid_q,
          write_vid_q, vertex_read_addr_q, vertex_read_data_q,
          vertex_write_req_q, vertex_write_resp_q)
      .invoke<detach>(NoopMerger, update_noop_qi1, update_noop_q)

      // PEs.
      .invoke<detach, kPeCount>(ProcElemS0, task_req_qi, task_resp_qi,
                                edge_req_q)
      .invoke<detach, kShardCount>(ProcElemS1, src_q, edge_read_data_q,
                                   update_req_q);
}

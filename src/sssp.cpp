#include <cassert>
#include <cstring>

#include <algorithm>

#include <tapa.h>

#include "sssp-kernel.h"

// Estimated DRAM latency.
constexpr int kMemLatency = 50;

constexpr int kPeCountR0 = 1;
constexpr int kPeCountR1 = (kPeCount - kPeCountR0) / 2;
constexpr int kPeCountR2 = kPeCount - kPeCountR0 - kPeCountR1;

// Verbosity definitions:
//   v=5: O(1)
//   v=6: O(#iteration)
//   v=7: O(#interval)
//   v=8: O(#vertex)
//   v=9: O(#edge)

void Control(const Iid interval_count, const Vid interval_size,
             tapa::mmap<uint64_t> metadata,
             // to VertexMem
             tapa::ostream<VertexReq>& vertex_req_q,
             // to UpdateHandler
             tapa::ostreams<Eid, kPeCount>& update_config_q,
             tapa::ostreams<TaskReq::Phase, kPeCount>& update_phase_q,
             // from UpdateHandler
             tapa::istreams<UpdateCount, kPeCount>& update_count_q,
             // to ProcElem
             tapa::ostreams<TaskReq, kPeCount>& task_req_q,
             // from ProcElem
             tapa::istreams<TaskResp, kPeCount>& task_resp_q) {
  CHECK_LE(interval_count, kMaxIntervalCount);
  CHECK_LE(interval_size, kMaxIntervalSize);

  // HLS crashes without this...
  RANGE(pe, kPeCount, {
    update_config_q[pe].close();
    update_phase_q[pe].close();
    task_req_q[pe].close();
  });

  // Control keeps track of all intervals.

  // Number of edges in each interval of each PE.
  Eid edge_count_local[kMaxIntervalCount][kPeCount];
#pragma HLS array_partition complete variable = edge_count_local dim = 2

  // Number of updates in each interval.
  Eid update_count_local[kMaxIntervalCount];
#pragma HLS array_partition cyclic factor = 2 variable = update_count_local

  // Memory offset of the 0-th edge in each interval (in unit of #edges).
  Eid eid_offsets[kMaxIntervalCount][kPeCount];
#pragma HLS array_partition complete variable = eid_offsets dim = 2

  // Temporary variables for accumulating the edge offsets.
  DECL_ARRAY(Eid, eid_offsets_acc, kPeCount, 0);

  // Collect stats about #edge and #update.
  Eid total_edge_count = 0;
  Eid total_update_count = 0;

load_config:
  for (Iid i = 0; i < interval_count * kPeCount; ++i) {
#pragma HLS pipeline II = 1
    Iid iid = i / kPeCount;
    int pe = i % kPeCount;
    Eid edge_count_delta = metadata[i];
    edge_count_local[iid][pe] = edge_count_delta;
    eid_offsets[iid][pe] = eid_offsets_acc[pe];
    eid_offsets_acc[pe] += edge_count_delta;
  }

  // Initialize UpdateMem, needed only once per execution.
config_update_offsets:
  for (Iid iid = 0; iid < interval_count; ++iid) {
    auto update_offset_v =
        metadata[interval_count * kPeCount + iid] / kUpdateVecLen;
    VLOG_F(7, info) << "update offset[" << iid << "]: " << update_offset_v;
    update_config_q[iid % kPeCount].write(update_offset_v);
  }

  // Tells UpdateHandler start to wait for phase requests.
  RANGE(pe, kPeCount, update_config_q[pe].close());

  bool all_done = false;
  int iter = 0;
bulk_steps:
  while (!all_done) {
    all_done = true;

    // Do the scatter phase for each interval, if active.
    // Wait until all PEs are done with the scatter phase.
    VLOG_F(6, info) << "Phase: " << TaskReq::kScatter;
    RANGE(pe, kPeCount, update_phase_q[pe].write(TaskReq::kScatter));

  scatter:
    for (Iid iid = 0; iid < interval_count; ++iid) {
#pragma HLS pipeline II = 1
      // Tell VertexMem to start broadcasting source vertices.
      vertex_req_q.write({iid});
      RANGE(pe, kPeCount, {
        total_edge_count += edge_count_local[iid][pe];
        task_req_q[pe].write({
            .phase = TaskReq::kScatter,
            .iid = iid,
            .edge_count_v = edge_count_local[iid][pe] / kEdgeVecLen,
            .eid_offset_v = eid_offsets[iid][pe] / kEdgeVecLen,
            .scatter_done = false,  // Unused for scatter.
        });
      });

      ap_wait();

      RANGE(pe, kPeCount, task_resp_q[pe].read());
    }

    // Tell PEs to tell UpdateHandlers that the scatter phase is done.
    TaskReq req = {};
    req.scatter_done = true;
    RANGE(pe, kPeCount, task_req_q[pe].write(req));

    // Get prepared for the gather phase.
    VLOG_F(6, info) << "Phase: " << TaskReq::kGather;
    RANGE(pe, kPeCount, update_phase_q[pe].write(TaskReq::kGather));

  reset_update_count:
    for (Iid iid = 0; iid < interval_count; ++iid) {
#pragma HLS pipeline II = 1
      update_count_local[iid] = 0;
    }

  collect_update_count:
    for (Iid iid_recv = 0;
         iid_recv < tapa::round_up<kPeCount>(interval_count);) {
#pragma HLS pipeline II = 1
#pragma HLS dependence false variable = update_count_local
      UpdateCount update_count_v;
      bool done = false;
      Iid iid;
      RANGE(pe, kPeCount, {
        if (!done && update_count_q[pe].try_read(update_count_v)) {
          done |= true;
          iid = update_count_v.addr * kPeCount + pe;
          ++iid_recv;
        }
      });
      if (done && iid < interval_count) {
        VLOG_F(7, recv) << "update_count_v: " << update_count_v;
        total_update_count += update_count_v.payload * kUpdateVecLen;
        update_count_local[iid] += update_count_v.payload;
      }
    }

    // updates.fence()
    ap_wait_n(80);

    // Do the gather phase for each interval.
    // Wait until all intervals are done with the gather phase.
    DECL_ARRAY(Iid, iid_send, kPeCount, 0);
  gather:
    for (Iid iid_recv = 0; iid_recv < interval_count;) {
#pragma HLS pipeline II = 1
      RANGE(pe, kPeCount, {
        Iid iid = iid_send[pe] * kPeCount + pe;
        if (iid < interval_count) {
          if (task_req_q[pe].try_write({
                  .phase = TaskReq::kGather,
                  .iid = iid,
                  .edge_count_v = update_count_local[iid],
                  .eid_offset_v = 0,      // Unused for gather.
                  .scatter_done = false,  // Unused for gather.
              })) {
            ++iid_send[pe];
          }
        }
      });

      TaskResp resp;
      RANGE(pe, kPeCount, {
        if (task_resp_q[pe].try_read(resp)) {
          VLOG_F(7, recv) << resp;
          if (resp.active) all_done = false;
          ++iid_recv;
        }
      });
    }
    VLOG_F(6, info) << "iter #" << iter << (all_done ? " " : " not ")
                    << "all done";
    ++iter;
  }
  // Terminates UpdateHandler.
  RANGE(pe, kPeCount, {
    update_phase_q[pe].close();
    task_req_q[pe].close();
  });

  metadata[interval_count * kPeCount + interval_count] = iter;
  metadata[interval_count * kPeCount + interval_count + 1] = total_edge_count;
  metadata[interval_count * kPeCount + interval_count + 2] = total_update_count;
}

void VertexMem(const Vid interval_size, tapa::istream<VertexReq>& scatter_req_q,
               tapa::istreams<VertexReq, kPeCountR0 + 1>& vertex_req_q,
               tapa::istreams<VertexAttrVec, kPeCountR0 + 1>& vertex_in_q,
               tapa::ostreams<VertexAttrVec, kPeCountR0 + 1>& vertex_out_q,
               tapa::async_mmap<VidVec>& parents,
               tapa::async_mmap<FloatVec>& distances) {
  constexpr int N = kPeCountR0 + 1;
  const Vid interval_size_v = interval_size / kVertexVecLen;
infinite_loop:
  for (;;) {
    // Prioritize scatter phase broadcast.
    VertexReq req;
    if (scatter_req_q.try_read(req)) {
      // Scatter phase
      //   Send distance to PEs.
      FloatVec resp;
      VertexAttrVec vertex_out;
      bool valid = false;
      DECL_ARRAY(bool, ready, N, false);
    scatter:
      for (Vid i_req = 0, i_resp = 0; i_resp < interval_size_v;) {
#pragma HLS pipeline II = 1
        // Send requests.
        if (i_req < interval_size_v && i_req < i_resp + kMemLatency &&
            distances.read_addr_try_write(req.iid * interval_size_v + i_req)) {
          ++i_req;
        }

        // Handle responses.
        UPDATE(valid, distances.read_data_try_read(resp));
        RANGE(i, kVertexVecLen,
              vertex_out.set(i, VertexAttr{kNullVertex, resp[i]}));
        if (valid) {
          RANGE(pe, N,
                UPDATE(ready[pe], vertex_out_q[pe].try_write(vertex_out)));
          if (All(ready)) {
            ++i_resp;
            valid = false;
            MemSet(ready, false);
          }
        }
      }
    } else {
      bool done = false;
      RANGE(pe, N, {
        if (!done && vertex_req_q[pe].try_read(req)) {
          done |= true;
          // Gather phase
          //   Send parent and distance to PEs.
          //   Recv parent and distance from PEs.

          VidVec resp_parent;
          FloatVec resp_distance;

          // valid_xx: resp_xx is valid
          bool valid_parent = false;
          bool valid_distance = false;

          // xx_ready_oo: write_xx has been written.
          bool addr_ready_parent = false;
          bool data_ready_parent = false;
          bool addr_ready_distance = false;
          bool data_ready_distance = false;

        gather:
          for (Vid i_rd_req_parent = 0, i_rd_req_distance = 0, i_rd_resp = 0,
                   i_wr = 0;
               i_wr < interval_size_v;) {
            _Pragma("HLS pipeline II = 1");
            // Send read requests.
            if (i_rd_req_parent < interval_size_v &&
                i_rd_req_parent < i_rd_resp + kMemLatency &&
                parents.read_addr_try_write(req.iid * interval_size_v +
                                            i_rd_req_parent)) {
              ++i_rd_req_parent;
            }
            if (i_rd_req_distance < interval_size_v &&
                i_rd_req_distance < i_rd_resp + kMemLatency &&
                distances.read_addr_try_write(req.iid * interval_size_v +
                                              i_rd_req_distance)) {
              ++i_rd_req_distance;
            }

            // Handle read responses.
            if (i_rd_resp < interval_size_v) {
              UPDATE(valid_parent, parents.read_data_try_read(resp_parent));
              UPDATE(valid_distance,
                     distances.read_data_try_read(resp_distance));
              VertexAttrVec vertex_out;
              RANGE(i, kVertexVecLen,
                    vertex_out.set(i, {resp_parent[i], resp_distance[i]}));
              if (valid_parent && valid_distance &&
                  vertex_out_q[pe].try_write(vertex_out)) {
                ++i_rd_resp;
                valid_parent = false;
                valid_distance = false;
              }
            }

            // Write to DRAM.
            if (!vertex_in_q[pe].empty()) {
              auto v = vertex_in_q[pe].peek(nullptr);
              VidVec parent_out;
              FloatVec distance_out;
              RANGE(i, kVertexVecLen, {
                parent_out.set(i, v[i].parent);
                distance_out.set(i, v[i].distance);
              });
              uint64_t addr = req.iid * interval_size_v + i_wr;
              UPDATE(addr_ready_distance, distances.write_addr_try_write(addr));
              UPDATE(data_ready_distance,
                     distances.write_data_try_write(distance_out));
              UPDATE(addr_ready_parent, parents.write_addr_try_write(addr));
              UPDATE(data_ready_parent,
                     parents.write_data_try_write(parent_out));
              if (addr_ready_distance && data_ready_distance &&
                  addr_ready_parent && data_ready_parent) {
                vertex_in_q[pe].read(nullptr);
                addr_ready_distance = false;
                data_ready_distance = false;
                addr_ready_parent = false;
                data_ready_parent = false;
                ++i_wr;
              }
            }
          }
        }
      });
    }
  }
}

template <uint64_t N>
void VertexRouterTemplated(
    // scalar
    const Vid interval_size,
    // upstream to VertexMem
    tapa::ostream<VertexReq>& mm_req_q, tapa::istream<VertexAttrVec>& mm_in_q,
    tapa::ostream<VertexAttrVec>& mm_out_q,
    // downstream to ProcElem
    tapa::istreams<VertexReq, N>& pe_req_q,
    tapa::istreams<VertexAttrVec, N>& pe_in_q,
    tapa::ostreams<VertexAttrVec, N>& pe_out_q) {
  TAPA_ELEM_TYPE(pe_req_q) pe_req;
  bool pe_req_valid = false;
  bool mm_req_ready = false;

  TAPA_ELEM_TYPE(mm_in_q) mm_in;
  bool mm_in_valid = false;
  DECL_ARRAY(bool, pe_out_ready, N, false);

  TAPA_ELEM_TYPE(pe_in_q) pe_in;
  bool pe_in_valid = false;
  bool mm_out_ready = false;

  int pe = 0;
  Vid mm2pe_count = 0;
  Vid pe2mm_count = 0;

infinite_loop:
  for (;;) {
#pragma HLS pipeline II = 1
    if (pe2mm_count == 0) {
      // Not processing a gather phase request.

      // Broadcast scatter phase data if any.
      UPDATE(mm_in_valid, mm_in_q.try_read(mm_in));
      if (mm_in_valid) {
#pragma HLS latency max = 1
        RANGE(i, N, UPDATE(pe_out_ready[i], pe_out_q[i].try_write(mm_in)));
        if (All(pe_out_ready)) {
#pragma HLS latency max = 1
          mm_in_valid = false;
          MemSet(pe_out_ready, false);
        }
      }

      // Accept gather phase requests.
      RANGE(i, N, {
        if (!pe_req_valid && pe_req_q[i].try_read(pe_req)) {
          pe_req_valid |= true;
          pe = i;
          mm2pe_count = pe2mm_count = interval_size / kVertexVecLen;
        }
      });
    } else {
      // Processing a gather phase request.

      // Forward vertex attribtues from ProcElem to VertexMem.
      if (!pe_in_valid) pe_in_valid = pe_in_q[pe].try_read(pe_in);
      if (pe_in_valid) {
        if (!mm_out_ready) mm_out_ready = mm_out_q.try_write(pe_in);
        if (mm_out_ready) {
          VLOG_F(7, fwd) << "gather phase: memory <- port " << pe
                         << "; remaining: " << pe2mm_count - 1;
          pe_in_valid = false;
          mm_out_ready = false;
          --pe2mm_count;
        }
      }

      // Forward vertex attribtues from VertexMem to ProcElem.
      if (!mm_in_valid) mm_in_valid = mm_in_q.try_read(mm_in);
      if (mm_in_valid && mm2pe_count > 0) {
        if (!pe_out_ready[pe]) pe_out_ready[pe] = pe_out_q[pe].try_write(mm_in);
        if (pe_out_ready[pe]) {
          VLOG_F(7, fwd) << "gather phase: memory -> port " << pe
                         << "; remaining: " << mm2pe_count - 1;
          mm_in_valid = false;
          pe_out_ready[pe] = false;
          --mm2pe_count;
        }
      }

      // Forward request from ProcElem to VertexMem.
      if (pe_req_valid) {
        if (!mm_req_ready) mm_req_ready = mm_req_q.try_write(pe_req);
        if (mm_req_ready) {
          VLOG_F(7, recv) << "fulfilling request from port " << pe;
          pe_req_valid = false;
          mm_req_ready = false;
        }
      }
    }
  }
}

void VertexRouterR1(
    // scalar
    const Vid interval_size,
    // upstream to VertexMem
    tapa::ostream<VertexReq>& mm_req_q, tapa::istream<VertexAttrVec>& mm_in_q,
    tapa::ostream<VertexAttrVec>& mm_out_q,
    // downstream to ProcElem
    tapa::istreams<VertexReq, kPeCountR1 + 1>& pe_req_q,
    tapa::istreams<VertexAttrVec, kPeCountR1 + 1>& pe_in_q,
    tapa::ostreams<VertexAttrVec, kPeCountR1 + 1>& pe_out_q) {
#pragma HLS inline region
  VertexRouterTemplated(interval_size, mm_req_q, mm_in_q, mm_out_q, pe_req_q,
                        pe_in_q, pe_out_q);
}

void VertexRouterR2(
    // scalar
    const Vid interval_size,
    // upstream to VertexMem
    tapa::ostream<VertexReq>& mm_req_q, tapa::istream<VertexAttrVec>& mm_in_q,
    tapa::ostream<VertexAttrVec>& mm_out_q,
    // downstream to ProcElem
    tapa::istreams<VertexReq, kPeCountR2>& pe_req_q,
    tapa::istreams<VertexAttrVec, kPeCountR2>& pe_in_q,
    tapa::ostreams<VertexAttrVec, kPeCountR2>& pe_out_q) {
#pragma HLS inline region
  VertexRouterTemplated(interval_size, mm_req_q, mm_in_q, mm_out_q, pe_req_q,
                        pe_in_q, pe_out_q);
}

// Handles edge read requests.
void EdgeMem(tapa::istream<Eid>& edge_req_q,
             tapa::ostream<EdgeVec>& edge_resp_q,
             tapa::async_mmap<EdgeVec> edges) {
  bool valid = false;
  EdgeVec edge_v;
infinite_loop:
  for (;;) {
#pragma HLS pipeline II = 1
    // Handle responses.
    if (UPDATE(valid, edges.read_data_try_read(edge_v))) {
      if (edge_resp_q.try_write(edge_v)) valid = false;
    }

    // Handle requests.
    if (!edge_req_q.empty() &&
        edges.read_addr_try_write(edge_req_q.peek(nullptr))) {
      edge_req_q.read(nullptr);
    }
  }
}

void UpdateMem(tapa::istream<uint64_t>& read_addr_q,
               tapa::ostream<UpdateVec>& read_data_q,
               tapa::istream<uint64_t>& write_addr_q,
               tapa::istream<UpdateVec>& write_data_q,
               tapa::async_mmap<UpdateVec> updates) {
  bool valid = false;
  UpdateVec update_v;
infinite_loop:
  for (;;) {
#pragma HLS pipeline II = 1
    // Handle read responses.
    if (UPDATE(valid, updates.read_data_try_read(update_v))) {
      if (read_data_q.try_write(update_v)) valid = false;
    }

    // Handle read requests.
    if (!read_addr_q.empty() &&
        updates.read_addr_try_write(read_addr_q.peek(nullptr))) {
      read_addr_q.read(nullptr);
    }

    // Handle write requests.
    if (!write_addr_q.empty() && !write_data_q.empty() &&
        updates.write_addr_try_write(write_addr_q.peek(nullptr))) {
      write_addr_q.read(nullptr);
      updates.write_data_write(write_data_q.read(nullptr));
    }
  }
}

void UpdateHandler(Iid interval_count,
                   // from Control
                   tapa::istream<Eid>& update_config_q,
                   tapa::istream<TaskReq::Phase>& update_phase_q,
                   // to Control
                   tapa::ostream<UpdateCount>& num_updates_out_q,
                   // from ProcElem via UpdateRouter
                   tapa::istream<UpdateReq>& update_req_q,
                   tapa::istream<UpdateVecPacket>& update_in_q,
                   // to ProcElem via UpdateReorderer
                   tapa::ostream<UpdateVec>& update_out_q,
                   // to and from UpdateMem
                   tapa::ostream<uint64_t>& updates_read_addr_q,
                   tapa::istream<UpdateVec>& updates_read_data_q,
                   tapa::ostream<uint64_t>& updates_write_addr_q,
                   tapa::ostream<UpdateVec>& updates_write_data_q) {
  // HLS crashes without this...
  update_config_q.open();
  update_phase_q.open();
  ap_wait();
  update_in_q.open();
  ap_wait();
  update_out_q.close();

  // Memory offsets of each update interval.
  Eid update_offsets[tapa::round_up_div<kPeCount>(kMaxIntervalCount)];
#pragma HLS resource variable = update_offsets latency = 4

  // Number of updates of each update interval in memory (in unit of UpdateVec).
  Eid update_counts[tapa::round_up_div<kPeCount>(kMaxIntervalCount)];

num_updates_init:
  for (Iid i = 0; i < tapa::round_up_div<kPeCount>(interval_count); ++i) {
#pragma HLS pipeline II = 1
    update_counts[i] = 0;
  }

  // Initialization; needed only once per execution.
  int update_offset_idx = 0;
update_offset_init:
  TAPA_WHILE_NOT_EOS(update_config_q) {
#pragma HLS pipeline II = 1
    update_offsets[update_offset_idx] = update_config_q.read(nullptr);
    ++update_offset_idx;
  }
  update_config_q.open();

update_phases:
  TAPA_WHILE_NOT_EOS(update_phase_q) {
    const auto phase = update_phase_q.read();
    VLOG_F(6, recv) << "Phase: " << phase;
    if (phase == TaskReq::kScatter) {
      // kScatter lasts until update_phase_q is non-empty.
      Iid last_last_iid = -1;
      Iid last_iid = -1;
      Eid last_update_idx = -1;
    update_writes:
      TAPA_WHILE_NOT_EOS(update_in_q) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = update_counts inter true distance = 2
        const auto peek_iid = update_in_q.peek(nullptr).addr;
        if (peek_iid != last_iid && peek_iid == last_last_iid) {
          // insert bubble
          last_last_iid = -1;
        } else {
          const auto update_with_iid = update_in_q.read(nullptr);
          VLOG_F(5, recv) << "UpdateWithIid: " << update_with_iid;
          const Iid iid = update_with_iid.addr;
          const UpdateVec& update_v = update_with_iid.payload;

          // number of updates already written to current interval, not
          // including the current update
          Eid update_idx;
          if (last_iid != iid) {
#pragma HLS latency min = 1 max = 1
            update_idx = update_counts[iid / kPeCount];
            if (last_iid != -1) {
              update_counts[last_iid / kPeCount] = last_update_idx;
            }
          } else {
            update_idx = last_update_idx;
          }

          // set for next iteration
          last_last_iid = last_iid;
          last_iid = iid;
          last_update_idx = update_idx + 1;

          Eid update_offset = update_offsets[iid / kPeCount] + update_idx;
          updates_write_addr_q.write(update_offset);
          updates_write_data_q.write(update_v);
        }
      }
      if (last_iid != -1) {
        update_counts[last_iid / kPeCount] = last_update_idx;
      }
      update_in_q.open();
      ap_wait_n(1);
    send_num_updates:
      for (Iid i = 0; i < tapa::round_up_div<kPeCount>(interval_count); ++i) {
        // TODO: store relevant intervals only
        VLOG_F(7, send) << "update_count_v[" << i << "]: " << update_counts[i];
        num_updates_out_q.write({i, update_counts[i]});
        update_counts[i] = 0;  // Reset for the next scatter phase.
      }
    } else {
      // Gather phase.
    recv_update_reqs:
      for (UpdateReq update_req; update_phase_q.empty();) {
        if (update_req_q.try_read(update_req)) {
          const auto iid = update_req.iid;
          const auto update_count_v = update_req.update_count_v;
          VLOG_F(7, recv) << "UpdateReq: " << update_req;

          bool valid = false;
          UpdateVec update_v;
        update_reads:
          for (Eid i_rd = 0, i_wr = 0; i_rd < update_count_v;) {
            auto read_addr = update_offsets[iid / kPeCount] + i_wr;
            if (i_wr < update_count_v &&
                updates_read_addr_q.try_write(read_addr)) {
              VLOG_F(9, req) << "UpdateVec[" << read_addr << "]";
              ++i_wr;
            }

            if (UPDATE(valid, updates_read_data_q.try_read(update_v)) &&
                update_out_q.try_write(update_v)) {
              VLOG_F(9, send) << "Update: " << update_v;
              ++i_rd;
              valid = false;
            }
          }
          update_out_q.close();
        }
      }
    }
  }
  VLOG_F(5, info) << "done";
  update_phase_q.open();
}

void ProcElem(
    // scalar
    const Vid interval_size,
    // from Control
    tapa::istream<TaskReq>& task_req_q,
    // to Control
    tapa::ostream<TaskResp>& task_resp_q,
    // to and from VertexMem
    tapa::ostream<VertexReq>& vertex_req_q,
    tapa::istream<VertexAttrVec>& vertex_in_q,
    tapa::ostream<VertexAttrVec>& vertex_out_q,
    // to and from EdgeMem
    tapa::ostream<Eid>& edge_req_q, tapa::istream<EdgeVec>& edge_resp_q,
    // to UpdateHandler
    tapa::ostream<UpdateReq>& update_req_q,
    // from UpdateHandler via UpdateReorderer
    tapa::istream<UpdateVec>& update_in_q,
    // to UpdateHandler via UpdateRouter
    tapa::ostream<UpdateVecPacket>& update_out_q) {
  // HLS crashes without this...
  task_req_q.open();
  ap_wait();
  update_out_q.close();
  ap_wait();
  update_in_q.open();

  VertexAttr vertices_local[kMaxIntervalSize];
#pragma HLS array_partition variable = vertices_local cyclic factor = \
    kVertexPartitionFactor
#pragma HLS resource variable = vertices_local core = RAM_S2P_URAM

task_requests:
  TAPA_WHILE_NOT_EOS(task_req_q) {
    const auto req = task_req_q.read();
    const Vid vid_offset = req.iid * interval_size;
    VLOG_F(6, recv) << "TaskReq: " << req;
    if (req.scatter_done) {
      update_out_q.close();
    } else {
      bool is_active = false;
      if (req.phase == TaskReq::kScatter) {
      vertex_reads:
        for (Vid i = 0; i * kVertexVecLen < interval_size; ++i) {
#pragma HLS pipeline II = 1
          auto vertex_vec = vertex_in_q.read();
          VLOG_F(8, recv) << "VertexAttrVec: " << vertex_vec;
          RANGE(j, kVertexVecLen,
                vertices_local[i * kVertexVecLen + j] = vertex_vec[j]);
        }

      edge_reads:
        for (Eid eid_resp = 0, eid_req = 0; eid_resp < req.edge_count_v;) {
#pragma HLS pipeline II = 1
          if (eid_req < req.edge_count_v && eid_resp < eid_req + kMemLatency &&
              edge_req_q.try_write(req.eid_offset_v + eid_req)) {
            ++eid_req;
          }
          EdgeVec edge_v;
          // empty edge is indicated by src == kNullVertex
          // first edge in a vector must have valid dst for routing purpose
          if (edge_resp_q.try_read(edge_v)) {
            VLOG_F(9, recv) << "Edge: " << edge_v;
            UpdateVecPacket update_v;
            update_v.addr = edge_v[0].dst / interval_size;
            update_v.payload.set(Update{kNullVertex, kNullVertex});
            bool is_valid_update = false;
            RANGE(i, kEdgeVecLen, {
              const auto& edge = edge_v[i];
              if (edge.src != kNullVertex) {
                auto addr = edge.src - vid_offset;
                CHECK_EQ(addr % kEdgeVecLen, i)
                    << " incorrect edge vector: " << edge_v
                    << " (vid_offset: " << vid_offset << ")";
                addr /= kEdgeVecLen;
                auto vertex_attr = vertices_local[addr * kEdgeVecLen + i];
                if (vertex_attr.distance < kInfDistance) {
                  is_valid_update |= true;
                  update_v.payload.set(
                      edge.dst % kEdgeVecLen,
                      {edge.src, edge.dst, vertex_attr.distance + edge.weight});
                }
              }
            });
            if (is_valid_update) {
              update_out_q.write(update_v);
              VLOG_F(9, send) << "Update: " << update_v;
            }
            ++eid_resp;
          }
        }
      } else {
      vertex_resets:
        for (Vid i = 0; i < interval_size; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS unroll factor = kVertexPartitionFactor
          vertices_local[i] = {kNullVertex, kInfDistance};
        }

        update_req_q.write({req.phase, req.iid, req.edge_count_v});
      update_reads:
        TAPA_WHILE_NOT_EOS(update_in_q) {
#pragma HLS pipeline II = 1
#pragma HLS dependence variable = vertices_local inter true distance = \
    kVertexUpdateDepDist
          auto update_v = update_in_q.read(nullptr);
          VLOG_F(9, recv) << "Update: " << update_v;
          RANGE(i, kUpdateVecLen, ({
                  auto update = update_v[i];
                  if (update.dst != kNullVertex) {
                    auto addr = update.dst - vid_offset;
                    CHECK_EQ(addr % kEdgeVecLen, i)
                        << " incorrect update vector: " << update_v
                        << " (vid_offset: " << vid_offset << ")";
                    addr /= kEdgeVecLen;
                    auto& vertex_attr = vertices_local[addr * kEdgeVecLen + i];
                    if (update.new_distance < vertex_attr.distance) {
                      vertex_attr = {update.src, update.new_distance};
                    }
                  }
                }));
        }
        update_in_q.open();

        vertex_req_q.write({req.iid});

      vertex_writes:
        for (Vid i = 0; i * kVertexVecLen < interval_size; ++i) {
#pragma HLS pipeline II = 1
          VertexAttrVec vertex_vec = vertex_in_q.read();
          bool is_active_local = false;
          RANGE(j, kVertexVecLen, ({
                  auto vertex = vertex_vec[j];
                  auto new_vertex = vertices_local[i * kVertexVecLen + j];
                  if (new_vertex.distance < vertex.distance) {
                    is_active_local |= true;
                    vertex = new_vertex;
                  }
                  vertex_vec.set(j, vertex);
                  VLOG_F(8, send) << "VertexAttr[" << j << "]: " << vertex;
                }));
          is_active |= is_active_local;
          vertex_out_q.write(vertex_vec);
          VLOG_F(8, send) << "VertexAttrVec: " << vertex_vec;
        }
      }
      TaskResp resp{is_active};
      task_resp_q.write(resp);
    }
  }
  task_req_q.open();
}

void SSSP(Iid interval_count, Vid interval_size, tapa::mmap<uint64_t> metadata,
          tapa::async_mmap<VidVec> parents,
          tapa::async_mmap<FloatVec> distances,
          tapa::async_mmaps<EdgeVec, kPeCount> edges,
          tapa::async_mmaps<UpdateVec, kPeCount> updates) {
  // between Control and ProcElem
  tapa::streams<TaskReq, kPeCount, 2> task_req("task_req");
  tapa::streams<TaskResp, kPeCount, 2> task_resp("task_resp");

  // between Control and VertexMem
  tapa::stream<VertexReq, 2> scatter_phase_vertex_req(
      "scatter_phase_vertex_req");

  // between ProcElem and VertexMem in region 0
  tapa::streams<VertexReq, kPeCountR0 + 1, 2> vertex_req_r0("vertex_req_r0");
  tapa::streams<VertexAttrVec, kPeCountR0 + 1, 2> vertex_pe2mm_r0(
      "vertex_pe2mm_r0");
  tapa::streams<VertexAttrVec, kPeCountR0 + 1, 2> vertex_mm2pe_r0(
      "vertex_mm2pe_r0");

  // between ProcElem and VertexMem in region 1
  tapa::streams<VertexReq, kPeCountR1 + 1, 2> vertex_req_r1("vertex_req_r1");
  tapa::streams<VertexAttrVec, kPeCountR1 + 1, 2> vertex_pe2mm_r1(
      "vertex_pe2mm_r1");
  tapa::streams<VertexAttrVec, kPeCountR1 + 1, 2> vertex_mm2pe_r1(
      "vertex_mm2pe_r1");

  // between ProcElem and VertexMem in region 2
  tapa::streams<VertexReq, kPeCountR2, 2> vertex_req_r2("vertex_req_r2");
  tapa::streams<VertexAttrVec, kPeCountR2, 2> vertex_pe2mm_r2(
      "vertex_pe2mm_r2");
  tapa::streams<VertexAttrVec, kPeCountR2, 2> vertex_mm2pe_r2(
      "vertex_mm2pe_r2");

  // between ProcElem and EdgeMem
  tapa::streams<Eid, kPeCount, 2> edge_req("edge_req");
  tapa::streams<EdgeVec, kPeCount, 2> edge_resp("edge_resp");

  // between Control and UpdateHandler
  tapa::streams<Eid, kPeCount, 2> update_config("update_config");
  tapa::streams<TaskReq::Phase, kPeCount, 2> update_phase("update_phase");

  // between UpdateHandler and ProcElem
  tapa::streams<UpdateReq, kPeCount, 2> update_req("update_req");

  // between UpdateHandler and UpdateMem
  tapa::streams<uint64_t, kPeCount, 2> update_read_addr("update_read_addr");
  tapa::streams<UpdateVec, kPeCount, 2> update_read_data("update_read_data");
  tapa::streams<uint64_t, kPeCount, 2> update_write_addr("update_write_addr");
  tapa::streams<UpdateVec, kPeCount, 2> update_write_data("update_write_data");

  tapa::streams<UpdateVecPacket, kPeCount, 2> update_pe2mm("update_pe2mm");
  tapa::streams<UpdateVec, kPeCount, 2> update_mm2pe("update_mm2pe");

  tapa::streams<UpdateCount, kPeCount, 2> update_count("update_count");

  tapa::task()
      .invoke<-1>(VertexMem, "VertexMem", interval_size,
                  scatter_phase_vertex_req, vertex_req_r0, vertex_pe2mm_r0,
                  vertex_mm2pe_r0, parents, distances)
      .invoke<-1>(VertexRouterR1, "VertexRouterR1", interval_size,
                  vertex_req_r0[kPeCountR0], vertex_mm2pe_r0[kPeCountR0],
                  vertex_pe2mm_r0[kPeCountR0], vertex_req_r1, vertex_pe2mm_r1,
                  vertex_mm2pe_r1)
      .invoke<-1>(VertexRouterR2, "VertexRouterR2", interval_size,
                  vertex_req_r1[kPeCountR1], vertex_mm2pe_r1[kPeCountR1],
                  vertex_pe2mm_r1[kPeCountR1], vertex_req_r2, vertex_pe2mm_r2,
                  vertex_mm2pe_r2)
      .invoke<-1, kPeCount>(EdgeMem, "EdgeMem", edge_req, edge_resp, edges)
      .invoke<-1, kPeCount>(UpdateMem, "UpdateMem", update_read_addr,
                            update_read_data, update_write_addr,
                            update_write_data, updates)
      .invoke<0>(ProcElem, "ProcElem[0]", interval_size, task_req[0],
                 task_resp[0], vertex_req_r0[0], vertex_mm2pe_r0[0],
                 vertex_pe2mm_r0[0], edge_req[0], edge_resp[0], update_req[0],
                 update_mm2pe[0], update_pe2mm[0])
      .invoke<0>(ProcElem, "ProcElem[1]", interval_size, task_req[1],
                 task_resp[1], vertex_req_r1[0], vertex_mm2pe_r1[0],
                 vertex_pe2mm_r1[0], edge_req[1], edge_resp[1], update_req[1],
                 update_mm2pe[1], update_pe2mm[1])
      .invoke<0>(ProcElem, "ProcElem[2]", interval_size, task_req[2],
                 task_resp[2], vertex_req_r1[1], vertex_mm2pe_r1[1],
                 vertex_pe2mm_r1[1], edge_req[2], edge_resp[2], update_req[2],
                 update_mm2pe[2], update_pe2mm[2])
      .invoke<0>(ProcElem, "ProcElem[3]", interval_size, task_req[3],
                 task_resp[3], vertex_req_r1[2], vertex_mm2pe_r1[2],
                 vertex_pe2mm_r1[2], edge_req[3], edge_resp[3], update_req[3],
                 update_mm2pe[3], update_pe2mm[3])
      .invoke<0>(ProcElem, "ProcElem[4]", interval_size, task_req[4],
                 task_resp[4], vertex_req_r2[0], vertex_mm2pe_r2[0],
                 vertex_pe2mm_r2[0], edge_req[4], edge_resp[4], update_req[4],
                 update_mm2pe[4], update_pe2mm[4])
      .invoke<0>(ProcElem, "ProcElem[5]", interval_size, task_req[5],
                 task_resp[5], vertex_req_r2[1], vertex_mm2pe_r2[1],
                 vertex_pe2mm_r2[1], edge_req[5], edge_resp[5], update_req[5],
                 update_mm2pe[5], update_pe2mm[5])
      .invoke<0>(ProcElem, "ProcElem[6]", interval_size, task_req[6],
                 task_resp[6], vertex_req_r2[2], vertex_mm2pe_r2[2],
                 vertex_pe2mm_r2[2], edge_req[6], edge_resp[6], update_req[6],
                 update_mm2pe[6], update_pe2mm[6])
      .invoke<0>(ProcElem, "ProcElem[7]", interval_size, task_req[7],
                 task_resp[7], vertex_req_r2[3], vertex_mm2pe_r2[3],
                 vertex_pe2mm_r2[3], edge_req[7], edge_resp[7], update_req[7],
                 update_mm2pe[7], update_pe2mm[7])
      .invoke<0>(Control, "Control", interval_count, interval_size, metadata,
                 scatter_phase_vertex_req, update_config, update_phase,
                 update_count, task_req, task_resp)
      .invoke<0, kPeCount>(UpdateHandler, "UpdateHandler", interval_count,
                           update_config, update_phase, update_count,
                           update_req, update_pe2mm, update_mm2pe,
                           update_read_addr, update_read_data,
                           update_write_addr, update_write_data);
}

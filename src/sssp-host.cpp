#include <cstdint>

#include <array>
#include <iomanip>
#include <vector>

#include <glog/logging.h>
#include <tapa.h>

#include "sssp.h"

using std::array;
using std::fixed;
using std::setfill;
using std::setprecision;
using std::setw;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

double Log(const aligned_vector<int64_t>& metadata,
           const array<aligned_vector<Edge>, kShardCount>& shards,
           int64_t connected_edge_count, int64_t spill_capacity,
           double elapsed_time, double refine_time) {
  {
    VLOG(3) << "kernel time: " << elapsed_time << " s";
    auto teps = connected_edge_count / (elapsed_time + refine_time);
    auto metadata_it = metadata.begin();
    const auto visited_edge_count = *(metadata_it++);
    const auto visited_vertex_count = *(metadata_it++);
    const auto push_noop_count = *(metadata_it++);
    const auto pop_noop_count = *(metadata_it++);
    const auto cycle_count = *(metadata_it++);
    // #(EF -> CVC) = visited_edge_count + 1, because root is not visited by EF
    // #(CVC -> EF) = pop_valid_count - 1, because root is not processed by CVC
    // #(CVC -> CGPQ) = #(EF -> CVC) - push_noop_count
    // #(CGPQ -> CVC) = #(CVC -> EF) + pop_noop_count
    const auto push_count = visited_edge_count - push_noop_count;
    const auto pop_count = visited_vertex_count + pop_noop_count;
    int64_t coarsened_edge_count = 0;
    for (auto& shard : shards) coarsened_edge_count += shard.size();
    VLOG(3) << "  TEPS:                 " << setfill(' ') << setw(10) << teps
            << " (" << 1e9 * elapsed_time / visited_edge_count
            << " ns/edge visited + " << refine_time << " s refinement)";
    VLOG(3) << "  #edges connected:     " << setfill(' ') << setw(10)
            << connected_edge_count;
    VLOG(3) << "  #edges visited:       " << setfill(' ') << setw(10)
            << visited_edge_count << " (" << fixed << setprecision(1)
            << std::showpos
            << 100. * visited_edge_count / coarsened_edge_count - 100
            << "% over " << std::noshowpos << coarsened_edge_count << ") ";
    VLOG(3) << "  #vertices visited:    " << setfill(' ') << setw(10)
            << visited_vertex_count << " (" << setw(4) << fixed
            << setprecision(1)
            << 100. * visited_vertex_count / visited_edge_count << "%)";
    VLOG(3) << "  #discarded by update: " << setfill(' ') << setw(10)
            << push_noop_count << " (" << setw(4) << fixed << setprecision(1)
            << 100. * push_noop_count / visited_edge_count << "%)";
    VLOG(3) << "  #discarded by filter: " << setfill(' ') << setw(10)
            << pop_noop_count << " (" << setw(4) << fixed << setprecision(1)
            << 100. * pop_noop_count / visited_edge_count << "%)";
    VLOG(3) << "  #push:                " << setfill(' ') << setw(10)
            << push_count;
    VLOG(3) << "  #pop:                 " << setfill(' ') << setw(10)
            << pop_count;
    VLOG(3) << "  cycle count:          " << setfill(' ') << setw(10)
            << cycle_count;

    for (int iid = 0; iid < kIntervalCount; ++iid) {
      for (int siid = 0; siid < kSubIntervalCount / kIntervalCount; ++siid) {
        VLOG(3) << "  interval[" << iid << "][" << siid << "]:";

        // Cache hit/miss.
        const auto read_hit = *(metadata_it++);
        const auto read_total = *(metadata_it++) + read_hit;
        const auto write_hit = *(metadata_it++);
        const auto write_total = *(metadata_it++) + write_hit;
        VLOG_IF(3, read_total)
            << "    read hit   : " << setfill(' ') << setw(10) << read_hit
            << " / " << read_total << " (" << fixed << setprecision(1)
            << 100. * read_hit / read_total << "%)";
        VLOG_IF(3, write_total)
            << "    write hit  : " << setfill(' ') << setw(10) << write_hit
            << " / " << write_total << " (" << fixed << setprecision(1)
            << 100. * write_hit / write_total << "%)";

        // Op counts.
        constexpr const char* kVertexUnitOpNamesAligned[] = {
            "#write_resp",  //
            "#read_resp ",  //
            "#push_busy ",  //
            "#pop_busy  ",  //
            "#noop_busy ",  //
            "#entry_busy",  //
            "#read_busy ",  //
            "#write_busy",  //
            "#idle      ",  //
        };
        constexpr int kVertexOpStatCount = sizeof(kVertexUnitOpNamesAligned) /
                                           sizeof(kVertexUnitOpNamesAligned[0]);

        int64_t op_counts[kVertexOpStatCount];
        int64_t total_op_count = read_total;
        for (int i = 0; i < kVertexOpStatCount; ++i) {
          op_counts[i] = *(metadata_it++);
          total_op_count += op_counts[i];
        }

        for (int i = 0; i < kVertexOpStatCount; ++i) {
          VLOG_IF(3, op_counts[i])
              << "    " << kVertexUnitOpNamesAligned[i] << ": " << setfill(' ')
              << setw(10) << op_counts[i] << " / " << total_op_count << " ("
              << fixed << setprecision(1) << setw(4)
              << 100. * op_counts[i] / total_op_count << "%)";
        }
      }
    }

    int64_t edge_vec_count = 0;
    for (int sid = 0; sid < kShardCount; ++sid) {
      VLOG(3) << "  shard[" << sid << "]:";

      constexpr const char* kEdgeUnitOpNamesAligned[] = {
          "active   ",  //
          "mem stall",  //
          "PE stall ",  //
      };
      const auto cycle_count = *(metadata_it++);
      for (int i = 0; i < kEdgeUnitStatCount - 1; ++i) {
        const auto op_count = *(metadata_it++);
        VLOG(3) << "    " << kEdgeUnitOpNamesAligned[i] << ": " << setfill(' ')
                << setw(10) << op_count << " ( " << fixed << setprecision(1)
                << setw(5) << 100. * op_count / cycle_count << "%)";
        if (i == 0) {
          edge_vec_count += op_count;
        }
      }
    }
    LOG(INFO) << "  " << setfill(' ') << fixed << setprecision(1) << setw(3)
              << (100. -
                  100. * visited_edge_count / (edge_vec_count * kEdgeVecLen))
              << "% edges are null";

    {
      for (int bank = 0; bank < kCgpqPushPortCount; ++bank) {
        VLOG(3) << "  bank[" << bank << "]:";

        const auto spill_count = *(metadata_it++);
        const auto max_heap_size = *(metadata_it++);
        const auto cycle_count = *(metadata_it++);
        VLOG(3) << "    spill count  : " << setfill(' ') << setw(10)
                << spill_count << " / " << spill_capacity;
        VLOG(3) << "    max heap size: " << setfill(' ') << setw(10)
                << max_heap_size << " / " << kCgpqCapacity;
        auto vlog = [&](const char* msg, int64_t var) {
          VLOG(3) << "    " << msg << ": " << setfill(' ') << setw(10) << var
                  << " / " << cycle_count << " (" << fixed << setprecision(1)
                  << 100. * var / cycle_count << "%)";
        };

        vlog("push blocked by full buffer   ", *(metadata_it++));
        vlog("push blocked by current refill", *(metadata_it++));
        vlog("push blocked by future refill ", *(metadata_it++));
        vlog("push blocked by bank conflict ", *(metadata_it++));

        vlog("pop blocked by full FIFO    ", *(metadata_it++));
        vlog("pop blocked by spill        ", *(metadata_it++));
        vlog("pop blocked by bank conflict", *(metadata_it++));
        vlog("pop blocked by alignment    ", *(metadata_it++));
      }
      return teps;
    }
  }
}

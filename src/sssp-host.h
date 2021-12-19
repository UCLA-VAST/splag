
#ifndef TAPA_SSSP_HOST_H_
#define TAPA_SSSP_HOST_H_

#include <cstdint>

#include <array>
#include <vector>

#include <tapa.h>

#include "sssp.h"

// Prints logging messages and returns TEPS.
double Log(
    const std::vector<int64_t, tapa::aligned_allocator<int64_t>>& metadata,
    const std::array<std::vector<Edge, tapa::aligned_allocator<Edge>>,
                     kShardCount>& shards,
    int64_t connected_edge_count, int64_t spill_capacity, double elapsed_time,
    double refine_time);

#endif  // TAPA_SSSP_HOST_H_

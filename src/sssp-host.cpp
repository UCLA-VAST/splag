#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <tapa.h>

#include "sssp.h"
#include "util.h"

using std::array;
using std::deque;
using std::make_unique;
using std::unordered_map;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

template <typename T>
bool IsValid(int64_t root, PackedEdgesView edges, WeightsView weights,
             const T* parents, const float* distances, int64_t vertex_count) {
  // Check that the parent of root is root.
  CHECK_EQ(parents[root], root);

  // Index weights with edges.
  unordered_map<T, unordered_map<T, float>> indexed_weights;
  for (int64_t eid = 0; eid < edges.size(); ++eid) {
    int64_t v0 = edges[eid].v0();
    int64_t v1 = edges[eid].v1();
    if (v0 > v1) std::swap(v0, v1);
    indexed_weights[v0][v1] = weights[eid];
  }

  auto parents_copy = make_unique<vector<T>>(parents, parents + vertex_count);
  for (int64_t dst = 0; dst < vertex_count; ++dst) {
    auto& ancestor = (*parents_copy)[dst];
    if (ancestor == kNullVertex) continue;

    // Check that the SSSP tree does not contain cycles by traversing all the
    // way to the `root`. If there is no cycle, we should be able to reach
    // `root` in less than `vertex_count` hops.
    for (int64_t hop = 0; ancestor != root; ++hop) {
      CHECK_NE(ancestor = (*parents_copy)[ancestor], kNullVertex);
      CHECK_LE(hop, vertex_count);
    }

    // Check that each tree edge connects vertices whose SSSP distances differ
    // by at most the weight of the edge.
    if (dst == root) continue;
    const int64_t src = parents[dst];
    int64_t v0 = src, v1 = dst;
    if (v0 > v1) std::swap(v0, v1);
    CHECK_GT(indexed_weights.count(v0), 0) << "v0: " << v0;
    CHECK_GT(indexed_weights[v0].count(v1), 0) << "v0: " << v0 << " v1: " << v1;
    CHECK_LE(distances[dst], distances[src] + indexed_weights[v0][v1]);
  }
  parents_copy.reset();

  for (int64_t eid = 0; eid < edges.size(); ++eid) {
    const PackedEdge& e = edges[eid];
    const int64_t v0 = e.v0();
    const int64_t v1 = e.v1();

    // Check that the two vertices are both or neither in the SSSP tree.
    if (parents[v0] == kNullVertex) {
      CHECK_EQ(parents[v1], kNullVertex);
      continue;
    }
    CHECK_NE(parents[v1], kNullVertex);

    // Check that every edge in the input list has vertices with distances that
    // differ by at most the weight of the edge or are not in the SSSP tree.
    if (distances[v0] < distances[v1]) {
      CHECK_LE(distances[v1], distances[v0] + weights[eid]);
    } else {
      CHECK_LE(distances[v0], distances[v1] + weights[eid]);
    }
  }
  return true;
}

void SSSP(Iid interval_count, Vid interval_size, tapa::mmap<uint64_t> metadata,
          tapa::async_mmap<VidVec> parents,
          tapa::async_mmap<FloatVec> distances,
          tapa::async_mmaps<EdgeVec, kPeCount> edges,
          tapa::async_mmaps<UpdateVec, kPeCount> updates);

int main(int argc, const char* argv[]) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "usage: " << argv[0] << " <edges file>";
    return 1;
  }

  // Memory-map inputs.
  MappedFile edges_mmap(argv[1]);
  MappedFile weights_mmap(edges_mmap.name() + ".weights");

  PackedEdgesView edges_view = edges_mmap.get_view<PackedEdge>();
  WeightsView weights_view = weights_mmap.get_view<float>();
  CHECK_EQ(edges_view.size(), weights_view.size()) << "inconsistent dataset";
  const int64_t edge_count = edges_view.size();

  // Determine vertex intervals.
  const int64_t vertex_count = edge_count / 16;
  CHECK_GE(vertex_count, kVertexVecLen);
  const int64_t interval_count =
      std::max(std::min({vertex_count / kVertexVecLen, int64_t{kPeCount},
                         int64_t{kMaxIntervalCount}}),
               vertex_count / kMaxIntervalSize);
  LOG(INFO) << "interval_count: " << interval_count;
  CHECK_EQ(vertex_count % interval_count, 0);
  const int64_t interval_size = vertex_count / interval_count;
  LOG(INFO) << "interval_size: " << interval_size;
  CHECK_GE(interval_size, kVertexVecLen);

  // Validate inputs and collect degree.
  vector<int64_t> degree(vertex_count);               // For TEPS calculation.
  vector<int64_t> degree_no_self_loop(vertex_count);  // For root sampling.
  vector<int64_t> interval_edge_counts(interval_count);
  array<Eid, kPeCount> pe_edge_counts = {};
  array<vector<Eid>, kPeCount> out_edge_counts;
  for (auto& edge_count : out_edge_counts) {
    edge_count.resize(interval_count);
  }

  // Stores edges in [src_iid][dst_iid][src_bank][dst_bank] bins.
  auto edge_bins = make_unique<unordered_map<
      Iid, unordered_map<
               Iid, array<array<deque<Edge>, kEdgeVecLen>, kEdgeVecLen>>>>();

  for (Eid eid = 0; eid < edge_count; ++eid) {
    auto& edge = edges_view[eid];
    const int64_t v0 = edge.v0();
    const int64_t v1 = edge.v1();

    for (int64_t v : {v0, v1}) {
      CHECK_GE(v, 0) << "invalid edge: " << edge;
      CHECK_LT(v, vertex_count) << "invalid edge: " << edge;
    }

    ++degree[v0];
    ++degree[v1];
    if (v0 != v1) {
      // Interval indices.
      const int p0 = v0 / interval_size;
      const int p1 = v1 / interval_size;

      // Bank indices.
      const int b0 = v0 % kEdgeVecLen;
      const int b1 = v1 % kEdgeVecLen;

      ++degree_no_self_loop[v0];
      ++degree_no_self_loop[v1];
      ++interval_edge_counts[p0];
      ++interval_edge_counts[p1];
      ++pe_edge_counts[p0 % kPeCount];
      ++pe_edge_counts[p1 % kPeCount];
      (*edge_bins)[p0][p1][b0][b1].push_back(
          Edge{static_cast<Vid>(v0), static_cast<Vid>(v1), weights_view[eid]});
      (*edge_bins)[p1][p0][b1][b0].push_back(
          Edge{static_cast<Vid>(v1), static_cast<Vid>(v0), weights_view[eid]});
    }
  }

  // Parition the edges.
  Eid total_edge_count = 0;
  Eid sum_of_max_edge_count = 0;
  vector<Eid> interval_in_edge_counts(interval_count);
  vector<Eid> interval_out_edge_counts(interval_count);
  array<aligned_vector<Edge>, kPeCount> edges;
  for (Iid src_iid = 0; src_iid < interval_count; ++src_iid) {
    vector<Eid> shard_edge_counts(interval_count);

    for (Iid dst_iid = 0; dst_iid < interval_count; ++dst_iid) {
      // There are kEdgeVecLen x kEdgeVecLen edge bins in each shard.
      // We select edges along diagonal directions until the bins are empty.
      for (bool done = false; !done;) {
        done = true;
        for (int i = 0; i < kEdgeVecLen; ++i) {
          bool active = false;
          Edge edge_v[kEdgeVecLen] = {};
          // An empty edge is indicated by src == kNullVertex.
          for (int j = 0; j < kEdgeVecLen; ++j) edge_v[j].src = kNullVertex;
          // First edge in a vector must have valid dst for routing purpose.
          edge_v[0].dst = dst_iid * interval_size;

          for (int j = 0; j < kEdgeVecLen; ++j) {
            deque<Edge>& bin = (*edge_bins)[src_iid][dst_iid][j % kEdgeVecLen]
                                           [(i + j) % kEdgeVecLen];
            if (!bin.empty()) {
              // Make sure the last kVertexUpdateDepDist edge vectors do
              // not write to the same dst.
              auto conflict = [&](const Edge& edge) -> bool {
                const auto& per_pe_edges = edges[dst_iid % kPeCount];
                return std::any_of(
                    per_pe_edges.rbegin(),
                    per_pe_edges.rbegin() +
                        std::min(kVertexUpdateDepDist * kEdgeVecLen,
                                 static_cast<int>(per_pe_edges.size())),
                    [&edge](auto& elem) -> bool {
                      return elem.dst == edge.dst && elem.src != kNullVertex;
                    });
              };
              if (!conflict(bin.back())) {
                edge_v[j] = bin.back();
                bin.pop_back();
              }
              active = true;
            }
          }
          if (!active) continue;  // All bins in this shard are empty.

          // Append newly assembled edge vector to edges.
          for (int j = 0; j < kEdgeVecLen; ++j) {
            edges[dst_iid % kPeCount].push_back(edge_v[j]);
          }
          shard_edge_counts[dst_iid] += kEdgeVecLen;
          interval_in_edge_counts[dst_iid] += kEdgeVecLen;
          interval_out_edge_counts[src_iid] += kEdgeVecLen;
          pe_edge_counts[dst_iid % kPeCount] += kEdgeVecLen;
          out_edge_counts[dst_iid % kPeCount][src_iid] += kEdgeVecLen;
          done = false;
        }
      }
    }

    total_edge_count += interval_out_edge_counts[src_iid];
    VLOG(7) << "edge count in interval[" << src_iid
            << "]: " << interval_out_edge_counts[src_iid];

    // If edges are not balanced among PEs, max #edge bottlenecks all PEs.
    Eid max_edge_count =
        (*std::max_element(out_edge_counts.begin(), out_edge_counts.end(),
                           [src_iid](auto& lhs, auto& rhs) -> bool {
                             return lhs[src_iid] < rhs[src_iid];
                           }))[src_iid];
    sum_of_max_edge_count += max_edge_count * kPeCount;
    VLOG(7) << "edge count in interval[" << src_iid
            << "]: " << max_edge_count * kPeCount << " (+" << std::fixed
            << std::setprecision(2)
            << (100. *
                (max_edge_count * kPeCount -
                 interval_out_edge_counts[src_iid]) /
                interval_out_edge_counts[src_iid])
            << "%)";

    for (Iid dst_iid = 0; dst_iid < interval_count; ++dst_iid) {
      VLOG(7) << "edge count in shard[" << src_iid << "][" << dst_iid
              << "]: " << shard_edge_counts[dst_iid];
    }
  }
  edge_bins.reset();
  for (auto& per_pe_edges : edges) {
    // Allocate at least 1 element to the kernel.
    if (per_pe_edges.size() == 0) {
      per_pe_edges.resize(kEdgeVecLen, {kNullVertex});
    }
    CHECK_EQ(per_pe_edges.size() % kEdgeVecLen, 0);
    CHECK_GT(per_pe_edges.size(), 0);
  }

  // Sample root vertices.
  vector<int64_t> population_vertices;
  population_vertices.reserve(vertex_count);
  vector<int64_t> sample_vertices;
  sample_vertices.reserve(64);
  for (int64_t i = 0; i < vertex_count; ++i) {
    if (degree_no_self_loop[i] > 0) population_vertices.push_back(i);
  }
  std::sample(population_vertices.begin(), population_vertices.end(),
              std::back_inserter(sample_vertices), 64, std::mt19937());

  // Other kernel arguments.
  aligned_vector<Vid> parents(tapa::round_up<kVertexVecLen>(vertex_count));
  aligned_vector<float> distances(tapa::round_up<kVertexVecLen>(vertex_count));
  aligned_vector<uint64_t> metadata(interval_count * (kPeCount + 1) + 1);
  for (int64_t iid = 0; iid < interval_count; ++iid) {
    for (int pe = 0; pe < kPeCount; ++pe) {
      metadata[kPeCount * iid + pe] = out_edge_counts[pe][iid];
    }
    if (iid / kPeCount > 0) {
      metadata[kPeCount * interval_count + iid] =
          metadata[kPeCount * interval_count + iid - kPeCount] +
          interval_in_edge_counts[iid - kPeCount];
    }
  }

  // Statistics.
  vector<double> teps;
  vector<int64_t> iteration_count;

  for (const auto root : sample_vertices) {
    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";
    LOG(INFO) << "root: " << root;

    std::fill(parents.begin(), parents.end(), kNullVertex);
    std::fill(distances.begin(), distances.end(), kInfDistance);
    parents[root] = root;
    distances[root] = 0.f;

    array<aligned_vector<Update>, kPeCount> updates;
    for (int pe = 0; pe < kPeCount; ++pe) {
      updates[pe].resize(edges[pe].size());
    }

    unsetenv("KERNEL_TIME_NS");
    const auto tic = steady_clock::now();
    SSSP(interval_count, interval_size, metadata,
         tapa::make_vec_async_mmap<Vid, kVertexVecLen>(parents),
         tapa::make_vec_async_mmap<float, kVertexVecLen>(distances),
         tapa::make_vec_async_mmaps<Edge, kEdgeVecLen, kPeCount>(edges),
         tapa::make_vec_async_mmaps<Update, kUpdateVecLen, kPeCount>(updates));
    double elapsed_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();
    if (auto env = getenv("KERNEL_TIME_NS")) {
      elapsed_time = 1e-9 * atoll(env);
    }

    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != kNullVertex) {
        connected_edge_count += degree[vid] / 2;
      }
    }
    teps.push_back(connected_edge_count / elapsed_time);
    iteration_count.push_back(*metadata.rbegin());

    if (!IsValid(root, edges_view, weights_view, parents.data(),
                 distances.data(), vertex_count)) {
      return 1;
    }
  }

  LOG(INFO) << "average #iteration: "
            << average<decltype(iteration_count), float>(iteration_count);
  printf("sssp harmonic_mean_TEPS:     !  %g\n", geo_mean(teps));

  return 0;
}

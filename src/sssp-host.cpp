#include <cstdint>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cnpy.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <tapa.h>

#include "sssp.h"
#include "util.h"

using std::array;
using std::deque;
using std::fixed;
using std::make_unique;
using std::setfill;
using std::setprecision;
using std::setw;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_int64(root, kNullVid, "optionally specifiy a single root vertex");
DEFINE_bool(sort, false, "sort edges for each vertex; may override --shuffle");
DEFINE_bool(
    shuffle, false,
    "randomly shuffle edges for each vertex; may be overridden by --sort");
DEFINE_int64(coarsen, 0, "remove vertices whose degree <= this value");
DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_bool(is_log_bucket, true, "use logarithm bucket instead of linear");
DEFINE_double(min_distance, 0, "min distance");
DEFINE_double(max_distance, 0, "max distance");
DEFINE_int32(interval, 1, "increment expected bid every this many cycles");

DEFINE_string(pq_size, "", "priority queue size history");
DEFINE_string(bucket_distribution, "", "bucket size history");
DEFINE_bool(bf_amount_of_work, false, "calculate bellman-ford amount of work");
DEFINE_string(mtx, "", "dump mtx file");

template <typename T>
bool IsValid(int64_t root, PackedEdgesView edges, WeightsView weights,
             const vector<unordered_map<T, float>>& indexed_weights,
             const T* parents, const float* distances, int64_t vertex_count) {
  constexpr auto kEpsilon = 1e-4;
  // Check that the parent of root is root.
  CHECK_EQ(parents[root], root);

  auto parents_copy = make_unique<vector<T>>(parents, parents + vertex_count);
  for (int64_t dst = 0; dst < vertex_count; ++dst) {
    auto& ancestor = (*parents_copy)[dst];
    if (ancestor == kNullVid) continue;

    // Check that the SSSP tree does not contain cycles by traversing all the
    // way to the `root`. If there is no cycle, we should be able to reach
    // `root` in less than `vertex_count` hops.
    for (int64_t hop = 0; ancestor != root; ++hop) {
      CHECK_NE(ancestor = (*parents_copy)[ancestor], kNullVid);
      CHECK_LE(hop, vertex_count);
    }

    // Check that each tree edge connects vertices whose SSSP distances differ
    // by at most the weight of the edge.
    if (dst == root) continue;
    const int64_t src = parents[dst];
    int64_t v0 = src, v1 = dst;
    if (v0 > v1) std::swap(v0, v1);
    CHECK_LT(v0, indexed_weights.size());
    const auto it = indexed_weights[v0].find(v1);
    CHECK(it != indexed_weights[v0].end()) << "v0: " << v0 << " v1: " << v1;
    CHECK_LE(distances[dst], distances[src] + it->second + kEpsilon);
  }
  parents_copy.reset();

  for (int64_t eid = 0; eid < edges.size(); ++eid) {
    const PackedEdge& e = edges[eid];
    const int64_t v0 = e.v0();
    const int64_t v1 = e.v1();

    // Check that the two vertices are both or neither in the SSSP tree.
    if (parents[v0] == kNullVid) {
      CHECK_EQ(parents[v1], kNullVid);
      continue;
    }
    CHECK_NE(parents[v1], kNullVid);

    // Check that every edge in the input list has vertices with distances that
    // differ by at most the weight of the edge or are not in the SSSP tree.
    CHECK_LE(std::abs(distances[v0] - distances[v1]), weights[eid] + kEpsilon)
        << ": v0 = " << v0 << ", v1 = " << v1;
  }
  return true;
}

vector<int64_t> SampleVertices(const vector<int64_t>& degree_no_self_loop) {
  // Sample root vertices.
  const int64_t vertex_count = degree_no_self_loop.size();
  if (FLAGS_root != kNullVid) {
    if (FLAGS_root < vertex_count && degree_no_self_loop[FLAGS_root] > 0) {
      LOG(INFO) << "respecting flag: --root=" << FLAGS_root;
      return {FLAGS_root};
    } else {
      LOG(WARNING) << "ignoring invalid flag: --root=" << FLAGS_root;
    }
  }
  vector<int64_t> population_vertices;
  population_vertices.reserve(vertex_count);
  vector<int64_t> sample_vertices;
  // Sample 4x vertices to allow vertices with too small connected component.
  sample_vertices.reserve(256);
  for (int64_t i = 0; i < vertex_count; ++i) {
    if (degree_no_self_loop[i] > 1) population_vertices.push_back(i);
  }
  std::sample(population_vertices.begin(), population_vertices.end(),
              std::back_inserter(sample_vertices), sample_vertices.capacity(),
              std::mt19937());
  return sample_vertices;
}

// Coarsen graph specified in `indexed_weights` and `degrees` and populate
// `edges` accordingly. `degrees` will be updated to reflect the
// degrees of the coarsened graph for sampling vertices.
deque<std::pair<Vid, unordered_map<Vid, float>>> Coarsen(
    vector<unordered_map<Vid, float>> indexed_weights, const int64_t edge_count,
    vector<int64_t>& degrees, array<aligned_vector<Edge>, kShardCount>& edges,
    vector<Index>& indices) {
  const int64_t vertex_count = degrees.size();
  CHECK_EQ(vertex_count, indexed_weights.size());
  unordered_map<int64_t, unordered_set<Vid>> degree_map;
  vector<unordered_map<Vid, float>> rindexed_weights(vertex_count);
  for (int64_t v0 = 0; v0 < vertex_count; ++v0) {
    degree_map[degrees[v0]].insert(v0);
    for (auto [v1, weight] : indexed_weights[v0]) {
      rindexed_weights[v1][v0] = weight;
    }
  }

  int64_t edge_count_delta = 0;

  auto get_edges = [&](int64_t v0) {
    vector<Edge> edges;
    edges.reserve(indexed_weights[v0].size() + rindexed_weights[v0].size());
    for (auto [v1, weight] : indexed_weights[v0]) {
      edges.push_back({.dst = v1, .weight = weight});
    }
    for (auto [v1, weight] : rindexed_weights[v0]) {
      edges.push_back({.dst = v1, .weight = weight});
    }
    return edges;
  };

  auto delete_edge = [&](int64_t v0, int64_t v1) {
    CHECK_NE(v0, v1);
    if (v0 > v1) std::swap(v0, v1);  // Use smaller vid as v0.
    indexed_weights[v0].erase(v1);
    rindexed_weights[v1].erase(v0);
    degree_map[degrees[v0]].erase(v0);
    CHECK_GE(--degrees[v0], 0);
    degree_map[degrees[v0]].insert(v0);
    degree_map[degrees[v1]].erase(v1);
    CHECK_GE(--degrees[v1], 0);
    degree_map[degrees[v1]].insert(v1);
    --edge_count_delta;
  };

  auto add_edge = [&](int64_t v0, int64_t v1, float weight) {
    CHECK_NE(v0, v1);
    if (v0 > v1) std::swap(v0, v1);  // Use smaller vid as v0.
    if (auto it = indexed_weights[v0].find(v1);
        it != indexed_weights[v0].end()) {
      // Edge already exists.
      CHECK(rindexed_weights[v1].find(v0) != rindexed_weights[v1].end());
      if (it->second < weight) return;
      indexed_weights[v0][v1] = weight;
      rindexed_weights[v1][v0] = weight;
      return;
    }
    CHECK(rindexed_weights[v1].find(v0) == rindexed_weights[v1].end());
    indexed_weights[v0][v1] = weight;
    rindexed_weights[v1][v0] = weight;
    degree_map[degrees[v0]].erase(v0);
    ++degrees[v0];
    degree_map[degrees[v0]].insert(v0);
    degree_map[degrees[v1]].erase(v1);
    ++degrees[v1];
    degree_map[degrees[v1]].insert(v1);
    ++edge_count_delta;
  };

  auto done = [&degree_map] {
    for (int64_t i = 1; i <= FLAGS_coarsen; ++i) {
      if (!degree_map[i].empty()) return false;
    }
    return true;
  };

  deque<std::pair<Vid, unordered_map<Vid, float>>> coarsen_records;
  while (!done()) {
    for (int64_t i = 1; i <= FLAGS_coarsen; ++i) {
      auto vertices = degree_map[i];
      for (auto v0 : vertices) {
        if (auto edges = get_edges(v0); edges.size() == i) {
          for (auto [v1, weight] : edges) delete_edge(v0, v1);
          for (int64_t a = 0; a < i; ++a) {
            for (int64_t b = a + 1; b < i; ++b) {
              add_edge(edges[a].dst, edges[b].dst,
                       edges[a].weight + edges[b].weight);
            }
          }
          // Prepend to make sure new records are replayed eariler when
          // traversed.
          unordered_map<Vid, float> neighbor_map;
          for (auto& [dst, weight] : edges) neighbor_map[dst] = weight;
          coarsen_records.emplace_front(v0, std::move(neighbor_map));
        }
      }
    }
  }

  LOG_IF(INFO, coarsen_records.size() != 0 || edge_count_delta != 0)
      << "coarsening removed " << coarsen_records.size() << " ("
      << std::setprecision(2) << 100. * coarsen_records.size() / vertex_count
      << "% of " << vertex_count << ") degree<=" << FLAGS_coarsen
      << " vertices, " << (edge_count_delta > 0 ? "added" : "removed") << " "
      << std::abs(edge_count_delta) << " (" << std::setprecision(2)
      << 100. * std::abs(edge_count_delta) / edge_count << "% of " << edge_count
      << ") edges";

  array<Eid, kShardCount> offset = {};
  indices.resize(vertex_count);
  for (Vid vid = 0; vid < vertex_count; ++vid) {
    const Vid count = degrees[vid];
    const auto sid = vid % kShardCount;
    indices[vid] = {.offset = offset[sid], .count = count};
    offset[sid] += tapa::round_up<kEdgeVecLen>(count);
  }

  for (int i = 0; i < kShardCount; ++i) edges[i].resize(offset[i]);
  vector<Vid> vertex_counts(vertex_count);
  for (Vid v0 = 0; v0 < vertex_count; ++v0) {
    for (auto [k, weight] : indexed_weights[v0]) {
      Vid v1 = k;
      for (auto [src, dst] : {std::tie(v0, v1), std::tie(v1, v0)}) {
        const auto src_sid = src % kShardCount;
        const auto index = indices[src];
        edges[src_sid][index.offset + vertex_counts[src]] = {
            .dst = Vid(dst),
            .weight = weight,
        };
        ++vertex_counts[src];
      }
    }
  }

  for (Vid vid = 0; vid < vertex_count; ++vid) {
    const auto& index = indices[vid];
    CHECK_EQ(vertex_counts[vid], index.count);
    CHECK_EQ(index.offset % kEdgeVecLen, 0);

    // Fill in null edges.
    for (int64_t i = index.count; i < tapa::round_up<kEdgeVecLen>(index.count);
         ++i) {
      edges[vid % kShardCount][index.offset + i] = {
          .dst = kNullVid,
          .weight = kInfDistance,
      };
    }
  }

  return coarsen_records;
}

void Refine(
    const deque<std::pair<Vid, unordered_map<Vid, float>>>& coarsen_records,
    array<aligned_vector<Vertex>, kIntervalCount>& vertices) {
  auto access_vertex = [&vertices](Vid vid) -> Vertex& {
    return vertices[vid % kIntervalCount][vid / kIntervalCount];
  };
  for (auto& [v0, edges] : coarsen_records) {
    // All neighbors must be in or not in the SSSP tree at the same time.
    // So if the first one is not, skip this coarsened vertex.
    if (access_vertex(edges.begin()->first).parent == kNullVid) continue;

    auto& attr0 = access_vertex(v0);
    for (auto& [v1, weight] : edges) {
      auto& attr1 = access_vertex(v1);
      if (edges.count(attr1.parent) &&
          almost_equal(attr1.distance,
                       edges.at(attr1.parent) +
                           access_vertex(attr1.parent).distance + weight)) {
        // Parent of v1 is v0.
        attr1.parent = v0;
      } else {
        // Parent of v1 is external; parent of v0 may be v1.
        const auto new_distance = attr1.distance + weight;
        if (new_distance < attr0.distance) {
          attr0 = {.parent = v1, .distance = new_distance};
        }
      }
    }
  }
}

void SSSP(Vid vertex_count, Task root, tapa::mmap<int64_t> metadata,
          tapa::mmaps<EdgeVec, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices, bool is_log_bucket,
          float min_distance, float max_distance, int32_t interval,
          tapa::mmaps<SpilledTaskPerMem, kCgpqPhysMemCount> cgpq_spill);

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "usage: " << argv[0] << " <edges file> [interval_count]";
    return 1;
  }

  // Memory-map inputs.
  MappedFile edges_mmap(argv[1]);
  MappedFile weights_mmap(edges_mmap.name() + ".weights");

  PackedEdgesView edges_view = edges_mmap.get_view<PackedEdge>();
  WeightsView weights_view = weights_mmap.get_view<float>();
  CHECK_EQ(edges_view.size(), weights_view.size()) << "inconsistent dataset";
  const int64_t edge_count = edges_view.size();

  // Validate inputs and collect degree.
  vector<int64_t> degree;               // For TEPS calculation.
  vector<int64_t> degree_no_self_loop;  // For root sampling.

  Eid edge_count_no_self_loop = 0;

  // Dedup edges.
  vector<unordered_map<Vid, float>> indexed_weights;
  int64_t max_degree;
  for (Eid eid = 0; eid < edge_count; ++eid) {
    const auto& edge = edges_view[eid];
    auto v0 = edge.v0();
    auto v1 = edge.v1();
    if (const auto required_size = std::max(v0, v1) + 1;
        required_size > degree.size()) {
      degree.resize(required_size);
      degree_no_self_loop.resize(required_size);
      indexed_weights.resize(required_size);
    }
    ++degree[v0];
    ++degree[v1];
    max_degree = std::max(max_degree, degree[v0]);
    max_degree = std::max(max_degree, degree[v1]);
    if (v0 != v1) {
      if (v0 > v1) std::swap(v0, v1);  // Use smaller vid as v0.
      if (auto it = indexed_weights[v0].find(v1);
          it != indexed_weights[v0].end()) {
        // Do not update the weight if the new weight is larger.
        if (weights_view[eid] > it->second) continue;
      } else {
        // Add a new edge if none exists.
        ++degree_no_self_loop[v0];
        ++degree_no_self_loop[v1];
        ++edge_count_no_self_loop;
      }
      indexed_weights[v0][v1] = weights_view[eid];
    }
  }

  // Determine vertex intervals.
  const int64_t vertex_count = degree.size();
  CHECK_EQ(degree_no_self_loop.size(), vertex_count);
  CHECK_EQ(indexed_weights.size(), vertex_count);
  CHECK_LT(vertex_count, (1 << kVidWidth));

  VLOG(3) << "total vertex count: " << vertex_count;
  VLOG(3) << "total edge count: " << edge_count;
  VLOG(3) << "avg degree: " << fixed << setprecision(1)
          << 1. * edge_count / vertex_count;
  VLOG(3) << "max degree: " << max_degree;

  if (!FLAGS_mtx.empty()) {
    std::ofstream ofs(FLAGS_mtx);
    ofs << vertex_count << " " << vertex_count << " " << edge_count * 2
        << std::endl;
    for (Eid eid = 0; eid < edge_count; ++eid) {
      const auto& edge = edges_view[eid];
      const auto weight = weights_view[eid];
      ofs << edge.v0() + 1 << " " << edge.v1() + 1 << " " << weight << std::endl
          << edge.v1() + 1 << " " << edge.v0() + 1 << " " << weight
          << std::endl;
      LOG_EVERY_N(INFO, 1000000)
          << eid << " / " << edge_count << " edges processed (" << fixed
          << setprecision(1) << 100. * eid / edge_count << "%)";
    }
    return 0;
  }

  if (VLOG_IS_ON(3)) {
    vector<int64_t> bins(7);
    constexpr int64_t kBase = 10;
    for (auto d : degree_no_self_loop) {
      int64_t bound = 1;
      for (auto& bin : bins) {
        if (d < bound) {
          ++bin;
          break;
        }
        bound *= kBase;
      }
    }
    int64_t bound = 1;
    int64_t last_bin = bins.back();
    bins.pop_back();
    for (auto bin : bins) {
      VLOG(3) << "  degree in [" << bound / kBase << ", " << bound
              << "): " << bin;
      bound *= kBase;
    }
    VLOG(3) << "  degree in [" << bound / kBase << ", +∞): " << last_bin;
  }

  // Allocate and fill edges for the kernel.
  array<aligned_vector<Edge>, kShardCount> edges;
  vector<Index> indices;
  const auto coarsen_records =
      Coarsen(indexed_weights, edge_count, degree_no_self_loop, edges, indices);

  if (FLAGS_sort || FLAGS_shuffle) {
    for (auto& shard : edges) {
      for (Vid vid = 0; vid < tapa::round_up_div<kShardCount>(vertex_count);
           ++vid) {
        const auto index = indices[vid];
        const auto first = shard.begin() + index.offset;
        const auto last = first + index.count;
        if (FLAGS_sort) {
          std::sort(first, last,
                    [](auto& a, auto& b) { return a.dst < b.dst; });
        } else {
          std::shuffle(first, last, std::mt19937());
        }
      }
    }
  }

  // Other kernel arguments.
  aligned_vector<int64_t> metadata(
      kGlobalStatCount + kSubIntervalCount * kVertexUniStatCount +
      kShardCount * kEdgeUnitStatCount + kQueueStatCount);
  array<aligned_vector<Vertex>, kIntervalCount> vertices;
  for (auto& interval : vertices) {
    interval.resize(tapa::round_up_div<kIntervalCount>(vertex_count));
  }
  array<aligned_vector<SpilledTaskPerMem>, kCgpqPhysMemCount> cgpq_spill;
  for (auto& spill : cgpq_spill) {
    spill.resize(1 << uint_spill_addr_t::width);
  }

  // Statistics.
  vector<double> teps;

  int valid_root_count = 0;
  for (const auto root : SampleVertices(degree_no_self_loop)) {
    ++valid_root_count;
    if (valid_root_count > 64) {
      break;
    }

    if (!FLAGS_pq_size.empty()) {
      std::vector<float> distances(vertex_count, kInfDistance);
      auto cmp = [&distances](auto a, auto b) {
        return distances[a] > distances[b];
      };
      std::priority_queue<int64_t, std::vector<int64_t>, decltype(cmp)> pq(cmp);

      distances[root] = 0.f;
      pq.push(root);

      std::vector<int64_t> pq_size_history;

      while (!pq.empty()) {
        const auto src = pq.top();
        pq.pop();

        const auto src_dist = distances[src];
        const auto src_index = indices[src];
        for (int64_t i = 0; i < src_index.count; ++i) {
          pq_size_history.push_back(pq.size());

          const auto edge = edges[src % kShardCount][src_index.offset + i];
          const auto new_dist = src_dist + edge.weight;
          if (new_dist < distances[edge.dst]) {
            distances[edge.dst] = new_dist;
            pq.push(edge.dst);
          }
        }
      }

      cnpy::npy_save(FLAGS_pq_size, pq_size_history);
      LOG(INFO) << "history of " << pq_size_history.size()
                << " iterations saved";
      return 0;
    }

    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";
    LOG(INFO) << "root: " << root;

    for (auto& interval : vertices) {
      std::fill(interval.begin(), interval.end(),
                Vertex{.parent = kNullVid, .distance = kInfDistance});
    }

    vertices[root % kIntervalCount][root / kIntervalCount] = {
        .parent = Vid(root), .distance = 0.f};
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      const auto index = indices[vid];
      CHECK_EQ(index.offset % kEdgeVecLen, 0) << "edges should be aligned";
      auto& vertex = vertices[vid % kIntervalCount][vid / kIntervalCount];
      vertex.offset = index.offset / kEdgeVecLen;
      vertex.degree = index.count;  // Needed to keep track of #task.
    }

    float root_min_distance = std::numeric_limits<float>::max();
    float root_max_distance = std::numeric_limits<float>::min();
    const auto& root_index = indices[root];
    for (int64_t i = 0; i < root_index.count; ++i) {
      const auto edge = edges[root % kShardCount][root_index.offset + i];
      root_min_distance = std::min(float(root_min_distance), edge.weight);
      root_max_distance = std::max(float(root_max_distance), edge.weight);
    }
    LOG(INFO) << "min distance of root's neighbors: " << root_min_distance;
    LOG(INFO) << "max distance of root's neighbors: " << root_max_distance;

    const float arg_min_distance =
        FLAGS_min_distance > 0 ? FLAGS_min_distance : root_min_distance;
    const float arg_max_distance =
        FLAGS_max_distance > 0 ? FLAGS_max_distance : root_min_distance + 0.5f;
    LOG(INFO) << "using min distance " << arg_min_distance;
    LOG(INFO) << "using max distance " << arg_max_distance;

    if (!FLAGS_bucket_distribution.empty()) {
      std::vector<float> distances(vertex_count, kInfDistance);
      auto cmp = [&distances](auto a, auto b) {
        return distances[a] > distances[b];
      };

      constexpr int kBucketCount = 32;

      std::array<std::queue<int64_t>, kBucketCount> buckets;

      distances[root] = 0.f;
      buckets[0].push(root);

      std::vector<std::array<int64_t, kBucketCount>>
          bucket_distribution_history;

      const auto norm =
          buckets.size() /
          (FLAGS_is_log_bucket
               ? (std::log(arg_max_distance) - std::log(arg_min_distance))
               : (arg_max_distance - arg_min_distance));

      for (;;) {
        int64_t src;
        bool is_empty = true;
        for (int bid = 0; bid < buckets.size(); ++bid) {
          if (!buckets[bid].empty()) {
            src = buckets[bid].front();
            buckets[bid].pop();
            is_empty = false;
            break;
          }
        }
        if (is_empty) {
          break;
        }

        const auto src_dist = distances[src];
        const auto src_index = indices[src];
        for (int64_t i = 0; i < src_index.count; ++i) {
          {
            std::array<int64_t, kBucketCount> bucket_distribution;
            for (int bid = 0; bid < buckets.size(); ++bid) {
              bucket_distribution[bid] = buckets[bid].size();
            }
            bucket_distribution_history.push_back(bucket_distribution);
          }

          const auto edge = edges[src % kShardCount][src_index.offset + i];
          const auto new_dist = src_dist + edge.weight;
          if (new_dist < distances[edge.dst]) {
            distances[edge.dst] = new_dist;

            const auto bid = std::max(
                std::min(
                    int((FLAGS_is_log_bucket
                             ? (std::log(new_dist) - std::log(arg_min_distance))
                             : (new_dist - arg_min_distance)) *
                        norm),
                    int(buckets.size() - 1)),
                0);
            buckets[bid].push(edge.dst);
          }
        }
      }

      cnpy::npy_save(FLAGS_bucket_distribution,
                     bucket_distribution_history[0].data(),
                     {bucket_distribution_history.size(), kBucketCount});
      LOG(INFO) << "history of " << bucket_distribution_history.size()
                << " iterations saved";
      return 0;
    }

    const double elapsed_time =
        1e-9 *
        tapa::invoke_in_new_process(
            SSSP, FLAGS_bitstream, Vid(vertex_count),
            Task{
                .vid = Vid(root),
                .vertex =
                    vertices[root % kIntervalCount][root / kIntervalCount],
            },
            tapa::write_only_mmap<int64_t>(metadata),
            tapa::read_only_mmaps<Edge, kShardCount>(edges)
                .vectorized<kEdgeVecLen>(),
            tapa::read_write_mmaps<Vertex, kIntervalCount>(vertices),
            FLAGS_is_log_bucket, arg_min_distance, arg_max_distance,
            FLAGS_interval,
            tapa::placeholder_mmaps<SpilledTaskPerMem, kCgpqPhysMemCount>(
                cgpq_spill));

    const auto tic = steady_clock::now();
    Refine(coarsen_records, vertices);
    const double refine_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();

    vector<Vid> parents(vertex_count);
    vector<float> distances(vertex_count);
    float max_distance = std::numeric_limits<float>::min();
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      const auto& vertex = vertices[vid % kIntervalCount][vid / kIntervalCount];
      parents[vid] = vertex.parent;
      distances[vid] = vertex.distance;
      if (vertex.parent != kNullVid) {
        max_distance = std::max(float(max_distance), vertex.distance);
      }
    }
    LOG(INFO) << "overall max distance (from root): " << max_distance;

    int64_t connected_vertex_count = 0;
    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != kNullVid) {
        ++connected_vertex_count;
        connected_edge_count += degree[vid] / 2;
      }
    }
    if (connected_vertex_count * 100 < vertex_count) {
      LOG(INFO) << "too few vertices in the connected component, skipping";
      continue;
    }

    if (FLAGS_bf_amount_of_work) {
      std::vector<float> distances_curr(vertex_count, kInfDistance);
      std::vector<float> distances_next(vertex_count, kInfDistance);
      std::vector<bool> is_active_ping(vertex_count, false);
      std::vector<bool> is_active_pong(vertex_count, false);

      auto is_active_curr = &is_active_ping;
      auto is_active_next = &is_active_pong;

      distances_curr[root] = 0.f;
      is_active_curr->at(root) = true;

      int64_t edges_visited = 0;

      for (;;) {
        bool has_update = false;
        std::fill(is_active_next->begin(), is_active_next->end(), false);

        for (int64_t src = 0; src < vertex_count; ++src) {
          if (is_active_curr->at(src)) {
            const auto src_dist = distances_curr[src];
            const auto src_index = indices[src];
            for (int64_t i = 0; i < src_index.count; ++i) {
              ++edges_visited;

              const auto edge = edges[src % kShardCount][src_index.offset + i];
              const auto new_dist = src_dist + edge.weight;
              if (new_dist < distances_next[edge.dst]) {
                has_update = true;
                is_active_next->at(edge.dst) = true;
                distances_next[edge.dst] = new_dist;
              }
            }
          }
        }

        std::swap(is_active_curr, is_active_next);
        distances_curr = distances_next;

        if (!has_update) {
          break;
        }
      }

      LOG(INFO) << "#edges visited: " << edges_visited << " (" << fixed
                << setprecision(1) << 1. * edges_visited / connected_edge_count
                << "x of " << connected_edge_count << ")";
    }

    VLOG(3) << "kernel time: " << elapsed_time << " s";
    teps.push_back(connected_edge_count / (elapsed_time + refine_time));
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
    for (auto& shard : edges) coarsened_edge_count += shard.size();
    VLOG(3) << "  TEPS:                 " << setfill(' ') << setw(10)
            << *teps.rbegin() << " (" << 1e9 * elapsed_time / visited_edge_count
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
                << spill_count << " / "
                << cgpq_spill[0].size() / kCgpqBankCountPerMem /
                       (kCgpqChunkSize / kSpilledTaskVecLen);
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
    }

    if (!IsValid(root, edges_view, weights_view, indexed_weights,
                 parents.data(), distances.data(), vertex_count)) {
      return 1;
    }
  }

  printf("sssp harmonic_mean_TEPS:     !  %g\n", harmonic_mean(teps));

  return 0;
}

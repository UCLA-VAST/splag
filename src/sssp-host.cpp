#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gflags/gflags.h>
#include <tapa.h>

#include "sssp.h"
#include "util.h"

using std::deque;
using std::make_unique;
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

template <typename T>
bool IsValid(int64_t root, PackedEdgesView edges, WeightsView weights,
             const vector<unordered_map<T, float>>& indexed_weights,
             const T* parents, const float* distances, int64_t vertex_count) {
  constexpr auto kEpsilon = 1e-5;
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
  sample_vertices.reserve(64);
  for (int64_t i = 0; i < vertex_count; ++i) {
    if (degree_no_self_loop[i] > 0) population_vertices.push_back(i);
  }
  std::sample(population_vertices.begin(), population_vertices.end(),
              std::back_inserter(sample_vertices), 64, std::mt19937());
  return sample_vertices;
}

// Coarsen graph specified in `indexed_weights` and `degrees` and populate
// `edges` accordingly. `degrees` will be updated to reflect the
// degrees of the coarsened graph for sampling vertices.
deque<std::pair<Vid, unordered_map<Vid, float>>> Coarsen(
    vector<unordered_map<Vid, float>> indexed_weights, vector<int64_t>& degrees,
    aligned_vector<Edge>& edges) {
  const int64_t vertex_count = degrees.size();
  CHECK_EQ(vertex_count, indexed_weights.size());
  const int64_t edge_count = vertex_count * 16;
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

  edges.resize(vertex_count);
  Eid offset = vertex_count;
  for (Vid vid = 0; vid < vertex_count; ++vid) {
    const Vid count = degrees[vid];
    edges[vid] = bit_cast<Edge>(Index{.offset = offset, .count = count});
    offset += count;
  }

  edges.resize(offset);
  vector<Vid> vertex_counts(vertex_count);
  for (Vid v0 = 0; v0 < vertex_count; ++v0) {
    for (auto [k, weight] : indexed_weights[v0]) {
      Vid v1 = k;
      for (auto [src, dst] : {std::tie(v0, v1), std::tie(v1, v0)}) {
        edges[bit_cast<Index>(edges[src]).offset + vertex_counts[src]] = {
            .dst = Vid(dst),
            .weight = weight,
        };
        ++vertex_counts[src];
      }
    }
  }

  for (Vid vid = 0; vid < vertex_count; ++vid) {
    CHECK_EQ(vertex_counts[vid], bit_cast<Index>(edges[vid]).count);
  }

  return coarsen_records;
}

void Refine(
    const deque<std::pair<Vid, unordered_map<Vid, float>>>& coarsen_records,
    aligned_vector<Vertex>& vertices) {
  for (auto& [v0, edges] : coarsen_records) {
    // All neighbors must be in or not in the SSSP tree at the same time.
    // So if the first one is not, skip this coarsened vertex.
    if (vertices[edges.begin()->first].parent == kNullVid) continue;

    auto& attr0 = vertices[v0];
    for (auto& [v1, weight] : edges) {
      auto& attr1 = vertices[v1];
      if (edges.count(attr1.parent) &&
          almost_equal(attr1.distance, edges.at(attr1.parent) +
                                           vertices[attr1.parent].distance +
                                           weight)) {
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

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::mmap<Edge> edges, tapa::mmap<Vertex> vertices,
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index);

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

  // Determine vertex intervals.
  const int64_t vertex_count = edge_count / 16;

  // Validate inputs and collect degree.
  vector<int64_t> degree(vertex_count);               // For TEPS calculation.
  vector<int64_t> degree_no_self_loop(vertex_count);  // For root sampling.

  Eid edge_count_no_self_loop = 0;

  // Dedup edges.
  vector<unordered_map<Vid, float>> indexed_weights(vertex_count);
  for (Eid eid = 0; eid < edge_count; ++eid) {
    const auto& edge = edges_view[eid];
    auto v0 = edge.v0();
    auto v1 = edge.v1();
    ++degree[v0];
    ++degree[v1];
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
  aligned_vector<Edge> edges;
  auto coarsen_records = Coarsen(indexed_weights, degree_no_self_loop, edges);

  if (FLAGS_sort || FLAGS_shuffle) {
    for (Vid vid = 0; vid < vertex_count; ++vid) {
      const auto index = bit_cast<Index>(edges[vid]);
      const auto first = edges.begin() + index.offset;
      const auto last = first + index.count;
      if (FLAGS_sort) {
        std::sort(first, last, [](auto& a, auto& b) { return a.dst < b.dst; });
      } else {
        std::random_shuffle(first, last);
      }
    }
  }

  // Other kernel arguments.
  aligned_vector<int64_t> metadata(5);
  aligned_vector<Vertex> vertices(vertex_count);
  aligned_vector<Task> heap_array(vertex_count);
  aligned_vector<Vid> heap_index(vertex_count);

  // Statistics.
  vector<double> teps;

  for (const auto root : SampleVertices(degree_no_self_loop)) {
    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";
    LOG(INFO) << "root: " << root;

    std::fill(vertices.begin(), vertices.end(),
              Vertex{.parent = kNullVid, .distance = kInfDistance});
    std::fill(heap_index.begin(), heap_index.end(), kNullVid);
    vertices[root] = {.parent = Vid(root), .distance = 0.f};

    unsetenv("KERNEL_TIME_NS");
    auto tic = steady_clock::now();
    SSSP(vertex_count, root, metadata, edges, vertices, heap_array, heap_index);
    double elapsed_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();
    if (auto env = getenv("KERNEL_TIME_NS")) {
      elapsed_time = 1e-9 * atoll(env);
      VLOG(3) << "using time reported by the kernel: " << elapsed_time << " s";
    }

    tic = steady_clock::now();
    Refine(coarsen_records, vertices);
    const double refine_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();

    vector<Vid> parents(vertex_count);
    vector<float> distances(vertex_count);
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      parents[vid] = vertices[vid].parent;
      distances[vid] = vertices[vid].distance;
    }

    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != kNullVid) {
        connected_edge_count += degree[vid] / 2;
      }
    }
    teps.push_back(connected_edge_count / (elapsed_time + refine_time));
    auto visited_edge_count = metadata[0];
    auto total_queue_size = metadata[1];
    auto queue_count = metadata[2];
    auto max_queue_size = metadata[3];
    auto visited_vertex_count = metadata[4];
    VLOG(3) << "  TEPS:                  " << *teps.rbegin() << " ("
            << 1e9 * elapsed_time / visited_edge_count << " ns/edge visited + "
            << refine_time << " s refinement)";
    VLOG(3) << "  #edges connected:      " << connected_edge_count;
    VLOG(3) << "  #edges visited:        " << visited_edge_count << " ("
            << std::fixed << std::setprecision(1) << std::showpos
            << 100. * visited_edge_count / edges.size() - 100 << "% over "
            << edges.size() << ") ";
    VLOG(3) << "  #vertices visited:     " << visited_vertex_count << " ("
            << std::fixed << std::setprecision(1) << std::showpos
            << 100. * visited_vertex_count / vertex_count - 100 << "% over "
            << std::noshowpos << vertex_count << ")";
    VLOG(3) << "  average size of queue: " << std::fixed << std::setprecision(1)
            << 1. * total_queue_size / queue_count << " ("
            << 100. * total_queue_size / queue_count / vertex_count << "% of "
            << vertex_count << ")";
    VLOG(3) << "  max size of queue:     " << max_queue_size << " ("
            << std::fixed << std::setprecision(1)
            << 100. * max_queue_size / vertex_count << "% of " << vertex_count
            << ")";
    VLOG(3) << "  queue operations:      " << queue_count;

    if (!IsValid(root, edges_view, weights_view, indexed_weights,
                 parents.data(), distances.data(), vertex_count)) {
      return 1;
    }
  }

  printf("sssp harmonic_mean_TEPS:     !  %g\n", geo_mean(teps));

  return 0;
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <tapa.h>

#include "sssp.h"
#include "util.h"

using std::make_unique;
using std::unordered_map;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

template <typename T>
struct mmap_allocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  T* allocate(size_t count) {
    void* ptr = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    if (ptr == MAP_FAILED) throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* ptr, std::size_t count) {
    if (munmap(ptr, count * sizeof(T)) != 0) throw std::bad_alloc();
  }
};

template <typename T>
using aligned_vector = std::vector<T, mmap_allocator<T>>;

template <typename T>
bool IsValid(int64_t root, PackedEdgesView edges, WeightsView weights,
             const vector<unordered_map<T, float>>& indexed_weights,
             const T* parents, const float* distances, int64_t vertex_count) {
  // Check that the parent of root is root.
  CHECK_EQ(parents[root], root);

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
    CHECK_LT(v0, indexed_weights.size());
    const auto it = indexed_weights[v0].find(v1);
    CHECK(it != indexed_weights[v0].end()) << "v0: " << v0 << " v1: " << v1;
    CHECK_LE(distances[dst], distances[src] + it->second);
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

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::async_mmap<Edge> edges, tapa::mmap<Index> indices,
          tapa::mmap<Vid> parents, tapa::mmap<float> distances,
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index);

int main(int argc, const char* argv[]) {
  FLAGS_logtostderr = true;
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
  CHECK_GE(vertex_count, kVertexVecLen);

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

  // Allocate and fill edges and indices for the kernel.
  aligned_vector<Index> indices(vertex_count);
  aligned_vector<Edge> edges(edge_count_no_self_loop * 2);  // Undirected edge.
  Eid offset = 0;
  for (Vid vid = 0; vid < vertex_count; ++vid) {
    const Vid count = degree_no_self_loop[vid];
    indices[vid] = {.offset = offset, .count = count};
    offset += count;
  }

  {
    vector<Vid> vertex_counts(vertex_count);
    for (Vid v0 = 0; v0 < vertex_count; ++v0) {
      for (auto [k, weight] : indexed_weights[v0]) {
        Vid v1 = k;
        for (auto [src, dst] : {std::tie(v0, v1), std::tie(v1, v0)}) {
          edges[indices[src].offset + vertex_counts[src]] = {
              .dst = Vid(dst),
              .weight = weight,
          };
          ++vertex_counts[src];
        }
      }
    }

    for (Vid vid = 0; vid < vertex_count; ++vid) {
      CHECK_EQ(vertex_counts[vid], indices[vid].count);
    }
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
  aligned_vector<int64_t> metadata(4);
  aligned_vector<Vid> parents(tapa::round_up<kVertexVecLen>(vertex_count));
  aligned_vector<float> distances(tapa::round_up<kVertexVecLen>(vertex_count));
  aligned_vector<Task> heap_array(vertex_count);
  aligned_vector<Vid> heap_index(vertex_count);

  // Statistics.
  vector<double> teps;

  for (const auto root : sample_vertices) {
    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";
    LOG(INFO) << "root: " << root;

    std::fill(parents.begin(), parents.end(), kNullVertex);
    std::fill(distances.begin(), distances.end(), kInfDistance);
    std::fill(heap_index.begin(), heap_index.end(), kNullVertex);
    parents[root] = root;
    distances[root] = 0.f;

    unsetenv("KERNEL_TIME_NS");
    const auto tic = steady_clock::now();
    SSSP(vertex_count, root, metadata, edges, indices, parents, distances,
         heap_array, heap_index);
    double elapsed_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();
    if (auto env = getenv("KERNEL_TIME_NS")) {
      elapsed_time = 1e-9 * atoll(env);
      VLOG(3) << "using time reported by the kernel: " << elapsed_time << " s";
    }

    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != kNullVertex) {
        connected_edge_count += degree[vid] / 2;
      }
    }
    teps.push_back(connected_edge_count / elapsed_time);
    VLOG(3) << "  TEPS:                  " << *teps.rbegin();
    auto visited_edge_count = metadata[0];
    auto total_queue_size = metadata[1];
    auto queue_count = metadata[2];
    auto max_queue_size = metadata[3];
    VLOG(3) << "  #edges connected:      " << connected_edge_count;
    VLOG(3) << "  #edges visited:        " << visited_edge_count << " ("
            << std::fixed << std::setprecision(1) << std::showpos
            << 100. * visited_edge_count / edges.size() - 100 << "% over "
            << edges.size() << ")";
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

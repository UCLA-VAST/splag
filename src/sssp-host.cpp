#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <tapa.h>

#include "util.h"

bool IsValid(int64_t root, PackedEdgesView edges, WeightsView weights,
             const int64_t* parents, const float* distances,
             int64_t vertex_count) {
  // Check that the parent of root is root.
  CHECK_EQ(parents[root], root);

  // Check that the SSSP tree does not contain cycles.
  std::vector<int64_t> parents_copy(vertex_count);
  for (int64_t i = 0; i < vertex_count; ++i) {
    parents_copy[i] = parents[i];
  }
  for (int64_t dst = 0; dst < vertex_count; ++dst) {
    auto& src = parents_copy[dst];
    if (src == -1) continue;

    // Traverse all the way to the root.
    int64_t hop = 0;
    while (src != root) {
      src = parents_copy[src];
      CHECK_NE(src, -1);
      ++hop;
      CHECK_LE(hop, vertex_count);
    }
  }

  // Check that every edge in the input list has vertices with distances that
  // differ by at most the weight of the edge or are not in the SSSP tree.
  for (int64_t eid = 0; eid < edges.size(); ++eid) {
    const PackedEdge& e = edges[eid];
    const int64_t v0 = e.v0();
    const int64_t v1 = e.v1();

    if (parents[v1] == v0) {
      CHECK_LE(distances[v0], distances[v1]);
      CHECK_LE(distances[v1], distances[v0] + weights[eid]);
    } else if (parents[v0] == v1) {
      CHECK_LE(distances[v1], distances[v0]);
      CHECK_LE(distances[v0], distances[v1] + weights[eid]);
    }
  }
  return true;
}

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
  const int64_t vertex_count = edge_count / 16;

  // Validate inputs and collect degree.
  std::vector<int64_t> degree(vertex_count);
  std::vector<int64_t> degree_no_self_loop(vertex_count);
  const int64_t partition_count = std::thread::hardware_concurrency();
  CHECK_EQ(vertex_count % partition_count, 0);
  const int64_t partition_size = vertex_count / partition_count;
  std::vector<int64_t> per_partition_degree(partition_count);
  for (auto& edge : edges_view) {
    const int64_t v0 = edge.v0();
    const int64_t v1 = edge.v1();

    for (int64_t v : {v0, v1}) {
      CHECK_GE(v, 0) << "invalid edge: " << edge;
      CHECK_LT(v, vertex_count) << "invalid edge: " << edge;
    }

    ++degree[v0];
    ++degree[v1];
    if (v0 != v1) {
      ++degree_no_self_loop[v0];
      ++degree_no_self_loop[v1];
      ++per_partition_degree[v0 / partition_size];
      ++per_partition_degree[v1 / partition_size];
    }
  }

  // Partition the edges.
  std::vector<std::vector<WeightedPackedEdge>> edge_partitions(partition_count);
  auto add_edge = [&edge_partitions, partition_size](int64_t src, int64_t dst,
                                                     float weight) {
    WeightedPackedEdge edge;
    edge.set_edge(src, dst);
    edge.weight = weight;
    edge_partitions[dst / partition_size].push_back(edge);
  };
  for (int64_t pid = 0; pid < partition_count; ++pid) {
    edge_partitions[pid].reserve(per_partition_degree[pid]);
    VLOG(5) << "#edges in partition #" << pid << ": "
            << per_partition_degree[pid];
  }
  for (int64_t eid = 0; eid < edge_count; ++eid) {
    const int64_t v0 = edges_view[eid].v0();
    const int64_t v1 = edges_view[eid].v1();
    const float weight = weights_view[eid];
    if (v0 != v1) {
      add_edge(v0, v1, weight);
      add_edge(v1, v0, weight);
    }
  }

  // Sample root vertices.
  std::vector<int64_t> population_vertices;
  population_vertices.reserve(vertex_count);
  std::vector<int64_t> sample_vertices;
  sample_vertices.reserve(64);
  for (int64_t i = 0; i < vertex_count; ++i) {
    if (degree_no_self_loop[i] > 0) population_vertices.push_back(i);
  }
  std::sample(population_vertices.begin(), population_vertices.end(),
              std::back_inserter(sample_vertices), 64, std::mt19937());

  // Allocate memory for outputs and statistics.
  auto parents_ptr = std::make_unique<int64_t[]>(vertex_count);
  auto distances_ptr = std::make_unique<float[]>(vertex_count);
  auto parents = parents_ptr.get();
  auto distances = distances_ptr.get();
  std::vector<double> teps;
  std::vector<int64_t> iteration_count;

  for (const auto root : sample_vertices) {
    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";

    std::fill_n(parents, vertex_count, -1);
    std::fill_n(distances, vertex_count, std::numeric_limits<float>::max());

    parents[root] = root;
    distances[root] = 0.f;

    // Bellman-Ford
    const auto tic = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < vertex_count; ++iteration) {
      std::atomic<bool> is_active = false;
#pragma omp parallel for
      for (int64_t pid = 0; pid < partition_count; ++pid) {
        for (const auto& edge : edge_partitions[pid]) {
          const int64_t src = edge.v0();
          const int64_t dst = edge.v1();
          const float weight = edge.weight;

          if (const float new_weight = distances[src] + weight;
              new_weight < distances[dst]) {
            is_active = true;
            parents[dst] = src;
            distances[dst] = new_weight;
          }
        }
      }
      if (!is_active) {
        iteration_count.push_back(iteration + 1);
        break;
      }
    }
    double elapsed_time =
        1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - tic)
                   .count();
    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != -1) {
        connected_edge_count += degree[vid] / 2;
      }
    }
    teps.push_back(connected_edge_count / elapsed_time);

    if (!IsValid(root, edges_view, weights_view, parents, distances,
                 vertex_count)) {
      return 1;
    }
  }

  LOG(INFO) << "average #iteration: "
            << average<decltype(iteration_count), float>(iteration_count);
  printf("sssp harmonic_mean_TEPS:     !  %g\n", geo_mean(teps));

  return 0;
}

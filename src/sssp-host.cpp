#include <algorithm>
#include <array>
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

using std::array;
using std::deque;
using std::fixed;
using std::make_unique;
using std::setfill;
using std::setprecision;
using std::setw;
using std::tuple;
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
    vector<unordered_map<Vid, float>> indexed_weights, const int64_t edge_count,
    vector<int64_t>& degrees, array<aligned_vector<Edge>, kShardCount>& edges) {
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

  const int64_t vertex_count_per_edge_partition =
      tapa::round_up_div<kShardCount>(vertex_count);
  array<Eid, kShardCount> offset;
  for (int i = 0; i < kShardCount; ++i) {
    edges[i].resize(vertex_count_per_edge_partition);
    offset[i] = vertex_count_per_edge_partition;
  }
  for (Vid vid = 0; vid < vertex_count; ++vid) {
    const Vid count = degrees[vid];
    const auto sid = vid % kShardCount;
    edges[sid][vid / kShardCount] =
        bit_cast<Edge>(Index{.offset = offset[sid], .count = count});
    offset[sid] += count;
  }

  for (int i = 0; i < kShardCount; ++i) edges[i].resize(offset[i]);
  vector<Vid> vertex_counts(vertex_count);
  for (Vid v0 = 0; v0 < vertex_count; ++v0) {
    for (auto [k, weight] : indexed_weights[v0]) {
      Vid v1 = k;
      for (auto [src, dst] : {std::tie(v0, v1), std::tie(v1, v0)}) {
        const auto src_sid = src % kShardCount;
        const auto index = bit_cast<Index>(edges[src_sid][src / kShardCount]);
        edges[src_sid][index.offset + vertex_counts[src]] = {
            .dst = Vid(dst),
            .weight = weight,
        };
        ++vertex_counts[src];
      }
    }
  }

  for (Vid vid = 0; vid < vertex_count; ++vid) {
    CHECK_EQ(
        vertex_counts[vid],
        bit_cast<Index>(edges[vid % kShardCount][vid / kShardCount]).count);
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
          tapa::mmaps<Edge, kShardCount> edges,
          tapa::mmaps<Vertex, kIntervalCount> vertices,
#ifdef TAPA_SSSP_COARSE_PRIORITY
          tapa::mmap<SpilledTask> cgpq_spill
#else   // TAPA_SSSP_COARSE_PRIORITY
          tapa::mmap<HeapElemPacked> heap_array,
          tapa::mmap<HeapIndexEntry> heap_index
#endif  // TAPA_SSSP_COARSE_PRIORITY
);

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
  CHECK_LT(bit_length(vertex_count), kVidWidth);

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
  const auto coarsen_records =
      Coarsen(indexed_weights, edge_count, degree_no_self_loop, edges);

  if (FLAGS_sort || FLAGS_shuffle) {
    for (auto& shard : edges) {
      for (Vid vid = 0; vid < tapa::round_up_div<kShardCount>(vertex_count);
           ++vid) {
        const auto index = bit_cast<Index>(shard[vid]);
        const auto first = shard.begin() + index.offset;
        const auto last = first + index.count;
        if (FLAGS_sort) {
          std::sort(first, last,
                    [](auto& a, auto& b) { return a.dst < b.dst; });
        } else {
          std::random_shuffle(first, last);
        }
      }
    }
  }

  // Other kernel arguments.
  aligned_vector<int64_t> metadata(9 + kSubIntervalCount * kVertexUniStatCount +
                                   kShardCount * kEdgeUnitStatCount +
                                   kQueueCount * kQueueStatCount +
                                   kSwitchCount * kSwitchStatCount);
  array<aligned_vector<Vertex>, kIntervalCount> vertices;
  for (auto& interval : vertices) {
    interval.resize(tapa::round_up_div<kIntervalCount>(vertex_count));
  }
  aligned_vector<SpilledTask> cgpq_spill(1 << 24);
  aligned_vector<HeapElemPacked> heap_array(
      GetAddrOfOffChipHeapElem(kLevelCount - 1,
                               GetCapOfLevel(kLevelCount - 1) - 1,
                               kQueueCount - 1) +
      1);
  aligned_vector<HeapIndexEntry> heap_index(vertex_count);

  // Statistics.
  vector<double> teps;

  for (int level = kOnChipLevelCount; level < kLevelCount; ++level) {
    VLOG(5) << "off-chip level " << level << " addr: ["
            << GetAddrOfOffChipHeapElem(level, 0, 0) << ", "
            << GetAddrOfOffChipHeapElem(level, GetCapOfLevel(level) - 1,
                                        kQueueCount - 1)
            << "]";
  }
  for (int level = 0; level < kLevelCount - 1; ++level) {
    // Child capacity should be the sum of all children's child capacity + 1.
    CHECK_EQ(GetChildCapOfLevel(level),
             GetChildCapOfLevel(level + 1) * kPiHeapWidth + 1);
  }

  for (int level = kOnChipLevelCount; level < kLevelCount; ++level) {
    const auto cap = GetChildCapOfLevel(level);
    HeapElemAxi init_elem;
    init_elem.valid = false;
    for (int i = 0; i < kPiHeapWidth; ++i) {
      init_elem.cap[i] = cap;
    }
    init_elem.size = 0;
    const auto init_elem_packed = HeapElemAxi::Pack({init_elem, init_elem});

    for (int qid = 0; qid < kQueueCount; ++qid) {
      for (int idx = 0; idx < GetCapOfLevel(level); idx += 2) {
        const auto addr = GetAddrOfOffChipHeapElem(level, idx, qid);
        heap_array[addr] = init_elem_packed;
      }
    }
  }

  for (const auto root : SampleVertices(degree_no_self_loop)) {
    CHECK_GE(root, 0) << "invalid root";
    CHECK_LT(root, vertex_count) << "invalid root";
    LOG(INFO) << "root: " << root;

    for (auto& interval : vertices) {
      std::fill(interval.begin(), interval.end(),
                Vertex{.parent = kNullVid, .distance = kInfDistance});
    }

    for (size_t i = 0; i < heap_index.size(); ++i) {
      heap_index[i].invalidate();
    }
    vertices[root % kIntervalCount][root / kIntervalCount] = {
        .parent = Vid(root), .distance = 0.f};
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      const auto index =
          bit_cast<Index>(edges[vid % kShardCount][vid / kShardCount]);
      auto& vertex = vertices[vid % kIntervalCount][vid / kIntervalCount];
      vertex.offset = index.offset;
      vertex.degree = index.count;
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
            tapa::read_only_mmaps<Edge, kShardCount>(edges),
            tapa::read_write_mmaps<Vertex, kIntervalCount>(vertices),
#ifdef TAPA_SSSP_COARSE_PRIORITY
            tapa::placeholder_mmap<SpilledTask>(cgpq_spill)
#else   // TAPA_SSSP_COARSE_PRIORITY
            tapa::read_only_mmap<HeapElemPacked>(heap_array),
            tapa::read_only_mmap<HeapIndexEntry>(heap_index)
#endif  // TAPA_SSSP_COARSE_PRIORITY
        );
    VLOG(3) << "kernel time: " << elapsed_time << " s";

    const auto tic = steady_clock::now();
    Refine(coarsen_records, vertices);
    const double refine_time =
        1e-9 * duration_cast<nanoseconds>(steady_clock::now() - tic).count();

    vector<Vid> parents(vertex_count);
    vector<float> distances(vertex_count);
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      const auto& vertex = vertices[vid % kIntervalCount][vid / kIntervalCount];
      parents[vid] = vertex.parent;
      distances[vid] = vertex.distance;
    }

    int64_t connected_edge_count = 0;
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      if (parents[vid] != kNullVid) {
        connected_edge_count += degree[vid] / 2;
      }
    }
    teps.push_back(connected_edge_count / (elapsed_time + refine_time));
    auto metadata_it = metadata.begin();
    auto visited_edge_count = *(metadata_it++);
    auto push_count = *(metadata_it++);
    auto pushpop_count = *(metadata_it++);
    auto pop_valid_count = *(metadata_it++);
    auto pop_noop_count = *(metadata_it++);
    ++metadata_it;
    auto cycle_count = *(metadata_it++);
    auto queue_full_count = *(metadata_it++);
    auto pe_full_count = *(metadata_it++);
    int64_t coarsened_edge_count = 0;
    for (auto& shard : edges) coarsened_edge_count += shard.size();
    VLOG(3) << "  TEPS:                  " << *teps.rbegin() << " ("
            << 1e9 * elapsed_time / visited_edge_count << " ns/edge visited + "
            << refine_time << " s refinement)";
    VLOG(3) << "  #edges connected:      " << connected_edge_count;
    VLOG(3) << "  #edges visited:        " << visited_edge_count << " ("
            << std::fixed << std::setprecision(1) << std::showpos
            << 100. * visited_edge_count / coarsened_edge_count - 100
            << "% over " << std::noshowpos << coarsened_edge_count << ") ";
    VLOG(3) << "  #PUSH:                 " << push_count;
    VLOG(3) << "  #PUSHPOP:              " << pushpop_count;
    VLOG(3) << "  #POP (valid):          " << pop_valid_count;
    VLOG(3) << "  #POP (noop):           " << pop_noop_count;
    VLOG(3) << "  cycle count:           " << cycle_count;
    VLOG_IF(3, queue_full_count)
        << "    queue full:          " << queue_full_count << " (" << std::fixed
        << std::setprecision(1) << 100. * queue_full_count / cycle_count
        << "%)";
    VLOG_IF(3, pe_full_count)
        << "    PE full:             " << pe_full_count << " (" << std::fixed
        << std::setprecision(1) << 100. * pe_full_count / cycle_count << "%)";

    for (int iid = 0; iid < kIntervalCount; ++iid) {
      for (int siid = 0; siid < kSubIntervalCount / kIntervalCount; ++siid) {
        VLOG(3) << "  interval[" << iid << "][" << siid << "]:";

        // Cache hit/miss.
        const auto read_hit = *(metadata_it++);
        const auto read_miss = *(metadata_it++);
        const auto write_hit = *(metadata_it++);
        const auto write_miss = *(metadata_it++);
        VLOG_IF(3, read_hit + read_miss)
            << "    read hit   : " << std::setfill(' ') << std::setw(10)
            << read_hit << " (" << std::fixed << std::setprecision(1)
            << 100. * read_hit / (read_hit + read_miss) << "%)";
        VLOG_IF(3, write_hit + write_miss)
            << "    write hit  : " << std::setfill(' ') << std::setw(10)
            << write_hit << " (" << std::fixed << std::setprecision(1)
            << 100. * write_hit / (write_hit + write_miss) << "%)";

        // Op counts.
        constexpr const char* kVertexUnitOpNamesAligned[] = {
            "#write_resp",  //
            "#read_resp ",  //
            "#req_hit   ",  //
            "#req_miss  ",  //
            "#req_busy  ",  //
            "#idle      ",  //
        };
        int64_t op_counts[sizeof(kVertexUnitOpNamesAligned) /
                          sizeof(kVertexUnitOpNamesAligned[0])];
        int64_t total_op_count = 0;
        for (int i = 0; i < sizeof(op_counts) / sizeof(op_counts[0]); ++i) {
          op_counts[i] = *(metadata_it++);
          total_op_count += op_counts[i];
        }
        for (int i = 0; i < sizeof(op_counts) / sizeof(op_counts[0]); ++i) {
          VLOG_IF(3, total_op_count)
              << "    " << kVertexUnitOpNamesAligned[i] << ": "
              << std::setfill(' ') << std::setw(10) << op_counts[i] << " ("
              << std::fixed << std::setprecision(1)
              << 100. * op_counts[i] / total_op_count << "%)";
        }
      }
    }

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
      }
    }

    for (int qid = 0; qid < kQueueCount; ++qid) {
      VLOG(3) << "  queue[" << qid << "]:";

#ifdef TAPA_SSSP_COARSE_PRIORITY
      const auto spill_count = *(metadata_it++);
      const auto max_heap_size = *(metadata_it++);
      const auto cycle_count = *(metadata_it++);
      const auto enqueue_full_count = *(metadata_it++);
      const auto enqueue_current_refill_count = *(metadata_it++);
      const auto enqueue_future_refill_count = *(metadata_it++);
      const auto enqueue_bank_conflict_count = *(metadata_it++);
      const auto dequeue_full_count = *(metadata_it++);
      const auto dequeue_bank_conflict_count = *(metadata_it++);
      VLOG(3) << "    spill count  : " << setfill(' ') << setw(10)
              << spill_count << " / " << cgpq_spill.size() / kCgpqChunkSize;
      VLOG(3) << "    max heap size: " << setfill(' ') << setw(10)
              << max_heap_size << " / " << kCgpqCapacity;
      auto vlog = [&](const char* msg, int64_t var) {
        VLOG(3) << "    " << msg << ": " << setfill(' ') << setw(10) << var
                << " / " << cycle_count << " (" << std::fixed
                << std::setprecision(1) << 100. * var / cycle_count << "%)";
      };
      vlog("push blocked by full buffer   ", enqueue_full_count);
      vlog("push blocked by current refill", enqueue_current_refill_count);
      vlog("push blocked by future refill ", enqueue_future_refill_count);
      vlog("push blocked by bank conflict ", enqueue_bank_conflict_count);
      vlog("pop blocked by full FIFO    ", dequeue_full_count);
      vlog("pop blocked by bank conflict", dequeue_bank_conflict_count);
#else   // TAPA_SSSP_COARSE_PRIORITY

      // Queue op counts.
      {
        constexpr const char* kQueueUnitOpNamesAligned[] = {
            "#idling  ",  //
            "#pushing ",  //
            "#popping ",  //
            "#indexing",  //
        };
        int64_t op_counts[sizeof(kQueueUnitOpNamesAligned) /
                          sizeof(kQueueUnitOpNamesAligned[0])];
        int64_t total_op_count = 0;
        for (int i = 0; i < sizeof(op_counts) / sizeof(op_counts[0]); ++i) {
          op_counts[i] = *(metadata_it++);
          total_op_count += op_counts[i];
        }
        const auto push_count = *(metadata_it++);
        const auto pop_count = *(metadata_it++);
        const auto pushpop_count = *(metadata_it++);
        const auto max_size = *(metadata_it++);
        auto total_size = *(metadata_it++) << 32;
        total_size += *(metadata_it++);

        VLOG_IF(3, max_size) << "    capacity: " << std::setfill(' ')
                             << std::setw(10) << kPiHeapCapacity;
        VLOG_IF(3, max_size) << "    max size: " << std::setfill(' ')
                             << std::setw(10) << max_size;
        VLOG_IF(3, total_op_count)
            << "    avg size: " << std::setfill(' ') << std::setw(10)
            << total_size / total_op_count;

        VLOG_IF(3, op_counts[1])
            << "    avg push    latency: " << std::setfill(' ') << std::setw(10)
            << op_counts[1] / push_count;
        VLOG_IF(3, op_counts[2])
            << "    avg     pop latency: " << std::setfill(' ') << std::setw(10)
            << op_counts[2] / pop_count;
        VLOG_IF(3, op_counts[3])
            << "    avg pushpop latency: " << std::setfill(' ') << std::setw(10)
            << op_counts[3] / pushpop_count;

        for (int i = 0; i < sizeof(op_counts) / sizeof(op_counts[0]); ++i) {
          VLOG_IF(3, total_op_count)
              << "    " << kQueueUnitOpNamesAligned[i] << ": "
              << std::setfill(' ') << std::setw(10) << op_counts[i] << " ("
              << std::fixed << std::setprecision(1)
              << 100. * op_counts[i] / total_op_count << "%)";
        }
      }

      // Index op counts.
      {
        int64_t state_counts[kPiHeapIndexOpTypeCount] = {};
        int64_t op_counts[kPiHeapIndexOpTypeCount] = {};
        int64_t total_op_count = 0;
        const char* kIndexOpNamesAligned[kPiHeapIndexOpTypeCount] = {
            "GET_STALE    ",  //
            "CLEAR_STALE  ",  //
            "UPDATE_INDEX ",  //
            "CLEAR_FRESH  ",  //
        };
        for (int level = 0; level < kLevelCount; ++level) {
          int64_t level_state_counts[kPiHeapIndexOpTypeCount];
          int64_t level_op_counts[kPiHeapIndexOpTypeCount];
          int64_t total_level_op_count = 0;
          for (int i = 0; i < kPiHeapIndexOpTypeCount; ++i) {
            level_state_counts[i] = *(metadata_it++);
            level_op_counts[i] = *(metadata_it++);
            total_level_op_count += level_op_counts[i];

            state_counts[i] += level_state_counts[i];
            op_counts[i] += level_op_counts[i];
            total_op_count += level_op_counts[i];
          }

          VLOG_IF(3, total_level_op_count) << "    level[" << level << "]:";
          for (int i = 0; i < kPiHeapIndexOpTypeCount; ++i) {
            VLOG_IF(3, total_level_op_count)
                << "      " << kIndexOpNamesAligned[i] << " : " << setfill(' ')
                << setw(10) << level_op_counts[i] << " ( " << fixed
                << setprecision(1) << setw(5)
                << 100. * level_op_counts[i] / total_level_op_count << "%) / "
                << setw(5) << 1. * level_state_counts[i] / level_op_counts[i];
          }
        }

        VLOG_IF(3, total_op_count) << "    total:";
        for (int i = 0; i < kPiHeapIndexOpTypeCount; ++i) {
          VLOG_IF(3, total_op_count)
              << "      " << kIndexOpNamesAligned[i] << " : " << setfill(' ')
              << setw(10) << op_counts[i] << " ( " << fixed << setprecision(1)
              << setw(5) << 100. * op_counts[i] / total_op_count << "%) / "
              << setw(5) << 1. * state_counts[i] / op_counts[i];
        }

        const auto read_hit = *(metadata_it++);
        const auto read_miss = *(metadata_it++);
        const auto write_hit = *(metadata_it++);
        const auto write_miss = *(metadata_it++);
        const auto idle_count = *(metadata_it++);
        const auto busy_count = *(metadata_it++);
        const auto collect_write_count = write_miss;
        const auto acquire_index_count = read_miss;

        // The `total_op_count` only counts the new requests; here the
        // `total_cycle_count` further includes the read/write acknowledges and
        // idle iteration. Note that II>1 so the real cycle count would be much
        // larger.
        const auto total_cycle_count = total_op_count + collect_write_count +
                                       acquire_index_count + idle_count +
                                       busy_count;

        VLOG(3) << "  index[" << qid << "]:";
        for (auto [name, item, total] :
             vector<tuple<const char*, int64_t, int64_t>>{
                 {"read hit ", read_hit, read_hit + read_miss},
                 {"write hit", write_hit, write_hit + write_miss},
                 {"idle         ", idle_count, total_cycle_count},
                 {"busy         ", busy_count, total_cycle_count},
                 {"collect write", collect_write_count, total_cycle_count},
                 {"acquire index", acquire_index_count, total_cycle_count},
                 {"new requests ", total_op_count, total_cycle_count},
             }) {
          VLOG_IF(3, total_cycle_count)
              << "    " << name << " : " << setfill(' ') << setw(10) << item
              << " ( " << fixed << setprecision(1) << setw(5)
              << 100. * item / total << "%)";
        }
      }
#endif  // TAPA_SSSP_COARSE_PRIORITY
    }

    for (int swid = 0; swid < kSwitchCount; ++swid) {
      VLOG(3) << "  switch[" << swid << "]:";

      constexpr const char* kSwitchStatNames[] = {
          "0 full    ",  //
          "1 full    ",  //
          "0 conflict",  //
          "1 conflict",  //
      };
      const auto total_count = *(metadata_it++);
      for (int i = 0; i < kSwitchStatCount - 1; ++i) {
        const auto stall_count = *(metadata_it++);
        VLOG(3) << "    " << kSwitchStatNames[i] << ": " << setfill(' ')
                << setw(10) << stall_count << " ( " << fixed << setprecision(1)
                << setw(5) << 100. * stall_count / total_count << "%)";
      }
    }

    if (!IsValid(root, edges_view, weights_view, indexed_weights,
                 parents.data(), distances.data(), vertex_count)) {
      return 1;
    }
  }

  printf("sssp harmonic_mean_TEPS:     !  %g\n", geo_mean(teps));

  return 0;
}

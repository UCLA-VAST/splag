#include <cstdint>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <openvdb/openvdb.h>
#include <tapa.h>

#include "sssp-host.h"
#include "sssp.h"
#include "util.h"

using std::array;
using std::fixed;
using std::make_unique;
using std::move;
using std::setfill;
using std::setprecision;
using std::setw;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;
using std::chrono::duration;
using std::chrono::steady_clock;

using openvdb::Coord;
using openvdb::FloatGrid;

using GiType = uint16_t;  // image_t in recut

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");
DEFINE_bool(is_log_bucket, false, "use logarithm bucket instead of linear");
DEFINE_double(min_distance, 0, "min distance");
DEFINE_double(max_distance, 32, "max distance");
DEFINE_string(markers, "", "path to markers directory");
DEFINE_string(gridname, "topology", "OpenVDB grid name");
DEFINE_bool(check, false, "check consistency of converted FPGA data");
DEFINE_string(clip, "",
              "relative bounding box for clipping input grid (if not empty)");
DEFINE_bool(compact, true, "compact edges");
DEFINE_bool(shrink, false, "shrink vector after edge compaction");
DEFINE_double(weight, 0.f, "use fixed edge weight if >0");

namespace {

constexpr float kGiVals[] = {
    22026.5, 20368,   18840.3, 17432.5, 16134.8, 14938.4, 13834.9, 12816.8,
    11877.4, 11010.2, 10209.4, 9469.8,  8786.47, 8154.96, 7571.17, 7031.33,
    6531.99, 6069.98, 5642.39, 5246.52, 4879.94, 4540.36, 4225.71, 3934.08,
    3663.7,  3412.95, 3180.34, 2964.5,  2764.16, 2578.14, 2405.39, 2244.9,
    2095.77, 1957.14, 1828.24, 1708.36, 1596.83, 1493.05, 1396.43, 1306.47,
    1222.68, 1144.62, 1071.87, 1004.06, 940.819, 881.837, 826.806, 775.448,
    727.504, 682.734, 640.916, 601.845, 565.329, 531.193, 499.271, 469.412,
    441.474, 415.327, 390.848, 367.926, 346.454, 326.336, 307.481, 289.804,
    273.227, 257.678, 243.089, 229.396, 216.541, 204.469, 193.129, 182.475,
    172.461, 163.047, 154.195, 145.868, 138.033, 130.659, 123.717, 117.179,
    111.022, 105.22,  99.7524, 94.5979, 89.7372, 85.1526, 80.827,  76.7447,
    72.891,  69.2522, 65.8152, 62.5681, 59.4994, 56.5987, 53.856,  51.2619,
    48.8078, 46.4854, 44.2872, 42.2059, 40.2348, 38.3676, 36.5982, 34.9212,
    33.3313, 31.8236, 30.3934, 29.0364, 27.7485, 26.526,  25.365,  24.2624,
    23.2148, 22.2193, 21.273,  20.3733, 19.5176, 18.7037, 17.9292, 17.192,
    16.4902, 15.822,  15.1855, 14.579,  14.0011, 13.4503, 12.9251, 12.4242,
    11.9464, 11.4905, 11.0554, 10.6401, 10.2435, 9.86473, 9.50289, 9.15713,
    8.82667, 8.51075, 8.20867, 7.91974, 7.64333, 7.37884, 7.12569, 6.88334,
    6.65128, 6.42902, 6.2161,  6.01209, 5.81655, 5.62911, 5.44938, 5.27701,
    5.11167, 4.95303, 4.80079, 4.65467, 4.51437, 4.37966, 4.25027, 4.12597,
    4.00654, 3.89176, 3.78144, 3.67537, 3.57337, 3.47528, 3.38092, 3.29013,
    3.20276, 3.11868, 3.03773, 2.9598,  2.88475, 2.81247, 2.74285, 2.67577,
    2.61113, 2.54884, 2.48881, 2.43093, 2.37513, 2.32132, 2.26944, 2.21939,
    2.17111, 2.12454, 2.07961, 2.03625, 1.99441, 1.95403, 1.91506, 1.87744,
    1.84113, 1.80608, 1.77223, 1.73956, 1.70802, 1.67756, 1.64815, 1.61976,
    1.59234, 1.56587, 1.54032, 1.51564, 1.49182, 1.46883, 1.44664, 1.42522,
    1.40455, 1.3846,  1.36536, 1.3468,  1.3289,  1.31164, 1.29501, 1.27898,
    1.26353, 1.24866, 1.23434, 1.22056, 1.2073,  1.19456, 1.18231, 1.17055,
    1.15927, 1.14844, 1.13807, 1.12814, 1.11864, 1.10956, 1.10089, 1.09262,
    1.08475, 1.07727, 1.07017, 1.06345, 1.05709, 1.05109, 1.04545, 1.04015,
    1.03521, 1.0306,  1.02633, 1.02239, 1.01878, 1.0155,  1.01253, 1.00989,
    1.00756, 1.00555, 1.00385, 1.00246, 1.00139, 1.00062, 1.00015, 1};

float GetWeight(GiType gi, GiType min = 32767, GiType max = 65534) {
  if (FLAGS_weight > 0) {
    return FLAGS_weight;
  }
  const int idx = float(gi - min) / max * 255;
  CHECK_GE(idx, 0);
  CHECK_LT(idx, 256);
  return kGiVals[idx] / kGiVals[0];
}

Coord ReadMarker(const string& filename) {
  VLOG(5) << "reading marker file '" << filename << "'";
  std::ifstream ifs(filename);
  string line;
  while (std::getline(ifs, line)) {
    if (line[0] == '#') {
      continue;
    }

    int32_t x, y, z;
    CHECK_EQ(sscanf(line.c_str(), "%d,%d,%d", &x, &y, &z), 3)
        << "unexpected marker file";
    return Coord(x, y, z);
  }
  return {};
}

vector<Coord> ReadMarkers(const string& dirname) {
  vector<Coord> markers;
  auto dir = opendir(dirname.c_str());
  while (auto ent = readdir(dir)) {
    if (ent->d_type == DT_REG) {
      markers.push_back(ReadMarker(dirname + '/' + ent->d_name));
    }
  }
  CHECK_ERR(closedir(dir));
  return markers;
}

class ElapsedTime {
  steady_clock::time_point first_time_point = steady_clock::now();
  steady_clock::time_point last_time_point = first_time_point;
  friend std::ostream& operator<<(std::ostream& os, ElapsedTime& obj) {
    const auto time_point = steady_clock::now();
    os << " (" << duration<float>(time_point - obj.last_time_point).count()
       << " s / " << duration<float>(time_point - obj.first_time_point).count()
       << " s)";
    obj.last_time_point = time_point;
    return os;
  }
};

constexpr int kEdgeCountPerVertex = 8;

}  // namespace

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    LOG(FATAL) << "usage: " << argv[0] << " <float.vdb file>";
    return 1;
  }

  int64_t edge_count;
  int64_t vertex_count;
  int32_t root;
  array<aligned_vector<Edge>, kShardCount> shards;
  array<aligned_vector<Vertex>, kIntervalCount> intervals;

  vector<Coord> vid2coord;

  thread bg_thread;

  ElapsedTime elapsed_time;

  // Pre-processing.
  {
    const auto markers = ReadMarkers(FLAGS_markers);
    openvdb::initialize();
    LOG(INFO) << "read markers and initialize openvdb" << elapsed_time;

    FloatGrid::Ptr gi_grid;
    {
      openvdb::io::File file(argv[1]);
      file.open();
      gi_grid = openvdb::gridPtrCast<FloatGrid>(file.readGrid(FLAGS_gridname));
      if (!FLAGS_clip.empty()) {
        int32_t x_min, y_min, z_min, x_max, y_max, z_max;
        CHECK_EQ(sscanf(FLAGS_clip.c_str(), "%d%d%d%d%d%d", &x_min, &y_min,
                        &z_min, &x_max, &y_max, &z_max),
                 6)
            << "unexpected clip argument";
        const auto bbox = gi_grid->evalActiveVoxelBoundingBox();
        gi_grid->clip({
            {
                bbox.min().x() + x_min,
                bbox.min().y() + y_min,
                bbox.min().z() + z_min,
            },
            {
                bbox.min().x() + x_max,
                bbox.min().y() + y_max,
                bbox.min().z() + z_max,
            },
        });
      }
      gi_grid->tree().voxelizeActiveTiles();
    }
    root = gi_grid->activeVoxelCount();
    vertex_count = root + 1;
    LOG(INFO) << "loaded openvdb with " << root << " voxels ("
              << 100. * gi_grid->activeVoxelCount() /
                     gi_grid->evalActiveVoxelBoundingBox().volume()
              << "%)" << elapsed_time;

    // Initialize vertices.
    for (int iid = 0; iid < kIntervalCount; ++iid) {
      intervals[iid].reserve((vertex_count - 1) / kIntervalCount + 1);
      for (int64_t i = 0; i < intervals[iid].capacity(); ++i) {
        const int32_t vid = i * kIntervalCount + iid;
        intervals[iid].push_back({
            .parent = kNullVid,
            .distance = kInfDistance,
            .offset = vid / kShardCount * kEdgeCountPerVertex / kEdgeVecLen,
            .degree = 0,
        });
      }
    }
    CHECK_EQ(kEdgeCountPerVertex % kEdgeVecLen, 0) << "edge alignment issue";
    for (int sid = 0; sid < kShardCount; ++sid) {
      auto size = tapa::round_up_div<kShardCount>(root) * kEdgeCountPerVertex;
      if (root % kShardCount == sid) {
        size += markers.size();
      }
      if (FLAGS_compact) {
        shards[sid].resize(size);
      } else {
        shards[sid].resize(size, {.dst = kNullVid, .weight = kInfDistance});
      }
    }
    LOG(INFO) << "allocated memory for accelerator" << elapsed_time;

    edge_count = 0;
    vid2coord.reserve(vertex_count - 1);
    auto coord2vid_grid = std::make_unique<openvdb::Int32Grid>();
    auto coord2vid_accessor = coord2vid_grid->getAccessor();
    const auto gi_accessor = gi_grid->getConstAccessor();
    for (auto it = gi_grid->beginValueOn(); it; ++it) {
      const GiType gi = *it;
      const auto coord = it.getCoord();

      const int32_t vid = vid2coord.size();
      vid2coord.push_back(coord);
      coord2vid_accessor.setValue(coord, vid);

      auto& vertex = intervals[vid % kIntervalCount][vid / kIntervalCount];

      for (const auto& neighbor_coord : {
               coord.offsetBy(1, 0, 0),
               coord.offsetBy(-1, 0, 0),
               coord.offsetBy(0, 1, 0),
               coord.offsetBy(0, -1, 0),
               coord.offsetBy(0, 0, 1),
               coord.offsetBy(0, 0, -1),
           }) {
        float neighbor_gi;
        if (gi_accessor.probeValue(neighbor_coord, neighbor_gi)) {
          int32_t neighbor_vid;
          if (coord2vid_accessor.probeValue(neighbor_coord, neighbor_vid)) {
            auto& neighbor_vertex = intervals[neighbor_vid % kIntervalCount]
                                             [neighbor_vid / kIntervalCount];

            const auto weight = (GetWeight(gi) + GetWeight(neighbor_gi)) / 2;
            shards[vid % kShardCount]
                  [vertex.offset * kEdgeVecLen + vertex.degree] = {
                      .dst = neighbor_vid,
                      .weight = weight,
                  };
            shards[neighbor_vid % kShardCount]
                  [neighbor_vertex.offset * kEdgeVecLen +
                   neighbor_vertex.degree] = {
                      .dst = vid,
                      .weight = weight,
                  };
            ++vertex.degree;
            ++neighbor_vertex.degree;
            ++edge_count;
          }
        }
      }
    }
    CHECK_EQ(vertex_count, vid2coord.size() + 1);
    LOG(INFO) << "initialized memory for accelerator" << elapsed_time;

    // Create the root vertex.
    {
      auto& root_vertex =
          intervals[root % kIntervalCount][root / kIntervalCount];
      root_vertex.parent = root;
      root_vertex.distance = 0.f;
      CHECK_EQ(root_vertex.offset,
               root / kShardCount * kEdgeCountPerVertex / kEdgeVecLen);
      CHECK_EQ(root_vertex.degree, 0);
      auto& root_shard = shards[root % kShardCount];
      for (auto& marker_coord : markers) {
        int32_t marker_vid;
        CHECK_EQ(gi_accessor.isValueOn(marker_coord),
                 coord2vid_accessor.isValueOn(marker_coord));
        if (coord2vid_accessor.probeValue(marker_coord, marker_vid)) {
          auto& marker_vertex = intervals[marker_vid % kIntervalCount]
                                         [marker_vid / kIntervalCount];
          root_shard[root_vertex.offset * kEdgeVecLen + root_vertex.degree] = {
              .dst = marker_vid,
              .weight = 0.f,
          };
          shards[marker_vid % kShardCount]
                [marker_vertex.offset * kEdgeVecLen + marker_vertex.degree] = {
                    .dst = root, .weight = 0.f};
          ++root_vertex.degree;
          ++marker_vertex.degree;
          ++edge_count;
        }
      }

      CHECK_GT(root_vertex.degree, 0) << "no marker in region";
      LOG(INFO) << "added " << root_vertex.degree << " markders"
                << elapsed_time;
    }

    if (FLAGS_compact)
      for (int sid = 0; sid < kShardCount; ++sid) {
        int32_t offset = 0;
        for (int vid = sid; vid < vertex_count; vid += kShardCount) {
          auto& vertex = intervals[vid % kIntervalCount][vid / kIntervalCount];
          if (offset < vertex.offset) {
            for (int32_t i = 0; i < vertex.degree; ++i) {
              shards[sid][offset * kEdgeVecLen + i] =
                  shards[sid][vertex.offset * kEdgeVecLen + i];
            }
            vertex.offset = offset;
          }
          std::fill_n(
              shards[sid].begin() + offset * kEdgeVecLen + vertex.degree,
              tapa::round_up<kEdgeVecLen>(vertex.degree) - vertex.degree,
              Edge{.dst = kNullVid, .weight = kInfDistance});
          offset += tapa::round_up_div<kEdgeVecLen>(vertex.degree);
        }
        shards[sid].resize(offset * kEdgeVecLen,
                           {.dst = kNullVid, .weight = kInfDistance});
        if (FLAGS_shrink) {
          shards[sid].shrink_to_fit();
        }
      }
    LOG(INFO) << "compacted edge memory for accelerator" << elapsed_time;

    if (FLAGS_check) {
      int marker_count = 0;
      int edge_count_for_checking = 0;
      for (int64_t vid = 0; vid < vertex_count - 1; ++vid) {
        const auto vertex =
            intervals[vid % kIntervalCount][vid / kIntervalCount];
        CHECK_EQ(vertex.parent, kNullVid);
        CHECK_EQ(vertex.distance, kInfDistance);
        CHECK_LE(vertex.offset,
                 vid / kShardCount * kEdgeCountPerVertex / kEdgeVecLen);
        CHECK_GE(vertex.degree, 0);
        CHECK_LE(vertex.degree, 7);
        const auto coord = vid2coord[vid];
        {
          int32_t retrieved_vid;
          CHECK(coord2vid_accessor.probeValue(coord, retrieved_vid)) << vid;
          CHECK_EQ(vid, retrieved_vid);
        }
        float gi;
        CHECK(gi_accessor.probeValue(coord, gi));
        const auto& shard = shards[vid % kShardCount];
        for (int i = 0; i < tapa::round_up<kEdgeVecLen>(vertex.degree); ++i) {
          const auto edge = shard[vertex.offset * kEdgeVecLen + i];
          if (i < vertex.degree) {
            ++edge_count_for_checking;
            if (edge.dst == root) {
              CHECK_EQ(edge.weight, 0.f);
              ++edge_count_for_checking;  // Count edges from root.
              ++marker_count;
              continue;
            }
            CHECK_GT(edge.weight, 0.f);
            const auto neighbor_coord = vid2coord[edge.dst];
            {
              const auto diff = neighbor_coord - coord;
              CHECK_EQ(
                  std::abs(diff.x()) + std::abs(diff.y()) + std::abs(diff.z()),
                  1)
                  << "vid: " << vid << " " << coord << " " << neighbor_coord;
            }
            float neighbor_gi;
            CHECK(gi_accessor.probeValue(neighbor_coord, neighbor_gi));
            const auto weight = (GetWeight(gi) + GetWeight(neighbor_gi)) / 2;
            CHECK_EQ(edge.weight, weight);
          } else {
            CHECK_EQ(edge.dst, kNullVid);
            CHECK_EQ(edge.weight, kInfDistance);
          }
        }
      }
      CHECK_EQ(edge_count_for_checking, edge_count * 2);

      const auto& root_vertex =
          intervals[root % kIntervalCount][root / kIntervalCount];
      CHECK_EQ(marker_count, root_vertex.degree);
      LOG(INFO) << "checked consistency" << elapsed_time;
    }

    bg_thread = thread([_0 = move(coord2vid_grid), _1 = move(gi_grid)] {});
  }

  CHECK_LT(vertex_count, (1 << kVidWidth));

  VLOG(3) << "total vertex count: " << vertex_count;
  VLOG(3) << "total edge count: " << edge_count;
  VLOG(3) << "avg degree: " << fixed << setprecision(1)
          << 1. * edge_count / vertex_count;

  // Other kernel arguments.
  aligned_vector<int64_t> metadata(
      kGlobalStatCount + kSubIntervalCount * kVertexUniStatCount +
      kShardCount * kEdgeUnitStatCount + kQueueStatCount);
  array<SpilledTaskPerMem*, kCgpqPhysMemCount> spill_raw_ptrs;
  array<unique_ptr<SpilledTaskPerMem>, kCgpqPhysMemCount> spill_unique_ptrs;
  array<uint64_t, kCgpqPhysMemCount> spill_sizes;
  for (int i = 0; i < kCgpqPhysMemCount; ++i) {
    spill_sizes[i] = 1 << uint_spill_addr_t::width;
    CHECK_LE(spill_sizes[i] * sizeof(SpilledTaskPerMem), 512 * 1024 * 1024);
    spill_raw_ptrs[i] = static_cast<SpilledTaskPerMem*>(
        aligned_alloc(4096, spill_sizes[i] * sizeof(SpilledTaskPerMem)));
    spill_unique_ptrs[i].reset(spill_raw_ptrs[i]);
  }

  {
    const float arg_min_distance = FLAGS_min_distance;
    const float arg_max_distance = FLAGS_max_distance;
    LOG(INFO) << "using min distance " << arg_min_distance;
    LOG(INFO) << "using max distance " << arg_max_distance;

    const double elapsed_time =
        1e-9 *
        tapa::invoke(
            SSSP, FLAGS_bitstream,
            Task{
                .vid = Vid(root),
                .vertex =
                    intervals[root % kIntervalCount][root / kIntervalCount],
            },
            tapa::write_only_mmap<int64_t>(metadata),
            tapa::read_only_mmaps<Edge, kShardCount>(shards)
                .vectorized<kEdgeVecLen>(),
            tapa::read_write_mmaps<Vertex, kIntervalCount>(intervals),
            FLAGS_is_log_bucket, arg_min_distance, arg_max_distance,
            tapa::placeholder_mmaps<SpilledTaskPerMem, kCgpqPhysMemCount>(
                spill_raw_ptrs, spill_sizes));

    vector<Vid> parents(vertex_count);
    vector<float> distances(vertex_count);
    float max_distance = std::numeric_limits<float>::min();
    for (int64_t vid = 0; vid < vertex_count; ++vid) {
      const auto& vertex =
          intervals[vid % kIntervalCount][vid / kIntervalCount];
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
        connected_edge_count +=
            intervals[vid % kIntervalCount][vid / kIntervalCount].degree;
      }
    }

    Log(metadata, shards, connected_edge_count,
        spill_sizes[0] / kCgpqBankCountPerMem /
            (kCgpqChunkSize / kSpilledTaskVecLen),
        elapsed_time, /*refine_time=*/0);
  }

  // Post-processing.
  {
    LOG(INFO) << "ran accelerator" << elapsed_time;

    openvdb::Int32Grid parent_grid;
    auto parent_accessor = parent_grid.getAccessor();
    for (int iid = 0; iid < kIntervalCount; ++iid) {
      for (int64_t i = 0; i < intervals[iid].size(); ++i) {
        const int32_t vid = i * kIntervalCount + iid;
        if (vid < vertex_count) {
          const auto& vertex = intervals[iid][i];
          if (vertex.parent != kNullVid) {
            const auto coord = vid2coord[vid];
            parent_accessor.setValue(coord, vertex.parent);
          }
        }
      }
    }

    LOG(INFO) << "selected " << parent_grid.activeVoxelCount()
              << " voxels with parents" << elapsed_time;

    if (bg_thread.joinable()) {
      bg_thread.join();
    }
  }

  return 0;
}

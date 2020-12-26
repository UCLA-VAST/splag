#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

using tapa::detach;
using tapa::istream;
using tapa::ostream;
using tapa::packet;
using tapa::seq;
using tapa::stream;
using tapa::streams;
using tapa::task;

using VidPkt = packet<PeId, Vid>;
using VertexPkt = packet<PeId, Vertex>;

// Estimated DRAM latency.
constexpr int kMemLatency = 50;

constexpr int kQueueCount = 4;

static_assert(kQueueCount % kIntervalCount == 0,
              "current implementation requires that queue count is a multiple "
              "of interval count");
static_assert(kQueueCount % kShardCount == 0,
              "current implementation requires that queue count is a multiple "
              "of shard count");

// Verbosity definitions:
//   v=5: O(1)
//   v=8: O(#vertex)
//   v=9: O(#edge)

// The queue is initialized with the root vertex. Each request is either a push
// or a pop.
//
// Each push request puts the task in the queue if there isn't a task for the
// same vertex in the queue, or decreases the priority of the existing task
// using the new value. Whether a new task is created is returned in the
// response.
//
// Each pop removes a task if the queue is not empty, otherwise the response
// indicates that the queue is empty.
void TaskQueue(
    // Scalar
    Vid qid,
    // Queue requests.
    tapa::istream<QueueOp>& queue_req_q,
    tapa::ostream<QueueOpResp>& queue_resp_q, tapa::mmap<Vertex> vertices,
    tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
#pragma HLS inline recursive
  // Heap rule: child <= parent
  Vid heap_size = 0;

  constexpr int kIndexCacheSize = 4096 * 4;
  tapa::packet<Vid, Vid> heap_index_cache[kIndexCacheSize];
#pragma HLS resource variable = heap_index_cache core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_index_cache
  int32_t read_hit = 0;
  int32_t read_miss = 0;
  int32_t write_hit = 0;
  int32_t write_miss = 0;
heap_index_cache_init:
  for (Vid i = 0; i < kIndexCacheSize; ++i) {
#pragma HLS pipeline II = 1
    heap_index_cache[i].addr = kNullVid;
  }
  auto get_heap_index = [&](Vid vid) -> Vid {
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr != vid) {
      if (entry.addr != kNullVid) {
        ++write_miss;
        heap_index[entry.addr] = entry.payload;
      }
      entry.addr = vid;
      entry.payload = heap_index[vid];
      ++read_miss;
    } else {
      ++read_hit;
    }
    return entry.payload;
  };
  auto set_heap_index_noalloc = [&](Vid vid, Vid index) {
    CHECK_NE(vid, kNullVid);
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr == vid) {
      ++write_hit;
      entry.addr = vid;
      entry.payload = index;
    } else {
      ++write_miss;
      heap_index[vid] = index;
    }
  };
  auto set_heap_index = [&](Vid vid, Vid index) {
    CHECK_NE(vid, kNullVid);
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    if (entry.addr != vid && entry.addr != kNullVid) {
      heap_index[entry.addr] = entry.payload;
      ++write_miss;
    } else {
      ++write_hit;
    }
    entry.addr = vid;
    entry.payload = index;
  };
  auto clear_heap_index = [&](Vid vid) {
    auto& entry = heap_index_cache[vid / kQueueCount % kIndexCacheSize];
    ++write_miss;
    if (entry.addr == vid) {
      entry.addr = kNullVid;
    }
    heap_index[vid] = kNullVid;
  };

  // #elements shared by the on-chip heap and the off-chip heap.
  constexpr int kHeapSharedSize = 4096;
  static_assert(is_power_of(kHeapSharedSize, kHeapOnChipWidth),
                "invalid heap configuration");
  static_assert(is_power_of(kHeapSharedSize, kHeapOffChipWidth),
                "invalid heap configuration");
  // #elements in the on-chip heap.
  constexpr int kHeapOnChipSize =
      (kHeapSharedSize * kHeapOnChipWidth - 1) / (kHeapOnChipWidth - 1);
  // #elements whose children are on chip.
  constexpr int kHeapOnChipBound = (kHeapOnChipSize - 1) / kHeapOnChipWidth;
  // #elements skipped in the off-chip heap (because they are on-chip).
  constexpr int kHeapOffChipSkipped =
      (kHeapOffChipWidth * kHeapOnChipSize * (kHeapOnChipWidth - 1) +
       kHeapOffChipWidth - kHeapOnChipWidth) /
      (kHeapOffChipWidth - 1) / kHeapOnChipWidth;
  // #elements difference between off-chip indices and mixed indices.
  constexpr int kHeapDiff = kHeapOnChipSize - kHeapOffChipSkipped;
  // #elements in the mixed heap whose parent is on chip.
  constexpr int kHeapOffChipBound =
      kHeapOnChipSize +
      kHeapOffChipWidth *
          (kHeapOnChipSize - (kHeapOnChipSize - 1) / kHeapOnChipWidth);

  /*
   *  parent of i:
   *    if i < kHeapOnChipSize:                           {on-chip}
   *      (i-1)/kHeapOnChipWidth                            {on-chip}
   *    elif i < kHeapOffChipBound                        {off-chip}
   *      (i-kHeapDiff-1)/kHeapOffChipWidth+kHeapDiff       {on-chip}
   *        = (i+(kHeapDiff*(kHeapOffChipWidth-1)-1))/kHeapOffChipWidth
   *    else:                                             {off-chip}
   *      (i-kHeapDiff-1)/kHeapOffChipWidth+kHeapDiff       {off-chip}
   *        = (i+(kHeapDiff*(kHeapOffChipWidth-1)-1))/kHeapOffChipWidth
   *
   *  first child of i:
   *    if i < kHeapOnChipBound:                                    {on-chip}
   *      i*kHeapOnChipWidth+1                                        {on-chip}
   *    elif i < kHeapOnChipSize:                                   {on-chip}
   *      (i-kHeapDiff)*kHeapOffChipWidth+kHeapDiff+1                 {off-chip}
   *        = i*kHeapOffChipWidth-kHeapDiff*(kHeapOffChipWidth-1)+1
   *    else:                                                       {off-chip}
   *      (i-kHeapDiff)*kHeapOffChipWidth+kHeapDiff+1                 {off-chip}
   *        = i*kHeapOffChipWidth-kHeapDiff*(kHeapOffChipWidth-1)+1
   */

  TaskOnChip heap_array_cache[kHeapOnChipSize];
#pragma HLS resource variable = heap_array_cache core = RAM_2P_URAM
#pragma HLS data_pack variable = heap_array_cache

  auto get_heap_array_index = [&](Vid i) {
    return (i - kHeapOnChipSize) * kQueueCount + qid;
  };
  auto get_heap_elem_on_chip = [&](Vid i) {
    CHECK_LT(i, kHeapOnChipSize);
    return Task(heap_array_cache[i]);
  };
  auto get_heap_elem_off_chip = [&](Vid i) {
    CHECK_GE(i, kHeapOnChipSize);
    return heap_array[get_heap_array_index(i)];
  };
  auto get_heap_elem = [&](Vid i) {
    return i < kHeapOnChipSize ? get_heap_elem_on_chip(i)
                               : get_heap_elem_off_chip(i);
  };
  auto set_heap_elem_on_chip = [&](Vid i, Task task) {
    CHECK_LT(i, kHeapOnChipSize);
    heap_array_cache[i] = TaskOnChip(task);
  };
  auto set_heap_elem_off_chip = [&](Vid i, Task task) {
    CHECK_GE(i, kHeapOnChipSize);
    heap_array[get_heap_array_index(i)] = task;
  };
  auto set_heap_elem = [&](Vid i, Task task) {
    i < kHeapOnChipSize ? set_heap_elem_on_chip(i, task)
                        : set_heap_elem_off_chip(i, task);
  };

  // Performance counters.
  int32_t heapify_up_count = 0;
  int32_t heapify_up_on_chip = 0;
  int32_t heapify_up_off_chip = 0;
  int32_t heapify_down_count = 0;
  int32_t heapify_down_on_chip = 0;
  int32_t heapify_down_off_chip = 0;

  CLEAN_UP(clean_up, [&]() {
    VLOG(3) << "average heapify up trip count (on-chip): "
            << 1. * heapify_up_on_chip / heapify_up_count;
    VLOG(3) << "average heapify up trip count (off-chip): "
            << 1. * heapify_up_off_chip / heapify_up_count;
    VLOG(3) << "average heapify down trip count (on-chip): "
            << 1. * heapify_down_on_chip / heapify_down_count;
    VLOG(3) << "average heapify down trip count (off-chip): "
            << 1. * heapify_down_off_chip / heapify_down_count;
    VLOG(3) << "read hit rate: " << read_hit * 100. / (read_hit + read_miss)
            << "%";
    VLOG(3) << "write hit rate: " << write_hit * 100. / (write_hit + write_miss)
            << "%";

    // Check that heap_index is restored to the initial state.
    CHECK_EQ(heap_size, 0);
    for (int i = 0; i < heap_index.size(); ++i) {
      CHECK_EQ(heap_index.get()[i], kNullVid) << "i = " << i;
    }
    for (int i = 0; i < kIndexCacheSize; ++i) {
      CHECK_EQ(heap_index_cache[i].addr, kNullVid) << "i = " << i;
    }
  });

spin:
  for (;;) {
    QueueOp req = queue_req_q.read();
    QueueOpResp resp{
        .queue_op = req.op,
        .task_op = TaskOp::NOOP,
        .task = req.task,
        .queue_size = 0,
    };
    switch (req.op) {
      case QueueOp::PUSH: {
        const auto new_task = req.task;
        CHECK_EQ(new_task.vid % kQueueCount, qid);
        CHECK_EQ(new_task.vid % kIntervalCount, qid % kIntervalCount);
        const Vid task_index = get_heap_index(new_task.vid);
        bool heapify = true;
        bool is_new = true;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const Task old_task = get_heap_elem(task_index);
          CHECK_EQ(old_task.vid, new_task.vid);
          if (new_task <= old_task) {
            heapify = false;
          } else {
            is_new = false;
          }
        }

        if (heapify &&
            !(bit_cast<uint32_t>(
                  vertices[new_task.vid / kIntervalCount].distance) <=
              bit_cast<uint32_t>(new_task.vertex.distance))) {
          if (is_new) {
            heapify_index = heap_size;
            ++heap_size;
            resp.task_op = TaskOp::NEW;
          }
        } else {
          heapify = false;
          clear_heap_index(new_task.vid);
        }

        if (heapify) {
          vertices[new_task.vid / kIntervalCount] = new_task.vertex;
          // Increase the priority of heap_array[i] if necessary.
          Vid i = heapify_index;
          const Task task_i = new_task;

          ++heapify_up_count;

        heapify_up_off_chip:
          for (; !(i < kHeapOffChipBound);) {
#pragma HLS pipeline
            ++heapify_up_off_chip;

            const auto parent =
                (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                kHeapOffChipWidth;
            const auto task_parent = get_heap_elem_off_chip(parent);
            if (task_i <= task_parent) break;

            set_heap_elem_off_chip(i, task_parent);
            set_heap_index(task_parent.vid, i);
            i = parent;
          }

          if (!(i < kHeapOnChipSize) && i < kHeapOffChipBound) {
            const auto parent =
                (i + (kHeapDiff * (kHeapOffChipWidth - 1) - 1)) /
                kHeapOffChipWidth;
            const auto task_parent = get_heap_elem_on_chip(parent);
            if (!(task_i <= task_parent)) {
              ++heapify_up_off_chip;

              set_heap_elem_off_chip(i, task_parent);
              set_heap_index(task_parent.vid, i);
              i = parent;
            } else {
              ++heapify_up_on_chip;
            }
          }

        heapify_up_on_chip:
          for (; i != 0 && i < kHeapOnChipSize;) {
#pragma HLS pipeline
            ++heapify_up_on_chip;

            const auto parent = (i - 1) / kHeapOnChipWidth;
            const auto task_parent = get_heap_elem_on_chip(parent);
            if (task_i <= task_parent) break;

            set_heap_elem_on_chip(i, task_parent);
            set_heap_index(task_parent.vid, i);
            i = parent;
          }

          set_heap_elem(i, task_i);
          set_heap_index(task_i.vid, i);
        }
        break;
      }
      case QueueOp::POP: {
        if (heap_size != 0) {
          const Task front(heap_array_cache[0]);
          clear_heap_index(front.vid);
          --heap_size;

          resp.task_op = TaskOp::NEW;
          resp.task = front;

          if (heap_size != 0) {
            ++heapify_down_count;

            // Find proper index `i` for `task_i`.
            const Task task_i = get_heap_elem(heap_size);
            Vid i = 0;

          heapify_down_on_chip:
            for (; i < kHeapOnChipBound;) {
#pragma HLS pipeline
              ++heapify_down_on_chip;

              Vid max = -1;
              Task task_max = task_i;
              for (int j = 1; j <= kHeapOnChipWidth; ++j) {
#pragma HLS unroll
                const Vid child = i * kHeapOnChipWidth + j;
                if (child < heap_size) {
                  const auto task_child = get_heap_elem_on_chip(child);
                  if (!(task_child <= task_max)) {
                    max = child;
                    task_max = task_child;
                  }
                }
              }
              if (max == -1) break;

              set_heap_elem_on_chip(i, task_max);
              set_heap_index(task_max.vid, i);
              i = max;
            }

            if (!(i < kHeapOnChipBound) && i < kHeapOnChipSize) {
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              for (int j = 1; j <= kHeapOffChipWidth; ++j) {
#pragma HLS unroll
                const Vid child = i * kHeapOffChipWidth -
                                  kHeapDiff * (kHeapOffChipWidth - 1) + j;
                if (child < heap_size) {
                  const auto task_child = get_heap_elem_off_chip(child);
                  if (!(task_child <= task_max)) {
                    max = child;
                    task_max = task_child;
                  }
                }
              }
              if (max != -1) {
                set_heap_elem_on_chip(i, task_max);
                set_heap_index(task_max.vid, i);
                i = max;
              }
            }

          heapify_down_off_chip:
            for (; !(i < kHeapOnChipSize);) {
#pragma HLS pipeline
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              for (int j = 1; j <= kHeapOffChipWidth; ++j) {
#pragma HLS unroll
                const Vid child = i * kHeapOffChipWidth -
                                  kHeapDiff * (kHeapOffChipWidth - 1) + j;
                if (child < heap_size) {
                  const auto task_child = get_heap_elem_off_chip(child);
                  if (!(task_child <= task_max)) {
                    max = child;
                    task_max = task_child;
                  }
                }
              }
              if (max == -1) break;

              set_heap_elem_off_chip(i, task_max);
              set_heap_index(task_max.vid, i);
              i = max;
            }

            set_heap_elem(i, task_i);
            set_heap_index(task_i.vid, i);
          }
        }
        break;
      }
    }
    resp.queue_size = heap_size;
    queue_resp_q.write(resp);
  }
}

// A VidMux merges two input streams into one.
void VidMux(istream<VidPkt>& in0, istream<VidPkt>& in1, ostream<VidPkt>& out) {
spin:
  for (bool flag = false;; flag = !flag) {
#pragma HLS pipeline II = 1
    VidPkt data;
    if (flag ? in0.try_read(data) : in1.try_read(data)) out.write(data);
  }
}

// A VertexMux merges two input streams into one based on a selection stream.
void VertexMux(istream<bool>& select_q, istream<VertexPkt>& in0,
               istream<VertexPkt>& in1, ostream<VertexPkt>& out) {
  DECL_BUF(bool, select);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    VertexPkt data;
    UPDATE(select, select_q.try_read(select),
           (select ? in1.try_read(data) : in0.try_read(data)) &&
               (out.write(data), true));
  }
}

// A VidDemux routes input streams based on the specified bit in Vid.
void VidDemux(int b, istream<VidPkt>& in, ostream<bool>& select_q,
              ostream<VidPkt>& out0, ostream<VidPkt>& out1) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    VidPkt data;
    if (in.try_read(data)) {
      const bool select =
          ap_uint<sizeof(data.payload) * CHAR_BIT>(data.payload).test(b);
      select ? out1.write(data) : out0.write(data);
      select_q.write(select);
    }
  }
}

// A VertexDemux routes input streams based on the specified bit in PeId.
void VertexDemux(int b, istream<VertexPkt>& in, ostream<VertexPkt>& out0,
                 ostream<VertexPkt>& out1) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    VertexPkt data;
    if (in.try_read(data)) {
      ap_uint<sizeof(data.addr) * CHAR_BIT>(data.addr).test(b)
          ? out1.write(data)
          : out0.write(data);
    }
  }
}

void VertexTagger(int id, istream<VidPkt>& pkt_q, ostream<PeId>& tag_q,
                  ostream<Vid>& addr_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    VidPkt pkt;
    if (pkt_q.try_read(pkt)) {
      CHECK_EQ(pkt.payload % kIntervalCount, id);
      tag_q.write(pkt.addr);
      addr_q.write(pkt.payload / kIntervalCount);
    }
  }
}

void VertexUntagger(istream<PeId>& tag_q, istream<Vertex>& data_q,
                    ostream<packet<PeId, Vertex>>& pkt_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!tag_q.empty() && !data_q.empty()) {
      pkt_q.write({tag_q.read(), data_q.read()});
    }
  }
}

void EdgeReadAddrArbiter(tapa::istreams<Vid, kPeCount>& req_q,
                         tapa::ostreams<PeId, kShardCount>& id_q,
                         tapa::ostreams<Vid, kShardCount>& addr_q) {
  DECL_ARRAY(PeId, id, kShardCount, PeId());
  DECL_ARRAY(bool, id_valid, kShardCount, false);
  DECL_ARRAY(Vid, addr, kShardCount, kNullVid);
  DECL_ARRAY(bool, addr_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      RANGE(pe_sid, kPeCount / kShardCount, {
        const auto pe = pe_sid * kShardCount + sid;
        if (!id_valid[sid] &&
            SET(addr_valid[sid], req_q[pe].try_read(addr[sid]))) {
          id[sid] = pe;
          id_valid[sid] = true;
        }
      });
    });

    RANGE(sid, kShardCount, {
      UNUSED RESET(id_valid[sid], id_q[sid].try_write(id[sid]));
      UNUSED RESET(addr_valid[sid], addr_q[sid].try_write(addr[sid]));
    });
  }
}

void EdgeReadDataArbiter(tapa::istreams<PeId, kShardCount>& id_q,
                         tapa::istreams<Edge, kShardCount>& data_in_q,
                         tapa::ostreams<Edge, kPeCount>& data_out_q) {
  DECL_ARRAY(PeId, id, kShardCount, PeId());
  DECL_ARRAY(bool, id_valid, kShardCount, false);
  DECL_ARRAY(Edge, data, kShardCount, Edge());
  DECL_ARRAY(bool, data_valid, kShardCount, false);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      UNUSED SET(id_valid[sid], id_q[sid].try_read(id[sid]));
      UNUSED SET(data_valid[sid], data_in_q[sid].try_read(data[sid]));
      const auto id_sid = id[sid] / kShardCount * kShardCount + sid;
      if (id_valid[sid] && data_valid[sid] &&
          data_out_q[id_sid].try_write(data[sid])) {
        id_valid[sid] = data_valid[sid] = false;
      }
    });
  }
}

void EdgeMem(tapa::istream<Vid>& read_addr_q, tapa::ostream<Edge>& read_data_q,
             tapa::async_mmap<Edge> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void VertexMem(tapa::istream<Vid>& read_addr_q,
               tapa::ostream<Vertex>& read_data_q,
               tapa::async_mmap<Vertex> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void ProcElemS0(
    // Task requests.
    tapa::istream<Vid>& task_req_q,
    // Stage #0 of task data.
    tapa::ostream<Edge>& task_s0_q,
    // Memory-maps.
    tapa::ostream<Vid>& edges_read_addr_q,
    tapa::istream<Edge>& edges_read_data_q) {
  DECL_BUF(Edge, edge);

spin:
  for (;;) {
    const auto src = task_req_q.read();
    const Index index = (edges_read_addr_q.write(src / kShardCount), ap_wait(),
                         bit_cast<Index>(edges_read_data_q.read()));
    task_s0_q.write({.dst = src, .weight = bit_cast<float>(index.count)});
    VLOG_F(9, strt) << "shard[" << src % kShardCount << "]: src: " << src
                    << " #dst: " << index.count;
  read_edges:
    for (Eid eid_req = 0, eid_resp = 0; eid_resp < index.count;) {
      if (eid_req < index.count && eid_req < eid_resp + kMemLatency &&
          edges_read_addr_q.try_write(index.offset + eid_req)) {
        ++eid_req;
      }

      if (UPDATE(edge, edges_read_data_q.try_read(edge),
                 task_s0_q.try_write(edge))) {
        VLOG_F(9, recv) << "shard[" << src % kShardCount << "]: src: " << src
                        << " #dst: " << index.count << " #req: " << eid_req
                        << " #resp: " << eid_resp;
        ++eid_resp;
      }
    }
    VLOG_F(9, done) << "shard[" << src % kShardCount << "]: src: " << src
                    << " #dst: " << index.count;
  }
}

void ProcElemS1(
    // PE id.
    PeId id,
    // Task data.
    tapa::istream<Edge>& task_s0_q, tapa::ostream<Update>& task_s1p0_q,
    tapa::ostream<float>& task_s1p1_q,
    // Memory-maps.
    ostream<VidPkt>& read_addr_q, istream<VertexPkt>& read_data_q) {
spin:
  for (;;) {
    Edge task_s0;
    if (task_s0_q.try_read(task_s0)) {
      const auto src = task_s0.dst;
      const auto count = bit_cast<Vid>(task_s0.weight);
      read_addr_q.write({id, src});
      ap_wait();
      const auto vertex_pkt = read_data_q.read();
      CHECK_EQ(vertex_pkt.addr, id);
      const auto src_distance = vertex_pkt.payload.distance;
      task_s1p0_q.write({
          .vid = src,
          .distance = src_distance,
          .count = count,
      });

      DECL_BUF(Edge, task_s0);
      DECL_BUF(VertexPkt, vertex);
      bool task_s1p0_written = false;
      bool read_addr_written = false;
      VLOG_F(9, strt) << "shard[" << src % kShardCount << "]: src: " << src
                      << " #dst: " << count;

    fwd:
      for (Vid i_req = 0, i_resp = 0; i_resp < count;) {
#pragma HLS pipeline II = 1
        if (i_req < count && i_req < i_resp + kMemLatency) {
          SET(task_s0_valid, task_s0_q.try_read(task_s0));
          if (task_s0_valid) {
            SET(task_s1p0_written,
                task_s1p0_q.try_write(
                    {.vid = task_s0.dst, .distance = task_s0.weight}));
            SET(read_addr_written, read_addr_q.try_write({id, task_s0.dst}));
          }
          if (task_s1p0_written && read_addr_written) {
            task_s0_valid = false;
            task_s1p0_written = false;
            read_addr_written = false;
            ++i_req;
          }
        }

        if (UPDATE(vertex, read_data_q.try_read(vertex),
                   task_s1p1_q.try_write(vertex.payload.distance))) {
          VLOG_F(9, recv) << "shard[" << src % kShardCount << "]: src: " << src
                          << " #dst: " << count << " #req: " << i_req
                          << " #resp: " << i_resp;
          ++i_resp;
        }
      }
      VLOG_F(9, done) << "shard[" << src % kShardCount << "]: src: " << src
                      << " #dst: " << count;
    }
  }
}

void ProcElemS2(
    // Task data.
    tapa::istream<Update>& task_s1p0_q, tapa::istream<float>& task_s1p1_q,
    tapa::ostream<TaskOp>& task_resp_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    const auto task_s1p0 = task_s1p0_q.read();
    const auto src = task_s1p0.vid;
    const auto src_distance = task_s1p0.distance;
    const auto count = task_s1p0.count;

  gen_updates:
    for (Vid i = 0; i < count; ++i) {
      const auto task_s1p0 = task_s1p0_q.read();
      const auto dst = task_s1p0.vid;
      const auto dst_distance = task_s1p1_q.read();
      const auto weight = task_s1p0.distance;

      const auto new_distance = src_distance + weight;
      if (new_distance < dst_distance) {
        VLOG_F(9, info) << "distances[" << dst << "] = " << dst_distance
                        << " -> distances[" << src << "] + " << weight << " = "
                        << new_distance;
        task_resp_q.write({
            .op = TaskOp::NEW,
            .task =
                {
                    .vid = dst,
                    .vertex =
                        {
                            .parent = src,
                            .distance = new_distance,
                        },
                },
        });
      }
    }
    task_resp_q.write({
        .op = TaskOp::DONE,
        .task = {.vid = count, .vertex = {.parent = src}},
    });
  }
}

void QueueReqArbiter(tapa::istream<QueueOp>& req_in_q,
                     tapa::ostreams<QueueOp, kQueueCount>& req_out_q) {
  static_assert(is_power_of(kQueueCount, 2), "invalid queue count");
  DECL_BUF(QueueOp, req);
  DECL_ARRAY(ap_uint<bit_length(kQueueCount / kShardCount)>, sid_base,
             kShardCount, 0);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UNUSED SET(req_valid, req_in_q.try_read(req));
    if (req_valid) {
      switch (req.op) {
        case QueueOp::PUSH:
          UNUSED RESET(req_valid,
                       req_out_q[req.task.vid % kQueueCount].try_write(req));
          break;
        case QueueOp::POP:
          CHECK_LT(req.task.vid, kShardCount);
          const auto qid =
              (sid_base[req.task.vid] * kShardCount + req.task.vid) %
              kQueueCount;
          if (req_out_q[qid].try_write(req)) {
            req_valid = false;
            ++sid_base[req.task.vid];
          }
          break;
      }
    }
  }
}

void QueueRespArbiter(tapa::istreams<QueueOpResp, kQueueCount>& resp_in_q,
                      tapa::ostream<QueueOpResp>& resp_out_q) {
  DECL_BUF(QueueOpResp, resp);
  DECL_ARRAY(int32_t, queue_size, kQueueCount, 0);
  int32_t total_queue_size = 0;
spin:
  for (uint8_t q = 0;; ++q) {
#pragma HLS pipeline II = 1
    if (SET(resp_valid, resp_in_q[q % kQueueCount].try_read(resp))) {
      total_queue_size += resp.queue_size - queue_size[q % kQueueCount];
      queue_size[q % kQueueCount] = resp.queue_size;
      resp.queue_size = total_queue_size;
    }
    RESET(resp_valid, resp_out_q.try_write(resp));
  }
}

constexpr int kTaskCapPerPe = 2;
constexpr int kTaskCapPerShard = kPeCount / kShardCount * kTaskCapPerPe;

void Dispatcher(
    // Scalar.
    const Vid root,
    // Metadata.
    tapa::mmap<int64_t> metadata,
    // Task and queue requests.
    tapa::ostreams<Vid, kPeCount>& task_req_q,
    tapa::istreams<TaskOp, kPeCount>& task_resp_q,
    tapa::ostream<QueueOp>& queue_req_q,
    tapa::istream<QueueOpResp>& queue_resp_q) {
  // Process finished tasks.
  bool task_buf_valid = false;
  TaskOp task_buf;
  bool queue_buf_valid = false;
  QueueOpResp queue_buf;

  int32_t task_count = 1;  // Number of active tasks.
  int32_t queue_size = 0;  // Number of tasks in the queue.

  // Number of POP requests sent but not acknowledged.
  DECL_ARRAY(int32_t, pop_count, kShardCount, 0);

  DECL_ARRAY(int32_t, task_count_per_shard, kShardCount, 0);

  DECL_ARRAY(bool, pe_active, kPeCount, false);

  task_req_q[root % kShardCount].write(root);
  ++task_count_per_shard[root % kShardCount];
  pe_active[root % kShardCount] = true;

  // Statistics.
  int32_t visited_vertex_count = 0;
  int32_t visited_edge_count = 0;
  int32_t queue_count = 0;
  int64_t total_queue_size = 0;
  int32_t max_queue_size = 0;
  int64_t total_task_count = 0;
  int64_t cycle_count = 0;
  int64_t queue_full_cycle_count = 0;
  int64_t pe_fullcycle_count = 0;
  DECL_ARRAY(int64_t, pe_active_count, kPeCount, 0);

  // Format log messages.
#define STATS(tag, content)                                                    \
  do {                                                                         \
    VLOG_F(9, tag) << content " | " << std::setfill(' ') << std::setw(1)       \
                   << task_count << " active + " << std::setw(2) << queue_size \
                   << " pending tasks";                                        \
    CHECK_GE(queue_size, 0);                                                   \
    CHECK_GE(task_count, 0);                                                   \
  } while (0)

spin:
  for (; queue_size != 0 || task_count != 0 || Any(pop_count); ++cycle_count) {
#pragma HLS pipeline II = 1
    RANGE(pe, kPeCount, pe_active[pe] && ++pe_active_count[pe]);
    total_task_count += task_count;
    const auto pe = cycle_count % kPeCount;
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          if (queue_buf.task_op == TaskOp::NOOP) {
            // PUSH request updated priority of existing tasks.
            --queue_size;
            STATS(recv, "QUEUE: DECR");
          }

          // Update statistics.
          ++queue_count;
          total_queue_size += queue_size;
          if (queue_size > max_queue_size) max_queue_size = queue_size;
          break;
        case QueueOp::POP:
          --pop_count[queue_buf.task.vid % kShardCount];
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            ++task_count;
            ++task_count_per_shard[queue_buf.task.vid % kShardCount];
            --queue_size;
            STATS(recv, "QUEUE: NEW ");
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            STATS(recv, "QUEUE: NOOP");
          }
          break;
      }
    }

    const Vid sid = cycle_count % kShardCount;
    if (task_buf_valid) {
      // Enqueue tasks generated from PEs.
      if (queue_req_q.try_write({.op = QueueOp::PUSH, .task = task_buf.task})) {
        task_buf_valid = false;
        STATS(send, "QUEUE: PUSH");
      } else {
        ++queue_full_cycle_count;
      }
    } else if (task_count_per_shard[sid] + pop_count[sid] < kTaskCapPerShard &&
               queue_size != 0) {
      // Dequeue tasks from the queue.
      if (queue_req_q.try_write({.op = QueueOp::POP, .task = {.vid = sid}})) {
        ++pop_count[sid];
        STATS(send, "QUEUE: POP ");
      } else {
        ++queue_full_cycle_count;
      }
    }

    // Assign tasks to PEs.
    const auto pe_req =
        (cycle_count * kShardCount + queue_buf.task.vid % kShardCount) %
        kPeCount;
    if (RESET(queue_buf_valid,
              task_req_q[pe_req].try_write(queue_buf.task.vid))) {
      pe_active[pe_req] = true;
    } else if (queue_buf_valid) {
      ++pe_fullcycle_count;
    }

    // Receive tasks generated from PEs.
    if (SET(task_buf_valid, task_resp_q[pe].try_read(task_buf))) {
      if (task_buf.op == TaskOp::DONE) {
        pe_active[pe] = false;
        task_buf_valid = false;
        --task_count;
        --task_count_per_shard[task_buf.task.vertex.parent % kShardCount];

        // Update statistics.
        ++visited_vertex_count;
        visited_edge_count += task_buf.task.vid;
        STATS(recv, "TASK : DONE");
      } else {
        ++queue_size;
        STATS(recv, "TASK : NEW ");
      }
    }
  }

  RANGE(sid, kShardCount, {
    CHECK_EQ(pop_count[sid], 0);
    CHECK_EQ(task_count_per_shard[sid], 0);
  });
  RANGE(pe, kPeCount, CHECK_EQ(pe_active[pe], false));

  metadata[0] = visited_edge_count;
  metadata[1] = total_queue_size;
  metadata[2] = queue_count;
  metadata[3] = max_queue_size;
  metadata[4] = visited_vertex_count;
  metadata[5] = total_task_count;
  metadata[6] = cycle_count;
  metadata[7] = queue_full_cycle_count;
  metadata[8] = pe_fullcycle_count;

meta:
  for (int pe = 0; pe < kPeCount; ++pe) {
#pragma HLS pipeline II = 1
    metadata[9 + pe] = pe_active_count[pe];
  }
}

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::async_mmaps<Edge, kShardCount> edges,
          tapa::async_mmaps<Vertex, kIntervalCount> vertices,
          // For queues.
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  tapa::stream<QueueOp, 256> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOp, kQueueCount, 2> queue_req_qi("queue_req_i");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  streams<Vid, kPeCount, kPeCount / kShardCount * 4> task_req_q("task_req");
  tapa::streams<Edge, kPeCount, 2> task_s0_q("task_stage0");
  tapa::streams<Update, kPeCount, 64> task_s1p0_q("task_stage1_part0");
  tapa::streams<float, kPeCount, 2> task_s1p1_q("task_stage1_part1");
  tapa::streams<TaskOp, kPeCount, 256> task_resp_q("task_resp");

  // For edges.
  tapa::streams<Vid, kPeCount, 2> edge_read_addr_q("edge_read_addr");
  tapa::streams<Edge, kPeCount, 2> edge_read_data_q("edge_read_data");
  tapa::streams<Vid, kShardCount, 2> edge_read_addr_qi("edges.read_addr");
  tapa::streams<Edge, kShardCount, 2> edge_read_data_qi("edges.read_data");
  tapa::streams<PeId, kShardCount, 64> edge_pe_qi("edges.pe");

  // For vertices.
  //   Connect PEs to the addr network.
  streams<VidPkt, kPeCount, 2> vertex_read_addr_q;
  streams<VidPkt, kPeCount / 2, 8> vertex_read_addr_qr1;
  streams<VidPkt, kPeCount / 4, 8> vertex_read_addr_qr2;
  //   Compose the addr network.
  streams<VidPkt, kIntervalCount, 8> vertex_read_addr_qi1;
  streams<VidPkt, kIntervalCount, 8> vertex_read_addr_0_qi0;
  streams<VidPkt, kIntervalCount, 8> vertex_read_addr_1_qi0;
  streams<bool, kIntervalCount, 64> vertex_select_qi0;
  streams<VidPkt, kIntervalCount, 8> vertex_read_addr_qi0;
  //   Connects the memory.
  streams<Vid, kIntervalCount, 2> vertex_read_addr_qi;
  streams<PeId, kIntervalCount, 64> vertex_tag_qi;
  streams<Vertex, kIntervalCount, 2> vertex_read_data_qi;
  //   Compose the data network.
  streams<VertexPkt, kIntervalCount, 8> vertex_read_data_qi0;
  streams<VertexPkt, kIntervalCount, 8> vertex_read_data_0_qi0;
  streams<VertexPkt, kIntervalCount, 8> vertex_read_data_1_qi0;
  streams<VertexPkt, kIntervalCount, 8> vertex_read_data_qi1;
  //   Connect the data network to PEs.
  streams<VertexPkt, kPeCount / 4, 8> vertex_read_data_qr2;
  streams<VertexPkt, kPeCount / 2, 8> vertex_read_data_qr1;
  streams<VertexPkt, kPeCount, 2> vertex_read_data_q;

  tapa::task()
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q)
      .invoke<-1, kQueueCount>(TaskQueue, tapa::seq(), queue_req_qi,
                               queue_resp_qi, vertices, heap_array, heap_index)
      .invoke<-1>(QueueReqArbiter, queue_req_q, queue_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)

      // For edges.
      .invoke<-1, kShardCount>(EdgeMem, edge_read_addr_qi, edge_read_data_qi,
                               edges)
      .invoke<-1>(EdgeReadAddrArbiter, edge_read_addr_q, edge_pe_qi,
                  edge_read_addr_qi)
      .invoke<-1>(EdgeReadDataArbiter, edge_pe_qi, edge_read_data_qi,
                  edge_read_data_q)

      // For vertices.
      // clang-format off
      .invoke<detach>(VidMux, vertex_read_addr_q[0], vertex_read_addr_q[ 8], vertex_read_addr_qr1[0])
      .invoke<detach>(VidMux, vertex_read_addr_q[1], vertex_read_addr_q[ 9], vertex_read_addr_qr1[1])
      .invoke<detach>(VidMux, vertex_read_addr_q[2], vertex_read_addr_q[10], vertex_read_addr_qr1[2])
      .invoke<detach>(VidMux, vertex_read_addr_q[3], vertex_read_addr_q[11], vertex_read_addr_qr1[3])
      .invoke<detach>(VidMux, vertex_read_addr_q[4], vertex_read_addr_q[12], vertex_read_addr_qr1[4])
      .invoke<detach>(VidMux, vertex_read_addr_q[5], vertex_read_addr_q[13], vertex_read_addr_qr1[5])
      .invoke<detach>(VidMux, vertex_read_addr_q[6], vertex_read_addr_q[14], vertex_read_addr_qr1[6])
      .invoke<detach>(VidMux, vertex_read_addr_q[7], vertex_read_addr_q[15], vertex_read_addr_qr1[7])
      .invoke<detach>(VidMux, vertex_read_addr_qr1[0], vertex_read_addr_qr1[4], vertex_read_addr_qr2[0])
      .invoke<detach>(VidMux, vertex_read_addr_qr1[1], vertex_read_addr_qr1[5], vertex_read_addr_qr2[1])
      .invoke<detach>(VidMux, vertex_read_addr_qr1[2], vertex_read_addr_qr1[6], vertex_read_addr_qr2[2])
      .invoke<detach>(VidMux, vertex_read_addr_qr1[3], vertex_read_addr_qr1[7], vertex_read_addr_qr2[3])
      .invoke<detach>(VidMux, vertex_read_addr_qr2[0], vertex_read_addr_qr2[2], vertex_read_addr_qi1[0])
      .invoke<detach>(VidMux, vertex_read_addr_qr2[1], vertex_read_addr_qr2[3], vertex_read_addr_qi1[1])
      .invoke<detach, kIntervalCount>(VidDemux, 0, vertex_read_addr_qi1, vertex_select_qi0, vertex_read_addr_0_qi0, vertex_read_addr_1_qi0)
      .invoke<detach>(VidMux, vertex_read_addr_0_qi0[0], vertex_read_addr_0_qi0[1], vertex_read_addr_qi0[0])
      .invoke<detach>(VidMux, vertex_read_addr_1_qi0[0], vertex_read_addr_1_qi0[1], vertex_read_addr_qi0[1])
      // clang-format on
      .invoke<detach, kIntervalCount>(VertexTagger, seq(), vertex_read_addr_qi0,
                                      vertex_tag_qi, vertex_read_addr_qi)
      .invoke<detach, kIntervalCount>(VertexMem, vertex_read_addr_qi,
                                      vertex_read_data_qi, vertices)
      .invoke<detach, kIntervalCount>(VertexUntagger, vertex_tag_qi,
                                      vertex_read_data_qi, vertex_read_data_qi0)
      // clang-format off
      .invoke<detach, kIntervalCount>(VertexDemux, 0, vertex_read_data_qi0, vertex_read_data_0_qi0, vertex_read_data_1_qi0)
      .invoke<detach>(VertexMux, vertex_select_qi0[0], vertex_read_data_0_qi0[0], vertex_read_data_0_qi0[1], vertex_read_data_qi1[0])
      .invoke<detach>(VertexMux, vertex_select_qi0[1], vertex_read_data_1_qi0[0], vertex_read_data_1_qi0[1], vertex_read_data_qi1[1])
      .invoke<detach>(VertexDemux, 1, vertex_read_data_qi1[0], vertex_read_data_qr2[0], vertex_read_data_qr2[2])
      .invoke<detach>(VertexDemux, 1, vertex_read_data_qi1[1], vertex_read_data_qr2[1], vertex_read_data_qr2[3])
      .invoke<detach>(VertexDemux, 2, vertex_read_data_qr2[0], vertex_read_data_qr1[0], vertex_read_data_qr1[4])
      .invoke<detach>(VertexDemux, 2, vertex_read_data_qr2[1], vertex_read_data_qr1[1], vertex_read_data_qr1[5])
      .invoke<detach>(VertexDemux, 2, vertex_read_data_qr2[2], vertex_read_data_qr1[2], vertex_read_data_qr1[6])
      .invoke<detach>(VertexDemux, 2, vertex_read_data_qr2[3], vertex_read_data_qr1[3], vertex_read_data_qr1[7])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[0], vertex_read_data_q[0], vertex_read_data_q[ 8])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[1], vertex_read_data_q[1], vertex_read_data_q[ 9])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[2], vertex_read_data_q[2], vertex_read_data_q[10])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[3], vertex_read_data_q[3], vertex_read_data_q[11])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[4], vertex_read_data_q[4], vertex_read_data_q[12])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[5], vertex_read_data_q[5], vertex_read_data_q[13])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[6], vertex_read_data_q[6], vertex_read_data_q[14])
      .invoke<detach>(VertexDemux, 3, vertex_read_data_qr1[7], vertex_read_data_q[7], vertex_read_data_q[15])
      // clang-format on

      // PEs.
      .invoke<-1, kPeCount>(ProcElemS0, task_req_q, task_s0_q, edge_read_addr_q,
                            edge_read_data_q)
      .invoke<detach, kPeCount>(ProcElemS1, seq(), task_s0_q, task_s1p0_q,
                                task_s1p1_q, vertex_read_addr_q,
                                vertex_read_data_q)
      .invoke<-1, kPeCount>(ProcElemS2, task_s1p0_q, task_s1p1_q, task_resp_q);
}

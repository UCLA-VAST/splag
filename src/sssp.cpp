#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

using tapa::detach;
using tapa::istream;
using tapa::mmap;
using tapa::ostream;
using tapa::packet;
using tapa::seq;
using tapa::stream;
using tapa::streams;
using tapa::task;

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
    tapa::ostream<QueueOpResp>& queue_resp_q, tapa::mmap<Task> heap_array,
    tapa::mmap<Vid> heap_index) {
#pragma HLS inline recursive
  // Heap rule: child <= parent
  Vid heap_size = 0;

  constexpr int kIndexCacheSize = 4096 * 4;
  tapa::packet<Vid, Vid> heap_index_cache[kIndexCacheSize];
#pragma HLS resource variable = heap_index_cache core = RAM_2P_URAM latency = 4
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
#pragma HLS resource variable = heap_array_cache core = RAM_2P_URAM latency = 2
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
    };
    switch (req.op) {
      case QueueOp::PUSH: {
        const auto new_task = Task(req.task);
        CHECK_EQ(new_task.vid % kQueueCount, qid);
        CHECK_EQ(new_task.vid % kIntervalCount, qid % kIntervalCount);
        const Vid task_index = get_heap_index(new_task.vid);
        bool heapify = true;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const Task old_task = get_heap_elem(task_index);
          CHECK_EQ(old_task.vid, new_task.vid);
          if (new_task <= old_task) {
            heapify = false;
            clear_heap_index(new_task.vid);
          }
        } else {
          heapify_index = heap_size;
          ++heap_size;
          resp.task_op = TaskOp::NEW;
        }

        if (heapify) {
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
        resp.task.set_vid(qid);
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

            auto heapify_down_cmp = [&](Task& task_max, Vid& max) {
              heapify_down_cmp:
                for (int j = 1; j <= kHeapOffChipWidth; ++j) {
#pragma HLS pipeline
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
            };

            if (!(i < kHeapOnChipBound) && i < kHeapOnChipSize) {
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              heapify_down_cmp(task_max, max);

              if (max != -1) {
                set_heap_elem_on_chip(i, task_max);
                set_heap_index(task_max.vid, i);
                i = max;
              }
            }

          heapify_down_off_chip:
            for (; !(i < kHeapOnChipSize);) {
              ++heapify_down_off_chip;

              Vid max = -1;
              Task task_max = task_i;
              heapify_down_cmp(task_max, max);
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
    queue_resp_q.write(resp);
  }
}

// A VidMux merges two input streams into one.
void VidMux(istream<Update>& in0, istream<Update>& in1, ostream<Update>& out) {
spin:
  for (bool flag = false;; flag = !flag) {
#pragma HLS pipeline II = 1
    Update data;
    if (flag ? in0.try_read(data) : in1.try_read(data)) out.write(data);
  }
}

// A UpdateMux merges two input streams into one based on a selection stream.
void UpdateMux(istream<bool>& select_q, istream<Update>& in0,
               istream<Update>& in1, ostream<Update>& out) {
  DECL_BUF(bool, select);
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    Update data;
    UPDATE(select, select_q.try_read(select),
           (select ? in1.try_read(data) : in0.try_read(data)) &&
               (out.write(data), true));
  }
}

// A VidDemux routes input streams based on the specified bit in Vid.
void VidDemux(int b, istream<Update>& in, ostream<bool>& select_q,
              ostream<Update>& out0, ostream<Update>& out1) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    Update data;
    if (in.try_read(data)) {
      const auto addr = data.payload.task.vid();
      const bool select = ap_uint<sizeof(addr) * CHAR_BIT>(addr).test(b);
      select ? out1.write(data) : out0.write(data);
      select_q.write(select);
    }
  }
}

// A UpdateDemux routes input streams based on the specified bit in PeId.
void UpdateDemux(int b, istream<Update>& in, ostream<Update>& out0,
                 ostream<Update>& out1) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    Update data;
    if (in.try_read(data)) {
      ap_uint<sizeof(data.addr) * CHAR_BIT>(data.addr).test(b)
          ? out1.write(data)
          : out0.write(data);
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
  for (ap_uint<bit_length(kPeCount / kShardCount) + 2> pe_sid = 0;; ++pe_sid) {
#pragma HLS pipeline II = 1
    RANGE(sid, kShardCount, {
      const auto pe = pe_sid / 8 * kShardCount + sid;
      if (!id_valid[sid] &&
          SET(addr_valid[sid], req_q[pe].try_read(addr[sid]))) {
        id[sid] = pe;
        id_valid[sid] = true;
      }
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
      RANGE(pe_sid, kPeCount / kShardCount, {
        const auto pe = pe_sid * kShardCount + sid;
        if (id_valid[sid] && data_valid[sid] && id[sid] == pe &&
            data_out_q[pe].try_write(data[sid])) {
          id_valid[sid] = data_valid[sid] = false;
        }
      });
    });
  }
}

void UpdateReqArbiter(tapa::istreams<Update, kPeCount>& in_q,
                      tapa::ostreams<Update, kIntervalCount>& out_q) {
  DECL_ARRAY(Update, update, kIntervalCount, Update());
  DECL_ARRAY(bool, update_valid, kIntervalCount, false);

spin:
  for (ap_uint<bit_length(kPeCount / kIntervalCount - 1)> pe_iid = 0;;
       ++pe_iid) {
#pragma HLS pipeline II = 1
    RANGE(iid, kIntervalCount, {
      const auto pe = pe_iid * kIntervalCount + iid;
      UNUSED SET(update_valid[iid], in_q[pe].try_read(update[iid]));
      UNUSED RESET(update_valid[iid], out_q[iid].try_write(update[iid]));
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

void ProcElemS0(istream<TaskOnChip>& task_in_q, ostream<TaskOnChip>& task_out_q,
                ostream<Vid>& edges_read_addr_q) {
spin:
  for (;;) {
    const auto task = task_in_q.read();
    task_out_q.write(task);

  read_edges:
    for (Eid i = 0; i < task.vertex().degree; ++i) {
#pragma HLS pipeline II = 1
      edges_read_addr_q.write(task.vertex().offset + i);
    }
  }
}

void ProcElemS1(PeId id, istream<TaskOnChip>& task_in_q,
                istream<Edge>& edges_read_data_q,
                ostream<Update>& update_out_q) {
spin:
  for (;;) {
    const auto task = task_in_q.read();

  fwd:
    for (Vid i = 0; i < task.vertex().degree;) {
#pragma HLS pipeline II = 1
      Edge edge;
      if (edges_read_data_q.try_read(edge)) {
        update_out_q.write({
            .addr = id,
            .payload =
                {
                    .op = TaskOp::NEW,
                    .task =
                        Task{
                            .vid = edge.dst,
                            .vertex =
                                {
                                    .parent = task.vid(),
                                    .distance =
                                        task.vertex().distance + edge.weight,
                                },
                        },
                },
        });
        ++i;
      }
    }

    update_out_q.write({
        .addr = id,
        .payload = {.op = TaskOp::DONE, .task = task},
    });
  }
}

void VertexReaderS0(
    // Input.
    istream<Update>& update_in_q,
    // Outputs.
    ostream<Update>& update_out_q, ostream<Vid>& vertex_read_addr_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!update_in_q.empty()) {
      const auto req = update_in_q.read(nullptr);
      update_out_q.write(req);
      vertex_read_addr_q.write(req.payload.task.vid() / kIntervalCount);
    }
  }
}

void VertexReaderS1(
    // Inputs.
    istream<Update>& update_in_q, istream<Vertex>& vertex_read_data_q,
    // Outputs.
    ostream<Update>& new_q, ostream<Update>& noop_q, ostream<bool>& op_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!update_in_q.empty() && !vertex_read_data_q.empty()) {
      auto req = update_in_q.read(nullptr);
      const auto vertex = vertex_read_data_q.read(nullptr);
      switch (req.payload.op) {
        case TaskOp::NEW:
          if (vertex <= req.payload.task.vertex()) {
            req.payload.op = TaskOp::NOOP;
          }
          break;
        case TaskOp::NOOP:
        case TaskOp::DONE:
          CHECK_EQ(req.payload.op, TaskOp::DONE);
          break;
      }

      switch (req.payload.op) {
        case TaskOp::NEW:
          new_q.write(req);
          break;
        case TaskOp::NOOP:
        case TaskOp::DONE:
          noop_q.write(req);
          break;
      }
      op_q.write(req.payload.op != TaskOp::NEW);
    }
  }
}

void VertexUpdater(istream<Update>& pkt_in_q, ostream<Update>& pkt_out_q,
                   mmap<Vertex> vertices) {
spin:
  for (;;) {
    if (!pkt_in_q.empty()) {
      auto pkt = pkt_in_q.read(nullptr);
      CHECK_EQ(pkt.payload.op, TaskOp::NEW);
      const auto addr = pkt.payload.task.vid() / kIntervalCount;
      const auto vertex = vertices[addr];
      if (vertex <= pkt.payload.task.vertex()) {
        pkt.payload.op = TaskOp::NOOP;
      } else {
        pkt.payload.op = TaskOp::NEW;  // Necessasry due to bug in Vivado HLS.
        pkt.payload.task.set_offset(vertex.offset);
        pkt.payload.task.set_degree(vertex.degree);
        vertices[addr] = pkt.payload.task.vertex();
      }
      pkt_out_q.write(pkt);
    }
  }
}

void UpdateFilter(istream<Update>& pkt_in_q, ostream<Update>& pkt_out_q) {
spin:
  for (;;) {
    if (!pkt_in_q.empty()) {
      const auto pkt = pkt_in_q.read(nullptr);
      if (pkt.payload.op != TaskOp::NOOP) {
        pkt_out_q.write(pkt);
      }
    }
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
                       req_out_q[req.task.vid() % kQueueCount].try_write(req));
          break;
        case QueueOp::POP:
          CHECK_LT(req.task.vid(), kShardCount);
          const auto qid =
              (sid_base[req.task.vid()] * kShardCount + req.task.vid()) %
              kQueueCount;
          if (req_out_q[qid].try_write(req)) {
            req_valid = false;
            ++sid_base[req.task.vid()];
          }
          break;
      }
    }
  }
}

void QueueRespArbiter(tapa::istreams<QueueOpResp, kQueueCount>& resp_in_q,
                      tapa::ostream<QueueOpResp>& resp_out_q) {
  DECL_BUF(QueueOpResp, resp);
spin:
  for (uint8_t q = 0;; ++q) {
#pragma HLS pipeline II = 1
    UPDATE(resp, resp_in_q[q % kQueueCount].try_read(resp),
           resp_out_q.try_write(resp));
  }
}

using uint_pe_t = ap_uint<bit_length(kPeCount)>;
using uint_pe_per_shart_t = ap_uint<bit_length(kPeCount / kShardCount)>;

void Dispatcher(
    // Scalar.
    const Task root,
    // Metadata.
    tapa::mmap<int64_t> metadata,
    // Task and queue requests.
    tapa::ostreams<TaskOnChip, kPeCount>& task_req_q,
    tapa::istreams<Update, kIntervalCount>& task_resp_q,
    tapa::ostream<QueueOp>& queue_req_q,
    tapa::istream<QueueOpResp>& queue_resp_q) {
  // Process finished tasks.
  bool task_buf_valid = false;
  Update task_buf;
  bool queue_buf_valid = false;
  QueueOpResp queue_buf;

  // Number of tasks whose parent task is sent to the PEs but not yet
  // acknowledged.
  int32_t active_task_count = root.vertex.degree + 1;

  // Number of tasks generated by the PEs but not yet sent to the PEs.
  int32_t pending_task_count = 0;

  // Number of POP requests sent but not acknowledged.
  DECL_ARRAY(uint_pe_per_shart_t, pop_count, kShardCount, 0);

  // Number of PUSH requests yet to send or not acknowledged.
  uint_pe_t push_count = 0;

  DECL_ARRAY(bool, queue_empty, kQueueCount, true);

  DECL_ARRAY(uint_pe_per_shart_t, task_count_per_shard, kShardCount, 0);
  DECL_ARRAY(ap_uint<1>, task_count_per_pe, kPeCount, 0);

  DECL_ARRAY(uint_pe_per_shart_t, pe_base_per_shard, kShardCount, 0);

  task_req_q[root.vid % kShardCount].write(root);
  ++task_count_per_shard[root.vid % kShardCount];
  ++task_count_per_pe[root.vid % kShardCount];

  // Statistics.
  int32_t visited_vertex_count = 1;
  int32_t visited_edge_count = 0;
  int32_t queue_count = 0;
  int64_t cycle_count = 0;
  int64_t queue_full_cycle_count = 0;
  int64_t pe_fullcycle_count = 0;
  DECL_ARRAY(int64_t, pe_active_count, kPeCount, 0);

  // Format log messages.
#define STATS(tag, content)                                              \
  do {                                                                   \
    VLOG_F(9, tag) << content " | " << std::setfill(' ') << std::setw(1) \
                   << active_task_count << " active + " << std::setw(2)  \
                   << pending_task_count << " pending tasks";            \
    CHECK_GE(active_task_count, 0);                                      \
    CHECK_GE(pending_task_count, 0);                                     \
  } while (0)

  auto shard_is_done = [&](int sid) {
    bool result = true;
    RANGE(i, kQueueCount / kShardCount,
          result &= queue_empty[i * kShardCount + sid]);
    return result;
  };

spin:
  for (; active_task_count || pending_task_count; ++cycle_count) {
#pragma HLS pipeline II = 1
    RANGE(pe, kPeCount, task_count_per_pe[pe] && ++pe_active_count[pe]);
    const auto pe = cycle_count % kPeCount;
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      queue_empty[queue_buf.task.vid() % kQueueCount] = false;
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          --push_count;
          if (queue_buf.task_op == TaskOp::NOOP) {
            // PUSH request updated priority of existing tasks.
            --pending_task_count;
            CHECK_GT(pending_task_count, 0);
            STATS(recv, "QUEUE: DECR");
          }

          // Update statistics.
          ++queue_count;
          break;
        case QueueOp::POP:
          --pop_count[queue_buf.task.vid() % kShardCount];
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            ++task_count_per_shard[queue_buf.task.vid() % kShardCount];
            STATS(recv, "QUEUE: NEW ");
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            queue_empty[queue_buf.task.vid() % kQueueCount] = true;
            STATS(recv, "QUEUE: NOOP");
          }
          break;
      }
    }

    const Vid sid = cycle_count % kShardCount;
    if (task_buf_valid) {
      // Enqueue tasks generated from PEs.
      if (queue_req_q.try_write({
              .op = QueueOp::PUSH,
              .task = task_buf.payload.task,
          })) {
        task_buf_valid = false;
        STATS(send, "QUEUE: PUSH");
      } else {
        ++queue_full_cycle_count;
      }
    } else if (task_count_per_shard[sid] + pop_count[sid] <
                   kPeCount / kShardCount &&
               !(shard_is_done(sid) && push_count == 0)) {
      // Dequeue tasks from the queue.
      if (queue_req_q.try_write(
              {.op = QueueOp::POP, .task = Task{.vid = sid}})) {
        ++pop_count[sid];
        STATS(send, "QUEUE: POP ");
      } else {
        ++queue_full_cycle_count;
      }
    }

    // Assign tasks to PEs.
    {
      const auto sid = queue_buf.task.vid() % kShardCount;
      const auto pe = (pe_base_per_shard[sid] * kShardCount + sid) % kPeCount;
      if (queue_buf_valid) {
        if (task_req_q[pe].try_write(queue_buf.task)) {
          active_task_count += queue_buf.task.vertex().degree + 1;
          --pending_task_count;
          queue_buf_valid = false;
          ++task_count_per_pe[pe];

          // Statistics.
          ++visited_vertex_count;
        } else {
          ++pe_fullcycle_count;
        }
        ++pe_base_per_shard[sid];
      }
    }

    // Receive tasks generated from PEs.
    if (SET(task_buf_valid,
            task_resp_q[cycle_count % kIntervalCount].try_read(task_buf))) {
      --active_task_count;
      const auto pe = task_buf.addr;
      switch (task_buf.payload.op) {
        case TaskOp::NEW:
          ++push_count;
          STATS(recv, "TASK : NEW ");
          ++pending_task_count;

          // Statistics.
          ++visited_edge_count;
          break;
        case TaskOp::NOOP:
          task_buf_valid = false;

          // Statistics.
          ++visited_edge_count;
          break;
        case TaskOp::DONE:
          task_buf_valid = false;
          --task_count_per_shard[task_buf.payload.task.vid() % kShardCount];
          --task_count_per_pe[pe];

          STATS(recv, "TASK : DONE");
          break;
      }
    }
  }

  RANGE(sid, kShardCount, {
    CHECK_EQ(pop_count[sid], 0);
    CHECK_EQ(task_count_per_shard[sid], 0);
  });
  RANGE(pe, kPeCount, CHECK_EQ(task_count_per_pe[pe], 0));
  CHECK_EQ(active_task_count, 0);
  CHECK_EQ(pending_task_count, 0);

  metadata[0] = visited_edge_count;
  metadata[2] = queue_count;
  metadata[4] = visited_vertex_count;
  metadata[6] = cycle_count;
  metadata[7] = queue_full_cycle_count;
  metadata[8] = pe_fullcycle_count;

meta:
  for (int pe = 0; pe < kPeCount; ++pe) {
#pragma HLS pipeline II = 1
    metadata[9 + pe] = pe_active_count[pe];
  }
}

void SSSP(Vid vertex_count, Task root, tapa::mmap<int64_t> metadata,
          tapa::async_mmaps<Edge, kShardCount> edges,
          tapa::async_mmaps<Vertex, kIntervalCount> vertices,
          // For queues.
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  tapa::stream<QueueOp, 256> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOp, kQueueCount, 2> queue_req_qi("queue_req_i");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  streams<TaskOnChip, kPeCount, 2> task_req_q("task_req");
  streams<TaskOnChip, kPeCount, 2> task_req_qi("task_req_i");
  streams<Update, kIntervalCount, 256> task_resp_q("task_resp");

  // For edges.
  tapa::streams<Vid, kPeCount, 2> edge_read_addr_q("edge_read_addr");
  tapa::streams<Edge, kPeCount, 2> edge_read_data_q("edge_read_data");
  tapa::streams<Vid, kShardCount, 2> edge_read_addr_qi("edges.read_addr");
  tapa::streams<Edge, kShardCount, 2> edge_read_data_qi("edges.read_data");
  tapa::streams<PeId, kShardCount, 256> edge_pe_qi("edges.pe");

  // For vertices.
  //   Connect PEs to the update request network.
  streams<Update, kPeCount, 2> update_req_q;
  //   Compose the update request network.
  streams<Update, kIntervalCount, 8> update_req_qi1;
  streams<Update, kIntervalCount, 8> update_req_0_qi0;
  streams<Update, kIntervalCount, 8> update_req_1_qi0;
  streams<bool, kIntervalCount, 256> update_select_qi0;
  streams<Update, kIntervalCount, 8> update_req_qi0;
  //   Connect the vertex readers and updaters.
  streams<Update, kIntervalCount, 64> update_qi0;
  streams<Update, kIntervalCount, 256> update_new_qi0;
  streams<Update, kIntervalCount, 2> update_new_qi1;
  streams<Update, kIntervalCount, 2> update_noop_qi;
  streams<bool, kIntervalCount, 256> update_op_qi;
  streams<Vid, kIntervalCount, 2> vertex_read_addr_q;
  streams<Vertex, kIntervalCount, 2> vertex_read_data_q;
  //   Compose the update response network.
  streams<Update, kIntervalCount, 8> update_resp_qi0;
  streams<Update, kIntervalCount, 8> update_resp_0_qi0;
  streams<Update, kIntervalCount, 8> update_resp_1_qi0;

  tapa::task()
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q)
      .invoke<-1, kQueueCount>(TaskQueue, tapa::seq(), queue_req_qi,
                               queue_resp_qi, heap_array, heap_index)
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
      .invoke<detach>(UpdateReqArbiter, update_req_q, update_req_qi1)
      .invoke<detach, kIntervalCount>(VidDemux, 0, update_req_qi1, update_select_qi0, update_req_0_qi0, update_req_1_qi0)
      .invoke<detach>(VidMux, update_req_0_qi0[0], update_req_0_qi0[1], update_req_qi0[0])
      .invoke<detach>(VidMux, update_req_1_qi0[0], update_req_1_qi0[1], update_req_qi0[1])
      // clang-format on
      .invoke<detach, kIntervalCount>(VertexMem, vertex_read_addr_q,
                                      vertex_read_data_q, vertices)
      .invoke<detach, kIntervalCount>(VertexReaderS0, update_req_qi0,
                                      update_qi0, vertex_read_addr_q)
      .invoke<detach, kIntervalCount>(VertexReaderS1, update_qi0,
                                      vertex_read_data_q, update_new_qi0,
                                      update_noop_qi, update_op_qi)
      .invoke<detach, kIntervalCount>(VertexUpdater, update_new_qi0,
                                      update_new_qi1, vertices)
      .invoke<detach, kIntervalCount>(UpdateMux, update_op_qi, update_new_qi1,
                                      update_noop_qi, update_resp_qi0)
      // clang-format off
      .invoke<detach, kIntervalCount>(UpdateDemux, 0, update_resp_qi0, update_resp_0_qi0, update_resp_1_qi0)
      .invoke<detach>(UpdateMux, update_select_qi0[0], update_resp_0_qi0[0], update_resp_0_qi0[1], task_resp_q[0])
      .invoke<detach>(UpdateMux, update_select_qi0[1], update_resp_1_qi0[0], update_resp_1_qi0[1], task_resp_q[1])
      // clang-format on

      // PEs.
      .invoke<detach, kPeCount>(ProcElemS0, task_req_q, task_req_qi,
                                edge_read_addr_q)
      .invoke<detach, kPeCount>(ProcElemS1, seq(), task_req_qi,
                                edge_read_data_q, update_req_q);
}

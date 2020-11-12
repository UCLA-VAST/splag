#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

// Estimated DRAM latency.
constexpr int kMemLatency = 50;

constexpr int kQueueCount = 4;

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
    tapa::ostream<QueueOpResp>& queue_resp_q,
    tapa::ostream<tapa::packet<Vid, Vertex>>& write_q,
    tapa::mmap<float> distances, tapa::mmap<Task> heap_array,
    tapa::mmap<Vid> heap_index) {
#pragma HLS inline recursive
  // Parent   of heap_array[i]: heap_array[(i - 1) / 2]
  // Children of heap_array[i]: heap_array[i * 2 + 1], heap_array[i * 2 + 2]
  // Heap rule: child <= parent
  Vid heap_size = 0;

  constexpr int kMaxOnChipSize = 4096 * 4;
  tapa::packet<Vid, Vid> heap_index_cache[kMaxOnChipSize];
#pragma HLS resource variable = heap_index_cache core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_index_cache
  int32_t read_hit = 0;
  int32_t read_miss = 0;
  int32_t write_hit = 0;
  int32_t write_miss = 0;
heap_index_cache_init:
  for (Vid i = 0; i < kMaxOnChipSize; ++i) {
#pragma HLS pipeline II = 1
    heap_index_cache[i].addr = kNullVid;
  }
  auto get_heap_index = [&](Vid vid) -> Vid {
    auto& entry = heap_index_cache[vid / kQueueCount % kMaxOnChipSize];
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
    auto& entry = heap_index_cache[vid / kQueueCount % kMaxOnChipSize];
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
    auto& entry = heap_index_cache[vid / kQueueCount % kMaxOnChipSize];
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
    auto& entry = heap_index_cache[vid / kQueueCount % kMaxOnChipSize];
    ++write_miss;
    if (entry.addr == vid) {
      entry.addr = kNullVid;
    }
    heap_index[vid] = kNullVid;
  };

  Task heap_array_cache[kMaxOnChipSize];
#pragma HLS array_partition variable = heap_array_cache cyclic factor = 2
#pragma HLS resource variable = heap_array_cache core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_array_cache

  auto get_heap_array_index = [&](Vid i) {
    return (i - kMaxOnChipSize) * kQueueCount + qid;
  };
  auto get_heap_elem = [&](Vid i) {
    return i < kMaxOnChipSize ? heap_array_cache[i]
                              : heap_array[get_heap_array_index(i)];
  };
  auto set_heap_elem = [&](Vid i, Task task) {
    if (i < kMaxOnChipSize) {
      heap_array_cache[i] = task;
    } else {
      heap_array[get_heap_array_index(i)] = task;
    }
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
    for (int i = 0; i < kMaxOnChipSize; ++i) {
      CHECK_EQ(heap_index_cache[i].addr, kNullVid) << "i = " << i;
    }
  });

spin:
  for (;;) {
    QueueOp req = queue_req_q.read();
    QueueOpResp resp{
        .queue_op = req.op,
        .task_op = TaskOp::NOOP,
        .task = {},
        .queue_size = 0,
    };
    switch (req.op) {
      case QueueOp::PUSH: {
        const auto new_task = req.task;
        const Vid task_index = get_heap_index(new_task.vid);
        bool heapify = false;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const Task old_task = get_heap_elem(task_index);
          CHECK_EQ(old_task.vid, new_task.vid);
          if (!(new_task <= old_task)) {
            heapify = true;
          }
        } else {
          if (!(bit_cast<uint32_t>(distances[new_task.vid]) <=
                bit_cast<uint32_t>(new_task.vertex.distance))) {
            heapify = true;
            heapify_index = heap_size;
            ++heap_size;
            resp.task_op = TaskOp::NEW;
          } else {
            clear_heap_index(new_task.vid);
          }
        }

        if (heapify) {
          write_q.write({new_task.vid, new_task.vertex});
          distances[new_task.vid] = new_task.vertex.distance;
          // Increase the priority of heap_array[i] if necessary.
          Vid i = heapify_index;
          const Task task_i = new_task;

          ++heapify_up_count;

          auto get_parent = [](Vid i) { return (i - 1) / 2; };
          Vid parent = get_parent(i);
          Task task_parent = get_heap_elem(parent);

        heapify_up_off_chip:
          for (; !(get_parent(parent) < kMaxOnChipSize) &&
                 !(task_i <= task_parent);) {
#pragma HLS pipeline
            ++heapify_up_off_chip;

            CHECK_GE(i, kMaxOnChipSize);
            heap_array[get_heap_array_index(i)] = task_parent;
            set_heap_index(task_parent.vid, i);

            i = parent;
            parent = get_parent(parent);
            CHECK_GE(parent, kMaxOnChipSize);
            task_parent = heap_array[get_heap_array_index(parent)];
          }

        heapify_up_on_chip:
          for (; i != 0 && !(task_i <= task_parent);) {
#pragma HLS pipeline II = 3
            ++heapify_up_on_chip;

            set_heap_elem(i, task_parent);
            set_heap_index(task_parent.vid, i);

            i = parent;
            parent = get_parent(parent);
            CHECK_LT(parent, kMaxOnChipSize);
            task_parent = heap_array_cache[parent];
          }

          set_heap_elem(i, task_i);
          set_heap_index(task_i.vid, i);
        }
        break;
      }
      case QueueOp::POP: {
        if (heap_size != 0) {
          const Task front = heap_array_cache[0];
          clear_heap_index(front.vid);
          --heap_size;

          resp.task_op = TaskOp::NEW;
          resp.task = front;

          if (heap_size != 0) {
            ++heapify_down_count;

            // Find proper index `i` for `task_i`.
            const Task task_i = get_heap_elem(heap_size);
            Vid i = 0;

            auto left_is_valid = [&] { return i * 2 + 1 < heap_size; };
            auto right_is_valid = [&] { return i * 2 + 2 < heap_size; };
            auto left = [&] { return (left_is_valid() ? i : 0) * 2 + 1; };
            auto right = [&] { return (right_is_valid() ? i + 1 : 1) * 2; };
            Task task_left = get_heap_elem(left());
            Task task_right = get_heap_elem(right());
            auto left_is_max = [&] {
              return !right_is_valid() ||
                     (left_is_valid() && task_right <= task_left);
            };
            auto max = [&] { return left_is_max() ? left() : right(); };
            Task task_max = left_is_max() ? task_left : task_right;
            auto not_heapified = [&] {
              return ((left_is_valid() && !(task_left <= task_i)) ||
                      (right_is_valid() && !(task_right <= task_i)));
            };

          heapify_down_on_chip:
            for (; max() * 2 + 2 < kMaxOnChipSize && not_heapified();) {
#pragma HLS pipeline II = 3
              ++heapify_down_on_chip;

              CHECK_LT(i, kMaxOnChipSize);
              heap_array_cache[i] = task_max;
              set_heap_index(task_max.vid, i);

              i = max();
              CHECK_LT(right(), kMaxOnChipSize);
              task_left = heap_array_cache[left()];
              task_right = heap_array_cache[right()];
              task_max = left_is_max() ? task_left : task_right;
            }

          heapify_down_off_chip:
            for (; not_heapified();) {
#pragma HLS pipeline
              ++heapify_down_off_chip;

              set_heap_elem(i, task_max);
              set_heap_index(task_max.vid, i);

              i = max();
              task_left = get_heap_elem(left());
              task_right = get_heap_elem(right());
              task_max = left_is_max() ? task_left : task_right;
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

void ReadAddrArbiter(tapa::istreams<Vid, kPeCount>& req_q,
                     tapa::ostream<PeId>& pe_q, tapa::ostream<Vid>& addr_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    bool done = false;
    RANGE(pe, kPeCount, {
      Vid addr;
      if (!done && req_q[pe].try_read(addr)) {
        done |= true;
        addr_q.write(addr);
        pe_q.write(pe);
      }
    });
  }
}

void EdgeReadDataArbiter(tapa::istream<PeId>& id_q,
                         tapa::istream<Edge>& data_in_q,
                         tapa::ostreams<Edge, kPeCount>& data_out_q) {
  ReadDataArbiter(id_q, data_in_q, data_out_q);
}

void IndexReadDataArbiter(tapa::istream<PeId>& id_q,
                          tapa::istream<Index>& data_in_q,
                          tapa::ostreams<Index, kPeCount>& data_out_q) {
  ReadDataArbiter(id_q, data_in_q, data_out_q);
}

void VertexReadDataArbiter(tapa::istream<PeId>& id_q,
                           tapa::istream<Vertex>& data_in_q,
                           tapa::ostreams<Vertex, kPeCount>& data_out_q) {
  ReadDataArbiter(id_q, data_in_q, data_out_q);
}

void EdgeMem(tapa::istream<Vid>& read_addr_q, tapa::ostream<Edge>& read_data_q,
             tapa::async_mmap<Edge> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void IndexMem(tapa::istream<Vid>& read_addr_q,
              tapa::ostream<Index>& read_data_q, tapa::async_mmap<Index> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
  ReadOnlyMem(read_addr_q, read_data_q, mem);
}

void VertexMem(tapa::istream<Vid>& read_addr_q,
               tapa::ostream<Vertex>& read_data_q,
               tapa::istream<Vid>& write_addr_q,
               tapa::istream<Vertex>& write_data_q,
               tapa::async_mmap<Vertex> mem) {
#pragma HLS data_pack variable = mem.read_data
#pragma HLS data_pack variable = mem.read_peek
#pragma HLS data_pack variable = mem.write_data
  ReadWriteMem(read_addr_q, read_data_q, write_addr_q, write_data_q, mem);
}

void ProcElemS0(
    // Task requests.
    tapa::istream<Vid>& task_req_q,
    // Stage #0 of task data.
    tapa::ostream<Edge>& task_s0_q,
    // Memory-maps.
    tapa::ostream<Vid>& edges_read_addr_q,
    tapa::istream<Edge>& edges_read_data_q,
    tapa::ostream<Vid>& indices_read_addr_q,
    tapa::istream<Index>& indices_read_data_q) {
  DECL_BUF(Edge, edge);

spin:
  for (;;) {
    const auto src = task_req_q.read();
    const Index index =
        (indices_read_addr_q.write(src), ap_wait(), indices_read_data_q.read());
    task_s0_q.write({.dst = src, .weight = bit_cast<float>(index.count)});
  read_edges:
    for (Eid eid_req = 0, eid_resp = 0; eid_resp < index.count;) {
      if (eid_req < index.count && eid_req < eid_resp + kMemLatency &&
          edges_read_addr_q.try_write(index.offset + eid_req)) {
        ++eid_req;
      }

      if (UPDATE(edge, edges_read_data_q.try_read(edge),
                 task_s0_q.try_write(edge))) {
        ++eid_resp;
      }
    }
  }
}

void ProcElemS1(
    // Task data.
    tapa::istream<Edge>& task_s0_q, tapa::ostream<Update>& task_s1p0_q,
    tapa::ostream<float>& task_s1p1_q,
    // Memory-maps.
    tapa::ostream<Vid>& read_addr_q, tapa::istream<Vertex>& read_data_q) {
spin:
  for (;;) {
    Edge task_s0;
    if (task_s0_q.try_read(task_s0)) {
      const auto src = task_s0.dst;
      const auto count = bit_cast<Vid>(task_s0.weight);
      const auto src_distance =
          (read_addr_q.write(src), ap_wait(), read_data_q.read().distance);
      task_s1p0_q.write({
          .vid = src,
          .distance = src_distance,
          .count = count,
      });

      DECL_BUF(Edge, task_s0);
      DECL_BUF(Vertex, vertex);
      bool task_s1p0_written = false;
      bool read_addr_written = false;

    fwd:
      for (Vid i_req = 0, i_resp = 0; i_resp < count;) {
#pragma HLS pipeline II = 1
        if (i_req < count && i_req < i_resp + kMemLatency) {
          SET(task_s0_valid, task_s0_q.try_read(task_s0));
          if (task_s0_valid) {
            SET(task_s1p0_written,
                task_s1p0_q.try_write(
                    {.vid = task_s0.dst, .distance = task_s0.weight}));
            SET(read_addr_written, read_addr_q.try_write(task_s0.dst));
          }
          if (task_s1p0_written && read_addr_written) {
            task_s0_valid = false;
            task_s1p0_written = false;
            read_addr_written = false;
            ++i_req;
          }
        }

        if (UPDATE(vertex, read_data_q.try_read(vertex),
                   task_s1p1_q.try_write(vertex.distance))) {
          ++i_resp;
        }
      }
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
        .task = {.vid = count, .vertex = {}},
    });
  }
}

void QueueReqArbiter(tapa::istream<QueueOp>& req_in_q,
                     tapa::ostreams<QueueOp, kQueueCount>& req_out_q) {
  DECL_BUF(QueueOp, req);
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
          RANGE(q, kQueueCount, req_out_q[q].write(req));
          req_valid = false;
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

void WriteAribter(
    tapa::istreams<tapa::packet<Vid, Vertex>, kQueueCount>& pkt_in_q,
    tapa::ostream<Vid>& addr_out_q, tapa::ostream<Vertex>& data_out_q) {
spin:
  for (uint8_t q = 0;; ++q) {
#pragma HLS pipeline II = 1
    tapa::packet<Vid, Vertex> pkt;
    if (pkt_in_q[q % kQueueCount].try_read(pkt)) {
      addr_out_q.write(pkt.addr);
      data_out_q.write(pkt.payload);
    }
  }
}

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

  task_req_q[0].write(root);

  // Statistics.
  int32_t visited_vertex_count = 0;
  int32_t visited_edge_count = 0;
  int32_t queue_count = 0;
  int64_t total_queue_size = 0;
  int32_t max_queue_size = 0;

  // Format log messages.
#define STATS(leve, tag, content)                                              \
  do {                                                                         \
    VLOG_F(9, tag) << content " | " << std::setfill(' ') << std::setw(1)       \
                   << task_count << " active + " << std::setw(2) << queue_size \
                   << " pending tasks";                                        \
    CHECK_GE(queue_size, 0);                                                   \
    CHECK_GE(task_count, 0);                                                   \
  } while (0)

spin:
  for (uint8_t pe = 0;
       queue_size != 0 || task_count != 0 || !queue_resp_q.empty();
       pe = pe == kPeCount - 1 ? 0 : pe + 1) {
#pragma HLS pipeline II = 1
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      queue_size = queue_buf.queue_size;
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          if (queue_buf.task_op == TaskOp::NOOP) {
            // PUSH request updated priority of existing tasks.
            STATS(9, recv, "QUEUE: DECR");
          }

          // Update statistics.
          ++queue_count;
          total_queue_size += queue_size;
          if (queue_size > max_queue_size) max_queue_size = queue_size;
          break;
        case QueueOp::POP:
          if (queue_buf.task_op == TaskOp::NEW) {
            // POP request returned a new task.
            STATS(9, recv, "QUEUE: NEW ");
          } else {
            // The queue is empty.
            queue_buf_valid = false;
            --task_count;
            STATS(9, recv, "QUEUE: NOOP");
          }
          break;
      }
    } else if (task_count < kPeCount * 2 && queue_size != 0) {
      // Dequeue tasks from the queue.
      if (queue_req_q.try_write({.op = QueueOp::POP, .task = {}})) {
        task_count += kQueueCount;
        STATS(9, send, "QUEUE: POP ");
      }
    } else if (RESET(task_buf_valid,
                     queue_req_q.try_write(
                         {.op = QueueOp::PUSH, .task = task_buf.task}))) {
      // Enqueue tasks generated from PEs.
      ++queue_size;
      STATS(9, send, "QUEUE: PUSH");
    }

    // Assign tasks to PEs.
    UNUSED RESET(queue_buf_valid, task_req_q[pe].try_write(queue_buf.task.vid));

    // Receive tasks generated from PEs.
    if (SET(task_buf_valid, task_resp_q[pe].try_read(task_buf))) {
      if (task_buf.op == TaskOp::DONE) {
        task_buf_valid = false;
        --task_count;

        // Update statistics.
        ++visited_vertex_count;
        visited_edge_count += task_buf.task.vid;
        STATS(9, recv, "TASK : DONE");
      }
    }
  }

  metadata[0] = visited_edge_count;
  metadata[1] = total_queue_size;
  metadata[2] = queue_count;
  metadata[3] = max_queue_size;
  metadata[4] = visited_vertex_count;
}

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::mmap<Edge> edges, tapa::mmap<Index> indices,
          tapa::mmap<Vertex> vertices,
          // For queues.
          tapa::mmap<float> distances_0, tapa::mmap<float> distances_1,
          tapa::mmap<float> distances_2, tapa::mmap<float> distances_3,
          tapa::mmap<Task> heap_array_0, tapa::mmap<Task> heap_array_1,
          tapa::mmap<Task> heap_array_2, tapa::mmap<Task> heap_array_3,
          tapa::mmap<Vid> heap_index_0, tapa::mmap<Vid> heap_index_1,
          tapa::mmap<Vid> heap_index_2, tapa::mmap<Vid> heap_index_3) {
  tapa::stream<QueueOp, 256> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<QueueOp, kQueueCount, 2> queue_req_qi("queue_req_i");
  tapa::streams<QueueOpResp, kQueueCount, 2> queue_resp_qi("queue_resp_i");

  tapa::streams<Vid, kPeCount, 2> task_req_q("task_req");
  tapa::streams<Edge, kPeCount, 2> task_s0_q("task_stage0");
  tapa::streams<Update, kPeCount, 64> task_s1p0_q("task_stage1_part0");
  tapa::streams<float, kPeCount, 2> task_s1p1_q("task_stage1_part1");
  tapa::streams<TaskOp, kPeCount, 256> task_resp_q("task_resp");

  // For edges.
  tapa::streams<Vid, kPeCount, 2> edge_read_addr_q("edge_read_addr");
  tapa::streams<Edge, kPeCount, 2> edge_read_data_q("edge_read_data");
  tapa::stream<Vid, 2> edge_read_addr_qi("edges.read_addr");
  tapa::stream<Edge, 2> edge_read_data_qi("edges.read_data");
  tapa::stream<PeId, 64> edge_pe_qi("edges.pe");

  // For indices.
  tapa::streams<Vid, kPeCount, 2> index_read_addr_q("index_read_addr");
  tapa::streams<Index, kPeCount, 2> index_read_data_q("index_read_data");
  tapa::stream<Vid, 2> index_read_addr_qi("indices.read_addr");
  tapa::stream<Index, 2> index_read_data_qi("indices.read_data");
  tapa::stream<PeId, 64> index_pe_qi("indices.pe");

  // For vertices.
  tapa::streams<Vid, kPeCount, 2> vertex_read_addr_q("vertex_read_addr");
  tapa::streams<Vertex, kPeCount, 2> vertex_read_data_q("vertex_read_data");
  tapa::stream<Vid, 2> vertex_write_addr_q("vertex_write_addr");
  tapa::stream<Vertex, 2> vertex_write_data_q("vertex_write_data");
  tapa::stream<Vid, 2> vertex_read_addr_qi("vertices.read_addr");
  tapa::stream<Vertex, 2> vertex_read_data_qi("vertices.read_data");
  tapa::streams<tapa::packet<Vid, Vertex>, kQueueCount, 2> vertex_write_qi(
      "vertices.write");
  tapa::stream<PeId, 64> vertex_pe_qi("vertices.pe");

  tapa::task()
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q)
      .invoke<-1>(TaskQueue, 0, queue_req_qi[0], queue_resp_qi[0],
                  vertex_write_qi[0], distances_0, heap_array_0, heap_index_0)
      .invoke<-1>(TaskQueue, 1, queue_req_qi[1], queue_resp_qi[1],
                  vertex_write_qi[1], distances_1, heap_array_1, heap_index_1)
      .invoke<-1>(TaskQueue, 2, queue_req_qi[2], queue_resp_qi[2],
                  vertex_write_qi[2], distances_2, heap_array_2, heap_index_2)
      .invoke<-1>(TaskQueue, 3, queue_req_qi[3], queue_resp_qi[3],
                  vertex_write_qi[3], distances_3, heap_array_3, heap_index_3)
      .invoke<-1>(QueueReqArbiter, queue_req_q, queue_req_qi)
      .invoke<-1>(QueueRespArbiter, queue_resp_qi, queue_resp_q)
      .invoke<-1>(WriteAribter, vertex_write_qi, vertex_write_addr_q,
                  vertex_write_data_q)

      // For edges.
      .invoke<-1>(EdgeMem, edge_read_addr_qi, edge_read_data_qi, edges)
      .invoke<-1>(ReadAddrArbiter, edge_read_addr_q, edge_pe_qi,
                  edge_read_addr_qi)
      .invoke<-1>(EdgeReadDataArbiter, edge_pe_qi, edge_read_data_qi,
                  edge_read_data_q)

      // For indices.
      .invoke<-1>(IndexMem, index_read_addr_qi, index_read_data_qi, indices)
      .invoke<-1>(ReadAddrArbiter, index_read_addr_q, index_pe_qi,
                  index_read_addr_qi)
      .invoke<-1>(IndexReadDataArbiter, index_pe_qi, index_read_data_qi,
                  index_read_data_q)

      // For vertices.
      .invoke<-1>(VertexMem, vertex_read_addr_qi, vertex_read_data_qi,
                  vertex_write_addr_q, vertex_write_data_q, vertices)
      .invoke<-1>(ReadAddrArbiter, vertex_read_addr_q, vertex_pe_qi,
                  vertex_read_addr_qi)
      .invoke<-1>(VertexReadDataArbiter, vertex_pe_qi, vertex_read_data_qi,
                  vertex_read_data_q)

      // PEs.
      .invoke<-1, kPeCount>(ProcElemS0, task_req_q, task_s0_q, edge_read_addr_q,
                            edge_read_data_q, index_read_addr_q,
                            index_read_data_q)
      .invoke<-1, kPeCount>(ProcElemS1, task_s0_q, task_s1p0_q, task_s1p1_q,
                            vertex_read_addr_q, vertex_read_data_q)
      .invoke<-1, kPeCount>(ProcElemS2, task_s1p0_q, task_s1p1_q, task_resp_q);
}

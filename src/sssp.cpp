#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

#include "sssp-kernel.h"

// Estimated DRAM latency.
constexpr int kMemLatency = 50;

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
    // Scalars.
    const Vid vertex_count, const Vid root,
    // Queue requests.
    tapa::istream<QueueOp>& queue_req_q,
    tapa::ostream<QueueOpResp>& queue_resp_q, tapa::ostream<Vid>& read_addr_q,
    tapa::istream<Vertex>& read_data_q, tapa::ostream<Vid>& write_addr_q,
    tapa::ostream<Vertex>& write_data_q, tapa::mmap<Task> heap_array_spill,
    tapa::mmap<Vid> heap_index_spill) {
#pragma HLS inline recursive
  // Parent   of heap_array[i]: heap_array[(i - 1) / 2]
  // Children of heap_array[i]: heap_array[i * 2 + 1], heap_array[i * 2 + 2]
  // Heap rule: child <= parent
  Vid heap_size = 0;

  constexpr int kMaxOnChipSize = 4096 * 16;
#ifdef __SYNTHESIS__
  Task heap_array[kMaxOnChipSize];
  Vid heap_index[kMaxOnChipSize];
heap_index_init:
  for (Vid i = 0; i < kMaxOnChipSize; ++i) {
#pragma HLS pipeline II = 1
    heap_index[i] = kNullVid;
  }
#else   // __SYNTHESIS__
  auto heap_array = heap_array_spill;
  auto heap_index = heap_index_spill;
#endif  // __SYNTHESIS__
#pragma HLS array_partition variable = heap_array cyclic factor = 2
#pragma HLS resource variable = heap_array core = RAM_2P_URAM latency = 2
#pragma HLS resource variable = heap_index core = RAM_2P_URAM latency = 2
#pragma HLS data_pack variable = heap_array
#pragma HLS data_pack variable = heap_index

  int64_t heapify_up_count = 0;
  int64_t heapify_up_total = 0;
  int64_t heapify_down_count = 0;
  int64_t heapify_down_total = 0;

  CLEAN_UP(clean_up, [&]() {
    VLOG(3) << "average heapify up trip count: "
            << 1. * heapify_up_total / heapify_up_count;
    VLOG(3) << "average heapify down trip count: "
            << 1. * heapify_down_total / heapify_down_count;

    // Check that heap_index is restored to the initial state.
    CHECK_EQ(heap_size, 0);
    for (int i = 0; i < vertex_count; ++i) {
      CHECK_EQ(heap_index[i], kNullVid) << "i = " << i;
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
        const Vid task_index = heap_index[new_task.vid];
        bool heapify = false;
        Vid heapify_index = task_index;
        if (task_index != kNullVid) {
          const Task old_task = heap_array[task_index];
          CHECK_EQ(old_task.vid, new_task.vid);
          if (!(new_task <= old_task)) {
            heapify = true;
            heap_array[task_index] = new_task;
          }
        } else {
          heapify = true;
          heap_array[heap_size] = new_task;
          heap_index[new_task.vid] = heapify_index = heap_size;
          ++heap_size;
          resp.task_op = TaskOp::NEW;
        }

        if (heapify) {
          // Increase the priority of heap_array[i] if necessary.
          Vid i = heapify_index;
          const Task task_i = heap_array[i];
          VLOG_F(10, h_up) << "start: array[" << i << "]  -> " << task_i;

          ++heapify_up_count;

        heapify_up:
          for (;;) {
#pragma HLS pipeline II = 3
            ++heapify_up_total;
            const Vid parent = (i - 1) / 2;
            const Task task_parent = heap_array[parent];

            if (i == 0 || task_i <= task_parent) break;
            heap_array[i] = task_parent;
            heap_index[task_parent.vid] = i;

            VLOG_F(10, h_up)
                << "iter:  array[" << parent << "]  -> " << task_parent;
            VLOG_F(10, h_up) << "       array[" << i << "] <-  " << task_parent;
            VLOG_F(10, h_up)
                << "       index[" << task_parent.vid << "] <-  " << i;
            i = parent;
          }
          heap_array[i] = task_i;
          heap_index[task_i.vid] = i;
          VLOG_F(10, h_up) << "       array[" << i << "] <-  " << task_i;
          VLOG_F(10, h_up) << "done:  index[" << task_i.vid << "] <-  " << i;
        }
        break;
      }
      case QueueOp::POP: {
        if (heap_size > 0) {
          const Task front = heap_array[0];
          heap_index[front.vid] = kNullVid;
          --heap_size;

          resp.task_op = TaskOp::NEW;
          resp.task = front;
          read_addr_q.write(front.vid);

          if (heap_size > 0) {
            ++heapify_down_count;

            // Find proper index `i` for `task_i`.
            const Task task_i = heap_array[heap_size];
            Vid i = 0;

          heapify_down:
            for (;;) {
#pragma HLS pipeline II = 3
              ++heapify_down_total;
              const Vid left = i * 2 + 1;
              const Vid right = i * 2 + 2;
              const bool left_is_valid = left < heap_size;
              const bool right_is_valid = right < heap_size;
              const Task task_left =
                  heap_array[(left_is_valid ? left : 1) / 2 * 2 + 1];
              const Task task_right =
                  heap_array[(right_is_valid ? right : 2) / 2 * 2];
              const bool left_is_ok = !left_is_valid || task_left <= task_i;
              const bool right_is_ok = !right_is_valid || task_right <= task_i;
              if (left_is_ok && right_is_ok) break;

              const bool left_is_max =
                  !right_is_valid || (left_is_valid && task_right <= task_left);
              const Vid max = left_is_max ? left : right;
              const Task task_max = left_is_max ? task_left : task_right;

              heap_array[i] = task_max;
              heap_index[task_max.vid] = i;

              i = max;
            }
            heap_array[i] = task_i;
            heap_index[task_i.vid] = i;
          }
          if (!(read_data_q.read() <= front.vertex)) {
            write_addr_q.write(front.vid);
            write_data_q.write(front.vertex);
            ap_wait_n(8);
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

void VertexReadAddrArbiter(tapa::istreams<Vid, kPeCount + 1>& req_q,
                           tapa::ostream<PeId>& pe_q,
                           tapa::ostream<Vid>& addr_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    bool done = false;
    RANGE(pe, kPeCount + 1, {
      Vid addr;
      if (!done && req_q[kPeCount - pe].try_read(addr)) {
        done |= true;
        addr_q.write(addr);
        pe_q.write(kPeCount - pe);
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
                           tapa::ostreams<Vertex, kPeCount + 1>& data_out_q) {
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

    fwd:
      for (Vid i_req = 0, i_resp = 0; i_resp < count;) {
        _Pragma("HLS pipeline II = 1");
        if (i_req < count && i_req < i_resp + kMemLatency) {
          const auto task_s0 = task_s0_q.read();
          const auto dst = task_s0.dst;
          task_s1p0_q.write({.vid = dst, .distance = task_s0.weight});
          read_addr_q.write(dst);
          ++i_req;
        }

        if (!read_data_q.empty()) {
          task_s1p1_q.write(read_data_q.read(nullptr).distance);
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

  int task_count = 1;                       // Number of active tasks.
  int queue_size = 0;                       // Number of tasks in the queue.
  DECL_ARRAY(bool, busy, kPeCount, false);  // Busy state of each PE.

  task_req_q[0].write(root);
  busy[0] = true;

  // Statistics.
  int64_t visited_vertex_count = 0;
  int64_t visited_edge_count = 0;
  int64_t queue_count = 0;
  int64_t total_queue_size = 0;
  int64_t max_queue_size = 0;

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
  for (; queue_size > 0 || task_count > 0 || !queue_resp_q.empty();) {
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
    }

    // Assign tasks to PEs.
    RANGE(pe, kPeCount,
          UNUSED SET(busy[pe],
                     RESET(queue_buf_valid,
                           task_req_q[pe].try_write(queue_buf.task.vid))));

    // Receive tasks generated from PEs.
    RANGE(pe, kPeCount, {
      if (SET(task_buf_valid, task_resp_q[pe].try_read(task_buf))) {
        if (task_buf.op == TaskOp::DONE) {
          task_buf_valid = false;
          --task_count;
          busy[pe] = false;

          // Update statistics.
          ++visited_vertex_count;
          visited_edge_count += task_buf.task.vid;
          STATS(9, recv, "TASK : DONE");
        }
      }
    });

    if (task_count < kPeCount && queue_size > 0) {
      // Dequeue tasks from the queue.
      if (queue_req_q.try_write({.op = QueueOp::POP, .task = {}})) {
        ++task_count;
        STATS(9, send, "QUEUE: POP ");
      }
    } else if (RESET(task_buf_valid,
                     queue_req_q.try_write(
                         {.op = QueueOp::PUSH, .task = task_buf.task}))) {
      // Enqueue tasks generated from PEs.
      ++queue_size;
      STATS(9, send, "QUEUE: PUSH");
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
          tapa::mmap<Vertex> vertices, tapa::mmap<Task> heap_array,
          tapa::mmap<Vid> heap_index) {
  tapa::stream<QueueOp, 256> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");

  tapa::streams<Vid, kPeCount, 2> task_req_q("task_req");
  tapa::streams<Edge, kPeCount, 2> task_s0_q("task_stage0");
  tapa::streams<Update, kPeCount, 64> task_s1p0_q("task_stage0_part0");
  tapa::streams<float, kPeCount, 2> task_s1p1_q("task_stage0_part1");
  tapa::streams<TaskOp, kPeCount, 2> task_resp_q("task_resp");

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
  tapa::streams<Vid, kPeCount + 1, 2> vertex_read_addr_q("vertex_read_addr");
  tapa::streams<Vertex, kPeCount + 1, 2> vertex_read_data_q("vertex_read_data");
  tapa::stream<Vid, 2> vertex_write_addr_q("vertex_write_addr");
  tapa::stream<Vertex, 2> vertex_write_data_q("vertex_write_data");
  tapa::stream<Vid, 2> vertex_read_addr_qi("vertices.read_addr");
  tapa::stream<Vertex, 2> vertex_read_data_qi("vertices.read_data");
  tapa::stream<PeId, 64> vertex_pe_qi("vertices.pe");

  tapa::task()
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q)
      .invoke<-1>(TaskQueue, vertex_count, root, queue_req_q, queue_resp_q,
                  vertex_read_addr_q[kPeCount], vertex_read_data_q[kPeCount],
                  vertex_write_addr_q, vertex_write_data_q, heap_array,
                  heap_index)

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
      .invoke<-1>(VertexReadAddrArbiter, vertex_read_addr_q, vertex_pe_qi,
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

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
    tapa::ostream<QueueOpResp>& queue_resp_q, tapa::mmap<Task> heap_array_spill,
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
    heap_index[i] = kNullVertex;
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
      CHECK_EQ(heap_index[i], kNullVertex) << "i = " << i;
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
        if (task_index != kNullVertex) {
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
          heap_index[front.vid] = kNullVertex;
          --heap_size;

          resp.task_op = TaskOp::NEW;
          resp.task = front;

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
        }
        break;
      }
    }
    resp.queue_size = heap_size;
    queue_resp_q.write(resp);
  }
}

void ProcElem(
    // Task requests.
    tapa::istream<Vid>& task_req_q, tapa::ostream<TaskOp>& task_resp_q,
    // Vertex requests.
    tapa::ostream<Update>& update_req_q, tapa::istream<TaskOp>& update_resp_q,
    // Memory-maps.
    tapa::mmap<Index> indices, tapa::async_mmap<Edge> edges) {
  DECL_BUF(Edge, edge);
  DECL_BUF(TaskOp, resp);

spin:
  for (;;) {
    const auto src = task_req_q.read();
    const Index index = indices[src];
    update_req_q.write({.vid = src, .weight = bit_cast<float>(index.count)});
  read_edges:
    for (Eid eid_req = 0, eid_resp = 0; eid_resp < index.count;) {
      if (eid_req < index.count && eid_req < eid_resp + kMemLatency &&
          edges.read_addr_try_write(index.offset + eid_req)) {
        ++eid_req;
      }

      UPDATE(edge, edges.read_data_try_read(edge),
             update_req_q.try_write({.vid = edge.dst, .weight = edge.weight}));

      if (UPDATE(resp, update_resp_q.try_read(resp),
                 resp.op == TaskOp::NOOP || task_resp_q.try_write(resp))) {
        ++eid_resp;
      }
    }
    task_resp_q.write({
        .op = TaskOp::DONE,
        .task = {.vid = index.count, .distance = 0.f},
    });
  }
}

void VertexMem(
    // Update requests.
    tapa::istreams<Update, kPeCount>& req_q,
    tapa::ostreams<TaskOp, kPeCount>& resp_q,
    // Memory-maps.
    tapa::mmap<Vid> parents, tapa::mmap<float> distances) {
  DECL_ARRAY(volatile Vid, active_srcs, kPeCount, kNullVertex);
  DECL_ARRAY(volatile Vid, active_dsts, kPeCount, kNullVertex);

  DECL_ARRAY(bool, valid, kPeCount, false);
  DECL_ARRAY(Update, updates, kPeCount, Update());

spin:
  for (;;) {
    RANGE(pe, kPeCount, {
      Update update = req_q[pe].read();
      const auto src = update.vid;
      const auto count = bit_cast<Vid>(update.weight);
      const auto src_distance = distances[src];

      for (Vid i = 0; i < count; ++i) {
        _Pragma("HLS pipeline II = 1");
        _Pragma("HLS dependence false variable = distances");
        const auto update = req_q[pe].read();
        const auto dst = update.vid;
        const auto weight = update.weight;
        const auto dst_distance = distances[dst];
        const auto new_distance = src_distance + weight;
        TaskOp resp{.op = TaskOp::NOOP, .task = {}};
        if (new_distance < dst_distance) {
          VLOG_F(9, info) << "distances[" << dst << "] = " << dst_distance
                          << " -> distances[" << src << "] + " << weight
                          << " = " << new_distance;
          distances[dst] = new_distance;
          parents[dst] = src;
          resp = {
              .op = TaskOp::NEW,
              .task = {.vid = dst, .distance = new_distance},
          };
        }

        resp_q[pe].write(resp);
      }
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
}

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::async_mmap<Edge> edges, tapa::mmap<Index> indices,
          tapa::mmap<Vid> parents, tapa::mmap<float> distances,
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  tapa::stream<QueueOp, 2> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<Vid, kPeCount, 2> task_req_q("task_req");
  tapa::streams<TaskOp, kPeCount, 2> task_resp_q("task_resp");
  tapa::streams<Update, kPeCount, 2> update_req_q("update_req");
  tapa::streams<TaskOp, kPeCount, 2> update_resp_q("update_resp");

  tapa::task()
      .invoke<-1>(TaskQueue, vertex_count, root, queue_req_q, queue_resp_q,
                  heap_array, heap_index)
      .invoke<-1, kPeCount>(ProcElem, task_req_q, task_resp_q, update_req_q,
                            update_resp_q, indices, edges)
      .invoke<-1>(VertexMem, update_req_q, update_resp_q, parents, distances)
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q);
}

#include <cassert>
#include <cstring>

#include <algorithm>
#include <iomanip>

#include <tapa.h>

// Non-synthesizable
#include <boost/heap/fibonacci_heap.hpp>
#include <memory>
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
    tapa::ostream<QueueOpResp>& queue_resp_q) {
  boost::heap::fibonacci_heap<Task> task_q;
  using handle_t = decltype(task_q)::handle_type;
  std::vector<std::unique_ptr<handle_t>> handle_map(vertex_count);
  handle_map[root] =
      std::make_unique<handle_t>(task_q.push({.vid = root, .distance = 0.}));

  int64_t queue_count = 0;
  int64_t total_queue_size = 0;
  int64_t max_queue_size = 0;

infinite_loop:
  for (QueueOp req;;) {
    if (queue_req_q.try_read(req)) {
      QueueOpResp resp = {
          .queue_op = req.op, .task_op = TaskOp::NOOP, .task = {}};
      switch (req.op) {
        case QueueOp::PUSH:
          RANGE(pe, kPeCount, {
            if (auto& handle = handle_map[req.task.vid]) {
              if (handle->node_->value < req.task) {
                task_q.increase(*handle, req.task);
              }
            } else {
              handle = std::make_unique<handle_t>(task_q.push(req.task));
              resp.task_op = TaskOp::NEW;
            }
            ++queue_count;
            total_queue_size += task_q.size();
            if (task_q.size() > max_queue_size) max_queue_size = task_q.size();
          });
          break;
        case QueueOp::POP:
          if (!task_q.empty()) {
            resp.task_op = TaskOp::NEW;
            resp.task = task_q.top();
            handle_map[task_q.top().vid].reset();
            task_q.pop();
          }
          break;
      }
      queue_resp_q.write(resp);
    }
  }
}

void ProcElem(
    // Task requests.
    tapa::istream<Vid>& task_req_q, tapa::ostream<TaskOp>& task_resp_q,
    // Vertex requests.
    tapa::ostream<Update>& update_req_q, tapa::istream<TaskOp>& update_resp_q,
    // Memory-maps.
    tapa::mmap<Index> indices, tapa::mmap<Edge> edges) {
infinite_loop:
  for (;;) {
    const auto src = task_req_q.read();
    CHECK_LT(src, indices.size());
    const Index& index = indices[src];
    for (Eid eid_req = 0, eid_resp = 0; eid_resp < index.count;) {
      if (eid_req < index.count &&
          update_req_q.try_write(
              {.src = src, .edge = edges[index.offset + eid_req]})) {
        ++eid_req;
      }

      TaskOp resp;
      if (update_resp_q.try_peek(resp) &&
          (resp.op == TaskOp::NOOP || task_resp_q.try_write(resp))) {
        update_resp_q.read(nullptr);
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

infinite_loop:
  for (;;) {
    RANGE(pe, kPeCount, {
      Update update;
      if (req_q[pe].try_peek(update) &&
          !Contains(active_srcs, update.edge.dst) &&
          !Contains(active_dsts, update.edge.dst) &&
          !Contains(active_dsts, update.src)) {
        // No conflict; lock both src and dst and read from the FIFO.
        active_srcs[pe] = update.src;
        active_dsts[pe] = update.edge.dst;
        req_q[pe].read(nullptr);

        // Process the update.
        TaskOp resp{.op = TaskOp::NOOP, .task = {}};
        auto new_distance = distances[update.src] + update.edge.weight;
        if (new_distance < distances[update.edge.dst]) {
          distances[update.edge.dst] = new_distance;
          parents[update.edge.dst] = update.src;
          resp = {
              .op = TaskOp::NEW,
              .task = {.vid = update.edge.dst, .distance = new_distance},
          };
        }

        // Unlock src and dst.
        active_srcs[pe] = kNullVertex;
        active_dsts[pe] = kNullVertex;
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

  int task_count = 0;                       // Number of active tasks.
  int queue_size = 1;                       // Number of tasks in the queue.
  DECL_ARRAY(bool, busy, kPeCount, false);  // Busy state of each PE.

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

  for (; queue_size > 0 || task_count > 0 || !queue_resp_q.empty();) {
    // Process response messages from the queue.
    if (SET(queue_buf_valid, queue_resp_q.try_read(queue_buf))) {
      switch (queue_buf.queue_op) {
        case QueueOp::PUSH:
          // PUSH requests do not need further processing.
          queue_buf_valid = false;
          if (queue_buf.task_op == TaskOp::NOOP) {
            // PUSH request updated priority of existing tasks.
            --queue_size;
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
            --queue_size;
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
    RANGE(
        pe, kPeCount,
        UNUSED(SET(busy[pe], RESET(queue_buf_valid, task_req_q[pe].try_write(
                                                        queue_buf.task.vid)))));

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

void SSSP(Vid root, tapa::mmap<int64_t> metadata, tapa::mmap<Edge> edges,
          tapa::mmap<Index> indices, tapa::mmap<Vid> parents,
          tapa::mmap<float> distances) {
  const Vid vertex_count = indices.size();

  //*
  tapa::stream<QueueOp, 2> queue_req_q("queue_req");
  tapa::stream<QueueOpResp, 2> queue_resp_q("queue_resp");
  tapa::streams<Vid, kPeCount, 2> task_req_q("task_req");
  tapa::streams<TaskOp, kPeCount, 2> task_resp_q("task_resp");
  tapa::streams<Update, kPeCount, 2> update_req_q("update_req");
  tapa::streams<TaskOp, kPeCount, 2> update_resp_q("update_resp");

  tapa::task()
      .invoke<-1>(TaskQueue, vertex_count, root, queue_req_q, queue_resp_q)
      .invoke<-1, kPeCount>(ProcElem, task_req_q, task_resp_q, update_req_q,
                            update_resp_q, indices, edges)
      .invoke<-1>(VertexMem, update_req_q, update_resp_q, parents, distances)
      .invoke<0>(Dispatcher, root, metadata, task_req_q, task_resp_q,
                 queue_req_q, queue_resp_q);
  // */

  /*
  boost::heap::fibonacci_heap<Task> task_q;
  using handle_t = decltype(task_q)::handle_type;
  std::vector<std::unique_ptr<handle_t>> handle_map(vertex_count);
  handle_map[root] =
      std::make_unique<handle_t>(task_q.push({.vid = root, .distance = 0.}));

  int64_t visited_edge_count = 0;
  int64_t queue_count = 0;
  int64_t total_queue_size = 0;
  int64_t max_queue_size = 0;

  auto update = [&distances, &parents](const Edge& edge, Vid src) -> float {
    float result = 0.f;
#pragma omp critical(vertex_lock)
    {
      auto new_distance = distances[src] + edge.weight;
      CHECK_NE(new_distance, result) << result << " indicates inactivity";
      if (new_distance < distances[edge.dst]) {
        distances[edge.dst] = new_distance;
        parents[edge.dst] = src;
        result = new_distance;
      }
    }
    return result;
  };

  while (!task_q.empty()) {
    std::vector<Task> tasks;
    tasks.reserve(kPeCount);
    for (int i = 0; !task_q.empty() && i < kPeCount; ++i) {
      tasks.push_back(task_q.top());
      task_q.pop();
      handle_map[tasks.back().vid].reset();
    }

#pragma omp parallel for
    for (int i = 0; i < tasks.size(); ++i) {
      const auto src = tasks[i].vid;
      const auto src_distance = tasks[i].distance;
      CHECK_LT(src, indices.size());
      const Index& index = indices[src];
      for (Eid eid = index.offset; eid < index.offset + index.count; ++eid) {
        const Edge& edge = edges[eid];
#pragma omp atomic
        ++visited_edge_count;
        if (auto new_distance = update(edge, src)) {
#pragma omp critical(queue_lock)
          {
            Task task{.vid = edge.dst, .distance = new_distance};
            if (auto& handle = handle_map[task.vid]) {
              if (handle->node_->value < task) task_q.increase(*handle, task);
            } else {
              handle = std::make_unique<handle_t>(task_q.push(task));
            }
            ++queue_count;
            total_queue_size += task_q.size();
            if (task_q.size() > max_queue_size) max_queue_size = task_q.size();
          }
        }
      }
    }
  }

  metadata[0] = visited_edge_count;
  metadata[1] = total_queue_size;
  metadata[2] = queue_count;
  metadata[3] = max_queue_size;
  // */
}

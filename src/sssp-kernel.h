#ifndef TAPA_SSSP_KERNEL_H_
#define TAPA_SSSP_KERNEL_H_

#include <cstddef>

#include <ap_int.h>

#include <algorithm>
#include <type_traits>

#include "sssp.h"

using uint_vid_t = ap_uint<kVidWidth>;

using uint_eid_t = ap_uint<kEidWidth>;

using uint_pe_qid_t = ap_uint<bit_length(kPeCount / kQueueCount - 1)>;

using uint_vertex_noop_t = ap_uint<bit_length(kSubIntervalCount)>;

using uint_queue_noop_t = ap_uint<bit_length(kQueueCount)>;

struct SourceVertex {
  Vid parent;
  float distance;
};

using EdgeReq = tapa::packet<Eid, SourceVertex>;

using TaskVec = tapa::vec_t<TaskOnChip, kEdgeVecLen>;

struct TaskCount {
  ap_uint<bit_length(kPeCount)> old_task_count;
  uint_eid_t new_task_count;
};

// Used in:
//
// ProcElemS1 -> (net)
// (net) -> VertexReaderS0 -> VertexReaderS1 -> VertexUpdater -> Dispatcher
struct TaskOp {
  enum Op { NEW, NOOP } op;
  TaskOnChip task;  // Valid only when op is NEW.
};

inline std::ostream& operator<<(std::ostream& os, const TaskOp& obj) {
  switch (obj.op) {
    case TaskOp::NEW:
      return os << "{ op: NEW, task: " << Task(obj.task) << " }";
    case TaskOp::NOOP:
      return os << "{ op: NOOP }";
  }
}

// Used in:
//
// Dispatcher -> TaskQueue
struct QueueOp {
  enum Op { PUSH, PUSHPOP, POP } op;
  TaskOnChip task;  // Valid only when op is PUSH.

  bool is_push() const { return op == QueueOp::PUSH; }
  bool is_pop() const { return op == QueueOp::POP; }
  bool is_pushpop() const { return op == QueueOp::PUSHPOP; }
};

// Used in:
//
// TaskQueue -> Dispatcher
struct QueueOpResp {
  QueueOp::Op queue_op;
  // Valid values are NEW and NOOP.
  // If queue_op is PUSH, NEW indicates a new task is created; NOOP indicates
  // the priority of existing task is increased.
  // If queue_op is POP, NEW indicates a task is dequeued and returned; NOOP
  // indicates the queue is empty.
  TaskOp::Op task_op;
  TaskOnChip task;  // Valid only when queue_op is POP and task_op is NEW.

  bool is_push() const { return queue_op == QueueOp::PUSH; }
  bool is_pop_noop() const {
    return queue_op == QueueOp::POP && task_op == TaskOp::NOOP;
  }
  bool is_pop_new() const {
    return queue_op != QueueOp::PUSH && task_op == TaskOp::NEW;
  }
};

inline std::ostream& operator<<(std::ostream& os, const QueueOpResp& obj) {
  os << "{queue_op: ";
  switch (obj.queue_op) {
    case QueueOp::PUSH:
      os << "PUSH";
      break;
    case QueueOp::PUSHPOP:
      os << "PUSHPOP";
      break;
    case QueueOp::POP:
      os << "POP";
      break;
  }
  os << ", task_op: ";
  switch (obj.task_op) {
    case TaskOp::NEW:
      os << "NEW";
      break;
    case TaskOp::NOOP:
      os << "NOOP";
      break;
  }
  if (obj.queue_op != QueueOp::PUSH && obj.task_op == TaskOp::NEW) {
    os << ", task: " << Task(obj.task);
  }
  return os << "}";
}

struct VertexCacheEntry {
  bool is_valid;
  bool is_reading;
  bool is_writing;
  bool is_dirty;

  VertexCacheEntry() {}

  VertexCacheEntry(std::nullptr_t) {
    is_valid = false;
    is_reading = false;
    is_writing = false;
    is_dirty = false;
  }

  Task GetTask() const {
    return {
        .vid = static_cast<Vid>(vid_),
        .vertex =
            {
                .parent = static_cast<Vid>(parent_),
                .distance = distance_,
                .offset = static_cast<Eid>(offset_),
                .degree = static_cast<Vid>(degree_),
            },
    };
  }

  void SetVid(Vid vid) { this->vid_ = vid; }

  void SetValue(const Vertex& vertex) {
    parent_ = vertex.parent;
    distance_ = vertex.distance;
  }

  void SetMetadata(const Vertex& vertex) {
    offset_ = vertex.offset;
    degree_ = vertex.degree;
  }

 private:
  uint_vid_t vid_;
  uint_vid_t parent_;
  float distance_;
  uint_eid_t offset_;
  uint_vid_t degree_;
};

// Convenient functions and macros.

#ifdef __SYNTHESIS__
#define CLEAN_UP(...)
#else  // __SYNTHESIS__
struct CleanUp {
  ~CleanUp() { clean_up_(); }
  std::function<void()> clean_up_;
};
#define CLEAN_UP(name, ...) \
  CleanUp name { __VA_ARGS__ }
#endif  // __SYNTHESIS__

// Prints logging messages with tag and function name as prefix.
#define LOG_F(level, tag) LOG(level) << #tag << "@" << __FUNCTION__ << ": "
#define VLOG_F(level, tag) VLOG(level) << #tag << "@" << __FUNCTION__ << ": "

/// Creates an unrolled loop with @c var iterating over [0, bound).
///
/// @param var    Name of the loop variable, may be referenced in @c body.
/// @param bound  Trip count of the loop.
/// @param ...    Loop body. Variadic arguments are used to allow commas.
#define RANGE(var, bound, ...)                \
  do {                                        \
    for (int var = 0; var < (bound); ++var) { \
      _Pragma("HLS unroll") __VA_ARGS__;      \
    }                                         \
  } while (0)

// Creates a pragma with stringified arguments.
#define DO_PRAGMA(x) _Pragma(#x)

/// Defines and initializes a fully partitioned array.
///
/// @param type Type of each array element.
/// @param var  Name of the declared array.
/// @param size Array size, in unit of elements, evaluated only once.
/// @param val  Initial value of all array elements, evaluated once per element.
#define DECL_ARRAY(type, var, size, val)                  \
  type var[(size)];                                       \
  static_assert(std::is_array<decltype(var)>::value,      \
                "'" #var "' is not an array");            \
  DO_PRAGMA(HLS array_partition complete variable = var); \
  RANGE(_i, sizeof(var) / sizeof(var[0]), var[_i] = (val))

/// Sets @c var, if @c var is not set and @c val evaluates to true.
///
/// @param var  A boolean lvalue that must not have side effects.
/// @param val  A boolean expression. Evaluated only if @c var is false.
/// @return     Whether @c var changed from false to true.
#define SET(var, val) !(var || !(var = (val)))

/// Resets @c var, if @c var is set and @c val evaluates to true.
///
/// @param var  A boolean lvalue that must not have side effects.
/// @param val  A boolean expression. Evaluated only if @c var is true.
/// @return     Whether @c var changed from true to false.
#define RESET(var, val) (var && !(var = !(val)))

/// Defines a variable and a boolean valid signal. To be used with @c UPDATE.
///
/// @param type Type of the variable.
/// @param var  Name of the declared variable.
#define DECL_BUF(type, var) \
                            \
  type var;                 \
  bool var##_valid = false

/// Produce @c buf if it is invalid, and consume it if valid.
/// @c buf should be declared using @c DECL_BUF.
///
/// @param buf      Name of the buffer variable.
/// @param producer A boolean expression that produces value to @c buf.
/// @param consumer A boolean expression that consumes value from @c buf.
/// @return         Whether @c consumer successfully consumed an element.
#define UPDATE(buf, producer, consumer) \
  SET(buf##_valid, producer), RESET(buf##_valid, consumer)

#define UNUSED (void)

/// Expands to a variable that is initialized with its own @c name as a string.
///
/// @param name Name of the variable.
/// @return     Variable initialized with @c name as a string.
#define VAR(name) name(#name)

/// Return @c value with an assumption that @c value % @c mod == @c rem.
inline int assume_mod(int value, int mod, int rem) {
#pragma HLS inline
  return value / mod * mod + rem;
}

/// Return @c value with an assertion that @c value % @c mod == @c rem.
inline int assert_mod(int value, int mod, int rem) {
#pragma HLS inline
  CHECK_EQ(value % mod, rem);
  return assume_mod(value, mod, rem);
}

template <typename T, int N, typename UnaryFunction>
inline void for_each(const T (&array)[N], UnaryFunction f) {
  for (int i = 0; i < N; ++i) {
#pragma HLS unroll
    f(array[i]);
  }
}

template <typename T, int N, typename BinaryOperation>
inline T accumulate(const T (&array)[N], T init, BinaryOperation op) {
#pragma HLS inline
  for_each(array, [&](T value) { init = op(init, value); });
  return init;
}

template <typename T, int N, typename UnaryPredicate>
inline bool all_of(const T (&array)[N], UnaryPredicate p) {
#pragma HLS inline
  bool result = true;
  for_each(array, [&](T value) { result &= p(value); });
  return result;
}

template <typename T, int N, typename UnaryPredicate>
inline bool any_of(const T (&array)[N], UnaryPredicate p) {
#pragma HLS inline
  bool result = false;
  for_each(array, [&](T value) { result |= p(value); });
  return result;
}

template <typename T, int N, typename UnaryPredicate>
inline bool none_of(const T (&array)[N], UnaryPredicate p) {
#pragma HLS inline
  return !all_of(array, [&](T value) { return !p(value); });
}

template <typename T, int N>
inline bool all_of(const T (&array)[N]) {
#pragma HLS inline
  bool result = true;
  for_each(array, [&](T value) { result &= bool(value); });
  return result;
}

template <typename T, int N>
inline bool any_of(const T (&array)[N]) {
#pragma HLS inline
  bool result = false;
  for_each(array, [&](T value) { result |= bool(value); });
  return result;
}

template <typename T, int N>
inline bool none_of(const T (&array)[N]) {
#pragma HLS inline
  return !all_of(array, [&](T value) { return !value; });
}

template <int begin, int len>
struct arbiter {
  template <typename T, uint64_t S, typename index_t>
  static bool find_non_empty(tapa::istreams<T, S>& in_qs,
                             const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= S, "begin + len must <= S");
    index_t idx_left, idx_right;
    const auto is_left_non_empty =
        arbiter<begin, len / 2>::find_non_empty(in_qs, priority, idx_left);
    const auto is_right_non_empty =
        arbiter<begin + len / 2, len - len / 2>::find_non_empty(in_qs, priority,
                                                                idx_right);
    idx = is_left_non_empty ^ is_right_non_empty
              ? is_left_non_empty ? idx_left : idx_right
          : ap_uint<len / 2>(priority.range(begin + len / 2 - 1, begin))
                  .or_reduce()
              ? idx_left
              : idx_right;
    return is_left_non_empty || is_right_non_empty;
  }

  template <typename T, uint64_t S, typename index_t>
  static bool find_non_full(tapa::ostreams<T, S>& out_qs,
                            const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= S, "begin + len must <= S");
    index_t idx_left, idx_right;
    const auto is_left_non_full =
        arbiter<begin, len / 2>::find_non_full(out_qs, priority, idx_left);
    const auto is_right_non_full =
        arbiter<begin + len / 2, len - len / 2>::find_non_full(out_qs, priority,
                                                               idx_right);
    idx = is_left_non_full ^ is_right_non_full
              ? is_left_non_full ? idx_left : idx_right
          : ap_uint<len / 2>(priority.range(begin + len / 2 - 1, begin))
                  .or_reduce()
              ? idx_left
              : idx_right;
    return is_left_non_full || is_right_non_full;
  }

  template <typename T, int N, typename pos_t>
  static T find_max(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= N, "begin + len must <= N");
    pos_t pos_left, pos_right;
    const auto max_left = arbiter<begin, len / 2>::find_max(array, pos_left);
    const auto max_right =
        arbiter<begin + len / 2, len - len / 2>::find_max(array, pos_right);
    if (!(max_left <= max_right)) {
      pos = pos_left;
      return max_left;
    }
    pos = pos_right;
    return max_right;
  }

  template <int N, typename pos_t>
  static bool find_false(const bool (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= N, "begin + len must <= N");
    pos_t pos_left, pos_right;
    const auto is_left_false =
        arbiter<begin, len / 2>::find_false(array, pos_left);
    const auto is_right_false =
        arbiter<begin + len / 2, len - len / 2>::find_false(array, pos_right);
    if (is_left_false) {
      pos = pos_left;
      return is_left_false;
    }
    pos = pos_right;
    return is_right_false;
  }
};

template <int begin>
struct arbiter<begin, 1> {
  template <typename T, uint64_t S, typename index_t>
  static bool find_non_empty(tapa::istreams<T, S>& in_qs,
                             const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < S, "begin must < S");
    idx = begin;
    if (in_qs[begin].empty()) {
      return false;
    }
    return true;
  }

  template <typename T, uint64_t S, typename index_t>
  static bool find_non_full(tapa::ostreams<T, S>& out_qs,
                            const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < S, "begin must < S");
    idx = begin;
    if (out_qs[begin].full()) {
      return false;
    }
    return true;
  }

  template <typename T, int N, typename pos_t>
  static T find_max(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < N, "beginmust < N");
    pos = begin;
    return array[begin];
  }

  template <int N, typename pos_t>
  static bool find_false(const bool (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < N, "beginmust < N");
    pos = begin;
    return !array[begin];
  }
};

/// Find a non-empty istream.
///
/// @param[in] in_qs    Input streams.
/// @param[in] priority One-hot encoding of the prioritzed input stream.
/// @param[out] idx     Index of the non-empty istream, invalid if none found.
/// @return             Whether a non-empty istream is found.
template <typename T, uint64_t S, typename index_t>
inline bool find_non_empty(tapa::istreams<T, S>& in_qs,
                           const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
  return arbiter<0, S>::find_non_empty(in_qs, priority, idx);
}
/// Find a non-full ostream.
///
/// @param[in] out_qs   Output streams.
/// @param[in] priority One-hot encoding of the prioritzed output stream.
/// @param[out] idx     Index of the non-full ostream, invalid if none found.
/// @return             Whether a non-full ostream is found.
template <typename T, uint64_t S, typename index_t>
inline bool find_non_full(tapa::ostreams<T, S>& in_qs,
                          const ap_uint<int(S)>& priority, index_t& idx) {
#pragma HLS inline
  return arbiter<0, S>::find_non_full(in_qs, priority, idx);
}

/// Find a false value in a **completely partitioned** boolean array.
///
/// @param[in] array  A completely partitioned boolean array.
/// @param[out] pos   Position of the false value, invalid if none found.
/// @return           Whether a false value is found.
template <int N, typename pos_t>
inline bool find_false(const bool (&array)[N], pos_t& idx) {
#pragma HLS inline
  return arbiter<0, N>::find_false(array, idx);
}

/// Find the maximum value in a **completely partitioned** array.
///
/// @param[in] array  A completely partitioned array.
/// @param[out] pos   Position of the maximum value.
/// @return           The maximum value.
template <typename T, int N, typename pos_t>
inline T find_max(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
  return arbiter<0, N>::find_max(array, pos);
}

template <typename T, int N>
struct shiftreg {
 public:
  static constexpr int depth = N;
  using value_type = T;
  T array[N];
  ap_uint<bit_length(N)> size = 0;

  T shift(T value) {
#pragma HLS inline
#pragma HLS array_partition complete variable = array
    const T result = array[0];
    for (int i = 1; i < depth; ++i) {
#pragma HLS unroll
      array[i - 1] = array[i];
    }
    array[depth - 1] = value;
    return result;
  }

  void push(T value) {
#pragma HLS inline
    CHECK_LT(size, depth);
    array[size++] = value;
  }

  T pop() {
#pragma HLS inline
    CHECK_GT(size, 0);
    --size;
    return shift({});
  }

  void set(T value) {
    for (int i = 0; i < depth; ++i) {
#pragma HLS unroll
      array[i] = value;
    }
    size = depth;
  }

  T front() const {
#pragma HLS inline
    CHECK(!is_empty());
    return array[0];
  }

  bool contains(T value) const {
#pragma HLS inline
#pragma HLS array_partition complete variable = array
    return recursor<0, depth>::contains(array, value, size);
  }

  bool is_empty() const {
#pragma HLS inline
    return size == 0;
  }

  bool full() const {
#pragma HLS inline
    return size >= depth;
  }

 private:
  template <int i, int n>
  struct recursor {
    static bool contains(const T (&array)[N], T value,
                         ap_uint<bit_length(N)> size) {
#pragma HLS inline
      static_assert(i >= 0, "i must >= 0");
      static_assert(n > 1, "n must > 1");
      static_assert(i + n <= depth, "i + n must <= depth");
      return recursor<i, n / 2>::contains(array, value, size) ||
             recursor<i + n / 2, n - n / 2>::contains(array, value, size);
    }
  };

  template <int i>
  struct recursor<i, 1> {
    static bool contains(const T (&array)[N], T value,
                         ap_uint<bit_length(N)> size) {
#pragma HLS inline
      static_assert(i >= 0, "i must >= 0");
      static_assert(i < depth, "i must < depth");
      return i < size && array[i] == value;
    }
  };
};

#ifndef __SYNTHESIS__
inline void ap_wait() {}
inline void ap_wait_n(int) {}
#endif  // __SYNTEHSIS__

// A fully pipelined lightweight proxy for read-only tapa::async_mmap.
template <typename data_t, typename addr_t>
inline void ReadOnlyMem(tapa::istream<addr_t>& read_addr_q,
                        tapa::ostream<data_t>& read_data_q,
                        tapa::async_mmap<data_t>& mem) {
#pragma HLS inline
  DECL_BUF(addr_t, read_addr);
  DECL_BUF(data_t, read_data);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(read_addr, read_addr_q.try_read(read_addr),
           mem.read_addr.try_write(read_addr));
    UPDATE(read_data, mem.read_data.try_read(read_data),
           read_data_q.try_write(read_data));
  }
}

// A fully pipelined lightweight proxy for write-only tapa::async_mmap.
template <typename data_t, typename addr_t>
inline void WriteOnlyMem(tapa::istream<addr_t>& write_addr_q,
                         tapa::istream<data_t>& write_data_q,
                         tapa::async_mmap<data_t>& mem) {
#pragma HLS inline
  DECL_BUF(addr_t, write_addr);
  DECL_BUF(data_t, write_data);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(write_addr, write_addr_q.try_read(write_addr),
           mem.write_addr.try_write(write_addr));
    UPDATE(write_data, write_data_q.try_read(write_data),
           mem.write_data.try_write(write_data));
  }
}

// A fully pipelined lightweight proxy for read-write tapa::async_mmap.
template <typename data_t, typename addr_t>
inline void ReadWriteMem(
    tapa::istream<addr_t>& read_addr_q, tapa::ostream<data_t>& read_data_q,
    tapa::istream<tapa::packet<addr_t, data_t>>& write_req_q,
    tapa::ostream<bool>& write_resp_q, tapa::async_mmap<data_t>& mem) {
#pragma HLS inline
  int32_t write_count = 0;

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!read_addr_q.empty() && !mem.read_addr.full()) {
      mem.read_addr.try_write(read_addr_q.read(nullptr));
    }
    if (!mem.read_data.empty() && !read_data_q.full()) {
      read_data_q.try_write(mem.read_data.read(nullptr));
    }
    if (!write_req_q.empty() && !mem.write_addr.full() &&
        !mem.write_data.full()) {
      const auto pkt = write_req_q.read(nullptr);
      mem.write_addr.try_write(pkt.addr);
      mem.write_data.try_write(pkt.payload);
    }
    if (write_count > 0 && write_resp_q.try_write(false)) {
      --write_count;
    }
    if (!mem.write_resp.empty()) {
      write_count += mem.write_resp.read(nullptr) + 1;
    }
  }
}

template <typename data_t, typename id_t, uint64_t N>
inline void ReadDataArbiter(tapa::istream<id_t>& id_q,
                            tapa::istream<data_t>& data_in_q,
                            tapa::ostreams<data_t, N>& data_out_q) {
#pragma HLS inline
  static_assert(N < std::numeric_limits<id_t>::max(), "invalid id type");
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    data_out_q[id_q.read()].write(data_in_q.read());
  }
}

template <uint64_t in_n, uint64_t out_n>
void TaskArbiterTemplate(tapa::istreams<TaskOnChip, in_n>& in_q,
                         tapa::ostreams<TaskOnChip, out_n>& out_q) {
  static_assert(in_n % out_n == 0 || out_n % in_n == 0);

spin:
  for (int base = 0;; ++base) {
#pragma HLS pipeline II = 1
    if constexpr (in_n > out_n) {  // #input >= #output.
      RANGE(oid, out_n, {
        const auto iid = assert_mod(base % in_n, out_n, oid);
        if (!in_q[iid].empty() && !out_q[oid].full()) {
          out_q[oid].try_write(in_q[iid].read(nullptr));
        }
      });
    } else {  //  #input < #output.
      RANGE(iid, in_n, {
        if (TaskOnChip task; in_q[iid].try_peek(task)) {
          const auto oid = assert_mod(task.vid() % out_n, in_n, iid);
          if (out_q[oid].try_write(task)) {
            in_q[iid].read(nullptr);
          }
        }
      });
    }
  }
}

template <typename T, uint64_t N>
void Duplicate(tapa::istream<T>& in_q, tapa::ostreams<T, N>& out_q) {
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    if (!in_q.empty()) {
      const auto done = in_q.read(nullptr);
      RANGE(i, N, out_q[i].write(done));
    }
  }
}

#endif  // TAPA_SSSP_KERNEL_H_

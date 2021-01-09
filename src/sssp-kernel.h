#ifndef TAPA_SSSP_KERNEL_H_
#define TAPA_SSSP_KERNEL_H_

#include <algorithm>
#include <type_traits>

#include <ap_int.h>

#include "sssp.h"

using PeId = uint8_t;

// Used in:
//
// ProcElemS1 -> Dispatcher
struct TaskOp {
  // ProcElemS1 -> Dispatcher: Valid values are NEW and DONE.
  enum Op { NEW, NOOP, DONE } op;
  Task task;  // Valid only when op is NEW.
};

inline std::ostream& operator<<(std::ostream& os, const TaskOp& obj) {
  switch (obj.op) {
    case TaskOp::NEW:
      return os << "{ op: NEW, task: " << obj.task << " }";
    case TaskOp::NOOP:
      return os << "{ op: NOOP }";
    case TaskOp::DONE:
      return os << "{ op: DONE }";
  }
}

// Used in:
//
// ProcElemS1 -> (net)
// (net) -> VertexReaderS0 -> VertexReaderS1 -> VertexUpdater -> (net)
// (net) -> ProcElemS1
using Update = tapa::packet<PeId, TaskOp>;

// Used in:
//
// Dispatcher -> TaskQueue
struct QueueOp {
  enum Op { PUSH, POP } op;
  Task task;  // Valid only when op is PUSH.
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
  Task task;  // Valid only when queue_op is POP and task_op is NEW.
};

inline std::ostream& operator<<(std::ostream& os, const QueueOpResp& obj) {
  os << "{queue_op: ";
  switch (obj.queue_op) {
    case QueueOp::PUSH:
      os << "PUSH";
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
    case TaskOp::DONE:
      os << "DONE";
      break;
  }
  if (obj.queue_op == QueueOp::POP && obj.task_op == TaskOp::NEW) {
    os << ", task: " << obj.task;
  }
  return os << "}";
}

class TaskOnChip {
 public:
  TaskOnChip() {}
  explicit TaskOnChip(const Task& task) {
    data.range(vid_msb, vid_lsb) = task.vid;
    data.range(parent_msb, parent_lsb) = task.vertex.parent;
    data.range(distance_msb, distance_lsb) =
        ap_uint<32>(bit_cast<uint32_t>(task.vertex.distance))
            .range(kFloatMsb, kFloatMsb - kFloatWidth + 1);
    data.range(offset_msb, offset_lsb) = task.vertex.offset;
    data.range(degree_msb, degree_lsb) = task.vertex.degree;
  }
  explicit operator Task() {
    ap_uint<32> distance = 0;
    distance.range(kFloatMsb, kFloatMsb - kFloatWidth + 1) =
        data.range(distance_msb, distance_lsb);
    return {
        .vid = Vid(data.range(vid_msb, vid_lsb)),
        .vertex =
            {
                .parent = Vid(data.range(parent_msb, parent_lsb)),
                .distance = bit_cast<float>(distance.to_uint()),
                .offset = Eid(data.range(offset_msb, offset_lsb)),
                .degree = Vid(data.range(degree_msb, degree_lsb)),
            },
    };
  }

 private:
  ap_uint<144> data;
  static constexpr int kVidWidth = 27;
  static constexpr int kFloatWidth = 31;
  static constexpr int kEidWidth = 32;
  static_assert(kVidWidth * 3 + kFloatWidth + kEidWidth ==
                    decltype(data)::width,
                "invalid TaskOnChip configuration");

  // kFloatWidth bits from kFloatMsb is used.
  static constexpr int kFloatMsb = 30;
  static_assert(kFloatMsb + 1 >= kFloatWidth,
                "invalid TaskOnChip configuration");

  static constexpr int vid_lsb = 0;
  static constexpr int vid_msb = vid_lsb + kVidWidth - 1;
  static constexpr int parent_lsb = vid_msb + 1;
  static constexpr int parent_msb = parent_lsb + kVidWidth - 1;
  static constexpr int distance_lsb = parent_msb + 1;
  static constexpr int distance_msb = distance_lsb + kFloatWidth - 1;
  static constexpr int offset_lsb = distance_msb + 1;
  static constexpr int offset_msb = offset_lsb + kEidWidth - 1;
  static constexpr int degree_lsb = offset_msb + 1;
  static constexpr int degree_msb = degree_lsb + kVidWidth - 1;
};

// Convenient functions and macros.

/// Returns whether singleton @p array contains @p value.
///
/// @tparam T     Type of each element in @p array and @p value.
/// @param array  An array of type @p T[1].
/// @param value  A value of type @p T.
/// @return       True if @p array contains @p value.
template <typename T>
inline bool Contains(const volatile T (&array)[1], const T& value) {
#pragma HLS inline
  return array[0] == value;
}

/// Returns whether @p array contains @p value.
///
/// @tparam T     Type of each element in @p array and @p value.
/// @tparam N     Number of elements in @p array.
/// @param array  An array of type @p T[N].
/// @param value  A value of type @p T.
/// @return       True if @p array contains @p value.
template <typename T, int N>
inline bool Contains(const volatile T (&array)[N], const T& value) {
#pragma HLS inline
  return Contains((const T(&)[N / 2])(array), value) ||
         Contains((const T(&)[N - N / 2])(array[N / 2]), value);
}

inline bool All(const bool (&array)[1]) {
#pragma HLS inline
  return array[0];
}

template <int N>
inline bool All(const bool (&array)[N]) {
#pragma HLS inline
  return All((const bool(&)[N / 2])(array)) &&
         All((const bool(&)[N - N / 2])(array[N / 2]));
}

template <typename T>
inline bool Any(const T (&array)[1]) {
#pragma HLS inline
  return array[0];
}

template <typename T, int N>
inline bool Any(const T (&array)[N]) {
#pragma HLS inline
  return Any((const T(&)[N / 2])(array)) &&
         Any((const T(&)[N - N / 2])(array[N / 2]));
}

template <typename T>
inline void MemSet(T (&array)[1], T value) {
#pragma HLS inline
  array[0] = value;
}

template <typename T, int N>
inline void MemSet(T (&array)[N], T value) {
#pragma HLS inline
  MemSet((T(&)[N / 2])(array), value);
  MemSet((T(&)[N - N / 2])(array[N / 2]), value);
}

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
           mem.read_addr_try_write(read_addr));
    UPDATE(read_data, mem.read_data_try_read(read_data),
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
           mem.write_addr_try_write(write_addr));
    UPDATE(write_data, write_data_q.try_read(write_data),
           mem.write_data_try_write(write_data));
  }
}

// A fully pipelined lightweight proxy for read-write tapa::async_mmap.
template <typename data_t, typename addr_t>
inline void ReadWriteMem(tapa::istream<addr_t>& read_addr_q,
                         tapa::ostream<data_t>& read_data_q,
                         tapa::istream<addr_t>& write_addr_q,
                         tapa::istream<data_t>& write_data_q,
                         tapa::async_mmap<data_t>& mem) {
#pragma HLS inline
  DECL_BUF(addr_t, read_addr);
  DECL_BUF(data_t, read_data);
  DECL_BUF(addr_t, write_addr);
  DECL_BUF(data_t, write_data);

spin:
  for (;;) {
#pragma HLS pipeline II = 1
    UPDATE(read_addr, read_addr_q.try_read(read_addr),
           mem.read_addr_try_write(read_addr));
    UPDATE(read_data, mem.read_data_try_read(read_data),
           read_data_q.try_write(read_data));
    UPDATE(write_addr, write_addr_q.try_read(write_addr),
           mem.write_addr_try_write(write_addr));
    UPDATE(write_data, write_data_q.try_read(write_data),
           mem.write_data_try_write(write_data));
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

#endif  // TAPA_SSSP_KERNEL_H_

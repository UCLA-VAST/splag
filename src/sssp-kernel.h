#ifndef TAPA_SSSP_KERNEL_H_
#define TAPA_SSSP_KERNEL_H_

#include <algorithm>
#include <type_traits>

#include "sssp.h"

// Used in:
//
// VertexMem -> ProcElem
// ProcElem  -> Dispatcher
struct TaskOp {
  // VertexMem -> ProcElem: Valid values are NEW and NOOP.
  // ProcElem  -> Dispatcher: Valid values are NEW and DONE.
  enum Op { NEW, NOOP, DONE = NOOP } op;
  Task task;  // Valid only when op is NEW.
};

inline std::ostream& operator<<(std::ostream& os, const TaskOp& obj) {
  switch (obj.op) {
    case TaskOp::NEW:
      return os << "{ op: NEW, task: " << obj.task << " }";
    default:
      return os << "{ op: NOOP/DONE }";
  }
}

// Used in:
//
// ProcElem -> VertexMem
struct Update {
  Vid vid;
  float weight;
};

inline std::ostream& operator<<(std::ostream& os, const Update& obj) {
  if (obj.weight < bit_cast<float>(0x10000000)) {
    // Weights are evenly distributed between 0 and 1; if it is this small,
    // almost certainly it should actually be interpreted as Vid.
    return os << "{ src: " << obj.vid
              << ", count: " << bit_cast<Vid>(obj.weight) << " }";
  }
  return os << "{ dst: " << obj.vid << ", weight: " << obj.weight << " }";
}

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
  Vid queue_size;
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
  }
  if (obj.queue_op == QueueOp::POP && obj.task_op == TaskOp::NEW) {
    os << ", task: " << obj.task;
  }
  return os << "}";
}

// Constants and types.
constexpr int kVertexPartitionFactor =
    kEdgeVecLen > kVertexVecLen ? kEdgeVecLen : kVertexVecLen;

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

#endif  // TAPA_SSSP_KERNEL_H_

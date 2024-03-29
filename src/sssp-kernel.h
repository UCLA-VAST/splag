#ifndef TAPA_SSSP_KERNEL_H_
#define TAPA_SSSP_KERNEL_H_

#include <cstddef>

#include <ap_int.h>

#include <algorithm>
#include <type_traits>

#include "sssp.h"

using uint_vid_t = ap_uint<kVidWidth>;

using uint_eid_t = ap_uint<kEidWidth>;

using uint_deg_t = ap_uint<kDegreeWidth>;

struct VertexNoop {
  using uint_vertex_noop_t = ap_uint<bit_length(kSubIntervalCount)>;

  uint_vertex_noop_t push_count;
  uint_vertex_noop_t pop_count;
};

struct SourceVertex {
  Vid vid;
  Vid parent;
  float distance;
};

using EdgeReq = tapa::packet<Eid, SourceVertex>;

using TaskVec = tapa::vec_t<TaskOnChip, kEdgeVecLen>;

struct TaskCount {
  ap_uint<bit_length(kPeCount)> old_task_count;
  uint_eid_t new_task_count;
};

struct VertexCacheEntry {
  bool is_valid;
  bool is_reading;
  bool is_writing;
  bool is_dirty;
  bool is_push;

  VertexCacheEntry() {}

  VertexCacheEntry(std::nullptr_t) {
    is_valid = false;
    is_reading = false;
    is_writing = false;
    is_dirty = false;
    is_push = false;
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
    CHECK_LT(vertex.offset, 1 << decltype(offset_)::width);
    CHECK_LT(vertex.degree, 1 << decltype(degree_)::width);
    offset_ = vertex.offset;
    degree_ = vertex.degree;
  }

 private:
  uint_vid_t vid_;
  uint_vid_t parent_;
  float distance_;
  uint_eid_t offset_;
  uint_deg_t degree_;
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

template <int divisor, int width>
ap_uint<width - log2(divisor)> div(ap_uint<width> dividend) {
  static_assert(is_power_of(divisor, 2));
  constexpr int divisor_width = log2(divisor);
  return dividend.range(width - 1, divisor_width);
}

template <int divisor, int width>
ap_uint<log2(divisor)> mod(ap_uint<width> dividend) {
  static_assert(is_power_of(divisor, 2));
  constexpr int divisor_width = log2(divisor);
  return dividend.range(divisor_width - 1, 0);
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

  template <typename T, uint64_t S, typename pos_t>
  static bool find_max_non_empty(tapa::istreams<T, S>& in_qs, T& elem,
                                 pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= S, "begin + len must <= S");
    pos_t pos_left, pos_right;
    T elem_left, elem_right;
#pragma HLS aggregate variable = elem_left bit
#pragma HLS aggregate variable = elem_right bit
    const auto is_left_non_empty =
        arbiter<begin, len / 2>::find_max_non_empty(in_qs, elem_left, pos_left);
    const auto is_right_non_empty =
        arbiter<begin + len / 2, len - len / 2>::find_max_non_empty(
            in_qs, elem_right, pos_right);
    const bool is_left_chosen =
        is_left_non_empty && (!is_right_non_empty || (elem_right < elem_left));
    pos = is_left_chosen ? pos_left : pos_right;
    elem = is_left_chosen ? elem_left : elem_right;
    return is_left_non_empty || is_right_non_empty;
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
    if (max_right < max_left) {
      pos = pos_left;
      return max_left;
    }
    pos = pos_right;
    return max_right;
  }

  template <typename T, int N, typename pos_t>
  static T find_min(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(len > 1, "len must > 1");
    static_assert(begin + len <= N, "begin + len must <= N");
    pos_t pos_left, pos_right;
    const auto min_left = arbiter<begin, len / 2>::find_min(array, pos_left);
    const auto min_right =
        arbiter<begin + len / 2, len - len / 2>::find_min(array, pos_right);
    if (min_left < min_right) {
      pos = pos_left;
      return min_left;
    }
    pos = pos_right;
    return min_right;
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

  template <typename T, uint64_t S, typename pos_t>
  static bool find_max_non_empty(tapa::istreams<T, S>& in_qs, T& elem,
                                 pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < S, "begin must < S");
    pos = begin;
    return in_qs[begin].try_peek(elem);
  }

  template <typename T, int N, typename pos_t>
  static T find_max(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
    static_assert(begin >= 0, "begin must >= 0");
    static_assert(begin < N, "beginmust < N");
    pos = begin;
    return array[begin];
  }

  template <typename T, int N, typename pos_t>
  static T find_min(const T (&array)[N], pos_t& pos) {
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

/// Find the maximum non-empty istream.
///
/// @param[in] in_qs    Input streams.
/// @param[out] elem    The maximum peeked value, invalid if none found.
/// @param[out] pos     Position of the maximum non-empty istream, invalid if
///                     none found.
/// @return             Whether a non-empty istream is found.
template <typename T, uint64_t S, typename pos_t>
inline bool find_max_non_empty(tapa::istreams<T, S>& in_qs, T& elem,
                               pos_t& pos) {
#pragma HLS inline
#pragma HLS aggregate variable = elem bit
  return arbiter<0, S>::find_max_non_empty(in_qs, elem, pos);
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

/// Find the minimum value in a **completely partitioned** array.
///
/// @param[in] array  A completely partitioned array.
/// @param[out] pos   Position of the maximum value.
/// @return           The minimum value.
template <typename T, int N, typename pos_t>
inline T find_min(const T (&array)[N], pos_t& pos) {
#pragma HLS inline
  return arbiter<0, N>::find_min(array, pos);
}

/// Find the minimum value in a **completely partitioned** array.
///
/// @param[in] array  A completely partitioned array.
/// @return           The minimum value.
template <typename T, int N>
inline T find_min(const T (&array)[N]) {
#pragma HLS inline
  ap_uint<bit_length(N - 1)> pos;
  return arbiter<0, N>::find_min(array, pos);
}

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
        const auto iid = assume_mod(base % in_n, out_n, oid);
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

/// Transpose a MxN stream array to a NxM stream array, effectively changing the
/// stride from 1 to M.
template <int M, int N, typename T, typename Checker>
void Transpose(tapa::istreams<T, M * N>& in_q, tapa::ostreams<T, M * N>& out_q,
               Checker checker) {
#pragma HLS inline
spin:
  for (;;) {
#pragma HLS pipeline II = 1
    RANGE(i, M, {
      RANGE(j, N, {
        if (!in_q[i * N + j].empty() && !out_q[j * M + i].full()) {
          const auto elem = in_q[i * N + j].read(nullptr);
          checker(elem, i * N + j, j * M + i);
          out_q[j * M + i].try_write(elem);
        }
      });
    });
  }
}

void Switch2x2Impl(int b, tapa::istream<TaskOnChip>& in_q0,
                   tapa::istream<TaskOnChip>& in_q1,
                   tapa::ostreams<TaskOnChip, 2>& out_q) {
  bool should_prioritize_1 = false;

spin:
  for (bool is_pkt_0_valid, is_pkt_1_valid;;) {
#pragma HLS pipeline II = 1
#pragma HLS latency max = 0
    const auto pkt_0 = in_q0.peek(is_pkt_0_valid);
    const auto pkt_1 = in_q1.peek(is_pkt_1_valid);

    const uint_vid_t addr_0 = pkt_0.vid();
    const uint_vid_t addr_1 = pkt_1.vid();

    const bool should_fwd_0_0 = is_pkt_0_valid && !addr_0.get_bit(b);
    const bool should_fwd_0_1 = is_pkt_0_valid && addr_0.get_bit(b);
    const bool should_fwd_1_0 = is_pkt_1_valid && !addr_1.get_bit(b);
    const bool should_fwd_1_1 = is_pkt_1_valid && addr_1.get_bit(b);

    const bool has_conflict = is_pkt_0_valid && is_pkt_1_valid &&
                              should_fwd_0_0 == should_fwd_1_0 &&
                              should_fwd_0_1 == should_fwd_1_1;

    const bool should_read_0 = !((!should_fwd_0_0 && !should_fwd_0_1) ||
                                 (should_prioritize_1 && has_conflict));
    const bool should_read_1 = !((!should_fwd_1_0 && !should_fwd_1_1) ||
                                 (!should_prioritize_1 && has_conflict));
    const bool should_write_0 = should_fwd_0_0 || should_fwd_1_0;
    const bool should_write_1 = should_fwd_1_1 || should_fwd_0_1;
    const bool shoud_write_0_0 =
        should_fwd_0_0 && (!should_fwd_1_0 || !should_prioritize_1);
    const bool shoud_write_1_1 =
        should_fwd_1_1 && (!should_fwd_0_1 || should_prioritize_1);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, check for conflict
    const bool is_0_written =
        should_write_0 && out_q[0].try_write(shoud_write_0_0 ? pkt_0 : pkt_1);
    const bool is_1_written =
        should_write_1 && out_q[1].try_write(shoud_write_1_1 ? pkt_1 : pkt_0);

    // if can forward through (0->0 or 1->1), do it
    // otherwise, round robin priority of both ins
    if (should_read_0 && (shoud_write_0_0 ? is_0_written : is_1_written)) {
      in_q0.read(nullptr);
    }
    if (should_read_1 && (shoud_write_1_1 ? is_1_written : is_0_written)) {
      in_q1.read(nullptr);
    }

    if (has_conflict) {
      should_prioritize_1 = !should_prioritize_1;
    }
  }
}

#endif  // TAPA_SSSP_KERNEL_H_

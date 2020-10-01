#include <cstddef>
#include <cstring>

#include <iostream>
#include <string_view>
#include <type_traits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <glog/logging.h>

class MappedFile {
 public:
  MappedFile(std::string_view file_name) : name_(file_name) {
    if (fd_ = open(name_.c_str(), O_RDONLY); fd_ < 0) {
      LOG(FATAL) << "failed to open '" << name_
                 << "': " << std::strerror(errno);
    }

    struct stat stat_buf;
    if (fstat(fd_, &stat_buf) != 0) {
      LOG(FATAL) << "failed to stat '" << name_
                 << "': " << std::strerror(errno);
    }
    size_ = stat_buf.st_size;

    if (ptr_ = mmap(/*addr=*/nullptr, size_, PROT_READ, MAP_PRIVATE, fd_,
                    /*offset=*/0);
        ptr_ == MAP_FAILED) {
      LOG(FATAL) << "failed to mmap '" << name_
                 << "': " << std::strerror(errno);
    }
  }

  ~MappedFile() {
    if (ptr_ != nullptr) {
      if (munmap(ptr_, size_) != 0) {
        LOG(WARNING) << "failed to munmap '" << name_
                     << "': " << std::strerror(errno);
      }
      ptr_ = nullptr;
      size_ = 0;
    }

    if (fd_ != -1) {
      if (close(fd_) != 0) {
        LOG(WARNING) << "failed to close '" << name_
                     << "': " << std::strerror(errno);
      }
      fd_ = -1;
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile(MappedFile&&) = delete;

  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile& operator=(MappedFile&&) = delete;

  const std::string& name() { return name_; }

  std::string_view view() { return get_view<char>(); }
  template <typename T>
  std::basic_string_view<T> get_view() {
    static_assert(std::is_standard_layout_v<T>);
    return std::basic_string_view(reinterpret_cast<T*>(ptr_),
                                  size_ / sizeof(T));
  }

 private:
  std::string name_;
  int fd_ = -1;
  void* ptr_ = nullptr;
  size_t size_ = 0;
};

class PackedEdge {
 public:
  int64_t v0() const {
    return v0_low |
           (static_cast<int64_t>(static_cast<int16_t>(high & 0xFFFF)) << 32);
  }

  int64_t v1() const {
    return v1_low |
           (static_cast<int64_t>(static_cast<int16_t>(high >> 16)) << 32);
  }

  void set_edge(int64_t v0, int64_t v1) {
    v0_low = static_cast<uint32_t>(v0);
    v1_low = static_cast<uint32_t>(v1);
    high = static_cast<uint32_t>(((v0 >> 32) & 0xFFFF) |
                                 (((v1 >> 32) & 0xFFFF) << 16));
  }

 private:
  uint32_t v0_low;
  uint32_t v1_low;
  uint32_t high;  // v1_high in high half, v0_high in low half
};

inline std::ostream& operator<<(std::ostream& os, const PackedEdge& e) {
  return os << e.v0() << " " << e.v1();
}

class WeightedPackedEdge : public PackedEdge {
 public:
  float weight;
};

inline std::ostream& operator<<(std::ostream& os, const WeightedPackedEdge& e) {
  return os << static_cast<PackedEdge>(e) << " " << e.weight;
}

using PackedEdgesView = std::basic_string_view<PackedEdge>;
using WeightsView = std::basic_string_view<float>;

template <typename Container, typename T = typename Container::value_type>
inline T geo_mean(const Container& container) {
  T log_product = 0.;
  int64_t count = 0;
  for (auto value : container) {
    log_product += log(value);
    ++count;
  }
  return exp(log_product / count);
}

template <typename Container, typename T = typename Container::value_type>
inline T average(const Container& container) {
  T sum = 0.;
  int64_t count = 0;
  for (auto value : container) {
    sum += value;
    ++count;
  }
  return sum / count;
}

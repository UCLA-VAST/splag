#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <string>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <frt.h>
#include <tapa.h>

#include "sssp.h"
#include "util.h"

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::mmap<Edge> edges, tapa::mmap<Index> indices,
          tapa::mmap<Vertex> vertices, tapa::mmap<float> distances,
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  auto kernel_time_ns_raw =
      mmap(nullptr, sizeof(int64_t), PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
  PCHECK(kernel_time_ns_raw != MAP_FAILED);
  auto deleter = [](int64_t* p) { PCHECK(munmap(p, sizeof(int64_t)) == 0); };
  std::unique_ptr<int64_t, decltype(deleter)> kernel_time_ns(
      reinterpret_cast<int64_t*>(kernel_time_ns_raw), deleter);
  if (pid_t pid = fork()) {
    // Parent.
    PCHECK(pid != -1);
    int status = 0;
    CHECK_EQ(wait(&status), pid);
    CHECK(WIFEXITED(status));
    CHECK_EQ(WEXITSTATUS(status), EXIT_SUCCESS);

    setenv("KERNEL_TIME_NS", std::to_string(*kernel_time_ns).c_str(),
           /*replace=*/1);
    return;
  }

  // Child.
  auto instance = fpga::Instance(getenv("BITSTREAM"));
  auto metadata_arg = fpga::ReadWrite(metadata.get(), metadata.size());
  auto edges_arg = fpga::WriteOnly(edges.get(), edges.size());
  auto indices_arg = fpga::WriteOnly(indices.get(), indices.size());
  auto vertices_arg = fpga::ReadWrite(vertices.get(), vertices.size());
  auto distances_arg = fpga::WriteOnly(distances.get(), distances.size());
  auto heap_array_arg = fpga::Placeholder(heap_array.get(), heap_array.size());
  auto heap_index_arg = fpga::Placeholder(heap_index.get(), heap_index.size());

  int arg_idx = 0;
  instance.SetArg(arg_idx++, vertex_count);
  instance.SetArg(arg_idx++, root);
  instance.AllocBuf(arg_idx, metadata_arg);
  instance.SetArg(arg_idx++, metadata_arg);
  instance.AllocBuf(arg_idx, edges_arg);
  instance.SetArg(arg_idx++, edges_arg);
  instance.AllocBuf(arg_idx, indices_arg);
  instance.SetArg(arg_idx++, indices_arg);
  instance.AllocBuf(arg_idx, vertices_arg);
  instance.SetArg(arg_idx++, vertices_arg);
  instance.AllocBuf(arg_idx, distances_arg);
  instance.SetArg(arg_idx++, distances_arg);
  instance.AllocBuf(arg_idx, heap_array_arg);
  instance.SetArg(arg_idx++, heap_array_arg);
  instance.AllocBuf(arg_idx, heap_index_arg);
  instance.SetArg(arg_idx++, heap_index_arg);

  instance.WriteToDevice();
  instance.Exec();
  instance.ReadFromDevice();
  instance.Finish();

  *kernel_time_ns = instance.ComputeTimeNanoSeconds();
  exit(EXIT_SUCCESS);
}

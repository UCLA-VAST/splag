#include <stdlib.h>

#include <algorithm>
#include <string>

#include <frt.h>
#include <tapa.h>

#include "sssp.h"

void SSSP(Vid vertex_count, Vid root, tapa::mmap<int64_t> metadata,
          tapa::mmap<Edge> edges, tapa::mmap<Index> indices,
          tapa::mmap<Vid> parents, tapa::mmap<float> distances,
          tapa::mmap<Task> heap_array, tapa::mmap<Vid> heap_index) {
  auto instance = fpga::Instance(getenv("BITSTREAM"));
  auto metadata_arg = fpga::ReadWrite(metadata.get(), metadata.size());
  auto edges_arg = fpga::WriteOnly(edges.get(), edges.size());
  auto indices_arg = fpga::WriteOnly(indices.get(), indices.size());
  auto parents_arg = fpga::ReadWrite(parents.get(), parents.size());
  auto distances_arg = fpga::ReadWrite(distances.get(), distances.size());
  auto heap_array_arg = fpga::ReadWrite(heap_array.get(), heap_array.size());
  auto heap_index_arg = fpga::ReadWrite(heap_index.get(), heap_index.size());

  int arg_idx = 0;
  instance.SetArg(arg_idx++, vertex_count);
  instance.SetArg(arg_idx++, root);
  instance.AllocBuf(arg_idx, metadata_arg);
  instance.SetArg(arg_idx++, metadata_arg);
  instance.AllocBuf(arg_idx, edges_arg);
  instance.SetArg(arg_idx++, edges_arg);
  instance.AllocBuf(arg_idx, indices_arg);
  instance.SetArg(arg_idx++, indices_arg);
  instance.AllocBuf(arg_idx, parents_arg);
  instance.SetArg(arg_idx++, parents_arg);
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

  setenv("KERNEL_TIME_NS",
         std::to_string(instance.ComputeTimeNanoSeconds()).c_str(),
         /*replace=*/1);
}

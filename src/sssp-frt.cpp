#include <stdlib.h>

#include <algorithm>
#include <string>

#include <frt.h>
#include <tapa.h>

#include "sssp.h"

void SSSP(Iid interval_count, Vid interval_size, Vid root,
          tapa::mmap<uint64_t> metadata, tapa::async_mmap<VidVec> parents,
          tapa::async_mmap<FloatVec> distances,
          tapa::async_mmaps<EdgeVec, kPeCount> edges,
          tapa::async_mmaps<UpdateVec, kPeCount> updates) {
  auto instance = fpga::Instance(getenv("BITSTREAM"));
  auto metadata_arg = fpga::ReadWrite(metadata.get(), metadata.size());
  auto parents_arg = fpga::ReadWrite(parents.get(), parents.size());
  auto distances_arg = fpga::ReadWrite(distances.get(), distances.size());
  int arg_idx = 0;
  instance.SetArg(arg_idx++, interval_count);
  instance.SetArg(arg_idx++, interval_size);
  instance.SetArg(arg_idx++, root);
  instance.AllocBuf(arg_idx, metadata_arg);
  instance.SetArg(arg_idx++, metadata_arg);
  instance.AllocBuf(arg_idx, parents_arg);
  instance.SetArg(arg_idx++, parents_arg);
  instance.AllocBuf(arg_idx, distances_arg);
  instance.SetArg(arg_idx++, distances_arg);
  for (int i = 0; i < kPeCount; ++i) {
    auto edges_arg = fpga::WriteOnly(edges[i].get(), edges[i].size());
    auto updates_arg = fpga::ReadOnly(updates[i].get(), updates[i].size());
    instance.AllocBuf(arg_idx + i, edges_arg);
    instance.SetArg(arg_idx + i, edges_arg);
    instance.AllocBuf(arg_idx + kPeCount + i, updates_arg);
    instance.SetArg(arg_idx + kPeCount + i, updates_arg);
  }

  instance.WriteToDevice();
  instance.Exec();
  instance.ReadFromDevice();
  instance.Finish();

  setenv("KERNEL_TIME_NS",
         std::to_string(instance.ComputeTimeNanoSeconds()).c_str(),
         /*replace=*/1);
}

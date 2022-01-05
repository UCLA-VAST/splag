# Accelerating <ins>S</ins>SSP for <ins>P</ins>ower-<ins>La</ins>w <ins>G</ins>raphs

SPLAG is an FPGA accelerator for the single-source shortest path (SSSP) problem, featuring:

+ A coarse-grained priority queue (CGPQ) that enables high-throughput priority-order graph traversal with a large frontier
+ A customized vertex cache (CVC) that reduces off-chip memory access and improves the random-access throughput to read and update vertex data
+ Outstanding performance & energy efficiency on a single U280 FPGA
  + Up to a 4.9× speedup over state-of-the-art SSSP accelerators
  + Up to a 2.6× speedup over 32-thread CPU running at 4.4 GHz
  + Up to a 0.9× speedup over an A100 GPU (that has 4.1× power budget and 3.4× HBM bandwidth)
  + Could be placed in the 14th position of the [Graph 500 benchmark](https://graph500.org/?page_id=944) for data intensive applications
    + The highest using a single FPGA with only a 45 W power budget
+ Open-source and fully parameterized [TAPA](https://github.com/UCLA-VAST/tapa) HLS C++ implementation
  + Easily portable to a different FPGA with a different configuration


## Prerequisites

+ [TAPA](https://github.com/UCLA-VAST/tapa)
+ [AutoBridge](https://github.com/Licheng-Guo/AutoBridge)
+ Xilinx Alveo U280 FPGA and its `xilinx_u280_xdma_201920_3` shell platform

## Getting Started

### Obtaining the Source Code

```bash
git clone https://github.com/UCLA-VAST/splag.git
cd splag
```

### Running Software Simulation

```bash
mkdir build
cd build
cmake ..
make sssp
./sssp ../data/graph500-scale-5
```

### Running High-Level Synthesis

```bash
make sssp-hw-xo
```

### Running Hardware Simulation

```bash
make sssp-cosim
```

### Building FPGA Bitstream

```bash
make SSSP.xilinx_u280_xdma_201920_3.hw_xclbin
```

### Running on Board

```bash
./sssp -v=3 --bitstream=SSSP.xilinx_u280_xdma_201920_3.hw.xclbin ../data/graph500-scale-5
```

## Data Format

Currently, SPLAG takes as input the same binary edge list format as [Graph 500](https://github.com/graph500/graph500).
Each dataset is stored as two separate files in the same directory, e.g., `dataset` and `dataset.weights`.
The `dataset` file is an array of [`PackedEdge`](https://github.com/UCLA-VAST/splag/blob/master/src/util.h#L80).
The `dataset.weights` file is an array of 32-bit `float`.
The dataset is assumed to be undirected.

See the [release page](https://github.com/UCLA-VAST/splag/releases/tag/fpga22) for more datasets and instructions on reproducing the experimental results in the [FPGA'22 paper](https://about.blaok.me/publication/splag).

## Publication

+ Yuze Chi, Licheng Guo, Jason Cong. Accelerating SSSP for Power-Law Graphs. In FPGA, 2022. [[PDF]](https://about.blaok.me/pub/fpga22-splag.pdf) [[Code]](https://github.com/UCLA-VAST/splag)


ARG CUDA_VERSION=11.3

FROM ubuntu:18.04 as u280-builder
# Requires an official XRT build from Xilinx
ARG XRT_VERSION=202110.2.11.634
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gcc-multilib \
  git \
  language-pack-en \
  libc6-dev-i386 \
  libjpeg-dev \
  libpython3-dev \
  libtinfo5 \
  libx11-6 \
  python3-numpy \
  python3-pip \
  python3-setuptools \
  unzip \
  zlib1g-dev \
  && curl --progress-bar -sSL \
  -o /tmp/xrt.deb \
  https://www.xilinx.com/bin/public/openDownload?filename=xrt_${XRT_VERSION}_18.04-amd64-xrt.deb \
  && curl --progress-bar -sSL \
  -o /tmp/frt.deb \
  https://github.com/Blaok/fpga-runtime/raw/gh-pages/pool/main/bionic/libfrt-dev_0.0.20211001.1_amd64.deb \
  && curl --progress-bar -sSL \
  -o /tmp/tapa.deb \
  https://github.com/Blaok/tapa/raw/gh-pages/pool/main/bionic/tapa_0.0.20211016.1_amd64.deb \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  -f /tmp/*.deb \
  && python3 -m pip --no-cache-dir install absl-py cmake matplotlib scipy tapa==0.0.20211016.1 \
  && apt remove -y \
  python3-pip \
  python3-setuptools \
  && apt autoremove -y \
  && rm -rf /var/lib/apt/lists/* /tmp/*.deb
ENV XILINX_XRT=/opt/xilinx/xrt
WORKDIR /usr/src/cnpy
RUN git clone https://github.com/rogersce/cnpy.git . \
  && git reset --hard 4e8810b1a8637695171ed346ce68f6984e585ef4 \
  && mkdir build \
  && cd build \
  && cmake .. -DCMAKE_BUILD_TYPE=Release \
  && make \
  && make install \
  && ldconfig \
  && rm -rf "${PWD}"
WORKDIR /usr/src/splag
RUN git clone https://github.com/UCLA-VAST/splag.git . \
  && git reset --hard af451dcdd465d766967bdf45ab8205398b635097
CMD scripts/build.sh build

FROM u280-builder as vu5p-builder
RUN git reset --hard 65ff67cf40042732b27842c092068fc69356359f

FROM u280-builder as u250-builder
RUN git reset --hard 99b064ebac367066d27741487f0eb1bdc8366e15

FROM u280-builder as u280-runner
RUN cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release \
  && make -C build sssp
RUN mkdir output
CMD scripts/run.sh data build output

FROM vu5p-builder as vu5p-runner
RUN cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release \
  && make -C build sssp
CMD scripts/run.sh data build output

FROM u250-builder as u250-runner
RUN cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release \
  && make -C build sssp
CMD scripts/run.sh data build output

FROM ubuntu:18.04 as cpu-builder
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  libboost-iostreams-dev \
  libboost-serialization-dev \
  libfmt-dev \
  libnuma-dev \
  llvm-10-dev \
  && curl --progress-bar -sSL \
  -o /tmp/cmake.sh \
  https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh \
  && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
  && rm -rf /var/lib/apt/lists/* /tmp/cmake.sh
WORKDIR /usr/src/galois
RUN git clone https://github.com/UCLA-VAST/Galois . \
  && git reset --hard 351402ff84ddbe10de7a0faf4b467887cb2342c4
RUN cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release \
  && make -C build sssp-cpu
WORKDIR /usr/src/splag
RUN git clone https://github.com/UCLA-VAST/splag.git . \
  && git reset --hard af451dcdd465d766967bdf45ab8205398b635097

FROM ubuntu:18.04 as cpu-runner
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  libnuma-dev \
  zsh \
  && rm -rf /var/lib/apt/lists/*
COPY --from=cpu-builder \
  /usr/src/galois/build/lonestar/analytics/cpu/sssp/sssp-cpu \
  /usr/src/galois/build/lonestar/analytics/cpu/sssp/sssp-cpu
COPY --from=cpu-builder \
  /usr/src/splag/scripts/run-cpu.sh \
  /usr/src/splag/scripts/run-cpu.sh
WORKDIR /usr/src/splag
CMD scripts/run-cpu.sh \
  /usr/src/galois/build/lonestar/analytics/cpu/sssp/sssp-cpu \
  data \
  output/hlwd-09.log output/rmat-21.log

FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu18.04 as gpu-builder
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git \
  libboost-dev \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/adds
RUN git clone https://github.com/UCLA-VAST/adds .
RUN make -C ads_float sssp
WORKDIR /usr/src/splag
RUN git clone https://github.com/UCLA-VAST/splag.git . \
  && git reset --hard af451dcdd465d766967bdf45ab8205398b635097

FROM nvidia/cuda:${CUDA_VERSION}.0-runtime-ubuntu18.04 as gpu-runner
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  zsh \
  && rm -rf /var/lib/apt/lists/*
COPY --from=gpu-builder \
  /usr/src/adds/ads_float/sssp \
  /usr/src/adds/ads_float/sssp
COPY --from=gpu-builder \
  /usr/src/splag/scripts/run-gpu.sh \
  /usr/src/splag/scripts/run-gpu.sh
WORKDIR /usr/src/splag
CMD scripts/run-gpu.sh \
  ../adds/ads_float/sssp \
  data \
  output/hlwd-09.log output/rmat-21.log

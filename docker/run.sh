#!/bin/bash
# Reproduces all experiments in the FPGA'22 paper in docker.
#
# Running FPGA experiments takes ~1 h. It requires:
#   0. valid docker installation, and
#   1. valid XRT installation of a recent official version from Xilinx, and
#   2. U280 board with xilinx_u280_xdma_201920_3 platform, or
#   3. U250 board with xilinx_u250_xdma_201830_2 platform.
#
# Running CPU experiments takes ~4 min. It requires:
#   0. valid docker installation, and
#   1. logs from the FPGA experiments in `output`.
#
# Running GPU experiments takes ~8 min. It requires:
#   0. valid docker installation with CUDA support, and
#   1. valid `nvidia-smi` command, and
#   2. logs from the FPGA experiments in `output`.
#
# Building the bitstreams takes ~14 h. It requires:
#   0. valid docker installation, and
#   1. valid Xilinx Vitis HLS 2020.2, and
#   2. valid Xilinx Vivado 2021.1.

set -eo pipefail

base_dir="$(realpath "${0%/*}")"
item="$1"
data_dir="${base_dir}"/data
output_dir="${base_dir}"/output
build_dir="${base_dir}/${item/-//}"

build_argv=()
run_argv=()

case "${item}" in
fpga | build-u280 | build-vu5p | build-u250)
  items=(u280 vu5p u250)

  xrt_version="$(cat /sys/bus/pci/drivers/xocl/module/version | cut -d, -f1)"
  declare -A xrt_versions
  xrt_versions=(
    [2.6.655]=202010.2.6.655
    [2.7.766]=202010.2.7.766
    [2.8.743]=202020.2.8.743
    [2.9.317]=202020.2.9.317
    [2.11.634]=202110.2.11.634
    [2.12.427]=202120.2.12.427
  )
  xrt_version="${xrt_versions[${xrt_version}]}"
  if [[ -z "${xrt_version}" ]]; then
    echo "ERROR: unsupported XRT version: ${xrt_version}" >&2
    exit 1
  fi
  build_argv+=(--build-arg=XRT_VERSION="${xrt_version}")

  if [[ "${item}" = fpga ]]; then
    for dev in /dev/xclmgmt* /dev/xfpga/* /dev/dri/renderD*; do
      run_argv+=(--device="${dev}:${dev}")
    done
  else
    items=("${item#build-}")
  fi

  img=splag-fpga-runners.xrt-${xrt_version#*.}.tgz
  ;;
cpu)
  items=(cpu)

  img=splag-cpu-runner.tgz
  ;;
gpu)
  items=(gpu)

  cuda_version="$(nvidia-smi | grep -o 'CUDA Version: [0-9]\+\.[0-9]\+')"
  cuda_version="${cuda_version#CUDA Version: }"

  build_argv+=(--build-arg=CUDA_VERSION="${cuda_version}")

  run_argv+=(--gpus=all)

  img=splag-gpu-runner.cuda-${cuda_version}.tgz
  ;;
*)
  echo "usage: $0 <fpga|cpu|gpu|build-u280|build-vu5p|build-u250>" >&2
  exit 1
  ;;
esac

if [[ "${item}" == build-* ]]; then
  job=builder

  if [[ ! -d "${XILINX_HLS}" ]]; then
    echo "ERROR: please environment variable 'XILINX_HLS'" >&2
    exit 1
  fi

  if [[ ! -d "${XILINX_VIVADO}" ]]; then
    echo "ERROR: please environment variable 'XILINX_VIVADO'" >&2
    exit 1
  fi

  platform_base=/opt/xilinx/platforms
  platforms=(xilinx_u280_xdma_201920_3 xilinx_u250_xdma_201830_2)

  for platform in "${platforms[@]}"; do
    if [[ ! -d "${platform_base}/${platform}" ]]; then
      echo "ERROR: please install platform '${platform}'" >&2
      exit 1
    fi
    run_argv+=(--volume="${platform_base}/${platform}:${platform_base}/${platform}")
  done

  if [[ "${XILINX_HLS%/}" != *2020.2 ]]; then
    echo "ERROR: Vitis 2020.2 is required" >&2
    exit 2
  fi

  if [[ "${XILINX_VIVADO%/}" != *2021.1 ]]; then
    echo "WARNING: Vivado 2021.1 was used in the FPGA'22 paper, yet a different version is used right now" >&2
  fi

  export XILINX_VITIS="$(
    source "${XILINX_VIVADO}"/settings64.sh && echo "${XILINX_VITIS}"
  )"
  xilinx_vitis_hls="$(
    source "${XILINX_VIVADO}"/settings64.sh && echo "${XILINX_HLS}"
  )"

  run_argv+=(
    --env=CPATH="${XILINX_HLS}"/include
    --env=XILINX_HLS
    --env=XILINX_VITIS
    --env=XILINX_VIVADO
    --volume="${XILINX_HLS}:${XILINX_HLS}"
    --volume="${XILINX_VITIS}:${XILINX_VITIS}"
    --volume="${XILINX_VIVADO}:${XILINX_VIVADO}"
    --volume="${xilinx_vitis_hls}:${xilinx_vitis_hls}"
    --volume="${build_dir}":/usr/src/splag/build
  )

  mkdir -p "${build_dir}"
else
  job=runner

  run_argv+=(
    --volume="${data_dir}":/usr/src/splag/data
    --volume="${output_dir}":/usr/src/splag/output
  )

  mkdir -p "${data_dir}"
  mkdir -p "${output_dir}"
fi

export DOCKER_BUILDKIT=1

if [[ -r "${img}" ]]; then
  gzip -cd <"${base_dir}/${img}" | docker load
else
  for item in "${items[@]}"; do
    argv docker build \
      --target="${item}-${job}" \
      --tag=splag-"${item}-${job}" \
      --network=host \
      --force-rm \
      "${build_argv[@]}" \
      - <"${base_dir}"/Dockerfile
  done
fi

for item in "${items[@]}"; do
  docker run \
    --name=splag-"${item}-${job}" \
    --interactive \
    --rm \
    --tty \
    --network=host \
    --env=HOME \
    --env=USER \
    --tmpfs="${HOME}" \
    --user="$(id -u):$(id -g)" \
    "${run_argv[@]}" \
    splag-"${item}-${job}"
done

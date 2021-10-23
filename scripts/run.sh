#!/bin/bash
set -eo pipefail
base_dir="$(realpath "${0%/*}")"
data_dir="$(realpath "$1")"
build_dir="$(realpath "$2")"
output_dir="$(realpath "$3")"

datasets=(
  hlwd-09
)

files=("${datasets[@]}")
for file in "${files[@]}"; do
  files+=("${file}".weights)
done
files+=(SSSP.xilinx_u250_xdma_201830_2.hw.xclbin)

mkdir -p "${data_dir}"
for file in "${files[@]}"; do
  if [[ ! -f "${data_dir}/${file}" ]]; then
    curl -L \
      "https://github.com/UCLA-VAST/splag/releases/download/fpga22/${file}.gz" |
      gzip -d >"${data_dir}"/.partial
    mv "${data_dir}"/.partial "${data_dir}/${file}"
  fi
done

for dataset in "${datasets[@]}"; do
  if [[ ! -f "${output_dir}/${dataset}.u250.log" ]]; then
    case "${dataset}" in
    amzn)
      # amzn has discrete edge weights so handle it differently
      args=(--min_distance=0.2 --max_distance=25.6)
      ;;
    hlwd-09)
      # hlwd-09 needs a greater upper bound; otherwise there are many outliers
      args=(--max_distance=3)
      ;;
    *)
      args=()
      ;;
    esac
    "${build_dir}"/sssp \
      -v=3 \
      --bitstream="${data_dir}/${files[-1]}" \
      "${args[@]}" \
      "${data_dir}/${dataset}" |&
      tee "${output_dir}/.partial.log"
    mv "${output_dir}/.partial.log" "${output_dir}/${dataset}.u250.log"
  fi
done

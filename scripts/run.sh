#!/bin/bash
set -eo pipefail
base_dir="$(realpath "${0%/*}")"
data_dir="$(realpath "$1")"
build_dir="$(realpath "$2")"
output_dir="$(realpath "$3")"

datasets=(
  amzn
  dblp
  digg
  flickr
  g500-15
  g500-16
  g500-17
  g500-18
  g500-19
  g500-20
  g500-21
  g500-22
  hlwd-09
  orkut
  rmat-21
  wiki
  youtube
)

files=(g500-13 g500-14 road-ny road-col road-fla "${datasets[@]}")
for file in "${files[@]}"; do
  files+=("${file}".weights)
done
files+=(metadata.json SSSP.xilinx_u280_xdma_201920_3.hw.xclbin)

mkdir -p "${data_dir}"
for file in "${files[@]}"; do
  if [[ ! -f "${data_dir}/${file}" ]]; then
    curl -L \
      "https://github.com/UCLA-VAST/splag/releases/download/fpga22/${file}.gz" |
      gzip -d >"${data_dir}"/.partial
    mv "${data_dir}"/.partial "${data_dir}/${file}"
  fi
done

logs=()
for dataset in "${datasets[@]}"; do
  if [[ ! -f "${output_dir}/${dataset}.log" ]]; then
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
    mv "${output_dir}/.partial.log" "${output_dir}/${dataset}.log"
  fi
  logs+=("${output_dir}/${dataset}.log")
done

if [[ ! -f "${output_dir}"/g500-15.bucket-distribution.npy ]]; then
  "${build_dir}"/sssp \
    --bucket_distribution="${output_dir}"/.partial.npy \
    "${data_dir}"/g500-15 \
    --max_distance=1.2
  mv \
    "${output_dir}"/.partial.npy \
    "${output_dir}"/g500-15.bucket-distribution.npy
fi

pq_sizes=''
for dataset in g500-13 g500-14 g500-15 road-ny road-col road-fla; do
  if [[ ! -f "${output_dir}/${dataset}.pq-size.npy" ]]; then
    "${build_dir}"/sssp \
      "${data_dir}/${dataset}" \
      --pq_size="${output_dir}"/.partial.npy
    mv "${output_dir}"/.partial.npy "${output_dir}/${dataset}.pq-size.npy"
  fi
  pq_sizes="${pq_sizes}${dataset}.pq-size.npy,"
done

if [[ ! -f ~/.local/share/fonts/.downloaded-LinLibertineTTF_5.3.0_2012_07_02.tgz ]]; then
  mkdir -p ~/.local/share/fonts
  curl -L \
    'https://sourceforge.net/projects/linuxlibertine/files/linuxlibertine/5.3.0/LinLibertineTTF_5.3.0_2012_07_02.tgz' |
    tar --directory ~/.local/share/fonts \
      --extract \
      --gzip \
      --wildcards \
      --file - \
      'LinLibertine_R*.ttf'
  touch ~/.local/share/fonts/.downloaded-LinLibertineTTF_5.3.0_2012_07_02.tgz
fi

pushd "${output_dir}"
LANG=en_US.UTF-8 "${base_dir}"/draw.py \
  --png_dir=. \
  --pdf_dir=. \
  "${logs[@]}" \
  --pq_size="${pq_sizes%,}" \
  --bucket_distribution=g500-15.bucket-distribution.npy \
  --metadata="${data_dir}"/metadata.json
popd

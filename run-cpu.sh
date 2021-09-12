#!/bin/zsh
set -e

sssp_bin="$1"
if [[ ! -x "${sssp_bin}" ]]; then
  echo "${sssp_bin} is not an executable" >&2
  exit 1
fi

dataset_dir="$2"
if [[ ! -d "${dataset_dir}" ]]; then
  echo "${dataset_dir} is not an directory" >&2
  exit 2
fi

shift 2

tmp_file="$(mktemp --suffix=splag.run-cpu.log)"

for file in "$@"; do
  if [[ ! -f "${file}" ]]; then
    echo "file '${file}' does not exist" >&2
    continue
  fi

  dataset="${file##*/}"
  dataset="${dataset%%.*}"
  bucket_count=128

  log_file="${file%/*}/${dataset}.${sssp_bin##*/}.log"
  exec {log_fd}>${log_file}

  trv_teps_recp=0.
  alg_teps_recp=0.
  query_count=0

  while read -r line; do
    line="${line#*] }"
    eval 'items=(${=line})' # make shfmt happy
    case "${line}" in
    root:*)
      root="${items[2]}"
      ;;
    "using max distance"*)
      max_distance="${items[4]}"
      ;;
    "using min distance"*)
      min_distance="${items[4]}"
      ;;
    "  #edges connected:"*)
      edge_connected="${items[3]}"
      delta=$(((max_distance - min_distance) / bucket_count))

      "${sssp_bin}" \
        "${dataset_dir}/${dataset}.gr" \
        --algo=deltaStep \
        --delta="${delta}" \
        -t="$(nproc)" \
        --startNode="${root}" \
        &>"${tmp_file}"

      while read -r line; do
        eval 'items=(${=line})' # make shfmt happy
        case "${line}" in
        'STAT, SSSP, Time, TMAX,'*)
          time_ms="${items[5]}"
          ;;
        'STAT, SSSP, WORK_COUNT, SINGLE,'*)
          edge_traversed="${items[5]}"
          ;;
        esac
      done <${tmp_file}

      trv_teps_recp=$((trv_teps_recp + 1000. * time_ms / edge_traversed))
      alg_teps_recp=$((alg_teps_recp + 1000. * time_ms / edge_connected))
      query_count=$((query_count + 1))
      printf 'query #%d: %.0f / %.0f MTEPS\n' \
        ${query_count} \
        $((query_count / trv_teps_recp)) \
        $((query_count / alg_teps_recp)) |
        tee >(cat >&2) >&"${log_fd}"

      cat "${tmp_file}" >&"${log_fd}"
      ;;
    esac
  done <"${file}"

  printf 'Throughput: %.0f / %.0f MTEPS (harmonic mean over %d queries)\n' \
    $((query_count / trv_teps_recp)) \
    $((query_count / alg_teps_recp)) \
    ${query_count} |
    tee >(cat >&2) >&"${log_fd}"
done

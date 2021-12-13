#!/bin/zsh
set -e

sssp_bin="$1"
if [[ ! -x "${sssp_bin}" ]]; then
  echo "${sssp_bin} is not an executable" >&2
  exit 1
fi

data_dir="$2"

shift 2

tmp_file="$(mktemp --suffix=splag.run-gpu.log)"

for file in "$@"; do
  if [[ ! -f "${file}" ]]; then
    echo "file '${file}' does not exist" >&2
    continue
  fi

  dataset="${file##*/}"
  dataset="${dataset%%.*}"

  mkdir -p "${data_dir}"
  if [[ ! -f "${data_dir}/${dataset}.gr" ]]; then
    curl -L \
      "https://github.com/UCLA-VAST/splag/releases/download/fpga22/${dataset}.gr.gz" |
      gzip -d >"${data_dir}"/.partial
    mv "${data_dir}"/.partial "${data_dir}/${dataset}.gr"
  fi

  log_file="${file%/*}/${dataset}.sssp-gpu.log"
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
    "  #edges connected:"*)
      edge_connected="${items[3]}"

      "${sssp_bin}" \
        "${data_dir}/${dataset}.gr" \
        -s "${root}" \
        >"${tmp_file}" 2>&1

      while read -r line; do
        eval 'items=(${=line})' # make shfmt happy
        case "${line}" in
        'total work is '*)
          edge_traversed="${items[4]}"
          ;;
        "${dataset}.gr Measured time for sample = "*)
          time_s="${items[7]%s}"
          ;;
        esac
      done <${tmp_file}

      trv_teps_recp=$((trv_teps_recp + 1e6 * time_s / edge_traversed))
      alg_teps_recp=$((alg_teps_recp + 1e6 * time_s / edge_connected))
      query_count=$((query_count + 1))
      printf "${dataset}: query #%02d: %.0f / %.0f MTEPS\n" \
        ${query_count} \
        $((query_count / trv_teps_recp)) \
        $((query_count / alg_teps_recp)) |
        tee >(cat >&2) >&"${log_fd}"

      cat "${tmp_file}" >&"${log_fd}"
      ;;
    esac
  done <"${file}"

  printf "${dataset}: Throughput: %.0f / %.0f MTEPS (harmonic mean over %d queries)\n" \
    $((query_count / trv_teps_recp)) \
    $((query_count / alg_teps_recp)) \
    ${query_count} |
    tee >(cat >&2) >&"${log_fd}"
done

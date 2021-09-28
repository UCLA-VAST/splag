#!/bin/bash
set -e
base_dir="$(realpath "${0%/*}")"
build_dir="$(realpath "$1")"

# Hack v++ so that it exits clean.
tmp_dir=/var/tmp/"${USER}"/bin
mkdir -p "${tmp_dir}"
cat >"${tmp_dir}"/v++ <<'EOF'
#!/bin/bash
source "${XILINX_VITIS}"/settings64.sh
cleanup() { kill -9 $(ps -s "${pid}" -o pid=) >/dev/null 2>/dev/null; }
setsid "$(basename "$0")" "$@" &
pid=$!
trap cleanup SIGINT SIGTERM SIGPIPE SIGCHLD EXIT
wait
EOF
chmod +x "${tmp_dir}"/v++
PATH="${tmp_dir}:${PATH}"

cmake -S. -B"${build_dir}" -DCMAKE_BUILD_TYPE=Release
make -C "${build_dir}" sssp-hw-xo
bash <<EOF
exec make -C ${build_dir@Q} SSSP.xilinx_u250_xdma_201830_2.hw_xclbin
EOF

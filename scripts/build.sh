#!/bin/bash
# Build all workloads. Portable: any location, any cwd.
set -e

source "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/_common.sh"
resolve_cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=$CUDA_HOME, sm_$ARCH"
for w in gemm reduce conv2d softmax attention mlp; do
  echo "== building $w =="
  cd "$LAB_ROOT/workloads/$w"
  nvcc -O3 -std=c++11 -arch="sm_$ARCH" -o "$w" "$w.cu"
  ls -la "$w"
done
echo "== build OK =="

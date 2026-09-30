#!/bin/bash
# Source this to get the right toolchain in your shell, from any location:
#   source /path/to/gpu-sdc-lab/scripts/env.sh
_here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
source "$_here/_common.sh"
resolve_cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "gpu-sdc-lab: LAB_ROOT=$LAB_ROOT CUDA_HOME=$CUDA_HOME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

#!/bin/bash
# Shared helpers for gpu-sdc-lab scripts.
# Source this from any location:  source /path/to/gpu-sdc-lab/scripts/_common.sh
# No absolute project paths are used; LAB_ROOT is derived from this file.

LAB_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

# Locate a CUDA toolkit without hardcoding a path:
#   $CUDA_HOME -> nvcc on PATH -> newest /usr/local/cuda*
resolve_cuda() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    export CUDA_HOME
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    CUDA_HOME="$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)"
  fi
  export CUDA_HOME
}

# Default GPU architecture (A100).
ARCH="${ARCH:-80}"

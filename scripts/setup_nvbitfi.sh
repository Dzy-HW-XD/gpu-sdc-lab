#!/bin/bash
# Build NVBit + NVBitFI with the A100 / CUDA-13 compatibility patches.
#
# Portable: works from any location and from any current directory. No
# absolute project paths. CUDA is auto-detected (or set CUDA_HOME=...).
#
# Compatibility notes (see fault_models/nvbitfi/README.md):
#   * profiler: ballot() is removed on sm_70+ -> __activemask/__ballot_sync
#   * injector: device-side printf()/assert() cause CUDA_ERROR_INVALID_SOURCE
#     on CUDA < 13.1, so they are compiled out.
set -e

source "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/_common.sh"
resolve_cuda

NVBIT_VER="${NVBIT_VER:-1.8.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TP="$LAB_ROOT/third_party"
mkdir -p "$TP"
cd "$TP"

echo "LAB_ROOT  = $LAB_ROOT"
echo "CUDA_HOME = $CUDA_HOME"

echo "== [1/6] NVBit release =="
if [ ! -d "$TP/nvbit_release_x86_64/core" ]; then
  curl -sL -o "nvbit-$NVBIT_VER.tar.bz2" \
    "https://github.com/NVlabs/NVBit/releases/download/v$NVBIT_VER/nvbit-Linux-x86_64-$NVBIT_VER.tar.bz2"
  tar xjf "nvbit-$NVBIT_VER.tar.bz2"
fi

echo "== [2/6] NVBitFI source =="
if [ ! -d "$TP/nvbitfi" ]; then
  git clone --depth 1 https://github.com/NVlabs/nvbitfi.git "$TP/nvbitfi"
fi

echo "== [3/6] install NVBitFI into NVBit tools/ =="
REL="$TP/nvbit_release_x86_64"
rm -rf "$REL/tools/nvbitfi"
cp -r "$TP/nvbitfi" "$REL/tools/nvbitfi"
FI="$REL/tools/nvbitfi"
find "$FI" -name '*.sh' | xargs chmod +x

echo "== [4/6] patch profiler ballot -> activemask/ballot_sync =="
sed -i 's/const int active_mask = ballot(1);/const int active_mask = __activemask();/' \
  "$FI/profiler/inject_funcs.cu"
sed -i 's/const int predicate_mask = ballot(predicate);/const int predicate_mask = __ballot_sync(__activemask(), predicate);/' \
  "$FI/profiler/inject_funcs.cu"

echo "== [5/6] patch injector device printf/assert (CUDA < 13.1) =="
if ! grep -q 'nvbitfi-lab patch' "$FI/injector/inject_funcs.cu"; then
  sed -i '/#include "arch.h"/a\
/* nvbitfi-lab patch: device-side printf\/assert unsupported on CUDA < 13.1 */\
#undef assert\
#define assert(x) ((void)0)\
#define printf(...) ((void)0)' "$FI/injector/inject_funcs.cu"
fi

echo "== [5b] patch NVBitFI: add dedicated 'atom' instruction group =="
python3 "$LAB_ROOT/scripts/patch_nvbitfi.py" "$FI"

echo "== [6/6] build tools for sm_$ARCH =="
( cd "$FI/injector" && make clean >/dev/null 2>&1 && make ARCH="$ARCH" )
( cd "$FI/profiler" && make clean >/dev/null 2>&1 && make ARCH="$ARCH" )

echo "NVBitFI ready: $FI/injector/injector.so"

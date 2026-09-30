#!/bin/bash
# Determinism smoke test for every workload: run each twice and compare all
# produced tensors. This is tool validation, NOT a fault-injection experiment.
set -e

source "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)/_common.sh"
resolve_cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

rc=0
for w in gemm reduce conv2d softmax attention mlp; do
  BIN="$LAB_ROOT/workloads/$w/$w"
  if [ ! -x "$BIN" ]; then echo "$w: MISSING BINARY"; rc=1; continue; fi
  "$BIN" --out "$TMP/${w}_a" >/dev/null 2>&1 || { echo "$w: RUN FAILED"; rc=1; continue; }
  "$BIN" --out "$TMP/${w}_b" >/dev/null 2>&1
  ok=1
  for f in "$TMP/${w}_a"*; do
    b="${f/_a/_b}"
    cmp -s "$f" "$b" || ok=0
  done
  n=$(ls "$TMP/${w}_a"* | wc -l)
  if [ $ok -eq 1 ]; then echo "$w: DETERMINISTIC ($n files)"; else echo "$w: DIFFERS"; rc=1; fi
done
exit $rc

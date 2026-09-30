# GEMM workload

Deterministic single-precision `C = A x B` used as the Phase-1 SDC workload.

- Row-major `A (M x K)`, `B (K x N)`, `C (M x N)`.
- Inputs are produced by a deterministic hash of the element index and the
  seed (`det_value` in `gemm.cu`). No randomness, no time dependence, so runs
  on the same binary/GPU are bit-identical -> valid golden oracle.
- One CUDA kernel `gemm_fp32` (naive tiled, `block = [16,16]`).
- Writes `<out>.bin` (raw little-endian float32) and `<out>.json` (metadata).

## CLI

```
gemm --M 1024 --N 1024 --K 1024 --dtype fp32 --seed 1234 \
     --pattern normal --device 0 --out /path/out
```

Exits non-zero (and prints `ERROR: ...`) on any CUDA error so the
RuntimeObserver can classify crashes.

## Patterns

`normal`, `small`, `large`, `near_zero`, `mixed` (magnitude scaling of the
deterministic inputs). Phase 1 uses `normal`; the others exist for the
data-pattern experiment and future input-aware SDC research.

## Extending

New workloads implement the interface in `core/workload/base.py`
(`prepare/run/collect_output/cleanup`) and register in
`core/workload/registry.py`. The runner/injector/observers are workload
agnostic.

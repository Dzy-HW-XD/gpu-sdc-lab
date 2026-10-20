# Tensor-Core BF16 GEMM + HMMA accumulator fault injection

This document describes the Tensor-Core direction of `gpu-sdc-lab`. It
supersedes the earlier **scalar (FFMA/FADD/FMUL, FP32 CUDA Core)** GEMM study,
which is kept for reference only (see `docs/ARCHIVE_scalar_gemm.md`).

## Execution path

```
BF16 A (MxK) , BF16 B (KxN)  --cuBLAS cublasGemmEx-->  FP32 C (MxN)
        (CUDA_R_16BF)         CUBLAS_COMPUTE_32F        (CUDA_R_32F)
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP
```

* Workload: **`gemm_bf16`** (`workloads/gemm_bf16/gemm_bf16.cu`,
  `core/workload/gemm_bf16.py`), built with `-lcublas`.
* The multiply runs on Tensor Cores as **HMMA** (`ampere_s16816gemm_bf16_*`
  kernels on A100), accumulating in **FP32 registers** — the injection target.
* Determinism: `cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED)` (no split-K
  atomics). Same inputs/GPU give **bit-identical** outputs (verified at
  512/1024/2048).

## Inputs

BF16 tensors are materialized as raw little-endian `uint16`
(`core/inputgen.py::to_bf16_bytes`, round-to-nearest-even). The fault-forecast
features (`scripts/features.py`) are computed on the pre-cast FP32 values.

Input distributions used by the TC campaign (LLM-activation oriented):

| family | meaning |
|---|---|
| `normal` | control (N(0,1)) |
| `heavy_tail_act` | few extreme "massive activations" + long tail |
| `residual_stream` | per-channel scale differences (log-normal) + correlation |
| `swiglu_gated` | gated/sparse activation `x*silu(gate)` |

Parameters (`outlier_frac`, `outlier_scale`, `channel_std`, `period`,
`rank_gain`, `gate_sparse`, ...) are configurable in `core/inputgen.py`.

## HMMA fault injection

NVBitFI's upstream handling of HMMA on `sm_80` was **wrong** (it assumed 2
destination registers); a HMMA.16816 accumulates into **4 FP32 registers**.
`scripts/patch_nvbitfi.py` now (idempotently):

1. adds a dedicated **`mma`** instruction group (`HMMA`/`IMMA`), like `atom`;
2. fixes `extractSize()` for HMMA to `128` (4 destination GPRs).

Fault model: NVBitFI injects one bit flip into the **destination register** of
the selected dynamic HMMA (i.e. an FP32 accumulator fragment).
`core/regclass.py` labels every HMMA/IMMA destination as `accumulator`.

Injection-site selection (as before): the runner picks a dynamic instruction
ordinal within the `mma` group; the **position** is the instruction's fraction
of the HMMA dynamic stream, interpreted as **early / middle / late of the FP32
accumulator lifetime** (proxy; the cuBLAS kernel's k-loop/accumulator mapping is
opaque — see note below).

Classification stays `MASKED / SDC / CRASH`, and a run whose output is bitwise
identical to golden is `MASKED` (never confused with a missed injection:
`NOT_INJECTED`/`NOT_TARGETED` are separate).

## Experiments

| id | axis | matrix | position | notes |
|---|---|---|---|---|
| `020_tc_input_bit_early` | input × bit 0–31 | 1024³ | first third of HMMA stream | |
| `021_tc_input_bit_middle` | input × bit 0–31 | 1024³ | middle third | |
| `022_tc_input_bit_late` | input × bit 0–31 | 1024³ | last third | |
| `023_tc_input_size` | input × size | 512/1024/2048 | — | |

All inject `groups: [mma]`, `fault_type: bitflip`, `injections_per_cell: 30`,
inputs `[normal, heavy_tail_act, residual_stream, swiglu_gated]`.

## Reproduce

```bash
# build workload + NVBitFI (HMMA patch) + workload binary
./scripts/setup_nvbitfi.sh          # clones/patches/builds NVBitFI (incl. mma group)
./scripts/build.sh                  # builds workloads (gemm_bf16 needs -lcublas)
./sdc-lab doctor

# single GPU1, sharded
SHARDS=8 bash _watch_tc.sh          # or: bash _watch_tc.sh

# collect + figures
python3 scripts/campaign.py collect --master results/_tc_campaign/master.csv \
    020_tc_input_bit_early 021_tc_input_bit_middle 022_tc_input_bit_late 023_tc_input_size
python3 scripts/heatmap.py results/_tc_campaign/cells.csv --axis bit \
    --experiment 021_tc_input_bit_middle --annotate --outdir results/_tc_campaign/figs
```

## Tolerance levels (L0–L5), offline

The injection run-time classification is exact (L0). In addition the same
injections are re-classified offline at levels L1..L5 = `(1e-7,1e-6) ..
(1e-3,1e-2)` (synchronous ×10) using the preserved raw outputs
(`scripts/tolerance_sweep.py`, `scripts/tolerance_heatmaps.py`); sees
`results/_tc_tolerance/TOLERANCE_LEVELS.md` and its `figs/` (24 heatmaps).

## Raw SDC magnitude vs bit

`scripts/plot_sdc_raw.py` (needs matplotlib; local analysis only) plots **every**
SDC sample as a point: x = bit 0–30, y = |C'−golden| (log scale), coloured by
input distribution, with the exponent region (23–30) shaded and a bottom panel
of N_sdc / N_masked per bit. The **sign bit (31) is excluded** and the plot is
truncated at bit 31. Instead of a pooled median line it overlays **two pooled
branch lines** (one per bit, pooled across inputs), split at the per-bit
**geometric mean** of `|Δ|` and smoothed with a **LOWESS** fit in log space:

* **explode** line (`max |Δ|` of the above-geo-mean samples, red solid) — the top
  of the exponent-bit blow-up (raw max ≈ 3e38 at bit 30; the smoothed line ≈ 1e34);
* **collapse** line (`min |Δ|` of the below-geo-mean samples, blue dashed) — the
  weakest exposure, value crushed toward zero.

The vertical gap between the two lines is the per-bit **exposure** (spread
between the strongest blow-up and the weakest crush). One figure per
(accumulator position × tolerance level):

```
results/_tc_campaign/figs/bit_diff_{early,middle,late}_L{0..5}.svg  (+ _counts.csv)
```

(`results/_tc_tolerance/samples.csv` holds the per-injection
`experiment_id,input,bit,level,classification,max_abs_error` used to plot each
level; produced by `scripts/tolerance_sweep.py --samples`.)

## Heatmaps

`results/_tc_campaign/figs/` contains `input × bit` SDC-rate heatmaps per
position (early/middle/late) and an `input × size` heatmap (exact / L0). The
per-tolerance-level versions live in `results/_tc_tolerance/figs/`. Axes, colour
scale, sample size and the statistical criterion are uniform; cells with misses
/ insufficient samples are marked (see `heatmap.py`).

## Note / open item (position semantics)

Because the matrix multiply is executed by **cuBLAS** (opaque kernel), the exact
mapping between the HMMA dynamic-ordinal and "which k-iteration of a given
accumulator" cannot be derived without disassembling the library kernel. The
campaign therefore uses the **HMMA-stream fraction** as the position variable
(approved proxy). If a precise per-accumulator k-phase is required later, the
cuBLAS kernel must be disassembled / a controlled `mma.sync` kernel used.

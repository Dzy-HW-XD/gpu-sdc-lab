# gpu-sdc-lab

[English](README.md) | [简体中文](README.zh-CN.md)

A long-lived, extensible **GPU Silent Data Corruption (SDC) lab**.

Phase 1 establishes one complete, reusable chain on a single A100:

```
Workload(GEMM) -> Golden oracle -> Fault injection(NVBitFI) -> Faulty run
      -> Observers(Output/Runtime/GPU) -> SDC classification(MASKED/SDC/CRASH)
      -> results/ (JSONL + CSV)
```

The framework is deliberately workload-, injector- and observer-agnostic:
new workloads, fault models and observation methods plug into the same
`Runner` without rewriting the platform.

## Architecture

```
                    SDC LAB
                       |
        +--------------+--------------+
        |              |              |
     Workload       Injector       Observer
     (gemm)        (nvbitfi)     output/runtime/gpu
        |              |              |
        +--------------+--------------+
                       |
                    Oracle (compare + classify)
                       |
                    results/
```

```
core/
  workload/    Workload interface + GEMM + registry
  injection/   FaultInjector interface + NVBitFI + registry
  oracle/      output comparator + MASKED/SDC/CRASH classifier
  observer/    OutputObserver / RuntimeObserver / GPUObserver
  runner/      orchestrates one experiment
  experiments.py  case generation + aggregation per experiment kind
  config.py, util.py, yamlmini.py
workloads/          gemm, reduce, conv2d, softmax, attention, mlp  (+README each)
fault_models/nvbitfi/  backend notes + compatibility patches
experiments/     001..014 experiment definitions
core/taxonomy.py  instruction taxonomy tree (fault-target selection)
core/inputgen.py  input-distribution generators (8 synthetic families + real)
scripts/          build.sh, setup_nvbitfi.sh, smoke_workloads.sh,
                  analyze.py, stats.py, features.py, predictor.py, sampling.py
docs/RESEARCH_PLAN.md  experiment program (for the paper)
results/         generated artifacts
scripts/         setup_nvbitfi.sh, build.sh, env.sh
sdc-lab          CLI entrypoint
```

## Environment (validated)

| item        | value                                        |
|-------------|----------------------------------------------|
| GPU         | NVIDIA A100-SXM4-80GB (sm_80), GPU index 1   |
| Driver      | 580.178.04 (CUDA 13.0)                       |
| Toolkit     | /usr/local/cuda-13.0 (nvcc 13.0.88)          |
| NVBit       | v1.8.1                                       |
| NVBitFI     | NVlabs/nvbitfi (patched, see below)          |
| Python      | 3.12 (stdlib only; no numpy/pandas required) |

## Install / build

```bash
# 1. fetch + patch + build NVBit and NVBitFI (no system changes)
./scripts/setup_nvbitfi.sh

# 2. build the GEMM workload
./scripts/build.sh

# 3. sanity check
./sdc-lab doctor
```

**Portable / relocatable.** The project contains no hardcoded paths: all paths
are derived from the project root at runtime (`realpath`), so it can live in any
directory and be invoked from any current directory or through a symlink. The
CUDA toolkit is auto-detected (`$CUDA_HOME` -> `nvcc` on `PATH` ->
`/usr/local/cuda*`); override with `CUDA_HOME=...` or `cuda_home:` in
`config.yaml`. `results/` and the `third_party/` build are relative to the root.

## Usage

```bash
# environment / GPU / tool presence
./sdc-lab doctor
./sdc-lab profile --M 1024 --N 1024 --K 1024

# baseline determinism (10x, no injection)
./sdc-lab baseline --experiment 001_gemm_baseline

# fault injection campaigns
./sdc-lab run --experiment 002_fault_position
./sdc-lab run --experiment 003_fault_type
./sdc-lab run --experiment 004_fault_bit
./sdc-lab run --experiment 005_gemm_size
./sdc-lab run --experiment 006_data_pattern
./sdc-lab run --experiment 007_reduce_atomic

# ad-hoc: 100 bit-flip injections, mixed targets
./sdc-lab run --workload gemm --injector nvbitfi \
    --fault-type bitflip --num-injections 100

# instruction taxonomy / workloads / input generators
./sdc-lab taxonomy
./sdc-lab list
./sdc-lab gen-input --gen cancellation --n 1048576 --seed 5 --out /tmp/x.bin

# input-sensitivity experiments (input/workload dependence)
./sdc-lab run --experiment 008_input_sensitivity_gemm
./sdc-lab run --experiment 014_cross_workload
```

Useful overrides: `--groups memory/atomic,arithmetic/fp32,memory/load,meta/gp`,
`--group fp32`, `--num-injections N`, `--positions`, `--bits`, `--M/--N/--K`,
`--gpu`, `--seed`, `--id`.

`run` prints:

```
Total   : 100
Masked  : 6
SDC     : 82
Crash   : 12
NotInj  : 0
SDC rate: 0.8200
```

## Results layout

```
results/002_fault_position/
  config.yaml                 # experiment definition
  experiment.json             # counts + per-dimension aggregation
  observations.jsonl          # one JSON object per injection
  summary.csv                 # overall counts
  fault_position_summary.csv  # position,total,masked,sdc,crash,sdc_rate
  report.md
  raw/0001_fp32/              # injection-info, injection-log, out.bin/json,
                              # stdout, stderr, result.json
```

`observations.jsonl` keeps **fault metadata** and **observation metadata**
separate, so fault-position -> SDC-probability -> observable-signature studies
can be done directly with e.g. `pandas.read_json(..., lines=True)`.

## Fault model (NVBitFI)

One fault = one dynamic thread-instruction destination register in one kernel
invocation, selected by `(kernel, invocation, instruction-group, dynamic
inst id)` plus `(bit-flip model, bit id seed)`. Fault types: `bitflip`,
`twobit`, `random`, `zero`. See `fault_models/nvbitfi/README.md`.

## Fault targets (taxonomy tree)

Targets are chosen from a small instruction tree (`core/taxonomy.py`); each leaf
maps to a NVBitFI group and optional opcode whitelist:

```
arithmetic: fp64, fp32, ffma*, fadd*, fmul*, fp16*, int*, mma*
            (* opcode-filtered: ffma/fadd/fmul filter the fp32 bucket, others filter "others")
memory:     load, atomic, store(not injectable)
control:    predicate, nodest(not injectable)
meta:       gppr, gp, other
```

`./sdc-lab taxonomy` prints it. Selection accepts a leaf (`--group atomic`), a
path (`--group memory/atomic`) or a whole category (`--group arithmetic`).
`memory/atomic` uses a dedicated NVBitFI `atom` group added by
`scripts/patch_nvbitfi.py`; leaves marked opcode-filtered sample inside the
shared `others` bucket and are only counted when the hit opcode is in the set
(otherwise `NOT_TARGETED`).

## Workloads

| name | computation | instruction coverage |
|---|---|---|
| `gemm` | `C=A×B` (FP32, deterministic) | fp32, ld, gp, pr |
| `reduce` | block reduction + `atomicAdd` | atom, fp32, ld, gp |
| `conv2d` | direct 2-D convolution | fp32, ld, gp |
| `softmax` | row-wise softmax | fp32, ld, gp, MUFU/others |
| `attention` | single-head explicit-math attention | fp32, ld, gp, MUFU/others |
| `mlp` | 2-layer MLP forward + one backward step | fp32, ld, gp, MUFU + multi-output |

All are deterministic (bit-exact golden). `mlp` emits six named tensors
(`A1, logits, probs, loss, dW1, dW2`) so error propagation can be observed per
layer. `scripts/smoke_workloads.sh` verifies determinism for all of them.

## Input generators

`core/inputgen.py` produces deterministic float32 tensors for the study:
`uniform, normal, lognormal, sparse, cancellation, near_overflow, correlated,
adversarial, ones, near_zero, near_one, extreme, small, large`, plus `real`
(load a tensor from disk). A workload with an `input_spec` (see
`experiments/008_*`, `015_*`) materialises these into `results/_inputs/` and
passes them to the kernel (`--input/--inputA/...`). `./sdc-lab gen-input`
writes one directly; `scripts/features.py` extracts interpretable features
(dynamic range, cancellation, exponent stats, ...). See
[docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md) for what each family means,
with concrete example values.

## Input-dependent campaign (015–019)

`kind: input_grid` sweeps an input family against one second variable while
holding the rest fixed (paired design), producing composite `input|<value>`
cells:

| experiment | second variable | notes |
|---|---|---|
| `015_input_bit_gemm` | FP32 bit 0–31 | core input × bit heatmap |
| `016_input_instruction_gemm` | FFMA / FADD / FMUL | opcode-filtered; FADD/FMUL use `--op-mode muladd` |
| `017_input_size_gemm` | 128/512/1024 | |
| `018_input_position_gemm` | early / middle / late | |
| `019_register_class_gemm` | accumulator / temporary | static SASS labelling, see `core/regclass.py` |

Run them and collect a single master CSV + Wilson-CI cell summary:

```bash
python3 scripts/campaign.py run 015_input_bit_gemm 016_input_instruction_gemm
python3 scripts/campaign.py collect 015_input_bit_gemm 016_input_instruction_gemm
python3 scripts/heatmap.py results/_campaign/cells.csv --axis bit
# wide-dynamic-range metrics: log colour scale + numeric values in every cell
python3 scripts/heatmap.py results/_campaign/cells.csv --axis bit \
    --metric mean_abs_err --log-color --annotate
python3 scripts/stats.py --cells results/_campaign/cells.csv uniform bit
python3 scripts/campaign_report.py results/_campaign/cells.csv \
    results/_campaign/CAMPAIGN_SUMMARY.md --master results/_campaign/master.csv
```

Tolerance sensitivity is computed **offline** from the preserved raw outputs
(no GPU re-run): `scripts/tolerance_sweep.py` re-classifies the same samples at
levels L1..L5 = `(1e-7,1e-6) .. (1e-3,1e-2)` (synchronous ×10 scaling), and
`scripts/tolerance_heatmaps.py` draws one input × bit heatmap per level. See
`results/_tolerance/TOLERANCE_LEVELS.md`.

```bash
python3 scripts/tolerance_sweep.py 015_input_bit_gemm 016_input_instruction_gemm \
    017_input_size_gemm 018_input_position_gemm 019_register_class_gemm \
    --out results/_tolerance
python3 scripts/tolerance_heatmaps.py \
    --cells results/_tolerance/tolerance_by_cell.csv \
    --experiment 015_input_bit_gemm --outdir results/_tolerance/figs
```

`campaign.py` merges every `observations.jsonl` into
`results/_campaign/master.csv` and `results/_campaign/cells.csv` (per-axis
Wilson CIs). `heatmap.py` renders stdlib-only SVG heatmaps + pivot CSVs.

## Instruction-type targeting

The NVBitFI `fp32` bucket is one group; `core/taxonomy.py` exposes opcode-
filtered views of it (`arithmetic/ffma`, `arithmetic/fadd`, `arithmetic/fmul`)
so a hit is only counted when the executed opcode matches. GEMM's inner loop is
pure FFMA, so `gemm` supports `--op-mode muladd`, which forces separate
`__fmul_rn`/`__fadd_rn` instructions (still deterministic) to materialise FADD
and FMUL in the same kernel. Default `--op-mode fma` is byte-identical to the
original GEMM.

## Register-class labelling

`core/regclass.py` disassembles the workload (`cuobjdump -sass`, cached under
`results/_sass/`) and labels each FP32 instruction's destination as
`accumulator` (destination is also an operand, i.e. an in-place running sum) or
`temporary`. The runner attaches `fault.register_class` to every record and
writes `register_class_summary.csv`. This is a static, best-effort heuristic.

## Analysis tools

`scripts/stats.py` (Wilson CI, McNemar, two-proportion z, Holm),
`scripts/features.py` (input features), `scripts/predictor.py` (ridge model +
leave-one-out), `scripts/sampling.py` (uniform vs input-aware SDC discovery),
`scripts/analyze.py` (per-experiment summary).

## SDC classification

| class   | condition |
|---------|-----------|
| CRASH   | timeout, non-zero exit, CUDA error / illegal access / misaligned address |
| MASKED  | run completes and output is bit-identical to golden |
| SDC     | run completes and output differs from golden (NaN/Inf flagged) |
| NOT_INJECTED | fault site never executed (bookkeeping only) |

Thresholds come from `oracle.abs_tol/rel_tol`, defaulting to exact
comparison, which is valid because the GEMM golden is bit-reproducible
(verified in experiment 001).

## Extending

- **Workload**: implement `core/workload/base.py` and register it.
- **Injector**: implement `core/injection/base.py` (e.g. software injector)
  and register it. The runner never imports NVBitFI directly.
- **Observer**: add an `Observer` (activation / gradient / loss / weight /
  DCGM) in `core/observer/`; the runner aggregates them.
- **Oracle / experiment kind**: add a builder in `core/experiments.py`.

## Compatibility notes (A100 + CUDA 13.0)

NVBitFI (2020) needed two source patches, applied automatically by
`scripts/setup_nvbitfi.sh`; no system/driver changes were made:

1. profiler: `ballot()` -> `__activemask()` / `__ballot_sync`.
2. injector: device-side `printf()/assert()` compiled out (they require
   CUDA >= 13.1 and otherwise raise `CUDA_ERROR_INVALID_SOURCE`).

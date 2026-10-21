# gpu-sdc-lab

[English](README.md) | [简体中文](README.zh-CN.md)

A reproducible **GPU Silent Data Corruption (SDC) laboratory**. The repository
has two layers:

- **Platform** — a reusable workload / injector / observer / oracle / runner
  pipeline. It is workload-, injector- and observer-agnostic.
- **Experiments** — YAML configurations built on the platform. The main
  experiment set is **001–019** (scalar FP32 GEMM + input-sensitivity studies)
  and **047** (attention Tensor-Core campaign).

No experiment forks the platform: every experiment is defined in
`experiments/<id>/experiment.yaml` and runs through the same `sdc-lab` /
`campaign.py` path.

---

## Platform

### Pipeline

```
Workload -> Golden oracle -> NVBitFI injection -> Faulty run
      -> Observers (output / runtime / GPU) -> SDC classification
      -> results/ (JSONL + CSV + figures)
```

Design principles:

- deterministic golden runs, so `MASKED` means bitwise identical;
- one fault per run, injected into one dynamic instruction destination register;
- fault metadata and observation metadata stored separately;
- figures and analysis are reproducible from the preserved raw outputs;
- the GPU-side tooling needs only Python's standard library.

### Repository layout

```
core/
  workload/       Workload interface + registry
  injection/      FaultInjector interface + NVBitFI backend
  oracle/         Output comparator + MASKED/SDC/CRASH classifier
  observer/       Output / Runtime / GPU observers
  runner/         Orchestrates one experiment
  experiments.py  Case generation + aggregation per experiment kind
  taxonomy.py     Instruction taxonomy (fault-target selection)
  inputgen.py     Synthetic input families + real-tensor loader
  regclass.py     Static SASS register classification
workloads/        gemm, gemm_bf16, reduce, conv2d, softmax, attention,
                  attention_forward, mlp
fault_models/     NVBitFI notes + compatibility patches
experiments/      Experiment definitions (001–047)
scripts/          Build, campaign, analysis, plotting, tolerance, statistics
docs/             Research plan, workload and experiment documentation
results/          Generated artifacts
sdc-lab           CLI entrypoint
```

### Workloads

| workload | computation | injectable path |
|---|---|---|
| `gemm` | FP32 `C=A×B`, deterministic CUDA-Core kernel | FFMA / FADD / FMUL |
| `gemm_bf16` | BF16×BF16 → FP32, cuBLAS Tensor Core | HMMA accumulator |
| `reduce` | block reduction + `atomicAdd` | atomics / FP32 / loads |
| `conv2d` | direct 2-D convolution | FP32 / loads |
| `softmax` | row-wise softmax | FP32 / MUFU / loads |
| `attention` | explicit FP32 single-head attention | FP32 / loads / MUFU |
| `attention_forward` | BF16 causal multi-head attention, five GEMMs | HMMA accumulator |
| `mlp` | 2-layer forward + one backward step | FP32 / loads / MUFU |

All workloads are deterministic. `mlp` emits six named tensors for per-layer
propagation studies.

### Fault model and targets

One fault = one dynamic thread-instruction destination register in one kernel
invocation. Fault types: `bitflip`, `twobit`, `random`, `zero`.

Targets are selected from an instruction taxonomy: `arithmetic` (fp32, ffma,
fadd, fmul, fp16, int, mma), `memory` (load, atomic), `control` (predicate)
and `meta` (gpr, gp). NVBitFI's `mma` group is patched to handle the four
FP32 destination registers of HMMA.16816. See
[docs/TC_BF16_HMMA.md](docs/TC_BF16_HMMA.md).

### Inputs

`core/inputgen.py` provides deterministic input families: `uniform`, `normal`,
`lognormal`, `sparse`, `cancellation`, `near_overflow`, `correlated`,
`adversarial`, `ones`, `near_zero`, `near_one`, `extreme`, `small`, `large`,
plus LLM-activation-oriented `heavy_tail_act`, `residual_stream`,
`swiglu_gated`, and `real` (load a tensor from disk). Inputs can be materialised
as FP32 or BF16. See [docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md).

### Analysis tools

- `scripts/campaign.py` — run/collect a campaign, merge `observations.jsonl`,
  compute per-cell Wilson intervals.
- `scripts/heatmap.py` — stdlib SVG heatmaps and pivot CSVs.
- `scripts/tolerance_sweep.py` — re-classify preserved outputs at L1–L5.
- `scripts/stats.py` — Wilson CI, McNemar, two-proportion z, Holm.
- `scripts/features.py` — interpretable input features.
- `scripts/predictor.py`, `scripts/sampling.py` — input-aware risk model and
  sampling.
- `scripts/analyze.py` — per-experiment summary.

### Environment (validated)

| item | value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (sm_80) |
| Driver | 580.178.04 (CUDA 13.0) |
| Toolkit | `/usr/local/cuda-13.0` (nvcc 13.0.88) |
| NVBit | v1.8.1 |
| NVBitFI | NVlabs/nvbitfi (patched) |
| Python | 3.12, standard library only on the GPU node |

### Build and run

```bash
./scripts/setup_nvbitfi.sh     # fetch + patch + build NVBit/NVBitFI
./scripts/build.sh             # build workloads
./sdc-lab doctor               # environment / GPU / tool check

./sdc-lab baseline --experiment 001_gemm_baseline
./sdc-lab run      --experiment 002_fault_position
./sdc-lab run      --experiment 047_attn_5gemm_full
```

`run` prints total / masked / SDC / crash / not-injected counts and the SDC
rate. Useful overrides: `--groups`, `--positions`, `--bits`, `--M/--N/--K`,
`--gpu`, `--seed`, `--num-injections`.

### Results layout

```
results/<experiment>/
  config.yaml                 # experiment definition
  experiment.json             # counts + aggregation
  observations.jsonl          # one JSON record per injection
  summary.csv                 # overall counts
  <axis>_summary.csv          # per-axis breakdown
  report.md                   # human-readable report (where generated)
  raw/                        # per-injection outputs (large; archive)
```

`observations.jsonl` keeps **fault metadata** and **observation metadata**
separate, so fault-site → SDC-probability → observable-signature studies can be
done directly. Large binaries (`*.bin`) and raw archives are intentionally not
tracked.

### SDC classification

| class | condition |
|---|---|
| `CRASH` | timeout, non-zero exit, CUDA error / illegal access |
| `MASKED` | completes and output is bit-identical to golden |
| `SDC` | completes and output differs from golden (NaN/Inf flagged) |
| `NOT_INJECTED` | fault site never executed (bookkeeping only) |

Thresholds come from `oracle.abs_tol/rel_tol`; exact comparison is valid because
the golden runs are bit-reproducible.

### Extending

- **Workload**: implement `core/workload/base.py` and register it.
- **Injector**: implement `core/injection/base.py` and register it.
- **Observer**: add an observer under `core/observer/`.
- **Oracle / experiment kind**: add a builder in `core/experiments.py`.

### Compatibility (A100 + CUDA 13.0)

NVBitFI (2020) needs two source patches, applied by `scripts/setup_nvbitfi.sh`:
`ballot()` → `__activemask()`/`__ballot_sync`, and removal of device-side
`printf()/assert()` (CUDA ≥ 13.1 only). No system driver changes are made.

---

## Experiments

All experiments below are configurations on the platform. The **main set** is
001–019 and 047; 020–023 and 030–040 are development/auxiliary runs listed at
the end.

### 001–007: platform bring-up and scalar GEMM basics

| ID | Question | Results |
|---|---|---|
| [001_gemm_baseline](experiments/001_gemm_baseline/experiment.yaml) | Is the golden GEMM bit-reproducible? | [report](results/001_gemm_baseline/report.md) |
| [002_fault_position](experiments/002_fault_position/experiment.yaml) | How does early/middle/late instruction position affect SDC? | [report](results/002_fault_position/report.md) |
| [003_fault_type](experiments/003_fault_type/experiment.yaml) | How do bitflip / twobit / random / zero faults compare? | [report](results/003_fault_type/report.md) |
| [004_fault_bit](experiments/004_fault_bit/experiment.yaml) | Which FP32 bit fields (sign / exponent / mantissa) are most damaging? | [report](results/004_fault_bit/report.md) |
| [005_gemm_size](experiments/005_gemm_size/experiment.yaml) | Does GEMM size change SDC probability? | [report](results/005_gemm_size/report.md) |
| [006_data_pattern](experiments/006_data_pattern/experiment.yaml) | Does the input data pattern change SDC probability? | [report](results/006_data_pattern/report.md) |
| [007_reduce_atomic](experiments/007_reduce_atomic/experiment.yaml) | How do atomics compare with FP32/load/global-memory faults? | [report](results/007_reduce_atomic/report.md) |

Group summary: [results/SDCLAB_SUMMARY.md](results/SDCLAB_SUMMARY.md).

### 008–014: input sensitivity across workloads

These paired experiments vary only the input distribution while holding the
fault site fixed. They established the input-dependent SDC direction used by
the paper; per-experiment result directories are not retained in this checkout.

| ID | Question | Links |
|---|---|---|
| [008_input_sensitivity_gemm](experiments/008_input_sensitivity_gemm/experiment.yaml) | Input-family sensitivity of GEMM | [plan](docs/RESEARCH_PLAN.md), [abstract](PAPER_ABSTRACT.md) |
| [009_input_sensitivity_reduce](experiments/009_input_sensitivity_reduce/experiment.yaml) | Input sensitivity of reduction / atomics | [plan](docs/RESEARCH_PLAN.md) |
| [010_input_sensitivity_softmax](experiments/010_input_sensitivity_softmax/experiment.yaml) | Input sensitivity of softmax | [plan](docs/RESEARCH_PLAN.md) |
| [011_input_sensitivity_conv2d](experiments/011_input_sensitivity_conv2d/experiment.yaml) | Input sensitivity of 2-D convolution | [plan](docs/RESEARCH_PLAN.md) |
| [012_input_sensitivity_attention](experiments/012_input_sensitivity_attention/experiment.yaml) | Input sensitivity of explicit attention | [plan](docs/RESEARCH_PLAN.md) |
| [013_input_sensitivity_mlp](experiments/013_input_sensitivity_mlp/experiment.yaml) | Input sensitivity and per-layer propagation in MLP | [plan](docs/RESEARCH_PLAN.md) |
| [014_cross_workload](experiments/014_cross_workload/experiment.yaml) | Cross-workload comparison | [plan](docs/RESEARCH_PLAN.md) |

### 015–019: input × variable campaign (scalar GEMM)

`kind: input_grid` sweeps an input family against one second variable in a
paired design. These are the scalar CUDA-Core GEMM campaign, archived for
reference; see [docs/ARCHIVE_scalar_gemm.md](docs/ARCHIVE_scalar_gemm.md).

| ID | Second variable | Results |
|---|---|---|
| [015_input_bit_gemm](experiments/015_input_bit_gemm/experiment.yaml) | FP32 bit 0–31 | [results](results/015_input_bit_gemm/) |
| [016_input_instruction_gemm](experiments/016_input_instruction_gemm/experiment.yaml) | FFMA / FADD / FMUL | [results](results/016_input_instruction_gemm/) |
| [017_input_size_gemm](experiments/017_input_size_gemm/experiment.yaml) | size 128 / 512 / 1024 | [results](results/017_input_size_gemm/) |
| [018_input_position_gemm](experiments/018_input_position_gemm/experiment.yaml) | early / middle / late | [results](results/018_input_position_gemm/) |
| [019_register_class_gemm](experiments/019_register_class_gemm/experiment.yaml) | accumulator / temporary register | [results](results/019_register_class_gemm/) |

Campaign-level links:
[per-cell summary](results/_campaign/CAMPAIGN_SUMMARY.md),
[L1–L5 tolerance](results/_tolerance/TOLERANCE_LEVELS.md).

### 047: attention five-GEMM campaign

`047_attn_5gemm_full` injects single bit flips into the HMMA accumulators of a
BF16 causal attention forward pass (`S=128, D=2048, H=32`). It covers five
GEMM stages (`qkv_q`, `qkv_k`, `qkv_v`, `qk`, `pv`), bits 24–30, early/middle/
late positions, and 30 injections per cell (3150 total). Classification uses
L4 (`|ΔY| > 1e-4 + 1e-3|Y|`).

Key results: **1507/3150 SDC (47.8%)**. QKV-Q, QK and PV are strictly
single-token; QKV-K is broad (median 20 affected token rows); QKV-V is
intermediate. 55 bit-30 runs produce NaN/Inf and are handled explicitly.

Links:
[campaign results](results/047_attn_5gemm_full/),
[figure notes](results/047_attn_5gemm_full/final_heatmaps_v2/README.md),
[token-count summary](results/047_attn_5gemm_full/final_heatmaps_v2/summary/token_count_distribution.png),
[golden-vs-delta](results/047_attn_5gemm_full/final_heatmaps_v2/summary/golden_vs_delta_hexbin.png),
[attention design](docs/ATTENTION.md),
[heatmap redesign](docs/HEATMAP_REDESIGN.md).

### Auxiliary / development runs

| ID | Purpose | Links |
|---|---|---|
| 020–023 | BF16 Tensor-Core input × bit/size (early/middle/late) | [TC summary](results/_tc_campaign/TC_SUMMARY.md), [tolerance](results/_tc_tolerance/TOLERANCE_LEVELS.md), [doc](docs/TC_BF16_HMMA.md) |
| 030–033 | Attention dev/target baselines | [attention doc](docs/ATTENTION.md), [configs](experiments/030_attn_single_dev_baseline/experiment.yaml) |
| 040 | Attention stage smoke test | [attention doc](docs/ATTENTION.md), [config](experiments/040_attn_stage_smoke/experiment.yaml) |

---

## Documentation index

- [PAPER_ABSTRACT.md](PAPER_ABSTRACT.md) — paper abstract (EN/ZH)
- [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) — experiment program
- [docs/INPUT_FAMILIES.md](docs/INPUT_FAMILIES.md) — input families
- [docs/TC_BF16_HMMA.md](docs/TC_BF16_HMMA.md) — Tensor-Core/HMMA campaign
- [docs/ATTENTION.md](docs/ATTENTION.md) — attention workload and 047 campaign
- [docs/HEATMAP_REDESIGN.md](docs/HEATMAP_REDESIGN.md) — figure redesign and captions
- [docs/LLM_SDC_PILOT.md](docs/LLM_SDC_PILOT.md) — LLM SDC pilot and phased plan
- [docs/LLM_STRUCTURE_INJECTION.md](docs/LLM_STRUCTURE_INJECTION.md) — LLM block structure and injection points
- [docs/EXPERIMENT_PLAN_HIGHBIT_OUTPUT.md](docs/EXPERIMENT_PLAN_HIGHBIT_OUTPUT.md) — high-bit activation plan
- [docs/REPORT_HIGHBIT_OUTPUT.md](docs/REPORT_HIGHBIT_OUTPUT.md) — high-bit output-corruption pilot
- [docs/REPORT_HIGHBIT_OUTPUT_200.md](docs/REPORT_HIGHBIT_OUTPUT_200.md) — strict clean 200-prompt Qwen experiment
- [docs/TEST_REPORT_HIGHBIT_OUTPUT_200.md](docs/TEST_REPORT_HIGHBIT_OUTPUT_200.md) — high-bit 200-prompt test report
- [docs/EXPERIMENT_PLAN_AGENT_GEMM.md](docs/EXPERIMENT_PLAN_AGENT_GEMM.md) — long-context agent GEMM injection plan
- [docs/ARCHIVE_scalar_gemm.md](docs/ARCHIVE_scalar_gemm.md) — scalar GEMM archive

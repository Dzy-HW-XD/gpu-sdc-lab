# Research plan — Input-dependent GPU SDC (Phase 2)

Target: a measurement-focused paper with a lightweight input-aware method.
Hardware: a single NVIDIA A100. The lab provides the full infrastructure; this
document is the experiment program. See `PAPER_ABSTRACT.md` for the abstract.

## Thesis

For the same fault site (kernel, instruction, register, bit, dynamic position),
whether a fault is **MASKED**, becomes **SDC**, or escalates to **CRASH** is
governed by the kernel's **input distribution**; that dependence is predictable
from interpretable input features and exploitable by input-aware injection.

## Hypotheses

- **H1** SDC probability varies significantly with input distribution (fixed site).
- **H2** interpretable input features predict the SDC rate (leave-one-family/workload-out).
- **H3** the dependence is jointly modulated by instruction class and FP32 bit field.
- **H4** uniform injection mis-estimates SDC relative to the real input distribution.
- **H5** input-aware sampling discovers more SDCs per injection than uniform.

## Design: paired injection

Hold the fault site fixed and vary only the input, so any outcome difference is
attributable to the data (paired design; high power on a single GPU).

## Workload suite

`gemm`, `reduce`, `conv2d`, `softmax`, `attention`, `mlp` (training step) —
covering arithmetic, memory, atomics, reductions, special functions, tensor-op
composition, and forward+backward propagation. All deterministic.

## Input families

Synthetic: `uniform, normal, lognormal, sparse, cancellation, near_overflow,
correlated, adversarial`; plus one `real` tensor. Defined in `core/inputgen.py`.

## Experiment matrix

| # | experiment (dir) | independent var | dimension | kind |
|---|---|---|---|---|
| E1 | 001–007 (existing) | position / type / bit / size / pattern / atomic | various | — |
| E2 | 008_input_sensitivity_gemm | input family | input | input_sensitivity |
| E2 | 009_input_sensitivity_reduce | input family (atomics) | input | input_sensitivity |
| E3 | 010/011/012 | input family (softmax/conv/attention) | input | input_sensitivity |
| E4 | 013_input_sensitivity_mlp | input family, per-layer outputs | input | input_sensitivity |
| E5 | 014_cross_workload | workload | workload | cross_workload |
| E6 | 015_input_bit_gemm | input × bit 0–31 | bit, input | input_grid(bit) |
| E6 | 016_input_instruction_gemm | input × FFMA/FADD/FMUL | instruction, input | input_grid(instruction) |
| E6 | 017_input_size_gemm | input × size | size, input | input_grid(size) |
| E6 | 018_input_position_gemm | input × early/middle/late | position, input | input_grid(position) |
| E6 | 019_register_class_gemm | input × accumulator/temp | register, input | input_sensitivity (+regclass) |
| E8 | predictor | input features | — | analysis (scripts/predictor.py) |
| E9 | input-aware sampling | policy | — | analysis (scripts/sampling.py) |

Templates use small `injections_per_input` (20); scale per the statistics below.

## Statistics

- Per-cell: ~385 injections for a ±5% Wilson CI at p≈0.5 (fewer for small p).
- Paired comparisons: exact McNemar; multiple groups: Holm–Bonferroni.
- Report Wilson CIs, effect sizes, and baseline noise (golden is bit-exact).
- Fixed seeds; repeat runs; power analysis before scaling.

## Lightweight method

Input features (`scripts/features.py`) → ridge/GB model (`scripts/predictor.py`)
→ input-aware injection/sampling (`scripts/sampling.py`); metric = SDCs
discovered per 1000 injections vs uniform. No external tool is required as a
baseline; the comparison is against uniform sampling within this lab.

## Threats to validity

Single architecture (A100); register-destination fault model only (no DRAM/ECC/
transient); synthetic inputs vs real data (mitigated by one real tensor);
per-op oracle exactness for deterministic kernels.

## Roadmap

1. Infrastructure: workloads, input generators, multi-output oracle, taxonomy,
   analysis scripts, experiment templates (done).
2. Input × variable campaign tooling: `input_grid` kind, FFMA/FADD/FMUL
   opcode-filtered targets, static register-class labelling
   (`core/regclass.py`), batch driver (`scripts/campaign.py`) and stdlib
   heatmaps (`scripts/heatmap.py`); run E6 (015–019) on gemm and produce the
   core input × bit figures with Wilson CIs.
3. Scale E2/E4 on gemm/reduce; produce core input-sensitivity figures with CIs.
4. Add E3 across the suite; produce instruction-class × input interactions.
5. E4 multi-layer propagation on `mlp`; E5 cross-workload consistency.
6. E8 predictor + E9 input-aware sampling; write up; artifact.

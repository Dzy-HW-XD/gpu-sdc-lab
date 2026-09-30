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
| E6 | bit × dynamic range | sign/exp/mantissa × input | bit, input | (fault_bit + inputs) |
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

1. Infrastructure (this commit): workloads, input generators, multi-output
   oracle, taxonomy, analysis scripts, experiment templates.
2. Scale E2/E4 on gemm/reduce; produce core input-sensitivity figures with CIs.
3. Add E3 across the suite; produce instruction-class × input interactions.
4. E4 multi-layer propagation on `mlp`; E5 cross-workload consistency.
5. E8 predictor + E9 input-aware sampling; write up; artifact.

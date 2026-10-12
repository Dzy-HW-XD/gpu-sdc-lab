# Tensor-Core campaign — tolerance levels (L0–L5)

Offline re-classification of the **same** HMMA accumulator injections under a
set of `(abs_tol, rel_tol)` levels (decoupled from the GPU run; see
`scripts/tolerance_sweep.py`). An element is corrupted iff
`|fault-golden| > abs_tol + rel_tol*|golden|`; a run whose output is bitwise
identical to golden is MASKED at every level.

| level | abs_tol | rel_tol | note |
|---|---|---|---|
| **L0** | 0 | 0 | strict / bit-exact (reference) |
| L1 | 1e-7 | 1e-6 | base point |
| L2 | 1e-6 | 1e-5 | ×10 |
| L3 | 1e-5 | 1e-4 | ×100 |
| L4 | 1e-4 | 1e-3 | ×1000 |
| L5 | 1e-3 | 1e-2 | ×10000 (≈ "1% relative is acceptable") |

## SDC rate by experiment × level

| experiment (accumulator position / axis) | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 020_tc_input_bit_early (first third of HMMA stream) | 0.929 | 0.814 | 0.702 | 0.597 | 0.499 | 0.398 |
| 021_tc_input_bit_middle | 0.970 | 0.851 | 0.747 | 0.649 | 0.540 | 0.434 |
| 022_tc_input_bit_late (last third) | 0.985 | 0.862 | 0.760 | 0.655 | 0.548 | 0.445 |
| 023_tc_input_size (input × 512/1024/2048) | 0.969 | 0.819 | 0.731 | 0.600 | 0.475 | 0.381 |

## Notes

- Tolerance strongly reduces the SDC rate for HMMA FP32 accumulators
  (e.g. 022: 0.985 → 0.445), i.e. a large fraction of accumulator bit flips
  produce only small relative output changes.
- The **early < middle < late** ordering holds at every level: faults later in
  the accumulator's lifetime remain more exposed even under a loose tolerance.
- Each experiment × level also has a per-cell (input distribution × bit / size)
  breakdown in `tolerance_by_cell.csv`, and per-input in `tolerance_by_input.csv`.

## Figures

`figs/` contains one input × bit (or input × size) SDC-rate heatmap per
experiment per level (24 SVGs + pivot CSVs):

```
figs/heatmap_020_tc_input_bit_early_bit_L{0..5}_sdc_rate.svg
figs/heatmap_021_tc_input_bit_middle_bit_L{0..5}_sdc_rate.svg
figs/heatmap_022_tc_input_bit_late_bit_L{0..5}_sdc_rate.svg
figs/heatmap_023_tc_input_size_size_L{0..5}_sdc_rate.svg
```

## Reproduce

```bash
python3 scripts/tolerance_sweep.py \
  020_tc_input_bit_early 021_tc_input_bit_middle 022_tc_input_bit_late 023_tc_input_size \
  --out results/_tc_tolerance \
  --levels L0=0:0,L1=1e-7:1e-6,L2=1e-6:1e-5,L3=1e-5:1e-4,L4=1e-4:1e-3,L5=1e-3:1e-2
python3 scripts/tolerance_heatmaps.py \
  --cells results/_tc_tolerance/tolerance_by_cell.csv \
  --outdir results/_tc_tolerance/figs
```

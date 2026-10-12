# Input-dependent FP32 GEMM SDC campaign

All results from `results/_campaign/cells.csv` (Wilson CIs included). Fault model: single-bit flip in the destination register of a targeted FP32 instruction; MASKED/SDC/CRASH classified against a bit-exact golden run. injection counts from `master.csv`.

## Overall (by experiment)

| experiment | sdc | n | rate |
|---|---|---|---|
| 020_tc_input_bit_early | 3566 | 3840 | 0.929 |
| 021_tc_input_bit_middle | 3723 | 3840 | 0.970 |
| 022_tc_input_bit_late | 3782 | 3840 | 0.985 |
| 023_tc_input_size | 349 | 360 | 0.969 |


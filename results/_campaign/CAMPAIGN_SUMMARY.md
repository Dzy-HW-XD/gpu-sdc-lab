# Input-dependent FP32 GEMM SDC campaign

All results from `results/_campaign/cells.csv` (Wilson CIs included). Fault model: single-bit flip in the destination register of a targeted FP32 instruction; MASKED/SDC/CRASH classified against a bit-exact golden run. injection counts from `master.csv`.

## Overall (by experiment)

| experiment | sdc | n | rate |
|---|---|---|---|
| 015_input_bit_gemm | 5130 | 5760 | 0.891 |
| 016_input_instruction_gemm | 690 | 760 | 0.908 |
| 017_input_size_gemm | 313 | 360 | 0.869 |
| 018_input_position_gemm | 326 | 360 | 0.906 |
| 019_register_class_gemm | 260 | 300 | 0.867 |

## SDC rate by FP32 bit region (015)

| region | sdc | n | rate |
|---|---|---|---|
| exponent | 1271 | 1440 | 0.883 |
| mantissa | 3723 | 4140 | 0.899 |
| sign | 136 | 180 | 0.756 |

## SDC rate by input (015, pooled over bits 0-31)

| input | sdc | n | rate |
|---|---|---|---|
| correlated | 906 | 960 | 0.944 |
| near_zero | 857 | 960 | 0.893 |
| normal | 855 | 960 | 0.891 |
| ones | 860 | 960 | 0.896 |
| sparse | 766 | 960 | 0.798 |
| uniform | 886 | 960 | 0.923 |

## SDC rate by instruction (016)

| opcode | sdc | n | rate |
|---|---|---|---|
| FADD | 187 | 204 | 0.917 |
| FFMA | 340 | 360 | 0.944 |
| FMUL | 163 | 196 | 0.832 |

## SDC rate by register class (019)

| register_class | sdc | n | rate |
|---|---|---|---|
| accumulator | 209 | 239 | 0.874 |
| temporary | 46 | 51 | 0.902 |

## SDC rate by GEMM size (017)

| size | sdc | n | rate |
|---|---|---|---|
| 1024x1024x1024 | 113 | 120 | 0.942 |
| 128x128x128 | 99 | 120 | 0.825 |
| 512x512x512 | 101 | 120 | 0.842 |

## SDC rate by injection position (018)

| position | sdc | n | rate |
|---|---|---|---|
| early | 108 | 120 | 0.900 |
| late | 112 | 120 | 0.933 |
| middle | 106 | 120 | 0.883 |


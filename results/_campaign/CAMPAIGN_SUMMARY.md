# Input-dependent FP32 GEMM SDC campaign

All results from `results/_campaign/cells.csv` (Wilson CIs included). Fault model: single-bit flip in the destination register of a targeted FP32 instruction; MASKED/SDC/CRASH classified against a bit-exact golden run. injection counts from `master.csv`.

## Overall (by experiment)

| experiment | sdc | n | rate |
|---|---|---|---|
| 015_input_bit_gemm | 7026 | 7680 | 0.915 |
| 016_input_instruction_gemm | 1065 | 1153 | 0.924 |
| 017_input_size_gemm | 491 | 540 | 0.909 |
| 018_input_position_gemm | 502 | 540 | 0.930 |
| 019_register_class_gemm | 389 | 420 | 0.926 |

## SDC rate by FP32 bit region (015)

| region | sdc | n | rate |
|---|---|---|---|
| exponent | 1780 | 1920 | 0.927 |
| mantissa | 5020 | 5520 | 0.909 |
| sign | 226 | 240 | 0.942 |

## SDC rate by input (015, pooled over bits 0-31)

| input | sdc | n | rate |
|---|---|---|---|
| correlated | 876 | 960 | 0.912 |
| extreme | 879 | 960 | 0.916 |
| near_overflow | 870 | 960 | 0.906 |
| near_zero | 885 | 960 | 0.922 |
| normal | 881 | 960 | 0.918 |
| ones | 883 | 960 | 0.920 |
| sparse | 883 | 960 | 0.920 |
| uniform | 869 | 960 | 0.905 |

## SDC rate by instruction (016)

| opcode | sdc | n | rate |
|---|---|---|---|
| FADD | 284 | 303 | 0.937 |
| FFMA | 537 | 553 | 0.971 |
| FMUL | 244 | 297 | 0.822 |

## SDC rate by register class (019)

| register_class | sdc | n | rate |
|---|---|---|---|
| accumulator | 307 | 317 | 0.968 |
| temporary | 72 | 74 | 0.973 |

## SDC rate by GEMM size (017)

| size | sdc | n | rate |
|---|---|---|---|
| 1024x1024x1024 | 167 | 180 | 0.928 |
| 128x128x128 | 162 | 180 | 0.900 |
| 512x512x512 | 162 | 180 | 0.900 |

## SDC rate by injection position (018)

| position | sdc | n | rate |
|---|---|---|---|
| early | 162 | 180 | 0.900 |
| late | 166 | 180 | 0.922 |
| middle | 174 | 180 | 0.967 |


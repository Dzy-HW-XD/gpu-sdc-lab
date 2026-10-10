# Input-dependent FP32 GEMM SDC campaign

All results from `results/_campaign/cells.csv` (Wilson CIs included). Fault model: single-bit flip in the destination register of a targeted FP32 instruction; MASKED/SDC/CRASH classified against a bit-exact golden run. injection counts from `master.csv`.

## Overall (by experiment)

| experiment | sdc | n | rate |
|---|---|---|---|
| 015_input_bit_gemm | 6479 | 7680 | 0.844 |
| 016_input_instruction_gemm | 865 | 1151 | 0.752 |
| 017_input_size_gemm | 455 | 540 | 0.843 |
| 018_input_position_gemm | 451 | 540 | 0.835 |
| 019_register_class_gemm | 354 | 420 | 0.843 |

## SDC rate by FP32 bit region (015)

| region | sdc | n | rate |
|---|---|---|---|
| exponent | 1485 | 1920 | 0.773 |
| mantissa | 4817 | 5520 | 0.873 |
| sign | 177 | 240 | 0.738 |

## SDC rate by input (015, pooled over bits 0-31)

| input | sdc | n | rate |
|---|---|---|---|
| correlated | 906 | 960 | 0.944 |
| extreme | 751 | 960 | 0.782 |
| near_overflow | 598 | 960 | 0.623 |
| near_zero | 857 | 960 | 0.893 |
| normal | 855 | 960 | 0.891 |
| ones | 860 | 960 | 0.896 |
| sparse | 766 | 960 | 0.798 |
| uniform | 886 | 960 | 0.923 |

## SDC rate by instruction (016)

| opcode | sdc | n | rate |
|---|---|---|---|
| FADD | 216 | 315 | 0.686 |
| FFMA | 486 | 552 | 0.880 |
| FMUL | 163 | 284 | 0.574 |

## SDC rate by register class (019)

| register_class | sdc | n | rate |
|---|---|---|---|
| accumulator | 284 | 336 | 0.845 |
| temporary | 62 | 69 | 0.899 |

## SDC rate by GEMM size (017)

| size | sdc | n | rate |
|---|---|---|---|
| 1024x1024x1024 | 158 | 180 | 0.878 |
| 128x128x128 | 149 | 180 | 0.828 |
| 512x512x512 | 148 | 180 | 0.822 |

## SDC rate by injection position (018)

| position | sdc | n | rate |
|---|---|---|---|
| early | 148 | 180 | 0.822 |
| late | 150 | 180 | 0.833 |
| middle | 153 | 180 | 0.850 |


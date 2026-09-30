# gpu-sdc-lab — Phase 1 summary

## 001_gemm_baseline (baseline)

- runs: 10, bitwise-identical: 10 -> DETERMINISTIC

## 002_fault_position

- injections: 100 | MASKED 6 | SDC 82 | CRASH 12 | NOT_INJECTED 0 | SDC rate 0.820

- by group: fp32(m0/s33/c1), gp(m1/s21/c11), ld(m5/s28/c0)

- SDC max abs err 1.66e+37, max rel-L2 1.518e+33, max corrupted elems 1



## 003_fault_type

- injections: 100 | MASKED 6 | SDC 92 | CRASH 2 | NOT_INJECTED 0 | SDC rate 0.920

- by group: fp32(m6/s92/c2)

- SDC max abs err 1.08e+37, max rel-L2 9.877e+32, max corrupted elems 1



## 004_fault_bit

- injections: 96 | MASKED 3 | SDC 90 | CRASH 3 | NOT_INJECTED 0 | SDC rate 0.938

- by group: fp32(m3/s90/c3)

- by FP32 region: exponent(m0/s24/c0), mantissa(m3/s63/c3), sign(m0/s3/c0)

- SDC max abs err 1.029e+20, max rel-L2 9.413e+15, max corrupted elems 1



## 005_gemm_size

- injections: 40 | MASKED 0 | SDC 32 | CRASH 8 | NOT_INJECTED 0 | SDC rate 0.800

- by group: fp32(m0/s32/c8)

- SDC max abs err 4451, max rel-L2 0.4224, max corrupted elems 1



## 006_data_pattern

- injections: 100 | MASKED 4 | SDC 92 | CRASH 4 | NOT_INJECTED 0 | SDC rate 0.920

- by group: fp32(m4/s92/c4)

- SDC max abs err 1.657e+33, max rel-L2 1.516e+35, max corrupted elems 1



## 007_reduce_atomic

- injections: 60 | MASKED 30 | SDC 27 | CRASH 3 | NOT_INJECTED 0 | SDC rate 0.450

- by group: atom(m15/s0/c0), fp32(m4/s11/c0), gp(m6/s6/c3), ld(m5/s10/c0)

- SDC max abs err 2.052, max rel-L2 0.00359, max corrupted elems 1




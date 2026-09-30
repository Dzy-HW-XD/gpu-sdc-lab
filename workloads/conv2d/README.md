# conv2d workload

Deterministic FP32 direct 2-D convolution (`NCHW`, batch 1, stride 1, same
padding). Exercises `ld`, `fp32`, `gp` (index arithmetic) and shared-memory-free
direct loops.

- Inputs: `X [Cin,H,W]`, `W [Cout,Cin,K,K]`; output `Y [Cout,H,W]`.
- Optional `--input` / `--weight` load raw float32; otherwise `det_value`.
- Observable: `Y` (bit-exact, fixed accumulation order).

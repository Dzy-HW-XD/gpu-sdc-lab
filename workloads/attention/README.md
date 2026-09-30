# attention workload

Deterministic single-head FP32 attention using an **explicit math path** (no
flash/SDPA, no TF32). `X [S,3D]` packs Q,K,V; `O = softmax(QKᵀ/√D)·V`, `O [S,D]`.
One block per query row with fixed-order reductions -> bit-exact golden.

Exercises `fp32`, `ld`, `gp`, and `MUFU`/others (exp).  This is the operator that
links GEMM + softmax in a real model, so it is the natural bridge to the
training-step workload.

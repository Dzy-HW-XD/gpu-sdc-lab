# softmax workload

Deterministic FP32 row-wise softmax (`Y = softmax(X, axis=1)`). One block per
row with fixed-order shared-memory reductions, so the golden is bit-exact.
Exercises `fp32`, `ld`, `gp`, and `MUFU`/others (`expf`).

- Input `X [R,C]`; output `Y [R,C]`.
- Optional `--input` loads raw float32; otherwise `det_value`.
- Observable: `Y`.

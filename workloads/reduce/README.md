# reduce workload

Deterministic FP32 block reduction, used to exercise **atomics** and shared
memory in the SDC lab.

- Deterministic input array (`det_value`, same generator as GEMM).
- Kernel `reduce_blocks`: each block sums a strided chunk, tree-reduces in
  shared memory, then thread 0:
  - writes the block sum to `partials[blockIdx.x]` (the observable),
  - `atomicAdd(&g_sum, block_sum)` (`ATOM.ADD.F32`),
  - `atomicAdd(&g_count, 1)` (`ATOM.ADD.S32`).
- The atomic *return values* are kept live (see `g_dummy`) so nvcc emits `ATOM`
  (with destination register) instead of `RED` (no destination) — this is what
  makes the `atom` instruction group injectable.

## Oracle

The observable is the `partials` array, which is **bit-exact** (per-block order
is fixed, unlike a global float atomic sum). The float atomic accumulator is
order-dependent and is recorded in metadata only; the integer atomic counter is
deterministic. So baseline reproducibility is bit-exact and exact comparison is
a valid SDC threshold.

## CLI

```
reduce --N 1048576 --blocks 256 --threads 256 --dtype fp32 \
       --seed 1234 --device 0 --out /path/out
```

Outputs `<out>.bin` (raw float32 partials) and `<out>.json` (metadata incl.
`atomic_sum`, `atomic_count`, `checksum`). Expects `ATOM` instructions to be
present: `./sdc-lab profile` on this workload should report a non-zero `atom`
count once the atom group patch is applied.

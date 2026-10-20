# Why a real fault injection can be MASKED — SASS trace

Case: experiment `040_attn_stage_smoke`, seq 2 — **qk, head 0, bit 23**
(`HMMA@0x19f0`, `regNo=62`, `thread=114`, `inst_id=3666/4096`), classified
**MASKED** (Y bitwise equal to golden).

Register evidence (from `observations.jsonl`):

```
mask        = 0x800000          (bit 23 = FP32 exponent LSB)
before_val  = 0xc3c66ebc        = -396.865
after_val   = 0xc3466ebc        = -198.433    (exponent 135 -> 134, value / 2)
```

So the injection **did** change the register; it was not a failed injection.

## SASS source

The qk GEMM kernel is `ampere_s16816gemm_bf16_64x64_sliced1x2_ldg8_stages_64x5_tn`
from `libcublasLt.so.13.6.0.2` (sm_80). `cuobjdump -sass/-xelf` on the whole
library crashes, so the cubin was located by scanning the sm_80 ELF images and
extracting `libcublasLt.so.5303.sm_80.cubin` (see
`scripts/attention_find_cubin.py`). The disassembly is
`results/_attn_sass/cub5303.sass`; the `_tn` function is
`results/_attn_sass/tn.sass`.

## Trace of the injected accumulator

```
/*11c0*/  CS2R R60, SRZ            // init accumulator quad R60..R63 = 0
/*11d0*/  CS2R R62, SRZ
...
/*1500*/  ...                      // k-loop body starts
/*1720*/  HMMA.16816.F32.BF16 R60, R108, R134, R60   // accumulate into R60 quad
/*19f0*/  HMMA.16816.F32.BF16 R60, R124, R142, R60   // <-- injection site
/*1a40*/  @P1 BRA.U 0x1500         // k-loop
...
/*2910*/  FMUL R110, R81, R60      // scale by alpha
/*2920*/  FMUL R111, R81, R62      // R62 is consumed here
...
/*2df0*/  @P3 STG.E.EF.128 [R76.64],  R104          // R104..R107
/*2e00*/  @P4 STG.E.EF.128 [R122.64], R108          // R108..R111  (R111 <- R62)
```

`HMMA Rd, ...` writes a **quad of FP32 registers** `Rd..Rd+3`. The injected
`regNo=62` is lane 2 of the `R60..R63` quad. R62 is:

1. initialized to zero before the loop,
2. accumulated in the k-loop (`Rc = R60`),
3. **read in the epilogue** (`FMUL R111, R81, R62`),
4. written out by the `STG.128 [R122.64], R108` store (R111 ∈ R108..R111).

There is **no write to R62 between the injection and the store** in this
function, so the corruption is not simply overwritten by a later instruction.

## Why it is still MASKED

The final store is **predicated**:

```
/*2b70*/  ISETP.LT.AND P3, PT, R85,  0x0, P2
/*2b80*/  ISETP.LT.AND P4, PT, R85, -0x4, P2
/*2df0*/  @P3 STG.E.EF.128 [R76.64],  R104
/*2e00*/  @P4 STG.E.EF.128 [R122.64], R108
```

`R85` is a bounds/offset value derived in the epilogue (`IADD3 R85, R86,
-c[0x0][0x1cc]`), i.e. these stores only fire for fragments whose output
element is inside the matrix/tile bounds. If the dynamic instruction instance
that was hit (`thread=114`, `inst_id=3666/4096`) belongs to a fragment whose
`P4` is false (padding / out-of-range / non-storing slice), the corrupted
`R111` is **never written**: the final `Y` stays bit-identical and the run is
classified MASKED.

This also explains why the same `(stage, head, bit)` can be SDC in one run and
MASKED in another: the sampled dynamic instance/thread decides whether the
fragment's store is taken.

## Related case

`pv, head 0, bit 23` (`HMMA@0x18c0`, reg 15) had `before_val = 0x0` ->
`after_val = 0x800000` = `2^-126`. Here the corrupted value is negligible and
is absorbed by the subsequent accumulation/rounding, independent of the store
predicate.

## Caveat / next step

The SASS identifies the **gate** (predicated epilogue store). To *prove* that
`P4` is false for thread 114 one would need the thread -> warp-fragment ->
`R85` mapping (evaluate the epilogue predicate for that lane). The register
liveness trace above already rules out the simpler "overwritten register"
explanation for this instance.

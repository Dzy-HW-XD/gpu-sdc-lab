# Attention SDC experiments (`attention_forward`)

Status: **Phase 1–3 implemented and validated on `node20` (A100, sm_80)**.
Phases 4–5 (target-size single-vs-multi campaign, extra input distributions)
are still open; see the end of this document.

## Model

Causal self-attention forward pass (standard, no flash/SDPA, no KV cache):

```
X [S,D] bf16
Q = X Wq, K = X Wk, V = X Wv          (bf16 x bf16 -> fp32, cuBLAS / tensor cores)
per head h (dh = D/H):
    Sh = Qh Kh^T / sqrt(dh)            (separate cuBLAS launch per head)
    Ph = causal_softmax(Sh)            (explicit fp32 kernel)
    Oh = Ph Vh                         (separate cuBLAS launch per head)
Y = concat(Oh) Wo                      (bf16 x bf16 -> fp32)
```

* `H=1, dh=D` is standard single-head; `H>1, dh=D/H` is multi-head.
* Causal mask: `j > i` is excluded from the softmax (probability exactly 0).
* Q/K/V are rounded to BF16 before the next GEMM (BF16 attention data path);
  softmax is FP32. cuBLAS atomics are disabled for bit-reproducible runs.

## Fault targeting: stage and head

NVBitFI selects a dynamic instruction ordinal within a kernel; it cannot see
"which head" inside a *batched* cuBLAS GEMM. Therefore score and PV are launched
**once per head**. The profiler then shows a deterministic launch map (verified
for `S=64,D=256,H=4`):

| launch index | kernel | role |
|---|---|---|
| 0,1,2 | tensor-core GEMM | Q, K, V projections |
| 6..9 | `ampere_s16816gemm_bf16..._tn` | score, heads 0..3 |
| 10 | `softmax_causal` | softmax (not injected) |
| 11..14 | `ampere_s16816gemm_bf16..._nn` | PV, heads 0..3 |
| 17 | tensor-core GEMM | output projection Wo |

`AttentionForwardWorkload.select_kernel()` maps `(stage, head)` to the
corresponding launch (it filters kernels with `mma > 0`, so cast/softmax/concat
launches are skipped). The existing `FaultSpec.kernel_count` then selects that
exact launch.

Stage labels used by the `attention_stage` experiment kind:

```
qkv_q, qkv_k, qkv_v   projection-level (affects all heads)
qk                    score GEMM of one head
pv                    PV GEMM of one head
wo                    output projection (not part of the first-phase plan)
```

## Intermediate tensors

`--dump 1` writes raw little-endian float32:

```
<out>.Q.bin     [S,D]       <out>.S.bin   [H,S,S]  raw scores
<out>.K.bin     [S,D]       <out>.P.bin   [H,S,S]  softmax probabilities
<out>.V.bin     [S,D]       <out>.O.bin   [H,S,dh] per-head outputs
<out>.Ocat.bin  [S,D]       <out>.Y.bin   [S,D]    final output
<out>.json      shapes, seed, backend, checksum
```

## Commands (on `node20`, `/home/dzy/gpu-sdc-lab`)

```bash
# build + structural validation (single and multi head)
./sdc-lab baseline --experiment 030_attn_single_dev_baseline   # S=64 D=256 H=1
./sdc-lab baseline --experiment 031_attn_multi_dev_baseline    # S=64 D=256 H=4
python3 scripts/attention_validate.py results/031_attn_multi_dev_baseline/raw/000/out

# target size
./sdc-lab baseline --experiment 032_attn_single_baseline        # S=128 D=2048 H=1
./sdc-lab baseline --experiment 033_attn_multi_baseline         # S=128 D=2048 H=32

# kernel launch map for injection targeting
python3 scripts/attention_profile.py --S 64 --D 256 --H 4 --tag attn_dev

# real single-fault injections (3 stages)
./sdc-lab run --experiment 040_attn_stage_smoke
# broader dev campaign (5 stage/head groups x bits {0,27,29})
./sdc-lab run --experiment 041_attn_stage_dev

# per-injection intermediate metrics + aggregates (stdlib)
python3 scripts/attention_collect.py results/041_attn_stage_dev

# dashboards (matplotlib; run where matplotlib is installed)
python scripts/attention_plots.py \
    --metrics <per-run metrics json> --outdir plots --runs-jsonl <attention_runs.jsonl>
```

`scripts/attention_selftest.py` checks the metric code with synthetic known
diffs (single changed element, per-head/per-token support, identical -> MASKED).

## Validation performed (real runs on node20)

| check | result |
|---|---|
| `030`/`031` baseline determinism (2 runs) | bitwise identical 2/2 |
| `032`/`033` baseline determinism (3 runs) | bitwise identical 3/3 |
| softmax row sums (dev & target) | max error ~1e-7 |
| causal mask violations | 0 |
| `Ocat` == concat(`O`) | exact (0 mismatch) |
| target-size runtime (3 un-instrumented runs) | ~2.3–2.4 s total |

### Real injection campaign `041_attn_stage_dev`

S=64, D=256, H=4; one fault per run; bit flip in an HMMA destination register.
21 injections, **0 NOT_INJECTED, 0 CRASH**:

| stage | head | n | MASKED | SDC | mean Y rel-L2 |
|---|---|---|---|---|---|
| qkv_q | – | 3 | 3 | 0 | 0 |
| qkv_k | – | 3 | 2 | 1 | 0.081 |
| qkv_v | – | 3 | 3 | 0 | 0 |
| qk | 0 | 3 | 2 | 1 | 0.036 |
| qk | 2 | 3 | 3 | 0 | 0 |
| pv | 0 | 3 | 3 | 0 | 0 |
| pv | 2 | 3 | 2 | **1** | **3.5e16** |

All SDCs occurred at **bit 29** (exponent); bits 0 and 27 were masked in this
sample. The PV/head-2/bit-29 fault was localized to head 2 (`O` head error
6.8e19, other heads exactly 0), to token row 9, and propagated through the
output projection to all 256 features of that token (Y rel-L2 1.0e17).

## Known limitations / open issues

* **Dev size only so far.** The target campaign (S=128, D=2048, H=1 vs H=32)
  has only been run as a *baseline*; injections at target size are Phase 4.
* **Per-head launches cost more.** B/C-MultiHead uses H score + H PV launches
  instead of one batched GEMM. This is required for head-attributable faults.
* **Softmax/concat/Wo are not injected** in Phase 1–3 (they are observation
  points only), per the proposal.
* **No numpy/matplotlib on node20.** Metrics/aggregation are stdlib; figures are
  generated locally from downloaded tensors/metrics.
* **One fault per run**, single GPU; no decode/KV-cache/LLM replay.
* `attention_forward` currently generates synthetic X and weights from the seed;
  input-family control (sparse/near-zero/correlated) is Phase 5.

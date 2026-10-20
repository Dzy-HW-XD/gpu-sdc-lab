# Attention SDC experiments (`attention_forward`)

Status: **Phase 1–3 implemented and validated on `node20` (A100, sm_80)**.
The current campaign is **`047_attn_5gemm_full`**: five GEMM stages, bits
24–30, random head for score/PV, classified with **L4** (`abs=1e-4`,
`rel=1e-3`) only. Phases 4–5 (single-vs-multi target campaign, input
distributions) remain open; see the end of this document.

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

# five-GEMM campaign (qkv_q/k/v, qk, pv), bits 24-30, L4 oracle, random head
python3 scripts/campaign.py --lab-config config.L4.yaml --gpu 0 run \
    --shard 0 --nshards 4 --tag gpu0 047_attn_5gemm_full
# ... repeat for gpu1/2/3; then merge/collect
python3 scripts/attention_merge_shards.py --eid 047_attn_5gemm_full
python3 scripts/attention_select_runs.py --exp results/047_attn_5gemm_full
python3 scripts/attention_head_errors.py --exp results/047_attn_5gemm_full
python3 scripts/attention_injection_trace.py --exp results/047_attn_5gemm_full

# dashboards (matplotlib; run where matplotlib is installed)
python scripts/attention_plots.py \
    --metrics <per-run metrics json> --outdir plots --runs-jsonl <runs.jsonl>
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

### Five-GEMM L4 campaign `047_attn_5gemm_full`

Target MHA: **S=128, D=2048, H=32, dh=64**; one fault per run; bit flip in an
HMMA destination register; classified with **L4 only**
(`|Y_f − Y_g| > 1e-4 + 1e-3·|Y_g|`).

* stages: `qkv_q`, `qkv_k`, `qkv_v` (shared projections) and `qk`, `pv`
  (per-head GEMMs; the head is chosen uniformly at random per injection from
  0..31, recorded in `labels.head`)
* bits **24..30**; 90 injections per (stage, bit) = 30 early + 30 middle +
  30 late
* total **3150** injections, sharded over 4 GPUs

**Results: 1507 / 3150 SDC (47.8%), 1643 MASKED, 0 CRASH / NOT_INJECTED.**

| stage | L4 SDC rate | median affected tokens | exactly 1 token |
|---|---|---|---|
| qkv_q | 0.544 | 1 | 93.6% |
| qkv_k | **0.633** | **16** | 17.0% |
| qkv_v | 0.549 | 2 | 44.5% |
| qk | 0.146 | 1 | 100% |
| pv | 0.519 | 1 | 99.4% |

Position: early 0.432, middle 0.483, late 0.520. Bit: 24 0.280, 25 0.393,
26 0.518, 27 0.582, 28 0.544, 29 0.578, 30 0.453.

Propagation (mean relative L2, representative runs):

* `qkv_q` / `qk`: S ~1e15 -> P/O/Ocat/Y ~1e-2 (softmax damps the score error)
* `qkv_k`: S ~1e31 -> P/O/Ocat/Y ~1e-2
* `qkv_v`: S/P ~1e-2 -> O/Ocat/Y ~1e35
* `pv`: S/P = 0 -> O/Ocat/Y ~1e35

Footprint: Q/Score/PV faults are **query-local** (one token row per run; across
30 random instances the support map covers many tokens). **K faults are broad**
(causal fan-out: queries i >= key j; median 16 tokens, up to ~70); V is
intermediate (median 2). This is why the final-Y heatmaps for `qkv_k` show
multiple rows while `qkv_q/qk/pv` show one row per run.

Outputs: `final_heatmaps/` (105 figures; 30-run mean `log10|Y_f-Y_g|`
(non-zero) with a **global fixed** `[-3, 6]` colour range, `>1e6` clipped;
30-run L4 support
fixed `0-1`; cumulative Token/Feature coverage; SDC count/rate, mean relL2 and
support ratio in the title/stats panel), `stage_heatmaps/` (103 S/P/O `|diff|`
figures with **global per-tensor** log ranges and fixed Token/Feature axes),
`stage_summary/`, `bit_sensitivity/`, `footprint/`, `injection_trace.csv/jsonl`.
There is no L0–L5 sweep; L4 is the single classification.

## Known limitations / open issues

* **L4 only.** `047` classifies with L4 (`abs=1e-4`, `rel=1e-3`); there is no
  L0–L5 sweep in the current campaign.
* **bit 30 can overflow.** Some bit-30 runs produce Inf/NaN; the comparator
  counts them as SDC, and the finite-error medians drop at bit 30.
* **Single-vs-multi comparison still open.** The target-size (S=128, D=2048)
  campaign is now multi-head (H=32); a matched single-head (H=1) injection
  campaign has not been run yet.
* **Per-head launches cost more.** Multi-head uses H score + H PV launches
  instead of one batched GEMM. This is required for head-attributable faults.
* **Softmax/concat/Wo are not injected** in Phase 1–3 (they are observation
  points only), per the proposal.
* **No numpy/matplotlib on node20.** Metrics/aggregation are stdlib; figures are
  generated locally from downloaded tensors/metrics.
* **One fault per run**, single GPU; no decode/KV-cache/LLM replay.
* `attention_forward` currently generates synthetic X and weights from the seed;
  input-family control (sparse/near-zero/correlated) is Phase 5.

## Publication figures (v2)

The v1 dashboard-style heatmaps have been superseded by a publication-quality
set; see `docs/HEATMAP_REDESIGN.md` for the critique, the plotting-skill
checklist, and caption templates.  New outputs:

* `results/047_attn_5gemm_full/final_heatmaps_v2/` — 105 cells, PDF + 600 dpi
  PNG, with a small review footer;
* `results/047_attn_5gemm_full/final_heatmaps_v2_paper/` — the same 105 cells
  as PDF without the footer, for manuscript captions;
* `results/047_attn_5gemm_full/final_heatmaps_v2/overview/` — a 5 x 3
  stage-by-position summary, recommended as the main-text figure.

The v2 data (`final_heatmaps_data_v2_nf/`) is non-finite aware: NaN/Inf
outputs are counted as affected (55 of 3150 runs, 11 cells, all at bit 30)
and shown in black on the error panel.  This fixes the v1 case where a cell
could report `SDC 13/30` but appear empty.

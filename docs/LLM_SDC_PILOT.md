# LLM SDC pilot and plan (A100, <=1.5B)

Status: **pilot completed** on node20, GPU 0, 2026-10-20.

## Environment

- GPU: NVIDIA A100-SXM4-80GB, shared with another user's vLLM (~25 GB used)
- PyTorch 2.14.1+cu126, transformers 5.20.2, Python 3.12
- venv: `/home/dzy/llm_sdc_env`
- Process memory cap: `torch.cuda.set_per_process_memory_fraction(0.20, 0)`
- Model: `EleutherAI/pythia-160m` (162M params, FP32), 620.5 MiB GPU memory
- No OOM; max allocated memory far below the free 56 GB.

## Pilot design

- Prompts (3):
  - `The capital of France is` -> ` Paris`
  - `The capital of Japan is` -> ` Tokyo`
  - `2 + 3 =` -> ` 5`
- Layers: 0, 6, 11 (early/middle/late)
- Weights: `attention.query_key_value.weight`, `mlp.dense_h_to_4h.weight`
- Bits: 22, 24, 26, 28, 30, 31 (FP32; 23–30 exponent, 31 sign)
- One bit flip in the **largest-magnitude weight element** of each weight
- 108 forward runs (3 prompts × 3 layers × 2 weights × 6 bits)
- Tensor-level injection, **not NVBitFI instruction-level**
- Baseline was deterministic (repeat max diff 0.0)

## Results

| bit | top-1 changes | non-finite logits | median finite max\|Δlogit\| | median NLL delta |
|---:|---:|---:|---:|---:|
| 22 | 0/18 | 0 | 0.00055 | 0 |
| 24 | 0/18 | 0 | 0.0023 | 0 |
| 26 | 0/18 | 0 | 0.0027 | 0 |
| 28 | 0/18 | 0 | 0.0027 | 0 |
| **30** | **13/18** | **8/18** | **577.3** | **4.15** |
| 31 | 0/18 | 0 | 0.010 | 0 |

Bit 30:
- Attention `qkv` weight: 8/9 runs produce NaN logits (huge QKV -> softmax NaN).
- MLP `h_to_4h` weight: 0 NaN; finite logit changes up to 578; 5/9 top-1 changes.
- Layer differences: layer 6/11 attention all NaN; layer 0 attention partially NaN; MLP effects depend on prompt/unit.

Interpretation:
- A single lower exponent/mantissa bit in one weight element is mostly masked at the task level (no top-1 change).
- The exponent MSB (bit 30 in FP32) is catastrophic: attention becomes NaN, MLP produces huge finite logits.
- Attention and MLP have different failure modes (NaN vs finite huge values).
- Task metrics (top-1/NLL) are much less sensitive than element-level logits; this is exactly why LLM experiments need task-level metrics.
- The 160M model is a pipeline pilot: its factual next-token prediction is weak (France/Japan both top-1 4441), so semantic conclusions need 410M/1B and better prompts.

## Caveats

- Tensor-level injection, not hardware/NVBitFI instruction-level.
- FP32 weights; real LLM inference often uses BF16 (exponent bits 7–14, not 24–30).
- One element per weight, chosen as the largest magnitude; not a random fault population.
- 3 short prompts; no generation, KV cache, or decode.
- GPU is shared with vLLM; memory cap protects against OOM but timing is not isolated.

## Plan (no 7B)

Models: Pythia-160M/410M/1B, Qwen2.5-0.5B/1.5B. 1.5B is the upper bound for
campaigns; no 7B/8B/70B in this phase.

| phase | question | model | target | injection | metrics |
|---|---|---|---|---|---|
| P1 | weight bit sensitivity | 410M/1B | qkv/o/mlp weights | tensor, random elements, bits 0–31 + BF16 bits | logits, NLL, top-1 |
| P2 | activation bit sensitivity | 160M/410M | hidden, Q/K/V, MLP out | tensor/hook | logits, row-relative, top-1 |
| P3 | KV-cache persistence | 160M/410M | K/V cache, layer/head/position/step | tensor | decode NLL, top-1, attention KL, KV cosine |
| P4 | layer/head/GQA | 410M/1B + Qwen2.5-1.5B | selected layers/heads | tensor + NVBitFI GEMM | task metrics, attention entropy |
| P5 | quantization | 410M/1B | BF16/FP16/INT8 weights/KV | tensor | bit sensitivity, task metrics |
| P6 | detection/mitigation | 1B | high-signal sites | — | detection rate, cost |

### A100 constraints

- 160M/410M/1B/1.5B all fit comfortably; use a 20% memory cap when sharing.
- No FP8 hardware on A100; use BF16/FP16/INT8; FP8 is emulation only.
- NVBitFI is suitable for cuBLAS GEMM (HMMA) but not FlashAttention/Triton/PagedAttention; start with eager PyTorch and tensor-level injection, then add NVBitFI for GEMMs.
- Do not use vLLM for the first controlled experiments (continuous batching/nondeterminism); use eager PyTorch, batch 1, greedy, fixed seeds.
- Store metrics and selected tensors only; LLM hidden states/dumps are large.

## Next two small experiments

1. **Activation bit flip** at layer 6, last token, bits 22/24/26/28/30/31; compare with weight faults.
2. **KV-cache bit flip** at decode step 1/8, 16-token generation; measure persistence and attention/KV metrics.

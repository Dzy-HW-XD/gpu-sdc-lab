# LLM block structure and injection points (prefill)

Figure: [`docs/figures/llm_structure_flow.pdf`](figures/llm_structure_flow.pdf)
(PNG: [`docs/figures/llm_structure_flow.png`](figures/llm_structure_flow.png))

The top panel is the **horizontal prefill flow** for the Pythia and Qwen2.5
families: tokens → embedding → repeated transformer blocks → final norm →
logits. The E/M/L markers show the early/middle/late injection depths. The
bottom panels zoom into one Pythia block and one Qwen2.5 block, marking the
W/A/N/R injection points. Decode and KV-cache injection are not part of this
figure.

## Model configuration

| model | family | layers | hidden | Q / KV heads | head dim | MLP | norm | activation | position | residual | context | vocab | embedding |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|
| Pythia-160M | GPT-NeoX | 12 | 768 | 12 / 12 (MHA) | 64 | 3072 (4×) | LayerNorm | GELU | partial RoPE (25%) | parallel | 2048 | 50304 | untied |
| Pythia-410M | GPT-NeoX | 24 | 1024 | 16 / 16 (MHA) | 64 | 4096 (4×) | LayerNorm | GELU | partial RoPE (25%) | parallel | 2048 | 50304 | untied |
| Pythia-1B | GPT-NeoX | 16 | 2048 | 8 / 8 (MHA) | 256 | 8192 (4×) | LayerNorm | GELU | partial RoPE (25%) | parallel | 2048 | 50304 | untied |
| Qwen2.5-0.5B | Qwen2 | 24 | 896 | 14 / 2 (GQA) | 64 | 4864 (5.4×) | RMSNorm | SiLU/SwiGLU | full RoPE (θ=1e6) | sequential | 32768 | 151936 | tied |
| Qwen2.5-1.5B | Qwen2 | 28 | 1536 | 12 / 2 (GQA) | 128 | 8960 (5.8×) | RMSNorm | SiLU/SwiGLU | full RoPE (θ=1e6) | sequential | 131072 | 151936 | tied |

## Injection points in the figure

| marker | meaning | Pythia module | Qwen2.5 module |
|---|---|---|---|
| `W_QKV` | fused Q/K/V projection | `gpt_neox.layers[l].attention.query_key_value.weight` | — |
| `W_Q` / `W_K` / `W_V` | separate Q/K/V projections | — | `model.layers[l].self_attn.q_proj/k_proj/v_proj` |
| `W_O` | attention output projection | `attention.dense.weight` | `self_attn.o_proj` |
| `W_up` | MLP up projection | `mlp.dense_h_to_4h.weight` | `mlp.up_proj` |
| `W_gate` | SwiGLU gate | — | `mlp.gate_proj` |
| `W_down` | MLP down projection | `mlp.dense_4h_to_h.weight` | `mlp.down_proj` |
| `A` | activation output (attention or MLP) | attention output / MLP output | attention output / MLP output |
| `N` | normalization | `input_layernorm` / `post_attention_layernorm` | `input_layernorm` / `post_attention_layernorm` |
| `R` | residual stream | residual stream | residual stream |

The figure marks **W** (weight), **A** (activation), **N** (norm) and **R**
(residual) injection points. Each injection is parameterized by layer `l` and
token position `p`; in prefill, only positions `>= p` can be affected through
the causal attention mask.

## Depth and E/M/L layers

The flow lanes use `Block N/2` as a generic middle depth; the exact per-model
depths are:

- Pythia-160M: 12 layers, E/M/L = 0/6/11
- Pythia-410M: 24 layers, E/M/L = 0/12/23
- Pythia-1B: 16 layers, E/M/L = 0/8/15
- Qwen2.5-0.5B: 24 layers, E/M/L = 0/12/23
- Qwen2.5-1.5B: 28 layers, E/M/L = 0/14/27

## Why the two block diagrams differ

**Pythia (parallel residual)**

```
x_out = x_in + Attn(LN(x_in)) + MLP(LN(x_in))
```

Attention and MLP are computed in parallel from the same normalized residual.
This makes attention-vs-MLP branch attribution relatively clean.

**Qwen2.5 (sequential pre-norm)**

```
x1 = x + Attn(RMSNorm(x))
x2 = x1 + MLP(RMSNorm(x1))
```

Attention output feeds the residual and then the same-layer MLP. This creates
intra-layer compounding; GQA also means one KV head serves multiple query
heads (7 for 0.5B, 6 for 1.5B).

## Reproduce

```bash
python scripts/llm_structure_diagram.py
```

The prefill propagation figure is separate:

```bash
python scripts/llm_prefill_propagation_plot.py \
  --data results/llm_pilot/prefill_activation \
  --out results/llm_pilot/prefill_activation/figures \
  --bit max
```

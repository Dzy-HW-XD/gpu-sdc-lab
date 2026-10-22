#!/usr/bin/env python3
"""Layer-14 qkv_v injection followed by clean downstream layers 15..27.

Captures four checkpoints:
  V, attention output, layer-14 block output, layer-27 block output.
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as qwen2

CAP = {}


def install_observer(layer):
    original = qwen2.eager_attention_forward

    def observed(module, query, key, value, attention_mask, scaling,
                 dropout=0.0, **kwargs):
        out = original(module, query, key, value, attention_mask, scaling,
                       dropout=dropout, **kwargs)
        if getattr(module, "layer_idx", -1) == layer:
            CAP["attn_context"] = out[0].detach()
            CAP["attn_weights"] = out[1].detach()
        return out

    qwen2.eager_attention_forward = observed
    try:
        qwen2.ALL_ATTENTION_FUNCTIONS["eager"] = observed
    except Exception:
        pass


def run_block(model, layer_idx, hidden_in, mask, pos_emb, position_ids):
    layer = model.model.layers[layer_idx]
    CAP.clear()
    v_holder = {}
    attn_holder = {}

    def v_hook(module, inputs, output):
        v_holder["v"] = output.detach()

    def attn_hook(module, inputs, output):
        attn_holder["out"] = output[0].detach()

    h1 = layer.self_attn.v_proj.register_forward_hook(v_hook)
    h2 = layer.self_attn.register_forward_hook(attn_hook)
    try:
        block_out = layer(
            hidden_in,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=pos_emb,
        )
    finally:
        h1.remove()
        h2.remove()
    return {
        "V": v_holder["v"],
        "attn_context": CAP.get("attn_context"),
        "attn_output": attn_holder["out"],
        "block_output": block_out,
    }


def run_downstream(model, start_layer, hidden, mask, pos_emb, position_ids):
    for i in range(start_layer + 1, len(model.model.layers)):
        hidden = model.model.layers[i](
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=pos_emb,
        )
    return hidden


def stats(x, ref):
    if not torch.is_tensor(x):
        return {"missing": True}
    x = x.float().cpu()
    ref = ref.float().cpu()
    finite = bool(torch.isfinite(x).all().item())
    n_nan = int((~torch.isfinite(x)).sum().item())
    if x.shape != ref.shape:
        return {"shape_mismatch": [list(x.shape), list(ref.shape)],
                "finite": finite, "n_nan_inf": n_nan,
                "bitwise_equal": False, "mismatch_count": -1}
    diff = x - ref
    absd = diff.abs()
    denom = ref.abs().mean().clamp_min(1e-30)
    xd = x.double()
    rd = ref.double()
    dd = xd - rd
    absd64 = dd.abs()
    denom64 = rd.abs().mean().clamp_min(1e-30)
    return {
        "bitwise_equal": bool(torch.equal(x, ref)),
        "mismatch_count": int((x != ref).sum().item()),
        "numel": int(x.numel()),
        "max_abs": float(absd64.max().item()),
        "mean_abs": float(absd64.mean().item()),
        "rel_l2": float((dd.norm() / rd.norm().clamp_min(1e-30)).item()),
        "rel_max": float((absd64.max() / denom64).item()),
        "finite": finite,
        "n_nan_inf": n_nan,
    }


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(x) for x in obj)
    if isinstance(obj, list):
        return [to_cuda(x) for x in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("clean", "profile", "inject"), required=True)
    ap.add_argument("--mem-fraction", type=float, default=0.20)
    args = ap.parse_args()

    t0 = time.time()
    rec = torch.load(args.input, map_location="cpu")
    model_name = rec["model"]
    layer_idx = int(rec["layer"])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda").eval()
    model.config.use_cache = False
    install_observer(layer_idx)

    hidden_in = to_cuda(rec["hidden_in"])
    position_ids = to_cuda(rec["position_ids"])
    pos_emb = to_cuda(rec["position_embeddings"])
    mask = to_cuda(rec["causal_mask"]) if torch.is_tensor(rec["causal_mask"]) else rec["causal_mask"]
    clean = {k: v.float() for k, v in rec["clean"].items()}

    with torch.inference_mode():
        outs = run_block(model, layer_idx, hidden_in, mask, pos_emb, position_ids)
        outs["final_output"] = run_downstream(
            model, layer_idx, outs["block_output"], mask, pos_emb, position_ids)

    result = {
        "mode": args.mode,
        "model": model_name,
        "layer": layer_idx,
        "prompt_tokens": int(rec["prompt_tokens"]),
        "elapsed_s": None,
        "max_mem_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "checkpoints": {},
        "runtime_error": None,
    }
    for name, tensor in outs.items():
        result["checkpoints"][name] = stats(tensor, clean[name])
    result["elapsed_s"] = round(time.time() - t0, 3)
    result["v_bitwise_equal"] = result["checkpoints"]["V"].get("bitwise_equal")
    result["attn_bitwise_equal"] = result["checkpoints"]["attn_output"].get("bitwise_equal")
    result["block_bitwise_equal"] = result["checkpoints"]["block_output"].get("bitwise_equal")
    result["final_bitwise_equal"] = result["checkpoints"]["final_output"].get("bitwise_equal")
    result["nan_inf_any"] = any(not result["checkpoints"][k].get("finite", True)
                                for k in ("V", "attn_output", "block_output", "final_output"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

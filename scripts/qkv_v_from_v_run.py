#!/usr/bin/env python3
"""Run layer14 attention/block from a precomputed qkv_v V matrix.

Variant V is produced by a standalone CUTLASS/cuBLAS qkv_v kernel.  The V
matrix is injected into layer14's v_proj; Q/K are recomputed normally, so the
attention path is identical to the Qwen block.
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


def load_v(path, S, N):
    b = bytearray(open(path, "rb").read())
    x = torch.frombuffer(b, dtype=torch.int16).view(torch.bfloat16).reshape(S, N)
    return x


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
    x = x.float().cpu()
    ref = ref.float().cpu()
    finite = bool(torch.isfinite(x).all().item())
    n_nan = int((~torch.isfinite(x)).sum().item())
    if x.shape != ref.shape:
        return {"shape_mismatch": [list(x.shape), list(ref.shape)],
                "bitwise_equal": False, "mismatch_count": -1,
                "finite": finite, "n_nan_inf": n_nan}
    xd = x.double()
    rd = ref.double()
    d = xd - rd
    ad = d.abs()
    return {
        "bitwise_equal": bool(torch.equal(x, ref)),
        "mismatch_count": int((x != ref).sum().item()),
        "numel": int(x.numel()),
        "max_abs": float(ad.max().item()),
        "mean_abs": float(ad.mean().item()),
        "rel_l2": float((d.norm() / rd.norm().clamp_min(1e-30)).item()),
        "rel_max": float((ad.max() / rd.abs().mean().clamp_min(1e-30)).item()),
        "finite": finite,
        "n_nan_inf": n_nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/qkv_v_block/inputs/layer14.pt")
    ap.add_argument("--v", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", default="", help="clean checkpoint .pt for fault mode")
    ap.add_argument("--mode", choices=("clean", "fault"), required=True)
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
    layer = model.model.layers[layer_idx]

    hidden_in = rec["hidden_in"].cuda()
    position_ids = rec["position_ids"].cuda()
    pos_emb = tuple(x.cuda() for x in rec["position_embeddings"])
    mask = rec["causal_mask"].cuda() if torch.is_tensor(rec["causal_mask"]) else rec["causal_mask"]
    S = int(hidden_in.shape[1])
    N = int(layer.self_attn.v_proj.out_features)
    v_raw = load_v(args.v, S, N).to(torch.float32)
    bias = layer.self_attn.v_proj.bias.detach().to(torch.float32).cpu()
    v = (v_raw + bias).to(torch.bfloat16).unsqueeze(0).cuda()

    # Replace only V; Q/K and all later operations remain the Qwen path.
    def v_hook(module, inputs, output):
        return v

    attn_holder = {}

    def attn_hook(module, inputs, output):
        attn_holder["out"] = output[0].detach()

    h1 = layer.self_attn.v_proj.register_forward_hook(v_hook)
    h2 = layer.self_attn.register_forward_hook(attn_hook)
    try:
        with torch.inference_mode():
            block = layer(
                hidden_in,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=pos_emb,
            )
            final = run_downstream(model, layer_idx, block, mask, pos_emb, position_ids)
    finally:
        h1.remove()
        h2.remove()

    cps = {"V": v, "attn_output": attn_holder["out"],
           "block_output": block, "final_output": final}

    result = {
        "mode": args.mode,
        "model": model_name,
        "layer": layer_idx,
        "prompt_tokens": S,
        "variant_file": args.v,
        "elapsed_s": round(time.time() - t0, 3),
        "checkpoints": {},
        "runtime_error": None,
    }
    if args.mode == "clean":
        out = {"clean": {k: t.detach().cpu() for k, t in cps.items()},
               "meta": {"S": S, "N": N, "model": model_name, "layer": layer_idx}}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        torch.save(out, args.out)
        result["saved"] = args.out
    else:
        clean = torch.load(args.clean, map_location="cpu")["clean"]
        for k in ("V", "attn_output", "block_output", "final_output"):
            result["checkpoints"][k] = stats(cps[k], clean[k])
        result["v_equal"] = result["checkpoints"]["V"]["bitwise_equal"]
        result["attn_equal"] = result["checkpoints"]["attn_output"]["bitwise_equal"]
        result["block_equal"] = result["checkpoints"]["block_output"]["bitwise_equal"]
        result["final_equal"] = result["checkpoints"]["final_output"]["bitwise_equal"]
        result["nan_inf_any"] = any(not result["checkpoints"][k]["finite"] for k in
                                     ("V", "attn_output", "block_output", "final_output"))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

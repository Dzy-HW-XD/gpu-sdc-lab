#!/usr/bin/env python3
"""Small LLM pilot: tensor-level bit flips in Pythia-160M weights.

This is a software/tensor-level injection pilot, not NVBitFI instruction
injection.  It validates the model/metrics pipeline before moving to
instruction-level HMMA faults.

For each (layer, weight, bit):
  - pick the largest-magnitude weight element
  - flip one bit in its FP32 representation
  - run 3 fixed prompts
  - record logit L2/max error, top-1 change, target NLL change
"""
import csv
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
OUT = "/home/dzy/gpu-sdc-lab/results/llm_pilot/pythia160_weight_bits.csv"
PROMPTS = [
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("2 + 3 =", " 5"),
]
LAYERS = [0, 6, 11]
WEIGHTS = [
    ("attn_qkv", "attention.query_key_value.weight"),
    ("mlp_h_to_4h", "mlp.dense_h_to_4h.weight"),
]
BITS = [22, 24, 26, 28, 30, 31]


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    # vLLM peers use ~25 GB; cap this process well below the remaining space.
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "device", torch.cuda.get_device_name(0), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    print("model loaded", sum(p.numel() for p in model.parameters()) / 1e6, "M params",
          "allocated", round(torch.cuda.max_memory_allocated() / 2**20, 1), "MiB", flush=True)

    inputs = []
    for prompt, target in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        tid = tok.encode(target, add_special_tokens=False)[0]
        inputs.append((prompt, target, ids, tid))

    with torch.no_grad():
        base = []
        for prompt, target, ids, tid in inputs:
            logits = model(input_ids=ids).logits[0, -1]
            nll = -torch.log_softmax(logits, dim=-1)[tid].item()
            top1 = int(torch.argmax(logits).item())
            base.append({"logits": logits.detach().cpu(), "nll": nll, "top1": top1})
            print("baseline", repr(prompt), "nll", round(nll, 6), "top1", top1, flush=True)
        # determinism check
        logits2 = model(input_ids=inputs[0][2]).logits[0, -1]
        print("baseline repeat max diff",
              float((logits2 - base[0]["logits"].to("cuda")).abs().max()), flush=True)

    rows = []
    layer_mods = model.gpt_neox.layers
    for layer_idx in LAYERS:
        for wname, wpath in WEIGHTS:
            mod = layer_mods[layer_idx]
            for part in wpath.split("."):
                mod = getattr(mod, part)
            w = mod
            orig = w.detach().clone()
            flat_idx = int(torch.argmax(w.detach().abs()).item())
            orig_val = float(w.detach().view(-1)[flat_idx].item())
            print("weight", layer_idx, wname, "shape", tuple(w.shape),
                  "idx", flat_idx, "orig", orig_val, flush=True)
            for bit in BITS:
                # restore then flip
                w.data.copy_(orig)
                ints = w.detach().view(torch.int32).view(-1)
                ints[flat_idx] = ints[flat_idx] ^ (1 << bit)
                fault_val = float(w.detach().view(-1)[flat_idx].item())
                for pi, (prompt, target, ids, tid) in enumerate(inputs):
                    with torch.no_grad():
                        logits = model(input_ids=ids).logits[0, -1]
                        d = logits - base[pi]["logits"].to("cuda")
                        nll = -torch.log_softmax(logits, dim=-1)[tid].item()
                        top1 = int(torch.argmax(logits).item())
                        rows.append({
                            "model": MODEL, "prompt_id": pi, "prompt": prompt,
                            "target": target, "layer": layer_idx,
                            "weight": wname, "bit": bit, "flat_idx": flat_idx,
                            "orig": orig_val, "fault": fault_val,
                            "max_abs_logit": float(d.abs().max().item()),
                            "rel_l2_logit": float((d.norm() / base[pi]["logits"].norm()).item()),
                            "top1_base": base[pi]["top1"], "top1_fault": top1,
                            "top1_change": int(top1 != base[pi]["top1"]),
                            "nll_base": base[pi]["nll"], "nll_fault": nll,
                            "nll_delta": nll - base[pi]["nll"],
                        })
                w.data.copy_(orig)
                print("  bit", bit, "fault_val", fault_val,
                      "top1_changes", sum(r["top1_change"] for r in rows[-3:]),
                      "max_dlogit", round(max(r["max_abs_logit"] for r in rows[-3:]), 6),
                      flush=True)
            w.data.copy_(orig)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prefill-only activation propagation pilot on Pythia-160M.

Injects one bit into one element of a layer's output activation for one token
position, then measures how the perturbation propagates to later layers and
later token positions in the same prefill forward.  No decode, no KV cache.

Outputs:
  prefill_propagation_matrix.csv  (run x out_layer x token)
  prefill_runs.csv                (one row per injection)
"""
import csv
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
OUTDIR = "/home/dzy/gpu-sdc-lab/results/llm_pilot/prefill_activation"
PROMPT = ("The capital of France is Paris. The capital of Japan is Tokyo. "
          "The capital of Italy is Rome.")
LAYERS = [0, 6, 11]
POSITIONS = [0, 10, 20]
BITS = [24, 28, 30]
MAX_LEN = 24


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.to("cuda").eval()
    model.config.use_cache = False

    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :MAX_LEN].to("cuda")
    seq = ids.shape[1]
    print("seq", seq, "prompt tokens", tok.decode(ids[0]), flush=True)

    with torch.no_grad():
        base = model(input_ids=ids, output_hidden_states=True)
    hs0 = base.hidden_states
    logits0 = base.logits[0, -1].detach()
    print("hidden layers", len(hs0), "shape", tuple(hs0[0].shape),
          "max mem", round(torch.cuda.max_memory_allocated() / 2**20, 1), "MiB", flush=True)

    runs = []
    matrix = []
    for layer in LAYERS:
        mod = model.gpt_neox.layers[layer]
        for pos in POSITIONS:
            # choose the largest-magnitude element of the golden activation
            # at this layer output / token position
            h = hs0[layer + 1][0, pos]
            idx = int(torch.argmax(h.abs()).item())
            for bit in BITS:
                def hook(module, inputs, output, pos=pos, idx=idx, bit=bit):
                    is_tuple = isinstance(output, tuple)
                    hs = output[0] if is_tuple else output
                    rest = tuple(output[1:]) if is_tuple else ()
                    orig_shape = hs.shape
                    if hs.dim() == 2:
                        hs3 = hs.view(1, hs.shape[0], hs.shape[1]).clone()
                    elif hs.dim() == 3:
                        hs3 = hs.clone()
                    else:
                        raise RuntimeError("unexpected hook output shape %s" % (orig_shape,))
                    hsi = hs3.view(torch.int32)
                    hsi[0, pos, idx] = hsi[0, pos, idx] ^ (1 << bit)
                    new = hs3.view(orig_shape)
                    return (new,) + rest if is_tuple else new
                handle = mod.register_forward_hook(hook)
                try:
                    with torch.no_grad():
                        out = model(input_ids=ids, output_hidden_states=True)
                finally:
                    handle.remove()
                hs1 = out.hidden_states
                logits1 = out.logits[0, -1]
                dlog = logits1 - logits0.to("cuda")
                top1_base = int(torch.argmax(logits0).item())
                top1_fault = int(torch.argmax(logits1).item())
                affected_layers = 0
                affected_min_token = seq
                max_rel = 0.0
                for l in range(len(hs0)):
                    for t in range(seq):
                        d = hs1[l][0, t] - hs0[l][0, t]
                        rel = float(d.norm() / torch.clamp(hs0[l][0, t].norm(), min=1e-12))
                        cos = float(F.cosine_similarity(
                            hs1[l][0, t].unsqueeze(0),
                            hs0[l][0, t].unsqueeze(0)).item())
                        mx = float(d.abs().max())
                        if l > layer:
                            if rel > 0:
                                affected_layers = max(affected_layers, l)
                                affected_min_token = min(affected_min_token, t)
                                max_rel = max(max_rel, rel)
                            matrix.append({"run_layer": layer, "pos": pos, "bit": bit,
                                           "out_layer": l, "token": t,
                                           "rel_l2": rel, "cosine": cos,
                                           "max_abs": mx})
                runs.append({"layer": layer, "pos": pos, "bit": bit,
                             "idx": idx, "fault_act": float(hs1[layer + 1][0, pos, idx]),
                             "logit_max_abs": float(dlog.abs().max()),
                             "logit_rel_l2": float(dlog.norm() / torch.clamp(logits0.norm(), min=1e-12)),
                             "top1_base": top1_base, "top1_fault": top1_fault,
                             "top1_change": int(top1_base != top1_fault),
                             "affected_layers": affected_layers,
                             "affected_min_token": affected_min_token if affected_min_token < seq else -1,
                             "max_hidden_rel": max_rel})
                print("layer", layer, "pos", pos, "bit", bit,
                      "top1_chg", int(top1_base != top1_fault),
                      "logit_max", round(float(dlog.abs().max()), 4),
                      "max_hidden_rel", round(max_rel, 6), flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "prefill_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader(); w.writerows(runs)
    with open(os.path.join(OUTDIR, "prefill_propagation_matrix.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
        w.writeheader(); w.writerows(matrix)
    print("wrote", OUTDIR, len(runs), "runs", len(matrix), "matrix rows", flush=True)


if __name__ == "__main__":
    main()

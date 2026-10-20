#!/usr/bin/env python3
"""High-bit activation fault injection -> generated-output corruption.

One FP32 activation element at the input to a chosen decoder block, at the
last prompt token, has one high-order bit flipped.  The model then generates a
greedy continuation.  Outputs are scored for literal garble, semantic
wrongness, and clean-reference preservation.

No KV-cache injection is performed.  Decode is used only to observe output.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_SPECS = {
    "EleutherAI/pythia-160m": {
        "short": "pythia160m", "label": "Pythia-160M", "layers": [0, 6, 11],
    },
    "EleutherAI/pythia-410m": {
        "short": "pythia410m", "label": "Pythia-410M", "layers": [0, 12, 23],
    },
    "EleutherAI/pythia-1b": {
        "short": "pythia1b", "label": "Pythia-1B", "layers": [0, 8, 15],
    },
    "Qwen/Qwen2.5-0.5B": {
        "short": "qwen05b", "label": "Qwen2.5-0.5B", "layers": [0, 12, 23],
    },
    "Qwen/Qwen2.5-1.5B": {
        "short": "qwen15b", "label": "Qwen2.5-1.5B", "layers": [0, 14, 27],
    },
}

FIELDS = [
    "model", "model_short", "condition", "prompt_id", "category", "prompt",
    "clean_output", "output", "layer", "layer_tag", "bit", "elem_idx",
    "old_int", "new_int", "bit_diff", "fault_value", "fault_finite",
    "runtime_error", "nonfinite_scores", "clean_correct", "fault_correct",
    "char_sim", "token_f1", "clean_nll", "fault_nll", "ppl_ratio",
    "replacement_rate", "control_rate", "empty", "degenerate",
    "injection_degenerate", "script_mix", "unexpected_script",
    "literal_garble", "semantic_wrong", "catastrophic", "partial",
    "preserved", "elapsed_s",
]


def norm_space(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def norm_compact(s):
    return re.sub(r"\s+", "", s.lower()).strip()


def expected_hit(text, expected):
    """Require the expected answer in the first sentence/line.

    This rejects rambling continuations where the answer appears only late.
    """
    if text is None:
        return False
    s = text.strip()
    first = re.split(r"(?<=[.!?])\s+|\n", s, maxsplit=1)[0]
    tok_re = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)
    out = tok_re.findall(first.lower())
    for e in expected:
        et = tok_re.findall(e.lower())
        if not et:
            continue
        if not any(out[i:i + len(et)] == et for i in range(len(out) - len(et) + 1)):
            return False
    return True


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_similarity(a, b):
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def token_f1(a, b):
    def toks(x):
        return re.findall(r"[a-z0-9]+", (x or "").lower())
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


def script_mix(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    bad = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if any(k in name for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL",
                                   "CYRILLIC", "ARABIC", "HEBREW", "THAI")):
            bad += 1
    return bad / len(letters)


def text_flags(s, clean_s=None):
    s = s or ""
    clean_s = clean_s or ""
    n = max(len(s), 1)
    replacement = sum(1 for ch in s if ch == "\ufffd") / n
    control = sum(
        1 for ch in s
        if unicodedata.category(ch).startswith("C") and ch not in "\n\r\t"
    ) / n
    empty = len(s.strip()) == 0
    smix = script_mix(s)

    words = re.findall(r"\S+", s.lower())
    degenerate = False
    if len(words) >= 8:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams:
            max_count = max(Counter(trigrams).values())
            rep_ratio = max_count / len(trigrams)
            bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
            distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
            degenerate = (max_count >= 4 and rep_ratio > 0.4) or distinct2 < 0.2

    clean_smix = script_mix(clean_s) if clean_s else 0.0
    unexpected_script = clean_smix < 0.05 and smix > 0.10
    return {
        "replacement_rate": replacement,
        "control_rate": control,
        "empty": int(empty),
        "degenerate": int(degenerate),
        "script_mix": smix,
        "unexpected_script": int(unexpected_script),
    }


def get_layers(model):
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("cannot locate decoder layers for %s" % type(model))


def make_hook(prompt_len, elem_idx, bit, state, sham=False):
    """Pre-hook on one decoder block; injects once, at the last prompt token."""
    def hook(module, args):
        if state.get("done"):
            return None
        hs = args[0]
        if not torch.is_tensor(hs) or hs.dim() != 3:
            return None
        if hs.shape[1] < prompt_len:
            return None
        pos = prompt_len - 1
        new_hs = hs.clone()
        ints = new_hs.view(torch.int32)
        old = int(ints[0, pos, elem_idx].item())
        if sham:
            new = old
        else:
            old_u = old & 0xFFFFFFFF
            new_u = old_u ^ (1 << bit)
            new = new_u - 2**32 if new_u >= 2**31 else new_u
            ints[0, pos, elem_idx] = new
        state.update({
            "done": True,
            "old_int": old,
            "new_int": new,
            "bit_diff": bin((old & 0xFFFFFFFF) ^ (new & 0xFFFFFFFF)).count("1"),
            "fault_value": float(new_hs[0, pos, elem_idx].item()),
        })
        if sham:
            return None
        return (new_hs,) + tuple(args[1:])
    return hook


@torch.no_grad()
def generate_one(model, tok, prompt, prompt_ids, prompt_len, layer_module,
                 elem_idx, bit, sham, max_new_tokens, device):
    state = {}
    hook = make_hook(prompt_len, elem_idx, bit, state, sham=sham)
    handle = layer_module.register_forward_pre_hook(hook)
    t0 = time.time()
    err = ""
    nonfinite = 0
    seq = None
    try:
        gen = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
        seq = gen.sequences
        if getattr(gen, "scores", None):
            scores = torch.stack(gen.scores, dim=0)
            nonfinite = int((~torch.isfinite(scores)).any(dim=-1).sum().item())
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, exc)
    finally:
        handle.remove()
    if seq is not None and not state.get("done") and not err:
        err = "hook not fired"
    elapsed = time.time() - t0
    if seq is None:
        return "", [], state, err, nonfinite, elapsed
    new_ids = seq[0, prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    return text, new_ids.tolist(), state, err, nonfinite, elapsed


@torch.no_grad()
def clean_nll(model, prompt_ids, prompt_len, clean_ids, layer_module,
              elem_idx, bit, sham):
    if not clean_ids:
        return None
    cont = torch.tensor([clean_ids], dtype=torch.long, device=prompt_ids.device)
    ids = torch.cat([prompt_ids, cont], dim=1)
    state = {}
    hook = make_hook(prompt_len, elem_idx, bit, state, sham=sham)
    handle = layer_module.register_forward_pre_hook(hook)
    try:
        logits = model(input_ids=ids, use_cache=False).logits
    except Exception:
        return None
    finally:
        handle.remove()
    try:
        lp = F.log_softmax(logits[:, prompt_len - 1:-1, :].float(), dim=-1)
        labels = ids[:, prompt_len:]
        nll = -lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
        return float(nll.item())
    except Exception:
        return None


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def write_summary(rows, path):
    groups = {}
    for r in rows:
        if r["condition"] == "clean_repeat":
            continue
        key = (r["condition"], r["layer_tag"], r["bit"])
        groups.setdefault(key, []).append(r)

    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        eval_rs = [r for r in rs if r["catastrophic"] not in ("", None)]
        n_eval = len(eval_rs)
        garbled = sum(int(r["literal_garble"]) for r in rs if r["literal_garble"] not in ("", None))
        wrong = sum(int(r["semantic_wrong"]) for r in eval_rs)
        cat = sum(int(r["catastrophic"]) for r in eval_rs)
        sims = [float(r["char_sim"]) for r in rs if r["char_sim"] not in ("", None)]
        ppls = [float(r["ppl_ratio"]) for r in rs
                if r["ppl_ratio"] not in ("", None) and math.isfinite(float(r["ppl_ratio"]))]
        nf = sum(int(r["nonfinite_scores"] or 0) for r in rs)
        ci = wilson(cat, n_eval)
        out.append({
            "condition": key[0], "layer_tag": key[1], "bit": key[2],
            "n": n, "n_eval": n_eval,
            "garbled_rate": garbled / n if n else "",
            "wrong_rate": wrong / n_eval if n_eval else "",
            "catastrophic_rate": cat / n_eval if n_eval else "",
            "catastrophic_ci_low": ci[0] if n_eval else "",
            "catastrophic_ci_high": ci[1] if n_eval else "",
            "mean_char_sim": statistics.mean(sims) if sims else "",
            "median_ppl_ratio": statistics.median(ppls) if ppls else "",
            "nonfinite_scores": nf,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--prompts", default="data/highbit_prompts.json")
    ap.add_argument("--outdir", default="results/llm_highbit_output")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default="")
    ap.add_argument("--bits", default="28,29,30,31")
    ap.add_argument("--clean-repeat", type=int, default=1)
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spec = MODEL_SPECS[args.model]
    layers = [int(x) for x in args.layers.split(",") if x != ""] or spec["layers"]
    bits = [int(x) for x in args.bits.split(",") if x != ""]

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)
    if args.limit:
        prompts = prompts[:args.limit]
    if not prompts:
        raise SystemExit("empty prompt suite")

    os.makedirs(args.outdir, exist_ok=True)
    runs_path = os.path.join(args.outdir, "runs_%s.csv" % spec["short"])
    summary_path = os.path.join(args.outdir, "summary_%s.csv" % spec["short"])
    meta_path = os.path.join(args.outdir, "meta_%s.json" % spec["short"])
    if os.path.exists(runs_path) and not args.overwrite:
        raise SystemExit("exists, use --overwrite: %s" % runs_path)

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
    print("model", args.model, "layers", layers, "bits", bits,
          "prompts", len(prompts), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float32)
    model.to("cuda").eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    layer_modules = get_layers(model)
    n_layers = len(layer_modules)
    print("n_layers", n_layers, "max_mem",
          round(torch.cuda.max_memory_allocated() / 2**20, 1), "MiB", flush=True)
    if any(l < 0 or l >= n_layers for l in layers):
        raise SystemExit("bad layer index for model")

    tags = {}
    for tag, idx in zip(("E", "M", "L"), layers):
        tags[idx] = tag

    rows = []
    t_start = time.time()
    for pi, p in enumerate(prompts, 1):
        prompt = p["prompt"]
        ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_len = int(ids.shape[1])
        print("[%d/%d] %s len=%d %r" % (pi, len(prompts), p["id"], prompt_len, prompt),
              flush=True)

        # clean baseline and clean activation
        with torch.no_grad():
            base = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        clean_out, clean_ids, _, err, _, _ = generate_one(
            model, tok, prompt, ids, prompt_len, layer_modules[layers[0]],
            0, 0, sham=True, max_new_tokens=args.max_new_tokens, device="cuda")
        if err:
            print("  clean generate error:", err, flush=True)
        clean_flags = text_flags(clean_out)
        clean_ok = expected_hit(clean_out, p["expected"])

        # clean NLL for the exact generated continuation
        clean_nll_val = clean_nll(
            model, ids, prompt_len, clean_ids, layer_modules[layers[0]],
            0, 0, sham=True)
        print("  clean correct=%d out=%r nll=%s" % (
            int(clean_ok), clean_out[:120], clean_nll_val), flush=True)

        clean_row = {
            "model": args.model, "model_short": spec["short"],
            "condition": "clean", "prompt_id": p["id"],
            "category": p["category"], "prompt": prompt,
            "clean_output": clean_out, "output": clean_out,
            "layer": "", "layer_tag": "", "bit": "", "elem_idx": "",
            "old_int": "", "new_int": "", "bit_diff": "", "fault_value": "",
            "fault_finite": "", "runtime_error": err, "nonfinite_scores": "",
            "clean_correct": int(clean_ok), "fault_correct": int(clean_ok),
            "char_sim": 1.0, "token_f1": 1.0,
            "clean_nll": clean_nll_val, "fault_nll": clean_nll_val,
            "ppl_ratio": 1.0,
            "replacement_rate": clean_flags["replacement_rate"],
            "control_rate": clean_flags["control_rate"],
            "empty": clean_flags["empty"],
            "degenerate": clean_flags["degenerate"],
            "injection_degenerate": 0,
            "script_mix": clean_flags["script_mix"],
            "unexpected_script": clean_flags["unexpected_script"],
            "literal_garble": int(
                clean_flags["replacement_rate"] > 0 or
                clean_flags["control_rate"] > 0 or
                clean_flags["empty"] or clean_flags["degenerate"]),
            "semantic_wrong": "", "catastrophic": "", "partial": "",
            "preserved": "", "elapsed_s": "",
        }
        rows.append(clean_row)

        for rep in range(1, args.clean_repeat):
            out2, _, _, err2, _, _ = generate_one(
                model, tok, prompt, ids, prompt_len, layer_modules[layers[0]],
                0, 0, sham=True, max_new_tokens=args.max_new_tokens, device="cuda")
            rows.append(dict(clean_row, condition="clean_repeat",
                             output=out2, runtime_error=err2,
                             elapsed_s=""))

        if args.clean_only:
            continue

        for layer_idx in layers:
            layer_module = layer_modules[layer_idx]
            h = base.hidden_states[layer_idx][0, -1]
            elem_idx = int(torch.argmax(h.abs()).item())
            tag = tags[layer_idx]

            # sham control for this layer
            sham_out, _, sham_state, sham_err, _, sham_elapsed = generate_one(
                model, tok, prompt, ids, prompt_len, layer_module,
                elem_idx, 0, sham=True, max_new_tokens=args.max_new_tokens,
                device="cuda")
            sham_flags = text_flags(sham_out, clean_out)
            sham_ok = expected_hit(sham_out, p["expected"])
            sham_sim = char_similarity(clean_out, sham_out)
            sham_lit = int(
                sham_flags["replacement_rate"] > 0 or
                sham_flags["control_rate"] > 0 or
                (sham_flags["empty"] and not clean_flags["empty"]) or
                (sham_flags["degenerate"] and not clean_flags["degenerate"]) or
                sham_flags["unexpected_script"])
            sham_sem = (int((not sham_ok) or sham_sim < 0.4)
                        if clean_ok else "")
            sham_cat = (int(sham_lit or (sham_sem == 1))
                        if clean_ok else "")
            sham_preserved = (int(not sham_cat and sham_ok and sham_sim >= 0.8)
                              if clean_ok else "")
            sham_partial = (int(not sham_cat and not sham_preserved)
                            if clean_ok else "")
            rows.append({
                "model": args.model, "model_short": spec["short"],
                "condition": "sham", "prompt_id": p["id"],
                "category": p["category"], "prompt": prompt,
                "clean_output": clean_out, "output": sham_out,
                "layer": layer_idx, "layer_tag": tag, "bit": "",
                "elem_idx": elem_idx, "old_int": sham_state.get("old_int", ""),
                "new_int": sham_state.get("new_int", ""),
                "bit_diff": sham_state.get("bit_diff", ""),
                "fault_value": sham_state.get("fault_value", ""),
                "fault_finite": int(math.isfinite(
                    float(sham_state.get("fault_value", 0.0)))),
                "runtime_error": sham_err, "nonfinite_scores": "",
                "clean_correct": int(clean_ok), "fault_correct": int(sham_ok),
                "char_sim": sham_sim,
                "token_f1": token_f1(clean_out, sham_out),
                "clean_nll": clean_nll_val, "fault_nll": "",
                "ppl_ratio": 1.0,
                "replacement_rate": sham_flags["replacement_rate"],
                "control_rate": sham_flags["control_rate"],
                "empty": sham_flags["empty"],
                "degenerate": sham_flags["degenerate"],
                "injection_degenerate": int(
                    sham_flags["degenerate"] and not clean_flags["degenerate"]),
                "script_mix": sham_flags["script_mix"],
                "unexpected_script": sham_flags["unexpected_script"],
                "literal_garble": sham_lit,
                "semantic_wrong": sham_sem,
                "catastrophic": sham_cat,
                "partial": sham_partial,
                "preserved": sham_preserved,
                "elapsed_s": sham_elapsed,
            })

            for bit in bits:
                out, _, state, err, nonfinite, elapsed = generate_one(
                    model, tok, prompt, ids, prompt_len, layer_module,
                    elem_idx, bit, sham=False,
                    max_new_tokens=args.max_new_tokens, device="cuda")
                flags = text_flags(out, clean_out)
                out_ok = expected_hit(out, p["expected"])
                sim = char_similarity(clean_out, out)
                inj_deg = int(flags["degenerate"] and not clean_flags["degenerate"])
                lit = int(
                    flags["replacement_rate"] > 0 or
                    flags["control_rate"] > 0 or
                    (flags["empty"] and not clean_flags["empty"]) or
                    inj_deg or flags["unexpected_script"])
                fault_nll_val = clean_nll(
                    model, ids, prompt_len, clean_ids, layer_module,
                    elem_idx, bit, sham=False)
                if clean_nll_val is not None and fault_nll_val is not None:
                    diff = fault_nll_val - clean_nll_val
                    ppl = math.exp(min(max(diff, -50.0), 50.0))
                else:
                    ppl = ""
                sem = int((not out_ok) or sim < 0.4 or
                           (ppl != "" and ppl > 5.0) or nonfinite > 0
                           ) if clean_ok else ""
                cat = int(lit or (sem == 1)) if clean_ok else ""
                preserved = (int(not cat and out_ok and sim >= 0.8)
                             if clean_ok else "")
                partial = (int(not cat and not preserved) if clean_ok else "")
                rows.append({
                    "model": args.model, "model_short": spec["short"],
                    "condition": "fault", "prompt_id": p["id"],
                    "category": p["category"], "prompt": prompt,
                    "clean_output": clean_out, "output": out,
                    "layer": layer_idx, "layer_tag": tag, "bit": bit,
                    "elem_idx": elem_idx,
                    "old_int": state.get("old_int", ""),
                    "new_int": state.get("new_int", ""),
                    "bit_diff": state.get("bit_diff", ""),
                    "fault_value": state.get("fault_value", ""),
                    "fault_finite": int(math.isfinite(
                        float(state.get("fault_value", 0.0)))),
                    "runtime_error": err, "nonfinite_scores": nonfinite,
                    "clean_correct": int(clean_ok), "fault_correct": int(out_ok),
                    "char_sim": sim,
                    "token_f1": token_f1(clean_out, out),
                    "clean_nll": clean_nll_val, "fault_nll": fault_nll_val,
                    "ppl_ratio": ppl,
                    "replacement_rate": flags["replacement_rate"],
                    "control_rate": flags["control_rate"],
                    "empty": flags["empty"], "degenerate": flags["degenerate"],
                    "injection_degenerate": inj_deg,
                    "script_mix": flags["script_mix"],
                    "unexpected_script": flags["unexpected_script"],
                    "literal_garble": lit,
                    "semantic_wrong": sem, "catastrophic": cat,
                    "partial": partial,
                    "preserved": preserved,
                    "elapsed_s": elapsed,
                })
                print("  L%d %s bit%d idx=%d out=%r%s%s" % (
                    layer_idx, tag, bit, elem_idx, out[:70],
                    " ERR=" + err if err else "",
                    " NF=%d" % nonfinite if nonfinite else ""), flush=True)

    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    write_summary(rows, summary_path)
    meta = {
        "model": args.model, "short": spec["short"], "layers": layers,
        "bits": bits, "prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "clean_repeat": args.clean_repeat,
        "elapsed_s": round(time.time() - t_start, 3),
        "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", runs_path, summary_path, meta_path, flush=True)
    print("elapsed_s", meta["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()

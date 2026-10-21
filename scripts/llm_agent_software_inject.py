#!/usr/bin/env python3
"""Software GEMM-output bit injection for long-context agent cases.

FP32 model, one bit flip in one FP32 GEMM output element, one fault per run.
Stages: qkv_q/k/v, qk, pv, wo, mlp_gate/up/down.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import time
import unicodedata

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import transformers.models.qwen2.modeling_qwen2 as qwen2
from transformers import AutoModelForCausalLM, AutoTokenizer


STAGES = ["qkv_q", "qkv_k", "qkv_v", "qk", "pv", "wo",
          "mlp_gate", "mlp_up", "mlp_down"]
CTX = {}


def norm(s):
    return " ".join((s or "").strip().split()).lower()


def matches(output, expected, category):
    out = norm(output)
    exp = norm(expected)
    if not exp:
        return False
    if out == exp or out.startswith(exp + " ") or out.startswith(exp + "."):
        return True
    if out.startswith(exp + "("):
        return True
    if category == "api_lookup" and exp in out:
        return True
    if category == "tool_routing" and out.split("(")[0].strip() == exp:
        return True
    return False


def char_sim(a, b):
    a, b = a or "", b or ""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # small Levenshtein
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def text_flags(s):
    s = s or ""
    n = max(len(s), 1)
    repl = sum(1 for c in s if c == "\ufffd") / n
    ctrl = sum(1 for c in s if unicodedata.category(c).startswith("C")
               and c not in "\n\r\t") / n
    empty = len(s.strip()) == 0
    words = re.findall(r"\S+", s.lower())
    deg = False
    if len(words) >= 8:
        tri = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if tri:
            from collections import Counter
            mx = max(Counter(tri).values())
            deg = mx / len(tri) > 0.45
    return repl, ctrl, int(empty), int(deg)


def flip_tensor(t, index, bit):
    """Return a clone with one bit flipped; t must be a float32 tensor."""
    x = t.clone().contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    ints = x.view(torch.int32)
    old = int(ints[index].item())
    old_u = old & 0xFFFFFFFF
    new_u = old_u ^ (1 << bit)
    new = new_u - 2**32 if new_u >= 2**31 else new_u
    ints[index] = new
    return x, old, new


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                     scaling, dropout=0.0, **kwargs):
    key_states = qwen2.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    ctx = CTX
    if (ctx and not ctx.get("done") and not ctx.get("sham")
            and ctx.get("stage") == "qk"
            and module.layer_idx == ctx.get("layer")):
        idx = (0, ctx["head"], ctx["pos"], ctx["key_pos"])
        attn_weights, old, new = flip_tensor(attn_weights, idx, ctx["bit"])
        ctx.update(done=True, old=old, new=new, bit_diff=1,
                   actual_idx=idx, fault_value=float(attn_weights[idx].item()))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    if (ctx and not ctx.get("done") and not ctx.get("sham")
            and ctx.get("stage") == "pv"
            and module.layer_idx == ctx.get("layer")):
        idx = (0, ctx["head"], ctx["pos"], ctx["elem"])
        attn_output, old, new = flip_tensor(attn_output, idx, ctx["bit"])
        ctx.update(done=True, old=old, new=new, bit_diff=1,
                   actual_idx=idx, fault_value=float(attn_output[idx].item()))
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# Install eagerly; model uses config._attn_implementation = "eager".
qwen2.eager_attention_forward = patched_eager_attention_forward
try:
    qwen2.ALL_ATTENTION_FUNCTIONS["eager"] = patched_eager_attention_forward
except Exception:
    pass


def get_layers(model):
    return model.model.layers


def module_for(layer, stage):
    if stage == "qkv_q":
        return layer.self_attn.q_proj
    if stage == "qkv_k":
        return layer.self_attn.k_proj
    if stage == "qkv_v":
        return layer.self_attn.v_proj
    if stage == "wo":
        return layer.self_attn.o_proj
    if stage == "mlp_gate":
        return layer.mlp.gate_proj
    if stage == "mlp_up":
        return layer.mlp.up_proj
    if stage == "mlp_down":
        return layer.mlp.down_proj
    return None


def linear_hook(module, inputs, output):
    ctx = CTX
    if not ctx or ctx.get("done") or ctx.get("sham"):
        return None
    t = output[0] if isinstance(output, tuple) else output
    idx = (0, ctx["pos"], ctx["elem"])
    new_t, old, new = flip_tensor(t, idx, ctx["bit"])
    ctx.update(done=True, old=old, new=new, bit_diff=1,
               actual_idx=idx, fault_value=float(new_t[idx].item()))
    if isinstance(output, tuple):
        return (new_t,) + tuple(output[1:])
    return new_t


def generate(model, tok, prompt_ids, max_new):
    with torch.inference_mode():
        gen = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()


def run_case(model, tok, case, max_new, stages, layers, bits, fracs,
             clean_only=False):
    messages = [
        {"role": "system", "content": case["system_context"] + "\n\n" +
         "Answer the user's question using only the document above. "
         "Return the shortest exact answer requested."},
        {"role": "user", "content": case["user_query"]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    S = int(ids.shape[1])
    clean = generate(model, tok, ids, max_new)
    clean_ok = matches(clean, case["expected_answer"], case["category"])
    rows = [{
        "model": model.config._name_or_path, "case_id": case["case_id"],
        "category": case["category"], "condition": "clean",
        "stage": "", "layer": "", "bit": "", "position": "",
        "prompt_tokens": S, "pos": "", "head": "", "key_pos": "", "elem": "",
        "old": "", "new": "", "bit_diff": "", "fault_value": "",
        "clean_output": clean, "output": clean, "expected": case["expected_answer"],
        "clean_correct": int(clean_ok), "fault_correct": int(clean_ok),
        "char_sim": 1.0, "literal_garble": 0,
        "semantic_wrong": "", "catastrophic": "", "partial": "", "preserved": "",
    }]
    if clean_only:
        return rows

    positions = [("early", int(0.1 * (S - 1))),
                 ("middle", int(0.5 * (S - 1))),
                 ("late", int(0.9 * (S - 1)))]
    if fracs is not None:
        positions = [(name, int(f * (S - 1))) for name, f in fracs]
    q_heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", 0) or
                   (model.config.hidden_size // q_heads))

    # sham: one middle position, all stages/layers
    for layer in layers:
        for stage in stages:
            pos_name, pos = "middle", int(0.5 * (S - 1))
            CTX.clear()
            CTX.update(stage=stage, layer=layer, pos=pos, bit=0,
                       sham=True, done=False, head=0, key_pos=0, elem=0)
            mod = module_for(get_layers(model)[layer], stage)
            handle = None
            if mod is not None and stage not in ("qk", "pv"):
                handle = mod.register_forward_hook(linear_hook)
            try:
                out = generate(model, tok, ids, max_new)
            finally:
                if handle:
                    handle.remove()
                CTX.clear()
            ok = matches(out, case["expected_answer"], case["category"])
            rows.append({
                "model": model.config._name_or_path, "case_id": case["case_id"],
                "category": case["category"], "condition": "sham",
                "stage": stage, "layer": layer, "bit": "", "position": pos_name,
                "prompt_tokens": S, "pos": pos, "head": "", "key_pos": "", "elem": "",
                "old": "", "new": "", "bit_diff": "", "fault_value": "",
                "clean_output": clean, "output": out, "expected": case["expected_answer"],
                "clean_correct": int(clean_ok), "fault_correct": int(ok),
                "char_sim": char_sim(clean, out), "literal_garble": 0,
                "semantic_wrong": int(not ok) if clean_ok else "",
                "catastrophic": int(not ok) if clean_ok else "",
                "partial": 0 if clean_ok else "", "preserved": int(ok) if clean_ok else "",
            })

    # fault matrix
    for layer in layers:
        for stage in stages:
            mod = module_for(get_layers(model)[layer], stage)
            if stage not in ("qk", "pv") and mod is None:
                continue
            for bit in bits:
                for pos_name, pos in positions:
                    rng = random.Random("%s|%s|%s|%s|%s" % (
                        case["case_id"], layer, stage, bit, pos_name))
                    head = rng.randrange(q_heads)
                    key_pos = rng.randrange(0, pos + 1) if stage == "qk" else 0
                    if stage == "qk":
                        elem = 0
                    elif stage == "pv":
                        elem = rng.randrange(head_dim)
                    else:
                        elem = rng.randrange(int(mod.weight.shape[0]))
                    CTX.clear()
                    CTX.update(stage=stage, layer=layer, pos=pos, bit=bit,
                               sham=False, done=False, head=head,
                               key_pos=key_pos, elem=elem)
                    handle = None
                    if mod is not None and stage not in ("qk", "pv"):
                        handle = mod.register_forward_hook(linear_hook)
                    try:
                        out = generate(model, tok, ids, max_new)
                    finally:
                        if handle:
                            handle.remove()
                        ctx = dict(CTX)
                        CTX.clear()
                    ok = matches(out, case["expected_answer"], case["category"])
                    repl, ctrl, empty, deg = text_flags(out)
                    lit = int(repl > 0 or ctrl > 0 or empty or deg)
                    sim = char_sim(clean, out)
                    sem = int((not ok) or sim < 0.4) if clean_ok else ""
                    cat = int(lit or (sem == 1)) if clean_ok else ""
                    preserved = int(not cat and ok and sim >= 0.8) if clean_ok else ""
                    partial = int(not cat and not preserved) if clean_ok else ""
                    rows.append({
                        "model": model.config._name_or_path, "case_id": case["case_id"],
                        "category": case["category"], "condition": "fault",
                        "stage": stage, "layer": layer, "bit": bit,
                        "position": pos_name, "prompt_tokens": S,
                        "pos": pos, "head": head, "key_pos": key_pos, "elem": elem,
                        "old": ctx.get("old", ""), "new": ctx.get("new", ""),
                        "bit_diff": ctx.get("bit_diff", ""),
                        "fault_value": ctx.get("fault_value", ""),
                        "clean_output": clean, "output": out,
                        "expected": case["expected_answer"],
                        "clean_correct": int(clean_ok), "fault_correct": int(ok),
                        "char_sim": sim, "literal_garble": lit,
                        "semantic_wrong": sem, "catastrophic": cat,
                        "partial": partial, "preserved": preserved,
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/agent_software_cases.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--layers", default="")
    ap.add_argument("--bits", default="28,29,30")
    ap.add_argument("--position-fracs", default="0.1,0.5,0.9")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--mem-fraction", type=float, default=0.12)
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    cases = [c for i, c in enumerate(cases) if i % args.nshards == args.shard]
    stages = [s for s in args.stages.split(",") if s]
    bits = [int(x) for x in args.bits.split(",") if x]
    fracs = None
    if args.position_fracs:
        fracs = [("p%d" % i, float(x)) for i, x in
                 enumerate(args.position_fracs.split(",")) if x]

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager")
    model.to("cuda").eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id
    layers = ([int(x) for x in args.layers.split(",") if x] or
              [0, model.config.num_hidden_layers // 2,
               model.config.num_hidden_layers - 1])
    print("model", args.model, "cases", len(cases), "layers", layers,
          "stages", stages, "bits", bits, "fracs", fracs, flush=True)

    rows = []
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        print("[%d/%d] %s" % (i, len(cases), c["case_id"]), flush=True)
        rows.extend(run_case(model, tok, c, args.max_new_tokens, stages,
                             layers, bits, fracs, args.clean_only))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out, len(rows), "rows", "elapsed", round(time.time() - t0, 1),
          "max_mem", round(torch.cuda.max_memory_allocated() / 2**20, 1), flush=True)


if __name__ == "__main__":
    main()

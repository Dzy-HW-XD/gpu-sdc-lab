#!/usr/bin/env python3
"""Clean calibration for the 12 long-context agent cases."""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(s):
    return " ".join((s or "").strip().split()).lower()


def matches(output, expected):
    out = normalize(output)
    exp = normalize(expected)
    return out == exp or out.startswith(exp + " ") or out.startswith(exp + "\n") or out.startswith(exp + ".")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/agent_software_cases.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--mem-fraction", type=float, default=0.15)
    args = ap.parse_args()
    cases = json.load(open(args.cases, encoding="utf-8"))
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
    rows = []
    for i, c in enumerate(cases, 1):
        messages = [
            {"role": "system", "content": c["system_context"] + "\n\n" + (
                "Answer the user's question using only the document above. "
                "Return the shortest exact answer requested.")},
            {"role": "user", "content": c["user_query"]},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=args.max_new_tokens, do_sample=False,
                                 num_beams=1, use_cache=True,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip()
        ok = matches(out, c["expected_answer"])
        rows.append({"case_id": c["case_id"], "category": c["category"],
                     "prompt_tokens": int(ids.shape[1]), "expected": c["expected_answer"],
                     "output": out, "clean_correct": int(ok)})
        print(c["case_id"], "tok", int(ids.shape[1]), "ok", int(ok), repr(out[:120]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("correct", sum(r["clean_correct"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()

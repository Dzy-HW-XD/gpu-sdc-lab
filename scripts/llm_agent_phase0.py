#!/usr/bin/env python3
"""Phase-0 long-context agent prefill runner.

Modes:
  clean   - prefill + decode, write the answer
  profile - prefill only, for NVBitFI kernel profiling
  inject  - prefill + decode under NVBitFI injection

The NVBitFI injector/profiler is provided by the caller through LD_PRELOAD.
"""
import argparse
import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="data/agent_phase0_case.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", choices=("clean", "profile", "inject"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--mem-fraction", type=float, default=0.20)
    args = ap.parse_args()

    case = json.load(open(args.case, encoding="utf-8"))
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda").eval()
    model.config.use_cache = True
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    messages = [
        {"role": "system", "content": case["system_context"] + "\n\n" +
         "Answer the user's question using only the document above. "
         "Return the shortest exact answer requested."},
        {"role": "user", "content": case["user_query"]},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    prompt_len = int(ids.shape[1])
    print("model", args.model, "mode", args.mode, "prompt_tokens", prompt_len, flush=True)

    result = {
        "case_id": case["case_id"],
        "category": case["category"],
        "model": args.model,
        "mode": args.mode,
        "prompt_tokens": prompt_len,
        "expected_answer": case["expected_answer"],
        "max_new_tokens": args.max_new_tokens,
    }

    t0 = time.time()
    with torch.inference_mode():
        if args.mode == "profile":
            # Profile the same generate() prefill/decode path used by inject.
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            result["prefill_logits_shape"] = list(gen.shape)
        else:
            gen = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            new_ids = gen[0, prompt_len:]
            text_out = tok.decode(new_ids, skip_special_tokens=True).strip()
            result["raw_output"] = text_out
            result["generated_tokens"] = len(new_ids)
    result["elapsed_s"] = round(time.time() - t0, 3)
    result["max_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare fixed inputs and clean checkpoints for one Qwen decoder block.

This captures the block input for a chosen layer from a clean full prefill and
saves all tensors needed to replay that block in isolation:

  hidden_in, causal_mask, position_embeddings, clean V / attn_context /
  attn_output / block_output.

No fault is injected here.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2 as qwen2

OUT_DIR = "results/qkv_v_block/inputs"
CAP = {}


def build_prompt(tok, case):
    messages = [
        {"role": "system", "content": case["system_context"] + "\n\n" +
         "Answer the user's question using only the document above. "
         "Return the shortest exact answer requested."},
        {"role": "user", "content": case["user_query"]},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
        "attn_weights": CAP.get("attn_weights"),
        "attn_output": attn_holder["out"],
        "block_output": block_out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--case", default="data/agent_phase0_case.json")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--mem-fraction", type=float, default=0.20)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    case = json.load(open(args.case, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager")
    model.to("cuda").eval()
    model.config.use_cache = False

    text = build_prompt(tok, case)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    S = int(ids.shape[1])
    position_ids = torch.arange(S, device="cuda").unsqueeze(0)
    print("prompt_tokens", S, "layer", args.layer, flush=True)

    with torch.inference_mode():
        hidden = model.model.embed_tokens(ids)
        pos_emb = model.model.rotary_emb(hidden, position_ids)
        mask = qwen2.create_causal_mask(
            config=model.config,
            inputs_embeds=hidden,
            attention_mask=torch.ones_like(ids),
            past_key_values=None,
            position_ids=position_ids,
        )
        install_observer(args.layer)
        input_hidden = None
        for i, layer in enumerate(model.model.layers):
            if i == args.layer:
                input_hidden = hidden
                outs = run_block(model, args.layer, hidden, mask, pos_emb, position_ids)
                break
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=pos_emb,
            )
        if input_hidden is None:
            raise RuntimeError("layer not reached")

        # A second clean run checks block replay determinism.
        outs2 = run_block(model, args.layer, input_hidden, mask, pos_emb, position_ids)
        for k in ("V", "attn_context", "attn_output", "block_output"):
            if not torch.equal(outs[k], outs2[k]):
                raise RuntimeError("non-deterministic checkpoint: " + k)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "layer%02d.pt" % args.layer)
    torch.save({
        "model": args.model,
        "case_id": case["case_id"],
        "layer": args.layer,
        "prompt_tokens": S,
        "hidden_in": input_hidden.detach().cpu(),
        "position_ids": position_ids.detach().cpu(),
        "position_embeddings": tuple(t.detach().cpu() for t in pos_emb),
        "causal_mask": mask.detach().cpu() if torch.is_tensor(mask) else mask,
        "clean": {k: outs[k].detach().cpu() for k in
                  ("V", "attn_context", "attn_output", "block_output")},
    }, path)
    meta = {
        "layer": args.layer,
        "path": path,
        "prompt_tokens": S,
        "hidden_shape": list(input_hidden.shape),
        "v_shape": list(outs["V"].shape),
        "attn_context_shape": list(outs["attn_context"].shape),
        "attn_output_shape": list(outs["attn_output"].shape),
        "block_output_shape": list(outs["block_output"].shape),
        "block_output_checksum": int(outs["block_output"].float().sum().item()),
    }
    json.dump(meta, open(os.path.join(args.out_dir, "layer%02d.json" % args.layer),
                         "w", encoding="utf-8"), indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()

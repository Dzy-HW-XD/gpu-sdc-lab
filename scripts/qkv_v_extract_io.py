#!/usr/bin/env python3
"""Extract layer14 qkv_v input A and W for the CUTLASS kernel variants."""
import os
import torch
from transformers import AutoModelForCausalLM

ROOT = "results/qkv_v_kernels/data"
os.makedirs(ROOT, exist_ok=True)
rec = torch.load("results/qkv_v_block/inputs/layer14.pt", map_location="cpu")
model = AutoModelForCausalLM.from_pretrained(
    rec["model"], dtype=torch.bfloat16, attn_implementation="eager")
layer = model.model.layers[int(rec["layer"])]
with torch.inference_mode():
    x = layer.input_layernorm(rec["hidden_in"].to(torch.bfloat16))
    w = layer.self_attn.v_proj.weight.detach().to(torch.bfloat16)
print("x", x.shape, x.dtype, "w", w.shape, w.dtype)
x.contiguous().view(torch.int16).numpy().tofile(os.path.join(ROOT, "A.bin"))
w.contiguous().view(torch.int16).numpy().tofile(os.path.join(ROOT, "W.bin"))
print("wrote", ROOT, os.path.getsize(os.path.join(ROOT, "A.bin")),
      os.path.getsize(os.path.join(ROOT, "W.bin")))

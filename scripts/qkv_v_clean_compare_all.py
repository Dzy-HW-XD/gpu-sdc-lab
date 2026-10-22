import json
import torch

rec = torch.load("results/qkv_v_block/inputs/layer14.pt", map_location="cpu")
k0 = rec["clean"]
paths = {"K1": "results/qkv_v_kernels/clean/k1/clean.pt",
         "K2a": "results/qkv_v_kernels/clean/k2/clean.pt",
         "K2b": "results/qkv_v_kernels/clean/k3/clean.pt",
         "K3": "results/qkv_v_kernels/clean/k4/clean.pt"}
clean = {k: torch.load(p, map_location="cpu")["clean"] for k, p in paths.items()}
out = {}
for label, c in clean.items():
    out[label] = {}
    for cp in ("V", "attn_output", "block_output", "final_output"):
        x = c[cp].float()
        y = k0[cp].float()
        # K1 is the controlled reference; compare every variant to K1.
        ref = clean["K1"][cp].float()
        d = x - ref
        out[label][cp] = {
            "vs_K0_equal": bool(torch.equal(x, y)),
            "vs_K0_mismatch": int((x != y).sum()),
            "vs_K0_max_abs": float((x - y).abs().max()),
            "vs_K0_rel_l2": float((x - y).norm() / y.norm().clamp_min(1e-30)),
            "vs_K1_equal": bool(torch.equal(x, ref)),
            "vs_K1_mismatch": int((x != ref).sum()),
            "vs_K1_max_abs": float(d.abs().max()),
            "vs_K1_rel_l2": float(d.norm() / ref.norm().clamp_min(1e-30)),
        }
json.dump(out, open("results/qkv_v_kernels/clean_compare.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(out, indent=2))

import json
import torch

base = "results/qkv_v_kernels"
p = base + "/clean_compare.json"
d = json.load(open(p, encoding="utf-8"))
rec = torch.load("results/qkv_v_block/inputs/layer14.pt", map_location="cpu")["clean"]
ref = torch.load(base + "/clean/k1/clean.pt", map_location="cpu")["clean"]
d["K0"] = {}
for cp in ("V", "attn_output", "block_output", "final_output"):
    x = rec[cp].float()
    r = ref[cp].float()
    diff = x - r
    d["K0"][cp] = {
        "vs_K1_equal": bool(torch.equal(x, r)),
        "vs_K1_mismatch": int((x != r).sum()),
        "vs_K1_max_abs": float(diff.abs().max()),
        "vs_K1_rel_l2": float(diff.norm() / r.norm().clamp_min(1e-30)),
    }
json.dump(d, open(p, "w", encoding="utf-8"), indent=2)
print(json.dumps(d["K0"], indent=2))

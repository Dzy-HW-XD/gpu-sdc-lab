import glob
import json
import os
import torch

base = "results/qkv_v_kernels"
out = {}
for binary, label in (("k1", "K1"), ("k2", "K2a"), ("k3", "K2b"), ("k4", "K3")):
    ref = torch.load(os.path.join(base, "clean", binary, "clean.pt"), map_location="cpu")["clean"]
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "clean_controls", binary, "clean_*.pt"))):
        c = torch.load(p, map_location="cpu")["clean"]
        eq = {k: bool(torch.equal(c[k], ref[k])) for k in ("V", "attn_output", "block_output", "final_output")}
        rows.append({"file": os.path.basename(p), **eq})
    out[label] = {"n": len(rows), "all_equal": all(all(r[k] for k in ("V", "attn_output", "block_output", "final_output")) for r in rows), "rows": rows}
json.dump(out, open(os.path.join(base, "clean_controls", "compare.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps({k: (v["n"], v["all_equal"]) for k, v in out.items()}, indent=2))

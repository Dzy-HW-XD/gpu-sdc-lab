import sys
sys.path.insert(0, ".")
from core.injection.nvbitfi import parse_profile_line

for l in ["00", "14", "27"]:
    p = f"results/qkv_v_block/profile/layer{l}_profile.txt"
    rows = [parse_profile_line(x) for x in open(p, encoding="utf-8", errors="ignore")]
    rows = [r for r in rows if r and r["groups"].get("mma", 0) > 0]
    print("layer", l, "mma rows", len(rows))
    for r in rows[:3]:
        print(" ", r["index"], r["name"], "ctas", r["ctas"], "instrs", r["instrs"], "mma", r["groups"]["mma"])

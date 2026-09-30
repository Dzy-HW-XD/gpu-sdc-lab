#!/usr/bin/env python3
"""Aggregate gpu-sdc-lab results into a markdown summary (stdlib only).

Usage: python3 scripts/analyze.py [results_dir] [out.md]
"""

import json
import os
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESULTS, "SDCLAB_SUMMARY.md")

EXPERIMENTS = [
    "001_gemm_baseline", "002_fault_position", "003_fault_type",
    "004_fault_bit", "005_gemm_size", "006_data_pattern",
    "007_reduce_atomic",
]

BIT_REGION = {}
for _b in range(32):
    if _b == 31:
        BIT_REGION[_b] = "sign"
    elif _b >= 23:
        BIT_REGION[_b] = "exponent"
    else:
        BIT_REGION[_b] = "mantissa"


def load(eid):
    p = os.path.join(RESULTS, eid, "observations.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def counts(records):
    c = {"MASKED": 0, "SDC": 0, "CRASH": 0, "NOT_INJECTED": 0}
    for r in records:
        c[r.get("classification", "NOT_INJECTED")] = c.get(r.get("classification"), 0) + 1
    c["injected"] = c["MASKED"] + c["SDC"] + c["CRASH"]
    c["sdc_rate"] = c["SDC"] / c["injected"] if c["injected"] else 0.0
    return c


def main():
    out = []
    out.append("# gpu-sdc-lab — Phase 1 summary\n")
    for eid in EXPERIMENTS:
        recs = load(eid)
        if not recs:
            continue
        if eid == "001_gemm_baseline":
            ident = sum(1 for r in recs if r["observation"]["bitwise_equal"])
            out.append("## %s (baseline)\n" % eid)
            out.append("- runs: %d, bitwise-identical: %d -> %s\n"
                       % (len(recs), ident,
                          "DETERMINISTIC" if ident == len(recs) else "NON-DETERMINISTIC"))
            continue

        c = counts(recs)
        out.append("## %s\n" % eid)
        out.append("- injections: %d | MASKED %d | SDC %d | CRASH %d "
                   "| NOT_INJECTED %d | SDC rate %.3f\n"
                   % (len(recs), c["MASKED"], c["SDC"], c["CRASH"],
                      c["NOT_INJECTED"], c["sdc_rate"]))

        # per-group breakdown
        bygroup = {}
        for r in recs:
            g = r["fault"]["group"]
            bygroup.setdefault(g, [0, 0, 0])
            cl = r["classification"]
            if cl == "MASKED":
                bygroup[g][0] += 1
            elif cl == "SDC":
                bygroup[g][1] += 1
            elif cl == "CRASH":
                bygroup[g][2] += 1
        out.append("- by group: " + ", ".join(
            "%s(m%d/s%d/c%d)" % (g, v[0], v[1], v[2]) for g, v in sorted(bygroup.items())) + "\n")

        if eid == "004_fault_bit":
            reg = {}
            for r in recs:
                b = r["fault"].get("bit_index")
                if b is None:
                    continue
                reg.setdefault(BIT_REGION[b], [0, 0, 0])
                cl = r["classification"]
                if cl == "MASKED":
                    reg[BIT_REGION[b]][0] += 1
                elif cl == "SDC":
                    reg[BIT_REGION[b]][1] += 1
                else:
                    reg[BIT_REGION[b]][2] += 1
            out.append("- by FP32 region: " + ", ".join(
                "%s(m%d/s%d/c%d)" % (k, v[0], v[1], v[2]) for k, v in sorted(reg.items())) + "\n")

        # error magnitude for SDC
        sdc = [r for r in recs if r["classification"] == "SDC"]
        if sdc:
            mx = max(r["observation"]["max_abs_error"] for r in sdc)
            rl = max(r["observation"]["relative_l2_error"] for r in sdc)
            ce = max(r["observation"]["corrupted_elements"] for r in sdc)
            out.append("- SDC max abs err %.4g, max rel-L2 %.4g, max corrupted elems %d\n"
                       % (mx, rl, ce))
        out.append("\n")

    text = "\n".join(out)
    print(text)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n[written] " + OUT)


if __name__ == "__main__":
    main()

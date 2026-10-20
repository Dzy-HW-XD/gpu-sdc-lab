#!/usr/bin/env python3
"""Per-head Frobenius error of O for every MASKED/SDC run (fast, stdlib).

Reads only O (plus the golden metadata), not every intermediate tensor, so it
scales to thousands of runs. Writes <exp>/attention_head_errors.jsonl, consumed
by attention_head_crosstalk.py.

Usage: attention_head_errors.py --exp results/046_attn_mha_bit_sweep
"""
import argparse
import array
import json
import math
import os
import sys


def read_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args(argv)

    obs_path = os.path.join(args.exp, "observations.jsonl")
    out_path = os.path.join(args.exp, "attention_head_errors.jsonl")
    meta_cache = {}
    n = 0
    with open(obs_path, encoding="utf-8") as f, \
            open(out_path, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cls = rec.get("classification")
            if cls not in ("MASKED", "SDC"):
                continue
            g = rec.get("golden_bin")
            fo_bin = rec.get("raw_out_bin")
            if not (g and fo_bin and g.endswith(".Y.bin")
                    and fo_bin.endswith(".Y.bin")):
                continue
            gp = g[:-len(".Y.bin")]
            fp = fo_bin[:-len(".Y.bin")]
            gO = gp + ".O.bin"
            fO = fp + ".O.bin"
            if not (os.path.exists(gO) and os.path.exists(fO)):
                continue
            meta = meta_cache.get(gp)
            if meta is None:
                with open(gp + ".json", encoding="utf-8") as mf:
                    meta = json.load(mf)
                meta_cache[gp] = meta
            H = int(meta["H"])
            S = int(meta["S"])
            dh = int(meta.get("dh", int(meta["D"]) // H))
            slab = S * dh
            gold = read_f32(gO)
            fault = read_f32(fO)
            if len(gold) != H * slab or len(fault) != H * slab:
                continue
            fro = []
            for h in range(H):
                base = h * slab
                ss = 0.0
                for i in range(base, base + slab):
                    d = fault[i] - gold[i]
                    ss += d * d
                fro.append(math.sqrt(ss))
            labels = rec.get("labels") or {}
            fo.write(json.dumps({
                "seq": rec.get("seq"),
                "stage": labels.get("stage"),
                "head": labels.get("head"),
                "position": labels.get("position", "all"),
                "bit": labels.get("bit"),
                "classification": cls,
                "o_head_fro": fro,
            }, sort_keys=True) + "\n")
            n += 1
    print("wrote %s (%d runs)" % (out_path, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())

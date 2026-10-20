#!/usr/bin/env python3
"""Merge sharded campaign observations into one experiment directory.

Each shard writes results/<eid>__gpuN/observations.jsonl with seq=1..N. This
merges them with globally unique seq (shard*100000 + seq), so per-run metric
files (seq_<seq>.json) cannot collide, and rewrites experiment_id to <eid>.

Usage: attention_merge_shards.py --eid 046_attn_mha_bit_sweep
"""
import argparse
import glob
import json
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.results, args.eid + "__gpu*")))
    outdir = os.path.join(args.results, args.eid)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "observations.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for d in shards:
            m = re.search(r"gpu(\d+)$", os.path.basename(d))
            sid = int(m.group(1)) if m else 0
            p = os.path.join(d, "observations.jsonl")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    r["seq"] = sid * 100000 + int(r.get("seq", 0))
                    r["experiment_id"] = args.eid
                    fo.write(json.dumps(r, sort_keys=True) + "\n")
                    n += 1
    print("merged %d records from %d shards -> %s" % (n, len(shards), out))


if __name__ == "__main__":
    main()

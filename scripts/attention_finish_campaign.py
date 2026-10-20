#!/usr/bin/env python3
"""Wait for a sharded attention campaign, then merge + collect + select.

Run this in tmux on the GPU node after launching the shards:
  python3 scripts/attention_finish_campaign.py --eid 046_attn_mha_bit_sweep

It polls until no `campaign.py --gpu` process remains, then:
  1. attention_merge_shards.py   (unique seq across shards)
  2. attention_collect.py        (L0 + L0..L5 tables, per-run metrics)
  3. attention_select_runs.py    (propagation_manifest.csv)
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
LAB_ROOT = os.path.dirname(HERE)


def campaign_running():
    out = subprocess.run(["pgrep", "-af", "campaign.py --gpu"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True).stdout
    return any("campaign.py --gpu" in line for line in out.splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", required=True)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=10800)
    args = ap.parse_args()
    os.chdir(LAB_ROOT)

    t0 = time.time()
    while campaign_running():
        if time.time() - t0 > args.timeout:
            raise SystemExit("timeout waiting for campaign")
        print("waiting for shards ...", flush=True)
        time.sleep(args.poll)
    print("campaign finished, merging", flush=True)

    exp = os.path.join("results", args.eid)
    steps = [
        [sys.executable, os.path.join(HERE, "attention_merge_shards.py"),
         "--eid", args.eid],
        [sys.executable, os.path.join(HERE, "attention_levels.py"), "--exp", exp],
        [sys.executable, os.path.join(HERE, "attention_select_runs.py"),
         "--exp", exp],
        [sys.executable, os.path.join(HERE, "attention_head_errors.py"),
         "--exp", exp],
    ]
    for cmd in steps:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

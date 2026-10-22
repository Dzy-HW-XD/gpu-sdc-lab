#!/usr/bin/env python3
"""Generate and run random HMMA-site injections for the qkv_v block.

Site sampling:
  - 3 layers x 3 bits x 45 random dynamic HMMA sites = 405 primary sites
  - 5 extra random sites per layer/bit as reserve

Each run is a separate process with NVBitFI injector.so. The child runs one
isolated Qwen decoder block and writes checkpoint comparisons to out.json.
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time

ROOT = "/home/dzy/gpu-sdc-lab"
INJECTOR = ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/injector/injector.so"
CUDA_BIN = "/usr/local/cuda/bin"
CUDA_LIB = "/usr/local/cuda/lib64"
RUNNER = ROOT + "/scripts/qkv_v_block_run.py"
INPUT_DIR = ROOT + "/results/qkv_v_block/inputs"
PROFILE_DIR = ROOT + "/results/qkv_v_block/profile"
OUT_DIR = ROOT + "/results/qkv_v_block/injections"
MANIFEST = ROOT + "/results/qkv_v_block/sites.csv"
LAYERS = [0, 14, 27]
BITS = [28, 29, 30]
SITES_PER_CELL = 45
RESERVE_PER_CELL = 5
SEED = 20261022


def profile_mma(layer):
    p = os.path.join(PROFILE_DIR, "layer%02d" % layer, "nvbitfi-injection-log-temp.txt")
    text = open(p, encoding="utf-8", errors="ignore").read()
    # qkv_v is the third HMMA kernel in each block.
    rows = []
    for line in text.splitlines():
        if not line.startswith("NVBit-igprofile"):
            continue
        m = re.search(r"index:\s*(\d+); kernel_name:\s*(.*?);", line)
        g = re.search(r"mma:\s*(\d+)", line)
        if not m or not g:
            continue
        mma = int(g.group(1))
        if mma <= 0:
            continue
        rows.append((int(m.group(1)), m.group(2).strip(), mma))
    if len(rows) < 3:
        raise RuntimeError("not enough MMA kernels in %s" % p)
    return rows[2]


def generate():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    sid = 0
    for layer in LAYERS:
        idx, name, mma = profile_mma(layer)
        for bit in BITS:
            for rep in range(SITES_PER_CELL + RESERVE_PER_CELL):
                inst_id = rng.randrange(mma)
                op_seed = rng.random()
                bit_seed = (bit + 0.5) / 32.0
                rows.append({
                    "site_id": "L%02d_b%d_%03d" % (layer, bit, rep),
                    "layer": layer,
                    "bit": bit,
                    "inst_id": inst_id,
                    "op_seed": "%.10f" % op_seed,
                    "bit_seed": "%.10f" % bit_seed,
                    "kernel_name": name,
                    "kernel_index": idx,
                    "mma_count": mma,
                    "reserve": int(rep >= SITES_PER_CELL),
                })
                sid += 1
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", MANIFEST, "rows", len(rows), "primary", sum(1 for r in rows if not r["reserve"]))


def parse_log(path):
    text = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
    rec = {"injected": "Injection data" in text,
           "error_not_injected": "Error not injected" in text,
           "kernel_error": "ERROR FAIL in kernel execution" in text,
           "mask": None, "before": None, "after": None,
           "reg": None, "opcode": None, "pc": None, "tid": None}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("mask:"):
            rec["mask"] = s.split(":", 1)[1].strip()
        elif s.startswith("beforeVal:"):
            try:
                b, a = s.split(";")
                rec["before"] = b.split(":", 1)[1].strip()
                rec["after"] = a.split(":", 1)[1].strip()
            except Exception:
                pass
        elif s.startswith("regNo:"):
            rec["reg"] = s.split(":", 1)[1].strip()
        elif s.startswith("opcode:"):
            rec["opcode"] = s.split(":", 1)[1].strip()
        elif s.startswith("pcOffset:"):
            rec["pc"] = s.split(":", 1)[1].strip()
        elif s.startswith("tid:"):
            rec["tid"] = s.split(":", 1)[1].strip()
    return rec


def run_one(site, gpu, resume=True):
    layer = int(site["layer"])
    run_dir = os.path.join(OUT_DIR, site["site_id"])
    os.makedirs(run_dir, exist_ok=True)
    out_json = os.path.join(run_dir, "out.json")
    info = os.path.join(run_dir, "nvbitfi-injection-info.txt")
    log = os.path.join(run_dir, "nvbitfi-injection-log-temp.txt")
    runlog = os.path.join(run_dir, "run.log")
    with open(info, "w", encoding="utf-8") as f:
        f.write("7\n0\n%s\n%d\n%d\n%s\n%s\n" % (
            site["kernel_name"], int(site["kernel_index"]), int(site["inst_id"]),
            site["op_seed"], site["bit_seed"]))
    if not (resume and os.path.exists(out_json) and os.path.exists(log)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["LD_PRELOAD"] = INJECTOR
        env["NOBANNER"] = "1"
        env["TOOL_VERBOSE"] = "0"
        env["PATH"] = CUDA_BIN + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = CUDA_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [sys.executable, RUNNER,
               "--input", os.path.join(INPUT_DIR, "layer%02d.pt" % layer),
               "--out", out_json, "--mode", "inject"]
        with open(runlog, "w", encoding="utf-8", errors="replace") as lf:
            p = subprocess.run(cmd, cwd=run_dir, env=env, stdout=lf,
                               stderr=subprocess.STDOUT, timeout=300)
        exit_code = p.returncode
    else:
        exit_code = 0

    meta = parse_log(log)
    rec = dict(site)
    rec.update({
        "run_dir": run_dir,
        "exit_code": exit_code,
        "injected": int(meta["injected"]),
        "error_not_injected": int(meta["error_not_injected"]),
        "kernel_error": int(meta["kernel_error"]),
        "mask": meta["mask"] or "",
        "before": meta["before"] or "",
        "after": meta["after"] or "",
        "reg": meta["reg"] or "",
        "opcode": meta["opcode"] or "",
        "pc": meta["pc"] or "",
        "tid": meta["tid"] or "",
        "runtime_error": "",
        "nan_inf_any": "",
        "v_equal": "",
        "attn_equal": "",
        "block_equal": "",
        "mismatch_v": "",
        "mismatch_attn": "",
        "mismatch_block": "",
        "max_abs_v": "",
        "max_abs_attn": "",
        "max_abs_block": "",
        "rel_l2_v": "",
        "rel_l2_attn": "",
        "rel_l2_block": "",
        "outcome_v": "",
        "outcome_attn": "",
        "outcome_block": "",
    })
    if exit_code != 0 or not os.path.exists(out_json):
        rec["runtime_error"] = "process_failed"
        rec["outcome_v"] = rec["outcome_attn"] = rec["outcome_block"] = "DUE"
        return rec
    try:
        out = json.load(open(out_json, encoding="utf-8"))
    except Exception as e:
        rec["runtime_error"] = "bad_json:%s" % e
        rec["outcome_v"] = rec["outcome_attn"] = rec["outcome_block"] = "DUE"
        return rec
    rec["runtime_error"] = out.get("runtime_error") or ""
    rec["nan_inf_any"] = int(out.get("nan_inf_any", False))
    cps = out.get("checkpoints", {})
    for short, key in (("v", "V"), ("attn", "attn_output"), ("block", "block_output")):
        c = cps.get(key, {})
        rec[short + "_equal"] = int(c.get("bitwise_equal", False))
        rec["mismatch_" + short] = c.get("mismatch_count", "")
        rec["max_abs_" + short] = c.get("max_abs", "")
        rec["rel_l2_" + short] = c.get("rel_l2", "")
        rec["outcome_" + short] = "Mask" if c.get("bitwise_equal", False) else "SDC"
    if rec["runtime_error"]:
        rec["outcome_v"] = rec["outcome_attn"] = rec["outcome_block"] = "DUE"
    if not meta["injected"]:
        rec["outcome_v"] = rec["outcome_attn"] = rec["outcome_block"] = "INVALID"
    return rec


def run(shard, nshards, limit, resume):
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rows = [r for r in rows if int(r["reserve"]) == 0]
    if limit and limit < len(rows):
        step = max(1, (len(rows) + limit - 1) // limit)
        rows = rows[::step][:limit]
    rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ROOT + "/results/qkv_v_block/logs", exist_ok=True)
    out_csv = ROOT + "/results/qkv_v_block/runs_s%d.csv" % shard
    fields = None
    done = 0
    t0 = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = None
        for site in rows:
            rec = run_one(site, gpu=shard, resume=resume)
            if w is None:
                fields = list(rec.keys())
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
            w.writerow(rec)
            f.flush()
            done += 1
            print("[s%d] %d/%d %s exit=%s inj=%s V=%s attn=%s block=%s" % (
                shard, done, len(rows), site["site_id"], rec["exit_code"],
                rec["injected"], rec["outcome_v"], rec["outcome_attn"],
                rec["outcome_block"]), flush=True)
    print("shard", shard, "done", done, "elapsed", round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.generate:
        generate()
    else:
        run(args.shard, args.nshards, args.limit, args.resume)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Layer-14 qkv_v full-stack campaign (C0..C3), 60 random sites per bit."""
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
RUNNER = ROOT + "/scripts/qkv_v_fullstack_run.py"
INPUT = ROOT + "/results/qkv_v_block/inputs/layer14.pt"
OUT_DIR = ROOT + "/results/qkv_v_full/injections"
MANIFEST = ROOT + "/results/qkv_v_full/sites.csv"
KERNEL_NAME = "ampere_bf16_s16816gemm_bf16_128x64_ldg8_relu_f2f_stages_64x4_tn"
KERNEL_INDEX = 357
MMA_COUNT = 14155776
LAYER = 14
BITS = [28, 29, 30]
SITES_PER_BIT = 60
RESERVE_PER_BIT = 6
SEED = 20261023
CPS = [("C0", "V"), ("C1", "attn_output"), ("C2", "block_output"), ("C3", "final_output")]


def generate():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    rng = random.Random(SEED)
    rows = []
    for bit in BITS:
        for rep in range(SITES_PER_BIT + RESERVE_PER_BIT):
            rows.append({
                "site_id": "L14_b%d_%03d" % (bit, rep),
                "layer": LAYER,
                "bit": bit,
                "inst_id": rng.randrange(MMA_COUNT),
                "op_seed": "%.10f" % rng.random(),
                "bit_seed": "%.10f" % ((bit + 0.5) / 32.0),
                "kernel_name": KERNEL_NAME,
                "kernel_index": KERNEL_INDEX,
                "mma_count": MMA_COUNT,
                "reserve": int(rep >= SITES_PER_BIT),
            })
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", MANIFEST, "rows", len(rows), "primary", sum(1 for r in rows if not r["reserve"]))


def parse_log(path):
    text = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
    rec = {"injected": "Injection data" in text,
           "error_not_injected": "Error not injected" in text,
           "kernel_error": "ERROR FAIL in kernel execution" in text,
           "mask": "", "before": "", "after": "", "reg": "", "opcode": "", "pc": "", "tid": ""}
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


def add_cp(rec, short, c):
    eq = bool(c.get("bitwise_equal", False))
    finite = bool(c.get("finite", True))
    rec[short + "_equal"] = int(eq)
    rec[short + "_mismatch"] = c.get("mismatch_count", "")
    rec[short + "_max_abs"] = c.get("max_abs", "")
    rec[short + "_mean_abs"] = c.get("mean_abs", "")
    rec[short + "_rel_l2"] = c.get("rel_l2", "")
    rec[short + "_rel_max"] = c.get("rel_max", "")
    rec[short + "_nan_inf"] = int(not finite)
    if rec.get("_crash"):
        rec["outcome_d0_" + short] = "Crash"
        rec["outcome_d1_" + short] = "Crash"
    elif not rec.get("injected"):
        rec["outcome_d0_" + short] = "Invalid"
        rec["outcome_d1_" + short] = "Invalid"
    else:
        rec["outcome_d0_" + short] = "Masked" if eq else "SDC"
        rec["outcome_d1_" + short] = ("DUE" if not finite else
                                      ("Masked" if eq else "SDC"))


def run_one(site, gpu, resume=True):
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
        cmd = [sys.executable, RUNNER, "--input", INPUT,
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
        "run_dir": run_dir, "exit_code": exit_code,
        "injected": int(meta["injected"]),
        "error_not_injected": int(meta["error_not_injected"]),
        "kernel_error": int(meta["kernel_error"]),
        "mask": meta["mask"], "before": meta["before"], "after": meta["after"],
        "reg": meta["reg"], "opcode": meta["opcode"], "pc": meta["pc"],
        "tid": meta["tid"], "runtime_error": "", "nan_inf_any": "",
        "_crash": exit_code != 0 or not os.path.exists(out_json),
    })
    if rec["_crash"]:
        rec["runtime_error"] = "process_failed"
        for short, _ in CPS:
            add_cp(rec, short, {})
        rec["run_d0"] = "Crash"
        rec["run_d1"] = "Crash"
        return rec
    try:
        out = json.load(open(out_json, encoding="utf-8"))
    except Exception as e:
        rec["_crash"] = True
        rec["runtime_error"] = "bad_json:%s" % e
        for short, _ in CPS:
            add_cp(rec, short, {})
        rec["run_d0"] = "Crash"
        rec["run_d1"] = "Crash"
        return rec
    rec["runtime_error"] = out.get("runtime_error") or ""
    rec["nan_inf_any"] = int(out.get("nan_inf_any", False))
    cps = out.get("checkpoints", {})
    for short, key in CPS:
        add_cp(rec, short, cps.get(key, {}))
    rec["run_d0"] = rec["outcome_d0_C3"]
    rec["run_d1"] = ("DUE" if rec["nan_inf_any"] else rec["outcome_d1_C3"])
    if not meta["injected"]:
        rec["run_d0"] = rec["run_d1"] = "Invalid"
    return rec


def run(shard, nshards, limit, resume):
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rows = [r for r in rows if int(r["reserve"]) == 0]
    if limit and limit < len(rows):
        step = max(1, (len(rows) + limit - 1) // limit)
        rows = rows[::step][:limit]
    rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ROOT + "/results/qkv_v_full/logs", exist_ok=True)
    out_csv = ROOT + "/results/qkv_v_full/runs_s%d.csv" % shard
    t0 = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = None
        for n, site in enumerate(rows, 1):
            rec = run_one(site, gpu=shard, resume=resume)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rec.keys()))
                w.writeheader()
            w.writerow(rec)
            f.flush()
            print("[s%d] %d/%d %s %s %s %s %s %s" % (
                shard, n, len(rows), site["site_id"], rec["run_d0"],
                rec["outcome_d0_C0"], rec["outcome_d0_C1"],
                rec["outcome_d0_C2"], rec["outcome_d0_C3"]), flush=True)
    print("shard", shard, "done", len(rows), "elapsed", round(time.time() - t0, 1))


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

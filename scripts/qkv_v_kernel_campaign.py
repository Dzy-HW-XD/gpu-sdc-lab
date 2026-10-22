#!/usr/bin/env python3
"""Full qkv_v kernel comparison campaign: K1/K2a/K2b/K3, 60 sites/bit."""
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
INJECTOR = ROOT + "/third_party/nvbit_release_x86_64/tools/nvbitfi/injector_long/injector.so"
CUDA_BIN = "/usr/local/cuda/bin"
CUDA_LIB = "/usr/local/cuda/lib64"
DATA = ROOT + "/results/qkv_v_kernels/data"
BASE = ROOT + "/results/qkv_v_kernels"
MANIFEST = BASE + "/sites.csv"
RUN_ROOT = BASE + "/runs"
VARIANTS = [("K1", "k1"), ("K2a", "k2"), ("K2b", "k3"), ("K3", "k4")]
BITS = [28, 29, 30]
SITES_PER_BIT = 60
RESERVE_PER_BIT = 6
SEED = 20261024
CPS = [("C0", "V"), ("C1", "attn_output"), ("C2", "block_output"), ("C3", "final_output")]


def kernel_info(binary):
    p = os.path.join(BASE, "profile", binary, "kernel.json")
    d = json.load(open(p, encoding="utf-8"))
    return d["index"], d["name"], d["mma"]


def generate():
    rng = random.Random(SEED)
    rows = []
    for label, binary in VARIANTS:
        idx, name, mma = kernel_info(binary)
        for bit in BITS:
            for rep in range(SITES_PER_BIT + RESERVE_PER_BIT):
                rows.append({
                    "site_id": "%s_b%d_%03d" % (label, bit, rep),
                    "variant": label, "binary": binary, "layer": 14, "bit": bit,
                    "inst_id": rng.randrange(mma),
                    "op_seed": "%.10f" % rng.random(),
                    "bit_seed": "%.10f" % ((bit + 0.5) / 32.0),
                    "kernel_index": idx, "kernel_name": name, "mma": mma,
                    "reserve": int(rep >= SITES_PER_BIT),
                    "clean": os.path.join(BASE, "clean", binary, "clean.pt"),
                })
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", MANIFEST, len(rows), "primary", sum(1 for r in rows if not r["reserve"]))


def parse_log(path):
    text = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
    rec = {"injected": "Injection data" in text,
           "error_not_injected": "Error not injected" in text,
           "kernel_error": "ERROR FAIL" in text,
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


def add_cp(rec, short, c, crash=False, invalid=False):
    eq = bool(c.get("bitwise_equal", False))
    finite = bool(c.get("finite", True))
    rec[short + "_equal"] = int(eq)
    rec[short + "_mismatch"] = c.get("mismatch_count", "")
    rec[short + "_max_abs"] = c.get("max_abs", "")
    rec[short + "_mean_abs"] = c.get("mean_abs", "")
    rec[short + "_rel_l2"] = c.get("rel_l2", "")
    rec[short + "_rel_max"] = c.get("rel_max", "")
    rec[short + "_nan_inf"] = int(not finite)
    if crash:
        rec["outcome_d0_" + short] = "Crash"
        rec["outcome_d1_" + short] = "Crash"
    elif invalid:
        rec["outcome_d0_" + short] = "Invalid"
        rec["outcome_d1_" + short] = "Invalid"
    else:
        rec["outcome_d0_" + short] = "Masked" if eq else "SDC"
        rec["outcome_d1_" + short] = "DUE" if not finite else ("Masked" if eq else "SDC")


def run_one(site, gpu, resume=True, injector=None, run_root=None):
    run_dir = os.path.join(run_root or RUN_ROOT, site["variant"], site["site_id"])
    os.makedirs(run_dir, exist_ok=True)
    v_path = os.path.join(run_dir, "V.bin")
    info = os.path.join(run_dir, "nvbitfi-injection-info.txt")
    injlog = os.path.join(run_dir, "nvbitfi-injection-log-temp.txt")
    fault_json = os.path.join(run_dir, "fault.json")
    with open(info, "w", encoding="utf-8") as f:
        f.write("7\n0\n%s\n%d\n%d\n%s\n%s\n" % (
            site["kernel_name"], int(site["kernel_index"]), int(site["inst_id"]),
            site["op_seed"], site["bit_seed"]))
    if not (resume and os.path.exists(fault_json) and os.path.exists(injlog)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["LD_PRELOAD"] = injector or INJECTOR
        env["NOBANNER"] = "1"
        env["TOOL_VERBOSE"] = "0"
        env["PATH"] = CUDA_BIN + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = CUDA_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        cmd = [os.path.join(BASE, "bin", site["binary"]),
               "--M", "2264", "--N", "256", "--K", "1536",
               "--inputA", os.path.join(DATA, "A.bin"),
               "--inputB", os.path.join(DATA, "W.bin"),
               "--out", v_path]
        with open(os.path.join(run_dir, "kernel.log"), "w", encoding="utf-8", errors="replace") as lf:
            pk = subprocess.run(cmd, cwd=run_dir, env=env, stdout=lf,
                                stderr=subprocess.STDOUT, timeout=300)
        if pk.returncode != 0 or not os.path.exists(v_path):
            return crash_rec(site, run_dir, pk.returncode, "kernel_failed")
        env2 = os.environ.copy()
        env2.pop("LD_PRELOAD", None)
        env2["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd2 = [sys.executable, ROOT + "/scripts/qkv_v_from_v_run.py",
                "--v", v_path, "--out", fault_json, "--mode", "fault",
                "--clean", site["clean"]]
        with open(os.path.join(run_dir, "downstream.log"), "w", encoding="utf-8", errors="replace") as lf:
            pd = subprocess.run(cmd2, env=env2, stdout=lf, stderr=subprocess.STDOUT, timeout=300)
        if pd.returncode != 0 or not os.path.exists(fault_json):
            return crash_rec(site, run_dir, pd.returncode, "downstream_failed")
    meta = parse_log(injlog)
    rec = dict(site)
    rec.update({"run_dir": run_dir, "injected": int(meta["injected"]),
                "error_not_injected": int(meta["error_not_injected"]),
                "kernel_error": int(meta["kernel_error"]),
                "mask": meta["mask"], "before": meta["before"], "after": meta["after"],
                "reg": meta["reg"], "opcode": meta["opcode"], "pc": meta["pc"],
                "tid": meta["tid"], "runtime_error": "", "nan_inf_any": ""})
    if not meta["injected"]:
        for short, _ in CPS:
            add_cp(rec, short, {}, invalid=True)
        rec["run_d0"] = rec["run_d1"] = "Invalid"
        return rec
    out = json.load(open(fault_json, encoding="utf-8"))
    cps = out.get("checkpoints", {})
    rec["nan_inf_any"] = int(out.get("nan_inf_any", False))
    for short, key in CPS:
        add_cp(rec, short, cps.get(key, {}))
    rec["run_d0"] = rec["outcome_d0_C3"]
    rec["run_d1"] = "DUE" if rec["nan_inf_any"] else rec["outcome_d1_C3"]
    return rec


def crash_rec(site, run_dir, code, why):
    rec = dict(site)
    rec.update({"run_dir": run_dir, "injected": 0, "error_not_injected": 0,
                "kernel_error": 0, "mask": "", "before": "", "after": "",
                "reg": "", "opcode": "", "pc": "", "tid": "",
                "runtime_error": why + ":%d" % code, "nan_inf_any": ""})
    for short, _ in CPS:
        add_cp(rec, short, {}, crash=True)
    rec["run_d0"] = rec["run_d1"] = "Crash"
    return rec


def run(shard, nshards, limit, resume):
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    rows = [r for r in rows if int(r["reserve"]) == 0]
    if limit and limit < len(rows):
        step = max(1, (len(rows) + limit - 1) // limit)
        rows = rows[::step][:limit]
    rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    os.makedirs(BASE, exist_ok=True)
    out_csv = os.path.join(BASE, "runs_s%d.csv" % shard)
    t0 = time.time()
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = None
        for n, site in enumerate(rows, 1):
            rec = run_one(site, gpu=shard, resume=resume)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rec.keys()))
                w.writeheader()
            w.writerow(rec); f.flush()
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

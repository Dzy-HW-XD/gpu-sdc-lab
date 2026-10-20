#!/usr/bin/env python3
"""Locate a cuBLAS(Lt) kernel's cubin by scanning the sm_80 ELF images.

Newer libcublasLt is too large for a single cuobjdump pass, so we list the ELF
images, extract them one at a time, and grep each cubin for the wanted symbol
substring. Prints the cubin path (copied under --out) on success.

Usage:
  attention_find_cubin.py [--lib LIB] [--want SUBSTR] [--out DIR]
"""
import argparse
import os
import shutil
import subprocess
import tempfile

CU13_3 = "/usr/local/cuda-13.3/bin/cuobjdump"
CU13_0 = "/usr/local/cuda-13.0/bin/cuobjdump"


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="/usr/local/cuda-13.3/lib64/libcublasLt.so.13")
    ap.add_argument("--want", default="128x1_nn_align8")
    ap.add_argument("--out", default="results/_attn_sass")
    ap.add_argument("--arch", default="sm_80")
    ap.add_argument("--near", default=None,
                    help="only scan a window around the ELF name containing this")
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lelf = run([CU13_3, "-lelf", args.lib]).stdout.decode("utf-8", "replace")
    names = []
    for ln in lelf.splitlines():
        if args.arch + ".cubin" in ln:
            names.append(ln.split(":")[-1].strip())
    print("ELF images (%s): %d" % (args.arch, len(names)), flush=True)
    if args.near:
        idx = next((i for i, n in enumerate(names) if args.near in n), None)
        if idx is None:
            print("near name not found:", args.near)
            return 1
        lo = max(0, idx - args.window)
        hi = min(len(names), idx + args.window + 1)
        print("scanning %d..%d around %s" % (lo, hi - 1, names[idx]), flush=True)
        names = names[lo:hi]

    tmp = tempfile.mkdtemp(prefix="cub_", dir=args.out)
    found = None
    missing = 0
    sizes = []
    try:
        for i, name in enumerate(names):
            run([CU13_0, "-xelf", name, args.lib], cwd=tmp)
            p = os.path.join(tmp, name)
            if not os.path.exists(p):
                missing += 1
                continue
            with open(p, "rb") as f:
                data = f.read()
            sizes.append((len(data), name))
            if args.want.encode() in data:
                found = os.path.join(args.out, name)
                shutil.copyfile(p, found)
                print("FOUND %s (size %d)" % (name, len(data)), flush=True)
                break
            os.remove(p)
            if (i + 1) % 50 == 0:
                print("checked %d/%d (missing %d)" % (i + 1, len(names), missing),
                      flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("extracted: %d, missing: %d" % (len(sizes), missing))
    for sz, nm in sorted(sizes, reverse=True)[:5]:
        print("  largest: %8d  %s" % (sz, nm))
    if not found:
        print("NOT FOUND")
        return 1
    print("cubin:", found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

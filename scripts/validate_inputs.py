#!/usr/bin/env python3
"""Gate: verify that an input_spec is materialized, passed to the kernel, and
actually changes the output.

Fails (exit 1) if:
  * input files are not materialized for a declared input_spec, or
  * the workload command does not carry the --input* flags, or
  * two different input families produce identical golden output (the bug where
    a workload's prepare() silently skipped input materialization).

Usage:
  python3 scripts/validate_inputs.py --workload gemm \
      --families uniform,near_zero,extreme,ones --M 128 --N 128 --K 128
"""

import argparse
import array
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)

from core.config import load_lab_config       # noqa: E402
from core.workload import get_workload        # noqa: E402
from core.experiments import make_input_spec  # noqa: E402
from core import util                         # noqa: E402


def _absmax(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    if not a:
        return 0.0
    return max(abs(x) for x in a)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--workload", default="gemm")
    p.add_argument("--families", default="uniform,near_zero,extreme,ones")
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--lab-config", default=os.path.join(LAB_ROOT, "config.yaml"))
    args = p.parse_args(argv)

    lab = load_lab_config(args.lab_config)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    base_out = util.ensure_dir(os.path.join(lab.results_dir, "_validate"))

    rows = []
    ok = True
    for fam in families:
        params = {"dtype": "fp32", "seed": args.seed, "pattern": "normal"}
        if args.workload == "gemm":
            params.update({"M": args.M, "N": args.N, "K": args.K})
        params["input_spec"] = make_input_spec(args.workload, fam, args.seed)

        wl = get_workload(args.workload, params, lab)
        wl.prepare()
        paths = getattr(wl, "input_paths", {})
        if not paths:
            print("FAIL: no input_paths materialized for family '%s'" % fam)
            ok = False
            continue

        rd = util.ensure_dir(os.path.join(base_out, wl.name, fam))
        cmd = wl.command(os.path.join(rd, "out"), device=0)
        has_flag = any(str(a).startswith("--input") for a in cmd)
        if not has_flag:
            print("FAIL: command has no --input* flag for family '%s'" % fam)
            ok = False

        res = wl.run(os.path.join(rd, "out"), device=0, cwd=rd)
        if res["exit_code"] != 0:
            print("FAIL: workload run failed for family '%s': %s"
                  % (fam, res.get("stderr", "")[-300:]))
            ok = False
            continue

        outbin = os.path.join(rd, "out") + wl.output_specs[0]["suffix"]
        out_hash = util.sha256_file(outbin)
        in_abs = {n: _absmax(p) for n, p in paths.items()}
        rows.append((fam, in_abs, has_flag, out_hash[:16],
                     _absmax(outbin)))

    print("\nworkload=%s  M=%d N=%d K=%d" % (args.workload, args.M, args.N, args.K))
    print("%-14s %-28s %-8s %-18s %s" %
          ("family", "input |A| (|B|)", "flag", "golden-sha(16)", "out |C|max"))
    for fam, in_abs, flag, h, oc in rows:
        iv = " / ".join("%.3g" % v for v in in_abs.values())
        print("%-14s %-28s %-8s %-18s %.4g" % (fam, iv, flag, h, oc))

    hashes = [r[3] for r in rows]
    if len(set(hashes)) != len(hashes):
        print("\nFAIL: different input families produced IDENTICAL golden output "
              "(input_spec is not reaching the kernel).")
        ok = False
    else:
        print("\nOK: inputs materialized, passed via --input*, and change the output.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

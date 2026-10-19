#!/usr/bin/env python3
"""Profile an attention_forward run and list its kernel launches.

NVBitFI's profiler reports one record per kernel launch, in launch order, with
per-instruction-group dynamic counts. This is the map needed to select a stage
(projection / score / PV / output) and, for per-head launches, a head.

Usage:
  attention_profile.py --S 64 --D 256 --H 4 --tag attn_dev
"""
import argparse
import os
import sys

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if LAB_ROOT not in sys.path:
    sys.path.insert(0, LAB_ROOT)

from core.config import load_lab_config  # noqa: E402
from core.workload import get_workload    # noqa: E402
from core.injection import get_injector   # noqa: E402
from core import util                     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, default=64)
    p.add_argument("--D", type=int, default=256)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tag", default="attention_profile")
    p.add_argument("--lab-config", default=os.path.join(LAB_ROOT, "config.yaml"))
    args = p.parse_args(argv)

    lab = load_lab_config(args.lab_config)
    params = {"S": args.S, "D": args.D, "H": args.H, "dtype": "bf16",
              "seed": args.seed, "dump": 0}
    wl = get_workload("attention_forward", params, lab)
    wl.prepare()
    cache = util.ensure_dir(os.path.join(lab.results_dir, "_profile", args.tag))
    inj = get_injector("nvbitfi", lab)
    prof = inj.profile(wl, cache)
    runs = prof.get("runs") or []
    print("kernels: %d" % len(runs))
    print("%4s  %-58s %10s %8s %8s" % ("idx", "name", "instrs", "mma", "fp32"))
    for r in runs:
        g = r.get("groups") or {}
        print("%4d  %-58s %10s %8s %8s"
              % (r.get("index"), (r.get("name") or "")[:58], r.get("instrs"),
                 g.get("mma"), g.get("fp32")))
    util.write_json(os.path.join(cache, "profile_list.json"),
                    {"params": params, "runs": runs})
    print("wrote", os.path.join(cache, "profile_list.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

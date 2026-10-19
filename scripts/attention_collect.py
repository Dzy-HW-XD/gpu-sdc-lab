#!/usr/bin/env python3
"""Collect per-injection attention metrics for a finished experiment.

Reads <experiment_dir>/observations.jsonl (written by the runner), compares each
faulty run's intermediate tensors against its golden run, and writes:

  attention_runs.jsonl          one record per injection (all classifications;
                                includes metrics when computable)
  attention_metrics/seq_*.json  per-run full metrics (computable runs only)
  attention_summary.csv         flat per-injection table
  attention_by_stage.csv        aggregate by (stage, head)

Usage:
  attention_collect.py results/041_attn_stage_dev [--abs-tol 0 --rel-tol 0]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import attention_metrics as M  # noqa: E402

SUFFIX = ".Y.bin"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--abs-tol", type=float, default=0.0)
    ap.add_argument("--rel-tol", type=float, default=0.0)
    ap.add_argument("--per-run-dir", default=None)
    args = ap.parse_args(argv)

    exp = args.experiment_dir
    obs_path = os.path.join(exp, "observations.jsonl")
    if not os.path.exists(obs_path):
        raise SystemExit("missing %s" % obs_path)
    run_dir = args.per_run_dir or os.path.join(exp, "attention_metrics")
    os.makedirs(run_dir, exist_ok=True)

    runs, rows = [], []
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cls = rec.get("classification")
            labels = rec.get("labels") or {}
            fault = rec.get("fault") or {}
            stage = labels.get("stage")
            head = labels.get("head")
            bit = labels.get("bit")
            seq = rec.get("seq")
            g = rec.get("golden_bin")
            fo = rec.get("raw_out_bin")
            mres = None
            if (cls in ("MASKED", "SDC") and g and fo
                    and g.endswith(SUFFIX) and fo.endswith(SUFFIX)):
                gp, fp = g[:-len(SUFFIX)], fo[:-len(SUFFIX)]
                if os.path.exists(gp + ".json") and os.path.exists(fp + ".json"):
                    try:
                        mres = M.compute(gp, fp, args.abs_tol, args.rel_tol)
                    except Exception as exc:  # keep going on a bad run
                        mres = {"error": str(exc), "tensors": {}}
                    mres.update({
                        "experiment_id": rec.get("experiment_id"),
                        "seq": seq, "stage": stage, "head": head, "bit": bit,
                        "run_classification": cls,
                        "fault": fault,
                        "runtime_sec": (rec.get("runtime") or {}).get("runtime_sec"),
                    })
                    with open(os.path.join(run_dir, "seq_%s.json" % seq),
                              "w", encoding="utf-8") as fh:
                        fh.write(json.dumps(mres, sort_keys=True) + "\n")
            run_rec = {
                "seq": seq, "stage": stage, "head": head, "bit": bit,
                "classification": cls,
                "injected": fault.get("injected"),
                "static_site": fault.get("static_site"),
                "inst_id": fault.get("inst_id"),
                "inst_count": fault.get("inst_count_profile"),
                "runtime_sec": (rec.get("runtime") or {}).get("runtime_sec"),
            }
            if mres and "tensors" in mres:
                y = mres["tensors"].get("Y", {})
                o = mres["tensors"].get("O", {})
                run_rec["y_rel_l2"] = y.get("relative_l2_error")
                run_rec["y_max_abs"] = y.get("max_abs_error")
                run_rec["y_changed"] = y.get("changed_elements")
                run_rec["o_head_fro"] = o.get("head_error_fro")
            runs.append(run_rec)
            rows.append({
                "seq": seq, "stage": stage, "head": head, "bit": bit,
                "classification": cls,
                "injected": fault.get("injected"),
                "static_site": fault.get("static_site"),
                "inst_id": fault.get("inst_id"),
                "inst_count": fault.get("inst_count_profile"),
                "y_rel_l2": run_rec.get("y_rel_l2", ""),
                "y_max_abs": run_rec.get("y_max_abs", ""),
                "y_changed": run_rec.get("y_changed", ""),
                "runtime_sec": run_rec.get("runtime_sec", ""),
            })

    with open(os.path.join(exp, "attention_runs.jsonl"), "w",
              encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    cols = ["seq", "stage", "head", "bit", "classification", "injected",
            "static_site", "inst_id", "inst_count", "y_rel_l2", "y_max_abs",
            "y_changed", "runtime_sec"]
    with open(os.path.join(exp, "attention_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    agg = {}
    for r in rows:
        key = (r["stage"], str(r["head"]))
        a = agg.setdefault(key, {"n": 0, "MASKED": 0, "SDC": 0, "CRASH": 0,
                                 "NOT_INJECTED": 0, "NOT_TARGETED": 0,
                                 "rel_l2_sum": 0.0, "rel_l2_n": 0})
        a["n"] += 1
        c = r["classification"]
        if c in a:
            a[c] += 1
        if r.get("y_rel_l2") not in ("", None):
            a["rel_l2_sum"] += float(r["y_rel_l2"])
            a["rel_l2_n"] += 1
    with open(os.path.join(exp, "attention_by_stage.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "head", "n", "MASKED", "SDC", "CRASH",
                    "NOT_INJECTED", "NOT_TARGETED", "mean_y_rel_l2"])
        for (stage, head) in sorted(agg):
            a = agg[(stage, head)]
            w.writerow([stage, head, a["n"], a["MASKED"], a["SDC"], a["CRASH"],
                        a["NOT_INJECTED"], a["NOT_TARGETED"],
                        ("%.6g" % (a["rel_l2_sum"] / a["rel_l2_n"]))
                        if a["rel_l2_n"] else ""])

    print("wrote %s/attention_runs.jsonl (%d runs)" % (exp, len(runs)))
    print("wrote %s/attention_summary.csv" % exp)
    print("wrote %s/attention_by_stage.csv" % exp)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Collect per-injection attention metrics for a finished experiment.

Reads <experiment_dir>/observations.jsonl (written by the runner), compares each
faulty run's intermediate tensors against its golden run, and writes:

  attention_runs.jsonl          one record per injection (all classifications;
                                includes L0 metrics when computable)
  attention_metrics/seq_*.json  per-run full metrics (computable runs only)
  attention_summary.csv         flat per-injection table (L0 / configured tol)
  attention_by_stage.csv        aggregate by (stage, head, position)
  attention_levels.csv          long table: one row per run x tolerance level
  attention_by_level.csv        aggregate by (stage, head, position, level)

Tolerance levels are re-derived from the final Y (offline, like
tolerance_sweep.py): an element is corrupted iff
  |Y_faulty - Y_golden| > abs_tol + rel_tol*|Y_golden|.
Default levels: L0 (0,0), L1..L5 = (1e-7,1e-6) .. (1e-3,1e-2).

Usage:
  attention_collect.py results/041_attn_stage_dev [--levels L0=0:0,L1=...]
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
LEVELS_DEFAULT = [
    ("L0", 0.0, 0.0),
    ("L1", 1e-7, 1e-6),
    ("L2", 1e-6, 1e-5),
    ("L3", 1e-5, 1e-4),
    ("L4", 1e-4, 1e-3),
    ("L5", 1e-3, 1e-2),
]


def parse_levels(spec):
    if not spec:
        return LEVELS_DEFAULT
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            name, pair = tok.split("=", 1)
        else:
            name, pair = "L%d" % len(out), tok
        a, _, r = pair.partition(":")
        out.append((name, float(a), float(r)))
    return out


def y_level_rows(golden_prefix, faulty_prefix, levels):
    """Reclassify the final Y at each tolerance level (loads Y once)."""
    g = M.load_f32(golden_prefix + SUFFIX)
    f = M.load_f32(faulty_prefix + SUFFIX)
    rows = []
    for name, a, r in levels:
        m = M.diff_metrics(g, f, a, r)
        cls = "SDC" if m["changed_elements"] > 0 else "MASKED"
        rows.append((name, cls, m))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--abs-tol", type=float, default=0.0)
    ap.add_argument("--rel-tol", type=float, default=0.0)
    ap.add_argument("--per-run-dir", default=None)
    ap.add_argument("--levels", default=None,
                    help="override, e.g. L0=0:0,L1=1e-7:1e-6")
    args = ap.parse_args(argv)
    levels = parse_levels(args.levels)

    exp = args.experiment_dir
    obs_path = os.path.join(exp, "observations.jsonl")
    if not os.path.exists(obs_path):
        raise SystemExit("missing %s" % obs_path)
    run_dir = args.per_run_dir or os.path.join(exp, "attention_metrics")
    os.makedirs(run_dir, exist_ok=True)

    runs, rows, level_records = [], [], []
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
            position = labels.get("position", "all")
            bit = labels.get("bit")
            seq = rec.get("seq")
            g = rec.get("golden_bin")
            fo = rec.get("raw_out_bin")
            gp = fp = None
            if (g and fo and g.endswith(SUFFIX) and fo.endswith(SUFFIX)):
                gp, fp = g[:-len(SUFFIX)], fo[:-len(SUFFIX)]

            mres = None
            if (cls in ("MASKED", "SDC") and gp and fp
                    and os.path.exists(gp + ".json") and os.path.exists(fp + ".json")):
                try:
                    mres = M.compute(gp, fp, args.abs_tol, args.rel_tol)
                except Exception as exc:  # keep going on a bad run
                    mres = {"error": str(exc), "tensors": {}}
                mres.update({
                    "experiment_id": rec.get("experiment_id"),
                    "seq": seq, "stage": stage, "head": head,
                    "position": position, "bit": bit,
                    "run_classification": cls,
                    "fault": fault,
                    "runtime_sec": (rec.get("runtime") or {}).get("runtime_sec"),
                })
                with open(os.path.join(run_dir, "seq_%s.json" % seq),
                          "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(mres, sort_keys=True) + "\n")

            # ---- L0..L5 reclassification -----------------------------
            lrows = []
            if cls in ("MASKED", "SDC") and gp and fp and os.path.exists(
                    gp + SUFFIX) and os.path.exists(fp + SUFFIX):
                try:
                    lrows = y_level_rows(gp, fp, levels)
                except Exception:
                    lrows = [(name, cls, None) for name, _, _ in levels]
            else:
                lrows = [(name, cls, None) for name, _, _ in levels]
            for name, lcls, lm in lrows:
                level_records.append({
                    "seq": seq, "stage": stage, "head": head,
                    "position": position, "bit": bit, "level": name,
                    "classification": lcls,
                    "y_rel_l2": (lm or {}).get("relative_l2_error", ""),
                    "y_max_abs": (lm or {}).get("max_abs_error", ""),
                    "y_changed": (lm or {}).get("changed_elements", ""),
                })

            run_rec = {
                "seq": seq, "stage": stage, "head": head,
                "position": position, "bit": bit,
                "classification": cls,
                "injected": fault.get("injected"),
                "static_site": fault.get("static_site"),
                "inst_id": fault.get("inst_id"),
                "inst_count": fault.get("inst_count_profile"),
                "inst_fraction": fault.get("inst_fraction"),
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
                "seq": seq, "stage": stage, "head": head,
                "position": position, "bit": bit,
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

    cols = ["seq", "stage", "head", "position", "bit", "classification",
            "injected", "static_site", "inst_id", "inst_count", "y_rel_l2",
            "y_max_abs", "y_changed", "runtime_sec"]
    with open(os.path.join(exp, "attention_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # ---- aggregate by (stage, head, position) -------------------------
    agg = {}
    for r in rows:
        key = (r["stage"], str(r["head"]), str(r["position"]))
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
        w.writerow(["stage", "head", "position", "n", "MASKED", "SDC",
                    "CRASH", "NOT_INJECTED", "NOT_TARGETED", "mean_y_rel_l2"])
        for key in sorted(agg):
            a = agg[key]
            w.writerow(list(key) + [a["n"], a["MASKED"], a["SDC"], a["CRASH"],
                                    a["NOT_INJECTED"], a["NOT_TARGETED"],
                                    ("%.6g" % (a["rel_l2_sum"] / a["rel_l2_n"]))
                                    if a["rel_l2_n"] else ""])

    # ---- long per-level table -----------------------------------------
    lcols = ["seq", "stage", "head", "position", "bit", "level",
             "classification", "y_rel_l2", "y_max_abs", "y_changed"]
    with open(os.path.join(exp, "attention_levels.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=lcols)
        w.writeheader()
        for r in level_records:
            w.writerow({k: r.get(k, "") for k in lcols})

    # ---- aggregate by (stage, head, position, level) ------------------
    lagg = {}
    for r in level_records:
        key = (r["stage"], str(r["head"]), str(r["position"]), r["level"])
        a = lagg.setdefault(key, {"n": 0, "MASKED": 0, "SDC": 0, "CRASH": 0,
                                  "NOT_INJECTED": 0, "NOT_TARGETED": 0})
        a["n"] += 1
        c = r["classification"]
        if c in a:
            a[c] += 1
    order = {name: i for i, (name, _, _) in enumerate(levels)}
    with open(os.path.join(exp, "attention_by_level.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "head", "position", "level", "n", "MASKED",
                    "SDC", "CRASH", "NOT_INJECTED", "NOT_TARGETED", "sdc_rate"])
        for key in sorted(lagg, key=lambda k: (k[0], str(k[1]), str(k[2]),
                                               order.get(k[3], 99))):
            a = lagg[key]
            denom = a["MASKED"] + a["SDC"] + a["CRASH"]
            w.writerow(list(key) + [a["n"], a["MASKED"], a["SDC"], a["CRASH"],
                                    a["NOT_INJECTED"], a["NOT_TARGETED"],
                                    "%.4f" % (a["SDC"] / denom) if denom else ""])

    print("wrote %s/attention_runs.jsonl (%d runs)" % (exp, len(runs)))
    print("wrote %s/attention_summary.csv / attention_by_stage.csv" % exp)
    print("wrote %s/attention_levels.csv / attention_by_level.csv (%d rows)"
          % (exp, len(level_records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

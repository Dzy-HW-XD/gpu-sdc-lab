#!/usr/bin/env python3
"""Attention tensor-diff metrics (stdlib only; runs on the GPU node).

Compares a faulty run prefix against its golden prefix and reports, for every
intermediate tensor dumped by `attention_forward`:

  max_abs_error, mean_abs_error, relative_l2_error,
  changed_elements / changed_ratio (with an abs+rel tolerance),
  nan/inf counts, error support (tokens / features),
  per-head Frobenius error (for 3-D tensors O, S, P),
  per-token / per-feature max-abs vectors (for 2-D tensors).

Usage:
  attention_metrics.py --golden <prefix> --faulty <prefix> \
      --abs-tol 0 --rel-tol 0 \
      --out-json out.json --out-jsonl runs.jsonl \
      [--experiment-id 040 --stage qk --head 3]
"""
import argparse
import array
import json
import math
import os
import sys

TENSORS = ["Q", "K", "V", "S", "P", "O", "Ocat", "Y"]


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def _finite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def shapes_from_meta(meta):
    S, D, H = int(meta["S"]), int(meta["D"]), int(meta["H"])
    dh = int(meta.get("dh", D // H))
    return {
        "Q": (S, D), "K": (S, D), "V": (S, D),
        "S": (H, S, S), "P": (H, S, S), "O": (H, S, dh),
        "Ocat": (S, D), "Y": (S, D),
    }


def diff_metrics(g, f, abs_tol, rel_tol):
    n = len(g)
    max_abs = 0.0
    sum_abs = 0.0
    sumsq_d = 0.0
    sumsq_g = 0.0
    changed = 0
    nan = inf = 0
    for i in range(n):
        gv = g[i]
        fv = f[i]
        if not _finite(fv):
            if fv != fv:
                nan += 1
            else:
                inf += 1
        if not _finite(gv) or not _finite(fv):
            if not (fv == gv or (fv != fv and gv != gv)):
                changed += 1
            continue
        d = fv - gv
        ad = d if d >= 0.0 else -d
        sum_abs += ad
        if ad > max_abs:
            max_abs = ad
        sumsq_d += d * d
        sumsq_g += gv * gv
        ag = gv if gv >= 0.0 else -gv
        if ad > abs_tol + rel_tol * ag:
            changed += 1
    finite_n = n - nan - inf
    rel_l2 = math.sqrt(sumsq_d / sumsq_g) if sumsq_g > 0.0 else math.sqrt(sumsq_d)
    return {
        "elements": n,
        "max_abs_error": max_abs,
        "mean_abs_error": (sum_abs / finite_n) if finite_n else 0.0,
        "relative_l2_error": rel_l2,
        "changed_elements": changed,
        "changed_ratio": (float(changed) / n) if n else 0.0,
        "nan_count": nan,
        "inf_count": inf,
    }


def matrix_metrics(g, f, shape, abs_tol, rel_tol):
    rows, cols = shape
    out = diff_metrics(g, f, abs_tol, rel_tol)
    row_max = [0.0] * rows
    col_max = [0.0] * cols
    row_changed = [0] * rows
    col_changed = [0] * cols
    for r in range(rows):
        base = r * cols
        for c in range(cols):
            gv = g[base + c]
            fv = f[base + c]
            if not (_finite(gv) and _finite(fv)):
                if not (fv == gv or (fv != fv and gv != gv)):
                    row_changed[r] = 1
                    col_changed[c] = 1
                continue
            d = fv - gv
            ad = d if d >= 0.0 else -d
            if ad > row_max[r]:
                row_max[r] = ad
            if ad > col_max[c]:
                col_max[c] = ad
            ag = gv if gv >= 0.0 else -gv
            if ad > abs_tol + rel_tol * ag:
                row_changed[r] = 1
                col_changed[c] = 1
    out["support_rows"] = sum(row_changed)
    out["support_cols"] = sum(col_changed)
    out["row_max_abs"] = row_max
    out["col_max_abs"] = col_max
    return out


def head_metrics(g, f, shape, abs_tol, rel_tol):
    H, A, B = shape
    out = diff_metrics(g, f, abs_tol, rel_tol)
    slab = A * B
    fro = []
    head_changed = []
    for h in range(H):
        base = h * slab
        ss = 0.0
        changed = 0
        for i in range(base, base + slab):
            gv, fv = g[i], f[i]
            if _finite(gv) and _finite(fv):
                d = fv - gv
                ss += d * d
                ag = gv if gv >= 0.0 else -gv
                if (d if d >= 0 else -d) > abs_tol + rel_tol * ag:
                    changed += 1
            elif not (fv == gv or (fv != fv and gv != gv)):
                changed += 1
        fro.append(math.sqrt(ss))
        head_changed.append(changed)
    out["head_error_fro"] = fro
    out["head_changed"] = head_changed
    return out


def compute(golden, faulty, abs_tol=0.0, rel_tol=0.0):
    """Return the full per-tensor metrics dict for one (golden, faulty) pair."""
    with open(golden + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    shapes = shapes_from_meta(meta)
    result = {
        "golden": golden,
        "faulty": faulty,
        "config": {k: meta.get(k) for k in ("S", "D", "H", "dh", "dtype",
                                            "accumulate", "mask", "seed")},
        "tolerance": {"abs": abs_tol, "rel": rel_tol},
        "tensors": {},
    }
    ok = True
    for name in TENSORS:
        gp = golden + "." + name + ".bin"
        fp = faulty + "." + name + ".bin"
        if not (os.path.exists(gp) and os.path.exists(fp)):
            continue
        g = load_f32(gp)
        f = load_f32(fp)
        shape = shapes[name]
        if len(g) != len(f) or len(g) != _prod(shape):
            result["tensors"][name] = {"error": "size mismatch",
                                       "shape": list(shape),
                                       "golden_len": len(g), "faulty_len": len(f)}
            ok = False
            continue
        if len(shape) == 2:
            m = matrix_metrics(g, f, shape, abs_tol, rel_tol)
        else:
            m = head_metrics(g, f, shape, abs_tol, rel_tol)
        m["shape"] = list(shape)
        result["tensors"][name] = m
    y = result["tensors"].get("Y", {})
    if y and "changed_elements" in y:
        result["classification"] = "SDC" if y["changed_elements"] > 0 else "MASKED"
    result["valid"] = bool(ok and y.get("changed_elements") is not None)
    return result


def _prod(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--golden", required=True)
    p.add_argument("--faulty", required=True)
    p.add_argument("--abs-tol", type=float, default=0.0)
    p.add_argument("--rel-tol", type=float, default=0.0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--head", default=None)
    p.add_argument("--classification", default=None)
    args = p.parse_args(argv)

    result = compute(args.golden, args.faulty, args.abs_tol, args.rel_tol)
    result.update({"experiment_id": args.experiment_id,
                   "stage": args.stage, "head": args.head})
    if args.classification:
        result["classification"] = args.classification

    text = json.dumps(result, sort_keys=True)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.out_jsonl:
        with open(args.out_jsonl, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    y = result["tensors"].get("Y", {})
    print("attention_metrics: stage=%s head=%s class=%s"
          % (args.stage, args.head, result.get("classification")))
    print("  Y: max_abs=%.4g rel_l2=%.4g changed=%s (%.3g)"
          % (y.get("max_abs_error", 0.0), y.get("relative_l2_error", 0.0),
             y.get("changed_elements"), y.get("changed_ratio", 0.0)))
    for name, m in result["tensors"].items():
        if "relative_l2_error" in m:
            print("  %-5s rel_l2=%.4g max_abs=%.4g changed=%d"
                  % (name, m["relative_l2_error"], m["max_abs_error"],
                     m["changed_elements"]))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

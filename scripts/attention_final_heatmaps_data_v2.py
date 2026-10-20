#!/usr/bin/env python3
"""Build L4 final-Y heatmap data, v2 (stdlib only; runs on the GPU node).

Difference from v1 (`attention_final_heatmaps_data.py`):
  v1 `meanlog` averaged log10|dY| over *all* runs with a non-zero change,
  including sub-threshold FP noise.  For broad stages such as qkv_k that made
  almost every element look affected (uniform red background).
  v2 averages only over runs in which the element actually exceeds L4, i.e.
  the conditional error magnitude, and additionally stores the per-element
  maximum and the affected-run count.

Outputs per cell (S x D):
  support/<tag>.bin    float32  affected runs / n_runs          (0..1)
  count/<tag>.bin      uint8    affected runs                   (0..n)
  meanlog/<tag>.bin    float32  mean log10|dY| over affected runs (NaN if none)
  maxlog/<tag>.bin     float32  max  log10|dY| over affected runs (NaN if none)

Usage:
  python3 scripts/attention_final_heatmaps_data_v2.py \
      --exp results/047_attn_5gemm_full \
      --out results/047_attn_5gemm_full/final_heatmaps_data_v2
"""
import argparse
import array
import collections
import csv
import json
import math
import os

SUFFIX = ".Y.bin"
LOG_NF = 40.0   # sentinel log10 magnitude for non-finite (NaN/Inf) outputs


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def write_bin(path, arr):
    with open(path, "wb") as f:
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--abs-tol", type=float, default=1e-4)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    args = ap.parse_args()
    out = args.out or os.path.join(args.exp, "final_heatmaps_data_v2")
    dirs = {k: os.path.join(out, k) for k in
            ("support", "count", "nfcount", "meanlog", "maxlog")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    obs_path = os.path.join(args.exp, "observations.jsonl")
    groups = collections.defaultdict(list)
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("classification") not in ("MASKED", "SDC"):
                continue
            g, fo = rec.get("golden_bin"), rec.get("raw_out_bin")
            if not (g and fo and g.endswith(SUFFIX) and fo.endswith(SUFFIX)):
                continue
            labels = rec.get("labels") or {}
            key = (labels.get("stage"), labels.get("bit"),
                   labels.get("position", "all"))
            groups[key].append(rec)

    meta_cache = {}
    golden_cache = {}
    footprint = []
    manifest = []
    log10 = math.log10
    abs_tol, rel_tol = args.abs_tol, args.rel_tol
    nan = float("nan")

    for key in sorted(groups, key=lambda k: (str(k[0]), int(k[1]), str(k[2]))):
        stage, bit, position = key
        recs = groups[key]
        gpath = recs[0]["golden_bin"]
        gprefix = gpath[:-len(SUFFIX)]
        meta = meta_cache.get(gprefix)
        if meta is None:
            with open(gprefix + ".json", encoding="utf-8") as mf:
                meta = json.load(mf)
            meta_cache[gprefix] = meta
        S, D = int(meta["S"]), int(meta["D"])
        N = S * D
        gold = golden_cache.get(gpath)
        if gold is None:
            gold = load_f32(gpath)
            golden_cache[gpath] = gold

        count = [0] * N
        nfcount = [0] * N
        sumlog = [0.0] * N
        maxlog = [-1e30] * N
        nvalid = 0
        nf_runs = 0
        best = None
        best_rel = -1.0
        for rec in recs:
            fpath = rec["raw_out_bin"]
            if not os.path.exists(fpath):
                continue
            fault = load_f32(fpath)
            if len(fault) != N:
                continue
            row_any = bytearray(S)
            col_any = bytearray(D)
            changed = 0
            nf_this = 0
            for i in range(N):
                gv = gold[i]
                fv = fault[i]
                if not (math.isfinite(gv) and math.isfinite(fv)):
                    # NaN/Inf is a catastrophic (non-finite) deviation; count
                    # it as affected and assign the sentinel magnitude.
                    count[i] += 1
                    nfcount[i] += 1
                    sumlog[i] += LOG_NF
                    if LOG_NF > maxlog[i]:
                        maxlog[i] = LOG_NF
                    nf_this += 1
                    changed += 1
                    row_any[i // D] = 1
                    col_any[i % D] = 1
                    continue
                d = fv - gv
                ad = -d if d < 0.0 else d
                if ad > abs_tol + rel_tol * (gv if gv >= 0.0 else -gv):
                    l = log10(ad) if ad > 0.0 else -1e30
                    count[i] += 1
                    sumlog[i] += l
                    if l > maxlog[i]:
                        maxlog[i] = l
                    changed += 1
                    row_any[i // D] = 1
                    col_any[i % D] = 1
            nvalid += 1
            if nf_this:
                nf_runs += 1
            rel = (rec.get("observation") or {}).get("relative_l2_error", "")
            labels = rec.get("labels") or {}
            footprint.append({
                "seq": rec.get("seq"), "stage": stage,
                "head": labels.get("head", ""), "position": position,
                "bit": bit, "classification": rec.get("classification"),
                "changed": changed, "affected_tokens": sum(row_any),
                "affected_features": sum(col_any), "nonfinite": nf_this,
                "y_rel_l2": rel,
            })
            try:
                rv = float(rel)
            except (TypeError, ValueError):
                rv = -1.0
            if rv > best_rel:
                best_rel = rv
                best = rec

        if not nvalid:
            continue
        labels = (best.get("labels") or {}) if best else {}
        head = labels.get("head", "")
        tag = "final_%s_h%s_%s_bit%s" % (
            stage, head if head not in ("", None) else "x", position, bit)

        support = array.array("f", (count[i] / float(nvalid) for i in range(N)))
        cnt8 = array.array("B", bytes(min(count[i], 255) for i in range(N)))
        nf8 = array.array("B", bytes(min(nfcount[i], 255) for i in range(N)))
        meanlog = array.array("f", (
            (sumlog[i] / count[i]) if count[i] else nan for i in range(N)))
        maxarr = array.array("f", (
            maxlog[i] if count[i] else nan for i in range(N)))

        write_bin(os.path.join(dirs["support"], tag + ".bin"), support)
        write_bin(os.path.join(dirs["count"], tag + ".bin"), cnt8)
        write_bin(os.path.join(dirs["nfcount"], tag + ".bin"), nf8)
        write_bin(os.path.join(dirs["meanlog"], tag + ".bin"), meanlog)
        write_bin(os.path.join(dirs["maxlog"], tag + ".bin"), maxarr)

        manifest.append({
            "tag": tag, "stage": stage, "head": labels.get("head", ""),
            "position": position, "bit": bit,
            "seq": best.get("seq") if best else "",
            "y_rel_l2": (best.get("observation", {}) or {}).get(
                "relative_l2_error", "") if best else "",
            "golden": best["golden_bin"][:-len(SUFFIX)] if best else "",
            "faulty": best["raw_out_bin"][:-len(SUFFIX)] if best else "",
            "support": os.path.join(dirs["support"], tag + ".bin"),
            "count": os.path.join(dirs["count"], tag + ".bin"),
            "nfcount": os.path.join(dirs["nfcount"], tag + ".bin"),
            "meanlog": os.path.join(dirs["meanlog"], tag + ".bin"),
            "maxlog": os.path.join(dirs["maxlog"], tag + ".bin"),
            "n_runs": nvalid,
            "nf_runs": nf_runs,
            "sdc_runs": sum(1 for r in recs
                            if r.get("classification") == "SDC"),
        })
        print("cell %-34s runs=%d S=%d D=%d affected=%.1f%% nf_runs=%d" % (
            tag, nvalid, S, D,
            100.0 * sum(1 for x in count if x) / N, nf_runs), flush=True)

    with open(os.path.join(out, "footprint.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["seq", "stage", "head", "position", "bit", "classification",
                "changed", "affected_tokens", "affected_features", "nonfinite",
                "y_rel_l2"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(footprint)
    with open(os.path.join(out, "final_heatmaps_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        cols = ["tag", "stage", "head", "position", "bit", "seq", "y_rel_l2",
                "golden", "faulty", "support", "count", "nfcount", "meanlog",
                "maxlog", "n_runs", "nf_runs", "sdc_runs"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(manifest)
    print("wrote %s (%d cells, %d runs)" % (out, len(manifest), len(footprint)))


if __name__ == "__main__":
    main()

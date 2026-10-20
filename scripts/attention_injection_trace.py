#!/usr/bin/env python3
"""Flatten the low-level injection provenance + outcome for every run.

Merges observations.jsonl (fault metadata, oracle metrics, runtime), the
L0-L5 table (attention_levels.csv) and the per-head O errors
(attention_head_errors.jsonl) into one CSV/JSONL trace.

Output columns include the SASS-level injection record:
  kernel, kernel_count, inst_id, inst_count, inst_fraction,
  pc_offset, opcode, static_site, register, register_class,
  mask, before_val, after_val, thread, injected, target_matched,
  error_not_injected, kernel_error
plus the observable outcome and L0..L5 classifications.

Usage: attention_injection_trace.py --exp results/046_attn_mha_bit_sweep
"""
import argparse
import csv
import json
import os
import sys

COLS = [
    "seq", "experiment_id", "stage", "head", "position", "bit",
    "classification", "classification_raw", "reason",
    "kernel", "kernel_count", "inst_id", "inst_count", "inst_fraction",
    "pc_offset", "opcode", "static_site", "register", "register_class",
    "mask", "before_val", "after_val", "thread",
    "injected", "target_matched", "error_not_injected", "kernel_error",
    "y_max_abs", "y_rel_l2", "y_changed", "y_changed_ratio",
    "nan_count", "inf_count", "bitwise_equal",
    "runtime_sec", "timeout", "crash",
    "level",
    "o_self_fro", "o_max_other_fro", "o_mean_all_fro",
]


def load_levels(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(str(r["seq"]), {})[r["level"]] = r["classification"]
    return out


def load_head_errors(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[str(r.get("seq"))] = r.get("o_head_fro")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args(argv)

    exp = args.exp
    obs_path = os.path.join(exp, "observations.jsonl")
    if not os.path.exists(obs_path):
        raise SystemExit("missing %s" % obs_path)
    hfro = load_head_errors(os.path.join(exp, "attention_head_errors.jsonl"))

    rows = []
    with open(obs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = rec.get("labels") or {}
            fault = rec.get("fault") or {}
            obs = rec.get("observation") or {}
            rt = rec.get("runtime") or {}
            seq = str(rec.get("seq"))
            fro = hfro.get(seq)
            head = labels.get("head")
            o_self = o_max_other = o_mean = ""
            if fro:
                if head is not None:
                    h = int(head)
                    o_self = fro[h] if h < len(fro) else ""
                    others = [v for i, v in enumerate(fro) if i != h]
                    o_max_other = max(others) if others else ""
                else:
                    o_max_other = max(fro) if fro else ""
                o_mean = sum(fro) / len(fro) if fro else ""
            rows.append({
                "seq": rec.get("seq"), "experiment_id": rec.get("experiment_id"),
                "stage": labels.get("stage"), "head": head,
                "position": labels.get("position", "all"), "bit": labels.get("bit"),
                "classification": rec.get("classification"),
                "classification_raw": rec.get("classification_raw"),
                "reason": rec.get("reason"),
                "kernel": fault.get("kernel"), "kernel_count": fault.get("kernel_count"),
                "inst_id": fault.get("inst_id"),
                "inst_count": fault.get("inst_count_profile"),
                "inst_fraction": fault.get("inst_fraction"),
                "pc_offset": fault.get("pc_offset"), "opcode": fault.get("opcode"),
                "static_site": fault.get("static_site"),
                "register": fault.get("register"),
                "register_class": fault.get("register_class"),
                "mask": fault.get("mask"), "before_val": fault.get("before_val"),
                "after_val": fault.get("after_val"), "thread": fault.get("thread"),
                "injected": fault.get("injected"),
                "target_matched": fault.get("target_matched"),
                "error_not_injected": fault.get("error_not_injected"),
                "kernel_error": fault.get("kernel_error"),
                "y_max_abs": obs.get("max_abs_error"),
                "y_rel_l2": obs.get("relative_l2_error"),
                "y_changed": obs.get("corrupted_elements"),
                "y_changed_ratio": obs.get("corrupted_element_ratio"),
                "nan_count": obs.get("nan_count"), "inf_count": obs.get("inf_count"),
                "bitwise_equal": obs.get("bitwise_equal"),
                "runtime_sec": rt.get("runtime_sec"), "timeout": rt.get("timeout"),
                "crash": rt.get("crash"),
                "level": "L4",
                "o_self_fro": o_self, "o_max_other_fro": o_max_other,
                "o_mean_all_fro": o_mean,
            })

    csv_path = os.path.join(exp, "injection_trace.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLS})
    jsonl_path = os.path.join(exp, "injection_trace.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print("wrote %s (%d rows)" % (csv_path, len(rows)))
    print("wrote %s" % jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

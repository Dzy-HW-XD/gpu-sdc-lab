#!/usr/bin/env python3
"""Post-process high-bit output-corruption runs.

The raw runs are produced by scripts/llm_highbit_output.py.  This script
recomputes the output labels with a stricter, position-aware answer check and
builds the cross-model summary tables used by the report and figure.

No GPU is needed.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict


MODEL_ORDER = [
    "pythia160m", "pythia410m", "pythia1b", "qwen05b", "qwen15b",
]
MODEL_LABEL = {
    "pythia160m": "Pythia-160M",
    "pythia410m": "Pythia-410M",
    "pythia1b": "Pythia-1B",
    "qwen05b": "Qwen2.5-0.5B",
    "qwen15b": "Qwen2.5-1.5B",
}
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)
SHORT_ANSWER_CATEGORIES = {"arithmetic", "list", "code"}


def toks(s):
    return TOKEN_RE.findall((s or "").lower())


def find_expected(text, expected):
    """First contiguous expected-token position in the first output line."""
    if not text:
        return None
    first = text.strip().splitlines()[0] if text.strip() else ""
    out = toks(first)
    positions = []
    for e in expected:
        et = toks(e)
        if not et:
            continue
        for i in range(len(out) - len(et) + 1):
            if out[i:i + len(et)] == et:
                positions.append(i)
    return min(positions) if positions else None


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_sim(a, b):
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def token_f1(a, b):
    ta = re.findall(r"[a-z0-9]+", (a or "").lower())
    tb = re.findall(r"[a-z0-9]+", (b or "").lower())
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


def script_mix(s):
    letters = [ch for ch in (s or "") if ch.isalpha()]
    if not letters:
        return 0.0
    bad = 0
    for ch in letters:
        name = unicodedata.name(ch, "")
        if any(k in name for k in ("CJK", "HIRAGANA", "KATAKANA", "HANGUL",
                                   "CYRILLIC", "ARABIC", "HEBREW", "THAI")):
            bad += 1
    return bad / len(letters)


def text_flags(s, clean_s=None):
    s = s or ""
    clean_s = clean_s or ""
    n = max(len(s), 1)
    replacement = sum(1 for ch in s if ch == "\ufffd") / n
    control = sum(
        1 for ch in s
        if unicodedata.category(ch).startswith("C") and ch not in "\n\r\t"
    ) / n
    empty = len(s.strip()) == 0
    smix = script_mix(s)

    words = re.findall(r"\S+", s.lower())
    degenerate = False
    if len(words) >= 8:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams:
            max_count = max(Counter(trigrams).values())
            rep_ratio = max_count / len(trigrams)
            bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
            distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
            degenerate = (max_count >= 4 and rep_ratio > 0.4) or distinct2 < 0.2

    clean_smix = script_mix(clean_s) if clean_s else 0.0
    return {
        "replacement_rate": replacement,
        "control_rate": control,
        "empty": int(empty),
        "degenerate": int(degenerate),
        "script_mix": smix,
        "unexpected_script": int(clean_smix < 0.05 and smix > 0.10),
    }


def parse_float(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def add_metrics(r, prompts):
    p = prompts[r["prompt_id"]]
    expected = p["expected"]
    category = p["category"]
    clean_out = r.get("clean_output", "")
    out = r.get("output", "")
    cflags = text_flags(clean_out)
    clean_pos = find_expected(clean_out, expected)
    clean_ok = clean_pos is not None and clean_pos <= 4

    r["clean_ok2"] = int(clean_ok)
    r["answer_pos"] = clean_pos if clean_pos is not None else ""
    r["answer_present2"] = ""
    r["answer_start2"] = ""
    r["literal_garble2"] = ""
    r["semantic_wrong2"] = ""
    r["catastrophic2"] = ""
    r["partial2"] = ""
    r["preserved2"] = ""
    r["fault_ok2"] = int(clean_ok)

    if r["condition"] == "clean":
        r["literal_garble2"] = int(
            cflags["replacement_rate"] > 0 or cflags["control_rate"] > 0 or
            cflags["empty"] or cflags["degenerate"] or
            cflags["unexpected_script"])
        return r
    if r["condition"] == "clean_repeat":
        rep_flags = text_flags(out, clean_out)
        r["literal_garble2"] = int(
            rep_flags["replacement_rate"] > 0 or rep_flags["control_rate"] > 0 or
            (rep_flags["empty"] and not cflags["empty"]) or
            (rep_flags["degenerate"] and not cflags["degenerate"]) or
            rep_flags["unexpected_script"])
        return r

    flags = text_flags(out, clean_out)
    pos = find_expected(out, expected)
    limit = 6 if category in SHORT_ANSWER_CATEGORIES else None
    present = pos is not None and (limit is None or pos <= limit)
    start = pos is not None and pos <= 4
    sim = char_sim(clean_out, out)
    inj_deg = int(flags["degenerate"] and not cflags["degenerate"])
    lit = int(
        flags["replacement_rate"] > 0 or flags["control_rate"] > 0 or
        (flags["empty"] and not cflags["empty"]) or inj_deg or
        flags["unexpected_script"])
    ppl = parse_float(r.get("ppl_ratio"))
    nf = int(r.get("nonfinite_scores") or 0)
    sem = int((not present) or sim < 0.4 or (ppl is not None and ppl > 5.0)
              or nf > 0) if clean_ok else ""
    cat = int(lit or (sem == 1)) if clean_ok else ""
    preserved = int(not cat and start and sim >= 0.8) if clean_ok else ""
    partial = int(not cat and not preserved) if clean_ok else ""

    r["answer_pos"] = pos if pos is not None else ""
    r["answer_present2"] = int(present)
    r["answer_start2"] = int(start)
    r["literal_garble2"] = lit
    r["semantic_wrong2"] = sem
    r["catastrophic2"] = cat
    r["partial2"] = partial
    r["preserved2"] = preserved
    r["fault_ok2"] = int(start)
    # preserve recomputed similarity for downstream tables
    r["char_sim"] = "%.6f" % sim
    r["token_f1"] = "%.6f" % token_f1(clean_out, out)
    return r


def summary(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        n = len(rs)
        eval_rs = [r for r in rs if r["catastrophic2"] not in ("", None)]
        n_eval = len(eval_rs)
        garbled = sum(int(r["literal_garble2"] or 0) for r in rs)
        wrong = sum(int(r["semantic_wrong2"] or 0) for r in eval_rs)
        cat = sum(int(r["catastrophic2"] or 0) for r in eval_rs)
        part = sum(int(r["partial2"] or 0) for r in eval_rs)
        pres = sum(int(r["preserved2"] or 0) for r in eval_rs)
        sims = [float(r["char_sim"]) for r in rs
                if parse_float(r.get("char_sim")) is not None]
        ppls = [parse_float(r.get("ppl_ratio")) for r in rs]
        ppls = [x for x in ppls if x is not None]
        nf = sum(int(r.get("nonfinite_scores") or 0) for r in rs)
        ci = wilson(cat, n_eval)
        d = {k: key[i] for i, k in enumerate(keys)}
        d.update({
            "n": n, "n_eval": n_eval,
            "garbled_rate": garbled / n if n else "",
            "wrong_rate": wrong / n_eval if n_eval else "",
            "catastrophic_rate": cat / n_eval if n_eval else "",
            "catastrophic_ci_low": ci[0] if n_eval else "",
            "catastrophic_ci_high": ci[1] if n_eval else "",
            "partial_rate": part / n_eval if n_eval else "",
            "preserved_rate": pres / n_eval if n_eval else "",
            "mean_char_sim": statistics.mean(sims) if sims else "",
            "median_ppl_ratio": statistics.median(ppls) if ppls else "",
            "nonfinite_scores": nf,
        })
        out.append(d)
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/llm_highbit_output")
    ap.add_argument("--prompts", default="data/highbit_prompts.json")
    ap.add_argument("--out", default="results/llm_highbit_output")
    args = ap.parse_args()

    prompts = {p["id"]: p for p in json.load(open(args.prompts, encoding="utf-8"))}
    rows = []
    files = sorted(glob.glob(os.path.join(args.data, "runs_*.csv")))
    for path in files:
        if os.path.basename(path) == "runs_all.csv":
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["model_short"] = r.get("model_short") or ""
                r["model_label"] = MODEL_LABEL.get(r["model_short"], r["model_short"])
                rows.append(add_metrics(r, prompts))
    rows.sort(key=lambda r: (MODEL_ORDER.index(r["model_short"])
                             if r["model_short"] in MODEL_ORDER else 99,
                             r["prompt_id"], r["condition"], r["layer_tag"], r["bit"]))

    # audit controls
    clean = [r for r in rows if r["condition"] == "clean"]
    repeats = [r for r in rows if r["condition"] == "clean_repeat"]
    clean_map = {(r["model_short"], r["prompt_id"]): r["output"] for r in clean}
    repeat_mismatch = sum(
        1 for r in repeats
        if clean_map.get((r["model_short"], r["prompt_id"]), None) != r["output"])
    sham = [r for r in rows if r["condition"] == "sham"]
    sham_mismatch = sum(1 for r in sham if r["char_sim"] != "1.000000")
    fault = [r for r in rows if r["condition"] == "fault"]
    bad_bitdiff = sum(1 for r in fault if str(r.get("bit_diff")) != "1")

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "runs_all.csv"), rows)
    write_csv(os.path.join(args.out, "summary_by_condition.csv"),
              summary(rows, ["model_short", "condition", "layer_tag", "bit"]))
    write_csv(os.path.join(args.out, "summary_by_model.csv"),
              summary(fault, ["model_short"]))
    write_csv(os.path.join(args.out, "summary_by_model_bit.csv"),
              summary(fault, ["model_short", "bit"]))
    write_csv(os.path.join(args.out, "summary_by_model_layer.csv"),
              summary(fault, ["model_short", "layer_tag"]))
    write_csv(os.path.join(args.out, "summary_by_model_layer_bit.csv"),
              summary(fault, ["model_short", "layer_tag", "bit"]))
    write_csv(os.path.join(args.out, "summary_by_category.csv"),
              summary(fault, ["model_short", "category"]))

    # outcome composition for bits 28/29 (the consistently damaging bit pair)
    comp = []
    for ms in MODEL_ORDER:
        rs = [r for r in fault if r["model_short"] == ms
              and int(r["bit"]) in (28, 29)
              and r["catastrophic2"] not in ("", None)]
        n = len(rs)
        g = sum(int(r["literal_garble2"] or 0) for r in rs)
        w = sum(int((not int(r["literal_garble2"] or 0)) and
                    int(r["semantic_wrong2"] or 0)) for r in rs)
        p = sum(int((not int(r["literal_garble2"] or 0)) and
                    not int(r["semantic_wrong2"] or 0) and
                    int(r["partial2"] or 0)) for r in rs)
        v = n - g - w - p
        comp.append({
            "model_short": ms, "model_label": MODEL_LABEL[ms],
            "n_eval": n, "garbled": g, "wrong": w, "partial": p,
            "preserved": v,
            "garbled_rate": g / n if n else "",
            "wrong_rate": w / n if n else "",
            "partial_rate": p / n if n else "",
            "preserved_rate": v / n if n else "",
        })
    write_csv(os.path.join(args.out, "outcome_by_model_2829.csv"), comp)

    meta = {
        "n_rows": len(rows), "n_fault": len(fault), "n_sham": len(sham),
        "n_clean": len(clean), "n_clean_repeat": len(repeats),
        "clean_repeat_mismatches": repeat_mismatch,
        "sham_char_sim_mismatches": sham_mismatch,
        "fault_bitdiff_not_one": bad_bitdiff,
        "outputs": sorted(os.path.basename(p) for p in files),
    }
    json.dump(meta, open(os.path.join(args.out, "analysis_meta.json"), "w",
                         encoding="utf-8"), indent=2)

    # compact human-readable examples
    lines = ["# High-bit output examples", ""]
    for ms in MODEL_ORDER:
        rs = [r for r in fault if r["model_short"] == ms
              and r["catastrophic2"] not in ("", None)]
        if not rs:
            continue
        lines.append("## " + MODEL_LABEL[ms])
        for label, predicate in [
            ("garbled", lambda r: int(r["literal_garble2"] or 0) == 1),
            ("wrong", lambda r: int(r["semantic_wrong2"] or 0) == 1
             and int(r["literal_garble2"] or 0) == 0),
            ("preserved", lambda r: int(r["preserved2"] or 0) == 1),
        ]:
            pick = next((r for r in rs if predicate(r)), None)
            if not pick:
                continue
            lines += [
                "",
                "**%s** — %s, layer %s, bit %s, char_sim %.3f" % (
                    label, pick["prompt_id"], pick["layer_tag"], pick["bit"],
                    float(pick["char_sim"])),
                "",
                "- prompt: `%s`" % pick["prompt"].replace("\n", "\\n"),
                "- clean: `%s`" % pick["clean_output"].replace("\n", "\\n")[:220],
                "- fault: `%s`" % pick["output"].replace("\n", "\\n")[:220],
            ]
        lines.append("")
    open(os.path.join(args.out, "examples.md"), "w", encoding="utf-8").write("\n".join(lines))

    print(json.dumps(meta, indent=2))
    print("\nmodel summary")
    for r in csv.DictReader(open(os.path.join(args.out, "summary_by_model.csv"), encoding="utf-8")):
        print(r["model_short"], "n_eval", r["n_eval"],
              "cat", round(float(r["catastrophic_rate"]), 3),
              "wrong", round(float(r["wrong_rate"]), 3),
              "garb", round(float(r["garbled_rate"]), 3))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistics helpers for SDC studies (stdlib only).

  wilson_ci(k, n)          confidence interval for a proportion
  mcnemar(b, c)            exact paired test (discordant counts)
  two_proportion_z(...)    unpaired test
  holm(pvalues)            multiple-comparison correction
  required_n(p_hat, ...)   sample size for a target CI half-width
"""

import math


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def two_proportion_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    return (z, 2 * (1 - _phi(abs(z))))


def mcnemar(b, c):
    """Exact two-sided McNemar for discordant counts b (fail->ok) and c (ok->fail)."""
    n = b + c
    if n == 0:
        return (1.0,)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return (min(1.0, 2 * tail),)


def holm(pvalues):
    """Return Holm-adjusted p-values (same order as input)."""
    idx = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvalues[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def required_n(p_hat=0.5, half_width=0.05, z=1.96):
    return int(math.ceil(z * z * p_hat * (1 - p_hat) / (half_width * half_width)))


def compare_inputs(cells_csv, baseline="uniform", axis="bit"):
    """Aggregate cells.csv per (experiment, input) and test each input against
    the baseline input (two-proportion z), Holm-corrected within each
    experiment. Returns a list of dict rows."""
    import csv as _csv
    per = {}
    with open(cells_csv, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("axis", axis) != axis:
                continue
            cell = r.get("cell", "")
            if "|" not in cell:
                continue
            inp = cell.split("|", 1)[0]
            key = (r["experiment_id"], inp)
            b = per.setdefault(key, [0, 0])  # sdc, total
            b[0] += int(r.get("sdc") or 0)
            b[1] += int(r.get("total") or 0)

    out = []
    experiments = sorted({k[0] for k in per})
    for eid in experiments:
        inputs = sorted(i for (e, i) in per if e == eid)
        if baseline not in inputs:
            continue
        bk, bn = per[(eid, baseline)]
        others = [i for i in inputs if i != baseline]
        pvals, stats = [], []
        for i in others:
            k, n = per[(eid, i)]
            z, p = two_proportion_z(k, n, bk, bn)
            stats.append((i, k, n, z, p))
            pvals.append(p)
        adj = holm(pvals) if pvals else []
        for (i, k, n, z, p), pa in zip(stats, adj):
            lo, hi = wilson_ci(k, n)
            out.append({
                "experiment_id": eid, "input": i, "baseline": baseline,
                "sdc": k, "total": n, "sdc_rate": (k / n) if n else 0.0,
                "ci_low": lo, "ci_high": hi, "z": z, "p": p, "p_holm": pa,
            })
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--cells":
        rows = compare_inputs(sys.argv[2],
                              sys.argv[3] if len(sys.argv) > 3 else "uniform",
                              sys.argv[4] if len(sys.argv) > 4 else "bit")
        print("%-34s %-14s %5s %6s %8s %8s" %
              ("experiment", "input", "sdc", "n", "rate", "p_holm"))
        for r in rows:
            print("%-34s %-14s %5d %6d %8.3f %8.4g" %
                  (r["experiment_id"], r["input"], r["sdc"], r["total"],
                   r["sdc_rate"], r["p_holm"]))
    elif len(sys.argv) >= 3:
        k, n = int(sys.argv[1]), int(sys.argv[2])
        lo, hi = wilson_ci(k, n)
        print("p=%.4f  95%% CI=[%.4f, %.4f]" % (k / n, lo, hi))
    else:
        print("usage: stats.py <k> <n>")
        print("       stats.py --cells <cells.csv> [baseline_input]")

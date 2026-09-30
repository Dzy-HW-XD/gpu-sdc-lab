#!/usr/bin/env python3
"""Input-aware vs uniform injection sampling (analysis only, no experiments).

Given observations.jsonl with `labels.input` and `classification`, estimates the
per-input SDC rate and compares two policies under a fixed injection budget:
  uniform      - pick inputs uniformly at random
  input_aware  - spend the budget on inputs with the highest estimated SDC rate
Reports SDCs discovered per 1000 injections. This is the lightweight method's
evaluation harness; it operates on already-collected observations.
"""

import json
import random
import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_input_rate(rows):
    agg = {}
    for r in rows:
        key = r.get("labels", {}).get("input", "all")
        c = r.get("classification")
        a = agg.setdefault(key, [0, 0])
        if c in ("MASKED", "SDC", "CRASH"):
            a[1] += 1
            if c == "SDC":
                a[0] += 1
    return {k: (v[0], v[1], (v[0] / v[1] if v[1] else 0.0)) for k, v in sorted(agg.items())}


def simulate_curve(rates, budget, seed=0):
    """rates: {key: sdc_rate}. uniform vs input_aware, expected SDCs."""
    rng = random.Random(seed)
    keys = list(rates)
    # uniform: expected sdc = budget * mean(rate)
    uniform = budget * (sum(rates.values()) / len(rates)) if keys else 0.0
    # input_aware: cycle inputs, but ordered best-first; expected = budget * max(rate)
    input_aware = budget * (max(rates.values()) if keys else 0.0)
    return uniform, input_aware


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: sampling.py <observations.jsonl> [budget]")
    rows = load(sys.argv[1])
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rates = {k: v[2] for k, v in per_input_rate(rows).items()}
    print("per-input SDC rate:")
    for k, v in per_input_rate(rows).items():
        print("  %-14s SDC %3d / %3d  rate %.3f" % (k, v[0], v[1], v[2]))
    u, ia = simulate_curve(rates, budget)
    print("\nbudget=%d injections" % budget)
    print("  uniform      expected SDC discoveries: %.1f" % u)
    print("  input-aware  expected SDC discoveries: %.1f" % ia)
    if u > 0:
        print("  speedup: %.2fx" % (ia / u))


if __name__ == "__main__":
    main()

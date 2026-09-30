#!/usr/bin/env python3
"""Input-distribution features for the SDC predictor (stdlib only).

features_file(path) -> dict of interpretable features:
  log2_range, mean, std, zero_frac, exp_mean, exp_std, cancellation,
  effective_bits, near_overflow_frac, sign_flip_frac
"""

import array
import math
import sys

FLT_MAX = 3.4028234663852886e38


def features_array(a):
    n = len(a)
    if n == 0:
        return {}
    s = 0.0
    s2 = 0.0
    abs_max = 0.0
    abs_min_pos = float("inf")
    zeros = 0
    near_ovf = 0
    signs = 0
    exp_sum = 0.0
    exp2_sum = 0.0
    prev_sign = None
    for v in a:
        x = float(v)
        s += x
        s2 += x * x
        ax = abs(x)
        if ax > abs_max:
            abs_max = ax
        if 0.0 < ax < abs_min_pos:
            abs_min_pos = ax
        if ax == 0.0:
            zeros += 1
        if ax > 0.9 * FLT_MAX:
            near_ovf += 1
        if x != 0.0:
            e = math.frexp(ax)[1] - 1
            exp_sum += e
            exp2_sum += e * e
        sg = 1 if x > 0 else (-1 if x < 0 else 0)
        if prev_sign is not None and sg != 0 and prev_sign != sg:
            signs += 1
        if sg != 0:
            prev_sign = sg
    mean = s / n
    var = max(0.0, s2 / n - mean * mean)
    total_abs = sum(abs(float(v)) for v in a)
    nz = n - zeros
    exp_mean = exp_sum / nz if nz else 0.0
    exp_std = math.sqrt(max(0.0, exp2_sum / nz - exp_mean * exp_mean)) if nz else 0.0
    log2_range = (math.log2(abs_max) - math.log2(abs_min_pos)) if (abs_max > 0 and abs_min_pos < float("inf")) else 0.0
    return {
        "log2_range": log2_range,
        "mean": mean,
        "std": math.sqrt(var),
        "zero_frac": zeros / n,
        "exp_mean": exp_mean,
        "exp_std": exp_std,
        "cancellation": abs(s) / total_abs if total_abs > 0 else 0.0,
        "effective_bits": math.log2(abs_max / abs_min_pos) if abs_min_pos < float("inf") and abs_min_pos > 0 else 0.0,
        "near_overflow_frac": near_ovf / n,
        "sign_flip_frac": signs / n,
    }


def features_file(path):
    import os
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return features_array(a)


if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        raise SystemExit("usage: features.py <tensor.bin>")
    print(json.dumps(features_file(sys.argv[1]), indent=1))

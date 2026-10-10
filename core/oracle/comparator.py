"""Output comparison oracle (pure stdlib).

Computes the numerical-distance metrics required by the SDC lab:
  max_abs_error, mean_abs_error, relative_l2_error,
  corrupted_element_count, corrupted_element_ratio, nan_count, inf_count.

Magnitude metrics (max/mean/relative-L2) are computed over **finite pairs
only** (both golden and faulty element finite). Elements where the faulty value
is NaN/Inf are counted as corrupted and tallied separately (nan_count/inf_count);
elements where the *golden* value is non-finite (possible when the input itself
overflows, e.g. an "extreme" distribution) are tallied as golden_nan_count /
golden_inf_count. `finite_pairs` tells the analysis how many elements actually
contributed to the magnitude metrics.

A bitwise hash fast-path avoids element-wise work for the common MASKED case.
"""

import array
import math
import os

from core import util


def read_floats(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def _empty_metrics(total):
    return {
        "total_elements": total,
        "fault_elements": total,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "relative_l2_error": 0.0,
        "corrupted_elements": 0,
        "corrupted_element_ratio": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "golden_nan_count": 0,
        "golden_inf_count": 0,
        "finite_pairs": total,
        "length_mismatch": False,
        "output_missing": False,
        "bitwise_equal": True,
    }


AGG_KEYS = ("total_elements", "fault_elements", "max_abs_error", "mean_abs_error",
            "relative_l2_error", "corrupted_elements", "corrupted_element_ratio",
            "nan_count", "inf_count", "golden_nan_count", "golden_inf_count",
            "finite_pairs", "length_mismatch", "output_missing", "bitwise_equal")


def compare_outputs(golden_prefix, fault_prefix, specs, abs_tol=0.0, rel_tol=0.0):
    """Compare every output in `specs` (each {"name","suffix","format"[,tol]}).

    Returns aggregate metrics plus a per-output breakdown.
    """
    per = {}
    agg = {
        "total_elements": 0, "fault_elements": 0, "max_abs_error": 0.0,
        "sum_abs": 0.0, "corrupted_elements": 0, "nan_count": 0,
        "inf_count": 0, "golden_nan_count": 0, "golden_inf_count": 0,
        "finite_pairs": 0, "length_mismatch": False, "output_missing": False,
        "bitwise_equal": True,
    }
    for s in specs:
        name = s["name"]
        gpath = golden_prefix + s["suffix"]
        fpath = fault_prefix + s["suffix"]
        tol = s.get("tol") or {}
        m = compare(gpath, fpath, tol.get("abs", abs_tol), tol.get("rel", rel_tol))
        per[name] = m
        agg["total_elements"] += m["total_elements"]
        agg["fault_elements"] += m["fault_elements"]
        agg["max_abs_error"] = max(agg["max_abs_error"], m["max_abs_error"])
        agg["corrupted_elements"] += m["corrupted_elements"]
        agg["nan_count"] += m["nan_count"]
        agg["inf_count"] += m["inf_count"]
        agg["golden_nan_count"] += m.get("golden_nan_count", 0)
        agg["golden_inf_count"] += m.get("golden_inf_count", 0)
        agg["finite_pairs"] += m.get("finite_pairs", 0)
        agg["sum_abs"] += m["mean_abs_error"] * m.get("finite_pairs", 0)
        agg["length_mismatch"] = agg["length_mismatch"] or m["length_mismatch"]
        agg["output_missing"] = agg["output_missing"] or m["output_missing"]
        agg["bitwise_equal"] = agg["bitwise_equal"] and m["bitwise_equal"]

    total = agg["total_elements"]
    fp = agg["finite_pairs"]
    out = {
        "total_elements": total,
        "fault_elements": agg["fault_elements"],
        "max_abs_error": agg["max_abs_error"],
        "mean_abs_error": (agg["sum_abs"] / fp) if fp else 0.0,
        "relative_l2_error": _agg_rel_l2(per),
        "corrupted_elements": agg["corrupted_elements"],
        "corrupted_element_ratio": (float(agg["corrupted_elements"]) / total) if total else 0.0,
        "nan_count": agg["nan_count"],
        "inf_count": agg["inf_count"],
        "golden_nan_count": agg["golden_nan_count"],
        "golden_inf_count": agg["golden_inf_count"],
        "finite_pairs": fp,
        "length_mismatch": agg["length_mismatch"],
        "output_missing": agg["output_missing"],
        "bitwise_equal": agg["bitwise_equal"],
        "per_output": per,
    }
    return out


def _agg_rel_l2(per):
    vals = [m["relative_l2_error"] for m in per.values()
            if m.get("total_elements") and m["relative_l2_error"] == m["relative_l2_error"]]
    return max(vals) if vals else 0.0


def compare(golden_bin, fault_bin, abs_tol=0.0, rel_tol=0.0):
    if not os.path.exists(fault_bin):
        m = _empty_metrics(0)
        m.update({"output_missing": True, "bitwise_equal": False,
                  "corrupted_elements": 1, "corrupted_element_ratio": 1.0,
                  "finite_pairs": 0})
        return m

    golden = read_floats(golden_bin)
    total = len(golden)

    if util.sha256_file(golden_bin) == util.sha256_file(fault_bin):
        return _empty_metrics(total)

    fault = read_floats(fault_bin)

    length_mismatch = len(fault) != total
    n = min(len(fault), total)

    max_abs = 0.0
    sum_abs = 0.0
    sumsq_d = 0.0
    sumsq_g = 0.0
    corrupted = 0
    nan_count = 0
    inf_count = 0
    golden_nan = 0
    golden_inf = 0
    finite_n = 0

    for i in range(n):
        gv = golden[i]
        fv = fault[i]
        f_finite = (fv == fv) and not (fv == math.inf or fv == -math.inf)
        if not f_finite:
            if fv != fv:
                nan_count += 1
            else:
                inf_count += 1
            corrupted += 1
            continue
        g_finite = (gv == gv) and not (gv == math.inf or gv == -math.inf)
        if not g_finite:
            if gv != gv:
                golden_nan += 1
            else:
                golden_inf += 1
            corrupted += 1
            continue
        finite_n += 1
        d = fv - gv
        ad = d if d >= 0.0 else -d
        sum_abs += ad
        if ad > max_abs:
            max_abs = ad
        sumsq_d += d * d
        sumsq_g += gv * gv
        if ad > (abs_tol + rel_tol * (gv if gv >= 0 else -gv)):
            corrupted += 1

    if length_mismatch:
        corrupted += abs(len(fault) - total)

    mean_abs = sum_abs / finite_n if finite_n else 0.0
    if sumsq_g > 0.0:
        rel_l2 = math.sqrt(sumsq_d / sumsq_g)
    else:
        rel_l2 = math.sqrt(sumsq_d)

    return {
        "total_elements": total,
        "fault_elements": len(fault),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_l2,
        "corrupted_elements": corrupted,
        "corrupted_element_ratio": (float(corrupted) / total) if total else 0.0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "golden_nan_count": golden_nan,
        "golden_inf_count": golden_inf,
        "finite_pairs": finite_n,
        "length_mismatch": length_mismatch,
        "output_missing": False,
        "bitwise_equal": False,
    }

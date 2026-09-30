"""Experiment specifications: case generation and aggregation.

Fault targets are chosen from the instruction taxonomy tree
(`core/taxonomy.py`). `groups`/`group` in an experiment may name a leaf, a path
(e.g. "memory/atomic"), or a whole category; a subtree expands to its leaves.
"""

from core import inputgen
from core import taxonomy

GEMM_DEFAULTS = {
    "M": 1024,
    "N": 1024,
    "K": 1024,
    "dtype": "fp32",
    "seed": 1234,
    "pattern": "normal",
    "block": [16, 16],
}

# kind -> dimension used for the per-dimension summary CSV
KIND_DIMENSION = {
    "uniform": "target",
    "fault": "target",
    "fault_position": "position",
    "fault_type": "fault_type",
    "fault_bit": "bit",
    "gemm_size": "size",
    "data_pattern": "pattern",
    "input_sensitivity": "input",
    "cross_workload": "workload",
}


def make_input_spec(workload, family, seed, real_path=None):
    """Build an input_spec dict for a workload/family (input names per workload)."""
    def g(offset):
        d = {"gen": family, "seed": seed + offset}
        if family == "real":
            if not real_path:
                raise ValueError("input family 'real' requires 'real_path'")
            d["path"] = real_path
        return d

    if workload == "gemm":
        return {"A": g(0), "B": g(1)}
    if workload in ("reduce", "softmax", "attention"):
        return {"X": g(0)}
    if workload == "conv2d":
        return {"X": g(0), "Wt": g(1)}
    if workload == "mlp":
        return {"X": g(0), "W1": g(1), "W2": g(2)}
    return {"X": g(0)}

CLASSES = ("MASKED", "SDC", "CRASH", "NOT_INJECTED", "NOT_TARGETED")


def base_params(expt):
    wl = expt.get("workload", "gemm")
    p = dict(GEMM_DEFAULTS) if wl == "gemm" else {}
    p.update(expt.get("workload_params") or {})
    return p


def _leaf_picker(expt):
    nodes = expt.get("groups")
    if not nodes and expt.get("group"):
        nodes = [expt["group"]]
    if not nodes:
        nodes = ["arithmetic/fp32", "memory/load", "meta/gp"]
    leaves = [lf for n in nodes for lf in taxonomy.resolve(n)]
    leaves = [lf for lf in leaves if lf["injectable"]]
    if not leaves:
        raise ValueError("no injectable targets for groups=%r" % (nodes,))
    state = {"i": 0}

    def pick():
        leaf = leaves[state["i"] % len(leaves)]
        state["i"] += 1
        return leaf

    return pick


def build_cases(expt):
    kind = expt.get("kind", "uniform")
    ftype = expt.get("fault_type", "bitflip")
    cases = []
    if kind == "baseline":
        return cases

    pick_leaf = _leaf_picker(expt)

    def add(labels, wp, ft, bit, rng_range):
        leaf = pick_leaf()
        labels = dict(labels)
        labels.setdefault("group", leaf["group"])
        labels.setdefault("target", leaf["path"])
        labels.setdefault("category", leaf["category"])
        case = {
            "labels": labels,
            "workload_params": wp,
            "fault_type": ft,
            "group": leaf["group"],
            "opcodes": leaf["opcodes"],
            "targeting": leaf["targeting"],
            "target": leaf["path"],
            "bit_index": bit,
            "inst_range": rng_range,
        }
        if labels.get("workload"):
            case["workload"] = labels["workload"]
        cases.append(case)

    if kind in ("uniform", "fault"):
        n = int(expt.get("num_injections", 100))
        wp = base_params(expt)
        for i in range(n):
            add({"case": i}, wp, ftype, None, (0.0, 1.0))

    elif kind == "fault_position":
        positions = int(expt.get("positions", 5))
        per = expt.get("injections_per_position")
        if per is None:
            per = max(1, int(expt.get("num_injections", 100)) // positions)
        wp = base_params(expt)
        for p in range(positions):
            lo, hi = p / float(positions), (p + 1) / float(positions)
            for _ in range(int(per)):
                add({"position": "P%d" % p}, wp, ftype, None, (lo, hi))

    elif kind == "fault_type":
        fts = expt.get("fault_types", ["bitflip", "twobit", "random", "zero"])
        per = int(expt.get("injections_per_type", 25))
        wp = base_params(expt)
        for ft in fts:
            for _ in range(per):
                add({"fault_type": ft}, wp, ft, None, (0.0, 1.0))

    elif kind == "fault_bit":
        bits = expt.get("bits", list(range(32)))
        per = int(expt.get("injections_per_bit", 3))
        wp = base_params(expt)
        for b in bits:
            for _ in range(per):
                add({"bit": int(b)}, wp, "bitflip", int(b), (0.0, 1.0))

    elif kind == "gemm_size":
        sizes = expt.get("sizes", [[256, 256, 256], [512, 512, 512],
                                   [1024, 1024, 1024], [2048, 2048, 2048]])
        per = int(expt.get("injections_per_size", 25))
        for (M, N, K) in sizes:
            wp = base_params(expt)
            wp.update({"M": int(M), "N": int(N), "K": int(K)})
            for _ in range(per):
                add({"size": "%dx%dx%d" % (M, N, K)}, wp, ftype, None, (0.0, 1.0))

    elif kind == "data_pattern":
        pats = expt.get("patterns", ["normal", "small", "large", "near_zero", "mixed"])
        per = int(expt.get("injections_per_pattern", 20))
        for pat in pats:
            wp = base_params(expt)
            wp["pattern"] = pat
            for _ in range(per):
                add({"pattern": pat}, wp, ftype, None, (0.0, 1.0))

    elif kind == "input_sensitivity":
        wl = expt.get("workload", "gemm")
        fams = expt.get("inputs", inputgen.FAMILIES)
        per = int(expt.get("injections_per_input", 100))
        seed0 = int(expt.get("input_seed", 1234))
        base = base_params(expt)
        for fam in fams:
            wp = dict(base)
            wp["input_spec"] = make_input_spec(wl, fam, seed0, expt.get("real_path"))
            for _ in range(per):
                add({"input": fam, "workload": wl}, wp, ftype, None, (0.0, 1.0))

    elif kind == "cross_workload":
        wls = expt.get("workloads",
                       ["gemm", "reduce", "conv2d", "softmax", "attention", "mlp"])
        per = int(expt.get("injections_per_workload", 100))
        fam = expt.get("input_family", "normal")
        wp_map = expt.get("workload_params", {}) or {}
        seed0 = int(expt.get("input_seed", 1234))
        for w in wls:
            base = dict(wp_map.get(w, {}))
            base["input_spec"] = make_input_spec(w, fam, seed0, expt.get("real_path"))
            for _ in range(per):
                add({"workload": w, "input": fam}, base, ftype, None, (0.0, 1.0))

    else:
        raise ValueError("unknown experiment kind '%s'" % kind)

    return cases


def _bucket_add(b, classification):
    key = {"MASKED": "masked", "SDC": "sdc", "CRASH": "crash",
           "NOT_TARGETED": "not_targeted"}.get(classification, "not_injected")
    b[key] += 1


def summarize_by(records, dim):
    """Aggregate records by records[i]['labels'][dim]."""
    buckets = {}
    for r in records:
        key = r.get("labels", {}).get(dim) if dim else "all"
        if key is None:
            key = "all"
        b = buckets.setdefault(key, {"masked": 0, "sdc": 0, "crash": 0,
                                     "not_injected": 0, "not_targeted": 0})
        _bucket_add(b, r.get("classification", "NOT_INJECTED"))

    rows = []
    for key in sorted(buckets.keys(), key=str):
        b = buckets[key]
        total = b["masked"] + b["sdc"] + b["crash"]
        rows.append({
            dim or "case": key,
            "total": total,
            "masked": b["masked"],
            "sdc": b["sdc"],
            "crash": b["crash"],
            "not_targeted": b["not_targeted"],
            "not_injected": b["not_injected"],
            "sdc_rate": (float(b["sdc"]) / total) if total else 0.0,
        })
    return rows


def overall_counts(records):
    counts = {c: 0 for c in CLASSES}
    for r in records:
        c = r.get("classification", "NOT_INJECTED")
        counts[c] = counts.get(c, 0) + 1
    counts["total"] = len(records)
    counts["sdc_rate"] = (
        float(counts["SDC"]) /
        max(1, counts["MASKED"] + counts["SDC"] + counts["CRASH"])
    )
    return counts


def to_csv(rows, columns):
    lines = [",".join(columns)]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                v = "%.6f" % v
            vals.append(str(v))
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"

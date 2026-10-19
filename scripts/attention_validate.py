#!/usr/bin/env python3
"""Structural validation for attention_forward dumps (stdlib only).

Checks, per dumped run prefix:
  * tensor sizes match S,D,H,dh
  * P rows (per head) sum to 1 and are non-negative
  * causal mask: P[i,j] == 0 for j > i
  * Ocat[:, h*dh:(h+1)*dh] == O[h]
  * Q/K/V/S/P/O/Ocat/Y are finite

Usage: attention_validate.py <run_prefix>   (e.g. results/.../raw/000/out)
"""
import array
import json
import math
import os
import sys


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, os.path.getsize(path) // a.itemsize)
    return a


def finite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def main():
    prefix = sys.argv[1]
    with open(prefix + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    S, D, H, dh = int(meta["S"]), int(meta["D"]), int(meta["H"]), int(meta["dh"])

    fails = []

    def want(name, n):
        p = prefix + name
        if not os.path.exists(p):
            fails.append("missing %s" % p)
            return None
        a = load_f32(p)
        if len(a) != n:
            fails.append("%s: %d elems, expected %d" % (name, len(a), n))
        return a

    Q = want(".Q.bin", S * D)
    K = want(".K.bin", S * D)
    V = want(".V.bin", S * D)
    Sc = want(".S.bin", H * S * S)
    P = want(".P.bin", H * S * S)
    O = want(".O.bin", H * S * dh)
    Ocat = want(".Ocat.bin", S * D)
    Y = want(".Y.bin", S * D)

    for name, a in (("Q", Q), ("K", K), ("V", V), ("S", Sc), ("P", P),
                    ("O", O), ("Ocat", Ocat), ("Y", Y)):
        if a is not None and any(not finite(v) for v in a):
            fails.append("%s contains non-finite values" % name)

    max_sum_err = 0.0
    mask_bad = 0
    neg = 0
    if P is not None and len(P) == H * S * S:
        for h in range(H):
            base = h * S * S
            for i in range(S):
                s = 0.0
                for j in range(S):
                    v = P[base + i * S + j]
                    if j > i and v != 0.0:
                        mask_bad += 1
                    if v < 0.0:
                        neg += 1
                    if finite(v):
                        s += v
                max_sum_err = max(max_sum_err, abs(s - 1.0))
    if max_sum_err > 1e-5:
        fails.append("softmax row-sum error %.3e > 1e-5" % max_sum_err)
    if mask_bad:
        fails.append("causal mask violated in %d entries" % mask_bad)
    if neg:
        fails.append("%d negative probabilities" % neg)

    max_cat = 0.0
    cat_bad = 0
    if O is not None and Ocat is not None and len(O) == H * S * dh and len(Ocat) == S * D:
        for i in range(S):
            for h in range(H):
                for j in range(dh):
                    d = abs(O[(h * S + i) * dh + j] - Ocat[i * D + h * dh + j])
                    max_cat = max(max_cat, d)
                    if d > 1e-6:
                        cat_bad += 1
    if cat_bad:
        fails.append("concat mismatch in %d entries (max %.3e)" % (cat_bad, max_cat))

    print("attention_validate: S=%d D=%d H=%d dh=%d" % (S, D, H, dh))
    print("  softmax max row-sum error : %.3e" % max_sum_err)
    print("  causal violations         : %d" % mask_bad)
    print("  negative probabilities    : %d" % neg)
    print("  concat max abs mismatch   : %.3e (%d)" % (max_cat, cat_bad))
    print("  finite tensors            : %s" % ("yes" if not any(
        "non-finite" in x for x in fails) else "no"))
    if fails:
        print("RESULT: FAIL")
        for x in fails:
            print("  -", x)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

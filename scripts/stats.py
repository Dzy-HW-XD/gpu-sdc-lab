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


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        k, n = int(sys.argv[1]), int(sys.argv[2])
        lo, hi = wilson_ci(k, n)
        print("p=%.4f  95%% CI=[%.4f, %.4f]" % (k / n, lo, hi))
    else:
        print("usage: stats.py <k> <n>")

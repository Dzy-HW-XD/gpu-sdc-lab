#!/usr/bin/env python3
"""Input-feature -> SDC-rate predictor (stdlib only, small ridge regression).

Usage:
  predictor.py dataset.csv [--target NAME] [--lam 1.0]

The CSV has a header row; all numeric columns except the target are features.
Reports in-sample R^2/RMSE and leave-one-row-out cross-validation. This is a
building block for the input-aware sampling method; it does not run injections.
"""

import csv
import math
import sys


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_csv(path, target):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty dataset")
    cols = list(rows[0].keys())
    if target not in cols:
        target = cols[-1]
    feats = [c for c in cols if c != target]
    X, y = [], []
    for r in rows:
        vals = [_to_float(r[c]) for c in feats]
        t = _to_float(r[target])
        if t is None or any(v is None for v in vals):
            continue
        X.append(vals)
        y.append(t)
    return feats, X, y


def standardize(X):
    d = len(X[0])
    mean = [sum(row[j] for row in X) / len(X) for j in range(d)]
    std = []
    for j in range(d):
        v = sum((row[j] - mean[j]) ** 2 for row in X) / len(X)
        std.append(math.sqrt(v) if v > 0 else 1.0)
    Xs = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    return Xs, mean, std


def _solve(A, b):
    """Gaussian elimination with partial pivoting (A: n x n)."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            M[col][col] = 1e-12
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def ridge_fit(X, y, lam=1.0):
    """Return weights for [bias, features...] on standardized X."""
    n = len(X)
    d = len(X[0])
    Xb = [[1.0] + row for row in X]
    dim = d + 1
    XtX = [[sum(Xb[i][a] * Xb[i][b] for i in range(n)) for b in range(dim)] for a in range(dim)]
    Xty = [sum(Xb[i][a] * y[i] for i in range(n)) for a in range(dim)]
    for a in range(1, dim):          # do not regularize the bias
        XtX[a][a] += lam
    return _solve(XtX, Xty)


def predict_one(w, x, mean, std):
    xs = [(x[j] - mean[j]) / std[j] for j in range(len(x))]
    return w[0] + sum(w[j + 1] * xs[j] for j in range(len(xs)))


def r2(y, yhat):
    m = sum(y) / len(y)
    ss_res = sum((y[i] - yhat[i]) ** 2 for i in range(len(y)))
    ss_tot = sum((y[i] - m) ** 2 for i in range(len(y)))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def rmse(y, yhat):
    return math.sqrt(sum((y[i] - yhat[i]) ** 2 for i in range(len(y))) / len(y))


def loo_cv(X, y, lam=1.0):
    preds = []
    for k in range(len(X)):
        Xt = X[:k] + X[k + 1:]
        yt = y[:k] + y[k + 1:]
        Xs, mean, std = standardize(Xt)
        w = ridge_fit(Xs, yt, lam)
        preds.append(predict_one(w, X[k], mean, std))
    return preds


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    path = args[0]
    target = "sdc_rate"
    lam = 1.0
    for i, a in enumerate(args):
        if a == "--target" and i + 1 < len(args):
            target = args[i + 1]
        if a == "--lam" and i + 1 < len(args):
            lam = float(args[i + 1])
    feats, X, y = load_csv(path, target)
    Xs, mean, std = standardize(X)
    w = ridge_fit(Xs, y, lam)
    yhat = [predict_one(w, X[i], mean, std) for i in range(len(X))]
    loo = loo_cv(X, y, lam)
    print("features:", ", ".join(feats))
    print("weights (standardized):", ["%.4f" % v for v in w])
    print("in-sample R^2=%.4f RMSE=%.4f" % (r2(y, yhat), rmse(y, yhat)))
    print("LOO      R^2=%.4f RMSE=%.4f" % (r2(y, loo), rmse(y, loo)))


if __name__ == "__main__":
    main()

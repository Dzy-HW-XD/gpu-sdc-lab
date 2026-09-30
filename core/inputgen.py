"""Input generators for input-aware SDC studies (stdlib only).

A generator turns (n, seed, params) into a deterministic float32 array. The
array is materialised to a raw little-endian .bin file that a workload reads
via `--inputA/--inputB/--input/...`. Reproducible: same spec -> same bytes.

Synthetic families target different error-propagation mechanisms:
  uniform        - control
  normal         - typical
  lognormal      - wide dynamic range (exponent-sensitive)
  sparse         - many zeros (faults often masked)
  cancellation   - large +/- terms that cancel (relative error explodes)
  near_overflow  - values close to FLT_MAX
  correlated     - low-rank / periodic structure
  adversarial    - engineered extreme exponents + cancellation
  real           - load a tensor from a file
"""

import array
import math
import os
import random

FLT_MAX = 3.4028234663852886e38
GENERATORS = {}


def register(name):
    def deco(fn):
        GENERATORS[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------
def _f32(x):
    # clamp/round to float32 so generated files match what the GPU sees
    v = float(x)
    if v != v:
        return 0.0
    if v > FLT_MAX:
        return FLT_MAX
    if v < -FLT_MAX:
        return -FLT_MAX
    return v


@register("uniform")
def _uniform(n, seed, lo=-1.0, hi=1.0):
    r = random.Random(seed)
    return [_f32(r.uniform(lo, hi)) for _ in range(n)]


@register("normal")
def _normal(n, seed, mu=0.0, sigma=1.0):
    r = random.Random(seed)
    return [_f32(r.gauss(mu, sigma)) for _ in range(n)]


@register("lognormal")
def _lognormal(n, seed, mu=0.0, sigma=1.0):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        v = math.exp(r.gauss(mu, sigma))
        out.append(_f32(v if r.random() < 0.5 else -v))
    return out


@register("sparse")
def _sparse(n, seed, zero_frac=0.9, base_sigma=1.0):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(0.0 if r.random() < zero_frac else _f32(r.gauss(0.0, base_sigma)))
    return out


@register("cancellation")
def _cancellation(n, seed, magnitude=1e4, noise=1e-3):
    r = random.Random(seed)
    out = []
    for i in range(n):
        big = magnitude * (1.0 + 0.01 * r.random())
        v = big if (i % 2 == 0) else -big
        # tiny perturbation so the residual is nonzero and sensitive
        v += r.uniform(-noise, noise) * magnitude
        out.append(_f32(v))
    return out


@register("near_overflow")
def _near_overflow(n, seed, frac=0.01, base_sigma=1.0):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        if r.random() < frac:
            out.append(_f32(FLT_MAX * (0.4 + 0.59 * r.random())))
        else:
            out.append(_f32(r.gauss(0.0, base_sigma)))
    return out


@register("correlated")
def _correlated(n, seed, period=64, noise=0.01):
    r = random.Random(seed)
    basis = [r.uniform(-1.0, 1.0) for _ in range(max(1, period))]
    out = []
    for i in range(n):
        out.append(_f32(basis[i % period] + noise * r.gauss(0.0, 1.0)))
    return out


@register("adversarial")
def _adversarial(n, seed):
    """Alternating huge exponent swings plus cancellation, to maximise the
    chance that a register-bit fault survives the reduction into the result."""
    r = random.Random(seed)
    out = []
    for i in range(n):
        e = (i % 61) - 30          # exponents from 2^-30 .. 2^30
        mag = math.ldexp(1.0, e)
        sign = 1.0 if (i % 2 == 0) else -1.0
        out.append(_f32(sign * mag * (1.0 + 0.01 * r.random())))
    return out


@register("real")
def _real(n, seed, path=None):
    if not path:
        raise ValueError("real generator requires 'path'")
    a = array.array("f")
    with open(path, "rb") as f:
        need = os.path.getsize(path) // a.itemsize
        a.fromfile(f, need)
    vals = list(a)
    if n and len(vals) >= n:
        return vals[:n]
    # tile/truncate to n
    out = []
    i = 0
    while len(out) < n:
        out.append(vals[i % len(vals)])
        i += 1
    return [float(x) for x in out[:n]]


# --------------------------------------------------------------------------
FAMILIES = ["uniform", "normal", "lognormal", "sparse", "cancellation",
            "near_overflow", "correlated", "adversarial"]


def generate(spec, n):
    """spec: {"gen": name, "seed": int, ...params} -> array('f', n)."""
    spec = dict(spec or {})
    name = spec.pop("gen", spec.pop("generator", "uniform"))
    seed = int(spec.pop("seed", 1234))
    if name not in GENERATORS:
        raise KeyError("unknown input generator '%s' (have: %s)"
                       % (name, ", ".join(sorted(GENERATORS))))
    vals = GENERATORS[name](n, seed, **spec)
    return array.array("f", vals)


def materialize(spec, n, path):
    arr = generate(spec, n)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        arr.tofile(f)
    return path

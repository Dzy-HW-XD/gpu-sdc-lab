#!/usr/bin/env python3
"""Self-test for attention_metrics.py using synthetic, known diffs (stdlib).

Builds a tiny attention result set in a temp dir, perturbs a few known
elements, and asserts the metrics are exactly the expected values. Also checks
that an identical pair is classified MASKED (not SDC).

Run: python3 scripts/attention_selftest.py
"""
import array
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)

import attention_metrics as M  # noqa: E402


def write_f32(path, values):
    a = array.array("f", values)
    with open(path, "wb") as f:
        a.tofile(f)


def write_run(prefix, S, D, H, dh):
    meta = {"S": S, "D": D, "H": H, "dh": dh, "dtype": "bf16",
            "accumulate": "fp32", "mask": "causal", "seed": 1}
    with open(prefix + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    shapes = {"Q": S * D, "K": S * D, "V": S * D, "S": H * S * S,
              "P": H * S * S, "O": H * S * dh, "Ocat": S * D, "Y": S * D}
    for name, n in shapes.items():
        write_f32(prefix + "." + name + ".bin", [0.0] * n)


def set_el(prefix, tensor, idx, value):
    p = prefix + "." + tensor + ".bin"
    a = array.array("f")
    with open(p, "rb") as f:
        a.fromfile(f, os.path.getsize(p) // 4)
    a[idx] = value
    with open(p, "wb") as f:
        a.tofile(f)


def main():
    S, D, H = 8, 16, 2
    dh = D // H
    tmp = tempfile.mkdtemp(prefix="attn_selftest_")
    try:
        gp = os.path.join(tmp, "golden", "out")
        fp = os.path.join(tmp, "faulty", "out")
        os.makedirs(os.path.dirname(gp))
        os.makedirs(os.path.dirname(fp))
        write_run(gp, S, D, H, dh)
        shutil.copy(gp + ".json", fp + ".json")
        for name in ("Q", "K", "V", "S", "P", "O", "Ocat", "Y"):
            shutil.copy(gp + "." + name + ".bin", fp + "." + name + ".bin")

        # Y[3,5] = 1.0  -> one changed element, max_abs 1.0
        set_el(fp, "Y", 3 * D + 5, 1.0)
        # O head 0, token 3, dim 2 = 0.5
        set_el(fp, "O", (0 * S + 3) * dh + 2, 0.5)
        # P head 1, token 2, key 3 = 0.25
        set_el(fp, "P", (1 * S + 2) * S + 3, 0.25)

        out_json = os.path.join(tmp, "metrics.json")
        rc = M.main(["--golden", gp, "--faulty", fp,
                     "--out-json", out_json, "--experiment-id", "selftest"])
        assert rc == 0, "metrics returned %d" % rc
        with open(out_json, encoding="utf-8") as f:
            r = json.load(f)

        y = r["tensors"]["Y"]
        assert y["changed_elements"] == 1, y["changed_elements"]
        assert abs(y["max_abs_error"] - 1.0) < 1e-6, y["max_abs_error"]
        assert y["support_rows"] == 1 and y["support_cols"] == 1
        assert r["classification"] == "SDC"

        o = r["tensors"]["O"]
        assert o["head_error_fro"][0] > 0.0 and o["head_error_fro"][1] == 0.0, \
            o["head_error_fro"]

        p = r["tensors"]["P"]
        assert p["head_changed"][1] == 1 and p["head_changed"][0] == 0, \
            p["head_changed"]

        # identical pair -> MASKED, zero change
        out2 = os.path.join(tmp, "metrics_zero.json")
        rc = M.main(["--golden", gp, "--faulty", gp, "--out-json", out2])
        assert rc == 0
        with open(out2, encoding="utf-8") as f:
            r0 = json.load(f)
        assert r0["classification"] == "MASKED", r0["classification"]
        assert r0["tensors"]["Y"]["changed_elements"] == 0

        print("SELFTEST: PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

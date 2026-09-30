"""SDC classification (Phase 1: MASKED / SDC / CRASH)."""

MASKED = "MASKED"
SDC = "SDC"
CRASH = "CRASH"
NOT_INJECTED = "NOT_INJECTED"

CORE_CLASSES = (MASKED, SDC, CRASH)


def classify(runtime_obs, comparison, injected, oracle_cfg=None):
    """Return (classification, reason)."""
    if runtime_obs.get("timeout"):
        return CRASH, "timeout"
    if runtime_obs.get("crash"):
        return CRASH, runtime_obs.get("crash_reason", "runtime error")
    if not injected:
        return NOT_INJECTED, "fault did not land on a dynamic instruction"

    if comparison.get("output_missing"):
        return CRASH, "output file missing after run"

    corrupted = comparison.get("corrupted_elements", 0)
    if corrupted == 0:
        return MASKED, "observable output identical to golden"

    nan = comparison.get("nan_count", 0)
    inf = comparison.get("inf_count", 0)
    total = comparison.get("total_elements", 0) or 1
    if (nan + inf) / total >= (oracle_cfg or {}).get("nan_inf_dominant_ratio", 0.5):
        return SDC, "output corrupted (NaN/Inf dominated)"
    return SDC, "output corrupted (silent data corruption)"

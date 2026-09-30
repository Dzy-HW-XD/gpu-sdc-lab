"""Experiment runner: orchestrates workload + injector + observers + oracle.

The runner only uses the abstractions, never concrete NVBitFI internals, so a
different injector / workload / observer can be swapped in. Fault targets come
from the taxonomy tree; opcode-filtered leaves are only counted when the hit
opcode matches, otherwise the run is marked NOT_TARGETED.
"""

import datetime
import os
import random
import sys

from core import util
from core import experiments as ex
from core.injection import get_injector
from core.injection.base import GROUP_IDS, BITFLIP_MODELS, FaultSpec
from core.observer import Observers
from core.oracle.comparator import compare
from core.oracle.classify import classify
from core.workload import get_workload


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def make_spec(case, kern, count, rng):
    lo, hi = case["inst_range"]
    frac = lo + rng.random() * (hi - lo)
    inst_id = int(frac * count)
    if inst_id >= count:
        inst_id = count - 1
    bfm = BITFLIP_MODELS[case["fault_type"]]
    bit = case.get("bit_index")
    if bit is not None and bfm == 0:
        bit_seed = (bit + 0.5) / 32.0
    elif bit is not None and bfm == 1:
        bit_seed = (bit + 0.5) / 31.0
    else:
        bit_seed = rng.random()
    return FaultSpec(
        group=GROUP_IDS[case["group"]],
        kernel_name=kern["name"],
        kernel_count=kern["index"],
        inst_id=inst_id,
        op_id_seed=rng.random(),
        bit_flip_model=bfm,
        bit_id_seed=bit_seed,
        fault_type=case["fault_type"],
        bit_index=bit,
    )


class Runner(object):
    def __init__(self, lab):
        self.lab = lab
        self.observers = Observers(lab)

    # ---- workload / oracle caches --------------------------------------
    def _workload(self, name, params):
        wl = get_workload(name, params, self.lab)
        wl.prepare()
        return wl

    def _golden_for(self, wl):
        sig = util.short_hash(wl.signature())
        cdir = util.ensure_dir(os.path.join(self.lab.results_dir, "_golden", sig))
        gprefix = os.path.join(cdir, "golden")
        first = gprefix + wl.output_specs[0]["suffix"]
        if not os.path.exists(first):
            run = wl.run(gprefix, device=0, cwd=cdir)
            if run["exit_code"] != 0 or not os.path.exists(first):
                raise RuntimeError("golden generation failed: %s\n%s"
                                   % (run.get("stdout"), run.get("stderr")))
            meta = wl.collect_output(gprefix).get("meta")
            util.write_json(os.path.join(cdir, "golden.meta.json"),
                            {"workload": wl.name, "params": wl.params,
                             "meta": meta, "runtime_sec": run["runtime_sec"]})
        return gprefix

    def _profile_for(self, wl):
        sig = util.short_hash(wl.signature())
        cdir = util.ensure_dir(os.path.join(self.lab.results_dir, "_profile", sig))
        jpath = os.path.join(cdir, "profile.json")
        if os.path.exists(jpath):
            return util.read_json(jpath)
        inj = get_injector("nvbitfi", self.lab)
        prof = inj.profile(wl, cdir)
        util.write_json(jpath, prof)
        return prof

    @staticmethod
    def _main_kernel(prof, wl):
        token = getattr(wl, "kernel_token", None) or wl.name
        runs = prof.get("runs", [])
        for r in runs:
            if token in (r.get("name") or ""):
                return r
        if runs:
            return max(runs, key=lambda r: r.get("instrs") or 0)
        return None

    # ---- baseline experiment -------------------------------------------
    def run_baseline(self, expt):
        eid = expt["id"]
        out = util.ensure_dir(os.path.join(self.lab.results_dir, eid))
        raw = util.ensure_dir(os.path.join(out, "raw"))
        wl = self._workload(expt.get("workload", "gemm"), ex.base_params(expt))
        repeats = int(expt.get("repeats", 10))
        abs_tol, rel_tol = self.lab.oracle["abs_tol"], self.lab.oracle["rel_tol"]
        obs_path = os.path.join(out, "observations.jsonl")
        if os.path.exists(obs_path):
            os.remove(obs_path)

        prefixes, records = [], []
        for i in range(repeats):
            rd = util.ensure_dir(os.path.join(raw, "%03d" % i))
            run = wl.run(os.path.join(rd, "out"), device=0, cwd=rd)
            prefixes.append(os.path.join(rd, "out"))
            with open(os.path.join(rd, "stdout.txt"), "w") as f:
                f.write(run["stdout"])
            with open(os.path.join(rd, "stderr.txt"), "w") as f:
                f.write(run["stderr"])

        for i in range(repeats):
            m = self.observers.output.observe_outputs(
                prefixes[0], prefixes[i], wl.output_specs)
            rec = {
                "experiment_id": eid, "seq": i, "timestamp": _now(),
                "kind": "baseline", "workload": wl.name, "params": wl.params,
                "observation": m,
                "classification": "IDENTICAL" if m["bitwise_equal"] else "DIFFERS",
            }
            util.append_jsonl(obs_path, rec)
            records.append(rec)

        varying = [r for r in records if not r["observation"]["bitwise_equal"]]
        dist = {
            "repeats": repeats,
            "bitwise_identical_runs": repeats - len(varying),
            "max_abs_error_max": max([r["observation"]["max_abs_error"] for r in records] or [0.0]),
            "relative_l2_error_max": max([r["observation"]["relative_l2_error"] for r in records] or [0.0]),
            "deterministic": len(varying) == 0,
        }
        util.write_json(os.path.join(out, "baseline_distribution.json"), dist)
        util.write_json(os.path.join(out, "experiment.json"),
                        {"config": expt, "baseline_distribution": dist})
        util.write_json(os.path.join(out, "config.yaml"), expt)
        self._write_report(out, eid, records, dist)
        return dist

    # ---- fault-injection experiment ------------------------------------
    def run_fault(self, expt):
        eid = expt["id"]
        out = util.ensure_dir(os.path.join(self.lab.results_dir, eid))
        raw = util.ensure_dir(os.path.join(out, "raw"))
        inj = get_injector(expt.get("injector", "nvbitfi"), self.lab)
        if not inj.available():
            raise RuntimeError("injector '%s' is not available (missing .so?)" % inj.name)

        wl_name = expt.get("workload", "gemm")
        cases = ex.build_cases(expt)
        rng = random.Random(int(expt.get("seed", 1234)))
        obs_path = os.path.join(out, "observations.jsonl")
        if os.path.exists(obs_path):
            os.remove(obs_path)
        util.write_json(os.path.join(out, "config.yaml"), expt)

        wl_cache = {}
        records = []
        total = len(cases)
        for seq, case in enumerate(cases, start=1):
            case_wl = case.get("workload") or wl_name
            wkey = util.short_hash("%s|%s" % (case_wl, sorted(case["workload_params"].items())))
            if wkey not in wl_cache:
                wl = self._workload(case_wl, case["workload_params"])
                wl_cache[wkey] = (wl, self._golden_for(wl), self._profile_for(wl))
            wl, gbin, prof = wl_cache[wkey]
            kern = self._main_kernel(prof, wl)
            if kern is None:
                raise RuntimeError("no kernel found in profile")
            count = int(kern["groups"].get(case["group"], 0))
            if count <= 0:
                raise RuntimeError("target group '%s' has zero instructions in '%s'"
                                   % (case["group"], kern.get("name")))

            spec = make_spec(case, kern, count, rng)
            rd = util.ensure_dir(os.path.join(raw, "%04d_%s" % (seq, spec.group_name)))
            res = inj.inject(wl, spec, rd, device=0)

            runtime_obs = self.observers.runtime.observe(res["run"])
            comparison = self.observers.output.observe_outputs(
                gbin, res["output"]["prefix"], wl.output_specs)
            gpu = self.observers.gpu.observe()
            meta = res["fault_meta"]
            injected = bool(meta["injected"] or meta["kernel_error"] or runtime_obs["crash"])
            classification, reason = classify(runtime_obs, comparison, injected, self.lab.oracle)
            classification_raw = classification

            opcodes = case.get("opcodes")
            op = meta.get("opcode")
            target_matched = True
            if opcodes is not None:
                target_matched = op in opcodes
                if not target_matched:
                    classification = "NOT_TARGETED"
                    reason = "opcode %s not in target set %s" % (op, opcodes)

            static_site = "%s@%s" % (meta["opcode"] or "?", meta["pc_offset"] or "?")
            labels = dict(case["labels"])
            labels.setdefault("group", case["group"])
            labels.setdefault("fault_type", case["fault_type"])
            labels["position_decile"] = "D%d" % min(9, int(10.0 * spec.inst_id / max(1, count)))

            record = {
                "experiment_id": eid,
                "seq": seq,
                "timestamp": _now(),
                "workload": wl.name,
                "params": wl.params,
                "fault": {
                    "type": spec.fault_type,
                    "model_id": spec.bit_flip_model,
                    "group": spec.group_name,
                    "group_id": spec.group,
                    "target": case.get("target"),
                    "targeting": case.get("targeting"),
                    "target_opcodes": opcodes,
                    "target_matched": target_matched,
                    "kernel": spec.kernel_name,
                    "kernel_count": spec.kernel_count,
                    "inst_id": spec.inst_id,
                    "inst_count_profile": count,
                    "inst_fraction": (float(spec.inst_id) / count) if count else 0.0,
                    "op_id_seed": spec.op_id_seed,
                    "bit_id_seed": spec.bit_id_seed,
                    "bit_index": spec.bit_index,
                    "injected": injected,
                    "static_site": static_site,
                    "register": meta["reg_no"],
                    "mask": meta["mask"],
                    "before_val": meta["before_val"],
                    "after_val": meta["after_val"],
                    "opcode": meta["opcode"],
                    "pc_offset": meta["pc_offset"],
                    "thread": meta["tid"],
                    "block": None,
                    "error_not_injected": bool(meta["error_not_injected"]),
                    "kernel_error": bool(meta["kernel_error"]),
                },
                "observation": comparison,
                "runtime": runtime_obs,
                "gpu": gpu,
                "classification": classification,
                "classification_raw": classification_raw,
                "reason": reason,
                "labels": labels,
            }
            for k in ("M", "N", "K", "dtype", "pattern"):
                if k in wl.params:
                    record[k] = wl.params[k]
            util.append_jsonl(obs_path, record)
            records.append(record)

            with open(os.path.join(rd, "stdout.txt"), "w") as f:
                f.write(res["run"]["stdout"])
            with open(os.path.join(rd, "stderr.txt"), "w") as f:
                f.write(res["run"]["stderr"])
            with open(os.path.join(rd, "injection-log.txt"), "w") as f:
                f.write(res["log"])
            util.write_json(os.path.join(rd, "result.json"), record)

            sys.stdout.write(
                "[%d/%d] %s target=%s site=%s reg=%s inst=%d/%d -> %s (%s)\n"
                % (seq, total, spec.group_name, case.get("target"), static_site,
                   meta["reg_no"], spec.inst_id, count, classification, reason)
            )
            sys.stdout.flush()

        self._finalize(out, eid, expt, records)
        return {"records": records, "counts": ex.overall_counts(records)}

    # ---- outputs --------------------------------------------------------
    def _finalize(self, out, eid, expt, records):
        counts = ex.overall_counts(records)
        dim = expt.get("dimension") or ex.KIND_DIMENSION.get(expt.get("kind", "uniform"))
        dim_rows = ex.summarize_by(records, dim) if dim else None
        cols = [dim or "case", "total", "masked", "sdc", "crash",
                "not_targeted", "not_injected", "sdc_rate"]

        with open(os.path.join(out, "summary.csv"), "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write("total,%d\n" % counts["total"])
            f.write("MASKED,%d\n" % counts["MASKED"])
            f.write("SDC,%d\n" % counts["SDC"])
            f.write("CRASH,%d\n" % counts["CRASH"])
            f.write("NOT_TARGETED,%d\n" % counts["NOT_TARGETED"])
            f.write("NOT_INJECTED,%d\n" % counts["NOT_INJECTED"])
            f.write("sdc_rate,%.6f\n" % counts["sdc_rate"])

        if dim_rows is not None:
            fname = {
                "position": "fault_position_summary.csv",
                "fault_type": "fault_type_summary.csv",
                "bit": "fault_bit_summary.csv",
                "size": "gemm_size_summary.csv",
                "pattern": "data_pattern_summary.csv",
                "target": "target_summary.csv",
            }.get(dim, "%s_summary.csv" % dim)
            with open(os.path.join(out, fname), "w", encoding="utf-8") as f:
                f.write(ex.to_csv(dim_rows, cols))

        # tree roll-ups
        extra = {}
        if any("target" in r.get("labels", {}) for r in records):
            rows = ex.summarize_by(records, "target")
            with open(os.path.join(out, "target_summary.csv"), "w", encoding="utf-8") as f:
                f.write(ex.to_csv(rows, ["target", "total", "masked", "sdc", "crash",
                                         "not_targeted", "not_injected", "sdc_rate"]))
            extra["target_summary"] = rows
        if any("category" in r.get("labels", {}) for r in records):
            rows = ex.summarize_by(records, "category")
            with open(os.path.join(out, "category_summary.csv"), "w", encoding="utf-8") as f:
                f.write(ex.to_csv(rows, ["category", "total", "masked", "sdc", "crash",
                                         "not_targeted", "not_injected", "sdc_rate"]))
            extra["category_summary"] = rows

        util.write_json(os.path.join(out, "experiment.json"),
                        {"config": expt, "counts": counts,
                         "dimension": dim, "dimension_rows": dim_rows, "rollups": extra})
        self._write_report(out, eid, records, counts)
        return {"counts": counts, "dimension_rows": dim_rows, "rollups": extra}

    def _write_report(self, out, eid, records, summary):
        lines = ["# Experiment %s" % eid, "", "## Summary", ""]
        if "sdc_rate" in summary:
            inj = summary.get("MASKED", 0) + summary.get("SDC", 0) + summary.get("CRASH", 0)
            lines += ["| classification | count |",
                      "|---|---|",
                      "| Total (targeted+injected) | %d |" % inj,
                      "| MASKED | %d |" % summary.get("MASKED", 0),
                      "| SDC | %d |" % summary.get("SDC", 0),
                      "| CRASH | %d |" % summary.get("CRASH", 0),
                      "| NOT_TARGETED | %d |" % summary.get("NOT_TARGETED", 0),
                      "| NOT_INJECTED | %d |" % summary.get("NOT_INJECTED", 0),
                      "| SDC rate | %.4f |" % summary.get("sdc_rate", 0.0)]
        else:
            lines += ["```", str(summary), "```"]
        lines += ["", "## Raw records", "",
                  "See `observations.jsonl` (one injection per line) and `raw/`.",
                  ""]
        with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

#!/usr/bin/env python3
"""Build a common 200-prompt set with 200/200 clean-strict passes.

The 187 old prompts that both Qwen2.5-0.5B and Qwen2.5-1.5B answered at the
first token are retained.  The 13 failures are replaced by easy capital
prompts, then the full 200 are re-checked with clean-only runs.
"""
import csv
import json
import os
import sys

sys.path.insert(0, "scripts")
import llm_highbit_analyze as ana

OLD = "data/highbit_prompts_200.json"
CLEAN05 = "results/llm_highbit_200_clean/runs_qwen05b.csv"
CLEAN15 = "results/llm_highbit_200_clean/runs_qwen15b.csv"
OUT = "data/highbit_prompts_200_calibrated.json"


def passed(path, prompts):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if r["condition"] == "clean"]
    out = set()
    for r in rows:
        p = prompts[r["prompt_id"]]
        if ana.find_expected(r["clean_output"], p["expected"]) == 0:
            out.add(r["prompt_id"])
    return out


def main():
    old = json.load(open(OLD, encoding="utf-8"))
    prompts = {p["id"]: p for p in old}
    common = sorted(passed(CLEAN05, prompts) & passed(CLEAN15, prompts),
                    key=lambda x: [p["id"] for p in old].index(x))
    assert len(common) == 187, len(common)

    extra_capitals = [
        ("Iceland", "Reykjavik"), ("Hungary", "Budapest"),
        ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
        ("Croatia", "Zagreb"), ("Serbia", "Belgrade"),
        ("Belarus", "Minsk"), ("Lithuania", "Vilnius"),
        ("Latvia", "Riga"), ("Estonia", "Tallinn"),
        ("Saudi Arabia", "Riyadh"), ("Qatar", "Doha"),
        ("Pakistan", "Islamabad"),
    ]
    examples = [("France", "Paris"), ("Japan", "Tokyo")]
    new = []
    for j, (country, capital) in enumerate(extra_capitals, 41):
        ex = [x for x in examples if x[0] != country][:2]
        prefix = "The capital of %s is %s. The capital of %s is %s. " % (
            ex[0][0], ex[0][1], ex[1][0], ex[1][1])
        new.append({
            "id": "fact_%02d" % j,
            "category": "factual",
            "prompt": prefix + "The capital of %s is" % country,
            "expected": [capital],
        })

    out = [prompts[i] for i in common] + new
    assert len(out) == 200, len(out)
    assert len({p["id"] for p in out}) == 200
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", OUT, len(out))
    print("retained", len(common), "new", len(new))


if __name__ == "__main__":
    main()

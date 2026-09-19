"""Repair a probe or pilot run directory that received rows from two
processes (for example, a stalled run that resumed after system sleep while
a --resume process was also running). Rule, fixed in advance and independent
of outcomes: for each probe index i, keep the FIRST-WRITTEN response; drop
the rest. Originals are kept as *.raw.jsonl; the summary JSON is recomputed.

  python -m harness.dedupe --run runs/probes-Y-20260919-004320
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
from typing import Dict, List
from .probes import wilson, binom_pvalue_greater


def dedupe(run_dir: str, phase: str = None) -> Dict:
    phase = phase or ("pilot" if os.path.exists(os.path.join(run_dir, "pilot_responses.jsonl")) else "probe")
    resp_path = os.path.join(run_dir, f"{phase}_responses.jsonl")
    tel_path = os.path.join(run_dir, "telemetry.jsonl")
    resp = [json.loads(l) for l in open(resp_path)]
    tel = [json.loads(l) for l in open(tel_path)]
    assert len(resp) == len(tel), "responses and telemetry rows must align one to one"
    keep, seen, dropped = [], set(), 0
    for r, t in zip(resp, tel):
        if r["i"] in seen:
            dropped += 1
            continue
        seen.add(r["i"])
        keep.append((r, t))
    keep.sort(key=lambda rt: rt[0]["i"])
    if dropped:
        for p in (resp_path, tel_path):
            raw = p.replace(".jsonl", ".raw.jsonl")
            if not os.path.exists(raw):
                shutil.copy(p, raw)
        with open(resp_path, "w") as f:
            for r, _ in keep:
                f.write(json.dumps(r) + "\n")
        with open(tel_path, "w") as f:
            for _, t in keep:
                f.write(json.dumps(t) + "\n")
    rows = [r for r, _ in keep]
    n = len(rows)
    k = sum(bool(r["correct"]) for r in rows)
    status: Dict[str, int] = {}
    per_comp: Dict[str, List[bool]] = {}
    per_family: Dict[str, List[bool]] = {}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
        per_comp.setdefault(r["component"], []).append(bool(r["correct"]))
        per_family.setdefault(r["family"], []).append(bool(r["correct"]))
    summary_path = os.path.join(run_dir, f"{phase}_{'Y' if 'Y' in run_dir.split('-') else ''}.json")
    # find the existing summary file to update
    existing = [f for f in os.listdir(run_dir) if f.startswith(phase + "_") and f.endswith(".json") and "responses" not in f]
    summary_path = os.path.join(run_dir, existing[0]) if existing else summary_path
    old = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    chance = old.get("chance", 0.25)
    lo, hi = wilson(k, n)
    pval = binom_pvalue_greater(k, n, chance)
    wrong = [r for r in rows if r["status"] != "failure" and not r["correct"]]
    mis: Dict[str, int] = {}
    for r in wrong:
        mis[r["misconception"]] = mis.get(r["misconception"], 0) + 1
    out = dict(old)
    out.update({"n": n, "correct": k, "accuracy": k / n, "ci95": [lo, hi], "p_greater_than_chance": pval,
                "at_floor": (k / n <= 0.30) and (pval >= 0.05), "parse_status_counts": status,
                "parse_failure_rate": status.get("failure", 0) / n, "valid": status.get("failure", 0) / n <= 0.05,
                "per_component": {c: sum(v) / len(v) for c, v in per_comp.items()},
                "per_family": {f: sum(v) / len(v) for f, v in per_family.items()},
                "misconceptions_chosen": dict(sorted(mis.items(), key=lambda kv: -kv[1])),
                "dedupe": {"rule": "first-written per probe index", "rows_dropped": dropped, "rows_kept": n}})
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    out = dedupe(a.run)
    print(json.dumps({k: out[k] for k in ("n", "correct", "accuracy", "ci95", "p_greater_than_chance", "at_floor",
                                          "parse_status_counts", "parse_failure_rate", "valid", "per_component",
                                          "misconceptions_chosen", "dedupe")}, indent=1))


if __name__ == "__main__":
    main()

"""Train the GBDT learner model from telemetry (Section IV-E) and compute
calibration diagnostics (Section IV-F): ECE by slice and tau-prime.

  python -m harness.train --runs runs/sweep-* --set demo --calibration platt --out models/gbdt_platt.pkl
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from .content import load_content
from .learner import History, Interaction, features, train_gbdt, ece


def replay(run_dirs: List[str], content) -> Tuple[List[List[float]], List[int], List[str], List[Dict]]:
    """Rebuild (features, label) pairs by replaying each student instance's
    instruction-phase rows in order. Returns X, y, instance ids, and the raw rows."""
    graph = content.graph()
    kc_index = {c.id: i for i, c in enumerate(content.components())}
    fam_index = {f.family_id: i for i, f in enumerate(content.families())}
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for d in run_dirs:
        p = os.path.join(d, "telemetry.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            if r.get("phase") == "instruction":
                groups[(r["run_id"], r["generation"], r["arm"])].append(r)
    X, y, ids, rows = [], [], [], []
    for key, rs in groups.items():
        h = History()
        cond = {"tau": rs[0]["tau"], "calibration_flag": rs[0]["calibration_flag"], "generation": rs[0]["generation"],
                "arm_index": 0 if rs[0]["arm"] == "chain" else 1}
        for r in rs:
            X.append(features(h, r["component"], graph, cond, kc_index, fam_index))
            y.append(1 if r["correct"] else 0)
            ids.append("|".join(map(str, key)))
            rows.append(r)
            h.interactions.append(Interaction(r["component"], r["family"], bool(r["correct"]), r["parse_status"],
                                              r["latency_ms"] or 0, r["context_length"] or 0, len(h.interactions) + 1))
    return X, y, ids, rows


def tau_prime(raw_scores: List[float], calibrated: List[float], tau: float) -> float:
    """tau'_k(tau) = inf{x : g_k(x) >= tau}, equation (11), estimated from the
    fitted map evaluated on the slice's raw scores."""
    pairs = sorted(zip(raw_scores, calibrated))
    for x, g in pairs:
        if g >= tau:
            return x
    return float("nan")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or globs")
    ap.add_argument("--set", required=True)
    ap.add_argument("--calibration", choices=["none", "platt", "isotonic"], default="platt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    dirs = [d for g in a.runs for d in glob.glob(g)]
    content = load_content(a.set)
    X, y, ids, rows = replay(dirs, content)
    if not X:
        raise SystemExit("no instruction-phase telemetry found")
    # split by student instance (Section IV-E): 60% train, 20% calibration, 20% held-out evaluation
    inst = sorted(set(ids)); random.Random(a.seed).shuffle(inst)
    n = len(inst); tr, ca = set(inst[: int(0.6 * n)]), set(inst[int(0.6 * n): int(0.8 * n)])
    sel = lambda S: [i for i, k in enumerate(ids) if k in S]
    itr, ica, iev = sel(tr), sel(ca), sel(set(inst[int(0.8 * n):]))
    model = train_gbdt([X[i] for i in itr], [y[i] for i in itr],
                       [X[i] for i in ica] if a.calibration != "none" else None,
                       [y[i] for i in ica] if a.calibration != "none" else None,
                       None if a.calibration == "none" else a.calibration,
                       {c.id: i for i, c in enumerate(content.components())},
                       {f.family_id: i for i, f in enumerate(content.families())})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    model.save(a.out)
    raw = model.model.predict_proba([X[i] for i in iev])[:, 1]
    labels = [y[i] for i in iev]
    print(json.dumps({"n_train": len(itr), "n_cal": len(ica), "n_eval": len(iev),
                      "ece_raw": ece(list(map(float, raw)), labels), "saved": a.out}, indent=1))


if __name__ == "__main__":
    main()

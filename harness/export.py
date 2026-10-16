"""Transcript and summary export for coding (v2).

  python -m harness.export packet --runs runs --sample analysis/coding_sample.json --out coding
  python -m harness.export rater  --runs runs --sample analysis/coding_sample.json --out coding_rater2 --n 20 --seed 0
  python -m harness.export summaries --runs runs --out analysis

Version 1 built the coding packet with a script that cut each student
summary at 1,200 characters (reviewer 2, item 4f). This module writes every
teacher turn and every summary in full. `rater` draws a stratified subset
(equal numbers per arm and generation) for a second human rater and writes
a sheet that asks only for the codes a non-specialist can give reliably.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import random
from typing import Dict, List

HEADER = ("TRANSCRIPT {tid}\nArm: {arm}   Generation: {gen}   Cell: {cell}\n"
          "Code the TEACHER's explanatory turns below against the codebook. The student's session follows for context.\n")
RULE = "=" * 70


def _teacher_turns(t: Dict, label: str) -> List[Dict]:
    msgs = t[label]["messages"]; out = []
    for i, m in enumerate(msgs):
        if m["role"] == "user" and m["content"].startswith("TEACH:") and i + 1 < len(msgs):
            out.append({"prompt": m["content"], "reply": msgs[i + 1]["content"]})
    return out


def write_transcript(runs_dir: str, entry: Dict, idx: int, out_dir: str, with_student: bool = True) -> str:
    cell, gen, arm = entry["cell"], int(entry["generation"]), entry["arm"]
    t = json.load(open(os.path.join(runs_dir, cell, f"transcript_gen{gen}.json")))
    teacher = f"T{gen-1}" if arm == "chain" else f"R{gen}"
    student = f"S{gen}" if arm == "chain" else f"Sprime{gen}"
    tid = f"{idx:02d}_{arm}_gen{gen}_{cell}"
    lines = [HEADER.format(tid=tid, arm=arm, gen=gen, cell=cell), RULE, "TEACHER TURNS", RULE]
    for turn in _teacher_turns(t, teacher):
        lines += [turn["reply"], "-" * 70]
    if with_student:
        lines += ["", RULE, "STUDENT SESSION (context only; do not code)", RULE]
        for m in t[student]["messages"]:
            lines += [f"[{m['role']}]", m["content"], ""]
    sp = os.path.join(runs_dir, cell, f"sigma_gen{gen}.json")
    if arm == "chain" and os.path.exists(sp):
        lines += ["", RULE, "STUDENT SUMMARY WRITTEN AFTER THIS SESSION (full text; context only)", RULE, json.load(open(sp))["sigma"]]
    path = os.path.join(out_dir, "transcripts", tid + ".txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return tid


def packet(runs_dir: str, sample_path: str, out_dir: str) -> List[str]:
    sample = json.load(open(sample_path)); ids = []
    for i, e in enumerate(sample, 1):
        ids.append(write_transcript(runs_dir, e, i, out_dir))
    with open(os.path.join(out_dir, "coding_sheet.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["transcript_id", "arm", "generation", "rater", "omission_of_component", "distortion_of_rule", "misconception_tags_observed", "notes"])
        for tid, e in zip(ids, sample):
            w.writerow([tid, e["arm"], e["generation"], "", "", "", "", ""])
    return ids


RATER_SHEET_HEADER = ["transcript_id", "arm", "generation", "rater",
                      "A1_mentions_tether", "A2_tether_scaled_by_shorter_cairn", "A3_rest_of_merge_rule_correct",
                      "B1_any_other_rule_wrong", "B2_which_rules_and_how", "notes"]


def write_rater_transcript(runs_dir: str, entry: Dict, idx: int, out_dir: str, part_b: bool) -> str:
    """Teacher turns only, the merge-rule explanation first (Part A), the other
    five after it (Part B, read only for the transcripts marked for it)."""
    cell, gen, arm = entry["cell"], int(entry["generation"]), entry["arm"]
    t = json.load(open(os.path.join(runs_dir, cell, f"transcript_gen{gen}.json")))
    teacher = f"T{gen-1}" if arm == "chain" else f"R{gen}"
    tid = f"{idx:02d}_{arm}_gen{gen}_{cell}"
    turns = _teacher_turns(t, teacher)
    merge = [x for x in turns if "merg" in x["prompt"].lower()]; others = [x for x in turns if "merg" not in x["prompt"].lower()]
    lines = [f"TRANSCRIPT {tid}", f"Part B required: {'YES' if part_b else 'no (Part A only)'}", "",
             RULE, "PART A: the teacher's explanation of the MERGE rule (C6)", RULE]
    lines += [merge[0]["reply"]] if merge else ["(the teacher never reached the merge rule; leave the Part A columns blank)"]
    lines += ["", RULE, "PART B: the teacher's explanations of the other rules (read only if Part B is required above)", RULE]
    for x in others:
        name = x["prompt"].split("'")[1] if "'" in x["prompt"] else x["prompt"]
        lines += [f"--- {name} ---", x["reply"], ""]
    path = os.path.join(out_dir, "transcripts", tid + ".txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return tid


def rater(runs_dir: str, sample_path: str, out_dir: str, n: int = 24, seed: int = 0, part_b_per_cell: int = 1) -> List[str]:
    """A stratified subset for the second human rater: n transcripts with equal
    numbers from each arm x generation cell of the 40-transcript sample,
    teacher turns only, the merge explanation first. One transcript per cell
    is marked for Part B (every rule); the rest need Part A (merge rule) only."""
    sample = json.load(open(sample_path)); rng = random.Random(seed)
    by_cell: Dict[str, List] = {}
    for i, e in enumerate(sample, 1):
        by_cell.setdefault(f"{e['arm']}-{e['generation']}", []).append((i, e))
    per = max(1, n // len(by_cell)); chosen = []
    for k in sorted(by_cell):
        lst = by_cell[k][:]; rng.shuffle(lst)
        for j, (i, e) in enumerate(lst[:per]):
            chosen.append((i, e, j < part_b_per_cell))
    chosen = sorted(chosen, key=lambda x: x[0])[:n]
    ids = []
    for i, e, pb in chosen:
        ids.append(write_rater_transcript(runs_dir, e, i, out_dir, pb))
    with open(os.path.join(out_dir, "rater_sheet.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(RATER_SHEET_HEADER)
        for tid, (i, e, pb) in zip(ids, chosen):
            w.writerow([tid, e["arm"], e["generation"], "", "", "", "", "" if pb else "n/a", "", ""])
    return ids


def summaries(runs_dir: str, out_dir: str) -> int:
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*-*"))):
        for p in sorted(glob.glob(os.path.join(d, "sigma_*gen*.json"))):
            j = json.load(open(p))
            out.append({"run": os.path.basename(d), "file": os.path.basename(p), "generation": j["generation"], "sigma": j["sigma"], "chars": len(j["sigma"])})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summaries_sigma.json"), "w") as f:
        json.dump(out, f, indent=1)
    return len(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("packet", "rater"):
        p = sub.add_parser(name); p.add_argument("--runs", required=True); p.add_argument("--sample", required=True); p.add_argument("--out", required=True)
        if name == "rater":
            p.add_argument("--n", type=int, default=24); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("summaries"); p.add_argument("--runs", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "packet":
        print(len(packet(a.runs, a.sample, a.out)), "transcripts written")
    elif a.cmd == "rater":
        print(len(rater(a.runs, a.sample, a.out, a.n, a.seed)), "transcripts written for the second rater")
    else:
        print(summaries(a.runs, a.out), "summaries exported")


if __name__ == "__main__":
    main()

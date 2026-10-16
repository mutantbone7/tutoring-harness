"""Version-2 commands that call the model (resubmission of Access-2026-47739).

  reprobe  Re-run the session-boundary persistence probe for every REFERENCE
           student of a finished run with that student's OWN summary. Version
           1 gave the reference student the chain student's summary (chain.py,
           line 82 of the released archive; reviewer 2, item 7a). The
           reference student's context is rebuilt from transcript_gen<n>.json,
           the SUMMARISE prompt is sent once, and the boundary probe is run on
           a fresh instance given that summary. Output goes to a separate
           directory; the original run directories are not modified.

  wording  Test whether the wording of the specification given to teachers
           causes the omission of the tether scaling (reviewer 2, item 4d):
           fresh teacher instances receive SPEC_Y, INSTRUCTION_Y (the Gate 2
           block) or SPEC_Y with the merge rule restated in INSTRUCTION_Y's
           form, teach all six components in order under the session's
           TEACH prompt and cap, and the merge explanation is scored with the
           released detector (revision.tether_scaling_stated).
"""
from __future__ import annotations
import glob
import json
import os
import time
from typing import Dict, List

from .chain import SUMMARY_PROMPT, instantiate
from .router import Router, Instance, spec_fingerprints
from .session import STUDENT_SYSTEM, TEACH_PROMPT, present
from .learner import History
from .telemetry import Telemetry


def reprobe(client, content, cfg, runs_dir: str, out_dir: str, tag: str = "main") -> Dict:
    forbidden = spec_fingerprints(content.specification()) | {b"ANSWER KEY", b"KEY["}
    router = Router(client, forbidden)
    done, rows = 0, 0
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{tag}-*"))):
        cell = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        summ = json.load(open(os.path.join(d, "run_summary.json")))
        seed = int(cell.split("-")[-1][1:])
        tel = Telemetry(out_dir, cell)
        for n in range(1, cfg.generations + 1):
            tp = os.path.join(d, f"transcript_gen{n}.json")
            if not os.path.exists(tp) or os.path.exists(os.path.join(tel.dir, f"sigma_ref_gen{n}.json")):
                continue
            t = json.load(open(tp)); src = t.get(f"Sprime{n}")
            if src is None:
                continue
            res_r = next(r for r in summ["results"] if r["arm"] == "reference" and r["generation"] == n)
            student = Instance(role="student", label=f"Sprime{n}", seed=src["seed"], model=cfg.model, system=src["system"])
            student.messages = [dict(m) for m in src["messages"]]
            sigma_r = router.send(student, SUMMARY_PROMPT, cfg.summary_token_cap).text
            tel.write_json(f"sigma_ref_gen{n}.json", {"generation": n, "sigma": sigma_r, "source": "reprobe (v2): summary requested from the reference student's replayed context, after its post-test, matching the version-1 timing of the chain summary",
                                                     "served_model": student.served_model})
            base = {"seed": seed, "generation": n, "arm": "reference", "teacher_model": cfg.model, "content_set": content.name,
                    "tau": float([p for p in cell.split("-") if p.startswith("tau")][0][3:]), "calibration_flag": "-cal-" in cell}
            comp = res_r.get("first_mastered") or content.topo_order()[0]
            fresh = Instance(role="student", label=f"Sprime{n}-boundary", seed=seed + 200 + 400, model=cfg.model,
                             system=STUDENT_SYSTEM + "\n\nYour notes from a previous session:\n" + sigma_r)
            h = History()
            for _ in range(cfg.persistence_items):
                present(router, fresh, content.sample_item(comp, fresh.rng), "", cfg.answer_token_cap, tel, base, "persist_boundary", h, None)
                rows += 1
            done += 1
            print(f"{cell} gen {n}: summary {len(sigma_r)} chars; boundary probe on {comp} done", flush=True)
        tel.write_json("reprobe_summary.json", {"cell": cell, "sessions": done, "turns": router.turn_log})
        tel.close()
    return {"sessions": done, "probe_rows": rows, "tokens_in": client.total_in, "tokens_out": client.total_out}


def _wording_status(path: str, components: List[str]) -> Dict:
    """Read a replies.jsonl and return, per (variant, teacher), the last
    record seen for each component. A teacher counts as complete when all
    components have a record; the omission is scored from its last C6 record."""
    last = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            last.setdefault((r["variant"], r["teacher"]), {})[r["component"]] = r
    status = {}
    for key, recs in last.items():
        complete = all(c in recs for c in components)
        status[key] = {"complete": complete, "scaling_stated": recs["C6"]["scaling_stated"] if complete else None}
    return status


def _wording_counts(status: Dict, variants: List[str]) -> Dict:
    counts = {v: {"omitted": 0, "of": 0} for v in variants}
    for (v, _), st in status.items():
        if v in counts and st["complete"]:
            counts[v]["of"] += 1
            counts[v]["omitted"] += int(not st["scaling_stated"])
    return counts


def wording(client, content, cfg, out_dir: str, n: int = 30, variants: List[str] = None, resume: str = None) -> Dict:
    """`resume` continues an interrupted run: the path of its directory, or
    "latest" for the most recent wording-<set>-* directory under out_dir.
    Teachers whose six replies are already on disk are skipped; a teacher
    that was cut off part-way is run again from its first prompt (its
    context cannot be rebuilt exactly), and the later records supersede the
    earlier ones when the file is read back. Records written by a resumed
    run carry "resumed": true."""
    from .revision import tether_scaling_stated
    from .content.novel import SPEC_Y, INSTRUCTION_Y
    merge_line = next(l for l in INSTRUCTION_Y.splitlines() if l.startswith("Merge:"))
    spec_merge_restated = "\n".join(
        (f"Rule C6 (merge): {merge_line[len('Merge: '):]} The shares alone do not fix the merged height, the tether does."
         if l.startswith("Rule C6") else l) for l in SPEC_Y.splitlines())
    texts = {"spec": SPEC_Y, "instruction": INSTRUCTION_Y, "spec_merge_restated": spec_merge_restated}
    variants = variants or list(texts)
    components = content.topo_order()
    if resume:
        if resume == "latest":
            prev = sorted(glob.glob(os.path.join(out_dir, f"wording-{content.name}-*")))
            if not prev:
                raise SystemExit(f"nothing to resume: no wording-{content.name}-* directory in {out_dir}")
            run_dir = prev[-1]
        else:
            run_dir = resume
        run_id = os.path.basename(os.path.normpath(run_dir))
        out_dir = os.path.dirname(os.path.normpath(run_dir)) or out_dir
        mode = "a"
    else:
        run_id = f"wording-{content.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        mode = "w"
    tel = Telemetry(out_dir, run_id)
    router = Router(client, set())
    out_path = os.path.join(tel.dir, "replies.jsonl")
    status = _wording_status(out_path, components) if resume else {}
    if resume:
        done = sum(1 for st in status.values() if st["complete"])
        print(f"resuming {run_id}: {done} complete teachers on disk", flush=True)
    with open(out_path, mode) as f:
        for v in variants:
            for i in range(n):
                if status.get((v, i), {}).get("complete"):
                    continue
                teacher = instantiate("teacher", f"W-{v}-{i}", 1000 + i, cfg.model, context=texts[v])
                recs = {}
                for c in components:
                    reply = router.send(teacher, TEACH_PROMPT.format(name=content.graph()[c].name), cfg.teach_token_cap).text
                    rec = {"variant": v, "teacher": i, "component": c, "reply": reply, "served_model": teacher.served_model}
                    if resume:
                        rec["resumed"] = True
                    if c == "C6":
                        rec["scaling_stated"] = tether_scaling_stated(reply)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    recs[c] = rec
                status[(v, i)] = {"complete": True, "scaling_stated": recs["C6"]["scaling_stated"]}
                cnt = _wording_counts(status, [v])[v]
                print(f"{v} teacher {i + 1}/{n}: omitted so far {cnt['omitted']}/{cnt['of']}", flush=True)
    counts = _wording_counts(status, variants)
    tel.write_json("wording_summary.json", {"counts": counts, "variants": {k: texts[k] for k in variants}, "turns": router.turn_log,
                                            "resumed": bool(resume)})
    tel.close()
    print(json.dumps(counts, indent=1))
    return counts

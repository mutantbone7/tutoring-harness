"""Gate 1 contamination probes (Section III-D) and the Gate 2 equating pilot
(Section III-A). Both are single-turn: no teacher, no policy."""
from __future__ import annotations
import json
import math
import os
import random
from typing import Dict, List
from .content.base import ContentSet, LABELS
from .parsing import parse_answer
from .router import Router, Instance, spec_fingerprints
from .telemetry import Telemetry

PROBE_SYSTEM = ("Answer the question. Keep any reasoning to a few sentences. Finish with one final line of the form "
                "'ANSWER: <letter>' and write nothing after it. If your reasoning runs long, stop and give the answer line.")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def binom_pvalue_greater(k: int, n: int, p0: float) -> float:
    """One-sided exact binomial test of H0: p = p0 against H1: p > p0."""
    from scipy.stats import binomtest
    return float(binomtest(k, n, p0, alternative="greater").pvalue)


def _already_done(telemetry: Telemetry, phase: str) -> List[Dict]:
    """Rows of a previous, interrupted run of the same phase in the same run directory."""
    import json, os
    path = os.path.join(telemetry.dir, "telemetry.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("phase") == phase:
                rows.append(r)
    return rows


def run_probes(client, content: ContentSet, cfg, telemetry: Telemetry, seed: int = 0, n: int = None,
               system: str = PROBE_SYSTEM, phase: str = "probe", resume: bool = False) -> Dict:
    n = n or cfg.n_probes
    forbidden = spec_fingerprints(content.specification()) | {b"ANSWER KEY", b"KEY["}
    router = Router(client, forbidden)
    rng = random.Random(seed)
    comps = content.components()
    k = 0
    per_comp: Dict[str, List[bool]] = {c.id: [] for c in comps}
    per_family: Dict[str, List[bool]] = {}
    served = None
    done = _already_done(telemetry, phase) if resume else []
    status_counts: Dict[str, int] = {}
    for r in done:                      # restore counts from the interrupted run
        status_counts[r["parse_status"]] = status_counts.get(r["parse_status"], 0) + 1
        k += bool(r["correct"])
        per_comp.setdefault(r["component"], []).append(bool(r["correct"]))
        per_family.setdefault(r["family"], []).append(bool(r["correct"]))
        served = r.get("student_model") or served
    for i in range(n):
        comp = comps[i % len(comps)]
        item = content.sample_item(comp.id, rng)     # advances the item RNG identically on resume
        if i < len(done):
            continue
        inst = Instance(role="probe", label=f"probe{i}", seed=seed * 100000 + i, model=cfg.model, system=system)
        prompt, key, tagmap = item.render(inst.rng)
        comp_ = router.send(inst, prompt, cfg.answer_token_cap, meta={"key": key, "item_id": item.item_id})
        served = comp_.model
        label, status = parse_answer(comp_.text)
        correct = status != "failure" and label == key
        k += correct
        per_comp[comp.id].append(correct)
        per_family.setdefault(item.family, []).append(correct)
        status_counts[status] = status_counts.get(status, 0) + 1
        with open(os.path.join(telemetry.dir, f"{phase}_responses.jsonl"), "a") as rf:
            rf.write(json.dumps({"i": i, "item_id": item.item_id, "component": comp.id, "family": item.family,
                                 "prompt": prompt, "response": comp_.text, "label": label, "status": status,
                                 "correct": correct, "misconception": tagmap.get(label) if (label and not correct) else None,
                                 "tokens_out": comp_.tokens_out, "stop_reason": getattr(comp_, "stop_reason", ""),
                                 "block_types": getattr(comp_, "block_types", "")}) + "\n")
        telemetry.log(seed=seed, generation=0, arm=phase, teacher_model=None, student_model=comp_.model,
                      content_set=content.name, tau=None, calibration_flag=None, component=comp.id,
                      item_id=item.item_id, family=item.family, held_out=False, correct=correct,
                      distractor=label if not correct else None, misconception=tagmap.get(label) if (label and not correct) else None,
                      parse_status=status, latency_ms=comp_.latency_ms, tokens_in=comp_.tokens_in,
                      tokens_out=comp_.tokens_out, context_position=1, context_length=inst.context_length(),
                      phase=phase, answer_label=label)
    acc = k / n
    lo, hi = wilson(k, n)
    pval = binom_pvalue_greater(k, n, content.chance_level)
    at_floor = (acc <= cfg.probe_fail_threshold) and (pval >= cfg.probe_alpha)
    failure_rate = status_counts.get("failure", 0) / n
    out = {"model": served or cfg.model, "content_set": content.name, "n": n, "correct": k, "accuracy": acc,
           "ci95": [lo, hi], "p_greater_than_chance": pval, "chance": content.chance_level,
           "at_floor": at_floor, "parse_status_counts": status_counts, "parse_failure_rate": failure_rate,
           "valid": failure_rate <= 0.05,
           "per_component": {c: (sum(v) / len(v) if v else None) for c, v in per_comp.items()},
           "per_family": {f: (sum(v) / len(v) if v else None) for f, v in per_family.items()},
           "isolation_violations": router.violations}
    telemetry.write_json(f"{phase}_{content.name}.json", out)
    return out


def run_pilot(client, content: ContentSet, cfg, telemetry: Telemetry, seed: int = 0, n: int = None, resume: bool = False) -> Dict:
    """Equating pilot: the minimal instruction block, then single-turn items.
    Difficulty per family is 1 - accuracy and feeds Item.difficulty."""
    system = PROBE_SYSTEM + "\n\n" + content.instruction_block()
    out = run_probes(client, content, cfg, telemetry, seed=seed, n=n, system=system, phase="pilot", resume=resume)
    out["difficulty_per_family"] = {f: (1 - a) if a is not None else None for f, a in out["per_family"].items()}
    telemetry.write_json(f"pilot_{content.name}.json", out)
    return out


def table3_rows(results: List[Dict]) -> str:
    """LaTeX rows for Table 3 of the manuscript."""
    rows = []
    for r in results:
        crit = {"Y": "at floor required", "S": "reported, not gated", "R": "reported"}.get(r["content_set"], "reported")
        rows.append(f"\\ {r['model']} & {r['content_set']} & {r['accuracy']:.3f} & [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] & {crit}; "
                    f"$p={r['p_greater_than_chance']:.3g}$ \\\\")
    return "\n".join(rows)

"""Algorithm 2: chain runner with transmission controls, line for line, plus
the persistence proxies of Section V-D run on every student instance.

What ran in the study (version 1) and what changed in version 2:

  * The teacher at generation n is a FRESH instance of the base model whose
    system prompt carries the generation-(n-1) chain student's summary
    sigma_{n-1} (line 15). It is not the promoted student instance itself;
    the first submission described it loosely as "promoted" (reviewer 2,
    item 4b). Nothing changed here; the description in the paper did.
  * Version 1 requested sigma_n AFTER the post-test, so a summary could
    reflect what the student inferred from post-test items (reviewer 2,
    item 4g). Version 2 requests it before the post-test when
    Config.summary_before_posttest is True (default). Set it to False to
    reproduce the study as run.
  * Version 1 passed the CHAIN student's summary to the REFERENCE student's
    boundary probe (reviewer 2, item 7a). Version 2 has the reference
    student write its own summary (Config.reference_summary, default True)
    and uses that; both summaries are logged.
  * Config.policy == "fixed" runs the non-adaptive fixed-sequence control
    (Section IV-E) in place of Algorithm 1 in both arms.
"""
from __future__ import annotations
from typing import Dict, List
from .content.base import ContentSet
from .learner import BKTLearner, GBDTLearner, History
from .router import Router, Instance, IsolationViolation, spec_fingerprints
from .session import run_session, present, TEACHER_SYSTEM
from .telemetry import Telemetry

SUMMARY_PROMPT = ("SUMMARISE: Write notes, in your own words, on everything you now understand about the rules "
                  "you were taught, as if preparing to teach someone else. Do not mention specific questions "
                  "or answers. Do not copy your tutor's words.")


def make_learner(cfg, calibrated: bool = True):
    path = cfg.learner if calibrated else cfg.learner_uncalibrated
    if path == "bkt":
        return BKTLearner()
    return GBDTLearner.load(path)


def instantiate(role: str, label: str, seed: int, model: str, context: str = "") -> Instance:
    """instantiate(M, seed) with an optional carried context (Sigma or sigma_n)."""
    system = ""
    if context:
        system = TEACHER_SYSTEM + "\n\nWhat you know:\n" + context
    return Instance(role=role, label=label, seed=seed, model=model, system=system)


def assert_clean(text: str, forbidden, label: str) -> None:
    b = text.encode("utf-8")
    for chunk in forbidden:
        if chunk in b:
            raise IsolationViolation(f"{label}: forbidden content present: {chunk[:60]!r}")


def persistence_probes(router: Router, student: Instance, sigma: str, content: ContentSet, cfg, telemetry: Telemetry,
                       base: Dict, first_mastered: str, model: str, seed: int) -> None:
    """Three proxies, reported jointly (Section V-D, decision recorded before Gate 3).
    `sigma` must be the summary written by THIS student (v2)."""
    h = History()
    m = cfg.persistence_items
    comp = first_mastered or content.topo_order()[0]
    for _ in range(m):    # (a) context-distance: the earliest-mastered component, re-tested at the far end of the context
        present(router, student, content.sample_item(comp, student.rng), "", cfg.answer_token_cap, telemetry, base, "persist_distance", h, None)
    fresh = Instance(role="student", label=student.label + "-boundary", seed=seed + 400, model=model,
                     system=student.system + "\n\nYour notes from a previous session:\n" + sigma)
    for _ in range(m):    # (b) session-boundary: summary-only re-instantiation
        present(router, fresh, content.sample_item(comp, fresh.rng), "", cfg.answer_token_cap, telemetry, base, "persist_boundary", h, None)
    for a, b in content.confusable_pairs():   # (c) interference: a, then confusable b, then a again
        for target in (a, b, a):
            for _ in range(max(1, m // 3)):
                present(router, student, content.sample_item(target, student.rng), "", cfg.answer_token_cap, telemetry, base, f"persist_interference_{target}", h, None)


def run_chain(client, content: ContentSet, cfg, tau: float, seed: int, calibrated: bool, telemetry: Telemetry,
              set_index: int = 0) -> List[Dict]:
    sigma_spec = content.specification()
    forbidden = spec_fingerprints(sigma_spec) | {b"ANSWER KEY", b"KEY["}
    router = Router(client, forbidden)
    M = cfg.model
    results = []
    before = getattr(cfg, "summary_before_posttest", True)
    ref_own = getattr(cfg, "reference_summary", True)

    def summarise(student: Instance) -> str:
        return router.send(student, SUMMARY_PROMPT, cfg.summary_token_cap).text

    T_prev = instantiate("teacher", "T0", seed, M, context=sigma_spec)              # line 1
    for n in range(1, cfg.generations + 1):                                          # line 2
        cond = {"set_index": set_index, "tau": tau, "calibration_flag": calibrated, "generation": n, "arm_index": 0}
        base = {"seed": seed, "generation": n, "arm": "chain", "teacher_model": M, "content_set": content.name,
                "tau": tau, "calibration_flag": calibrated, "_condition": cond}
        S_n = instantiate("student", f"S{n}", seed + n, M)                            # line 4
        res = run_session(router, T_prev, S_n, content, tau, make_learner(cfg, calibrated), cfg, telemetry, base,
                          before_posttest=summarise if before else None)              # line 5
        router.check_isolation(S_n)                                                   # line 6
        results.append({"arm": "chain", "generation": n, **{k: v for k, v in res.items() if k not in ("history", "sigma")}})   # line 7
        R_n = instantiate("teacher", f"R{n}", seed + 100 + n, M, context=sigma_spec) # line 9
        S_pn = instantiate("student", f"Sprime{n}", seed + 200 + n, M)                # line 10
        base_r = dict(base, arm="reference", _condition=dict(cond, arm_index=1))
        res_r = run_session(router, R_n, S_pn, content, tau, make_learner(cfg, calibrated), cfg, telemetry, base_r,
                            before_posttest=(summarise if (before and ref_own) else None))  # line 11
        results.append({"arm": "reference", "generation": n, **{k: v for k, v in res_r.items() if k not in ("history", "sigma")}})  # line 12
        sigma_n = res["sigma"] if before else summarise(S_n)                          # line 14
        telemetry.write_json(f"sigma_gen{n}.json", {"generation": n, "sigma": sigma_n, "written_before_posttest": before})
        if ref_own:
            sigma_r = res_r["sigma"] if before else summarise(S_pn)
            telemetry.write_json(f"sigma_ref_gen{n}.json", {"generation": n, "sigma": sigma_r, "written_before_posttest": before})
        else:
            sigma_r = sigma_n                                                         # the version-1 behaviour (bug)
        T_n = instantiate("teacher", f"T{n}", seed + 300 + n, M, context=sigma_n)    # line 15
        assert_clean(T_n.system, forbidden, f"T{n}")                                  # line 16
        persistence_probes(router, S_n, sigma_n, content, cfg, telemetry, base, res["first_mastered"], M, seed)      # line 17
        persistence_probes(router, S_pn, sigma_r, content, cfg, telemetry, base_r, res_r["first_mastered"], M, seed + 200)
        telemetry.write_transcript(f"gen{n}", [T_prev, S_n, R_n, S_pn, T_n])
        T_prev = T_n
    telemetry.write_json("run_summary.json", {"results": results, "isolation_violations": router.violations,
                                              "turns": router.turn_log,
                                              "tokens_in": sum(t["tokens_in"] for t in router.turn_log),
                                              "tokens_out": sum(t["tokens_out"] for t in router.turn_log),
                                              "cache_read": sum(t.get("cache_read", 0) for t in router.turn_log),
                                              "cache_write": sum(t.get("cache_write", 0) for t in router.turn_log),
                                              "retries": sum(t["retries"] for t in router.turn_log),
                                              "summary_before_posttest": before, "reference_summary": ref_own,
                                              "policy": getattr(cfg, "policy", "mastery")})
    return results

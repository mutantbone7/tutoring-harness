"""Revision analysis (resubmission of Access-2026-47739).

  python -m harness.revision --runs runs --out analysis_v2 --floor 0.064 --floor-r 0.960 --floor-s 0.815

Everything here is computed from the logged run directories (telemetry.jsonl,
run_summary.json, sigma_gen*.json, transcript_gen*.json) and from the
bootstrap pass that trained the learner model. It adds to analysis.py the
quantities the reviewers asked for:

  * equations (5) to (7) as pre-registered, from the ceiling-control cells;
  * the policy against the rule "advance on the first correct answer";
  * calibration maps refitted without the L2 penalty and with isotonic
    regression, with raw accuracy beside every plateau;
  * mastery decisions re-scored on an estimate of the latent P(L) and on a
    learner model trained without the condition group;
  * post-test accuracy by component and by whether the component was taught;
  * generation-1 chain-minus-reference differences as a noise baseline;
  * tether-scaling omission by arm, generation and seed, with the detector;
  * t-based cell intervals, cluster-robust pooled intervals, the mixed model
    with run and item-family random intercepts, the eight-contrast family
    with Benjamini-Hochberg adjustment, minimum detectable effects;
  * the knee estimator's degeneracy and a change-point reading of the step;
  * survival analysis restricted to attempted components, clustered by
    session; the dense grid on a pooled basis;
  * persistence proxies, attempt-cap rates, over- and under-practice,
    abandonment, time to mastery; the replication cell's descriptives;
  * the knowledge-tracing baselines with a condition-blind GBDT and a DKT
    with a component-specific prior.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import random
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import load, wilson, ece, platt_fit, tau_prime, seed_variance

warnings.filterwarnings("ignore")

T975_2 = float(stats.t.ppf(0.975, 2))      # 4.303, three seeds


# ----------------------------------------------------------------------------
# Intervals
# ----------------------------------------------------------------------------
def t_ci(x: List[float]) -> Tuple[float, float, float, int]:
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    n = len(x)
    if n == 0:
        return (float("nan"),) * 3 + (0,)
    m = float(x.mean())
    if n < 2:
        return (m, float("nan"), float("nan"), n)
    se = x.std(ddof=1) / math.sqrt(n); q = float(stats.t.ppf(0.975, n - 1))
    return (m, m - q * se, m + q * se, n)


def cluster_boot_ci(values: np.ndarray, clusters: np.ndarray, reps: int = 4000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean with a cluster-bootstrap percentile interval (resample clusters)."""
    rng = np.random.RandomState(seed)
    cl = np.asarray(clusters); vals = np.asarray(values, float)
    ids = np.unique(cl); groups = [vals[cl == i] for i in ids]
    boots = []
    for _ in range(reps):
        pick = rng.randint(0, len(groups), len(groups))
        boots.append(np.concatenate([groups[k] for k in pick]).mean())
    return (float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def cluster_robust_ci(values: np.ndarray, clusters: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean with a cluster-robust (sandwich) standard error, t with G-1 df."""
    import statsmodels.api as sm
    y = np.asarray(values, float); X = np.ones((len(y), 1))
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(clusters)[0]})
    se = float(res.bse[0]); G = len(set(clusters)); q = float(stats.t.ppf(0.975, G - 1))
    return (float(y.mean()), float(y.mean() - q * se), float(y.mean() + q * se), se)


# ----------------------------------------------------------------------------
# 1. The decomposition as pre-registered (equations 5 to 7)
# ----------------------------------------------------------------------------
def decomposition(sess: pd.DataFrame, controls: pd.DataFrame, floors: Dict[str, float]) -> Dict:
    """gain(X) = a1(X) - a0(X) per set at the matching cell (tau = 0.85,
    calibrated), per arm and pooled over arms; then the three differences."""
    out = {"floors": floors, "gain": {}, "differences": {}}
    y = sess[(np.isclose(sess.tau, 0.85)) & (sess.cal)]
    cells = {"Y": y}
    for st in ("R", "S"):
        cells[st] = controls[(controls.set == st) & (np.isclose(controls.tau, 0.85)) & (controls.cal)]
    for st, df in cells.items():
        out["gain"][st] = {}
        for arm in ("chain", "reference", "pooled"):
            d = df if arm == "pooled" else df[df.arm == arm]
            m, lo, hi, n = t_ci((d.post_accuracy - floors[st]).tolist())
            out["gain"][st][arm] = {"mean": m, "ci_t": [lo, hi], "n_sessions": n,
                                    "post_accuracy": float(d.post_accuracy.mean()) if n else float("nan"),
                                    "n_items": int(18 * n)}
    for arm in ("chain", "reference", "pooled"):
        gR, gS, gY = (out["gain"][s][arm]["mean"] for s in ("R", "S", "Y"))
        out["differences"][arm] = {"eq5_gainR_minus_gainS": gR - gS, "eq6_gainS_minus_gainY": gS - gY,
                                   "eq7_gainR_minus_gainY": gR - gY}
    out["note"] = ("Under equations (2) to (4), gain(S) - gain(Y) = e_sigma >= 0 and gain(R) - gain(S) = "
                   "e_upsilon + kappa >= 0. Both are negative here because a_0(R) and a_0(S) leave no room "
                   "for a positive gain: the additive model with a common alpha does not hold at the ceiling, "
                   "so (6) is not estimable from these cells.")
    return out


# ----------------------------------------------------------------------------
# 2. Policy against the first-correct rule; cap rates; practice; abandonment
# ----------------------------------------------------------------------------
def _instruction_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[rows.phase == "instruction"].copy()


def policy_vs_first_correct(sess: pd.DataFrame, rows: pd.DataFrame) -> Dict:
    inst = _instruction_rows(rows)
    out = {"mastered_components": 0, "mastered_on_first_item": 0, "mastered_on_first_correct": 0,
           "by_arm_cal": {}, "by_tau_cal": {}}
    recs = []
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)].sort_values("context_position")
        for c, k in s["items_to_mastery"].items():
            att = g[g.component == c]
            if k is None:
                continue
            corr = att.correct.astype(bool).tolist()
            first_correct = next((i + 1 for i, v in enumerate(corr) if v), None)
            recs.append({"cell": s.cell, "tau": s.tau, "cal": s.cal, "arm": s.arm, "component": c, "k": k,
                         "first_correct": first_correct, "on_first_item": k == 1,
                         "on_first_correct": first_correct == k})
    df = pd.DataFrame(recs)
    out["mastered_components"] = int(len(df))
    out["mastered_on_first_item"] = int(df.on_first_item.sum())
    out["mastered_on_first_correct"] = int(df.on_first_correct.sum())
    out["later_than_first_correct"] = int((df.k > df.first_correct.fillna(99)).sum())
    out["mastered_without_correct_answer"] = int(df.first_correct.isna().sum())
    for (cal,), g in df.groupby(["cal"]):
        out["by_arm_cal"][f"cal={cal}"] = {"n": int(len(g)), "on_first_item": float(g.on_first_item.mean()),
                                           "on_first_correct": float(g.on_first_correct.mean())}
    for (tau, cal), g in df.groupby(["tau", "cal"]):
        out["by_tau_cal"][f"tau={tau}|cal={cal}"] = {"n": int(len(g)), "on_first_item": float(g.on_first_item.mean()),
                                                     "on_first_correct": float(g.on_first_correct.mean())}
    # Session-level: would the first-correct rule have produced the same set of mastered components?
    same = 0
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)]
        fc = {c for c, gg in g.groupby("component") if gg.correct.astype(bool).any()}
        same += int(fc == set(s["mastered"]))
    out["sessions_same_mastered_set_as_first_correct"] = {"n": int(len(sess)), "same": same}
    return out


def termination_and_practice(sess: pd.DataFrame, rows: pd.DataFrame) -> Dict:
    inst = _instruction_rows(rows); post = rows[rows.phase == "post"]
    out = {"outcomes_by_cell": {}, "attempt_cap_rate_by_tau_cal": {}, "all_six_mastered_vs_completed": {},
           "over_practice": {}, "under_practice": {}, "abandonment": {}, "time_to_mastery_s": {}}
    for (tau, cal), g in sess.groupby(["tau", "cal"]):
        key = f"tau={tau}|cal={cal}"
        oc = g.outcome.value_counts().to_dict()
        out["outcomes_by_cell"][key] = {k: int(v) for k, v in oc.items()}
        out["attempt_cap_rate_by_tau_cal"][key] = float((g.outcome == "non-mastery").mean())
        out["all_six_mastered_vs_completed"][key] = {"all_six_mastered": float((g.n_mastered == 6).mean()),
                                                     "terminated_by_mastery": float(g.completed.mean()), "n": int(len(g))}
    out["abandonment"] = {"sessions_at_cap": int((sess.outcome == "cap").sum()), "n": int(len(sess)),
                          "note": "abandonment = session reaching the 40-exchange cap; none did"}
    # over-practice: instruction items on a component after its first correct answer and before mastery
    # under-practice: components declared mastered whose post-test accuracy was below 2 of 3
    over, under = defaultdict(list), defaultdict(list)
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)].sort_values("context_position")
        p = post[(post.cell == s.cell) & (post.generation == s.generation) & (post.arm == s.arm)]
        key = f"tau={s.tau}|cal={s.cal}"
        for c, k in s["items_to_mastery"].items():
            att = g[g.component == c]
            if len(att) == 0:
                continue
            corr = att.correct.astype(bool).tolist()
            fc = next((i + 1 for i, v in enumerate(corr) if v), None)
            if k is not None and fc is not None:
                over[key].append(max(0, k - fc))
            if k is not None:
                acc = p[p.component == c].correct.astype(float).mean()
                under[key].append(float(acc < 2 / 3))
    out["over_practice"] = {k: {"mean_items_beyond_first_correct": float(np.mean(v)), "n": len(v)} for k, v in over.items()}
    out["under_practice"] = {k: {"share_mastered_components_below_two_thirds_post": float(np.mean(v)), "n": len(v)} for k, v in under.items()}
    # time to mastery: seconds from the session's first instruction row to the mastery row, per component
    tt = defaultdict(list)
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)].sort_values("context_position")
        if len(g) == 0:
            continue
        t0 = pd.to_datetime(g.timestamp.iloc[0])
        for c, k in s["items_to_mastery"].items():
            if k is None:
                continue
            att = g[g.component == c]
            if len(att) >= k:
                tt[c].append((pd.to_datetime(att.timestamp.iloc[k - 1]) - t0).total_seconds())
    out["time_to_mastery_s"] = {c: {"median": float(np.median(v)), "n": len(v)} for c, v in sorted(tt.items())}
    return out


# ----------------------------------------------------------------------------
# 3. Accuracy by component, by taught status, and the Gate 2 gap
# ----------------------------------------------------------------------------
def by_component(sess: pd.DataFrame, rows: pd.DataFrame, gate2: Dict[str, float]) -> Dict:
    inst = _instruction_rows(rows); post = rows[rows.phase == "post"]; tr = rows[rows.phase == "transfer"]
    taught = set()
    for (cell, gen, arm), g in inst.groupby(["cell", "generation", "arm"]):
        for c in g.component.unique():
            taught.add((cell, gen, arm, c))
    post = post.copy()
    post["taught"] = [((r.cell, r.generation, r.arm, r.component) in taught) for r in post.itertuples()]
    out = {"post_by_component": {}, "post_by_component_taught": {}, "transfer_by_component": {}, "gate2": gate2, "gap_to_gate2": {}}
    for c, g in post.groupby("component"):
        k, n = int(g.correct.sum()), int(len(g)); lo, hi = wilson(k, n)
        out["post_by_component"][c] = {"accuracy": k / n, "n": n, "wilson": [lo, hi]}
        if c in gate2:
            out["gap_to_gate2"][c] = gate2[c] - k / n
        for t, gg in g.groupby("taught"):
            kk, nn = int(gg.correct.sum()), int(len(gg)); lo, hi = wilson(kk, nn)
            out["post_by_component_taught"][f"{c}|taught={t}"] = {"accuracy": kk / nn, "n": nn, "wilson": [lo, hi]}
    for c, g in tr.groupby("component"):
        k, n = int(g.correct.sum()), int(len(g)); lo, hi = wilson(k, n)
        out["transfer_by_component"][c] = {"accuracy": k / n, "n": n, "wilson": [lo, hi]}
    # the tau = 0.95 calibrated cells: only C1 taught
    sub = post[(np.isclose(post.tau, 0.95)) & post.cal]
    out["tau095_cal"] = {"n_items": int(len(sub)), "accuracy": float(sub.correct.mean()),
                         "untaught_items": int((~sub.taught).sum()), "untaught_accuracy": float(sub[~sub.taught].correct.mean()) if (~sub.taught).any() else float("nan"),
                         "by_component": {c: float(g.correct.mean()) for c, g in sub.groupby("component")}}
    # overall gap between post-test after teaching and Gate 2 under the instruction block
    tot = float(post.correct.mean())
    out["overall"] = {"post_accuracy": tot, "gate2_overall": gate2.get("overall"), "gap": (gate2.get("overall") or float("nan")) - tot}
    return out


# ----------------------------------------------------------------------------
# 4. Generation-1 noise baseline
# ----------------------------------------------------------------------------
def generation1_noise(sess: pd.DataFrame) -> Dict:
    g1 = sess[sess.generation == 1]
    recs = []
    for (tau, cal, seed), g in g1.groupby(["tau", "cal", "seed"]):
        ch = g[g.arm == "chain"]; rf = g[g.arm == "reference"]
        if len(ch) == 1 and len(rf) == 1:
            recs.append({"tau": tau, "cal": cal, "seed": seed,
                         "d_post": float(ch.post_accuracy.iloc[0] - rf.post_accuracy.iloc[0]),
                         "d_transfer": float(ch.transfer_accuracy.iloc[0] - rf.transfer_accuracy.iloc[0]),
                         "d_completed": float(ch.completed.iloc[0]) - float(rf.completed.iloc[0])})
    df = pd.DataFrame(recs)
    out = {"n_pairs": int(len(df))}
    for k in ("d_post", "d_transfer", "d_completed"):
        out[k] = {"mean": float(df[k].mean()), "sd": float(df[k].std(ddof=1)), "min": float(df[k].min()), "max": float(df[k].max()),
                  "mean_abs": float(df[k].abs().mean())}
    # cell-level (mean over the three seeds) chain minus reference at generation 1
    cells = []
    for (tau, cal), g in df.groupby(["tau", "cal"]):
        cells.append({"tau": tau, "cal": cal, "d_post": float(g.d_post.mean()), "d_transfer": float(g.d_transfer.mean())})
    out["cell_means"] = cells
    out["transfer_chain_minus_reference_all_generations"] = {}
    for (tau, cal), g in sess.groupby(["tau", "cal"]):
        a = g[g.arm == "chain"].transfer_accuracy.mean(); b = g[g.arm == "reference"].transfer_accuracy.mean()
        out["transfer_chain_minus_reference_all_generations"][f"tau={tau}|cal={cal}"] = float(a - b)
    return out


# ----------------------------------------------------------------------------
# 5. Statistics: cluster-robust pooled intervals, t-based cells, mixed models,
#    the eight-contrast family, minimum detectable effects
# ----------------------------------------------------------------------------
def pooled_intervals(sess: pd.DataFrame, floor: float) -> Dict:
    out = {}
    cl = sess.cell.values
    g = (sess.post_accuracy - floor).values
    out["gain_pooled"] = {"session_iid_1.96se": list(_iid(g)), "cluster_robust_t": list(cluster_robust_ci(g, cl)[:3]),
                          "cluster_bootstrap": list(cluster_boot_ci(g, cl)), "n_sessions": int(len(g)), "n_clusters": int(len(set(cl)))}
    t = sess.transfer_accuracy.values
    out["transfer_pooled"] = {"session_iid_1.96se": list(_iid(t)), "cluster_robust_t": list(cluster_robust_ci(t, cl)[:3]),
                              "cluster_bootstrap": list(cluster_boot_ci(t, cl))}
    out["gain_by_tau"] = {}
    for tau, s in sess.groupby("tau"):
        gg = (s.post_accuracy - floor).values
        out["gain_by_tau"][str(tau)] = {"session_iid_1.96se": list(_iid(gg)), "cluster_robust_t": list(cluster_robust_ci(gg, s.cell.values)[:3]),
                                        "cluster_bootstrap": list(cluster_boot_ci(gg, s.cell.values))}
    out["post_by_cal"] = {}
    for cal, s in sess.groupby("cal"):
        out["post_by_cal"][str(cal)] = {"cluster_robust_t": list(cluster_robust_ci(s.post_accuracy.values, s.cell.values)[:3])}
    # calibrated minus uncalibrated with cluster-robust SE (OLS with a cal indicator)
    import statsmodels.formula.api as smf
    df = sess.copy(); df["cal_i"] = df.cal.astype(int); df["chain"] = (df.arm == "chain").astype(int); df["gen0"] = df.generation - 1
    df["tau_c"] = df.tau.astype(str)
    m = smf.ols("post_accuracy ~ cal_i + C(tau_c) + chain*gen0", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cell)[0]})
    ci = m.conf_int()
    out["calibrated_minus_uncalibrated"] = {"estimate": float(m.params["cal_i"]), "ci_cluster": [float(ci.loc["cal_i", 0]), float(ci.loc["cal_i", 1])], "p": float(m.pvalues["cal_i"])}
    out["by_arm_generation"] = {}
    for (arm, gen), s in sess.groupby(["arm", "generation"]):
        out["by_arm_generation"][f"{arm}-{int(gen)}"] = {"mean": float(s.post_accuracy.mean()), "t_ci": list(t_ci(s.post_accuracy.tolist())[:3]),
                                                        "n": int(len(s))}
    return out


def _iid(x):
    x = np.asarray(x, float); se = x.std(ddof=1) / math.sqrt(len(x))
    return (float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se))


def table8_t(sess: pd.DataFrame, floor: float, rows: pd.DataFrame, floors: Dict[str, float] = None) -> str:
    """Table 8 rows with t-based intervals over the three seeds (t_{0.975,2} = 4.30)."""
    lines = []
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna()]
    for (st, tau, cal, gen, arm), g in sess.groupby(["set", "tau", "cal", "generation", "arm"]):
        fl = (floors or {}).get(st, floor)
        m, lo, hi, n = t_ci((g.post_accuracy - fl).tolist()); tr = float(g.transfer_accuracy.mean())
        itm = [v for s_ in g.items_to_mastery for v in s_.values()]
        med = np.median([v for v in itm if v is not None]) if any(v is not None for v in itm) else float("nan")
        cens = np.mean([v is None for v in itm])
        sub = inst[(inst.set == st) & (inst.tau == tau) & (inst.cal == cal) & (inst.generation == gen) & (inst.arm == arm)]
        e = ece(sub.p_next.astype(float), sub.correct.astype(int)) if len(sub) else float("nan")
        if n >= 2 and not math.isnan(lo):
            ci = f"[{max(lo, -1):.2f}, {min(hi, 1):.2f}]" if hi - lo > 1e-9 else "(identical seeds)"
        else:
            ci = "(one seed)"
        medtxt = "n/a" if math.isnan(med) else f"{med:.0f}"
        lines.append(f"{st} & {tau:.2f} & {'on' if cal else 'off'} & {int(gen)} & {arm} & {m:.3f} {ci} & {tr:.3f} & {medtxt} ({cens:.0%}\\% cens.) & {g.completed.mean():.2f} & {e:.3f} \\\\")
    return "\n".join(lines)


def mixed_models(sess: pd.DataFrame, rows: pd.DataFrame) -> Dict:
    """(a) the fitted model of the first submission (seed random intercept);
    (b) the specified model: post-test item correctness with random intercepts
    for run and item family (equation 11); (c) a session-level model with the
    run (chain lineage) as the random intercept."""
    import statsmodels.formula.api as smf
    out = {}
    df = sess.dropna(subset=["post_accuracy"]).copy()
    df["chain"] = (df.arm == "chain").astype(int); df["gen0"] = df.generation - 1
    df["tau_c"] = df.tau.astype(str); df["cal_i"] = df.cal.astype(int)

    def coefs(m, names):
        ci = m.conf_int()
        return {n: {"estimate": float(m.params[n]), "ci": [float(ci.loc[n, 0]), float(ci.loc[n, 1])], "p": float(m.pvalues[n])} for n in names}
    names = ["gen0", "chain:gen0", "chain"]
    ma = smf.mixedlm("post_accuracy ~ chain*gen0 + C(tau_c) + cal_i", df, groups=df["seed"]).fit(reml=False)
    out["seed_random_intercept"] = coefs(ma, names)
    mc = smf.mixedlm("post_accuracy ~ chain*gen0 + C(tau_c) + cal_i", df, groups=df["cell"]).fit(reml=False)
    out["run_random_intercept_session_level"] = coefs(mc, names)
    # item level, equation (11): run (cell) random intercept, item-family variance component
    post = rows[rows.phase == "post"].copy()
    post["y"] = post.correct.astype(float); post["chain"] = (post.arm == "chain").astype(int); post["gen0"] = post.generation - 1
    post["tau_c"] = post.tau.astype(str); post["cal_i"] = post.cal.astype(int)
    try:
        mb = smf.mixedlm("y ~ chain*gen0 + C(tau_c) + cal_i", post, groups=post["cell"], re_formula="1",
                         vc_formula={"item": "0 + C(family)"}).fit(reml=False)
        out["run_and_item_family_item_level"] = coefs(mb, names)
        out["run_and_item_family_item_level"]["variance_components"] = {"run": float(mb.cov_re.iloc[0, 0]), "item_family": float(mb.vcomp[0]) if len(mb.vcomp) else None, "residual": float(mb.scale)}
    except Exception as e:
        out["run_and_item_family_item_level"] = {"error": str(e)}
    # equation (11) as specified: crossed random intercepts for the run (the session that produced
    # the responses) and the item, fitted as variance components under a single group; REML
    post["sess"] = post.cell + "|" + post.generation.astype(str) + "|" + post.arm; post["one"] = 1
    for name, vc in (("crossed_session_and_item", {"sess": "0 + C(sess)", "item": "0 + C(item_id)"}),
                     ("crossed_lineage_and_item", {"run": "0 + C(cell)", "item": "0 + C(item_id)"}),
                     ("item_only", {"item": "0 + C(item_id)"})):
        try:
            md = smf.mixedlm("y ~ chain*gen0 + C(tau_c) + cal_i", post, groups=post["one"], vc_formula=vc).fit(reml=True, method=["lbfgs", "powell"], maxiter=2000)
            out[name] = coefs(md, names)
            out[name]["variance_components"] = {k: float(v) for k, v in zip(list(vc), md.vcomp)}
            out[name]["residual"] = float(md.scale); out[name]["converged"] = bool(md.converged)
        except Exception as e:
            out[name] = {"error": str(e)}
    # item fixed effects with cluster-robust (lineage) standard errors, as a check on the crossed model
    m_fe = smf.ols("y ~ chain*gen0 + C(tau_c) + cal_i + C(item_id)", post).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(post.cell)[0]})
    ci_fe = m_fe.conf_int()
    out["item_fixed_effects_cluster_lineage"] = {"chain:gen0": {"estimate": float(m_fe.params["chain:gen0"]), "ci": [float(ci_fe.loc["chain:gen0", 0]), float(ci_fe.loc["chain:gen0", 1])], "p": float(m_fe.pvalues["chain:gen0"])}}
    # leave-one-out item base rate by arm and generation: were the post-test draws of equal difficulty?
    base = post.groupby("item_id").y.agg(["sum", "count"])
    loo = [((base.loc[r.item_id, "sum"] - r.y) / (base.loc[r.item_id, "count"] - 1)) if base.loc[r.item_id, "count"] > 1 else np.nan for r in post.itertuples()]
    post["loo"] = loo
    out["item_difficulty_drawn_by_arm_generation"] = {f"{arm}-{int(g)}": float(v) for (arm, g), v in post.groupby(["arm", "generation"]).loo.mean().items()}
    # transfer items with crossed session and item intercepts
    tr = rows[rows.phase == "transfer"].copy()
    tr["y"] = tr.correct.astype(float); tr["chain"] = (tr.arm == "chain").astype(int); tr["gen0"] = tr.generation - 1
    tr["tau_c"] = tr.tau.astype(str); tr["cal_i"] = tr.cal.astype(int); tr["sess"] = tr.cell + "|" + tr.generation.astype(str) + "|" + tr.arm; tr["one"] = 1
    try:
        mt2 = smf.mixedlm("y ~ chain*gen0 + C(tau_c) + cal_i", tr, groups=tr["one"], vc_formula={"sess": "0 + C(sess)", "item": "0 + C(item_id)"}).fit(reml=True, method=["lbfgs", "powell"], maxiter=2000)
        out["transfer_crossed_session_and_item"] = coefs(mt2, names); out["transfer_crossed_session_and_item"]["converged"] = bool(mt2.converged)
    except Exception as e:
        out["transfer_crossed_session_and_item"] = {"error": str(e)}
    # tau x generation in the chain arm, run random intercept
    ch = df[df.chain == 1].copy(); ch["tau_num"] = ch.tau
    m2 = smf.mixedlm("post_accuracy ~ gen0*tau_num + cal_i", ch, groups=ch["cell"]).fit(reml=False)
    out["chain_gen_x_tau_run_intercept"] = coefs(m2, ["gen0:tau_num"])["gen0:tau_num"]
    # tau main effect on acquisition, 0.95 minus 0.70, run random intercept
    df["t95"] = (np.isclose(df.tau, 0.95)).astype(int); df["t85"] = (np.isclose(df.tau, 0.85)).astype(int)
    m3 = smf.mixedlm("post_accuracy ~ t85 + t95 + cal_i + chain*gen0", df, groups=df["cell"]).fit(reml=False)
    out["tau_095_minus_070_run_intercept"] = coefs(m3, ["t95", "t85"])
    # transfer
    dft = sess.dropna(subset=["transfer_accuracy"]).copy(); dft["chain"] = (dft.arm == "chain").astype(int); dft["gen0"] = dft.generation - 1
    dft["tau_c"] = dft.tau.astype(str); dft["cal_i"] = dft.cal.astype(int)
    mt = smf.mixedlm("transfer_accuracy ~ chain*gen0 + C(tau_c) + cal_i", dft, groups=dft["cell"]).fit(reml=False)
    out["transfer_run_random_intercept"] = coefs(mt, names)
    return out


def bh(pvals: List[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p); ranked = p[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_full = np.empty(n); adj_full[order] = np.minimum(adj, 1.0)
    return adj_full.tolist(), (adj_full <= q).tolist()


def auc_boot_p(pl_a, pl_b, reps: int = 2000, seed: int = 0, window: int = None) -> Dict:
    """Paired bootstrap over test instances of AUC(a) - AUC(b); two-sided p from the bootstrap distribution."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    insts = sorted(set(i for i, _, _, _ in pl_a))
    def auc(pl, pick):
        sel = [(p, y) for i, pos, p, y in pl if i in pick and (window is None or pos < window)]
        y = np.array([t[1] for t in sel]); p = np.array([t[0] for t in sel])
        return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")
    full = set(insts); d0 = auc(pl_a, full) - auc(pl_b, full)
    boots = []
    for _ in range(reps):
        pick = defaultdict(int)
        for k in rng.randint(0, len(insts), len(insts)):
            pick[insts[k]] += 1
        # weighted resample: repeat instances by multiplicity
        a = [(i, pos, p, y) for i, pos, p, y in pl_a for _ in range(pick[i])]
        b = [(i, pos, p, y) for i, pos, p, y in pl_b for _ in range(pick[i])]
        boots.append(auc(a, set(pick)) - auc(b, set(pick)))
    boots = np.array([v for v in boots if not math.isnan(v)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"estimate": float(d0), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p": float(min(1.0, max(p, 1 / reps)))}


def ece_boot_p(p_raw, p_cal, y, reps: int = 4000, seed: int = 0) -> Dict:
    rng = np.random.RandomState(seed); p_raw = np.asarray(p_raw, float); p_cal = np.asarray(p_cal, float); y = np.asarray(y, int)
    d0 = ece(p_cal, y) - ece(p_raw, y); boots = []
    for _ in range(reps):
        k = rng.randint(0, len(y), len(y)); boots.append(ece(p_cal[k], y[k]) - ece(p_raw[k], y[k]))
    boots = np.array(boots); p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"estimate": float(d0), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "p": float(min(1.0, max(p, 1 / reps)))}


def mde_simulation(boot_sess: pd.DataFrame, n_seeds: int = 3, power: float = 0.8, alpha: float = 0.05, sims: int = 200, seed: int = 0) -> Dict:
    """Minimum detectable effects by simulation from the bootstrap pass's
    variance components (between-seed variance of cell means and within-cell
    residual variance of post-test accuracy), for the design as run: 18 cells
    (3 tau x 2 calibration x ... on Y) x 4 generations x 2 arms x n_seeds.
    Contrasts: (a) tau main effect, 0.95 minus 0.70; (b) chain slope per
    generation net of reference. Analysis model: mixed model with a seed
    random intercept, as fitted. Two-sided alpha."""
    import statsmodels.formula.api as smf
    rng = np.random.RandomState(seed)
    b = boot_sess.dropna(subset=["post_accuracy"])
    cells = b.groupby(["tau", "generation", "arm", "seed"]).post_accuracy.mean().reset_index()
    v_seed = float(cells.groupby(["tau", "generation", "arm"]).post_accuracy.var(ddof=1).mean())
    v_resid = float((b.post_accuracy - b.groupby(["tau", "generation", "arm"]).post_accuracy.transform("mean")).var(ddof=1))
    base = float(b.post_accuracy.mean())
    taus = [0.70, 0.85, 0.95]
    def simulate(delta_tau, delta_slope):
        recs = []
        for s in range(n_seeds):
            u = rng.normal(0, math.sqrt(v_seed))
            for tau in taus:
                for cal in (0, 1):
                    for gen in (1, 2, 3, 4):
                        for arm in ("chain", "reference"):
                            mu = base + u + delta_tau * (tau - 0.70) / 0.25 + (delta_slope * (gen - 1) if arm == "chain" else 0)
                            recs.append({"seed": s, "tau_c": str(tau), "cal_i": cal, "gen0": gen - 1, "chain": int(arm == "chain"),
                                         "t95": int(tau == 0.95), "t85": int(tau == 0.85), "y": min(1, max(0, mu + rng.normal(0, math.sqrt(v_resid))))})
        return pd.DataFrame(recs)
    def power_at(delta, which):
        hits = 0
        for _ in range(sims):
            df = simulate(delta if which == "tau" else 0, delta if which == "slope" else 0)
            try:
                if which == "tau":
                    m = smf.mixedlm("y ~ t85 + t95 + cal_i + chain*gen0", df, groups=df["seed"]).fit(reml=False); p = float(m.pvalues["t95"])
                else:
                    m = smf.mixedlm("y ~ chain*gen0 + C(tau_c) + cal_i", df, groups=df["seed"]).fit(reml=False); p = float(m.pvalues["chain:gen0"])
                hits += int(p < alpha)
            except Exception:
                pass
        return hits / sims
    out = {"variance_components_bootstrap_pass": {"between_seed_var_cell_means": v_seed, "within_cell_residual_var": v_resid, "mean_post": base},
           "n_seeds": n_seeds, "alpha_two_sided": alpha, "power": power, "sims_per_point": sims}
    for which, grid in (("tau", [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]), ("slope", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08])):
        curve = {str(d): power_at(d, which) for d in grid}
        mde = next((d for d in grid if curve[str(d)] >= power), None)
        out[f"mde_{which}"] = {"power_curve": curve, "mde": mde}
    return out


# ----------------------------------------------------------------------------
# 6. Survival restricted to attempted components, clustered by session
# ----------------------------------------------------------------------------
def survival_v2(sess: pd.DataFrame, rows: pd.DataFrame, perms: int = 2000, seed: int = 0) -> Dict:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    inst = _instruction_rows(rows)
    recs = []; never = defaultdict(int); total = defaultdict(int)
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)]
        sid = f"{s.cell}|{s.generation}|{s.arm}"
        for c, v in s["items_to_mastery"].items():
            total[s.tau] += 1
            att = int((g.component == c).sum())
            if att == 0:
                never[s.tau] += 1
                continue
            recs.append({"session": sid, "tau": s.tau, "cal": int(s.cal), "component": c,
                         "time": v if v is not None else att, "event": 0 if v is None else 1,
                         "censor_type": None if v is not None else ("attempt_cap" if att >= 8 else "session_ended")})
    df = pd.DataFrame(recs)
    out = {"by_tau": {}, "never_attempted": {str(t): {"never": int(never[t]), "of": int(total[t])} for t in sorted(total)}}
    for tau in sorted(df.tau.unique()):
        sub = df[df.tau == tau]; km = KaplanMeierFitter().fit(sub.time, sub.event); med = km.median_survival_time_
        out["by_tau"][str(tau)] = {"n_attempted": int(len(sub)), "censored": float(1 - sub.event.mean()),
                                   "censored_at_cap": int((sub.censor_type == "attempt_cap").sum()),
                                   "censored_session_ended": int((sub.censor_type == "session_ended").sum()),
                                   "median_items_to_mastery": None if med == float("inf") else float(med)}
    lr = multivariate_logrank_test(df.time, df.tau, df.event); out["logrank_p_naive"] = float(lr.p_value)
    # session-level permutation of the tau label for the log-rank statistic
    rng = np.random.RandomState(seed); stat0 = float(lr.test_statistic)
    sess_tau = df.groupby("session").tau.first(); keys = list(sess_tau.index); labels = sess_tau.values.copy(); hits = 0
    for _ in range(perms):
        rng.shuffle(labels); mp = dict(zip(keys, labels))
        st = multivariate_logrank_test(df.time, df.session.map(mp), df.event).test_statistic
        hits += int(st >= stat0)
    out["logrank_p_session_permutation"] = float((hits + 1) / (perms + 1))
    df["sid"] = pd.factorize(df.session)[0]
    cph = CoxPHFitter().fit(df[["time", "event", "tau", "cal", "sid"]], "time", "event", cluster_col="sid")
    hr = float(np.exp(cph.params_["tau"])); ci = cph.confidence_intervals_.loc["tau"]
    out["cox_clustered"] = {"hr_per_unit_tau": hr, "ci": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))], "p": float(cph.summary.loc["tau", "p"])}
    try:
        from lifelines.statistics import proportional_hazard_test
        ph = proportional_hazard_test(cph, df[["time", "event", "tau", "cal", "sid"]], time_transform="rank")
        out["ph_test_p"] = float(ph.summary["p"].min())
    except Exception:
        out["ph_test_p"] = None
    # competing-risk view: at each tau, share of components mastered, censored at cap, censored by session end, never reached
    out["disposition_by_tau"] = {}
    for tau in sorted(total):
        sub = df[df.tau == tau]
        out["disposition_by_tau"][str(tau)] = {"mastered": int(sub.event.sum()), "censored_at_cap": int((sub.censor_type == "attempt_cap").sum()),
                                               "censored_session_ended": int((sub.censor_type == "session_ended").sum()), "never_attempted": int(never[tau]), "total": int(total[tau])}
    return out


# ----------------------------------------------------------------------------
# 7. Dense grid on a pooled basis; knee degeneracy; change point
# ----------------------------------------------------------------------------
def dense_v2(dense: pd.DataFrame, floor: float, seed: int = 0) -> Dict:
    d = dense.copy(); d["gain"] = d.post_accuracy - floor
    taus = sorted(d.tau.unique())
    out = {"taus": taus, "gain_pooled": [float(d[d.tau == t].gain.mean()) for t in taus],
           "gain_chain": [float(d[(d.tau == t) & (d.arm == "chain")].gain.mean()) for t in taus],
           "gain_reference": [float(d[(d.tau == t) & (d.arm == "reference")].gain.mean()) for t in taus],
           "completion_pooled": [float(d[d.tau == t].completed.mean()) for t in taus],
           "all_six_mastered_pooled": [float((d[d.tau == t].n_mastered == 6).mean()) for t in taus],
           "n_per_level": [int((d.tau == t).sum()) for t in taus]}
    # the pre-registered estimator on arbitrary curves
    x = np.array(taus)
    def knee(y):
        coef = np.polyfit(x, y, 3); f = np.polyval(coef, x)
        d2 = np.abs(np.gradient(np.gradient(f, x), x)); return float(x[int(np.argmax(d2))])
    rng = np.random.RandomState(seed); returned = defaultdict(int)
    for _ in range(20000):
        returned[knee(rng.uniform(0, 1, len(x)))] += 1
    out["knee_estimator_on_random_curves"] = {str(k): int(v) for k, v in sorted(returned.items())}
    # change point: single breakpoint of a two-level step fitted by least squares to session gains, pooled arms
    def step(df):
        best = None
        for j in range(1, len(taus)):
            lo = df[df.tau < taus[j]].gain; hi = df[df.tau >= taus[j]].gain
            sse = float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, (taus[j - 1] + taus[j]) / 2, float(lo.mean()), float(hi.mean()))
        return best
    b = step(d); out["change_point"] = {"between": [max(t for t in taus if t < b[1]), min(t for t in taus if t > b[1])], "level_below": b[2], "level_above": b[3]}
    seeds = sorted(d.seed.unique()); boots = []
    for _ in range(2000):
        pick = rng.choice(seeds, len(seeds)); df = pd.concat([d[d.seed == s] for s in pick])
        boots.append(step(df)[1])
    out["change_point"]["bootstrap_over_seeds"] = {str(k): int(v) for k, v in sorted(pd.Series(boots).value_counts().items())}
    # bootstrap over sessions within level
    boots2 = []
    levels = {t: d[d.tau == t] for t in taus}
    for _ in range(2000):
        df = pd.concat([g.iloc[rng.randint(0, len(g), len(g))] for g in levels.values()])
        boots2.append(step(df)[1])
    out["change_point"]["bootstrap_over_sessions"] = {str(k): int(v) for k, v in sorted(pd.Series(boots2).value_counts().items())}
    return out


# ----------------------------------------------------------------------------
# 8. Persistence proxies, summaries, misconceptions
# ----------------------------------------------------------------------------
def persistence_v2(rows: pd.DataFrame) -> Dict:
    pers = rows[rows.phase.str.startswith("persist")]
    out = {}
    for arm in ("chain", "reference"):
        out[arm] = {}
        for ph, g in pers[pers.arm == arm].groupby("phase"):
            k, n = int(g.correct.sum()), int(len(g)); lo, hi = wilson(k, n)
            out[arm][ph] = {"n": n, "accuracy": k / n, "wilson": [lo, hi], "components": sorted(g.component.unique().tolist())}
    out["note"] = ("persist_boundary in the reference arm received the chain student's summary (chain.py, line 82 of the "
                   "released version) and is reported as invalid; the distance and boundary probes test the first "
                   "component mastered, which was C1 in every session where one was mastered.")
    return out


def export_summaries(runs_dir: str) -> List[Dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*-*"))):
        for p in sorted(glob.glob(os.path.join(d, "sigma_gen*.json"))):
            j = json.load(open(p)); out.append({"run": os.path.basename(d), "generation": j["generation"], "sigma": j["sigma"], "chars": len(j["sigma"]),
                                                "used_as_next_teacher_context": j["generation"] < 4})
    return out


# ----------------------------------------------------------------------------
# 9. Tether-scaling detector and transcript tabulations
# ----------------------------------------------------------------------------
SCALE_PATTERNS = [
    r"\b(shorter|smaller|lower|shortest|smallest|lesser)\s+(cairn|height|pile|one|of the two|of the\b)",
    r"\bthe\s+(shorter|smaller|lower|shortest|smallest)\b(?!\s+(result|than|answer|merged))",
    r"\bmin\s*\(", r"\bminimum\s+(height|of|cairn)", r"\bmin\s+height", r"\bmin\(q",
    r"per\s+(ten|10)\s+marks", r"every\s+(ten|10)\s+marks", r"each\s+(ten|10)\s+marks", r"for\s+each\s+(full\s+)?(ten|10)\b",
    r"(ten|10)\s+marks\s+of", r"/\s*10\s*\)", r"÷\s*10", r"divided by (ten|10)", r"one mark per ten", r"per ten\b", r"tenth of",
]
TETHER_SENT = re.compile(r"[^.\n]*\b(tether|\bb\b)[^.\n]*", re.I)


def tether_scaling_stated(text: str) -> bool:
    """True if any sentence that mentions the tether (or the symbol b) also
    carries a scaling cue: the shorter (smaller, minimum) cairn, or a per-ten-marks
    factor. Verbatim reproduction is not required; paraphrase counts."""
    for m in TETHER_SENT.finditer(text):
        s = m.group(0)
        if any(re.search(p, s, re.I) for p in SCALE_PATTERNS):
            return True
    return False


def merge_explanations(runs_dir: str, tag: str = "main") -> pd.DataFrame:
    recs = []
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{tag}-*"))):
        cell = os.path.basename(d); parts = cell.split("-")
        tau = float([p for p in parts if p.startswith("tau")][0][3:]); cal = "-cal-" in cell; seed = int(parts[-1][1:])
        for n in (1, 2, 3, 4):
            p = os.path.join(d, f"transcript_gen{n}.json")
            if not os.path.exists(p):
                continue
            t = json.load(open(p))
            for arm, label in (("chain", f"T{n-1}"), ("reference", f"R{n}")):
                inst = t.get(label)
                if inst is None:
                    continue
                msgs = inst["messages"]; reply = None; taught = []
                for i, m in enumerate(msgs):
                    if m["role"] == "user" and m["content"].startswith("TEACH:"):
                        name = m["content"].split("'")[1] if "'" in m["content"] else m["content"]
                        taught.append(name)
                        if "merg" in m["content"].lower() and i + 1 < len(msgs):
                            reply = msgs[i + 1]["content"]
                recs.append({"cell": cell, "tau": tau, "cal": cal, "seed": seed, "generation": n, "arm": arm, "teacher": label,
                             "n_teach_turns": len(taught), "merge_taught": reply is not None,
                             "scaling_stated": tether_scaling_stated(reply) if reply else None, "reply_chars": len(reply) if reply else 0})
    return pd.DataFrame(recs)


def omission_tables(df: pd.DataFrame) -> Dict:
    m = df[df.merge_taught].copy(); m["omitted"] = ~m.scaling_stated.astype(bool)
    out = {"n_merge_explanations": int(len(m)), "omitted_total": int(m.omitted.sum()), "by_arm": {}, "by_arm_generation": {}, "by_arm_seed": {}, "by_arm_generation_seed": {}}
    for arm, g in m.groupby("arm"):
        out["by_arm"][arm] = {"omitted": int(g.omitted.sum()), "of": int(len(g))}
    for (arm, gen), g in m.groupby(["arm", "generation"]):
        out["by_arm_generation"][f"{arm}-{int(gen)}"] = {"omitted": int(g.omitted.sum()), "of": int(len(g))}
    for (arm, seed), g in m.groupby(["arm", "seed"]):
        out["by_arm_seed"][f"{arm}-s{int(seed)}"] = {"omitted": int(g.omitted.sum()), "of": int(len(g))}
    for (arm, gen, seed), g in m.groupby(["arm", "generation", "seed"]):
        out["by_arm_generation_seed"][f"{arm}-{int(gen)}-s{int(seed)}"] = {"omitted": int(g.omitted.sum()), "of": int(len(g))}
    # chain teachers whose own teacher (previous generation) omitted vs stated
    return out


def summaries_tether(summaries: List[Dict]) -> Dict:
    mains = [s for s in summaries if s["run"].startswith("main-")]
    used = [s for s in mains if s["used_as_next_teacher_context"]]
    def mentions(s): return bool(re.search(r"\btether", s["sigma"], re.I))
    return {"n_main": len(mains), "n_used_as_context": len(used),
            "mention_tether_used": sum(mentions(s) for s in used), "scaling_stated_used": sum(tether_scaling_stated(s["sigma"]) for s in used),
            "mention_tether_all": sum(mentions(s) for s in mains), "scaling_stated_all": sum(tether_scaling_stated(s["sigma"]) for s in mains),
            "truncated_at_cap_chars_min_max": [min(s["chars"] for s in mains), max(s["chars"] for s in mains)]}


# ----------------------------------------------------------------------------
# 10. Calibration refits, P(L) and condition-blind re-scoring, KT baselines
# ----------------------------------------------------------------------------
def retrain_learners(boot_dirs: List[str], content, seed: int = 0) -> Dict:
    """Reproduce the frozen learner (GBDT + Platt on the bootstrap pass) and
    train the variants the reviewers asked for."""
    from .train import replay
    from .learner import train_gbdt, FEATURE_NAMES
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    X, y, ids, _ = replay(boot_dirs, content)
    inst = sorted(set(ids)); random.Random(seed).shuffle(inst)
    n = len(inst); tr, ca = set(inst[: int(0.6 * n)]), set(inst[int(0.6 * n): int(0.8 * n)])
    sel = lambda S: [i for i, k in enumerate(ids) if k in S]
    itr, ica, iev = sel(tr), sel(ca), sel(set(inst[int(0.8 * n):]))
    kci = {c.id: i for i, c in enumerate(content.components())}; fi = {f.family_id: i for i, f in enumerate(content.families())}
    ncond = FEATURE_NAMES.index("set_index")
    Xb = [x[:ncond] for x in X]           # condition-blind feature set (first five groups)
    full = train_gbdt([X[i] for i in itr], [y[i] for i in itr], [X[i] for i in ica], [y[i] for i in ica], "platt", kci, fi)
    blind = train_gbdt([Xb[i] for i in itr], [y[i] for i in itr], [Xb[i] for i in ica], [y[i] for i in ica], "platt", kci, fi)
    raw_cal = full.model.predict_proba([X[i] for i in ica])[:, 1]; y_cal = np.array([y[i] for i in ica])
    maps = {"platt_l2_default": full.calibrator,
            "platt_unpenalised": LogisticRegression(C=1e12, max_iter=1000).fit(raw_cal.reshape(-1, 1), y_cal),
            "isotonic": IsotonicRegression(out_of_bounds="clip").fit(raw_cal, y_cal)}
    grid = np.linspace(0, 1, 1001)
    def apply(mp, s):
        return mp.predict(s) if hasattr(mp, "transform") else mp.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]
    raw_ev = full.model.predict_proba([X[i] for i in iev])[:, 1]; y_ev = np.array([y[i] for i in iev])
    out = {"n_train": len(itr), "n_cal": len(ica), "n_eval": len(iev), "calibration_set_accuracy": float(y_cal.mean()), "maps": {}}
    for name, mp in maps.items():
        g = apply(mp, grid)
        out["maps"][name] = {"ceiling": float(g.max()), "floor": float(g.min()), "value_at_raw_0.5": float(apply(mp, [0.5])[0]),
                             "value_at_raw_0.9": float(apply(mp, [0.9])[0]), "value_at_raw_0.99": float(apply(mp, [0.99])[0]),
                             "ece_eval": ece(apply(mp, raw_ev), y_ev),
                             "reachable": {str(t): bool(g.max() >= t) for t in (0.70, 0.85, 0.95)},
                             "tau_prime": {str(t): (float(grid[np.where(g >= t)[0][0]]) if (g >= t).any() else None) for t in (0.70, 0.85, 0.95)}}
    out["ece_eval_raw"] = ece(raw_ev, y_ev)
    out["raw_score_range_on_calibration_set"] = [float(raw_cal.min()), float(raw_cal.max())]
    return {"full": full, "blind": blind, "maps": maps, "summary": out, "kci": kci, "fi": fi, "ncond": ncond}


def per_component_maps(rows: pd.DataFrame) -> Dict:
    """Per-component maps fitted on the uncalibrated arm's raw scores, three
    ways, with raw accuracy and score range beside every plateau."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna() & (~rows.cal)].copy()
    inst["p"] = inst.p_next.astype(float); inst["y"] = inst.correct.astype(int)
    grid = np.linspace(0, 1, 1001); out = {}
    for c, g in inst.groupby("component"):
        rec = {"n": int(len(g)), "raw_accuracy": float(g.y.mean()), "mean_raw_score": float(g.p.mean()), "score_range": [float(g.p.min()), float(g.p.max())]}
        if len(set(g.y)) < 2:
            rec["status"] = "one outcome class; no map"
        else:
            for name, mp in (("platt_l2_default", LogisticRegression().fit(g.p.values.reshape(-1, 1), g.y)),
                             ("platt_unpenalised", LogisticRegression(C=1e12, max_iter=1000).fit(g.p.values.reshape(-1, 1), g.y)),
                             ("isotonic", IsotonicRegression(out_of_bounds="clip").fit(g.p, g.y))):
                v = mp.predict(grid) if hasattr(mp, "transform") else mp.predict_proba(grid.reshape(-1, 1))[:, 1]
                vs = mp.predict(g.p) if hasattr(mp, "transform") else mp.predict_proba(g.p.values.reshape(-1, 1))[:, 1]
                rec[name] = {"plateau_on_grid": float(v.max()), "min_on_grid": float(v.min()),
                             "max_on_observed_scores": float(vs.max()), "min_on_observed_scores": float(vs.min()),
                             "tau_prime_grid": {str(t): (float(grid[np.where(v >= t)[0][0]]) if (v >= t).any() else None) for t in (0.70, 0.85, 0.95)},
                             "status_at_0.85": ("everywhere" if v.min() >= 0.85 else "unreachable" if v.max() < 0.85 else "interior"),
                             "status_at_0.85_observed": ("everywhere" if vs.min() >= 0.85 else "unreachable" if vs.max() < 0.85 else "interior"),
                             "status_at_0.95_observed": ("everywhere" if vs.min() >= 0.95 else "unreachable" if vs.max() < 0.95 else "interior")}
        out[c] = rec
    return out


def rescore(sess: pd.DataFrame, rows: pd.DataFrame, content, learners: Dict, boot_dirs: List[str]) -> Dict:
    """Replay every logged instruction history and re-score each mastery
    decision (a) with the condition-blind learner, (b) on BKT's latent P(L)
    fitted per component on the bootstrap pass."""
    from .learner import History, Interaction, features
    inst = _instruction_rows(rows); graph = content.graph()
    full, blind, ncond = learners["full"], learners["blind"], learners["ncond"]
    # BKT per component from the bootstrap pass (same grid search as the H1 baseline)
    from .analysis import _sequences
    _, brows = load(os.path.dirname(boot_dirs[0]), "bootstrap")
    seqs = _sequences(brows)
    def bkt_ll(params, seq_list):
        pi, pl, pg, ps = params; ll = 0.0
        for s in seq_list:
            L = pi
            for yv in s:
                pc = L * (1 - ps) + (1 - L) * pg; ll += math.log(pc if yv else 1 - pc)
                num = L * (1 - ps) if yv else L * ps; den = pc if yv else 1 - pc; L = num / den; L = L + (1 - L) * pl
        return ll
    grid = [(pi, pl, pg, ps) for pi in (0.1, 0.3, 0.5, 0.7) for pl in (0.1, 0.3, 0.5, 0.7) for pg in (0.1, 0.25, 0.4) for ps in (0.05, 0.15, 0.3)]
    bkt = {}
    for c in sorted({c for s in seqs.values() for c, _ in s}):
        sl = [[yv for cc, yv in s if cc == c] for s in seqs.values()]; sl = [s for s in sl if s]
        bkt[c] = max(grid, key=lambda p: bkt_ll(p, sl))
    out = {"bkt_params": {c: dict(zip(("p_init", "p_learn", "p_guess", "p_slip"), v)) for c, v in bkt.items()}, "decisions": 0,
           "blind_changes": 0, "blind_changes_by_cal": {}, "pl_mastery": {}, "pl_vs_logged": {}}
    changes = defaultdict(lambda: [0, 0]); pl_recs = []
    for _, s in sess.iterrows():
        g = inst[(inst.cell == s.cell) & (inst.generation == s.generation) & (inst.arm == s.arm)].sort_values("context_position")
        if len(g) == 0:
            continue
        cond = {"tau": float(s.tau), "calibration_flag": bool(s.cal), "generation": int(s.generation), "arm_index": 0 if s.arm == "chain" else 1}
        h = History(); c = "C1"; L = {}; first_pl = {}
        rs = list(g.itertuples())
        for i, r in enumerate(rs):
            if r.action in ("START", "ADVANCE", "REMEDIATE", "PRACTICE"):
                c = r.component
            h.interactions.append(Interaction(r.component, r.family, bool(r.correct), r.parse_status, int(r.latency_ms or 0), int(r.context_length or 0), len(h.interactions) + 1))
            # BKT latent P(L) update for the answered component
            pi, pl, pg, ps = bkt[r.component]; Lc = L.get(r.component, pi)
            pc = Lc * (1 - ps) + (1 - Lc) * pg
            num = Lc * (1 - ps) if r.correct else Lc * ps; den = pc if r.correct else 1 - pc
            Lc = num / den; Lc = Lc + (1 - Lc) * pl; L[r.component] = Lc
            if r.component not in first_pl and Lc >= s.tau:
                first_pl[r.component] = h.attempts(r.component)
            if i + 1 < len(rs):
                p_logged = float(rs[i + 1].p_next)
                f = features(h, c, graph, cond, learners["kci"], learners["fi"])
                raw_blind = float(blind.model.predict_proba([f[:ncond]])[0, 1])
                p_blind = float(blind.calibrator.predict_proba([[raw_blind]])[0, 1]) if s.cal else raw_blind
                out["decisions"] += 1
                key = f"cal={s.cal}"; changes[key][1] += 1
                if (p_logged >= s.tau) != (p_blind >= s.tau):
                    out["blind_changes"] += 1; changes[key][0] += 1
        for comp, k in s["items_to_mastery"].items():
            att = h.attempts(comp)
            if att:
                pl_recs.append({"tau": s.tau, "cal": s.cal, "component": comp, "logged_k": k, "pl_k": first_pl.get(comp), "attempts": att})
    out["blind_changes_by_cal"] = {k: {"changed": v[0], "of": v[1]} for k, v in changes.items()}
    df = pd.DataFrame(pl_recs)
    for tau, g in df.groupby("tau"):
        out["pl_mastery"][str(tau)] = {"attempted_components": int(len(g)), "mastered_logged": int(g.logged_k.notna().sum()),
                                       "mastered_under_PL": int(g.pl_k.notna().sum()),
                                       "median_items_logged": float(g.logged_k.median()) if g.logged_k.notna().any() else None,
                                       "median_items_under_PL": float(g.pl_k.median()) if g.pl_k.notna().any() else None,
                                       "same_decision": int(((g.logged_k.notna()) == (g.pl_k.notna())).sum())}
    return out


def kt_baselines_v2(rows: pd.DataFrame, content, learners: Dict, seed: int = 0) -> Dict:
    """As analysis.kt_baselines, with the GBDT restricted to the first five
    feature groups and DKT's first prediction set to the training base rate
    of the first item's component; AUC intervals by instance bootstrap; the
    H1 tests of the eight-contrast family."""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from .analysis import _sequences
    from .learner import History, Interaction, features, train_gbdt
    seqs = _sequences(rows); ids = sorted(seqs); rng = random.Random(seed); rng.shuffle(ids)
    n_tr = int(0.7 * len(ids)); train, test = ids[:n_tr], ids[n_tr:]
    kcs = sorted({c for s in seqs.values() for c, _ in s}); kidx = {c: i for i, c in enumerate(kcs)}
    preds = defaultdict(list)   # model -> [(inst, pos, p, y)]
    def bkt_ll(params, seq_list):
        pi, pl, pg, ps = params; ll = 0.0
        for s in seq_list:
            L = pi
            for yv in s:
                pc = L * (1 - ps) + (1 - L) * pg; ll += math.log(pc if yv else 1 - pc)
                num = L * (1 - ps) if yv else L * ps; den = pc if yv else 1 - pc; L = num / den; L = L + (1 - L) * pl
        return ll
    grid = [(pi, pl, pg, ps) for pi in (0.1, 0.3, 0.5, 0.7) for pl in (0.1, 0.3, 0.5, 0.7) for pg in (0.1, 0.25, 0.4) for ps in (0.05, 0.15, 0.3)]
    bkt = {}
    for c in kcs:
        sl = [[yv for cc, yv in seqs[i] if cc == c] for i in train]; sl = [s for s in sl if s]
        bkt[c] = max(grid, key=lambda p: bkt_ll(p, sl)) if sl else (0.3, 0.3, 0.25, 0.15)
    for i in test:
        L = {}
        for pos, (c, yv) in enumerate(seqs[i]):
            pi, pl, pg, ps = bkt[c]; Lc = L.get(c, pi); pc = Lc * (1 - ps) + (1 - Lc) * pg; preds["BKT"].append((i, pos, pc, yv))
            num = Lc * (1 - ps) if yv else Lc * ps; den = pc if yv else 1 - pc; L[c] = num / den + (1 - num / den) * pl
    def pfa_feats(seq):
        X, Y = [], []; succ = defaultdict(int); fail = defaultdict(int)
        for c, yv in seq:
            x = np.zeros(3 * len(kcs)); j = kidx[c]; x[j] = 1; x[len(kcs) + j] = succ[c]; x[2 * len(kcs) + j] = fail[c]; X.append(x); Y.append(yv)
            if yv: succ[c] += 1
            else: fail[c] += 1
        return X, Y
    Xtr, Ytr = [], []
    for i in train:
        X, Y = pfa_feats(seqs[i]); Xtr += X; Ytr += Y
    pfa = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(Ytr))
    for i in test:
        X, Y = pfa_feats(seqs[i])
        for pos, (x, yv) in enumerate(zip(X, Y)):
            preds["PFA"].append((i, pos, float(pfa.predict_proba([x])[0, 1]), yv))
    base = {c: float(np.mean([yv for i in train for cc, yv in seqs[i] if cc == c] or [0.5])) for c in kcs}
    try:
        import autograd.numpy as anp
        from autograd import grad
        H = 16; nin = 2 * len(kcs); nout = len(kcs); r0 = np.random.RandomState(0)
        params = {"Wz": r0.randn(nin, H) * 0.1, "Uz": r0.randn(H, H) * 0.1, "bz": np.zeros(H), "Wr": r0.randn(nin, H) * 0.1, "Ur": r0.randn(H, H) * 0.1, "br": np.zeros(H),
                  "Wh": r0.randn(nin, H) * 0.1, "Uh": r0.randn(H, H) * 0.1, "bh": np.zeros(H), "Wo": r0.randn(H, nout) * 0.1, "bo": np.zeros(nout)}
        def onehot(c, yv):
            v = np.zeros(nin); v[kidx[c] + (len(kcs) if yv else 0)] = 1; return v
        def forward(p, seq):
            h = anp.zeros(H); outs = []
            for c, yv in seq:
                x = onehot(c, yv)
                z = 1 / (1 + anp.exp(-(anp.dot(x, p["Wz"]) + anp.dot(h, p["Uz"]) + p["bz"]))); r = 1 / (1 + anp.exp(-(anp.dot(x, p["Wr"]) + anp.dot(h, p["Ur"]) + p["br"])))
                hh = anp.tanh(anp.dot(x, p["Wh"]) + anp.dot(r * h, p["Uh"]) + p["bh"]); h = (1 - z) * h + z * hh
                outs.append(1 / (1 + anp.exp(-(anp.dot(h, p["Wo"]) + p["bo"]))))
            return outs
        def loss(p, batch):
            tot = 0.0; n = 0
            for seq in batch:
                outs = forward(p, seq)
                for t in range(len(seq) - 1):
                    c, yv = seq[t + 1]; q = outs[t][kidx[c]]; tot = tot - (yv * anp.log(q + 1e-6) + (1 - yv) * anp.log(1 - q + 1e-6)); n += 1
            return tot / max(n, 1)
        g = grad(loss); m = {k: np.zeros_like(v) for k, v in params.items()}; v2 = {k: np.zeros_like(v) for k, v in params.items()}
        trseqs = [seqs[i] for i in train]; lr = 0.01
        for step in range(60):
            rng.shuffle(trseqs)
            for b in range(0, len(trseqs), 16):
                gr = g(params, trseqs[b:b + 16])
                for k in params:
                    m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2; params[k] = params[k] - lr * m[k] / (np.sqrt(v2[k]) + 1e-8)
        for i in test:
            outs = forward(params, seqs[i]); c0, y0 = seqs[i][0]
            preds["DKT"].append((i, 0, base[c0], y0))          # component-specific prior for the first prediction
            for t in range(len(seqs[i]) - 1):
                c, yv = seqs[i][t + 1]; preds["DKT"].append((i, t + 1, float(outs[t][kidx[c]]), yv))
    except Exception:
        preds["DKT"] = []
    graph = content.graph(); kci, fi, ncond = learners["kci"], learners["fi"], learners["ncond"]
    inst = rows[rows.phase == "instruction"].copy(); inst["inst_id"] = inst.cell + "|" + inst.generation.astype(str) + "|" + inst.arm
    def replay(ids_, blind):
        X, Y, P, I = [], [], [], []
        for i in ids_:
            g_ = inst[inst.inst_id == i].sort_values("context_position"); h = History()
            cond = {"tau": float(g_.tau.iloc[0]), "calibration_flag": bool(g_.cal.iloc[0]), "generation": int(g_.generation.iloc[0]), "arm_index": 0 if g_.arm.iloc[0] == "chain" else 1}
            for pos, (_, r) in enumerate(g_.iterrows()):
                f = features(h, r.component, graph, cond, kci, fi); X.append(f[:ncond] if blind else f); Y.append(int(r.correct)); P.append(pos); I.append(i)
                h.interactions.append(Interaction(r.component, r.family, bool(r.correct), r.parse_status, int(r.latency_ms or 0), int(r.context_length or 0), pos + 1))
        return X, Y, P, I
    for name, blind in (("GBDT", True), ("GBDT (with condition features)", False)):
        Xtr, Ytr, _, _ = replay(train, blind); Xte, Yte, Pte, Ite = replay(test, blind)
        gb = train_gbdt(Xtr, Ytr, kc_index=kci, family_index=fi)
        for i, pos, p, yv in zip(Ite, Pte, gb.model.predict_proba(Xte)[:, 1], Yte):
            preds[name].append((i, pos, float(p), yv))
    for i in test:
        for pos, (c, yv) in enumerate(seqs[i]):
            preds["Base rate"].append((i, pos, base[c], yv))
    def metrics(pl):
        if not pl:
            return None
        y = np.array([t[3] for t in pl]); p = np.array([t[2] for t in pl])
        out = {"n": int(len(pl)), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"), "rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
        for w in (5, 10, 20):
            sel = [t for t in pl if t[1] < w]; yy = np.array([t[3] for t in sel]); pp = np.array([t[2] for t in sel])
            out[f"auc_first{w}"] = float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else float("nan")
        # instance-bootstrap interval for the pooled AUC
        insts = sorted(set(t[0] for t in pl)); rr = np.random.RandomState(0); boots = []
        byi = defaultdict(list)
        for t in pl:
            byi[t[0]].append(t)
        for _ in range(1000):
            pick = [insts[k] for k in rr.randint(0, len(insts), len(insts))]
            sel = [t for i in pick for t in byi[i]]; yy = np.array([t[3] for t in sel]); pp = np.array([t[2] for t in sel])
            if len(set(yy)) > 1:
                boots.append(roc_auc_score(yy, pp))
        out["auc_ci_instance_bootstrap"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        return out
    res = {"n_train_instances": len(train), "n_test_instances": len(test), "models": {k: metrics(v) for k, v in preds.items()}}
    res["tests"] = {"DKT_minus_GBDT_first5": auc_boot_p(preds["DKT"], preds["GBDT"], window=5) if preds["DKT"] else None,
                    "best_human_KT_minus_base_rate_pooled": auc_boot_p(max((preds[k] for k in ("BKT", "PFA", "DKT") if preds[k]), key=lambda pl: metrics(pl)["auc"]), preds["Base rate"]),
                    "GBDT_minus_base_rate_pooled": auc_boot_p(preds["GBDT"], preds["Base rate"])}
    return res


# ----------------------------------------------------------------------------
# 11. Replication cell descriptives
# ----------------------------------------------------------------------------
def replication_v2(rep: pd.DataFrame, rrows: pd.DataFrame, floor_rep: float) -> Dict:
    if len(rep) == 0:
        return {}
    out = {"n_sessions": int(len(rep)), "floor": floor_rep, "gain_pooled": list(t_ci((rep.post_accuracy - floor_rep).tolist())[:3]),
           "by_arm_generation": {}, "transfer_by_arm_generation": {}, "completion": float(rep.completed.mean()), "all_six_mastered": float((rep.n_mastered == 6).mean()),
           "outcomes": rep.outcome.value_counts().to_dict()}
    for (arm, g), s in rep.groupby(["arm", "generation"]):
        out["by_arm_generation"][f"{arm}-{int(g)}"] = {"post": float(s.post_accuracy.mean()), "transfer": float(s.transfer_accuracy.mean()), "n": int(len(s))}
    inst = rrows[rrows.phase == "instruction"]
    out["parse"] = {k: int(v) for k, v in inst.parse_status.value_counts().items()}; out["unscorable_rate"] = float((inst.parse_status == "failure").mean())
    p = inst[inst.p_next.notna()]
    out["ece_calibrated_probabilities"] = ece(p.p_next.astype(float), p.correct.astype(int))
    out["per_component"] = {}
    for c, g in p.groupby("component"):
        out["per_component"][c] = {"n": int(len(g)), "accuracy": float(g.correct.mean()), "mean_p": float(g.p_next.astype(float).mean()), "max_p": float(g.p_next.astype(float).max())}
    post = rrows[rrows.phase == "post"]
    out["post_by_component"] = {c: float(g.correct.mean()) for c, g in post.groupby("component")}
    out["ignores_tether_post_by_arm_generation"] = {f"{arm}-{int(g)}": float((s.misconception == "ignores-tether").mean()) for (arm, g), s in post.groupby(["arm", "generation"])}
    out["decomposition"] = "not available: the replication ran on set Y only, so equations (5) to (7) cannot be formed; gain(Y) is reported"
    return out


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="analysis_v2")
    ap.add_argument("--floor", type=float, required=True); ap.add_argument("--floor-r", type=float, default=0.960); ap.add_argument("--floor-s", type=float, default=0.815)
    ap.add_argument("--floor-rep", type=float, default=0.075)
    ap.add_argument("--gate2", default='{"C1":1.0,"C2":1.0,"C3":1.0,"C4":1.0,"C5":0.904,"C6":1.0,"overall":0.984}', help="Gate 2 per-component accuracy under the instruction block (JSON)")
    ap.add_argument("--mde-sims", type=int, default=200)
    ap.add_argument("--skip", default="", help="comma-separated stages to skip (mde,kt,rescore)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from .content import load_content
    content = load_content("Y"); skip = set(a.skip.split(",")) if a.skip else set()
    sess, rows = load(a.runs, "main"); controls, crows = load(a.runs, "control"); dense, _ = load(a.runs, "dense")
    rep, rrows = load(a.runs, "replication"); boot, _ = load(a.runs, "bootstrap")
    floors = {"Y": a.floor, "R": a.floor_r, "S": a.floor_s}
    res = {"n_sessions": int(len(sess))}
    res["decomposition"] = decomposition(sess, controls, floors)
    res["policy_vs_first_correct"] = policy_vs_first_correct(sess, rows)
    res["termination_practice"] = termination_and_practice(sess, rows)
    res["by_component"] = by_component(sess, rows, json.loads(a.gate2))
    res["generation1_noise"] = generation1_noise(sess)
    res["intervals"] = pooled_intervals(sess, a.floor)
    res["mixed_models"] = mixed_models(sess, rows)
    res["survival"] = survival_v2(sess, rows)
    res["dense"] = dense_v2(dense, a.floor) if len(dense) else {}
    res["persistence"] = persistence_v2(rows)
    summaries = export_summaries(a.runs)
    with open(os.path.join(a.out, "summaries_sigma.json"), "w") as f:
        json.dump(summaries, f, indent=1)
    res["summaries"] = summaries_tether(summaries)
    mx = merge_explanations(a.runs, "main"); mx.to_csv(os.path.join(a.out, "merge_explanations_main.csv"), index=False)
    res["tether_omission"] = omission_tables(mx)
    if len(rep):
        mr = merge_explanations(a.runs, "replication"); mr.to_csv(os.path.join(a.out, "merge_explanations_replication.csv"), index=False)
        res["tether_omission_replication"] = omission_tables(mr)
    res["per_component_maps"] = per_component_maps(rows)
    boot_dirs = sorted(glob.glob(os.path.join(a.runs, "bootstrap-*")))
    learners = retrain_learners(boot_dirs, content)
    res["policy_map_refits"] = learners["summary"]
    if "rescore" not in skip:
        res["rescore"] = rescore(sess, rows, content, learners, boot_dirs)
    if "kt" not in skip:
        res["kt_baselines"] = kt_baselines_v2(rows, content, learners)
    res["replication"] = replication_v2(rep, rrows, a.floor_rep)
    res["seed_variance_post"] = seed_variance(sess, "post_accuracy")
    if "mde" not in skip and len(boot):
        res["mde"] = mde_simulation(boot, sims=a.mde_sims)
    # calibration tests for the eight-contrast family
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna()].copy(); inst["p"] = inst.p_next.astype(float); inst["y"] = inst.correct.astype(int)
    raw = inst[~inst.cal]; key = raw.cell + "|" + raw.generation.astype(str) + "|" + raw.arm
    units = sorted(set(key)); rng = random.Random(0); rng.shuffle(units); fit_units = set(units[: len(units) // 2])
    fit = raw[key.isin(fit_units)]; ev = raw[~key.isin(fit_units)]
    pl = platt_fit(fit.p, fit.y)
    res["calibration_tests"] = {"delta_ece_platt": ece_boot_p(ev.p.values, pl.predict_proba(ev.p.values.reshape(-1, 1))[:, 1], ev.y.values)}
    # transported calibration: generation-3 map on generation 1 vs generation-1 map on generation 1
    g1 = raw[raw.generation == 1]; g3 = raw[raw.generation == 3]; m1 = platt_fit(g1.p, g1.y); m3 = platt_fit(g3.p, g3.y)
    res["calibration_tests"]["transported_g3_to_g1_minus_g1_to_g1"] = ece_boot_p(m1.predict_proba(g1.p.values.reshape(-1, 1))[:, 1], m3.predict_proba(g1.p.values.reshape(-1, 1))[:, 1], g1.y.values)
    # the eight-contrast family
    mm = res["mixed_models"]; kt = res.get("kt_baselines", {}).get("tests", {})
    fam = [
        ("H1a: pooled AUC, best human-derived KT minus base rate", kt.get("best_human_KT_minus_base_rate_pooled", {}).get("p")),
        ("H1b: AUC first 5 exchanges, DKT minus GBDT", kt.get("DKT_minus_GBDT_first5", {}).get("p")),
        ("H2: ECE, Platt minus raw, held-out units", res["calibration_tests"]["delta_ece_platt"]["p"]),
        ("H3a: tau 0.95 minus 0.70 on post-test accuracy", mm["tau_095_minus_070_run_intercept"]["t95"]["p"]),
        ("H3b: Cox hazard ratio per unit tau (clustered)", res["survival"]["cox_clustered"]["p"]),
        ("H5a: chain slope per generation net of reference (crossed session and item intercepts)", mm.get("crossed_session_and_item", {}).get("chain:gen0", {}).get("p", 1.0)),
        ("H5b: tau x generation in the chain arm", mm["chain_gen_x_tau_run_intercept"]["p"]),
        ("H6: transported-map ECE minus same-slice ECE (generation 1)", res["calibration_tests"]["transported_g3_to_g1_minus_g1_to_g1"]["p"]),
    ]
    ps = [p if p is not None else 1.0 for _, p in fam]; adj, rej = bh(ps)
    res["bh_family"] = [{"contrast": n, "p_raw": p, "p_bh": q, "significant_q05": r} for (n, p), q, r in zip(fam, adj, rej)]
    with open(os.path.join(a.out, "revision_results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(a.out, "table8_rows_t.tex"), "w") as f:
        f.write(table8_t(sess, a.floor, rows))
    with open(os.path.join(a.out, "table8_control_rows_t.tex"), "w") as f:
        f.write(table8_t(controls, a.floor, crows, floors=floors))
    if len(rep):
        with open(os.path.join(a.out, "table8_replication_rows_t.tex"), "w") as f:
            f.write(table8_t(rep, a.floor_rep, rrows, floors={"Y": a.floor_rep}))
    print(json.dumps({k: res[k] for k in ("decomposition", "policy_vs_first_correct", "generation1_noise", "tether_omission", "bh_family")}, indent=1, default=float))


if __name__ == "__main__":
    main()

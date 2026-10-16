"""Confirmatory analysis over run directories (Section V-E of the manuscript).

  python -m harness.analysis --runs runs --out analysis --floor 0.064

Reads main-*, control-*, dense-*, replication-* run directories, computes
every pre-registered quantity, and writes results.json, LaTeX rows for
Tables 8 and 9, Figures 2 to 4 as PDF, and the transcript-coding sample.
Bounds are labelled as bounds; nothing here reports a point where the
manuscript promises a bound.
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load(runs_dir: str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sess, rows = [], []
    for d in sorted(glob.glob(os.path.join(runs_dir, f"{tag}-*"))):
        if not os.path.exists(os.path.join(d, "run_summary.json")):
            continue
        cell = os.path.basename(d)
        parts = cell.split("-")
        content = parts[1]
        tau = float([p for p in parts if p.startswith("tau")][0][3:])
        cal = "-cal-" in cell
        seed = int(parts[-1][1:])
        s = json.load(open(os.path.join(d, "run_summary.json")))
        for res in s["results"]:
            r = dict(res); r.update({"cell": cell, "set": content, "tau": tau, "cal": cal, "seed": seed})
            r["n_mastered"] = len(res["mastered"])
            r["completed"] = res["outcome"] == "mastered"
            sess.append(r)
        for line in open(os.path.join(d, "telemetry.jsonl")):
            r = json.loads(line); r.update({"cell": cell, "set": content, "tau": tau, "cal": cal, "seed": seed}); rows.append(r)
    return pd.DataFrame(sess), pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mean_ci(x: List[float]) -> Tuple[float, float, float]:
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    m = x.mean(); se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return (m, m - 1.96 * se, m + 1.96 * se)


def seed_variance(sess: pd.DataFrame, value: str) -> float:
    """Between-seed variance of cell means, equation (12): mean over cells of
    the variance across seeds of the cell mean."""
    cells = sess.groupby(["set", "tau", "cal", "generation", "arm", "seed"])[value].mean().reset_index()
    v = cells.groupby(["set", "tau", "cal", "generation", "arm"])[value].var(ddof=1)
    return float(v.mean())


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def ece(p, y, bins=10) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1.0))
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def platt_fit(p, y):
    from sklearn.linear_model import LogisticRegression
    p = np.asarray(p, float).reshape(-1, 1); y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return None
    return LogisticRegression().fit(p, y)


def tau_prime(model, tau: float, grid=np.linspace(0.0, 1.0, 1001)) -> Dict:
    """Smallest raw score whose calibrated value reaches tau, equation (11).
    Status 'interior' when the map crosses tau inside (0, 1); 'everywhere'
    when it is at or above tau at every score; 'unreachable' when its
    maximum over [0, 1] stays below tau; 'undefined' when the slice has one
    outcome class and no map can be fitted."""
    if model is None:
        return {"tau_prime": float("nan"), "status": "undefined", "plateau": float("nan")}
    g = model.predict_proba(grid.reshape(-1, 1))[:, 1]
    hit = np.where(g >= tau)[0]
    plateau = float(g.max())
    if not len(hit):
        return {"tau_prime": float("nan"), "status": "unreachable", "plateau": plateau}
    x = float(grid[hit[0]])
    return {"tau_prime": x, "status": "everywhere" if hit[0] == 0 else "interior", "plateau": plateau}


def calibration_analysis(rows: pd.DataFrame) -> Dict:
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna()].copy()
    inst["p"] = inst.p_next.astype(float); inst["y"] = inst.correct.astype(int)
    out = {}
    for cal in (False, True):
        sub = inst[inst.cal == cal]
        out["calibrated" if cal else "uncalibrated"] = {"n": int(len(sub)), "mean_p": float(sub.p.mean()), "accuracy": float(sub.y.mean()), "ece": ece(sub.p, sub.y)}
    raw = inst[~inst.cal]                       # raw GBDT scores
    # pooled maps fitted on raw scores (instance-level split by session cell/generation/arm)
    key = raw.cell + "|" + raw.generation.astype(str) + "|" + raw.arm
    units = sorted(set(key)); rng = random.Random(0); rng.shuffle(units)
    fit_units = set(units[: len(units) // 2])
    fit = raw[key.isin(fit_units)]; ev = raw[~key.isin(fit_units)]
    pl = platt_fit(fit.p, fit.y)
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(fit.p, fit.y) if len(fit) else None
    out["pooled"] = {"ece_raw": ece(ev.p, ev.y),
                     "ece_platt": ece(pl.predict_proba(ev.p.values.reshape(-1, 1))[:, 1], ev.y) if pl else float("nan"),
                     "ece_isotonic": ece(iso.predict(ev.p), ev.y) if iso else float("nan"), "n_eval": int(len(ev))}
    # slices: generation, component, interaction-count bin
    raw = raw.copy()
    raw["ibin"] = pd.cut(raw.context_position.astype(float), [0, 6, 12, 20, 200], labels=["1-6", "7-12", "13-20", "21+"]).astype(str)
    slices = {"generation": raw.generation.astype(str), "component": raw.component, "interaction_bin": raw.ibin}
    out["slices"] = {}; out["tau_prime"] = {}; out["transported"] = {}
    for name, col in slices.items():
        levels = sorted(set(col))
        maps = {}
        for lv in levels:
            sub = raw[col == lv]
            maps[lv] = platt_fit(sub.p, sub.y)
            out["slices"][f"{name}={lv}"] = {"n": int(len(sub)), "ece_raw": ece(sub.p, sub.y)}
        for tau in (0.70, 0.85, 0.95):
            tp = {lv: tau_prime(maps[lv], tau) for lv in levels}
            interior = [v["tau_prime"] for v in tp.values() if v["status"] == "interior"]
            statuses = {st: sum(1 for v in tp.values() if v["status"] == st) for st in ("interior", "everywhere", "unreachable", "undefined")}
            out["tau_prime"][f"{name}@{tau}"] = {"by_slice": tp, "statuses": statuses,
                                                  "interior_range": (max(interior) - min(interior)) if len(interior) > 1 else float("nan"),
                                                  "plateaus": {lv: v["plateau"] for lv, v in tp.items()}}
        tr = {}
        for a in levels:
            for b in levels:
                if maps[a] is None:
                    continue
                sub = raw[col == b]
                tr[f"{a}->{b}"] = ece(maps[a].predict_proba(sub.p.values.reshape(-1, 1))[:, 1], sub.y)
        out["transported"][name] = tr
    return out


# ----------------------------------------------------------------------------
# Generational and threshold models
# ----------------------------------------------------------------------------
def generation_model(sess: pd.DataFrame, value: str) -> Dict:
    import statsmodels.formula.api as smf
    df = sess.dropna(subset=[value]).copy()
    df["chain"] = (df.arm == "chain").astype(int); df["gen0"] = df.generation - 1
    df["tau_c"] = df.tau.astype(str); df["cal_i"] = df.cal.astype(int)
    m = smf.mixedlm(f"{value} ~ chain*gen0 + C(tau_c) + cal_i", df, groups=df["seed"]).fit(reml=False)
    ci = m.conf_int()
    def coef(name):
        return {"estimate": float(m.params[name]), "ci": [float(ci.loc[name, 0]), float(ci.loc[name, 1])], "p": float(m.pvalues[name])}
    out = {"reference_slope_per_generation": coef("gen0"), "chain_minus_reference_slope": coef("chain:gen0"), "chain_intercept_shift": coef("chain"), "n": int(len(df))}
    # H5 interaction tau x generation in the chain arm
    ch = df[df.chain == 1].copy(); ch["tau_num"] = ch.tau
    m2 = smf.mixedlm(f"{value} ~ gen0*tau_num + cal_i", ch, groups=ch["seed"]).fit(reml=False)
    ci2 = m2.conf_int()
    out["chain_gen_x_tau"] = {"estimate": float(m2.params["gen0:tau_num"]), "ci": [float(ci2.loc["gen0:tau_num", 0]), float(ci2.loc["gen0:tau_num", 1])], "p": float(m2.pvalues["gen0:tau_num"])}
    return out


def survival_analysis(sess: pd.DataFrame) -> Dict:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    recs = []
    for _, s in sess.iterrows():
        for c, v in s["items_to_mastery"].items():
            recs.append({"tau": s.tau, "seed": s.seed, "component": c, "arm": s.arm, "cal": int(s.cal),
                         "time": v if v is not None else 8, "event": 0 if v is None else 1})
    df = pd.DataFrame(recs)
    out = {"by_tau": {}}
    for tau in sorted(df.tau.unique()):
        sub = df[df.tau == tau]; km = KaplanMeierFitter().fit(sub.time, sub.event)
        med = km.median_survival_time_
        out["by_tau"][str(tau)] = {"n": int(len(sub)), "censored": float(1 - sub.event.mean()), "median_items_to_mastery": None if med == float("inf") else float(med)}
    lr = multivariate_logrank_test(df.time, df.tau, df.event)
    out["logrank_p"] = float(lr.p_value)
    cph = CoxPHFitter().fit(df[["time", "event", "tau", "cal"]], "time", "event", cluster_col=None)
    hr = float(np.exp(cph.params_["tau"])); ci = cph.confidence_intervals_.loc["tau"]
    out["cox_hazard_ratio_per_unit_tau"] = {"hr": hr, "ci": [float(np.exp(ci.iloc[0])), float(np.exp(ci.iloc[1]))], "p": float(cph.summary.loc["tau", "p"])}
    try:
        from lifelines.statistics import proportional_hazard_test
        ph = proportional_hazard_test(cph, df[["time", "event", "tau", "cal"]], time_transform="rank")
        out["ph_test_p"] = float(ph.summary["p"].min())
    except Exception as e:
        out["ph_test_p"] = None
    return out


def threshold_curve(dense: pd.DataFrame, floor: float) -> Dict:
    """Acquisition per item on the dense grid; knee = tau at which the second
    difference of the fitted curve is largest in magnitude; bootstrap over seeds.

    v2: both arms pooled, as the figure caption states (reviewer 2, item 6; v1
    used the chain arm only). The pre-registered knee estimator is kept for the
    record; revision.dense_v2 shows that on a 12-point grid it can only return
    0.78 or 0.96 and adds a change-point reading."""
    if len(dense) == 0:
        return {}
    d = dense.copy()
    d["gain"] = d.post_accuracy - floor; d["api"] = d.gain / d.interactions.clip(lower=1)
    taus = sorted(d.tau.unique())
    def knee(df):
        y = np.array([df[df.tau == t].api.mean() for t in taus]); x = np.array(taus)
        coef = np.polyfit(x, y, 3); f = np.polyval(coef, x)
        d2 = np.abs(np.gradient(np.gradient(f, x), x))
        return float(x[int(np.argmax(d2))])
    seeds = sorted(d.seed.unique()); rng = random.Random(0); boots = []
    for _ in range(500):
        pick = [rng.choice(seeds) for _ in seeds]
        boots.append(knee(pd.concat([d[d.seed == s] for s in pick])))
    return {"taus": taus, "acquisition_per_item": [float(d[d.tau == t].api.mean()) for t in taus],
            "gain": [float(d[d.tau == t].gain.mean()) for t in taus],
            "completion": [float(d[d.tau == t].completed.mean()) for t in taus],
            "interactions": [float(d[d.tau == t].interactions.mean()) for t in taus],
            "gain_by_arm": {arm: [float(d[(d.tau == t) & (d.arm == arm)].gain.mean()) for t in taus] for arm in ("chain", "reference")},
            "knee": knee(d), "knee_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


# ----------------------------------------------------------------------------
# Knowledge-tracing baselines (H1)
# ----------------------------------------------------------------------------
def _sequences(rows: pd.DataFrame):
    inst = rows[rows.phase == "instruction"].copy()
    inst["inst_id"] = inst.cell + "|" + inst.generation.astype(str) + "|" + inst.arm
    seqs = {}
    for k, g in inst.groupby("inst_id"):
        g = g.sort_values("context_position")
        seqs[k] = list(zip(g.component.tolist(), g.correct.astype(int).tolist()))
    return seqs


def kt_baselines(rows: pd.DataFrame, content) -> Dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    seqs = _sequences(rows); ids = sorted(seqs); rng = random.Random(0); rng.shuffle(ids)
    n_tr = int(0.7 * len(ids)); train, test = ids[:n_tr], ids[n_tr:]
    kcs = sorted({c for s in seqs.values() for c, _ in s}); kidx = {c: i for i, c in enumerate(kcs)}
    preds = defaultdict(list)   # model -> [(pos, p, y)]

    # BKT: per-KC parameters by grid search on train log-likelihood
    def bkt_ll(params, seq_list):
        pi, pl, pg, ps = params; ll = 0.0
        for s in seq_list:
            L = pi
            for y in s:
                pc = L * (1 - ps) + (1 - L) * pg
                ll += math.log(pc if y else 1 - pc)
                num = L * (1 - ps) if y else L * ps
                den = pc if y else 1 - pc
                L = num / den; L = L + (1 - L) * pl
        return ll
    bkt = {}
    grid = [(pi, pl, pg, ps) for pi in (0.1, 0.3, 0.5, 0.7) for pl in (0.1, 0.3, 0.5, 0.7) for pg in (0.1, 0.25, 0.4) for ps in (0.05, 0.15, 0.3)]
    for c in kcs:
        seq_list = [[y for cc, y in seqs[i] if cc == c] for i in train]
        seq_list = [s for s in seq_list if s]
        bkt[c] = max(grid, key=lambda p: bkt_ll(p, seq_list)) if seq_list else (0.3, 0.3, 0.25, 0.15)
    for i in test:
        L = {}
        for pos, (c, y) in enumerate(seqs[i]):
            pi, pl, pg, ps = bkt[c]; Lc = L.get(c, pi)
            pc = Lc * (1 - ps) + (1 - Lc) * pg; preds["BKT"].append((pos, pc, y))
            num = Lc * (1 - ps) if y else Lc * ps; den = pc if y else 1 - pc
            L[c] = num / den + (1 - num / den) * pl

    # PFA: logistic regression on per-KC success/failure counts
    def pfa_feats(seq):
        X, Y = [], []; succ = defaultdict(int); fail = defaultdict(int)
        for c, y in seq:
            x = np.zeros(3 * len(kcs)); j = kidx[c]; x[j] = 1; x[len(kcs) + j] = succ[c]; x[2 * len(kcs) + j] = fail[c]
            X.append(x); Y.append(y)
            if y: succ[c] += 1
            else: fail[c] += 1
        return X, Y
    Xtr, Ytr = [], []
    for i in train:
        X, Y = pfa_feats(seqs[i]); Xtr += X; Ytr += Y
    pfa = LogisticRegression(max_iter=1000).fit(np.array(Xtr), np.array(Ytr))
    for i in test:
        X, Y = pfa_feats(seqs[i])
        for pos, (x, y) in enumerate(zip(X, Y)):
            preds["PFA"].append((pos, float(pfa.predict_proba([x])[0, 1]), y))

    # DKT: one-layer GRU (hidden 16) trained by autograd Adam on next-response prediction
    try:
        import autograd.numpy as anp
        from autograd import grad
        H = 16; nin = 2 * len(kcs); nout = len(kcs); r0 = np.random.RandomState(0)
        params = {"Wz": r0.randn(nin, H) * 0.1, "Uz": r0.randn(H, H) * 0.1, "bz": np.zeros(H),
                  "Wr": r0.randn(nin, H) * 0.1, "Ur": r0.randn(H, H) * 0.1, "br": np.zeros(H),
                  "Wh": r0.randn(nin, H) * 0.1, "Uh": r0.randn(H, H) * 0.1, "bh": np.zeros(H),
                  "Wo": r0.randn(H, nout) * 0.1, "bo": np.zeros(nout)}
        def onehot(c, y):
            v = np.zeros(nin); v[kidx[c] + (len(kcs) if y else 0)] = 1; return v
        def forward(p, seq):
            h = anp.zeros(H); outs = []
            for c, y in seq:
                x = onehot(c, y)
                z = 1 / (1 + anp.exp(-(anp.dot(x, p["Wz"]) + anp.dot(h, p["Uz"]) + p["bz"])))
                r = 1 / (1 + anp.exp(-(anp.dot(x, p["Wr"]) + anp.dot(h, p["Ur"]) + p["br"])))
                hh = anp.tanh(anp.dot(x, p["Wh"]) + anp.dot(r * h, p["Uh"]) + p["bh"])
                h = (1 - z) * h + z * hh
                outs.append(1 / (1 + anp.exp(-(anp.dot(h, p["Wo"]) + p["bo"]))))
            return outs
        def loss(p, batch):
            tot = 0.0; n = 0
            for seq in batch:
                outs = forward(p, seq)
                for t in range(len(seq) - 1):
                    c, y = seq[t + 1]; q = outs[t][kidx[c]]
                    tot = tot - (y * anp.log(q + 1e-6) + (1 - y) * anp.log(1 - q + 1e-6)); n += 1
            return tot / max(n, 1)
        g = grad(loss); m = {k: np.zeros_like(v) for k, v in params.items()}; v2 = {k: np.zeros_like(v) for k, v in params.items()}
        trseqs = [seqs[i] for i in train]; lr = 0.01
        for step in range(60):
            rng.shuffle(trseqs)
            for b in range(0, len(trseqs), 16):
                gr = g(params, trseqs[b:b + 16])
                for k in params:
                    m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
                    params[k] = params[k] - lr * m[k] / (np.sqrt(v2[k]) + 1e-8)
        base0 = {c: np.mean([y for i in train for cc, y in seqs[i] if cc == c] or [0.5]) for c in kcs}
        for i in test:
            outs = forward(params, seqs[i])
            # v2: the first prediction uses the component's training base rate, not a constant 0.5 (reviewer 2, item 10d)
            preds["DKT"].append((0, float(base0[seqs[i][0][0]]), seqs[i][0][1]))
            for t in range(len(seqs[i]) - 1):
                c, y = seqs[i][t + 1]; preds["DKT"].append((t + 1, float(outs[t][kidx[c]]), y))
    except Exception as e:
        preds["DKT"] = []

    # GBDT (Table 6 features, replayed) and the fixed-sequence control (train base rate per KC)
    from .learner import History, Interaction, features, train_gbdt, N_CONDITION_BLIND
    graph = content.graph(); kci = {c.id: i for i, c in enumerate(content.components())}; fi = {f.family_id: i for i, f in enumerate(content.families())}
    inst = rows[rows.phase == "instruction"].copy(); inst["inst_id"] = inst.cell + "|" + inst.generation.astype(str) + "|" + inst.arm
    def replay(ids_):
        X, Y, P = [], [], []
        for i in ids_:
            g_ = inst[inst.inst_id == i].sort_values("context_position"); h = History()
            cond = {"tau": float(g_.tau.iloc[0]), "calibration_flag": bool(g_.cal.iloc[0]), "generation": int(g_.generation.iloc[0]), "arm_index": 0 if g_.arm.iloc[0] == "chain" else 1}
            for pos, (_, r) in enumerate(g_.iterrows()):
                # v2: the H1 comparison uses the first five feature groups only, as Section IV-E states (reviewer 2, item 2d)
                X.append(features(h, r.component, graph, cond, kci, fi)[:N_CONDITION_BLIND]); Y.append(int(r.correct)); P.append(pos)
                h.interactions.append(Interaction(r.component, r.family, bool(r.correct), r.parse_status, int(r.latency_ms or 0), int(r.context_length or 0), pos + 1))
        return X, Y, P
    Xtr, Ytr, _ = replay(train); Xte, Yte, Pte = replay(test)
    gb = train_gbdt(Xtr, Ytr, kc_index=kci, family_index=fi)
    for pos, p, y in zip(Pte, gb.model.predict_proba(Xte)[:, 1], Yte):
        preds["GBDT"].append((pos, float(p), y))
    base = {c: np.mean([y for i in train for cc, y in seqs[i] if cc == c] or [0.5]) for c in kcs}
    for i in test:
        for pos, (c, y) in enumerate(seqs[i]):
            preds["Base rate"].append((pos, float(base[c]), y))   # v1 called this "Fixed-sequence"; it is a per-component base-rate predictor, not the fixed-sequence control of Section IV-E

    def metrics(pl):
        if not pl:
            return None
        y = np.array([t[2] for t in pl]); p = np.array([t[1] for t in pl])
        out = {"n": int(len(pl)), "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"), "rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
        for w in (5, 10, 20):
            sel = [t for t in pl if t[0] < w]; yy = np.array([t[2] for t in sel]); pp = np.array([t[1] for t in sel])
            out[f"auc_first{w}"] = float(roc_auc_score(yy, pp)) if len(set(yy)) > 1 else float("nan")
        return out
    return {"n_train_instances": len(train), "n_test_instances": len(test), "models": {k: metrics(v) for k, v in preds.items()}}


# ----------------------------------------------------------------------------
# Misconceptions, persistence, coding sample
# ----------------------------------------------------------------------------
def misconceptions(rows: pd.DataFrame) -> Dict:
    post = rows[rows.phase == "post"]
    out = {}
    for arm in ("chain", "reference"):
        out[arm] = {}
        for g in sorted(post.generation.unique()):
            sub = post[(post.arm == arm) & (post.generation == g)]
            counts = sub.misconception.value_counts()
            out[arm][str(int(g))] = {"n": int(len(sub)), "rates": {k: float(v / len(sub)) for k, v in counts.items()}}
    inst = rows[(rows.phase == "instruction") & (~rows.correct.astype(bool))]
    out["instruction_totals"] = {k: int(v) for k, v in inst.misconception.value_counts().items()}
    return out


def persistence(rows: pd.DataFrame) -> Dict:
    pers = rows[rows.phase.str.startswith("persist")]
    return {ph: {"n": int((pers.phase == ph).sum()), "accuracy": float(pers[pers.phase == ph].correct.astype(float).mean())} for ph in sorted(pers.phase.unique())}


def coding_sample(sess: pd.DataFrame, per: int = 5) -> List[Dict]:
    out = []
    for arm in ("chain", "reference"):
        for g in sorted(sess.generation.unique()):
            sub = sess[(sess.arm == arm) & (sess.generation == g)].sort_values(["seed", "tau", "cal"]).head(per)
            for _, s in sub.iterrows():
                out.append({"arm": arm, "generation": int(g), "cell": s.cell, "transcript": f"transcript_gen{int(g)}.json"})
    return out


# ----------------------------------------------------------------------------
# Tables and figures
# ----------------------------------------------------------------------------
def table8(sess: pd.DataFrame, floor: float, rows: pd.DataFrame, floors: Dict[str, float] = None) -> str:
    lines = []
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna()]
    for (st, tau, cal, gen, arm), g in sess.groupby(["set", "tau", "cal", "generation", "arm"]):
        fl = (floors or {}).get(st, floor)
        gain = mean_ci((g.post_accuracy - fl).tolist()); tr = mean_ci(g.transfer_accuracy.tolist())
        if len(g) == 1:
            gain = (gain[0], float("nan"), float("nan"))
        itm = [v for s_ in g.items_to_mastery for v in s_.values()]
        med = np.median([v for v in itm if v is not None]) if any(v is not None for v in itm) else float("nan")
        cens = np.mean([v is None for v in itm])
        sub = inst[(inst.set == st) & (inst.tau == tau) & (inst.cal == cal) & (inst.generation == gen) & (inst.arm == arm)]
        e = ece(sub.p_next.astype(float), sub.correct.astype(int)) if len(sub) else float("nan")
        ci = f"[{gain[1]:.3f}, {gain[2]:.3f}]" if not math.isnan(gain[1]) else "(one seed)"
        lines.append(f"{st} & {tau:.2f} & {'on' if cal else 'off'} & {int(gen)} & {arm} & {gain[0]:.3f} {ci} & {tr[0]:.3f} & {med:.0f} ({cens:.0%}\\% cens.) & {g.completed.mean():.2f} & {e:.3f} \\\\")
    return "\n".join(lines)


def figures(out_dir: str, res: Dict, sess: pd.DataFrame, rows: pd.DataFrame, dense: pd.DataFrame, floor: float, controls: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    # Figure 2: KT baselines
    kt = res.get("kt_baselines", {}).get("models", {})
    if kt:
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.4))
        names = [k for k in ("BKT", "PFA", "DKT", "GBDT", "Base rate") if kt.get(k)]
        ax[0].bar(names, [kt[k]["auc"] for k in names], color="0.4"); ax[0].set_ylabel("AUC"); ax[0].set_ylim(0.4, 1.0); ax[0].axhline(0.5, ls=":", c="k", lw=0.8)
        ax[0].set_title("(a) Pooled")
        for k in names:
            ax[1].plot([5, 10, 20], [kt[k][f"auc_first{w}"] for w in (5, 10, 20)], marker="o", label=k)
        ax[1].set_xlabel("Cold-start window (interactions)"); ax[1].set_ylabel("AUC"); ax[1].set_title("(b) Cold start"); ax[1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_learner.pdf")); plt.close(fig)
    # Figure 3: calibration
    cal = res["calibration"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    inst = rows[(rows.phase == "instruction") & rows.p_next.notna()]
    for lab, sub, c in (("uncalibrated", inst[~inst.cal], "0.2"), ("calibrated (Platt)", inst[inst.cal], "0.6")):
        p = sub.p_next.astype(float).values; y = sub.correct.astype(int).values
        bins = np.linspace(0, 1, 11); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 5: xs.append(p[m].mean()); ys.append(y[m].mean())
        ax[0].plot(xs, ys, marker="o", color=c, label=f"{lab}, ECE {ece(p, y):.2f}")
    ax[0].plot([0, 1], [0, 1], ls=":", c="k", lw=0.8); ax[0].set_xlabel("Stated probability"); ax[0].set_ylabel("Observed accuracy"); ax[0].set_title("(a) Reliability"); ax[0].legend(fontsize=6)
    sl = cal["slices"]; keys = [k for k in sl if k.startswith("generation=")] + [k for k in sl if k.startswith("component=")]
    ax[1].bar(range(len(keys)), [sl[k]["ece_raw"] for k in keys], color="0.4"); ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([k.split("=")[1] for k in keys], rotation=90, fontsize=6)
    ax[1].set_ylabel("ECE (raw)"); ax[1].set_title("(b) By generation and component")
    comps = sorted(cal["tau_prime"]["component@0.85"]["plateaus"])
    plate = [cal["tau_prime"]["component@0.85"]["plateaus"][c] for c in comps]
    ax[2].bar(range(len(comps)), [0 if math.isnan(v) else v for v in plate], color="0.4")
    for tau, ls in ((0.70, ":"), (0.85, "--"), (0.95, "-")):
        ax[2].axhline(tau, ls=ls, c="k", lw=0.8, label=f"$\\tau$={tau}")
    ax[2].set_xticks(range(len(comps))); ax[2].set_xticklabels(comps, fontsize=6); ax[2].set_ylim(0, 1.05)
    ax[2].set_ylabel("Calibrated map ceiling"); ax[2].set_title("(c) Reachability of $\\tau$ by component"); ax[2].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_calibration.pdf")); plt.close(fig)
    # Figure 4: generations (Y plus controls if present)
    sets = ["Y"] + [s for s in ("R", "S") if len(controls) and (controls.set == s).any()]
    fig, ax = plt.subplots(2, len(sets), figsize=(2.4 * len(sets) + 0.6, 4.2), squeeze=False)
    for j, st in enumerate(sets):
        src = sess if st == "Y" else controls[controls.set == st]
        for arm, ls in (("chain", "-"), ("reference", "--")):
            for tau, c in ((0.7, "0.2"), (0.85, "0.5"), (0.95, "0.8")):
                sub = src[(src.arm == arm) & (np.isclose(src.tau, tau))]
                if not len(sub):
                    continue
                gens = sorted(sub.generation.unique())
                m = [sub[sub.generation == g].post_accuracy.mean() for g in gens]
                ax[0, j].plot(gens, m, ls=ls, color=c, marker="o", ms=3, label=f"{arm} $\\tau$={tau}")
                post = rows if st == "Y" else None
        ax[0, j].set_title(f"Set {st}"); ax[0, j].set_ylim(0, 1); ax[0, j].set_ylabel("Post-test accuracy" if j == 0 else "")
        ax[0, j].set_xticks([1, 2, 3, 4])
    mis = res["misconceptions"]
    for arm, ls in (("chain", "-"), ("reference", "--")):
        gens = sorted(int(g) for g in mis[arm]); ax[1, 0].plot(gens, [mis[arm][str(g)]["rates"].get("ignores-tether", 0) for g in gens], ls=ls, color="0.3", marker="o", ms=3, label=f"{arm}: ignores-tether")
    ax[1, 0].set_ylabel("Post-test misconception rate"); ax[1, 0].set_xlabel("Generation"); ax[1, 0].set_xticks([1, 2, 3, 4]); ax[1, 0].legend(fontsize=6)
    for j in range(1, len(sets)):
        ax[1, j].axis("off")
    ax[0, 0].legend(fontsize=5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig4_generations.pdf")); plt.close(fig)


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs"); ap.add_argument("--out", default="analysis")
    ap.add_argument("--floor", type=float, required=True, help="cold accuracy of the base model on Y (Table 3)")
    ap.add_argument("--set", default="Y")
    ap.add_argument("--floor-r", type=float, default=0.960, help="cold accuracy of the base model on R")
    ap.add_argument("--floor-s", type=float, default=float("nan"), help="cold accuracy of the base model on S")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from .content import load_content
    content = load_content(a.set)
    sess, rows = load(a.runs, "main")
    controls, crows = load(a.runs, "control")
    dense, _ = load(a.runs, "dense")
    rep, rrows = load(a.runs, "replication")
    res = {"n_sessions": int(len(sess)), "n_interactions": int((rows.phase == "instruction").sum()),
           "parse": rows[rows.phase == "instruction"].parse_status.value_counts().to_dict()}
    res["gain"] = {}
    for tau, g in sess.groupby("tau"):
        m, lo, hi = mean_ci((g.post_accuracy - a.floor).tolist()); res["gain"][str(tau)] = {"mean": m, "ci": [lo, hi], "n": int(len(g))}
    m, lo, hi = mean_ci((sess.post_accuracy - a.floor).tolist()); res["gain"]["pooled"] = {"mean": m, "ci": [lo, hi]}
    res["transfer_pooled"] = dict(zip(("mean", "lo", "hi"), mean_ci(sess.transfer_accuracy.tolist())))
    res["seed_variance_post"] = seed_variance(sess, "post_accuracy")
    res["by_cal"] = {str(c): dict(zip(("mean", "lo", "hi"), mean_ci(g.post_accuracy.tolist()))) for c, g in sess.groupby("cal")}
    res["by_arm_generation"] = {f"{arm}-{int(g)}": dict(zip(("mean", "lo", "hi"), mean_ci(s.post_accuracy.tolist()))) for (arm, g), s in sess.groupby(["arm", "generation"])}
    res["completion_by_tau"] = {str(t): float(g.completed.mean()) for t, g in sess.groupby("tau")}
    res["generation_model_post"] = generation_model(sess, "post_accuracy")
    res["generation_model_transfer"] = generation_model(sess, "transfer_accuracy")
    res["survival"] = survival_analysis(sess)
    res["calibration"] = calibration_analysis(rows)
    res["misconceptions"] = misconceptions(rows)
    res["persistence"] = persistence(rows)
    res["kt_baselines"] = kt_baselines(rows, content)
    res["threshold_curve"] = threshold_curve(dense, a.floor) if len(dense) else {}
    if len(controls):
        res["controls"] = {f"{st}-{arm}-{int(g)}": float(s.post_accuracy.mean()) for (st, arm, g), s in controls.groupby(["set", "arm", "generation"])}
    if len(rep):
        res["replication"] = {"by_arm_generation": {f"{arm}-{int(g)}": dict(zip(("mean", "lo", "hi"), mean_ci(s.post_accuracy.tolist()))) for (arm, g), s in rep.groupby(["arm", "generation"])},
                              "transfer_by_arm_generation": {f"{arm}-{int(g)}": dict(zip(("mean", "lo", "hi"), mean_ci(s.transfer_accuracy.tolist()))) for (arm, g), s in rep.groupby(["arm", "generation"])},
                              "n_sessions": int(len(rep)), "parse": rrows[rrows.phase == "instruction"].parse_status.value_counts().to_dict()}
        try:   # a single tau and calibration level makes the full model singular; fit the reduced one
            import statsmodels.formula.api as smf
            df = rep.dropna(subset=["post_accuracy"]).copy(); df["chain"] = (df.arm == "chain").astype(int); df["gen0"] = df.generation - 1
            m = smf.mixedlm("post_accuracy ~ chain*gen0", df, groups=df["seed"]).fit(reml=False); ci = m.conf_int()
            res["replication"]["chain_minus_reference_slope"] = {"estimate": float(m.params["chain:gen0"]), "ci": [float(ci.loc["chain:gen0", 0]), float(ci.loc["chain:gen0", 1])], "p": float(m.pvalues["chain:gen0"])}
        except Exception as e:
            res["replication"]["chain_minus_reference_slope"] = None
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    with open(os.path.join(a.out, "table8_rows.tex"), "w") as f:
        f.write(table8(sess, a.floor, rows))
    if len(controls):
        with open(os.path.join(a.out, "table8_control_rows.tex"), "w") as f:
            f.write(table8(controls, a.floor, crows, floors={"R": a.floor_r, "S": a.floor_s}))
    with open(os.path.join(a.out, "coding_sample.json"), "w") as f:
        json.dump(coding_sample(sess), f, indent=1)
    figures(a.out, res, sess, rows, dense, a.floor, controls)
    print(json.dumps({k: res[k] for k in ("n_sessions", "gain", "by_cal", "generation_model_post", "survival", "completion_by_tau")}, indent=1, default=float))
    print("kt:", json.dumps(res["kt_baselines"]["models"], default=float))
    print("tau_prime component@0.85:", res["calibration"]["tau_prime"]["component@0.85"])


if __name__ == "__main__":
    main()

"""Analyse the runs added after review (fixed-sequence control, reference-arm
probe re-run, wording test) and the September pilot files.

  python analyze_v2runs.py <v2runs_root> <september_runs_root> out.json
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from harness.analysis import load, wilson  # noqa: E402
from harness.revision import t_ci, cluster_robust_ci  # noqa: E402


def main(v2root, seproot, out_path):
    out = {}
    runs = os.path.join(v2root, "runs")
    # ---- fixed-sequence control
    fixed, frows = load(runs, "fixed")
    if len(fixed):
        sess, rows = load(seproot, "main")
        g1 = sess[sess.generation == 1]; g1_85 = g1[(np.isclose(g1.tau, 0.85)) & g1.cal]
        fx_post = fixed.post_accuracy.values; fx_tr = fixed.transfer_accuracy.values
        out["fixed_control"] = {
            "n_sessions": int(len(fixed)), "interactions_mean": float(fixed.interactions.mean()),
            "post_accuracy": list(t_ci(fx_post.tolist())[:3]), "transfer": list(t_ci(fx_tr.tolist())[:3]),
            "post_by_component": {c: float(g.correct.mean()) for c, g in frows[frows.phase == "post"].groupby("component")},
            "adaptive_gen1_tau085_cal_post": list(t_ci(g1_85.post_accuracy.tolist())[:3]), "adaptive_gen1_tau085_cal_n": int(len(g1_85)),
            "adaptive_gen1_all_post": list(cluster_robust_ci(g1.post_accuracy.values, g1.cell.values)[:3]), "adaptive_gen1_all_n": int(len(g1)),
            "adaptive_gen1_all_interactions_mean": float(g1.interactions.mean()),
            "adaptive_gen1_all_transfer": list(cluster_robust_ci(g1.transfer_accuracy.values, g1.cell.values)[:3]),
        }
        t, p = stats.ttest_ind(fx_post, g1.post_accuracy.values, equal_var=False)
        out["fixed_control"]["welch_t_vs_adaptive_gen1_all"] = {"t": float(t), "p": float(p), "difference": float(fx_post.mean() - g1.post_accuracy.mean())}
        t2, p2 = stats.ttest_ind(fx_post, g1_85.post_accuracy.values, equal_var=False)
        out["fixed_control"]["welch_t_vs_adaptive_gen1_tau085_cal"] = {"t": float(t2), "p": float(p2), "difference": float(fx_post.mean() - g1_85.post_accuracy.mean())}
        pf = frows[frows.phase == "post"]; out["fixed_control"]["ignores_tether_post"] = float((pf.misconception == "ignores-tether").mean())
        out["fixed_control"]["tokens"] = {"in": int(sum(json.load(open(f))["tokens_in"] for f in glob.glob(os.path.join(runs, "fixed-*", "run_summary.json")))),
                                          "out": int(sum(json.load(open(f))["tokens_out"] for f in glob.glob(os.path.join(runs, "fixed-*", "run_summary.json"))))}
    # ---- reprobe
    rp = sorted(glob.glob(os.path.join(runs, "reprobe", "*", "telemetry.jsonl")))
    if rp:
        recs = []
        for f in rp:
            for l in open(f):
                r = json.loads(l)
                if r["phase"] == "persist_boundary":
                    recs.append(r)
        df = pd.DataFrame(recs); k, n = int(df.correct.sum()), int(len(df)); lo, hi = wilson(k, n)
        sig = []
        for f in sorted(glob.glob(os.path.join(runs, "reprobe", "*", "sigma_ref_gen*.json"))):
            j = json.load(open(f)); sig.append({"run": f.split("/")[-2], "generation": j["generation"], "chars": len(j["sigma"]), "served_model": j.get("served_model")})
        out["reprobe"] = {"n_items": n, "accuracy": k / n, "wilson": [lo, hi], "n_sessions": len(sig), "components": sorted(df.component.unique().tolist()),
                          "served_models": sorted({s["served_model"] for s in sig if s["served_model"]}), "summary_chars_min_max": [min(s["chars"] for s in sig), max(s["chars"] for s in sig)]}
    # ---- wording
    wd = sorted(glob.glob(os.path.join(runs, "wording-*", "wording_summary.json")))
    if wd:
        j = json.load(open(wd[-1])); counts = j["counts"]; res = {}
        for v, c in counts.items():
            lo, hi = wilson(c["omitted"], c["of"]); res[v] = {"omitted": c["omitted"], "of": c["of"], "rate": c["omitted"] / c["of"], "wilson": [lo, hi]}
        keys = list(counts)
        for i in range(len(keys)):
            for jx in range(i + 1, len(keys)):
                a, b = counts[keys[i]], counts[keys[jx]]
                _, p = stats.fisher_exact([[a["omitted"], a["of"] - a["omitted"]], [b["omitted"], b["of"] - b["omitted"]]])
                res[f"fisher_{keys[i]}_vs_{keys[jx]}"] = float(p)
        out["wording"] = res
    # ---- pilot per component
    pil = sorted(glob.glob(os.path.join(runs, "pilot-Y-*", "pilot_Y.json"))) or sorted(glob.glob(os.path.join(v2root, "**", "pilot_Y.json"), recursive=True))
    if pil:
        j = json.load(open(pil[-1])); out["pilot_Y"] = {"accuracy": j.get("accuracy"), "n": j.get("n"), "per_family": j.get("per_family"), "file": pil[-1]}
    json.dump(out, open(out_path, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])

"""Agreement between the human second rater and the model coding (and the
released detector) on the 24-transcript subset.

  python rater_agreement.py rater_sheet_completed.csv coding_sheet_Claude.csv merge_explanations_main.csv out.json
"""
import csv
import json
import re
import sys


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n; pb = sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def yn(v):
    v = (v or "").strip().lower()
    if v in ("yes", "y", "true", "1"):
        return 1
    if v in ("no", "n", "false", "0"):
        return 0
    return None


def main(rater_path, model_path, det_path, out_path):
    rater = {r["transcript_id"]: r for r in csv.DictReader(open(rater_path))}
    model = {r["transcript_id"]: r for r in csv.DictReader(open(model_path))}
    det = {}
    for r in csv.DictReader(open(det_path)):
        if r["merge_taught"] == "True":
            key = (r["cell"], int(r["generation"]), r["arm"]); det[key] = r["scaling_stated"] == "True"
    out = {"n": len(rater), "A2_tether_scaled": {}, "B1_other_rule_wrong": {}, "A3_rest_of_merge_correct": {}, "disagreements": []}
    a_r, a_m, a_d, b_r, b_m, c_r, c_m = [], [], [], [], [], [], []
    for tid, r in rater.items():
        m = model[tid]; dist = m["distortion_of_rule"]
        num, arm, gen, cell = tid.split("_", 3); gen = int(gen[3:])
        model_scaled = 0 if re.search(r"tether not scaled|flat tether", dist) else 1
        model_other = 1 if re.search(r"\bC[1-5]:", dist) else 0
        model_rest_ok = 0 if re.search(r"single floor|shares not complementary|rounds the sum|shares in tenths not divided|applied to whole result|shares computed from", dist) else 1
        v = yn(r["A2_tether_scaled_by_shorter_cairn"])
        if v is not None:
            a_r.append(v); a_m.append(model_scaled); a_d.append(int(det.get((cell, gen, arm), model_scaled == 1)))
            if v != model_scaled:
                out["disagreements"].append({"transcript": tid, "code": "A2", "rater": v, "model": model_scaled})
        v3 = yn(r["A3_rest_of_merge_rule_correct"])
        if v3 is not None:
            c_r.append(v3); c_m.append(model_rest_ok)
            if v3 != model_rest_ok:
                out["disagreements"].append({"transcript": tid, "code": "A3", "rater": v3, "model": model_rest_ok})
        vb = yn(r["B1_any_other_rule_wrong"])
        if vb is not None:
            b_r.append(vb); b_m.append(model_other)
            if vb != model_other:
                out["disagreements"].append({"transcript": tid, "code": "B1", "rater": vb, "model": model_other})
    if a_r:
        k, po = kappa(a_r, a_m); kd, pod = kappa(a_r, a_d)
        out["A2_tether_scaled"] = {"n": len(a_r), "kappa_vs_model": k, "agreement_vs_model": po, "kappa_vs_detector": kd, "agreement_vs_detector": pod,
                                  "rater_says_scaled": sum(a_r), "model_says_scaled": sum(a_m)}
    if c_r:
        k, po = kappa(c_r, c_m); out["A3_rest_of_merge_correct"] = {"n": len(c_r), "kappa_vs_model": k, "agreement_vs_model": po}
    if b_r:
        k, po = kappa(b_r, b_m); out["B1_other_rule_wrong"] = {"n": len(b_r), "kappa_vs_model": k, "agreement_vs_model": po, "rater_says_wrong": sum(b_r), "model_says_wrong": sum(b_m)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])

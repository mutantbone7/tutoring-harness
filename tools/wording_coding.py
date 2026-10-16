"""Score the wording test (harness v2 `wording` command; reviewer 2, item 4d).

  python wording_coding.py <wording run dir> <out dir>

Writes wording_results.json and wording_coding.csv. Two codings are reported:
the released detector (revision.tether_scaling_stated) on every merge
explanation, and a reading of the rule statement of each explanation given
the specification context, made by the model-assisted coder of Section VI-F
(the assistant model, reading the full reply) and recorded here as a fixed
dictionary so that the classification is inspectable and re-checkable
against the replies in replies.jsonl. For the two formula contexts every
explanation is checked mechanically for the formula's third term.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from harness.revision import tether_scaling_stated  # noqa: E402
from harness.analysis import wilson  # noqa: E402
from scipy import stats  # noqa: E402

# Reading of the rule statement in each specification-context explanation.
#   correct      the tether b is scaled by the shorter cairn (per ten marks, or divided by 10)
#   flat         the tether is a fixed number added to the sum; no scaling anywhere in the rule statement
#   whole_sum    the per-ten-marks factor is attached to the whole sum, or read as a unit of measure, not to the tether
SPEC_CODES = {
    0: "flat", 1: "correct", 2: "correct", 3: "flat", 4: "correct", 5: "flat", 6: "correct", 7: "correct",
    8: "whole_sum", 9: "flat", 10: "whole_sum", 11: "flat", 12: "whole_sum", 13: "flat", 14: "correct",
    15: "flat", 16: "whole_sum", 17: "whole_sum", 18: "flat", 19: "correct", 20: "flat", 21: "flat",
    22: "flat", 23: "correct", 24: "whole_sum", 25: "flat", 26: "whole_sum", 27: "flat", 28: "whole_sum", 29: "whole_sum",
}
FORMULA_TERM = re.compile(r"b\s*[×x\*]\s*floor\s*\(\s*min\s*\(\s*q[₁1]\s*,\s*q[₂2]\s*\)\s*/\s*10\s*\)", re.I)


def main(run_dir, out_dir):
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "replies.jsonl")) if l.strip()]
    summ = json.load(open(os.path.join(run_dir, "wording_summary.json")))
    c6 = [r for r in recs if r["component"] == "C6"]
    variants = ["spec", "instruction", "spec_merge_restated"]
    out = {"run": os.path.basename(run_dir.rstrip("/")), "n_replies": len(recs), "served_models": sorted({r["served_model"] for r in recs}),
           "detector": {}, "spec_rule_statement_codes": {}, "formula_contexts": {}, "token_cap": {}}
    rows = []
    for v in variants:
        rs = [r for r in c6 if r["variant"] == v]
        om = sum(1 for r in rs if not tether_scaling_stated(r["reply"]))
        lo, hi = wilson(om, len(rs))
        out["detector"][v] = {"omitted": om, "of": len(rs), "rate": om / len(rs), "wilson95": [lo, hi],
                              "agrees_with_run_time_detector": all(tether_scaling_stated(r["reply"]) == r["scaling_stated"] for r in rs)}
        for r in rs:
            det = tether_scaling_stated(r["reply"])
            code = SPEC_CODES.get(r["teacher"]) if v == "spec" else ("correct" if FORMULA_TERM.search(r["reply"]) else "formula_term_absent")
            rows.append({"variant": v, "teacher": r["teacher"], "detector_scaling_stated": det, "rule_statement_code": code, "reply_chars": len(r["reply"])})
    for a, b in (("spec", "instruction"), ("spec", "spec_merge_restated"), ("instruction", "spec_merge_restated")):
        da, db = out["detector"][a], out["detector"][b]
        _, p = stats.fisher_exact([[da["omitted"], da["of"] - da["omitted"]], [db["omitted"], db["of"] - db["omitted"]]])
        out["detector"][f"fisher_{a}_vs_{b}"] = float(p)
    codes = [r for r in rows if r["variant"] == "spec"]
    out["spec_rule_statement_codes"] = {c: sum(1 for r in codes if r["rule_statement_code"] == c) for c in ("correct", "flat", "whole_sum")}
    out["spec_rule_statement_codes"]["detector_omitted_by_code"] = {c: sum(1 for r in codes if r["rule_statement_code"] == c and not r["detector_scaling_stated"]) for c in ("correct", "flat", "whole_sum")}
    for v in ("instruction", "spec_merge_restated"):
        vs = [r for r in rows if r["variant"] == v]
        out["formula_contexts"][v] = {"formula_third_term_present": sum(1 for r in vs if r["rule_statement_code"] == "correct"), "of": len(vs)}
    turns = summ["turns"]
    c6_turns = [t for k, t in enumerate(turns) if k % 6 == 5]
    out["token_cap"] = {"teach_token_cap": 500, "c6_turns": len(c6_turns), "c6_turns_at_cap": sum(1 for t in c6_turns if t["tokens_out"] >= 500),
                        "all_turns_at_cap": sum(1 for t in turns if t["tokens_out"] >= 500), "all_turns": len(turns)}
    out["tokens"] = {"in": sum(t["tokens_in"] for t in turns), "out": sum(t["tokens_out"] for t in turns), "retries": sum(t["retries"] for t in turns)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "wording_results.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "wording_coding.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])

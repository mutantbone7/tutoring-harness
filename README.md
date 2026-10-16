# Tutoring harness (version 2)

Software for the experiment described in Section IV of *Elicitation or
Acquisition? Contamination-Controlled Measurement of In-Context Learning and
Pedagogical Transfer in Language Model Agents* (IEEE Access manuscript
Access-2026-47739). It implements the message router with the isolation
check, Algorithm 1 (policy), Algorithm 2 (chain runner with transmission
controls), the telemetry schema, the Gate 1 contamination probes, the Gate 2
equating pilot, GBDT training with Platt or isotonic calibration, the
confirmatory analysis, and the revision analysis of the resubmission.

Version 1 (Zenodo 10.5281/zenodo.22851579, version 1) is the code the study
ran on. Version 2 is the same code with the fixes listed under "What changed"
below. Every change that affects a result is flagged in the manuscript's
pre-registration addendum and reproduced from the logged version-1 data by
`harness/revision.py`, so a reader can see the effect of each fix without
rerunning the model.

## Install

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q          # 31 tests, no network, about 2 s

The Anthropic client reads `ANTHROPIC_API_KEY` from the environment. Every
command accepts `--fake`, which swaps in a deterministic fake model so the
whole pipeline can be exercised without an API call.

## Commands

    python -m harness.cli --fake probes --set Y --n 40
    python -m harness.cli --fake pilot  --set Y --n 40
    python -m harness.cli --fake chain  --set Y --tau 0.85 --seed 1
    python -m harness.cli --fake sweep  --sets Y --seeds 1 2 3 --tag main
    python -m harness.train --runs "runs/bootstrap-*" --set Y --calibration platt --out models/gbdt_platt.pkl
    python -m harness.analysis --runs runs --out analysis --floor 0.064 --floor-s 0.815
    python -m harness.revision --runs runs --out analysis_v2 --floor 0.064

Version-2 commands (all accept `--fake`):

    python -m harness.cli --policy fixed sweep --sets Y --seeds 1 2 3 4 5 6 --taus 0.85 --calibration on --generations 1 --tag fixed
    python -m harness.cli reprobe --runs runs --tag main --set Y
    python -m harness.cli wording --set Y --n 30
    python -m harness.cli --provider openai --base-url https://api.openai.com/v1 --model gpt-4.1-mini probes --set Y --n 200
    python -m harness.export packet    --runs runs --sample analysis/coding_sample.json --out coding
    python -m harness.export rater     --runs runs --sample analysis/coding_sample.json --out coding_rater2 --n 20
    python -m harness.export summaries --runs runs --out analysis_v2

Drop `--fake` to use the model named in `Config.model` (`--model` overrides
it; the study used `claude-haiku-4-5-20251001`). Set a spend cap in the
Console before doing so.

Outputs go to `runs/<run-id>/`: `telemetry.jsonl` (one line per interaction:
the Table 5 fields plus policy action and learner probability),
`transcript_gen<n>.json` (every instance's full context),
`sigma_gen<n>.json` and `sigma_ref_gen<n>.json` (the chain and reference
students' summaries), `run_summary.json` (outcomes, isolation-violation
count, token totals, retry count, and the version-2 flags in force), and
`config.json` (the configuration the run used).

## What a session contains

One TEACH turn per component when the policy advances or remediates to it,
capped at `teach_token_cap` (500) tokens; four-option items with the
feedback template "Your previous answer was correct / not correct" and no
teacher response to errors; a post-test of `n_assess` (3) items per
component in the same context, plus 3 held-out items for each component
that has a held-out family (C3 and C4); then the persistence probes. The
prompts are in `harness/session.py` and `harness/chain.py`, and the
specification texts in `harness/content/novel.py` and `finance.py`.

## What changed in version 2

Fixes to the code the study ran on (version 1), each with the reviewer item
that prompted it:

* `chain.py`: the reference student's boundary probe received the chain
  student's summary (item 7a). Now each student writes its own summary
  (`Config.reference_summary`), and both are logged.
* `chain.py`: the summary was requested after the post-test (item 4g). Now
  it is requested before it (`Config.summary_before_posttest`; set False to
  reproduce version 1).
* `learner.py`, `train.py`: the policy's learner used tau, calibration,
  generation and arm as features (item 2d). Training now excludes them by
  default (`--condition-features` restores version 1); `N_CONDITION_BLIND`
  marks the boundary. The H1 baseline in `analysis.py` uses the first five
  groups only.
* `learner.py`: the Platt map was scikit-learn's default L2-penalised
  logistic regression, which on 145 calibration exchanges capped the map at
  0.923 (item 2b). `--platt-penalty none` (the new default) fits the
  unpenalised map; `l2` reproduces version 1.
* `content/base.py`, `policy.py`, `cli.py --pilot`: the pilot difficulty was
  never loaded, so the feature was the constant 0.5, and the family feature
  was the previous item's (item 2e). Both are wired to the next item now.
* `session.py`, `cli.py --policy fixed`: the non-adaptive fixed-sequence
  control of Section IV-E, which version 1 did not run (item 2f). The
  "Fixed-sequence" entry of version 1's `analysis.py` was a per-component
  base-rate predictor and is now named "Base rate".
* `analysis.py`: DKT's first prediction is the component's training base
  rate, not 0.5 (item 10d); the dense-grid curve pools both arms as the
  figure caption says (item 6).
* `content/novel.py`, `finance.py`: the header label "DRAFT, not frozen"
  left from the Gate 0 review was removed (item 4e). No rule changed; the
  rule text is byte-identical to commit 814ddc1.
* `revision.py`: the released tether-scaling detector
  (`tether_scaling_stated`), the eight-contrast family with
  Benjamini-Hochberg adjustment, minimum detectable effects, cluster-robust
  and t-based intervals, the mixed model with run and item-family
  intercepts, survival analysis on attempted components clustered by
  session, the change-point reading of the dense grid, and everything else
  the reviewers asked for (items 1a, 2a to 2d, 3b, 3c, 4a, 4c, 4h, 5a to
  5g, 7a, 7b, 10d).
* `export.py`: the coding packet and the summaries are exported in full
  (version 1's packet cut summaries at 1,200 characters, item 4f).
* `extras.py`: `reprobe` re-runs the reference-arm boundary probe with the
  reference student's own summary; `wording` tests whether the
  specification's wording causes the tether omission (item 4d).
* `client.py`: `OpenAICompatibleClient` for a second provider (reviewer 3).

## Isolation check, stated precisely

Before every student turn the router checks the byte content of the
student's context against every line of the specification of 40 or more
characters and against the answer-key markers, and raises
`IsolationViolation`, which aborts the run and is counted. It catches
verbatim reproduction only; a paraphrase of the specification by a teacher
is not a violation and is not meant to be, since conveying the rules is the
teacher's job. The answer key never enters any prompt: it exists only in the
harness side channel (`meta`).

## Other notes

* `demo` is a toy rule system for tests. R, S and Y are
  `harness/content/finance.py` and `novel.py`, frozen at commit 814ddc1.
* The learner inside the policy is BKT with fixed parameters until a GBDT has
  been trained from a bootstrap pass; then `Config.learner` and
  `Config.learner_uncalibrated` point at the saved models.
* Sampling parameters are never set: the API for current models accepts
  neither temperature nor a seed. Harness seeds govern item order and
  instantiation, following the s+n, s+100+n, s+200+n, s+300+n pattern.
* The served model version string is recorded per turn in
  `run_summary.json` under `turns[].model`.

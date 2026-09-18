# Tutoring harness

Software for the experiment described in Section IV of *Elicitation or
Acquisition?* Implements the message router with the isolation invariant,
Algorithm 1 (policy), Algorithm 2 (chain runner with transmission controls),
the Table 5 telemetry schema, the Gate 1 contamination probes, the Gate 2
equating pilot, and GBDT training with Platt or isotonic calibration.

## Install

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q          # 10 tests, no network, ~1 s

The real client reads `ANTHROPIC_API_KEY` from the environment. Every
command accepts `--fake`, which swaps in a deterministic fake model so the
whole pipeline can be exercised without an API call.

## Commands

    python -m harness.cli --fake probes --set demo --n 40
    python -m harness.cli --fake pilot  --set demo --n 40
    python -m harness.cli --fake chain  --set demo --tau 0.85 --seed 1
    python -m harness.cli --fake sweep  --sets demo --seeds 1 2 3
    python -m harness.train --runs "runs/sweep-*" --set demo --calibration platt --out models/gbdt_platt.pkl

Drop `--fake` to use the model named in `Config.model` (default
`claude-fable-5-1`). Set a spend cap in the Console before doing so.

Outputs go to `runs/<run-id>/`: `telemetry.jsonl` (one line per interaction,
Table 5 fields plus policy action and learner probability),
`transcript_gen<n>.json` (every instance's full context), `sigma_gen<n>.json`
(the carried summary), `run_summary.json` (outcomes, isolation-violation
count, token totals, retry count), and for probes and pilots a JSON with
accuracy, Wilson interval, one-sided binomial p-value and the gate verdict,
plus a ready LaTeX row for Table 3.

## What is and is not decided here

* `demo` is the only registered content set. It is a toy rule system that
  exists to test the harness. It is not R, S or Y and produces no
  reportable result. R, S and Y are written by the authors as subclasses of
  `harness.content.base.ContentSet` (specification text, KC graph, item
  families with oracles and misconception-tagged distractors, execution
  traces, instruction block, confusable pairs) and registered in
  `harness/content/__init__.py`. `check_held_out_traces` mechanically verifies
  the structural-distinctness rule of Section III-E.
* The learner model that runs inside the policy is BKT with fixed parameters
  until a GBDT has been trained. The GBDT cannot exist before agent
  telemetry does, so the first pass is a bootstrap. Record this at Gate 3.
  After `harness.train`, point `Config.learner` (calibrated) and
  `Config.learner_uncalibrated` at the saved models.
* Numeric defaults in `harness/config.py` marked GATE 3 (`attempt_cap`,
  `review_interval`, `session_cap`, `summary_token_cap`, `teach_token_cap`,
  `persistence_items`) are harness defaults. Fix them at pre-registration and
  make the paper match.
* Isolation is enforced as the manuscript says: before every student turn the
  router checks the byte content of the student's context against every line
  of the specification of 40 or more characters and against answer-key
  markers, and raises `IsolationViolation`, which aborts the run and is
  counted. Teachers may hold the specification; students and probes may not.
  Verbatim reproduction by a teacher of a specification line into a student's
  context therefore aborts the run by design.
* Fixed text (curriculum phrase, item stems, feedback templates) comes from
  the harness. Teacher explanatory turns are generated live. The summary
  sigma_n is produced from the student instance by a fixed prompt capped at
  `summary_token_cap`, and is checked for specification and answer-key
  content before it becomes the next teacher's context (Algorithm 2 line 16).
* Persistence proxies (context-distance, session-boundary, interference) run
  on every student instance after its session and are reported jointly.
* Sampling parameters are never set: the API for current models accepts
  neither temperature nor a seed. Harness seeds govern item order and
  instantiation, following the s+n, s+100+n, s+200+n, s+300+n pattern.

## Not yet implemented

Survival analysis, the mixed model of equation (12), the tau-prime dispersion
table and the figures are analysis steps over `telemetry.jsonl` and are
written once real telemetry exists. The model version string that the API
serves is recorded per turn in `run_summary.json` under `turns[].model`.

"""Command line entry points.

  python -m harness.cli probes  --set demo --n 40 --fake
  python -m harness.cli pilot   --set demo --n 40 --fake
  python -m harness.cli chain   --set demo --tau 0.85 --seed 1 --fake
  python -m harness.cli sweep   --sets demo --seeds 1 2 --fake

Drop --fake to call the real model (needs ANTHROPIC_API_KEY and credits).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from .config import Config
from .content import load_content
from .telemetry import Telemetry


def make_client(cfg: Config, fake: bool):
    if fake:
        from .client import FakeClient
        return FakeClient(model=cfg.model)
    from .client import ModelClient
    return ModelClient(cfg.model, cfg.max_retries, cfg.prompt_caching)


def _run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="harness")
    ap.add_argument("--config", default=None, help="JSON config file (see Config)")
    ap.add_argument("--fake", action="store_true", help="use the deterministic fake client (no API calls)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None, help="override the model identifier in the config (Gate 2 candidate screening)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes"); p.add_argument("--set", required=True); p.add_argument("--n", type=int); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue the most recent interrupted run of this command and set")
    p = sub.add_parser("pilot"); p.add_argument("--set", required=True); p.add_argument("--n", type=int); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue the most recent interrupted run of this command and set")
    p = sub.add_parser("chain"); p.add_argument("--set", required=True); p.add_argument("--tau", type=float, required=True)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--uncalibrated", action="store_true")
    p = sub.add_parser("dedupe"); p.add_argument("--run", required=True, help="repair a run that received rows from two processes")
    p = sub.add_parser("inspect"); p.add_argument("--run", required=True, help="a run directory under the output dir")
    p = sub.add_parser("sweep"); p.add_argument("--sets", nargs="+", required=True); p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--taus", nargs="*", type=float)
    p.add_argument("--calibration", choices=["on", "off", "both"], default="both")
    p.add_argument("--workers", type=int, default=None, help="parallel sessions (default: Config.concurrency)")
    p.add_argument("--tag", default="sweep", help="run-directory prefix (e.g. bootstrap, dense, replication)")
    p.add_argument("--generations", type=int, default=None, help="override Config.generations (dense sweep uses 1)")
    p.add_argument("--learner", default=None, help="calibrated learner: 'bkt' or a GBDT .pkl path")
    p.add_argument("--learner-uncalibrated", default=None, help="uncalibrated learner: 'bkt' or a GBDT .pkl path")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config) if a.config else Config()
    if a.out:
        cfg.output_dir = a.out
    if a.model:
        cfg.model = a.model
    os.makedirs(cfg.output_dir, exist_ok=True)
    client = make_client(cfg, a.fake)

    if a.cmd in ("probes", "pilot"):
        from .probes import run_probes, run_pilot, table3_rows
        content = load_content(a.set)
        tag = f"{a.cmd}-{a.set}" + (f"-{cfg.model}" if a.model else "")
        run_id = _run_id(tag)
        if a.resume:
            import glob
            prev = sorted(glob.glob(os.path.join(cfg.output_dir, f"{tag}-*")))
            if not prev:
                raise SystemExit(f"nothing to resume: no {a.cmd}-{a.set}-* directory in {cfg.output_dir}")
            run_id = os.path.basename(prev[-1])
            print(f"resuming {run_id}")
        tel = Telemetry(cfg.output_dir, run_id)
        fn = run_probes if a.cmd == "probes" else run_pilot
        res = fn(client, content, cfg, tel, seed=a.seed, n=a.n, resume=a.resume)
        tel.close()
        print(json.dumps({k: v for k, v in res.items() if k not in ("per_family",)}, indent=1))
        if res["parse_failure_rate"] > 0.05:
            print(f"\nWARNING: {res['parse_failure_rate']:.1%} of replies could not be parsed; the run is not a valid "
                  f"measurement. Read {tel.dir}/{'probe' if a.cmd == 'probes' else 'pilot'}_responses.jsonl.")
        if a.cmd == "probes":
            print("\nTable 3 row:\n" + table3_rows([res]))
    elif a.cmd == "dedupe":
        from .dedupe import main as dedupe_main
        dedupe_main(["--run", a.run])
        return
    elif a.cmd == "inspect":
        import collections
        rows = [json.loads(l) for l in open(os.path.join(a.run, "telemetry.jsonl"))]
        print("rows:", len(rows), " phases:", dict(collections.Counter(r["phase"] for r in rows)))
        print("parse status:", dict(collections.Counter(r["parse_status"] for r in rows)))
        print("accuracy:", round(sum(bool(r["correct"]) for r in rows) / len(rows), 3))
        wrong = [r for r in rows if r["parse_status"] != "failure" and not r["correct"]]
        print("misconceptions chosen:", dict(collections.Counter(r["misconception"] for r in wrong).most_common()))
        print("mean output tokens:", round(sum(r["tokens_out"] or 0 for r in rows) / len(rows), 1),
              " max:", max(r["tokens_out"] or 0 for r in rows))
        for name in ("probe_responses.jsonl", "pilot_responses.jsonl"):
            path = os.path.join(a.run, name)
            if os.path.exists(path):
                fails = [json.loads(l) for l in open(path)]
                fails = [f for f in fails if f["status"] == "failure"][:3]
                for f in fails:
                    print("\n--- unparsed reply for", f["item_id"], "---\n" + f["response"][-600:])
        return
        from .chain import run_chain
        content = load_content(a.set)
        tel = Telemetry(cfg.output_dir, _run_id(f"chain-{a.set}-tau{a.tau}-s{a.seed}"))
        res = run_chain(client, content, cfg, a.tau, a.seed, not a.uncalibrated, tel)
        tel.close()
        for r in res:
            print(json.dumps(r))
    elif a.cmd == "sweep":
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .chain import run_chain
        taus = a.taus or cfg.taus
        if a.generations:
            cfg.generations = a.generations
        if a.learner:
            cfg.learner = a.learner
        if a.learner_uncalibrated:
            cfg.learner_uncalibrated = a.learner_uncalibrated
        cal_levels = {"on": (True,), "off": (False,), "both": (True, False)}[a.calibration]
        cells = []
        for si, name in enumerate(a.sets):
            for tau in taus:
                for calibrated in cal_levels:
                    for seed in a.seeds:
                        rid = f"{a.tag}-{name}-tau{tau}-{'cal' if calibrated else 'uncal'}-s{seed}"
                        if os.path.exists(os.path.join(cfg.output_dir, rid, "run_summary.json")):
                            print(rid, "already complete; skipped")
                            continue
                        cells.append((rid, name, si, tau, calibrated, seed))
        print(f"{len(cells)} cells to run, {a.workers or cfg.concurrency} in parallel")

        def run_cell(cell):
            rid, name, si, tau, calibrated, seed = cell
            content = load_content(name)
            import shutil
            d = os.path.join(cfg.output_dir, rid)
            if os.path.exists(d):
                shutil.rmtree(d)              # a partial cell is restarted from its first session
            tel = Telemetry(cfg.output_dir, rid)
            cfg.save(os.path.join(tel.dir, "config.json"))
            try:
                res = run_chain(client, content, cfg, tau, seed, calibrated, tel, set_index=si)
            finally:
                tel.close()
            return rid, len(res)

        workers = 1 if a.fake else (a.workers or cfg.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_cell, c) for c in cells]
            for f in as_completed(futures):
                try:
                    rid, n = f.result()
                    print(rid, "done;", n, "sessions", flush=True)
                except Exception as e:
                    print("CELL FAILED:", e, flush=True)
        if not a.fake:
            print(json.dumps({"tokens_in": client.total_in, "tokens_out": client.total_out,
                              "cache_read": client.total_cache_read, "cache_write": client.total_cache_write,
                              "retries": client.total_retries}))


if __name__ == "__main__":
    main()

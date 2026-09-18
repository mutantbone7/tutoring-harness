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
    return ModelClient(cfg.model, cfg.max_retries)


def _run_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="harness")
    ap.add_argument("--config", default=None, help="JSON config file (see Config)")
    ap.add_argument("--fake", action="store_true", help="use the deterministic fake client (no API calls)")
    ap.add_argument("--out", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probes"); p.add_argument("--set", required=True); p.add_argument("--n", type=int); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("pilot"); p.add_argument("--set", required=True); p.add_argument("--n", type=int); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("chain"); p.add_argument("--set", required=True); p.add_argument("--tau", type=float, required=True)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--uncalibrated", action="store_true")
    p = sub.add_parser("sweep"); p.add_argument("--sets", nargs="+", required=True); p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--taus", nargs="*", type=float)
    a = ap.parse_args(argv)

    cfg = Config.load(a.config) if a.config else Config()
    if a.out:
        cfg.output_dir = a.out
    os.makedirs(cfg.output_dir, exist_ok=True)
    client = make_client(cfg, a.fake)

    if a.cmd in ("probes", "pilot"):
        from .probes import run_probes, run_pilot, table3_rows
        content = load_content(a.set)
        tel = Telemetry(cfg.output_dir, _run_id(f"{a.cmd}-{a.set}"))
        fn = run_probes if a.cmd == "probes" else run_pilot
        res = fn(client, content, cfg, tel, seed=a.seed, n=a.n)
        tel.close()
        print(json.dumps({k: v for k, v in res.items() if k not in ("per_family",)}, indent=1))
        if a.cmd == "probes":
            print("\nTable 3 row:\n" + table3_rows([res]))
    elif a.cmd == "chain":
        from .chain import run_chain
        content = load_content(a.set)
        tel = Telemetry(cfg.output_dir, _run_id(f"chain-{a.set}-tau{a.tau}-s{a.seed}"))
        res = run_chain(client, content, cfg, a.tau, a.seed, not a.uncalibrated, tel)
        tel.close()
        for r in res:
            print(json.dumps(r))
    elif a.cmd == "sweep":
        from .chain import run_chain
        taus = a.taus or cfg.taus
        for si, name in enumerate(a.sets):
            content = load_content(name)
            for tau in taus:
                for calibrated in (True, False):
                    for seed in a.seeds:
                        rid = f"sweep-{name}-tau{tau}-{'cal' if calibrated else 'uncal'}-s{seed}"
                        tel = Telemetry(cfg.output_dir, rid)
                        cfg.save(os.path.join(tel.dir, "config.json"))
                        res = run_chain(client, content, cfg, tau, seed, calibrated, tel, set_index=si)
                        tel.close()
                        print(rid, "done;", len(res), "sessions")


if __name__ == "__main__":
    main()

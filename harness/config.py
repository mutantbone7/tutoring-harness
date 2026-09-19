"""Run configuration. Every numeric default below is a harness default, not a
value the manuscript has fixed. The ones marked GATE 3 must be recorded at
pre-registration and must then match the paper."""
from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class Config:
    model: str = "claude-fable-5-1"          # base model M (Section V-A)
    output_dir: str = "runs"
    taus: List[float] = field(default_factory=lambda: [0.70, 0.85, 0.95])
    dense_taus: List[float] = field(default_factory=lambda: [0.70, 0.74, 0.78, 0.82, 0.85, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98])
    generations: int = 4                     # N_g
    attempt_cap: int = 8                     # A_max, Algorithm 1 line 12      GATE 3: 8 (pre-registered 19 Sep. 2026)
    review_interval: int = 6                 # k, Algorithm 1 line 3            GATE 3: 6 (pre-registered 19 Sep. 2026)
    session_cap: int = 40                    # L, interactions per session      GATE 3: 40 (pre-registered 19 Sep. 2026)
    summary_token_cap: int = 600             # B_sigma, Algorithm 2 line 14     GATE 3: 600 (pre-registered 19 Sep. 2026)
    teach_token_cap: int = 500               # teacher explanatory turn length  GATE 3: 500 (pre-registered 19 Sep. 2026)
    answer_token_cap: int = 2500             # reply cap; 300 truncated 36% of Y probe replies and 1500 truncated 6.4% of Y pilot replies (2026-09-19)
    persistence_items: int = 4               # items per persistence proxy      GATE 3: 4 (pre-registered 19 Sep. 2026)
    n_probes: int = 500                      # Gate 1 probes per model per set
    probe_fail_threshold: float = 0.30       # Gate 1: accuracy above this fails
    probe_alpha: float = 0.05                # Gate 1: one-sided binomial test
    max_retries: int = 8
    prompt_caching: bool = True              # cache breakpoints on system prompt and latest turn
    concurrency: int = 4                     # parallel sessions in a sweep (rate-limit permitting)
    n_assess: int = 3                        # post-session items per component            GATE 3: 3 (pre-registered 19 Sep. 2026)
    calibrated: bool = True
    learner: str = "bkt"                     # calibrated learner: "bkt" (bootstrap) or a GBDT .pkl path
    learner_uncalibrated: str = "bkt"        # uncalibrated learner for the calibration-off arm

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

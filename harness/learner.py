"""Learner model f (Section IV-E), features (Table 6), and calibration
(Section IV-F).

Two implementations of f share one interface:
  * BKTLearner: per-component Bayesian knowledge tracing with fixed
    parameters. This is the BOOTSTRAP model: the GBDT of the paper cannot be
    trained before agent telemetry exists, so the first data-collection pass
    runs on BKT and the GBDT is trained from its telemetry. Record this at
    Gate 3.
  * GBDTLearner: sklearn HistGradientBoostingClassifier over the Table 6
    features, optionally wrapped by a Platt (logistic) or isotonic map fitted
    on held-out student instances.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

FEATURE_NAMES = [
    "attempts_c", "successes_c", "rate_c", "attempts_prereq", "successes_prereq",
    "since_last_c", "intervening", "position",
    "latency_z", "attempt_index", "parse_fail_last",
    "difficulty", "kc_index", "family_index",
    "rolling_acc", "total_interactions", "context_length",
    "set_index", "tau", "calibration_flag", "generation", "arm_index",
]


@dataclass
class Interaction:
    component: str
    family: str
    correct: bool
    parse_status: str
    latency_ms: int
    context_length: int
    position: int
    difficulty: float = 0.5


@dataclass
class History:
    interactions: List[Interaction] = field(default_factory=list)
    mastered: List[str] = field(default_factory=list)

    def attempts(self, c: str) -> int:
        return sum(1 for i in self.interactions if i.component == c)

    def successes(self, c: str) -> int:
        return sum(1 for i in self.interactions if i.component == c and i.correct)


def features(h: History, c: str, graph, condition: Dict, kc_index: Dict[str, int],
             family_index: Dict[str, int], next_difficulty: float = 0.5) -> List[float]:
    """Table 6 feature vector for the next item of component c."""
    n = len(h.interactions)
    att, suc = h.attempts(c), h.successes(c)
    prereqs = graph[c].prerequisites
    att_p = sum(h.attempts(q) for q in prereqs)
    suc_p = sum(h.successes(q) for q in prereqs)
    last_idx = max((i for i, x in enumerate(h.interactions) if x.component == c), default=None)
    since = (n - last_idx) if last_idx is not None else n + 1
    lat = [x.latency_ms for x in h.interactions]
    if len(lat) >= 2:
        mu = sum(lat) / len(lat)
        sd = math.sqrt(sum((l - mu) ** 2 for l in lat) / (len(lat) - 1)) or 1.0
        latency_z = (lat[-1] - mu) / sd
    else:
        latency_z = 0.0
    fam = h.interactions[-1].family if h.interactions else ""
    return [
        att, suc, (suc / att) if att else 0.5, att_p, suc_p,
        since, since - 1, n,
        latency_z, att + 1, 1.0 if (h.interactions and h.interactions[-1].parse_status == "failure") else 0.0,
        next_difficulty, kc_index.get(c, -1), family_index.get(fam, -1),
        (sum(1 for x in h.interactions if x.correct) / n) if n else 0.5, n,
        h.interactions[-1].context_length if h.interactions else 0,
        condition.get("set_index", 0), condition.get("tau", 0.85), 1.0 if condition.get("calibration_flag") else 0.0,
        condition.get("generation", 1), condition.get("arm_index", 0),
    ]


class LearnerModel:
    name = "abstract"

    def predict(self, h: History, c: str, graph, condition: Dict, **kw) -> float:  # pragma: no cover
        raise NotImplementedError

    def observe(self, h: History, inter: Interaction) -> None:
        pass


class BKTLearner(LearnerModel):
    name = "bkt"

    def __init__(self, p_init=0.20, p_learn=0.15, p_guess=0.25, p_slip=0.10):
        self.p_init, self.p_learn, self.p_guess, self.p_slip = p_init, p_learn, p_guess, p_slip
        self.p_known: Dict[str, float] = {}

    def predict(self, h: History, c: str, graph, condition: Dict, **kw) -> float:
        pl = self.p_known.get(c, self.p_init)
        return pl * (1 - self.p_slip) + (1 - pl) * self.p_guess

    def observe(self, h: History, inter: Interaction) -> None:
        c = inter.component
        pl = self.p_known.get(c, self.p_init)
        if inter.correct:
            num = pl * (1 - self.p_slip)
            den = num + (1 - pl) * self.p_guess
        else:
            num = pl * self.p_slip
            den = num + (1 - pl) * (1 - self.p_guess)
        post = num / den if den else pl
        self.p_known[c] = post + (1 - post) * self.p_learn


class GBDTLearner(LearnerModel):
    name = "gbdt"

    def __init__(self, model, calibrator=None, kc_index=None, family_index=None):
        self.model, self.calibrator = model, calibrator
        self.kc_index, self.family_index = kc_index or {}, family_index or {}

    def predict(self, h: History, c: str, graph, condition: Dict, next_difficulty: float = 0.5, **kw) -> float:
        x = [features(h, c, graph, condition, self.kc_index, self.family_index, next_difficulty)]
        p = float(self.model.predict_proba(x)[0, 1])
        if self.calibrator is not None:
            p = float(self.calibrator.transform([p])[0]) if hasattr(self.calibrator, "transform") \
                else float(self.calibrator.predict_proba([[p]])[0, 1])
        return min(max(p, 1e-6), 1 - 1e-6)

    @classmethod
    def load(cls, path: str) -> "GBDTLearner":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)


def train_gbdt(X: Sequence[Sequence[float]], y: Sequence[int], X_cal=None, y_cal=None,
               calibration: Optional[str] = None, kc_index=None, family_index=None) -> GBDTLearner:
    """Fit the GBDT and, if requested, a Platt or isotonic map on held-out data.
    Splits must be made by student instance by the caller (Section IV-E)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300)
    model.fit(X, y)
    cal = None
    if calibration and X_cal is not None:
        raw = model.predict_proba(X_cal)[:, 1]
        if calibration == "platt":
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression().fit(raw.reshape(-1, 1), y_cal)
        elif calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            cal = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        else:
            raise ValueError(calibration)
    return GBDTLearner(model, cal, kc_index, family_index)


def ece(probs: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """Expected calibration error, equation (8)."""
    n = len(probs)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total

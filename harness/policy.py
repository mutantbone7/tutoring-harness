"""Algorithm 1 (Policy), line for line. Returns one of
ADVANCE(c'), PRACTICE(c), REMEDIATE(q), REVIEW(r), TERMINATE(reason)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .learner import History, LearnerModel


@dataclass
class Action:
    kind: str                       # ADVANCE | PRACTICE | REMEDIATE | REVIEW | TERMINATE
    component: Optional[str] = None
    reason: Optional[str] = None
    p: float = 0.0


class Policy:
    def __init__(self, content, tau: float, learner: LearnerModel, attempt_cap: int, review_interval: int, condition: Dict):
        self.content, self.tau, self.f = content, tau, learner
        self.A_max, self.k = attempt_cap, review_interval
        self.graph = content.graph()
        self.topo = content.topo_order()
        self.condition = condition
        self.mastered: set = set()
        self.review_queue: List[Tuple[int, str]] = []   # (due position, component)

    def _p(self, h: History, c: str) -> float:
        """f(features(h, c)) with the next item's family and pilot difficulty
        (v2; the study as run passed the previous item's family and 0.5)."""
        fam = self.content.next_family(c)
        return self.f.predict(h, c, self.graph, self.condition,
                              next_difficulty=self.content.family_difficulty(fam), next_family=fam)

    def decide(self, h: History, c: str) -> Action:
        p = self._p(h, c)                                             # line 1
        if p >= self.tau:                                             # line 2
            self.mastered.add(c)                                      # line 3
            h.mastered = sorted(self.mastered)
            self.review_queue.append((len(h.interactions) + self.k, c))
            if self.review_queue and self.review_queue[0][0] <= len(h.interactions):   # line 4
                _, r = self.review_queue.pop(0)
                return Action("REVIEW", r, p=p)
            for cand in self.topo:                                    # lines 5-6
                if cand not in self.mastered and all(q in self.mastered for q in self.graph[cand].prerequisites):
                    return Action("ADVANCE", cand, p=p)               # line 7
            return Action("TERMINATE", reason="mastered", p=p)        # line 8
        for q in self.topo:                                           # lines 9-10
            if q in self.graph[c].prerequisites:
                if self._p(h, q) < self.tau:                          # line 11
                    return Action("REMEDIATE", q, p=p)
        if h.attempts(c) >= self.A_max:                               # line 12
            return Action("TERMINATE", c, reason="non-mastery", p=p)
        return Action("PRACTICE", c, p=p)                             # line 13

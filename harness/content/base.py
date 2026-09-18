"""Content-set interface shared by R, S and Y (Section III-C).

A content set supplies: the specification text Sigma given to teachers, the
knowledge-component graph, item families with numerical parameters, a pure
oracle from parameters to answer and misconception-tagged distractors, an
execution trace per family (for the held-out-family test of Section III-E),
the minimal instruction block used by the equating pilot, and confusable
component pairs for the interference proxy.

Nothing in this file encodes financial content. The R, S and Y sets are
written by the authors and frozen at Gate 0.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import random

LABELS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    prerequisites: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Distractor:
    value: str
    misconception: str          # tag carried into telemetry


@dataclass
class Item:
    item_id: str
    component: str
    family: str
    params: Dict[str, float]
    stem: str
    answer: str                 # correct option text (from the oracle)
    distractors: List[Distractor]
    trace: Tuple[str, ...]      # oracle execution trace (rule ids in order)
    difficulty: float = 0.5     # pilot estimate; 0.5 until the pilot runs
    held_out: bool = False      # set by ContentSet.sample_item from the family

    def render(self, rng: random.Random) -> Tuple[str, str, Dict[str, Optional[str]]]:
        """Return (prompt text, correct label, label->misconception map).
        Option order is shuffled with the harness RNG; the answer key never
        leaves this function except as the returned label, which the harness
        keeps out of every model context."""
        options = [(self.answer, None)] + [(d.value, d.misconception) for d in self.distractors]
        rng.shuffle(options)
        lines = [self.stem, ""]
        tagmap: Dict[str, Optional[str]] = {}
        key = None
        for label, (text, tag) in zip(LABELS, options):
            lines.append(f"{label}. {text}")
            tagmap[label] = tag
            if tag is None:
                key = label
        lines.append("")
        lines.append("Answer with your reasoning if you wish, then end with exactly one final line of the form")
        lines.append("ANSWER: <letter>")
        lines.append(f"where <letter> is one of {', '.join(LABELS[:len(options)])}.")
        assert key is not None
        return "\n".join(lines), key, tagmap


class ItemFamily:
    """A templated family: draws parameters and asks the oracle for the item."""

    def __init__(self, family_id: str, component: str, held_out: bool = False):
        self.family_id = family_id
        self.component = component
        self.held_out = held_out

    def sample(self, rng: random.Random) -> Item:  # pragma: no cover - abstract
        raise NotImplementedError


class ContentSet:
    """Abstract base. Subclasses set `name` to one of R, S, Y (or a demo)."""
    name: str = "abstract"
    chance_level: float = 1.0 / len(LABELS)

    # --- specification and instruction text ---------------------------------
    def specification(self) -> str:
        raise NotImplementedError

    def instruction_block(self) -> str:
        """Minimal instruction block for the equating pilot (Section III-A)."""
        raise NotImplementedError

    # --- structure ------------------------------------------------------------
    def components(self) -> List[Component]:
        raise NotImplementedError

    def families(self) -> List[ItemFamily]:
        raise NotImplementedError

    def confusable_pairs(self) -> List[Tuple[str, str]]:
        """Pairs of components used by the interference proxy."""
        return []

    # --- derived helpers ------------------------------------------------------
    def graph(self) -> Dict[str, Component]:
        return {c.id: c for c in self.components()}

    def topo_order(self) -> List[str]:
        graph = self.graph()
        order: List[str] = []
        seen: set = set()

        def visit(cid: str) -> None:
            if cid in seen:
                return
            for p in graph[cid].prerequisites:
                visit(p)
            seen.add(cid)
            order.append(cid)

        for cid in graph:
            visit(cid)
        return order

    def instructed_families(self, component: Optional[str] = None) -> List[ItemFamily]:
        return [f for f in self.families() if not f.held_out and (component is None or f.component == component)]

    def held_out_families(self, component: Optional[str] = None) -> List[ItemFamily]:
        return [f for f in self.families() if f.held_out and (component is None or f.component == component)]

    def sample_item(self, component: str, rng: random.Random, held_out: bool = False) -> Item:
        fams = self.held_out_families(component) if held_out else self.instructed_families(component)
        if not fams:
            raise ValueError(f"no {'held-out' if held_out else 'instructed'} families for {component}")
        fam = rng.choice(fams)
        item = fam.sample(rng)
        item.held_out = fam.held_out
        return item

    # --- structural distinctness (Section III-E) ------------------------------
    def check_held_out_traces(self, rng: random.Random, n: int = 50) -> Dict[str, bool]:
        """A held-out family is structurally distinct when its trace applies a
        rule, or composes rules in an order, that no instructed family
        produces. Checked mechanically over n samples per family."""
        instructed_traces = set()
        for fam in self.instructed_families():
            for _ in range(n):
                instructed_traces.add(fam.sample(rng).trace)
        result = {}
        for fam in self.held_out_families():
            traces = {fam.sample(rng).trace for _ in range(n)}
            result[fam.family_id] = all(t not in instructed_traces for t in traces)
        return result

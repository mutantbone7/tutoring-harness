"""DRAFT content set Y: novel surface AND novel structure, for the author to
review and freeze at Gate 0 (Section III-B).

Y mirrors R component for component (same six-node prerequisite graph, one
family per component, two held-out families composed in the same orders),
but its rules are an invented integer system with no financial or standard
mathematical counterpart. The rules use floor operations so that no closed
form exists: the oracle iterates, and so must a learner. Each R misconception
has a Y analogue that plays the same role (fails to update the base,
ignores the second parameter, ignores the coupling, undercounts steps).

The "cairn" system. A cairn has an integer height q and a stride k.
  C1 pulse:   q -> q + k + floor(q / k)        (stride plus one per full stride)
  C2 t pulses: apply C1 t times                (no closed form)
  C3 leak:    after a pulse, height drops by floor(q' / d), d the leak size
  C4 origin:  given the height after t pulses and k, recover the start height
              (unique, because C1 is strictly increasing in q)
  C5 drain:   each cycle is a pulse followed by a fixed drain m; count the
              cycles until the height reaches zero or below
  C6 merge:   two cairns with shares w and 1-w (in tenths) and tether b in
              {-2..2}: floor(w q1) + floor((1-w) q2) + b * floor(min(q1,q2)/10)

Nothing here is frozen. Gate 1 verifies that a cold model is at chance on Y;
Gate 2 equates its difficulty with R and S by adjusting parameter ranges.
"""
from __future__ import annotations
import random
from typing import Dict, List, Tuple
from .base import ContentSet, Component, ItemFamily, Item, Distractor


def pulse(q: int, k: int) -> int:
    return q + k + q // k


def pulses(q: int, k: int, t: int) -> int:
    for _ in range(t):
        q = pulse(q, k)
    return q


def leak(q: int, d: int) -> int:
    return q - q // d


def origin_ge(target: int, k: int, t: int) -> int:
    """Smallest q whose height after t pulses is at least target."""
    q = 1
    while pulses(q, k, t) < target:
        q += 1
    return q


def origin(target: int, k: int, t: int) -> int:
    """Inverse of `pulses`: unique because pulse is strictly increasing in q."""
    q = origin_ge(target, k, t)
    assert pulses(q, k, t) == target, "target not reachable"
    return q


def drain_cycles(q: int, k: int, m: int, cap: int = 500) -> int:
    n = 0
    while q > 0 and n < cap:
        q = pulse(q, k) - m
        n += 1
    return n


def merge(q1: int, q2: int, w10: int, b: int) -> int:
    return (w10 * q1) // 10 + ((10 - w10) * q2) // 10 + b * (min(q1, q2) // 10)


class IntFamily(ItemFamily):
    trace: Tuple[str, ...] = ()
    unit = "marks"

    def params(self, rng):  # pragma: no cover - abstract
        raise NotImplementedError

    def stem(self, p):  # pragma: no cover
        raise NotImplementedError

    def correct(self, p):  # pragma: no cover
        raise NotImplementedError

    def wrong(self, p):  # pragma: no cover
        raise NotImplementedError

    def sample(self, rng: random.Random) -> Item:
        for _ in range(300):
            p = self.params(rng)
            ans = int(self.correct(p))
            ds = [(int(v), tag) for v, tag in self.wrong(p)]
            vals = [ans] + [v for v, _ in ds]
            if len(set(vals)) == 4 and all(v > 0 for v in vals):
                break
        else:
            raise RuntimeError(f"{self.family_id}: could not draw distinct options")
        pid = "-".join(f"{k}{v}" for k, v in p.items())
        return Item(item_id=f"Y-{self.family_id}-{pid}", component=self.component, family=self.family_id,
                    params=p, stem=self.stem(p), answer=f"{ans} {self.unit}",
                    distractors=[Distractor(f"{v} {self.unit}", tag) for v, tag in ds], trace=self.trace)


class Pulse(IntFamily):
    trace = ("C1",)

    def params(self, rng):
        return {"q": rng.randint(10, 120), "k": rng.randint(2, 9)}

    def stem(self, p):
        return f"A cairn stands at {p['q']} marks with a stride of {p['k']}. After one pulse, what is its height?"

    def correct(self, p):
        return pulse(p["q"], p["k"])

    def wrong(self, p):
        q, k = p["q"], p["k"]
        return [(q + k, "forgets-stride-count"), (q + q // k, "forgets-stride"), (q + k + q % k, "remainder-for-count")]


class Pulses(IntFamily):
    trace = ("C1", "C2")

    def params(self, rng):
        return {"q": rng.randint(10, 80), "k": rng.randint(3, 9), "t": rng.randint(2, 5)}

    def stem(self, p):
        return f"A cairn stands at {p['q']} marks with a stride of {p['k']}. After {p['t']} pulses, what is its height?"

    def correct(self, p):
        return pulses(p["q"], p["k"], p["t"])

    def wrong(self, p):
        q, k, t = p["q"], p["k"], p["t"]
        return [(q + t * (k + q // k), "linear"), (pulses(q, k, t - 1), "off-by-one"), (q + t * k + q // k, "counts-once")]


class PulseLeak(IntFamily):
    trace = ("C1", "C3")

    def params(self, rng):
        return {"q": rng.randint(20, 150), "k": rng.randint(2, 9), "d": rng.randint(3, 12)}

    def stem(self, p):
        return (f"A cairn stands at {p['q']} marks with a stride of {p['k']} and a leak of {p['d']}. "
                f"After one pulse and its leak, what is its height?")

    def correct(self, p):
        return leak(pulse(p["q"], p["k"]), p["d"])

    def wrong(self, p):
        q, k, d = p["q"], p["k"], p["d"]
        return [(pulse(q, k), "ignores-leak"), (pulse(leak(q, d), k), "leak-first"), (pulse(q, k) - d, "subtracts-leak-size")]


class Origin(IntFamily):
    trace = ("C1", "C2", "C4")

    def params(self, rng):
        q0, k, t = rng.randint(5, 60), rng.randint(3, 9), rng.randint(2, 4)
        return {"h": pulses(q0, k, t), "k": k, "t": t}

    def stem(self, p):
        return (f"A cairn with a stride of {p['k']} stands at {p['h']} marks after {p['t']} pulses. "
                f"What was its height before the pulses?")

    def correct(self, p):
        return origin(p["h"], p["k"], p["t"])

    def wrong(self, p):
        h, k, t = p["h"], p["k"], p["t"]
        return [(h - t * k, "ignores-stride-count"), (origin_ge(h, k, t - 1), "off-by-one"),
                (max(1, h - t * (k + h // k)), "linear-reverse")]


class Drain(IntFamily):
    trace = ("C1", "C2", "C5")
    unit = "cycles"

    def params(self, rng):
        q, k = rng.randint(30, 150), rng.randint(3, 9)
        growth = k + q // k
        m = growth + rng.randint(3, 15)          # drain exceeds the pulse so the cairn empties
        return {"q": q, "k": k, "m": m}

    def stem(self, p):
        return (f"A cairn stands at {p['q']} marks with a stride of {p['k']}. Each cycle it takes one pulse and then "
                f"loses a fixed drain of {p['m']} marks. After how many cycles does its height first reach zero or below?")

    def correct(self, p):
        return drain_cycles(p["q"], p["k"], p["m"])

    def wrong(self, p):
        q, k, m = p["q"], p["k"], p["m"]
        return [(-(-q // m), "ignores-pulses"), (-(-pulse(q, k) // m), "pulse-once"), (-(-q // (m - k)), "ignores-stride-count")]


class Merge(IntFamily):
    trace = ("C6",)

    def params(self, rng):
        return {"q1": rng.randint(20, 150), "q2": rng.randint(20, 150), "w": rng.choice([3, 4, 5, 6, 7]),
                "b": rng.choice([-2, -1, 1, 2])}

    def stem(self, p):
        return (f"Two cairns stand at {p['q1']} and {p['q2']} marks. They are merged with a share of {p['w']} tenths "
                f"from the first and the rest from the second, with a tether of {p['b']}. What is the merged height?")

    def correct(self, p):
        return merge(p["q1"], p["q2"], p["w"], p["b"])

    def wrong(self, p):
        q1, q2, w, b = p["q1"], p["q2"], p["w"], p["b"]
        return [(merge(q1, q2, w, 0), "ignores-tether"), ((q1 + q2) // 2, "naive-merge"), (merge(q1, q2, w, -b), "tether-sign")]


class PulseLeakCycles(IntFamily):
    """Held-out: t cycles of pulse then leak. Trace (C1, C2, C3)."""
    trace = ("C1", "C2", "C3")

    def params(self, rng):
        return {"q": rng.randint(20, 100), "k": rng.randint(3, 9), "d": rng.randint(4, 12), "t": rng.randint(2, 4)}

    def stem(self, p):
        return (f"A cairn stands at {p['q']} marks with a stride of {p['k']} and a leak of {p['d']}. "
                f"After {p['t']} cycles of a pulse followed by its leak, what is its height?")

    def correct(self, p):
        q = p["q"]
        for _ in range(p["t"]):
            q = leak(pulse(q, p["k"]), p["d"])
        return q

    def wrong(self, p):
        q, k, d, t = p["q"], p["k"], p["d"], p["t"]
        one = leak(pulse(q, k), d)
        return [(pulses(q, k, t), "ignores-leak"), (q + t * (one - q), "linear"),
                (self.correct({"q": q, "k": k, "d": d, "t": t - 1}), "off-by-one")]


class OriginWithLeak(IntFamily):
    """Held-out: recover the start height given the height after t pulse-and-leak cycles. Trace (C1, C3, C2, C4)."""
    trace = ("C1", "C3", "C2", "C4")

    def params(self, rng):
        q0, k, d, t = rng.randint(10, 60), rng.randint(3, 9), rng.randint(4, 12), rng.randint(2, 3)
        q = q0
        for _ in range(t):
            q = leak(pulse(q, k), d)
        return {"h": q, "k": k, "d": d, "t": t, "_q0": q0}

    def stem(self, p):
        return (f"A cairn with a stride of {p['k']} and a leak of {p['d']} stands at {p['h']} marks after {p['t']} "
                f"cycles of a pulse followed by its leak. What is the smallest starting height that produces this?")

    def correct(self, p):
        q0 = 1
        while True:
            q = q0
            for _ in range(p["t"]):
                q = leak(pulse(q, p["k"]), p["d"])
            if q >= p["h"]:
                return q0
            q0 += 1

    def wrong(self, p):
        h, k, d, t = p["h"], p["k"], p["d"], p["t"]
        return [(origin_ge(h, k, t), "ignores-leak"),
                (max(1, h - t * (k + h // k) + t * (h // d)), "linear"), (max(1, h - k - h // k + h // d), "off-by-one")]

    def sample(self, rng: random.Random) -> Item:
        item = super().sample(rng)
        item.params.pop("_q0", None)
        return item


SPEC_Y = """SPECIFICATION: cairn rule system (DRAFT, not frozen)
A cairn has a height in marks, a stride, and sometimes a leak. Heights, strides and leaks are whole numbers.
Rule C1 (pulse): one pulse raises the height by the stride plus one mark for every full stride contained in the current height; a height of 20 with stride 6 becomes 20 + 6 + 3 = 29.
Rule C2 (repeated pulses): pulses are applied one after another, and the count of full strides is taken afresh from the height at each pulse; there is no shortcut that uses the starting count for every pulse.
Rule C3 (leak): after a pulse, a cairn with a leak of size d loses one mark for every d marks in its new height; the leak is applied to the pulsed height, not to the height before the pulse.
Rule C4 (origin): because a pulse always raises a taller cairn to a taller height, the height after a given number of pulses fixes the starting height uniquely; it is found by undoing the pulses, not by subtracting strides.
Rule C5 (drain): in a drained cairn each cycle is a pulse followed by the removal of a fixed drain; the cairn empties only if the drain exceeds what a pulse adds, and the number of cycles is larger than the height divided by the drain suggests.
Rule C6 (merge): two cairns merged with shares w and 1 - w in tenths and a tether b have height floor(w q1) + floor((1 - w) q2) + b times one mark per ten marks of the shorter cairn; the shares alone do not fix the merged height, the tether does.
Common errors: using the starting stride count for every pulse; ignoring the leak; counting cycles as height over drain; merging without the tether.
"""

INSTRUCTION_Y = """You will answer multiple-choice questions about cairns. A cairn has a whole-number height in marks and a whole-number stride; some cairns also have a leak.
Pulse: one pulse raises the height by the stride plus one mark for every full stride contained in the current height; a height of 20 with stride 6 becomes 20 + 6 + 3 = 29.
Repeated pulses: apply the pulse rule once per pulse, recounting the full strides from the new height each time.
Leak: after a pulse, a cairn with leak d loses one mark for every d marks in its new pulsed height, rounding down.
Origin: the height after a given number of pulses fixes the starting height uniquely; find the starting height that produces the stated height.
Drain: each cycle is one pulse followed by removing a fixed drain; count the cycles until the height first reaches zero or below.
Merge: two cairns with heights q1 and q2, a share w in tenths and a tether b merge to floor(w q1 / 10) + floor((10 - w) q2 / 10) + b times floor(min(q1, q2) / 10).
Answer each question with a single letter."""


class YContentSet(ContentSet):
    name = "Y"

    def specification(self) -> str:
        return SPEC_Y

    def instruction_block(self) -> str:
        return INSTRUCTION_Y

    def components(self) -> List[Component]:
        return [
            Component("C1", "the pulse"),
            Component("C2", "repeated pulses", prerequisites=("C1",)),
            Component("C3", "the leak", prerequisites=("C1",)),
            Component("C4", "finding the origin", prerequisites=("C2",)),
            Component("C5", "draining a cairn", prerequisites=("C2",)),
            Component("C6", "merging cairns"),
        ]

    def families(self) -> List[ItemFamily]:
        return [Pulse("pulse", "C1"), Pulses("pulses", "C2"), PulseLeak("pulse-leak", "C3"), Origin("origin", "C4"),
                Drain("drain", "C5"), Merge("merge", "C6"),
                PulseLeakCycles("pulse-leak-cycles", "C3", held_out=True), OriginWithLeak("origin-with-leak", "C4", held_out=True)]

    def confusable_pairs(self) -> List[Tuple[str, str]]:
        return [("C2", "C4"), ("C1", "C3")]

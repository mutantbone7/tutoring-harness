"""DEMO CONTENT SET. Not R, S or Y. Exists only so the harness can be run and
tested end to end before the frozen specifications exist. Its rules are
arbitrary and carry no financial meaning. Do not report results from it."""
from __future__ import annotations
import random
from typing import List, Tuple
from .base import ContentSet, Component, ItemFamily, Item, Distractor

SPEC = """DEMO SPECIFICATION (not a study content set)
Rule K1 (scaling): a quantity q scaled by factor m gives q * m.
Rule K2 (repeated scaling): scaling by m for t steps gives q * m^t.
Rule K3 (offset adjustment): an offset d applied after scaling gives (q * m) - d.
Misconceptions: linear (q * m * t instead of q * m^t); offset-first ((q - d) * m).
"""

INSTRUCTION_BLOCK = """You will answer multiple-choice questions about a rule system.
Scaling multiplies a quantity by a factor. Repeated scaling applies the factor once per step.
An offset is subtracted after scaling. Answer each question with a single letter."""


class F_Scale(ItemFamily):
    def sample(self, rng: random.Random) -> Item:
        q, m = rng.randint(2, 30), rng.randint(2, 9)
        ans = q * m
        return Item(
            item_id=f"demo-K1-scale-{q}-{m}", component="K1", family="scale",
            params={"q": q, "m": m},
            stem=f"A quantity of {q} units is scaled by a factor of {m}. What is the result?",
            answer=str(ans),
            distractors=[Distractor(str(q + m), "add-not-multiply"), Distractor(str(q * m + q), "double-count"), Distractor(str(q * (m - 1)), "off-by-one")],
            trace=("K1",),
        )


class F_Repeat(ItemFamily):
    def sample(self, rng: random.Random) -> Item:
        q, m, t = rng.randint(2, 12), rng.randint(2, 4), rng.randint(2, 4)
        ans = q * m ** t
        return Item(
            item_id=f"demo-K2-repeat-{q}-{m}-{t}", component="K2", family="repeat",
            params={"q": q, "m": m, "t": t},
            stem=f"A quantity of {q} units is scaled by a factor of {m} for {t} steps. What is the result?",
            answer=str(ans),
            distractors=[Distractor(str(q * m * t), "linear"), Distractor(str(q * m ** (t - 1)), "off-by-one"), Distractor(str(q + m * t), "add-not-multiply")],
            trace=("K1", "K2"),
        )


class F_Offset(ItemFamily):
    def sample(self, rng: random.Random) -> Item:
        q, m, d = rng.randint(5, 30), rng.randint(2, 6), rng.randint(1, 10)
        ans = q * m - d
        return Item(
            item_id=f"demo-K3-offset-{q}-{m}-{d}", component="K3", family="offset",
            params={"q": q, "m": m, "d": d},
            stem=f"A quantity of {q} units is scaled by {m}, then an offset of {d} is applied. What is the result?",
            answer=str(ans),
            distractors=[Distractor(str((q - d) * m), "offset-first"), Distractor(str(q * m + d), "sign-error"), Distractor(str(q * (m - d)), "offset-on-factor")],
            trace=("K1", "K3"),
        )


class F_RepeatOffset(ItemFamily):
    """Held-out family: composes K2 then K3, an order no instructed family produces."""
    def sample(self, rng: random.Random) -> Item:
        q, m, t, d = rng.randint(2, 10), rng.randint(2, 3), rng.randint(2, 3), rng.randint(1, 9)
        ans = q * m ** t - d
        return Item(
            item_id=f"demo-K3-repoff-{q}-{m}-{t}-{d}", component="K3", family="repeat-offset",
            params={"q": q, "m": m, "t": t, "d": d},
            stem=f"A quantity of {q} units is scaled by {m} for {t} steps, then an offset of {d} is applied. What is the result?",
            answer=str(ans),
            distractors=[Distractor(str(q * m * t - d), "linear"), Distractor(str((q - d) * m ** t), "offset-first"), Distractor(str(q * m ** t + d), "sign-error")],
            trace=("K1", "K2", "K3"),
        )


class DemoContentSet(ContentSet):
    name = "demo"

    def specification(self) -> str:
        return SPEC

    def instruction_block(self) -> str:
        return INSTRUCTION_BLOCK

    def components(self) -> List[Component]:
        return [
            Component("K1", "scaling"),
            Component("K2", "repeated scaling", prerequisites=("K1",)),
            Component("K3", "offset adjustment", prerequisites=("K1",)),
        ]

    def families(self) -> List[ItemFamily]:
        return [F_Scale("scale", "K1"), F_Repeat("repeat", "K2"), F_Offset("offset", "K3"),
                F_RepeatOffset("repeat-offset", "K3", held_out=True)]

    def confusable_pairs(self) -> List[Tuple[str, str]]:
        return [("K2", "K3")]

"""DRAFT content sets R (real financial surface and structure) and S (real
structure, renewed surface), for the author to review and freeze at Gate 0.

R draws its components from what the Big Three and Big Five (Lusardi and
Mitchell), the PISA financial literacy framework and the FINRA survey test:
percentages and simple interest, compounding, inflation and real return,
discounting, minimum payments on revolving debt, and diversification.
Distractors are computed by applying the documented misconceptions cited in
Section II-F of the manuscript: linear extrapolation of compound growth
(Stango and Zinman), money illusion (Shafir, Diamond and Tversky), naive
diversification (Benartzi and Thaler) and minimum-payment underestimation
(Soll, Keeney and Larrick), plus generic computational errors.

S is R with every surface term, unit and symbol renamed through `S_SURFACE`
and the same oracle functions, so its isomorphism to R holds by construction
(Section III-B). Items are numeric so that a frontier model is not at
ceiling by recall alone; the Gate 2 pilot decides whether headroom exists.

Nothing here is frozen. The author reviews every family against the
instruments, adjusts, commits, and records the hash in the manuscript.
"""
from __future__ import annotations
import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
from .base import ContentSet, Component, ItemFamily, Item, Distractor


# ----------------------------------------------------------------------------
# Surface: everything a learner sees that is not structure.
# ----------------------------------------------------------------------------
@dataclass
class Surface:
    name: str
    cur: str = "$"                 # currency symbol placed before amounts
    pct: str = "%"                 # rate unit placed after numbers
    year: str = "year"
    years: str = "years"
    month: str = "month"
    months: str = "months"
    terms: Dict[str, str] = field(default_factory=dict)   # vocabulary substitutions applied to text

    def t(self, s: str) -> str:
        """Apply vocabulary substitutions, longest key first, matching the
        capitalised form as well, then repair indefinite articles."""
        for k, v in sorted(self.terms.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(k, v)
            s = s.replace(k[0].upper() + k[1:], v[0].upper() + v[1:])
        s = re.sub(r"\b([Aa])n (?=[bcdfghjklmnpqrstvwxyz])", r"\1 ", s)   # "an vault" -> "a vault"
        s = re.sub(r"\ba (?=[aeiou])", "an ", s)                          # "a accrual" -> "an accrual"
        return s

    def money(self, x: float) -> str:
        return f"{self.cur}{x:,.2f}"

    def rate(self, r: float) -> str:
        return f"{r:g}{self.pct}"

    def yrs(self, t: int) -> str:
        return f"{t} {self.year if t == 1 else self.years}"

    def mos(self, n: float) -> str:
        return f"{n:g} {self.month if n == 1 else self.months}"


R_SURFACE = Surface(name="R")

S_SURFACE = Surface(
    name="S", cur="K", pct=" pph", year="cycle", years="cycles", month="phase", months="phases",
    terms={
        "personal-finance": "vault-and-ledger", "compounded monthly": "layered once per phase",
        "savings account": "holding vault", "account": "vault", "interest rate": "accrual rate",
        "interest": "accrual", "compounded annually": "layered once per cycle", "compounded": "layered",
        "inflation": "drift", "prices": "tariffs", "purchasing power": "claim power",
        "credit card": "revolving ledger", "balance": "standing", "minimum payment": "floor remittance",
        "payment": "remittance", "APR": "cycle accrual rate", "stock": "unit", "stocks": "units",
        "mutual fund": "pooled bundle", "portfolio": "bundle", "standard deviation": "spread",
        "correlation": "co-drift", "deposit": "lodge", "invest": "lodge", "bond": "fixed-claim note",
        "loan": "advance", "bank": "depository", "money": "kestrels", "percent": "parts per hundred",
        "per year": "per cycle", "years": "cycles", "year": "cycle", "months": "phases", "month": "phase",
    },
)


def _r2(x: float) -> float:
    return round(x, 2)


# ----------------------------------------------------------------------------
# Families. Each has: parameter sampler, correct formula, distractor formulas
# with misconception tags, a stem template, and an execution trace.
# ----------------------------------------------------------------------------
class NumericFamily(ItemFamily):
    """Numeric multiple-choice family. Resamples until the four options are
    distinct at two decimals, so no distractor collides with the answer."""

    def __init__(self, family_id, component, surface: Surface, held_out=False):
        super().__init__(family_id, component, held_out)
        self.sf = surface

    def params(self, rng: random.Random) -> Dict[str, float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def stem(self, p: Dict[str, float]) -> str:  # pragma: no cover
        raise NotImplementedError

    def correct(self, p: Dict[str, float]) -> float:  # pragma: no cover
        raise NotImplementedError

    def wrong(self, p: Dict[str, float]) -> List[Tuple[float, str]]:  # pragma: no cover
        raise NotImplementedError

    trace: Tuple[str, ...] = ()
    fmt: str = "money"        # money | number | months

    def render_value(self, x: float) -> str:
        if self.fmt == "money":
            return self.sf.money(x)
        if self.fmt == "months":
            return self.sf.mos(_r2(x))
        return f"{_r2(x):g}"

    def sample(self, rng: random.Random) -> Item:
        for _ in range(200):
            p = self.params(rng)
            ans = _r2(self.correct(p))
            ds = [(_r2(v), tag) for v, tag in self.wrong(p)]
            vals = [ans] + [v for v, _ in ds]
            if len(set(vals)) == 4 and all(v > 0 for v in vals):
                break
        else:
            raise RuntimeError(f"{self.family_id}: could not draw distinct options")
        pid = "-".join(f"{k}{v:g}" for k, v in p.items())
        return Item(
            item_id=f"{self.sf.name}-{self.family_id}-{pid}", component=self.component, family=self.family_id,
            params=p, stem=self.sf.t(self.stem(p)), answer=self.render_value(ans),
            distractors=[Distractor(self.render_value(v), tag) for v, tag in ds], trace=self.trace,
        )


# C1 -------------------------------------------------------------------------
class SimpleInterest(NumericFamily):
    trace = ("C1",)

    def params(self, rng):
        return {"P": rng.choice(range(200, 5001, 50)), "r": rng.choice([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8])}

    def stem(self, p):
        return (f"You deposit {self.sf.money(p['P'])} in a savings account paying simple interest of "
                f"{self.sf.rate(p['r'])} per year. How much interest is earned in one year?")

    def correct(self, p):
        return p["P"] * p["r"] / 100

    def wrong(self, p):
        return [(p["P"] * p["r"], "percent-as-multiplier"), (p["P"] * p["r"] / 1000, "decimal-shift"),
                (p["P"] + p["r"], "add-rate-to-principal")]


# C2 -------------------------------------------------------------------------
class CompoundBalance(NumericFamily):
    trace = ("C1", "C2")

    def params(self, rng):
        return {"P": rng.choice(range(100, 5001, 100)), "r": rng.choice([2, 3, 4, 5, 6, 7, 8, 10]),
                "t": rng.randint(3, 12)}

    def stem(self, p):
        return (f"You deposit {self.sf.money(p['P'])} in an account paying {self.sf.rate(p['r'])} per year, "
                f"compounded annually. After {self.sf.yrs(p['t'])}, how much is in the account?")

    def correct(self, p):
        return p["P"] * (1 + p["r"] / 100) ** p["t"]

    def wrong(self, p):
        return [(p["P"] * (1 + p["r"] / 100 * p["t"]), "linear"),
                (p["P"] * (1 + p["r"] / 100) ** (p["t"] - 1), "off-by-one"),
                (p["P"] * (1 + p["r"] * p["t"] / 100) ** 1 + p["P"] * p["r"] / 100, "rate-scaling")]


# C3 -------------------------------------------------------------------------
class RealPurchasingPower(NumericFamily):
    trace = ("C1", "C3")
    fmt = "money"

    def params(self, rng):
        return {"P": rng.choice(range(500, 5001, 100)), "r": rng.choice([0.5, 1, 1.5, 2, 3, 4, 5]),
                "i": rng.choice([1, 2, 3, 4, 5, 6, 8])}

    def stem(self, p):
        return (f"You have {self.sf.money(p['P'])} in an account paying {self.sf.rate(p['r'])} per year. "
                f"Inflation is {self.sf.rate(p['i'])} per year. After one year, what is the purchasing power of "
                f"the account in today's prices?")

    def correct(self, p):
        return p["P"] * (1 + p["r"] / 100) / (1 + p["i"] / 100)

    def wrong(self, p):
        return [(p["P"] * (1 + p["r"] / 100), "money-illusion"),
                (p["P"] * (1 + (p["r"] - p["i"]) / 100), "rate-subtraction"),
                (p["P"] * (1 + p["i"] / 100) / (1 + p["r"] / 100), "inverted")]


# C4 -------------------------------------------------------------------------
class PresentValue(NumericFamily):
    trace = ("C1", "C2", "C4")

    def params(self, rng):
        return {"F": rng.choice(range(1000, 20001, 500)), "r": rng.choice([2, 3, 4, 5, 6, 8, 10]),
                "t": rng.randint(2, 10)}

    def stem(self, p):
        return (f"A bond pays {self.sf.money(p['F'])} in {self.sf.yrs(p['t'])} and nothing before then. If the "
                f"interest rate is {self.sf.rate(p['r'])} per year compounded annually, what is the bond worth today?")

    def correct(self, p):
        return p["F"] / (1 + p["r"] / 100) ** p["t"]

    def wrong(self, p):
        return [(p["F"] / (1 + p["r"] / 100 * p["t"]), "linear"),
                (p["F"] * (1 - p["r"] / 100 * p["t"]), "linear-subtractive"),
                (p["F"] / (1 + p["r"] / 100) ** (p["t"] - 1), "off-by-one")]


# C5 -------------------------------------------------------------------------
class MinimumPaymentMonths(NumericFamily):
    trace = ("C1", "C2", "C5")
    fmt = "months"

    def params(self, rng):
        B = rng.choice(range(1000, 6001, 250))
        apr = rng.choice([12, 15, 18, 20, 24])
        r = apr / 12 / 100
        M = _r2(max(B * r * rng.choice([1.5, 2, 2.5, 3]), 25))   # payment above interest so the loan clears
        return {"B": B, "apr": apr, "M": M}

    def stem(self, p):
        return (f"A credit card balance is {self.sf.money(p['B'])} at an APR of {self.sf.rate(p['apr'])}, "
                f"compounded monthly. You pay a fixed {self.sf.money(p['M'])} each month and make no new charges. "
                f"About how many months until the balance is repaid?")

    def correct(self, p):
        r = p["apr"] / 12 / 100
        return -math.log(1 - r * p["B"] / p["M"]) / math.log(1 + r)

    def wrong(self, p):
        r = p["apr"] / 12 / 100
        return [(p["B"] / p["M"], "minimum-payment-underestimate"),
                (p["B"] * (1 + r) / p["M"], "interest-once"),
                (p["B"] * (1 + p["apr"] / 100) / p["M"], "annual-rate-as-monthly")]


# C6 -------------------------------------------------------------------------
class PortfolioRisk(NumericFamily):
    trace = ("C6",)
    fmt = "number"

    def params(self, rng):
        return {"s1": rng.choice([10, 12, 15, 18, 20, 25]), "s2": rng.choice([8, 10, 12, 15, 20, 30]),
                "w": rng.choice([0.3, 0.4, 0.5, 0.6, 0.7]), "rho": rng.choice([-0.5, -0.2, 0, 0.2, 0.5, 0.8])}

    def stem(self, p):
        return (f"Stock A has a standard deviation of returns of {self.sf.rate(p['s1'])} and stock B of "
                f"{self.sf.rate(p['s2'])}. The correlation between them is {p['rho']}. A portfolio puts "
                f"{self.sf.rate(p['w']*100)} in A and the rest in B. What is the portfolio's standard deviation, "
                f"in the same units?")

    def correct(self, p):
        w = p["w"]
        return math.sqrt((w * p["s1"]) ** 2 + ((1 - w) * p["s2"]) ** 2 + 2 * w * (1 - w) * p["rho"] * p["s1"] * p["s2"])

    def wrong(self, p):
        w = p["w"]
        return [(w * p["s1"] + (1 - w) * p["s2"], "no-diversification-benefit"),
                (math.sqrt((w * p["s1"]) ** 2 + ((1 - w) * p["s2"]) ** 2), "ignores-correlation"),
                ((p["s1"] + p["s2"]) / 2, "naive-diversification")]


# Held-out (transfer) families: compose rules in orders no instructed family produces.
class RealCompoundBalance(NumericFamily):
    """C2 then C3: real value of a compounded deposit after t years of inflation. Trace (C1,C2,C3)."""
    trace = ("C1", "C2", "C3")

    def params(self, rng):
        return {"P": rng.choice(range(500, 5001, 100)), "r": rng.choice([3, 4, 5, 6, 7, 8]),
                "i": rng.choice([1, 2, 3, 4, 5]), "t": rng.randint(2, 8)}

    def stem(self, p):
        return (f"You deposit {self.sf.money(p['P'])} at {self.sf.rate(p['r'])} per year, compounded annually, "
                f"while inflation runs at {self.sf.rate(p['i'])} per year. After {self.sf.yrs(p['t'])}, what is the "
                f"purchasing power of the account in today's prices?")

    def correct(self, p):
        return p["P"] * ((1 + p["r"] / 100) / (1 + p["i"] / 100)) ** p["t"]

    def wrong(self, p):
        return [(p["P"] * (1 + p["r"] / 100) ** p["t"], "money-illusion"),
                (p["P"] * (1 + (p["r"] - p["i"]) / 100 * p["t"]), "linear"),
                (p["P"] * ((1 + p["r"] / 100) / (1 + p["i"] / 100)) ** (p["t"] - 1), "off-by-one")]


class PresentValueOfRealAmount(NumericFamily):
    """C3 then C4: today's money needed to hold a real target after t years of inflation. Trace (C1,C3,C2,C4)."""
    trace = ("C1", "C3", "C2", "C4")

    def params(self, rng):
        return {"F": rng.choice(range(1000, 10001, 500)), "r": rng.choice([3, 4, 5, 6, 8]),
                "i": rng.choice([1, 2, 3, 4]), "t": rng.randint(2, 8)}

    def stem(self, p):
        return (f"You want to have, in {self.sf.yrs(p['t'])}, the purchasing power that {self.sf.money(p['F'])} has "
                f"today. Inflation is {self.sf.rate(p['i'])} per year and your account pays {self.sf.rate(p['r'])} per "
                f"year, compounded annually. How much must you deposit today?")

    def correct(self, p):
        return p["F"] * (1 + p["i"] / 100) ** p["t"] / (1 + p["r"] / 100) ** p["t"]

    def wrong(self, p):
        return [(p["F"] / (1 + p["r"] / 100) ** p["t"], "money-illusion"),
                (p["F"] * (1 + p["i"] / 100 * p["t"]) / (1 + p["r"] / 100 * p["t"]), "linear"),
                (p["F"] * (1 + p["r"] / 100) ** p["t"] / (1 + p["i"] / 100) ** p["t"], "inverted")]


# ----------------------------------------------------------------------------
# Specification and instruction text (shared structure, surface-mapped).
# ----------------------------------------------------------------------------
SPEC_R = """SPECIFICATION: personal-finance rule system (DRAFT, not frozen)
Rule C1 (simple interest): interest for one year on a principal P at rate r percent is P times r divided by 100.
Rule C2 (compound interest): a principal P at rate r percent compounded annually for t years grows to P times (1 + r/100) to the power t; interest earned in each year is added to the principal before the next year's interest is computed.
Rule C3 (inflation and real return): if prices rise by i percent per year, money that grows by r percent buys (1 + r/100) divided by (1 + i/100) times as much after one year; nominal growth and real growth are different quantities.
Rule C4 (present value): an amount F received in t years is worth F divided by (1 + r/100) to the power t today, when money can earn r percent per year compounded annually.
Rule C5 (revolving debt): a balance B at a monthly rate m = APR/12/100 with a fixed monthly payment M is repaid after n months where n = minus log(1 minus m times B over M) divided by log(1 + m); paying only slightly more than the monthly interest takes far longer than B divided by M suggests.
Rule C6 (diversification): the standard deviation of a two-asset portfolio with weights w and 1 minus w, asset standard deviations s1 and s2 and correlation rho is the square root of (w s1) squared plus ((1 minus w) s2) squared plus 2 w (1 minus w) rho s1 s2; the number of holdings is not what reduces risk, the correlation between them is.
Common errors: treating compound growth as linear; confusing nominal with real amounts; assuming a balance is repaid in balance-over-payment months; averaging asset risks as if correlation were one.
"""

INSTRUCTION_R = """You will answer numerical multiple-choice questions about interest, inflation, debt and investment risk.
Simple interest: one year's interest on principal P at rate r percent is P times r/100.
Compound interest: P at r percent compounded annually for t years grows to P times (1 + r/100) to the power t.
Inflation: money that grows by r percent buys (1 + r/100)/(1 + i/100) times as much after one year when prices rise by i percent.
Present value: an amount F due in t years is worth F divided by (1 + r/100) to the power t today.
Revolving debt: a balance B at monthly rate m = APR/1200 with fixed payment M is repaid after n = -ln(1 - mB/M)/ln(1 + m) months.
Portfolio risk: with weights w and 1 - w, deviations s1 and s2 and correlation rho, the portfolio deviation is the square root of (w s1)^2 + ((1 - w) s2)^2 + 2 w (1 - w) rho s1 s2.
Answer each question with a single letter."""


class FinanceContentSet(ContentSet):
    """Structure shared by R and S; the surface is injected."""
    surface: Surface = R_SURFACE

    def __init__(self):
        self.name = self.surface.name

    def specification(self) -> str:
        return self.surface.t(SPEC_R)

    def instruction_block(self) -> str:
        return self.surface.t(INSTRUCTION_R)

    def components(self) -> List[Component]:
        t = self.surface.t
        return [
            Component("C1", t("simple interest")),
            Component("C2", t("compound interest"), prerequisites=("C1",)),
            Component("C3", t("inflation and real return"), prerequisites=("C1",)),
            Component("C4", t("present value"), prerequisites=("C2",)),
            Component("C5", t("repaying revolving debt"), prerequisites=("C2",)),
            Component("C6", t("diversification and portfolio risk")),
        ]

    def families(self) -> List[ItemFamily]:
        sf = self.surface
        return [
            SimpleInterest("simple-interest", "C1", sf),
            CompoundBalance("compound-balance", "C2", sf),
            RealPurchasingPower("real-purchasing-power", "C3", sf),
            PresentValue("present-value", "C4", sf),
            MinimumPaymentMonths("minimum-payment-months", "C5", sf),
            PortfolioRisk("portfolio-risk", "C6", sf),
            RealCompoundBalance("real-compound-balance", "C3", sf, held_out=True),
            PresentValueOfRealAmount("present-value-real", "C4", sf, held_out=True),
        ]

    def confusable_pairs(self) -> List[Tuple[str, str]]:
        return [("C2", "C4"), ("C1", "C3")]


class RContentSet(FinanceContentSet):
    surface = R_SURFACE


class SContentSet(FinanceContentSet):
    surface = S_SURFACE

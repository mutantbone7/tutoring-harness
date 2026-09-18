"""Oracle unit tests against hand-computed cases and property tests for the
invariants the specifications imply (Section III-C)."""
import math, random
import pytest
from harness.content import load_content
from harness.content.finance import (SimpleInterest, CompoundBalance, RealPurchasingPower, PresentValue,
                                     MinimumPaymentMonths, PortfolioRisk, RealCompoundBalance,
                                     PresentValueOfRealAmount, R_SURFACE, S_SURFACE)
from harness.content import novel
from harness.content.base import LABELS

R = load_content("R"); S = load_content("S"); Y = load_content("Y")


# ---- hand-computed cases, set R -------------------------------------------
def test_r_oracles_hand_computed():
    assert SimpleInterest("f", "C1", R_SURFACE).correct({"P": 1000, "r": 3}) == 30
    assert abs(CompoundBalance("f", "C2", R_SURFACE).correct({"P": 1000, "r": 5, "t": 3}) - 1157.625) < 1e-9
    assert abs(RealPurchasingPower("f", "C3", R_SURFACE).correct({"P": 1000, "r": 2, "i": 5}) - 971.4285714) < 1e-6
    assert abs(PresentValue("f", "C4", R_SURFACE).correct({"F": 1000, "r": 10, "t": 2}) - 826.4462810) < 1e-6
    n = MinimumPaymentMonths("f", "C5", R_SURFACE).correct({"B": 1000, "apr": 12, "M": 100})
    assert abs(n - (-math.log(0.9) / math.log(1.01))) < 1e-9 and 10.5 < n < 10.7
    assert abs(PortfolioRisk("f", "C6", R_SURFACE).correct({"s1": 10, "s2": 20, "w": 0.5, "rho": 0}) - math.sqrt(125)) < 1e-9
    assert abs(PortfolioRisk("f", "C6", R_SURFACE).correct({"s1": 10, "s2": 20, "w": 0.5, "rho": 1}) - 15) < 1e-9
    assert abs(RealCompoundBalance("f", "C3", R_SURFACE).correct({"P": 1000, "r": 5, "i": 5, "t": 4}) - 1000) < 1e-9
    assert abs(PresentValueOfRealAmount("f", "C4", R_SURFACE).correct({"F": 1000, "r": 5, "i": 5, "t": 4}) - 1000) < 1e-9


def test_r_misconceptions_reproduce_named_errors():
    p = {"P": 1000, "r": 5, "t": 3}
    wrong = dict((tag, v) for v, tag in CompoundBalance("f", "C2", R_SURFACE).wrong(p))
    assert wrong["linear"] == 1150                       # simple-interest thinking
    assert abs(wrong["off-by-one"] - 1102.5) < 1e-9
    p = {"P": 1000, "r": 2, "i": 5}
    wrong = dict((tag, v) for v, tag in RealPurchasingPower("f", "C3", R_SURFACE).wrong(p))
    assert wrong["money-illusion"] == 1020               # nominal reported as real
    p = {"B": 1000, "apr": 12, "M": 100}
    wrong = dict((tag, v) for v, tag in MinimumPaymentMonths("f", "C5", R_SURFACE).wrong(p))
    assert wrong["minimum-payment-underestimate"] == 10  # balance over payment


# ---- hand-computed cases, set Y -------------------------------------------
def test_y_oracles_hand_computed():
    assert novel.pulse(20, 6) == 29                      # the worked example in the specification
    assert novel.pulses(20, 6, 2) == 39
    assert novel.leak(29, 5) == 24
    assert novel.origin(39, 6, 2) == 20
    assert novel.drain_cycles(30, 3, 20) == 3            # 30->23->13->0
    assert novel.merge(56, 111, 6, -2) == 67
    assert novel.merge(56, 111, 6, 0) == 77


def test_y_pulse_is_strictly_increasing_so_origin_is_unique():
    for k in range(2, 10):
        heights = [novel.pulses(q, k, 3) for q in range(1, 300)]
        assert all(b > a for a, b in zip(heights, heights[1:]))


# ---- structural properties --------------------------------------------------
@pytest.mark.parametrize("content", [R, S, Y])
def test_options_distinct_positive_and_key_unique(content):
    rng = random.Random(7)
    for fam in content.families():
        for _ in range(100):
            item = fam.sample(rng)
            vals = [item.answer] + [d.value for d in item.distractors]
            assert len(set(vals)) == 4
            prompt, key, tagmap = item.render(rng)
            assert key in LABELS and tagmap[key] is None and sum(v is None for v in tagmap.values()) == 1
            assert "SPECIFICATION" not in prompt


@pytest.mark.parametrize("content", [R, S, Y])
def test_held_out_families_are_structurally_distinct(content):
    assert all(content.check_held_out_traces(random.Random(1), 50).values())


def test_s_is_r_under_surface_substitution():
    """Same oracle on the same parameters; only the surface differs."""
    rng_r, rng_s = random.Random(11), random.Random(11)
    for fr, fs in zip(R.families(), S.families()):
        ir, is_ = fr.sample(rng_r), fs.sample(rng_s)
        assert ir.params == is_.params and ir.trace == is_.trace and ir.family == is_.family
        assert [d.misconception for d in ir.distractors] == [d.misconception for d in is_.distractors]
        assert ir.stem != is_.stem
    for word in ("interest", "inflation", "portfolio", "stock", "credit card", "$"):
        assert word not in S.specification() and word not in S.instruction_block()


def test_y_mirrors_r_graph_and_traces():
    assert [(c.id, c.prerequisites) for c in R.components()] == [(c.id, c.prerequisites) for c in Y.components()]
    assert sorted(f.trace for f in R.families()) == sorted(f.trace for f in Y.families())
    assert sorted(f.held_out for f in R.families()) == sorted(f.held_out for f in Y.families())
    for word in ("interest", "money", "price", "bond", "debt", "invest", "rate", "%", "$"):
        assert word not in Y.specification().lower() and word not in Y.instruction_block().lower()

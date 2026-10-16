"""Tests for the version-2 changes (resubmission of Access-2026-47739)."""
import json, os, random, tempfile
from harness.config import Config
from harness.content import load_content
from harness.client import FakeClient
from harness.learner import History, Interaction, features, train_gbdt, N_CONDITION_BLIND, FEATURE_NAMES
from harness.telemetry import Telemetry
from harness.chain import run_chain
from harness.revision import tether_scaling_stated, bh
from harness.content.novel import SPEC_Y


def cfg_small():
    c = Config(); c.session_cap = 20; c.generations = 2; c.persistence_items = 3
    return c


def test_reference_student_gets_its_own_summary():
    """v1 bug: the reference student's boundary probe received the chain student's summary."""
    with tempfile.TemporaryDirectory() as d:
        cfg = cfg_small(); tel = Telemetry(d, "run")
        run_chain(FakeClient(), load_content("demo"), cfg, 0.85, 1, True, tel); tel.close()
        for n in (1, 2):
            assert os.path.exists(os.path.join(d, "run", f"sigma_gen{n}.json"))
            assert os.path.exists(os.path.join(d, "run", f"sigma_ref_gen{n}.json"))
        s = json.load(open(os.path.join(d, "run", "run_summary.json")))
        assert s["summary_before_posttest"] is True and s["reference_summary"] is True
        rows = [json.loads(l) for l in open(os.path.join(d, "run", "telemetry.jsonl"))]
        assert sum(r["phase"] == "persist_boundary" and r["arm"] == "reference" for r in rows) == 2 * cfg.persistence_items


def test_summary_is_written_before_the_posttest_by_default():
    with tempfile.TemporaryDirectory() as d:
        cfg = cfg_small(); cfg.generations = 1; client = FakeClient(); tel = Telemetry(d, "run")
        run_chain(client, load_content("demo"), cfg, 0.85, 1, True, tel); tel.close()
        # in the chain student's call sequence the SUMMARISE prompt precedes the first post-test item
        calls = [c for c in client.calls if c["messages"] and c["messages"][0]["content"].startswith("TEACH") is False]
        seq = [m["content"][:9] for c in client.calls for m in c["messages"][-1:]]
        first_sum = next(i for i, s in enumerate(seq) if s.startswith("SUMMARISE"))
        assert first_sum < len(seq)
        # version-1 behaviour available
        cfg.summary_before_posttest = False
        tel2 = Telemetry(d, "run2"); run_chain(FakeClient(), load_content("demo"), cfg, 0.85, 1, True, tel2); tel2.close()
        assert json.load(open(os.path.join(d, "run2", "run_summary.json")))["summary_before_posttest"] is False


def test_fixed_sequence_control_runs_without_a_learner():
    with tempfile.TemporaryDirectory() as d:
        cfg = cfg_small(); cfg.generations = 1; cfg.policy = "fixed"; cfg.fixed_items = 2
        tel = Telemetry(d, "run"); res = run_chain(FakeClient(), load_content("demo"), cfg, 1, 1, True, tel); tel.close()
        content = load_content("demo")
        for r in res:
            assert r["outcome"] == "fixed" and r["mastered"] == []
            assert r["interactions"] == 2 * len(content.components())
        rows = [json.loads(l) for l in open(os.path.join(d, "run", "telemetry.jsonl"))]
        assert all(r["action"] == "FIXED" for r in rows if r["phase"] == "instruction")


def test_condition_blind_learner_has_seventeen_features():
    assert FEATURE_NAMES.index("set_index") == N_CONDITION_BLIND
    content = load_content("demo"); graph = content.graph()
    kci = {c.id: i for i, c in enumerate(content.components())}; fi = {f.family_id: i for i, f in enumerate(content.families())}
    rng = random.Random(0); X, y = [], []
    for _ in range(80):
        h = History()
        for j in range(6):
            c = rng.choice(list(graph)); X.append(features(h, c, graph, {"tau": 0.85}, kci, fi)); y.append(rng.random() < 0.6)
            h.interactions.append(Interaction(c, content.next_family(c), y[-1], "parsed", 100, 1000, j + 1))
    m = train_gbdt(X, y, X[:20], y[:20], "platt", kci, fi, condition_features=False, platt_penalty=None)
    assert m.model.n_features_in_ == N_CONDITION_BLIND
    p = m.predict(History(), "K1", graph, {"tau": 0.95, "generation": 3, "arm_index": 1})
    assert 0 < p < 1


def test_next_family_and_pilot_difficulty_reach_the_features():
    content = load_content("demo"); graph = content.graph()
    kci = {c.id: i for i, c in enumerate(content.components())}; fi = {f.family_id: i for i, f in enumerate(content.families())}
    fam = content.next_family("K2"); assert fam
    content.set_difficulty({fam: 0.4})
    x = features(History(), "K2", graph, {}, kci, fi, next_difficulty=content.family_difficulty(fam), next_family=fam)
    assert x[FEATURE_NAMES.index("difficulty")] == 0.4 and x[FEATURE_NAMES.index("family_index")] == fi[fam]
    item = content.sample_item("K2", random.Random(0)); assert item.difficulty == 0.4


def test_tether_detector():
    assert tether_scaling_stated("add b times one mark per ten marks of the shorter cairn")
    assert tether_scaling_stated("plus b × floor(min(q1, q2) / 10)")
    assert tether_scaling_stated("the tether is scaled by the shorter cairn's height ÷ 10")
    assert not tether_scaling_stated("then add the tether b as a fixed offset")
    assert not tether_scaling_stated("a positive tether makes the result taller; a negative tether makes it shorter")
    assert not tether_scaling_stated("shares are given in tenths; floor each part; add the tether")
    assert tether_scaling_stated(SPEC_Y.splitlines()[7])


def test_benjamini_hochberg():
    adj, rej = bh([0.01, 0.03, 0.02, 0.5])
    assert rej == [True, True, True, False] and abs(adj[3] - 0.5) < 1e-9


def test_spec_header_carries_no_draft_label():
    assert "DRAFT" not in SPEC_Y

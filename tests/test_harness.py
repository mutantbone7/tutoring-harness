import os, random, tempfile
import pytest
from harness.config import Config
from harness.content import load_content
from harness.client import FakeClient
from harness.parsing import parse_answer
from harness.router import Router, Instance, IsolationViolation, spec_fingerprints
from harness.learner import BKTLearner, History, Interaction, features, ece
from harness.policy import Policy
from harness.telemetry import Telemetry
from harness.session import run_session
from harness.chain import run_chain
from harness.probes import run_probes, run_pilot, wilson


def cfg_small():
    c = Config(); c.session_cap = 20; c.generations = 2; c.persistence_items = 3; c.n_probes = 24
    return c


def test_parser_leniency_rule():
    assert parse_answer("reasoning...\nANSWER: B") == ("B", "parsed")
    assert parse_answer("I choose C") == ("C", "fallback")
    assert parse_answer("Either A or B") == (None, "failure")
    assert parse_answer("") == (None, "failure")
    assert parse_answer("answer: d.") == ("D", "parsed")


def test_held_out_family_is_structurally_distinct():
    content = load_content("demo")
    res = content.check_held_out_traces(random.Random(0))
    assert res == {"repeat-offset": True}


def test_item_key_never_in_prompt():
    content = load_content("demo")
    rng = random.Random(1)
    item = content.sample_item("K2", rng)
    prompt, key, tagmap = item.render(rng)
    assert "KEY" not in prompt and key in "ABCD" and tagmap[key] is None
    assert sum(1 for v in tagmap.values() if v is None) == 1


def test_isolation_assertion_aborts_on_spec_leak():
    content = load_content("demo")
    router = Router(FakeClient(), spec_fingerprints(content.specification()))
    s = Instance(role="student", label="S1", seed=1, model="fake")
    router.send(s, "hello", 10)                       # clean turn passes
    leak = content.specification().splitlines()[1]    # a spec line >= 40 chars
    with pytest.raises(IsolationViolation):
        router.send(s, "note: " + leak, 10)
    assert router.violations == 1


def test_teacher_context_is_not_checked_but_student_is():
    content = load_content("demo")
    router = Router(FakeClient(), spec_fingerprints(content.specification()))
    t = Instance(role="teacher", label="T0", seed=0, model="fake", system=content.specification())
    router.send(t, "TEACH: something", 10)            # teachers may hold the spec


def test_policy_follows_algorithm_1():
    content = load_content("demo")
    class Fixed(BKTLearner):
        def __init__(self, p): self.p = p
        def predict(self, h, c, graph, condition, **kw): return self.p
    h = History()
    pol = Policy(content, 0.85, Fixed(0.9), attempt_cap=3, review_interval=2, condition={})
    a = pol.decide(h, "K1")
    assert a.kind == "ADVANCE" and a.component in ("K2", "K3")    # K1 mastered, prerequisites satisfied
    pol2 = Policy(content, 0.85, Fixed(0.1), attempt_cap=3, review_interval=2, condition={})
    a = pol2.decide(h, "K2")
    assert a.kind == "REMEDIATE" and a.component == "K1"           # line 11 reuses tau
    pol3 = Policy(content, 0.85, Fixed(0.1), attempt_cap=1, review_interval=2, condition={})
    h.interactions.append(Interaction("K1", "scale", False, "parsed", 5, 10, 1))
    a = pol3.decide(h, "K1")
    assert a.kind == "TERMINATE" and a.reason == "non-mastery"     # line 12 at A_max


def test_bkt_updates_and_features_shape():
    content = load_content("demo")
    f = BKTLearner(); h = History()
    p0 = f.predict(h, "K1", content.graph(), {})
    i = Interaction("K1", "scale", True, "parsed", 5, 10, 1); h.interactions.append(i); f.observe(h, i)
    assert f.predict(h, "K1", content.graph(), {}) > p0
    x = features(h, "K2", content.graph(), {"tau": 0.85}, {"K1": 0, "K2": 1, "K3": 2}, {"scale": 0})
    assert len(x) == 22
    assert abs(ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 1]) - 0.25) < 1e-9 or ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 1]) >= 0


def test_session_runs_and_logs_table5_fields():
    content = load_content("demo"); cfg = cfg_small()
    with tempfile.TemporaryDirectory() as d:
        tel = Telemetry(d, "t")
        router = Router(FakeClient(gain=0.4), spec_fingerprints(content.specification()))
        T = Instance(role="teacher", label="T0", seed=0, model="fake", system="tutor\n" + content.specification())
        S = Instance(role="student", label="S1", seed=1, model="fake")
        base = {"seed": 1, "generation": 1, "arm": "chain", "teacher_model": "fake", "content_set": "demo",
                "tau": 0.85, "calibration_flag": True, "_condition": {"tau": 0.85}}
        res = run_session(router, T, S, content, 0.85, BKTLearner(), cfg, tel, base)
        tel.close()
        assert res["outcome"] in ("mastered", "non-mastery", "cap")
        assert 1 <= res["interactions"] <= cfg.session_cap
        rows = [r for r in tel.rows if r["phase"] == "instruction"]
        assert len(rows) == res["interactions"]
        for k in ("run_id", "seed", "generation", "arm", "component", "item_id", "correct", "parse_status",
                  "latency_ms", "tokens_in", "tokens_out", "context_position", "timestamp"):
            assert k in rows[0]
        assert any(r["phase"] == "post" for r in tel.rows) and any(r["phase"] == "transfer" for r in tel.rows)
        assert router.violations == 0


def test_chain_seed_pattern_and_arms():
    content = load_content("demo"); cfg = cfg_small()
    with tempfile.TemporaryDirectory() as d:
        tel = Telemetry(d, "c")
        res = run_chain(FakeClient(gain=0.4), content, cfg, 0.85, seed=10, calibrated=True, telemetry=tel)
        tel.close()
        assert [(r["arm"], r["generation"]) for r in res] == [("chain", 1), ("reference", 1), ("chain", 2), ("reference", 2)]
        import json
        tr = json.load(open(os.path.join(tel.dir, "transcript_gen1.json")))
        assert tr["S1"]["seed"] == 11 and tr["R1"]["seed"] == 111 and tr["Sprime1"]["seed"] == 211 and tr["T1"]["seed"] == 311
        assert content.specification().splitlines()[1] not in tr["T1"]["system"]     # line 16: sigma_n replaces Sigma
        assert content.specification().splitlines()[1] in tr["T0"]["system"]          # line 1: T0 holds Sigma
        assert os.path.exists(os.path.join(tel.dir, "sigma_gen1.json"))
        phases = {r["phase"] for r in tel.rows}
        assert {"persist_distance", "persist_boundary", "persist_interference_K2"} <= phases


def test_probes_and_pilot_statistics():
    content = load_content("demo"); cfg = cfg_small()
    with tempfile.TemporaryDirectory() as d:
        tel = Telemetry(d, "p")
        res = run_probes(FakeClient(base_p=0.25, gain=0.0), content, cfg, tel, seed=0, n=24)
        assert res["n"] == 24 and 0 <= res["accuracy"] <= 1 and "p_greater_than_chance" in res
        lo, hi = wilson(6, 24); assert lo < 0.25 < hi
        res2 = run_pilot(FakeClient(base_p=0.6, gain=0.0), content, cfg, tel, seed=0, n=24)
        assert set(res2["difficulty_per_family"]) <= {"scale", "repeat", "offset"}
        tel.close()


def test_probes_resume_continues_same_items_without_repeating_calls():
    content = load_content("demo"); cfg = cfg_small()
    with tempfile.TemporaryDirectory() as d:
        tel = Telemetry(d, "p")
        fake = FakeClient(base_p=0.3, gain=0.0)
        run_probes(fake, content, cfg, tel, seed=5, n=10)                 # a "full" run of 10
        full_ids = [r["item_id"] for r in tel.rows]
        tel.close()
        tel2 = Telemetry(d, "q")
        fake2 = FakeClient(base_p=0.3, gain=0.0)
        run_probes(fake2, content, cfg, tel2, seed=5, n=4)                # interrupted after 4
        tel2.close()
        tel3 = Telemetry(d, "q")                                          # same run dir, append
        fake3 = FakeClient(base_p=0.3, gain=0.0)
        res = run_probes(fake3, content, cfg, tel3, seed=5, n=10, resume=True)
        tel3.close()
        import json
        rows = [json.loads(l) for l in open(os.path.join(tel3.dir, "telemetry.jsonl"))]
        assert [r["item_id"] for r in rows] == full_ids                  # identical item sequence
        assert len(fake3.calls) == 6 and res["n"] == 10                   # only the missing 6 were called

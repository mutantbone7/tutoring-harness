"""The wording command can be interrupted and resumed without re-running
finished teachers (v2, added 16 Oct 2026 after a laptop sleep cut a run)."""
import json
import os

from harness.client import Completion
from harness.config import Config
from harness.content import load_content
from harness.extras import wording, _wording_status


class FakeClient:
    def __init__(self, stop_after=None):
        self.calls = 0
        self.stop_after = stop_after
        self.total_in = self.total_out = self.total_retries = 0
        self.total_cache_read = self.total_cache_write = self.total_refusals = 0

    def complete(self, system, messages, max_tokens, meta=None):
        self.calls += 1
        if self.stop_after is not None and self.calls > self.stop_after:
            raise KeyboardInterrupt
        text = ("Take the shorter cairn's height and scale it by the tether: the merged height is the "
                "shorter cairn's height times the tether over ten.") if self.calls % 2 else "Add the shares and merge."
        return Completion(text=text, model="fake", tokens_in=10, tokens_out=5, latency_ms=1, retries=0)


def test_wording_resume_skips_finished_teachers(tmp_path):
    cfg = Config(model="fake", output_dir=str(tmp_path))
    content = load_content("Y")
    # first run: 3 teachers of one variant, cut off during the second teacher (after 6 + 2 calls)
    try:
        wording(FakeClient(stop_after=8), content, cfg, str(tmp_path), n=3, variants=["spec"])
    except KeyboardInterrupt:
        pass
    run_dir = [d for d in os.listdir(tmp_path) if d.startswith("wording-Y-")][0]
    path = os.path.join(tmp_path, run_dir, "replies.jsonl")
    lines = [json.loads(l) for l in open(path)]
    assert len(lines) == 8
    st = _wording_status(path, content.topo_order())
    assert st[("spec", 0)]["complete"] and not st[("spec", 1)]["complete"]
    # resume: teacher 0 is skipped, teacher 1 is run again from its first prompt, teacher 2 is run
    client = FakeClient()
    counts = wording(client, content, cfg, str(tmp_path), n=3, variants=["spec"], resume="latest")
    assert client.calls == 12
    lines = [json.loads(l) for l in open(path)]
    assert len(lines) == 20 and sum(1 for r in lines if r.get("resumed")) == 12
    st = _wording_status(path, content.topo_order())
    assert all(st[("spec", i)]["complete"] for i in range(3))
    assert counts["spec"]["of"] == 3
    summ = json.load(open(os.path.join(tmp_path, run_dir, "wording_summary.json")))
    assert summ["resumed"] is True and summ["counts"]["spec"]["of"] == 3
    # no new run directory was created
    assert len([d for d in os.listdir(tmp_path) if d.startswith("wording-Y-")]) == 1

"""Telemetry writer: one JSON line per interaction with the fields of
Table 5, plus the policy action and the learner-model probability that drove
it. Transcripts are written per instance at the end of a session."""
from __future__ import annotations
import json
import os
import time
from typing import Dict, List

FIELDS = ["run_id", "seed", "generation", "arm", "teacher_model", "student_model", "content_set",
          "tau", "calibration_flag", "component", "item_id", "family", "held_out", "correct",
          "distractor", "misconception", "parse_status", "latency_ms", "tokens_in", "tokens_out",
          "context_position", "context_length", "timestamp", "phase", "action", "p_next",
          "attempt_index", "answer_label"]


class Telemetry:
    def __init__(self, out_dir: str, run_id: str):
        self.dir = os.path.join(out_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self._f = open(os.path.join(self.dir, "telemetry.jsonl"), "a")
        self.rows: List[Dict] = []

    def log(self, **row) -> None:
        row.setdefault("run_id", self.run_id)
        row.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        for k in FIELDS:
            row.setdefault(k, None)
        self.rows.append(row)
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def write_transcript(self, name: str, instances) -> None:
        with open(os.path.join(self.dir, f"transcript_{name}.json"), "w") as f:
            json.dump({i.label: {"role": i.role, "seed": i.seed, "model": i.model, "served_model": i.served_model,
                                 "system": i.system, "messages": i.messages} for i in instances}, f, indent=1)

    def write_json(self, name: str, obj) -> None:
        with open(os.path.join(self.dir, name), "w") as f:
            json.dump(obj, f, indent=1)

    def close(self) -> None:
        self._f.close()

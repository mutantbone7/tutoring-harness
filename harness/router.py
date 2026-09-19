"""Instances and the message router with the isolation invariant
(Section IV-B). The student context never contains the specification, an
answer key, or the teacher's private reasoning. The check is an assertion on
the byte content of the outgoing context, run on every turn; a violation
raises IsolationViolation, which aborts the run and is counted."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class IsolationViolation(AssertionError):
    pass


def spec_fingerprints(spec: str, min_len: int = 40) -> Set[bytes]:
    """Byte fingerprints of the specification: every line of at least
    min_len characters. Verbatim reproduction of any one of them in a student
    context is a violation; paraphrase is not."""
    return {line.strip().encode("utf-8") for line in spec.splitlines() if len(line.strip()) >= min_len}


@dataclass
class Instance:
    role: str                       # teacher | student | reference | probe
    label: str                      # e.g. T0, S1, R1, Sprime1
    seed: int
    model: str
    system: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    rng: random.Random = field(init=False)
    served_model: Optional[str] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def context_bytes(self) -> bytes:
        parts = [self.system] + [m["content"] for m in self.messages]
        return "\n".join(parts).encode("utf-8")

    def context_length(self) -> int:
        return len(self.context_bytes())


class Router:
    def __init__(self, client, forbidden: Set[bytes]):
        self.client = client
        self.forbidden: Set[bytes] = set(forbidden)
        self.violations = 0
        self.turn_log: List[Dict] = []

    def forbid(self, *chunks: str) -> None:
        for c in chunks:
            if c:
                self.forbidden.add(c.encode("utf-8"))

    def check_isolation(self, inst: Instance) -> None:
        if inst.role not in ("student", "probe"):
            return
        ctx = inst.context_bytes()
        for chunk in self.forbidden:
            if chunk in ctx:
                self.violations += 1
                raise IsolationViolation(
                    f"forbidden content in context of {inst.label}: {chunk[:60]!r}")

    def send(self, inst: Instance, content: str, max_tokens: int, meta: Optional[Dict] = None):
        """Append a user turn, enforce isolation, call the model, append its reply."""
        inst.messages.append({"role": "user", "content": content})
        self.check_isolation(inst)
        comp = self.client.complete(inst.system, inst.messages, max_tokens, meta=meta)
        inst.messages.append({"role": "assistant", "content": comp.text})
        inst.served_model = comp.model
        self.turn_log.append({"to": inst.label, "role": inst.role, "tokens_in": comp.tokens_in,
                              "tokens_out": comp.tokens_out, "cache_read": getattr(comp, "cache_read", 0),
                              "cache_write": getattr(comp, "cache_write", 0), "latency_ms": comp.latency_ms,
                              "retries": comp.retries, "model": comp.model})
        return comp

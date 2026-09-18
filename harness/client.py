"""Model client: one place that talks to the API. Records the served model
string, token counts and latency per turn; retries rate limits and transient
errors with exponential backoff; never sets a temperature or a seed, because
the interface for current models accepts neither (Section IV-B)."""
from __future__ import annotations
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Completion:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    retries: int


class ModelClient:
    def __init__(self, model: str, max_retries: int = 8):
        from anthropic import Anthropic  # imported here so tests can run without the SDK
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
        self._client = Anthropic(max_retries=0)  # the harness owns retry logic and counts it
        self.model = model
        self.max_retries = max_retries
        self.total_in = 0
        self.total_out = 0
        self.total_retries = 0

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int, meta: Optional[Dict] = None) -> Completion:
        """`meta` is harness-side bookkeeping (never sent to the API)."""
        import anthropic
        attempt = 0
        while True:
            t0 = time.time()
            try:
                resp = self._client.messages.create(
                    model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                )
                latency = int((time.time() - t0) * 1000)
                text = "".join(getattr(b, "text", "") for b in resp.content)
                self.total_in += resp.usage.input_tokens
                self.total_out += resp.usage.output_tokens
                return Completion(text, resp.model, resp.usage.input_tokens, resp.usage.output_tokens, latency, attempt)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                attempt += 1
                self.total_retries += 1
                if attempt > self.max_retries:
                    raise RuntimeError(f"giving up after {attempt} attempts: {e}") from e
                time.sleep(min(60.0, (2 ** attempt) + random.random()))


class FakeClient:
    """Deterministic stand-in for tests and dry runs. Answers items with a
    probability of being correct that rises with the number of teaching turns
    it has seen for the item's component; never calls the network."""

    def __init__(self, model: str = "fake-model", base_p: float = 0.3, gain: float = 0.25, seed: int = 0):
        self.model = model
        self.base_p, self.gain = base_p, gain
        self.rng = random.Random(seed)
        self.total_in = self.total_out = self.total_retries = 0
        self.calls: List[Dict] = []

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int, meta: Optional[Dict] = None) -> Completion:
        self.calls.append({"system": system, "messages": list(messages), "meta": meta})
        last = messages[-1]["content"]
        text_so_far = "\n".join(m["content"] for m in messages)
        if "TEACH:" in last:
            text = "Here is an explanation of the rule in my own words: apply the factor, then adjust."
        elif "SUMMARISE:" in last:
            text = "Notes: quantities are scaled by factors; repeated scaling compounds; offsets come after."
        elif "ANSWER:" in last:
            # the harness passes the key in meta (side channel, never in the prompt)
            key = (meta or {}).get("key")
            taught = text_so_far.count("Here is an explanation")
            p = min(0.95, self.base_p + self.gain * taught)
            if key is not None and self.rng.random() < p:
                text = f"I think the answer is clear.\nANSWER: {key}"
            else:
                wrong = [l for l in "ABCD" if l != key]
                text = f"Not sure.\nANSWER: {self.rng.choice(wrong)}"
        else:
            text = "OK"
        n_in = sum(len(m["content"]) for m in messages) // 4
        n_out = len(text) // 4
        self.total_in += n_in
        self.total_out += n_out
        return Completion(text, self.model, n_in, n_out, 5, 0)

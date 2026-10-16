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
    cache_read: int = 0
    cache_write: int = 0
    stop_reason: str = ""
    block_types: str = ""


class ModelClient:
    """Thread-safe: one instance may be shared by parallel sessions."""

    def __init__(self, model: str, max_retries: int = 8, prompt_caching: bool = True):
        from anthropic import Anthropic  # imported here so tests can run without the SDK
        import threading
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment")
        self._client = Anthropic(max_retries=0)  # the harness owns retry logic and counts it
        self.model = model
        self.max_retries = max_retries
        self.prompt_caching = prompt_caching
        self._lock = threading.Lock()
        self.total_in = self.total_out = self.total_retries = 0
        self.total_cache_read = self.total_cache_write = 0
        self.total_refusals = 0
        self.refusal_retries = 3

    def _payload(self, system: str, messages: List[Dict[str, str]]):
        """With caching on, the system prompt and the latest user turn carry
        cache breakpoints, so the growing conversation prefix is read from
        cache on every turn instead of being re-billed in full."""
        if not self.prompt_caching:
            return system, messages
        sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] if system else system
        msgs = [dict(m) for m in messages]
        last = msgs[-1]
        msgs[-1] = {"role": last["role"], "content": [{"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}]}
        return sys_blocks, msgs

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int, meta: Optional[Dict] = None) -> Completion:
        """`meta` is harness-side bookkeeping (never sent to the API)."""
        import anthropic
        attempt = 0
        refusals = 0
        while True:
            t0 = time.time()
            try:
                sys_payload, msg_payload = self._payload(system, messages)
                kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": msg_payload}
                if sys_payload:
                    kwargs["system"] = sys_payload
                resp = self._client.messages.create(**kwargs)
                latency = int((time.time() - t0) * 1000)
                text = "".join(getattr(b, "text", "") for b in resp.content)
                if getattr(resp, "stop_reason", None) == "refusal":
                    # An API-level refusal is not an answer. Retry up to
                    # refusal_retries times; every refusal is counted so the
                    # rate can be reported. If all attempts refuse, the empty
                    # reply is returned and parses as a failure.
                    with self._lock:
                        self.total_refusals += 1
                    if refusals < self.refusal_retries:
                        refusals += 1
                        continue
                u = resp.usage
                cr = getattr(u, "cache_read_input_tokens", 0) or 0
                cw = getattr(u, "cache_creation_input_tokens", 0) or 0
                with self._lock:
                    self.total_in += u.input_tokens
                    self.total_out += u.output_tokens
                    self.total_cache_read += cr
                    self.total_cache_write += cw
                comp = Completion(text, resp.model, u.input_tokens, u.output_tokens, latency, attempt, cr, cw,
                                  str(getattr(resp, "stop_reason", "")), ",".join(getattr(b, "type", "?") for b in resp.content))
                comp.refusals = refusals
                return comp
            except anthropic.BadRequestError as e:
                if self.prompt_caching and "cache_control" in str(e):
                    self.prompt_caching = False       # provider rejected caching; continue without it
                    continue
                raise
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                attempt += 1
                with self._lock:
                    self.total_retries += 1
                if attempt > self.max_retries:
                    raise RuntimeError(f"giving up after {attempt} attempts: {e}") from e
                time.sleep(min(60.0, (2 ** attempt) + random.random()))


class OpenAICompatibleClient:
    """Second-provider client (v2, reviewer 3, item 2): any endpoint that
    speaks the OpenAI chat-completions protocol (OpenAI, Google's
    OpenAI-compatible endpoint, Groq, Together, a local server). Uses httpx,
    which the anthropic SDK already depends on, so nothing new is installed.
    The key is read from the environment variable named by `api_key_env`
    (default OPENAI_API_KEY). No temperature or seed is sent, matching the
    Anthropic client. Prompt caching is provider-side and not requested."""

    def __init__(self, model: str, base_url: str, max_retries: int = 8, api_key_env: str = "OPENAI_API_KEY"):
        import threading
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set in the environment")
        self._key, self.base_url = key, base_url.rstrip("/")
        self.model, self.max_retries = model, max_retries
        self.prompt_caching = False
        self._lock = threading.Lock()
        self.total_in = self.total_out = self.total_retries = 0
        self.total_cache_read = self.total_cache_write = 0
        self.total_refusals = 0

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int, meta: Optional[Dict] = None) -> Completion:
        import httpx
        msgs = ([{"role": "system", "content": system}] if system else []) + [dict(m) for m in messages]
        body = {"model": self.model, "messages": msgs, "max_tokens": max_tokens}
        attempt = 0
        while True:
            t0 = time.time()
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=body, timeout=180.0,
                               headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code == 400 and "max_tokens" in r.text and "max_completion_tokens" not in body:
                    body["max_completion_tokens"] = body.pop("max_tokens"); continue   # newer OpenAI models
                r.raise_for_status()
                j = r.json(); latency = int((time.time() - t0) * 1000)
                ch = j["choices"][0]; text = (ch.get("message") or {}).get("content") or ""
                u = j.get("usage") or {}
                n_in, n_out = int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0)
                with self._lock:
                    self.total_in += n_in; self.total_out += n_out
                comp = Completion(text, j.get("model", self.model), n_in, n_out, latency, attempt, 0, 0,
                                  str(ch.get("finish_reason", "")), "text")
                comp.refusals = 0
                return comp
            except (RuntimeError, httpx.HTTPError) as e:
                attempt += 1
                with self._lock:
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

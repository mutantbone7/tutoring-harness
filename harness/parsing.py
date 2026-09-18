"""Response parsing and scoring (Section IV-C). Strict parser first; the
fallback accepts a response only when exactly one option label appears in the
final non-empty line. Anything else is a parse failure, scored incorrect."""
from __future__ import annotations
import re
from typing import Optional, Tuple
from .content.base import LABELS

_STRICT = re.compile(r"^\s*ANSWER\s*:\s*([A-D])\s*\.?\s*$", re.IGNORECASE)
_LABEL = re.compile(r"\b([A-D])\b")


def parse_answer(text: str) -> Tuple[Optional[str], str]:
    """Return (label or None, parse_status) with status in {parsed, fallback, failure}."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, "failure"
    last = lines[-1]
    m = _STRICT.match(last)
    if m:
        return m.group(1).upper(), "parsed"
    found = {x.upper() for x in _LABEL.findall(last) if x.upper() in LABELS}
    if len(found) == 1:
        return found.pop(), "fallback"
    return None, "failure"

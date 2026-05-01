"""
Helpers for parsing structured JSON out of LLM/VLM responses.

Models are prompted to return JSON only, but real responses often include
markdown fences, lead-in text, or trailing commentary. These helpers avoid
greedy regex extraction and return the first valid JSON object found.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    """Return the first JSON object found in a model response."""
    if not isinstance(raw, str):
        raise TypeError(f"Expected a string response, got {type(raw).__name__}")

    text = raw.strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(text))

    decoder = json.JSONDecoder()
    errors: list[str] = []

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            errors.append("top-level JSON value was not an object")
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

        for idx, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    detail = "; ".join(errors[-3:]) if errors else "no JSON object delimiter found"
    raise ValueError(f"No valid JSON object found in model response ({detail})")

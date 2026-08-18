"""
summariser.py — condenses older conversation turns into a single running
summary. Defaults to the same extractive approach as the compressor;
optionally calls an LLM for an abstractive summary when a callable
``llm_call`` is supplied (Section 4.3 — "there is an optional mode that
calls the LLM itself to produce an abstractive summary").
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..compression.compressor import compress_prompt


def _flatten(history: List[Dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history)


def extractive_summary(history: List[Dict[str, str]], target_ratio: float = 0.35) -> str:
    """Condense older turns via the same extractive pipeline used for
    single prompts (segment -> score -> select -> dedupe)."""
    if not history:
        return ""
    flat = _flatten(history)
    return compress_prompt(flat, target_ratio=target_ratio).compressed_text


def abstractive_summary(
    history: List[Dict[str, str]], llm_call: Optional[Callable[[str], str]] = None
) -> str:
    """Produce an abstractive summary of older turns by delegating to a
    caller-supplied ``llm_call(prompt: str) -> str`` function. Falls back
    to the extractive summary if no LLM call is supplied, since spending
    an extra API call on every summarisation defeats the purpose of a
    token-saving tool unless the caller has explicitly opted in."""
    if llm_call is None:
        return extractive_summary(history)
    flat = _flatten(history)
    instruction = (
        "Summarise the following conversation history concisely, "
        "preserving any facts, decisions, or instructions the user gave:\n\n"
        f"{flat}"
    )
    return llm_call(instruction)

"""
manager.py — rolling-window orchestration for the Context Window
Manager, mirroring the pseudocode in report Section 3.4.2:

    if token_count(history) < max_tokens: return history
    recent = history[-keep_recent_turns:]
    older = history[:-keep_recent_turns]
    summary = extractive_summary(older)  # or LLM call, if abstractive mode is on
    return [system_message(summary)] + recent
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .budget import history_tokens
from .summariser import abstractive_summary, extractive_summary


def system_message(content: str) -> Dict[str, str]:
    return {"role": "system", "content": f"[Condensed earlier conversation] {content}"}


def manage_context(
    history: List[Dict[str, str]],
    max_tokens: int = 4000,
    keep_recent_turns: int = 6,
    abstractive: bool = False,
    llm_call: Optional[Callable[[str], str]] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, str]]:
    """Return a possibly-condensed message history that stays within
    ``max_tokens``. Nothing is changed if the history is already under
    budget. Otherwise, everything except the most recent
    ``keep_recent_turns`` messages is condensed into a single running
    summary re-injected as a system message ahead of the retained turns."""
    if history_tokens(history, model=model) < max_tokens:
        return history  # nothing to do yet

    if len(history) <= keep_recent_turns:
        return history  # not enough history to condense

    recent = history[-keep_recent_turns:]
    older = history[:-keep_recent_turns]

    if abstractive:
        summary = abstractive_summary(older, llm_call=llm_call)
    else:
        summary = extractive_summary(older)

    return [system_message(summary)] + recent

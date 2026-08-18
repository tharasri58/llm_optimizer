"""budget.py — token-budget tracking for the Context Window Manager."""
from __future__ import annotations

from typing import Dict, List

from ..reporting.token_counter import count_tokens


def message_tokens(message: Dict[str, str], model: str = "gpt-4o-mini") -> int:
    """Token count for a single {"role": ..., "content": ...} message,
    plus a small fixed overhead per message (role/name framing tokens),
    consistent with how chat APIs bill message structure."""
    overhead = 4
    return overhead + count_tokens(message.get("content", ""), model=model)


def history_tokens(history: List[Dict[str, str]], model: str = "gpt-4o-mini") -> int:
    """Total token count across a full multi-turn message history."""
    return sum(message_tokens(m, model=model) for m in history)


def is_over_budget(
    history: List[Dict[str, str]], max_tokens: int, model: str = "gpt-4o-mini"
) -> bool:
    return history_tokens(history, model=model) >= max_tokens

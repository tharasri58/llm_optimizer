"""
anthropic_wrapper.py — wraps anthropic.messages-style calls the same way
openai_wrapper.py wraps OpenAI's client (Section 4.4). Covers standard
chat-completion style calls; tool-calling is not yet covered for the
Anthropic wrapper (Table 5.3 / Section 6.2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..compression.compressor import compress_prompt
from ..context_manager.manager import manage_context
from ..reporting.token_counter import count_tokens


def _is_tool_related(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") in (
                    "tool_use",
                    "tool_result",
                ):
                    return True
    return False


class OptimisedAnthropic:
    """Drop-in wrapper around an ``anthropic.Anthropic`` client (or a
    compatible test double)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        compression_target: float = 0.5,
        max_context_tokens: int = 4000,
        client: Any = None,
    ):
        self.compression_target = compression_target
        self.max_context_tokens = max_context_tokens
        self._last_savings_report: Dict[str, Any] = {}

        if client is not None:
            self._client = client
        else:
            import anthropic  # imported lazily so the package doesn't hard-depend on it

            self._client = anthropic.Anthropic(api_key=api_key)

        self.messages = _MessagesNamespace(self)

    def last_savings_report(self) -> Dict[str, Any]:
        return self._last_savings_report


class _MessagesNamespace:
    def __init__(self, wrapper: "OptimisedAnthropic"):
        self._wrapper = wrapper

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        **kwargs,
    ):
        wrapper = self._wrapper
        baseline_tokens = sum(
            count_tokens(str(m.get("content", "")), model=model) for m in messages
        )
        if system:
            baseline_tokens += count_tokens(system, model=model)

        if _is_tool_related(messages):
            optimised_messages, optimised_system = messages, system
        else:
            if len(messages) > 1:
                optimised_messages = manage_context(
                    messages, max_tokens=wrapper.max_context_tokens, model=model
                )
            else:
                m = messages[0]
                result = compress_prompt(
                    str(m.get("content", "")),
                    target_ratio=wrapper.compression_target,
                    model=model,
                )
                optimised_messages = [{**m, "content": result.compressed_text}]

            optimised_system = None
            if system:
                sys_result = compress_prompt(
                    system, target_ratio=wrapper.compression_target, model=model
                )
                optimised_system = sys_result.compressed_text

        optimised_tokens = sum(
            count_tokens(str(m.get("content", "")), model=model) for m in optimised_messages
        )
        if optimised_system:
            optimised_tokens += count_tokens(optimised_system, model=model)

        reduction = (
            round(100 * (1 - optimised_tokens / baseline_tokens), 2)
            if baseline_tokens
            else 0.0
        )
        wrapper._last_savings_report = {
            "baseline_tokens": baseline_tokens,
            "optimised_tokens": optimised_tokens,
            "reduction": f"{reduction}%",
        }

        call_kwargs = dict(model=model, messages=optimised_messages, max_tokens=max_tokens, **kwargs)
        if optimised_system:
            call_kwargs["system"] = optimised_system
        return wrapper._client.messages.create(**call_kwargs)

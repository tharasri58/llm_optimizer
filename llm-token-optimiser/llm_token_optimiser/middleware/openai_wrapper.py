"""
openai_wrapper.py — wraps openai.ChatCompletion-style calls so a calling
application can swap in OptimisedOpenAI in place of the real client with
no other code changes (Section 4.4).

Function/tool-calling payloads are detected and passed through
unmodified rather than compressed — an earlier version applied
sentence-level compression to structured tool fields and occasionally
corrupted them (Section 4.4, noted as a partially-resolved limitation:
the OpenAI integration suite has one known-failing tool-calling case).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..compression.compressor import compress_prompt
from ..context_manager.manager import manage_context
from ..reporting.token_counter import count_tokens


def _is_tool_related(messages: List[Dict[str, Any]]) -> bool:
    """True if the message list contains structured tool/function-call
    fields that should not be run through sentence-level compression."""
    for m in messages:
        if "tool_calls" in m or "tool_call_id" in m or m.get("role") == "tool":
            return True
        if "function_call" in m:
            return True
    return False


class OptimisedOpenAI:
    """Drop-in wrapper around an ``openai.OpenAI`` client (or a
    compatible test double) that compresses outgoing messages before
    forwarding the call, and leaves the response untouched."""

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
            import openai  # imported lazily so the package doesn't hard-depend on it

            self._client = openai.OpenAI(api_key=api_key)

        self.chat = _ChatNamespace(self)


class _ChatNamespace:
    def __init__(self, wrapper: "OptimisedOpenAI"):
        self._wrapper = wrapper
        self.completions = _CompletionsNamespace(wrapper)


class _CompletionsNamespace:
    def __init__(self, wrapper: "OptimisedOpenAI"):
        self._wrapper = wrapper

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        wrapper = self._wrapper
        baseline_tokens = sum(
            count_tokens(m.get("content", "") or "", model=model) for m in messages
        )

        if _is_tool_related(messages):
            optimised_messages = messages  # passed through unmodified, per Section 4.4
        elif len(messages) > 1:
            optimised_messages = manage_context(
                messages, max_tokens=wrapper.max_context_tokens, model=model
            )
        else:
            m = messages[0]
            result = compress_prompt(
                m.get("content", ""), target_ratio=wrapper.compression_target, model=model
            )
            optimised_messages = [{**m, "content": result.compressed_text}]

        optimised_tokens = sum(
            count_tokens(m.get("content", "") or "", model=model) for m in optimised_messages
        )
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

        return wrapper._client.chat.completions.create(
            model=model, messages=optimised_messages, **kwargs
        )


def _last_savings_report(self: OptimisedOpenAI) -> Dict[str, Any]:
    return self._last_savings_report


OptimisedOpenAI.last_savings_report = _last_savings_report

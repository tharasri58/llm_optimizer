"""Adaptive LLM middleware for OpenAI, Anthropic, or custom local-model clients."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..compression.compressor import compress_prompt
from ..context_manager.manager import manage_context
from ..reporting.token_counter import count_tokens


def _is_tool_related_openai(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if "tool_calls" in m or "tool_call_id" in m or m.get("role") == "tool":
            return True
        if "function_call" in m:
            return True
    return False


def _is_tool_related_anthropic(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


class OptimisedLLM:
    """Adaptive wrapper for supported LLM clients.

    This wrapper supports OpenAI-style clients with
    ``client.chat.completions.create(...)`` and Anthropic-style clients
    with ``client.messages.create(...)``.

    It also adapts to a passed client instance by detecting the supported
    interface automatically.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        compression_target: float = 0.5,
        max_context_tokens: int = 4000,
        client: Any = None,
    ):
        self.compression_target = compression_target
        self.max_context_tokens = max_context_tokens
        self._last_savings_report: Dict[str, Any] = {}
        self._provider = provider.lower() if provider else None

        if client is not None:
            self._client = client
            self._provider = self._provider or self._detect_provider(client)
        elif self._provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        elif self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(
                "provider must be 'openai' or 'anthropic' when no client is provided"
            )

        self.chat = _ChatNamespace(self)
        self.messages = _MessagesNamespace(self)

    def _detect_provider(self, client: Any) -> str:
        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            return "openai"
        if hasattr(client, "messages"):
            return "anthropic"
        raise ValueError(
            "Unable to detect client provider. Pass provider='openai' or provider='anthropic', "
            "or use a client with an OpenAI-style chat.completions.create or Anthropic-style messages.create method."
        )

    def last_savings_report(self) -> Dict[str, Any]:
        return self._last_savings_report


class _ChatNamespace:
    def __init__(self, wrapper: "OptimisedLLM"):
        self._wrapper = wrapper
        self.completions = _CompletionsNamespace(wrapper)


class _CompletionsNamespace:
    def __init__(self, wrapper: "OptimisedLLM"):
        self._wrapper = wrapper

    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs):
        wrapper = self._wrapper
        baseline_tokens = sum(
            count_tokens(m.get("content", "") or "", model=model) for m in messages
        )

        if _is_tool_related_openai(messages):
            optimised_messages = messages
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

        if wrapper._provider != "openai":
            raise RuntimeError(
                "OpenAI-style chat.completions.create is only available for OpenAI-style clients"
            )

        return wrapper._client.chat.completions.create(
            model=model, messages=optimised_messages, **kwargs
        )


class _MessagesNamespace:
    def __init__(self, wrapper: "OptimisedLLM"):
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

        if _is_tool_related_anthropic(messages):
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

        if wrapper._provider != "anthropic":
            raise RuntimeError(
                "Anthropic-style messages.create is only available for Anthropic-style clients"
            )

        call_kwargs = dict(model=model, messages=optimised_messages, max_tokens=max_tokens, **kwargs)
        if optimised_system:
            call_kwargs["system"] = optimised_system
        return wrapper._client.messages.create(**call_kwargs)

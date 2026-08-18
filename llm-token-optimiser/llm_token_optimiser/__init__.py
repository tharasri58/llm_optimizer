"""LLM Token Optimiser — a middleware library that reduces token
consumption in OpenAI / Anthropic LLM API calls via prompt compression
and rolling conversation-history management."""

from .compression.compressor import compress_prompt
from .context_manager.manager import manage_context
from .middleware.anthropic_wrapper import OptimisedAnthropic
from .middleware.llm_wrapper import OptimisedLLM
from .middleware.openai_wrapper import OptimisedOpenAI
from .reporting.token_counter import count_tokens, estimate_cost

__version__ = "0.1.0"

__all__ = [
    "compress_prompt",
    "manage_context",
    "OptimisedOpenAI",
    "OptimisedAnthropic",
    "OptimisedLLM",
    "count_tokens",
    "estimate_cost",
]

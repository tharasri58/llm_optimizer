"""
token_counter.py — tiktoken-based token counting plus published-pricing
cost estimation, used by both the compression pipeline and the
Dashboard/CLI Tool.
"""
from __future__ import annotations

# Published per-1K-input-token pricing snapshots, used only to produce an
# estimated cost figure (Section 1.3, objective 5 / Section 8.3) — this is
# not wired to a real billing account.
PRICING_PER_1K_INPUT_TOKENS = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "claude-3-5-sonnet": 3.00,
    "claude-3-5-haiku": 0.80,
}

_encoding_cache: dict = {}


def _get_encoding(model: str):
    """Return a cached tiktoken encoding for ``model``, or None if
    tiktoken isn't installed, or its BPE data can't be fetched/loaded
    (e.g. no network access) — callers fall back to the heuristic
    estimator in that case rather than raising."""
    if model in _encoding_cache:
        return _encoding_cache[model]
    enc = None
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
    _encoding_cache[model] = enc
    return enc


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in ``text``. Uses tiktoken when installed; falls back
    to a ~4-chars-per-token heuristic (a commonly cited approximation for
    English text) so the rest of the pipeline still works without it."""
    if not text:
        return 0
    enc = _get_encoding(model)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / 4))


def estimate_cost(tokens: int, model: str = "gpt-4o-mini") -> float:
    """Estimate USD cost for ``tokens`` input tokens using published
    per-1K pricing. Returns 0.0 for an unrecognised model rather than
    raising, since this is an estimate, not a billing system."""
    price_per_1k = PRICING_PER_1K_INPUT_TOKENS.get(model, 0.0)
    return round((tokens / 1000) * price_per_1k, 6)

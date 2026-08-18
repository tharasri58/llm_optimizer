"""
compressor.py — orchestrates the Token Compression Module pipeline
described in Section 3.4.1:

    segment -> score -> flag_instructional -> select_top_n
             -> prune_sentences -> remove_near_duplicates -> reorder_original
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..reporting.token_counter import count_tokens
from .pruner import prune_sentences
from .redundancy import remove_near_duplicates
from .scorer import flag_instructional, tfidf_score
from .segmenter import segment


@dataclass
class CompressionResult:
    compressed_text: str
    baseline_tokens: int
    optimised_tokens: int

    @property
    def reduction_pct(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return round(
            100 * (1 - self.optimised_tokens / self.baseline_tokens), 2
        )

    def as_dict(self) -> dict:
        return {
            "baseline_tokens": self.baseline_tokens,
            "optimised_tokens": self.optimised_tokens,
            "reduction": f"{self.reduction_pct}%",
        }


def _select_top_n(
    sentences: List[str],
    scores: List[float],
    protect: List[bool],
    target_ratio: float,
) -> List[int]:
    """Return the indices to keep, ranked by score with protected
    sentences always kept, targeting roughly ``target_ratio`` of the
    original sentence count (but never fewer than the protected set)."""
    n = len(sentences)
    n_keep = max(1, round(n * target_ratio))
    protected_idx = [i for i, p in enumerate(protect) if p]
    n_keep = max(n_keep, len(protected_idx))

    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    kept = set(protected_idx)
    for i in ranked:
        if len(kept) >= n_keep:
            break
        kept.add(i)
    return sorted(kept)  # sorted = reorder_original


def compress_prompt(
    prompt: str,
    target_ratio: float = 0.5,
    redundancy_threshold: float = 0.92,
    model: str = "gpt-4o-mini",
) -> CompressionResult:
    """Compress a single prompt via extractive summarisation, protecting
    instructional/negation-bearing sentences, then prune filler words/
    phrases within each kept sentence and remove near-duplicate
    sentences. Mirrors the pseudocode in report Section 3.4.1."""
    baseline_tokens = count_tokens(prompt, model=model)

    sentences = segment(prompt)
    if not sentences:
        return CompressionResult(prompt, baseline_tokens, baseline_tokens)

    scores = tfidf_score(sentences)
    protect = flag_instructional(sentences)

    keep_idx = _select_top_n(sentences, scores, protect, target_ratio)
    kept = [sentences[i] for i in keep_idx]
    kept_protect = [protect[i] for i in keep_idx]

    # Stop-word/filler pruning happens within each kept sentence, before
    # dedup — this also helps dedup catch near-duplicates that filler
    # wording was masking (e.g. "Basically, X." vs "X.").
    pruned = prune_sentences(kept, kept_protect)

    deduped = remove_near_duplicates(pruned, kept_protect, redundancy_threshold)

    compressed_text = " ".join(deduped)
    optimised_tokens = count_tokens(compressed_text, model=model)

    # Compression should never make the prompt longer.
    if optimised_tokens > baseline_tokens:
        compressed_text, optimised_tokens = prompt, baseline_tokens

    return CompressionResult(compressed_text, baseline_tokens, optimised_tokens)

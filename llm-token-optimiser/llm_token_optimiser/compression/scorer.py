"""
scorer.py — TF-IDF sentence importance scoring, and the instructional-word
protection list described in Section 3.4.1 of the report.

The protection list exists because an earlier, unprotected stop-word pass
dropped a "not" from a test prompt and silently inverted its meaning —
statistically low-importance words are not always semantically
unimportant. Negations and imperative verbs are exempted from removal
regardless of their TF-IDF score.
"""
from __future__ import annotations

import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

NEGATIONS = {
    "not", "no", "never", "none", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "without", "except", "unless",
}

IMPERATIVE_VERBS = {
    "summarise", "summarize", "exclude", "include", "ignore", "must",
    "should", "never", "always", "avoid", "ensure", "do not", "required",
    "mandatory", "only", "strictly",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def flag_instructional(sentences: List[str]) -> List[bool]:
    """Return a boolean mask flagging which sentences contain negations or
    imperative/instructional language and should be protected from removal."""
    flags = []
    for sentence in sentences:
        lower = sentence.lower()
        words = set(_WORD_RE.findall(lower))
        is_instructional = bool(words & NEGATIONS) or any(
            phrase in lower for phrase in IMPERATIVE_VERBS
        )
        flags.append(is_instructional)
    return flags


def tfidf_score(sentences: List[str]) -> List[float]:
    """Score each sentence by its TF-IDF weight relative to the other
    sentences in the same prompt (the prompt is treated as a small corpus,
    one sentence per "document")."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # e.g. every sentence is pure stop-words / punctuation
        return [1.0] * len(sentences)

    # A sentence's score is the sum of its TF-IDF weights — longer,
    # information-dense sentences score higher than short filler ones.
    scores = matrix.sum(axis=1)
    return [float(s) for s in scores.A1]

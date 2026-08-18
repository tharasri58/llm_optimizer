"""
pruner.py — stop-word / filler-phrase pruning for the Token Compression
Module (the "stop-word pruning" step described alongside extractive
summarisation and redundancy elimination in Section 3.4.1 / Table 3.1
of the report).

Sentence-level selection (scorer.py) and sentence-level deduplication
(redundancy.py) operate on whole sentences — a sentence is either kept
intact or dropped entirely. This module is the missing third technique:
it operates *within* a kept sentence, stripping low-information filler
words and phrases so the sentence still reads naturally but costs fewer
tokens, rather than every kept sentence surviving byte-for-byte.

Sentences flagged as instructional by scorer.flag_instructional() are
left completely untouched by design. The report documents a real bug
(Section 3.4.1) where an unprotected stop-word pass dropped a "not" and
inverted a prompt's meaning; rather than trying to enumerate every word
that is safe to remove from an instructional sentence, this module
sidesteps that risk entirely by never pruning those sentences at all.
"""
from __future__ import annotations

import re
from typing import List

# Multi-word filler phrases. Sorted longest-first so a longer phrase is
# matched before a shorter one that could otherwise match a substring
# of it first (e.g. "at the end of the day" before "the day").
_FILLER_PHRASES = [
    "at the end of the day",
    "for what it's worth",
    "needless to say",
    "generally speaking",
    "as a matter of fact",
    "in my opinion",
    "in other words",
    "so to speak",
    "to be honest",
    "as such",
    "i feel like",
    "i think that",
    "i would say",
    "kind of",
    "sort of",
    "in fact",
    "of course",
]
_FILLER_PHRASES.sort(key=len, reverse=True)

# Single-word filler / intensifier words. Deliberately disjoint from
# scorer.NEGATIONS and scorer.IMPERATIVE_VERBS so this list can never
# strip a word that protection is relying on elsewhere.
_FILLER_WORDS = {
    "basically", "actually", "essentially", "really", "very", "just",
    "quite", "rather", "somewhat", "literally", "totally", "definitely",
    "certainly", "obviously", "particularly", "simply", "truly",
    "honestly", "clearly", "virtually", "practically",
}

_PHRASE_PATTERNS = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), "")
    for phrase in _FILLER_PHRASES
]
_WORD_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(w) for w in sorted(_FILLER_WORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:])")
_LEADING_COMMA_RE = re.compile(r"^\s*,\s*")


def prune_sentence(sentence: str) -> str:
    """Strip filler phrases and filler/intensifier words from ``sentence``,
    tidying up any whitespace/punctuation left behind.

    Never returns an empty string for non-empty input — if pruning would
    strip a sentence down to nothing, the original sentence is returned
    unchanged rather than silently dropping content the selector chose
    to keep.
    """
    if not sentence or not sentence.strip():
        return sentence

    pruned = sentence
    for pattern, repl in _PHRASE_PATTERNS:
        pruned = pattern.sub(repl, pruned)
    pruned = _WORD_PATTERN.sub("", pruned)

    pruned = _WHITESPACE_RE.sub(" ", pruned)
    pruned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", pruned)
    pruned = _LEADING_COMMA_RE.sub("", pruned)
    pruned = pruned.strip()

    return pruned if pruned else sentence


def prune_sentences(sentences: List[str], protect: List[bool]) -> List[str]:
    """Apply prune_sentence() to every sentence in ``sentences`` except
    those flagged True in ``protect`` (instructional / negation-bearing
    per flag_instructional), which are returned unchanged."""
    return [
        sentence if is_protected else prune_sentence(sentence)
        for sentence, is_protected in zip(sentences, protect)
    ]

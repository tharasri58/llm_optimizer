"""
segmenter.py — sentence segmentation for the Token Compression Module.

Uses NLTK's punkt tokenizer when available (matches what is described in
the report), and falls back to a lightweight regex splitter so the rest
of the pipeline never hard-fails on a machine without the NLTK corpora
downloaded.
"""
from __future__ import annotations

import re
from typing import List

_FALLBACK_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _regex_segment(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _FALLBACK_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def segment(text: str) -> List[str]:
    """Split ``text`` into a list of sentences.

    Tries NLTK's punkt tokenizer first (as described in the report,
    Section 3.4.1); falls back to a regex-based splitter if NLTK or its
    punkt data isn't available, so the module degrades gracefully rather
    than raising an import error.
    """
    if not text or not text.strip():
        return []

    try:
        import nltk

        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            return sentences
    except ImportError:
        pass

    return _regex_segment(text)

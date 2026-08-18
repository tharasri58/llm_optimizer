"""
redundancy.py — embedding-based near-duplicate removal.

The report (Section 2.3 / 4.2) describes this step using Sentence-BERT
embeddings. sentence-transformers is an optional, heavy dependency, so
this module uses it when installed and transparently falls back to a
TF-IDF cosine-similarity embedding otherwise — the interface and the
threshold-based logic are identical either way, only the embedding
source changes.
"""
from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_st_model = None
_st_load_attempted = False


def _get_sentence_transformer():
    global _st_model, _st_load_attempted
    if _st_load_attempted:
        return _st_model
    _st_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        _st_model = None
    return _st_model


def _embed(sentences: List[str]):
    model = _get_sentence_transformer()
    if model is not None:
        return model.encode(sentences)
    # Fallback: TF-IDF vectors as a lightweight embedding proxy.
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        return vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return None


def remove_near_duplicates(
    sentences: List[str],
    protect: List[bool] | None = None,
    threshold: float = 0.92,
) -> List[str]:
    """Drop sentences that are near-duplicates (cosine similarity >=
    ``threshold``) of a sentence already kept. Protected sentences
    (instructional / negation-bearing, per ``flag_instructional``) are
    never dropped, even if they duplicate another kept sentence.

    This matters most for RAG-style prompts built from overlapping
    document chunks, which routinely contain several sentences that are,
    for all practical purposes, saying the same thing twice.
    """
    if len(sentences) <= 1:
        return list(sentences)

    protect = protect or [False] * len(sentences)
    embeddings = _embed(sentences)
    if embeddings is None:
        return list(sentences)

    sims = cosine_similarity(embeddings)
    kept_idx: List[int] = []
    for i, sentence in enumerate(sentences):
        is_duplicate = False
        if not protect[i]:
            for j in kept_idx:
                if sims[i, j] >= threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            kept_idx.append(i)

    return [sentences[i] for i in kept_idx]

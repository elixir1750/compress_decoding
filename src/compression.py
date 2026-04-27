"""Prompt compression baselines and BP-RPC implementation."""

from __future__ import annotations

import random
import re
from typing import Callable, Iterable, List, Sequence, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentences(text: str) -> List[str]:
    """Split English text into non-empty sentences using punctuation boundaries."""
    if not text or not text.strip():
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return sentences if sentences else [text.strip()]


def count_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens in text without adding special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_budget(text: str, tokenizer, budget: int) -> str:
    """Token-level truncate text to fit within the token budget."""
    if budget is None:
        return text
    if budget <= 0:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= budget:
        return text
    return tokenizer.decode(token_ids[:budget], skip_special_tokens=True)


def _join_sentences(sentences: Iterable[str]) -> str:
    """Join selected sentences into a compact prompt."""
    return " ".join(s.strip() for s in sentences if s and s.strip()).strip()


def _sentence_token_counts(sentences: Sequence[str], tokenizer) -> List[int]:
    """Count tokens for each sentence."""
    return [count_tokens(sentence, tokenizer) for sentence in sentences]


def _select_until_budget(indices: Sequence[int], token_counts: Sequence[int], budget: int) -> Set[int]:
    """Select sentence indices in the given order while respecting budget."""
    selected: Set[int] = set()
    used = 0
    for idx in indices:
        sentence_tokens = token_counts[idx]
        if sentence_tokens <= 0:
            continue
        if used + sentence_tokens <= budget:
            selected.add(idx)
            used += sentence_tokens
    return selected


def _pseudo_query(prompt_text: str, tokenizer, query_token_len: int) -> str:
    """Decode the final query_token_len tokens as a pseudo-query."""
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not token_ids:
        return ""
    query_ids = token_ids[-max(query_token_len, 1) :]
    return tokenizer.decode(query_ids, skip_special_tokens=True)


def _tfidf_relevance(sentences: Sequence[str], query: str) -> np.ndarray:
    """Compute TF-IDF cosine relevance between each sentence and query."""
    if not sentences:
        return np.array([], dtype=float)
    if not query or not query.strip():
        return np.zeros(len(sentences), dtype=float)

    corpus = list(sentences) + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    sentence_matrix = matrix[:-1]
    query_vector = matrix[-1]
    scores = cosine_similarity(sentence_matrix, query_vector).reshape(-1)
    return np.asarray(scores, dtype=float)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1], returning zeros when all values match."""
    if values.size == 0:
        return values
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def full_prompt(prompt_text: str, tokenizer, budget: int | None = None) -> str:
    """Return the original prompt without compression."""
    return prompt_text


def first_k(prompt_text: str, tokenizer, budget: int) -> str:
    """Select sentences from the beginning until the token budget is reached."""
    sentences = split_sentences(prompt_text)
    token_counts = _sentence_token_counts(sentences, tokenizer)
    selected = _select_until_budget(range(len(sentences)), token_counts, budget)
    compressed = _join_sentences(sentences[i] for i in range(len(sentences)) if i in selected)
    return truncate_to_budget(compressed, tokenizer, budget)


def last_k(prompt_text: str, tokenizer, budget: int) -> str:
    """Select sentences from the end until the token budget is reached, preserving order."""
    sentences = split_sentences(prompt_text)
    token_counts = _sentence_token_counts(sentences, tokenizer)
    selected = _select_until_budget(range(len(sentences) - 1, -1, -1), token_counts, budget)
    compressed = _join_sentences(sentences[i] for i in range(len(sentences)) if i in selected)
    return truncate_to_budget(compressed, tokenizer, budget)


def random_k(prompt_text: str, tokenizer, budget: int, seed: int = 42) -> str:
    """Randomly select sentences until budget is reached, then restore original order."""
    sentences = split_sentences(prompt_text)
    token_counts = _sentence_token_counts(sentences, tokenizer)
    indices = list(range(len(sentences)))
    random.Random(seed).shuffle(indices)
    selected = _select_until_budget(indices, token_counts, budget)
    compressed = _join_sentences(sentences[i] for i in range(len(sentences)) if i in selected)
    return truncate_to_budget(compressed, tokenizer, budget)


def tfidf_k(prompt_text: str, tokenizer, budget: int, query_token_len: int = 128) -> str:
    """Select sentences by TF-IDF relevance to a pseudo-query from the prompt tail."""
    sentences = split_sentences(prompt_text)
    if not sentences:
        return ""

    try:
        query = _pseudo_query(prompt_text, tokenizer, query_token_len)
        relevance = _tfidf_relevance(sentences, query)
        ranked_indices = sorted(range(len(sentences)), key=lambda i: float(relevance[i]), reverse=True)
    except Exception:
        return last_k(prompt_text, tokenizer, budget)

    token_counts = _sentence_token_counts(sentences, tokenizer)
    selected = _select_until_budget(ranked_indices, token_counts, budget)
    compressed = _join_sentences(sentences[i] for i in range(len(sentences)) if i in selected)
    return truncate_to_budget(compressed, tokenizer, budget)


def bp_rpc(
    prompt_text: str,
    tokenizer,
    budget: int,
    alpha: float = 0.7,
    beta: float = 0.3,
    query_token_len: int = 128,
    keep_head: int = 1,
    keep_tail: int = 2,
) -> str:
    """Boundary-Preserved Recency-Aware Prompt Compression.

    BP-RPC preserves prompt boundaries, then fills the remaining budget using
    TF-IDF relevance to a pseudo-query and a recency score.
    """
    sentences = split_sentences(prompt_text)
    n_sentences = len(sentences)
    if n_sentences == 0:
        return ""
    if budget <= 0:
        return ""

    head_indices = set(range(min(max(keep_head, 0), n_sentences)))
    tail_start = max(n_sentences - max(keep_tail, 0), 0)
    tail_indices = set(range(tail_start, n_sentences))
    boundary_indices = head_indices | tail_indices

    token_counts = _sentence_token_counts(sentences, tokenizer)

    # If boundaries exceed budget, prefer tail context and only keep head
    # sentences when the tail-preserving budget still has room.
    boundary_tokens = sum(token_counts[i] for i in boundary_indices)
    if boundary_tokens > budget:
        selected = set(tail_indices)
        tail_text = _join_sentences(sentences[i] for i in range(n_sentences) if i in selected)
        if count_tokens(tail_text, tokenizer) >= budget:
            return truncate_to_budget(tail_text, tokenizer, budget)

        used_tokens = sum(token_counts[i] for i in selected)
        for idx in sorted(head_indices):
            if idx not in selected and used_tokens + token_counts[idx] <= budget:
                selected.add(idx)
                used_tokens += token_counts[idx]

        boundary_text = _join_sentences(sentences[i] for i in range(n_sentences) if i in selected)
        return truncate_to_budget(boundary_text, tokenizer, budget)

    try:
        query = _pseudo_query(prompt_text, tokenizer, query_token_len)
        relevance = _minmax_normalize(_tfidf_relevance(sentences, query))
    except Exception:
        return last_k(prompt_text, tokenizer, budget)

    selected = set(boundary_indices)
    used_tokens = boundary_tokens
    candidates = [i for i in range(n_sentences) if i not in selected]

    scored_candidates = []
    for i in candidates:
        recency = i / max(n_sentences - 1, 1)
        score = alpha * float(relevance[i]) + beta * float(recency)
        scored_candidates.append((score, i))

    for _, idx in sorted(scored_candidates, key=lambda item: item[0], reverse=True):
        sentence_tokens = token_counts[idx]
        if sentence_tokens <= 0:
            continue
        if used_tokens + sentence_tokens <= budget:
            selected.add(idx)
            used_tokens += sentence_tokens

    compressed = _join_sentences(sentences[i] for i in range(n_sentences) if i in selected)
    return truncate_to_budget(compressed, tokenizer, budget)


def get_compressor(name: str) -> Callable:
    """Return a compressor function by method name."""
    compressors = {
        "full": full_prompt,
        "first": first_k,
        "last": last_k,
        "random": random_k,
        "tfidf": tfidf_k,
        "bp_rpc": bp_rpc,
    }
    normalized = name.lower().strip()
    if normalized not in compressors:
        raise ValueError(f"Unknown compressor '{name}'. Available: {sorted(compressors)}")
    return compressors[normalized]

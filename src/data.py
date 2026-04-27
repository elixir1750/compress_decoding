"""Data loading and prompt-target construction utilities."""

from __future__ import annotations

import re
from typing import Dict, List

from datasets import load_dataset


FALLBACK_TEXTS = [
    (
        "Natural language processing studies how computers process and generate human language. "
        "Early systems relied on hand-written rules, while modern systems often use neural networks. "
        "A language model estimates the probability of text one token at a time. "
        "During inference, long prompts can increase latency because the model must process every input token. "
        "Prompt compression attempts to preserve useful context while reducing the number of tokens. "
        "A simple baseline keeps the first part of the prompt. "
        "Another baseline keeps the most recent part of the prompt. "
        "A stronger method can preserve boundaries, estimate relevance, and prefer recent information. "
        "This lightweight experiment is designed for laptops and small language models. "
    )
    * 80,
    (
        "Machine learning experiments should be reproducible and easy to inspect. "
        "A clear project structure helps readers understand data loading, model evaluation, and benchmarking. "
        "For small computers, it is important to avoid unnecessary training and heavy serving frameworks. "
        "CPU inference is slower than GPU inference, but careful prompt reduction can still show measurable effects. "
        "Perplexity can be computed only on target tokens by masking the prompt portion of the labels. "
        "Generation benchmarks should report total time, generated tokens, and throughput. "
        "These metrics reveal the trade-off between quality and speed. "
    )
    * 90,
    (
        "Information retrieval often uses term frequency and inverse document frequency to score relevance. "
        "A pseudo-query can be extracted from the end of a prompt when no explicit query is available. "
        "Sentences similar to the pseudo-query may contain useful facts for continuation. "
        "However, the beginning and ending of a prompt can carry structural information. "
        "Boundary preservation protects instructions, setup, and recent context. "
        "Recency-aware scoring further favors sentences closer to the model continuation point. "
    )
    * 100,
]


def split_sentences_for_pairs(text: str) -> List[str]:
    """Split text into sentences for sentence-boundary pair construction."""
    if not text or not text.strip():
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return sentences if sentences else [text.strip()]


def load_text_samples(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 20,
) -> List[str]:
    """Load lightweight text samples, falling back to built-in text on failure.

    WikiText raw rows are often short paragraphs. This loader merges consecutive
    non-empty rows into longer samples so default 1024-token prompts can be built
    without requiring a large dataset scan at evaluation time.
    """
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        texts: List[str] = []
        buffer: List[str] = []
        buffer_words = 0
        min_words_per_sample = 900

        for item in dataset:
            text = str(item.get("text", "")).strip()
            words = text.split()
            if len(words) < 5:
                continue

            buffer.append(text)
            buffer_words += len(words)
            if buffer_words >= min_words_per_sample:
                texts.append(" ".join(buffer))
                buffer = []
                buffer_words = 0

            if len(texts) >= max_samples:
                break

        if buffer and len(texts) < max_samples and buffer_words >= 50:
            texts.append(" ".join(buffer))

        if texts:
            return texts
    except Exception as exc:
        print(f"Warning: failed to load dataset '{dataset_name}/{dataset_config}' ({exc}). Using fallback texts.")

    return FALLBACK_TEXTS[: max(1, min(max_samples, len(FALLBACK_TEXTS)))]


def build_prompt_target_pairs(
    texts: List[str],
    tokenizer,
    prompt_len: int = 1024,
    target_len: int = 128,
    max_pairs: int = 20,
    pair_mode: str = "sentence",
) -> List[Dict[str, object]]:
    """Build prompt-target pairs from texts.

    In ``token`` mode, pairs are built with the original fixed token slicing
    strategy. In ``sentence`` mode, the prompt ends at a sentence boundary and
    the target begins from the following sentence, which better matches the
    sentence-level compression methods used in this project.
    """
    if pair_mode not in {"token", "sentence"}:
        raise ValueError("pair_mode must be either 'token' or 'sentence'")

    if pair_mode == "sentence":
        return build_sentence_prompt_target_pairs(
            texts,
            tokenizer,
            prompt_len=prompt_len,
            target_len=target_len,
            max_pairs=max_pairs,
        )

    return build_token_prompt_target_pairs(
        texts,
        tokenizer,
        prompt_len=prompt_len,
        target_len=target_len,
        max_pairs=max_pairs,
    )


def build_token_prompt_target_pairs(
    texts: List[str],
    tokenizer,
    prompt_len: int = 1024,
    target_len: int = 128,
    max_pairs: int = 20,
) -> List[Dict[str, object]]:
    """Build prompt-target pairs by fixed token slicing."""
    pairs: List[Dict[str, object]] = []
    required_tokens = prompt_len + target_len

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < required_tokens:
            continue

        prompt_tokens = token_ids[:prompt_len]
        target_tokens = token_ids[prompt_len : prompt_len + target_len]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)

        pairs.append(
            {
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_len": len(prompt_tokens),
                "target_len": len(target_tokens),
            }
        )
        if len(pairs) >= max_pairs:
            break

    return pairs


def build_sentence_prompt_target_pairs(
    texts: List[str],
    tokenizer,
    prompt_len: int = 1024,
    target_len: int = 128,
    max_pairs: int = 20,
) -> List[Dict[str, object]]:
    """Build prompt-target pairs where target text starts at a sentence boundary."""
    pairs: List[Dict[str, object]] = []
    min_prompt_tokens = max(1, int(prompt_len * 0.6))

    for text in texts:
        sentences = split_sentences_for_pairs(text)
        if len(sentences) < 2:
            continue

        prompt_sentences: List[str] = []
        prompt_tokens: List[int] = []
        sentence_idx = 0

        while sentence_idx < len(sentences):
            candidate_prompt = " ".join(prompt_sentences + [sentences[sentence_idx]])
            candidate_tokens = tokenizer.encode(candidate_prompt, add_special_tokens=False)
            if prompt_sentences and len(candidate_tokens) > prompt_len:
                break

            prompt_sentences.append(sentences[sentence_idx])
            prompt_tokens = candidate_tokens
            sentence_idx += 1

            if len(prompt_tokens) >= prompt_len:
                break

        if len(prompt_tokens) < min_prompt_tokens or sentence_idx >= len(sentences):
            continue

        target_sentences: List[str] = []
        target_tokens: List[int] = []
        while sentence_idx < len(sentences) and len(target_tokens) < target_len:
            target_sentences.append(sentences[sentence_idx])
            target_text = " ".join(target_sentences)
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
            sentence_idx += 1

        if len(target_tokens) < target_len:
            continue

        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        target_text = tokenizer.decode(target_tokens[:target_len], skip_special_tokens=True)
        pairs.append(
            {
                "prompt_text": prompt_text,
                "target_text": target_text,
                "prompt_len": len(prompt_tokens),
                "target_len": min(len(target_tokens), target_len),
            }
        )
        if len(pairs) >= max_pairs:
            break

    return pairs

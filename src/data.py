"""Data loading and prompt-target construction utilities."""

from __future__ import annotations

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
) -> List[Dict[str, object]]:
    """Build prompt-target pairs by slicing tokenized texts."""
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

#!/usr/bin/env python3
"""Run generation benchmark for prompt compression methods."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import benchmark_generation
from src.compression import count_tokens, get_compressor
from src.data import build_prompt_target_pairs, load_text_samples
from src.utils import get_device, load_model_and_tokenizer, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark prompt compression generation latency.")
    parser.add_argument("--model_name", default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--prompt_len", type=int, default=1024)
    parser.add_argument("--keep_ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--methods", nargs="+", default=["full", "first", "last", "random", "tfidf", "bp_rpc"])
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output", default="results/benchmark_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds used for the Random baseline. Use multiple seeds for more robust random averages.",
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default=None)
    parser.add_argument("--cache_dir", default=None, help="Optional HuggingFace cache directory.")
    parser.add_argument("--local_files_only", action="store_true", help="Load the model from local cache only.")
    return parser.parse_args()


def compress_prompt(method: str, prompt_text: str, tokenizer, budget: int, seed: int) -> str:
    """Apply a named compressor with method-specific arguments."""
    compressor = get_compressor(method)
    if method == "full":
        return compressor(prompt_text, tokenizer, None)
    if method == "random":
        return compressor(prompt_text, tokenizer, budget, seed=seed)
    return compressor(prompt_text, tokenizer, budget)


def method_seed_runs(methods: list[str], random_seeds: list[int]) -> list[Tuple[str, Optional[int]]]:
    """Expand methods so Random can be benchmarked over one or more seeds."""
    runs: list[Tuple[str, Optional[int]]] = []
    for method in methods:
        if method == "random":
            runs.extend((method, seed) for seed in random_seeds)
        else:
            runs.append((method, None))
    return runs


def main() -> None:
    """Run benchmark and save a CSV file."""
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    texts = load_text_samples(args.dataset_name, args.dataset_config, args.split, args.max_samples)
    pairs = build_prompt_target_pairs(
        texts,
        tokenizer,
        prompt_len=args.prompt_len,
        target_len=args.max_new_tokens,
        max_pairs=args.max_samples,
    )
    if not pairs:
        raise RuntimeError("No prompt samples could be built. Try smaller --prompt_len.")

    rows = []
    method_runs = method_seed_runs(args.methods, args.random_seeds)
    total = len(pairs) * len(args.keep_ratios) * len(method_runs)
    progress = tqdm(total=total, desc="Benchmarking")

    for sample_id, pair in enumerate(pairs):
        prompt_text = str(pair["prompt_text"])
        original_prompt_tokens = count_tokens(prompt_text, tokenizer)

        for keep_ratio in args.keep_ratios:
            for method, random_seed in method_runs:
                budget = args.prompt_len if method == "full" else max(1, int(args.prompt_len * keep_ratio))
                run_seed = args.seed if random_seed is None else random_seed
                compressed_prompt = compress_prompt(method, prompt_text, tokenizer, budget, run_seed)
                compressed_prompt_tokens = count_tokens(compressed_prompt, tokenizer)
                metrics = benchmark_generation(
                    model,
                    tokenizer,
                    compressed_prompt,
                    device,
                    max_new_tokens=args.max_new_tokens,
                )

                rows.append(
                    {
                        "sample_id": sample_id,
                        "method": method,
                        "random_seed": "" if random_seed is None else random_seed,
                        "keep_ratio": keep_ratio,
                        "original_prompt_tokens": original_prompt_tokens,
                        "compressed_prompt_tokens": compressed_prompt_tokens,
                        "generated_tokens": metrics["generated_tokens"],
                        "total_time": metrics["total_time"],
                        "time_per_output_token": metrics["time_per_output_token"],
                        "throughput_tokens_per_sec": metrics["throughput_tokens_per_sec"],
                    }
                )
                progress.update(1)

    progress.close()
    output_path = PROJECT_ROOT / args.output
    os.makedirs(output_path.parent, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()

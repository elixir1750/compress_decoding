"""Generation benchmark utilities."""

from __future__ import annotations

import time
from typing import Dict

import torch


def benchmark_generation(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 32,
) -> Dict[str, float]:
    """Benchmark greedy generation latency for one prompt."""
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_tokens = int(input_ids.shape[1])

    if prompt_tokens > 0 and max_new_tokens > 0:
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids[:, -min(prompt_tokens, 32) :],
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    total_time = time.perf_counter() - start

    generated_tokens = max(int(outputs.shape[1] - input_ids.shape[1]), 0)
    time_per_output_token = total_time / generated_tokens if generated_tokens else float("inf")
    throughput = generated_tokens / total_time if total_time > 0 else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_time": float(total_time),
        "time_per_output_token": float(time_per_output_token),
        "throughput_tokens_per_sec": float(throughput),
    }

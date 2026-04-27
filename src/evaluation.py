"""Perplexity evaluation utilities."""

from __future__ import annotations

import math
from typing import Dict

import torch


def evaluate_ppl(model, tokenizer, prompt_text: str, target_text: str, device: str) -> Dict[str, float]:
    """Evaluate loss and perplexity on target tokens only."""
    full_text = prompt_text + target_text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_token_count = min(len(prompt_ids), input_ids.shape[1])
    labels = input_ids.clone()
    labels[:, :prompt_token_count] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    loss_value = float(loss.detach().cpu().item())
    ppl = float("inf") if loss_value > 50 else float(math.exp(loss_value))

    return {
        "loss": loss_value,
        "ppl": ppl,
        "num_input_tokens": int(input_ids.shape[1]),
        "num_prompt_tokens": int(prompt_token_count),
        "num_target_tokens": int(input_ids.shape[1] - prompt_token_count),
    }

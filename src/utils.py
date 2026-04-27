"""Shared utilities for reproducibility and model loading."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    """Return Apple MPS when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible lightweight experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def load_model_and_tokenizer(
    model_name: str = "EleutherAI/pythia-70m",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load a causal LM and tokenizer from HuggingFace or a local cache/path."""
    if device is None:
        device = get_device()

    load_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except OSError as exc:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        offline_hint = (
            "You passed --local_files_only, but the model was not found locally."
            if local_files_only
            else "The model download failed. This is usually a network/SSL or HuggingFace access issue."
        )
        raise RuntimeError(
            f"{offline_hint}\n"
            f"Model: {model_name}\n"
            f"HF endpoint: {endpoint}\n"
            f"Cache dir: {cache_dir or 'default HuggingFace cache'}\n\n"
            "Try one of these:\n"
            "1. Use a reachable mirror before running the script:\n"
            "   export HF_ENDPOINT=https://hf-mirror.com\n"
            "2. Pre-download the model, then run with --local_files_only.\n"
            "3. Pass a local model directory with --model_name /path/to/pythia-70m.\n"
            "4. If the network is unstable, rerun after the partial HuggingFace cache finishes/clears.\n"
        ) from exc

    model.to(device)
    model.eval()
    return model, tokenizer

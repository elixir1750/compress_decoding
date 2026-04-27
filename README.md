# BP-RPC: Boundary-Preserved Recency-Aware Prompt Compression

[中文说明](README_zh.md)

BP-RPC is a lightweight, training-free prompt compression project for language model inference acceleration. It reduces the number of input tokens before inference, aiming to lower prefill latency and total generation time while preserving enough context for reasonable output quality.

The default model is `EleutherAI/pythia-70m` from HuggingFace. The code is designed for MacBook Air style environments: CPU first, optional Apple MPS detection, no CUDA, no vLLM, and no training.

## Results Preview

Example results from a lightweight CPU run show that BP-RPC preserves target-token PPL much better than First-K and Random at low keep ratios, while staying competitive with Last-K and TF-IDF.

![PPL vs Keep Ratio](docs/figures/ppl_vs_keep_ratio.png)

The speed-quality trade-off plot is useful for report discussion, but benchmark numbers should be interpreted as example measurements because laptop CPU latency can be noisy.

![Speedup vs Quality Trade-off](docs/figures/speedup_vs_ppl_tradeoff.png)

See [docs/results_summary.md](docs/results_summary.md) for a compact table summary of the example run.

## Method Overview

The project compares six prompt handling methods:

- **Full Prompt**: uses the original prompt without compression.
- **First-K**: keeps sentences from the beginning until the token budget is reached.
- **Last-K**: keeps sentences from the end until the token budget is reached.
- **Random**: randomly keeps sentences under the budget with a fixed seed.
- **TF-IDF**: builds a pseudo-query from the final prompt tokens and keeps sentences most relevant to it.
- **BP-RPC**: combines boundary preservation, pseudo-query relevance, and recency-aware scoring.

BP-RPC keeps the first `keep_head` sentences and last `keep_tail` sentences, then scores remaining sentences with:

```text
score_i = alpha * relevance_i + beta * recency_i
```

where `relevance_i` is TF-IDF cosine similarity to the pseudo-query and `recency_i` is larger for sentences closer to the end of the prompt.

## Installation

```bash
pip install -r requirements.txt
```

## Run Evaluation

Perplexity is computed only on target tokens. The prompt portion of the labels is masked with `-100`.

```bash
python scripts/run_eval.py --max_samples 10 --prompt_len 1024 --target_len 128
```

For a faster first run on MacBook Air:

```bash
python scripts/run_eval.py --max_samples 5 --prompt_len 512 --target_len 64 --device cpu
```

If HuggingFace downloads fail because of SSL or network interruptions, try a mirror or a local model directory:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_eval.py --max_samples 5 --prompt_len 512 --target_len 64 --device cpu

huggingface-cli download EleutherAI/pythia-70m --local-dir models/pythia-70m
python scripts/run_eval.py --model_name models/pythia-70m --max_samples 5 --prompt_len 512 --target_len 64 --device cpu --local_files_only
```

## Run Benchmark

Generation uses greedy decoding with `do_sample=False`.

```bash
python scripts/run_benchmark.py --max_samples 5 --prompt_len 1024 --max_new_tokens 32
```

For a faster first run:

```bash
python scripts/run_benchmark.py --max_samples 3 --prompt_len 512 --max_new_tokens 16 --device cpu
```

## Plot Results

After `results/eval_results.csv` or `results/benchmark_results.csv` exists, generate report-ready figures with:

```bash
python scripts/plot_results.py
```

Figures are saved to `results/figures/` by default. The script creates PPL, loss, compressed-token, latency, throughput, speedup, and speedup-vs-PPL trade-off plots when the required CSV columns are available.

If PPL outliers make the plot hard to read, clip the y-axis data:

```bash
python scripts/plot_results.py --max_ppl 200
```

## Output Files

- `results/eval_results.csv`: perplexity results with method, keep ratio, compressed prompt tokens, loss, and PPL.
- `results/benchmark_results.csv`: generation timing results with compressed prompt tokens, total time, time per output token, and throughput.
- `results/figures/`: generated PNG/PDF/SVG figures from result CSV files.

## Recommended Tables

Useful report tables and plots:

- **PPL vs keep ratio**: compare quality degradation as prompts become shorter.
- **Compressed tokens vs latency**: show how token count affects inference time.
- **Speedup vs quality trade-off**: compare latency reduction against PPL increase.

## Notes

- MacBook Air users should start with `--max_samples 5`.
- If MPS has compatibility issues, force CPU with `--device cpu`.
- If HuggingFace access fails, set `HF_ENDPOINT=https://hf-mirror.com` or pass a local model path with `--model_name`.
- Dataset loading falls back to built-in English long texts if HuggingFace download fails.
- PPL is computed only on target tokens, not on the prompt.
- This is an inference-only experiment; it does not train or fine-tune the model.

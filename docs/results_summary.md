# Example Results Summary

This page summarizes one lightweight CPU run. The exact numbers may vary across machines, Python environments, and background system load.

## Target-token PPL

Evaluation setup:

- Samples: 5
- Prompt length: 512 tokens
- Target length: 64 tokens
- Model: `EleutherAI/pythia-70m`
- Device: CPU

| Method | 0.25 | 0.50 | 0.75 |
|---|---:|---:|---:|
| Full | 51.13 | 51.13 | 51.13 |
| First-K | 73.03 | 62.96 | 60.46 |
| Last-K | 58.88 | 53.40 | 54.78 |
| Random | 67.02 | 56.90 | 54.17 |
| TF-IDF | 60.10 | 54.51 | 51.47 |
| BP-RPC | 59.70 | 54.96 | 53.11 |

BP-RPC is substantially better than First-K and Random at 25% and 50% keep ratios, while staying close to the stronger Last-K and TF-IDF baselines.

## Generation Speedup

Benchmark setup:

- Samples: 2
- Prompt length: 512 tokens
- Max new tokens: 16
- Device: CPU

| Method | 0.25 | 0.50 | 0.75 |
|---|---:|---:|---:|
| Full | 1.00x | 1.00x | 1.00x |
| First-K | 1.04x | 0.67x | 1.14x |
| Last-K | 1.05x | 1.03x | 0.96x |
| Random | 0.96x | 0.70x | 0.98x |
| TF-IDF | 1.20x | 1.01x | 1.06x |
| BP-RPC | 1.06x | 1.14x | 1.07x |

These speed numbers are illustrative. Laptop CPU benchmarks can be noisy, so the latency experiment is best interpreted together with compressed token counts and repeated runs.

## Figures

![PPL vs Keep Ratio](figures/ppl_vs_keep_ratio.png)

![Speedup vs Quality Trade-off](figures/speedup_vs_ppl_tradeoff.png)

#!/usr/bin/env python3
"""Print robust summaries for evaluation and benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["full", "first", "last", "random", "tfidf", "bp_rpc"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Summarize result CSV files.")
    parser.add_argument("--eval_csv", default="results/eval_results.csv")
    parser.add_argument("--benchmark_csv", default="results/benchmark_results.csv")
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path


def method_sort_key(method: str) -> tuple[int, str]:
    """Sort methods in a stable report-friendly order."""
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return len(METHOD_ORDER), method


def print_metric_table(df: pd.DataFrame, metric: str, lower_is_better: bool = True) -> None:
    """Print mean, median, and std for a metric by keep ratio and method."""
    if df.empty or metric not in df.columns:
        return

    print(f"\n== {metric} by method ==")
    data = df.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])

    for ratio in sorted(data["keep_ratio"].unique()):
        ratio_df = data[data["keep_ratio"] == ratio]
        summary = (
            ratio_df.groupby("method")[metric]
            .agg(["mean", "median", "std", "count"])
            .reset_index()
        )
        summary = summary.sort_values("mean", ascending=lower_is_better)
        print(f"\nkeep_ratio={ratio}")
        for _, row in summary.iterrows():
            print(
                f"{row['method']:7s} "
                f"mean={row['mean']:8.3f} "
                f"median={row['median']:8.3f} "
                f"std={row['std']:8.3f} "
                f"n={int(row['count'])}"
            )


def print_eval_winners(eval_df: pd.DataFrame) -> None:
    """Print per-sample winner counts after averaging repeated random seeds."""
    required = {"sample_id", "method", "keep_ratio", "ppl"}
    if eval_df.empty or not required.issubset(eval_df.columns):
        return

    print("\n== compressed-method winner counts by sample ==")
    df = eval_df.copy()
    df["ppl"] = pd.to_numeric(df["ppl"], errors="coerce")
    df = df.dropna(subset=["ppl"])
    df = df[df["method"] != "full"]
    per_method = df.groupby(["keep_ratio", "sample_id", "method"], as_index=False)["ppl"].mean()

    for ratio in sorted(per_method["keep_ratio"].unique()):
        ratio_df = per_method[per_method["keep_ratio"] == ratio]
        winners = ratio_df.loc[ratio_df.groupby("sample_id")["ppl"].idxmin()]
        counts = winners["method"].value_counts().to_dict()
        print(f"keep_ratio={ratio}: {dict(sorted(counts.items(), key=lambda kv: method_sort_key(kv[0])))}")


def print_speedup_table(benchmark_df: pd.DataFrame) -> None:
    """Print speedup relative to Full Prompt by method."""
    required = {"sample_id", "method", "keep_ratio", "total_time"}
    if benchmark_df.empty or not required.issubset(benchmark_df.columns):
        return

    df = benchmark_df.copy()
    df["total_time"] = pd.to_numeric(df["total_time"], errors="coerce")
    df = df.dropna(subset=["total_time"])
    full = df[df["method"] == "full"][["sample_id", "keep_ratio", "total_time"]].rename(
        columns={"total_time": "full_total_time"}
    )
    merged = df.merge(full, on=["sample_id", "keep_ratio"], how="inner")
    merged = merged[merged["total_time"] > 0]
    merged["speedup_vs_full"] = merged["full_total_time"] / merged["total_time"]
    print_metric_table(merged, "speedup_vs_full", lower_is_better=False)


def main() -> None:
    """Print result summaries."""
    args = parse_args()
    eval_path = resolve_path(args.eval_csv)
    benchmark_path = resolve_path(args.benchmark_csv)

    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    benchmark_df = pd.read_csv(benchmark_path) if benchmark_path.exists() else pd.DataFrame()

    print_metric_table(eval_df, "ppl", lower_is_better=True)
    print_eval_winners(eval_df)
    print_speedup_table(benchmark_df)


if __name__ == "__main__":
    main()

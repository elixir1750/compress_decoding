#!/usr/bin/env python3
"""Plot evaluation and benchmark CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["full", "first", "last", "random", "tfidf", "bp_rpc"]
METHOD_LABELS = {
    "full": "Full",
    "first": "First-K",
    "last": "Last-K",
    "random": "Random",
    "tfidf": "TF-IDF",
    "bp_rpc": "BP-RPC",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create figures from result CSV files.")
    parser.add_argument("--eval_csv", default="results/eval_results.csv", help="Path to eval_results.csv.")
    parser.add_argument(
        "--benchmark_csv",
        default="results/benchmark_results.csv",
        help="Path to benchmark_results.csv.",
    )
    parser.add_argument("--output_dir", default="results/figures", help="Directory for output figures.")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output figure format.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI for raster outputs.")
    parser.add_argument(
        "--agg",
        default="mean",
        choices=["mean", "median"],
        help="Aggregation used for line plots. Median is more robust to PPL outliers.",
    )
    parser.add_argument(
        "--max_ppl",
        type=float,
        default=None,
        help="Optional upper clipping value for PPL plots to reduce outlier distortion.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    """Resolve a possibly relative path against the project root."""
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path


def method_sort_key(method: str) -> tuple[int, str]:
    """Sort methods in a stable, report-friendly order."""
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method), method
    return len(METHOD_ORDER), method


def method_label(method: str) -> str:
    """Return display label for a method name."""
    return METHOD_LABELS.get(method, method)


def setup_style() -> None:
    """Set a clean matplotlib style for report figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 5.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def save_current_figure(output_dir: Path, filename: str, fmt: str, dpi: int) -> Path:
    """Save the current matplotlib figure and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.{fmt}"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return output_path


def aggregate_metric(df: pd.DataFrame, metric: str, agg: str = "mean") -> pd.DataFrame:
    """Aggregate a metric by method and keep ratio."""
    required = {"method", "keep_ratio", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting {metric}: {sorted(missing)}")

    plot_df = df.copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if agg == "median":
        agg_df = (
            plot_df.groupby(["method", "keep_ratio"])[metric]
            .agg(["median", "mean", "std"])
            .reset_index()
            .rename(columns={"median": metric, "mean": f"{metric}_mean", "std": f"{metric}_std"})
        )
        return agg_df

    agg_df = (
        plot_df.groupby(["method", "keep_ratio"])[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": metric, "std": f"{metric}_std"})
    )
    return agg_df


def plot_line_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_dir: Path,
    filename: str,
    fmt: str,
    dpi: int,
    y_clip: Optional[float] = None,
    agg: str = "mean",
) -> Optional[Path]:
    """Plot a metric against keep ratio for all methods."""
    if df.empty or metric not in df.columns:
        return None

    agg_df = aggregate_metric(df, metric, agg=agg)
    if y_clip is not None:
        agg_df[metric] = agg_df[metric].clip(upper=y_clip)

    plt.figure()
    for method in sorted(agg_df["method"].unique(), key=method_sort_key):
        method_df = agg_df[agg_df["method"] == method].sort_values("keep_ratio")
        if method_df.empty:
            continue
        plt.plot(
            method_df["keep_ratio"],
            method_df[metric],
            marker="o",
            linewidth=2,
            label=method_label(method),
        )

    plt.xlabel("Keep ratio")
    plt.ylabel(ylabel)
    agg_label = "Median" if agg == "median" else "Mean"
    plt.title(f"{title} ({agg_label})")
    plt.legend(ncol=2)
    output_name = filename if agg == "mean" else f"{filename}_{agg}"
    return save_current_figure(output_dir, output_name, fmt, dpi)


def plot_compressed_tokens(
    df: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    dpi: int,
    agg: str = "mean",
) -> Optional[Path]:
    """Plot compressed prompt token count against keep ratio."""
    if "compressed_prompt_tokens" not in df.columns:
        return None
    return plot_line_metric(
        df,
        "compressed_prompt_tokens",
        "Compressed prompt tokens",
        "Compressed Tokens vs Keep Ratio",
        output_dir,
        "compressed_tokens_vs_keep_ratio",
        fmt,
        dpi,
        agg=agg,
    )


def compute_speedup(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Compute speedup relative to Full Prompt for each sample and keep ratio."""
    required = {"sample_id", "method", "keep_ratio", "total_time"}
    missing = required - set(benchmark_df.columns)
    if missing:
        raise ValueError(f"Missing columns for speedup: {sorted(missing)}")

    df = benchmark_df.copy()
    df["total_time"] = pd.to_numeric(df["total_time"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["total_time"])
    full = df[df["method"] == "full"][["sample_id", "keep_ratio", "total_time"]].rename(
        columns={"total_time": "full_total_time"}
    )
    merged = df.merge(full, on=["sample_id", "keep_ratio"], how="left")
    merged = merged.dropna(subset=["full_total_time"])
    merged = merged[merged["total_time"] > 0]
    merged["speedup_vs_full"] = merged["full_total_time"] / merged["total_time"]
    return merged


def plot_speedup(
    benchmark_df: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    dpi: int,
    agg: str = "mean",
) -> Optional[Path]:
    """Plot generation speedup relative to the full prompt baseline."""
    if benchmark_df.empty:
        return None

    speedup_df = compute_speedup(benchmark_df)
    if speedup_df.empty:
        return None

    agg_func = "median" if agg == "median" else "mean"
    agg_df = (
        speedup_df.groupby(["method", "keep_ratio"], as_index=False)["speedup_vs_full"]
        .agg(agg_func)
        .sort_values(["method", "keep_ratio"])
    )

    plt.figure()
    for method in sorted(agg_df["method"].unique(), key=method_sort_key):
        method_df = agg_df[agg_df["method"] == method].sort_values("keep_ratio")
        plt.plot(
            method_df["keep_ratio"],
            method_df["speedup_vs_full"],
            marker="o",
            linewidth=2,
            label=method_label(method),
        )

    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Keep ratio")
    plt.ylabel("Speedup vs Full Prompt")
    agg_label = "Median" if agg == "median" else "Mean"
    plt.title(f"Speedup vs Keep Ratio ({agg_label})")
    plt.legend(ncol=2)
    filename = "speedup_vs_keep_ratio" if agg == "mean" else f"speedup_vs_keep_ratio_{agg}"
    return save_current_figure(output_dir, filename, fmt, dpi)


def plot_quality_speed_tradeoff(
    eval_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    output_dir: Path,
    fmt: str,
    dpi: int,
    max_ppl: Optional[float] = None,
    agg: str = "mean",
) -> Optional[Path]:
    """Plot speedup against target-token PPL."""
    if eval_df.empty or benchmark_df.empty:
        return None
    if "ppl" not in eval_df.columns:
        return None

    ppl = aggregate_metric(eval_df, "ppl", agg=agg)[["method", "keep_ratio", "ppl"]]
    if max_ppl is not None:
        ppl["ppl"] = ppl["ppl"].clip(upper=max_ppl)

    speedup = compute_speedup(benchmark_df)
    agg_func = "median" if agg == "median" else "mean"
    speedup = speedup.groupby(["method", "keep_ratio"], as_index=False)["speedup_vs_full"].agg(agg_func)
    merged = ppl.merge(speedup, on=["method", "keep_ratio"], how="inner")
    if merged.empty:
        return None

    plt.figure()
    for method in sorted(merged["method"].unique(), key=method_sort_key):
        method_df = merged[merged["method"] == method].sort_values("keep_ratio")
        plt.scatter(method_df["ppl"], method_df["speedup_vs_full"], s=56, label=method_label(method))
        for _, row in method_df.iterrows():
            plt.annotate(
                f"{row['keep_ratio']:.2g}",
                (row["ppl"], row["speedup_vs_full"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    plt.xlabel("Target-token PPL")
    plt.ylabel("Speedup vs Full Prompt")
    agg_label = "Median" if agg == "median" else "Mean"
    plt.title(f"Speedup vs Quality Trade-off ({agg_label})")
    plt.legend(ncol=2)
    filename = "speedup_vs_ppl_tradeoff" if agg == "mean" else f"speedup_vs_ppl_tradeoff_{agg}"
    return save_current_figure(output_dir, filename, fmt, dpi)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    """Read a CSV file if it exists, otherwise return an empty DataFrame."""
    if not path.exists():
        print(f"Warning: {path} does not exist, skipping related plots.")
        return pd.DataFrame()
    return pd.read_csv(path)


def print_outputs(paths: Iterable[Optional[Path]]) -> None:
    """Print generated figure paths."""
    generated = [path for path in paths if path is not None]
    if not generated:
        print("No figures generated. Check that the input CSV files exist and contain expected columns.")
        return

    print("Generated figures:")
    for path in generated:
        print(f"- {path}")


def main() -> None:
    """Create result figures."""
    args = parse_args()
    setup_style()

    eval_csv = resolve_path(args.eval_csv)
    benchmark_csv = resolve_path(args.benchmark_csv)
    output_dir = resolve_path(args.output_dir)

    eval_df = read_csv_if_exists(eval_csv)
    benchmark_df = read_csv_if_exists(benchmark_csv)

    outputs = []
    outputs.append(
        plot_line_metric(
            eval_df,
            "ppl",
            "Target-token PPL",
            "PPL vs Keep Ratio",
            output_dir,
            "ppl_vs_keep_ratio",
            args.format,
            args.dpi,
            y_clip=args.max_ppl,
            agg=args.agg,
        )
    )
    outputs.append(
        plot_line_metric(
            eval_df,
            "loss",
            "Target-token loss",
            "Loss vs Keep Ratio",
            output_dir,
            "loss_vs_keep_ratio",
            args.format,
            args.dpi,
            agg=args.agg,
        )
    )
    token_source = eval_df if not eval_df.empty else benchmark_df
    outputs.append(plot_compressed_tokens(token_source, output_dir, args.format, args.dpi, agg=args.agg))
    outputs.append(
        plot_line_metric(
            benchmark_df,
            "total_time",
            "Total generation time (s)",
            "Latency vs Keep Ratio",
            output_dir,
            "latency_vs_keep_ratio",
            args.format,
            args.dpi,
            agg=args.agg,
        )
    )
    outputs.append(
        plot_line_metric(
            benchmark_df,
            "throughput_tokens_per_sec",
            "Throughput (tokens/s)",
            "Throughput vs Keep Ratio",
            output_dir,
            "throughput_vs_keep_ratio",
            args.format,
            args.dpi,
            agg=args.agg,
        )
    )
    outputs.append(plot_speedup(benchmark_df, output_dir, args.format, args.dpi, agg=args.agg))
    outputs.append(
        plot_quality_speed_tradeoff(
            eval_df,
            benchmark_df,
            output_dir,
            args.format,
            args.dpi,
            max_ppl=args.max_ppl,
            agg=args.agg,
        )
    )
    print_outputs(outputs)


if __name__ == "__main__":
    main()

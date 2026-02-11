#!/usr/bin/env python3
"""
Plot comparison of different partition strategies for term1's p_acc_call_fail.

This script visualizes how the "tool rescue success rate" (p_acc_call_fail) changes
under three different sample partition strategies:
1. Dynamic partition: D_fail(t) - changes at each step
2. Fixed partition: D_fail(0) - fixed from baseline
3. Intersection partition: D_fail(0) ∩ D_fail(t) - persistently difficult samples
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from plot_paper_figures import (
    PAPER_FIG_CONFIG,
    get_sorted_benchmarks,
    load_experiment_with_baseline,
    setup_plot_style,
    smooth_curve,
)

# Default perception benchmarks
PERCEPTION_BENCHMARKS = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]


def aggregate_partition_metrics(
    df: pd.DataFrame,
    benchmarks: list[str],
    metric_suffix: str,
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
) -> dict[str, Any]:
    """Aggregate a specific metric across benchmarks.

    Args:
        df: DataFrame with experiment data
        benchmarks: List of benchmark names
        metric_suffix: Suffix for metric columns (e.g., '', '_0', '_0_t')
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method

    Returns:
        Dict with aggregated data
    """
    # Metric name with suffix
    p_acc_call_fail_col = f"p_acc_call_fail{metric_suffix}"

    all_values = []
    common_steps = None

    for benchmark in benchmarks:
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            continue

        benchmark_data = benchmark_data.sort_values("step")
        steps = benchmark_data["step"].values

        if common_steps is None:
            common_steps = steps.copy()

        if p_acc_call_fail_col not in benchmark_data.columns:
            continue

        # Use raw values
        values = benchmark_data[p_acc_call_fail_col].values
        all_values.append(values)

    if len(all_values) == 0:
        return None

    # Aggregate (mean across benchmarks)
    aggregated_raw = np.mean(all_values, axis=0)

    # Smooth
    if smoothing_factor > 0 or smoothing_method != "none":
        aggregated_smooth = smooth_curve(
            aggregated_raw, common_steps, smoothing_factor, smoothing_method
        )
    else:
        aggregated_smooth = aggregated_raw.copy()

    return {
        "steps": common_steps,
        "values_raw": aggregated_raw,
        "values_smooth": aggregated_smooth,
    }


def plot_partition_comparison(
    experiment_names: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "partition_comparison_p_acc_call_fail.pdf",
    captions: list[str] | None = None,
    aggregated_benchmarks: list[str] | None = None,
):
    """Plot p_acc_call_fail under three partition strategies.

    Creates a 1xN horizontal grid where each column is one experiment showing three curves:
    - Dynamic partition: D_fail(t) - orange
    - Fixed partition: D_fail(0) - green
    - Intersection partition: D_fail(0) ∩ D_fail(t) - red

    Captions are placed below each subplot as (a) Caption1, (b) Caption2, etc.

    Args:
        experiment_names: List of experiment names to plot
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        captions: List of captions for each experiment (e.g., ["Qwen2.5-VL-Instruct", "Qwen3-VL-Instruct"])
        aggregated_benchmarks: List of benchmark names to aggregate
    """
    if aggregated_benchmarks is None:
        aggregated_benchmarks = PERCEPTION_BENCHMARKS

    setup_plot_style()

    n_experiments = len(experiment_names)

    # Create 1xN figure (horizontal layout)
    fig, axes = plt.subplots(
        1,
        n_experiments,
        figsize=(6 * n_experiments, 5),
        dpi=PAPER_FIG_CONFIG["dpi"],
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    # Ensure axes is iterable
    if n_experiments == 1:
        axes = [axes]

    # Process each experiment
    for exp_idx, exp_name in enumerate(experiment_names):
        print(f"\nProcessing experiment {exp_idx + 1}/{n_experiments}: {exp_name}")

        # Load experiment data with baseline performance
        df = load_experiment_with_baseline(exp_name)
        if df is None:
            print(f"Failed to load experiment '{exp_name}'")
            continue

        selected_benchmarks = get_sorted_benchmarks(df)
        target_benchmarks = [b for b in selected_benchmarks if b in aggregated_benchmarks]

        if not target_benchmarks:
            print(f"No matching benchmarks found for {exp_name}")
            continue

        # Get current axis
        ax = axes[exp_idx]

        # Aggregate three metrics
        # 1. Dynamic partition (no suffix)
        data_dynamic = aggregate_partition_metrics(
            df, target_benchmarks, "", smoothing_factor, smoothing_method
        )

        # 2. Fixed partition (_0 suffix)
        data_fixed = aggregate_partition_metrics(
            df, target_benchmarks, "_0", smoothing_factor, smoothing_method
        )

        # 3. Intersection partition (_0_t suffix)
        data_intersection = aggregate_partition_metrics(
            df, target_benchmarks, "_0_t", smoothing_factor, smoothing_method
        )

        if data_dynamic is None:
            print(f"No data found for {exp_name}")
            continue

        steps = data_dynamic["steps"]

        # Plot dynamic partition
        ax.plot(
            steps,
            data_dynamic["values_smooth"],
            color="darkorange",
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}}(t))$ (Dynamic)",
            alpha=0.9,
            zorder=3,
        )

        # Plot fixed partition if available
        if data_fixed is not None:
            ax.plot(
                steps,
                data_fixed["values_smooth"],
                color="green",
                linestyle="--",
                linewidth=2.5,
                marker="s",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}}(0))$ (Fixed)",
                alpha=0.9,
                zorder=2,
            )

        # Plot intersection partition if available
        if data_intersection is not None:
            ax.plot(
                steps,
                data_intersection["values_smooth"],
                color="red",
                linestyle="-.",
                linewidth=2.5,
                marker="^",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}}(0) \\cap \\mathcal{D}_{\\text{fail}}(t))$ (Intersection)",
                alpha=0.9,
                zorder=1,
            )

        # Styling
        # Set axis labels with larger font size
        ax.set_xlabel("Training Steps", fontsize=16, fontweight="medium")
        ax.set_ylabel(
            "$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}})$",
            fontsize=16,
            fontweight="medium",
        )
        # ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

        # Legend
        ax.legend(
            loc="best", fontsize=10, frameon=True, fancybox=True, shadow=True, framealpha=0.95
        )

        # Scale x-axis labels by 8
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # Add caption below subplot
        if captions and exp_idx < len(captions):
            caption_label = chr(ord("a") + exp_idx)  # 'a', 'b', 'c', ...
            ax.text(
                0.5,
                -0.15,
                f"({caption_label}) {captions[exp_idx]}",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="top",
            )

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Add space for captions
    plt.savefig(output_path, dpi=PAPER_FIG_CONFIG["dpi"], bbox_inches="tight")
    print(f"\n{'='*80}")
    print(f"Saved figure to: {output_path}")
    print(f"{'='*80}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot partition strategy comparison for p_acc_call_fail"
    )
    parser.add_argument(
        "--exp_names", type=str, nargs="+", required=True, help="List of experiment names"
    )
    parser.add_argument(
        "--captions", type=str, nargs="+", default=None, help="List of captions for each experiment"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="List of benchmarks to aggregate (default: perception benchmarks)",
    )
    parser.add_argument("--output_dir", type=str, default="paper_figures", help="Output directory")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="partition_comparison_p_acc_call_fail.pdf",
        help="Output filename",
    )
    parser.add_argument("--smoothing_factor", type=float, default=0.99, help="Smoothing factor")

    args = parser.parse_args()

    plot_partition_comparison(
        experiment_names=args.exp_names,
        smoothing_factor=args.smoothing_factor,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        captions=args.captions,
        aggregated_benchmarks=args.benchmarks,
    )


# Example usage:
# python3 recipe/o3/plot_v3/plot_partition_comparison.py \
#     --exp_names \
#         qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#     --captions "Qwen2.5-VL-7B-Instruct" \
#     --benchmarks vstar hrbench4k hrbench8k visualprobeasy visualprobmedium visualprobhard

if __name__ == "__main__":
    main()

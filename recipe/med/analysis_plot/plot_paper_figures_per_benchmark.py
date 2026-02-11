#!/usr/bin/env python3
"""
Plot per-benchmark absolute performance figures.
Each benchmark gets its own subplot in a 2x3 grid showing w/tool and w/o tool absolute performance.
S_tool is calculated and displayed in the title.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from plot_paper_figures import (
    PAPER_FIG_CONFIG,
    aggregate_benchmarks,
    get_sorted_benchmarks,
    load_experiment_with_baseline,
    setup_plot_style,
    smooth_curve,
)

# Benchmark display names mapping
BENCHMARK_DISPLAY_NAMES = {
    "vstar": "VStar",
    "hrbench4k": "HRBench(4k)",
    "hrbench8k": "HRBench(8k)",
    "visualprobeasy": "VisualProb(easy)",
    "visualprobmedium": "VisualProb(medium)",
    "visualprobhard": "VisualProb(hard)",
}


def plot_paper_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_performance.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark absolute performance.

    Creates a 2x3 grid where each subplot shows one benchmark's absolute performance:
    - w/tool performance (solid line)
    - w/o tool performance (dashed line)
    - S_tool displayed in title

    Args:
        experiment_name: Experiment name to plot
        benchmarks: List of 6 benchmark names
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        caption: Caption for the entire figure
    """
    if len(benchmarks) != 6:
        raise ValueError(f"Expected 6 benchmarks, got {len(benchmarks)}")

    setup_plot_style()

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(
        2, 3, figsize=(18, 12), dpi=PAPER_FIG_CONFIG["dpi"], facecolor="white"
    )
    fig.patch.set_facecolor("white")

    # Load experiment data
    df = load_experiment_with_baseline(experiment_name)
    if df is None:
        print(f"Failed to load experiment '{experiment_name}'")
        return

    selected_benchmarks = get_sorted_benchmarks(df)
    target_benchmarks = [b for b in benchmarks if b in selected_benchmarks]

    if not target_benchmarks:
        print(f"No matching benchmarks found for {experiment_name}")
        return

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot each benchmark
    for bench_idx, benchmark in enumerate(target_benchmarks):
        print(
            f"\nProcessing benchmark {bench_idx + 1}/{len(target_benchmarks)}: {benchmark}"
        )

        ax = axes[bench_idx]

        # Aggregate data for this single benchmark to calculate S_tool
        agg_data = aggregate_benchmarks(
            df, [benchmark], smoothing_factor, smoothing_method
        )

        if agg_data is None:
            ax.set_visible(False)
            continue

        steps = agg_data["steps"]
        w_tool_smooth = agg_data["w_tool_smooth"]
        wo_tool_smooth = agg_data["wo_tool_smooth"]

        # Calculate S_tool (without plotting the area chart)
        agg_data["w_tool_raw"]
        aggregated_wo_tool_raw = agg_data["wo_tool_raw"]

        diff_aggregated_raw = w_tool_smooth - wo_tool_smooth
        positive_area = np.trapezoid(np.maximum(diff_aggregated_raw, 0), steps)
        negative_area = np.trapezoid(np.minimum(diff_aggregated_raw, 0), steps)
        abs_base_area = np.trapezoid(np.abs(aggregated_wo_tool_raw), steps)
        abs_tool_area = positive_area + abs(negative_area)
        total_change = abs_base_area + abs_tool_area
        S_tool = abs_tool_area / total_change if total_change > 0 else 0

        # Get absolute scores for this benchmark
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        benchmark_data = benchmark_data.sort_values("step")

        if (
            "w_tool_score" in benchmark_data.columns
            and "wo_tool_score" in benchmark_data.columns
        ):
            w_tool_abs_raw = benchmark_data["w_tool_score"].values
            wo_tool_abs_raw = benchmark_data["wo_tool_score"].values

            # Smooth
            if smoothing_factor > 0 or smoothing_method != "none":
                w_tool_abs = smooth_curve(
                    w_tool_abs_raw, steps, smoothing_factor, smoothing_method
                )
                wo_tool_abs = smooth_curve(
                    wo_tool_abs_raw, steps, smoothing_factor, smoothing_method
                )
            else:
                w_tool_abs = w_tool_abs_raw.copy()
                wo_tool_abs = wo_tool_abs_raw.copy()

            # Dense interpolation for smooth plotting
            num_interp_points = max(200, len(steps) * 10)
            steps_dense = np.linspace(steps[0], steps[-1], num_interp_points)
            w_tool_abs_dense = np.interp(steps_dense, steps, w_tool_abs)
            wo_tool_abs_dense = np.interp(steps_dense, steps, wo_tool_abs)

            # Plot w/ tool
            ax.plot(
                steps_dense,
                w_tool_abs_dense,
                color="black",
                linestyle="-",
                linewidth=PAPER_FIG_CONFIG["linewidth"],
                label="$\\mathrm{Acc}_\\mathrm{w}$",
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                steps,
                w_tool_abs,
                color="black",
                marker="o",
                s=PAPER_FIG_CONFIG["scatter_size_large"],
                edgecolors="white",
                linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
                alpha=0.8,
                zorder=4,
            )

            # Plot w/o tool
            ax.plot(
                steps_dense,
                wo_tool_abs_dense,
                color="black",
                linestyle="--",
                linewidth=PAPER_FIG_CONFIG["linewidth"],
                label="$\\mathrm{Acc}_\\mathrm{wo}$",
                alpha=0.7,
                zorder=2,
            )
            ax.scatter(
                steps,
                wo_tool_abs,
                color="black",
                marker="s",
                s=PAPER_FIG_CONFIG["scatter_size_small"],
                edgecolors="white",
                linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
                alpha=0.7,
                zorder=3,
            )

        # Styling
        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        ax.set_title(
            f"{benchmark_display} ($S_{{\\text{{tool}}}}$={S_tool:.2f})",
            fontsize=14,
            fontweight="bold",
            pad=-100,
        )
        ax.set_xlabel(
            "Steps", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )
        ax.set_ylabel(
            "Accuracy", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )
        ax.grid(
            True,
            alpha=PAPER_FIG_CONFIG["grid_alpha"],
            linewidth=PAPER_FIG_CONFIG["grid_linewidth"],
        )
        ax.set_axisbelow(True)

        # Legend (only for first subplot)
        if bench_idx == 0:
            ax.legend(
                loc="best",
                fontsize=PAPER_FIG_CONFIG["legend_fontsize"],
                frameon=True,
                fancybox=True,
                shadow=True,
            )

        # Scale x-axis labels by 8
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

    # Add overall caption
    if caption:
        fig.suptitle(caption, fontsize=18, fontweight="bold", y=0.995)

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*not compatible with tight_layout.*"
        )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)
        if caption:
            plt.subplots_adjust(top=0.96)

    plt.savefig(output_path, dpi=PAPER_FIG_CONFIG["dpi"], bbox_inches="tight")
    print(f"\n{'='*80}")
    print(f"Saved figure to: {output_path}")
    print(f"{'='*80}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-benchmark absolute performance"
    )
    parser.add_argument("experiment_name", type=str, help="Experiment name to plot")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        required=True,
        help="List of 6 benchmarks to plot (required). Example: --benchmarks vstar hrbench4k hrbench8k visualprobeasy visualprobmedium visualprobhard",
    )
    parser.add_argument(
        "--caption", type=str, default=None, help="Caption for the entire figure"
    )
    parser.add_argument(
        "--output_dir", type=str, default="paper_figures", help="Output directory"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="per_benchmark_performance.pdf",
        help="Output filename",
    )
    parser.add_argument(
        "--smoothing_factor", type=float, default=0.99, help="Smoothing factor"
    )

    args = parser.parse_args()

    plot_paper_per_benchmark(
        experiment_name=args.experiment_name,
        benchmarks=args.benchmarks,
        smoothing_factor=args.smoothing_factor,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        caption=args.caption,
    )


if __name__ == "__main__":
    main()

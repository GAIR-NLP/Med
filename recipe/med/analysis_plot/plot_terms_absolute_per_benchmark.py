#!/usr/bin/env python3
"""
Plot per-benchmark term1-4 absolute values.
Each benchmark gets its own subplot in a 2x3 grid.
Exactly replicates the plotting logic from plot_term_absolute_values.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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

# Benchmark display names mapping
BENCHMARK_DISPLAY_NAMES = {
    "vstar": "VStar",
    "hrbench4k": "HRBench(4k)",
    "hrbench8k": "HRBench(8k)",
    "visualprobeasy": "VisualProb(easy)",
    "visualprobmedium": "VisualProb(medium)",
    "visualprobhard": "VisualProb(hard)",
}


def plot_terms_absolute_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_terms_absolute.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark term1-4 absolute values with stacked bars and curves.

    Creates a 2x3 grid where each subplot shows one benchmark's term decomposition
    (exactly replicating the plotting logic from plot_term_absolute_values).

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
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=PAPER_FIG_CONFIG["dpi"], facecolor="white")
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
        ax = axes[bench_idx]

        # Filter data for this benchmark
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            ax.set_visible(False)
            continue

        benchmark_data = benchmark_data.sort_values("step")
        common_steps = benchmark_data["step"].values

        # Check if term1-4 data exists
        if not all(col in benchmark_data.columns for col in ["term1", "term2", "term3", "term4"]):
            ax.set_visible(False)
            continue

        # Get term values (raw, no aggregation needed)
        aggregated_term1_raw = benchmark_data["term1"].values
        aggregated_term2_raw = benchmark_data["term2"].values
        aggregated_term3_raw = benchmark_data["term3"].values
        aggregated_term4_raw = benchmark_data["term4"].values

        # Get w_tool and wo_tool scores if available
        if "wo_tool_score" in benchmark_data.columns:
            aggregated_wo_tool_raw = benchmark_data["wo_tool_score"].values
        else:
            aggregated_wo_tool_raw = None

        if "w_tool_score" in benchmark_data.columns:
            aggregated_w_tool_raw = benchmark_data["w_tool_score"].values
        else:
            aggregated_w_tool_raw = None

        # Smooth the values
        if smoothing_factor > 0 or smoothing_method != "none":
            aggregated_term1 = smooth_curve(
                aggregated_term1_raw, common_steps, smoothing_factor, smoothing_method
            )
            aggregated_term2 = smooth_curve(
                aggregated_term2_raw, common_steps, smoothing_factor, smoothing_method
            )
            aggregated_term3 = smooth_curve(
                aggregated_term3_raw, common_steps, smoothing_factor, smoothing_method
            )
            aggregated_term4 = smooth_curve(
                aggregated_term4_raw, common_steps, smoothing_factor, smoothing_method
            )

            if aggregated_wo_tool_raw is not None:
                aggregated_wo_tool = smooth_curve(
                    aggregated_wo_tool_raw, common_steps, smoothing_factor, smoothing_method
                )
            else:
                aggregated_wo_tool = None

            if aggregated_w_tool_raw is not None:
                aggregated_w_tool = smooth_curve(
                    aggregated_w_tool_raw, common_steps, smoothing_factor, smoothing_method
                )
            else:
                aggregated_w_tool = None
        else:
            aggregated_term1 = aggregated_term1_raw.copy()
            aggregated_term2 = aggregated_term2_raw.copy()
            aggregated_term3 = aggregated_term3_raw.copy()
            aggregated_term4 = aggregated_term4_raw.copy()
            aggregated_wo_tool = (
                aggregated_wo_tool_raw.copy() if aggregated_wo_tool_raw is not None else None
            )
            aggregated_w_tool = (
                aggregated_w_tool_raw.copy() if aggregated_w_tool_raw is not None else None
            )

        # Calculate metrics
        total_gain = aggregated_term1 + aggregated_term2
        total_harm = aggregated_term3 + aggregated_term4

        if aggregated_w_tool is not None and aggregated_wo_tool is not None:
            acc_gain = aggregated_w_tool - aggregated_wo_tool
        else:
            acc_gain = None

        # Bar width and positioning
        step_interval = (common_steps[1] - common_steps[0]) if len(common_steps) > 1 else 1.0
        bar_width = step_interval * 0.4

        offset_left = -bar_width / 2
        offset_right = bar_width / 2

        # ========== Side-by-side stacked bars (exactly copied) ==========
        # Left group: T1 + T2 (Gains)
        ax.bar(
            common_steps + offset_left,
            aggregated_term1,
            width=bar_width,
            bottom=0,
            color="darkgreen",
            alpha=0.8,
            label="T1: Call Gain",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        ax.bar(
            common_steps + offset_left,
            aggregated_term2,
            width=bar_width,
            bottom=aggregated_term1,
            color="lightgreen",
            alpha=0.75,
            label="T2: Schema Gain",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # Right group: T3 + T4 (Harms)
        ax.bar(
            common_steps + offset_right,
            aggregated_term3,
            width=bar_width,
            bottom=0,
            color="red",
            alpha=0.8,
            label="T3: Call Harm",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        ax.bar(
            common_steps + offset_right,
            aggregated_term4,
            width=bar_width,
            bottom=aggregated_term3,
            color="lightcoral",
            alpha=0.75,
            label="T4: Schema Harm",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # ========== Plot curves (exactly copied) ==========
        # Gain curve
        ax.plot(
            common_steps,
            total_gain,
            color="darkgreen",
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6,
            markerfacecolor="lightgreen",
            markeredgecolor="darkgreen",
            markeredgewidth=1.5,
            label="Gain (T1+T2)",
            alpha=0.95,
            zorder=5,
        )

        # Harm curve
        ax.plot(
            common_steps,
            total_harm,
            color="darkred",
            linestyle="-",
            linewidth=2.5,
            marker="s",
            markersize=6,
            markerfacecolor="lightcoral",
            markeredgecolor="darkred",
            markeredgewidth=1.5,
            label="Harm (T3+T4)",
            alpha=0.95,
            zorder=5,
        )

        # Net gain curve
        if acc_gain is not None:
            ax.plot(
                common_steps,
                acc_gain,
                color="black",
                linestyle="--",
                linewidth=2.5,
                marker="D",
                markersize=6,
                markerfacecolor="gold",
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Net Gain (Gain - Harm)",
                alpha=0.95,
                zorder=6,
            )

        # Styling
        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        ax.set_title(benchmark_display, fontsize=14, fontweight="bold", pad=-100)
        ax.set_xlabel("Steps", fontsize=14, fontweight="medium")
        ax.set_ylabel("Probability", fontsize=14, fontweight="medium")

        # Add horizontal line at y=0
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.5, alpha=0.5, zorder=1)

        ax.grid(
            True, alpha=PAPER_FIG_CONFIG["grid_alpha"], linewidth=PAPER_FIG_CONFIG["grid_linewidth"]
        )
        ax.set_axisbelow(True)

        # Legend (only for first subplot)
        if bench_idx == 0:
            ax.legend(loc="best", fontsize=10, frameon=True, fancybox=True, shadow=True)

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
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
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
    parser = argparse.ArgumentParser(description="Plot per-benchmark term1-4 absolute values")
    parser.add_argument("experiment_name", type=str, help="Experiment name to plot")
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="List of 6 benchmarks to plot (default: perception benchmarks)",
    )
    parser.add_argument("--caption", type=str, default=None, help="Caption for the entire figure")
    parser.add_argument("--output_dir", type=str, default="paper_figures", help="Output directory")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="per_benchmark_terms_absolute.pdf",
        help="Output filename",
    )
    parser.add_argument("--smoothing_factor", type=float, default=0.99, help="Smoothing factor")

    args = parser.parse_args()

    benchmarks = args.benchmarks if args.benchmarks else PERCEPTION_BENCHMARKS

    plot_terms_absolute_per_benchmark(
        experiment_name=args.experiment_name,
        benchmarks=benchmarks,
        smoothing_factor=args.smoothing_factor,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        caption=args.caption,
    )


if __name__ == "__main__":
    main()

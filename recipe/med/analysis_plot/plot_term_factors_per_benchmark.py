#!/usr/bin/env python3
"""
Plot per-benchmark term1 factor decomposition (Mass × Policy × Quality).
Each benchmark gets its own subplot in a 2x3 grid showing term1's three factors with dual y-axes.
Exactly replicates the plotting logic from plot_term_factor_decomposition (term1 column only).
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


def plot_term_factors_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_term_factors.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark term1 factor decomposition with dual y-axes.

    Creates a 2x3 grid where each subplot shows one benchmark's term1 decomposition:
    - Left y-axis: term1 value (thick line)
    - Right y-axis: three factors (thin lines, 0-1 scale)

    Exactly replicates the plotting logic from plot_term_factor_decomposition.

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

    # Required columns for term1 factor decomposition
    required_cols = ["term1", "p_fail", "p_call_fail", "p_acc_call_fail"]

    # Plot each benchmark
    for bench_idx, benchmark in enumerate(target_benchmarks):
        ax_left = axes[bench_idx]
        ax_right = ax_left.twinx()  # Create right y-axis

        # Filter data for this benchmark
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            ax_left.set_visible(False)
            ax_right.set_visible(False)
            continue

        benchmark_data = benchmark_data.sort_values("step")
        common_steps = benchmark_data["step"].values

        # Check if all required columns exist
        if not all(col in benchmark_data.columns for col in required_cols):
            ax_left.set_visible(False)
            ax_right.set_visible(False)
            continue

        # Get data
        aggregated_data = {}
        for col in required_cols:
            values_raw = benchmark_data[col].values
            # Smooth
            if smoothing_factor > 0 or smoothing_method != "none":
                aggregated_data[col] = smooth_curve(
                    values_raw, common_steps, smoothing_factor, smoothing_method
                )
            else:
                aggregated_data[col] = values_raw.copy()

        # Term1 configuration (exactly copied from original)
        term_key = "term1"
        mass_key = "p_fail"
        policy_key = "p_call_fail"
        quality_key = "p_acc_call_fail"
        title = "Term 1: Call Gain"
        mass_label = "$P(\\mathcal{D}_{\\text{fail}})$"
        policy_label = "$P(c \\mid \\mathcal{D}_{\\text{fail}})$"
        quality_label = "$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}})$"
        term_color = "darkgreen"
        term_label = "Call Gain"

        # ========== Left axis: Term (exactly copied) ==========
        line_term = ax_left.plot(
            common_steps,
            aggregated_data[term_key],
            color=term_color,
            linestyle="-",
            linewidth=3.0,
            marker="o",
            markersize=8,
            label=term_label,
            alpha=0.95,
            zorder=5,
        )[0]

        # ========== Right axis: Three factors (exactly copied) ==========
        line_mass = ax_right.plot(
            common_steps,
            aggregated_data[mass_key],
            color="gray",
            linestyle="-",
            linewidth=1.5,
            label=mass_label,
            alpha=0.7,
            zorder=3,
        )[0]

        line_policy = ax_right.plot(
            common_steps,
            aggregated_data[policy_key],
            color="steelblue",
            linestyle="-",
            linewidth=1.5,
            label=policy_label,
            alpha=0.7,
            zorder=3,
        )[0]

        line_quality = ax_right.plot(
            common_steps,
            aggregated_data[quality_key],
            color="darkorange",
            linestyle="-",
            linewidth=1.5,
            label=quality_label,
            alpha=0.7,
            zorder=3,
        )[0]

        # ========== Axis labels and formatting (exactly copied) ==========
        ax_left.set_xlabel("Steps", fontsize=14, fontweight="medium")
        ax_left.set_ylabel("Probability", fontsize=14, fontweight="medium", color=term_color)
        ax_right.set_ylabel("Factor Probability", fontsize=14, fontweight="medium")

        ax_left.tick_params(axis="y", labelcolor=term_color)
        ax_right.set_ylim(0, 1.0)  # Fixed 0-1 for factors

        # Title
        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        ax_left.set_title(benchmark_display, fontsize=14, fontweight="bold", pad=10)

        # Grid
        ax_left.grid(True, alpha=0.3, linewidth=0.5)
        ax_left.set_axisbelow(True)

        # Legend: combine lines from both axes (only for first subplot)
        if bench_idx == 0:
            lines = [line_term, line_mass, line_policy, line_quality]
            labels = [l.get_label() for l in lines]
            ax_left.legend(
                lines, labels, loc="best", fontsize=9, frameon=True, fancybox=True, shadow=True
            )

        # Scale x-axis by 8
        from matplotlib.ticker import FuncFormatter

        ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

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
    parser = argparse.ArgumentParser(description="Plot per-benchmark term1 factor decomposition")
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
        default="per_benchmark_term_factors.pdf",
        help="Output filename",
    )
    parser.add_argument("--smoothing_factor", type=float, default=0.99, help="Smoothing factor")

    args = parser.parse_args()

    benchmarks = args.benchmarks if args.benchmarks else PERCEPTION_BENCHMARKS

    plot_term_factors_per_benchmark(
        experiment_name=args.experiment_name,
        benchmarks=benchmarks,
        smoothing_factor=args.smoothing_factor,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        caption=args.caption,
    )


if __name__ == "__main__":
    main()

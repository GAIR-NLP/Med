#!/usr/bin/env python3
"""
Plot per-benchmark figures for Vision Tool-Use RL experiments.
Supports three types of per-benchmark analysis:
- MEASURE: Absolute performance (w/tool vs w/o tool)
- EXPLAIN: Term decomposition (term1-4)
- DIAGNOSE: Factor decomposition (Mass × Policy × Quality)
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from plot_paper_figures import (  # noqa: E402
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


def plot_measure_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_measure.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark absolute performance (MEASURE).

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


def plot_explain_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_explain.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark term decomposition (EXPLAIN).

    Creates a 2x3 grid showing term1-4 absolute values with stacked bars:
    - Left bars: T1 (Call Gain) + T2 (Schema Gain)
    - Right bars: T3 (Call Harm) + T4 (Schema Harm)
    - Curves: Total Gain, Total Harm, Net Gain

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
        2, 3, figsize=(20, 12), dpi=PAPER_FIG_CONFIG["dpi"], facecolor="white"
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
        ax = axes[bench_idx]

        # Filter data for this benchmark
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            ax.set_visible(False)
            continue

        benchmark_data = benchmark_data.sort_values("step")
        common_steps = benchmark_data["step"].values

        # Check if term1-4 data exists
        if not all(
            col in benchmark_data.columns
            for col in ["term1", "term2", "term3", "term4"]
        ):
            ax.set_visible(False)
            continue

        # Get term values
        aggregated_term1_raw = benchmark_data["term1"].values
        aggregated_term2_raw = benchmark_data["term2"].values
        aggregated_term3_raw = benchmark_data["term3"].values
        aggregated_term4_raw = benchmark_data["term4"].values

        # Get w_tool and wo_tool scores if available
        aggregated_wo_tool_raw = (
            benchmark_data["wo_tool_score"].values
            if "wo_tool_score" in benchmark_data.columns
            else None
        )
        aggregated_w_tool_raw = (
            benchmark_data["w_tool_score"].values
            if "w_tool_score" in benchmark_data.columns
            else None
        )

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
            aggregated_wo_tool = (
                smooth_curve(
                    aggregated_wo_tool_raw,
                    common_steps,
                    smoothing_factor,
                    smoothing_method,
                )
                if aggregated_wo_tool_raw is not None
                else None
            )
            aggregated_w_tool = (
                smooth_curve(
                    aggregated_w_tool_raw,
                    common_steps,
                    smoothing_factor,
                    smoothing_method,
                )
                if aggregated_w_tool_raw is not None
                else None
            )
        else:
            aggregated_term1 = aggregated_term1_raw.copy()
            aggregated_term2 = aggregated_term2_raw.copy()
            aggregated_term3 = aggregated_term3_raw.copy()
            aggregated_term4 = aggregated_term4_raw.copy()
            aggregated_wo_tool = (
                aggregated_wo_tool_raw.copy()
                if aggregated_wo_tool_raw is not None
                else None
            )
            aggregated_w_tool = (
                aggregated_w_tool_raw.copy()
                if aggregated_w_tool_raw is not None
                else None
            )

        # Calculate metrics
        total_gain = aggregated_term1 + aggregated_term2
        total_harm = aggregated_term3 + aggregated_term4
        acc_gain = (
            aggregated_w_tool - aggregated_wo_tool
            if aggregated_w_tool is not None and aggregated_wo_tool is not None
            else None
        )

        # Bar width and positioning
        step_interval = (
            (common_steps[1] - common_steps[0]) if len(common_steps) > 1 else 1.0
        )
        bar_width = step_interval * 0.4
        offset_left = -bar_width / 2
        offset_right = bar_width / 2

        # Stacked bars: Left group (T1 + T2)
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

        # Stacked bars: Right group (T3 + T4)
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

        # Plot curves
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
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.5, alpha=0.5, zorder=1)
        ax.grid(
            True,
            alpha=PAPER_FIG_CONFIG["grid_alpha"],
            linewidth=PAPER_FIG_CONFIG["grid_linewidth"],
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


def plot_diagnose_per_benchmark(
    experiment_name: str,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "per_benchmark_diagnose.pdf",
    caption: str | None = None,
):
    """Plot per-benchmark factor decomposition (DIAGNOSE).

    Creates a 2x3 grid showing term1 decomposition with dual y-axes:
    - Left y-axis: term1 value (thick line)
    - Right y-axis: three factors (thin lines, 0-1 scale)
      - Mass: P(D_fail)
      - Policy: P(c | D_fail)
      - Quality: P(√ | c, D_fail)

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
        2, 3, figsize=(20, 12), dpi=PAPER_FIG_CONFIG["dpi"], facecolor="white"
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

        # Get and smooth data
        aggregated_data = {}
        for col in required_cols:
            values_raw = benchmark_data[col].values
            if smoothing_factor > 0 or smoothing_method != "none":
                aggregated_data[col] = smooth_curve(
                    values_raw, common_steps, smoothing_factor, smoothing_method
                )
            else:
                aggregated_data[col] = values_raw.copy()

        # Term1 configuration
        term_key = "term1"
        mass_key = "p_fail"
        policy_key = "p_call_fail"
        quality_key = "p_acc_call_fail"
        term_label = "Call Gain"
        mass_label = "$P(\\mathcal{D}_{\\text{fail}})$"
        policy_label = "$P(c \\mid \\mathcal{D}_{\\text{fail}})$"
        quality_label = "$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}})$"
        term_color = "darkgreen"

        # Left axis: Term
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

        # Right axis: Three factors
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

        # Styling
        ax_left.set_xlabel("Steps", fontsize=14, fontweight="medium")
        ax_left.set_ylabel(
            "Probability", fontsize=14, fontweight="medium", color=term_color
        )
        ax_right.set_ylabel("Factor Probability", fontsize=14, fontweight="medium")
        ax_left.tick_params(axis="y", labelcolor=term_color)
        ax_right.set_ylim(0, 1.0)

        # Title
        benchmark_display = BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)
        ax_left.set_title(benchmark_display, fontsize=14, fontweight="bold", pad=10)

        # Grid
        ax_left.grid(True, alpha=0.3, linewidth=0.5)
        ax_left.set_axisbelow(True)

        # Legend (only for first subplot)
        if bench_idx == 0:
            lines = [line_term, line_mass, line_policy, line_quality]
            labels = [line.get_label() for line in lines]
            ax_left.legend(
                lines,
                labels,
                loc="best",
                fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True,
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
        description="Plot per-benchmark figures (MEASURE/EXPLAIN/DIAGNOSE)"
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
        default="per_benchmark",
        help="Base output filename (default: per_benchmark). Will generate {filename}_measure.pdf, {filename}_explain.pdf, {filename}_diagnose.pdf",
    )
    parser.add_argument(
        "--smoothing_factor", type=float, default=0.99, help="Smoothing factor"
    )
    parser.add_argument(
        "--smoothing_method",
        type=str,
        default="time_weighted_ema",
        choices=["none", "savgol", "ema", "time_weighted_ema"],
        help="Smoothing method (default: time_weighted_ema)",
    )
    parser.add_argument(
        "--figure_type",
        type=str,
        default="all",
        choices=["all", "measure", "explain", "diagnose"],
        help="Type of figure to generate (default: all)",
    )

    args = parser.parse_args()

    # Generate requested figure(s)
    if args.figure_type in ["all", "measure"]:
        print("\n" + "=" * 80)
        print("Generating MEASURE figure (per-benchmark)...")
        print("=" * 80)
        plot_measure_per_benchmark(
            experiment_name=args.experiment_name,
            benchmarks=args.benchmarks,
            smoothing_factor=args.smoothing_factor,
            smoothing_method=args.smoothing_method,
            output_dir=args.output_dir,
            output_filename=f"{args.output_filename}_measure.pdf",
            caption=args.caption,
        )

    if args.figure_type in ["all", "explain"]:
        print("\n" + "=" * 80)
        print("Generating EXPLAIN figure (per-benchmark)...")
        print("=" * 80)
        plot_explain_per_benchmark(
            experiment_name=args.experiment_name,
            benchmarks=args.benchmarks,
            smoothing_factor=args.smoothing_factor,
            smoothing_method=args.smoothing_method,
            output_dir=args.output_dir,
            output_filename=f"{args.output_filename}_explain.pdf",
            caption=args.caption,
        )

    if args.figure_type in ["all", "diagnose"]:
        print("\n" + "=" * 80)
        print("Generating DIAGNOSE figure (per-benchmark)...")
        print("=" * 80)
        plot_diagnose_per_benchmark(
            experiment_name=args.experiment_name,
            benchmarks=args.benchmarks,
            smoothing_factor=args.smoothing_factor,
            smoothing_method=args.smoothing_method,
            output_dir=args.output_dir,
            output_filename=f"{args.output_filename}_diagnose.pdf",
            caption=args.caption,
        )


if __name__ == "__main__":
    main()

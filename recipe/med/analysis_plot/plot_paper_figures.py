#!/usr/bin/env python3
"""
Plot paper figures for Vision Tool-Use RL experiments.
Creates multi-experiment comparison figures with aggregated performance metrics.
"""

# Import utility functions from calculate_and_plot_area.py
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from calculate_and_plot_area import (
    get_sorted_benchmarks,
    load_experiment_data,
    smooth_curve,
)

# Model baseline performance mapping
# Maps model identifier prefix to baseline model path
MODEL_BASELINE_MAPPING = {
    "qwen25vl_instruct": "baseline/Qwen2.5-VL-7B-Instruct",
    "qwen3vl_instruct": "baseline/Qwen3-VL-8B-Instruct",
}

# Base directory for evaluation results
RESULTS_DIR = Path("evals")

# ==============================================================================
# PAPER FIGURE STYLE CONFIGURATION
# ==============================================================================
# Global configuration for paper figure styling.
# Modify these values to adjust all figure sizes, fonts, colors, etc.
# This makes it easy to maintain consistent styling across all paper figures.
PAPER_FIG_CONFIG = {
    # Figure settings
    "fig_width": 13,  # Width per figure
    "fig_height_per_row": 5.5,  # Height per experiment row
    "dpi": 400,
    # Font sizes
    "title_fontsize": 14,
    "label_fontsize": 18,
    "legend_fontsize": 11,
    "tick_fontsize": 10,
    "row_label_fontsize": 16,
    "stool_fontsize": 14,
    "bar_label_fontsize": 12,
    "colorbar_tick_fontsize": 14,
    "colorbar_label_fontsize": 14,
    # Line and marker styles
    "linewidth": 2.5,
    "scatter_size_large": 40,
    "scatter_size_small": 30,
    "scatter_edgewidth": 1.5,
    "grid_linewidth": 0.8,
    "grid_alpha": 0.4,
    # Progress bar settings
    "bar_height": 0.018,
    "bar_width": 0.3,
    "bar_y_position": 0.96,
    "bar_x_start": 0.35,
    "bar_border_linewidth": 1.5,
    "bar_alpha": 0.6,
    # Title padding
    "title_pad_with_bar": 20,  # Padding when progress bar is present
    "title_pad_normal": 20,
    # Colorbar settings
    "colorbar_width": "3%",
    "colorbar_height": "90%",
    "colorbar_bbox_x": 0.01,
    # Colors
    "title_color": "black",
    "positive_color": "green",
    "negative_color": "red",
    "base_hatch_color": "gray",
    # Row label position offset
    "row_label_y_offset": -0.02,
}


def identify_model_from_exp_name(exp_name: str) -> str | None:
    """Identify which model an experiment belongs to based on its name.

    Args:
        exp_name: Experiment name (e.g., "qwen25vl_instruct_50_25/...")

    Returns:
        Model identifier prefix (e.g., "qwen25vl_instruct") or None if not found
    """
    # Extract first part before any path separator
    exp_base = exp_name.split("/")[0] if "/" in exp_name else exp_name

    # Check which model prefix matches
    for model_prefix in MODEL_BASELINE_MAPPING.keys():
        if exp_base.startswith(model_prefix):
            return model_prefix

    print(f"Warning: Could not identify model for experiment '{exp_name}'")
    return None


def load_baseline_performance(model_path: str) -> pd.DataFrame | None:
    """Load baseline performance for a given model.

    Args:
        model_path: Path to baseline model (e.g., "verl_model/Qwen2.5-VL-7B-Instruct")

    Returns:
        DataFrame with baseline performance or None if not found
    """
    baseline_dir = RESULTS_DIR / model_path

    if not baseline_dir.exists():
        print(f"Warning: Baseline directory not found: {baseline_dir}")
        return None

    # Look for the CSV file with pattern {model_name}_results.csv
    model_name = Path(model_path).name
    csv_file = baseline_dir / f"{model_name}_results.csv"

    if not csv_file.exists():
        print(f"Warning: Baseline CSV not found: {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded baseline performance from: {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading baseline CSV {csv_file}: {e}")
        return None


def merge_baseline_performance(exp_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline performance (step=0) into experiment DataFrame.

    Inserts baseline performance as step=0 rows if they don't exist.

    Args:
        exp_df: Experiment DataFrame
        baseline_df: Baseline performance DataFrame

    Returns:
        DataFrame with baseline rows inserted at step=0
    """
    exp_df = exp_df.copy()

    # Get minimum step in baseline (should be 0 or close to it)
    min_baseline_step = baseline_df["step"].min() if "step" in baseline_df.columns else 0

    # Filter baseline to get only the earliest step
    baseline_step0 = baseline_df[baseline_df["step"] == min_baseline_step].copy()

    # Set step to 0 for baseline data
    baseline_step0["step"] = 0

    # Check if exp_df already has step=0 data
    has_step0 = (exp_df["step"] == 0).any() if "step" in exp_df.columns else False

    if has_step0:
        print("  Warning: Experiment already has step=0 data, skipping baseline insertion")
        return exp_df

    # Concatenate baseline at the beginning
    combined_df = pd.concat([baseline_step0, exp_df], ignore_index=True)

    # Sort by step and benchmark
    combined_df = combined_df.sort_values(["benchmark", "step"]).reset_index(drop=True)

    print(f"  Added {len(baseline_step0)} baseline rows at step=0")

    return combined_df


def load_experiment_with_baseline(exp_name: str) -> pd.DataFrame | None:
    """Load experiment data and merge with baseline performance.

    Args:
        exp_name: Experiment name

    Returns:
        DataFrame with experiment data and baseline columns, or None if failed
    """
    # Load experiment data
    experiment_data = load_experiment_data(exp_name)
    if exp_name not in experiment_data:
        print(f"Experiment '{exp_name}' not found!")
        return None

    exp_df = experiment_data[exp_name]

    # Identify model and load baseline
    model_prefix = identify_model_from_exp_name(exp_name)
    if model_prefix is None:
        print(f"Using experiment data without baseline for '{exp_name}'")
        return exp_df

    model_path = MODEL_BASELINE_MAPPING[model_prefix]
    baseline_df = load_baseline_performance(model_path)

    if baseline_df is None:
        print(f"Using experiment data without baseline for '{exp_name}'")
        return exp_df

    # Merge baseline into experiment
    exp_df = merge_baseline_performance(exp_df, baseline_df)
    print(f"Successfully merged baseline for '{exp_name}' (model: {model_prefix})")

    return exp_df


def setup_plot_style():
    """Configure matplotlib style for paper figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = PAPER_FIG_CONFIG["dpi"]
    plt.rcParams["savefig.dpi"] = PAPER_FIG_CONFIG["dpi"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = PAPER_FIG_CONFIG["label_fontsize"]
    plt.rcParams["axes.titlesize"] = PAPER_FIG_CONFIG["title_fontsize"]
    plt.rcParams["axes.labelsize"] = PAPER_FIG_CONFIG["label_fontsize"]
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = PAPER_FIG_CONFIG["grid_alpha"]


def aggregate_benchmarks(
    df: pd.DataFrame,
    benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
):
    """Aggregate performance across benchmarks.

    Args:
        df: DataFrame with experiment data
        benchmarks: List of benchmark names to aggregate
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method to use

    Returns:
        Dictionary with aggregated data
    """
    all_normalized_w_tool = []
    all_normalized_wo_tool = []
    all_call_rates = []
    common_steps = None

    for benchmark in benchmarks:
        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            continue

        benchmark_data = benchmark_data.sort_values("step")
        steps = benchmark_data["step"].values

        if common_steps is None:
            common_steps = steps.copy()

        if (
            "w_tool_score" not in benchmark_data.columns
            or "wo_tool_score" not in benchmark_data.columns
        ):
            continue

        # Use raw data (no smoothing at benchmark level)
        w_tool_values_raw = benchmark_data["w_tool_score"].values
        wo_tool_values_raw = benchmark_data["wo_tool_score"].values

        # Method 1: First calculate drift (f_w and f_wo), then normalize
        # This directly corresponds to the paper definition

        # Step 1: Calculate drift relative to start
        w_tool_drift = w_tool_values_raw - w_tool_values_raw[0]
        wo_tool_drift = wo_tool_values_raw - wo_tool_values_raw[0]

        # Step 2: Find max absolute drift for normalization
        # Use the SAME scale for both w and wo to preserve gap
        max_abs_w = np.max(np.abs(w_tool_drift))
        max_abs_wo = np.max(np.abs(wo_tool_drift))
        max_abs = max(max_abs_w, max_abs_wo)

        # Step 3: Normalize drifts to [-1, 1] range
        if max_abs > 0:
            w_tool_normalized = w_tool_drift / max_abs
            wo_tool_normalized = wo_tool_drift / max_abs
        else:
            w_tool_normalized = w_tool_drift
            wo_tool_normalized = wo_tool_drift

        # Store normalized drift data
        all_normalized_w_tool.append(w_tool_normalized)
        all_normalized_wo_tool.append(wo_tool_normalized)

        if "call_num" in benchmark_data.columns and "w_tool_total" in benchmark_data.columns:
            call_num = benchmark_data["call_num"].values
            w_tool_total = benchmark_data["w_tool_total"].values
            call_rate = call_num / w_tool_total
        else:
            call_rate = np.zeros_like(steps)

        all_call_rates.append(call_rate)

    if len(all_normalized_w_tool) == 0:
        return None

    # Aggregate normalized raw data
    aggregated_w_tool_raw = np.mean(all_normalized_w_tool, axis=0)
    aggregated_wo_tool_raw = np.mean(all_normalized_wo_tool, axis=0)
    aggregated_call_rate = (
        np.mean(all_call_rates, axis=0) if len(all_call_rates) > 0 else np.zeros_like(common_steps)
    )

    # Smooth the aggregated data for plotting
    if smoothing_factor > 0 or smoothing_method != "none":
        aggregated_w_tool = smooth_curve(
            aggregated_w_tool_raw, common_steps, smoothing_factor, smoothing_method
        )
        aggregated_wo_tool = smooth_curve(
            aggregated_wo_tool_raw, common_steps, smoothing_factor, smoothing_method
        )
    else:
        aggregated_w_tool = aggregated_w_tool_raw.copy()
        aggregated_wo_tool = aggregated_wo_tool_raw.copy()

    return {
        "steps": common_steps,
        "w_tool_raw": aggregated_w_tool_raw,
        "wo_tool_raw": aggregated_wo_tool_raw,
        "w_tool_smooth": aggregated_w_tool,
        "wo_tool_smooth": aggregated_wo_tool,
        "call_rate": aggregated_call_rate,
    }


def plot_explain_figure(
    experiment_names: list[str],
    aggregated_benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "term_absolute_values.pdf",
    captions: list[str] | None = None,
):
    """Plot term1-4 ABSOLUTE values with side-by-side Gain/Harm stacks.

    For each checkpoint, shows two stacked bar groups meeting at the checkpoint:
    - Left group (T1+T2): Gain components, right edge aligns with checkpoint
      - T1 (bottom, dark green): Call Gain
      - T2 (top, light green): Schema Gain
    - Right group (T3+T4): Harm components, left edge aligns with checkpoint
      - T3 (bottom, red): Call Harm
      - T4 (top, light coral): Schema Harm

    The two groups seamlessly connect at the checkpoint with no gap.

    Also plots three curves:
    - Gain Curve (green): traces T1+T2 top (total positive contribution)
    - Harm Curve (red): traces T3+T4 top (total negative contribution, shown positive)
    - Net Gain (black dashed): Gain - Harm = Acc_w - Acc_wo

    Relationship: Net Gain = Gain Curve - Harm Curve

    Visual semantics:
    - Left bars (green): positive contributions
    - Right bars (red): negative contributions (costs)
    - Vertical distance between Gain and Harm curves = Net accuracy gain

    Args:
        experiment_names: List of experiment names to plot
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        captions: List of captions for each experiment
        aggregated_benchmarks: List of benchmark names to aggregate
    """
    setup_plot_style()

    n_experiments = len(experiment_names)
    fig, axes = plt.subplots(
        n_experiments,
        1,
        figsize=(8, 6 * n_experiments),
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

        # Filter selected benchmarks based on aggregated_benchmarks
        target_benchmarks = [b for b in selected_benchmarks if b in aggregated_benchmarks]

        if not target_benchmarks:
            print(f"No matching benchmarks found for {exp_name}")
            continue

        # Aggregate term1-4 across benchmarks
        all_term1 = []
        all_term2 = []
        all_term3 = []
        all_term4 = []
        all_wo_tool = []
        all_w_tool = []
        common_steps = None

        for benchmark in target_benchmarks:
            benchmark_data = df[df["benchmark"] == benchmark].copy()
            if benchmark_data.empty:
                continue

            benchmark_data = benchmark_data.sort_values("step")
            steps = benchmark_data["step"].values

            if common_steps is None:
                common_steps = steps.copy()

            # Check if term1-4 data exists
            if all(col in benchmark_data.columns for col in ["term1", "term2", "term3", "term4"]):
                # Use raw values (no normalization) - preserve physical meaning
                all_term1.append(benchmark_data["term1"].values)
                all_term2.append(benchmark_data["term2"].values)
                all_term3.append(benchmark_data["term3"].values)
                all_term4.append(benchmark_data["term4"].values)

                # Also read wo_tool and w_tool scores (no normalization)
                if "wo_tool_score" in benchmark_data.columns:
                    all_wo_tool.append(benchmark_data["wo_tool_score"].values)
                if "w_tool_score" in benchmark_data.columns:
                    all_w_tool.append(benchmark_data["w_tool_score"].values)

        if len(all_term1) == 0:
            print(f"No term1-4 data found for {exp_name}")
            continue

        # Aggregate (mean across benchmarks)
        aggregated_term1_raw = np.mean(all_term1, axis=0)
        aggregated_term2_raw = np.mean(all_term2, axis=0)
        aggregated_term3_raw = np.mean(all_term3, axis=0)
        aggregated_term4_raw = np.mean(all_term4, axis=0)

        if len(all_wo_tool) > 0:
            aggregated_wo_tool_raw = np.mean(all_wo_tool, axis=0)
        else:
            aggregated_wo_tool_raw = None

        if len(all_w_tool) > 0:
            aggregated_w_tool_raw = np.mean(all_w_tool, axis=0)
        else:
            aggregated_w_tool_raw = None

        # Smooth the original values first
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

        # Get axis for this experiment
        ax = axes[exp_idx]

        # Calculate metrics
        # Gain = term1 + term2 (total positive contribution)
        # Harm = term3 + term4 (total negative contribution)
        # Acc_gain = Gain - Harm
        total_gain = aggregated_term1 + aggregated_term2
        total_harm = aggregated_term3 + aggregated_term4

        if aggregated_w_tool is not None and aggregated_wo_tool is not None:
            acc_gain = aggregated_w_tool - aggregated_wo_tool
        else:
            acc_gain = None

        # Bar width and positioning for side-by-side stacked bars
        step_interval = (common_steps[1] - common_steps[0]) if len(common_steps) > 1 else 1.0
        bar_width = step_interval * 0.4  # Width of each stacked bar group

        # T1+T2: right edge aligns with checkpoint (center = checkpoint - bar_width/2)
        offset_left = -bar_width / 2

        # T3+T4: left edge aligns with checkpoint (center = checkpoint + bar_width/2)
        offset_right = bar_width / 2

        # ========== Side-by-side stacked bars ==========
        # Left group (T1+T2): Gain components, right edge aligns with checkpoint
        # Right group (T3+T4): Harm components (now positive values), left edge aligns with checkpoint
        # The two groups meet at the checkpoint with no gap

        # Left group: T1 + T2 (Gains)
        # term1: Call Gain (dark green) - bottom layer
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

        # term2: Schema Gain (light green) - stacked on T1
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

        # Right group: T3 + T4 (Harms - now POSITIVE upward stacking)
        # term3: Call Harm (red) - bottom layer (positive value, upward)
        ax.bar(
            common_steps + offset_right,
            aggregated_term3,  # Positive value now
            width=bar_width,
            bottom=0,
            color="red",
            alpha=0.8,
            label="T3: Call Harm",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # term4: Schema Harm (light coral) - stacked on T3 (positive value, upward)
        ax.bar(
            common_steps + offset_right,
            aggregated_term4,  # Positive value now
            width=bar_width,
            bottom=aggregated_term3,
            color="lightcoral",
            alpha=0.75,
            label="T4: Schema Harm",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

        # ========== Plot curves ==========

        # Gain curve: traces the top of T1+T2 (total positive contribution)
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

        # Harm curve: traces the top of T3+T4 (total harm, shown as positive height)
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

        # Net gain curve (Gain - Harm = Acc_w - Acc_wo)
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

        # Title (only for first subplot)
        if exp_idx == 0:
            ax.set_title(
                "($\\mathrm{Acc}_\\mathrm{w} - \\mathrm{Acc}_\\mathrm{wo}$) Decomposition",
                fontsize=PAPER_FIG_CONFIG["title_fontsize"],
                fontweight="bold",
                pad=20,
            )
        ax.set_xlabel("Steps", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium")
        ax.set_ylabel(
            "Probability", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )

        # Add a horizontal line at y=0
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.5, alpha=0.5, zorder=1)

        ax.grid(
            True, alpha=PAPER_FIG_CONFIG["grid_alpha"], linewidth=PAPER_FIG_CONFIG["grid_linewidth"]
        )
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=9.5, frameon=True, fancybox=True, shadow=True)

        # Scale x-axis labels by 8
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # Row label (a), (b), (c), ... - centered below each row
        if captions and exp_idx < len(captions):
            row_label = captions[exp_idx]
        else:
            row_label = exp_name

        label = chr(ord("a") + exp_idx)
        fig.text(
            0.55,  # Horizontal center
            1.0
            - (exp_idx + 1) / n_experiments
            - PAPER_FIG_CONFIG["row_label_y_offset"],  # Below this row
            f"({label}) {row_label}",
            fontsize=PAPER_FIG_CONFIG["row_label_fontsize"],
            fontweight="bold",
            ha="center",
            va="top",
        )

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)  # Adjust vertical spacing for row labels

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / output_filename

    fig.savefig(
        filepath,
        dpi=PAPER_FIG_CONFIG["dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )

    print(f"\nFigure saved to: {filepath}")
    plt.close(fig)
    return filepath


def plot_diagnose_figure(
    experiment_names: list[str],
    aggregated_benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "figures",
    output_filename: str = "term_factor_decomposition.pdf",
    captions: list[str] | None = None,
):
    """Plot term1-4 factor decomposition with dual y-axes (Mass × Policy × Quality).

    Creates an Nx4 grid where each row is one experiment and each column is one term:
    - Column 0: Term1 (Call Gain) = p_fail × p_call_fail × p_acc_call_fail
    - Column 1: Term2 (Schema Gain) = p_fail × p_nocall_fail × p_acc_nocall_fail
    - Column 2: Term3 (Call Harm) = p_succ × p_call_succ × p_err_call_succ
    - Column 3: Term4 (Schema Harm) = p_succ × p_nocall_succ × p_err_nocall_succ

    Each subplot has:
    - Left y-axis: Term value (auto-scaled, thick line with markers)
    - Right y-axis: Three factors (0-1 scale, thin lines)

    Args:
        experiment_names: List of experiment names to plot
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        captions: List of captions for each experiment
        aggregated_benchmarks: List of benchmark names to aggregate
    """
    setup_plot_style()

    n_experiments = len(experiment_names)

    # Create Nx4 figure for all experiments
    fig, axes = plt.subplots(
        n_experiments,
        4,
        figsize=(20, 5 * n_experiments),
        dpi=PAPER_FIG_CONFIG["dpi"],
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    # Ensure axes is 2D
    if n_experiments == 1:
        axes = axes.reshape(1, -1)

    # Process each experiment
    for exp_idx, exp_name in enumerate(experiment_names):
        print(f"\nProcessing experiment {exp_idx + 1}/{n_experiments}: {exp_name}")

        # Load experiment data
        df = load_experiment_with_baseline(exp_name)
        if df is None:
            print(f"Failed to load experiment '{exp_name}'")
            continue
        selected_benchmarks = get_sorted_benchmarks(df)

        # Filter selected benchmarks based on aggregated_benchmarks
        target_benchmarks = [b for b in selected_benchmarks if b in aggregated_benchmarks]

        if not target_benchmarks:
            print(f"No matching benchmarks found for {exp_name}")
            continue

        # Aggregate data across benchmarks
        required_cols = [
            "term1",
            "term2",
            "term3",
            "term4",
            "p_fail",
            "p_succ",
            "p_call_fail",
            "p_nocall_fail",
            "p_acc_call_fail",
            "p_acc_nocall_fail",
            "p_call_succ",
            "p_nocall_succ",
            "p_err_call_succ",
            "p_err_nocall_succ",
        ]

        aggregated_data = {}
        common_steps = None

        for benchmark in target_benchmarks:
            benchmark_data = df[df["benchmark"] == benchmark].copy()
            if benchmark_data.empty:
                continue

            benchmark_data = benchmark_data.sort_values("step")
            steps = benchmark_data["step"].values

            if common_steps is None:
                common_steps = steps.copy()

            # Check if all required columns exist
            if all(col in benchmark_data.columns for col in required_cols):
                for col in required_cols:
                    if col not in aggregated_data:
                        aggregated_data[col] = []
                    aggregated_data[col].append(benchmark_data[col].values)

        if len(aggregated_data) == 0:
            print(f"No factor data found for {exp_name}")
            continue

        # Average across benchmarks
        for key in aggregated_data:
            aggregated_data[key] = np.mean(aggregated_data[key], axis=0)

        # Smooth all curves
        if smoothing_factor > 0 or smoothing_method != "none":
            for key in aggregated_data:
                aggregated_data[key] = smooth_curve(
                    aggregated_data[key], common_steps, smoothing_factor, smoothing_method
                )

        # Subplot configurations: (col, term_key, factors, labels)
        subplot_configs = [
            # (col, term, mass, policy, quality, title, mass_label, policy_label, quality_label, term_color)
            (
                0,
                "term1",
                "p_fail",
                "p_call_fail",
                "p_acc_call_fail",
                "Term 1: Call Gain",
                "$P(\\mathcal{D}_{\\text{fail}})$",
                "$P(c \\mid \\mathcal{D}_{\\text{fail}})$",
                "$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}})$",
                "darkgreen",
            ),
            (
                1,
                "term2",
                "p_fail",
                "p_nocall_fail",
                "p_acc_nocall_fail",
                "Term 2: Schema Gain",
                "$P(\\mathcal{D}_{\\text{fail}})$",
                "$P(\\neg c \\mid \\mathcal{D}_{\\text{fail}})$",
                "$P(\\checkmark \\mid \\neg c, \\mathcal{D}_{\\text{fail}})$",
                "lightgreen",
            ),
            (
                2,
                "term3",
                "p_succ",
                "p_call_succ",
                "p_err_call_succ",
                "Term 3: Call Harm",
                "$P(\\mathcal{D}_{\\text{succ}})$",
                "$P(c \\mid \\mathcal{D}_{\\text{succ}})$",
                "$P(\\times \\mid c, \\mathcal{D}_{\\text{succ}})$",
                "red",
            ),
            (
                3,
                "term4",
                "p_succ",
                "p_nocall_succ",
                "p_err_nocall_succ",
                "Term 4: Schema Harm",
                "$P(\\mathcal{D}_{\\text{succ}})$",
                "$P(\\neg c \\mid \\mathcal{D}_{\\text{succ}})$",
                "$P(\\times \\mid \\neg c, \\mathcal{D}_{\\text{succ}})$",
                "lightcoral",
            ),
        ]

        # Plot each subplot in this row
        for (
            col,
            term_key,
            mass_key,
            policy_key,
            quality_key,
            title,
            mass_label,
            policy_label,
            quality_label,
            term_color,
        ) in subplot_configs:
            ax_left = axes[exp_idx, col]
            ax_right = ax_left.twinx()  # Create right y-axis

            # Term label mapping (without "Term x:" prefix)
            term_labels = {
                "term1": "Call Gain",
                "term2": "Schema Gain",
                "term3": "Call Harm",
                "term4": "Schema Harm",
            }

            # Left axis: Term (thick line with markers at every data point)
            line_term = ax_left.plot(
                common_steps,
                aggregated_data[term_key],
                color=term_color,
                linestyle="-",
                linewidth=3.0,
                marker="o",
                markersize=8,
                label=term_labels[term_key],
                alpha=0.95,
                zorder=5,
            )[0]

            # Right axis: Three factors (thin lines)
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

            # Axis labels and formatting
            # X-axis label only on last row
            if exp_idx == n_experiments - 1:
                ax_left.set_xlabel("Steps", fontsize=16, fontweight="medium")

            # Left Y-axis label only on leftmost column
            if col == 0:
                ax_left.set_ylabel(
                    "Probability", fontsize=16, fontweight="medium", color=term_color
                )

            # Right Y-axis label only on rightmost column
            if col == 3:
                ax_right.set_ylabel("Factor Probability", fontsize=16, fontweight="medium")

            ax_left.tick_params(axis="y", labelcolor=term_color)
            ax_right.set_ylim(0, 1.0)  # Fixed 0-1 for factors

            # Title (only for first row to save space)
            if exp_idx == 0:
                ax_left.set_title(title, fontsize=14, fontweight="bold", pad=10)

            # Grid
            ax_left.grid(True, alpha=0.3, linewidth=0.5)
            ax_left.set_axisbelow(True)

            # Legend: combine lines from both axes (only for first row)
            if exp_idx == 0:
                lines = [line_term, line_mass, line_policy, line_quality]
                labels = [l.get_label() for l in lines]
                ax_left.legend(
                    lines, labels, loc="best", fontsize=9, frameon=True, fancybox=True, shadow=True
                )

            # Scale x-axis by 8
            from matplotlib.ticker import FuncFormatter

            ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # Add row label (a), (b), (c) with model name - centered below each row
        if captions and exp_idx < len(captions):
            row_label = captions[exp_idx]
        else:
            row_label = exp_name

        label = chr(ord("a") + exp_idx)
        fig.text(
            0.5,  # Horizontal center
            1 - (exp_idx + 1) / n_experiments - 0.02,  # Below this row
            f"({label}) {row_label}",
            fontsize=PAPER_FIG_CONFIG["row_label_fontsize"],
            fontweight="bold",
            ha="center",
            va="top",
        )

    # Overall title
    fig.suptitle(
        "Term Factor Decomposition (Mass × Policy × Quality)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.35)  # Adjust vertical spacing for row labels

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / output_filename

    fig.savefig(
        filepath,
        dpi=PAPER_FIG_CONFIG["dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )

    print(f"\nFigure saved to: {filepath}")
    plt.close(fig)

    return filepath


def plot_measure_figure(
    experiment_names: list[str],
    aggregated_benchmarks: list[str],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "perception_aggregated.pdf",
    captions: list[str] | None = None,
):
    """Plot paper figure with multiple experiments.

    Each experiment gets one row with two subplots:
    - Left: Δ Performance (relative to start)
    - Right: Absolute performance (normalized)

    Args:
        experiment_names: List of experiment names to plot
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        captions: List of captions for each experiment
        aggregated_benchmarks: List of benchmark names to aggregate
    """
    setup_plot_style()

    n_experiments = len(experiment_names)
    fig, axes = plt.subplots(
        n_experiments,
        2,
        figsize=(
            PAPER_FIG_CONFIG["fig_width"],
            PAPER_FIG_CONFIG["fig_height_per_row"] * n_experiments,
        ),
        dpi=PAPER_FIG_CONFIG["dpi"],
        facecolor="white",
    )
    fig.patch.set_facecolor("white")

    # Ensure axes is 2D
    if n_experiments == 1:
        axes = axes.reshape(1, -1)

    # Process each experiment
    for exp_idx, exp_name in enumerate(experiment_names):
        print(f"\nProcessing experiment {exp_idx + 1}/{n_experiments}: {exp_name}")

        # Load experiment data with baseline performance
        df = load_experiment_with_baseline(exp_name)
        if df is None:
            print(f"Failed to load experiment '{exp_name}'")
            continue
        selected_benchmarks = get_sorted_benchmarks(df)

        # Filter selected benchmarks based on aggregated_benchmarks
        target_benchmarks = [b for b in selected_benchmarks if b in aggregated_benchmarks]

        if not target_benchmarks:
            print(f"No matching benchmarks found for {exp_name}")
            continue

        # Aggregate data
        agg_data = aggregate_benchmarks(df, target_benchmarks, smoothing_factor, smoothing_method)

        if agg_data is None:
            print(f"Failed to aggregate data for {exp_name}")
            continue

        # Get axes for this experiment
        ax_left = axes[exp_idx, 0]  # Δ Performance
        ax_right = axes[exp_idx, 1]  # Absolute Performance

        steps = agg_data["steps"]
        w_tool_smooth = agg_data["w_tool_smooth"]
        wo_tool_smooth = agg_data["wo_tool_smooth"]
        call_rate = agg_data["call_rate"]

        # Dense interpolation for smooth fills
        num_interp_points = max(200, len(steps) * 10)
        steps_dense = np.linspace(steps[0], steps[-1], num_interp_points)
        w_tool_dense = np.interp(steps_dense, steps, w_tool_smooth)
        wo_tool_dense = np.interp(steps_dense, steps, wo_tool_smooth)
        call_rate_dense = np.interp(steps_dense, steps, call_rate)

        # ============================================
        # LEFT SUBPLOT: Δ Performance (relative)
        # ============================================
        ax_left.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1)

        # w/ tool curve
        ax_left.plot(
            steps_dense,
            w_tool_dense,
            color="black",
            linestyle="-",
            linewidth=PAPER_FIG_CONFIG["linewidth"],
            label="$\\mathrm{f}_{w}$",
            alpha=0.8,
            zorder=3,
        )
        ax_left.scatter(
            steps,
            w_tool_smooth,
            color="black",
            marker="o",
            s=PAPER_FIG_CONFIG["scatter_size_large"],
            edgecolors="white",
            linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
            alpha=0.8,
            zorder=4,
        )

        # w/o tool curve
        ax_left.plot(
            steps_dense,
            wo_tool_dense,
            color="black",
            linestyle="--",
            linewidth=PAPER_FIG_CONFIG["linewidth"],
            label="$\\mathrm{f}_{wo}$",
            alpha=0.7,
            zorder=2,
        )
        ax_left.scatter(
            steps,
            wo_tool_smooth,
            color="black",
            marker="s",
            s=PAPER_FIG_CONFIG["scatter_size_small"],
            edgecolors="white",
            linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
            alpha=0.7,
            zorder=3,
        )

        # Fill area between w/o tool curve and x-axis with hatch pattern
        ax_left.fill_between(
            steps_dense,
            0,
            wo_tool_dense,
            facecolor="none",
            edgecolor=PAPER_FIG_CONFIG["base_hatch_color"],
            hatch="///",
            alpha=0.4,
            linewidth=0,
            zorder=0.5,
            label="$|B_{\\text{wo}}|$",
        )

        # Filled areas with alpha based on call rate
        alpha_values = 0.1 + call_rate_dense * 0.7

        positive_labeled = False
        negative_labeled = False
        for i in range(len(steps_dense) - 1):
            x_fill = [steps_dense[i], steps_dense[i + 1], steps_dense[i + 1], steps_dense[i]]
            y_fill = [wo_tool_dense[i], wo_tool_dense[i + 1], w_tool_dense[i + 1], w_tool_dense[i]]

            alpha = alpha_values[i]

            if w_tool_dense[i] >= wo_tool_dense[i] and w_tool_dense[i + 1] >= wo_tool_dense[i + 1]:
                if not positive_labeled:
                    ax_left.fill(
                        x_fill,
                        y_fill,
                        color=PAPER_FIG_CONFIG["positive_color"],
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                        label="$|B_{\\Delta_\\mathrm{tool}}|^+$",
                    )
                    positive_labeled = True
                else:
                    ax_left.fill(
                        x_fill,
                        y_fill,
                        color=PAPER_FIG_CONFIG["positive_color"],
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                    )
            elif w_tool_dense[i] < wo_tool_dense[i] and w_tool_dense[i + 1] < wo_tool_dense[i + 1]:
                if not negative_labeled:
                    ax_left.fill(
                        x_fill,
                        y_fill,
                        color=PAPER_FIG_CONFIG["negative_color"],
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                        label="$|B_{\\Delta_\\mathrm{tool}}|^-$",
                    )
                    negative_labeled = True
                else:
                    ax_left.fill(
                        x_fill,
                        y_fill,
                        color=PAPER_FIG_CONFIG["negative_color"],
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                    )

        # Calculate S_tool for aggregated data
        aggregated_w_tool_raw = agg_data["w_tool_raw"]
        aggregated_wo_tool_raw = agg_data["wo_tool_raw"]

        # diff_aggregated_raw = w_tool_smooth - wo_tool_smooth
        diff_aggregated_raw = aggregated_w_tool_raw - aggregated_wo_tool_raw
        positive_area_agg = np.trapezoid(np.maximum(diff_aggregated_raw, 0), steps)
        negative_area_agg = np.trapezoid(np.minimum(diff_aggregated_raw, 0), steps)
        abs_base_area_agg = np.trapezoid(np.abs(aggregated_wo_tool_raw), steps)
        abs_tool_area_agg = positive_area_agg + abs(negative_area_agg)
        total_change_agg = abs_base_area_agg + abs_tool_area_agg
        S_tool_agg = abs_tool_area_agg / total_change_agg if total_change_agg > 0 else 0

        # Add progress bar below title
        from matplotlib.patches import Rectangle

        bar_height = PAPER_FIG_CONFIG["bar_height"]
        bar_y_position = PAPER_FIG_CONFIG["bar_y_position"]
        bar_width = PAPER_FIG_CONFIG["bar_width"]
        bar_x_start = PAPER_FIG_CONFIG["bar_x_start"]

        # Calculate proportions
        total_area_agg = abs_base_area_agg + abs_tool_area_agg
        base_frac_agg = abs_base_area_agg / total_area_agg if total_area_agg > 0 else 0
        tool_pos_frac_agg = positive_area_agg / total_area_agg if total_area_agg > 0 else 0
        tool_neg_frac_agg = abs(negative_area_agg) / total_area_agg if total_area_agg > 0 else 0

        # Background bar (white with black border)
        ax_left.add_patch(
            Rectangle(
                (bar_x_start, bar_y_position),
                bar_width,
                bar_height,
                transform=ax_left.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=PAPER_FIG_CONFIG["bar_border_linewidth"],
                zorder=10,
            )
        )

        # Base portion (left, hatch pattern)
        base_width = base_frac_agg * bar_width
        ax_left.add_patch(
            Rectangle(
                (bar_x_start, bar_y_position),
                base_width,
                bar_height,
                transform=ax_left.transAxes,
                facecolor="none",
                edgecolor=PAPER_FIG_CONFIG["base_hatch_color"],
                hatch="///",
                linewidth=0,
                zorder=11,
            )
        )

        # Tool+ portion (green)
        tool_pos_width = tool_pos_frac_agg * bar_width
        ax_left.add_patch(
            Rectangle(
                (bar_x_start + base_width, bar_y_position),
                tool_pos_width,
                bar_height,
                transform=ax_left.transAxes,
                facecolor=PAPER_FIG_CONFIG["positive_color"],
                edgecolor="none",
                alpha=PAPER_FIG_CONFIG["bar_alpha"],
                zorder=11,
            )
        )

        # Tool- portion (red)
        tool_neg_width = tool_neg_frac_agg * bar_width
        ax_left.add_patch(
            Rectangle(
                (bar_x_start + base_width + tool_pos_width, bar_y_position),
                tool_neg_width,
                bar_height,
                transform=ax_left.transAxes,
                facecolor=PAPER_FIG_CONFIG["negative_color"],
                edgecolor="none",
                alpha=PAPER_FIG_CONFIG["bar_alpha"],
                zorder=11,
            )
        )

        # Add labels
        ax_left.text(
            bar_x_start - 0.02,
            bar_y_position + bar_height / 2,
            "Intrinsic",
            transform=ax_left.transAxes,
            fontsize=PAPER_FIG_CONFIG["bar_label_fontsize"],
            verticalalignment="center",
            horizontalalignment="right",
            fontweight="medium",
        )
        ax_left.text(
            bar_x_start + bar_width + 0.02,
            bar_y_position + bar_height / 2,
            "Tool-induced",
            transform=ax_left.transAxes,
            fontsize=PAPER_FIG_CONFIG["bar_label_fontsize"],
            verticalalignment="center",
            horizontalalignment="left",
            fontweight="medium",
        )

        # Add S_tool value above the bar
        ax_left.text(
            bar_x_start + bar_width / 2,
            bar_y_position + bar_height + 0.01,
            f"$S_{{\\text{{tool}}}}$ = {S_tool_agg:.2f}",
            transform=ax_left.transAxes,
            fontsize=PAPER_FIG_CONFIG["stool_fontsize"],
            verticalalignment="bottom",
            horizontalalignment="center",
            fontweight="bold",
        )

        # Add call rate colorbar
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        cax = inset_axes(
            ax_left,
            width=PAPER_FIG_CONFIG["colorbar_width"],
            height=PAPER_FIG_CONFIG["colorbar_height"],
            loc="center right",
            bbox_to_anchor=(PAPER_FIG_CONFIG["colorbar_bbox_x"], 0, 1, 1),
            bbox_transform=ax_left.transAxes,
            borderpad=0,
        )

        n_colors = 100

        # Top half: green gradient (positive tool effect)
        for i in range(n_colors):
            y_pos = 0.5 + i / (2 * n_colors)
            call_rate_val = i / (n_colors - 1)
            alpha_val = 0.1 + call_rate_val * 0.7

            cax.add_patch(
                Rectangle(
                    (0, y_pos),
                    1,
                    1 / (2 * n_colors),
                    facecolor=PAPER_FIG_CONFIG["positive_color"],
                    alpha=alpha_val,
                    edgecolor="none",
                )
            )

        # Bottom half: red gradient (negative tool effect)
        for i in range(n_colors):
            y_pos = (n_colors - 1 - i) / (2 * n_colors)
            call_rate_val = i / (n_colors - 1)
            alpha_val = 0.1 + call_rate_val * 0.7

            cax.add_patch(
                Rectangle(
                    (0, y_pos),
                    1,
                    1 / (2 * n_colors),
                    facecolor=PAPER_FIG_CONFIG["negative_color"],
                    alpha=alpha_val,
                    edgecolor="none",
                )
            )

        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.set_xticks([])
        cax.set_yticks([0, 0.5, 1])
        cax.set_yticklabels(["1", "0", "1"], fontsize=PAPER_FIG_CONFIG["colorbar_tick_fontsize"])
        cax.yaxis.tick_right()
        cax.set_ylabel(
            "Call Rate",
            fontsize=PAPER_FIG_CONFIG["colorbar_label_fontsize"],
            rotation=270,
            labelpad=15,
        )
        cax.yaxis.set_label_position("right")

        cax.axhline(y=0.5, color="gray", linewidth=1, alpha=0.5)

        # Styling
        ax_left.set_title(
            "Δ Accuracy (Per-benchmark Normalized)",
            fontsize=PAPER_FIG_CONFIG["title_fontsize"],
            fontweight="bold",
            pad=PAPER_FIG_CONFIG["title_pad_with_bar"],
        )
        ax_left.set_xlabel(
            "Steps", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )
        ax_left.set_ylabel(
            "Normalized Δ Accuracy",
            fontsize=PAPER_FIG_CONFIG["label_fontsize"],
            fontweight="medium",
        )
        ax_left.grid(
            True, alpha=PAPER_FIG_CONFIG["grid_alpha"], linewidth=PAPER_FIG_CONFIG["grid_linewidth"]
        )
        ax_left.set_axisbelow(True)
        ax_left.legend(
            loc="best",
            bbox_to_anchor=(0.25, 0.65),
            fontsize=PAPER_FIG_CONFIG["legend_fontsize"],
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax_left.set_ylim(None, 1)  # Upper limit at 1 for normalized drift ([-1, 1] range)

        # Scale x-axis labels by 8
        from matplotlib.ticker import FuncFormatter

        ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # ============================================
        # RIGHT SUBPLOT: Absolute Performance (normalized)
        # ============================================
        # Aggregate absolute scores (not relative to start)
        all_w_tool_abs = []
        all_wo_tool_abs = []

        for benchmark in aggregated_benchmarks:
            benchmark_data = df[df["benchmark"] == benchmark].copy()
            if benchmark_data.empty:
                continue

            benchmark_data = benchmark_data.sort_values("step")

            if (
                "w_tool_score" not in benchmark_data.columns
                or "wo_tool_score" not in benchmark_data.columns
            ):
                continue

            w_tool_raw = benchmark_data["w_tool_score"].values
            wo_tool_raw = benchmark_data["wo_tool_score"].values

            # Use raw values (no normalization) - preserve physical meaning
            all_w_tool_abs.append(w_tool_raw)
            all_wo_tool_abs.append(wo_tool_raw)

        # Aggregate (mean across benchmarks)
        aggregated_w_tool_abs_raw = np.mean(all_w_tool_abs, axis=0)
        aggregated_wo_tool_abs_raw = np.mean(all_wo_tool_abs, axis=0)

        # Smooth
        if smoothing_factor > 0 or smoothing_method != "none":
            aggregated_w_tool_abs = smooth_curve(
                aggregated_w_tool_abs_raw, steps, smoothing_factor, smoothing_method
            )
            aggregated_wo_tool_abs = smooth_curve(
                aggregated_wo_tool_abs_raw, steps, smoothing_factor, smoothing_method
            )
        else:
            aggregated_w_tool_abs = aggregated_w_tool_abs_raw.copy()
            aggregated_wo_tool_abs = aggregated_wo_tool_abs_raw.copy()

        # Interpolate for smooth plotting
        w_tool_abs_dense = np.interp(steps_dense, steps, aggregated_w_tool_abs)
        wo_tool_abs_dense = np.interp(steps_dense, steps, aggregated_wo_tool_abs)

        # Plot w/ tool
        ax_right.plot(
            steps_dense,
            w_tool_abs_dense,
            color="black",
            linestyle="-",
            linewidth=PAPER_FIG_CONFIG["linewidth"],
            label="$\\mathrm{Acc}_\\mathrm{w}$",
            alpha=0.8,
            zorder=3,
        )
        ax_right.scatter(
            steps,
            aggregated_w_tool_abs,
            color="black",
            marker="o",
            s=PAPER_FIG_CONFIG["scatter_size_large"],
            edgecolors="white",
            linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
            alpha=0.8,
            zorder=4,
        )

        # Plot w/o tool
        ax_right.plot(
            steps_dense,
            wo_tool_abs_dense,
            color="black",
            linestyle="--",
            linewidth=PAPER_FIG_CONFIG["linewidth"],
            label="$\\mathrm{Acc}_\\mathrm{wo}$",
            alpha=0.7,
            zorder=2,
        )
        ax_right.scatter(
            steps,
            aggregated_wo_tool_abs,
            color="black",
            marker="s",
            s=PAPER_FIG_CONFIG["scatter_size_small"],
            edgecolors="white",
            linewidths=PAPER_FIG_CONFIG["scatter_edgewidth"],
            alpha=0.7,
            zorder=3,
        )

        # Styling
        ax_right.set_title(
            "Accuracy",
            fontsize=PAPER_FIG_CONFIG["title_fontsize"],
            fontweight="bold",
            pad=PAPER_FIG_CONFIG["title_pad_normal"],
        )
        ax_right.set_xlabel(
            "Steps", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )
        ax_right.set_ylabel(
            "Accuracy", fontsize=PAPER_FIG_CONFIG["label_fontsize"], fontweight="medium"
        )
        ax_right.grid(
            True, alpha=PAPER_FIG_CONFIG["grid_alpha"], linewidth=PAPER_FIG_CONFIG["grid_linewidth"]
        )
        ax_right.set_axisbelow(True)
        # ax_right.set_ylim(0, 1)  # Accuracy range [0, 1]
        ax_right.legend(
            loc="best",
            fontsize=PAPER_FIG_CONFIG["legend_fontsize"],
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Scale x-axis labels by 8
        from matplotlib.ticker import FuncFormatter

        ax_right.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # Row label (a), (b), (c), ... - centered below each row
        label = chr(ord("a") + exp_idx)
        fig.text(
            0.5,  # Horizontal center
            1
            - (exp_idx + 1) / n_experiments
            - PAPER_FIG_CONFIG["row_label_y_offset"],  # Below this row
            f"({label}) {captions[exp_idx]}",
            fontsize=PAPER_FIG_CONFIG["row_label_fontsize"],
            fontweight="bold",
            ha="center",
            va="top",
        )

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        plt.tight_layout()  # No need for rect since labels are now centered below
        plt.subplots_adjust(wspace=0.30, hspace=0.45)  # wspace控制横向，hspace控制纵向``

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / output_filename

    fig.savefig(
        filepath,
        # format="pdf",
        dpi=PAPER_FIG_CONFIG["dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )

    print(f"\nFigure saved to: {filepath}")
    plt.close(fig)
    return filepath


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot paper figures for multiple experiments")
    parser.add_argument(
        "experiment_names",
        type=str,
        nargs="+",
        help="List of experiment names to plot",
    )
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.99,
        help="Smoothing factor (default: 0.99)",
    )
    parser.add_argument(
        "--smoothing_method",
        type=str,
        default="time_weighted_ema",
        choices=["none", "savgol", "ema", "time_weighted_ema"],
        help="Smoothing method (default: time_weighted_ema)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_figures",
        help="Output directory (default: paper_figures)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="perception_aggregated.pdf",
        help="Output filename (default: perception_aggregated.pdf)",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="*",
        default=None,
        help="Optional captions for each experiment (list of strings)",
    )
    parser.add_argument(
        "--aggregated_benchmarks",
        type=str,
        nargs="+",
        required=True,
        help="Benchmarks to aggregate (required). Example: --aggregated_benchmarks vstar hrbench4k mathvision",
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
        print("Generating MEASURE figure...")
        print("=" * 80)
        plot_measure_figure(
            args.experiment_names,
            args.aggregated_benchmarks,
            args.smoothing_factor,
            args.smoothing_method,
            args.output_dir,
            "perception_aggregated.pdf",
            args.captions,
        )

    if args.figure_type in ["all", "explain"]:
        print("\n" + "=" * 80)
        print("Generating EXPLAIN figure...")
        print("=" * 80)
        plot_explain_figure(
            args.experiment_names,
            args.aggregated_benchmarks,
            args.smoothing_factor,
            args.smoothing_method,
            args.output_dir,
            "term_absolute_values.pdf",
            args.captions,
        )

    if args.figure_type in ["all", "diagnose"]:
        print("\n" + "=" * 80)
        print("Generating DIAGNOSE figure...")
        print("=" * 80)
        plot_diagnose_figure(
            args.experiment_names,
            args.aggregated_benchmarks,
            args.smoothing_factor,
            args.smoothing_method,
            args.output_dir,
            "term_factor_decomposition.pdf",
            args.captions,
        )

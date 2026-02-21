#!/usr/bin/env python3
"""
Plot term factor decomposition with FIXED D_fail and D_succ partition.

This script differs from plot_term_factor_decomposition in that:
- D_fail and D_succ are fixed based on step 0 (baseline) performance
- All subsequent steps use this fixed partition to compute terms and factors
- This allows us to see how the same samples evolve over training
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from create_csv import (
    analyze_trajectories,
    extract_step_from_path,
    load_benchmark_trajectories,
)
from plot_paper_figures import (
    MODEL_BASELINE_MAPPING,
    PAPER_FIG_CONFIG,
    RESULTS_DIR,
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


def get_baseline_model_from_exp_name(exp_name: str) -> str | None:
    """Extract baseline model path from experiment name."""
    for prefix, baseline_path in MODEL_BASELINE_MAPPING.items():
        if exp_name.startswith(prefix):
            return baseline_path
    return None


def load_baseline_partition(
    evaluation_dir: str, exp_name: str, benchmark: str, baseline_step: int = 0
) -> set | None:
    """Load the baseline D_succ partition (wo_tool_correct_indices) at step 0.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_name: Experiment name
        benchmark: Benchmark name
        baseline_step: The step to use as baseline (default: 0)

    Returns:
        Set of sample indices that were correct w/o tool at baseline, or None if not found
    """
    # Find baseline checkpoint directory
    exp_path = os.path.join(evaluation_dir, exp_name)

    baseline_dir = None
    for root, dirs, files in os.walk(exp_path):
        if f"global_step_{baseline_step}" in root and benchmark in root:
            if "evaluation_results.json" in files:
                baseline_dir = root
                break

    if baseline_dir is None:
        print(f"Warning: Could not find baseline checkpoint for {benchmark}")
        return None

    # Load trajectories
    trajectories = load_benchmark_trajectories(baseline_dir, benchmark)
    wo_tool_trajs = trajectories["wo_tool_trajectories"]

    if not wo_tool_trajs:
        print(f"Warning: No w/o tool trajectories found for {benchmark} at baseline")
        return None

    # Determine D_succ at baseline (samples correct w/o tool)
    baseline_correct_indices = set()
    for idx, traj in enumerate(wo_tool_trajs):
        accuracy_original = traj.get("accuracy_reward_original", 0)
        accuracy_llm = traj.get("accuracy_reward_llm", 0)

        if accuracy_original == 1 or accuracy_llm == 1:
            baseline_correct_indices.add(idx)

    return baseline_correct_indices


def compute_fixed_partition_metrics(
    evaluation_dir: str, exp_name: str, target_benchmarks: list[str], baseline_step: int = 0
) -> dict[str, Any]:
    """Compute metrics across all steps using fixed baseline partition.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_name: Experiment name
        target_benchmarks: List of benchmarks to process
        baseline_step: The step to use as baseline (default: 0)

    Returns:
        Dict mapping benchmark -> step -> metrics
    """
    exp_path = os.path.join(evaluation_dir, exp_name)

    if not os.path.exists(exp_path):
        print(f"Warning: Experiment path {exp_path} does not exist")
        return {}

    print(f"\nProcessing experiment: {exp_name}")

    # Load baseline partitions for all benchmarks
    baseline_partitions = {}
    for benchmark in target_benchmarks:
        baseline_correct = load_baseline_partition(
            evaluation_dir, exp_name, benchmark, baseline_step
        )
        if baseline_correct is not None:
            baseline_partitions[benchmark] = baseline_correct
            print(
                f"  Loaded baseline partition for {benchmark}: "
                f"{len(baseline_correct)} correct samples at step {baseline_step}"
            )

    # Collect all checkpoint directories
    checkpoint_data = {}  # {benchmark: {step: checkpoint_dir}}

    for root, dirs, files in os.walk(exp_path):
        if "evaluation_results.json" not in files:
            continue

        bench_name = os.path.basename(root)
        if bench_name not in target_benchmarks:
            continue

        if bench_name not in baseline_partitions:
            continue

        step = extract_step_from_path(root)

        if bench_name not in checkpoint_data:
            checkpoint_data[bench_name] = {}
        checkpoint_data[bench_name][step] = root

    # Compute metrics for each benchmark and step
    results = {}  # {benchmark: {step: metrics}}

    for benchmark in tqdm(target_benchmarks, desc="Processing benchmarks"):
        if benchmark not in baseline_partitions:
            continue

        if benchmark not in checkpoint_data:
            continue

        baseline_correct = baseline_partitions[benchmark]
        results[benchmark] = {}

        for step in sorted(checkpoint_data[benchmark].keys()):
            checkpoint_dir = checkpoint_data[benchmark][step]

            # Load trajectories
            trajectories = load_benchmark_trajectories(checkpoint_dir, benchmark)

            if not trajectories["wo_tool_trajectories"] or not trajectories["w_tool_trajectories"]:
                continue

            # Analyze with FIXED baseline partition
            metrics = analyze_trajectories(trajectories, baseline_correct_indices=baseline_correct)
            metrics["step"] = step

            results[benchmark][step] = metrics

    return results


def aggregate_fixed_partition_metrics(
    benchmark_metrics: dict[str, dict[int, dict[str, Any]]],
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
) -> dict[str, Any]:
    """Aggregate metrics across benchmarks with smoothing.

    Args:
        benchmark_metrics: Dict mapping benchmark -> step -> metrics
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method

    Returns:
        Dict with aggregated and smoothed data
    """
    # Collect data from all benchmarks
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

    for benchmark, step_metrics in benchmark_metrics.items():
        if not step_metrics:
            continue

        steps = sorted(step_metrics.keys())
        if common_steps is None:
            common_steps = np.array(steps)

        # Extract values for each column
        for col in required_cols:
            if col not in aggregated_data:
                aggregated_data[col] = []

            values = [step_metrics[step][col] for step in steps]
            aggregated_data[col].append(values)

    if len(aggregated_data) == 0:
        return None

    # Average across benchmarks
    for key in aggregated_data:
        aggregated_data[key] = np.mean(aggregated_data[key], axis=0)

    # Smooth all curves
    if smoothing_factor > 0 or smoothing_method != "none":
        for key in aggregated_data:
            aggregated_data[key] = smooth_curve(
                aggregated_data[key], common_steps, smoothing_factor, smoothing_method
            )

    aggregated_data["steps"] = common_steps

    return aggregated_data


def plot_term_factor_decomposition_fixed_partition(
    experiment_names: list[str],
    evaluation_dir: str = None,
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "paper_figures",
    output_filename: str = "term_factor_decomposition_fixed_partition.pdf",
    captions: list[str] | None = None,
    aggregated_benchmarks: list[str] | None = None,
    baseline_step: int = 0,
):
    """Plot term1-4 factor decomposition with FIXED D_fail/D_succ partition.

    Creates an Nx4 grid where each row is one experiment and each column is one term:
    - Column 0: Term1 (Call Gain) = p_fail × p_call_fail × p_acc_call_fail
    - Column 1: Term2 (Schema Gain) = p_fail × p_nocall_fail × p_acc_nocall_fail
    - Column 2: Term3 (Call Harm) = p_succ × p_call_succ × p_err_call_succ
    - Column 3: Term4 (Schema Harm) = p_succ × p_nocall_succ × p_err_nocall_succ

    KEY DIFFERENCE: D_fail and D_succ are fixed based on step 0 (baseline) performance.

    Args:
        experiment_names: List of experiment names to plot
        evaluation_dir: Path to evaluation directory
        smoothing_factor: Smoothing parameter
        smoothing_method: Smoothing method
        output_dir: Output directory
        output_filename: Output filename
        captions: List of captions for each experiment
        aggregated_benchmarks: List of benchmark names to aggregate
        baseline_step: Step to use for baseline partition (default: 0)
    """
    if evaluation_dir is None:
        evaluation_dir = str(RESULTS_DIR)

    if aggregated_benchmarks is None:
        aggregated_benchmarks = PERCEPTION_BENCHMARKS

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
        print(f"\n{'='*80}")
        print(f"Processing experiment {exp_idx + 1}/{n_experiments}: {exp_name}")
        print(f"Using FIXED partition from step {baseline_step}")
        print(f"{'='*80}")

        # Compute metrics with fixed partition
        benchmark_metrics = compute_fixed_partition_metrics(
            evaluation_dir, exp_name, aggregated_benchmarks, baseline_step=baseline_step
        )

        if not benchmark_metrics:
            print(f"No data found for {exp_name}")
            continue

        # Aggregate across benchmarks
        aggregated_data = aggregate_fixed_partition_metrics(
            benchmark_metrics, smoothing_factor, smoothing_method
        )

        if aggregated_data is None:
            print(f"Failed to aggregate data for {exp_name}")
            continue

        common_steps = aggregated_data["steps"]

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
                "$P(\\mathcal{D}_{\\text{fail}}^{(0)})$",
                "$P(c \\mid \\mathcal{D}_{\\text{fail}}^{(0)})$",
                "$P(\\checkmark \\mid c, \\mathcal{D}_{\\text{fail}}^{(0)})$",
                "darkgreen",
            ),
            (
                1,
                "term2",
                "p_fail",
                "p_nocall_fail",
                "p_acc_nocall_fail",
                "Term 2: Schema Gain",
                "$P(\\mathcal{D}_{\\text{fail}}^{(0)})$",
                "$P(\\neg c \\mid \\mathcal{D}_{\\text{fail}}^{(0)})$",
                "$P(\\checkmark \\mid \\neg c, \\mathcal{D}_{\\text{fail}}^{(0)})$",
                "lightgreen",
            ),
            (
                2,
                "term3",
                "p_succ",
                "p_call_succ",
                "p_err_call_succ",
                "Term 3: Call Harm",
                "$P(\\mathcal{D}_{\\text{succ}}^{(0)})$",
                "$P(c \\mid \\mathcal{D}_{\\text{succ}}^{(0)})$",
                "$P(\\times \\mid c, \\mathcal{D}_{\\text{succ}}^{(0)})$",
                "red",
            ),
            (
                3,
                "term4",
                "p_succ",
                "p_nocall_succ",
                "p_err_nocall_succ",
                "Term 4: Schema Harm",
                "$P(\\mathcal{D}_{\\text{succ}}^{(0)})$",
                "$P(\\neg c \\mid \\mathcal{D}_{\\text{succ}}^{(0)})$",
                "$P(\\times \\mid \\neg c, \\mathcal{D}_{\\text{succ}}^{(0)})$",
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
            term_values = aggregated_data[term_key]
            ax_left.plot(
                common_steps,
                term_values,
                color=term_color,
                linestyle="-",
                linewidth=3,
                marker="o",
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label=term_labels[term_key],
                zorder=3,
            )

            # Right axis: Three factors (thin lines)
            mass_values = aggregated_data[mass_key]
            policy_values = aggregated_data[policy_key]
            quality_values = aggregated_data[quality_key]

            ax_right.plot(
                common_steps,
                mass_values,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=mass_label,
                zorder=2,
            )
            ax_right.plot(
                common_steps,
                policy_values,
                color="blue",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                label=policy_label,
                zorder=2,
            )
            ax_right.plot(
                common_steps,
                quality_values,
                color="purple",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.7,
                label=quality_label,
                zorder=2,
            )

            # Styling
            ax_left.set_title(
                title + f" (Fixed $\\mathcal{{D}}$ from step {baseline_step})",
                fontsize=12,
                fontweight="bold",
                pad=-100,
            )
            ax_left.set_xlabel("Steps", fontsize=11)
            ax_left.set_ylabel(
                term_labels[term_key], fontsize=11, color=term_color, fontweight="bold"
            )
            ax_left.tick_params(axis="y", labelcolor=term_color)
            ax_left.grid(True, alpha=0.3, linewidth=0.5)
            ax_left.set_axisbelow(True)

            # Right axis styling
            ax_right.set_ylabel("Probability", fontsize=11)
            ax_right.set_ylim(0, 1)
            ax_right.tick_params(axis="y")

            # Legends
            if col == 0:  # Only show legend on first column
                ax_left.legend(loc="upper left", fontsize=9, framealpha=0.9)
                ax_right.legend(loc="upper right", fontsize=8, framealpha=0.9)

            # Scale x-axis labels by 8 (assuming training steps are in units of 8)
            from matplotlib.ticker import FuncFormatter

            ax_left.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x * 8)}"))

        # Add caption below each row
        if captions and exp_idx < len(captions):
            fig.text(
                0.5,
                1 - (exp_idx + 1) / n_experiments + 0.02,
                captions[exp_idx],
                ha="center",
                fontsize=13,
                fontweight="bold",
            )

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    plt.tight_layout()
    plt.savefig(output_path, dpi=PAPER_FIG_CONFIG["dpi"], bbox_inches="tight")
    print(f"\n{'='*80}")
    print(f"Saved figure to: {output_path}")
    print(f"{'='*80}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot term factor decomposition with fixed D_fail/D_succ partition"
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results",
        help="Path to evaluation directory",
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
        default="term_factor_decomposition_fixed_partition.pdf",
        help="Output filename",
    )
    parser.add_argument("--smoothing_factor", type=float, default=0.99, help="Smoothing factor")
    parser.add_argument(
        "--baseline_step",
        type=int,
        default=0,
        help="Step to use for baseline partition (default: 0)",
    )

    args = parser.parse_args()

    plot_term_factor_decomposition_fixed_partition(
        experiment_names=args.exp_names,
        evaluation_dir=args.evaluation_dir,
        smoothing_factor=args.smoothing_factor,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        captions=args.captions,
        aggregated_benchmarks=args.benchmarks,
        baseline_step=args.baseline_step,
    )


# Example usage:
# python3 recipe/o3/plot_v3/plot_term_factor_fixed_partition.py \
#     --exp_names \
#         qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#     --captions "Qwen2.5-VL-7B-Instruct" \
#     --benchmarks vstar hrbench4k hrbench8k visualprobeasy visualprobmedium visualprobhard \
#     --baseline_step 0

if __name__ == "__main__":
    main()

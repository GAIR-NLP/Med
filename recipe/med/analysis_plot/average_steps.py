#!/usr/bin/env python3
"""
Average scores across steps 180, 190, and 200 for multiple experiments.

This script:
1. Reads CSV files created by create_csv.py for each experiment
2. Filters for steps 180, 190, and 200
3. Filters for specific benchmarks: hrbench4k, visualprobeasy, vstar
4. Averages w/o tool and w/ tool scores across these three steps
5. Outputs per-benchmark scores and overall average
"""
import argparse
import os
from typing import Any

import pandas as pd

TARGET_STEPS = [180, 190, 200]
TARGET_BENCHMARKS = ["hrbench4k", "visualprobeasy", "vstar"]


def load_experiment_csv(evaluation_dir: str, exp_name: str) -> pd.DataFrame | None:
    """Load CSV file for an experiment.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_name: Experiment name

    Returns:
        DataFrame or None if not found
    """
    exp_dir = os.path.join(evaluation_dir, exp_name)
    csv_name = exp_name.split("/")[-1]
    csv_path = os.path.join(exp_dir, f"{csv_name}_results.csv")

    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def compute_average_scores(
    df: pd.DataFrame, steps: list[int], benchmarks: list[str]
) -> dict[str, Any]:
    """Compute average scores across specified steps and benchmarks.

    Args:
        df: DataFrame with experiment results
        steps: List of step numbers to average
        benchmarks: List of benchmark names to include

    Returns:
        Dict with per-benchmark averages and overall average
    """
    # Filter for target steps and benchmarks
    df_filtered = df[df["step"].isin(steps) & df["benchmark"].isin(benchmarks)]

    if df_filtered.empty:
        print(f"Warning: No data found for steps {steps} and benchmarks {benchmarks}")
        return None

    # Compute per-benchmark averages
    per_benchmark = {}
    for bench in benchmarks:
        df_bench = df_filtered[df_filtered["benchmark"] == bench]
        if not df_bench.empty:
            per_benchmark[bench] = {
                "wo_tool": df_bench["wo_tool_score"].mean(),
                "w_tool": df_bench["w_tool_score"].mean(),
                "count": len(df_bench),
            }
        else:
            per_benchmark[bench] = {"wo_tool": None, "w_tool": None, "count": 0}

    # Compute overall average across all filtered data
    overall_wo = df_filtered["wo_tool_score"].mean()
    overall_w = df_filtered["w_tool_score"].mean()

    return {
        "per_benchmark": per_benchmark,
        "overall": {"wo_tool": overall_wo, "w_tool": overall_w, "count": len(df_filtered)},
    }


def process_experiments(evaluation_dir: str, exp_names: list[str]) -> list[dict[str, Any]]:
    """Process all experiments and compute averaged scores.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_names: List of experiment names

    Returns:
        List of dicts with experiment results
    """
    results = []

    for exp_name in exp_names:
        print(f"\nProcessing: {exp_name}")

        # Load CSV
        df = load_experiment_csv(evaluation_dir, exp_name)
        if df is None:
            continue

        # Compute averages
        avg_scores = compute_average_scores(df, TARGET_STEPS, TARGET_BENCHMARKS)

        if avg_scores is not None:
            print("  Per-benchmark scores:")
            for bench in TARGET_BENCHMARKS:
                bench_data = avg_scores["per_benchmark"][bench]
                if bench_data["wo_tool"] is not None:
                    print(
                        f"    {bench}: W/O={bench_data['wo_tool']:.4f}, W/={bench_data['w_tool']:.4f}"
                    )
                else:
                    print(f"    {bench}: No data")

            overall = avg_scores["overall"]
            print(f"  Overall: W/O={overall['wo_tool']:.4f}, W/={overall['w_tool']:.4f}")

            results.append({"experiment": exp_name, "avg_scores": avg_scores})
        else:
            print("  No data for target steps/benchmarks")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Average scores across steps 180, 190, and 200 for specific benchmarks"
    )
    parser.add_argument(
        "--evaluation_dir", type=str, required=True, help="Path to evaluation directory"
    )
    parser.add_argument(
        "--exp_names", type=str, nargs="+", required=True, help="List of experiment names"
    )
    parser.add_argument(
        "--output", type=str, default="average_scores.csv", help="Output CSV file path"
    )

    args = parser.parse_args()

    # Process all experiments
    results = process_experiments(args.evaluation_dir, args.exp_names)

    if not results:
        print("\nNo results found")
        return

    # Prepare data for CSV output
    csv_rows = []
    for result in results:
        exp_name = result["experiment"]
        avg_scores = result["avg_scores"]

        # Add per-benchmark rows
        for bench in TARGET_BENCHMARKS:
            bench_data = avg_scores["per_benchmark"][bench]
            if bench_data["wo_tool"] is not None:
                csv_rows.append(
                    {
                        "experiment": exp_name,
                        "benchmark": bench,
                        "wo_tool_score": bench_data["wo_tool"],
                        "w_tool_score": bench_data["w_tool"],
                        "improvement": bench_data["w_tool"] - bench_data["wo_tool"],
                        "count": bench_data["count"],
                    }
                )

        # Add overall average row
        overall = avg_scores["overall"]
        csv_rows.append(
            {
                "experiment": exp_name,
                "benchmark": "AVERAGE",
                "wo_tool_score": overall["wo_tool"],
                "w_tool_score": overall["w_tool"],
                "improvement": overall["w_tool"] - overall["wo_tool"],
                "count": overall["count"],
            }
        )

    # Save to CSV
    df_output = pd.DataFrame(csv_rows)
    df_output.to_csv(args.output, index=False)
    print(f"\n{'='*120}")
    print(f"Saved results to: {args.output}")
    print(f"{'='*120}")

    # Print detailed summary table
    print("\nDETAILED SUMMARY TABLE")
    print(f"{'='*120}")

    for result in results:
        exp_name = result["experiment"]
        avg_scores = result["avg_scores"]

        print(f"\n{exp_name}")
        print(f"{'-'*120}")
        print(
            f"{'Benchmark':<20} {'W/O Tool':>12} {'W/ Tool':>12} {'Improvement':>14} {'Count':>8}"
        )
        print(f"{'-'*120}")

        # Print per-benchmark scores
        for bench in TARGET_BENCHMARKS:
            bench_data = avg_scores["per_benchmark"][bench]
            if bench_data["wo_tool"] is not None:
                print(
                    f"{bench:<20} "
                    f"{bench_data['wo_tool']:>12.4f} "
                    f"{bench_data['w_tool']:>12.4f} "
                    f"{bench_data['w_tool'] - bench_data['wo_tool']:>+14.4f} "
                    f"{bench_data['count']:>8}"
                )
            else:
                print(f"{bench:<20} {'N/A':>12} {'N/A':>12} {'N/A':>14} {0:>8}")

        # Print overall average
        print(f"{'-'*120}")
        overall = avg_scores["overall"]
        print(
            f"{'AVERAGE':<20} "
            f"{overall['wo_tool']:>12.4f} "
            f"{overall['w_tool']:>12.4f} "
            f"{overall['w_tool'] - overall['wo_tool']:>+14.4f} "
            f"{overall['count']:>8}"
        )

    print(f"\n{'='*120}")


# Example usage:
# python3 recipe/o3/plot_v3/average_steps.py \
# --evaluation_dir /jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results \
# --exp_names \
#     qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.5_toolcall_ratio0.25_qwen2_5_stratified_tool_reward_discriminative_range0.1 \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.5_toolcall_ratio0.5_qwen2_5_stratified_tool_reward_discriminative \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.5_toolcall_ratio0.75_qwen2_5_stratified_tool_reward_discriminative \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.75_tool_call_ratio0.25_qwen2_5_stratified_tool_reward_discriminative_range0.1 \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.75_tool_call_ratio0.5_qwen2_5_stratified_tool_reward_discriminative_range0.1 \
#     qwen25vl_instruct_ablation_toolcall_ratio/all_data_natural_ratio0.75_tool_call_ratio0.75_qwen2_5_stratified_tool_reward_discriminative_range0.1 \
# --output average_scores.csv
if __name__ == "__main__":
    main()

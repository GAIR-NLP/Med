#!/usr/bin/env python3
"""
Create a comparison table showing Acc_wo, Acc_schema_nc, and Acc_w for baseline models.

- Acc_wo: w/o tool score from evaluation_results
- Acc_schema_nc: w/ tool score from evaluation_results_schema_no_call (force no call, only schema added)
- Acc_w: w/ tool score from evaluation_results
"""
import argparse
import os

import pandas as pd

BENCHMARK_ORDER = {
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
}


def create_comparison_table(
    evaluation_dir: str, schema_no_call_dir: str, model_names: list[str], output_path: str = None
) -> pd.DataFrame:
    """Create comparison table for multiple models.

    Args:
        evaluation_dir: Path to evaluation_results directory
        schema_no_call_dir: Path to evaluation_results_schema_no_call directory
        model_names: List of model names (e.g., ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"])
        output_path: Optional path to save CSV output

    Returns:
        DataFrame with comparison results
    """
    all_results = []

    for model_name in model_names:
        # Read evaluation_results CSV
        eval_csv = os.path.join(
            evaluation_dir, "verl_model", model_name, f"{model_name}_results.csv"
        )

        # Read schema_no_call CSV
        schema_csv = os.path.join(
            schema_no_call_dir, "verl_model", model_name, f"{model_name}_results.csv"
        )

        if not os.path.exists(eval_csv):
            print(f"Warning: {eval_csv} not found, skipping {model_name}")
            continue

        if not os.path.exists(schema_csv):
            print(f"Warning: {schema_csv} not found, skipping {model_name}")
            continue

        # Load data
        df_eval = pd.read_csv(eval_csv)
        df_schema = pd.read_csv(schema_csv)

        # Filter for step 0 only (baseline)
        df_eval = df_eval[df_eval["step"] == 0].copy()
        df_schema = df_schema[df_schema["step"] == 0].copy()

        # Merge on benchmark
        df_merged = df_eval[["benchmark", "wo_tool_score", "w_tool_score"]].merge(
            df_schema[["benchmark", "w_tool_score"]], on="benchmark", suffixes=("", "_schema_nc")
        )

        # Rename columns
        df_merged = df_merged.rename(
            columns={
                "wo_tool_score": "Acc_wo",
                "w_tool_score_schema_nc": "Acc_schema_nc",
                "w_tool_score": "Acc_w",
            }
        )

        # Add model column
        df_merged["model"] = model_name

        # Reorder columns
        df_merged = df_merged[["model", "benchmark", "Acc_wo", "Acc_schema_nc", "Acc_w"]]

        all_results.append(df_merged)

    # Concatenate all results
    if not all_results:
        print("No results found")
        return pd.DataFrame()

    df_final = pd.concat(all_results, ignore_index=True)

    # Sort by model and benchmark
    df_final = df_final.sort_values(["model", "benchmark"])

    # Save to CSV if output path is provided
    if output_path:
        df_final.to_csv(output_path, index=False)
        print(f"Saved comparison table to: {output_path}")

    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Create comparison table for Acc_wo, Acc_schema_nc, and Acc_w"
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results",
        help="Path to evaluation_results directory",
    )
    parser.add_argument(
        "--schema_no_call_dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results_schema_no_call",
        help="Path to evaluation_results_schema_no_call directory",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"],
        help="List of model names",
    )
    parser.add_argument(
        "--output", type=str, default="comparison_table.csv", help="Output CSV file path"
    )

    args = parser.parse_args()

    # Create comparison table
    df = create_comparison_table(
        args.evaluation_dir, args.schema_no_call_dir, args.model_names, args.output
    )

    # Print table
    if not df.empty:
        print("\nComparison Table:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)


# python3 create_comparison_table.py \
#       --evaluation_dir /jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results \
#       --schema_no_call_dir /jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results_schema_no_call \
if __name__ == "__main__":
    main()

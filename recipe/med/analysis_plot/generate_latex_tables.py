#!/usr/bin/env python3
"""
Generate LaTeX tables comparing Acc_wo, Acc_schema_nc, and Acc_w for baseline models.
"""
import argparse
import os

import pandas as pd

# Define the 6 benchmarks we care about
BENCHMARKS = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]

# Benchmark display names for table headers
BENCHMARK_DISPLAY_NAMES = {
    "vstar": "VStar",
    "hrbench4k": "HRBench\n(4k)",
    "hrbench8k": "HRBench\n(8k)",
    "visualprobeasy": "VisualProbe\n(Easy)",
    "visualprobmedium": "VisualProbe\n(Medium)",
    "visualprobhard": "VisualProbe\n(Hard)",
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL",
}


def load_model_data(evaluation_dir: str, schema_no_call_dir: str, model_name: str) -> pd.DataFrame:
    """Load and merge data for a single model.

    Args:
        evaluation_dir: Path to evaluation_results directory
        schema_no_call_dir: Path to evaluation_results_schema_no_call directory
        model_name: Model name

    Returns:
        DataFrame with columns: benchmark, Acc_wo, Acc_schema_nc, Acc_w
    """
    # Read evaluation_results CSV
    eval_csv = os.path.join(evaluation_dir, "verl_model", model_name, f"{model_name}_results.csv")

    # Read schema_no_call CSV
    schema_csv = os.path.join(
        schema_no_call_dir, "verl_model", model_name, f"{model_name}_results.csv"
    )

    # Load data
    df_eval = pd.read_csv(eval_csv)
    df_schema = pd.read_csv(schema_csv)

    # Filter for step 0 and selected benchmarks
    df_eval = df_eval[(df_eval["step"] == 0) & (df_eval["benchmark"].isin(BENCHMARKS))].copy()
    df_schema = df_schema[
        (df_schema["step"] == 0) & (df_schema["benchmark"].isin(BENCHMARKS))
    ].copy()

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

    return df_merged[["benchmark", "Acc_wo", "Acc_schema_nc", "Acc_w"]]


def format_number(value: float, decimal_places: int = 1, as_percentage: bool = True) -> str:
    """Format a number for LaTeX table.

    Args:
        value: Number to format
        decimal_places: Number of decimal places
        as_percentage: If True, multiply by 100 and treat as percentage

    Returns:
        Formatted string
    """
    if as_percentage:
        value = value * 100

    format_str = f"{{:.{decimal_places}f}}"
    return format_str.format(value)


def format_gap(gap: float, decimal_places: int = 1) -> str:
    """Format gap value with color coding.

    Args:
        gap: Gap value (can be positive or negative)
        decimal_places: Number of decimal places

    Returns:
        Formatted string with LaTeX color coding
    """
    gap_percentage = gap * 100
    format_str = f"{{:+.{decimal_places}f}}"  # Include sign
    formatted = format_str.format(gap_percentage)

    if gap < 0:
        return f"\\textcolor{{red}}{{{formatted}}}"
    else:
        return formatted


def generate_table1(model_data: dict[str, pd.DataFrame]) -> str:
    """Generate Table 1: Average performance across 6 benchmarks.

    Args:
        model_data: Dict mapping model_name -> DataFrame

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Model} & \\textbf{Intrinsic} & \\textbf{Schema-Only} & \\textbf{Gap} & \\textbf{Tool} \\\\"
    )
    lines.append(
        " & ($Acc_{\\text{wo}}$) & ($Acc_{\\text{nc}}$) & ($\\Delta$) & ($Acc_{\\text{w}}$) \\\\"
    )
    lines.append("\\midrule")

    for model_name in ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"]:
        if model_name not in model_data:
            continue

        df = model_data[model_name]

        # Calculate averages
        avg_wo = df["Acc_wo"].mean()
        avg_nc = df["Acc_schema_nc"].mean()
        avg_w = df["Acc_w"].mean()
        gap = avg_nc - avg_wo

        display_name = MODEL_DISPLAY_NAMES[model_name]

        line = f"{display_name} & {format_number(avg_wo)} & {format_number(avg_nc)} & {format_gap(gap)} & {format_number(avg_w)} \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")

    return "\n".join(lines)


def generate_table2(model_data: dict[str, pd.DataFrame]) -> str:
    """Generate Table 2: Per-benchmark performance.

    Args:
        model_data: Dict mapping model_name -> DataFrame

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Setting} & \\textbf{VStar} & \\textbf{HRBench} & \\textbf{HRBench} & \\textbf{VisualProbe} & \\textbf{VisualProbe} & \\textbf{VisualProbe} & \\textbf{Avg.} \\\\"
    )
    lines.append(" & & (4k) & (8k) & (Easy) & (Medium) & (Hard) & \\\\")
    lines.append("\\midrule")

    for model_name in ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"]:
        if model_name not in model_data:
            continue

        df = model_data[model_name]

        # Ensure benchmarks are in correct order
        df = df.set_index("benchmark").reindex(BENCHMARKS).reset_index()

        # Model section header
        display_name = MODEL_DISPLAY_NAMES[model_name]
        if model_name == "Qwen2.5-VL-7B-Instruct":
            lines.append(
                f"\\multicolumn{{8}}{{l}}{{\\textit{{\\textbf{{{display_name}-Instruct-7B}}}}}} \\\\"
            )
        else:
            lines.append(
                f"\\multicolumn{{8}}{{l}}{{\\textit{{\\textbf{{{display_name}-Instruct-8B}}}}}} \\\\"
            )

        # Row A: Intrinsic
        row_a_parts = ["(A) Intrinsic ($Acc_{\\text{wo}}$)"]
        for _, row in df.iterrows():
            row_a_parts.append(format_number(row["Acc_wo"]))
        avg_wo = df["Acc_wo"].mean()
        row_a_parts.append(format_number(avg_wo))
        lines.append(" & ".join(row_a_parts) + " \\\\")

        # Row B: Distracted
        row_b_parts = ["(B) Distracted ($Acc_{\\text{schema\\_nc}}$)"]
        for _, row in df.iterrows():
            row_b_parts.append(format_number(row["Acc_schema_nc"]))
        avg_nc = df["Acc_schema_nc"].mean()
        row_b_parts.append(format_number(avg_nc))
        lines.append(" & ".join(row_b_parts) + " \\\\")

        # Row C: Tool-Available
        row_c_parts = ["(C) Tool-Available ($Acc_{\\text{w}}$)"]
        for _, row in df.iterrows():
            row_c_parts.append(format_number(row["Acc_w"]))
        avg_w = df["Acc_w"].mean()
        row_c_parts.append(format_number(avg_w))
        lines.append(" & ".join(row_c_parts) + " \\\\")

        # Row Gap: Interference Gap
        row_gap_parts = ["\\rowcolor{gray!10} \\textbf{Interference Gap (B - A)}"]
        for _, row in df.iterrows():
            gap = row["Acc_schema_nc"] - row["Acc_wo"]
            row_gap_parts.append(format_gap(gap))
        avg_gap = avg_nc - avg_wo
        row_gap_parts.append(format_gap(avg_gap))
        lines.append(" & ".join(row_gap_parts) + " \\\\")

        # Add midrule between models
        if model_name == "Qwen2.5-VL-7B-Instruct":
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX comparison tables")
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
        "--output_table1",
        type=str,
        default="table1_average.tex",
        help="Output file for Table 1 (average)",
    )
    parser.add_argument(
        "--output_table2",
        type=str,
        default="table2_perbench.tex",
        help="Output file for Table 2 (per-benchmark)",
    )

    args = parser.parse_args()

    # Load data for both models
    model_data = {}
    for model_name in ["Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"]:
        try:
            df = load_model_data(args.evaluation_dir, args.schema_no_call_dir, model_name)
            model_data[model_name] = df
            print(f"Loaded data for {model_name}: {len(df)} benchmarks")
        except Exception as e:
            print(f"Error loading data for {model_name}: {e}")

    if not model_data:
        print("No data loaded, exiting")
        return

    # Generate Table 1
    table1 = generate_table1(model_data)
    with open(args.output_table1, "w") as f:
        f.write(table1)
    print(f"\nTable 1 saved to: {args.output_table1}")
    print("\n" + "=" * 80)
    print("TABLE 1: Average Performance")
    print("=" * 80)
    print(table1)
    print("=" * 80)

    # Generate Table 2
    table2 = generate_table2(model_data)
    with open(args.output_table2, "w") as f:
        f.write(table2)
    print(f"\nTable 2 saved to: {args.output_table2}")
    print("\n" + "=" * 80)
    print("TABLE 2: Per-Benchmark Performance")
    print("=" * 80)
    print(table2)
    print("=" * 80)


if __name__ == "__main__":
    main()

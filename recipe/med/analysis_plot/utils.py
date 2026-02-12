#!/usr/bin/env python3
"""
Utility functions for MED analysis and plotting.

This module contains commonly used functions for:
- Loading experiment data and baseline performance
- Smoothing curves with various methods
- Getting sorted benchmarks
- Plot styling
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ==============================================================================
# CONSTANTS
# ==============================================================================

RESULTS_DIR = Path("evals")
CSV_PATTERN = "**/*_results.csv"

BENCHMARK_ORDER = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]

# Model baseline performance mapping
MODEL_BASELINE_MAPPING = {
    "qwen25vl_instruct": "baseline/Qwen2.5-VL-7B-Instruct",
    "qwen3vl_instruct": "baseline/Qwen3-VL-8B-Instruct",
}


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================


def load_experiment_data(exp_name: str | None = None) -> dict[str, pd.DataFrame]:
    """Load experiment data from CSV files.

    Args:
        exp_name: Optional experiment name to load. If None, loads all experiments.
                  Must exactly match the directory name.

    Returns:
        Dictionary mapping experiment names to DataFrames.
    """
    experiment_data = {}

    if not RESULTS_DIR.exists():
        print(f"Results directory does not exist: {RESULTS_DIR}")
        return {}

    # If specific experiment requested, load only that one
    if exp_name is not None:
        exp_dir = RESULTS_DIR / exp_name
        if not exp_dir.exists():
            print(f"Experiment directory does not exist: {exp_dir}")
            return {}

        # CSV file name uses only the last part of the path
        exp_basename = Path(exp_name).name
        csv_file = exp_dir / f"{exp_basename}_results.csv"
        if not csv_file.exists():
            print(f"CSV file does not exist: {csv_file}")
            return {}

        try:
            df = pd.read_csv(csv_file)
            experiment_data[exp_name] = df
            print(f"Loaded experiment: {exp_name}")
        except Exception as e:
            print(f"Failed to load file {csv_file}: {e}")

        return experiment_data

    # Otherwise, load all experiments
    csv_files = list(RESULTS_DIR.glob(CSV_PATTERN))

    if not csv_files:
        print(f"No CSV files found in {RESULTS_DIR}")
        return {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            experiment_name = csv_file.parent.name
            experiment_data[experiment_name] = df
        except Exception as e:
            print(f"Failed to load file {csv_file}: {e}")

    return experiment_data


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

    return None


def load_baseline_performance(model_path: str) -> pd.DataFrame | None:
    """Load baseline performance for a given model.

    Args:
        model_path: Path to baseline model (e.g., "baseline/Qwen2.5-VL-7B-Instruct")

    Returns:
        DataFrame with baseline performance or None if not found
    """
    baseline_dir = RESULTS_DIR / model_path

    if not baseline_dir.exists():
        return None

    # Look for the CSV file with pattern {model_name}_results.csv
    model_name = Path(model_path).name
    csv_file = baseline_dir / f"{model_name}_results.csv"

    if not csv_file.exists():
        return None

    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"Error loading baseline CSV {csv_file}: {e}")
        return None


def merge_baseline_performance(
    exp_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> pd.DataFrame:
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
    min_baseline_step = (
        baseline_df["step"].min() if "step" in baseline_df.columns else 0
    )

    # Filter baseline to get only the earliest step
    baseline_step0 = baseline_df[baseline_df["step"] == min_baseline_step].copy()

    # Set step to 0 for baseline data
    baseline_step0["step"] = 0

    # Check if exp_df already has step=0 data
    has_step0 = (exp_df["step"] == 0).any() if "step" in exp_df.columns else False

    if has_step0:
        return exp_df

    # Concatenate baseline at the beginning
    combined_df = pd.concat([baseline_step0, exp_df], ignore_index=True)

    # Sort by step and benchmark
    combined_df = combined_df.sort_values(["benchmark", "step"]).reset_index(drop=True)

    return combined_df


def load_experiment_with_baseline(exp_name: str) -> dict[str, pd.DataFrame]:
    """Load experiment data and merge with baseline performance.

    Args:
        exp_name: Experiment name

    Returns:
        Dictionary with experiment data (with baseline merged if available)
    """
    # Load experiment data
    experiment_data = load_experiment_data(exp_name)
    if exp_name not in experiment_data:
        return {}

    exp_df = experiment_data[exp_name]

    # Identify model and load baseline
    model_prefix = identify_model_from_exp_name(exp_name)
    if model_prefix is None:
        return experiment_data

    model_path = MODEL_BASELINE_MAPPING[model_prefix]
    baseline_df = load_baseline_performance(model_path)

    if baseline_df is None:
        return experiment_data

    # Merge baseline into experiment
    exp_df = merge_baseline_performance(exp_df, baseline_df)
    experiment_data[exp_name] = exp_df

    return experiment_data


# ==============================================================================
# SMOOTHING FUNCTIONS
# ==============================================================================


def time_weighted_ema(
    y_values: np.ndarray, steps: np.ndarray, smoothing_factor: float = 0.1
) -> np.ndarray:
    """Apply time-weighted exponential moving average smoothing.

    Args:
        y_values: Values to smooth
        steps: Time steps corresponding to y_values
        smoothing_factor: Smoothing parameter (0-1)

    Returns:
        Smoothed values
    """
    if len(y_values) <= 1:
        return y_values.astype(float)

    smoothing_weight = min(np.sqrt(smoothing_factor), 0.999)
    range_of_x = steps[-1] - steps[0] if len(steps) > 1 else 1.0
    if range_of_x == 0:
        range_of_x = 1.0

    VIEWPORT_SCALE = 1000.0

    result = np.zeros_like(y_values, dtype=float)
    last_y = 0.0
    debias_weight = 0.0

    for i in range(len(y_values)):
        prev_x = steps[i - 1] if i > 0 else steps[0]
        change_in_x = ((steps[i] - prev_x) / range_of_x) * VIEWPORT_SCALE
        smoothing_weight_adj = smoothing_weight**change_in_x
        last_y = last_y * smoothing_weight_adj + y_values[i]
        debias_weight = debias_weight * smoothing_weight_adj + 1.0
        result[i] = last_y / debias_weight

    return result


def exponential_moving_average(
    y_values: np.ndarray, smoothing_factor: float = 0.1
) -> np.ndarray:
    """Apply standard exponential moving average smoothing.

    Args:
        y_values: Values to smooth
        smoothing_factor: Smoothing parameter (0-1)

    Returns:
        Smoothed values
    """
    if len(y_values) <= 1:
        return y_values

    result = np.zeros_like(y_values)
    result[0] = y_values[0]

    alpha = smoothing_factor
    for i in range(1, len(y_values)):
        result[i] = alpha * y_values[i] + (1 - alpha) * result[i - 1]

    return result


def smooth_curve(
    y_values: np.ndarray,
    steps: np.ndarray | None = None,
    smoothing_factor: float = 0.1,
    method: str = "time_weighted_ema",
) -> np.ndarray:
    """Apply smoothing to a curve using specified method.

    Args:
        y_values: Values to smooth
        steps: Time steps (required for time_weighted_ema)
        smoothing_factor: Smoothing parameter
        method: Smoothing method ("none", "time_weighted_ema", "ema", "savgol")

    Returns:
        Smoothed values
    """
    if smoothing_factor == 0 or len(y_values) < 2:
        return y_values

    if method == "none":
        return y_values
    elif method == "time_weighted_ema":
        if steps is None:
            return exponential_moving_average(y_values, smoothing_factor)
        return time_weighted_ema(y_values, steps, smoothing_factor)
    elif method == "ema":
        return exponential_moving_average(y_values, smoothing_factor)
    elif method == "savgol":
        if len(y_values) < 5:
            return y_values
        window_length = max(
            5, min(len(y_values) // 3, int(len(y_values) * smoothing_factor))
        )
        if window_length % 2 == 0:
            window_length += 1
        try:
            return savgol_filter(y_values, window_length, 3)
        except:
            return y_values
    else:
        return smooth_curve(y_values, steps, smoothing_factor, "savgol")


# ==============================================================================
# BENCHMARK UTILITIES
# ==============================================================================


def get_sorted_benchmarks(df):
    """Get benchmarks sorted by predefined order, excluding those not in BENCHMARK_ORDER.

    Args:
        df: DataFrame containing a 'benchmark' column

    Returns:
        List of benchmark names sorted by BENCHMARK_ORDER
    """
    all_benchmarks = df["benchmark"].unique()

    filtered_benchmarks = [b for b in all_benchmarks if b in BENCHMARK_ORDER]

    def sort_key(benchmark):
        return BENCHMARK_ORDER.index(benchmark)

    return sorted(filtered_benchmarks, key=sort_key)


# ==============================================================================
# PLOT STYLING
# ==============================================================================


def setup_plot_style():
    """Configure common matplotlib style settings."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 400
    plt.rcParams["savefig.dpi"] = 400
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

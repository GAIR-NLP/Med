from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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


def time_weighted_ema(
    y_values: np.ndarray, steps: np.ndarray, smoothing_factor: float = 0.1
) -> np.ndarray:
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


def get_sorted_benchmarks(df):
    """Get benchmarks sorted by predefined order, excluding those not in BENCHMARK_ORDER."""
    all_benchmarks = df["benchmark"].unique()

    filtered_benchmarks = [b for b in all_benchmarks if b in BENCHMARK_ORDER]

    def sort_key(benchmark):
        return BENCHMARK_ORDER.index(benchmark)

    return sorted(filtered_benchmarks, key=sort_key)


def create_subplot_grid(n_benchmarks):
    """Create subplot grid based on number of benchmarks."""
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        dpi=400,
        facecolor="white",
        edgecolor="none",
    )
    fig.patch.set_facecolor("white")

    if n_benchmarks == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    return fig, axes, n_rows, n_cols


def style_axis(ax, benchmark, ylabel, legend_loc="best"):
    """Apply common styling to an axis."""
    from matplotlib.ticker import MultipleLocator

    ax.set_title(
        f"{benchmark}", fontsize=16, fontweight="bold", color="#2C3E50", pad=20
    )
    ax.set_xlabel("Training Step", fontsize=12, color="#34495E", fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=12, color="#34495E", fontweight="medium")
    ax.grid(True, alpha=0.4, linewidth=0.8, color="#BDC3C7")
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#7F8C8D")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        colors="#2C3E50",
        width=1,
        length=4,
    )

    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.legend(loc=legend_loc, fontsize=10, frameon=True, fancybox=True, shadow=True)


def hide_empty_subplots(axes, n_benchmarks, n_rows, n_cols):
    """Hide empty subplots in the grid."""
    for idx in range(n_benchmarks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)


def save_figure(fig, output_dir, filename, subfolder=None):
    """Save figure to file.

    Args:
        fig: Matplotlib figure to save
        output_dir: Base output directory
        filename: Name of the file to save
        subfolder: Optional subfolder within output_dir (e.g., 'aggregated_curves')
    """
    output_path = Path(output_dir)

    # Add subfolder if specified
    if subfolder is not None:
        output_path = output_path / subfolder

    filepath = output_path / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath,
        format="png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )

    print(f"Plot saved to: {filepath}")
    plt.close(fig)
    return filepath


def plot_experiment_curves(
    experiment_name: str,
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "images",
    show_area: bool = False,
    relative_to_start: bool = True,
    experiment_data: dict[str, pd.DataFrame] | None = None,
):
    if experiment_data is None:
        experiment_data = load_experiment_data(experiment_name)

    if experiment_name not in experiment_data:
        print(f"Experiment '{experiment_name}' not found in data")
        print(f"Available experiments: {list(experiment_data.keys())}")
        return

    df = experiment_data[experiment_name]
    selected_benchmarks = get_sorted_benchmarks(df)

    setup_plot_style()
    n_benchmarks = len(selected_benchmarks)
    fig, axes, n_rows, n_cols = create_subplot_grid(n_benchmarks)

    line_color = "black"
    positive_area_color = "green"
    negative_area_color = "red"

    y_axis_settings = {}
    for benchmark in selected_benchmarks:
        benchmark_data = df[df["benchmark"] == benchmark]
        if benchmark_data.empty:
            continue

        all_values = []
        w_tool_col, wo_tool_col = "w_tool_score", "wo_tool_score"

        if w_tool_col in benchmark_data.columns:
            values = benchmark_data[w_tool_col].dropna().values
            if relative_to_start and len(values) > 0:
                values = values - values[0]
            all_values.extend(values.tolist())

        if wo_tool_col in benchmark_data.columns:
            values = benchmark_data[wo_tool_col].dropna().values
            if relative_to_start and len(values) > 0:
                values = values - values[0]
            all_values.extend(values.tolist())

        if all_values:
            data_min = min(all_values)
            data_max = max(all_values)
            padding = (data_max - data_min) * 0.05

            if relative_to_start:
                y_axis_settings[benchmark] = {
                    "min": min(0, data_min - padding),
                    "max": max(0, data_max + padding),
                }
            else:
                y_axis_settings[benchmark] = {
                    "min": max(0, data_min - padding),
                    "max": min(1, data_max + padding),
                }

    for idx, benchmark in enumerate(selected_benchmarks):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            continue

        benchmark_data = benchmark_data.sort_values("step")
        steps = benchmark_data["step"].values

        w_tool_col = "w_tool_score"
        wo_tool_col = "wo_tool_score"
        ylabel = (
            "Accuracy Change (Relative to Start)" if relative_to_start else "Accuracy"
        )

        w_tool_values_stored = None
        wo_tool_values_stored = None
        w_tool_values_dense = None
        wo_tool_values_dense = None
        steps_dense = None

        if w_tool_col in benchmark_data.columns:
            w_tool_values = benchmark_data[w_tool_col].values
            w_tool_values_raw = w_tool_values.copy()

            if relative_to_start:
                w_tool_values = w_tool_values - w_tool_values[0]

            if smoothing_factor > 0 or smoothing_method != "none":
                w_tool_values = smooth_curve(
                    w_tool_values, steps, smoothing_factor, smoothing_method
                )

            w_tool_values_stored = w_tool_values.copy()

            num_interp_points = max(200, len(steps) * 10)
            steps_dense = np.linspace(steps[0], steps[-1], num_interp_points)
            w_tool_values_dense = np.interp(steps_dense, steps, w_tool_values)

            ax.plot(
                steps_dense,
                w_tool_values_dense,
                color=line_color,
                linestyle="-",
                linewidth=2.5,
                label="w/ tool",
                alpha=0.8,
                zorder=3,
            )

            ax.scatter(
                steps,
                w_tool_values,
                color=line_color,
                marker="o",
                s=25,
                edgecolors="white",
                linewidths=1,
                alpha=0.8,
                zorder=4,
            )

            if relative_to_start:
                np.trapezoid(w_tool_values, steps)
            else:
                np.trapezoid(w_tool_values - w_tool_values[0], steps)

        if wo_tool_col in benchmark_data.columns:
            wo_tool_values = benchmark_data[wo_tool_col].values
            wo_tool_values_raw = wo_tool_values.copy()

            if relative_to_start:
                wo_tool_values = wo_tool_values - wo_tool_values[0]

            if smoothing_factor > 0 or smoothing_method != "none":
                wo_tool_values = smooth_curve(
                    wo_tool_values, steps, smoothing_factor, smoothing_method
                )

            wo_tool_values_stored = wo_tool_values.copy()

            num_interp_points = max(200, len(steps) * 10)
            steps_dense = np.linspace(steps[0], steps[-1], num_interp_points)
            wo_tool_values_dense = np.interp(steps_dense, steps, wo_tool_values)

            ax.plot(
                steps_dense,
                wo_tool_values_dense,
                color=line_color,
                linestyle="--",
                linewidth=2.5,
                label="w/o tool",
                alpha=0.7,
                zorder=2,
            )

            ax.scatter(
                steps,
                wo_tool_values,
                color=line_color,
                marker="s",
                s=16,
                edgecolors="white",
                linewidths=1,
                alpha=0.7,
                zorder=3,
            )

            # Fill area between w/o tool curve and x-axis with hatch pattern (only when relative_to_start)
            if relative_to_start:
                ax.fill_between(
                    steps_dense,
                    0,
                    wo_tool_values_dense,
                    facecolor="none",
                    edgecolor="gray",
                    hatch="///",
                    alpha=0.4,
                    linewidth=0,
                    zorder=0.5,
                    label="$B_{\\text{base}}$",
                )

        if w_tool_values_stored is not None and wo_tool_values_stored is not None:
            # Calculate difference: if relative_to_start, compute after each subtracts its own starting point
            if relative_to_start:
                w_tool_relative = w_tool_values_raw - w_tool_values_raw[0]
                wo_tool_relative = wo_tool_values_raw - wo_tool_values_raw[0]
                diff_values_raw = w_tool_relative - wo_tool_relative
            else:
                diff_values_raw = w_tool_values_raw - wo_tool_values_raw

            positive_area = np.trapezoid(np.maximum(diff_values_raw, 0), steps)
            negative_area = np.trapezoid(np.minimum(diff_values_raw, 0), steps)
            positive_area + negative_area

            # Calculate |B_base| - absolute change in base capability
            if relative_to_start:
                abs_base_area = np.trapezoid(
                    np.abs(wo_tool_values_raw - wo_tool_values_raw[0]), steps
                )
            else:
                abs_base_area = np.trapezoid(
                    np.abs(wo_tool_values_raw - wo_tool_values_raw[0]), steps
                )

            # Calculate |B_tool| = B_tool+ + |B_tool-|
            abs_tool_area = positive_area + abs(negative_area)

            # Calculate tool sensitivity score S_tool
            total_change = abs_base_area + abs_tool_area
            S_tool = abs_tool_area / total_change if total_change > 0 else 0

            if show_area and relative_to_start:
                call_rate = None
                if (
                    "call_num" in benchmark_data.columns
                    and "call_sum" in benchmark_data.columns
                ):
                    call_num = benchmark_data["call_num"].values
                    call_sum = benchmark_data["call_sum"].values
                    w_tool_total = benchmark_data["w_tool_total"].values
                    wo_tool_total = benchmark_data["wo_tool_total"].values
                    assert np.all(w_tool_total == wo_tool_total)
                    call_rate = call_num / w_tool_total
                    call_sum / (call_num + 1e-6)

                if (
                    steps_dense is not None
                    and w_tool_values_dense is not None
                    and wo_tool_values_dense is not None
                ):
                    w_tool_interp = w_tool_values_dense
                    wo_tool_interp = wo_tool_values_dense
                    steps_interp = steps_dense
                else:
                    num_interp_points = max(200, len(steps) * 10)
                    steps_interp = np.linspace(steps[0], steps[-1], num_interp_points)
                    w_tool_interp = np.interp(steps_interp, steps, w_tool_values_stored)
                    wo_tool_interp = np.interp(
                        steps_interp, steps, wo_tool_values_stored
                    )

                diff_interp = w_tool_interp - wo_tool_interp

                if call_rate is not None:
                    call_rate_interp = np.interp(steps_interp, steps, call_rate)
                    alpha_values = 0.1 + call_rate_interp * 0.7
                else:
                    alpha_values = np.ones(num_interp_points) * 0.3
                positive_labeled = False
                negative_labeled = False
                for i in range(len(steps_interp) - 1):
                    x_fill = [
                        steps_interp[i],
                        steps_interp[i + 1],
                        steps_interp[i + 1],
                        steps_interp[i],
                    ]
                    y_fill = [
                        wo_tool_interp[i],
                        wo_tool_interp[i + 1],
                        w_tool_interp[i + 1],
                        w_tool_interp[i],
                    ]

                    alpha = alpha_values[i]

                    if diff_interp[i] >= 0 and diff_interp[i + 1] >= 0:
                        if not positive_labeled:
                            ax.fill(
                                x_fill,
                                y_fill,
                                color=positive_area_color,
                                alpha=alpha,
                                edgecolor="none",
                                zorder=1.5,
                                label="$B_{\\text{tool}}^{+}$",
                            )
                            positive_labeled = True
                        else:
                            ax.fill(
                                x_fill,
                                y_fill,
                                color=positive_area_color,
                                alpha=alpha,
                                edgecolor="none",
                                zorder=1.5,
                            )
                    elif diff_interp[i] < 0 and diff_interp[i + 1] < 0:
                        if not negative_labeled:
                            ax.fill(
                                x_fill,
                                y_fill,
                                color=negative_area_color,
                                alpha=alpha,
                                edgecolor="none",
                                zorder=1.5,
                                label="$B_{\\text{tool}}^{-}$",
                            )
                            negative_labeled = True
                        else:
                            ax.fill(
                                x_fill,
                                y_fill,
                                color=negative_area_color,
                                alpha=alpha,
                                edgecolor="none",
                                zorder=1.5,
                            )

                if call_rate is not None:
                    from matplotlib.patches import Rectangle
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                    cax = inset_axes(
                        ax,
                        width="3%",
                        height="90%",
                        loc="center right",
                        bbox_to_anchor=(0.05, 0, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0,
                    )

                    n_colors = 100

                    for i in range(n_colors):
                        y_pos = 0.5 + i / (2 * n_colors)
                        call_rate_val = i / (n_colors - 1)
                        alpha_val = 0.1 + call_rate_val * 0.7

                        cax.add_patch(
                            Rectangle(
                                (0, y_pos),
                                1,
                                1 / (2 * n_colors),
                                facecolor="green",
                                alpha=alpha_val,
                                edgecolor="none",
                            )
                        )

                    for i in range(n_colors):
                        y_pos = (n_colors - 1 - i) / (2 * n_colors)
                        call_rate_val = i / (n_colors - 1)
                        alpha_val = 0.1 + call_rate_val * 0.7

                        cax.add_patch(
                            Rectangle(
                                (0, y_pos),
                                1,
                                1 / (2 * n_colors),
                                facecolor="red",
                                alpha=alpha_val,
                                edgecolor="none",
                            )
                        )

                    cax.set_xlim(0, 1)
                    cax.set_ylim(0, 1)
                    cax.set_xticks([])
                    cax.set_yticks([0, 0.5, 1])
                    cax.set_yticklabels(["1", "0", "1"], fontsize=8)
                    cax.yaxis.tick_right()
                    cax.set_ylabel("Call Rate", fontsize=9, rotation=270, labelpad=15)
                    cax.yaxis.set_label_position("right")

                    cax.axhline(y=0.5, color="gray", linewidth=1, alpha=0.5)

            if relative_to_start:
                # Add progress bar below title (matching aggregated curves style)
                from matplotlib.patches import Rectangle

                bar_height = 0.018
                bar_y_position = 0.96
                bar_width = 0.3
                bar_x_start = 0.35

                # Calculate proportions and S_tool
                total_area = abs_base_area + abs_tool_area
                base_frac = abs_base_area / total_area if total_area > 0 else 0
                tool_pos_frac = positive_area / total_area if total_area > 0 else 0
                tool_neg_frac = abs(negative_area) / total_area if total_area > 0 else 0
                S_tool = abs_tool_area / total_area if total_area > 0 else 0

                # Background bar (white with black border)
                ax.add_patch(
                    Rectangle(
                        (bar_x_start, bar_y_position),
                        bar_width,
                        bar_height,
                        transform=ax.transAxes,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=1.5,
                        zorder=10,
                    )
                )

                # Base portion (left, hatch pattern)
                base_width = base_frac * bar_width
                ax.add_patch(
                    Rectangle(
                        (bar_x_start, bar_y_position),
                        base_width,
                        bar_height,
                        transform=ax.transAxes,
                        facecolor="none",
                        edgecolor="gray",
                        hatch="///",
                        linewidth=0,
                        zorder=11,
                    )
                )

                # Tool+ portion (green)
                tool_pos_width = tool_pos_frac * bar_width
                ax.add_patch(
                    Rectangle(
                        (bar_x_start + base_width, bar_y_position),
                        tool_pos_width,
                        bar_height,
                        transform=ax.transAxes,
                        facecolor="green",
                        edgecolor="none",
                        alpha=0.6,
                        zorder=11,
                    )
                )

                # Tool- portion (red)
                tool_neg_width = tool_neg_frac * bar_width
                ax.add_patch(
                    Rectangle(
                        (bar_x_start + base_width + tool_pos_width, bar_y_position),
                        tool_neg_width,
                        bar_height,
                        transform=ax.transAxes,
                        facecolor="red",
                        edgecolor="none",
                        alpha=0.6,
                        zorder=11,
                    )
                )

                # Add "Base" label on the left
                ax.text(
                    bar_x_start - 0.02,
                    bar_y_position + bar_height / 2,
                    "Base",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="center",
                    horizontalalignment="right",
                    fontweight="medium",
                )

                # Add "Tool" label on the right
                ax.text(
                    bar_x_start + bar_width + 0.02,
                    bar_y_position + bar_height / 2,
                    "Tool",
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="center",
                    horizontalalignment="left",
                    fontweight="medium",
                )

                # Add S_tool value above the bar
                ax.text(
                    bar_x_start + bar_width / 2,
                    bar_y_position + bar_height + 0.01,
                    f"$S_{{\\text{{tool}}}}$ = {S_tool:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontweight="bold",
                )

        if relative_to_start:
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        style_axis(ax, benchmark, ylabel, legend_loc="upper left")

        if benchmark in y_axis_settings:
            ax.set_ylim(
                y_axis_settings[benchmark]["min"], y_axis_settings[benchmark]["max"]
            )
        else:
            ax.set_ylim(0, 1)

    hide_empty_subplots(axes, n_benchmarks, n_rows, n_cols)
    plt.tight_layout()

    suffix = "_relative" if relative_to_start else ""
    filename = f"{experiment_name}_accuracy{suffix}.png"
    save_figure(fig, output_dir, filename, subfolder="experiment_curves")

    return fig


def plot_aggregated_curves(
    experiment_name: str,
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "images",
    experiment_data: dict[str, pd.DataFrame] | None = None,
):
    if experiment_data is None:
        experiment_data = load_experiment_data(experiment_name)

    if experiment_name not in experiment_data:
        print(f"Experiment '{experiment_name}' not found in data")
        print(f"Available experiments: {list(experiment_data.keys())}")
        return

    df = experiment_data[experiment_name]
    selected_benchmarks = get_sorted_benchmarks(df)

    reasoning_benchmarks = selected_benchmarks[:6]
    perception_benchmarks = selected_benchmarks[6:12]

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=400, facecolor="white")

    groups = [
        ("Visual Reasoning", reasoning_benchmarks, axes[0]),
        ("Visual Perception", perception_benchmarks, axes[1]),
    ]

    for group_name, benchmarks, ax in groups:
        all_normalized_w_tool = []  # Normalized raw data from each benchmark
        all_normalized_wo_tool = []  # Normalized raw data from each benchmark
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

            # Normalize raw data: relative to start
            w_tool_relative = w_tool_values_raw - w_tool_values_raw[0]
            wo_tool_relative = wo_tool_values_raw - wo_tool_values_raw[0]

            # Find max for normalization
            max_abs_w = np.max(np.abs(w_tool_relative))
            max_abs_wo = np.max(np.abs(wo_tool_relative))
            max_abs = max(max_abs_w, max_abs_wo)

            if max_abs > 0:
                w_tool_normalized = w_tool_relative / max_abs
                wo_tool_normalized = wo_tool_relative / max_abs
            else:
                w_tool_normalized = w_tool_relative
                wo_tool_normalized = wo_tool_relative

            # Store normalized raw data
            all_normalized_w_tool.append(w_tool_normalized)
            all_normalized_wo_tool.append(wo_tool_normalized)

            if (
                "call_num" in benchmark_data.columns
                and "w_tool_total" in benchmark_data.columns
            ):
                call_num = benchmark_data["call_num"].values
                w_tool_total = benchmark_data["w_tool_total"].values
                call_rate = call_num / w_tool_total
            else:
                call_rate = np.zeros_like(steps)

            all_call_rates.append(call_rate)

        if len(all_normalized_w_tool) == 0:
            continue

        # Aggregate normalized raw data
        aggregated_w_tool_raw = np.mean(all_normalized_w_tool, axis=0)
        aggregated_wo_tool_raw = np.mean(all_normalized_wo_tool, axis=0)
        aggregated_call_rate = (
            np.mean(all_call_rates, axis=0)
            if len(all_call_rates) > 0
            else np.zeros_like(common_steps)
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

        num_interp_points = max(200, len(common_steps) * 10)
        steps_dense = np.linspace(common_steps[0], common_steps[-1], num_interp_points)
        w_tool_dense = np.interp(steps_dense, common_steps, aggregated_w_tool)
        wo_tool_dense = np.interp(steps_dense, common_steps, aggregated_wo_tool)
        call_rate_dense = np.interp(steps_dense, common_steps, aggregated_call_rate)

        ax.plot(
            steps_dense,
            w_tool_dense,
            color="black",
            linestyle="-",
            linewidth=2.8,
            label="w/ tool",
            alpha=0.8,
            zorder=3,
        )

        ax.scatter(
            common_steps,
            aggregated_w_tool,
            color="black",
            marker="o",
            s=30,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.8,
            zorder=4,
        )

        ax.plot(
            steps_dense,
            wo_tool_dense,
            color="black",
            linestyle="--",
            linewidth=2.8,
            label="w/o tool",
            alpha=0.7,
            zorder=2,
        )

        ax.scatter(
            common_steps,
            aggregated_wo_tool,
            color="black",
            marker="s",
            s=25,
            edgecolors="white",
            linewidths=1.5,
            alpha=0.7,
            zorder=3,
        )

        ax.axhline(
            y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=1
        )

        # Fill area between w/o tool curve and x-axis with hatch pattern
        ax.fill_between(
            steps_dense,
            0,
            wo_tool_dense,
            facecolor="none",
            edgecolor="gray",
            hatch="///",
            alpha=0.4,
            linewidth=0,
            zorder=0.5,
            label="$B_{\\text{base}}$",
        )

        alpha_values = 0.1 + call_rate_dense * 0.7

        positive_labeled = False
        negative_labeled = False
        for i in range(len(steps_dense) - 1):
            x_fill = [
                steps_dense[i],
                steps_dense[i + 1],
                steps_dense[i + 1],
                steps_dense[i],
            ]
            y_fill = [
                wo_tool_dense[i],
                wo_tool_dense[i + 1],
                w_tool_dense[i + 1],
                w_tool_dense[i],
            ]

            alpha = alpha_values[i]

            if (
                w_tool_dense[i] >= wo_tool_dense[i]
                and w_tool_dense[i + 1] >= wo_tool_dense[i + 1]
            ):
                if not positive_labeled:
                    ax.fill(
                        x_fill,
                        y_fill,
                        color="green",
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                        label="$B_{\\text{tool}}^{+}$",
                    )
                    positive_labeled = True
                else:
                    ax.fill(
                        x_fill,
                        y_fill,
                        color="green",
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                    )
            elif (
                w_tool_dense[i] < wo_tool_dense[i]
                and w_tool_dense[i + 1] < wo_tool_dense[i + 1]
            ):
                if not negative_labeled:
                    ax.fill(
                        x_fill,
                        y_fill,
                        color="red",
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                        label="$B_{\\text{tool}}^{-}$",
                    )
                    negative_labeled = True
                else:
                    ax.fill(
                        x_fill,
                        y_fill,
                        color="red",
                        alpha=alpha,
                        edgecolor="none",
                        zorder=1.5,
                    )

        # Calculate S_tool for aggregated data using RAW aggregated data
        # diff_aggregated_raw = aggregated_w_tool_raw - aggregated_wo_tool_raw
        diff_aggregated_raw = aggregated_w_tool - aggregated_wo_tool
        positive_area_agg = np.trapezoid(
            np.maximum(diff_aggregated_raw, 0), common_steps
        )
        negative_area_agg = np.trapezoid(
            np.minimum(diff_aggregated_raw, 0), common_steps
        )
        abs_base_area_agg = np.trapezoid(np.abs(aggregated_wo_tool_raw), common_steps)
        abs_tool_area_agg = positive_area_agg + abs(negative_area_agg)
        total_change_agg = abs_base_area_agg + abs_tool_area_agg
        S_tool_agg = abs_tool_area_agg / total_change_agg if total_change_agg > 0 else 0

        ax.set_title(
            f"{group_name} (n={len(benchmarks)})",
            fontsize=16,
            fontweight="bold",
            color="#2C3E50",
            pad=40,  # Increased padding to make room for progress bar
        )

        # Add progress bar below title
        from matplotlib.patches import Rectangle

        bar_height = 0.018  # Narrower
        bar_y_position = 0.96
        bar_width = 0.3  # Narrower
        bar_x_start = 0.35

        # Calculate proportions
        total_area_agg = abs_base_area_agg + abs_tool_area_agg
        base_frac_agg = abs_base_area_agg / total_area_agg if total_area_agg > 0 else 0
        tool_pos_frac_agg = (
            positive_area_agg / total_area_agg if total_area_agg > 0 else 0
        )
        tool_neg_frac_agg = (
            abs(negative_area_agg) / total_area_agg if total_area_agg > 0 else 0
        )

        # Background bar (white with black border)
        ax.add_patch(
            Rectangle(
                (bar_x_start, bar_y_position),
                bar_width,
                bar_height,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
            )
        )

        # Base portion (left, hatch pattern)
        base_width = base_frac_agg * bar_width
        ax.add_patch(
            Rectangle(
                (bar_x_start, bar_y_position),
                base_width,
                bar_height,
                transform=ax.transAxes,
                facecolor="none",
                edgecolor="gray",
                hatch="///",
                linewidth=0,
                zorder=11,
            )
        )

        # Tool+ portion (green)
        tool_pos_width = tool_pos_frac_agg * bar_width
        ax.add_patch(
            Rectangle(
                (bar_x_start + base_width, bar_y_position),
                tool_pos_width,
                bar_height,
                transform=ax.transAxes,
                facecolor="green",
                edgecolor="none",
                alpha=0.6,
                zorder=11,
            )
        )

        # Tool- portion (red)
        tool_neg_width = tool_neg_frac_agg * bar_width
        ax.add_patch(
            Rectangle(
                (bar_x_start + base_width + tool_pos_width, bar_y_position),
                tool_neg_width,
                bar_height,
                transform=ax.transAxes,
                facecolor="red",
                edgecolor="none",
                alpha=0.6,
                zorder=11,
            )
        )

        # Add labels
        ax.text(
            bar_x_start - 0.02,
            bar_y_position + bar_height / 2,
            "Base",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="right",
            fontweight="medium",
        )
        ax.text(
            bar_x_start + bar_width + 0.02,
            bar_y_position + bar_height / 2,
            "Tool",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="left",
            fontweight="medium",
        )

        # Add S_tool value in the center
        ax.text(
            bar_x_start + bar_width / 2,
            bar_y_position + bar_height + 0.01,
            f"$S_{{\\text{{tool}}}}$ = {S_tool_agg:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            fontweight="bold",
        )

        ax.set_xlabel(
            "Training Step", fontsize=12, color="#34495E", fontweight="medium"
        )
        ax.set_ylabel(
            "Normalized Δ Performance",
            fontsize=12,
            color="#34495E",
            fontweight="medium",
        )
        ax.grid(True, alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#7F8C8D")

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            colors="#2C3E50",
            width=1,
            length=4,
        )

        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(20))

        ax.legend(loc="best", fontsize=10, frameon=True, fancybox=True, shadow=True)

        all_values = np.concatenate([aggregated_w_tool, aggregated_wo_tool])
        y_min = np.min(all_values)
        y_max = np.max(all_values)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else 0.1
        ax.set_ylim(y_min - padding, y_max + padding)

        if group_name == "Visual Perception":
            from matplotlib.patches import Rectangle
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            cax = inset_axes(
                ax,
                width="3%",
                height="90%",
                loc="center right",
                bbox_to_anchor=(0.05, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )

            n_colors = 100

            for i in range(n_colors):
                y_pos = 0.5 + i / (2 * n_colors)
                call_rate_val = i / (n_colors - 1)
                alpha_val = 0.1 + call_rate_val * 0.7

                cax.add_patch(
                    Rectangle(
                        (0, y_pos),
                        1,
                        1 / (2 * n_colors),
                        facecolor="green",
                        alpha=alpha_val,
                        edgecolor="none",
                    )
                )

            for i in range(n_colors):
                y_pos = (n_colors - 1 - i) / (2 * n_colors)
                call_rate_val = i / (n_colors - 1)
                alpha_val = 0.1 + call_rate_val * 0.7

                cax.add_patch(
                    Rectangle(
                        (0, y_pos),
                        1,
                        1 / (2 * n_colors),
                        facecolor="red",
                        alpha=alpha_val,
                        edgecolor="none",
                    )
                )

            cax.set_xlim(0, 1)
            cax.set_ylim(0, 1)
            cax.set_xticks([])
            cax.set_yticks([0, 0.5, 1])
            cax.set_yticklabels(["1", "0", "1"], fontsize=8)
            cax.yaxis.tick_right()
            cax.set_ylabel("Call Rate", fontsize=9, rotation=270, labelpad=15)
            cax.yaxis.set_label_position("right")

            cax.axhline(y=0.5, color="gray", linewidth=1, alpha=0.5)

    plt.tight_layout()

    filename = f"{experiment_name}_aggregated_performance.png"
    save_figure(fig, output_dir, filename, subfolder="aggregated_curves")

    return fig


def plot_call_metrics(
    experiment_name: str,
    smoothing_factor: float = 0.99,
    smoothing_method: str = "time_weighted_ema",
    output_dir: str = "images",
    experiment_data: dict[str, pd.DataFrame] | None = None,
):
    if experiment_data is None:
        experiment_data = load_experiment_data(experiment_name)

    if experiment_name not in experiment_data:
        print(f"Experiment '{experiment_name}' not found in data")
        print(f"Available experiments: {list(experiment_data.keys())}")
        return

    df = experiment_data[experiment_name]
    selected_benchmarks = get_sorted_benchmarks(df)

    setup_plot_style()
    n_benchmarks = len(selected_benchmarks)
    fig, axes, n_rows, n_cols = create_subplot_grid(n_benchmarks)

    call_rate_color = "#3498DB"
    avg_count_color = "#E67E22"

    for idx, benchmark in enumerate(selected_benchmarks):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]

        benchmark_data = df[df["benchmark"] == benchmark].copy()
        if benchmark_data.empty:
            continue

        benchmark_data = benchmark_data.sort_values("step")
        steps = benchmark_data["step"].values

        if (
            "call_num" not in benchmark_data.columns
            or "call_sum" not in benchmark_data.columns
        ):
            continue

        call_num = benchmark_data["call_num"].values
        call_sum = benchmark_data["call_sum"].values
        w_tool_total = benchmark_data["w_tool_total"].values

        call_rate = call_num / w_tool_total
        average_call_count = call_sum / (call_num + 1e-6)

        if smoothing_factor > 0 or smoothing_method != "none":
            call_rate = smooth_curve(
                call_rate, steps, smoothing_factor, smoothing_method
            )
            average_call_count = smooth_curve(
                average_call_count, steps, smoothing_factor, smoothing_method
            )

        ax.plot(
            steps,
            call_rate,
            color=call_rate_color,
            linestyle="-",
            linewidth=2.8,
            marker="o",
            markersize=6,
            label="Call Rate",
            markerfacecolor=call_rate_color,
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
            zorder=3,
        )

        ax.set_ylabel("Call Rate", fontsize=12, color="#34495E", fontweight="semibold")
        ax.tick_params(axis="y", labelcolor=call_rate_color, width=1.2)
        ax.spines["left"].set_edgecolor(call_rate_color)
        ax.spines["left"].set_alpha(0.3)

        ax2 = ax.twinx()
        ax2.plot(
            steps,
            average_call_count,
            color=avg_count_color,
            linestyle="-",
            linewidth=2.8,
            marker="D",
            markersize=5,
            label="Avg Call Count",
            markerfacecolor=avg_count_color,
            markeredgewidth=1.5,
            markeredgecolor="white",
            alpha=0.9,
            zorder=2,
        )

        ax2.set_ylabel(
            "Average Call Count", fontsize=12, color="#34495E", fontweight="semibold"
        )
        ax2.tick_params(axis="y", labelcolor=avg_count_color, width=1.2)
        ax2.spines["right"].set_edgecolor(avg_count_color)
        ax2.spines["right"].set_alpha(0.3)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(1.2)

        ax.set_title(
            f"{benchmark}", fontsize=16, fontweight="bold", color="#2C3E50", pad=20
        )
        ax.set_xlabel(
            "Training Step", fontsize=12, color="#34495E", fontweight="medium"
        )
        ax.grid(True, alpha=0.4, linewidth=0.8, color="#BDC3C7")
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("#7F8C8D")

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            colors="#2C3E50",
            width=1,
            length=4,
        )

        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(20))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

    hide_empty_subplots(axes, n_benchmarks, n_rows, n_cols)
    plt.tight_layout()

    filename = f"{experiment_name}_call_metrics.png"
    save_figure(fig, output_dir, filename, subfolder="call_metrics")

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot experiment performance curves")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
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
        default="area_images",
        help="Output directory (default: images)",
    )
    parser.add_argument(
        "--show_area",
        action="store_true",
        help="Show filled area under curves",
    )
    parser.add_argument(
        "--smooth_then_diff",
        action="store_true",
        help="Smooth first then calculate difference (default: diff then smooth)",
    )
    parser.add_argument(
        "--no_relative_to_start",
        action="store_true",
        help="Plot absolute values instead of relative to starting point",
    )

    args = parser.parse_args()

    experiment_data = load_experiment_with_baseline(args.experiment_name)

    plot_experiment_curves(
        args.experiment_name,
        args.smoothing_factor,
        args.smoothing_method,
        args.output_dir,
        args.show_area,
        relative_to_start=not args.no_relative_to_start,
        experiment_data=experiment_data,
    )

    plot_experiment_curves(
        args.experiment_name,
        args.smoothing_factor,
        args.smoothing_method,
        args.output_dir,
        show_area=False,
        relative_to_start=False,
        experiment_data=experiment_data,
    )

    plot_aggregated_curves(
        args.experiment_name,
        args.smoothing_factor,
        args.smoothing_method,
        args.output_dir,
        experiment_data=experiment_data,
    )

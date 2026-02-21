#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for aggregated metrics.

This script loads raw trajectory data and computes paired nonparametric bootstrap CIs:
- Paired: same sample index across checkpoints
- Stratified: resample within each benchmark independently
- Equal-weight aggregation: simple mean across 6 benchmarks
- 95% CI: percentile [2.5%, 97.5%]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Evaluation results directory
RESULTS_DIR = Path("../../../evaluation_results")

# Model baseline performance mapping (same as create_csv.py)
MODEL_BASELINE_MAPPING = {
    "qwen25vl_instruct": "verl_model/Qwen2.5-VL-7B-Instruct",
    "qwen3vl_instruct": "verl_model/Qwen3-VL-8B-Instruct",
    "qwen3vl_thinking": "verl_model/Qwen3-VL-8B-Thinking",
    "glm46v": "verl_model/GLM-4.6V-Flash",
}

# Default perception benchmarks
PERCEPTION_BENCHMARKS = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]


def load_trajectory_jsonl(jsonl_path: Path) -> list[dict]:
    """Load trajectory data from a JSONL file."""
    if not jsonl_path.exists():
        return []

    trajectories = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {jsonl_path}: {e}")
        return []

    return trajectories


def load_benchmark_trajectories(checkpoint_dir: Path, benchmark_name: str) -> dict[str, list[dict]]:
    """Load both single_turn and tool agent trajectories for a benchmark at a checkpoint.

    Args:
        checkpoint_dir: Directory for the checkpoint (e.g., .../global_step_0000010)
        benchmark_name: Name of the benchmark (e.g., 'charxiv2rq')

    Returns:
        Dict with keys 'wo_tool_trajectories' and 'w_tool_trajectories'
    """
    benchmark_dir = checkpoint_dir / benchmark_name
    trajectories_dir = benchmark_dir / "trajectories"

    if not trajectories_dir.exists():
        return {"wo_tool_trajectories": [], "w_tool_trajectories": []}

    # Path to trajectory files
    single_turn_path = (
        trajectories_dir / f"{benchmark_name}_bench_single_turn_agent_trajectories.jsonl"
    )
    tool_agent_path = trajectories_dir / f"{benchmark_name}_bench_tool_agent_trajectories.jsonl"

    return {
        "wo_tool_trajectories": load_trajectory_jsonl(single_turn_path),
        "w_tool_trajectories": load_trajectory_jsonl(tool_agent_path),
    }


def get_baseline_model_from_exp_name(exp_name: str) -> str | None:
    """Extract baseline model path from experiment name.

    Args:
        exp_name: Experiment name like "qwen25vl_instruct_75_50/natural_0.75_..."

    Returns:
        Baseline model path or None if not found
    """
    for prefix, baseline_path in MODEL_BASELINE_MAPPING.items():
        if exp_name.startswith(prefix):
            return baseline_path
    return None


def load_baseline_trajectories(
    evaluation_dir: Path, baseline_model_path: str, benchmarks: list[str]
) -> dict[str, dict[str, list[dict]]]:
    """Load baseline model's trajectories for each benchmark.

    Args:
        evaluation_dir: Evaluation directory
        baseline_model_path: Baseline model path (e.g., "verl_model/Qwen2.5-VL-7B-Instruct")
        benchmarks: List of benchmark names

    Returns:
        Dict mapping {benchmark: {'wo_tool_trajectories': [...], 'w_tool_trajectories': [...]}}
    """
    baseline_exp_path = evaluation_dir / baseline_model_path
    if not baseline_exp_path.exists():
        raise FileNotFoundError(f"Baseline path not found: {baseline_exp_path}")

    print(f"Loading baseline from: {baseline_model_path}")

    baseline_data = {}

    for benchmark in benchmarks:
        # Try direct benchmark directory (baseline models don't have global_step directories)
        benchmark_dir = baseline_exp_path / benchmark

        if benchmark_dir.exists():
            trajectories = load_benchmark_trajectories(baseline_exp_path, benchmark)
            if trajectories["wo_tool_trajectories"] and trajectories["w_tool_trajectories"]:
                baseline_data[benchmark] = trajectories
                print(f"  {benchmark}: {len(trajectories['wo_tool_trajectories'])} samples")
            else:
                print(f"  Warning: No trajectories found for {benchmark} in baseline")
        else:
            print(f"  Warning: {benchmark} not found in baseline")

    return baseline_data


def get_all_checkpoint_info(
    exp_name: str, evaluation_dir: Path = RESULTS_DIR
) -> tuple[str, list[tuple[int, Path]]]:
    """Get information about all checkpoints without loading data.

    Args:
        exp_name: Experiment name
        evaluation_dir: Base evaluation directory

    Returns:
        Tuple of (baseline_model_path, checkpoint_list)
        - baseline_model_path: Path to baseline model
        - checkpoint_list: [(step, checkpoint_dir), ...] sorted by step
    """
    exp_path = evaluation_dir / exp_name
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    # 1. Get baseline model path
    baseline_model_path = get_baseline_model_from_exp_name(exp_name)
    if not baseline_model_path:
        raise ValueError(f"No baseline model mapping found for experiment: {exp_name}")

    # 2. Find all checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith("global_step_")]
    )

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {exp_path}")

    checkpoint_list = []
    for ckpt_dir in checkpoint_dirs:
        step = int(ckpt_dir.name.replace("global_step_", ""))
        checkpoint_list.append((step, ckpt_dir))

    checkpoint_list = sorted(checkpoint_list, key=lambda x: x[0])

    return baseline_model_path, checkpoint_list


def load_checkpoint_data(
    checkpoint_dir: Path, benchmarks: list[str], is_baseline: bool = False
) -> dict[str, dict[str, list[dict]]]:
    """Load data for a single checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory
        benchmarks: List of benchmark names
        is_baseline: Whether this is a baseline model directory

    Returns:
        Dict mapping {benchmark: {'wo_tool_trajectories': [...], 'w_tool_trajectories': [...]}}
    """
    data = {}
    for benchmark in benchmarks:
        trajectories = load_benchmark_trajectories(checkpoint_dir, benchmark)

        if not trajectories["wo_tool_trajectories"] or not trajectories["w_tool_trajectories"]:
            if not is_baseline:
                print(f"    Warning: Missing trajectories for {benchmark}, skipping")
            continue

        data[benchmark] = trajectories

    return data


def get_num_samples(
    evaluation_dir: Path, baseline_model_path: str, benchmarks: list[str]
) -> dict[str, int]:
    """Get number of samples for each benchmark by loading baseline.

    Returns:
        Dict mapping {benchmark: num_samples}
    """
    baseline_path = evaluation_dir / baseline_model_path
    num_samples = {}

    for benchmark in benchmarks:
        trajectories = load_benchmark_trajectories(baseline_path, benchmark)
        if trajectories["wo_tool_trajectories"]:
            num_samples[benchmark] = len(trajectories["wo_tool_trajectories"])

    return num_samples


def load_all_trajectories(
    evaluation_dir: Path,
    baseline_model_path: str,
    checkpoint_list: list[tuple[int, Path]],
    benchmarks: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Load all checkpoint data into memory at once.

    Args:
        evaluation_dir: Base evaluation directory
        baseline_model_path: Path to baseline model
        checkpoint_list: [(step, checkpoint_dir), ...] for all checkpoints
        benchmarks: List of benchmark names

    Returns:
        y_wo: {benchmark: (T, N_b)} - tool-free correctness
        y_w: {benchmark: (T, N_b)} - tool-available correctness
        call: {benchmark: (T, N_b)} - whether tool was called
        steps: array of checkpoint steps [0, 10, 20, ..., 200]
    """
    # Prepare checkpoint info: [(step, checkpoint_dir), ...] with step 0 prepended
    baseline_path = evaluation_dir / baseline_model_path
    all_checkpoints = [(0, baseline_path)] + checkpoint_list
    steps = np.array([step for step, _ in all_checkpoints])
    T = len(steps)

    print(f"Loading {T} checkpoints: {steps}")

    y_wo = {}
    y_w = {}
    call = {}

    for benchmark in benchmarks:
        # First, get number of samples from baseline
        baseline_data = load_checkpoint_data(baseline_path, [benchmark], is_baseline=True)
        if benchmark not in baseline_data:
            print(f"  Warning: {benchmark} not found in baseline, skipping")
            continue

        N_b = len(baseline_data[benchmark]["wo_tool_trajectories"])

        y_wo_b = np.zeros((T, N_b), dtype=int)
        y_w_b = np.zeros((T, N_b), dtype=int)
        call_b = np.zeros((T, N_b), dtype=int)

        # Load all checkpoints for this benchmark
        for t_idx, (step, ckpt_dir) in enumerate(all_checkpoints):
            is_baseline = step == 0
            ckpt_data = load_checkpoint_data(ckpt_dir, [benchmark], is_baseline=is_baseline)

            if benchmark not in ckpt_data:
                print(f"    Warning: {benchmark} not found at step {step}")
                continue

            wo_tool_trajs = ckpt_data[benchmark]["wo_tool_trajectories"]
            w_tool_trajs = ckpt_data[benchmark]["w_tool_trajectories"]

            # Extract y_wo
            for i, traj in enumerate(wo_tool_trajs):
                acc_orig = traj.get("accuracy_reward_original", 0)
                acc_llm = traj.get("accuracy_reward_llm", 0)
                y_wo_b[t_idx, i] = 1 if (acc_orig == 1 or acc_llm == 1) else 0

            # Extract y_w and call
            for i, traj in enumerate(w_tool_trajs):
                acc_orig = traj.get("accuracy_reward_original", 0)
                acc_llm = traj.get("accuracy_reward_llm", 0)
                y_w_b[t_idx, i] = 1 if (acc_orig == 1 or acc_llm == 1) else 0

                tool_call_counts = traj.get("tool_call_counts", 0)
                call_b[t_idx, i] = 1 if tool_call_counts > 0 else 0

        y_wo[benchmark] = y_wo_b
        y_w[benchmark] = y_w_b
        call[benchmark] = call_b

        print(f"  {benchmark}: loaded {T} checkpoints × {N_b} samples")

    return y_wo, y_w, call, steps


def bootstrap_all_metrics(
    y_wo: dict[str, np.ndarray],
    y_w: dict[str, np.ndarray],
    call: dict[str, np.ndarray],
    steps: np.ndarray,
    benchmarks: list[str],
    B: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CI for all metrics in a single pass.

    For each bootstrap replicate:
    1. Generate resampling indices for each benchmark
    2. Compute entire curves Acc_wo[t], Acc_w[t] using same indices
    3. Compute all metrics from these curves:
       - Accuracy (init, final, delta)
       - S_tool (using all checkpoints)
       - P(✓|c,D_fail) (init, final, delta)
       - P(×|c,D_succ) (init, final, delta)

    Args:
        y_wo: {benchmark: (T, N_b)} - tool-free correctness
        y_w: {benchmark: (T, N_b)} - tool-available correctness
        call: {benchmark: (T, N_b)} - whether tool was called
        steps: array of checkpoint steps
        benchmarks: List of benchmark names
        B: Number of bootstrap replicates
        seed: Random seed

    Returns:
        Dict with all metrics and their CIs
    """
    rng = np.random.default_rng(seed)
    T = len(steps)
    t_init = 0
    t_final = T - 1

    # Storage for bootstrap replicates
    acc_wo_init_bs = []
    acc_wo_final_bs = []
    acc_w_init_bs = []
    acc_w_final_bs = []
    s_tool_bs = []
    p_acc_call_fail_init_bs = []
    p_acc_call_fail_final_bs = []
    p_err_call_succ_init_bs = []
    p_err_call_succ_final_bs = []

    for r in tqdm(range(B), desc="Bootstrap"):
        # 1. Generate resampling indices for each benchmark
        sample_indices = {}
        for benchmark in benchmarks:
            N_b = y_wo[benchmark].shape[1]
            sample_indices[benchmark] = rng.integers(0, N_b, size=N_b)

        # 2. Compute entire curves using same indices
        # For S_tool: need per-benchmark normalization
        Acc_wo_b_t = np.zeros((len(benchmarks), T))
        Acc_w_b_t = np.zeros((len(benchmarks), T))

        # Normalized drift for S_tool calculation
        drift_wo_b_t_normalized = np.zeros((len(benchmarks), T))
        drift_w_b_t_normalized = np.zeros((len(benchmarks), T))

        for b_idx, benchmark in enumerate(benchmarks):
            idx = sample_indices[benchmark]
            # For all t at once
            for t in range(T):
                Acc_wo_b_t[b_idx, t] = y_wo[benchmark][t, idx].mean()
                Acc_w_b_t[b_idx, t] = y_w[benchmark][t, idx].mean()

            # Calculate drift relative to t=0
            drift_wo = Acc_wo_b_t[b_idx, :] - Acc_wo_b_t[b_idx, 0]
            drift_w = Acc_w_b_t[b_idx, :] - Acc_w_b_t[b_idx, 0]

            # Find max absolute drift (use SAME scale for both w and wo)
            max_abs_wo = np.max(np.abs(drift_wo))
            max_abs_w = np.max(np.abs(drift_w))
            max_abs = max(max_abs_wo, max_abs_w)

            # Normalize to [-1, 1]
            if max_abs > 0:
                drift_wo_b_t_normalized[b_idx, :] = drift_wo / max_abs
                drift_w_b_t_normalized[b_idx, :] = drift_w / max_abs
            else:
                drift_wo_b_t_normalized[b_idx, :] = drift_wo
                drift_w_b_t_normalized[b_idx, :] = drift_w

        # 3. Aggregate across benchmarks (equal-weight)
        # For accuracy metrics: use raw values
        Acc_wo_t = Acc_wo_b_t.mean(axis=0)  # Shape: (T,)
        Acc_w_t = Acc_w_b_t.mean(axis=0)  # Shape: (T,)

        # For S_tool: use normalized drift
        drift_wo_t_normalized = drift_wo_b_t_normalized.mean(axis=0)  # Shape: (T,)
        drift_w_t_normalized = drift_w_b_t_normalized.mean(axis=0)  # Shape: (T,)

        # 4. Compute metrics for this replicate

        # 4.1 Accuracy
        acc_wo_init_bs.append(Acc_wo_t[t_init])
        acc_wo_final_bs.append(Acc_wo_t[t_final])
        acc_w_init_bs.append(Acc_w_t[t_init])
        acc_w_final_bs.append(Acc_w_t[t_final])

        # 4.2 S_tool (using normalized drift and all checkpoints)
        # f_wo: normalized wo_tool drift
        # delta_tool: normalized tool-induced change
        f_wo = drift_wo_t_normalized
        delta_tool = drift_w_t_normalized - drift_wo_t_normalized

        # Use trapz for integration (as in plot_paper_figures.py)
        B_wo = np.trapezoid(np.abs(f_wo), steps)
        B_tool = np.trapezoid(np.abs(delta_tool), steps)

        if B_wo + B_tool > 1e-10:
            s_tool_bs.append(B_tool / (B_wo + B_tool))
        else:
            s_tool_bs.append(np.nan)

        # 4.3 P(✓|c, D_fail) - init and final
        # Need to aggregate across benchmarks with masks
        for t_idx in [t_init, t_final]:
            p_per_b = []
            for benchmark in benchmarks:
                idx = sample_indices[benchmark]
                mask_fail = y_wo[benchmark][t_idx, idx] == 0
                mask_call = call[benchmark][t_idx, idx] == 1
                mask = mask_fail & mask_call

                if mask.sum() > 0:
                    p_per_b.append(y_w[benchmark][t_idx, idx][mask].mean())
                else:
                    p_per_b.append(np.nan)

            p_per_b = np.array(p_per_b)
            if np.isnan(p_per_b).all():
                p_agg = np.nan
            else:
                p_agg = np.nanmean(p_per_b)

            if t_idx == t_init:
                p_acc_call_fail_init_bs.append(p_agg)
            else:
                p_acc_call_fail_final_bs.append(p_agg)

        # 4.4 P(×|c, D_succ) - init and final
        for t_idx in [t_init, t_final]:
            q_per_b = []
            for benchmark in benchmarks:
                idx = sample_indices[benchmark]
                mask_succ = y_wo[benchmark][t_idx, idx] == 1
                mask_call = call[benchmark][t_idx, idx] == 1
                mask = mask_succ & mask_call

                if mask.sum() > 0:
                    q_per_b.append((1 - y_w[benchmark][t_idx, idx][mask]).mean())
                else:
                    q_per_b.append(np.nan)

            q_per_b = np.array(q_per_b)
            if np.isnan(q_per_b).all():
                q_agg = np.nan
            else:
                q_agg = np.nanmean(q_per_b)

            if t_idx == t_init:
                p_err_call_succ_init_bs.append(q_agg)
            else:
                p_err_call_succ_final_bs.append(q_agg)

    # Convert to arrays and compute CIs
    acc_wo_init_bs = np.array(acc_wo_init_bs)
    acc_wo_final_bs = np.array(acc_wo_final_bs)
    acc_w_init_bs = np.array(acc_w_init_bs)
    acc_w_final_bs = np.array(acc_w_final_bs)

    delta_wo_bs = acc_wo_final_bs - acc_wo_init_bs
    delta_w_bs = acc_w_final_bs - acc_w_init_bs

    s_tool_bs = np.array(s_tool_bs)
    s_tool_bs = s_tool_bs[~np.isnan(s_tool_bs)]

    p_acc_call_fail_init_bs = np.array(p_acc_call_fail_init_bs)
    p_acc_call_fail_final_bs = np.array(p_acc_call_fail_final_bs)
    p_acc_call_fail_init_clean = p_acc_call_fail_init_bs[~np.isnan(p_acc_call_fail_init_bs)]
    p_acc_call_fail_final_clean = p_acc_call_fail_final_bs[~np.isnan(p_acc_call_fail_final_bs)]

    valid_mask_p = ~np.isnan(p_acc_call_fail_init_bs) & ~np.isnan(p_acc_call_fail_final_bs)
    delta_p_bs = p_acc_call_fail_final_bs[valid_mask_p] - p_acc_call_fail_init_bs[valid_mask_p]

    p_err_call_succ_init_bs = np.array(p_err_call_succ_init_bs)
    p_err_call_succ_final_bs = np.array(p_err_call_succ_final_bs)
    p_err_call_succ_init_clean = p_err_call_succ_init_bs[~np.isnan(p_err_call_succ_init_bs)]
    p_err_call_succ_final_clean = p_err_call_succ_final_bs[~np.isnan(p_err_call_succ_final_bs)]

    valid_mask_q = ~np.isnan(p_err_call_succ_init_bs) & ~np.isnan(p_err_call_succ_final_bs)
    delta_q_bs = p_err_call_succ_final_bs[valid_mask_q] - p_err_call_succ_init_bs[valid_mask_q]

    # Helper function to compute CI with delta format
    def compute_ci_with_delta(data):
        mean = data.mean()
        ci_low = np.percentile(data, 2.5)
        ci_high = np.percentile(data, 97.5)
        delta_lower = mean - ci_low  # How much below mean
        delta_upper = ci_high - mean  # How much above mean
        return {
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "delta_lower": delta_lower,
            "delta_upper": delta_upper,
        }

    return {
        "accuracy": {
            "wo_tool": {
                "init": compute_ci_with_delta(acc_wo_init_bs),
                "final": compute_ci_with_delta(acc_wo_final_bs),
                "delta": compute_ci_with_delta(delta_wo_bs),
            },
            "w_tool": {
                "init": compute_ci_with_delta(acc_w_init_bs),
                "final": compute_ci_with_delta(acc_w_final_bs),
                "delta": compute_ci_with_delta(delta_w_bs),
            },
        },
        "s_tool": compute_ci_with_delta(s_tool_bs),
        "p_acc_call_fail": {
            "init": compute_ci_with_delta(p_acc_call_fail_init_clean),
            "final": compute_ci_with_delta(p_acc_call_fail_final_clean),
            "delta": compute_ci_with_delta(delta_p_bs),
        },
        "p_err_call_succ": {
            "init": compute_ci_with_delta(p_err_call_succ_init_clean),
            "final": compute_ci_with_delta(p_err_call_succ_final_clean),
            "delta": compute_ci_with_delta(delta_q_bs),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for aggregated metrics"
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment name (e.g., qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew)",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=None,
        help="List of benchmarks (default: perception benchmarks)",
    )
    parser.add_argument(
        "--evaluation_dir", type=str, default=None, help="Evaluation results directory"
    )
    parser.add_argument(
        "--B", type=int, default=1000, help="Number of bootstrap replicates (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: ci_results/<exp_name>_ci.json)",
    )

    args = parser.parse_args()

    benchmarks = args.benchmarks if args.benchmarks else PERCEPTION_BENCHMARKS
    evaluation_dir = Path(args.evaluation_dir) if args.evaluation_dir else RESULTS_DIR

    print(f"\n{'='*80}")
    print("Computing Bootstrap Confidence Intervals")
    print(f"{'='*80}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Bootstrap replicates: {args.B}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")

    # Get checkpoint information
    print("Getting checkpoint information...")
    baseline_model_path, checkpoint_list = get_all_checkpoint_info(
        args.experiment_name, evaluation_dir
    )

    print(f"Baseline model: {baseline_model_path}")
    print(f"Found {len(checkpoint_list)} checkpoints")
    print(f"Steps: {[step for step, _ in checkpoint_list]}\n")

    # Load all checkpoint data into memory
    print("Loading all checkpoints into memory...")
    y_wo, y_w, call, steps = load_all_trajectories(
        evaluation_dir, baseline_model_path, checkpoint_list, benchmarks
    )

    print(f"\nLoaded {len(steps)} checkpoints: {steps}")
    print("Sample counts:")
    for b in benchmarks:
        if b in y_wo:
            print(f"  {b}: {y_wo[b].shape[1]} samples")
    print()

    # Compute all CIs in a single pass
    print("Computing bootstrap CIs for all metrics...")
    results = bootstrap_all_metrics(y_wo, y_w, call, steps, benchmarks, args.B, args.seed)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("ci_results")
        output_dir.mkdir(exist_ok=True)
        exp_name_clean = args.experiment_name.replace("/", "_")
        output_path = output_dir / f"{exp_name_clean}_ci.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    # Print summary with delta format
    def format_result(r):
        """Format result as: mean ± delta (or +upper/-lower if asymmetric)"""
        mean = r["mean"]
        delta_lower = r["delta_lower"]
        delta_upper = r["delta_upper"]

        # Check if symmetric (within tolerance)
        if abs(delta_lower - delta_upper) < 0.001:
            return f"{mean:.4f} ± {delta_upper:.4f}"
        else:
            return f"{mean:.4f} +{delta_upper:.4f}/-{delta_lower:.4f}"

    print("Summary:")
    print("  Accuracy (wo_tool):")
    print(f"    Init:  {format_result(results['accuracy']['wo_tool']['init'])}")
    print(f"    Final: {format_result(results['accuracy']['wo_tool']['final'])}")
    print(f"    Delta: {format_result(results['accuracy']['wo_tool']['delta'])}")

    print("\n  Accuracy (w_tool):")
    print(f"    Init:  {format_result(results['accuracy']['w_tool']['init'])}")
    print(f"    Final: {format_result(results['accuracy']['w_tool']['final'])}")
    print(f"    Delta: {format_result(results['accuracy']['w_tool']['delta'])}")

    print(f"\n  S_tool: {format_result(results['s_tool'])}")

    print("\n  P(✓|c,D_fail):")
    print(f"    Init:  {format_result(results['p_acc_call_fail']['init'])}")
    print(f"    Final: {format_result(results['p_acc_call_fail']['final'])}")
    print(f"    Delta: {format_result(results['p_acc_call_fail']['delta'])}")

    print("\n  P(×|c,D_succ):")
    print(f"    Init:  {format_result(results['p_err_call_succ']['init'])}")
    print(f"    Final: {format_result(results['p_err_call_succ']['final'])}")
    print(f"    Delta: {format_result(results['p_err_call_succ']['delta'])}")


if __name__ == "__main__":
    main()

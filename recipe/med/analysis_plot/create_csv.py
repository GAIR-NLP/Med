#!/usr/bin/env python3
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm
from utils import BENCHMARK_ORDER

# Model baseline performance mapping
MODEL_BASELINE_MAPPING = {
    "qwen25vl_instruct": "baseline/Qwen2.5-VL-7B-Instruct",
    "qwen3vl_instruct": "baseline/Qwen3-VL-8B-Instruct",
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


def extract_step_from_path(path: str) -> int:
    """Extract step number from global_step_XXXXXXX directory name."""
    parts = path.split("/")
    for part in parts:
        if part.startswith("global_step_"):
            step_str = part.replace("global_step_", "")
            return int(step_str)
    return 0


def load_trajectory_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Load trajectory data from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file

    Returns:
        list of trajectory records (each line is a dict)
    """
    if not os.path.exists(jsonl_path):
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


def load_benchmark_trajectories(
    benchmark_dir: str, benchmark_name: str
) -> dict[str, list[dict[str, Any]]]:
    """Load both single_turn and tool agent trajectories for a benchmark.

    Args:
        benchmark_dir: Directory containing the benchmark data (e.g., .../global_step_XXX/vstar)
        benchmark_name: Name of the benchmark (e.g., 'vstar')

    Returns:
        dict with keys 'wo_tool_trajectories' and 'w_tool_trajectories'
    """
    trajectories_dir = os.path.join(benchmark_dir, "trajectories")

    if not os.path.exists(trajectories_dir):
        return {"wo_tool_trajectories": [], "w_tool_trajectories": []}

    single_turn_path = os.path.join(
        trajectories_dir, f"{benchmark_name}_bench_single_turn_agent_trajectories.jsonl"
    )
    tool_agent_path = os.path.join(
        trajectories_dir, f"{benchmark_name}_bench_tool_agent_trajectories.jsonl"
    )

    return {
        "wo_tool_trajectories": load_trajectory_jsonl(single_turn_path),
        "w_tool_trajectories": load_trajectory_jsonl(tool_agent_path),
    }


def analyze_trajectories(
    trajectories: dict[str, list[dict[str, Any]]],
    baseline_correct_indices: set | None = None,
    baseline_incorrect_indices: set | None = None,
) -> dict[str, Any]:
    """Analyze trajectories and compute metrics.

    Args:
        trajectories: dict with 'wo_tool_trajectories' and 'w_tool_trajectories'
        baseline_correct_indices: Optional set of indices that were correct in baseline (step 0).
                                  If provided, use this as fixed D_succ.
        baseline_incorrect_indices: Optional set of indices that were incorrect in baseline (step 0).
                                    If provided along with baseline_correct_indices, use both for intersection partition.
                                    If None but baseline_correct_indices is provided, compute as complement.

    Returns:
        dict containing computed metrics
    """
    wo_tool_trajs = trajectories["wo_tool_trajectories"]
    w_tool_trajs = trajectories["w_tool_trajectories"]

    # Determine D_succ and D_fail
    if baseline_correct_indices is not None:
        # Use fixed baseline划分
        wo_tool_correct_indices = baseline_correct_indices
        total = len(wo_tool_trajs)

        if baseline_incorrect_indices is not None:
            # Both provided: use explicit D_fail (for intersection partition)
            wo_tool_incorrect_indices = baseline_incorrect_indices
        else:
            # Only correct provided: compute D_fail as complement
            wo_tool_incorrect_indices = set(range(total)) - baseline_correct_indices
    else:
        # Use current checkpoint's w/o tool performance (original behavior)
        wo_tool_correct_indices = set()
        wo_tool_incorrect_indices = set()

        for idx, traj in enumerate(wo_tool_trajs):
            accuracy = traj.get("accuracy_reward", 0)

            if accuracy == 1:
                wo_tool_correct_indices.add(idx)
            else:
                wo_tool_incorrect_indices.add(idx)

    w_tool_correct_indices = set()
    w_tool_incorrect_indices = set()
    w_tool_call_indices = set()
    w_tool_no_call_indices = set()
    call_sum = 0

    for idx, traj in enumerate(w_tool_trajs):
        accuracy = traj.get("accuracy_reward", 0)
        tool_call_counts = traj.get("tool_call_counts", 0)

        if accuracy == 1:
            w_tool_correct_indices.add(idx)
        else:
            w_tool_incorrect_indices.add(idx)

        if tool_call_counts > 0:
            w_tool_call_indices.add(idx)
        else:
            w_tool_no_call_indices.add(idx)

        call_sum += tool_call_counts

    wo_tool_total = len(wo_tool_trajs)
    w_tool_total = len(w_tool_trajs)

    assert (
        wo_tool_total == w_tool_total
    ), f"Mismatch: wo_tool has {wo_tool_total} samples, w_tool has {w_tool_total} samples"

    total = wo_tool_total

    wo_tool_accuracy = total and len(wo_tool_correct_indices) / total or 0.0
    w_tool_accuracy = total and len(w_tool_correct_indices) / total or 0.0

    # Count variables (renamed from N_hard/N_win to use fail/succ terminology)
    N_fail = len(wo_tool_incorrect_indices)  # |D_fail|
    N_succ = len(wo_tool_correct_indices)  # |D_succ|

    term1_indices = (
        wo_tool_incorrect_indices & w_tool_correct_indices & w_tool_call_indices
    )
    N_call_fail = len(
        wo_tool_incorrect_indices & w_tool_call_indices
    )  # |D_fail ∩ call|
    term1 = total and len(term1_indices) / total or 0.0

    term2_indices = (
        wo_tool_incorrect_indices & w_tool_correct_indices & w_tool_no_call_indices
    )
    term2 = total and len(term2_indices) / total or 0.0

    term3_indices = (
        wo_tool_correct_indices & w_tool_incorrect_indices & w_tool_call_indices
    )
    term3 = total and len(term3_indices) / total or 0.0

    term4_indices = (
        wo_tool_correct_indices & w_tool_incorrect_indices & w_tool_no_call_indices
    )
    term4 = total and len(term4_indices) / total or 0.0

    # Verify term1-4 decomposition (only for dynamic partition)
    # For fixed (_0) or intersection (_0_t) partitions, these checks may not hold
    # because the partition is based on a different time point or intersection
    if baseline_correct_indices is None and baseline_incorrect_indices is None:
        accuracy_diff = w_tool_accuracy - wo_tool_accuracy
        term_sum = term1 + term2 - term3 - term4

        assert (
            abs(accuracy_diff - term_sum) < 1e-5
        ), f"Mismatch: accuracy_diff={accuracy_diff:.6f}, term_sum={term_sum:.6f}, diff={abs(accuracy_diff - term_sum):.6e}"

    call_num = len(w_tool_call_indices)

    # ========== Calculate detailed probabilities for term decomposition ==========

    # Domain probabilities
    p_fail = total and N_fail / total or 0.0  # P(D_fail)
    p_succ = total and N_succ / total or 0.0  # P(D_succ)

    # Call probabilities in fail domain
    p_call_fail = N_fail and N_call_fail / N_fail or 0.0  # P(c | D_fail)
    p_nocall_fail = (
        N_fail
        and len(wo_tool_incorrect_indices & w_tool_no_call_indices) / N_fail
        or 0.0
    )  # P(¬c | D_fail)

    # Call probabilities in succ domain
    p_call_succ = (
        N_succ and len(wo_tool_correct_indices & w_tool_call_indices) / N_succ or 0.0
    )  # P(c | D_succ)
    p_nocall_succ = (
        N_succ and len(wo_tool_correct_indices & w_tool_no_call_indices) / N_succ or 0.0
    )  # P(¬c | D_succ)

    # Accuracy probabilities in fail domain
    n_nocall_fail = len(wo_tool_incorrect_indices & w_tool_no_call_indices)

    p_acc_call_fail = (
        N_call_fail
        and len(
            wo_tool_incorrect_indices & w_tool_call_indices & w_tool_correct_indices
        )
        / N_call_fail
        or 0.0
    )  # P(✓ | c, D_fail)
    p_acc_nocall_fail = (
        n_nocall_fail
        and len(
            wo_tool_incorrect_indices & w_tool_no_call_indices & w_tool_correct_indices
        )
        / n_nocall_fail
        or 0.0
    )  # P(✓ | ¬c, D_fail)

    # Error probabilities in succ domain
    n_call_succ = len(wo_tool_correct_indices & w_tool_call_indices)
    n_nocall_succ = len(wo_tool_correct_indices & w_tool_no_call_indices)

    p_err_call_succ = (
        n_call_succ
        and len(
            wo_tool_correct_indices & w_tool_call_indices & w_tool_incorrect_indices
        )
        / n_call_succ
        or 0.0
    )  # P(× | c, D_succ)
    p_err_nocall_succ = (
        n_nocall_succ
        and len(
            wo_tool_correct_indices & w_tool_no_call_indices & w_tool_incorrect_indices
        )
        / n_nocall_succ
        or 0.0
    )  # P(× | ¬c, D_succ)

    # Complementary probabilities (for completeness and verification)
    p_acc_call_succ = 1.0 - p_err_call_succ  # P(✓ | c, D_succ)
    p_acc_nocall_succ = 1.0 - p_err_nocall_succ  # P(✓ | ¬c, D_succ)
    p_err_call_fail = 1.0 - p_acc_call_fail  # P(× | c, D_fail)
    p_err_nocall_fail = 1.0 - p_acc_nocall_fail  # P(× | ¬c, D_fail)

    # Verify term calculations using the detailed probabilities
    term1_check = p_fail * p_call_fail * p_acc_call_fail
    term2_check = p_fail * p_nocall_fail * p_acc_nocall_fail
    term3_check = p_succ * p_call_succ * p_err_call_succ
    term4_check = p_succ * p_nocall_succ * p_err_nocall_succ

    # Optional: verify that the recalculated terms match
    if abs(term1 - term1_check) > 1e-5:
        print(f"Warning: term1 mismatch: {term1:.6f} vs {term1_check:.6f}")
    if abs(term2 - term2_check) > 1e-5:
        print(f"Warning: term2 mismatch: {term2:.6f} vs {term2_check:.6f}")
    if abs(term3 - term3_check) > 1e-5:
        print(f"Warning: term3 mismatch: {term3:.6f} vs {term3_check:.6f}")
    if abs(term4 - term4_check) > 1e-5:
        print(f"Warning: term4 mismatch: {term4:.6f} vs {term4_check:.6f}")

    return {
        "term1": term1,
        "term2": term2,
        "term3": term3,
        "term4": term4,
        "call_num": call_num,
        "call_sum": call_sum,
        "wo_tool_accuracy": wo_tool_accuracy,
        "w_tool_accuracy": w_tool_accuracy,
        "wo_tool_total": wo_tool_total,
        "w_tool_total": w_tool_total,
        "wo_tool_correct_num": len(wo_tool_correct_indices),
        "wo_tool_incorrect_num": len(wo_tool_incorrect_indices),
        "w_tool_correct_num": len(w_tool_correct_indices),
        "w_tool_incorrect_num": len(w_tool_incorrect_indices),
        "w_tool_call_num": len(w_tool_call_indices),
        "w_tool_no_call_num": len(w_tool_no_call_indices),
        "term1_num": len(term1_indices),
        "term2_num": len(term2_indices),
        "term3_num": len(term3_indices),
        "term4_num": len(term4_indices),
        "N_fail": N_fail,
        "N_succ": N_succ,
        "N_call_fail": N_call_fail,
        # Detailed probabilities for term decomposition
        "p_fail": p_fail,
        "p_succ": p_succ,
        "p_call_fail": p_call_fail,
        "p_nocall_fail": p_nocall_fail,
        "p_call_succ": p_call_succ,
        "p_nocall_succ": p_nocall_succ,
        "p_acc_call_fail": p_acc_call_fail,
        "p_acc_nocall_fail": p_acc_nocall_fail,
        "p_err_call_succ": p_err_call_succ,
        "p_err_nocall_succ": p_err_nocall_succ,
        "p_acc_call_succ": p_acc_call_succ,
        "p_acc_nocall_succ": p_acc_nocall_succ,
        "p_err_call_fail": p_err_call_fail,
        "p_err_nocall_fail": p_err_nocall_fail,
    }


def parse_evaluation_results(json_path: str) -> dict[str, Any]:
    """Parse evaluation_results.json and extract required metrics."""
    with open(json_path) as f:
        data = json.load(f)

    return {
        "wo_tool_score": data.get("w/o tool"),
        "w_tool_score": data.get("w/ tool"),
    }


def process_single_benchmark(
    root: str, step0_correct_indices: set | None = None
) -> dict[str, Any] | None:
    """Process a single benchmark directory.

    Computes THREE partition metrics:
    - Dynamic: term1, term2, ... (based on current step t's w/o tool performance)
      D_fail = D_fail(t), D_succ = D_succ(t)

    - Fixed: term1_0, term2_0, ... (based on BASELINE MODEL's w/o tool performance)
      D_fail = D_fail(0), D_succ = D_succ(0)
      Note: "step 0" refers to the baseline model, not the experiment's first checkpoint

    - Intersection: term1_0_t, term2_0_t, ... (intersection of baseline and step t for BOTH partitions)
      D_fail = D_fail(0) ∩ D_fail(t) - samples that fail in BOTH baseline and step t (persistently difficult)
      D_succ = D_succ(0) ∩ D_succ(t) - samples that succeed in BOTH baseline and step t (persistently easy)
      Note: This excludes samples that change status between baseline and step t

    Args:
        root: Path to benchmark directory
        step0_correct_indices: Optional set of indices correct in baseline model (step 0)

    Returns:
        dict with benchmark results or None if error
    """
    json_path = os.path.join(root, "evaluation_results.json")
    step = extract_step_from_path(root)
    bench_name = os.path.basename(root)

    if bench_name not in BENCHMARK_ORDER:
        return None

    try:
        metrics = parse_evaluation_results(json_path)
        trajectories = load_benchmark_trajectories(root, bench_name)

        # Compute dynamic partition metrics (original behavior)
        analysis_dynamic = analyze_trajectories(
            trajectories, baseline_correct_indices=None
        )
        analysis_dynamic.pop("wo_tool_accuracy")
        analysis_dynamic.pop("w_tool_accuracy")

        result = {
            "step": step,
            "benchmark": bench_name,
            "json_path": json_path,
            **metrics,
            **analysis_dynamic,
        }

        # Compute fixed and intersection partition metrics if step 0 data available
        if step0_correct_indices is not None:
            # 1. Fixed partition (based on step 0 only)
            analysis_fixed = analyze_trajectories(
                trajectories, baseline_correct_indices=step0_correct_indices
            )
            analysis_fixed.pop("wo_tool_accuracy", None)
            analysis_fixed.pop("w_tool_accuracy", None)

            # Add fixed partition metrics with _0 suffix
            for key, value in analysis_fixed.items():
                result[f"{key}_0"] = value

            # 2. Intersection partition (step 0 ∩ step t)
            # Get current step's correct and incorrect indices
            wo_tool_trajs = trajectories["wo_tool_trajectories"]
            total = len(wo_tool_trajs)

            current_correct_indices = set()
            current_incorrect_indices = set()
            for idx, traj in enumerate(wo_tool_trajs):
                accuracy = traj.get("accuracy_reward", 0)
                if accuracy == 1:
                    current_correct_indices.add(idx)
                else:
                    current_incorrect_indices.add(idx)

            # Compute step 0's incorrect indices
            step0_incorrect_indices = set(range(total)) - step0_correct_indices

            # Compute intersections for BOTH D_succ and D_fail
            # D_succ_intersection = D_succ(0) ∩ D_succ(t) - persistently easy samples
            intersection_correct_indices = (
                step0_correct_indices & current_correct_indices
            )

            # D_fail_intersection = D_fail(0) ∩ D_fail(t) - persistently difficult samples
            intersection_incorrect_indices = (
                step0_incorrect_indices & current_incorrect_indices
            )

            # Analyze with intersection partition (both correct and incorrect provided)
            analysis_intersection = analyze_trajectories(
                trajectories,
                baseline_correct_indices=intersection_correct_indices,
                baseline_incorrect_indices=intersection_incorrect_indices,
            )
            analysis_intersection.pop("wo_tool_accuracy", None)
            analysis_intersection.pop("w_tool_accuracy", None)

            # Add intersection partition metrics with _0_t suffix
            for key, value in analysis_intersection.items():
                result[f"{key}_0_t"] = value

        return result
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None


def load_experiment_step0_partition(
    evaluation_dir: str, exp_name: str
) -> dict[str, set]:
    """Load initial step partition from current experiment (FALLBACK method).

    NOTE: This is a fallback when no baseline model mapping exists in MODEL_BASELINE_MAPPING.
    Normally, step 0 should come from the baseline model, not the experiment's first checkpoint.

    For RL experiments: loads from the EARLIEST step (e.g., global_step_10 if step 0 doesn't exist)
    For baseline models: loads from direct benchmark directories (since there's no training steps)

    Args:
        evaluation_dir: Evaluation directory
        exp_name: Experiment name (e.g., "qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew"
                  or "verl_model/Qwen2.5-VL-7B-Instruct")

    Returns:
        dict mapping benchmark_name -> set of correct indices at initial step
    """
    exp_path = os.path.join(evaluation_dir, exp_name)
    if not os.path.exists(exp_path):
        print(f"Warning: Experiment path {exp_path} does not exist")
        return {}

    print(f"Loading initial step partition from experiment: {exp_name}")

    step0_indices = {}

    # Strategy 1: Find the EARLIEST global_step directory
    # Collect all step directories first
    step_dirs = {}  # {benchmark: {step: directory}}

    for root, _, files in os.walk(exp_path):
        if "evaluation_results.json" in files:
            bench_name = os.path.basename(root)
            if bench_name not in BENCHMARK_ORDER:
                continue

            # Check if this is in a global_step directory
            if "global_step_" in root:
                step = extract_step_from_path(root)
                if bench_name not in step_dirs:
                    step_dirs[bench_name] = {}
                step_dirs[bench_name][step] = root

    # For each benchmark, find the minimum step
    found_global_step = False
    if step_dirs:
        found_global_step = True
        for bench_name, steps_dict in step_dirs.items():
            min_step = min(steps_dict.keys())
            min_step_dir = steps_dict[min_step]

            try:
                trajectories = load_benchmark_trajectories(min_step_dir, bench_name)
                wo_tool_trajs = trajectories["wo_tool_trajectories"]

                correct_indices = set()
                for idx, traj in enumerate(wo_tool_trajs):
                    accuracy = traj.get("accuracy_reward", 0)

                    if accuracy == 1:
                        correct_indices.add(idx)

                step0_indices[bench_name] = correct_indices
                print(
                    f"  {bench_name}: {len(correct_indices)}/{len(wo_tool_trajs)} correct at step {min_step}"
                )

            except Exception as e:
                print(f"  Error loading step {min_step} for {bench_name}: {e}")

    # Strategy 2: If no global_step directories found, try direct benchmark directories (for baseline models)
    if not found_global_step:
        print(
            "  No global_step directories found, trying direct benchmark directories (baseline model)..."
        )
        for bench_name in BENCHMARK_ORDER:
            bench_dir = os.path.join(exp_path, bench_name)
            eval_results_path = os.path.join(bench_dir, "evaluation_results.json")

            if os.path.exists(eval_results_path):
                try:
                    trajectories = load_benchmark_trajectories(bench_dir, bench_name)
                    wo_tool_trajs = trajectories["wo_tool_trajectories"]

                    correct_indices = set()
                    for idx, traj in enumerate(wo_tool_trajs):
                        accuracy = traj.get("accuracy_reward", 0)

                        if accuracy == 1:
                            correct_indices.add(idx)

                    step0_indices[bench_name] = correct_indices
                    print(
                        f"  {bench_name}: {len(correct_indices)}/{len(wo_tool_trajs)} correct (baseline)"
                    )

                except Exception as e:
                    print(f"  Error loading baseline for {bench_name}: {e}")

    return step0_indices


def load_baseline_correct_indices(
    evaluation_dir: str, baseline_model_path: str
) -> dict[str, set]:
    """Load baseline model's w/o tool correct indices for each benchmark.

    Args:
        evaluation_dir: Evaluation directory
        baseline_model_path: Baseline model path (e.g., "verl_model/Qwen2.5-VL-7B-Instruct")

    Returns:
        dict mapping benchmark_name -> set of correct indices in baseline
    """
    baseline_exp_path = os.path.join(evaluation_dir, baseline_model_path)
    if not os.path.exists(baseline_exp_path):
        print(f"Warning: Baseline path {baseline_exp_path} does not exist")
        return {}

    print(f"Loading baseline from: {baseline_model_path}")

    baseline_indices = {}

    # Strategy 1: Try to find global_step_0 directory first
    found_global_step = False
    for root, _, files in os.walk(baseline_exp_path):
        if "global_step_0" in root and "evaluation_results.json" in files:
            found_global_step = True
            bench_name = os.path.basename(root)
            if bench_name not in BENCHMARK_ORDER:
                continue

            try:
                trajectories = load_benchmark_trajectories(root, bench_name)
                wo_tool_trajs = trajectories["wo_tool_trajectories"]

                correct_indices = set()
                for idx, traj in enumerate(wo_tool_trajs):
                    accuracy = traj.get("accuracy_reward", 0)

                    if accuracy == 1:
                        correct_indices.add(idx)

                baseline_indices[bench_name] = correct_indices
                print(
                    f"  {bench_name}: {len(correct_indices)}/{len(wo_tool_trajs)} correct"
                )

            except Exception as e:
                print(f"  Error loading baseline for {bench_name}: {e}")

    # Strategy 2: If no global_step_0 found, look for benchmark directories directly under baseline path
    if not found_global_step:
        print("  No global_step_0 found, trying direct benchmark directories...")
        for bench_name in BENCHMARK_ORDER:
            bench_dir = os.path.join(baseline_exp_path, bench_name)
            eval_results_path = os.path.join(bench_dir, "evaluation_results.json")

            if os.path.exists(eval_results_path):
                try:
                    trajectories = load_benchmark_trajectories(bench_dir, bench_name)
                    wo_tool_trajs = trajectories["wo_tool_trajectories"]

                    correct_indices = set()
                    for idx, traj in enumerate(wo_tool_trajs):
                        accuracy = traj.get("accuracy_reward", 0)

                        if accuracy == 1:
                            correct_indices.add(idx)

                    baseline_indices[bench_name] = correct_indices
                    print(
                        f"  {bench_name}: {len(correct_indices)}/{len(wo_tool_trajs)} correct"
                    )

                except Exception as e:
                    print(f"  Error loading baseline for {bench_name}: {e}")

    return baseline_indices


def find_evaluation_files(
    evaluation_dir: str,
    exp_name: str,
    max_workers: int = 8,
    step0_map: dict[str, set] | None = None,
) -> list[dict[str, Any]]:
    """Find all evaluation_results.json files for a given experiment.

    Args:
        evaluation_dir: Evaluation directory
        exp_name: Experiment name
        max_workers: Number of parallel workers
        step0_map: Optional step 0 partition (benchmark_name -> set of correct indices at step 0)

    Returns:
        list of evaluation results
    """
    exp_path = os.path.join(evaluation_dir, exp_name)
    if not os.path.exists(exp_path):
        print(f"Warning: Experiment path {exp_path} does not exist")
        return []

    print(f"Processing experiment: {exp_name}")

    # Collect all evaluation paths first
    eval_paths = []
    for root, _, files in os.walk(exp_path):
        if "evaluation_results.json" in files:
            bench_name = os.path.basename(root)
            if bench_name in BENCHMARK_ORDER:
                step0_indices = step0_map.get(bench_name, None) if step0_map else None
                eval_paths.append((root, step0_indices))

    # Process in parallel with progress bar
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_benchmark, root, step0_idx): root
            for root, step0_idx in eval_paths
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading benchmarks"
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    return results


def create_csv_for_experiment(
    evaluation_dir: str, exp_name: str, step0_map: dict[str, set] | None = None
) -> None:
    """Create CSV file for a single experiment.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_name: Experiment name
        step0_map: Optional step 0 partition indices
    """
    print(f"Processing experiment: {exp_name}")

    # Find all evaluation files
    results = find_evaluation_files(evaluation_dir, exp_name, step0_map=step0_map)

    if not results:
        print(f"No evaluation results found for {exp_name}")
        return

    df = pd.DataFrame(results)

    # Sort by step and benchmark
    df = df.sort_values(["step", "benchmark"])

    # Save CSV to experiment directory
    exp_dir = os.path.join(evaluation_dir, exp_name)
    csv_name = exp_name.split("/")[-1]
    csv_path = os.path.join(exp_dir, f"{csv_name}_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV to: {csv_path}")
    print(f"Total records: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create CSV files from evaluation results"
    )
    parser.add_argument(
        "--evaluation_dir", type=str, required=True, help="Path to evaluation directory"
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        required=True,
        help="list of experiment names",
    )

    args = parser.parse_args()

    # Cache for baseline models
    baseline_cache: dict[str, dict[str, set]] = {}

    # Process each experiment
    for exp_name in args.exp_names:
        print(f"\n{'='*80}")
        print(f"Processing experiment: {exp_name}")
        print(f"{'='*80}")

        # Determine step 0 partition source
        step0_map = None

        # First, try to find baseline model from mapping
        baseline_model_path = get_baseline_model_from_exp_name(exp_name)

        if baseline_model_path:
            print(f"Found baseline model mapping: {baseline_model_path}")

            # Check cache first
            if baseline_model_path in baseline_cache:
                print("Using cached baseline partition")
                step0_map = baseline_cache[baseline_model_path]
            else:
                # Load from baseline model
                step0_map = load_baseline_correct_indices(
                    args.evaluation_dir, baseline_model_path
                )
                if step0_map:
                    baseline_cache[baseline_model_path] = step0_map
                    print(f"Loaded baseline partition for {len(step0_map)} benchmarks")
                else:
                    print(
                        f"Warning: Failed to load baseline from {baseline_model_path}"
                    )

        # If no baseline model found or failed to load, fallback to experiment's earliest step
        if not step0_map:
            print("Fallback: Loading step 0 partition from experiment itself")
            step0_map = load_experiment_step0_partition(args.evaluation_dir, exp_name)

            if step0_map:
                print(f"Loaded initial step partition for {len(step0_map)} benchmarks")
            else:
                print("Warning: No step 0 partition found")
                print("Will only compute dynamic partition metrics (without _0 suffix)")

        # Create CSV with both dynamic and fixed partition metrics
        create_csv_for_experiment(args.evaluation_dir, exp_name, step0_map=step0_map)


if __name__ == "__main__":
    main()

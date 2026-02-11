#!/usr/bin/env python3
"""
Extract sample details that satisfy specific conditions:
1. p_acc_call_fail: Samples that are incorrect w/o tool, called tool, and correct w/ tool (term1)
2. p_err_call_succ: Samples that are correct w/o tool, called tool, and incorrect w/ tool (term3)

Output: JSON file with structure:
{
    "exp_name": {
        "step": {
            "benchmark_name": {
                "p_acc_call_fail": [
                    {"idx": 0, "id": "sample_id", "messages": [...]},
                    ...
                ],
                "p_err_call_succ": [
                    {"idx": 5, "id": "sample_id", "messages": [...]},
                    ...
                ]
            }
        }
    }
}
"""
import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

# Only process the last 6 benchmarks
SELECTED_BENCHMARKS = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]


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
        List of trajectory records (each line is a dict)
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
        benchmark_dir: Directory containing the benchmark data (e.g., .../global_step_XXX/charxiv2rq)
        benchmark_name: Name of the benchmark (e.g., 'charxiv2rq')

    Returns:
        Dict with keys 'wo_tool_trajectories' and 'w_tool_trajectories'
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


def extract_sample_indices(
    trajectories: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]] | None:
    """Extract sample indices that satisfy specific conditions.

    Args:
        trajectories: Dict with 'wo_tool_trajectories' and 'w_tool_trajectories'

    Returns:
        Dict with 'p_acc_call_fail' and 'p_err_call_succ' containing sample details, or None if no non-empty indices
    """
    wo_tool_trajs = trajectories["wo_tool_trajectories"]
    w_tool_trajs = trajectories["w_tool_trajectories"]

    # Identify wo_tool correct/incorrect indices
    wo_tool_correct_indices = set()
    wo_tool_incorrect_indices = set()

    for idx, traj in enumerate(wo_tool_trajs):
        accuracy_original = traj.get("accuracy_reward_original", 0)
        accuracy_llm = traj.get("accuracy_reward_llm", 0)

        if accuracy_original == 1 or accuracy_llm == 1:
            wo_tool_correct_indices.add(idx)
        else:
            wo_tool_incorrect_indices.add(idx)

    # Identify w_tool correct/incorrect/call/no_call indices
    w_tool_correct_indices = set()
    w_tool_incorrect_indices = set()
    w_tool_call_indices = set()
    w_tool_no_call_indices = set()

    for idx, traj in enumerate(w_tool_trajs):
        accuracy_original = traj.get("accuracy_reward_original", 0)
        accuracy_llm = traj.get("accuracy_reward_llm", 0)
        tool_call_counts = traj.get("tool_call_counts", 0)

        if accuracy_original == 1 or accuracy_llm == 1:
            w_tool_correct_indices.add(idx)
        else:
            w_tool_incorrect_indices.add(idx)

        if tool_call_counts > 0:
            w_tool_call_indices.add(idx)
        else:
            w_tool_no_call_indices.add(idx)

    # Extract indices for the two conditions:
    # 1. p_acc_call_fail: D_fail ∩ call ∩ w_tool_correct (same as term1_indices)
    p_acc_call_fail_indices = (
        wo_tool_incorrect_indices & w_tool_call_indices & w_tool_correct_indices
    )

    # 2. p_err_call_succ: D_succ ∩ call ∩ w_tool_incorrect (same as term3_indices)
    p_err_call_succ_indices = (
        wo_tool_correct_indices & w_tool_call_indices & w_tool_incorrect_indices
    )

    # Return None if both are empty
    if not p_acc_call_fail_indices and not p_err_call_succ_indices:
        return None

    # Extract detailed information for each sample (from w/ tool trajectories)
    p_acc_call_fail_samples = []
    for idx in sorted(p_acc_call_fail_indices):
        sample_info = {
            "idx": idx,
            "id": w_tool_trajs[idx].get("result_dict", {}).get("id", ""),
            "messages": w_tool_trajs[idx].get("messages", []),
        }
        p_acc_call_fail_samples.append(sample_info)

    p_err_call_succ_samples = []
    for idx in sorted(p_err_call_succ_indices):
        sample_info = {
            "idx": idx,
            "id": w_tool_trajs[idx].get("result_dict", {}).get("id", ""),
            "messages": w_tool_trajs[idx].get("messages", []),
        }
        p_err_call_succ_samples.append(sample_info)

    return {"p_acc_call_fail": p_acc_call_fail_samples, "p_err_call_succ": p_err_call_succ_samples}


def process_single_benchmark(
    benchmark_path: str, step: int, bench_name: str
) -> tuple[int, str, dict[str, list[dict[str, Any]]]] | None:
    """Process a single benchmark directory.

    Args:
        benchmark_path: Path to benchmark directory
        step: Step number
        bench_name: Benchmark name

    Returns:
        Tuple of (step, bench_name, sample_details) or None if error/empty
    """
    try:
        trajectories = load_benchmark_trajectories(benchmark_path, bench_name)

        if not trajectories["wo_tool_trajectories"] or not trajectories["w_tool_trajectories"]:
            return None

        sample_details = extract_sample_indices(trajectories)

        if sample_details is None:
            return None

        return (step, bench_name, sample_details)

    except Exception as e:
        print(f"Error processing {bench_name} at step {step}: {e}")
        return None


def extract_indices_for_experiment(
    evaluation_dir: str, exp_name: str, max_workers: int = 8
) -> dict[int, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Extract sample indices for all benchmarks and steps in an experiment.

    Args:
        evaluation_dir: Path to evaluation directory
        exp_name: Experiment name
        max_workers: Number of parallel workers

    Returns:
        Dict mapping step -> benchmark_name -> {p_acc_call_fail, p_err_call_succ}
    """
    exp_path = os.path.join(evaluation_dir, exp_name)
    if not os.path.exists(exp_path):
        print(f"Warning: Experiment path {exp_path} does not exist")
        return {}

    print(f"Processing experiment: {exp_name}")

    # Collect all benchmark paths first
    benchmark_tasks = []
    for root, dirs, files in os.walk(exp_path):
        if "evaluation_results.json" not in files:
            continue

        bench_name = os.path.basename(root)
        if bench_name not in SELECTED_BENCHMARKS:
            continue

        step = extract_step_from_path(root)
        benchmark_tasks.append((root, step, bench_name))

    # Process in parallel with progress bar
    results = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_benchmark, path, step, bench_name): (step, bench_name)
            for path, step, bench_name in benchmark_tasks
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing {exp_name}"
        ):
            result = future.result()
            if result is not None:
                step, bench_name, sample_details = result
                results[step][bench_name] = sample_details

    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Extract sample indices that satisfy specific conditions"
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
        "--output", type=str, default="sample_indices.json", help="Output JSON file path"
    )

    args = parser.parse_args()

    # Extract indices for all experiments
    all_results = {}

    for exp_name in args.exp_names:
        exp_results = extract_indices_for_experiment(args.evaluation_dir, exp_name)
        if exp_results:
            all_results[exp_name] = exp_results

    if not all_results:
        print("No results found")
        return

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved sample details to: {args.output}")

    # Print summary (only non-zero counts)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for exp_name, exp_data in sorted(all_results.items()):
        print(f"\n{exp_name}:")
        for step in sorted(exp_data.keys()):
            print(f"  Step {step}:")
            for bench_name, sample_details in sorted(exp_data[step].items()):
                n_acc_call_fail = len(sample_details["p_acc_call_fail"])
                n_err_call_succ = len(sample_details["p_err_call_succ"])
                # Only print if at least one is non-zero
                if n_acc_call_fail > 0 or n_err_call_succ > 0:
                    print(
                        f"    {bench_name}: p_acc_call_fail={n_acc_call_fail}, p_err_call_succ={n_err_call_succ}"
                    )
    print("=" * 80)


# Example usage:
# python3 extract_sample_indices.py \
#     --evaluation_dir /jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results \
#     --exp_names \
#         qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#         qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#     --output sample_indices.json
if __name__ == "__main__":
    main()

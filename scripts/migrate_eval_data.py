#!/usr/bin/env python3
"""
Migrate and simplify evaluation data from verl_vision to Med/evals.

This script:
1. Extracts pass@1_accuracy from evaluation_results.json summary
2. Filters trajectory jsonl files to keep only essential fields
3. Preserves the original directory structure
"""

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional


# Benchmark categories
REASONING_BENCHMARKS = [
    "charxiv2rq",
    "mathvision",
    "mathvista",
    "mme",
    "mmmu",
    "mmmupro",
]

PERCEPTION_BENCHMARKS = [
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
    "zerobench",
]

ALL_BENCHMARKS = REASONING_BENCHMARKS + PERCEPTION_BENCHMARKS

# Fields to keep in trajectory files
TRAJECTORY_FIELDS = [
    "uid",
    "response_length",
    "tool_call_counts",
    "messages",
    "accuracy_reward",
]


def extract_pass_at_1(value: str) -> float:
    """Extract numeric value from formatted string like '0.5610 (56.10%)'."""
    match = re.match(r"([0-9.]+)", value)
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot parse pass@1 value: {value}")


def extract_summary(eval_results: Dict) -> Dict[str, float]:
    """
    Extract simplified summary from evaluation_results.json.

    Args:
        eval_results: Full evaluation results dict

    Returns:
        {"w/o tool": 0.561, "w/ tool": 0.479}
    """
    summary = eval_results.get("summary", {})

    # Find single_turn and tool agent entries
    single_turn_key = None
    tool_agent_key = None

    for key in summary.keys():
        if "single_turn_agent" in key:
            single_turn_key = key
        elif "tool_agent" in key:
            tool_agent_key = key

    if not single_turn_key or not tool_agent_key:
        raise ValueError(f"Missing agent keys in summary. Keys: {list(summary.keys())}")

    wo_tool = extract_pass_at_1(summary[single_turn_key]["pass@1_accuracy"])
    w_tool = extract_pass_at_1(summary[tool_agent_key]["pass@1_accuracy"])

    return {"w/o tool": wo_tool, "w/ tool": w_tool}


def filter_trajectory_line(line: str, generate_uid: bool = True) -> str:
    """
    Filter a single trajectory JSON line to keep only essential fields.

    Args:
        line: JSON line string
        generate_uid: If True, generate a new uid when existing uid is None or 'None'

    Returns:
        Filtered JSON line string
    """
    data = json.loads(line)
    filtered = {k: data[k] for k in TRAJECTORY_FIELDS if k in data}

    # Generate new uid if it's None or the string 'None'
    uid_value = filtered.get("uid")
    if generate_uid and (uid_value is None or uid_value == "None"):
        filtered["uid"] = uuid.uuid4().hex

    return json.dumps(filtered)


def migrate_benchmark(
    source_bench_dir: Path,
    target_bench_dir: Path,
    override: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Migrate a single benchmark directory.

    Args:
        source_bench_dir: Source benchmark directory (e.g., .../charxiv2rq)
        target_bench_dir: Target benchmark directory
        override: Whether to override existing files
        verbose: Print detailed logs

    Returns:
        True if successful, False if skipped or failed
    """
    eval_results_path = source_bench_dir / "evaluation_results.json"
    trajectories_dir = source_bench_dir / "trajectories"

    # Check if source files exist
    if not eval_results_path.exists():
        if verbose:
            print(f"  ⚠️  Skipped: evaluation_results.json not found")
        return False

    if not trajectories_dir.exists():
        if verbose:
            print(f"  ⚠️  Skipped: trajectories directory not found")
        return False

    # Check if target already exists
    if target_bench_dir.exists() and not override:
        if verbose:
            print(f"  ⏭️  Skipped: target already exists (use --override to overwrite)")
        return False

    # Create target directory
    target_bench_dir.mkdir(parents=True, exist_ok=True)
    target_trajectories_dir = target_bench_dir / "trajectories"
    target_trajectories_dir.mkdir(exist_ok=True)

    # 1. Migrate evaluation_results.json
    try:
        with eval_results_path.open() as f:
            eval_results = json.load(f)

        simplified_summary = extract_summary(eval_results)

        target_eval_path = target_bench_dir / "evaluation_results.json"
        with target_eval_path.open("w") as f:
            json.dump(simplified_summary, f, indent=2)

        if verbose:
            print(f"  ✅ Migrated evaluation_results.json: {simplified_summary}")
    except Exception as e:
        print(f"  ❌ Failed to migrate evaluation_results.json: {e}")
        return False

    # 2. Migrate trajectory files
    trajectory_files = list(trajectories_dir.glob("*.jsonl"))
    for traj_file in trajectory_files:
        try:
            target_traj_file = target_trajectories_dir / traj_file.name

            with traj_file.open() as fin, target_traj_file.open("w") as fout:
                for line in fin:
                    line = line.strip()
                    if line:
                        filtered_line = filter_trajectory_line(line)
                        fout.write(filtered_line + "\n")

            if verbose:
                print(f"  ✅ Migrated trajectory: {traj_file.name}")
        except Exception as e:
            print(f"  ❌ Failed to migrate {traj_file.name}: {e}")
            return False

    return True


def parse_step_range(step_range: Optional[str]) -> Optional[List[int]]:
    """
    Parse step range string.

    Args:
        step_range: e.g., "10-50" or "10,20,30"

    Returns:
        List of step numbers or None for all steps
    """
    if not step_range:
        return None

    # Range format: "10-50"
    if "-" in step_range:
        start, end = step_range.split("-")
        return list(range(int(start), int(end) + 10, 10))

    # Comma-separated: "10,20,30"
    if "," in step_range:
        return [int(s.strip()) for s in step_range.split(",")]

    # Single value: "10"
    return [int(step_range)]


def migrate_experiment(
    source_dir: Path,
    target_dir: Path,
    exp_group: str,
    exp_name: str,
    benchmarks: Optional[List[str]] = None,
    steps: Optional[str] = None,
    override: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """
    Migrate an entire experiment.

    Args:
        source_dir: Source evaluation_results directory
        target_dir: Target evals directory
        exp_group: Experiment group name
        exp_name: Experiment name
        benchmarks: List of benchmark names to migrate (None for all)
        steps: Step range to migrate (None for all)
        override: Whether to override existing files
        dry_run: Preview without actually migrating
        verbose: Print detailed logs
    """
    source_exp_dir = source_dir / exp_group / exp_name
    target_exp_dir = target_dir / exp_group / exp_name

    if not source_exp_dir.exists():
        print(f"❌ Source experiment not found: {source_exp_dir}")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating: {exp_group}/{exp_name}")

    # Parse step range
    step_list = parse_step_range(steps)

    # Find all global_step directories
    step_dirs = sorted(source_exp_dir.glob("global_step_*"))

    if not step_dirs:
        print(f"  ⚠️  No global_step directories found")
        return

    # Filter by step range if specified
    if step_list:
        step_dirs = [
            d for d in step_dirs if int(d.name.split("_")[-1]) in step_list
        ]

    print(f"  Found {len(step_dirs)} steps to process")

    total_migrated = 0
    total_skipped = 0
    total_failed = 0

    for step_dir in step_dirs:
        step_name = step_dir.name
        print(f"\n  📁 {step_name}")

        # Get all benchmark directories
        bench_dirs = [d for d in step_dir.iterdir() if d.is_dir()]

        # Filter by benchmark list if specified
        if benchmarks:
            bench_dirs = [d for d in bench_dirs if d.name in benchmarks]

        for bench_dir in bench_dirs:
            bench_name = bench_dir.name
            target_bench_dir = target_exp_dir / step_name / bench_name

            print(f"    🔄 {bench_name}...", end=" ")

            if dry_run:
                print("[DRY RUN]")
                continue

            success = migrate_benchmark(
                bench_dir, target_bench_dir, override=override, verbose=verbose
            )

            if success:
                total_migrated += 1
                if not verbose:
                    print("✅")
            elif target_bench_dir.exists() and not override:
                total_skipped += 1
                if not verbose:
                    print("⏭️  (exists)")
            else:
                total_failed += 1
                if not verbose:
                    print("❌")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    print(f"  ✅ Migrated: {total_migrated}")
    print(f"  ⏭️  Skipped: {total_skipped}")
    print(f"  ❌ Failed: {total_failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate evaluation data from verl_vision to Med/evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate entire experiment group
  %(prog)s qwen3vl_instruct_75_50

  # Migrate single experiment
  %(prog)s qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1

  # Migrate with specific benchmarks and steps
  %(prog)s qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1 --benchmarks charxiv2rq,mathvision --steps 10-100

  # Preview migration
  %(prog)s qwen3vl_instruct_75_50 --dry-run
        """
    )

    parser.add_argument(
        "exp_path",
        nargs="?",
        help="Experiment path: 'group' to migrate entire group, or 'group/exp_name' for single experiment (e.g., qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1)",
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results"),
        help="Source evaluation_results directory",
    )

    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("evals"),
        help="Target evals directory (relative to Med project root)",
    )

    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark names (e.g., charxiv2rq,mathvision). If not specified, migrate all benchmarks.",
    )

    parser.add_argument(
        "--steps",
        help="Step range to migrate (e.g., 10-100, or 10,20,30). If not specified, migrate all steps.",
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing files in target directory",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without actually copying files",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed logs",
    )

    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List all available benchmarks and exit",
    )

    args = parser.parse_args()

    # List benchmarks
    if args.list_benchmarks:
        print("Reasoning benchmarks:")
        for b in REASONING_BENCHMARKS:
            print(f"  - {b}")
        print("\nPerception benchmarks:")
        for b in PERCEPTION_BENCHMARKS:
            print(f"  - {b}")
        return

    # Check exp_path is provided
    if not args.exp_path:
        parser.error("exp_path is required (use --list-benchmarks to list available benchmarks)")

    # Parse benchmarks
    benchmark_list = None
    if args.benchmarks:
        benchmark_list = [b.strip() for b in args.benchmarks.split(",")]

    # Resolve target directory (relative to Med project root)
    if not args.target_dir.is_absolute():
        # Assume script is in Med/scripts/
        project_root = Path(__file__).parent.parent
        args.target_dir = project_root / args.target_dir

    # Parse exp_path into exp_group and exp_name
    if "/" in args.exp_path:
        # Single experiment: group/exp_name
        parts = args.exp_path.split("/", 1)
        exp_group = parts[0]
        exp_name = parts[1]

        # Migrate single experiment
        migrate_experiment(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            exp_group=exp_group,
            exp_name=exp_name,
            benchmarks=benchmark_list,
            steps=args.steps,
            override=args.override,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    else:
        # Entire group: just group name
        exp_group = args.exp_path

        source_group_dir = args.source_dir / exp_group
        if not source_group_dir.exists():
            print(f"❌ Experiment group not found: {source_group_dir}")
            return

        exp_names = [d.name for d in source_group_dir.iterdir() if d.is_dir()]
        print(f"Found {len(exp_names)} experiments in group '{exp_group}':")
        for exp_name in exp_names:
            print(f"  - {exp_name}")

        if args.dry_run:
            print("\n[DRY RUN] Would migrate all experiments listed above.")
            return

        confirm = input("\nProceed with migration? [y/N]: ")
        if confirm.lower() != "y":
            print("Migration cancelled.")
            return

        for exp_name in exp_names:
            migrate_experiment(
                source_dir=args.source_dir,
                target_dir=args.target_dir,
                exp_group=exp_group,
                exp_name=exp_name,
                benchmarks=benchmark_list,
                steps=args.steps,
                override=args.override,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()

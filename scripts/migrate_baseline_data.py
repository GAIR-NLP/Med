#!/usr/bin/env python3
"""
Migrate baseline evaluation data from verl_vision/verl_model to Med/evals/baseline.

Baseline models don't have global_step directories, only direct benchmark folders.
"""

import argparse
import sys
from pathlib import Path

# Import from the main migration script
sys.path.insert(0, str(Path(__file__).parent))
from migrate_eval_data import (
    REASONING_BENCHMARKS,
    PERCEPTION_BENCHMARKS,
    ALL_BENCHMARKS,
    migrate_benchmark,
)


def migrate_baseline_model(
    source_dir: Path,
    target_dir: Path,
    model_name: str,
    benchmarks: list[str] | None = None,
    override: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """
    Migrate a baseline model's evaluation data.

    Args:
        source_dir: Source verl_model directory
        target_dir: Target evals/baseline directory
        model_name: Model name (e.g., Qwen2.5-VL-7B-Instruct)
        benchmarks: List of benchmark names to migrate (None for all)
        override: Whether to override existing files
        dry_run: Preview without actually migrating
        verbose: Print detailed logs
    """
    source_model_dir = source_dir / model_name
    target_model_dir = target_dir / model_name

    if not source_model_dir.exists():
        print(f"❌ Source model not found: {source_model_dir}")
        return

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating baseline: {model_name}")

    # Find all benchmark directories (ignore CSV files and other non-directories)
    bench_dirs = [d for d in source_model_dir.iterdir() if d.is_dir()]

    # Filter by benchmark list if specified
    if benchmarks:
        bench_dirs = [d for d in bench_dirs if d.name in benchmarks]

    if not bench_dirs:
        print(f"  ⚠️  No benchmark directories found")
        return

    print(f"  Found {len(bench_dirs)} benchmarks to process")

    total_migrated = 0
    total_skipped = 0
    total_failed = 0

    for bench_dir in sorted(bench_dirs):
        bench_name = bench_dir.name
        target_bench_dir = target_model_dir / bench_name

        print(f"  🔄 {bench_name}...", end=" ")

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
        description="Migrate baseline evaluation data from verl_vision to Med/evals/baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate specific baseline model
  %(prog)s Qwen2.5-VL-7B-Instruct

  # Migrate multiple baseline models
  %(prog)s Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct

  # Migrate with specific benchmarks
  %(prog)s Qwen2.5-VL-7B-Instruct --benchmarks charxiv2rq,mathvision

  # Preview migration
  %(prog)s Qwen2.5-VL-7B-Instruct --dry-run
        """,
    )

    parser.add_argument(
        "model_names",
        nargs="+",
        help="Baseline model names (e.g., Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct)",
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_vision/evaluation_results/verl_model"
        ),
        help="Source verl_model directory",
    )

    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("evals/baseline"),
        help="Target baseline directory (relative to Med project root)",
    )

    parser.add_argument(
        "--benchmarks",
        help="Comma-separated benchmark names (e.g., charxiv2rq,mathvision). If not specified, migrate all benchmarks.",
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
        "-v",
        "--verbose",
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

    # Parse benchmarks
    benchmark_list = None
    if args.benchmarks:
        benchmark_list = [b.strip() for b in args.benchmarks.split(",")]

    # Resolve target directory (relative to Med project root)
    if not args.target_dir.is_absolute():
        # Assume script is in Med/scripts/
        project_root = Path(__file__).parent.parent
        args.target_dir = project_root / args.target_dir

    # Migrate each model
    for model_name in args.model_names:
        migrate_baseline_model(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            model_name=model_name,
            benchmarks=benchmark_list,
            override=args.override,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()

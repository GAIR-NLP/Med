#!/usr/bin/env python3
"""
Analyze data sources from parquet files by extracting and counting image path prefixes.

This script:
1. Loads all parquet files from a specified directory
2. Extracts image_path from extra_info
3. Parses the path string and gets the directory prefix (without filename)
4. Counts occurrences of each prefix
"""
import argparse
import ast
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def extract_path_prefix(image_path_str: str) -> str | None:
    """Extract directory prefix from image_path string.

    Args:
        image_path_str: String like "['/path/to/dir/file.jpg']"

    Returns:
        Directory prefix like '/path/to/dir' or None if parsing fails
    """
    try:
        # Parse the string to get the list
        path_list = ast.literal_eval(image_path_str)

        if not path_list or len(path_list) == 0:
            return None

        # Get the first path
        first_path = path_list[0]

        # Remove the filename (everything after the last slash)
        prefix = first_path.rsplit("/", 1)[0]

        return prefix
    except Exception:
        return None


def analyze_parquet_sources(parquet_dir: str) -> Counter:
    """Analyze data sources from all parquet files in a directory.

    Args:
        parquet_dir: Directory containing parquet files

    Returns:
        Counter object with prefix counts
    """
    parquet_dir = Path(parquet_dir)

    # Find all parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return Counter()

    print(f"Found {len(parquet_files)} parquet files")

    prefix_counter = Counter()

    # Process each parquet file
    for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            # Load dataset
            ds = load_dataset("parquet", data_files=str(parquet_file))

            # Process each sample in train split
            for sample in ds["train"]:
                extra_info = sample.get("extra_info", {})
                image_path_str = extra_info.get("image_path")

                if not image_path_str:
                    continue

                # Check for special cases first
                if "visualprobe" in image_path_str.lower():
                    prefix_counter["visualprobe"] += 1
                elif "thyme" in image_path_str.lower():
                    prefix_counter["thyme"] += 1
                else:
                    # Extract prefix for normal cases
                    prefix = extract_path_prefix(image_path_str)
                    if prefix:
                        prefix_counter[prefix] += 1
            print(prefix_counter)

        except Exception as e:
            print(f"Error processing {parquet_file.name}: {e}")
            continue

    return prefix_counter


def print_source_statistics(prefix_counter: Counter):
    """Print source statistics in a formatted way.

    Args:
        prefix_counter: Counter object with prefix counts
    """
    if not prefix_counter:
        print("No data found")
        return

    total_samples = sum(prefix_counter.values())

    print("\n" + "=" * 100)
    print("DATA SOURCE STATISTICS")
    print("=" * 100)
    print(f"Total samples: {total_samples:,}")
    print(f"Unique source prefixes: {len(prefix_counter):,}")
    print("\n" + "-" * 100)
    print(f"{'Source Prefix':<80} {'Count':>10} {'Percentage':>10}")
    print("-" * 100)

    # Sort by count (descending)
    for prefix, count in prefix_counter.most_common():
        percentage = (count / total_samples) * 100
        print(f"{prefix:<80} {count:>10,} {percentage:>9.2f}%")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze data sources from parquet files")
    parser.add_argument(
        "--parquet_dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/natural_ratio_0.75_tool_call_ratio_0.5",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional output file to save statistics"
    )

    args = parser.parse_args()

    # Analyze sources
    prefix_counter = analyze_parquet_sources(args.parquet_dir)

    # Print statistics
    print_source_statistics(prefix_counter)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            total_samples = sum(prefix_counter.values())
            f.write("DATA SOURCE STATISTICS\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total samples: {total_samples:,}\n")
            f.write(f"Unique source prefixes: {len(prefix_counter):,}\n\n")
            f.write(f"{'Source Prefix':<80} {'Count':>10} {'Percentage':>10}\n")
            f.write("-" * 100 + "\n")

            for prefix, count in prefix_counter.most_common():
                percentage = (count / total_samples) * 100
                f.write(f"{prefix:<80} {count:>10,} {percentage:>9.2f}%\n")

            f.write("=" * 100 + "\n")

        print(f"\nStatistics saved to: {args.output}")


# Example usage:
# python3 recipe/o3/plot_v3/analyze_data_sources.py \
#     --parquet_dir /jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/natural_ratio_0.75_tool_call_ratio_0.5 \
#     --output source_statistics.txt

if __name__ == "__main__":
    main()

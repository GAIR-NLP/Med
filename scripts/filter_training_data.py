#!/usr/bin/env python3
"""
Filter training data to keep only samples from specified data sources.

Source: natural_ratio_0.75_tool_call_ratio_0.5 (35 parquet files, 17,248 samples)
Output: Med_training_data (filtered parquet files, ~14,880 samples)

Matching logic:
- thyme: top-level data_source == 'thyme_tool_agent'
- visualprobe: top-level data_source == 'visualprob_tool_agent'
- Other 10 sources: data_source == 'xiatatutu_tool_agent' AND
  lowercase grouped name appears in extra_info.image_path (lowercase)
  Note: est-vqa must be checked before st-vqa to avoid substring collision.

Expected counts (paper Table):
  thyme=7913, visualprobe=4335, st-vqa=580, MathV-360k=546, llavaov=382,
  rvlcdip=261, Video-R1-data=201, screenqa=158, FinChart-Bench=136,
  plotqa=126, SPIQA=122(~124), EST-VQA=118  |  Total=14,878
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 13 data sources to keep, ordered so that est-vqa is checked before st-vqa
KEEP_SOURCES_TOP_LEVEL = {
    "thyme_tool_agent": "thyme",
    "visualprob_tool_agent": "visualprobe",
}

# For xiatatutu_tool_agent, match by image_path substring (lowercase).
# Order matters: est-vqa before st-vqa to avoid substring collision.
KEEP_SOURCES_IMAGE_PATH = [
    "est-vqa",
    "st-vqa",
    "mathv-360k",
    "llavaov",
    "rvlcdip",
    "video-r1-data",
    "screenqa",
    "finchart-bench",
    "plotqa",
    "spiqa",
]


def classify_sample(row: pd.Series) -> str | None:
    """Classify a sample into a grouped data source name.

    Returns the grouped name if the sample should be kept, None otherwise.
    """
    ds = row["data_source"]

    # Direct match for thyme and visualprobe
    if ds in KEEP_SOURCES_TOP_LEVEL:
        return KEEP_SOURCES_TOP_LEVEL[ds]

    # For xiatatutu, match by image_path
    if ds == "xiatatutu_tool_agent":
        extra_info = row.get("extra_info", {})
        if isinstance(extra_info, dict):
            image_path = str(extra_info.get("image_path", "")).lower()
            for source in KEEP_SOURCES_IMAGE_PATH:
                if source in image_path:
                    return source

    return None


def filter_and_save(src_dir: str, dst_dir: str):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {src_path}")
        return

    print(f"Source: {src_path}")
    print(f"Output: {dst_path}")
    print(f"Found {len(parquet_files)} parquet files")
    print()

    total_counter = Counter()
    total_kept = 0
    total_discarded = 0
    all_kept_dfs = []

    for pf in tqdm(parquet_files, desc="Processing"):
        df = pd.read_parquet(pf)

        # Classify each sample
        labels = df.apply(classify_sample, axis=1)
        mask = labels.notnull()

        kept_df = df[mask].copy()
        kept_labels = labels[mask]

        # Count per source
        for label in kept_labels:
            total_counter[label] += 1

        total_kept += len(kept_df)
        total_discarded += len(df) - len(kept_df)
        all_kept_dfs.append(kept_df)

    # Concatenate all kept samples
    merged = pd.concat(all_kept_dfs, ignore_index=True)

    # Save as multiple parquet files (500 samples each to avoid overflow)
    chunk_size = 500
    num_chunks = (len(merged) + chunk_size - 1) // chunk_size
    total_size = 0.0

    print(f"\nSaving {len(merged)} samples into {num_chunks} parquet files...")
    for i in range(num_chunks):
        chunk = merged.iloc[i * chunk_size : (i + 1) * chunk_size]
        output_file = dst_path / f"train-{i:05d}-of-{num_chunks:05d}.parquet"
        chunk.to_parquet(output_file, index=False)
        total_size += output_file.stat().st_size

    # Print summary
    print()
    print("=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"{'Source':<20} {'Count':>8}")
    print("-" * 30)
    for source, count in total_counter.most_common():
        print(f"{source:<20} {count:>8,}")
    print("-" * 30)
    print(f"{'Total kept':<20} {total_kept:>8,}")
    print(f"{'Total discarded':<20} {total_discarded:>8,}")
    print(f"{'Grand total':<20} {total_kept + total_discarded:>8,}")
    print("=" * 60)
    print(f"\nSaved to: {dst_path}")
    print(f"Files: {num_chunks} parquet files")
    print(f"Total size: {total_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Filter training data to keep only specified data sources"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/natural_ratio_0.75_tool_call_ratio_0.5",
        help="Source directory containing parquet files",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/Med_training_data",
        help="Output directory for filtered data",
    )
    args = parser.parse_args()

    filter_and_save(args.src, args.dst)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Deduplicate training data against evaluation data using pHash.

Removes training samples whose image pHash has Hamming distance < threshold
to any evaluation image. Saves using HuggingFace datasets with proper Features.
Outputs data source composition before and after dedup.

Usage:
    python scripts/dedup_train_eval_phash.py --dry-run
    python scripts/dedup_train_eval_phash.py
    python scripts/dedup_train_eval_phash.py --threshold 5 --workers 16
"""
import argparse
import io
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import PIL.Image
from datasets import Dataset, Features, Image, List, Value
from tqdm import tqdm

FEATURES = Features(
    {
        "images": List(Image(mode=None, decode=True)),
        "data_source": Value("string"),
        "prompt": List({"content": Value("string"), "role": Value("string")}),
        "ability": Value("string"),
        "reward_model": {
            "answer": Value("string"),
            "format_ratio": Value("float32"),
            "ground_truth": Value("string"),
            "length_ratio": Value("float32"),
            "style": Value("string"),
            "verifier": Value("string"),
            "verifier_parm": {
                "det_verifier_normalized": Value("bool"),
                "det_reward_ratio": {
                    "iou_max_label_first": Value("float32"),
                    "iou_max_iou_first": Value("float32"),
                    "iou_completeness": Value("float32"),
                    "map": Value("float32"),
                    "map50": Value("float32"),
                    "map75": Value("float32"),
                },
            },
        },
        "extra_info": {
            "answer": Value("string"),
            "data_source": Value("string"),
            "id": Value("string"),
            "image_path": Value("string"),
            "question": Value("string"),
            "split": Value("string"),
            "index": Value("string"),
            "prompt_length": Value("int32"),
            "tools_kwargs": {
                "crop_and_zoom": {
                    "create_kwargs": {
                        "raw_query": Value("string"),
                        "image": Image(mode=None, decode=True),
                    }
                }
            },
            "need_tools_kwargs": Value("bool"),
        },
        "agent_name": Value("string"),
    }
)

# Data source classification (reused from filter_training_data.py)
_TOP_LEVEL_SOURCES = {
    "thyme_tool_agent": "thyme",
    "visualprob_tool_agent": "visualprobe",
}
# Order matters: est-vqa before st-vqa to avoid substring collision
_IMAGE_PATH_SOURCES = [
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


def classify_data_source(row_dict: dict) -> str:
    """Classify a sample into grouped data source name."""
    ds = row_dict.get("data_source", "")
    if ds in _TOP_LEVEL_SOURCES:
        return _TOP_LEVEL_SOURCES[ds]
    if ds == "xiatatutu_tool_agent":
        extra_info = row_dict.get("extra_info", {})
        if isinstance(extra_info, dict):
            ip = str(extra_info.get("image_path", "")).lower()
            for src in _IMAGE_PATH_SOURCES:
                if src in ip:
                    return src
    return "other"


# --------------- pHash computation ---------------


def image_bytes_to_phash(image_bytes: bytes) -> int | None:
    """Compute 64-bit pHash from raw image bytes."""
    try:
        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        h = imagehash.phash(img, hash_size=8)
        return int(str(h), 16)
    except Exception:
        return None


def _hash_one_image(args: tuple) -> tuple[int | None, str, int]:
    """Worker: hash a single image. Returns (hash, filename, row_idx)."""
    image_bytes, filename, row_idx = args
    return (image_bytes_to_phash(image_bytes), filename, row_idx)


def extract_image_tasks(parquet_dir: str) -> list[tuple[bytes, str, int]]:
    """Extract (image_bytes, filename, row_index) from all parquets."""
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    tasks = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for i, row in df.iterrows():
            images = row.get("images")
            if images is None or len(images) == 0:
                continue
            img_data = images[0]
            if isinstance(img_data, dict):
                raw = img_data.get("bytes", b"")
                if raw:
                    tasks.append((raw, pf.name, i))
    return tasks


def compute_hashes_parallel(
    parquet_dir: str, label: str, workers: int
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Compute pHash for all images using multiprocessing."""
    print(f"  Loading {label} images from parquets...")
    tasks = extract_image_tasks(parquet_dir)
    print(f"  Found {len(tasks):,} images, hashing with {workers} workers...")

    hashes = []
    index = []

    with Pool(workers) as pool:
        for h, fname, row_idx in tqdm(
            pool.imap(_hash_one_image, tasks, chunksize=64),
            total=len(tasks),
            desc=f"Hashing {label}",
        ):
            if h is not None:
                hashes.append(h)
                index.append((fname, row_idx))

    return np.array(hashes, dtype=np.uint64), index


# --------------- Hamming distance ---------------


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint64 array."""
    x = arr.astype(np.uint64)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return (x * h01) >> np.uint64(56)


def find_duplicates(
    train_hashes: np.ndarray,
    eval_hashes: np.ndarray,
    threshold: int = 5,
    batch_size: int = 1000,
) -> set[int]:
    """Find training indices with Hamming distance < threshold to any eval hash."""
    dup_indices = set()
    n_train = len(train_hashes)
    n_eval = len(eval_hashes)

    print(f"Comparing {n_train:,} train x {n_eval:,} eval (threshold={threshold})")

    for start in tqdm(range(0, n_train, batch_size), desc="Comparing"):
        end = min(start + batch_size, n_train)
        batch = train_hashes[start:end]
        xor = batch[:, np.newaxis] ^ eval_hashes[np.newaxis, :]
        dist = popcount_u64(xor)
        min_dist = dist.min(axis=1)
        hits = np.where(min_dist < threshold)[0]
        for h in hits:
            dup_indices.add(start + h)

    return dup_indices


def print_source_comparison(before: Counter, after: Counter):
    """Print data source composition before and after dedup."""
    all_sources = sorted(set(before) | set(after), key=lambda s: before.get(s, 0), reverse=True)
    total_before = sum(before.values())
    total_after = sum(after.values())

    print(f"\n  {'Source':<20} {'Before':>8} {'After':>8} {'Removed':>8} {'%Removed':>10}")
    print("  " + "-" * 58)
    for src in all_sources:
        b = before.get(src, 0)
        a = after.get(src, 0)
        r = b - a
        pct = f"{r/b*100:.1f}%" if b > 0 else "-"
        print(f"  {src:<20} {b:>8,} {a:>8,} {r:>8,} {pct:>10}")
    print("  " + "-" * 58)
    r_total = total_before - total_after
    pct_total = f"{r_total/total_before*100:.1f}%" if total_before > 0 else "-"
    print(f"  {'TOTAL':<20} {total_before:>8,} {total_after:>8,} " f"{r_total:>8,} {pct_total:>10}")


# --------------- Main ---------------


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate training vs eval using pHash, remove duplicates"
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/Med_training_data",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/Med_evaluation_data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data/Med_training_data_dedup",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Hamming distance threshold (exclusive, <threshold)",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of parallel workers for hashing"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report duplicates, don't save")
    args = parser.parse_args()

    # Step 1: Compute eval hashes
    print("=" * 60)
    print("Step 1: Computing evaluation set pHashes")
    print("=" * 60)
    eval_hashes, eval_index = compute_hashes_parallel(args.eval_dir, "eval", args.workers)
    print(f"  Eval images hashed: {len(eval_hashes):,}")

    # Step 2: Compute training hashes
    print()
    print("=" * 60)
    print("Step 2: Computing training set pHashes")
    print("=" * 60)
    train_hashes, train_index = compute_hashes_parallel(args.train_dir, "train", args.workers)
    print(f"  Train images hashed: {len(train_hashes):,}")

    # Step 3: Find duplicates
    print()
    print("=" * 60)
    print(f"Step 3: Finding duplicates (Hamming distance < {args.threshold})")
    print("=" * 60)
    dup_indices = find_duplicates(train_hashes, eval_hashes, args.threshold, args.batch_size)
    print(f"\n  Duplicates found: {len(dup_indices):,} / {len(train_hashes):,}")

    dup_by_file = {}
    for idx in dup_indices:
        fname, row_idx = train_index[idx]
        dup_by_file.setdefault(fname, []).append(row_idx)

    if dup_indices:
        print("\n  Duplicates per training file:")
        for fname in sorted(dup_by_file):
            print(f"    {fname}: {len(dup_by_file[fname])} duplicates")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Step 4: Remove duplicates and save, track data source composition
    print()
    print("=" * 60)
    print("Step 4: Removing duplicates & saving")
    print("=" * 60)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dup_keys = set()
    for idx in dup_indices:
        dup_keys.add(train_index[idx])

    before_counter = Counter()
    after_counter = Counter()
    total_samples = 0
    total_size = 0.0
    train_files = sorted(Path(args.train_dir).glob("*.parquet"))
    shard_idx = 0

    for pf in tqdm(train_files, desc="Filtering & saving"):
        df = pd.read_parquet(pf)
        rows = []
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            src = classify_data_source(row_dict)
            before_counter[src] += 1

            if (pf.name, i) not in dup_keys:
                rows.append(row_dict)
                after_counter[src] += 1

        if rows:
            shard_ds = Dataset.from_list(rows, features=FEATURES)
            out_file = output_path / f"train-{shard_idx:05d}.parquet"
            shard_ds.to_parquet(str(out_file))
            total_samples += len(shard_ds)
            total_size += out_file.stat().st_size
            shard_idx += 1
            del shard_ds
        del rows, df

    # Rename with total shard count
    final_files = sorted(output_path.glob("train-*.parquet"))
    num_shards = len(final_files)
    for f in final_files:
        idx_str = f.stem.split("-")[1]
        new_name = f.parent / f"train-{idx_str}-of-{num_shards:05d}.parquet"
        f.rename(new_name)

    print(f"\n  Original:     {sum(before_counter.values()):,} samples")
    print(f"  Removed:      {len(dup_indices):,} duplicates")
    print(f"  Remaining:    {total_samples:,} samples")
    print(f"  Saved to:     {output_path}")
    print(f"  Files:        {num_shards} parquet files")
    print(f"  Total size:   {total_size / (1024*1024):.1f} MB")

    # Data source composition
    print()
    print("=" * 60)
    print("Data Source Composition (Before vs After)")
    print("=" * 60)
    print_source_comparison(before_counter, after_counter)


if __name__ == "__main__":
    main()

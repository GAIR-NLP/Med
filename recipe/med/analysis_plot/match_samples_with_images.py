#!/usr/bin/env python3
"""
Match extracted samples with their original images from parquet files.

This script:
1. Loads the extracted sample indices JSON
2. For each experiment/step/benchmark, loads the corresponding parquet file
3. Matches samples by idx and extracts images
4. Combines images with messages into a new dataset
"""
import argparse
import json
import os
from typing import Any

from datasets import load_dataset

# Parquet file configurations
PARQUET_DIR = "/jfs-dialogue-mmos02-rs02/yema/code/verl_data/tool_call/test"

PARQUET_FILES = {
    "hrbench4k": "hrbench4k_bench_tool_agent_format_0.0_length_0.0_maxlen_12800_num_800.parquet",
    "hrbench8k": "hrbench8k_bench_tool_agent_format_0.0_length_0.0_maxlen_12800_num_800.parquet",
    "visualprobeasy": "visualprobeasy_bench_tool_agent_format_0.0_length_0.0_maxlen_12919_num_141.parquet",
    "visualprobmedium": "visualprobmedium_bench_tool_agent_format_0.0_length_0.0_maxlen_12934_num_268.parquet",
    "visualprobhard": "visualprobhard_bench_tool_agent_format_0.0_length_0.0_maxlen_12878_num_106.parquet",
    "vstar": "vstar_bench_tool_agent_format_0.0_length_0.0_maxlen_10564_num_191.parquet",
}


def load_parquet_dataset(benchmark_name: str) -> Any:
    """Load parquet dataset for a benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Loaded dataset
    """
    parquet_file = PARQUET_FILES.get(benchmark_name)
    if not parquet_file:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    parquet_path = os.path.join(PARQUET_DIR, parquet_file)
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Load parquet using datasets
    ds = load_dataset("parquet", data_files=parquet_path)
    return ds


def match_samples_with_images(sample_json_path: str, exp_names: list[str] = None) -> dict[str, Any]:
    """Match samples with their images from parquet files.

    Args:
        sample_json_path: Path to the extracted sample JSON
        exp_names: Optional list of experiment names to process (if None, process all)

    Returns:
        Dict with matched samples including images
    """
    # TODO: Change this to process different steps if needed
    TARGET_STEP = 200

    # Load sample JSON
    with open(sample_json_path) as f:
        sample_data = json.load(f)

    # Cache for loaded datasets (benchmark_name -> dataset)
    dataset_cache = {}

    # Results
    matched_results = {}

    # Process experiments
    experiments_to_process = exp_names if exp_names else list(sample_data.keys())

    for experiment in experiments_to_process:
        if experiment not in sample_data:
            print(f"Warning: Experiment {experiment} not found in sample JSON")
            continue

        print(f"\nProcessing experiment: {experiment}")
        matched_results[experiment] = {}

        exp_data = sample_data[experiment]

        # Only process the target step
        step_str = str(TARGET_STEP)
        if step_str not in exp_data:
            print(f"  Warning: Step {TARGET_STEP} not found in {experiment}")
            continue

        step_data = exp_data[step_str]
        print(f"  Processing step {TARGET_STEP}")
        matched_results[experiment][TARGET_STEP] = {}

        # Process each benchmark
        for bench_name, sample_details in step_data.items():
            print(f"    Benchmark: {bench_name}")

            # Load dataset if not cached
            if bench_name not in dataset_cache:
                try:
                    ds = load_parquet_dataset(bench_name)
                    dataset_cache[bench_name] = ds
                    print(f"      Loaded dataset: {len(ds['train'])} samples")
                except Exception as e:
                    print(f"      Error loading dataset for {bench_name}: {e}")
                    continue

            ds = dataset_cache[bench_name]

            # Match samples
            matched_samples = {"p_acc_call_fail": [], "p_err_call_succ": []}

            # Process p_acc_call_fail samples
            for sample in sample_details.get("p_acc_call_fail", []):
                idx = sample["idx"]
                try:
                    # Get original sample from dataset
                    original_sample = ds["train"][idx]

                    # Extract image
                    images = original_sample.get("images", [])
                    image = images[0] if images else None

                    # Create matched sample
                    matched_sample = {
                        "idx": idx,
                        "id": sample["id"],
                        "messages": sample["messages"],
                        "image": image,  # PIL Image object
                        "original_data": original_sample,  # Include full original data
                    }
                    matched_samples["p_acc_call_fail"].append(matched_sample)

                except Exception as e:
                    print(f"      Error processing sample {idx}: {e}")

            # Process p_err_call_succ samples
            for sample in sample_details.get("p_err_call_succ", []):
                idx = sample["idx"]
                try:
                    # Get original sample from dataset
                    original_sample = ds["train"][idx]

                    # Extract image
                    images = original_sample.get("images", [])
                    image = images[0] if images else None

                    # Create matched sample
                    matched_sample = {
                        "idx": idx,
                        "id": sample["id"],
                        "messages": sample["messages"],
                        "image": image,  # PIL Image object
                        "original_data": original_sample,  # Include full original data
                    }
                    matched_samples["p_err_call_succ"].append(matched_sample)

                except Exception as e:
                    print(f"      Error processing sample {idx}: {e}")

            matched_results[experiment][TARGET_STEP][bench_name] = matched_samples

            n_acc = len(matched_samples["p_acc_call_fail"])
            n_err = len(matched_samples["p_err_call_succ"])
            print(f"      Matched: p_acc_call_fail={n_acc}, p_err_call_succ={n_err}")

    return matched_results


def save_matched_results_to_jsonl(
    matched_results: dict[str, Any], output_dir: str = "matched_samples_jsonl"
) -> None:
    """Save matched results to jsonl files with images saved separately.

    For each experiment, creates:
    - A jsonl file with metadata
    - An images directory with saved PIL images

    JSONL format per line:
    {
        "step": step number,
        "data_source": benchmark name,
        "category": 'p_acc_call_fail' or 'p_err_call_succ',
        "idx": sample index,
        "id": sample id,
        "messages": messages list,
        "image_path": relative path to saved image,
        "original_data": full original data
    }

    Args:
        matched_results: Results from match_samples_with_images
        output_dir: Directory to save jsonl files and images
    """
    os.makedirs(output_dir, exist_ok=True)

    for exp_name, exp_data in matched_results.items():
        print(f"\nSaving {exp_name} to jsonl...")

        # Create safe filename from experiment name
        safe_filename = exp_name.replace("/", "_")

        # Create experiment-specific directories
        exp_output_dir = os.path.join(output_dir, safe_filename)
        images_dir = os.path.join(exp_output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # JSONL output path
        jsonl_path = os.path.join(exp_output_dir, f"{safe_filename}.jsonl")

        sample_count = 0
        with open(jsonl_path, "w") as f:
            for step, step_data in exp_data.items():
                for benchmark_name, categories in step_data.items():
                    # Process p_acc_call_fail samples
                    for sample in categories.get("p_acc_call_fail", []):
                        # Save image
                        image_filename = (
                            f"step{step}_{benchmark_name}_p_acc_call_fail_idx{sample['idx']}.png"
                        )
                        image_path = os.path.join(images_dir, image_filename)

                        if sample["image"] is not None:
                            sample["image"].save(image_path)
                            relative_image_path = f"images/{image_filename}"
                        else:
                            relative_image_path = None

                        # Prepare original_data without images (not JSON serializable)
                        original_data = sample.get("original_data", {})
                        if isinstance(original_data, dict):
                            original_data = {
                                k: v for k, v in original_data.items() if k != "images"
                            }
                        original_data["extra_info"].pop("tools_kwargs", None)

                        # Create record
                        sample_record = {
                            "step": step,
                            "data_source": benchmark_name,
                            "category": "p_acc_call_fail",
                            "idx": sample["idx"],
                            "id": sample["id"],
                            "messages": sample["messages"],
                            "image_path": relative_image_path,
                            "original_data": original_data,
                        }

                        # Write to jsonl
                        f.write(json.dumps(sample_record) + "\n")
                        sample_count += 1

                    # Process p_err_call_succ samples
                    for sample in categories.get("p_err_call_succ", []):
                        # Save image
                        image_filename = (
                            f"step{step}_{benchmark_name}_p_err_call_succ_idx{sample['idx']}.png"
                        )
                        image_path = os.path.join(images_dir, image_filename)

                        if sample["image"] is not None:
                            sample["image"].save(image_path)
                            relative_image_path = f"images/{image_filename}"
                        else:
                            relative_image_path = None

                        # Prepare original_data without images (not JSON serializable)
                        original_data = sample.get("original_data", {})
                        if isinstance(original_data, dict):
                            original_data = {
                                k: v for k, v in original_data.items() if k != "images"
                            }
                        original_data["extra_info"].pop("tools_kwargs", None)

                        # Create record
                        sample_record = {
                            "step": step,
                            "data_source": benchmark_name,
                            "category": "p_err_call_succ",
                            "idx": sample["idx"],
                            "id": sample["id"],
                            "messages": sample["messages"],
                            "image_path": relative_image_path,
                            "original_data": original_data,
                        }

                        # Write to jsonl

                        f.write(json.dumps(sample_record) + "\n")
                        sample_count += 1

        print(f"  Saved {sample_count} samples to {jsonl_path}")
        print(f"  Images saved to {images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Match extracted samples with images from parquet files"
    )
    parser.add_argument(
        "--sample_json", type=str, required=True, help="Path to extracted sample JSON file"
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        default=None,
        help="List of experiment names to process (if not provided, process all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="matched_samples_jsonl",
        help="Output directory for jsonl files and images",
    )

    args = parser.parse_args()

    # Match samples with images (currently hardcoded to process step 200)
    matched_results = match_samples_with_images(args.sample_json, args.exp_names)

    # Save to jsonl + images
    save_matched_results_to_jsonl(matched_results, args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for exp_name, exp_data in matched_results.items():
        print(f"\n{exp_name}:")
        for step, step_data in sorted(exp_data.items()):
            print(f"  Step {step}:")
            for bench_name, samples in step_data.items():
                n_acc = len(samples["p_acc_call_fail"])
                n_err = len(samples["p_err_call_succ"])
                if n_acc > 0 or n_err > 0:
                    print(f"    {bench_name}: p_acc_call_fail={n_acc}, p_err_call_succ={n_err}")
    print("=" * 80)


# Example usage:
# Process single experiment:
# python3 match_samples_with_images.py \
#     --sample_json sample_indices.json \
#     --exp_names "qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew" \
#     --output_dir matched_samples_jsonl
#
# Process multiple experiments:
# python3 match_samples_with_images.py \
#     --sample_json sample_indices.json \
#     --exp_names \
#         "qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew" \
#         "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew" \
#     --output_dir matched_samples_jsonl
#
# Process all experiments:
# python3 match_samples_with_images.py \
#     --sample_json sample_indices.json \
#     --output_dir matched_samples_jsonl
#
# Output structure:
# matched_samples_jsonl/
#   └── qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew/
#       ├── qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew.jsonl
#       └── images/
#           ├── step200_hrbench8k_p_acc_call_fail_idx0.png
#           ├── step200_hrbench8k_p_acc_call_fail_idx5.png
#           └── ...
if __name__ == "__main__":
    main()

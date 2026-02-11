from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("../../../evaluation_results")
CSV_PATTERN = "**/*_results.csv"

BENCHMARK_ORDER = [
    "charxiv2rq",
    "mathvision",
    "mathvista",
    "mme",
    "mmmu",
    "mmmupro",
    "vstar",
    "hrbench4k",
    "hrbench8k",
    "visualprobeasy",
    "visualprobmedium",
    "visualprobhard",
]

EXPS = [
    # "qwen25vl_instruct_75_50/all_data_natural_ratio0.75_tool_call_ratio0.5_qwen2_5_stratified_tool_reward_discriminative_range0.1",
    # "qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew",
    # "qwen25vl_instruct_75_50/natural_ratio0.75_toolcall_ratio0.50_cons_intrin_0.1_single_v3",
    # "qwen25vl_instruct_75_50/natural_ratio0.75_toolcall_ratio0.50_cons_intrin_0.1_v3",
    # "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1",
    # "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1_instri_0.1",
    "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1_instri_0.1_new_rew",
    # "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_intrin_0.1_single",
    # "qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew",
    # "qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_0.1",
    # "qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_0.1_instri_0.1_new_rew",
    # "qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_0.1_intri_0.1",
    # "qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_intrin_0.1_single",
    # "qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_no_rew",
]


def load_experiment_data() -> dict[str, pd.DataFrame]:
    experiment_data = {}

    if not RESULTS_DIR.exists():
        print(f"Results directory does not exist: {RESULTS_DIR}")
        return {}

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

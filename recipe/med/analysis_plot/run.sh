#!/usr/bin/env bash

SMOOTHING_FACTOR=0.99

# No Rew
python calculate_and_plot_area.py qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_no_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py glm46v_thinking_75_50/glm46v_natural_0.75_toolcall_0.5_no_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR


# Single
python calculate_and_plot_area.py qwen25vl_instruct_75_50/natural_ratio0.75_toolcall_ratio0.50_cons_intrin_0.1_single_v3  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_intrin_0.1_single  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_intrin_0.1_single  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py glm46v_thinking_75_50/glm46v_natural_0.75_toolcall_0.5_single  --show_area --smoothing_factor $SMOOTHING_FACTOR

# Consistency
python calculate_and_plot_area.py qwen25vl_instruct_75_50/all_data_natural_ratio0.75_tool_call_ratio0.5_qwen2_5_stratified_tool_reward_discriminative_range0.1  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_0.1  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py glm46v_thinking_75_50/glm46v_natural_0.75_toolcall_0.5_cons_0.1  --show_area --smoothing_factor $SMOOTHING_FACTOR


# Consistency + Intrinsic
python calculate_and_plot_area.py qwen25vl_instruct_75_50/natural_ratio0.75_toolcall_ratio0.50_cons_intrin_0.1_v3 --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1_instri_0.1_new_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py qwen3vl_thinking_75_50/qwen3vl_thinking_natural_0.75_toolcall_0.5_cons_0.1_instri_0.1_new_rew  --show_area --smoothing_factor $SMOOTHING_FACTOR
python calculate_and_plot_area.py glm46v_thinking_75_50/glm46v_natural_0.75_toolcall_0.5_cons_0.1_intri_0.1  --show_area --smoothing_factor $SMOOTHING_FACTOR

#!/usr/bin/env bash

SMOOTHING_FACTOR=0.99

# Reasoning benchmarks (first 6 in BENCHMARK_ORDER)
REASON_BENCHMARKS=("charxiv2rq" "mathvision" "mathvista" "mme" "mmmu" "mmmupro")

# Vision Perception benchmarks (last 6 in BENCHMARK_ORDER)
PERCEPTION_BENCHMARKS=("vstar" "hrbench4k" "hrbench8k" "visualprobeasy" "visualprobmedium" "visualprobhard")

python plot_paper_figures.py \
  qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
  qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
    --smoothing_factor $SMOOTHING_FACTOR \
    --output_filename exp1.pdf \
    --captions "Qwen2.5-VL-Instruct" "Qwen3-VL-Instruct" \
    --aggregated_benchmarks "${PERCEPTION_BENCHMARKS[@]}"

# python plot_terms_absolute.py \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#     --smoothing_factor $SMOOTHING_FACTOR \
#     --captions "Qwen2.5-VL-Instruct" "Qwen3-VL-Instruct" \
#     --output_filename exp2.pdf \
#     --aggregated_benchmarks "${PERCEPTION_BENCHMARKS[@]}"

# python plot_term_factors.py \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#     --smoothing_factor $SMOOTHING_FACTOR \
#     --captions "Qwen2.5-VL-Instruct" "Qwen3-VL-Instruct" \
#     --output_filename exp3.pdf \
#     --aggregated_benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_partition_comparison.py \
#   --exp_names \
#     qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#     qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --captions "Qwen2.5-VL-Instruct" "Qwen3-VL-Instruct" \
#   --output_filename exp4.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"

# Per-benchmark plots
# python3 plot_paper_per_benchmark.py \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen2.5-VL-Instruct" \
#   --output_filename exp1_per_benchmark_qwen25vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_paper_per_benchmark.py \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen3-VL-Instruct" \
#   --output_filename exp1_per_benchmark_qwen3vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_terms_absolute_per_benchmark.py \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen2.5-VL-Instruct" \
#   --output_filename exp2_per_benchmark_qwen25vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_terms_absolute_per_benchmark.py \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen3-VL-Instruct" \
#   --output_filename exp2_per_benchmark_qwen3vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_term_factors_per_benchmark.py \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen2.5-VL-Instruct" \
#   --output_filename exp3_per_benchmark_qwen25vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"
#
# python3 plot_term_factors_per_benchmark.py \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#   --smoothing_factor $SMOOTHING_FACTOR \
#   --caption "Qwen3-VL-Instruct" \
#   --output_filename exp3_per_benchmark_qwen3vl.pdf \
#   --benchmarks "${PERCEPTION_BENCHMARKS[@]}"

#!/bin/bash
set -e

# Experiment names
EXP_NAMES=(
    "qwen25vl_instruct_7b_crop_and_zoom_rl"
    "qwen3vl_instruct_8b_crop_and_zoom_rl"
)

# Benchmarks to aggregate (perception benchmarks)
BENCHMARKS=(
    "vstar"
    "hrbench4k"
    "hrbench8k"
    "visualprobeasy"
    "visualprobmedium"
    "visualprobhard"
)

# Captions for experiments
CAPTIONS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-8B-Instruct"
)

# Smoothing parameters
SMOOTHING_FACTOR=0.99
SMOOTHING_METHOD="time_weighted_ema"

# Output directory
OUTPUT_DIR="figures"

# Generate all three figures
python3 recipe/med/analysis_plot/plot_paper_figures.py \
    "${EXP_NAMES[@]}" \
    --aggregated_benchmarks "${BENCHMARKS[@]}" \
    --captions "${CAPTIONS[@]}" \
    --smoothing_factor $SMOOTHING_FACTOR \
    --smoothing_method $SMOOTHING_METHOD \
    --output_dir $OUTPUT_DIR \
    --figure_type all

echo ""
echo "All figures generated successfully in $OUTPUT_DIR/"
echo "  - perception_aggregated.pdf (MEASURE)"
echo "  - term_absolute_values.pdf (EXPLAIN)"
echo "  - term_factor_decomposition.pdf (DIAGNOSE)"

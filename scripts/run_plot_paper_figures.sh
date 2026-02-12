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

# Output directory and filename
OUTPUT_DIR="figures"
OUTPUT_FILENAME="qwen25vl_qwen3vl"

# Generate all three aggregated figures
python3 recipe/med/analysis_plot/plot_paper_figures.py \
    "${EXP_NAMES[@]}" \
    --aggregated_benchmarks "${BENCHMARKS[@]}" \
    --captions "${CAPTIONS[@]}" \
    --smoothing_factor $SMOOTHING_FACTOR \
    --smoothing_method $SMOOTHING_METHOD \
    --output_dir $OUTPUT_DIR \
    --output_filename $OUTPUT_FILENAME \
    --figure_type all

echo ""
echo "Aggregated figures generated successfully in $OUTPUT_DIR/"
echo "  - ${OUTPUT_FILENAME}_measure.pdf (MEASURE)"
echo "  - ${OUTPUT_FILENAME}_explain.pdf (EXPLAIN)"
echo "  - ${OUTPUT_FILENAME}_diagnose.pdf (DIAGNOSE)"

# Generate per-benchmark figures for each experiment
echo ""
echo "Generating per-benchmark figures..."
for i in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$i]}"
    caption="${CAPTIONS[$i]}"
    exp_num=$((i + 1))

    echo ""
    echo "Processing experiment ${exp_num}/${#EXP_NAMES[@]}: ${exp_name}"

    python3 recipe/med/analysis_plot/plot_paper_figures_per_benchmark.py \
        "$exp_name" \
        --benchmarks "${BENCHMARKS[@]}" \
        --caption "$caption" \
        --smoothing_factor $SMOOTHING_FACTOR \
        --smoothing_method $SMOOTHING_METHOD \
        --output_dir $OUTPUT_DIR \
        --output_filename "${OUTPUT_FILENAME}_per_bench_exp${exp_num}" \
        --figure_type all
done

echo ""
echo "All figures generated successfully in $OUTPUT_DIR/"
echo "Aggregated figures:"
echo "  - ${OUTPUT_FILENAME}_measure.pdf (MEASURE)"
echo "  - ${OUTPUT_FILENAME}_explain.pdf (EXPLAIN)"
echo "  - ${OUTPUT_FILENAME}_diagnose.pdf (DIAGNOSE)"
echo "Per-benchmark figures:"
for i in "${!EXP_NAMES[@]}"; do
    exp_num=$((i + 1))
    echo "  Experiment ${exp_num} (${CAPTIONS[$i]}):"
    echo "    - ${OUTPUT_FILENAME}_per_bench_exp${exp_num}_measure.pdf (MEASURE)"
    echo "    - ${OUTPUT_FILENAME}_per_bench_exp${exp_num}_explain.pdf (EXPLAIN)"
    echo "    - ${OUTPUT_FILENAME}_per_bench_exp${exp_num}_diagnose.pdf (DIAGNOSE)"
done

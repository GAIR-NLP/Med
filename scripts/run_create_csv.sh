#!/bin/bash
set -e

python3 recipe/med/analysis_plot/create_csv.py \
    --evaluation_dir evals \
    --exp_names \
        qwen25vl_instruct_7b_crop_and_zoom_rl \
        qwen3vl_instruct_8b_crop_and_zoom_rl \
        baseline/Qwen2.5-VL-7B-Instruct \
        baseline/Qwen3-VL-8B-Instruct

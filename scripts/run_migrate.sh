# Migrate baseline models
bash scripts/migrate_baseline_data.sh \
  Qwen2.5-VL-7B-Instruct \
  Qwen3-VL-8B-Instruct \
  --perception \
  --verbose
#
# # Migrate experiments
# bash scripts/migrate_eval_data.sh \
#   qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_no_rew \
#   qwen25vl_instruct_75_50/natural_0.75_toolcall_0.5_no_rew \
#   --perception \
#   --steps 10-200 \
#   --verbose

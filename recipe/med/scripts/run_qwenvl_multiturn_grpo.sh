#!/usr/bin/env bash

set -x

# Distributed training
export NUM_NODES=$(echo $VC_WORKER_HOSTS | tr ',' '\n' | wc -l)
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
echo "NUM_NODES: $NUM_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"

# data
export DATA_FILTER_OVERLONG_PROMPTS=${DATA_FILTER_OVERLONG_PROMPTS:-False}
export DATA_SHUFFLE=${DATA_SHUFFLE:-True}
export DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-8192}
export ENFORCE_SINGLE_TURN=${ENFORCE_SINGLE_TURN:-False}

export ROLLOUT_MODE=${ROLLOUT_MODE:-"async"}
export DO_EVAL=${DO_EVAL:-True}
# training
export ACTOR_KL_LOSS_USE=${ACTOR_KL_LOSS_USE:-False}
export ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-"low_var_kl"}
export ACTOR_USE_LIGER=${ACTOR_USE_LIGER:-False}
export USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
export USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
export LOSS_MODE=${LOSS_MODE:-"vanilla"}

export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$((ROLLOUT_N * (DATA_MAX_PROMPT_LENGTH + DATA_MAX_RES_LENGTH)))}

# rollout
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}

# fine-grained lr
export WARMUP_STYLE=${WARMUP_STYLE:-"constant"}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.05}

export ACTOR_LR_VIT=${ACTOR_LR_VIT:-$ACTOR_LR}
export ACTOR_LR_CONNECTOR=${ACTOR_LR_CONNECTOR:-$ACTOR_LR}
export ACTOR_LR_LLM=${ACTOR_LR_LLM:-$ACTOR_LR}
# ACTOR_LR_FREEZE: can be only null or a list (must contain only in 'vit', 'connector', 'llm'), e.g., "['vit', 'connector']"
export ACTOR_LR_FREEZE=${ACTOR_LR_FREEZE:-null}
export FREEZE_VISION_TOWER=${FREEZE_VISION_TOWER:-True}

# eval
export EVAL_BEFORE_TRAIN=${EVAL_BEFORE_TRAIN:-True}
export EVAL_TEMP=${EVAL_TEMP:-0}
export EVAL_TOPP=${EVAL_TOPP:-1}
export EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-False}
export EVAL_TOPK=${EVAL_TOPK:--1}

# base dir for config
export BASE_DIR=${BASE_DIR:-"/verl_vision"}
export ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-False}

# Tool
export ENABLE_MULTI_TURN=${ENABLE_MULTI_TURN:-False}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-1}
export MAX_USER_TURNS=${MAX_USER_TURNS:-1}
export MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-1280}

# misc
export USE_SHM=${USE_SHM:-False}
export FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-"triton"}
export ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"vllm"}

if [[ "$ACTOR_LOAD_PATH" == *"Qwen3-VL"* ]]; then
  export IMAGE_PATCH_SIZE=${IMAGE_PATCH_SIZE:-16}
else
  export IMAGE_PATCH_SIZE=${IMAGE_PATCH_SIZE:-14}
fi

export ENABLE_MULTIMODAL_MASK=${ENABLE_MULTIMODAL_MASK:-False}
export TOKENIZATION_SANITY_CHECK_MODE=${TOKENIZATION_SANITY_CHECK_MODE:-"disable"}

export ACC_SCALE_RANGE=${ACC_SCALE_RANGE:-"[0, 1.0]"}
export FORMAT_SCALE_RANGE=${FORMAT_SCALE_RANGE:-"[0, 1.0]"}
export TOOL_INTRINSIC_SCALE_RANGE=${TOOL_INTRINSIC_SCALE_RANGE:-"[0, 1.0]"}
export TOOL_CONSISTENCY_SCALE_RANGE=${TOOL_CONSISTENCY_SCALE_RANGE:-"[0, 1.0]"}

export FILTER_OVERLONG_MASK=${FILTER_OVERLONG_MASK:-False}


if [[ "$ACTOR_LOAD_PATH" == *"GLM"* ]]; then
  export TOOL_FORMAT="glm4"
else
  export TOOL_FORMAT="hermes"
fi

echo $TOOL_FORMAT


python3 -m verl.trainer.main_ppo \
  --config-path="${BASE_DIR}"/recipe/o3/config \
  --config-name="o3" \
  do_eval="$DO_EVAL" \
  data.train_files="$DATA_TRAIN_FILE" \
  data.val_files="$DATA_VAL_FILE" \
  data.train_batch_size="$DATA_GENERATION_BATCH_SIZE" \
  data.val_batch_size="$DATA_VAL_BATCH_SIZE" \
  data.max_prompt_length="$DATA_MAX_PROMPT_LENGTH" \
  data.max_response_length="$DATA_MAX_RES_LENGTH" \
  data.return_raw_chat="$RETURN_RAW_CHAT" \
  data.return_multi_modal_inputs="$RETURN_MULTI_MODAL_INPUTS" \
  data.filter_overlong_prompts="$DATA_FILTER_OVERLONG_PROMPTS" \
  data.truncation="error" \
  data.image_key="$DATA_IMAGE_KEYWORD" \
  data.enable_multimodal_mask="$ENABLE_MULTIMODAL_MASK" \
  data.shuffle="$DATA_SHUFFLE" \
  data.enforce_single_turn="$ENFORCE_SINGLE_TURN" \
  data.image_patch_size="$IMAGE_PATCH_SIZE" \
  data.trust_remote_code=True \
  actor_rollout_ref.model.path="$ACTOR_LOAD_PATH" \
  actor_rollout_ref.model.use_shm="$USE_SHM" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding="$USE_REMOVE_PADDING" \
  actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="$FUSED_KERNEL_BACKEND" \
  actor_rollout_ref.model.use_liger="$ACTOR_USE_LIGER" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="$ENTROPY_FROM_LOGITS_WITH_CHUNKING" \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.freeze_vision_tower="$FREEZE_VISION_TOWER" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$ACTOR_PPO_GLOBAL_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_PPO_MICRO_BSZ" \
  actor_rollout_ref.actor.use_dynamic_bsz="$USE_DYNAMIC_BSZ" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.clip_ratio="$ACTOR_CLIP_RATIO" \
  actor_rollout_ref.actor.clip_ratio_low="$ACTOR_CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$ACTOR_CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.loss_agg_mode="$ACTOR_LOSS_AGG_MODE" \
  actor_rollout_ref.actor.policy_loss.loss_mode="$LOSS_MODE" \
  actor_rollout_ref.actor.use_kl_loss="$ACTOR_KL_LOSS_USE" \
  actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEFF" \
  actor_rollout_ref.actor.kl_loss_type="$ACTOR_KL_LOSS_TYPE" \
  actor_rollout_ref.actor.entropy_coeff="$ACTOR_ENTROPY_COEFF" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.actor.optim.warmup_style="$WARMUP_STYLE" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_FSDP_OMT_OFFLOAD" \
  actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_P_MICRO_BSZ" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MAX_GPU_MEM" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMP" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$EVAL_TEMP" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$EVAL_TOPP" \
  actor_rollout_ref.rollout.val_kwargs.top_k="$EVAL_TOPK" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$EVAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE" \
  +actor_rollout_ref.rollout.limit_images=15 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOG_P_MICRO_BSZ" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$ACTOR_FSDP_PARAM_OFFLOAD" \
  actor_rollout_ref.rollout.multi_turn.enable="$ENABLE_MULTI_TURN" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="$TOKENIZATION_SANITY_CHECK_MODE" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.format="$TOOL_FORMAT" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.adv_estimator="$ALGO_ADV_ESTIMATOR" \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef="$ALGO_KL_COEF" \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$TRAIN_PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes="$NUM_NODES" \
  trainer.default_local_dir="$TRAIN_SAVE_PATH/$EXP_NAME" \
  trainer.val_before_train="$EVAL_BEFORE_TRAIN" \
  trainer.resume_mode=auto \
  trainer.save_freq="$TRAIN_SAVE_FREQ" \
  trainer.test_freq="$TRAIN_TEST_FREQ" \
  trainer.total_epochs="$TRAIN_TOTAL_EPOCHS" \
  trainer.trajectory_data_dir="/verl_vision/trajectories" \
  trainer.filter_overlong_mask="$FILTER_OVERLONG_MASK" \
  reward_model.reward_manager=remote \
  reward_model.reward_kwargs.acc_scale_range="$ACC_SCALE_RANGE" \
  reward_model.reward_kwargs.format_scale_range="$FORMAT_SCALE_RANGE" \
  reward_model.reward_kwargs.tool_intrinsic_scale_range="$TOOL_INTRINSIC_SCALE_RANGE" \
  reward_model.reward_kwargs.tool_consistency_scale_range="$TOOL_CONSISTENCY_SCALE_RANGE"

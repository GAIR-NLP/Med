#!/usr/bin/env bash

set -x

export BASE_DIR=${BASE_DIR:-"your-code-dir"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./evaluation_results"}

export REMOTE_REWARD_JOB_ID=${REMOTE_REWARD_JOB_ID:-"j-xxxxxxxx"}

export NUM_NODES=${NUM_NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export USE_SHM=${USE_SHM:-True}

export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}

export ACTOR_LOAD_PATH=${ACTOR_LOAD_PATH:-"your-ckpt-path/Qwen2.5-VL-7B-Instruct"}

export DATA_VAL_FILE=${DATA_VAL_FILE:-"[your-eval-data-files]"}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}

export ENABLE_MULTI_TURN=${ENABLE_MULTI_TURN:-False}
export MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-1}
export MAX_USER_TURNS=${MAX_USER_TURNS:-1}
export MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}

export DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-12800}
export DATA_MAX_RES_LENGTH=${DATA_MAX_RES_LENGTH:-8192}

export ROLLOUT_MODE=${ROLLOUT_MODE:-"async"}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-1280}

export RETURN_RAW_CHAT=${RETURN_RAW_CHAT:-False}

export EVAL_PROJECT_NAME=${EVAL_PROJECT_NAME:-"Eval"}

export ROLLOUT_MAX_GPU_MEM=${ROLLOUT_MAX_GPU_MEM:-0.85}

export PASSK=${PASSK:-1}

export VERIFICATION_LOAD_PATH=${VERIFICATION_LOAD_PATH:-"your-ckpt-path/Qwen2.5-32B-Instruct"}
export VERIFICATION_TP_SIZE=${VERIFICATION_TP_SIZE:-1}

export VERIFICATION_ENABLE=${VERIFICATION_ENABLE:-"True"}
export VERIFICATION_MAX_TOKENS=${VERIFICATION_MAX_TOKENS:-1024}

export ACC_SCALE_RANGE=${ACC_SCALE_RANGE:-"[0, 1.0]"}
export FORMAT_SCALE_RANGE=${FORMAT_SCALE_RANGE:-"[0, 0.0]"}
export TOOL_INTRINSIC_SCALE_RANGE=${TOOL_INTRINSIC_SCALE_RANGE:-"[0, 0.0]"}
export TOOL_CONSISTENCY_SCALE_RANGE=${TOOL_CONSISTENCY_SCALE_RANGE:-"[0, 0.0]"}
export USE_RAY_ACTOR=${USE_RAY_ACTOR:-"True"}
export ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"sglang"}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-"False"}
export DATA_VAL_BATCH_SIZE=${DATA_VAL_BATCH_SIZE:-4096}

export EVAL_TEMP=${EVAL_TEMP:-0.0}
export EVAL_TOPP=${EVAL_TOPP:-1}
export EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE:-False}
export EVAL_TOPK=${EVAL_TOPK:--1}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

# # Set evaluation parameters based on model type
if [[ "$ACTOR_LOAD_PATH" == *"Qwen3-VL-8B-Thinking"* ]] || [[ "$EXP_NAME" == *"qwen3vl_thinking"* ]]; then
    export EVAL_TEMP=1.0
    export EVAL_TOPP=0.95
    export EVAL_DO_SAMPLE=True
    export EVAL_TOPK=20
    export DATA_MAX_RES_LENGTH=25600
fi

if [[ "$ACTOR_LOAD_PATH" == *"GLM"* ]] || [[ "$ACTOR_LOAD_PATH" == *"glm"* ]]; then
    export EVAL_TEMP=0.8
    export EVAL_TOPP=0.6
    export EVAL_DO_SAMPLE=True
    export EVAL_TOPK=2
    export DATA_MAX_RES_LENGTH=16384
    export REPETITION_PENALTY=1.1
fi

if [[ "$ACTOR_LOAD_PATH" == *"GLM"* ]] || [[ "$ACTOR_LOAD_PATH" == *"glm"* ]]; then
  export TOOL_FORMAT="glm4"
else
  export TOOL_FORMAT="hermes"
fi

echo $TOOL_FORMAT

echo "EVAL_TEMP=${EVAL_TEMP}"
echo "EVAL_TOPP=${EVAL_TOPP}"
echo "EVAL_DO_SAMPLE=${EVAL_DO_SAMPLE}"
echo "EVAL_TOPK=${EVAL_TOPK}"
echo "DATA_MAX_RES_LENGTH=${DATA_MAX_RES_LENGTH}"


python3 -m recipe.med.eval.main_eval \
  --config-path="${BASE_DIR}"/recipe/med/config \
  data.val_files="$DATA_VAL_FILE" \
  data.max_prompt_length="$DATA_MAX_PROMPT_LENGTH" \
  data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  data.filter_overlong_prompts_workers="$DATALOADER_NUM_WORKERS" \
  data.max_response_length="$DATA_MAX_RES_LENGTH" \
  data.val_batch_size="$DATA_VAL_BATCH_SIZE" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.return_raw_chat="$RETURN_RAW_CHAT" \
  data.return_multi_modal_inputs="$RETURN_MULTI_MODAL_INPUTS" \
  critic.enable=False \
  actor_rollout_ref.model.path="$ACTOR_LOAD_PATH" \
  actor_rollout_ref.model.use_shm="$USE_SHM" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.val_kwargs.n="$PASSK" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$EVAL_TEMP" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$EVAL_TOPP" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$EVAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.top_k="$EVAL_TOPK" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$ROLLOUT_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.multi_turn.enable="$ENABLE_MULTI_TURN" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.format="$TOOL_FORMAT" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_MAX_GPU_MEM" \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.repetition_penalty="$REPETITION_PENALTY" \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$EVAL_PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes="$NUM_NODES" \
  reward_model.reward_manager=remote \
  reward_model.reward_kwargs.acc_scale_range="$ACC_SCALE_RANGE" \
  reward_model.reward_kwargs.format_scale_range="$FORMAT_SCALE_RANGE" \
  reward_model.reward_kwargs.tool_intrinsic_scale_range="$TOOL_INTRINSIC_SCALE_RANGE" \
  reward_model.reward_kwargs.tool_consistency_scale_range="$TOOL_CONSISTENCY_SCALE_RANGE" \
  llm_verification.enabled="$VERIFICATION_ENABLE" \
  llm_verification.verifier_model.path="$VERIFICATION_LOAD_PATH" \
  llm_verification.verifier_model.max_tokens="$VERIFICATION_MAX_TOKENS" \
  llm_verification.verifier_model.tensor_parallel_size="$VERIFICATION_TP_SIZE" \
  llm_verification.verifier_model.use_ray_actor="$USE_RAY_ACTOR" \
  llm_verification.verifier_model.backend="$ROLLOUT_BACKEND" \
  output_dir="$OUTPUT_DIR" \
  +reward_model.reward_kwargs.remote_reward_job_id="$REMOTE_REWARD_JOB_ID" \


export BASE_DIR=/inspire/ssd/project/cq-scientific-data-factory/mitiantian-253108120109/yema/code/Med

export REMOTE_REWARD_JOB_ID=j-1e8ohg4srz

export NUM_NODES=1
export GPUS_PER_NODE=4

export ACTOR_LOAD_PATH=/inspire/hdd/global_user/mitiantian-253108120109/yema/ckpts/Qwen3-VL-8B-Instruct

export DATA_VAL_FILE=[/inspire/ssd/project/cq-scientific-data-factory/mitiantian-253108120109/yema/qb-ilm_data/Med_eval_data/vstar_bench_single_turn_format_0.0_length_0.0_maxlen_10564_num_191.parquet]

  # /inspire/ssd/project/cq-scientific-data-factory/mitiantian-253108120109/yema/qb-ilm_data/Med_eval_data/vstar_bench_tool_agent_format_0.0_length_0.0_maxlen_10564_num_191.parquet
export WANDB_MODE=offline

export VERIFICATION_LOAD_PATH=$ACTOR_LOAD_PATH

bash recipe/med/scripts/eval.sh


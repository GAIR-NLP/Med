#!/usr/bin/env bash

export BASE_DIR=${BASE_DIR:-"your-code-dir"}

export PYTHONPATH=${BASE_DIR}/:${PYTHONPATH}

export DATA_TRAIN_FILE=${DATA_TRAIN_FILE:-"[your-data-files]"}

echo $DATA_TRAIN_FILE

cd ${BASE_DIR}

sleep 10

ray status

sleep 10

bash recipe/med/scripts/serve_vision_tool.sh

sleep 10

ray status

sleep 10

bash recipe/med/scripts/train.sh

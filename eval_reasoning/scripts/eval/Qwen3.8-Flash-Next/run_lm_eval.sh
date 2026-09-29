#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
PORT=${2:-8000}
OUTPUT_DIR=${3}

EXTRA_ARGS=()
if [ -n "${OUTPUT_DIR}" ]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

python run_lm_eval.py \
  --model_path ${MODEL_PATH} \
  --api_url http://127.0.0.1:${PORT}/v1 \
  "${EXTRA_ARGS[@]}"

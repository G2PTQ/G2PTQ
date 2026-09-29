#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
PORT=${2:-8000}

python run_eval.py \
  --config configs/Qwen3.8-Flash-Next.yaml \
  --model_path ${MODEL_PATH} \
  --api_url http://127.0.0.1:${PORT}/v1 \
  --datasets arxivmath ifbench live_code_bench gpqa_diamond

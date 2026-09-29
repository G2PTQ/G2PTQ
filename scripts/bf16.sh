#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
DEVICE=${2}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})

FIRST_DEVICE=$(echo "${DEVICE}" | cut -d',' -f1)
NPROC=$(echo "${DEVICE}" | awk -F',' '{print NF}')

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

# Execute the distributed run
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=${NPROC} --rdzv_endpoint=localhost:2940${FIRST_DEVICE} ./ptq.py \
    --model ${MODEL_PATH} \
    --exp bf16 --deterministic \
    --lm_eval --lm_eval_batch_size 32 \

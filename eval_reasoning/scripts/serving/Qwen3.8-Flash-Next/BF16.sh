#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
DEVICE=${2:-0,1,2,3,4,5,6,7}

FIRST_DEVICE=$(echo "${DEVICE}" | cut -d',' -f1)
NPROC=$(echo "${DEVICE}" | awk -F',' '{print NF}')

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

vllm serve ${MODEL_PATH} --port 800${FIRST_DEVICE} \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --enable-expert-parallel \
    --tensor-parallel-size ${NPROC} \
    --moe-backend triton \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \

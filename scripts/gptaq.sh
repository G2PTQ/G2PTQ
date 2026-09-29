#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
BITS=${2}           # 4
DEVICE=${3}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=1024
SEQ_LEN=2048

FIRST_DEVICE=$(echo "${DEVICE}" | cut -d',' -f1)
NPROC=$(echo "${DEVICE}" | awk -F',' '{print NF}')

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

# Execute the distributed run
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=${NPROC} --rdzv_endpoint=localhost:2940${FIRST_DEVICE} ./ptq.py \
    --model ${MODEL_PATH} \
    --exp gptaq --deterministic --enable_torch_compile \
    --dataset neuralmagic --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptaq --w_bits ${BITS} --w_clip --act_order --bsz 4 --layer_bsz 64 \
    --rotate \
    --lm_eval --lm_eval_batch_size 32 \
    --offload_inps --offload_hessians \

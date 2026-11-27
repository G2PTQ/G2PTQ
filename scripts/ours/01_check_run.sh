#!/bin/bash
# Go/no-go: small G2PTQ run on the 4B model (16 calibration sequences, no autotune).
# Confirms it runs, fits in 40 GB, and gives the time per run. Then a 50-prompt IFEval sanity check.
set -euo pipefail
source scripts/ours/config.sh

${LAUNCH} --model ${MODEL} --exp check_g2ptq --deterministic --nsamples 16 --seq_len ${SEQ_LEN} \
  --output_dir ${OUT} --cache_dir ${CACHE} --eval_datasets calib_heldout_if \
  --dataset calib_ifchat --w_bits 4 --w_groupsize 128 --w_clip --bsz 4 --layer_bsz 64 \
  --offload_inps --offload_hessians --save_dequant_model \
  --w_method g2ptq --num_groups 4 --act_order --gptq_backend triton --gptq_graph --kl_topk 256

echo "Time and memory:"; grep -E "Quantization Time|Peak GPU Memory" ${OUT}/${MODEL_NAME}/check_g2ptq/logs/*.log | tail -2 || true
${PY_EVAL} ours/eval/ifeval_vllm.py --model ${OUT}/${MODEL_NAME}/check_g2ptq/dequant_model \
  --out_dir ${OUT}/${MODEL_NAME}/check_g2ptq/ifeval_limit50 --limit 50

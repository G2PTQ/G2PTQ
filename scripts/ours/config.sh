#!/bin/bash
# Shared settings for all scripts in scripts/ours/. Edit here, not in the individual scripts.

MODEL=${MODEL:-./modelzoo/Qwen3.5/Qwen3.5-4B}
MODEL_NAME=$(basename "${MODEL}")
GPU=${GPU:-0}
# Python for vLLM steps (IFEval, self-generated data). vLLM pins its own torch, so it usually
# lives in a separate environment, e.g. PY_EVAL=/content/venv-eval/bin/python on Colab.
PY_EVAL=${PY_EVAL:-python}
export CUDA_VISIBLE_DEVICES=${GPU}

NSAMPLES=128
SEQ_LEN=2048
SEED=42
OUT=./outputs
CACHE=./cache

# Calibration sets: raw text, instruction-following chat, self-generated
CALIB_SETS=(wikitext2 calib_ifchat calib_selfgen)
# Held-out selection data (never used for calibration)
EVAL_SETS=(calib_heldout_if wikitext2)

# Size budget and bit settings (set TYPE_MAP from 02_bit_search.sh; UNIFORM_* from `bpw.py --match`)
BUDGET=3.5
TYPE_GS=128
TYPE_MAP=./outputs/bit_search/best_bits_map.json
UNIFORM_BITS=3
UNIFORM_GS=32

# Token-weight settings (tune LAMBDA on held-out KL with 03_token_weights.sh --sweep)
LAMBDA=1.0
TW_MODE=source

LAUNCH="python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2950${GPU} ./ptq.py"

COMMON="--model ${MODEL} --deterministic --nsamples ${NSAMPLES} --seq_len ${SEQ_LEN} --seed ${SEED} \
 --output_dir ${OUT} --cache_dir ${CACHE} --eval_datasets ${EVAL_SETS[*]} --w_clip --bsz 4 --layer_bsz 64 \
 --offload_inps --offload_hessians --save_dequant_model"

G2PTQ_ARGS="--w_method g2ptq --num_groups 4 --act_order --gptq_backend triton --gptq_graph \
 --enable_torch_compile --kl_topk 256 --true_sequential_ratio 1.0 \
 --autotune --autotune_nsamples 64 --autotune_min 0.0 --autotune_max 3.0 --autotune_stepsize 0.1"

GPTQ_ARGS="--w_method gptq --act_order --gptq_backend triton --gptq_graph"

uniform_args() { echo "--w_bits ${UNIFORM_BITS} --w_groupsize ${UNIFORM_GS}"; }
typeaware_args() {
  [ -f "${TYPE_MAP}" ] || { echo "Missing ${TYPE_MAP}: run scripts/ours/02_bit_search.sh first" >&2; exit 1; }
  echo "--w_bits 4 --w_groupsize ${TYPE_GS} --w_bits_map ${TYPE_MAP}"
}
tw_path() { echo "${CACHE}/token_weights/${MODEL_NAME}-$1-${TW_MODE}-lam${LAMBDA}.pt"; }

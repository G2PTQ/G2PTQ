#!/bin/bash
# Idea 1: per-type sensitivity + search under the size budget (RTN), confirm the top 2 with GPTQ.
# Writes outputs/bit_search/best_bits_map.json (used as TYPE_MAP).
set -euo pipefail
source scripts/ours/config.sh

python ours/search/bit_search.py --model ${MODEL} --budget ${BUDGET} --w_groupsize ${TYPE_GS} \
  --dataset calib_ifchat --nsamples ${NSAMPLES} --seq_len ${SEQ_LEN} --eval_datasets ${EVAL_SETS[*]} \
  --confirm_top 2 --out_dir ${OUT}/bit_search --output_dir ${OUT} --cache_dir ${CACHE} --seed ${SEED} \
  --launcher "${LAUNCH}" "$@"

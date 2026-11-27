#!/bin/bash
# Idea 2: compute token weights for every calibration set with the current LAMBDA / TW_MODE.
#   bash scripts/ours/03_token_weights.sh            # weights for all calibration sets
#   bash scripts/ours/03_token_weights.sh --sweep    # tune lambda on calib_ifchat (uniform bits, G2PTQ)
set -euo pipefail
source scripts/ours/config.sh

compute() {  # $1 = calibration set, $2 = lambda, $3 = mode
  local out="${CACHE}/token_weights/${MODEL_NAME}-$1-$3-lam$2.pt"
  [ -f "${out}" ] || python ours/token_weights/compute_token_weights.py --model ${MODEL} --dataset $1 \
    --nsamples ${NSAMPLES} --seq_len ${SEQ_LEN} --seed ${SEED} --cache_dir ${CACHE} \
    --lam $2 --mode $3 --out "${out}"
}

if [ "${1:-}" = "--sweep" ]; then
  # Held-out KL for a few settings; pick the best and set LAMBDA / TW_MODE in config.sh
  for mode in source prediction; do
    for lam in 0.0 0.5 1.0 2.0; do
      compute calib_ifchat ${lam} ${mode}
      ${LAUNCH} ${COMMON} ${G2PTQ_ARGS} $(uniform_args) --dataset calib_ifchat \
        --exp sweep_tw_${mode}_lam${lam} \
        --token_weights ${CACHE}/token_weights/${MODEL_NAME}-calib_ifchat-${mode}-lam${lam}.pt
    done
  done
  grep -H "KL-" ${OUT}/${MODEL_NAME}/sweep_tw_*/metrics.json
  exit 0
fi

for calib in "${CALIB_SETS[@]}"; do compute ${calib} ${LAMBDA} ${TW_MODE}; done

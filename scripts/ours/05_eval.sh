#!/bin/bash
# IFEval on the BF16 model and every saved dequantized model, then the results table and
# the main paired comparisons. Finished evaluations are skipped.
set -euo pipefail
source scripts/ours/config.sh

[ -f ${OUT}/${MODEL_NAME}/bf16/ifeval/summary.json ] || \
  ${PY_EVAL} ours/eval/ifeval_vllm.py --model ${MODEL} --out_dir ${OUT}/${MODEL_NAME}/bf16/ifeval
for d in ${OUT}/${MODEL_NAME}/*/dequant_model; do
  exp_dir=$(dirname ${d})
  [ -f ${exp_dir}/ifeval/summary.json ] && continue
  ${PY_EVAL} ours/eval/ifeval_vllm.py --model ${d} --out_dir ${exp_dir}/ifeval
done

python ours/eval/collect_results.py --model ${MODEL} --output_dir ${OUT} --out ${OUT}/results.csv

# Paired comparisons pooled over calibration sets (B vs A)
compare() {  # $1 = A prefix, $2 = B prefix, $3 = label
  local pairs=()
  for c in "${CALIB_SETS[@]}"; do
    a=${OUT}/${MODEL_NAME}/$1_${c}/ifeval/responses.jsonl; b=${OUT}/${MODEL_NAME}/$2_${c}/ifeval/responses.jsonl
    [ -f ${a} ] && [ -f ${b} ] && pairs+=("${a}:${b}")
  done
  [ ${#pairs[@]} -gt 0 ] || return 0
  echo "== $3"
  python ours/eval/bootstrap.py --pairs "${pairs[@]}" --json ${OUT}/bootstrap_$2_vs_$1.json
}
compare g2ptq_uniform   g2ptq_typeaware      "Idea 1: type-aware vs uniform bits (G2PTQ)"
compare g2ptq_uniform   g2ptq_uniform_tw     "Idea 2: token weights vs plain (uniform bits)"
compare g2ptq_uniform   g2ptq_typeaware_tw   "Both ideas vs plain G2PTQ"
compare gptq_uniform    g2ptq_typeaware_tw   "Both ideas vs GPTQ"

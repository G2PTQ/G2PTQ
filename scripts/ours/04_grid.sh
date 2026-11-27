#!/bin/bash
# Main experiment: 2x2 (uniform / type-aware bits) x (plain / token-weighted G2PTQ) x 3 calibration sets,
# plus GPTQ baselines (uniform and type-aware) x 3 calibration sets. Finished runs are skipped.
#   bash scripts/ours/04_grid.sh [calibration set ...]   # default: all of CALIB_SETS
set -euo pipefail
source scripts/ours/config.sh
SETS=("$@"); [ ${#SETS[@]} -gt 0 ] || SETS=("${CALIB_SETS[@]}")

run() {  # $1 = exp name, rest = ptq.py args
  local exp=$1; shift
  if [ -f "${OUT}/${MODEL_NAME}/${exp}/metrics.json" ]; then echo "skip ${exp}"; return; fi
  ${LAUNCH} ${COMMON} --exp ${exp} "$@"
}

for calib in "${SETS[@]}"; do
  tw=$(tw_path ${calib})
  [ -f "${tw}" ] || { echo "Missing ${tw}: run scripts/ours/03_token_weights.sh" >&2; exit 1; }
  run g2ptq_uniform_${calib}      ${G2PTQ_ARGS} $(uniform_args)   --dataset ${calib}
  run g2ptq_typeaware_${calib}    ${G2PTQ_ARGS} $(typeaware_args) --dataset ${calib}
  run g2ptq_uniform_tw_${calib}   ${G2PTQ_ARGS} $(uniform_args)   --dataset ${calib} --token_weights ${tw}
  run g2ptq_typeaware_tw_${calib} ${G2PTQ_ARGS} $(typeaware_args) --dataset ${calib} --token_weights ${tw}
  run gptq_uniform_${calib}       ${GPTQ_ARGS}  $(uniform_args)   --dataset ${calib}
  run gptq_typeaware_${calib}     ${GPTQ_ARGS}  $(typeaware_args) --dataset ${calib}
done

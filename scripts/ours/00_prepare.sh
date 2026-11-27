#!/bin/bash
# Download the model and datasets, build the calibration sets, generate the self-generated set.
# Run once, from the repo root, on the GPU machine (needs Hugging Face access).
set -euo pipefail
source scripts/ours/config.sh

hf download Qwen/Qwen3.5-4B --local-dir "${MODEL}"
hf download Salesforce/wikitext --repo-type dataset --local-dir ./datasets/wikitext

# IFEval checkers need NLTK's punkt tokenizer
${PY_EVAL} -c "import nltk; nltk.download('punkt_tab')"

# Tulu 3 Persona IF + Conifer, decontaminated against IFEval and IFBench
python ours/data/build_calib.py --out_dir ./datasets/calib

# The BF16 model answers the same prompts (needs vLLM)
${PY_EVAL} ours/data/generate_selfgen.py --model "${MODEL}" --calib_dir ./datasets/calib

# Size of the candidate settings, for choosing UNIFORM_* in config.sh
python ours/size/bpw.py --model "${MODEL}" --w_groupsize ${TYPE_GS} --enumerate --budget ${BUDGET}
python ours/size/bpw.py --model "${MODEL}" --match ${BUDGET}

# Reasoning Evaluation

This repository provides evaluations for the exported quantized models.


## Contents

- [Preparations](#preparations)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Preparation](#model-preparation)
- [Usage](#usage)
  - [Serving a Checkpoint](#serving-a-checkpoint)
  - [Reasoning Benchmarks](#reasoning-benchmarks)
  - [QA Benchmarks](#qa-benchmarks)
  - [Summarizing Results](#summarizing-results)


## Preparations

### Installation

```bash
conda create -n evalscope python=3.12 -y
conda activate evalscope

uv pip install -r requirements.txt --torch-backend=auto
```


### Data Preparation

#### Reasoning Benchmarks

`download_datasets.py` fetches every benchmark from ModelScope into `./datasets/<name>`:

```bash
python download_datasets.py
```

| Dataset           | ModelScope repo                                              |
| ----------------- | ------------------------------------------------------------ |
| `gpqa_diamond`    | [AI-ModelScope/gpqa_diamond](https://modelscope.cn/datasets/AI-ModelScope/gpqa_diamond) |
| `live_code_bench` | [evalscope/livecodebench_code_generation_lite_parquet](https://modelscope.cn/datasets/evalscope/livecodebench_code_generation_lite_parquet) |
| `arxivmath`       | [evalscope/arxivmath](https://modelscope.cn/datasets/evalscope/arxivmath) |
| `ifbench`         | [allenai/IFBench_test](https://modelscope.cn/datasets/allenai/IFBench_test) |


#### QA Benchmarks

Please refer to the parent directory's `README.md` for detailed instructions about data preparation.

```bash
ln -s ../datasets ./
```

### Model Preparation

Please refer to the parent directory's `README.md` for detailed instructions about model preparation.

```bash
ln -s ../modelzoo ./
```

## Usage

### Serving a Checkpoint

Serve a BF16 model (`scripts/serving/Qwen3.8-Flash-Next/BF16.sh`):

```bash
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
```

Serve a W4A16 model (`scripts/serving/Qwen3.8-Flash-Next/W4A16.sh`):

```bash
#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
DEVICE=${2:-0,1,2,3}

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
  --moe-backend marlin \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
```

### Reasoning Benchmarks

Run the reasoning benchmarks with `scripts/eval/Qwen3.8-Flash-Next/run_eval.sh`:

```bash
#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
PORT=${2:-8000}

python run_eval.py \
  --config configs/Qwen3.8-Flash-Next.yaml \
  --model_path ${MODEL_PATH} \
  --api_url http://127.0.0.1:${PORT}/v1 \
  --datasets arxivmath ifbench live_code_bench gpqa_diamond
```

Other commonly used arguments for `run_eval.py`:

- **Sampling.** `--config` selects a YAML profile under `configs/`. A profile may set only the six sampling keys (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repetition_penalty`).

### QA Benchmarks

Run the same seven QA tasks as the parent repo's `ptq.py --lm_eval` (`piqa`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `lambada_openai`, `ceval-valid`) against a served checkpoint (`scripts/eval/Qwen3.8-Flash-Next/run_lm_eval.sh`):

```bash
#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # modelzoo/Qwen4/Qwen3.8-Flash-Next
PORT=${2:-8000}
OUTPUT_DIR=${3}

EXTRA_ARGS=()
if [ -n "${OUTPUT_DIR}" ]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

python run_lm_eval.py \
  --model_path ${MODEL_PATH} \
  --api_url http://127.0.0.1:${PORT}/v1 \
  "${EXTRA_ARGS[@]}"
```

### Summarizing Results

`summarize_results.py` turns the finished per-seed reports into one markdown table:

```bash
# every run under outputs/
python summarize_results.py

# specific runs, in the given order, plus a per-seed breakdown, written to a file
python summarize_results.py \
  outputs/Qwen3.8-Flash-Next-GPTQ-W4A16 \
  outputs/Qwen3.8-Flash-Next-G2PTQ-W4A16 \
  --per_seed -o results.md
```

Narrow the view with `--seeds 42 43` or `--datasets arxivmath ifbench`.
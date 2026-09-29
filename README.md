# G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation

[![arXiv](https://img.shields.io/badge/G2PTQ-2609.31009-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2609.31009)

G$^2$PTQ is a unified PTQ framework with Generalized Gradient Compensation that integrates both first- and second-order information under a globally supervised, block-wise optimization objective. By refreshing gradient and Hessian estimates before quantizing each Transformer block, G$^2$PTQ avoids the staleness of prior global methods. Furthermore, to stabilize the exact first-order compensation, we introduce a trust-region scaling mechanism that dynamically bounds the gradient step to prevent exploding weight updates. Finally, we derive efficient implementations for block-wise Hessian approximation and exact gradient compensation. Experimental results on various model families and bit-widths demonstrate that G$^2$PTQ enables better alignment with the full-precision model, outperforming state-of-the-art baselines.


## News 🔥

- [2026/09] Initial public release. Check our paper [here](https://arxiv.org/abs/2609.31009).


## Contents

- [Key Features](#key-features)
- [Preparations](#preparations)
  - [Installation](#installation)
  - [Optional CUDA Extensions](#optional-cuda-Extensions)
  - [Data Preparation](#data-preparation)
  - [Model Preparation](#model-preparation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [References](#references)


## Key Features

- **Comprehensive Support for Weight Quantization Methods.** We implement G$^2$PTQ alongside RTN, GPTQ, GPTAQ, and GuidedQuant baselines within a unified pipeline, featuring optional support for weight clipping and rotation.
- **Extensible Model and Dataset Coverage.** We currently support a broad range of model architectures, including Llama, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, and Qwen4-Exp, alongside a comprehensive suite of calibration datasets, such as NeuralMagic and Open-Perfectblend. Furthermore, support for additional models and calibration datasets can be seamlessly integrated via a flexible registration mechanism.
- **Quantization Efficiency Optimization**. 1) **Distributed Quantization.** We support data-parallel quantization, distributing both calibration data processing and linear layer quantization across multiple devices. 2) **Flexible Offloading.** Model weights can be offloaded to the CPU or disk, with all ranks accessing a single shared copy. Calibration hidden states and Hessians can also be offloaded to the CPU, facilitated by asynchronous memory transfers. 3) **Kernel Fusion.** We provide Triton kernels for lazy-batch intra-block error compensation, complemented by optional CUDA graph support. 4) **Layer-batched Quantization**. Linear layers sharing identical shapes can be batched and quantized concurrently.
- **Quantized Model Export.** We support exporting quantized models to the `compressed-tensors` format,  ensuring seamless compatibility with inference engines such as vLLM and SGLang.
- **Comprehensive Evaluation.** We support a diverse array of benchmarks for assessing model degradation, encompassing distributional metrics such as KL, EAR, and PPL, as well as downstream tasks including commonsense question answering. Additionally, we facilitate the evaluation of exported quantized models on complex reasoning benchmarks.


## Preparations

### Installation

Install the project dependencies from the repository root:

```bash
conda create -n g2ptq python=3.12 -y
conda activate g2ptq
uv pip install -r requirements.txt --torch-backend=auto

git submodule update --init third-party/fast-hadamard-transform
python -m pip install -v --no-build-isolation -e ./third-party/fast-hadamard-transform
```

**Note:** To run models like Qwen3.8-Flash-Next, please upgrade to `transformers==5.16.1`.


### Optional CUDA Extensions

FlashAttention-3 and causal-conv1d are model-dependent extensions. While not strictly required for every model, they are highly recommended when quantizing models that utilize corresponding attention or linear-attention implementations.


#### FlashAttention-3

```bash
git submodule update --init third-party/flash-attention
git -C third-party/flash-attention submodule update --init csrc/cutlass
python -m pip install packaging ninja

FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_DISABLE_SM80=TRUE \
MAX_JOBS=4 \
NVCC_THREADS=2 \
python -m pip install -v --no-build-isolation \
    ./third-party/flash-attention/hopper
```

Once built, select it with `--attn_implementation flash_attention_3`.

#### causal-conv1d

```bash
git submodule update --init third-party/causal-conv1d
python -m pip install packaging ninja

CAUSAL_CONV1D_FORCE_BUILD=TRUE \
MAX_JOBS=4 \
python -m pip install -v --no-build-isolation \
    ./third-party/causal-conv1d
```


### Data Preparation

The dataset loaders access local files stored in the `./datasets` directory. Each dataset must be downloaded into this local directory prior to use.

#### Calibration and KL/EAR/PPL Evaluation

| Dataset             | Local dir                                | URL                                                          |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| `neuralmagic`       | `./datasets/LLM_compression_calibration` | [https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration](https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration) |
| `wikitext2`         | `./datasets/wikitext`                    | [https://huggingface.co/datasets/Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) |
| `ultrachat_2k`      | `./datasets/ultrachat_2k`                | [https://huggingface.co/datasets/neuralmagic/ultrachat_2k](https://huggingface.co/datasets/neuralmagic/ultrachat_2k) |
| `numinamath`        | `./datasets/NuminaMath-1.5`              | [https://huggingface.co/datasets/AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) |
| `open_perfectblend` | `./datasets/open-perfectblend`           | [https://huggingface.co/datasets/mlabonne/open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) |

#### Commonsense QA Evaluation

QA evaluation resolves datasets through local `lm-eval` task configs under `./datasets/lm_eval_configs/tasks`. First copy the task configs out of the installed package:

```bash
LM_EVAL_TASKS=$(python -c "import lm_eval.tasks, os; print(os.path.dirname(lm_eval.tasks.__file__))")
mkdir -p ./datasets/lm_eval_configs/tasks
cp -r "${LM_EVAL_TASKS}/." ./datasets/lm_eval_configs/tasks/
```

Then set `dataset_path` in each QA task's config to the matching local directory below:

| Dataset         | Local dir                   | URL                                                          |
| --------------- | --------------------------- | ------------------------------------------------------------ |
| ARC-C and ARC-E | `./datasets/ai2_arc`        | [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) |
| C-eval          | `./datasets/ceval-exam`     | [https://huggingface.co/datasets/ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) |
| HellaSwag       | `./datasets/hellaswag`      | [https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| LAMBADA         | `./datasets/lambada_openai` | [https://huggingface.co/datasets/EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai) |
| PIQA            | `./datasets/piqa`           | [https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) |
| WinoGrande      | `./datasets/winogrande`     | [https://huggingface.co/datasets/allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) |


### Model Preparation

Download model checkpoints into `./modelzoo`. For example:

```bash
hf download Qwen/Qwen3-0.6B \
    --local-dir ./modelzoo/Qwen3/Qwen3-0.6B
```

## Usage

Run weight-only quantization with the following script (`scripts/g2ptq.sh`):

```bash
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
    --exp g2ptq --deterministic --enable_torch_compile \
    --dataset neuralmagic --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method g2ptq --w_bits ${BITS} --w_clip --num_groups 4 --act_order --bsz 4 --layer_bsz 64 \
    --rotate \
    --gptq_backend triton --gptq_graph \
    --lm_eval --lm_eval_batch_size 32 \
    --autotune --autotune_nsamples 64 --autotune_min 0.0 --autotune_max 3.0 --autotune_stepsize 0.1 \
    --kl_topk 256 \
    --true_sequential_ratio 1.0 \
    --offload_inps --offload_hessians
```

Scripts for more quantization methods are in `scripts`. For GuidedQuant, run `scripts/save_grads.sh` prior to quantization

Other commonly used arguments for `ptq.py`:

- **W4A4**. `--a_bits 4`,  `--a_clip_ratio 0.9`.
- **G$^2$PTQ Variants**. Setting `--true_sequential_ratio 1.0`  selects the G$^2$PTQ$^\star$ variant, whereas `0.0` yields the plain G$^2$PTQ.
- **Quantized Model Export**. `--export_compressed_tensors`, `--export_mtp`, `--disable_online_rot`.
- **Evaluations on Reasoning Benchmarks**. See `eval_reasoning`.
- **Hessian-based Weight Clipping**. `--w_clip`, `--w_hclip`.


## Acknowledgements

This project is based on the work of the following projects:

- [QuaRot](https://github.com/spcl/QuaRot)
- [SpinQuant](https://github.com/facebookresearch/SpinQuant)
- [GPTQ](https://github.com/IST-DASLab/gptq)
- [GPTAQ](https://github.com/Intelligent-Computing-Lab-Yale/GPTAQ)
- [GuidedQuant](https://github.com/snu-mllab/GuidedQuant)
- [FlatQuant](https://github.com/ruikangliu/FlatQuant)
- [llmcompressor](https://github.com/vllm-project/llm-compressor)
- [compressed-tensors](https://github.com/vllm-project/compressed-tensors)


## References

If you find G$^2$PTQ helpful, please cite our paper:

```bibtex
@article{liu2026g2ptq,
  title={G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation},
  author={Liu, Ruikang and Bai, Haoli and Sun, Yuxuan and Zhang, Qian and Cai, Wenzheng and Hao, Yanqi and Wang, Feiyu and Zhong, Weidong and Wang, Zhuang and Yang, Tong and Zhou, Xiangsheng},
  journal={arXiv preprint arXiv:2609.31009},
  year={2026}
}
```
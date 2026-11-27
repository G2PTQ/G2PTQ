# Layer-type-aware quantization + token-weighted G²PTQ (AI Engineering mega assignment)

Everything we added on top of the upstream G²PTQ code. Upstream behaviour is unchanged unless
one of the new options is passed.

- **Idea 1, type-aware bits:** `--w_bits_map` gives DeltaNet, attention and FFN layers their own
  bit width (and optionally group size). `ours/search/bit_search.py` picks the assignment.
- **Idea 2, token-weighted G²PTQ:** `--token_weights` multiplies each calibration token's
  contribution to G²PTQ's Hessian (H′ = Σₜ wₜFₜ) and block loss (g′ = ∇ Σₜ wₜℓₜ).
  `ours/token_weights/compute_token_weights.py` computes the weights from attention received
  and DeltaNet gate retention.

## Run order on the GPU machine

All scripts run from the repo root and read `scripts/ours/config.sh`.

| Step | Command | What it does | Rough cost |
|---|---|---|---|
| 0 | `bash scripts/ours/00_prepare.sh` (eval env) | Download Qwen3.5-4B and wikitext; build the calibration sets; generate the self-generated set; print size options | ~30 min |
| 1 | `bash scripts/ours/01_check_run.sh` (quant env; last line needs eval env) | **Go/no-go:** small G²PTQ run on 4B (16 sequences) + 50-prompt IFEval. Check time and peak memory in the log | short |
| 2 | `bash scripts/ours/02_bit_search.sh` (quant env) | Idea 1: sensitivity per type, search under `BUDGET`, confirm top 2 with GPTQ → `outputs/bit_search/best_bits_map.json` | ~11 RTN + 2 GPTQ runs |
| 3a | `bash scripts/ours/03_token_weights.sh --sweep` (quant env) | Idea 2: try λ ∈ {0, 0.5, 1, 2} × {source, prediction} on held-out KL; set `LAMBDA`/`TW_MODE` in config.sh | 8 G²PTQ runs (cut if short on time) |
| 3b | `bash scripts/ours/03_token_weights.sh` (quant env) | Token weights for every calibration set | minutes |
| 4 | `bash scripts/ours/04_grid.sh [calib sets]` (quant env) | Main grid: 2×2 G²PTQ + GPTQ baselines, per calibration set; skips finished runs | 18 runs |
| 5 | `bash scripts/ours/05_eval.sh` (eval env) | IFEval for BF16 + every model, `outputs/results.csv`, pooled paired bootstraps | ~19 IFEval runs |

Before step 4, set `UNIFORM_BITS`/`UNIFORM_GS` in `config.sh` so the uniform baseline has (about) the
same size as the type-aware map: `python ours/size/bpw.py --model <model> --w_bits_map outputs/bit_search/best_bits_map.json --w_groupsize 128`
prints the type-aware size, and `python ours/size/bpw.py --model <model> --match <that bpw>` lists uniform settings near it.

Steps 4 and 5 take calibration-set names, so the two of us can split them across GPUs:
`GPU=0 bash scripts/ours/04_grid.sh wikitext2 calib_ifchat` and `GPU=1 bash scripts/ours/04_grid.sh calib_selfgen`.

## Colab

Open `ours/colab/G2PTQ_colab.ipynb` in Colab (A100 runtime). It installs both environments,
runs steps 0–5 in order, and backs up results to Google Drive. The full grid saves 18
dequantized checkpoints of about 9 GB each, so keep an eye on disk space.

## Environment

1. The upstream environment (`requirements.txt`, Python 3.12) for quantization, plus
   `flash-linear-attention` and `causal-conv1d` (see the main README). On an A100, skip FlashAttention-3 (Hopper only).
2. A **separate** environment for vLLM (vLLM pins its own torch): `pip install -r ours/requirements-eval.txt`
   plus a vLLM release that supports Qwen3.5. Point `PY_EVAL` at its Python (for example
   `export PY_EVAL=/content/venv-eval/bin/python`); the scripts use it for IFEval and the
   self-generated data, and the quantization environment for everything else.

## Files

| Path | Purpose |
|---|---|
| `gptq_utils/bits_map.py` | `--w_bits_map` parsing; per-module (bits, group size); 16-bit modules are skipped |
| `gptq_utils/token_weights.py` | Loads weights (checks the calibration-token hash), applies them per batch |
| `gptq_utils/g2ptq_utils.py`, `gptq_utils.py` | Patched to use per-module settings and token weights |
| `ptq.py` | `--save_dequant_model` (BF16 checkpoint of the quantized model, servable by vLLM) and `metrics.json` |
| `utils/datasets/calib_chat.py` | `calib_ifchat`, `calib_selfgen`, `calib_heldout_if`: chat-templated, packed at conversation boundaries |
| `ours/data/build_calib.py` | Tulu 3 Persona IF + Conifer, n-gram decontaminated against IFEval and IFBench |
| `ours/data/generate_selfgen.py` | BF16 model answers the calibration prompts (thinking off) |
| `ours/token_weights/compute_token_weights.py` | Attention received + DeltaNet gate retention → weights (mean 1) |
| `ours/size/bpw.py` | Bits per weight and GB from safetensors headers; `--enumerate`, `--match` |
| `ours/search/bit_search.py` | Sensitivity + budgeted search + GPTQ confirmation |
| `ours/eval/ifeval_vllm.py` | IFEval generation (vLLM, greedy, 1280 tokens, thinking off) and per-prompt scoring; IFBench generation |
| `ours/eval/bootstrap.py` | Paired bootstrap (95% CI, p-value, discordant counts), pooled across calibration sets |
| `ours/eval/collect_results.py` | One CSV with settings, size, KL/PPL, IFEval, time, memory |
| `tests/` | CPU tests (below) |

## Tests (CPU, no GPU needed)

```bash
python tests/test_bits_map.py
python tests/test_token_weights.py           # all-ones weights == plain G²PTQ; weights scale H and g correctly
python tests/test_token_weight_signals.py    # retention math vs a naive loop, sink handling, normalisation
python tests/test_calib_data.py <tokenizer>  # decontamination, splits, deterministic packing
python tests/test_eval_tools.py              # IFEval scoring wrapper, bootstrap
```

`tests/cpu_smoke_ptq.py` runs the real `ptq.py` on CPU by stubbing the CUDA-only parts. It was
used to check RTN, GPTQ and G²PTQ end to end on a tiny random Qwen3.5 (3 DeltaNet + 1 attention
layer): bit maps put every layer on the intended grid, DeltaNet gate projections stay untouched,
all-ones token weights reproduce plain G²PTQ bit for bit, and the dequantized checkpoint saves.
It is a development tool, not a substitute for the GPU check run.

## Known limits

- **AWQ baseline is not implemented yet.** Qwen3.5's zero-centred RMSNorm (`x·(1+w)`) and DeltaNet
  layers need custom smoothing mappings, so an off-the-shelf AWQ run could silently be wrong.
  GPTQ is the baseline for now.
- **IFBench:** `ifeval_vllm.py --benchmark ifbench` writes `ifbench_input.jsonl`; score it with the
  official IFBench repository.
- **Mixed-bit export:** `--export_compressed_tensors` only supports uniform 4/8-bit. Use
  `--save_dequant_model` for evaluation; sizes come from `ours/size/bpw.py`.
- **Token weights** are G²PTQ-only and dense-model-only (they're rejected for MoE layers).

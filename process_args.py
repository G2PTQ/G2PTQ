import argparse
import os
import logging

import transformers
import torch

from utils import dist_utils
from utils.datasets import get_dataset_names
from utils.log_utils import init_logging


def parse_gen():
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("--model", type=str, required=True, help="The model to quantize")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    parser.add_argument("--deterministic", action="store_true", help="Ensure reproducibility")
    # Paths
    parser.add_argument("--output_dir", type=str, default="./outputs", help="The directory to save results in")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="The directory to cache results in")
    parser.add_argument("--offload_folder", type=str, default="./offload_weights", help="The directory for model weights disk offload")
    parser.add_argument("--offload_extra_cpu_mem", type=float, default=100e9, help="CPU memory to reserve outside model offload, in bytes")
    # Datasets
    parser.add_argument("--dataset", type=str, default="neuralmagic",
                        choices=get_dataset_names(), help="The dataset to use")
    parser.add_argument("--nsamples", type=int, default=1024, help="The number of examples to use")
    parser.add_argument("--seq_len", type=int, default=2048, help="The sequence length to use in calibration")
    parser.add_argument("--eval_seq_len", type=int, default=2048, help="The sequence length to use in PPL&KL evaluation")
    # Gradient profiling
    parser.add_argument("--mode", type=str, default="gradients", choices=["tokens", "gradients"],
                        help="The mode to run in")
    # Quantization configs
    parser.add_argument("--w_bits", type=int, default=16, help="Weight bits")
    parser.add_argument("--a_bits", type=int, default=16, help="Activation bits")
    parser.add_argument("--k_bits", type=int, default=16, help="K cache bits")
    parser.add_argument("--v_bits", type=int, default=16, help="V cache bits")
    parser.add_argument("--w_groupsize", type=int, default=-1, help="Weight group size")
    parser.add_argument("--a_groupsize", type=int, default=-1, help="Activation group size")
    parser.add_argument("--k_groupsize", type=int, default=-1, help="K cache group size")
    parser.add_argument("--v_groupsize", type=int, default=-1, help="V cache group size")
    parser.add_argument("--w_asym", action="store_true", help="Weight asymmetric quantization")
    parser.add_argument("--a_asym", action="store_true", help="Activation asymmetric quantization")
    parser.add_argument("--k_asym", action="store_true", help="K cache asymmetric quantization")
    parser.add_argument("--v_asym", action="store_true", help="V cache asymmetric quantization")
    parser.add_argument("--w_clip", action="store_true", help="Enable weight clipping")
    parser.add_argument("--w_hclip", action="store_true", help="Use diagonal-Hessian-weighted --w_clip")
    parser.add_argument("--a_clip_ratio", type=float, default=1.0, help="Activation clipping ratio")
    parser.add_argument("--k_clip_ratio", type=float, default=1.0, help="K cache clipping ratio")
    parser.add_argument("--v_clip_ratio", type=float, default=1.0, help="V cache clipping ratio")
    parser.add_argument("--ignore_attn", action="store_true", help="Keep attention modules in high-precision")
    parser.add_argument("--export_compressed_tensors", action="store_true", help="Export quantized model to compressed-tensors format")
    parser.add_argument("--export_mtp", action="store_true", help="Export MTP weights to compressed-tensors format")
    # Rotate
    parser.add_argument("--optimized_rotation_path", type=str, default=None, help="The path to rotation ckpt")
    parser.add_argument("--rotate", action="store_true", help="Rotate model (SpinQuant)")
    parser.add_argument("--disable_online_rot", action="store_true", help="Disable online rotations.")
    # GPTQ
    parser.add_argument("--w_method", type=str, default="gptq",
                        choices=["rtn", "gptq", "gptaq", "gptq_guided", "g2ptq"], help="Weight quantization method to use")
    parser.add_argument(
        "--gptq_backend",
        type=str,
        default="torch",
        choices=["torch", "triton"],
        help="Sequential compensation backend for GPTQ-family methods",
    )
    parser.add_argument(
        "--gptq_graph",
        action="store_true",
        help="Capture the selected GPTQ-family compensation backend in a CUDA Graph",
    )
    parser.add_argument("--act_order", action="store_true", help="Activation reorder (with static groups)")
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of groups $g$ to use for block-diagonal Hessian")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--alpha_kl", type=float, default=0.05, help="Dynamic down-scaling factor for the KL loss gradient update term")
    parser.add_argument("--alpha_mse", type=float, default=0.01, help="Dynamic down-scaling factor for the MSE loss gradient update term")
    parser.add_argument("--alpha_warmup_ratio", type=float, default=0, help="Dynamic down-scaling factor warmup ratio")
    parser.add_argument("--kl_ratio", type=float, default=0.0, help="Only the last few layers use the KL loss")
    parser.add_argument("--enable_linear_patch", action="store_true", help="Use a linear patch to approximate unquantized Transformer blocks")
    parser.add_argument("--residual_forcing_ratio", type=float, default=0.0, help="Use GT residual for the earlier layers")
    parser.add_argument("--linear_patch_strength", type=float, default=1.0, help="Weight of the additional linear-patch loss relative to the local block loss")
    parser.add_argument("--linear_patch_ratio", type=float, default=0.5, help="Only the last few layers use the linear-patch loss")
    parser.add_argument("--kl_topk", type=int, default=-1, help="Top-k KL loss")
    parser.add_argument("--true_sequential_ratio", type=float, default=0, help="Sequentially quantize modules in the last few Transformer blocks")
    parser.add_argument("--moe_gate_align", action="store_true", help="Align MoE gating logits")
    parser.add_argument("--moe_gate_align_strength", type=float, default=1, help="MoE gating logits align strength")
    parser.add_argument("--moe_gate_forcing", action="store_true", help="Use teacher forcing in MoE gating")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for model forward and backward")
    parser.add_argument("--load_qmodel_path", type=str, default=None, help="The path to load quantized model ckpt")
    parser.add_argument("--save_qmodel_path", type=str, default=None, help="The path to save quantized model ckpt")
    parser.add_argument("--offload_inps", action="store_true", help="Offload inputs to CPU")
    parser.add_argument("--offload_hessians", action="store_true", help="Offload Hessians and gradients to CPU when not in use")
    parser.add_argument("--layer_bsz", type=int, default=1, help="Batch size for batched layer quantization")
    parser.add_argument("--onload_gptq_bsz", type=int, default=-1,
                        help="Max number of G2PTQ instances whose Hessians are onloaded and "
                             "accumulated at once during the calibration forward. -1 = all at "
                             "once (default, current behavior).")
    # Autotune
    parser.add_argument("--autotune", action="store_true", help="Autotune the dynamic down-scaling factor")
    parser.add_argument("--load_autotune", type=str, default=None, help="Load dynamic down-scaling factor from autotune config")
    parser.add_argument("--autotune_nsamples", type=int, default=64, help="The number of examples to use")
    parser.add_argument("--autotune_min", type=float, default=0.0, help="Autotune lower bound")
    parser.add_argument("--autotune_max", type=float, default=3.0, help="Autotune upper bound")
    parser.add_argument("--autotune_stepsize", type=float, default=0.1, help="Autotune stepsize")
    parser.add_argument("--autotune_early_stop_margin", type=float, default=None, help="Autotune early stop criterion")
    parser.add_argument("--autotune_outlier_thresh", type=float, default=1, help="Ignore hidden state outliers in autotuning")
    # Ablations
    parser.add_argument("--disable_kl", action="store_true",
                        help="Ablation: always use the MSE block objective, never KL")
    parser.add_argument("--disable_grad", action="store_true",
                        help="Ablation: drop the gradient-guided compensation term")
    parser.add_argument("--disable_refresh", action="store_true",
                        help="Ablation: feed full-precision inputs to every block instead of the "
                             "previous block's quantized outputs (implies --disable_grad)")
    parser.add_argument("--approx_grad", action="store_true",
                        help="Ablation: replace the true gradient compensation with a shrink "
                             "toward the original weight (implies --disable_grad)")
    parser.add_argument("--approx_grad_beta", type=float, default=0.2,
                        help="Shrink factor for --approx_grad")
    # Eval
    parser.add_argument("--lm_eval", action="store_true", help="Enable QA eval")
    parser.add_argument("--lm_eval_batch_size", type=int, default=32, help="Batch size for QA tasks")
    parser.add_argument("--eval_datasets", type=str, nargs="+", choices=get_dataset_names(),
                        default=["wikitext2", "ultrachat_2k", "numinamath"],
                        help="Datasets for PPL & KL eval")
    # Exp
    parser.add_argument("--attn_implementation", type=str, default=None,
                        choices=["sdpa", "flash_attention_3"],
                        help="Attention backend. Default (unset) lets transformers choose (sdpa).")
    parser.add_argument("--enable_torch_compile", action="store_true",
                        help="torch.compile the quantization kernels (can perturb PPL/KL slightly)")
    parser.add_argument("--enable_debug", action="store_true", help="Enable debugging")

    args = parser.parse_args()

    if args.autotune and args.load_autotune:
        parser.error(
            "--autotune and --load_autotune are mutually exclusive: the search would overwrite "
            "the loaded alphas."
        )

    if (
        args.gptq_backend != "torch" or args.gptq_graph
    ) and args.w_method not in {
        "gptq", "g2ptq", "gptaq", "gptq_guided"
    }:
        parser.error(
            "--gptq_backend triton and --gptq_graph require a GPTQ-family "
            "weight method"
        )

    # Ablations
    if args.disable_refresh or args.approx_grad:
        args.disable_grad = True
    if args.disable_refresh:
        args.true_sequential_ratio = 0
    if args.disable_grad:
        args.autotune = False
        args.load_autotune = None
    if args.approx_grad and args.gptq_backend != "torch":
        parser.error("--approx_grad is only implemented for --gptq_backend torch")

    # set paths & others
    args.model_name = args.model.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.exp)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tokens_cache_path = (f"{args.cache_dir}/tokens/"
                              f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt")
    args.autotune_cache_path = os.path.join(args.output_dir, "autotune.json")
    args.export_path = os.path.join(args.output_dir, "export_model")
    if args.export_compressed_tensors:
        assert not args.w_asym and (args.w_bits in [4, 8]), "Currently, only the export of 4/8-bit symmetrically quantized models is supported."
        assert (not args.rotate) or args.disable_online_rot, "Online rotations are not supportded in inference engines."
    if args.num_groups is not None:
        model_name = args.model_name + ("_rot" if args.rotate else "")
        args.saliency_cache_path = (f"{args.cache_dir}/saliency/"
                                    f"{model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}")
        args.gradients_cache_path = (f"{args.cache_dir}/gradients/"
                                    f"{model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}.pt")
    else:
        args.saliency_cache_path = None
        args.gradients_cache_path = None
    args.offload_folder = os.path.join(args.offload_folder, args.model_name)
    if args.autotune_early_stop_margin is None:
        args.autotune_early_stop_margin = args.autotune_stepsize

    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
    transformers.set_seed(args.seed, deterministic=args.deterministic)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    init_logging(args.log_dir)
    logging.info(args)

    if args.enable_torch_compile and args.deterministic:
        logging.warning(
            "--enable_torch_compile with --deterministic: inductor may reorder reductions in the "
            "weight-clip search, which can flip near-ties and pick a different clip scale. "
            "Results stay close but are not bit-identical to the eager run."
        )

    # Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.enable_debug:
        import debugpy
        debugpy.listen(5678 + dist_utils.get_rank())
        logging.info("Waiting for debugger attach")
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    return args

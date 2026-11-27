import json
import logging
import os

import torch
from compressed_tensors.offload.module import remove_module_offload

from process_args import parse_gen
from gptq_utils.main import quantize_weights
from utils import data_utils, dist_utils, eval_utils, model_utils, rotation_utils, offload_utils, \
                  memory_utils, quant_utils, hadamard_utils, compressed_tensors_utils, compile_utils


def main(args):
    compile_utils.configure(args.enable_torch_compile)

    analyzer = model_utils.ModelAnalyzer(
        args.model,
        args.seq_len,
        args.ignore_attn,
        offload_folder=args.offload_folder,
        offload_extra_cpu_mem=args.offload_extra_cpu_mem,
        attn_implementation=args.attn_implementation,
    )
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    # Generate reference logits for KL eval
    test_loader_dict, ref_logits_dict = {}, {}
    for eval_dataset in args.eval_datasets:
        test_loader = data_utils.get_loaders(eval_dataset, split="test", tokenizer=tokenizer,
                                             seq_len=args.eval_seq_len, num_samples=args.nsamples)
        ref_logits, orig_lm_head = eval_utils.get_ref_logits(args, analyzer, eval_dataset, test_loader)
        test_loader_dict[eval_dataset] = test_loader
        ref_logits_dict[eval_dataset] = ref_logits

    # Rotate the weights
    if args.rotate:
        rotation_utils.rotate_model(args, analyzer)
        memory_utils.cleanup_memory()

        quant_utils.add_actquant(analyzer)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if (not args.disable_online_rot) and "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(qlayers[name].weight.shape[1])
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = False
    else:
        quant_utils.add_actquant(analyzer)

    if (args.k_bits < 16 or args.v_bits < 16) and not analyzer.spec.SUPPORTS_KV_QUANT:
        raise NotImplementedError(
            f"KV-cache quantization (--k_bits / --v_bits < 16) is not supported for "
            f"{analyzer.model_arch}."
        )

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        rope_function_name = "apply_rotary_pos_emb"
        layers = analyzer.get_layers()
        k_quant_config = {
            "k_bits": args.k_bits,
            "k_groupsize": args.k_groupsize,
            "k_sym": not (args.k_asym),
            "k_clip_ratio": args.k_clip_ratio,
        }
        n_wrapped = 0
        for layer in layers:
            # None on layers that hold no KV cache (linear attention in a hybrid stack).
            attn = analyzer.spec.get_kv_attn_module(layer)
            if attn is None:
                continue
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                attn,
                rope_function_name,
                head_dim=analyzer.head_dim,
                **k_quant_config,
            )
            n_wrapped += 1
        logging.info(
            f"K-cache quantization: wrapped {n_wrapped}/{len(layers)} layers."
        )

    # Quantize model weights
    with analyzer.enter_quantization_context(args):
        model = quantize_weights(args, analyzer)

    # Eval
    metric_vals = eval_utils.kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict)
    if dist_utils.is_main():
        write_metrics(args, metric_vals)
    del orig_lm_head, ref_logits_dict
    memory_utils.cleanup_memory()
    if dist_utils.is_main():
        # Fully materialize the model before QA eval
        offload_utils.set_onload_device(model, torch.device("cpu"))
        for module in model.modules():
            remove_module_offload(module, onload_tensors=True)

        if args.save_dequant_model:
            save_dequant_model(args, analyzer)

        if args.lm_eval:
            dist_utils.distribute_model(analyzer)
            eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)

    # Save model to compressed-tensors format
    if dist_utils.is_main() and args.export_compressed_tensors:
        quant_utils.remove_actquant(analyzer)
        analyzer.stack_experts_weights(weight_packed=True)    # save 3D weights
        analyzer.recover_tie_word_embeddings()
        compressed_tensors_utils.save_to_compressed_tensors(
            source_model=args.model,
            config=analyzer.config,
            model=analyzer.model,
            tokenizer=analyzer.tokenizer,
            processor=analyzer.processor,
            save_path=args.export_path,
            bits=args.w_bits,
            group_size=args.w_groupsize,
            sym=not args.w_asym,
            export_mtp=args.export_mtp,
        )


def write_metrics(args, metric_vals):
    """Write KL/PPL/EAR plus run settings to <output_dir>/metrics.json (read by ours/eval/collect_results.py)."""
    from gptq_utils.bits_map import describe
    record = {
        "exp": args.exp,
        "model": args.model,
        "w_method": args.w_method,
        "w_bits": args.w_bits,
        "w_groupsize": args.w_groupsize,
        "w_bits_map": describe(args),
        "w_bits_map_json": ({r.pattern: ({"bits": b, "groupsize": g} if g is not None else b)
                             for r, b, g in args.w_bits_map} if args.w_bits_map else None),
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seq_len": args.seq_len,
        "token_weights": args.token_weights,
        "rotate": args.rotate,
        "metrics": {k: float(v) for k, v in metric_vals.items()},
        **getattr(args, "quant_stats", {}),
    }
    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(record, f, indent=2)
    logging.info(f"Wrote metrics to {args.metrics_path}")


def save_dequant_model(args, analyzer):
    """Save the fake-quantized weights as an ordinary BF16 checkpoint (vLLM-servable).

    The quantizers write dequantized values back into each Linear's weight, so a plain
    save_pretrained() gives a model whose outputs match the quantized model exactly; the
    storage size is accounted for separately (ours/size/bpw.py).
    """
    from gptq_utils.bits_map import describe
    logging.info(f"Saving dequantized model to {args.dequant_path}")
    quant_utils.remove_actquant(analyzer)
    analyzer.recover_tie_word_embeddings()
    os.makedirs(args.dequant_path, exist_ok=True)
    analyzer.model.save_pretrained(args.dequant_path, safe_serialization=True)
    analyzer.tokenizer.save_pretrained(args.dequant_path)
    if getattr(analyzer, "processor", None) is not None:
        analyzer.processor.save_pretrained(args.dequant_path)
    with open(os.path.join(args.dequant_path, "g2ptq_quant_info.json"), "w") as f:
        json.dump({"source_model": args.model, "exp": args.exp, "w_method": args.w_method,
                   "settings": describe(args)}, f, indent=2)


if __name__ == "__main__":
    dist_utils.init_process_group()
    args = parse_gen()
    main(args)

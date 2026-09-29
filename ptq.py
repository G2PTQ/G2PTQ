import logging

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
    eval_utils.kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict)
    del orig_lm_head, ref_logits_dict
    memory_utils.cleanup_memory()
    if dist_utils.is_main():
        # Fully materialize the model before QA eval
        offload_utils.set_onload_device(model, torch.device("cpu"))
        for module in model.modules():
            remove_module_offload(module, onload_tensors=True)

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


if __name__ == "__main__":
    dist_utils.init_process_group()
    args = parse_gen()
    main(args)

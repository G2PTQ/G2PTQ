import os
import logging

import torch
from compressed_tensors.offload.module import remove_module_offload

from process_args import parse_gen
from utils.model_utils import ModelAnalyzer
from utils.data_utils import get_tokens
from utils.gradients import get_gradients
from utils import dist_utils, offload_utils, rotation_utils, memory_utils, quant_utils, hadamard_utils


def save_gradients(
    model,
    mode='gradient',
    dataset="wikitext2", seq_len=2048, nsamples=1024,
    seed=42,
    num_groups=4,
    **kwargs,
):
    if mode == 'tokens':
        logging.info("Running: [Tokens]")
    elif mode == 'gradients':
        logging.info("Running: [Tokens -> Gradients]")

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    logging.info(f"Running Quantization on {model_name} using {dataset} for gradient calculation")

    # ------------------- Load model -------------------

    analyzer = ModelAnalyzer(model, args.seq_len, attn_implementation=args.attn_implementation)
    model = analyzer.model

    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(analyzer)
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

    # ------------------- Set cache paths -------------------

    tokens_cache_path = args.tokens_cache_path
    gradients_cache_path = args.gradients_cache_path
    saliency_cache_path = args.saliency_cache_path

    logging.info(f"Tokens cache path: {tokens_cache_path}")
    logging.info(f"Gradients cache path: {gradients_cache_path}")
    logging.info(f"Saliency cache path: {saliency_cache_path}")

    # ------------------- Get tokens -------------------

    logging.info("------------------- Get tokens -------------------")
    logging.info(f"Getting tokens for {dataset} with sequence length {seq_len} and {nsamples} examples")
    tokens = get_tokens(dataset, "train", analyzer.tokenizer, seq_len, nsamples, tokens_cache_path, seed)
    logging.info("Tokens loading complete.")

    if mode == 'tokens':
        return

    # ------------------- Gradients -------------------

    logging.info("------------------- Gradients -------------------")

    logging.info("Beginning gradient calculation...")
    # # Calculate or load gradients
    # if os.path.exists(gradients_cache_path):
    #     # if the user wants to recalculate the gradients, delete the cached gradients
    #     logging.info(f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate.")
    #     os.remove(gradients_cache_path)

    if dist_utils.is_main():
        # Fully materialize the model
        offload_utils.set_onload_device(model, torch.device("cpu"))
        for module in model.modules():
            remove_module_offload(module, onload_tensors=True)

        model_gradients = get_gradients(
            analyzer=analyzer,
            input_tokens=tokens,
            gradients_path=gradients_cache_path,
            saliency_path=saliency_cache_path,
            num_groups=num_groups,
        )
        
        logging.info("Gradient calculation complete.")


if __name__ == "__main__":
    dist_utils.init_process_group()
    args = parse_gen()
    # only pass options that are not None
    save_gradients(**{k: v for k, v in args.__dict__.items() if v is not None})

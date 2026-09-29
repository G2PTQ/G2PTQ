import os
import time
import math
import logging
import resource

import torch
import transformers

from gptq_utils import gptq_utils, gptaq_utils, gptq_guided_utils, g2ptq_utils
from gptq_utils.graph_utils import clear_graph_cache
from utils import data_utils, model_utils, dist_utils


def quantize_weights(args, analyzer: model_utils.ModelAnalyzer):
    transformers.set_seed(args.seed)

    model = analyzer.model

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            logging.info("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        else:
            trainloader = data_utils.get_tokens(args.dataset, "train", analyzer.tokenizer, args.seq_len,
                                                args.nsamples, args.tokens_cache_path, args.seed)
            
            # Split dataset
            if dist_utils.is_dist_available_and_initialized():
                rank = dist_utils.get_rank()
                world_size = dist_utils.get_world_size()
                start = math.floor(args.nsamples * (rank / world_size))
                end = math.floor(args.nsamples * ((rank + 1) / world_size))
                trainloader = trainloader[start: end]

            if isinstance(trainloader[0], torch.Tensor):
                assert trainloader[0].dim() == 1
                logging.info("Reformatting input tokens to tuple + Unsqueeze")
                trainloader = [(x.unsqueeze(0), None) for x in trainloader]

            dev = analyzer.onload_device
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
            start_time = time.time()
            try:
                if args.w_method == "rtn":
                    quantizers = gptq_utils.rtn_fwrd(args, analyzer, dev)
                elif args.w_method == "gptq":
                    quantizers = gptq_utils.gptq_fwrd(args, analyzer, trainloader, dev)
                elif args.w_method == "gptaq":
                    quantizers = gptaq_utils.gptq_fwrd(args, analyzer, trainloader, dev)
                elif args.w_method == "gptq_guided":
                    quantizers = gptq_guided_utils.gptq_fwrd(args, analyzer, trainloader, dev)
                elif args.w_method == "g2ptq":
                    quantizers = g2ptq_utils.gptq_fwrd(args, analyzer, trainloader, dev)
            finally:
                if args.gptq_graph:
                    clear_graph_cache()
            torch.cuda.synchronize(dev)
            end_time = time.time()
            peak_gpu_mem = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            peak_cpu_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            logging.info(f"Model Quantization Time: {((end_time - start_time) / 3600):.2f} hours")
            logging.info(f"Peak GPU Memory Usage during Model Quantization: {peak_gpu_mem:.2f} GB")
            logging.info(f"Peak CPU Memory Usage during Model Quantization: {peak_cpu_mem:.2f} GB")
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path and dist_utils.is_main():
            os.makedirs(os.path.dirname(args.save_qmodel_path), exist_ok=True)
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    return model

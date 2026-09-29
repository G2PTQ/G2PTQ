import logging
import os
import copy
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from compressed_tensors.offload import disable_offloading

import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.models.huggingface import eval_logger
eval_logger.level = logging.ERROR

from utils import memory_utils, dist_utils, model_utils, log_utils, offload_utils


def _get_effective_bsz(nsamples, bsz):
    bsz = min(bsz, nsamples)
    for b in range(bsz, 0, -1):
        if nsamples % b == 0:
            return b
    return 1


@torch.no_grad()
def _get_logits(args, analyzer: model_utils.ModelAnalyzer, testenc, dev):
    model = analyzer.model
    use_cache = analyzer.config.use_cache
    analyzer.config.use_cache = False
    layers = analyzer.get_layers()

    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // args.eval_seq_len
    input_ids = input_ids[:, :nsamples * args.eval_seq_len].view(nsamples, args.eval_seq_len).to(dev)

    # Use the largest factor of nsamples <= args.bsz so all batches are equal-sized;
    # this keeps the cached batch-dependent kwargs shape-consistent across batches.
    bsz = _get_effective_bsz(nsamples, args.bsz)

    dtype = next(iter(model.parameters())).dtype
    inps = offload_utils.alloc_pinned(
        (nsamples, args.eval_seq_len, analyzer.residual_width), dtype=dtype,
        device="cpu" if args.offload_inps else dev, pin=False,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module, bsz):
            super().__init__()
            self.module = module
            self.bsz = bsz
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, *args, **kwargs):
            inps[cache["i"]: cache["i"] + self.bsz] = inp.to(inps.device)
            cache["i"] += self.bsz
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs.get("position_ids")
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0], bsz)
    with (
        disable_offloading(),
        analyzer.capture_block_internals() as block_internals
    ):
        for i in range(0, nsamples, bsz):
            try:
                model(input_ids[i: i + bsz])
            except ValueError:
                pass
    layers[0] = layers[0].module

    memory_utils.cleanup_memory()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]
    kwargs = {}

    topk_buffer = analyzer.alloc_index_buffer(nsamples, args.eval_seq_len, inps.device)

    for i in tqdm(range(len(layers)), ncols=80, desc="Forwarding Layers"):
        layer = layers[i]
        with disable_offloading():
            for j, (inps_batch, topk_batch) in offload_utils.prefetch_generator(
                (inps, topk_buffer), nsamples, bsz, dev, args.offload_inps
            ):
                model_utils.run_block_layer(
                    analyzer, layer, inps_batch, out_buffer=inps,
                    prev_topk_indices=topk_batch, index_buffer=topk_buffer,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    block_internals=block_internals,
                    layer_idx=i, sample_idx=j, bsz=bsz, dev=dev,
                    **kwargs,
                )

        del layer
        memory_utils.cleanup_memory()

    analyzer.config.use_cache = use_cache
    memory_utils.cleanup_memory()

    # Get model logits
    norm = analyzer.get_layernorm_before_head()
    lm_logits = []
    with disable_offloading():
        for j, (inps_batch,) in offload_utils.prefetch_generator((inps,), nsamples, bsz, dev, args.offload_inps):
            hidden_states = norm(inps_batch)
            lm_logits.append(hidden_states.cpu())
    lm_logits = torch.cat(lm_logits, dim=0)

    memory_utils.cleanup_memory(verbos=True)

    return lm_logits, input_ids


@torch.no_grad()
def get_ref_logits(args, analyzer: model_utils.ModelAnalyzer, dataset, dataloader):
    cache_dir = os.path.join(args.cache_dir, "ref_logits")
    os.makedirs(cache_dir, exist_ok=True)
    ref_logits_path = f'{cache_dir}/{args.model_name}_{dataset}_test_{args.eval_seq_len}.cache'
    if not os.path.exists(ref_logits_path):
        logging.info(f"Generating reference logits for {dataset}...")
        ref_logits, _ = _get_logits(args, analyzer, dataloader, analyzer.onload_device)
        if dist_utils.is_main():
            torch.save(ref_logits, ref_logits_path)
    else:
        logging.info(f"Loading reference logits for {dataset}...")
        ref_logits = torch.load(ref_logits_path).cpu()
    # Save original head (before rotation)
    head_weight = analyzer.get_lm_head().weight.data
    orig_lm_head = nn.Linear(analyzer.config.hidden_size, analyzer.config.vocab_size, bias=False,
                             device="cpu", dtype=head_weight.dtype)
    orig_lm_head.weight.data.copy_(head_weight)
    memory_utils.cleanup_memory()
    return ref_logits, orig_lm_head


@torch.no_grad()
def _logits_eval(args, analyzer: model_utils.ModelAnalyzer, orig_lm_head, dataloader, ref_logits_list):
    """Compute PPL, KL divergence, and EAR between quantized and reference model.

    Returns:
        Tuple of (ppl, kl_loss, ear) where:
            - ppl: Perplexity on the test set
            - kl_loss: KL divergence vs reference logits
            - ear: Expected Acceptance Rate (token agreement probability)
    """
    model = analyzer.model
    dev = analyzer.onload_device

    logits_list, input_ids_list = _get_logits(args, analyzer, dataloader, dev)
    orig_lm_head.to(dev)

    kl_loss = 0
    ear_sum = 0
    nlls = []
    with disable_offloading():
        for logits, ref_logits, input_ids in tqdm(zip(logits_list, ref_logits_list, input_ids_list), ncols=80,
                                                  total=len(ref_logits_list), desc="Computing PPL & KL & EAR"):
            logits = analyzer.post_process_logits(model.lm_head(logits.to(dev)))
            ref_logits = analyzer.post_process_logits(orig_lm_head(ref_logits.to(dev)))

            # NLL loss
            shift_labels = input_ids[None, 1:].to(dev)
            shift_logits = logits[None, :-1, :].to(dev)
            loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels,
                                   reduction="none")
            neg_log_likelihood = loss.float().mean(dim=1)
            nlls.append(neg_log_likelihood)

            # kl loss
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="none",
            )
            kl_loss += loss.float().sum(-1).mean()

            # EAR (Expected Acceptance Rate)
            ear_sum += compute_ear(logits, ref_logits)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    kl_loss /= len(ref_logits_list)
    ear = ear_sum / len(ref_logits_list)

    memory_utils.cleanup_memory()

    return ppl.item(), kl_loss.item(), ear


def compute_ear(logits, ref_logits):
    """Compute Expected Acceptance Rate (EAR) between two logit distributions.

    EAR measures the maximum token-agreement probability under optimal coupling,
    computed as the sum of element-wise minimum probabilities across the vocabulary.

    Reference: "Statistically-Lossless Quantization of Large Language Models"
               (arXiv:2605.02404v2, Section 3.2, Equation 3)

    Args:
        logits: Quantized model logits [seq_len, vocab_size]
        ref_logits: Reference (original) model logits [seq_len, vocab_size]

    Returns:
        EAR value in [0, 1]. EAR ≥ 0.99 means ≥99% token agreement.
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)          # [seq_len, vocab_size]
    ref_probs = F.softmax(ref_logits, dim=-1)  # [seq_len, vocab_size]

    # Optimal coupling: min(p,q) gives max probability of X∼p and Y∼q agreeing
    min_probs = torch.min(probs, ref_probs)    # [seq_len, vocab_size]
    ear_per_position = min_probs.sum(dim=-1)   # [seq_len]

    # Average over all positions
    ear = ear_per_position.mean()

    return ear.item()


def assert_metric_vals_synced(metric_vals, rtol=1e-3, atol=1e-6):
    """Assert that ``kl_ppl_eval``'s ``metric_vals`` match across all ranks.

    All ranks share the same (offloaded) weights and reference logits, so KL/PPL/EAR must be
    identical everywhere. A mismatch means the shared-offload state desynced across ranks
    (e.g. a param write that didn't reach the shared master). This is a no-op when running
    on a single rank / without a process group.

    :param metric_vals: the ``OrderedDict`` of formatted metric strings returned by the
        per-rank eval; values are parsed back to floats for a numerical comparison.
    """
    if not dist_utils.is_dist_available_and_initialized() or dist_utils.get_world_size() == 1:
        return

    keys = list(metric_vals.keys())
    local = torch.tensor([float(metric_vals[k]) for k in keys], dtype=torch.float64, device="cuda")

    # Gather rank 0's values to every rank and compare.
    ref = local.clone()
    dist.broadcast(ref, src=0)

    mismatches = [
        (k, local[i].item(), ref[i].item())
        for i, k in enumerate(keys)
        if not torch.isclose(local[i], ref[i], rtol=rtol, atol=atol)
    ]
    if mismatches:
        report = "\n  ".join(f"{k}: rank{dist_utils.get_rank()}={v:.6g} vs rank0={r:.6g}"
                             for k, v, r in mismatches)
        raise AssertionError(
            f"metric_vals differ across ranks (rank {dist_utils.get_rank()}):\n  {report}"
        )


def kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict):
    metric_vals = OrderedDict()
    for eval_dataset in args.eval_datasets:
        logging.info(f"Evaluating KL&PPL&EAR on {eval_dataset}")
        ppl, kl_loss, ear = _logits_eval(args, analyzer, orig_lm_head, test_loader_dict[eval_dataset], ref_logits_dict[eval_dataset])
        metric_vals[f"KL-{eval_dataset}"] = f"{kl_loss:.2e}"
        metric_vals[f"PPL-{eval_dataset}"] = f"{ppl:.2f}"
        metric_vals[f"EAR-{eval_dataset}"] = f"{ear:.4f}"
        logging.info(f"KL&PPL&EAR on {eval_dataset}: {kl_loss:.2e}, {ppl:.2f}, {ear:.4f}")
    assert_metric_vals_synced(metric_vals)
    pretty_print_results(metric_vals)


def qa_eval(model, tokenizer, lm_eval_batch_size=32):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=lm_eval_batch_size)

    tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai", "ceval-valid"]
    task_manager = lm_eval.tasks.TaskManager(include_path="./datasets/lm_eval_configs/tasks", include_defaults=False)
    task_names = lm_eval_utils.pattern_match(tasks, task_manager.all_tasks)
    results, results_str = {}, {}
    for task_name in task_names:
        logging.info(f"Evaluating {task_name}...")
        hflm.batch_size_per_gpu = lm_eval_batch_size
        with log_utils.disable_logging_context():
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], task_manager=task_manager)['results']
        result = result[task_name]
        acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
        results[task_name] = acc
        logging.info(f"acc: {acc}%")
    results_str.update({task: f"{result:.2f}" for task, result in results.items()})
    results_str['acc_avg'] = f"{sum(results.values()) / len(task_names):.2f}"
    pretty_print_results(results_str)


def pretty_print_results(data):
    headers = list(data.keys())
    values = [str(v) for v in data.values()]

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_row = "| " + " | ".join(values) + " |"

    logging.info("\n" + header_row + "\n" + separator_row + "\n" + data_row)

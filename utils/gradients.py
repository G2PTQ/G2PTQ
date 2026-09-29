import os
import logging
from typing import Optional, Tuple
from tqdm import tqdm

import torch

from utils.dist_utils import distribute_model
from utils import model_utils


def get_gradients(
    analyzer: model_utils.ModelAnalyzer,
    input_tokens,
    gradients_path: Optional[str] = None,
    saliency_path: Optional[str] = None,
    num_groups: Optional[int] = None,
):
    """
    Calculates weight gradients for the given input tokens. Optionally also calculates
    'saliency' (mean absolute gradient w.r.t. each module's output activations, grouped
    by channel) if 'saliency_path' is provided. In that case, we save one file per layer
    under 'saliency_path' directory (e.g., l0.pt, l1.pt, ...).

    Args:
        analyzer:        Analyzer object with `.model`, `.get_layers()`, `.get_modules(layer)`.
        input_tokens:    Collection of token tensors, each shape [seq_len].
        saliency_path:   Directory in which to save the saliency files (one file per layer).
                         If None, no saliency is computed/saved.
        num_groups:      Number of groups to chunk the channel dimension for saliency.
                         E.g. if hidden_dim=4096 and num_groups=4, each group has 1024 channels.

    Returns:
        gradients (list of dict): The list of per-layer, per-module weight gradients.
    """

    logging.info(f"Calculating gradients on {len(input_tokens)} tokens...")

    # ----------------------------------------------------------------
    # 2) Prepare model
    # ----------------------------------------------------------------
    model = analyzer.model
    if torch.cuda.device_count() > 1:
        distribute_model(analyzer)
    else:
        model.cuda()

    layers = analyzer.get_layers()

    # We'll use these to decide whether to hook/save a given layer
    start_layer, end_layer = (None, None)

    # ----------------------------------------------------------------
    # 3) If we want saliency, set up forward hooks
    # ----------------------------------------------------------------
    # We'll store a list-of-dicts parallel to `layers`:
    #   saliency_data[i_layer][module_name] = list of [bsz, seq_len, num_groups]
    saliency_data = None
    saliency_hooks = []

    if saliency_path is not None:
        # We'll store chunk-lists for all layers
        saliency_data = [
            {module_name: [] for module_name in analyzer.get_quantizable_modules(layer).keys()}
            for layer in layers
        ]

        def make_forward_hook(layer_idx, module_name):
            def forward_hook(module, inp, out):
                # We'll store gradient on 'out', so we must retain it
                out.retain_grad()

                def grad_hook(grad):
                    """
                    grad shape typically [bsz, seq_len, hidden_dim].
                    We group the channels, take abs, then average.
                    """
                    bsz, seq_len, hidden_dim = grad.shape
                    group_size = hidden_dim // num_groups

                    grad_squared = (grad.float() * 1e3).pow(2).view(bsz, seq_len, num_groups, group_size)
                    mean_squared_grad = grad_squared.mean(dim=-1)  # -> [bsz, seq_len, num_groups]

                    # Move to CPU and store
                    saliency_data[layer_idx][module_name].append(mean_squared_grad.bfloat16().cpu())

                # Attach the gradient hook to 'out'
                out.register_hook(grad_hook)
            return forward_hook

        # Attach hooks only for layers in [start_layer, end_layer) if set
        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    # skip hooking this layer
                    continue

            # Register forward hooks for each module
            for module_name, module in analyzer.get_quantizable_modules(layer).items():
                h = module.register_forward_hook(make_forward_hook(layer_idx, module_name))
                saliency_hooks.append(h)

    # ----------------------------------------------------------------
    # 4) Weight-gradient hook
    # ----------------------------------------------------------------
    # def weight_grad_hook(grad):
    #     return grad

    # weight_hooks = []
    # for layer_idx in layers:
    #     for module in analyzer.get_quantizable_modules(layer_idx).values():
    #         weight_hooks.append(module.weight.register_hook(weight_grad_hook))

    gradients = [{module_name: 0 for module_name in analyzer.get_quantizable_modules(layer)} for layer in layers]
    # hessians = [{module_name: 0 for module_name in analyzer.get_quantizable_modules(layer)} for layer in layers]

    # ----------------------------------------------------------------
    # 5) Forward/backward pass over data
    # ----------------------------------------------------------------
    for i, tokens in enumerate(tqdm(input_tokens, desc="Calculating gradients and Hessians")):
        model.zero_grad()
        tokens = tokens.to(model.device).unsqueeze(0)
        outputs = model(input_ids=tokens, labels=tokens)
        loss = outputs.loss
        loss.backward()

        for layer_idx, layer in enumerate(layers):
            for module_name, module in analyzer.get_quantizable_modules(layer).items():
                grad = module.weight.grad.data.float()
                d_row, d_col = grad.shape
                
                group_size = d_row // num_groups
                grad_grouped = grad.view(num_groups, group_size, d_col)
                grad_mean = grad_grouped.mean(dim=1)
                
                gradients[layer_idx][module_name] += grad_mean.cpu()

                # fisher = torch.einsum('ngi,ngj->nij', grad_grouped, grad_grouped) / group_size
                # hessians[layer_idx][module_name] += fisher.cpu()

    # ----------------------------------------------------------------
    # 6) Remove hooks
    # ----------------------------------------------------------------
    # for h in weight_hooks:
    #     h.remove()

    for h in saliency_hooks:
        h.remove()

    # ----------------------------------------------------------------
    # 7) Move model back to CPU
    # ----------------------------------------------------------------
    model.cpu()

    # ----------------------------------------------------------------
    # 8) Harvest the weight gradients and Hessians
    # ----------------------------------------------------------------
    for layer_idx, layer in enumerate(layers):
        for module_name in analyzer.get_quantizable_modules(layer).keys():
            gradients[layer_idx][module_name] = gradients[layer_idx][module_name] / len(input_tokens)
            # hessians[layer_idx][module_name] = hessians[layer_idx][module_name] / len(input_tokens)

    # ----------------------------------------------------------------
    # 9) Save saliency per layer, if computed
    # ----------------------------------------------------------------
    if saliency_path is not None:
        logging.info(f"Saving saliency files to {saliency_path}...")

        # Ensure directory exists
        os.makedirs(saliency_path, exist_ok=True)

        # For each layer, gather module data -> single dictionary, then save
        for layer_idx, layer in enumerate(layers):
            if (start_layer is not None) and (end_layer is not None):
                if not (start_layer <= layer_idx < end_layer):
                    continue

            # Build dict of { module_name -> cat_tensor or None }
            layer_dict = {}
            for module_name, chunk_list in saliency_data[layer_idx].items():
                if len(chunk_list) > 0:
                    cat_tensor = torch.cat(chunk_list, dim=0)  # shape: [N, seq_len, num_groups]
                else:
                    cat_tensor = None
                layer_dict[module_name] = cat_tensor

            # If there's no data at all (empty?), you can choose to skip saving
            # But we'll save anyway for consistency
            filename = os.path.join(saliency_path, f"l{layer_idx}.pt")

            # if os.path.exists(filename):
            #     input(f"[WARNING] File {filename} already exists. "
            #           "Press Enter to overwrite or Ctrl+C to cancel.")

            # Save each layer's dictionary to l{layer_idx}.pt
            torch.save(layer_dict, filename)

    # ----------------------------------------------------------------
    # 10) Save the gradients and hessians (if needed)
    # ----------------------------------------------------------------
    # if gradients_path is not None:
    #     logging.info(f"Saving gradients to {gradients_path}...")
    #     if not gradients_path.endswith('.pt'):
    #         gradients_path = gradients_path + '.pt'
    #     os.makedirs(os.path.dirname(gradients_path), exist_ok=True)
    #     torch.save(gradients, gradients_path)

    # if hessians_path is not None:
    #     logging.info(f"Saving hessians to {hessians_path}...")
    #     if not hessians_path.endswith('.pt'):
    #         hessians_path = hessians_path + '.pt'
    #     os.makedirs(os.path.dirname(hessians_path), exist_ok=True)
    #     torch.save(hessians, hessians_path)

    # ----------------------------------------------------------------
    # 11) Return the gradients
    # ----------------------------------------------------------------
    return gradients

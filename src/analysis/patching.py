"""Activation Patching for Causal Purity (H3).

Tests whether topographic clusters causally control behavior by patching
activations and measuring the effect on classification logits.

Reference: Research plan Section 3.4
"""
import torch
from typing import Dict, List


def identify_selective_cluster(
    model, probe_loader, class_idx: int,
    num_units: int = None, device: str = 'cpu'
) -> List[int]:
    """Identify the set of units most selective for a given class.
    
    Args:
        model: Trained model (TinyViT).
        probe_loader: DataLoader with labeled probe images.
        class_idx: Target class to find selective units for.
        num_units: Number of top selective units to return.
        device: Device to run on.
    
    Returns:
        List of unit indices most selective for class_idx.
    """
    from src.analysis.monosemanticity import compute_class_selectivity
    # Handle both DatasetFolder (has classes attr) and Subset
    ds = probe_loader.dataset
    if hasattr(ds, 'dataset'):
        ds = ds.dataset  # unwrap Subset
    n_classes = len(ds.classes) if hasattr(ds, 'classes') else NUM_CLASSES if 'NUM_CLASSES' in dir() else 100
    selectivity = compute_class_selectivity(
        model, probe_loader,
        num_classes=n_classes,
        device=device
    )
    class_scores = selectivity[:, class_idx]
    if num_units is None:
        num_units = max(1, len(class_scores) // 10)
    top_units = class_scores.topk(min(num_units, len(class_scores))).indices.tolist()
    return top_units


def _collect_source_activation(model, source_loader, layer_idx, device):
    """Get average activations at the specified layer from source images."""
    avg_acts = None
    count = 0
    
    def hook_fn(module, input, output):
        nonlocal avg_acts, count
        if output.dim() == 3:
            acts = output.mean(dim=1).detach().cpu()  # (B, D)
        else:
            acts = output.detach().cpu()
        if avg_acts is None:
            avg_acts = acts.sum(dim=0)
        else:
            avg_acts += acts.sum(dim=0)
        count += acts.shape[0]
    
    hooks = []
    if layer_idx is not None:
        h = model.blocks[layer_idx].register_forward_hook(hook_fn)
    else:
        h = model.norm.register_forward_hook(hook_fn)
    hooks.append(h)
    
    with torch.no_grad():
        for images, _ in source_loader:
            _ = model(images.to(device))
    
    for h in hooks:
        h.remove()
    
    return avg_acts / count  # (D,)


def _patched_forward(model, images, patch_units, source_avg, layer_idx, device):
    """Forward pass with selected unit activations replaced by source values."""
    patched = [None]
    
    def patch_hook(module, input, output):
        out = output.clone()
        if out.dim() == 3:
            for u in patch_units:
                if u < out.shape[2]:
                    out[:, :, u] = source_avg[u].to(device)
        else:
            for u in patch_units:
                if u < out.shape[1]:
                    out[:, u] = source_avg[u].to(device)
        patched[0] = out
        return out
    
    hook = (model.blocks[layer_idx].register_forward_hook(patch_hook)
            if layer_idx is not None
            else model.norm.register_forward_hook(patch_hook))
    
    with torch.no_grad():
        _ = model(images.to(device))
    
    hook.remove()
    
    # Extract CLS and run through remaining layers
    if patched[0].dim() == 3:
        cls = patched[0][:, 0, :]
        cls = model.norm(cls)
    else:
        cls = patched[0]
    
    return model.head(cls).cpu()


def run_patching_experiment(
    model, test_loader, patch_units: List[int],
    source_loader, layer_idx: int = None, device: str = 'cpu'
) -> Dict:
    """Run activation patching experiment.
    
    For each test image, run model normally and with patched activations,
    then measure delta_logit = clean_logit - patched_logit.
    
    Args:
        model: Trained model.
        test_loader: Test images to evaluate on.
        patch_units: Unit indices to patch.
        source_loader: Source images for replacement activations.
        layer_idx: Which transformer block to patch (None = final norm output).
        device: Device to run on.
    
    Returns:
        Dict with delta_logits, clean_logits, patched_logits.
    """
    model.eval()
    source_avg = _collect_source_activation(model, source_loader, layer_idx, device)
    
    clean_list = []
    patched_list = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            clean_out = model(images.to(device))
            clean_list.append(clean_out.cpu())
            patched_out = _patched_forward(
                model, images, patch_units, source_avg, layer_idx, device
            )
            patched_list.append(patched_out)
    
    clean_logits = torch.cat(clean_list, dim=0)
    patched_logits = torch.cat(patched_list, dim=0)
    
    return {
        'delta_logits': clean_logits - patched_logits,
        'clean_logits': clean_logits,
        'patched_logits': patched_logits,
    }

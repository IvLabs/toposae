"""Activation Patching for Causal Purity (H3).

Tests whether topographic clusters causally control behavior by patching
activations and measuring the effect on classification logits.
"""
import torch
from typing import Dict, List


def identify_selective_cluster(
    model, probe_loader, class_idx: int,
    num_units: int = None, device: str = "cpu", num_classes: int = 100,
) -> List[int]:
    """Find the top-K units most selective for class_idx."""
    from src.analysis.monosemanticity import compute_class_selectivity
    selectivity = compute_class_selectivity(
        model, probe_loader, num_classes=num_classes, device=device
    )
    class_scores = selectivity[:, class_idx]
    if num_units is None:
        num_units = max(1, len(class_scores) // 10)
    top_units = class_scores.topk(min(num_units, len(class_scores))).indices.tolist()
    return top_units


def _collect_source_activation(model, source_loader, layer_idx: int, device: str):
    """Average block output across all source images."""
    avg_acts = None
    count = 0

    def hook_fn(module, input, output):
        nonlocal avg_acts, count
        # output: (B, N, D) for ViT; take CLS token mean across sequence for robustness
        acts = output.detach().cpu()  # (B, N, D)
        summed = acts.sum(dim=0)       # (N, D) — sum over batch
        if avg_acts is None:
            avg_acts = summed
        else:
            avg_acts += summed
        count += acts.shape[0]

    hook = model.blocks[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        for images, _ in source_loader:
            model(images.to(device))
    hook.remove()

    return avg_acts / count  # (N, D)


def run_patching_experiment(
    model, test_loader, patch_units: List[int],
    source_loader, layer_idx: int = None, device: str = "cpu",
) -> Dict:
    """Patch selected units in block `layer_idx` and measure delta logit.

    Uses a hook to intercept block output, replaces `patch_units` dimensions
    with source averages, and lets the rest of the forward pass proceed
    normally. No manual model.norm / model.head calls needed.
    """
    model.eval()

    source_avg = _collect_source_activation(model, source_loader, layer_idx, device)
    source_avg = source_avg.to(device)          # (N, D)
    unit_idx = torch.tensor(patch_units, device=device)

    clean_list, patched_list = [], []

    with torch.no_grad():
        # ── Clean forward ──────────────────────────────────────────────────
        for images, _ in test_loader:
            out = model(images.to(device))
            clean_list.append(out.cpu())

        # ── Patched forward ────────────────────────────────────────────────
        def patch_hook(module, input, output):
            out = output.clone()  # (B, N, D)
            # Replace selected feature dimensions with source averages
            out[:, :, unit_idx] = source_avg[:, unit_idx].unsqueeze(0)
            return out

        hook = model.blocks[layer_idx].register_forward_hook(patch_hook)
        for images, _ in test_loader:
            out = model(images.to(device))
            patched_list.append(out.cpu())
        hook.remove()

    clean_logits   = torch.cat(clean_list,   dim=0)
    patched_logits = torch.cat(patched_list, dim=0)

    return {
        "delta_logits":   clean_logits - patched_logits,
        "clean_logits":   clean_logits,
        "patched_logits": patched_logits,
    }

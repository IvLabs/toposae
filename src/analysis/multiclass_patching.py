"""Multi-class activation patching for EXP_016.

Runs the H3 patching protocol across many classes rather than just class 0.
This gives n_seeds × n_classes paired observations per variant, providing
the statistical power to properly test H3.
"""
from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def _get_class_indices(dataset, class_idx: int) -> list[int]:
    """Return sample indices belonging to class_idx."""
    indices = []
    # Handle both ImageFolder (has .samples) and Subset wrappers
    base = dataset
    offset = 0
    if isinstance(dataset, Subset):
        subset_indices = dataset.indices
        base = dataset.dataset
        for local_i, global_i in enumerate(subset_indices):
            _, label = base.samples[global_i]
            if label == class_idx:
                indices.append(local_i)
        return indices

    for i, (_, label) in enumerate(base.samples):
        if label == class_idx:
            indices.append(i)
    return indices


def _collect_source_avg(model: nn.Module, loader: DataLoader, layer_idx: int, device: str) -> torch.Tensor:
    """Average block output (CLS token dim) across all source images."""
    avg = None
    count = 0

    def hook(module, input, output):
        nonlocal avg, count
        acts = output.detach().cpu()  # (B, N, D)
        s = acts.sum(dim=0)           # (N, D)
        nonlocal avg
        avg = s if avg is None else avg + s
        count += acts.shape[0]

    h = model.blocks[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        for imgs, _ in loader:
            model(imgs.to(device))
    h.remove()
    return avg / count  # (N, D)


@torch.no_grad()
def _mean_abs_delta_logit(
    model: nn.Module,
    test_loader: DataLoader,
    patch_units: list[int],
    source_avg: torch.Tensor,
    layer_idx: int,
    class_idx: int,
    device: str,
) -> float:
    """Mean |Δlogit| for class_idx when patching patch_units."""
    source_avg = source_avg.to(device)
    unit_idx = torch.tensor(patch_units, device=device)

    clean_logits, patched_logits = [], []

    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        clean_logits.append(model(imgs).cpu())

    def patch_hook(module, input, output):
        out = output.clone()
        out[:, :, unit_idx] = source_avg[:, unit_idx].unsqueeze(0)
        return out

    h = model.blocks[layer_idx].register_forward_hook(patch_hook)
    for imgs, _ in test_loader:
        patched_logits.append(model(imgs.to(device)).cpu())
    h.remove()

    clean  = torch.cat(clean_logits,   dim=0)[:, class_idx]
    patched = torch.cat(patched_logits, dim=0)[:, class_idx]
    return (clean - patched).abs().mean().item()


def run_multiclass_patching(
    model: nn.Module,
    val_dataset,
    layer_idx: int,
    num_units: int = 16,
    n_classes: int = 20,
    n_test_per_class: int = 50,
    n_source: int = 200,
    device: str = "cpu",
    num_total_classes: int = 100,
    seed: int = 0,
) -> dict[int, dict]:
    """Run patching for n_classes randomly sampled classes.

    For each class:
    - Identify top-num_units selective units (cluster)
    - Sample a random unit set of the same size
    - Measure |Δlogit| for cluster vs random
    - Return per-class ratio

    Returns:
        dict mapping class_idx -> {cluster_dlogit, random_dlogit, ratio}
    """
    model.eval()
    rng = random.Random(seed)

    # Sample classes uniformly (skip any with < 10 val images)
    all_classes = list(range(num_total_classes))
    rng.shuffle(all_classes)

    results = {}
    hidden_dim = None

    for class_idx in all_classes:
        if len(results) >= n_classes:
            break

        class_idx_indices = _get_class_indices(val_dataset, class_idx)
        other_indices = [i for i in range(len(val_dataset)) if i not in set(class_idx_indices)]

        if len(class_idx_indices) < 10:
            continue

        # Probe loader: images of this class (for selectivity and test patching)
        probe_n = min(n_test_per_class, len(class_idx_indices))
        probe_subset = Subset(val_dataset, class_idx_indices[:probe_n])
        probe_loader = DataLoader(probe_subset, batch_size=16, shuffle=False, num_workers=2)

        # Source loader: images NOT of this class
        source_n = min(n_source, len(other_indices))
        source_subset = Subset(val_dataset, other_indices[:source_n])
        source_loader = DataLoader(source_subset, batch_size=16, shuffle=False, num_workers=2)

        # Identify selective units for this class
        from src.analysis.monosemanticity import compute_class_selectivity
        selectivity = compute_class_selectivity(
            model, probe_loader, num_classes=num_total_classes, device=device
        )  # (D, C)
        if hidden_dim is None:
            hidden_dim = selectivity.shape[0]

        class_scores = selectivity[:, class_idx]
        cluster = class_scores.topk(min(num_units, hidden_dim)).indices.tolist()

        # Random unit baseline (same size, fixed per class for reproducibility)
        rng_units = random.Random(seed + class_idx)
        random_units = rng_units.sample(range(hidden_dim), min(num_units, hidden_dim))

        # Collect source average activation (once per class)
        source_avg = _collect_source_avg(model, source_loader, layer_idx, device)

        # Measure Δlogit for cluster and random
        cluster_d = _mean_abs_delta_logit(
            model, probe_loader, cluster, source_avg, layer_idx, class_idx, device
        )
        random_d = _mean_abs_delta_logit(
            model, probe_loader, random_units, source_avg, layer_idx, class_idx, device
        )

        ratio = cluster_d / random_d if random_d > 1e-8 else float("nan")
        results[class_idx] = {
            "cluster_dlogit": cluster_d,
            "random_dlogit": random_d,
            "ratio": ratio,
        }

    return results

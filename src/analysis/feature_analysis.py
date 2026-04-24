"""Feature-level analysis for EXP_012 (SAE monosemanticity) and EXP_011 (PCA).

EXP_012: Measures monosemanticity at the SAE feature level rather than
         the raw neuron level (H1 null rescue). For each SAE feature,
         compute its class selectivity score using the same M_u formula
         as H1 but applied to SAE features instead of neurons.

EXP_011: PCA dimensionality control. Finds d_95 (components for 95%
         variance) and compares against SAE L0 to check whether the L0
         reduction is simply a dimensionality effect.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Activation extraction ─────────────────────────────────────────────────────

def extract_layer_activations(
    model: nn.Module,
    loader: DataLoader,
    layer_idx: int,
    device: str,
    cache_path: Optional[Path] = None,
) -> tuple[torch.Tensor, list[int]]:
    """Extract CLS-token activations from block `layer_idx`.

    Returns (activations [N, D], labels [N]).
    If cache_path is given and exists, loads from disk instead.
    """
    if cache_path is not None and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["acts"], data["labels"]

    acts, labels = [], []

    def hook(module, input, output):
        acts.append(output[:, 0, :].detach().cpu())

    h = model.blocks[layer_idx].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            model(imgs.to(device))
            labels.extend(lbls.tolist())
    h.remove()

    acts_tensor   = torch.cat(acts, dim=0)
    labels_list   = labels

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"acts": acts_tensor, "labels": labels_list}, cache_path)

    return acts_tensor, labels_list


# ── EXP_012: SAE feature monosemanticity ─────────────────────────────────────

def compute_feature_selectivity(
    sae_hidden: torch.Tensor,
    labels: list[int],
    num_classes: int,
) -> torch.Tensor:
    """Compute mean activation per SAE feature per class.

    Args:
        sae_hidden: (N, M) feature activations from the SAE encoder.
        labels: (N,) class indices.
        num_classes: Total number of classes.

    Returns:
        (M, C) tensor of mean activations per feature per class.
    """
    M = sae_hidden.shape[1]
    selectivity = torch.zeros(M, num_classes)
    counts = torch.zeros(num_classes)

    label_tensor = torch.tensor(labels)
    for c in range(num_classes):
        mask = label_tensor == c
        if mask.any():
            selectivity[:, c] = sae_hidden[mask].mean(dim=0)
            counts[c] = mask.sum().float()

    return selectivity  # (M, C)


def feature_monosemanticity_scores(selectivity: torch.Tensor) -> torch.Tensor:
    """M_u = max(s_u) / sum(s_u) for each SAE feature u.

    Same formula as the H1 neuron-level score but applied to features.
    Returns (M,) tensor in [1/C, 1].
    """
    sel = selectivity.clamp(min=0.0)
    row_sums = sel.sum(dim=1, keepdim=True).clamp(min=1e-10)
    normed = sel / row_sums
    return normed.max(dim=1).values  # (M,)


def sae_monosemanticity_summary(
    scores: torch.Tensor,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict:
    return {
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "max": float(scores.max()),
        **{f"frac_gt_{t}": float((scores > t).float().mean()) for t in thresholds},
    }


# ── EXP_011: PCA dimensionality ───────────────────────────────────────────────

def pca_effective_dim(acts: torch.Tensor, variance_threshold: float = 0.95) -> dict:
    """Find the number of PCA components needed to explain `variance_threshold` variance.

    Args:
        acts: (N, D) activation matrix.
        variance_threshold: Cumulative variance fraction (default 0.95).

    Returns:
        dict with d_95, explained variance per component, total variance.
    """
    X = acts.float().numpy()
    X -= X.mean(axis=0, keepdims=True)

    # SVD-based PCA (more numerically stable than eig for tall matrices)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    var = (s ** 2) / (len(X) - 1)
    cumvar = np.cumsum(var) / var.sum()

    d_thresh = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return {
        "d_95": d_thresh,
        "total_variance": float(var.sum()),
        "top10_variance_frac": float(cumvar[min(9, len(cumvar) - 1)]),
    }

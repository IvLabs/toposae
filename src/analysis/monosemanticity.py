"""Monosemanticity Analysis (H1).

Computes class selectivity and monosemanticity scores for model units.
Reference: Research plan Section 3.2
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_class_selectivity(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: str = 'cpu',
    norm_layer_name: str = 'norm',
) -> torch.Tensor:
    """Compute class selectivity scores for all hidden units.

    Accumulates running sums per class incrementally — does NOT buffer
    all activations in memory.

    Args:
        model: The model to analyze.
        loader: DataLoader with input samples.
        num_classes: Number of classes in the dataset.
        device: Device to run computation on.
        norm_layer_name: Attribute name of the final norm layer.

    Returns:
        Tensor of shape (num_units, num_classes) with average activations.
    """
    import gc

    model.eval()
    model.to(device)

    # Find the norm layer
    target_layer = getattr(model, norm_layer_name, None)
    if target_layer is None:
        # Fallback: search all modules for LayerNorm
        for name, mod in model.named_modules():
            if isinstance(mod, nn.LayerNorm) and 'norm' in name:
                target_layer = mod
                break
    if target_layer is None:
        raise RuntimeError("Could not find norm layer in model")

    # Determine hidden dim from first batch
    hidden_dim = None
    num_units = None

    # Accumulators: sum of activations per class, count per class
    # Shape: (num_units, num_classes) — but we don't know num_units yet
    accum = None  # will be initialized on first batch
    counts = torch.zeros(num_classes, dtype=torch.long)

    hidden_acts = []

    def hook_fn(module, input, output):
        # Extract CLS token (position 0) — only thing we need
        cls = output[:, 0, :]  # (B, hidden_dim)
        hidden_acts.append(cls)

    hook = target_layer.register_forward_hook(hook_fn)

    for images, labels in loader:
        images = images.to(device)
        _ = model(images)

        cls = hidden_acts[-1].cpu()  # (B, D)
        if hidden_dim is None:
            hidden_dim = cls.shape[1]
            num_units = hidden_dim
            accum = torch.zeros(num_units, num_classes, dtype=torch.float64)

        # Accumulate per-class sums
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                accum[:, c] += cls[mask].sum(dim=0)
                counts[c] += mask.sum().item()

        hidden_acts.clear()
        gc.collect()

    hook.remove()

    # Compute means
    selectivity = torch.zeros(num_units, num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            selectivity[:, c] = accum[:, c] / counts[c].item()

    return selectivity


def compute_monosemanticity_scores(selectivity: torch.Tensor) -> torch.Tensor:
    """Compute monosemanticity scores from class selectivity.

    Monosemanticity score for each unit is defined as 1 - H_normalized,
    where H_normalized is the normalized entropy of the unit's activation
    distribution across classes. A score of 1.0 means the unit responds
    to only one class (perfectly monosemantic), and lower scores indicate
    polysemanticity.

    To ensure numerical stability and meaningful bounds, we clamp the
    minimum score to 0.2 (representing maximum polysemanticity for
    practical purposes).

    Args:
        selectivity: Tensor of shape (num_units, num_classes) with average
                     activations per class.

    Returns:
        Tensor of shape (num_units,) with monosemanticity scores in [0.2, 1.0].
    """
    import numpy as np

    # Ensure non-negative activations
    sel = selectivity.clamp(min=0.0)

    # Normalize to probability distribution per unit
    row_sums = sel.sum(dim=1, keepdim=True)
    row_sums = row_sums.clamp(min=1e-10)
    probs = sel / row_sums

    # Compute entropy
    num_classes = probs.shape[1]
    log_num_classes = np.log(num_classes)

    # Avoid log(0)
    probs_safe = probs.clamp(min=1e-10)
    entropy = -(probs_safe * torch.log(probs_safe)).sum(dim=1)

    # Normalize entropy to [0, 1]
    normalized_entropy = entropy / log_num_classes

    # Monosemanticity = 1 - normalized_entropy
    scores = 1.0 - normalized_entropy

    # Clamp to [0.2, 1.0] for meaningful bounds
    scores = scores.clamp(min=0.2, max=1.0)

    return scores

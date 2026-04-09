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
    device: str = 'cpu'
) -> torch.Tensor:
    """Compute class selectivity scores for all hidden units.

    For each unit (neuron) in the model's hidden layers, computes the average
    activation per class. Returns a tensor of shape (num_units, num_classes).

    Args:
        model: The model to analyze.
        loader: DataLoader with input samples.
        num_classes: Number of classes in the dataset.
        device: Device to run computation on.

    Returns:
        Tensor of shape (num_units, num_classes) with average activations.
    """
    model.eval()
    model.to(device)

    # Collect activations from the last layer norm (before head)
    activations_list = []
    labels_list = []

    hook_handle = None
    hidden_activations = []

    def hook_fn(module, input, output):
        hidden_activations.append(output)

    # Register hook on the norm layer to get hidden representations
    hook_handle = model.norm.register_forward_hook(hook_fn)

    for images, labels in loader:
        images = images.to(device)
        _ = model(images)
        activations_list.append(hidden_activations[-1].cpu())
        labels_list.append(labels)

    hook_handle.remove()

    # Concatenate all activations and labels
    all_activations = torch.cat(activations_list, dim=0)  # (N, seq_len, hidden_dim)
    all_labels = torch.cat(labels_list, dim=0)  # (N,)

    # Use CLS token activations (first position)
    cls_activations = all_activations[:, 0, :]  # (N, hidden_dim)

    # Compute mean activation per class
    num_units = cls_activations.shape[1]
    selectivity = torch.zeros(num_units, num_classes)

    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            selectivity[:, c] = cls_activations[mask].mean(dim=0)

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

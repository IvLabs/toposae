"""SAE Analysis for Superposition (H2).

TODO: Implement when ready for H2 testing.
Reference: Research plan Section 3.3
"""


def train_sae(activations, config):
    """Train Sparse Autoencoder on residual stream activations."""
    raise NotImplementedError(
        "SAE analysis pending - prototype focuses on H1 (monosemanticity scores). "
        "See research plan Section 3.3 for SAE training protocol."
    )


def evaluate_sae(sae, test_activations):
    """Evaluate SAE on held-out activations."""
    raise NotImplementedError(
        "SAE analysis pending - prototype focuses on H1 (monosemanticity scores)."
    )

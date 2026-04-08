"""Activation Patching for Causal Purity (H3).

TODO: Implement when ready for H3 testing.
Reference: Research plan Section 3.4
"""


def identify_selective_cluster(model, probe_data, class_idx):
    """Identify spatial cluster most selective for given class."""
    raise NotImplementedError(
        "Activation patching analysis pending - prototype focuses on H1 (monosemanticity scores). "
        "See research plan Section 3.4 for patching protocol."
    )


def run_patching_experiment(model, cluster_units, patch_source_images, test_images):
    """Run activation patching experiment."""
    raise NotImplementedError(
        "Activation patching analysis pending - prototype focuses on H1 (monosemanticity scores)."
    )

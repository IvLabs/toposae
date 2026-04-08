"""Tests for TinyVisionTransformer model."""
import pytest
import torch
from src.models.tiny_vit import TinyViT


def test_tiny_vit_output_shape():
    """Test that TinyViT produces correct output shapes."""
    model = TinyViT(
        num_classes=100, depth=4, hidden_dim=128, num_heads=4,
        patch_size=16, image_size=128, dropout=0.1
    )
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    assert output.shape == (2, 100), f"Expected (2, 100), got {output.shape}"


def test_tiny_vit_different_batch_sizes():
    """Test that TinyViT handles different batch sizes."""
    model = TinyViT(num_classes=100, depth=4, hidden_dim=128, num_heads=4, patch_size=16, image_size=128)
    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 3, 128, 128)
        output = model(x)
        assert output.shape[0] == batch_size


def test_tiny_vit_gradient_flow():
    """Test that gradients flow through the model."""
    model = TinyViT(num_classes=10, depth=4, hidden_dim=128, num_heads=4)
    x = torch.randn(2, 3, 128, 128)
    output = model(x)
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if 'proj' in name:
            assert param.grad is not None, f"No gradient for {name}"


def test_tiny_vit_get_attention_layers():
    """Test that we can extract attention projection layers for TopoLoss."""
    model = TinyViT(num_classes=100, depth=4, hidden_dim=128, num_heads=4)
    proj_layers = model.get_attention_proj_layers()
    assert len(proj_layers) == 4, f"Expected 4 layers, got {len(proj_layers)}"
    for name, layer in proj_layers.items():
        assert hasattr(layer, 'weight'), f"Layer {name} has no weight attribute"

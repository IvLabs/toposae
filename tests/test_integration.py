"""Integration test: full training run with synthetic data."""
import os
import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tiny_vit import TinyViT
from src.data.imagenet import SyntheticDataset
from torch.utils.data import DataLoader


def test_training_loop_with_synthetic_data():
    """Test that the full training loop runs without errors."""
    from src.experiments.train import train_epoch, validate, setup_topo_loss
    from torch.cuda.amp import GradScaler
    
    model = TinyViT(num_classes=10, depth=2, hidden_dim=64, num_heads=2, image_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_data = SyntheticDataset(num_samples=100, num_classes=10, image_size=64)
    val_data = SyntheticDataset(num_samples=20, num_classes=10, image_size=64)
    train_loader = DataLoader(train_data, batch_size=8, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    topo_loss, alpha = setup_topo_loss(model, {'experiment': {'alpha': 0.0}})
    scaler = GradScaler()
    
    metrics = train_epoch(model, train_loader, optimizer, topo_loss, alpha,
        accumulation_steps=2, device=device, scaler=scaler, epoch=1)
    
    assert 'train_loss' in metrics
    assert 'train_acc' in metrics
    assert metrics['train_acc'] >= 0.0
    
    val_metrics = validate(model, val_loader, device)
    assert 'val_acc' in val_metrics
    assert val_metrics['val_acc'] >= 0.0


def test_analysis_pipeline():
    """Test that monosemanticity analysis runs on model outputs."""
    from src.analysis.monosemanticity import compute_monosemanticity_scores, compute_class_selectivity
    
    model = TinyViT(num_classes=5, depth=2, hidden_dim=64, num_heads=2, image_size=64)
    data = SyntheticDataset(num_samples=50, num_classes=5, image_size=64)
    loader = DataLoader(data, batch_size=10, num_workers=0)
    
    selectivity = compute_class_selectivity(model, loader, num_classes=5, device='cpu')
    assert selectivity.shape[1] == 5
    
    scores = compute_monosemanticity_scores(selectivity)
    assert scores.shape[0] == selectivity.shape[0]
    assert (scores >= 0.2).all() and (scores <= 1.0).all()

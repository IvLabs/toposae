#!/usr/bin/env python
"""Training script for topographic Vision Transformers.

Usage:
    python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 0.0 --run_id baseline
    python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 0.1 --run_id topo_weak
    python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 1.0 --run_id topo_strong
"""
import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config, merge_configs, save_config
from src.models.tiny_vit import TinyViT
from src.data.imagenet import get_dataloaders
from src.utils.visualization import plot_training_curves


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config):
    model_cfg = config['model']
    if model_cfg['type'] == 'tiny_vit':
        return TinyViT(
            num_classes=config['data']['num_classes'],
            depth=model_cfg['depth'],
            hidden_dim=model_cfg['hidden_dim'],
            num_heads=model_cfg['num_heads'],
            patch_size=model_cfg['patch_size'],
            image_size=config['data']['image_size'],
            dropout=model_cfg.get('dropout', 0.1),
        )
    raise ValueError(f"Unknown model type: {model_cfg['type']}")


def setup_topo_loss(model, config):
    """Initialize TopoLoss. Returns (topo_loss_instance, alpha)."""
    alpha = config['experiment'].get('alpha', 0.0)
    if alpha == 0.0:
        return None, 0.0
    try:
        from topoloss import TopoLoss, LaplacianPyramid
        proj_layers = model.get_attention_proj_layers()
        topo_loss = TopoLoss(
            losses=[
                LaplacianPyramid.from_layer(
                    model=model, layer=layer,
                    factor_h=8.0, factor_w=8.0,
                    scale=config['topo_loss'].get('sigma', 2.0),
                )
                for layer in proj_layers.values()
            ]
        )
        return topo_loss, alpha
    except ImportError:
        print("Warning: topoloss not installed. TopoLoss disabled.")
        return None, 0.0


def train_epoch(model, train_loader, optimizer, topo_loss, alpha, accumulation_steps, device, scaler, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_topo_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        images, labels = images.to(device), labels.to(device)
        
        with autocast():
            outputs = model(images)
            loss_ce = torch.nn.functional.cross_entropy(outputs, labels) / accumulation_steps
        
        scaler.scale(loss_ce).backward()
        total_loss += loss_ce.item() * accumulation_steps
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if topo_loss is not None and alpha > 0:
                loss_topo = topo_loss.compute(model=model)
                scaler.scale(alpha * loss_topo).backward()
                total_topo_loss += loss_topo.item()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return {
        'train_loss': total_loss / len(train_loader),
        'topo_loss': total_topo_loss / max(1, len(train_loader) // accumulation_steps),
        'train_acc': correct / total,
    }


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return {'val_loss': total_loss / len(val_loader), 'val_acc': correct / total}


def save_checkpoint(model, optimizer, epoch, metrics, config, output_dir, is_best=False):
    checkpoint_dir = os.path.join(output_dir, config['experiment']['run_id'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
        'metrics': metrics,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
    if is_best:
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_best.pt'))
    save_config(config, os.path.join(checkpoint_dir, 'config.yaml'))


def main():
    parser = argparse.ArgumentParser(description='Train topographic ViT')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.alpha is not None:
        config = merge_configs(config, {'experiment': {'alpha': args.alpha}})
    if args.run_id is not None:
        config = merge_configs(config, {'experiment': {'run_id': args.run_id}})
    if args.epochs is not None:
        config = merge_configs(config, {'training': {'epochs': args.epochs}})
    
    set_seed(config['experiment']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=config['data']['data_dir'],
        num_classes=config['data']['num_classes'],
        image_size=config['data']['image_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )
    
    print("Creating model...")
    model = create_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    topo_loss, alpha = setup_topo_loss(model, config)
    print(f"TopoLoss alpha: {alpha}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=config['training']['lr_min'])
    scaler = GradScaler()
    
    print(f"Starting training for {config['training']['epochs']} epochs...")
    all_metrics = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'topo_loss': []}
    best_acc = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, topo_loss, alpha,
            config['data']['accumulation_steps'], device, scaler, epoch,
            grad_clip=config['training'].get('grad_clip', 1.0))
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{config['training']['epochs']} ({elapsed:.1f}s) - "
              f"Train Loss: {train_metrics['train_loss']:.4f} "
              f"Val Acc: {val_metrics['val_acc']:.4f} "
              f"TopoLoss: {train_metrics['topo_loss']:.4f}")
        
        all_metrics['train_loss'].append(train_metrics['train_loss'])
        all_metrics['val_loss'].append(val_metrics['val_loss'])
        all_metrics['val_acc'].append(val_metrics['val_acc'])
        all_metrics['topo_loss'].append(train_metrics['topo_loss'])
        
        is_best = val_metrics['val_acc'] > best_acc
        if is_best:
            best_acc = val_metrics['val_acc']
        if epoch % config['training']['checkpoint_every'] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_metrics, config,
                          config['output']['checkpoints_dir'], is_best)
    
    figures_dir = config['output']['figures_dir']
    run_id = config['experiment']['run_id']
    plot_training_curves(all_metrics, os.path.join(figures_dir, f'{run_id}_training_curves.png'),
                        title=f'Training Curves - {run_id}')
    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == '__main__':
    main()

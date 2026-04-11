#!/usr/bin/env python
"""Collect and save residual stream activations for all 3 ImageNet-100 models.

Saves activations to results/data/activations/<run_id>_layer<N>.pt
So analysis can load them from disk instead of recomputing forward passes.
"""
import torch, sys, os
sys.path.insert(0, ".")
from src.models.tiny_vit import TinyViT
from src.data.imagenet import get_dataloaders
from src.analysis.sae import collect_residual_stream

DEVICE = "cpu"
NUM_CLASSES = 100
IMAGE_SIZE = 128
DATA_DIR = "data/imagenet-100"
MIDDLE_LAYER = 2  # block 2 of 4
OUT_DIR = "results/data/activations"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading ImageNet-100 data...")
train_loader, val_loader = get_dataloaders(DATA_DIR, NUM_CLASSES, IMAGE_SIZE, batch_size=64, num_workers=2)

for run_id in ["baseline_imagenet100", "topo_weak_imagenet100", "topo_strong_imagenet100"]:
    print(f"\n{'='*50}")
    print(f"  {run_id}")
    print(f"{'='*50}")
    
    ckpt_path = f"results/data/checkpoints/{run_id}/checkpoint_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: checkpoint not found")
        continue
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    
    model = TinyViT(
        num_classes=cfg["data"]["num_classes"],
        depth=cfg["model"]["depth"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_heads=cfg["model"]["num_heads"],
        patch_size=cfg["model"]["patch_size"],
        image_size=cfg["data"]["image_size"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    layer = cfg["model"]["depth"] // 2
    print(f"  Collecting from layer {layer}...")
    
    # Train
    print(f"    Train ({len(train_loader.dataset)} images)...")
    train_acts = collect_residual_stream(model, train_loader, layer, DEVICE)
    print(f"    Shape: {train_acts.shape}, Size: {train_acts.element_size() * train_acts.nelement() / 1e6:.1f} MB")
    train_path = f"{OUT_DIR}/{run_id}_train_layer{layer}.pt"
    torch.save(train_acts, train_path)
    print(f"    Saved: {train_path}")
    
    # Val
    print(f"    Val ({len(val_loader.dataset)} images)...")
    val_acts = collect_residual_stream(model, val_loader, layer, DEVICE)
    print(f"    Shape: {val_acts.shape}, Size: {val_acts.element_size() * val_acts.nelement() / 1e6:.1f} MB")
    val_path = f"{OUT_DIR}/{run_id}_val_layer{layer}.pt"
    torch.save(val_acts, val_path)
    print(f"    Saved: {val_path}")

print(f"\nDone! All activations saved to {OUT_DIR}/")
total = sum(f.stat().st_size for f in os.scandir(OUT_DIR) if f.is_file())
print(f"Total disk usage: {total / 1e6:.1f} MB")

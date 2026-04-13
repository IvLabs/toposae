#!/usr/bin/env python
"""Collect and save residual stream activations for all 3 ViT-S/16 ImageNet-100 models.

Saves activations to results/data/activations/vit_s16_<run_id>_train_layer<N>.pt
Uses python3.10 + timm.
"""
import torch, sys, os
sys.path.insert(0, ".")

import timm
from src.data.imagenet import get_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100
IMAGE_SIZE = 224
DATA_DIR = "data/imagenet-100"
MIDDLE_LAYER = 6  # block 6 of 12
OUT_DIR = "results/data/activations"
RUN_IDS = ["baseline", "topo_weak", "topo_strong"]
NUM_CLASSES = 100

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print("Loading ImageNet-100 data (224x224)...")
train_loader, val_loader = get_dataloaders(DATA_DIR, NUM_CLASSES, IMAGE_SIZE, batch_size=64, num_workers=4)


def collect_residual_stream(model, dataloader, layer_idx, device):
    """Collect CLS token activations from a specific transformer layer."""
    cls_activations = []
    hooks = []

    def hook_fn(module, input, output):
        cls_activations.append(output[:, 0, :].detach().cpu())

    hook = model.blocks[layer_idx].register_forward_hook(hook_fn)
    hooks.append(hook)

    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            _ = model(images.to(device))

    for hook in hooks:
        hook.remove()

    return torch.cat(cls_activations, dim=0)


for run_id in RUN_IDS:
    print(f"\n{'='*50}")
    print(f"  {run_id}")
    print(f"{'='*50}")

    ckpt_path = f"results/data/checkpoints/{run_id}/checkpoint_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: checkpoint not found")
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Load timm model
    model = timm.create_model(
        'vit_small_patch16_224',
        num_classes=NUM_CLASSES,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    layer = MIDDLE_LAYER
    print(f"  Epoch: {ckpt['epoch']}, Val Acc: {ckpt['metrics']['val_acc']:.4f}")
    print(f"  Collecting from block {layer} (of 12)...")

    # Train
    print(f"    Train ({len(train_loader.dataset)} images)...")
    train_acts = collect_residual_stream(model, train_loader, layer, DEVICE)
    print(f"    Shape: {train_acts.shape}, Size: {train_acts.element_size() * train_acts.nelement() / 1e6:.1f} MB")
    train_path = f"{OUT_DIR}/vit_s16_{run_id}_train_layer{layer}.pt"
    torch.save(train_acts, train_path)
    print(f"    Saved: {train_path}")

    # Val
    print(f"    Val ({len(val_loader.dataset)} images)...")
    val_acts = collect_residual_stream(model, val_loader, layer, DEVICE)
    print(f"    Shape: {val_acts.shape}, Size: {val_acts.element_size() * val_acts.nelement() / 1e6:.1f} MB")
    val_path = f"{OUT_DIR}/vit_s16_{run_id}_val_layer{layer}.pt"
    torch.save(val_acts, val_path)
    print(f"    Saved: {val_path}")

print(f"\nDone! All activations saved to {OUT_DIR}/")
total = sum(f.stat().st_size for f in os.scandir(OUT_DIR) if f.is_file())
print(f"Total disk usage: {total / 1e6:.1f} MB")

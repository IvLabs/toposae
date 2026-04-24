#!/usr/bin/env python3
"""Train ResNet-18 with/without TopoLoss on ImageNet-100.

Usage (single run):
    python scripts/train_resnet18.py --alpha 0.0 --seed 42 --run-id resnet18_baseline_s42
    python scripts/train_resnet18.py --alpha 0.1 --seed 42 --run-id resnet18_weak_s42
    python scripts/train_resnet18.py --alpha 1.0 --seed 42 --run-id resnet18_strong_s42

See scripts/run_resnet18_all.sh to queue all 9 runs.

Checkpoints saved to: ../topo/topo_checkpoints/{run_id}/checkpoint_best.pt
Format matches ViT-S checkpoints for compatibility with analysis scripts.
"""
import argparse
import gc
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import timm
from src.data.imagenet import get_imagenet_transforms

CKPT_DIR = ROOT.parent / "topo" / "topo_checkpoints"
DATA_DIR = ROOT.parent / "topo" / "data" / "imagenet-100"

# TopoLoss target layers — all conv layers in ResNet-18 stages 1-4
TOPO_LAYERS = [
    "layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2",
    "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2",
    "layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2",
    "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size: int, num_workers: int = 4):
    train_tf = get_imagenet_transforms(224, is_training=True)
    val_tf   = get_imagenet_transforms(224, is_training=False)
    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_topo_loss(model: nn.Module, alpha: float, factor_h: float = 4.0, factor_w: float = 4.0):
    if alpha == 0.0:
        return None
    from topoloss import TopoLoss
    from topoloss.losses.laplacian_pyramid import LaplacianPyramid
    losses = []
    for layer_name in TOPO_LAYERS:
        # Walk the module tree by name
        layer = model
        for part in layer_name.split("."):
            layer = getattr(layer, part)
        losses.append(LaplacianPyramid.from_layer(
            model=model, layer=layer,
            factor_h=factor_h, factor_w=factor_w, scale=1.0,
        ))
    return TopoLoss(losses=losses)


def cosine_lr(optimizer, epoch: int, total_epochs: int,
              base_lr: float, min_lr: float, warmup_epochs: int):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        correct += (logits.argmax(1) == lbls.to(device)).sum().item()
        total   += lbls.size(0)
    return correct / total


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  run_id: {args.run_id}  |  α={args.alpha}  |  seed={args.seed}")

    set_seed(args.seed)

    print("Loading data...")
    train_loader, val_loader = get_loaders(batch_size=128)

    model = timm.create_model("resnet18", num_classes=100, pretrained=False).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    topo = build_topo_loss(model, args.alpha)
    if topo:
        print(f"TopoLoss active on {len(topo.losses)} conv layers")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scaler    = GradScaler()
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = CKPT_DIR / args.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc   = 0.0
    history    = {"train_loss": [], "val_acc": [], "topo_loss": []}

    for epoch in range(args.epochs):
        lr = cosine_lr(optimizer, epoch, args.epochs,
                       base_lr=1e-3, min_lr=1e-5, warmup_epochs=5)
        model.train()
        running_loss = running_topo = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:03d}/{args.epochs}", ncols=90, leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast():
                logits = model(imgs)
                ce_loss = criterion(logits, lbls)
                tl = topo.compute(model) if topo else torch.tensor(0.0, device=device)
                loss = ce_loss + args.alpha * tl

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += ce_loss.item()
            running_topo += tl.item()
            n_batches    += 1
            pbar.set_postfix(ce=f"{ce_loss.item():.3f}", topo=f"{tl.item():.4f}")

        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / n_batches
        avg_topo = running_topo / n_batches
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["topo_loss"].append(avg_topo)

        print(f"  Ep {epoch+1:3d}  lr={lr:.6f}  ce={avg_loss:.4f}  "
              f"topo={avg_topo:.4f}  val_acc={val_acc:.4f}"
              + (" ← best" if val_acc > best_acc else ""))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "alpha": args.alpha,
                "seed": args.seed,
                "run_id": args.run_id,
                "metrics": {"val_acc": val_acc},
                "history": history,
            }, ckpt_dir / "checkpoint_best.pt")

        # Also save latest for resuming
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "alpha": args.alpha,
            "seed": args.seed,
            "run_id": args.run_id,
            "best_acc": best_acc,
            "history": history,
        }, ckpt_dir / "checkpoint_latest.pt")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Best val_acc={best_acc:.4f} → {ckpt_dir / 'checkpoint_best.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",   type=float, required=True, help="TopoLoss weight")
    parser.add_argument("--seed",    type=int,   required=True, help="Random seed")
    parser.add_argument("--run-id",  type=str,   required=True, help="Checkpoint directory name")
    parser.add_argument("--epochs",  type=int,   default=90)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

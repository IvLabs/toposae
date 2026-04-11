#!/usr/bin/env python
"""Run full analysis (H1, H2, H3) on all 3 ImageNet-100 models.

Loads pre-computed activations from results/data/activations/ if available,
otherwise computes them on-the-fly (slower, more memory).
"""
import torch, sys, os
sys.path.insert(0, ".")
from src.models.tiny_vit import TinyViT
from src.data.imagenet import get_dataloaders
from src.analysis.monosemanticity import compute_monosemanticity_scores, compute_class_selectivity
from src.analysis.sae import train_sae, collect_residual_stream
from src.analysis.patching import identify_selective_cluster, run_patching_experiment
from src.utils.visualization import plot_monosemanticity_distribution, plot_comparison_bar_chart
from torch.utils.data import DataLoader

DEVICE = "cpu"
NUM_CLASSES = 100
IMAGE_SIZE = 128
DATA_DIR = "data/imagenet-100"
ACT_DIR = "results/data/activations"

print("Loading real ImageNet-100 data...")
train_loader, val_loader = get_dataloaders(DATA_DIR, NUM_CLASSES, IMAGE_SIZE, batch_size=32, num_workers=2)
test_loader = val_loader

results_h1 = {}
results_h2 = {}
results_h3 = {}

for run_id in ["baseline_imagenet100", "topo_weak_imagenet100", "topo_strong_imagenet100"]:
    print(f"\n{'='*60}")
    print(f"  {run_id.upper()}")
    print(f"{'='*60}")
    
    ckpt_path = f"results/data/checkpoints/{run_id}/checkpoint_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint not found, skipping")
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
    
    label = run_id.replace("_imagenet100", "").replace("_", " ").title()
    mid = cfg["model"]["depth"] // 2
    print(f"  Label: {label}")
    print(f"  Epoch: {ckpt['epoch']}, Val Acc: {ckpt['metrics']['val_acc']:.4f}")
    print(f"  Middle layer: {mid}")

    # Load or collect activations
    train_acts_path = f"{ACT_DIR}/{run_id}_train_layer{mid}.pt"
    val_acts_path = f"{ACT_DIR}/{run_id}_val_layer{mid}.pt"
    
    if os.path.exists(train_acts_path) and os.path.exists(val_acts_path):
        print(f"  Loading pre-computed activations from disk...")
        train_acts = torch.load(train_acts_path, map_location=DEVICE, weights_only=True)
        test_acts = torch.load(val_acts_path, map_location=DEVICE, weights_only=True)
    else:
        print(f"  Collecting activations (full dataset, ~65 MB)...")
        train_acts = collect_residual_stream(model, train_loader, mid, DEVICE)
        test_acts = collect_residual_stream(model, test_loader, mid, DEVICE)
        print(f"  Train: {train_acts.shape} ({train_acts.element_size()*train_acts.nelement()/1e6:.1f} MB)")
        print(f"  Val:   {test_acts.shape} ({test_acts.element_size()*test_acts.nelement()/1e6:.1f} MB)")

    # ========== H1: Monosemanticity ==========
    print(f"\n  [H1] Computing monosemanticity scores...")
    selectivity = compute_class_selectivity(model, test_loader, NUM_CLASSES, DEVICE)
    scores = compute_monosemanticity_scores(selectivity)
    results_h1[label] = scores
    print(f"    Mean score: {scores.mean().item():.4f}")
    print(f"    Frac > 0.5: {(scores > 0.5).float().mean().item():.4f}")
    print(f"    Median:     {scores.median().item():.4f}")

    # ========== H2: SAE (full dataset) ==========
    print(f"\n  [H2] Training SAE (full {len(train_acts)} activations)...")
    sae_cfg = dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3, epochs=50, batch_size=256)
    sae_out = train_sae(train_acts, sae_cfg, test_acts, DEVICE)
    m = sae_out["metrics"]
    results_h2[label] = {
        "L0 Norm": m["l0_norm"],
        "Dead Feature %": m["dead_feature_fraction"] * 100,
        "Recon Loss": m["reconstruction_loss"],
    }
    print(f"    L0={m['l0_norm']:.1f}, Dead={m['dead_feature_fraction']*100:.1f}%, Recon={m['reconstruction_loss']:.6f}")

    # ========== H3: Patching ==========
    print(f"\n  [H3] Running patching experiment...")
    cluster = identify_selective_cluster(model, train_loader, 0, num_units=16, device=DEVICE)
    pr = run_patching_experiment(model, test_loader, cluster, test_loader, layer_idx=mid, device=DEVICE)
    td = pr["delta_logits"][:, 0].abs().mean().item()
    ru = torch.randperm(train_acts.shape[1])[:len(cluster)].tolist()
    rr = run_patching_experiment(model, test_loader, ru, test_loader, layer_idx=mid, device=DEVICE)
    rd = rr["delta_logits"][:, 0].abs().mean().item()
    results_h3[label] = {
        "Cluster |dlogit|": td,
        "Random |dlogit|": rd,
        "Ratio": td / rd if rd > 0 else 0,
    }
    print(f"    Cluster={td:.4f}, Random={rd:.4f}, Ratio={results_h3[label]['Ratio']:.2f}")

# ========== Summary Tables ==========
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")

print("\n=== H1: Monosemanticity ===")
for lbl, s in results_h1.items():
    print(f"  {lbl:15s}: mean={s.mean().item():.4f}, median={s.median().item():.4f}, frac>0.5={(s>0.5).float().mean().item():.4f}")

print("\n=== H2: SAE ===")
for lbl, m in results_h2.items():
    print(f"  {lbl:15s}: L0={m['L0 Norm']:.1f}, Dead={m['Dead Feature %']:.1f}%, Recon={m['Recon Loss']:.6f}")

print("\n=== H3: Patching ===")
for lbl, m in results_h3.items():
    print(f"  {lbl:15s}: cluster={m['Cluster |dlogit|']:.4f}, random={m['Random |dlogit|']:.4f}, ratio={m['Ratio']:.2f}x")

# ========== Plots ==========
print(f"\nGenerating plots...")
plot_monosemanticity_distribution(results_h1, "results/figures/imagenet100_monosemanticity.png", "ImageNet-100 Monosemanticity Scores")

h1_summary = {lbl: {
    "Mean": s.mean().item(), "Median": s.median().item(), "Max": s.max().item(),
    "Frac>0.5": (s > 0.5).float().mean().item()
} for lbl, s in results_h1.items()}
plot_comparison_bar_chart(h1_summary, "results/figures/imagenet100_h1_summary.png", "H1 Monosemanticity Summary")

plot_comparison_bar_chart(results_h2, "results/figures/imagenet100_sae.png", "H2 SAE Metrics")

h3_plot = {lbl: {"Cluster |dlogit|": m["Cluster |dlogit|"], "Random |dlogit|": m["Random |dlogit|"]} for lbl, m in results_h3.items()}
plot_comparison_bar_chart(h3_plot, "results/figures/imagenet100_patching.png", "H3 Patching |dlogit|")

print("\nAll done! Figures saved to results/figures/imagenet100_*")

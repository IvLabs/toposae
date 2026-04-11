#!/usr/bin/env python
"""Run multi-seed training and analysis for robustness check.

Trains all 3 variants (α=0.0, 0.1, 1.0) across multiple seeds,
then runs H1/H2/H3 analysis and reports mean ± std with statistical tests.

Usage:
    # Quick test (2 seeds, 20 epochs):
    python scripts/run_multi_seed.py --seeds 42 123 --epochs 20

    # Full run (5 seeds, 50 epochs):
    python scripts/run_multi_seed.py --seeds 42 123 456 789 1024 --epochs 50
"""
import argparse
import os
import sys
import json
import numpy as np
import scipy.stats as stats
sys.path.insert(0, ".")


def train_seed(seed, alpha, run_id, epochs, config_path):
    """Train a single model variant."""
    import subprocess
    cmd = [
        sys.executable, "src/experiments/train.py",
        "--config", config_path,
        "--alpha", str(alpha),
        "--run_id", run_id,
        "--seed", str(seed),
        "--epochs", str(epochs),
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None
    
    # Extract val_acc from last epoch line
    for line in result.stdout.split("\n"):
        if "Val Acc:" in line:
            parts = line.split("Val Acc:")[1].strip().split()
            return float(parts[0])
    return None


def run_analysis(ckpt_path, data_dir="data/imagenet-100", device="cpu"):
    """Run H1/H2/H3 analysis on a single checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from src.models.tiny_vit import TinyViT
    from src.data.imagenet import get_dataloaders
    from src.analysis.monosemanticity import compute_monosemanticity_scores, compute_class_selectivity
    from src.analysis.sae import train_sae, collect_residual_stream
    from src.analysis.patching import identify_selective_cluster, run_patching_experiment

    train_loader, val_loader = get_dataloaders(data_dir, 100, 128, batch_size=32, num_workers=2)
    
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
    
    mid = cfg["model"]["depth"] // 2
    
    # H1
    sel = compute_class_selectivity(model, val_loader, 100, device)
    scores = compute_monosemanticity_scores(sel)
    
    # H2
    acts = collect_residual_stream(model, train_loader, mid, device)
    test_acts = collect_residual_stream(model, val_loader, mid, device)
    sae_out = train_sae(acts, dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3, epochs=50, batch_size=256), test_acts, device)
    
    # H3
    cluster = identify_selective_cluster(model, train_loader, 0, num_units=16, device=device)
    pr = run_patching_experiment(model, val_loader, cluster, val_loader, layer_idx=mid, device=device)
    td = pr["delta_logits"][:, 0].abs().mean().item()
    ru = torch.randperm(acts.shape[1])[:len(cluster)].tolist()
    rr = run_patching_experiment(model, val_loader, ru, val_loader, layer_idx=mid, device=device)
    rd = rr["delta_logits"][:, 0].abs().mean().item()
    
    return {
        "mono_mean": scores.mean().item(),
        "mono_frac_05": (scores > 0.5).float().mean().item(),
        "l0": sae_out["metrics"]["l0_norm"],
        "recon": sae_out["metrics"]["reconstruction_loss"],
        "patch_ratio": td / rd if rd > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--config", type=str, default="configs/exp_001_ultra_minimal.yaml")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just run analysis on existing checkpoints")
    parser.add_argument("--skip-analysis", action="store_true", help="Only train, skip analysis")
    args = parser.parse_args()

    alphas = [0.0, 0.1, 1.0]
    labels = ["baseline", "topo_weak", "topo_strong"]
    
    results = {lbl: {s: None for s in args.seeds} for lbl in labels}
    
    if not args.skip_training:
        print(f"\n{'='*60}")
        print(f"  MULTI-SEED TRAINING: {len(args.seeds)} seeds × 3 variants = {len(args.seeds)*3} runs")
        print(f"  Epochs: {args.epochs}")
        print(f"{'='*60}\n")
        
        for seed in args.seeds:
            for alpha, label in zip(alphas, labels):
                run_id = f"{label}_seed{seed}"
                print(f"\n  Seed {seed}, {label} (α={alpha}):")
                acc = train_seed(seed, alpha, run_id, args.epochs, args.config)
                if acc is not None:
                    print(f"    Val Acc: {acc:.4f}")
    
    if not args.skip_analysis:
        print(f"\n{'='*60}")
        print(f"  MULTI-SEED ANALYSIS")
        print(f"{'='*60}\n")
        
        for seed in args.seeds:
            for alpha, label in zip(alphas, labels):
                run_id = f"{label}_seed{seed}"
                ckpt_path = f"results/data/checkpoints/{run_id}/checkpoint_best.pt"
                if not os.path.exists(ckpt_path):
                    print(f"  SKIP {run_id}: checkpoint not found")
                    continue
                print(f"\n  Analyzing {run_id}...")
                metrics = run_analysis(ckpt_path)
                results[label][seed] = metrics
                print(f"    mono={metrics['mono_mean']:.4f}, L0={metrics['l0']:.1f}, patch_ratio={metrics['patch_ratio']:.2f}x")
    
    # Summary
    print(f"\n{'='*60}")
    print("  MULTI-SEED SUMMARY")
    print(f"{'='*60}\n")
    
    for label in labels:
        seeds_with_data = {s: m for s, m in results[label].items() if m is not None}
        if not seeds_with_data:
            print(f"{label:12s}: no data")
            continue
        
        for metric in ["mono_mean", "mono_frac_05", "l0", "recon", "patch_ratio"]:
            values = [m[metric] for m in seeds_with_data.values()]
            mean, std = np.mean(values), np.std(values)
            print(f"{label:12s} {metric:15s}: {mean:.4f} ± {std:.4f}")
        print()
    
    # Statistical tests
    print("="*60)
    print("  STATISTICAL SIGNIFICANCE (paired t-test)")
    print("="*60)
    print()
    
    for metric in ["mono_mean", "l0", "patch_ratio"]:
        print(f"\n  {metric}:")
        for label in ["topo_weak", "topo_strong"]:
            baseline_vals = [m[metric] for s, m in results["baseline"].items() if results[label][s] is not None and m is not None]
            topo_vals = [m[metric] for s, m in results[label].items() if results[label][s] is not None and m is not None]
            if len(baseline_vals) >= 2:
                t, p = stats.ttest_rel(baseline_vals, topo_vals)
                sig = "✓" if p < 0.05 else "✗"
                print(f"    {label:12s} vs baseline: t={t:.3f}, p={p:.4f} {sig}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP_012b: SAE feature-level monosemanticity on 25-30k training images (change 8).

The original EXP_012 used 5k val images, giving 82% feature activity per token
(insufficient sparsity for reliable selectivity measurement). This script uses
25k training images, providing a more representative distribution.

Two outcomes are informative:
  - Still null: H1/H3 dissociation claim strengthens (monosemanticity gains absent
    in both bases).
  - Positive: story shifts to 'feature-basis monosemanticity emerges, not neuron-basis'.

Usage:
    python scripts/run_exp012_large_eval.py [--n-train 25000] [--dry-run]

Reads:  ~/topo/topo_checkpoints/{run_id}/checkpoint_best.pt
        ~/topo/data/imagenet-100/train/
Writes: results/json/exp012b_sae_monosemanticity_large.json
        results/figures/exp012b_sae_monosemanticity_large.png
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import timm
from src.analysis.feature_analysis import (
    extract_layer_activations, compute_feature_selectivity,
    feature_monosemanticity_scores, sae_monosemanticity_summary,
)
from src.analysis.sae import SparseAutoencoder, train_sae
from src.data.imagenet import get_imagenet_transforms

CKPT_DIR  = ROOT.parent / "topo" / "topo_checkpoints"
DATA_DIR  = ROOT.parent / "topo" / "data" / "imagenet-100"
ACT_DIR   = ROOT / "results" / "data" / "activations"
JSON_DIR  = ROOT / "results" / "json"
FIGS_DIR  = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
ACT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 100
IMAGE_SIZE   = 224
MIDDLE_LAYER = 6

# Use the same 3-seed, 3-alpha runs as the main analysis
RUNS = [
    ("baseline_s42",    0.0, 42),
    ("baseline_s123",   0.0, 123),
    ("baseline_s456",   0.0, 456),
    ("topo_weak_s42",   0.1, 42),
    ("topo_weak_s123",  0.1, 123),
    ("topo_weak_s456",  0.1, 456),
    ("topo_strong_s42", 1.0, 42),
    ("topo_strong_s123",1.0, 123),
    ("topo_strong_s456",1.0, 456),
]

ALPHA_LABELS = {0.0: "Baseline\n(α=0)", 0.1: "α=0.1", 1.0: "α=1.0"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.1: "#DD8452", 1.0: "#55A868"}

# Use higher L1 penalty than EXP_012 (0.001 → 0.005) to increase sparsity;
# keep other hypers the same.
SAE_CFG = dict(expansion_factor=4, l1_penalty=0.005, lr=1e-3, epochs=50, batch_size=256)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def get_train_loader(n_train: int, seed: int) -> DataLoader:
    """Return a DataLoader for a stratified subset of the training set."""
    train_dir = DATA_DIR / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"ImageNet-100 train not found at {train_dir}")
    transform = get_imagenet_transforms(IMAGE_SIZE, is_training=False)  # no aug for eval
    dataset   = datasets.ImageFolder(str(train_dir), transform=transform)

    # Stratified subsample: n_train // NUM_CLASSES images per class
    per_class = max(1, n_train // NUM_CLASSES)
    rng       = np.random.default_rng(seed)
    indices   = []
    class_to_idxs: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_idxs.setdefault(label, []).append(idx)
    for label, idxs in class_to_idxs.items():
        chosen = rng.choice(idxs, size=min(per_class, len(idxs)), replace=False)
        indices.extend(chosen.tolist())

    print(f"  Train subset: {len(indices)} images ({per_class} per class)")
    return DataLoader(
        Subset(dataset, indices), batch_size=64, shuffle=False,
        num_workers=0, pin_memory=False,
    ), dataset, indices


def run_all(n_train: int, device: str, resume: dict | None = None) -> dict:
    results = dict(resume) if resume else {}

    for run_id, alpha, seed in RUNS:
        if run_id in results:
            print(f"  SKIP (cached): {run_id}")
            continue
        print(f"\n{'─'*55}")
        print(f"  {run_id}  (α={alpha}, seed={seed})")
        print(f"{'─'*55}")

        ckpt_path = CKPT_DIR / run_id / "checkpoint_best.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found")
            continue

        # ── Load model ────────────────────────────────────────────────────────
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = timm.create_model("vit_small_patch16_224", num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        # ── Build train loader ────────────────────────────────────────────────
        train_loader, train_dataset, train_indices = get_train_loader(n_train, seed)

        # ── Extract activations (no caching — train set is large) ─────────────
        cache = ACT_DIR / f"{run_id}_layer{MIDDLE_LAYER}_train{n_train}.pt"
        print(f"  Extracting activations (layer {MIDDLE_LAYER}, {n_train} images)...")
        acts, labels = extract_layer_activations(model, train_loader, MIDDLE_LAYER, device, cache)
        print(f"  Acts shape: {acts.shape}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ── Train SAE ─────────────────────────────────────────────────────────
        print(f"  Training SAE ({SAE_CFG['epochs']} epochs, L1={SAE_CFG['l1_penalty']})...")
        split      = int(0.8 * len(acts))
        train_acts = acts[:split].to(device)
        val_acts   = acts[split:].to(device)
        sae_out    = train_sae(train_acts, SAE_CFG, val_acts, device)
        sae: SparseAutoencoder = sae_out["model"]
        sae.eval()

        h2_metrics = sae_out["metrics"]
        print(f"  SAE L0={h2_metrics['l0_norm']:.1f}  "
              f"dead={h2_metrics['dead_feature_fraction']*100:.1f}%")

        # ── Per-feature selectivity ───────────────────────────────────────────
        print(f"  Computing SAE feature selectivity...")
        with torch.no_grad():
            hidden = sae.encode(acts.to(device)).cpu()  # (N, M)

        selectivity = compute_feature_selectivity(hidden, labels, NUM_CLASSES)
        scores      = feature_monosemanticity_scores(selectivity)
        summary     = sae_monosemanticity_summary(scores)

        print(f"  Feature mono mean={summary['mean']:.4f}  "
              f"frac>0.5={summary['frac_gt_0.5']:.4f}")

        results[run_id] = {
            "alpha":       alpha,
            "seed":        seed,
            "n_train":     len(acts),
            "sae_l0":      h2_metrics["l0_norm"],
            "sae_dead_pct": h2_metrics["dead_feature_fraction"] * 100,
            "feature_mono": summary,
            "scores_sample": scores[::4].tolist(),
        }

        del sae, sae_out, hidden, acts, selectivity, scores
        torch.cuda.empty_cache()
        gc.collect()

    return results


def aggregate_by_alpha(results: dict) -> dict:
    by_alpha: dict[float, list] = {}
    for r in results.values():
        by_alpha.setdefault(r["alpha"], []).append(r)
    return by_alpha


def print_table(by_alpha: dict):
    print("\n── EXP_012b: SAE Feature Monosemanticity (large eval) ────────────────")
    print(f"{'α':<6} {'n':>3}  {'Mean M_u':>10} {'Frac>0.3':>10} {'Frac>0.5':>10} {'SAE L0':>8}")
    print("─" * 55)
    for alpha in sorted(by_alpha):
        runs  = by_alpha[alpha]
        means  = [r["feature_mono"]["mean"]        for r in runs]
        frac03 = [r["feature_mono"]["frac_gt_0.3"] for r in runs]
        frac05 = [r["feature_mono"]["frac_gt_0.5"] for r in runs]
        l0s    = [r["sae_l0"]                       for r in runs]
        print(f"{alpha:<6} {len(runs):>3}  "
              f"{np.mean(means):>10.4f} "
              f"{np.mean(frac03):>10.4f} "
              f"{np.mean(frac05):>10.4f} "
              f"{np.mean(l0s):>8.1f}")


def plot_results(by_alpha: dict):
    alphas = sorted(by_alpha.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("EXP_012b: SAE Feature Monosemanticity (25k train images)",
                 fontsize=12, fontweight="bold")

    metrics = [
        ("mean",        "Mean feature M_u",      "Higher = more monosemantic"),
        ("frac_gt_0.3", "Frac features M_u>0.3", ""),
        ("frac_gt_0.5", "Frac features M_u>0.5", ""),
    ]

    x = np.arange(len(alphas))
    for ax, (key, title, sub) in zip(axes, metrics):
        for i, alpha in enumerate(alphas):
            vals = [r["feature_mono"][key] for r in by_alpha[alpha]]
            mean, std = np.mean(vals), np.std(vals)
            ax.bar(i, mean, yerr=std, color=ALPHA_COLORS[alpha], alpha=0.85,
                   width=0.5, capsize=5, ecolor="black")
            ax.text(i, mean + std + 0.001, f"{mean:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas])
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp012b_sae_monosemanticity_large.png"
    plt.savefig(out)
    plt.close()
    print(f"\n  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=25000,
                        help="Number of training images to use (default: 25000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use n_train=500 for a quick smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.n_train = 500

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Train images: {args.n_train}")

    out = JSON_DIR / "exp012b_sae_monosemanticity_large.json"
    resume = json.loads(out.read_text()) if out.exists() else {}
    # Only resume if existing file has the right n_train (not dry-run results)
    if resume and any(v.get("n_train", 0) < 1000 for v in resume.values()):
        print("  Ignoring stale dry-run cache.")
        resume = {}

    results  = run_all(args.n_train, device, resume=resume)
    by_alpha = aggregate_by_alpha(results)

    print_table(by_alpha)
    plot_results(by_alpha)

    out = JSON_DIR / "exp012b_sae_monosemanticity_large.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results → {out}")
    print("Done.")


if __name__ == "__main__":
    main()

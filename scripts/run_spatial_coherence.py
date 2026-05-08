#!/usr/bin/env python3
"""EXP_019: Spatial-coherence analysis for top-k class-specialized units (change 1b).

For each (class, seed, alpha), compute mean pairwise grid distance of the
top-k=16 neurons most activated by that class.  The grid is the 16×24
cortical sheet used by TopoLoss for the 384-dim attn.proj output.

Predicted result: distance decreases monotonically with alpha, confirming
that top-activated units become spatially co-localized under TopoLoss.

Usage:
    python scripts/run_spatial_coherence.py [--n-classes 20] [--dry-run]

Reads:  ~/topo/topo_checkpoints/{run_id}/checkpoint_best.pt
        ~/topo/data/imagenet-100/val/
Writes: results/json/spatial_coherence.json
        results/figures/spatial_coherence.png
"""
import argparse
import gc
import json
import sys
from itertools import combinations
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
from src.analysis.monosemanticity import compute_class_selectivity
from src.data.imagenet import get_imagenet_transforms

CKPT_DIR     = ROOT.parent / "topo" / "topo_checkpoints"
DATA_DIR     = ROOT.parent / "topo" / "data" / "imagenet-100"
JSON_DIR     = ROOT / "results" / "json"
FIGS_DIR     = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 100
IMAGE_SIZE   = 224
EMBED_DIM    = 384   # ViT-S/16 residual stream dimension
GRID_H       = 16    # find_cortical_sheet_size(384) → height=16, width=24
GRID_W       = 24
NUM_UNITS    = 16    # same k as H3

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

ALPHA_LABELS = {0.0: "Baseline (α=0)", 0.1: "α=0.1", 1.0: "α=1.0"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.1: "#DD8452", 1.0: "#55A868"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def unit_to_grid(unit_idx: int) -> tuple[int, int]:
    """Map flat unit index to (row, col) on the 16×24 cortical sheet."""
    return unit_idx // GRID_W, unit_idx % GRID_W


def mean_pairwise_grid_distance(unit_indices: list[int]) -> float:
    """Mean Euclidean pairwise distance on the 16×24 grid."""
    if len(unit_indices) < 2:
        return 0.0
    coords = np.array([unit_to_grid(i) for i in unit_indices], dtype=float)
    dists = [
        np.sqrt(((coords[a] - coords[b]) ** 2).sum())
        for a, b in combinations(range(len(coords)), 2)
    ]
    return float(np.mean(dists))


def random_baseline_distance(n_units: int = NUM_UNITS, n_samples: int = 1000, seed: int = 0) -> float:
    """Expected pairwise distance for a random size-k set."""
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_samples):
        idxs = rng.choice(EMBED_DIM, size=n_units, replace=False).tolist()
        dists.append(mean_pairwise_grid_distance(idxs))
    return float(np.mean(dists))


def get_class_indices(dataset, class_idx: int) -> list[int]:
    return [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]


def load_model(run_id: str, device: str) -> torch.nn.Module:
    ckpt_path = CKPT_DIR / run_id / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = timm.create_model("vit_small_patch16_224", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)


def compute_coherence_for_run(run_id: str, val_dataset, n_classes: int, device: str, seed: int) -> dict:
    """Compute per-class spatial coherence for a single checkpoint."""
    model = load_model(run_id, device)

    rng = np.random.default_rng(seed)
    all_classes = rng.permutation(NUM_CLASSES).tolist()

    per_class: dict[int, float] = {}

    for class_idx in all_classes:
        if len(per_class) >= n_classes:
            break

        class_indices = get_class_indices(val_dataset, class_idx)
        if len(class_indices) < 10:
            continue

        probe_n      = min(50, len(class_indices))
        probe_loader = DataLoader(
            Subset(val_dataset, class_indices[:probe_n]),
            batch_size=16, shuffle=False, num_workers=2,
        )

        selectivity = compute_class_selectivity(
            model, probe_loader, num_classes=NUM_CLASSES, device=device
        )  # (384, 100)

        class_scores  = selectivity[:, class_idx]
        top_k_indices = class_scores.topk(NUM_UNITS).indices.tolist()
        dist          = mean_pairwise_grid_distance(top_k_indices)
        per_class[class_idx] = dist

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-classes", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.n_classes = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Classes per model: {args.n_classes}")

    val_dir   = DATA_DIR / "val"
    transform = get_imagenet_transforms(IMAGE_SIZE, is_training=False)
    val_ds    = datasets.ImageFolder(str(val_dir), transform=transform)

    rand_baseline = random_baseline_distance()
    print(f"\nRandom baseline mean pairwise distance: {rand_baseline:.3f}")

    all_results: dict[str, dict] = {}

    for run_id, alpha, seed in RUNS:
        print(f"\n{'─'*55}")
        print(f"  {run_id}  (α={alpha}, seed={seed})")
        print(f"{'─'*55}")

        try:
            per_class = compute_coherence_for_run(run_id, val_ds, args.n_classes, device, seed)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        dists = list(per_class.values())
        mean_d = float(np.mean(dists))
        std_d  = float(np.std(dists))
        print(f"  Mean grid distance: {mean_d:.3f} ± {std_d:.3f}  (random baseline: {rand_baseline:.3f})")

        all_results[run_id] = {
            "alpha":      alpha,
            "seed":       seed,
            "per_class":  {str(k): v for k, v in per_class.items()},
            "mean_dist":  mean_d,
            "std_dist":   std_d,
            "random_baseline": rand_baseline,
        }

    # ── Aggregate by alpha ────────────────────────────────────────────────────
    by_alpha: dict[float, list[float]] = {}
    for r in all_results.values():
        by_alpha.setdefault(r["alpha"], []).extend(r["per_class"].values())

    print("\n── EXP_019: Spatial Coherence Summary ────────────────────────────────")
    print(f"{'α':<6}  {'n':>4}  {'Mean dist':>10}  {'Std':>8}")
    print("─" * 35)
    for alpha in sorted(by_alpha):
        dists = by_alpha[alpha]
        print(f"{alpha:<6}  {len(dists):>4}  {np.mean(dists):>10.3f}  {np.std(dists):>8.3f}")
    print(f"{'random':<6}  {'—':>4}  {rand_baseline:>10.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = JSON_DIR / "spatial_coherence.json"
    out_data = {"results": all_results, "random_baseline": rand_baseline}
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results → {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4))
    alphas = sorted(by_alpha)
    means  = [float(np.mean(by_alpha[a])) for a in alphas]
    stds   = [float(np.std(by_alpha[a]))  for a in alphas]

    for i, (alpha, mean, std) in enumerate(zip(alphas, means, stds)):
        ax.bar(i, mean, yerr=std, color=ALPHA_COLORS[alpha], alpha=0.85,
               width=0.55, capsize=5, ecolor="black",
               label=ALPHA_LABELS[alpha])
        ax.text(i, mean + std + 0.05, f"{mean:.2f}", ha="center", fontsize=9)

    ax.axhline(rand_baseline, color="gray", linestyle="--", linewidth=1.2,
               label=f"Random baseline ({rand_baseline:.2f})")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas])
    ax.set_ylabel("Mean pairwise grid distance (lower = more spatially coherent)")
    ax.set_title("Spatial coherence of top-k class units (16×24 grid)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    out_fig = FIGS_DIR / "spatial_coherence.png"
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()
    print(f"  Figure → {out_fig}")
    print("Done.")


if __name__ == "__main__":
    main()

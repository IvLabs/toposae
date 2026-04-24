#!/usr/bin/env python3
"""EXP_012: SAE-feature-level monosemanticity (H1 rescue).

The entropy-based H1 score (neuron level) was null. This experiment
measures monosemanticity at the SAE feature level — each feature's
M_u = max(s_u)/sum(s_u) across classes — which is the metric used in
the Anthropic monosemanticity literature.

Goes in the paper appendix as the 'better H1 measure'.

Usage:
    python scripts/run_exp012_sae_monosemanticity.py

Reads:  ../topo/topo_checkpoints/{run_id}/checkpoint_best.pt
        ../topo/data/imagenet-100/val/
Writes: results/json/exp012_sae_monosemanticity.json
        results/figures/exp012_sae_monosemanticity.png
        results/data/activations/{run_id}_layer6_val.pt   (cache)
"""
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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

ALPHA_LABELS = {0.0: "Baseline\n(α=0)", 0.1: "TopoWeak\n(α=0.1)", 1.0: "TopoStrong\n(α=1.0)"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.1: "#C44E52", 1.0: "#CCB974"}

SAE_CFG = dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3, epochs=50, batch_size=256)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def get_val_loader():
    val_dir = DATA_DIR / "val"
    transform = get_imagenet_transforms(IMAGE_SIZE, is_training=False)
    dataset = datasets.ImageFolder(str(val_dir), transform=transform)
    return torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )


def run_all(device: str) -> dict:
    val_loader = get_val_loader()
    results = {}

    for run_id, alpha, seed in RUNS:
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

        # ── Extract / load activations ────────────────────────────────────────
        cache = ACT_DIR / f"{run_id}_layer{MIDDLE_LAYER}_val.pt"
        print(f"  Extracting activations (layer {MIDDLE_LAYER})...")
        acts, labels = extract_layer_activations(model, val_loader, MIDDLE_LAYER, device, cache)
        print(f"  Acts shape: {acts.shape}  (cached: {cache.exists()})")

        del model; torch.cuda.empty_cache(); gc.collect()

        # ── Train SAE ─────────────────────────────────────────────────────────
        print(f"  Training SAE ({SAE_CFG['epochs']} epochs)...")
        split = int(0.8 * len(acts))
        train_acts = acts[:split].to(device)
        val_acts   = acts[split:].to(device)
        sae_out    = train_sae(train_acts, SAE_CFG, val_acts, device)
        sae: SparseAutoencoder = sae_out["model"]
        sae.eval()

        # ── Compute per-feature class selectivity ────────────────────────────
        print(f"  Computing SAE feature selectivity...")
        with torch.no_grad():
            hidden = sae.encode(acts.to(device)).cpu()  # (N, M)

        selectivity = compute_feature_selectivity(hidden, labels, NUM_CLASSES)
        scores      = feature_monosemanticity_scores(selectivity)
        summary     = sae_monosemanticity_summary(scores)

        h2_metrics = sae_out["metrics"]
        print(f"  SAE L0={h2_metrics['l0_norm']:.1f}  dead={h2_metrics['dead_feature_fraction']*100:.1f}%")
        print(f"  Feature mono mean={summary['mean']:.4f}  frac>0.5={summary['frac_gt_0.5']:.4f}")

        results[run_id] = {
            "alpha": alpha, "seed": seed,
            "sae_l0": h2_metrics["l0_norm"],
            "sae_dead_pct": h2_metrics["dead_feature_fraction"] * 100,
            "feature_mono": summary,
            # Store score distribution for plotting (every 4th value to save space)
            "scores_sample": scores[::4].tolist(),
        }

        del sae, sae_out, hidden, acts, selectivity, scores
        torch.cuda.empty_cache(); gc.collect()

    return results


def aggregate_by_alpha(results: dict) -> dict:
    by_alpha: dict[float, list] = {}
    for r in results.values():
        by_alpha.setdefault(r["alpha"], []).append(r)
    return by_alpha


def print_table(by_alpha: dict):
    print("\n── EXP_012: SAE Feature Monosemanticity ──────────────────────────────")
    print(f"{'α':<6} {'n':>3}  {'Mean M_u':>10} {'Frac>0.3':>10} {'Frac>0.5':>10} {'SAE L0':>8}")
    print("─" * 55)
    for alpha in sorted(by_alpha):
        runs = by_alpha[alpha]
        means     = [r["feature_mono"]["mean"]         for r in runs]
        frac03    = [r["feature_mono"]["frac_gt_0.3"]  for r in runs]
        frac05    = [r["feature_mono"]["frac_gt_0.5"]  for r in runs]
        l0s       = [r["sae_l0"]                        for r in runs]
        print(f"{alpha:<6} {len(runs):>3}  "
              f"{np.mean(means):>10.4f} "
              f"{np.mean(frac03):>10.4f} "
              f"{np.mean(frac05):>10.4f} "
              f"{np.mean(l0s):>8.1f}")


def plot_results(by_alpha: dict):
    alphas = sorted(by_alpha.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EXP_012: SAE Feature Monosemanticity (Appendix H1)",
                 fontsize=12, fontweight="bold")

    metrics = [
        ("mean",        "Mean feature M_u",    "Higher = more monosemantic features"),
        ("frac_gt_0.3", "Frac features M_u>0.3", ""),
        ("frac_gt_0.5", "Frac features M_u>0.5", ""),
    ]

    x = np.arange(len(alphas))
    for ax, (key, title, sub) in zip(axes, metrics):
        for i, alpha in enumerate(alphas):
            vals = [r["feature_mono"][key] for r in by_alpha[alpha]]
            mean, std = np.mean(vals), np.std(vals)
            ax.bar(i, mean, yerr=std, color=ALPHA_COLORS[alpha], alpha=0.8,
                   width=0.5, capsize=5, ecolor="black")
            ax.text(i, mean + std + 0.001, f"{mean:.3f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas])
        ax.set_title(f"{title}\n{sub}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp012_sae_monosemanticity.png"
    plt.savefig(out)
    plt.close()
    print(f"\n  Saved → {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    results   = run_all(device)
    by_alpha  = aggregate_by_alpha(results)

    print_table(by_alpha)
    plot_results(by_alpha)

    out = JSON_DIR / "exp012_sae_monosemanticity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results → {out}")
    print("Done.")


if __name__ == "__main__":
    main()

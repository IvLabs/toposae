#!/usr/bin/env python3
"""EXP_016: H3 activation patching across multiple classes.

Runs patching for N_CLASSES classes per model instead of just class 0.
Produces n_seeds × N_CLASSES paired observations per variant, giving
proper statistical power for H3.

Usage:
    python scripts/run_exp016_multiclass_patching.py [--n-classes 20] [--dry-run]

Reads:  ../topo/topo_checkpoints/{run_id}/checkpoint_best.pt
        data/imagenet-100/val/
Writes: results/json/exp016_multiclass_patching.json
        results/figures/exp016_h3_multiclass.png
        (prints stats table to stdout)
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

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import timm
from scipy import stats as scipy_stats
from src.analysis.multiclass_patching import run_multiclass_patching
from src.data.imagenet import get_imagenet_transforms
from torchvision import datasets

CKPT_DIR   = ROOT.parent / "topo" / "topo_checkpoints"
JSON_DIR   = ROOT / "results" / "json"
FIGS_DIR   = ROOT / "results" / "figures"
DATA_DIR   = ROOT.parent / "topo" / "data" / "imagenet-100"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 100
IMAGE_SIZE   = 224
MIDDLE_LAYER = 6
NUM_UNITS    = 16

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

ALPHA_LABELS = {0.0: "Baseline (α=0)", 0.1: "TopoWeak (α=0.1)", 1.0: "TopoStrong (α=1.0)"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.1: "#C44E52", 1.0: "#CCB974"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_val_dataset():
    val_dir = DATA_DIR / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"ImageNet-100 val not found at {val_dir}")
    transform = get_imagenet_transforms(IMAGE_SIZE, is_training=False)
    return datasets.ImageFolder(str(val_dir), transform=transform)


def load_model(run_id: str, device: str) -> torch.nn.Module:
    ckpt_path = CKPT_DIR / run_id / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = timm.create_model("vit_small_patch16_224", num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def run_all(n_classes: int, device: str) -> dict:
    val_dataset = load_val_dataset()
    all_results = {}

    for run_id, alpha, seed in RUNS:
        print(f"\n{'─'*55}")
        print(f"  {run_id}  (α={alpha}, seed={seed})")
        print(f"{'─'*55}")

        try:
            model = load_model(run_id, device)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        per_class = run_multiclass_patching(
            model=model,
            val_dataset=val_dataset,
            layer_idx=MIDDLE_LAYER,
            num_units=NUM_UNITS,
            n_classes=n_classes,
            n_test_per_class=50,
            n_source=200,
            device=device,
            num_total_classes=NUM_CLASSES,
            seed=seed,
        )

        ratios = [v["ratio"] for v in per_class.values() if not np.isnan(v["ratio"])]
        print(f"  Classes evaluated: {len(per_class)}")
        print(f"  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
        print(f"  Median ratio: {np.median(ratios):.3f}")

        all_results[run_id] = {
            "alpha": alpha, "seed": seed,
            "per_class": {str(k): v for k, v in per_class.items()},
            "ratios": ratios,
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios)),
        }

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return all_results


def compute_stats(all_results: dict) -> list[dict]:
    """Paired t-test across n_seeds × n_classes obs per variant."""
    alphas = sorted({v["alpha"] for v in all_results.values()})
    baseline_alpha = 0.0
    stats_out = []

    for treat_alpha in [a for a in alphas if a != baseline_alpha]:
        # Pool ratios across seeds for baseline and treatment
        base_ratios = []
        treat_ratios = []

        seeds = sorted({v["seed"] for v in all_results.values()})
        for seed in seeds:
            b = next((v for v in all_results.values() if v["alpha"] == baseline_alpha and v["seed"] == seed), None)
            t = next((v for v in all_results.values() if v["alpha"] == treat_alpha  and v["seed"] == seed), None)
            if b is None or t is None:
                continue
            # Match by class index (same classes evaluated per seed)
            b_classes = set(b["per_class"].keys())
            t_classes = set(t["per_class"].keys())
            shared = sorted(b_classes & t_classes)
            for cls in shared:
                br = b["per_class"][cls]["ratio"]
                tr = t["per_class"][cls]["ratio"]
                if not (np.isnan(br) or np.isnan(tr)):
                    base_ratios.append(br)
                    treat_ratios.append(tr)

        if len(base_ratios) < 2:
            continue

        base_arr  = np.array(base_ratios)
        treat_arr = np.array(treat_ratios)
        diffs = treat_arr - base_arr
        t_stat, p_val = scipy_stats.ttest_1samp(diffs, 0)

        # Cohen's d on the differences
        d = diffs.mean() / (diffs.std(ddof=1) + 1e-12)

        # Bootstrap 95% CI
        rng = np.random.default_rng(0)
        boot = rng.choice(diffs, size=(10_000, len(diffs)), replace=True).mean(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

        stats_out.append({
            "comparison": f"α={baseline_alpha} vs α={treat_alpha}",
            "n_obs": len(base_ratios),
            "mean_diff": float(diffs.mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "ci_95": ci,
            "significant": p_val < 0.05,
        })

    return stats_out


def print_stats(stats_list: list[dict]):
    print("\n── EXP_016: H3 multi-class paired test ───────────────────────────────")
    header = f"{'Comparison':<28} {'n':>5} {'Δ mean':>8} {'t':>7} {'p':>8} {'d':>6}  95% CI"
    print(header)
    print("─" * len(header))
    for s in stats_list:
        sig = " *" if s["significant"] else "  "
        print(
            f"{s['comparison']:<28} {s['n_obs']:>5} {s['mean_diff']:>+8.3f} "
            f"{s['t_stat']:>7.3f} {s['p_value']:>8.4f} {s['cohens_d']:>6.2f}  "
            f"[{s['ci_95'][0]:+.3f}, {s['ci_95'][1]:+.3f}]{sig}"
        )


def plot_results(all_results: dict, stats_list: list[dict]):
    alphas = sorted({v["alpha"] for v in all_results.values()})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("EXP_016: H3 Patch Ratio across Classes (ViT-S/16, ImageNet-100)",
                 fontsize=12, fontweight="bold")

    # ── Left: per-seed boxplot of per-class ratios ────────────────────────────
    ax = axes[0]
    positions = np.arange(len(alphas)) * 2
    for i, alpha in enumerate(alphas):
        runs = [v for v in all_results.values() if v["alpha"] == alpha]
        all_ratios = []
        for r in runs:
            all_ratios.extend(r["ratios"])
        bp = ax.boxplot(all_ratios, positions=[positions[i]], widths=0.8,
                        patch_artist=True, notch=False,
                        boxprops=dict(facecolor=ALPHA_COLORS[alpha], alpha=0.7),
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker=".", markersize=4, alpha=0.5))

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="random baseline (ratio=1)")
    ax.set_xticks(positions)
    ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas], fontsize=9)
    ax.set_ylabel("Patch Ratio (cluster / random |Δlogit|)")
    ax.set_title("Distribution of per-class patch ratios\n(pooled across 3 seeds)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Right: mean ratio per alpha with CI from stats ────────────────────────
    ax = axes[1]
    for i, alpha in enumerate(alphas):
        runs = [v for v in all_results.values() if v["alpha"] == alpha]
        all_ratios = np.array([r for rv in runs for r in rv["ratios"]])
        mean = all_ratios.mean()
        se = all_ratios.std() / np.sqrt(len(all_ratios))
        ax.bar(i, mean, color=ALPHA_COLORS[alpha], alpha=0.8, width=0.5,
               yerr=1.96 * se, capsize=6, ecolor="black")
        ax.text(i, mean + 1.96 * se + 0.02, f"{mean:.2f}", ha="center", fontsize=9)

    # Annotate significance
    for s in stats_list:
        if s["significant"]:
            treat_alpha = float(s["comparison"].split("α=")[-1])
            idx = alphas.index(treat_alpha)
            ax.text(idx, ax.get_ylim()[1] * 0.95, "***" if s["p_value"] < 0.001
                    else "**" if s["p_value"] < 0.01 else "*",
                    ha="center", fontsize=13, color="red")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas], fontsize=9)
    ax.set_ylabel("Mean Patch Ratio ± 1.96 SE")
    ax.set_title("Mean patch ratio per variant\n(* = p<0.05 vs baseline)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp016_h3_multiclass.png"
    plt.savefig(out)
    plt.close()
    print(f"\n  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-classes", type=int, default=20,
                        help="Number of classes to test per model (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use n_classes=3 for a quick smoke test")
    args = parser.parse_args()

    if args.dry_run:
        args.n_classes = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Classes per model: {args.n_classes}")
    print(f"Expected obs per comparison: 3 seeds × {args.n_classes} classes = {3 * args.n_classes}")

    all_results = run_all(args.n_classes, device)

    out_path = JSON_DIR / "exp016_multiclass_patching.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Raw results → {out_path}")

    stats_list = compute_stats(all_results)
    print_stats(stats_list)

    # Append stats to JSON
    combined = {"results": all_results, "stats": stats_list}
    out_path.write_text(json.dumps(combined, indent=2, default=lambda o: bool(o) if isinstance(o, (bool, np.bool_)) else str(o)))

    plot_results(all_results, stats_list)
    print("\nDone.")


if __name__ == "__main__":
    main()

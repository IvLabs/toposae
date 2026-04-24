#!/usr/bin/env python3
"""EXP_011: PCA dimensionality control.

Tests whether the SAE L0 reduction is simply a dimensionality effect.
If PCA d_95 is flat across alpha but SAE L0 drops at alpha=1.0, we
show SAE captures something beyond raw dimensionality.

Uses cached activations from EXP_012 (run that first).

Usage:
    python scripts/run_exp011_pca.py

Reads:  results/data/activations/{run_id}_layer6_val.pt   (from EXP_012)
        results/json/*_analysis.json                       (for SAE L0)
Writes: results/figures/exp011_pca_control.png
        results/json/exp011_pca.json
"""
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

from src.analysis.feature_analysis import pca_effective_dim

ACT_DIR  = ROOT / "results" / "data" / "activations"
JSON_DIR = ROOT / "results" / "json"
FIGS_DIR = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

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

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_sae_l0(run_id: str) -> float | None:
    """Pull SAE L0 from the existing per-run analysis JSON."""
    p = JSON_DIR / f"{run_id}_analysis.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["h2"]["l0_norm"]


def run_all() -> dict:
    results = {}
    for run_id, alpha, seed in RUNS:
        cache = ACT_DIR / f"{run_id}_layer{MIDDLE_LAYER}_val.pt"
        if not cache.exists():
            print(f"  SKIP {run_id}: no cached activations (run EXP_012 first)")
            continue

        data = torch.load(cache, map_location="cpu", weights_only=True)
        acts = data["acts"]  # (N, D)

        print(f"  {run_id}: acts {acts.shape} → PCA...")
        pca = pca_effective_dim(acts)
        sae_l0 = load_sae_l0(run_id)

        print(f"    d_95={pca['d_95']}  SAE_L0={sae_l0:.1f}" if sae_l0 else
              f"    d_95={pca['d_95']}  SAE_L0=N/A")

        results[run_id] = {
            "alpha": alpha, "seed": seed,
            "d_95": pca["d_95"],
            "top10_var_frac": pca["top10_variance_frac"],
            "sae_l0": sae_l0,
        }

    return results


def aggregate_by_alpha(results: dict) -> dict:
    by_alpha: dict[float, list] = {}
    for r in results.values():
        by_alpha.setdefault(r["alpha"], []).append(r)
    return by_alpha


def print_table(by_alpha: dict):
    print("\n── EXP_011: PCA Dimensionality Control ───────────────────────────────")
    print(f"{'α':<6} {'n':>3}  {'d_95 (mean±std)':>20} {'SAE L0 (mean±std)':>22}  {'Δd_95':>7}  {'ΔL0':>7}")
    print("─" * 70)

    baseline_d95 = baseline_l0 = None
    for alpha in sorted(by_alpha):
        runs   = by_alpha[alpha]
        d95s   = [r["d_95"] for r in runs]
        l0s    = [r["sae_l0"] for r in runs if r["sae_l0"] is not None]

        if alpha == 0.0:
            baseline_d95 = np.mean(d95s)
            baseline_l0  = np.mean(l0s) if l0s else None

        delta_d = f"{np.mean(d95s) - baseline_d95:+.1f}" if baseline_d95 else "—"
        delta_l = f"{np.mean(l0s) - baseline_l0:+.1f}"   if (l0s and baseline_l0) else "—"

        print(f"{alpha:<6} {len(runs):>3}  "
              f"{np.mean(d95s):>8.1f} ± {np.std(d95s):>5.1f}   "
              f"{np.mean(l0s) if l0s else float('nan'):>8.1f} ± {np.std(l0s) if l0s else 0:>5.1f}   "
              f"{delta_d:>7}  {delta_l:>7}")


def plot_results(by_alpha: dict, results: dict):
    alphas = sorted(by_alpha.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EXP_011: PCA Dimensionality vs SAE L0 — Ruling Out the Dimensionality Confound",
                 fontsize=11, fontweight="bold")

    x = np.arange(len(alphas))

    # ── Left: d_95 per variant ────────────────────────────────────────────────
    ax = axes[0]
    for i, alpha in enumerate(alphas):
        vals = [r["d_95"] for r in by_alpha[alpha]]
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=ALPHA_COLORS[alpha],
               alpha=0.8, width=0.5, capsize=5, ecolor="black")
        ax.text(i, np.mean(vals) + np.std(vals) + 0.5, f"{np.mean(vals):.0f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas])
    ax.set_ylabel("PCA d_95 (components for 95% var)")
    ax.set_title("PCA dimensionality\n(if this drops with α, confound is real)")
    ax.grid(axis="y", alpha=0.3)

    # ── Middle: SAE L0 per variant ────────────────────────────────────────────
    ax = axes[1]
    for i, alpha in enumerate(alphas):
        vals = [r["sae_l0"] for r in by_alpha[alpha] if r["sae_l0"] is not None]
        if not vals:
            continue
        ax.bar(i, np.mean(vals), yerr=np.std(vals), color=ALPHA_COLORS[alpha],
               alpha=0.8, width=0.5, capsize=5, ecolor="black")
        ax.text(i, np.mean(vals) + np.std(vals) + 5, f"{np.mean(vals):.0f}",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([ALPHA_LABELS[a] for a in alphas])
    ax.set_ylabel("SAE L0 norm (active features/image)")
    ax.set_title("SAE L0 norm\n(this drops at α=1.0)")
    ax.grid(axis="y", alpha=0.3)

    # ── Right: scatter d_95 vs L0 (key diagnostic plot) ──────────────────────
    ax = axes[2]
    for run_id, r in results.items():
        if r["sae_l0"] is None:
            continue
        alpha = r["alpha"]
        ax.scatter(r["d_95"], r["sae_l0"],
                   color=ALPHA_COLORS.get(alpha, "gray"), s=80, zorder=3)

    # Add handles
    for alpha in alphas:
        ax.scatter([], [], color=ALPHA_COLORS[alpha], s=80,
                   label=ALPHA_LABELS[alpha].replace("\n", " "))
    ax.legend(fontsize=8)
    ax.set_xlabel("PCA d_95")
    ax.set_ylabel("SAE L0 norm")
    ax.set_title("d_95 vs L0 scatter\n(divergence = SAE beyond dimensionality)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp011_pca_control.png"
    plt.savefig(out)
    plt.close()
    print(f"\n  Saved → {out}")


def main():
    print("EXP_011: PCA dimensionality control")
    print("(requires cached activations from EXP_012)\n")

    results  = run_all()
    if not results:
        print("No results — run EXP_012 first to populate activation cache.")
        return

    by_alpha = aggregate_by_alpha(results)
    print_table(by_alpha)
    plot_results(by_alpha, results)

    out = JSON_DIR / "exp011_pca.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results → {out}")
    print("Done.")


if __name__ == "__main__":
    main()

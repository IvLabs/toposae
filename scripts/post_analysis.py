#!/usr/bin/env python3
"""Post-hoc analysis: multi-seed aggregation, alpha sweep plots, layer-wise plots.

Run after all training + analysis is done:
    python scripts/post_analysis.py

Reads from results/json/, writes to results/figures/ and results/RESULTS_FINAL.md.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
JSON_DIR = ROOT / "results" / "json"
FIGS_DIR = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_LABELS = {0.0: "Baseline", 0.01: "α=0.01", 0.1: "TopoWeak", 0.5: "α=0.5", 1.0: "TopoStrong"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.01: "#55A868", 0.1: "#C44E52", 0.5: "#8172B2", 1.0: "#CCB974"}
SEEDS = [42, 123, 456]
ALPHAS_MAIN = [0.0, 0.1, 1.0]
ALPHAS_SWEEP = [0.0, 0.01, 0.1, 0.5, 1.0]


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_all_results() -> dict:
    """Load all per-run analysis JSONs."""
    results = {}
    for p in JSON_DIR.glob("*_analysis.json"):
        run_id = p.stem.replace("_analysis", "")
        results[run_id] = json.loads(p.read_text())
    return results


def load_layerwise_results() -> dict:
    results = {}
    for p in JSON_DIR.glob("*_layerwise.json"):
        run_id = p.stem.replace("_layerwise", "")
        results[run_id] = json.loads(p.read_text())
    return results


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ─── Multi-seed Aggregation ───────────────────────────────────────────────────

def compute_multiseed_stats(results: dict) -> dict:
    """Group by alpha, compute mean ± std across seeds for H1/H2/H3."""
    by_alpha: dict[float, list] = {}
    for run_id, r in results.items():
        # Only include multi-seed runs (run_id ends with _sN)
        if "_s" not in run_id:
            continue
        alpha = r["alpha"]
        if alpha not in ALPHAS_MAIN:
            continue
        by_alpha.setdefault(alpha, []).append(r)

    stats = {}
    for alpha, runs in sorted(by_alpha.items()):
        def agg(keys):
            vals = [r]
            for k in keys:
                vals = [v[k] for v in vals]
            return vals

        def get_metric(key_path):
            parts = key_path.split(".")
            values = []
            for r in runs:
                v = r
                for p in parts:
                    v = v[p]
                values.append(v)
            return np.mean(values), np.std(values), values

        stats[alpha] = {
            "n_seeds": len(runs),
            "val_acc":        get_metric("val_acc"),
            "h1.mono_mean":   get_metric("h1.mono_mean"),
            "h1.frac_gt_05":  get_metric("h1.frac_gt_05"),
            "h2.l0_norm":     get_metric("h2.l0_norm"),
            "h2.dead_pct":    get_metric("h2.dead_pct"),
            "h2.recon_loss":  get_metric("h2.recon_loss"),
            "h3.patch_ratio": get_metric("h3.patch_ratio"),
        }
    return stats


def plot_multiseed_summary(stats: dict):
    if not stats:
        print("  No multi-seed data found — skipping multi-seed plot")
        return

    alphas = sorted(stats.keys())
    labels = [ALPHA_LABELS.get(a, str(a)) for a in alphas]
    colors = [ALPHA_COLORS.get(a, "gray") for a in alphas]
    x = np.arange(len(alphas))

    metrics = [
        ("h2.l0_norm",     "H2: SAE L0 Norm",          "Lower = sparser features"),
        ("h2.dead_pct",    "H2: Dead Features (%)",     "Higher = more wasted capacity"),
        ("h3.patch_ratio", "H3: Patch Ratio",           "Higher = cleaner causal isolation"),
        ("h1.mono_mean",   "H1: Monosemanticity Mean",  "Higher = more monosemantic"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Multi-seed Results (mean ± std, 3 seeds)", fontsize=13, fontweight="bold")

    for ax, (metric_key, title, subtitle) in zip(axes, metrics):
        means = [stats[a][metric_key][0] for a in alphas]
        stds  = [stats[a][metric_key][1] for a in alphas]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.6, ecolor="black")
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.001,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGS_DIR / "multiseed_summary.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ─── Alpha Sweep ──────────────────────────────────────────────────────────────

def get_sweep_data(results: dict) -> dict:
    """Collect seed=42 results across all 5 alpha values."""
    sweep = {}
    for run_id, r in results.items():
        if r["seed"] != 42:
            continue
        alpha = r["alpha"]
        if alpha in ALPHAS_SWEEP:
            sweep[alpha] = r
    return sweep


def plot_alpha_sweep(sweep: dict):
    if len(sweep) < 2:
        print("  Not enough α sweep data — skipping sweep plot")
        return

    alphas = sorted(sweep.keys())
    metrics = {
        "H2: L0 Norm":      [sweep[a]["h2"]["l0_norm"]    for a in alphas],
        "H2: Dead %":       [sweep[a]["h2"]["dead_pct"]   for a in alphas],
        "H3: Patch Ratio":  [sweep[a]["h3"]["patch_ratio"]for a in alphas],
        "H1: Mono Mean":    [sweep[a]["h1"]["mono_mean"]  for a in alphas],
        "Val Accuracy":     [sweep[a]["val_acc"]           for a in alphas],
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle("Alpha Sweep — ViT-S/16, ImageNet-100, seed=42", fontsize=12, fontweight="bold")

    for ax, (title, vals) in zip(axes, metrics.items()):
        ax.plot(alphas, vals, "o-", color="#C44E52", linewidth=2, markersize=8)
        for a, v in zip(alphas, vals):
            ax.annotate(f"{v:.3f}", (a, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("α (TopoLoss weight)")
        ax.set_xscale("symlog", linthresh=0.005)
        ax.set_xticks(alphas)
        ax.set_xticklabels([str(a) for a in alphas], rotation=30, ha="right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "alpha_sweep.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ─── Layer-wise Plot ─────────────────────────────────────────────────────────

def plot_layerwise(layerwise: dict):
    # Filter to seed=42 canonical runs
    seed42_runs = {rid: d for rid, d in layerwise.items() if d["seed"] == 42}
    if not seed42_runs:
        print("  No layerwise data found — skipping layer-wise plot")
        return

    # Sort by alpha
    sorted_runs = sorted(seed42_runs.items(), key=lambda x: x[1]["alpha"])

    metrics_to_plot = [
        ("l0_norm",    "SAE L0 Norm per Layer",    "Avg active features"),
        ("dead_pct",   "Dead Features % per Layer", "% never-active features"),
        ("recon_loss", "Recon Loss per Layer",      "SAE reconstruction MSE"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Layer-wise SAE Analysis — ViT-S/16, seed=42", fontsize=12, fontweight="bold")

    for ax, (metric_key, title, ylabel) in zip(axes, metrics_to_plot):
        for run_id, data in sorted_runs:
            alpha = data["alpha"]
            depth = data["depth"]
            layers = sorted(data["layers"].keys(), key=int)
            vals = [data["layers"][l][metric_key] for l in layers]
            layer_nums = [int(l) for l in layers]
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")
            color = ALPHA_COLORS.get(alpha, "gray")
            ax.plot(layer_nums, vals, "o-", label=label, color=color, linewidth=2, markersize=6)

        ax.set_title(f"{title}\n{ylabel}", fontsize=10)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(depth))

    plt.tight_layout()
    out = FIGS_DIR / "layerwise_sae.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ─── Markdown Report ─────────────────────────────────────────────────────────

def generate_final_report(results: dict, stats: dict, sweep: dict):
    import time
    lines = [
        "# Final Results — Topo Monosemanticity (ViT-S/16, ImageNet-100)",
        "",
        f"> Generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Multi-Seed Summary (mean ± std, seeds 42/123/456)",
        "",
        "| α | Val Acc | H1 Mono↑ | H2 L0↓ | H2 Dead↑ | H3 Ratio↑ |",
        "|---|---------|----------|--------|----------|-----------|",
    ]
    for alpha in sorted(stats.keys()):
        s = stats[alpha]
        lines.append(
            f"| {alpha} | "
            f"{s['val_acc'][0]:.4f}±{s['val_acc'][1]:.4f} | "
            f"{s['h1.mono_mean'][0]:.4f}±{s['h1.mono_mean'][1]:.4f} | "
            f"{s['h2.l0_norm'][0]:.1f}±{s['h2.l0_norm'][1]:.1f} | "
            f"{s['h2.dead_pct'][0]:.1f}±{s['h2.dead_pct'][1]:.1f} | "
            f"{s['h3.patch_ratio'][0]:.3f}±{s['h3.patch_ratio'][1]:.3f} |"
        )

    lines += ["", "## Alpha Sweep (seed=42)", "",
              "| α | Val Acc | H1 Mono | H2 L0 | H3 Ratio |",
              "|---|---------|---------|-------|----------|"]
    for alpha in sorted(sweep.keys()):
        r = sweep[alpha]
        lines.append(
            f"| {alpha} | {r['val_acc']:.4f} | {r['h1']['mono_mean']:.4f} | "
            f"{r['h2']['l0_norm']:.1f} | {r['h3']['patch_ratio']:.3f} |"
        )

    lines += [
        "", "## Figures", "",
        "- `results/figures/multiseed_summary.png` — H1/H2/H3 bar chart with error bars",
        "- `results/figures/alpha_sweep.png` — all 5 metrics vs α",
        "- `results/figures/layerwise_sae.png` — L0/dead/recon per layer",
        "", "---", "",
    ]

    out = ROOT / "results" / "RESULTS_FINAL.md"
    out.write_text("\n".join(lines))
    print(f"  Final report → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    setup_style()
    print("Loading results...")
    results   = load_all_results()
    layerwise = load_layerwise_results()
    print(f"  {len(results)} run results, {len(layerwise)} layerwise results")

    print("\nMulti-seed aggregation...")
    stats = compute_multiseed_stats(results)
    plot_multiseed_summary(stats)

    print("\nAlpha sweep...")
    sweep = get_sweep_data(results)
    plot_alpha_sweep(sweep)

    print("\nLayer-wise plot...")
    plot_layerwise(layerwise)

    print("\nGenerating final report...")
    generate_final_report(results, stats, sweep)

    print("\nDone. All outputs in results/figures/ and results/RESULTS_FINAL.md")


if __name__ == "__main__":
    main()

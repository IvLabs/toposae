#!/usr/bin/env python3
"""EXP_010 + EXP_015: Accuracy-L0 confound check and paired statistical tests.

Usage:
    python scripts/run_statistical_analysis.py

Reads:  results/json/*_analysis.json
Writes: results/figures/exp010_accuracy_l0.png
        results/figures/exp015_paired_stats.png
        results/json/statistical_analysis.json
        (prints stats table to stdout)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.analysis.statistical import (
    load_results, filter_main,
    accuracy_vs_l0_regression,
    paired_tests,
    results_to_dict,
    METRICS,
)

JSON_DIR  = ROOT / "results" / "json"
FIGS_DIR  = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_LABELS = {0.0: "Baseline\n(α=0)", 0.1: "TopoWeak\n(α=0.1)", 1.0: "TopoStrong\n(α=1.0)"}
ALPHA_COLORS = {0.0: "#4C72B0", 0.1: "#C44E52", 1.0: "#CCB974"}
SEED_MARKERS = {42: "o", 123: "s", 456: "^"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── EXP_010 plot ─────────────────────────────────────────────────────────────

def plot_accuracy_l0(records, reg):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "EXP_010: Accuracy vs L0 — ruling out the 'weaker model' confound",
        fontsize=12, fontweight="bold",
    )

    accs = np.array([r["val_acc"] for r in records])
    l0s  = np.array([r["h2"]["l0_norm"] for r in records])

    # ── Left: scatter + regression line ──────────────────────────────────────
    ax = axes[0]
    xs = np.linspace(accs.min() - 0.005, accs.max() + 0.005, 100)
    ax.plot(xs, reg.slope * xs + reg.intercept, "--", color="gray", linewidth=1.5,
            label=f"OLS (R²={reg.r_squared:.3f}, p={reg.p_value:.3f})")

    for r in records:
        alpha = r["alpha"]
        seed  = r["seed"]
        ax.scatter(r["val_acc"], r["h2"]["l0_norm"],
                   color=ALPHA_COLORS.get(alpha, "gray"),
                   marker=SEED_MARKERS.get(seed, "o"),
                   s=80, zorder=3,
                   label=f"{ALPHA_LABELS.get(alpha, alpha)} s{seed}")

    # Add a clean legend (one per alpha, deduplicated)
    seen = set()
    handles, labels_ = [], []
    for r in sorted(records, key=lambda x: x["alpha"]):
        key = r["alpha"]
        if key not in seen:
            seen.add(key)
            handles.append(plt.scatter([], [], color=ALPHA_COLORS.get(key, "gray"), s=80))
            labels_.append(ALPHA_LABELS.get(key, f"α={key}"))
    # Add regression line handle
    handles.append(plt.Line2D([0], [0], linestyle="--", color="gray"))
    labels_.append(f"OLS fit")
    ax.legend(handles, labels_, fontsize=8, loc="upper left")

    ax.set_xlabel("Validation Accuracy")
    ax.set_ylabel("SAE L0 Norm (layer 6)")
    ax.set_title("Scatter: accuracy vs L0\n(if confound, all points lie on the line)")
    ax.grid(alpha=0.3)

    # ── Right: residuals per variant ─────────────────────────────────────────
    ax = axes[1]
    by_alpha: dict[float, list] = {}
    for r in records:
        by_alpha.setdefault(r["alpha"], []).append(reg.residuals[r["run_id"]])

    alphas = sorted(by_alpha.keys())
    positions = np.arange(len(alphas))
    for i, alpha in enumerate(alphas):
        res = by_alpha[alpha]
        ax.scatter([i] * len(res), res, color=ALPHA_COLORS.get(alpha, "gray"),
                   s=80, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(res)] * 2, color=ALPHA_COLORS.get(alpha, "gray"),
                linewidth=2)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels([ALPHA_LABELS.get(a, str(a)) for a in alphas])
    ax.set_ylabel("L0 Residual (observed − predicted from accuracy)")
    ax.set_title("Residuals: negative = sparser than accuracy predicts\n"
                 "(topo_strong below zero → effect beyond accuracy)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp010_accuracy_l0.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ── EXP_015 plot ─────────────────────────────────────────────────────────────

def plot_paired_stats(tests):
    # Group by metric (3 metrics × 2 comparisons = 6 entries → 3 subplots, 2 bars each)
    metrics = list(dict.fromkeys(t.metric for t in tests))  # preserve order, dedupe
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    fig.suptitle("EXP_015: Paired t-tests — treatment vs baseline (n=3 seeds)",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        subset = [t for t in tests if t.metric == metric]
        x = np.arange(len(subset))
        colors = ["#C44E52", "#CCB974"]

        for i, test in enumerate(subset):
            mean_diff = test.mean_diff
            ci_lo = test.ci_low
            ci_hi = test.ci_high
            err_lo = mean_diff - ci_lo
            err_hi = ci_hi - mean_diff

            bar = ax.bar(i, mean_diff, color=colors[i], alpha=0.8, width=0.5,
                         yerr=[[err_lo], [err_hi]], capsize=6, ecolor="black")

            # Significance annotation
            sig = "***" if test.p_value < 0.001 else ("**" if test.p_value < 0.01
                  else ("*" if test.p_value < 0.05 else "ns"))
            top = ci_hi + abs(ci_hi - ci_lo) * 0.15
            ax.text(i, top, f"{sig}\nd={test.cohens_d:.2f}\np={test.p_value:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([t.comparison.replace(" vs ", "\nvs ") for t in subset], fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.set_ylabel("Mean diff (treatment − baseline)")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGS_DIR / "exp015_paired_stats.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out}")


# ── Stdout table ──────────────────────────────────────────────────────────────

def print_stats_table(tests):
    print("\n── EXP_015: Paired t-tests (n=3 seeds, baseline = α=0.0) ──────────────")
    header = f"{'Metric':<22} {'Comparison':<24} {'Δ mean':>8} {'t':>7} {'p':>8} {'d':>7} {'95% CI'}"
    print(header)
    print("─" * len(header))
    for t in tests:
        sig = " *" if t.p_value < 0.05 else "  "
        print(
            f"{t.metric:<22} {t.comparison:<24} "
            f"{t.mean_diff:>+8.2f} {t.t_stat:>7.3f} {t.p_value:>8.4f} "
            f"{t.cohens_d:>7.2f}  [{t.ci_low:+.2f}, {t.ci_high:+.2f}]{sig}"
        )

def print_regression_summary(reg):
    print("\n── EXP_010: L0 ~ val_acc regression ───────────────────────────────────")
    print(f"  slope={reg.slope:.1f}, intercept={reg.intercept:.1f}, "
          f"R²={reg.r_squared:.4f}, p={reg.p_value:.4f}")
    print("  Residuals (negative = sparser than accuracy alone predicts):")
    for run_id, res in sorted(reg.residuals.items()):
        print(f"    {run_id:<26} {res:+.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    all_records = load_results(JSON_DIR)
    records = filter_main(all_records)
    print(f"  {len(records)} runs (alpha ∈ {{0.0, 0.1, 1.0}}, seeds {{42, 123, 456}})")

    if len(records) < 3:
        print("ERROR: fewer than 3 runs found — check that *_analysis.json files exist in results/json/")
        sys.exit(1)

    print("\nEXP_010: accuracy vs L0 regression...")
    reg = accuracy_vs_l0_regression(records)
    plot_accuracy_l0(records, reg)
    print_regression_summary(reg)

    print("\nEXP_015: paired t-tests...")
    tests = paired_tests(records)
    plot_paired_stats(tests)
    print_stats_table(tests)

    out = JSON_DIR / "statistical_analysis.json"
    out.write_text(json.dumps(results_to_dict(reg, tests), indent=2))
    print(f"\n  Stats saved → {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate layer-wise SAE L0 figure from existing JSON data (change 2 in revision plan).

Reads:  results/json/{run_id}_layerwise.json  (already computed)
Writes: results/figures/layerwise_sae_l0.pdf
        results/figures/layerwise_sae_l0.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).parent.parent.resolve()
JSON_DIR = ROOT / "results" / "json"
FIGS_DIR = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Only s42 has layer-wise data; use it for the figure.
RUNS = [
    ("baseline_s42",    0.0, "Baseline (α=0)"),
    ("topo_weak_s42",   0.1, "α=0.1"),
    ("topo_strong_s42", 1.0, "α=1.0"),
]

COLORS = {0.0: "#4C72B0", 0.1: "#DD8452", 1.0: "#55A868"}
STYLES = {0.0: "-",        0.1: "--",       1.0: "-"}
WIDTHS = {0.0: 1.8,        0.1: 1.8,        1.0: 2.2}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})


def load_layerwise(run_id: str) -> dict | None:
    path = JSON_DIR / f"{run_id}_layerwise.json"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return None
    return json.loads(path.read_text())


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for run_id, alpha, label in RUNS:
        data = load_layerwise(run_id)
        if data is None:
            continue

        layers_dict = data["layers"]
        layers   = sorted(int(k) for k in layers_dict)
        l0s      = [layers_dict[str(l)]["l0_norm"]  for l in layers]
        dead_pct = [layers_dict[str(l)]["dead_pct"] for l in layers]

        kw = dict(color=COLORS[alpha], linestyle=STYLES[alpha],
                  linewidth=WIDTHS[alpha], marker="o", markersize=4, label=label)

        axes[0].plot(layers, l0s,      **kw)
        axes[1].plot(layers, dead_pct, **kw)

    # ── L0 panel ─────────────────────────────────────────────────────────────
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean active features (L0)")
    axes[0].set_title("SAE L0 per layer")
    axes[0].set_xticks(range(12))
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    # shade the region where effect first appears (layers 3-10)
    axes[0].axvspan(3, 10, alpha=0.06, color="#55A868", label="_nolegend_")
    axes[0].annotate("effect band\n(layers 3–10)", xy=(6.5, axes[0].get_ylim()[0]),
                     xytext=(6.5, axes[0].get_ylim()[0]),
                     fontsize=7, color="#55A868", ha="center")

    # ── Dead-feature panel ────────────────────────────────────────────────────
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Dead-feature fraction (%)")
    axes[1].set_title("Dead features per layer")
    axes[1].set_xticks(range(12))
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.suptitle("Layer-wise SAE analysis (ViT-S/16, seed=42)", fontsize=11, y=1.01)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = FIGS_DIR / f"layerwise_sae_l0.{ext}"
        plt.savefig(out)
        print(f"Saved → {out}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()

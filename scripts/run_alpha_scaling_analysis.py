#!/usr/bin/env python3
"""Alpha-scaling analysis: why does optimal α differ between TinyViT and ViT-S?

Hypothesis: α=1.0 exerts disproportionate pressure on smaller/narrower models
because weight magnitudes are smaller → the topo term dominates CE loss.

Zero-cost checks (no new training):
  1. Weight norm analysis — compare per-layer RMS weight norms across models.
     If ViT-S weights are larger, the same α exerts less *relative* pressure.
  2. Topo/CE loss ratio — from logged training history in ViT-S checkpoints.
     Estimates what α would produce equivalent relative pressure for TinyViT.
  3. Effective-α curve — if we rescale α by mean weight norm squared, does the
     optimal α collapse to the same value across models?

Reads:
    ../topo/topo_checkpoints/{run_id}/checkpoint_best.pt

Writes:
    results/json/alpha_scaling.json
    results/figures/alpha_scaling.png

Usage:
    python scripts/run_alpha_scaling_analysis.py
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
CKPT_DIR = ROOT.parent / "topo" / "topo_checkpoints"
JSON_DIR  = ROOT / "results" / "json"
FIGS_DIR  = ROOT / "results" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# ── Model specs ───────────────────────────────────────────────────────────────
RUNS = {
    "tinyvit_base":   {"path": "baseline_imagenet100",    "model": "tinyvit", "alpha": 0.0},
    "tinyvit_weak":   {"path": "topo_weak_imagenet100",   "model": "tinyvit", "alpha": 0.1},
    "tinyvit_strong": {"path": "topo_strong_imagenet100", "model": "tinyvit", "alpha": 1.0},
    "vits_base":      {"path": "baseline_s42",            "model": "vits",    "alpha": 0.0},
    "vits_weak":      {"path": "topo_weak_s42",           "model": "vits",    "alpha": 0.1},
    "vits_strong":    {"path": "topo_strong_s42",         "model": "vits",    "alpha": 1.0},
}

# Layers TopoLoss targets (layers='all' in both configs → all weight matrices
# in attention + MLP blocks, excluding norm/bias/embedding)
TOPO_LAYER_PATTERNS = [
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
]

MODEL_DIMS = {
    "tinyvit": {"width": 128, "depth": 4,  "heads": 4},
    "vits":    {"width": 384, "depth": 12, "heads": 6},
}


def rms_norm(tensor: torch.Tensor) -> float:
    """Root-mean-square weight magnitude (Frobenius / sqrt(n_elements))."""
    return (tensor.float().pow(2).mean().sqrt()).item()


def load_checkpoint(run_id: str) -> dict:
    path = CKPT_DIR / run_id / "checkpoint_best.pt"
    if not path.exists():
        print(f"  MISSING: {path}")
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_weight_norms(state_dict: dict) -> dict:
    """Per-layer RMS norm for all TopoLoss-targeted layers."""
    norms = {}
    for name, param in state_dict.items():
        if any(pat in name for pat in TOPO_LAYER_PATTERNS):
            norms[name] = rms_norm(param)
    return norms


def get_training_history(ckpt: dict) -> dict | None:
    """Extract topo_loss and train_loss arrays from checkpoint config metrics."""
    cfg = ckpt.get("config", {})
    metrics = cfg.get("_all_metrics", {})
    if not metrics:
        return None
    return {k: v for k, v in metrics.items() if isinstance(v, list)}


def analyse_topo_ce_ratio(history: dict) -> dict:
    """At convergence (last 10 epochs), compute mean topo_loss / mean_ce_loss."""
    topo = np.array(history.get("topo_loss", []))
    ce   = np.array(history.get("train_loss", []))
    if len(topo) == 0 or topo.sum() == 0:
        return {"topo_ce_ratio": None, "note": "baseline (α=0)"}
    n = min(10, len(topo))
    ratio = topo[-n:].mean() / ce[-n:].mean()
    return {
        "topo_ce_ratio_convergence": float(ratio),
        "mean_topo_last10": float(topo[-n:].mean()),
        "mean_ce_last10":   float(ce[-n:].mean()),
    }


def main():
    print("Alpha-scaling analysis: weight norms + topo/CE ratios\n")

    results = {}

    for run_name, meta in RUNS.items():
        print(f"── {run_name} (α={meta['alpha']}) ──")
        ckpt = load_checkpoint(meta["path"])
        if not ckpt:
            continue

        sd      = ckpt.get("model_state_dict", {})
        norms   = extract_weight_norms(sd)
        history = get_training_history(ckpt)
        ratio   = analyse_topo_ce_ratio(history) if history else {}

        mean_rms = float(np.mean(list(norms.values()))) if norms else None
        print(f"  Topo-layer RMS norm (mean): {mean_rms:.5f}" if mean_rms else "  No norms")
        if ratio.get("topo_ce_ratio_convergence"):
            print(f"  Topo/CE ratio at convergence: {ratio['topo_ce_ratio_convergence']:.4f}")

        results[run_name] = {
            "alpha":    meta["alpha"],
            "model":    meta["model"],
            "mean_rms_norm": mean_rms,
            "layer_norms":   norms,
            **ratio,
        }

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Weight norm summary ──────────────────────────────────────────")
    print(f"{'Run':<22} {'Model':<10} {'α':>5}  {'Mean RMS':>10}  {'Topo/CE':>10}")
    print("─" * 65)
    for name, r in results.items():
        rms   = f"{r['mean_rms_norm']:.5f}" if r["mean_rms_norm"] else "—"
        ratio = f"{r['topo_ce_ratio_convergence']:.4f}" if r.get("topo_ce_ratio_convergence") else "—"
        print(f"{name:<22} {r['model']:<10} {r['alpha']:>5}  {rms:>10}  {ratio:>10}")

    # ── Effective α analysis ──────────────────────────────────────────────────
    # If topo_loss ∝ mean_w² (squared weight magnitudes), then for the same
    # absolute topo term, α_eff = α * (mean_rms_target² / mean_rms_ref²)
    # where ref = ViT-S baseline.
    ref_rms = results.get("vits_base", {}).get("mean_rms_norm")
    if ref_rms:
        print(f"\n── Effective α (normalised to ViT-S weight scale) ──────────────")
        print(f"  Reference (ViT-S baseline) mean RMS: {ref_rms:.5f}")
        for name, r in results.items():
            rms = r.get("mean_rms_norm")
            if rms is None or r["alpha"] == 0.0:
                continue
            # Topo loss ∝ w², so if TinyViT RMS is smaller by factor k,
            # same α produces k² less absolute topo loss → need α/k² to match
            scale = (rms / ref_rms) ** 2
            alpha_eff = r["alpha"] * scale
            results[name]["alpha_eff_normalised"] = float(alpha_eff)
            print(f"  {name:<22} α={r['alpha']}  RMS={rms:.5f}  "
                  f"scale={scale:.3f}  α_eff={alpha_eff:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Alpha-Scaling Analysis: Why Optimal α Differs Across Model Sizes",
                 fontsize=11, fontweight="bold")

    colors = {"tinyvit": "#C44E52", "vits": "#4C72B0"}
    markers = {0.0: "o", 0.1: "s", 1.0: "^"}

    # Left: mean RMS norm by model × alpha
    ax = axes[0]
    for model_type in ["tinyvit", "vits"]:
        xs, ys, labs = [], [], []
        for name, r in results.items():
            if r["model"] == model_type and r.get("mean_rms_norm"):
                xs.append(r["alpha"])
                ys.append(r["mean_rms_norm"])
        ax.plot(xs, ys, "o-", color=colors[model_type], label=model_type, linewidth=2, markersize=8)
    ax.set_xlabel("α")
    ax.set_ylabel("Mean RMS weight norm")
    ax.set_title("Weight norms shrink under topo pressure")
    ax.legend()
    ax.grid(alpha=0.3)

    # Middle: topo/CE ratio at convergence (ViT-S only — TinyViT history unavailable)
    ax = axes[1]
    vits_runs = [(r["alpha"], r.get("topo_ce_ratio_convergence"))
                 for r in results.values()
                 if r["model"] == "vits" and r.get("topo_ce_ratio_convergence")]
    if vits_runs:
        alphas_v, ratios_v = zip(*sorted(vits_runs))
        ax.bar(range(len(alphas_v)), ratios_v, color=colors["vits"], alpha=0.8, width=0.5)
        ax.set_xticks(range(len(alphas_v)))
        ax.set_xticklabels([f"α={a}" for a in alphas_v])
        ax.set_ylabel("Topo loss / CE loss (convergence)")
        ax.set_title("Topo/CE ratio at convergence\n(ViT-S only — history logged)")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No topo history\navailable", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_title("Topo/CE ratio (unavailable)")

    # Right: effective α after weight-norm normalisation
    ax = axes[2]
    for model_type in ["tinyvit", "vits"]:
        xs, ys = [], []
        for name, r in sorted(results.items(), key=lambda x: x[1]["alpha"]):
            if r["model"] == model_type and "alpha_eff_normalised" in r:
                xs.append(r["alpha"])
                ys.append(r["alpha_eff_normalised"])
        if xs:
            ax.plot(xs, ys, "o--", color=colors[model_type], label=model_type, linewidth=2, markersize=8)
    ax.plot([0, 1.05], [0, 1.05], "k:", linewidth=1, label="α_eff = α (ViT-S ref)")
    ax.set_xlabel("Nominal α")
    ax.set_ylabel("α_eff (normalised to ViT-S scale)")
    ax.set_title("Effective pressure after norm correction\n(if TinyViT line >> ViT-S → over-pressured)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_fig = FIGS_DIR / "alpha_scaling.png"
    plt.savefig(out_fig)
    plt.close()
    print(f"\n  Saved → {out_fig}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = JSON_DIR / "alpha_scaling.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  Saved → {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()

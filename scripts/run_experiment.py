#!/usr/bin/env python
"""Master experiment runner: Train + Analyze + Report.

Usage:
    # Quick: TinyViT, 3 variants, 1 seed, 50 epochs
    python scripts/run_experiment.py --config configs/exp_001_ultra_minimal.yaml --epochs 50

    # Full: ViT-S/16, 3 variants, 1 seed, 90 epochs
    python scripts/run_experiment.py --config configs/vit_s16.yaml --epochs 90

    # Multi-seed: ViT-S/16, 3 variants, 5 seeds, 90 epochs each
    python scripts/run_experiment.py --config configs/vit_s16.yaml --seeds 42 123 456 789 1024 --epochs 90

    # Analysis only (skip training, use existing checkpoints)
    python scripts/run_experiment.py --config configs/vit_s16.yaml --skip-training
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import scipy.stats as stats

sys.path.insert(0, ".")

from src.utils.config import load_config, merge_configs, save_config
from src.models.tiny_vit import TinyViT
from src.data.imagenet import get_dataloaders
from src.analysis.monosemanticity import compute_monosemanticity_scores, compute_class_selectivity
from src.analysis.sae import train_sae, collect_residual_stream
from src.analysis.patching import identify_selective_cluster, run_patching_experiment
from src.utils.visualization import (
    plot_monosemanticity_distribution,
    plot_comparison_bar_chart,
    plot_training_curves,
)

# ─── Model Factory ───────────────────────────────────────────────────────────

def create_model(config, device):
    """Create model from config. Supports TinyViT and timm models."""
    model_cfg = config["model"]
    mtype = model_cfg["type"]
    
    if mtype == "tiny_vit":
        model = TinyViT(
            num_classes=config["data"]["num_classes"],
            depth=model_cfg.get("depth", 4),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_heads=model_cfg.get("num_heads", 4),
            patch_size=model_cfg.get("patch_size", 16),
            image_size=config["data"]["image_size"],
            dropout=model_cfg.get("dropout", 0.1),
        )
    elif mtype in ("vit_s_16", "vit_b_16", "vit_l_16"):
        import timm
        drop_path = model_cfg.get("drop_path_rate", 0.1)
        pretrained = model_cfg.get("pretrained", False)
        model = timm.create_model(
            mtype, pretrained=pretrained,
            num_classes=config["data"]["num_classes"],
            drop_path_rate=drop_path,
            img_size=config["data"]["image_size"],
        )
        # Ensure get_attention_proj_layers exists
        if not hasattr(model, "get_attention_proj_layers"):
            model.get_attention_proj_layers = lambda: {
                f"block_{i}_attn_proj": model.blocks[i].attn.proj
                for i in range(len(model.blocks))
            }
    else:
        import timm
        model = timm.create_model(
            mtype, pretrained=model_cfg.get("pretrained", False),
            num_classes=config["data"]["num_classes"],
            img_size=config["data"]["image_size"],
        )
        if not hasattr(model, "get_attention_proj_layers"):
            model.get_attention_proj_layers = lambda: {
                f"block_{i}_attn_proj": model.blocks[i].attn.proj
                for i in range(len(model.blocks))
            }
    
    return model.to(device)


def get_model_depth(config):
    """Get number of transformer blocks."""
    mtype = config["model"]["type"]
    if mtype == "tiny_vit":
        return config["model"].get("depth", 4)
    elif mtype == "vit_s_16":
        return 12
    elif mtype == "vit_b_16":
        return 12
    elif mtype == "vit_l_16":
        return 24
    else:
        import timm
        # Try to create a dummy model to get depth
        m = timm.create_model(mtype, pretrained=False, num_classes=10)
        return len(m.blocks) if hasattr(m, "blocks") else 4


# ─── Training ────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_topo_loss(model, config):
    alpha = config["experiment"].get("alpha", 0.0)
    if alpha == 0.0:
        return None, 0.0
    try:
        from topoloss import TopoLoss, LaplacianPyramid
        proj_layers = model.get_attention_proj_layers()
        topo_loss = TopoLoss(
            losses=[
                LaplacianPyramid.from_layer(
                    model=model, layer=layer,
                    factor_h=8.0, factor_w=8.0,
                    scale=config["topo_loss"].get("sigma", 2.0),
                )
                for layer in proj_layers.values()
            ]
        )
        return topo_loss, alpha
    except ImportError:
        print("  Warning: topoloss not installed. TopoLoss disabled.")
        return None, 0.0


def train_epoch(model, train_loader, optimizer, topo_loss, alpha,
                accumulation_steps, device, scaler, epoch, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_topo_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    ):
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss_ce = torch.nn.functional.cross_entropy(outputs, labels) / accumulation_steps
        scaler.scale(loss_ce).backward()
        total_loss += loss_ce.item() * accumulation_steps

        if (batch_idx + 1) % accumulation_steps == 0:
            if topo_loss is not None and alpha > 0:
                loss_topo = topo_loss.compute(model=model)
                scaler.scale(alpha * loss_topo).backward()
                total_topo_loss += loss_topo.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "train_loss": total_loss / len(train_loader),
        "topo_loss": total_topo_loss / max(1, len(train_loader) // accumulation_steps),
        "train_acc": correct / total,
    }


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    correct = total = total_loss = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return {"val_loss": total_loss / len(val_loader), "val_acc": correct / total}


def save_checkpoint(model, optimizer, epoch, metrics, config, output_dir, is_best=False):
    run_id = config["experiment"]["run_id"]
    ckpt_dir = os.path.join(output_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }
    torch.save(checkpoint, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt"))
    if is_best:
        torch.save(checkpoint, os.path.join(ckpt_dir, "checkpoint_best.pt"))
    save_config(config, os.path.join(ckpt_dir, "config.yaml"))


def train_model(config, device):
    """Train a single model variant. Returns (all_metrics, best_acc)."""
    set_seed(config["experiment"]["seed"])
    print(f"\n  Device: {device}")

    print("  Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        num_classes=config["data"]["num_classes"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    print(f"  Creating model ({config['model']['type']})...")
    model = create_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    topo_loss, alpha = setup_topo_loss(model, config)
    print(f"  TopoLoss alpha: {alpha}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
        eta_min=config["training"]["lr_min"],
    )
    scaler = GradScaler()

    print(f"  Training for {config['training']['epochs']} epochs...")
    all_metrics = {"train_loss": [], "val_loss": [], "val_acc": [], "topo_loss": []}
    best_acc = 0
    t0 = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        te = time.time()
        train_m = train_epoch(
            model, train_loader, optimizer, topo_loss, alpha,
            config["data"]["accumulation_steps"], device, scaler, epoch,
            grad_clip=config["training"].get("grad_clip", 1.0),
        )
        val_m = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - te
        print(f"    Epoch {epoch}/{config['training']['epochs']} ({elapsed:.0f}s) "
              f"Train Loss: {train_m['train_loss']:.4f} "
              f"Val Acc: {val_m['val_acc']:.4f} "
              f"TopoLoss: {train_m['topo_loss']:.4f}")

        for k in all_metrics:
            all_metrics[k].append(train_m.get(k, val_m.get(k, 0)))

        is_best = val_m["val_acc"] > best_acc
        if is_best:
            best_acc = val_m["val_acc"]
        if epoch % config["training"]["checkpoint_every"] == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_m, config,
                          config["output"]["checkpoints_dir"], is_best)

    total_time = time.time() - t0
    print(f"  Done! Best val acc: {best_acc:.4f} ({total_time/60:.1f} min)")

    # Plot training curves
    run_id = config["experiment"]["run_id"]
    figures_dir = config["output"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    plot_training_curves(all_metrics,
                        os.path.join(figures_dir, f"{run_id}_training_curves.png"),
                        title=f"Training Curves - {run_id}")

    return all_metrics, best_acc


# ─── Analysis ────────────────────────────────────────────────────────────────

def run_analysis(run_id, config, device):
    """Run H1/H2/H3 analysis on a trained model checkpoint."""
    ckpt_path = f"{config['output']['checkpoints_dir']}/{run_id}/checkpoint_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  SKIP: {ckpt_path} not found")
        return None

    print(f"\n  Loading checkpoint: {run_id}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    
    model = create_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_classes = cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size"]
    mid_layer = get_model_depth(cfg) // 2

    train_loader, val_loader = get_dataloaders(
        cfg["data"]["data_dir"], num_classes, image_size,
        batch_size=32, num_workers=2,
    )

    label = run_id  # Use run_id as label
    print(f"  Model: {cfg['model']['type']}, Epoch: {ckpt['epoch']}, "
          f"Val Acc: {ckpt['metrics']['val_acc']:.4f}")

    # ─── H1: Monosemanticity ──────────────────────────────────────────────
    print(f"  [H1] Monosemanticity scores...")
    sel = compute_class_selectivity(model, val_loader, num_classes, device)
    scores = compute_monosemanticity_scores(sel)
    h1 = {
        "mono_mean": scores.mean().item(),
        "mono_median": scores.median().item(),
        "mono_max": scores.max().item(),
        "frac_gt_05": (scores > 0.5).float().mean().item(),
    }
    print(f"    Mean={h1['mono_mean']:.4f}, Median={h1['mono_median']:.4f}, "
          f"Max={h1['mono_max']:.4f}, Frac>0.5={h1['frac_gt_05']:.4f}")

    # ─── H2: SAE ──────────────────────────────────────────────────────────
    print(f"  [H2] Collecting activations (layer {mid_layer})...")
    train_acts = collect_residual_stream(model, train_loader, mid_layer, device)
    test_acts = collect_residual_stream(model, val_loader, mid_layer, device)
    print(f"    Train: {train_acts.shape}, Val: {test_acts.shape}")
    
    print(f"  [H2] Training SAE...")
    sae_cfg = dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3,
                   epochs=50, batch_size=256)
    sae_out = train_sae(train_acts, sae_cfg, test_acts, device)
    m = sae_out["metrics"]
    h2 = {
        "l0_norm": m["l0_norm"],
        "dead_pct": m["dead_feature_fraction"] * 100,
        "recon_loss": m["reconstruction_loss"],
    }
    print(f"    L0={h2['l0_norm']:.1f}, Dead={h2['dead_pct']:.1f}%, "
          f"Recon={h2['recon_loss']:.6f}")

    # ─── H3: Patching ─────────────────────────────────────────────────────
    print(f"  [H3] Activation patching...")
    cluster = identify_selective_cluster(model, train_loader, 0, num_units=16, device=device)
    pr = run_patching_experiment(model, val_loader, cluster, val_loader,
                                 layer_idx=mid_layer, device=device)
    td = pr["delta_logits"][:, 0].abs().mean().item()
    ru = torch.randperm(train_acts.shape[1])[:len(cluster)].tolist()
    rr = run_patching_experiment(model, val_loader, ru, val_loader,
                                  layer_idx=mid_layer, device=device)
    rd = rr["delta_logits"][:, 0].abs().mean().item()
    h3 = {
        "cluster_dlogit": td,
        "random_dlogit": rd,
        "patch_ratio": td / rd if rd > 0 else 0,
    }
    print(f"    Cluster={h3['cluster_dlogit']:.4f}, Random={h3['random_dlogit']:.4f}, "
          f"Ratio={h3['patch_ratio']:.2f}x")

    return {"h1": h1, "h2": h2, "h3": h3, "val_acc": ckpt["metrics"]["val_acc"],
            "epoch": ckpt["epoch"], "model_type": cfg["model"]["type"]}


# ─── Report Generation ───────────────────────────────────────────────────────

def generate_results_md(all_results, output_path="RESULTS.md"):
    """Generate RESULTS.md with all experiment results."""
    lines = [
        "# Experiment Results — Topo Monosemanticity Research",
        "",
        "> Auto-generated by `scripts/run_experiment.py`",
        "",
        "## Setup",
        "",
        "- **Dataset:** ImageNet-100 (clane9/imagenet-100, Hugging Face)",
        "- **Images:** 126,688 train / 5,000 val, 100 classes",
        "- **Models:** TinyViT (4L/128D), ViT-S/16 (12L), ViT-B/16 (12L)",
        "- **Training:** AdamW, cosine LR, mixed precision",
        "- **TopoLoss:** Official topoloss library (ICLR 2025 Spotlight)",
        "- **SAE:** 4× expansion, L1=0.001, 50 epochs",
        "- **Patching:** Top-16 most selective units for class 0",
        "",
    ]

    # Group results by model type
    model_types = {}
    for run_id, r in all_results.items():
        if r is None:
            continue
        mtype = r["model_type"]
        if mtype not in model_types:
            model_types[mtype] = []
        model_types[mtype].append((run_id, r))

    for mtype, entries in model_types.items():
        lines.append(f"## {mtype.upper()}")
        lines.append("")

        # Training table
        lines.append("### Training Results")
        lines.append("")
        lines.append("| Run | α | Epoch | Val Acc |")
        lines.append("|-----|---|-------|---------|")
        for run_id, r in entries:
            alpha = run_id.split("_")[-1] if "seed" not in run_id else "?"
            # Extract alpha from run_id pattern
            if "baseline" in run_id: alpha = "0.0"
            elif "topo_weak" in run_id: alpha = "0.1"
            elif "topo_strong" in run_id: alpha = "1.0"
            lines.append(f"| {run_id} | {alpha} | {r['epoch']} | {r['val_acc']:.4f} |")
        lines.append("")

        # H1 table
        lines.append("### H1: Monosemanticity")
        lines.append("")
        lines.append("| Run | Mean | Median | Max | Frac > 0.5 |")
        lines.append("|-----|------|--------|-----|------------|")
        for run_id, r in entries:
            h = r["h1"]
            lines.append(f"| {run_id} | {h['mono_mean']:.4f} | {h['mono_median']:.4f} | {h['mono_max']:.4f} | {h['frac_gt_05']:.4f} |")
        lines.append("")

        # H2 table
        lines.append("### H2: SAE")
        lines.append("")
        lines.append("| Run | L0 Norm | Dead % | Recon Loss |")
        lines.append("|-----|---------|--------|------------|")
        for run_id, r in entries:
            h = r["h2"]
            lines.append(f"| {run_id} | {h['l0_norm']:.1f} | {h['dead_pct']:.1f} | {h['recon_loss']:.6f} |")
        lines.append("")

        # H3 table
        lines.append("### H3: Causal Patching")
        lines.append("")
        lines.append("| Run | Cluster |Δlogit| | Random |Δlogit| | Ratio |")
        lines.append("|-----|----------|----------|-------|")
        for run_id, r in entries:
            h = r["h3"]
            lines.append(f"| {run_id} | {h['cluster_dlogit']:.4f} | {h['random_dlogit']:.4f} | {h['patch_ratio']:.2f}× |")
        lines.append("")

        # Key findings
        if len(entries) == 3:
            baseline = [e for e in entries if "baseline" in e[0]][0][1]
            weak = [e for e in entries if "topo_weak" in e[0]][0][1]
            strong = [e for e in entries if "topo_strong" in e[0]][0][1]
            
            lines.append("### Key Findings")
            lines.append("")
            mono_change = (weak["h1"]["mono_mean"] - baseline["h1"]["mono_mean"]) / baseline["h1"]["mono_mean"] * 100
            lines.append(f"- **H1:** TopoWeak mono_mean = {weak['h1']['mono_mean']:.4f} vs baseline {baseline['h1']['mono_mean']:.4f} ({mono_change:+.1f}%)")
            lines.append(f"- **H2:** TopoStrong L0 = {strong['h2']['l0_norm']:.1f} vs baseline {baseline['h2']['l0_norm']:.1f} ({(strong['h2']['l0_norm']-baseline['h2']['l0_norm'])/baseline['h2']['l0_norm']*100:+.1f}%)")
            lines.append(f"- **H3:** TopoWeak patch ratio = {weak['h3']['patch_ratio']:.2f}× vs baseline {baseline['h3']['patch_ratio']:.2f}×")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "*")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nResults written to {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Master experiment runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine runs to execute
    alphas = [(0.0, "baseline"), (0.1, "topo_weak"), (1.0, "topo_strong")]
    seeds = args.seeds or [base_config["experiment"].get("seed", 42)]

    run_configs = []
    for seed in seeds:
        for alpha, label in alphas:
            run_id = f"{label}_seed{seed}" if len(seeds) > 1 else label
            run_id = args.run_id if args.run_id and len(run_configs) == 0 else run_id
            
            cfg = load_config(args.config)
            cfg["experiment"]["alpha"] = alpha
            cfg["experiment"]["run_id"] = run_id
            cfg["experiment"]["seed"] = seed
            if args.epochs:
                cfg["training"]["epochs"] = args.epochs
            if args.alpha is not None:
                cfg["experiment"]["alpha"] = args.alpha
            if args.seed is not None:
                cfg["experiment"]["seed"] = args.seed
            if args.run_id is not None and len(run_configs) == 0:
                cfg["experiment"]["run_id"] = args.run_id
            
            run_configs.append(cfg)

    # Train
    all_results = {}
    for cfg in run_configs:
        run_id = cfg["experiment"]["run_id"]
        if args.skip_training:
            print(f"\nSkipping training for {run_id}")
        else:
            print(f"\n{'='*60}")
            print(f"  TRAINING: {run_id} (α={cfg['experiment']['alpha']}, seed={cfg['experiment']['seed']})")
            print(f"{'='*60}")
            _, best_acc = train_model(cfg, device)

    # Analyze
    if not args.skip_analysis:
        print(f"\n{'='*60}")
        print("  ANALYSIS")
        print(f"{'='*60}")
        
        for cfg in run_configs:
            run_id = cfg["experiment"]["run_id"]
            print(f"\n  Analyzing {run_id}...")
            result = run_analysis(run_id, cfg, device)
            all_results[run_id] = result

        # Generate plots
        print(f"\n  Generating figures...")
        figures_dir = base_config["output"]["figures_dir"]
        os.makedirs(figures_dir, exist_ok=True)

        h1_data = {}
        for run_id, r in all_results.items():
            if r and "h1" in r:
                scores = torch.tensor([r["h1"]["mono_mean"]])  # placeholder
                # We need actual score tensors for the distribution plot
                # Re-compute scores for plotting
                ckpt_path = f"{base_config['output']['checkpoints_dir']}/{run_id}/checkpoint_best.pt"
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    model = create_model(ckpt["config"], device)
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    _, val_loader = get_dataloaders(
                        ckpt["config"]["data"]["data_dir"],
                        ckpt["config"]["data"]["num_classes"],
                        ckpt["config"]["data"]["image_size"],
                        batch_size=32, num_workers=2,
                    )
                    sel = compute_class_selectivity(model, val_loader,
                        ckpt["config"]["data"]["num_classes"], device)
                    h1_data[run_id] = compute_monosemanticity_scores(sel)

        if h1_data:
            plot_monosemanticity_distribution(
                h1_data, os.path.join(figures_dir, f"{base_config['experiment']['name']}_monosemanticity.png"),
                f"Monosemanticity Scores - {base_config['experiment']['name']}")

        # Generate RESULTS.md
        generate_results_md(all_results)

    print(f"\n{'='*60}")
    print("  ALL DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Master experiment runner: Train + Analyze + Report.

Supports full resume from any checkpoint. All outputs go under /home/toposae/toposae/.
Logs are tee'd to logs/<run_id>.log automatically.

Usage:
    # ViT-S/16, 3 seeds, 90 epochs — full pipeline
    python scripts/run_experiment.py --config configs/vit_s16.yaml \\
        --seeds 42 123 456 --epochs 90

    # ViT-B/16, single seed
    python scripts/run_experiment.py --config configs/vit_b16.yaml --epochs 90

    # Analysis only (skip training, use existing checkpoints)
    python scripts/run_experiment.py --config configs/vit_s16.yaml --skip-training

    # Resume interrupted run
    python scripts/run_experiment.py --config configs/vit_s16.yaml \\
        --seeds 42 123 456 --epochs 90
        (same command — training auto-resumes from latest checkpoint)

    # Alpha sweep
    python scripts/run_experiment.py --config configs/vit_s16.yaml \\
        --alphas 0.0 0.01 0.1 0.5 1.0 --epochs 90
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, merge_configs, save_config
from src.models.tiny_vit import TinyViT
from src.data.imagenet import get_dataloaders
from src.analysis.monosemanticity import compute_monosemanticity_scores, compute_class_selectivity
from src.analysis.sae import train_sae, collect_residual_stream, collect_all_layers
from src.analysis.patching import identify_selective_cluster, run_patching_experiment
from src.utils.visualization import (
    plot_monosemanticity_distribution,
    plot_comparison_bar_chart,
    plot_training_curves,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Logging ─────────────────────────────────────────────────────────────────

_log_file = None

def setup_logging(log_path: Path):
    global _log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a", buffering=1)

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + "\n")

def close_logging():
    if _log_file:
        _log_file.close()


# ─── ntfy notifications ───────────────────────────────────────────────────────

def _fmt_mins(seconds: float) -> str:
    m = int(seconds / 60)
    return f"{m}m" if m < 60 else f"{m//60}h{m%60:02d}m"

def notify(title: str, msg: str, priority: str = "default", tags: str = ""):
    """Send a push notification via ntfy.sh. Reads NTFY_TOPIC from env.
    Silent no-op if topic not set or network fails — never breaks training.
    """
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        import urllib.request, json as _json
        payload = {"topic": topic, "title": title, "message": msg, "priority": priority}
        if tags:
            payload["tags"] = [t.strip() for t in tags.split(",")]
        req = urllib.request.Request(
            "https://ntfy.sh",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ─── Paths ───────────────────────────────────────────────────────────────────

def abs_path(rel: str) -> Path:
    """Resolve a relative path against the project root."""
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / p
    return p


# ─── Model Factory ───────────────────────────────────────────────────────────

def create_model(config):
    model_cfg = config["model"]
    mtype = model_cfg["type"]
    num_classes = config["data"]["num_classes"]
    image_size = config["data"]["image_size"]

    if mtype == "tiny_vit":
        model = TinyViT(
            num_classes=num_classes,
            depth=model_cfg.get("depth", 4),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_heads=model_cfg.get("num_heads", 4),
            patch_size=model_cfg.get("patch_size", 16),
            image_size=image_size,
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        import timm
        model = timm.create_model(
            mtype,
            pretrained=model_cfg.get("pretrained", False),
            num_classes=num_classes,
            drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
            img_size=image_size,
        )
        if not hasattr(model, "get_attention_proj_layers"):
            model.get_attention_proj_layers = lambda: {
                f"block_{i}_attn_proj": model.blocks[i].attn.proj
                for i in range(len(model.blocks))
            }
        if not hasattr(model, "blocks"):
            raise RuntimeError(f"timm model {mtype} has no .blocks attribute — unsupported")

    return model.to(DEVICE)


def get_model_depth(config) -> int:
    mtype = config["model"]["type"]
    if mtype == "tiny_vit":
        return config["model"].get("depth", 4)
    # For timm models, create and check
    import timm
    m = timm.create_model(mtype, pretrained=False, num_classes=2)
    depth = len(m.blocks) if hasattr(m, "blocks") else 12
    del m
    return depth


# ─── Seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── TopoLoss ────────────────────────────────────────────────────────────────

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
        log("  WARNING: topoloss not installed — TopoLoss disabled.")
        return None, 0.0


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def checkpoint_dir(config) -> Path:
    run_id = config["experiment"]["run_id"]
    return abs_path(config["output"]["checkpoints_dir"]) / run_id


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, config, is_best=False):
    ckpt_d = checkpoint_dir(config)
    ckpt_d.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }
    epoch_path = ckpt_d / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, epoch_path)
    if is_best:
        torch.save(state, ckpt_d / "checkpoint_best.pt")
    save_config(config, str(ckpt_d / "config.yaml"))

    # Keep only last 3 periodic checkpoints to save disk space
    all_epoch_ckpts = sorted(ckpt_d.glob("checkpoint_epoch_*.pt"))
    for old in all_epoch_ckpts[:-3]:
        old.unlink(missing_ok=True)


def find_latest_checkpoint(config) -> tuple[Path | None, int]:
    """Return (path, epoch) of most recent checkpoint, or (None, 0)."""
    ckpt_d = checkpoint_dir(config)
    if not ckpt_d.exists():
        return None, 0
    epoch_ckpts = sorted(ckpt_d.glob("checkpoint_epoch_*.pt"))
    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        epoch = int(latest.stem.split("_")[-1])
        return latest, epoch
    return None, 0


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler):
    log(f"  Resuming from {path}")
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])
    return state["epoch"], state.get("metrics", {}), state.get("all_metrics", {})


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, train_loader, optimizer, topo_loss, alpha,
                accumulation_steps, scaler, epoch, grad_clip=1.0):
    from tqdm import tqdm
    model.train()
    total_ce = total_topo = correct = total = 0
    optimizer.zero_grad()
    n_acc_steps = 0

    for batch_idx, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Ep{epoch}", leave=False, ncols=80)
    ):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with autocast("cuda"):
            outputs = model(images)
            loss_ce = F.cross_entropy(outputs, labels) / accumulation_steps

        scaler.scale(loss_ce).backward()
        total_ce += loss_ce.item() * accumulation_steps

        is_last_batch = (batch_idx + 1 == len(train_loader))
        if (batch_idx + 1) % accumulation_steps == 0 or is_last_batch:
            if topo_loss is not None and alpha > 0:
                loss_topo = topo_loss.compute(model=model)
                scaler.scale(alpha * loss_topo).backward()
                total_topo += loss_topo.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            n_acc_steps += 1

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "train_loss": total_ce / len(train_loader),
        "topo_loss": total_topo / max(1, n_acc_steps),
        "train_acc": correct / total,
    }


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    correct = total = total_loss = 0
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        total_loss += F.cross_entropy(outputs, labels).item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return {"val_loss": total_loss / len(val_loader), "val_acc": correct / total}


def train_model(config) -> tuple[dict, float]:
    """Train one model variant. Automatically resumes from latest checkpoint."""
    run_id = config["experiment"]["run_id"]
    seed = config["experiment"]["seed"]
    alpha = config["experiment"]["alpha"]
    epochs = config["training"]["epochs"]
    log(f"\n{'='*60}")
    log(f"  TRAIN: {run_id}  α={alpha}  seed={seed}  epochs={epochs}")
    log(f"{'='*60}")

    set_seed(seed)

    data_dir = str(abs_path(config["data"]["data_dir"]))
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        num_classes=config["data"]["num_classes"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"].get("num_workers", 4),
    )

    model = create_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {config['model']['type']}  params={n_params:,}  device={DEVICE}")

    topo_loss, alpha_val = setup_topo_loss(model, config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config["training"].get("lr_min", 1e-5),
    )
    scaler = GradScaler("cuda")

    # Resume support
    start_epoch = 0
    best_acc = 0.0
    all_metrics = {"train_loss": [], "val_loss": [], "val_acc": [], "topo_loss": []}

    latest_ckpt, latest_epoch = find_latest_checkpoint(config)
    if latest_ckpt:
        start_epoch, last_metrics, saved_all_metrics = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, scaler
        )
        best_acc = last_metrics.get("val_acc", 0.0)
        if saved_all_metrics:
            all_metrics = saved_all_metrics
        log(f"  Resumed at epoch {start_epoch}, best_acc={best_acc:.4f}")
    else:
        log(f"  Starting fresh")

    if start_epoch >= epochs:
        log(f"  Already completed ({start_epoch}/{epochs} epochs), skipping.")
        return all_metrics, best_acc

    if start_epoch == 0:
        notify(f"▶ Started: {run_id}",
               f"α={alpha}  seed={seed}  {epochs} epochs\nModel: {config['model']['type']}",
               tags="arrow_forward")
    else:
        notify(f"↩ Resumed: {run_id}",
               f"From epoch {start_epoch}/{epochs}  best_acc={best_acc:.4f}",
               tags="repeat")

    accumulation_steps = config["data"].get("accumulation_steps", 1)
    checkpoint_every = config["training"].get("checkpoint_every", 10)
    t0 = time.time()
    epoch_times = []
    notify_every = 10  # send ntfy every N epochs

    for epoch in range(start_epoch + 1, epochs + 1):
        te = time.time()
        train_m = train_epoch(
            model, train_loader, optimizer, topo_loss, alpha_val,
            accumulation_steps, scaler, epoch,
            grad_clip=config["training"].get("grad_clip", 1.0),
        )
        val_m = validate(model, val_loader)
        scheduler.step()

        epoch_elapsed = time.time() - te
        epoch_times.append(epoch_elapsed)
        remaining_epochs = epochs - epoch
        avg_epoch_t = sum(epoch_times[-5:]) / len(epoch_times[-5:])  # rolling 5-ep avg
        eta_s = avg_epoch_t * remaining_epochs

        log(f"  Ep {epoch:03d}/{epochs} ({epoch_elapsed:.0f}s) "
            f"loss={train_m['train_loss']:.4f} "
            f"topo={train_m['topo_loss']:.4f} "
            f"val_acc={val_m['val_acc']:.4f} "
            f"eta={_fmt_mins(eta_s)}")

        for k in ("train_loss", "topo_loss"):
            all_metrics[k].append(train_m[k])
        all_metrics["val_loss"].append(val_m["val_loss"])
        all_metrics["val_acc"].append(val_m["val_acc"])

        is_best = val_m["val_acc"] > best_acc
        if is_best:
            best_acc = val_m["val_acc"]

        # ntfy every N epochs
        if epoch % notify_every == 0 or epoch == epochs:
            notify(
                f"{'🏆 ' if is_best else ''}Ep {epoch}/{epochs} — {run_id}",
                f"val_acc={val_m['val_acc']:.4f}  loss={train_m['train_loss']:.4f}\n"
                f"ETA: {_fmt_mins(eta_s)}  best={best_acc:.4f}",
                tags="chart_with_upwards_trend",
            )

        # Save on schedule or when best
        if epoch % checkpoint_every == 0 or is_best or epoch == epochs:
            config["_all_metrics"] = all_metrics
            save_checkpoint(model, optimizer, scheduler, scaler,
                            epoch, val_m, config, is_best=is_best)
            config.pop("_all_metrics", None)

    total_time = time.time() - t0
    log(f"  Done: best_val_acc={best_acc:.4f}  total={total_time/60:.1f}min")
    notify(f"✅ Done: {run_id}",
           f"best_val_acc={best_acc:.4f}  total={_fmt_mins(total_time)}",
           priority="high", tags="white_check_mark")

    figs_dir = abs_path(config["output"]["figures_dir"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        all_metrics,
        str(figs_dir / f"{run_id}_training_curves.png"),
        title=f"Training — {run_id}",
    )
    return all_metrics, best_acc


# ─── Analysis ────────────────────────────────────────────────────────────────

def run_analysis(run_id: str, config: dict) -> dict | None:
    ckpt_path = checkpoint_dir(config) / "checkpoint_best.pt"
    if not ckpt_path.exists():
        log(f"  SKIP analysis: {ckpt_path} not found")
        return None

    log(f"\n{'='*60}")
    log(f"  ANALYZE: {run_id}")
    log(f"{'='*60}")

    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    cfg.pop("_all_metrics", None)

    model = create_model(cfg)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    log(f"  Checkpoint epoch={state['epoch']}  val_acc={state['metrics']['val_acc']:.4f}")

    num_classes = cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size"]
    depth = get_model_depth(cfg)
    mid_layer = depth // 2

    data_dir = str(abs_path(cfg["data"]["data_dir"]))
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        num_classes=num_classes,
        image_size=image_size,
        batch_size=64,
        num_workers=4,
    )

    # ── H1: Monosemanticity ──────────────────────────────────────────────────
    log(f"  [H1] Computing monosemanticity (norm layer CLS)...")
    sel = compute_class_selectivity(model, val_loader, num_classes, DEVICE)
    scores = compute_monosemanticity_scores(sel)
    h1 = {
        "mono_mean":   scores.mean().item(),
        "mono_median": scores.median().item(),
        "mono_max":    scores.max().item(),
        "frac_gt_05":  (scores > 0.5).float().mean().item(),
    }
    log(f"    mean={h1['mono_mean']:.4f} median={h1['mono_median']:.4f} "
        f"max={h1['mono_max']:.4f} frac>0.5={h1['frac_gt_05']:.4f}")

    # ── H2: SAE ──────────────────────────────────────────────────────────────
    log(f"  [H2] Collecting activations (layer {mid_layer} of {depth})...")
    train_acts = collect_residual_stream(model, train_loader, mid_layer, DEVICE)
    val_acts   = collect_residual_stream(model, val_loader,   mid_layer, DEVICE)
    log(f"    train={tuple(train_acts.shape)}  val={tuple(val_acts.shape)}")

    log(f"  [H2] Training SAE (4× expansion, L1=0.001, 50ep)...")
    sae_cfg = dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3, epochs=50, batch_size=512)
    sae_out = train_sae(train_acts, sae_cfg, val_acts, DEVICE)
    m = sae_out["metrics"]
    h2 = {
        "l0_norm":    m["l0_norm"],
        "dead_pct":   m["dead_feature_fraction"] * 100,
        "recon_loss": m["reconstruction_loss"],
    }
    log(f"    L0={h2['l0_norm']:.1f}  dead={h2['dead_pct']:.1f}%  recon={h2['recon_loss']:.6f}")

    # ── H3: Causal Patching ───────────────────────────────────────────────────
    log(f"  [H3] Activation patching (layer {mid_layer}, class 0, 16 units)...")
    cluster = identify_selective_cluster(
        model, train_loader, class_idx=0, num_units=16, device=DEVICE, num_classes=num_classes
    )
    pr_cluster = run_patching_experiment(
        model, val_loader, cluster, train_loader, layer_idx=mid_layer, device=DEVICE
    )
    cluster_d = pr_cluster["delta_logits"][:, 0].abs().mean().item()

    rng = torch.Generator()
    rng.manual_seed(0)
    random_units = torch.randperm(train_acts.shape[1], generator=rng)[:len(cluster)].tolist()
    pr_random = run_patching_experiment(
        model, val_loader, random_units, train_loader, layer_idx=mid_layer, device=DEVICE
    )
    random_d = pr_random["delta_logits"][:, 0].abs().mean().item()

    h3 = {
        "cluster_dlogit": cluster_d,
        "random_dlogit":  random_d,
        "patch_ratio":    cluster_d / random_d if random_d > 0 else 0.0,
    }
    log(f"    cluster={h3['cluster_dlogit']:.4f}  random={h3['random_dlogit']:.4f}  "
        f"ratio={h3['patch_ratio']:.2f}×")

    result = {
        "run_id":     run_id,
        "model_type": cfg["model"]["type"],
        "alpha":      cfg["experiment"]["alpha"],
        "seed":       cfg["experiment"]["seed"],
        "val_acc":    state["metrics"]["val_acc"],
        "epoch":      state["epoch"],
        "h1": h1,
        "h2": h2,
        "h3": h3,
    }

    # Save per-run JSON
    json_dir = ROOT / "results" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(json_dir / f"{run_id}_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─── Layer-wise Analysis ─────────────────────────────────────────────────────

def run_layerwise_analysis(run_id: str, config: dict) -> dict | None:
    """For each transformer layer: train SAE, record L0/dead/recon.

    Uses a single forward pass to collect all layers simultaneously.
    Results saved to results/json/<run_id>_layerwise.json.
    """
    ckpt_path = checkpoint_dir(config) / "checkpoint_best.pt"
    if not ckpt_path.exists():
        log(f"  SKIP layerwise: {ckpt_path} not found")
        return None

    log(f"\n  [LAYERWISE] {run_id}")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = state["config"]
    cfg.pop("_all_metrics", None)

    model = create_model(cfg)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    depth = get_model_depth(cfg)
    data_dir = str(abs_path(cfg["data"]["data_dir"]))

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        num_classes=cfg["data"]["num_classes"],
        image_size=cfg["data"]["image_size"],
        batch_size=128,
        num_workers=4,
    )

    log(f"  Collecting all {depth} layers in one pass (train + val)...")
    train_all = collect_all_layers(model, train_loader, depth, DEVICE)
    val_all   = collect_all_layers(model, val_loader,   depth, DEVICE)
    log(f"  Collection done. Training SAE per layer...")

    sae_cfg = dict(expansion_factor=4, l1_penalty=0.001, lr=1e-3, epochs=50, batch_size=512)
    layer_results = {}

    for layer_idx in range(depth):
        train_acts = train_all[layer_idx]
        val_acts   = val_all[layer_idx]
        sae_out = train_sae(train_acts, sae_cfg, val_acts, DEVICE)
        m = sae_out["metrics"]
        layer_results[layer_idx] = {
            "l0_norm":    m["l0_norm"],
            "dead_pct":   m["dead_feature_fraction"] * 100,
            "recon_loss": m["reconstruction_loss"],
        }
        log(f"    layer {layer_idx:02d}: L0={m['l0_norm']:.1f}  "
            f"dead={m['dead_feature_fraction']*100:.1f}%  recon={m['reconstruction_loss']:.6f}")

    result = {
        "run_id":     run_id,
        "alpha":      cfg["experiment"]["alpha"],
        "seed":       cfg["experiment"]["seed"],
        "depth":      depth,
        "layers":     layer_results,
    }
    json_dir = ROOT / "results" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    with open(json_dir / f"{run_id}_layerwise.json", "w") as f:
        json.dump(result, f, indent=2)
    log(f"  Layerwise saved → results/json/{run_id}_layerwise.json")
    notify(f"📊 Layerwise done: {run_id}", f"{depth} layers analyzed", tags="bar_chart")
    return result


# ─── Reporting ───────────────────────────────────────────────────────────────

def generate_results_md(all_results: dict, output_path: Path):
    lines = [
        "# Experiment Results — Topo Monosemanticity Research",
        "",
        f"> Auto-generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "- **Dataset:** ImageNet-100 (clane9/imagenet-100, HF)",
        "- **Models:** TinyViT (4L/128D), ViT-S/16 (12L/384D), ViT-B/16 (12L/768D)",
        "- **Training:** AdamW, cosine LR, mixed precision, grad clip 1.0",
        "- **SAE:** 4× expansion, L1=0.001, 50 epochs, batch=512",
        "- **Patching:** Top-16 selective units, class 0",
        "",
    ]

    # Group by model type
    by_model: dict[str, list] = {}
    for run_id, r in all_results.items():
        if r is None:
            continue
        by_model.setdefault(r["model_type"], []).append((run_id, r))

    for mtype, entries in by_model.items():
        lines += [f"## {mtype}", ""]

        # Training
        lines += ["### Training", "", "| Run | α | Seed | Epoch | Val Acc |",
                  "|-----|---|------|-------|---------|"]
        for rid, r in entries:
            lines.append(f"| {rid} | {r['alpha']} | {r['seed']} | {r['epoch']} | {r['val_acc']:.4f} |")
        lines.append("")

        # H1
        lines += ["### H1: Monosemanticity", "",
                  "| Run | Mean | Median | Max | Frac > 0.5 |",
                  "|-----|------|--------|-----|------------|"]
        for rid, r in entries:
            h = r["h1"]
            lines.append(f"| {rid} | {h['mono_mean']:.4f} | {h['mono_median']:.4f} | "
                         f"{h['mono_max']:.4f} | {h['frac_gt_05']:.4f} |")
        lines.append("")

        # H2
        lines += ["### H2: SAE", "",
                  "| Run | L0 Norm | Dead % | Recon Loss |",
                  "|-----|---------|--------|------------|"]
        for rid, r in entries:
            h = r["h2"]
            lines.append(f"| {rid} | {h['l0_norm']:.1f} | {h['dead_pct']:.1f} | {h['recon_loss']:.6f} |")
        lines.append("")

        # H3
        lines += ["### H3: Causal Patching", "",
                  "| Run | Cluster |Δlogit| | Random |Δlogit| | Ratio |",
                  "|-----|----------|----------|-------|"]
        for rid, r in entries:
            h = r["h3"]
            lines.append(f"| {rid} | {h['cluster_dlogit']:.4f} | {h['random_dlogit']:.4f} | "
                         f"{h['patch_ratio']:.2f}× |")
        lines.append("")

    lines += ["---", f"*Generated {time.strftime('%Y-%m-%d %H:%M:%S')}*", ""]
    output_path.write_text("\n".join(lines))
    log(f"  Results → {output_path}")


def generate_summary_json(all_results: dict, output_path: Path):
    output_path.write_text(json.dumps(all_results, indent=2))
    log(f"  Summary JSON → {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def build_run_configs(args, base_config: dict) -> list[dict]:
    """Build list of per-run configs from CLI args."""
    alphas_map = {
        0.0:  "baseline",
        0.1:  "topo_weak",
        1.0:  "topo_strong",
        0.01: "topo_001",
        0.5:  "topo_05",
    }

    alphas = args.alphas if args.alphas else [0.0, 0.1, 1.0]
    seeds  = args.seeds  if args.seeds  else [base_config["experiment"].get("seed", 42)]

    run_configs = []
    for seed in seeds:
        for alpha in alphas:
            label = alphas_map.get(alpha, f"alpha{alpha}")
            run_id = f"{label}_s{seed}"

            cfg = load_config(args.config)
            cfg["experiment"]["alpha"]  = alpha
            cfg["experiment"]["seed"]   = seed
            cfg["experiment"]["run_id"] = run_id
            if args.epochs:
                cfg["training"]["epochs"] = args.epochs

            # Force absolute data path
            cfg["data"]["data_dir"] = str(abs_path(cfg["data"]["data_dir"]))
            cfg["output"]["checkpoints_dir"] = str(ROOT / "results" / "data" / "checkpoints")
            cfg["output"]["figures_dir"]     = str(ROOT / "results" / "figures")

            run_configs.append(cfg)

    return run_configs


def main():
    parser = argparse.ArgumentParser(description="Topo monosemanticity experiment runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds",  type=int, nargs="+", default=None,
                        help="Seeds to run (default: config seed)")
    parser.add_argument("--alphas", type=float, nargs="+", default=None,
                        help="Alpha values (default: 0.0 0.1 1.0)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-training",  action="store_true")
    parser.add_argument("--skip-analysis",  action="store_true")
    parser.add_argument("--layerwise",      action="store_true",
                        help="Run per-layer SAE analysis after standard analysis")
    args = parser.parse_args()

    base_config = load_config(args.config)
    exp_name = base_config["experiment"]["name"]

    log_path = ROOT / "logs" / f"{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_path)
    log(f"Experiment: {exp_name}")
    log(f"Config: {args.config}")
    log(f"Device: {DEVICE}")
    log(f"Log: {log_path}")

    run_configs = build_run_configs(args, base_config)
    log(f"Runs planned: {len(run_configs)}")
    for cfg in run_configs:
        log(f"  {cfg['experiment']['run_id']}  α={cfg['experiment']['alpha']}  "
            f"seed={cfg['experiment']['seed']}  epochs={cfg['training']['epochs']}")

    # ── Training ─────────────────────────────────────────────────────────────
    if not args.skip_training:
        for cfg in run_configs:
            run_id = cfg["experiment"]["run_id"]
            try:
                train_model(cfg)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log(f"  ERROR training {run_id}: {e}\n{tb}")
                notify(f"💥 CRASH: {run_id}", f"{type(e).__name__}: {e}\n\n{tb[:400]}",
                       priority="urgent", tags="rotating_light")

    # ── Analysis ─────────────────────────────────────────────────────────────
    all_results = {}
    if not args.skip_analysis:
        for cfg in run_configs:
            run_id = cfg["experiment"]["run_id"]
            try:
                result = run_analysis(run_id, cfg)
                all_results[run_id] = result
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log(f"  ERROR analyzing {run_id}: {e}\n{tb}")
                notify(f"💥 CRASH (analysis): {run_id}", f"{type(e).__name__}: {e}\n\n{tb[:400]}",
                       priority="urgent", tags="rotating_light")
                all_results[run_id] = None

        # Load any previously computed results for runs we didn't re-analyze
        json_dir = ROOT / "results" / "json"
        for cfg in run_configs:
            run_id = cfg["experiment"]["run_id"]
            if run_id not in all_results or all_results[run_id] is None:
                cached = json_dir / f"{run_id}_analysis.json"
                if cached.exists():
                    all_results[run_id] = json.loads(cached.read_text())

        # Reports
        out_dir = ROOT / "results"
        generate_results_md(all_results, out_dir / "RESULTS_NEW.md")
        generate_summary_json(all_results, ROOT / "results" / "json" / "summary.json")

        # ── Layer-wise analysis ───────────────────────────────────────────────
        if args.layerwise:
            log(f"\n  Running layer-wise SAE analysis...")
            for cfg in run_configs:
                run_id = cfg["experiment"]["run_id"]
                try:
                    run_layerwise_analysis(run_id, cfg)
                except Exception as e:
                    log(f"  ERROR layerwise {run_id}: {e}")
                    import traceback; traceback.print_exc()

        # Monosemanticity distribution plot
        figs_dir = ROOT / "results" / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        scores_dict = {}
        for cfg in run_configs:
            run_id = cfg["experiment"]["run_id"]
            ckpt_path = checkpoint_dir(cfg) / "checkpoint_best.pt"
            if not ckpt_path.exists():
                continue
            try:
                state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                m = create_model(state["config"])
                m.load_state_dict(state["model_state_dict"])
                m.eval()
                data_dir = str(abs_path(state["config"]["data"]["data_dir"]))
                _, vl = get_dataloaders(
                    data_dir=data_dir,
                    num_classes=state["config"]["data"]["num_classes"],
                    image_size=state["config"]["data"]["image_size"],
                    batch_size=64, num_workers=4,
                )
                sel = compute_class_selectivity(m, vl, state["config"]["data"]["num_classes"], DEVICE)
                scores_dict[run_id] = compute_monosemanticity_scores(sel)
            except Exception as e:
                log(f"  WARNING: could not get scores for {run_id}: {e}")

        if scores_dict:
            plot_monosemanticity_distribution(
                scores_dict,
                str(figs_dir / f"{exp_name}_monosemanticity.png"),
                title=f"Monosemanticity — {exp_name}",
            )

    log(f"\n{'='*60}")
    log(f"  ALL DONE")
    log(f"{'='*60}")
    close_logging()


if __name__ == "__main__":
    main()

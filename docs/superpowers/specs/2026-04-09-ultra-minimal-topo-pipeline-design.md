# Design Spec: Ultra-Minimal Topo Monosemanticity Pipeline

**Date:** 2026-04-09
**Author:** AI Assistant
**Status:** Draft - Pending Review

---

## 1. Purpose & Scope

Build a complete, functional research pipeline for studying monosemanticity in topographic Vision Transformers, optimized for 4GB VRAM (RTX 3050). This is a **prototype-scale** implementation designed to:

1. Validate the entire training → analysis → visualization pipeline
2. Produce meaningful (if limited) scientific results
3. Be **easily migratable** to larger compute (A100, Colab Pro) without code breakage

**Key constraint:** Must fit in 4GB VRAM using gradient accumulation + mixed precision.

**Scale:** ImageNet-100, Tiny ViT (4 layers, 128 dim), 50 epochs, 3 model variants.

---

## 2. Architecture

### 2.1 Module Structure

```
src/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── tiny_vit.py          # TinyVisionTransformer class
│   └── topo_loss.py         # TopoLoss module (independent of model)
├── experiments/
│   ├── __init__.py
│   └── train.py             # Training loop, config loading
├── analysis/
│   ├── __init__.py
│   └── monosemanticity.py   # Selectivity scoring, statistical tests
├── data/
│   ├── __init__.py
│   └── imagenet.py          # Dataset utilities (configurable class count)
└── utils/
    ├── __init__.py
    ├── config.py            # Configuration management
    └── visualization.py     # Plotting utilities (cortical heatmaps, distributions)

configs/
└── exp_001_ultra_minimal.yaml

results/
├── data/                    # Checkpoints, cached activations
├── figures/                 # Training curves, analysis plots
└── summaries/               # Per-experiment markdown reports
```

### 2.2 Module Boundaries

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `models.tiny_vit` | Tiny ViT forward pass | PyTorch only |
| `models.topo_loss` | TopoLoss computation given cortical positions | PyTorch only (no model dependency) |
| `experiments.train` | Training loop, logging, checkpointing | models.*, data.*, utils.config |
| `analysis.monosemanticity` | Score computation, stats | PyTorch, scipy |
| `data.imagenet` | Dataset loading, transforms | torchvision, PIL |
| `utils.config` | YAML config → dict | PyYAML |
| `utils.visualization` | Plot generation | matplotlib, seaborn |

**Migration safety rule:** No module imports from a different module's subdirectory except through public interfaces. This ensures you can swap `tiny_vit.py` → `vit_s_16.py` without touching anything else.

---

## 3. Tiny ViT Design

### 3.1 Architecture Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Depth | 4 layers | Enough for hierarchy, fits in 4GB |
| Hidden dim | 128 | Small but expressive |
| Attention heads | 4 | Head dim = 32 (standard) |
| Patch size | 16 | Same as ViT-S/16 for fair comparison |
| Image size | 128×128 | Reduces memory, can increase later |
| Position embedding | Learnable | Standard ViT approach |
| CLS token | Yes | For classification |

### 3.2 Forward Pass

```
Input (B, 3, 128, 128)
  → PatchEmbed (B, 64, 128)  # 8×8 patches
  → PosEmbed (add learnable positions)
  → [TransformerBlock × 4]   # Each: MHA → MLP → LayerNorm
  → CLS token extraction (B, 128)
  → Linear head (B, 100)     # 100-class classification
```

### 3.3 Cortical Sheet Assignment

Each output channel in **each attention layer** gets a fixed 2D position:
- 128 channels → √128 ≈ 11.3 → use 11×12 grid (132 positions, 4 unused)
- Positions are assigned once at initialization, never change
- This creates the "cortical sheet" for TopoLoss computation
- **Edge handling:** Boundary units only compute similarity with valid neighbors (non-padded positions). Padding positions are excluded from neighborhood definitions to avoid artificial edge effects.
- TopoLoss is computed for **all 4 attention layers** and summed: `L_topo = Σ_l L_topo^l`

---

## 4. TopoLoss Implementation

### 4.1 Formulation

Following TopoNets (Deb et al., 2025) — **weight-level smoothness**, not activation-level:

```
For each pair of neighboring units (u, v) on cortical sheet:
  sim(u, v) = cosine_similarity(W_u, W_v)

L_topo = -Σ_{(u,v) ∈ neighborhoods} sim(u, v) × exp(-dist(u, v) / σ)

Total loss: L = L_CE + α × L_topo
```

Where:
- `W_u` = weight vector for unit u (the output projection weights for that channel)
- `dist(u, v)` = Euclidean distance on the 2D cortical sheet
- `σ` = length scale hyperparameter (default: 2.0)
- `α` = topographic strength: {0, 0.1, 1.0}

### 4.2 Implementation Details

1. **Neighborhood definition:** 8-connected grid neighbors (up, down, left, right, diagonals)
2. **Weight extraction:** Hook into the output projection layer of each attention block
3. **Computation:** Cheap — only runs on weight matrices, not per-batch activations
4. **Gradient flow:** TopoLoss contributes to weight updates through standard backprop

### 4.3 Key Design Decision

TopoLoss is implemented as a **standalone module** that takes:
- Model weights (via parameter extraction)
- Cortical position map
- Hyperparameters (α, σ)

It returns a scalar loss. This means it works with **any model architecture**, not just TinyViT.

---

## 5. Training Protocol

### 5.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | ImageNet-100 (~130K train, ~6.5K val) |
| Image size | 128×128 |
| Physical batch size | 8 |
| Gradient accumulation steps | 4 |
| Effective batch size | 32 |
| Epochs | 50 |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.05) |
| LR schedule | Cosine annealing to 1e-5 |
| Warmup | 5 epochs |
| Mixed precision | FP16 (torch.cuda.amp) |
| Data augmentation | Random crop, horizontal flip, color jitter |

### 5.2 Three Model Variants

| Variant | α (TopoLoss weight) | Purpose |
|---------|---------------------|---------|
| Baseline | 0.0 | Control: standard training |
| TopoWeak | 0.1 | Light topographic pressure |
| TopoStrong | 1.0 | Strong topographic pressure |

All other hyperparameters are **identical** across variants.

### 5.3 Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        with autocast():
            outputs = model(images)
            loss_ce = cross_entropy(outputs, labels) / accumulation_steps
        
        loss_ce.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # TopoLoss computed once per accumulation cycle (not per-batch)
            loss_topo = topo_loss(model)  # computed on weights, all layers
            (alpha * loss_topo).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # Validation pass
    model.eval()
    val_loss, val_acc = validate(model, val_loader)
    if val_acc > best_acc:
        save_checkpoint(model, optimizer, epoch, 'best')
```

**Key:** `loss_ce` is accumulated over batches, but `loss_topo` (weight-level, batch-independent) is computed once per accumulation cycle. This keeps α consistent regardless of accumulation steps.

### 5.4 Expected Training Time

- ~2-4 hours per variant on RTX 3050 (4GB)
- Total: ~6-12 hours for all three variants
- Can run sequentially overnight

---

## 6. Analysis Pipeline

### 6.1 Monosemanticity Score

For each unit u in each model variant:

1. **Collect activations** for all images in a held-out probe set (100 images per class from ImageNet-100 validation set)
   - **Note:** The full research plan uses the THINGS dataset for concept categories. For this ultra-minimal pipeline, we use ImageNet-100 validation images as a proxy. This is a prototype-level simplification — the analysis code is structured so swapping to THINGS later requires only changing the data loader, not the scoring logic.
2. **Compute mean activation** per class: μ_{u,c} = mean(activations of u for class c)
3. **Selectivity vector:** s_u = [μ_{u,1}, μ_{u,2}, ..., μ_{u,100}]
4. **Score:** M_u = max(s_u) / sum(s_u) ∈ [1/100, 1]
   - M_u = 1: fires ONLY for one class (monosemantic)
   - M_u = 1/100: fires equally for all (maximally polysemantic)

### 6.2 Additional Analysis Modules (Stubs for Migration)

The full research plan includes two additional analyses. For this ultra-minimal pipeline, we provide **stub modules** with clear interfaces so they can be implemented later without breaking the pipeline.

#### 6.2.1 SAE Analysis (H2 - Superposition) — STUB

```python
# src/analysis/sae.py — Stub for migration
# TODO: Implement SAE training protocol per research plan Section 3.3
# - Extract residual stream activations at middle layer
# - Train SAE with 8x expansion, L1 sparsity
# - Compare L0 norm, dead feature fraction between models

def train_sae(activations, config):
    """Stub: returns SAE metrics dict"""
    raise NotImplementedError("SAE analysis pending — prototype focuses on H1")

def evaluate_sae(sae, test_activations):
    """Stub: returns L0, reconstruction loss, dead features"""
    raise NotImplementedError("SAE analysis pending — prototype focuses on H1")
```

#### 6.2.2 Activation Patching (H3 - Causal Purity) — STUB

```python
# src/analysis/patching.py — Stub for migration
# TODO: Implement activation patching per research plan Section 3.4
# - Identify face cluster (or most selective cluster for any class)
# - Patch activations and measure Δlogit
# - Compare topographic cluster vs. random unit set

def identify_selective_cluster(model, probe_data, class_idx):
    """Stub: returns set of unit indices most selective for class"""
    raise NotImplementedError("Patching analysis pending — prototype focuses on H1")

def run_patching_experiment(model, cluster_units, patch_source_images, test_images):
    """Stub: returns Δlogit distribution"""
    raise NotImplementedError("Patching analysis pending — prototype focuses on H1")
```

**Rationale:** Including stubs ensures the module structure is finalized now. When you're ready for H2/H3, you implement the functions — no refactoring of existing code needed.

### 6.3 Metrics to Report

| Metric | Description | Comparison |
|--------|-------------|------------|
| Mean monosemanticity | Average M_u across all units | Baseline vs. TopoWeak vs. TopoStrong |
| Fraction > 0.5 | % of units with high selectivity | Same |
| Distribution plot | Histogram of M_u values | Overlay all three variants |
| Top-k selectivity maps | Heatmap of most selective units on cortical sheet | Visual clustering check |

### 6.4 Statistical Testing

- **t-test** between baseline and topo variants (mean M_u)
- **Effect size:** Cohen's d
- **Significance threshold:** p < 0.05

---

## 7. Visualization Outputs

### 7.1 Figures to Generate

1. **Training curves:** Loss, accuracy over epochs (3 lines: baseline, topo-weak, topo-strong)
2. **TopoLoss curves:** L_topo over epochs (should decrease for topo variants, stay flat for baseline)
3. **Monosemanticity distributions:** Overlaid histograms or KDE plots
4. **Cortical heatmaps:** 2D grid showing which units are face-selective, body-selective, etc.
5. **Summary table:** All metrics in a clean table for PROGRESS.md

### 7.2 Format

- All figures: 300 DPI, PNG + PDF (publication quality)
- Captions with key insights
- Consistent color scheme across all plots

---

## 8. Configuration System

### 8.1 Config File: `configs/exp_001_ultra_minimal.yaml`

```yaml
experiment:
  name: exp_001_ultra_minimal
  seed: 42
  alpha: 0.0  # Per-run override: 0.0, 0.1, 1.0
  run_id: baseline  # Descriptive label for this run

data:
  dataset: imagenet-100
  num_classes: 100
  image_size: 128
  data_dir: /path/to/imagenet
  batch_size: 8
  accumulation_steps: 4
  num_workers: 4

model:
  type: tiny_vit
  depth: 4
  hidden_dim: 128
  num_heads: 4
  patch_size: 16
  dropout: 0.1

training:
  epochs: 50
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.05
  warmup_epochs: 5
  lr_min: 0.00001
  mixed_precision: true
  grad_clip: 1.0
  checkpoint_every: 10  # epochs
  save_best: true

topo_loss:
  sigma: 2.0
  layers: all  # Which layers to apply TopoLoss: 'all' or list of indices

analysis:
  probe_images_per_class: 100
  monosemanticity_threshold: 0.5

output:
  results_dir: results
  checkpoints_dir: results/data/checkpoints
  figures_dir: results/figures
  summaries_dir: results/summaries
```

### 8.2 Per-Run Config Generation

For reproducibility, each run gets its own config file derived from the base config:

```bash
# Run script generates per-run configs
python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 0.0 --run_id baseline
python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 0.1 --run_id topo_weak
python src/experiments/train.py --config configs/exp_001_ultra_minimal.yaml --alpha 1.0 --run_id topo_strong
```

The script creates: `results/data/checkpoints/<run_id>/config.yaml` with the exact parameters used.

### 8.3 Checkpoint Format

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': current_epoch,
    'config': config_dict,  # Full config snapshot
    'config_hash': hash(config_dict),  # For verification
    'random_state': {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    },
    'metrics': {'train_loss': ..., 'val_acc': ..., 'topo_loss': ...},
}
```

### 8.4 Migration Config (Future)

To scale to cloud compute, **only change these fields:**

```yaml
data:
  dataset: imagenet-1k
  num_classes: 1000
  image_size: 224
  batch_size: 128
  accumulation_steps: 1

model:
  type: vit_s_16  # Swap to timm model
  # ... or keep tiny_vit and increase dimensions

training:
  epochs: 90  # Standard schedule
```

**Nothing else changes.** All downstream code reads from config.

---

## 9. Error Handling & Edge Cases

### 9.1 OOM Protection

- Gradient checkpointing option if 4GB is still too tight
- Automatic batch size reduction on OOM (halve until it fits)
- Clear CUDA cache after each epoch

### 9.2 Data Loading

- Fallback to smaller worker count if memory pressure
- Graceful handling of missing ImageNet paths
- Progress bars for long operations (activation collection)

### 9.3 Training Stability

- Gradient clipping (max_norm=1.0)
- Loss scaling for mixed precision
- Checkpoint every 10 epochs + best model

---

## 10. Success Criteria

| Criterion | Threshold | Priority |
|-----------|-----------|----------|
| All 3 models train without OOM | No CUDA OOM errors | MUST HAVE |
| TopoLoss decreases during training | Visible reduction in L_topo for α>0 | MUST HAVE |
| Monosemanticity analysis completes | Scores computed for all units | MUST HAVE |
| Results reproducible | Same seed → same results (within tolerance) | MUST HAVE |
| Validation accuracy > chance | >5% on ImageNet-100 (random = 1%) | MUST HAVE |
| Code migration-ready | Config-only scaling, no hard-coded values | SHOULD HAVE |
| Statistical significance | p < 0.05 for M_u difference | NICE TO HAVE |
| SAE/Patching stubs in place | Modules exist with clear TODO markers | SHOULD HAVE |

---

## 11. Migration Safety Checklist

To ensure this prototype doesn't break when scaling:

- [ ] All hyperparameters in config files (no hard-coded values in code)
- [ ] Model interface matches timm's `timm.create_model()` signature (forward returns logits)
- [ ] TopoLoss works with any model that exposes weight matrices via hook or method
- [ ] Analysis code reads model outputs/activations, not model internals
- [ ] Data loading abstracted behind a `get_dataloaders(config)` function
- [ ] Visualization code reads from results files, not training loops
- [ ] Clear separation: `src/models/` (architecture) vs `src/experiments/` (training) vs `src/analysis/` (evaluation)
- [ ] Random seed management: torch, numpy, python.random all seeded from config
- [ ] Checkpoint format includes config hash for verification
- [ ] Per-run configs saved to results directory for reproducibility
- [ ] SAE and patching modules have stub interfaces defined
- [ ] Validation loop included in training (not just training loss)
- [ ] TopoLoss computed on all layers, not just final layer
- [ ] Gradient accumulation doesn't affect TopoLoss α scaling

---

## 12. Timeline Estimate

| Task | Duration |
|------|----------|
| Environment setup + dependencies | 1 hour |
| Implement TinyViT + TopoLoss | 4-6 hours |
| Implement training loop + config system | 3-4 hours |
| Debug + verify training runs | 2-3 hours |
| Implement analysis pipeline | 3-4 hours |
| Run 3 model variants (sequential) | 6-12 hours |
| Generate figures + update PROGRESS.md | 2-3 hours |
| **Total** | **~21-33 hours** |

---

**END OF DESIGN SPEC**

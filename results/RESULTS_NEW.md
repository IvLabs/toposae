# Experiment Results — Topo Monosemanticity Research

> Auto-generated 2026-04-24 01:12:55

## Setup
- **Dataset:** ImageNet-100 (clane9/imagenet-100, HF)
- **Models:** TinyViT (4L/128D), ViT-S/16 (12L/384D), ViT-B/16 (12L/768D)
- **Training:** AdamW, cosine LR, mixed precision, grad clip 1.0
- **SAE:** 4× expansion, L1=0.001, 50 epochs, batch=512
- **Patching:** Top-16 selective units, class 0

## vit_small_patch16_224

### Training

| Run | α | Seed | Epoch | Val Acc |
|-----|---|------|-------|---------|
| topo_001_s42 | 0.01 | 42 | 88 | 0.6150 |

### H1: Monosemanticity

| Run | Mean | Median | Max | Frac > 0.5 |
|-----|------|--------|-----|------------|
| topo_001_s42 | 0.2317 | 0.2096 | 1.0000 | 0.0026 |

### H2: SAE

| Run | L0 Norm | Dead % | Recon Loss |
|-----|---------|--------|------------|
| topo_001_s42 | 1313.1 | 0.3 | 0.005097 |

### H3: Causal Patching

| Run | Cluster |Δlogit| | Random |Δlogit| | Ratio |
|-----|----------|----------|-------|
| topo_001_s42 | 0.3965 | 0.3358 | 1.18× |

---
*Generated 2026-04-24 01:12:55*

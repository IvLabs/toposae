# Experiment Results — Topo Monosemanticity Research

> Auto-generated 2026-04-19 23:07:07

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
| baseline_s123 | 0.0 | 123 | 83 | 0.6194 |

### H1: Monosemanticity

| Run | Mean | Median | Max | Frac > 0.5 |
|-----|------|--------|-----|------------|
| baseline_s123 | 0.2324 | 0.2106 | 0.5794 | 0.0052 |

### H2: SAE

| Run | L0 Norm | Dead % | Recon Loss |
|-----|---------|--------|------------|
| baseline_s123 | 1256.2 | 0.3 | 0.012608 |

### H3: Causal Patching

| Run | Cluster |Δlogit| | Random |Δlogit| | Ratio |
|-----|----------|----------|-------|
| baseline_s123 | 0.3259 | 0.2839 | 1.15× |

---
*Generated 2026-04-19 23:07:07*

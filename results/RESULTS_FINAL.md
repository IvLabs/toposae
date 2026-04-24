# Final Results — Topo Monosemanticity (ViT-S/16, ImageNet-100)

> Generated 2026-04-24 01:37:46

## Multi-Seed Summary (mean ± std, seeds 42/123/456)

| α | Val Acc | H1 Mono↑ | H2 L0↓ | H2 Dead↑ | H3 Ratio↑ |
|---|---------|----------|--------|----------|-----------|
| 0.0 | 0.6123±0.0054 | 0.2340±0.0019 | 1279.9±13.9 | 0.2±0.1 | 1.211±0.042 |
| 0.1 | 0.6143±0.0022 | 0.2348±0.0025 | 1308.0±18.1 | 0.2±0.1 | 1.404±0.078 |
| 1.0 | 0.5959±0.0048 | 0.2359±0.0016 | 1130.5±7.4 | 3.9±0.7 | 2.757±0.658 |

## Alpha Sweep (seed=42)

| α | Val Acc | H1 Mono | H2 L0 | H3 Ratio |
|---|---------|---------|-------|----------|
| 0.0 | 0.6064 | 0.2328 | 1283.1 | 1.269 |
| 0.01 | 0.6150 | 0.2317 | 1313.1 | 1.181 |
| 0.1 | 0.6138 | 0.2383 | 1287.2 | 1.403 |
| 1.0 | 0.5974 | 0.2374 | 1120.4 | 3.685 |

## Figures

- `results/figures/multiseed_summary.png` — H1/H2/H3 bar chart with error bars
- `results/figures/alpha_sweep.png` — all 5 metrics vs α
- `results/figures/layerwise_sae.png` — L0/dead/recon per layer

---

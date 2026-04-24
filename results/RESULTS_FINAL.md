# Final Results — Topo Monosemanticity (ViT-S/16, ImageNet-100)

> Generated 2026-04-24 12:36:04

## Multi-Seed Summary (mean ± std, seeds 42/123/456)

| α | Val Acc | H1 Mono↑ | H2 L0↓ | H2 Dead↑ | H3 Ratio↑ |
|---|---------|----------|--------|----------|-----------|
| 0.0 | 0.6123±0.0054 | 0.2340±0.0019 | 1273.9±15.0 | 0.2±0.1 | 1.189±0.011 |
| 0.1 | 0.6143±0.0022 | 0.2348±0.0025 | 1306.4±19.9 | 0.2±0.1 | 1.404±0.078 |
| 1.0 | 0.5959±0.0048 | 0.2359±0.0016 | 1133.8±3.2 | 3.8±0.8 | 2.786±0.700 |

## Alpha Sweep (seed=42)

| α | Val Acc | H1 Mono | H2 L0 | H3 Ratio |
|---|---------|---------|-------|----------|
| 0.0 | 0.6064 | 0.2328 | 1265.1 | 1.203 |
| 0.01 | 0.6150 | 0.2317 | 1313.4 | 1.170 |
| 0.1 | 0.6138 | 0.2383 | 1282.6 | 1.402 |
| 1.0 | 0.5974 | 0.2374 | 1130.4 | 3.772 |

## Figures

- `results/figures/multiseed_summary.png` — H1/H2/H3 bar chart with error bars
- `results/figures/alpha_sweep.png` — all 5 metrics vs α
- `results/figures/layerwise_sae.png` — L0/dead/recon per layer

---

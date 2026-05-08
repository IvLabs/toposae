# TopoSAE — Topographic Training as a Path to Monosemanticity

Investigates whether spatially-constrained topographic training (TopoLoss) induces monosemanticity and reduces feature superposition in vision transformers. Primary model: **ViT-S/16** on **ImageNet-100**.

Target venue: ICML 2026 Mechanistic Interpretability Workshop.

---

## Key Results

| Hypothesis | Result | Evidence |
|---|---|---|
| **H1** Polysemanticity ↓ | **Null** (neuron + SAE-feature level) | Entropy M_u flat across α; confirmed at 5k and 25k images |
| **H2** Feature superposition ↓ | **Supported at α=1.0** | SAE L0 −11%, dead features 19×, p=0.007, d=7.1 |
| **H3** Causal purity ↑ | **Strongly supported** | Patching ratio 2.79× vs 1.68×, p<0.0001, d=0.95 (20 classes, 3 seeds) |
| **EXP_019** Spatial coherence ↑ | **Supported at α=1.0** | Top-k units 6.1% more co-localized, p=0.0005, d=0.66 |

**Core claim:** Topographic training concentrates causally relevant circuitry in spatial clusters (H3) and reduces feature superposition (H2) without improving neuron-level selectivity (H1 null). Effects emerge at α=1.0; α≤0.1 leaves feature geometry largely unchanged.

**Architecture variability:** TinyViT (4L/128D) shows weaker H2 (−2.1%) and H3 peaks at α=0.1 rather than α=1.0. Weight-norm analysis reveals TinyViT is over-pressured at α=1.0 (effective pressure 2.6× ViT-S). H3 may additionally require sufficient model depth.

---

## Setup

```bash
# Clone and install
git clone git@github.com:Ashu-00/toposae.git
cd toposae
pip install -r requirements.txt

# Download ImageNet-100
python scripts/download_data.py
```

**Requirements:** Python 3.9+, PyTorch, timm, numpy, scikit-learn, matplotlib.

---

## How to Run

### Training
```bash
# Single run
python src/experiments/train.py --alpha 1.0 --seed 42

# Multi-seed sweep (α ∈ {0.0, 0.1, 1.0}, seeds 42/123/456)
bash scripts/run_all.sh
```

### Analysis
```bash
# Main H1/H2/H3 analysis (ViT-S)
python scripts/run_analysis_vit_s16.py

# Statistical tests (EXP_010/015)
python scripts/run_statistical_analysis.py

# H3 multiclass patching (EXP_016)
python scripts/run_exp016_multiclass_patching.py

# SAE monosemanticity — 25k train images (EXP_012b)
python scripts/run_exp012_large_eval.py

# Spatial coherence of top-k class units (EXP_019)
python scripts/run_spatial_coherence.py

# Layer-wise SAE L0 figure
python scripts/plot_layerwise_sae_l0.py
```

---

## Repo Structure

```
toposae/
├── src/
│   ├── models/             # Model definitions (ViT-S, TinyViT, ResNet-18)
│   ├── experiments/        # Training loop + TopoLoss
│   ├── analysis/           # SAE, patching, monosemanticity metrics
│   ├── utils/
│   └── visualization/
├── scripts/                # One-off experiment and analysis scripts
├── configs/                # Training configs
├── notebooks/
├── results/
│   ├── figures/            # All generated plots
│   ├── json/               # Raw numeric outputs
│   ├── data/               # Cached activations
│   ├── RESULTS_FINAL.md    # Full results writeup
│   └── NEW_EXPERIMENTS.md  # Revision experiments (EXP_019, EXP_012b, layer-wise fig)
└── PROGRESS.md             # Experiment log and tracking
```

---

## Experiments Index

| ID | Description | Status | Key output |
|---|---|---|---|
| EXP_010 | Accuracy–L0 confound (OLS) | ✅ | `json/statistical_analysis.json` |
| EXP_011 | PCA dimensionality control | ✅ | `json/exp011_pca.json` |
| EXP_012 | SAE feature mono (5k val) | ✅ null | `json/exp012_sae_monosemanticity.json` |
| EXP_012b | SAE feature mono (25k train) | ✅ null confirmed | `json/exp012b_sae_monosemanticity_large.json` |
| EXP_015 | Multi-seed paired t-tests | ✅ | `json/statistical_analysis.json` |
| EXP_016 | H3 multiclass patching (20 classes) | ✅ | `json/exp016_multiclass_patching.json` |
| EXP_018 | α-scaling / weight-norm analysis | ✅ | `json/alpha_scaling.json` |
| EXP_019 | Spatial coherence of top-k units | ✅ | `json/spatial_coherence.json` |
| TinyViT | 4L architecture variability | ✅ exploratory | `results/RESULTS_FINAL.md §9` |
| ResNet-18 | Architecture variability | ⏸ paused | — |

See [`results/RESULTS_FINAL.md`](results/RESULTS_FINAL.md) for full quantitative results and [`results/NEW_EXPERIMENTS.md`](results/NEW_EXPERIMENTS.md) for the three revision additions.

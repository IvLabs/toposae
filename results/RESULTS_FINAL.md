# Final Results — Topographic Training as a Path to Monosemanticity

> **Model:** ViT-S/16 (`vit_small_patch16_224`)  
> **Dataset:** ImageNet-100 (126,688 train / 5,000 val, 100 classes)  
> **Variants:** α ∈ {0.0 (Baseline), 0.1 (TopoWeak), 1.0 (TopoStrong)}  
> **Seeds:** {42, 123, 456} for all main results  
> **Last updated:** 2026-04-24  
> **Target venue:** ICML 2026 Mechanistic Interpretability Workshop

---

## 1. Training Summary

| α | Label | Val Acc (mean±std) | Epochs |
|---|---|---|---|
| 0.0 | Baseline | 0.6123 ± 0.0054 | ~87 |
| 0.1 | TopoWeak | 0.6143 ± 0.0022 | ~67 |
| 1.0 | TopoStrong | 0.5959 ± 0.0048 | ~79 |

Accuracy cost at α=1.0: **−1.6 pp** (0.612 → 0.596).

---

## 2. H1 — Monosemanticity (entropy-based, neuron level)

**Result: NULL**

| α | Mean M_u | Median M_u | Frac > 0.5 |
|---|---|---|---|
| 0.0 | 0.2340 ± 0.0019 | — | 0.26% |
| 0.1 | 0.2348 ± 0.0025 | — | 0.52% |
| 1.0 | 0.2359 ± 0.0016 | — | 0.78% |

No meaningful difference across variants. Entropy-based monosemanticity is insensitive to the reorganisation produced by TopoLoss.

**SAE feature-level H1 (EXP_012):** Also null. SAE trained on 5k val images achieves L0=82% of features active — insufficient sparsity for feature-level selectivity analysis. Requires train-set activations (~126k images) for a valid measurement; left for future work.

---

## 3. H2 — Feature Superposition (SAE, layer 6)

**Result: SIGNIFICANT at α=1.0**

### 3a. Multi-seed aggregate (seeds 42/123/456)

| α | SAE L0↓ | Dead %↑ | Recon Loss |
|---|---|---|---|
| 0.0 | 1273.9 ± 15.0 | 0.2 ± 0.1 | — |
| 0.1 | 1306.4 ± 19.9 | 0.2 ± 0.1 | — |
| 1.0 | **1133.8 ± 3.2** | **3.8 ± 0.8** | — |

L0 reduction at α=1.0: **−11.0%** vs baseline.  
Dead feature increase: **19×** (0.2% → 3.8%).

### 3b. Paired t-tests (EXP_015, n=3 seeds)

| Comparison | Metric | Δ mean | t | p | Cohen's d |
|---|---|---|---|---|---|
| α=0.0 vs α=0.1 | L0 | +32.5 | −4.15 | 0.054 | 2.39 |
| α=0.0 vs α=1.0 | L0 | **−140.1** | 12.23 | **0.007** | **−7.06** |
| α=0.0 vs α=0.1 | Dead % | +0.07 | −0.66 | 0.580 | 0.38 |
| α=0.0 vs α=1.0 | Dead % | **+3.58** | −5.72 | **0.029** | **3.30** |

### 3c. Accuracy-L0 confound check (EXP_010)

OLS regression L0 ~ val_acc: R²=0.724, p=0.004.  
Accuracy explains ~72% of L0 variance. Topo_strong residuals: −54, −37, +21 (2/3 seeds below the regression line — trend toward sparser-than-accuracy-predicts but not conclusive across all seeds).  
**Interpretation:** L0 reduction is partially explained by accuracy drop; framed as a partial confound, not a full explanation.

### 3d. PCA dimensionality control (EXP_011)

| α | PCA d_95 | Δd_95 | SAE L0 | ΔL0 |
|---|---|---|---|---|
| 0.0 | 166.7 ± 3.3 | — | 1273.9 | — |
| 0.1 | 158.3 ± 3.8 | −8.3 | 1306.4 | **+32.5** |
| 1.0 | 142.3 ± 3.4 | −24.3 | 1133.8 | −140.1 |

At α=0.1: PCA d_95 decreases but SAE L0 *increases* — the two metrics dissociate, showing SAE L0 captures something beyond raw effective dimensionality. At α=1.0 both decrease in similar proportions (~15% vs ~11%).

### 3e. Layer-wise SAE (seed=42)

Effect distributed across depth, not layer-specific. Strongest gap at layers 5–10. α=0.01 and α=0.1 layer-wise curves track baseline; α=1.0 separates cleanly from layer 3 onward (L0 plateaus at ~1080–1130 vs baseline's rising curve to ~1475).

---

## 4. H3 — Causal Purity (Activation Patching)

**Result: HIGHLY SIGNIFICANT, large effect**

### 4a. Single-class result (class 0, multi-seed, EXP_015)

| α | Cluster \|Δlogit\| | Random \|Δlogit\| | Ratio | Cohen's d (vs baseline) |
|---|---|---|---|---|
| 0.0 | — | — | 1.189 ± 0.011 | — |
| 0.1 | — | — | 1.404 ± 0.078 | 2.39 |
| 1.0 | — | — | **2.786 ± 0.700** | **1.89** |

Class-0 t-test: p=0.082 (n=3 seeds insufficient).

### 4b. Multi-class result (20 classes, EXP_016) — primary H3 statistic

| α | Mean ratio (20 classes) | Δ vs baseline | n | p | Cohen's d |
|---|---|---|---|---|---|
| 0.0 | 1.68 ± 0.42 | — | 60 | — | — |
| 0.1 | 2.27 ± 0.93 | +0.59 | 60 | **<0.0001** | **0.61** |
| 1.0 | 2.79 ± 1.24 | +1.12 | 60 | **<0.0001** | **0.95** |

Per-seed mean ratios at α=1.0: 2.53, 2.89, 2.97 — consistent across all three seeds.  
**Interpretation:** Topographic clusters are 1.7–2.8× more causally sufficient than random unit sets, monotonically increasing with α, robust across 20 classes and 3 seeds.

---

## 5. Alpha Sweep (seed=42)

| α | Val Acc | H1 Mono | H2 L0 | H2 Dead% | H3 Ratio |
|---|---|---|---|---|---|
| 0.00 | 0.6064 | 0.2328 | 1265.1 | 0.20 | 1.20 |
| 0.01 | 0.6150 | 0.2317 | 1313.4 | 0.26 | 1.17 |
| 0.10 | 0.6138 | 0.2383 | 1282.6 | 0.39 | 1.40 |
| 1.00 | 0.5974 | 0.2374 | 1130.4 | 3.32 | 3.77 |

Threshold behaviour: H2 and H3 effects emerge primarily at α=1.0. Light topographic pressure (α≤0.1) leaves the feature geometry largely unchanged.

---

## 6. Per-seed H3 robustness (α=1.0)

| Seed | Cluster \|Δlogit\| | Random \|Δlogit\| | Ratio |
|---|---|---|---|
| 42 | 1.282 | 0.340 | 3.77× |
| 123 | 0.911 | 0.386 | 2.36× |
| 456 | 0.892 | 0.401 | 2.22× |

All three seeds individually exceed 2× — effect not driven by a single outlier.

---

## 7. Figures inventory

| File | Content |
|---|---|
| `figures/multiseed_summary.png` | H1/H2/H3 bar chart, mean±std, 3 seeds |
| `figures/alpha_sweep.png` | All metrics vs α (seed=42) |
| `figures/layerwise_sae.png` | L0/dead/recon per layer, seed=42, 4 α values |
| `figures/exp010_accuracy_l0.png` | Accuracy vs L0 scatter + residuals |
| `figures/exp011_pca_control.png` | PCA d_95 vs SAE L0, scatter + bars |
| `figures/exp012_sae_monosemanticity.png` | SAE feature mono (null result) |
| `figures/exp015_paired_stats.png` | Paired t-test forest plot |
| `figures/exp016_h3_multiclass.png` | H3 ratios across 20 classes, boxplot + bars |

---

## 8. Interpretation summary

| Hypothesis | Result | Stat evidence | Paper placement |
|---|---|---|---|
| H1: Polysemanticity ↓ | **Null** (both entropy and SAE-feature measures) | — | Appendix |
| H2: L0 sparsity ↓ | **Supported at α=1.0** (−11%, p=0.007, d=7.1) | Paired t-test | Main §Results |
| H3: Causal purity ↑ | **Strongly supported** (p<0.0001, d=0.95) | Paired t-test, 60 obs | Main §Results (primary) |
| H4: Brain alignment | **Not attempted** (NSD data access required) | — | Future work |

**Core claim:** Topographic training concentrates causally relevant circuitry in spatial clusters (H3) and reduces feature superposition at sufficient pressure (H2), without improving neuron-level selectivity (H1 null). The two effects are partially, but not fully, explained by the small accuracy cost at α=1.0.

---

## 9. Architecture Variability

### 9a. TinyViT (4L / 128D / 4H, ImageNet-100, 50 epochs, seed=42)

> Preliminary — single seed, single-class patching (class 0, layer 2). Different methodology from ViT-S multi-seed/multi-class analysis. Treat as exploratory.

**Training**

| α | Val Acc | Best Epoch |
|---|---|---|
| 0.0 | 0.4718 | 49 |
| 0.1 | 0.4624 | 48 |
| 1.0 | 0.4570 | 50 |

**H1 (entropy mono)**

| α | Mean M_u | Frac > 0.5 |
|---|---|---|
| 0.0 | 0.2525 | 2.34% |
| 0.1 | **0.2599** | **3.12%** |
| 1.0 | 0.2491 | 1.56% |

**H2 (SAE L0, full 126K activations)**

| α | L0 | Dead % | Recon Loss |
|---|---|---|---|
| 0.0 | 444.9 | 2.0% | 5.9e-5 |
| 0.1 | 443.7 | 2.5% | 5.2e-5 |
| 1.0 | **435.5** | **2.9%** | **4.9e-5** |

H2 reduction at α=1.0: **−2.1%** (vs −11% for ViT-S). Effect present but weaker.

**H3 (patching ratio, class 0, layer 2)**

| α | Cluster \|Δlogit\| | Random \|Δlogit\| | Ratio |
|---|---|---|---|
| 0.0 | 1.767 | 1.479 | 1.19× |
| 0.1 | **1.884** | 1.331 | **1.42×** |
| 1.0 | 1.662 | 1.487 | 1.12× |

**Key contrast with ViT-S:**

| Model | Optimal α for H3 | H3 peak ratio | Width | Depth |
|---|---|---|---|---|
| TinyViT | **0.1** (predicted optimal: 0.38) | 1.42× | 128 | 4L |
| ViT-S/16 | **1.0** | 2.79× (20-class) | 384 | 12L |

### 9b. Alpha-Scaling Hypothesis (EXP_018)

**Why does optimal α differ across model sizes?**

TopoLoss penalises weight differences between spatially adjacent units: `topo_loss ∝ Σ(w_i − w_j)²`. Larger weight magnitudes → larger absolute topo gradients at the same nominal α. We measured `attn.proj` RMS weight norms (the only layers targeted by TopoLoss in this codebase) from saved checkpoints:

| Model | α | attn.proj RMS norm | Norm growth vs own baseline |
|---|---|---|---|
| TinyViT | 0.0 | 0.0695 | — |
| TinyViT | 1.0 | **0.2346** | **+237%** (explosion) |
| ViT-S | 0.0 | 0.0432 | — |
| ViT-S | 1.0 | 0.0548 | +27% (controlled) |

**Key finding:** TinyViT at α=1.0 suffers weight norm explosion (3.4× baseline), while ViT-S remains stable (1.27×). This is the mechanism behind H3 degradation at α=1.0 for small models.

**Effective α calculation** (topo_loss ∝ rms², reference = ViT-S baseline):

| Run | Nominal α | α_eff (ViT-S scale) |
|---|---|---|
| TinyViT α=0.1 | 0.1 | 0.37 |
| TinyViT α=1.0 | 1.0 | **2.6** (over-pressured) |
| ViT-S α=0.1 | 0.1 | 0.11 |
| ViT-S α=1.0 | 1.0 | 1.12 |

TinyViT at α=1.0 is experiencing ~2.6× the topographic pressure ViT-S sees at α=1.0.

**Predicted optimal α for TinyViT:** `α* ≈ 0.38`  
(= ViT-S optimal α=1.0 × (rms_vits_baseline / rms_tinyvit_baseline)² = (0.0432/0.0695)²)

**Planned experiment:** One new TinyViT run at α=0.38, seed=42, 50 epochs (same setup as existing TinyViT runs).  
Prediction: H3 ratio at α=0.38 > H3 at α=0.1 and α=1.0, approaching ViT-S α=1.0 performance.  
If confirmed: validates the weight-norm scaling rule as a practical design recommendation.

**Design rule:** `α_target = α_ref × (rms_ref_baseline / rms_target_baseline)²`

Script: `scripts/run_alpha_scaling_analysis.py`  
Output: `results/json/alpha_scaling.json`, `results/figures/alpha_scaling.png`

### 9c. ResNet-18 (ImageNet-100, 40 epochs, seed=42)

> Paused in favour of ViT-Ti/16 sweep. Baseline (α=0.0) + Strong (α=1.0). Will resume after ViT-Ti runs complete.

---

## 10. Experiments index

| ID | Script | Status | Output |
|---|---|---|---|
| EXP_010 | `scripts/run_statistical_analysis.py` | ✅ Done | `json/statistical_analysis.json` |
| EXP_011 | `scripts/run_exp011_pca.py` | ✅ Done | `json/exp011_pca.json` |
| EXP_012 | `scripts/run_exp012_sae_monosemanticity.py` | ✅ Done (null — val-only SAE) | `json/exp012_sae_monosemanticity.json` |
| EXP_015 | `scripts/run_statistical_analysis.py` | ✅ Done | `json/statistical_analysis.json` |
| EXP_016 | `scripts/run_exp016_multiclass_patching.py` | ✅ Done | `json/exp016_multiclass_patching.json` |
| EXP_017 | Post-hoc SOM control | ⬜ Not started | — |
| TinyViT | 4L/128D/4H training (preliminary) | ✅ Done (exploratory) | `../topo/RESULTS.md` |
| ResNet-18 | Baseline + Strong, seed=42, 40ep | ⏸ Paused (ViT-Ti prioritised) | `../topo/topo_checkpoints/resnet18_*/` |
| EXP_018 | α-scaling: weight norm analysis | ✅ Done | `json/alpha_scaling.json`, `figures/alpha_scaling.png` |
| TinyViT α=0.38 | Predicted optimal run, seed=42, 50ep | ⬜ Ready to launch | `../topo/topo_checkpoints/topo_mid_s42/` |

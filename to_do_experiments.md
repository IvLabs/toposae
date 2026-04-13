# To-Do Experiments — Topo Monosemanticity Paper

> Target: NeurIPS/ICLR Mech Interp Workshop (4-page short paper)
> Created: 2026-04-14

---

## Priority Legend

| Priority | Meaning |
|---|---|
| 🔴 **Must Have** | Paper cannot be submitted without this |
| 🟡 **Should Have** | Paper is weak without this |
| 🟢 **Nice to Have** | Strengthens the paper but optional for workshop |
| 🔵 **Stretch** | For main conference submission or follow-up |

---

## EXP_005: Multi-Seed Replication (ViT-S/16)

**Priority:** 🔴 **Must Have**

**Why:** Everything is seed=42. Reviewers will immediately ask "is this robust?" The H2 L0 reduction of 5% and H3 ratio of 1.17× need statistical significance.

**Seeds:** {42 ✓ done, 123, 456} — 3 seeds total

### ViT-S/16 (12L / 384D / ImageNet-100 / 90 epochs)

**Already done:** seed=42 (3 runs: baseline, topo_weak, topo_strong)
**Remaining:** 6 runs (seeds 123, 456 × 3 variants)

**Compute:** ~6-8h per run on A100 = ~36-48h total

### TinyViT (H1 exploration only)

**Already done:** seed=42 (3 runs)
**Remaining:** 6 runs (seeds 123, 456 × 3 variants) — local RTX 3050, ~2-3h each = ~12-18h

**Purpose:** Only for H1 (monosemanticity) exploration — entropy-based vs SAE-based comparison. Not included in main paper results. Quick local sanity check.

### Per-run analysis (ViT-S/16):

- H2 (SAE) on all 6 new models — 6 SAE runs
- H3 (patching) on all 6 new models — 12 patching runs (cluster + random each)
- H1 deprioritized (null result in seed=42)

**Success criteria:**
- H2: L0 reduction ≥5% with paired t-test p < 0.05 (topo_strong vs baseline)
- H3: Ratio > 1.1 with paired t-test p < 0.05
- Report effect sizes (Cohen's d)

**Output table for paper:**

| Seed | Metric | Baseline | TopoWeak | TopoStrong |
|---|---|---|---|---|
| 42 | L0 | 1061.5 | 1034.6 | 1010.8 |
| 123 | L0 | ... | ... | ... |
| 456 | L0 | ... | ... | ... |
| **Mean ± SE** | **L0** | **X ± Y** | **X ± Y** | **X ± Y** |
| **Cohen's d** | **vs Baseline** | — | **d = ?** | **d = ?** |

---

## EXP_009: Layer-Wise SAE Analysis

**Priority:** 🟡 **Should Have**

**Why:** Currently measuring only layer 6 (middle). Reviewer will ask: "Is the effect concentrated in one layer, or distributed?" Layer-wise analysis shows the *structure* of the effect (Level 2 depth).

**What:**
- For all 3 seeds of ViT-S/16 models (9 models total), extract CLS activations from **all 12 blocks**
- Train SAE on each layer (same config: 4× expansion, L1=0.001, 50 epochs)
- Measure L0 norm, dead features, recon loss per layer, per seed, per variant

**Compute:** 9 models × 12 layers × 3 variants = 324 SAE runs (many can be parallelized). Each ~1h on RTX 3050.

**Output figure:** Line plot — L0 norm (y) vs layer depth (x), 3 lines (baseline, topo_weak, topo_strong) with error bars across seeds. Expected: topo lines diverge from baseline progressively through layers.

---

## EXP_010: Accuracy vs L0 Trade-off Analysis

**Priority:** 🟡 **Should Have**

**Why:** Killer alternative explanation: "TopoStrong is just a weaker model, of course it has lower L0." We need to explicitly test whether L0 reduction is an accuracy confound.

**What:**
- Collect (accuracy, L0) pairs across all seeds and all variants (9 data points from EXP_005: 3 seeds × 3 variants)
- Fit linear regression: L0 ~ accuracy
- Test: Does TopoStrong sit *below* the regression line? If yes → topo drives sparsity beyond accuracy alone.
- If TopoStrong sits *on* the line → reviewer is right, it's just accuracy.

**Compute:** Zero additional compute — uses existing results.

**Output figure:** Scatter plot — accuracy (x) vs L0 (y), with regression line. Color points by variant.

---

## EXP_011: PCA Baseline Comparison

**Priority:** 🟡 **Should Have**

**Why:** Distinguishes our H2 from TopoNets' "lower dimensionality" claim. If PCA dimensionality reduction correlates with SAE L0, reviewer says "you just rediscovered dimensionality." If they diverge, we show SAE captures something different.

**What:**
- On all 3 seeds of ViT-S/16 activations (layer 6, collected during EXP_005):
  - Run PCA, find # components for 95% variance explained
  - Compare to SAE L0 norm (active features per image)
  - If PCA finds similar "effective dimensionality" for all 3 variants but SAE finds different L0 → SAE captures something PCA misses

**Compute:** No training. PCA is fast (~minutes).

**Output:** Table comparing PCA dimensionality vs SAE L0 per model.

---

## EXP_012: SAE-Based Monosemanticity (Better H1)

**Priority:** 🟢 **Nice to Have**

**Why:** Current H1 (entropy-based score) is null. The entropy metric is too blunt. Anthropic's actual approach: count SAE features that are single-concept-selective. This is the *right* way to measure monosemanticity.

**What:**
- Use trained SAEs from EXP_005 (3 seeds × 3 variants = 9 SAEs)
- For each SAE feature, compute its activation distribution across ImageNet-100 classes
- Count features that respond to only 1 class (monosemantic features)
- Compare: baseline vs topo_weak vs topo_strong, aggregated across seeds

**Compute:** No new training. Just analysis on existing SAEs + activations.

**Output:** Bar chart — % monosemantic features per model.

**Risk:** This could also be null. But even a null result is informative — "topo doesn't help monosemanticity even by SAE measures."

---

## EXP_013: α Sweep (Fine-Grained)

**Priority:** 🟢 **Nice to Have**

**Why:** We only tested α ∈ {0, 0.1, 1.0}. The "sweet spot flip" between TinyViT (α=0.1 wins) and ViT-S/16 (α=1.0 wins) is interesting but under-sampled. Adding α=0.01 and α=0.5 would reveal the curve.

**What:**
- Train ViT-S/16 with α ∈ {0.01, 0.5} (seed=42 only, 2 runs)
- Run H2 analysis on each

**Compute:** 2 training runs = ~12-16h on A100.

**Output:** Line plot — L0 norm (y) vs α (x), showing the curve shape.

---

## EXP_014: SAE Feature Correlation Analysis

**Priority:** 🟢 **Nice to Have**

**Why:** Shows *how* topo reduces superposition. If topo features are less correlated with each other, it means they're more independent — directly supporting the "reduced interference" claim.

**What:**
- For each trained SAE (3 seeds × 3 variants = 9 SAEs), compute pairwise feature correlation matrix
- Measure: mean absolute correlation, fraction of highly correlated pairs (|r| > 0.5)
- Compare: baseline vs topo models

**Compute:** Analysis only, no training.

**Output:** Heatmaps of correlation matrices (3 side-by-side). Summary stat table.

---

## EXP_007: ViT-B/16 on ImageNet-100

**Priority:** 🔵 **Stretch**

**Why:** Tests if the effect scales further. If ViT-S/16 shows -5% L0, does ViT-B/16 show -8% or does it saturate?

**What:**
- Train ViT-B/16 (timm `vit_b_16`), 3 variants, 3 seeds (42, 123, 456), 90 epochs
- Run H2 analysis on all 9 models

**Compute:** 9 runs (6 new + seed=42), ~8-10h each on A100 = ~72-90h. Needs 24+ GB VRAM.

**Note:** Only worth doing if ViT-S/16 multi-seed results are strong. Otherwise it's a weak result at larger scale.

---

## EXP_008: Multi-Seed on ViT-B/16

**Priority:** 🔵 **Stretch**

**Why:** Full robustness at largest scale. Only for main conference submission.

**What:** Same as EXP_005 but for ViT-B/16 (3 seeds × 3 variants = 9 runs).

**Compute:** 9 runs × ~10h = 90h on A100. Expensive.

---

## Suggested Execution Order

For a **4-page workshop paper**, here's the minimum viable path:

| Step | Experiment | Duration | Depends On |
|---|---|---|---|
| 1 | **EXP_005** (Multi-seed ViT-S/16: seeds 123, 456) | ~2 days GPU | None |
| 2 | **EXP_009** (Layer-wise SAE, all 3 seeds) | ~1 day local | EXP_005 |
| 3 | **EXP_010** (Accuracy vs L0) | ~2 hours | EXP_005 |
| 4 | **EXP_011** (PCA baseline) | ~2 hours | EXP_005 |
| 5 | **EXP_012** (SAE-based H1) | ~1 day local | EXP_005 |
| — | **Write paper** | ~3 days | Steps 1-5 |
| 6 | EXP_013 (α sweep) | ~1 day GPU | Optional |
| 7 | EXP_014 (Feature correlation) | ~4 hours | EXP_005 |
| 8 | EXP_007 (ViT-B/16) | ~2 days GPU | Optional |

**Total for workshop paper:** ~7-9 days (mostly GPU wait time for EXP_005)

---

## Paper Figure Inventory (4-page target)

| Figure | Content | From Experiment |
|---|---|---|
| **Fig 1** | Conceptual: topo → spatial clustering → fewer co-active features | — |
| **Fig 2** | H2 main result: L0 norm comparison (multi-seed, mean ± SE) | EXP_005 |
| **Fig 3** | Layer-wise L0: topo effect through transformer depth | EXP_009 |
| **Fig 4** | Accuracy vs L0: foreclosing "weaker model" objection | EXP_010 |
| **Fig 5** | H3 patching: causal isolability ratio (multi-seed) | EXP_005 |
| **Table 1** | Training results (accuracy, loss) across seeds | EXP_005 |
| **Table 2** | Statistical summary (t-tests, Cohen's d) | EXP_005 analysis |

**Appendix (if needed):**
- SAE training curves
- Full layer-wise tables
- PCA comparison
- Feature correlation heatmaps
- α sweep results

---

## Compute Budget Summary

| Phase | GPU Hours | Where |
|---|---|---|
| EXP_005 (6 ViT-S/16 runs) | ~48h | A100 / Colab Pro |
| EXP_013 (2 α sweep runs) | ~16h | A100 / Colab Pro |
| EXP_007 (9 ViT-B/16 runs) | ~90h | A100 24GB+ |
| Analysis (SAE × layers, PCA, etc.) | ~24h | RTX 3050 (local) |
| **Workshop minimum** | **~48h GPU + ~24h local** | |
| **Full plan (all experiments)** | **~154h GPU + ~48h local** | |

---

*Last updated: 2026-04-14*

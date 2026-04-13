# Research Ideas & Framing — Topo Monosemanticity

> Notes from analysis discussion — 2026-04-14

---

## 1. Original Hypotheses

| Hypothesis | Claim | Operationalization |
|---|---|---|
| **H1: Monosemanticity** | Topo training reduces polysemantic neurons | Class selectivity → entropy-based monosemanticity score |
| **H2: Superposition** | SAE on topo model needs fewer active features | L0 norm, dead feature fraction, reconstruction loss |
| **H3: Causal Purity** | Patching topo cluster suppresses classification more cleanly | Δlogit ratio: cluster vs random unit patching |
| **H4: Brain Alignment** | Topo cortical sheet predicts fMRI voxel layout | Encoding model, spatial correlation (not done) |

---

## 2. Experiments Completed

| Experiment | Models | Dataset | Status |
|---|---|---|---|
| EXP_001-004 | TinyViT (4L/128D, 50 ep) | ImageNet-100 (128×128) | ✅ |
| EXP_006 | ViT-S/16 (12L/384D, 90 ep) | ImageNet-100 (224×224) | ✅ |
| EXP_005 | Multi-seed (5 seeds × 3 variants) | — | ❌ |
| EXP_007 | ViT-B/16 | — | ❌ |
| EXP_004 (H4) | Brain Alignment (NSD/fMRI) | — | ❌ |

---

## 3. Results Summary

### H1: Monosemanticity

| Model | Baseline | TopoWeak | TopoStrong | Supported? |
|---|---|---|---|---|
| TinyViT | 0.2525 | **0.2599** (+3%) | 0.2491 | ✅ Weakly (TopoWeak only) |
| ViT-S/16 | 0.2331 | 0.2353 | 0.2295 | ❌ No meaningful effect |

**Problem:** Entropy-based score is too blunt. All models ≈ 0.23. The metric treats "responding to 5 related concepts" same as "responding to 5 unrelated ones."

### H2: Superposition (SAE)

| Model | Baseline L0 | TopoWeak L0 | TopoStrong L0 | Supported? |
|---|---|---|---|---|
| TinyViT | 444.9 | 443.7 | **435.5** (-2%) | ✅ Weakly |
| ViT-S/16 | 1061.5 | 1034.6 | **1010.8** (-5%) | ✅ Yes, clearer |

**Strongest signal:** Monotonic L0 reduction with α. Effect strengthens with model scale. Dead features also increase monotonically.

### H3: Causal Purity

| Model | Baseline Ratio | TopoWeak Ratio | TopoStrong Ratio | Supported? |
|---|---|---|---|---|
| TinyViT | 1.19× | **1.42×** | 1.12× | ✅ (TopoWeak) |
| ViT-S/16 | 0.96× | 0.97× | **1.17×** | ✅ (TopoStrong only) |

**Interesting flip:** "Sweet spot" for α shifts with model scale. Larger models need stronger topo pressure for causal clustering.

---

## 4. H2 vs TopoNets Section 3.3 — Critical Distinction

### What TopoNets 3.3 Measures

| Metric | What It Tests |
|---|---|
| **L1 weight pruning** | Remove small weights, measure accuracy drop |
| **Weight downsampling** | Reduce parameter count, measure accuracy drop |
| **Claim** | Topo models have "sparser weight distributions" |

### What Our H2 Measures

| Metric | What It Tests |
|---|---|
| **SAE L0 norm** | How many features active per input |
| **Dead features** | Wasted capacity in feature basis |
| **Claim** | Topo models use fewer simultaneous features per input |

### The Difference

| Aspect | TopoNets 3.3 (Weights) | Our H2 (Activations) |
|---|---|---|
| **What's sparse?** | Model parameters | Feature usage per forward pass |
| **Analysis level** | Static — weight inspection | Dynamic — per-input activation patterns |
| **What it means** | Knowledge stored efficiently | Computation uses fewer features |
| **Mech interp relevance** | Model compression | Feature superposition / polysemanticity |

**Key framing:** TopoNets asks "can you prune this model?" We ask "does this model compute with fewer features simultaneously?" These are orthogonal. A model can have sparse weights but still have every neuron firing for everything.

---

## 5. Broader Impact Framing

### The Problem in Mech Interp Right Now

1. **SAEs are expensive** — Anthropic spent millions. Community needs ways to make models "SAE-friendly" by design.
2. **Superposition is the bottleneck** — it's why we can't interpret LLMs. Features are compressed, entangled, polysemantic.
3. **Current solutions are post-hoc** — SAEs, dictionary learning, sparse probing. All try to *recover* clean features from a messy model.

### What Our Paper Shows

> **Topographic training pressure is a *pre-hoc* approach to monosemanticity.** Instead of training normal + disentangling with SAEs, build disentanglement into training.

### Why This Matters

| If H2 holds | Impact |
|---|---|
| **AI Safety** | Inherently interpretable models — don't need post-hoc SAEs for every new model release |
| **Model Design** | Simple loss term → 5% fewer active features. At GPT-4 scale = millions of parameters saved per forward pass |
| **Neuroscience** | Validates brain's design: topography is a *computational* optimization, not just spatial/wiring optimization |
| **Topography Literature** | Shifts from "topo looks like brain" → "topo makes models more interpretable" — new evaluation paradigm |

### Bridge Paper Positioning

This paper bridges two communities that don't talk:
- **NeuroAI** — care about brain alignment, don't use SAEs
- **Mech Interp** — use SAEs, don't care about topography

**Novelty:** First application of SAEs + activation patching to topographic models.

---

## 6. Risks & Mitigations

### If Results Are Strong

- Lead with H2 as the main result (monotonic, scalable, clean)
- H3 as supporting evidence (causal isolability at larger scales)
- H1 as exploratory (acknowledge entropy metric is blunt)

### If Results Are Null

- Reframe: "Topography alone is not sufficient for monosemanticity"
- Still publishable — negative result narrows the architectural prior search space
- Position as: we tested topo, here's what does/doesn't work

### Reviewer Concerns

1. **"Everything is seed=42"** → EXP_005 (multi-seed) is critical
2. **"Is topo just a weaker model?"** → Accuracy drop 3-5% needs addressing. Could reframe as trade-off: interpretability vs accuracy
3. **"TopoNets already showed sparsity"** → Clearly distinguish weight sparsity (theirs) from activation sparsity (ours)

---

## 7. Suggested Next Steps

1. **EXP_005 — Multi-seed runs** (5 seeds × 3 variants on ViT-S/16)
   - Critical for statistical significance
   - Will tell us if H2 effect is robust or seed-dependent

2. **Better H1 metric** — Use SAE features as monosemanticity measure
   - Count how many SAE features are single-concept-selective
   - This is what Anthropic actually did (not entropy)

3. **PCA baseline comparison** — Show SAE captures something PCA misses
   - SAE finds overcomplete basis (M > N); PCA limited to N components

4. **Layer-wise analysis** — Check if effect concentrates in specific layers

5. **α sweep** — Test α = {0, 0.01, 0.1, 0.5, 1.0} to find optimal pressure

---

## 8. Paper Structure (Tentative)

**Title ideas:**
- "Topographic Training Reduces Feature Superposition in Vision Transformers"
- "Does Topography Help Interpretability? A Mechanistic Analysis of Topographic ViTs"
- "Spatial Organization, Sparse Features: A Mech Interp Study of Topographic Networks"

**Abstract structure:**
1. Problem: Polysemanticity/superposition blocks interpretability
2. Prior: TopoNets shows topo models have spatial organization
3. Gap: Nobody tested if topo reduces feature entanglement
4. Method: Train topo ViTs, analyze with SAEs + patching
5. Results: H2 (L0 ↓), H3 (causal purity at scale), H1 (null)
6. Impact: Pre-hoc approach to monosemanticity

---

## 9. Key References to Cite

### Mech Interp
- Elhage et al. (2022) — Superposition hypothesis
- Bricken et al. (2023) — SAEs for monosemanticity
- Templeton et al. (2024) — Scaling SAEs to Claude 3

### Topography
- Deb et al. (2025) — TopoNets (ICLR 2025 Spotlight)
- Rathi et al. (2025) — TopoLM (ICLR 2025 Oral)
- Margalit et al. (2024) — TDANN unified framework

### Neuroscience
- Allen et al. (2022) — NSD fMRI dataset
- Hubel & Wiesel — Original topographic maps (if space)

---

*Last updated: 2026-04-14*

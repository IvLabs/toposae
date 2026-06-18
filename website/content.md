# Content — copy for every section

Use this text as-is. **Do not paraphrase numbers.** Every statistic here is taken
from the paper; if a number changes in the paper, change it here too.

---

## Meta

- **Title:** Topographic Training Concentrates Causal Circuits Without Improving Neuron Monosemanticity
- **Authors:** Gautam Ranka, Shubham Pandere\*, Aiden Ross D'souza\*  (\* equal contribution)
- **Affiliation:** IvLabs, VNIT Nagpur, India
- **Venue:** Mechanistic Interpretability Workshop at the 43rd International Conference on Machine Learning (ICML 2026), Seoul, South Korea
- **Correspondence:** gautam.ranka@ivlabs.in · shubham.pandere@ivlabs.in · aiden.dsouza@ivlabs.in
- **Links:** Paper (PDF) · Code: https://github.com/IvLabs/toposae · Video (manim walkthrough)

## TL;DR (hero, one bold sentence)

> A sub-0.1k-parameter training-time penalty (TopoLoss) concentrates a vision
> transformer's causal circuits into tight spatial clusters — **2.79× more causally
> sufficient** than random neuron sets — yet leaves individual neurons no more
> monosemantic. Interpretability gains here are a property of *circuits*, not *neurons*.

## Headline stat cards (hero)

| Number   | Label                                              |
|----------|----------------------------------------------------|
| `2.79×`  | causal sufficiency of topographic clusters (vs random) |
| `d = 0.95` | effect size for the H3 result (p < 0.0001)        |
| `<0.1k`  | extra parameters added by TopoLoss                 |

---

## Abstract (verbatim from paper)

Mechanistic interpretability of vision transformers seeks to decompose model
computation into human-readable units, but learned representations entangle many
concepts in each neuron. Feature superposition is widely treated as the central
obstacle to this decomposition, yet most mitigations (sparse autoencoders,
dictionary learning) are post-hoc and leave the underlying network unchanged. We
ask whether a spatial-locality training loss (TopoLoss) can act as a lightweight,
training-time prior that improves interpretability of standard mech-interp tools.
Training ViT on ImageNet-100 across multiple TopoLoss weights α, we measure causal
sufficiency of topographic clusters via activation patching and feature geometry
via sparse autoencoders fit to the same residual stream. At α = 1.0, topographic
clusters are 2.79× more causally sufficient than random unit sets of the same size,
with the effect increasing monotonically in α. SAE L0 sparsity decreases by 11% and
dead-feature fraction rises 19-fold, yet standard neuron-level monosemanticity
scores are unchanged, indicating that topographic pressure acts at circuit level,
concentrating causal mass into spatially local structures without disentangling
individual neurons. This dissociation suggests current neuron-level monosemanticity
metrics are insensitive to a class of real interpretability gains, and positions
cheap architectural priors as a viable training-time complement to post-hoc tooling.

---

## Method (TopoLoss)

**Equation (KaTeX):**

`\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{topo}}, \qquad \mathcal{L}_{\text{topo}} = \alpha \sum_{(i,j)\in\mathcal{E}} \lVert w_i - w_j \rVert^2`

**Gloss (write ~3 sentences):**
- TopoLoss adds a weight-smoothness penalty between units that are *adjacent* on a
  fixed 2D topographic grid, pushing functionally similar units to co-locate —
  echoing cortical maps in biological vision (following Khosla et al., 2025).
- Applied to the attention-projection weights (`attn.proj`) of all 12 ViT-S/16
  blocks; units are arranged on a √d × √d grid matching the head dimension.
- α controls pressure strength. It adds fewer than 0.1k parameters and costs about
  1.6 percentage points of accuracy at the strongest setting (α = 1.0).

**Setup line:** ViT-S/16 trained on ImageNet-100, α ∈ {0.0, 0.1, 1.0}, three seeds
{42, 123, 456}. A 4-layer TinyViT (d = 128) serves as a preliminary scale probe.

---

## Three questions (card row)

| Hyp. | Question | Metric | Verdict |
|------|----------|--------|---------|
| **H1 — Monosemanticity** | Do individual neurons become more selective? | Entropy-based selectivity `Mᵤ` | **NULL** |
| **H2 — Superposition** | Does SAE-measured feature complexity decrease? | SAE `L0` norm + dead-feature fraction | **SUPPORTED \*** (accuracy caveat) |
| **H3 — Causal purity** | Are topographic clusters more causally sufficient than random unit sets? | Activation-patching logit-delta ratio | **SUPPORTED** |

---

## H3 — Causal circuits concentrate (lead result)

**Headline:** Topographic clusters carry disproportionate causal weight.

- At α = 1.0, topographic clusters reach a patching ratio of **2.79 ± 1.24** vs
  **1.68 ± 0.42** at baseline — a 66% increase.
- The effect is **monotone in α: 1.68 → 2.27 → 2.79** (α = 0.0, 0.1, 1.0).
- Multi-class expansion (n = 60, 20 classes × 3 seeds): **p < 0.0001, d = 0.95**.
- Robust across all three seeds independently (per-seed ratios at α = 1.0: 2.53,
  2.89, 2.97 — no single outlier seed drives it).
- **Spatial coherence:** the top-k = 16 class units at α = 1.0 are **6.1% more
  spatially concentrated** than baseline (mean grid distance 9.948 vs 10.599;
  p = 0.0005, d = 0.66), confirming the selected units really do co-localize.

**α-toggle data (for the segmented control):**

| α    | Patch ratio (cluster / random \|Δlogit\|) | Caption                          |
|------|-------------------------------------------|----------------------------------|
| 0.0  | 1.68 ± 0.42                               | Baseline — near the random line  |
| 0.1  | 2.27 ± 0.93                               | Weak topography (+0.59 vs base)  |
| 1.0  | 2.79 ± 1.24                               | Strong topography (+1.12 vs base)|

---

## H2 — Superposition drops, with a caveat

**Headline:** SAEs find sparser, more specialized features — but accuracy is a confound.

- At α = 1.0, SAE **L0 decreases 11%** (1133.8 vs 1273.9; p = 0.007, d = 7.06).
- **Dead-feature fraction rises 19-fold** (3.8% vs 0.2%).
- The α = 0.1 model shows neither effect reliably (threshold between 0.1 and 1.0).
- The L0 reduction is distributed across depth (layers 3–10, peak at layer 10),
  not localized to one block.

**Caveat box (label: "Caveat"):**
> We treat the L0 reduction as suggestive, not confound-free. The α = 1.0 model
> costs about 1.6 pp accuracy (61.2% → 59.6%), and an OLS regression of L0 on
> validation accuracy across all 9 ViT-S checkpoints yields R² = 0.72 (p = 0.004) —
> accuracy explains substantial L0 variance. Two signs the effect isn't purely
> accuracy-driven (a PCA-dimensionality dissociation at α = 0.1, and seeds below the
> regression line) are reported in the paper. An accuracy-matched ablation is left
> to future work.

---

## H1 — Neurons don't change (the null)

**Headline:** Individual neurons are no more monosemantic.

- Entropy-based selectivity `Mᵤ` is **flat across α**: 0.234 ± 0.002 (α = 0) vs
  0.236 ± 0.002 (α = 1.0); Δ = 0.0019, not significant.
- The SAE-feature-level `Mᵤ` is **also null**, including at the larger 25k-image
  evaluation (no features reach the monosemantic threshold).
- **Interpretation:** monosemanticity measures how one neuron spreads across
  classes — a marginal statistic insensitive to the *joint* arrangement of many
  neurons. TopoLoss reorganizes neurons into functionally coherent spatial clusters
  (H3) without changing per-neuron selectivity (H1). Consistent with superposition:
  spatial co-location, not per-neuron selectivity, is the relevant unit of circuit purity.

---

## Takeaway

A penalty of fewer than 0.1k parameters, added to the training loss, systematically
shifts circuit *geometry* in a way existing tools can measure (d = 0.95 for H3) — at
a modest cost (−1.6 pp accuracy at α = 1.0) and without disentangling individual
neurons. This positions cheap architectural priors as a viable *training-time
complement* to post-hoc tooling like SAEs, and suggests current neuron-level
monosemanticity metrics are blind to a class of real interpretability gains.

**Scale caveat:** In a 4-layer TinyViT the H2 trend replicates (L0 −2.1% at α = 1.0)
but H3 saturates at just 1.42× (α = 0.1), far below the ViT-S result of 2.79× —
consistent with a depth/scale requirement for circuit formation, though a
controlled depth-sweep is needed to confirm. Limitations: a single dataset
(ImageNet-100), a partial accuracy confound on H2, and a single seed for TinyViT.

---

## Video

Caption: "A short animated walkthrough of TopoLoss and the H1/H2/H3 results."
Placeholder text until the render is added: "Video walkthrough coming soon."

---

## BibTeX (cite block)

```bibtex
@inproceedings{ranka2026topographic,
  title     = {Topographic Training Concentrates Causal Circuits Without Improving Neuron Monosemanticity},
  author    = {Ranka, Gautam and Pandere, Shubham and D'souza, Aiden Ross},
  booktitle = {Mechanistic Interpretability Workshop at the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
> Note: confirm the exact `booktitle`/proceedings string and add a DOI/URL once the
> workshop finalizes them.

---

## Glossary (for metric tooltips)

- **α (alpha):** weight on the TopoLoss term; 0 = baseline, 1.0 = strong topography.
- **L0:** number of SAE features active per token — lower means sparser, more
  specialized representations.
- **Mᵤ:** entropy-based neuron monosemanticity / selectivity score; 1 = perfectly
  selective to one class, higher = more selective.
- **d (Cohen's d):** standardized effect size; ~0.5 medium, ~0.8+ large.
- **Patching ratio:** causal effect (|Δlogit|) of patching a topographic cluster
  divided by that of patching a same-size random unit set; >1 means the cluster
  carries disproportionate causal weight.
- **SAE:** sparse autoencoder, a post-hoc tool that decomposes activations into
  sparse, interpretable features.

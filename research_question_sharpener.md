# Research Question Sharpener

> Filled out: 2026-04-14

---

## 1 THE WHAT

### What are you claiming or showing?

We show that topographic training pressure — a spatially smooth loss applied to Vision Transformer units — **reduces feature superposition in the model's internal representations**. Specifically, a Sparse Autoencoder trained on a topographic ViT-S/16 recovers a feature basis with **5% fewer simultaneously active features per image** (L0 norm) compared to an identically trained baseline. At the same time, topographic models show **causal isolability**: selective patching of topo-induced feature clusters meaningfully impacts classification logits (1.17× vs 0.96× for baseline), suggesting topo colocalizes causally important circuitry. This matters because superposition is the primary bottleneck to mechanistic interpretability in modern neural networks — if a simple training-time prior can reduce it, we can build more interpretable models by design rather than relying on expensive post-hoc disentanglement.

### The Surprise

TopoNets (Deb et al., 2025, ICLR Spotlight) showed that topographic training produces spatially organized features and weight-level sparsity (resilience to pruning). **Nobody tested whether it reduces feature entanglement during computation.** The community assumes topo is about spatial organization and brain alignment — we show it also changes *how the model computes*, not just *where features sit*. Specifically, who would update:

- **TopoNets authors** would learn their method does more than they claimed (activation-level sparsity, not just weight sparsity)
- **Mech interp researchers** (Anthropic, EleutherAI) who currently assume topo is irrelevant to superposition would see it as a potential pre-hoc solution
- **NeuroAI researchers** would see topography as a *computational* optimization, not just a wiring/biological fidelity optimization

### One-Sentence Version

**Topographic training reduces feature superposition in ViTs, producing internally sparser feature representations and more causally isolable circuits — a pre-hoc approach to model interpretability.**

### Alignment with LossFunk

*(Needs manual check against lossfunk.com — but the core topic aligns with: mechanistic interpretability, representation learning, neural network feature geometry, and brain-inspired model design.)*

---

## 2 THE WHY — Fruitfulness

### High fruitfulness (crossroads)

This is a **crossroads result**. If topo genuinely reduces superposition, it opens:

1. **Does the effect scale?** → Apply to ViT-L, ViT-H, LLMs. Does L0 reduction grow or shrink with model size?
2. **Can we optimize α per layer?** → Instead of fixed α, learn layer-specific topo pressure. Does early-layer topo help more than late-layer?
3. **Does topo + SAE beat SAE-only on interpretability benchmarks?** → Can a topo model be interpreted with a smaller/cheaper SAE? Does feature visualization work better?
4. **Is topo a substitute for or complement to MoE?** → Both route computation. Does topo achieve similar feature isolation without the architectural complexity?

### New questions this result would open

| Question | Interesting because of our finding, or pre-existing? |
|---|---|
| Does topo work on LLMs the same way? | Pre-existing (TopoLM exists), but our SAE angle is new |
| Is there an optimal α per model size? | **New** — we found the "sweet spot" flips between TinyViT (α=0.1) and ViT-S/16 (α=1.0) |
| Can topo replace post-hoc SAEs? | **New** — nobody tested topo as a pre-hoc interpretability method |
| Does topo improve feature visualization quality? | **New** — direct downstream application of our finding |

### Who would build on this?

- **EleutherAI / mechanistic interp community** — they're actively searching for model design choices that reduce superposition
- **TopoNets / TopoLM authors** — our result extends their method's evaluation paradigm
- **NeuroAI labs** (McDermott, Yamins, Khaligh-Razavi) — bridges brain alignment with interpretability
- **AI safety researchers** (Anthropic, Redwood) — if topo makes models more interpretable by design, it's a safety-relevant training modification

### Crossroads or corridor?

**Crossroads.** It gives:
- **NeuroAI** a new reason to use topo (interpretability, not just brain alignment)
- **Mech interp** a new pre-hoc tool (topo training, not just post-hoc SAEs)
- **Model design** a new knob (α) for the accuracy-interpretability trade-off

---

## 3 THE HOW

### Killer Alternative Explanation

**Skeptic:** "Your topo model is just weaker (lower accuracy) — of course it has fewer active features. A smaller/dumber model is easier to interpret."

**Control that forecloses this:** We explicitly measure the accuracy-interpretability trade-off. TopoStrong has -5% L0 but also -1.5% accuracy. If L0 reduction were purely an accuracy effect, the relationship would be linear — but we find that TopoWeak has similar accuracy to baseline but no L0 improvement, while TopoStrong shows the L0 effect. This suggests the topo loss itself drives sparsity, not just weaker performance.

**Second skeptic escape route:** "This is seed-dependent. Run 5 seeds and it disappears."

**Control:** EXP_005 — multi-seed replication (5 seeds × 3 variants = 15 runs).

### Experimental Design

| Component | Details |
|---|---|
| **Models** | ViT-S/16 (timm), 3 variants: Baseline (α=0), TopoWeak (α=0.1), TopoStrong (α=1.0) |
| **Dataset** | ImageNet-100 (126K train, 5K val) |
| **Training** | 90 epochs, AdamW lr=5e-4, batch=128, same seed across variants |
| **H1 Measure** | Monosemanticity score (class selectivity → normalized entropy) |
| **H2 Measure** | SAE (4× expansion, L1=0.001, 50 epochs) → L0 norm, dead features, recon loss |
| **H3 Measure** | Activation patching: selective cluster vs random, Δlogit ratio |
| **Success criteria** | H2: L0 reduction ≥5% with p<0.05 across seeds. H3: ratio >1.1 with p<0.05 |
| **Robustness** | 5 seeds (EXP_005), layer-wise analysis, α sweep (0, 0.01, 0.1, 0.5, 1.0) |

**Feasibility:** Each ViT-S/16 training run takes ~6-8h on A100. Analysis runs on local RTX 3050. Already completed seed=42 for TinyViT + ViT-S/16.

### What Rigor Looks Like Here

- **5 seeds** with paired t-tests between topo and baseline
- **Effect sizes** (Cohen's d) for H1 and H2
- **Ablation:** Same architecture, same data, same training recipe — only α varies
- **Layer-wise breakdown:** Show effect at each transformer block, not just middle layer
- **PCA baseline:** Compare SAE L0 to PCA dimensionality — show SAE captures something PCA misses
- **Pre-registration:** Document hypotheses and success criteria before running EXP_005

---

## 4 THE SO WHAT

### Depth Level

| Level | Can our design produce it? |
|---|---|
| **0** — "We propose X, it works" | Yes, trivially |
| **1** — "X works + strong evidence" | Yes — if multi-seed holds |
| **2** — "WHY X works + reveals structure" | **Target level** — we show topo → spatial clustering → fewer co-active features → causal isolability. This is a mechanism chain. |
| **3** — "General principle that reframes" | Potential — if we show "spatial priors reduce superposition" generalizes beyond topo to other architectural priors |

**Current design is Level 2.** We can explain *why* topo reduces superposition (spatial proximity → weight similarity → fewer conflicting features per unit → lower L0). To reach Level 3, we'd need to show this applies to other spatial priors beyond topo.

### Impact Type

- **■ Conceptual** — Changes how experts think about topography: from "brain-like spatial organization" to "computational feature disentanglement"
- **■ Practical** — Enables pre-hoc interpretability: train topo models that are easier to analyze with SAEs
- **■ Methodological** — First application of SAEs + activation patching to topographic models; others can adopt this evaluation pipeline

### Broadest Truthful Audience

- **AI safety / alignment researchers** — If topo makes models more interpretable by design, it's a training-time safety intervention
- **Model compression researchers** — TopoNets already showed weight sparsity; we add activation sparsity
- **Computational neuroscientists** — Validates the hypothesis that cortical topography serves a computational purpose (reducing interference), not just a wiring purpose
- **LLM architects** — If topo works on ViTs, does it work on transformers generally? Could this be applied to LLMs?

### The 'Science Version'

**Title:** *"Brain-like brain organization makes AI interpretable"*

**Press release lead:** *"Neural networks trained with a brain-inspired spatial organization develop cleaner, less entangled internal representations — making them easier to interpret and potentially safer to deploy. The finding suggests the brain's topographic maps serve not just to minimize wiring, but to reduce computational interference between features."*

---

*Last updated: 2026-04-14*

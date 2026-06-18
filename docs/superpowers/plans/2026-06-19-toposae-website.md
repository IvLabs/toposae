# TopoSAE Project-Page Website Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a static, single-page academic project website for the TopoSAE paper, served from `docs/` on GitHub Pages.

**Architecture:** Three files (`docs/index.html`, `docs/style.css`, `docs/main.js`) plus a `docs/assets/` directory. No build step, no framework, no external state. Each file has a single responsibility: structure, presentation, and progressive enhancement respectively. JS is strictly additive — page must be complete and readable with it disabled.

**Tech Stack:** Vanilla HTML5, CSS custom properties, ES2020 vanilla JS, KaTeX (CDN, no install), `ffmpeg` (for asset prep only).

**Spec reference:** `website/SPEC.md`, `website/design-system.md`, `website/content.md`, `website/asset-manifest.md`.

---

## Chunk 1: Assets + Scaffold

### Task 1: Prepare figure assets

**Files:**
- Create: `docs/assets/` directory and all figure files

- [ ] **Step 1.1: Create the docs directory tree**
```bash
mkdir -p docs/assets
touch docs/.nojekyll
```

- [ ] **Step 1.2: Copy and verify figures exist at source**
```bash
ls results/figures/exp016_h3_multiclass.png \
      results/figures/exp016_h3_multiclass.pdf \
      results/figures/layerwise_sae_l0.png \
      results/figures/layerwise_sae_l0.pdf \
      results/figures/spatial_coherence.png \
      results/figures/monosemanticity_summary.png
```
Expected: all six paths print without error.

- [ ] **Step 1.3: Copy figures into docs/assets/**
```bash
cp results/figures/exp016_h3_multiclass.png  docs/assets/fig2_h3_patching.png
cp results/figures/exp016_h3_multiclass.pdf  docs/assets/fig2_h3_patching.pdf
cp results/figures/layerwise_sae_l0.png      docs/assets/fig3_layerwise_sae.png
cp results/figures/layerwise_sae_l0.pdf      docs/assets/fig3_layerwise_sae.pdf
cp results/figures/spatial_coherence.png     docs/assets/spatial_coherence.png
cp results/figures/monosemanticity_summary.png docs/assets/monosemanticity_summary.png
```

> **Note on Fig 1 (overview 3-panel):** The paper's Figure 1 is a composite panel.
> Check if `results/figures/` contains a single composite PNG. If not, use
> `patching_comparison.png` and `sae_comparison.png` side-by-side in the Method
> section markup, or ask the author to export the composite. This task does not
> block the rest of the plan — skip Fig 1 temporarily and use a placeholder.

- [ ] **Step 1.4: Copy paper PDF**
```bash
cp drafts/new_paper/draft_extracted/paper.pdf docs/assets/toposae.pdf
```

- [ ] **Step 1.5: Encode video for web delivery**

Requires `ffmpeg`. If not installed: `sudo apt install ffmpeg`.

```bash
ffmpeg -i media/videos/manim_animation/1080p60/TopoSAEPresentation.mp4 \
  -vf scale=1280:-2 -c:v libx264 -crf 22 -preset slow \
  -movflags +faststart docs/assets/toposae_walkthrough.mp4

ffmpeg -i media/videos/manim_animation/1080p60/TopoSAEPresentation.mp4 \
  -vframes 1 -q:v 2 docs/assets/video_poster.jpg
```
Expected: two files created. Check sizes — the video should be well under 1GB; if it is above 500MB, lower CRF to 26 for a smaller file.

- [ ] **Step 1.6: Verify assets directory**
```bash
ls -lh docs/assets/
```
Expected: at minimum `toposae.pdf`, `fig2_h3_patching.png`, `fig2_h3_patching.pdf`, `fig3_layerwise_sae.png`, `fig3_layerwise_sae.pdf`, `toposae_walkthrough.mp4`, `video_poster.jpg`.

- [ ] **Step 1.7: Commit**
```bash
git add docs/.nojekyll docs/assets/
git commit -m "feat(website): add prepared assets for project page"
```

---

### Task 2: HTML skeleton

**Files:**
- Create: `docs/index.html`

Build the page structure in one pass — all sections, correct IDs, no inline styles. No content yet except section headings. The goal is a valid, complete HTML document with landmarks and anchor targets.

- [ ] **Step 2.1: Create `docs/index.html` with this exact content**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Topographic Training Concentrates Causal Circuits — TopoSAE</title>
  <meta name="description" content="A sub-0.1k-parameter training-time penalty concentrates a ViT's causal circuits 2.79× without improving neuron monosemanticity. ICML 2026 Mech-Interp Workshop.">
  <link rel="stylesheet" href="style.css">
  <!-- KaTeX for equations -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]})"></script>
</head>
<body>

<!-- ── STICKY NAV ──────────────────────────────────────────── -->
<header class="site-nav" role="banner">
  <div class="nav-inner">
    <a class="nav-wordmark" href="#top">TopoSAE</a>
    <nav aria-label="Page sections">
      <a href="#abstract">Abstract</a>
      <a href="#method">Method</a>
      <a href="#results">Results</a>
      <a href="#video">Video</a>
      <a href="#cite">Cite</a>
      <a href="https://github.com/IvLabs/toposae" class="nav-github" aria-label="GitHub repository" target="_blank" rel="noopener">
        <!-- GitHub icon inline SVG inserted by Task 3 -->
        <svg aria-hidden="true" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/></svg>
      </a>
    </nav>
  </div>
</header>

<main id="top">

  <!-- ── 0 · HERO ─────────────────────────────────────────── -->
  <section class="section-hero" aria-labelledby="hero-title">
    <div class="container">
      <h1 id="hero-title">Topographic Training Concentrates Causal Circuits Without Improving Neuron Monosemanticity</h1>

      <p class="authors">
        Gautam Ranka,
        <span class="equal">Shubham Pandere<sup>*</sup></span>,
        <span class="equal">Aiden Ross D'souza<sup>*</sup></span>
      </p>
      <p class="affiliation">IvLabs, VNIT Nagpur, India &nbsp;·&nbsp; <sup>*</sup>Equal contribution</p>
      <p class="venue">Mechanistic Interpretability Workshop @ ICML 2026, Seoul</p>

      <div class="hero-buttons">
        <a href="assets/toposae.pdf" class="btn btn-primary" target="_blank" rel="noopener">📄 Paper</a>
        <a href="https://github.com/IvLabs/toposae" class="btn btn-secondary" target="_blank" rel="noopener">💻 Code</a>
        <a href="#video" class="btn btn-secondary">▶ Video</a>
      </div>

      <p class="tldr"><strong>TL;DR:</strong> A sub-0.1k-parameter training-time penalty (TopoLoss) concentrates a vision transformer's causal circuits into tight spatial clusters — <strong>2.79× more causally sufficient</strong> than random neuron sets — yet leaves individual neurons no more monosemantic. Interpretability gains here are a property of <em>circuits</em>, not <em>neurons</em>.</p>

      <div class="stat-cards" aria-label="Key results">
        <div class="stat-card">
          <span class="stat-number" data-target="2.79" data-suffix="×" data-decimals="2">2.79×</span>
          <span class="stat-label">causal sufficiency of topographic clusters vs. random</span>
        </div>
        <div class="stat-card">
          <span class="stat-number" data-target="0.95" data-prefix="d = " data-decimals="2">d = 0.95</span>
          <span class="stat-label">effect size (p &lt; 0.0001)</span>
        </div>
        <div class="stat-card">
          <span class="stat-number" data-target="0.1" data-prefix="&lt;" data-suffix="k" data-decimals="1">&lt;0.1k</span>
          <span class="stat-label">extra parameters added by TopoLoss</span>
        </div>
      </div>
    </div>
  </section>

  <!-- ── 1 · ABSTRACT ─────────────────────────────────────── -->
  <section id="abstract" class="section-alt" aria-labelledby="abstract-heading">
    <div class="container container-prose">
      <h2 id="abstract-heading">Abstract</h2>
      <p>
        Mechanistic interpretability of vision transformers seeks to decompose model
        computation into human-readable units, but learned representations entangle many
        concepts in each neuron. Feature superposition is widely treated as the central
        obstacle to this decomposition, yet most mitigations (sparse autoencoders,
        dictionary learning) are post-hoc and leave the underlying network unchanged.
        We ask whether a spatial-locality training loss (TopoLoss) can act as a
        lightweight, training-time prior that improves interpretability of standard
        mech-interp tools. Training ViT on ImageNet-100 across multiple TopoLoss
        weights α, we measure causal sufficiency of topographic clusters via activation
        patching and feature geometry via sparse autoencoders fit to the same residual
        stream. At α = 1.0, topographic clusters are 2.79× more causally sufficient
        than random unit sets of the same size, with the effect increasing monotonically
        in α. SAE L0 sparsity decreases by 11% and dead-feature fraction rises 19-fold,
        yet standard neuron-level monosemanticity scores are unchanged, indicating that
        topographic pressure acts at circuit level, concentrating causal mass into
        spatially local structures without disentangling individual neurons. This
        dissociation suggests current neuron-level monosemanticity metrics are
        insensitive to a class of real interpretability gains, and positions cheap
        architectural priors as a viable training-time complement to post-hoc tooling.
      </p>
    </div>
  </section>

  <!-- ── 2 · METHOD ───────────────────────────────────────── -->
  <section id="method" aria-labelledby="method-heading">
    <div class="container">
      <h2 id="method-heading">Method: TopoLoss</h2>
      <div class="equation-block">
        $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{topo}}, \qquad \mathcal{L}_{\text{topo}} = \alpha \sum_{(i,j)\in\mathcal{E}} \lVert w_i - w_j \rVert^2$$
      </div>
      <div class="method-gloss">
        <p>
          TopoLoss adds a weight-smoothness penalty between units that are <em>adjacent</em>
          on a fixed 2D topographic grid, pushing functionally similar units to co-locate —
          echoing cortical maps in biological vision. Applied to the attention-projection
          weights (<code>attn.proj</code>) of all 12 ViT-S/16 blocks; units are arranged
          on a $\sqrt{d} \times \sqrt{d}$ grid matching the head dimension.
          <abbr class="tooltip" data-tip="Weight on the TopoLoss term; 0 = baseline, 1.0 = strong topography.">α</abbr>
          controls pressure strength: it adds fewer than 0.1k parameters and costs
          about 1.6 pp accuracy at the strongest setting (α = 1.0).
        </p>
        <p class="method-setup">
          <strong>Setup:</strong> ViT-S/16 trained on ImageNet-100, α ∈ {0.0, 0.1, 1.0},
          three seeds {42, 123, 456}. A 4-layer TinyViT (d = 128) serves as a preliminary
          scale probe.
        </p>
      </div>

      <figure class="fig-block fig-wide">
        <!-- If the composite Fig 1 PNG is available, swap this for that file -->
        <div class="fig-placeholder">
          <p>Figure 1 — Overview panels (TopoLoss schematic · H2 SAE sparsity · H3 causal purity).<br>
          <em>Author: replace this placeholder with the composite Fig 1 PNG from the paper.</em></p>
        </div>
        <figcaption>
          <strong>Figure 1.</strong> <em>(a)</em> TopoLoss penalises weight differences between adjacent units,
          pushing functionally similar units to co-localise. <em>(b)</em> SAE L0 (mean active features)
          decreases monotonically with α, while dead-feature fraction rises 19-fold at α = 1.0.
          <em>(c)</em> Per-class causal patching ratios. Topographic clusters at α = 1.0 are
          2.79× more causally sufficient than random unit sets (p &lt; 0.0001, d = 0.95).
        </figcaption>
      </figure>
    </div>
  </section>

  <!-- ── 3 · THREE QUESTIONS ──────────────────────────────── -->
  <section id="results" class="section-alt" aria-labelledby="results-heading">
    <div class="container">
      <h2 id="results-heading">Three Interpretability Axes</h2>
      <p class="results-intro">We audit TopoLoss on three independent questions about what changes when you impose topographic structure at training time.</p>
      <div class="hypothesis-cards">
        <div class="h-card">
          <div class="h-tag">H1</div>
          <h3>Monosemanticity</h3>
          <p>Do individual neurons become more class-selective?</p>
          <p class="h-metric">Metric: entropy-based selectivity <abbr class="tooltip" data-tip="Normalised negative entropy of a neuron's class-activation distribution; 1 = perfectly selective.">Mᵤ</abbr></p>
          <span class="verdict verdict-null">NULL</span>
        </div>
        <div class="h-card">
          <div class="h-tag">H2</div>
          <h3>Superposition</h3>
          <p>Does SAE-measured feature complexity decrease?</p>
          <p class="h-metric">Metric: SAE <abbr class="tooltip" data-tip="Number of SAE features active per token — lower means sparser, more specialized representations.">L0</abbr> norm + dead-feature fraction</p>
          <span class="verdict verdict-warn">SUPPORTED *</span>
        </div>
        <div class="h-card">
          <div class="h-tag">H3</div>
          <h3>Causal Purity</h3>
          <p>Are topographic clusters more causally sufficient than random sets?</p>
          <p class="h-metric">Metric: activation-patching <abbr class="tooltip" data-tip="Causal effect of patching a topographic cluster divided by that of a same-size random set; &gt;1 means disproportionate causal weight.">logit-delta ratio</abbr></p>
          <span class="verdict verdict-pos">SUPPORTED</span>
        </div>
      </div>
    </div>
  </section>

  <!-- ── 4 · H3 (lead result) ─────────────────────────────── -->
  <section id="h3" aria-labelledby="h3-heading">
    <div class="container">
      <div class="result-header">
        <h2 id="h3-heading">H3 — Causal circuits concentrate</h2>
        <span class="verdict verdict-pos">SUPPORTED</span>
      </div>
      <p>
        Topographic clusters carry disproportionate causal weight. At α = 1.0,
        topographic clusters reach a patching ratio of <strong>2.79 ± 1.24</strong> vs
        <strong>1.68 ± 0.42</strong> at baseline — a 66% increase.
        The effect is <strong>monotone in α</strong> and robust across all three seeds
        independently (per-seed ratios: 2.53, 2.89, 2.97).
        Multi-class expansion (n = 60): <strong>p &lt; 0.0001,
        <abbr class="tooltip" data-tip="Standardised effect size; ~0.5 medium, ~0.8+ large.">d</abbr> = 0.95</strong>.
      </p>

      <!-- α-toggle -->
      <div class="alpha-toggle" role="group" aria-label="Select α value to see result">
        <button class="alpha-btn active" data-alpha="0.0">α = 0.0</button>
        <button class="alpha-btn" data-alpha="0.1">α = 0.1</button>
        <button class="alpha-btn" data-alpha="1.0">α = 1.0</button>
      </div>
      <div class="alpha-readout" aria-live="polite">
        <span class="alpha-ratio">1.68 ± 0.42</span>
        <span class="alpha-caption">Baseline — near the random line</span>
      </div>

      <figure class="fig-block">
        <a href="assets/fig2_h3_patching.pdf" target="_blank" rel="noopener" aria-label="Open Figure 2 full-resolution PDF">
          <img src="assets/fig2_h3_patching.png"
               alt="Scatter plot of per-class causal patch ratios for Baseline (α=0), TopoWeak (α=0.1), and TopoStrong (α=1.0). Mean ratio rises monotonically: 1.68, 2.27, 2.79. Both TopoWeak and TopoStrong are significantly above baseline (p<0.0001). The dashed line marks ratio=1 (random baseline)."
               loading="lazy">
        </a>
        <figcaption>
          <strong>Figure 2.</strong> H3: Causal patching ratios across 20 classes × 3 seeds (n = 60).
          Each point is the mean cluster/random logit-delta ratio for one class × seed combination.
          Both TopoWeak (α = 0.1) and TopoStrong (α = 1.0) are significantly above baseline (p &lt; 0.0001,
          Cohen's d ≥ 0.61). The dashed line marks the random-cluster baseline (ratio = 1).
        </figcaption>
      </figure>

      <p class="supporting">
        <strong>Spatial coherence:</strong> The top-k = 16 class units at α = 1.0 are
        6.1% more spatially concentrated than baseline (mean grid distance 9.948 vs 10.599;
        p = 0.0005, d = 0.66), confirming the selected units genuinely co-localise on
        the topographic sheet.
      </p>
    </div>
  </section>

  <!-- ── 5 · H2 ────────────────────────────────────────────── -->
  <section id="h2" class="section-alt" aria-labelledby="h2-heading">
    <div class="container">
      <div class="result-header">
        <h2 id="h2-heading">H2 — Superposition drops</h2>
        <span class="verdict verdict-warn">SUPPORTED *</span>
      </div>
      <p>
        At α = 1.0, SAE <abbr class="tooltip" data-tip="Number of SAE features active per token — lower means sparser, more specialized representations.">L0</abbr>
        decreases <strong>11%</strong> (1133.8 vs 1273.9; p = 0.007,
        <abbr class="tooltip" data-tip="Standardised effect size.">d</abbr> = 7.06),
        and <strong>dead-feature fraction rises 19-fold</strong> (3.8% vs 0.2%).
        The L0 reduction is distributed across depth (layers 3–10, peak at layer 10).
        The α = 0.1 model shows neither effect reliably, confirming a threshold.
      </p>

      <div class="caveat-box" role="note" aria-label="Caveat">
        <strong>Caveat:</strong> We treat the L0 reduction as suggestive, not
        confound-free. The α = 1.0 model costs ~1.6 pp accuracy (61.2% → 59.6%),
        and OLS regression of L0 on validation accuracy across all 9 ViT-S checkpoints
        yields R² = 0.72 (p = 0.004) — accuracy explains substantial L0 variance. Two
        signs the effect isn't purely accuracy-driven are reported in the paper. An
        accuracy-matched ablation is left to future work.
      </div>

      <figure class="fig-block">
        <a href="assets/fig3_layerwise_sae.pdf" target="_blank" rel="noopener" aria-label="Open Figure 3 full-resolution PDF">
          <img src="assets/fig3_layerwise_sae.png"
               alt="Two line charts showing SAE L0 and dead-feature fraction per ViT layer (0–11) for Baseline (α=0, blue), TopoWeak (α=0.1, orange dashed), and TopoStrong (α=1.0, green). L0 for α=1.0 diverges below baseline at layer 3 and remains lower through layer 10, with peak gap at layer 10. Dead-feature fraction for α=1.0 is sharply elevated."
               loading="lazy">
        </a>
        <figcaption>
          <strong>Figure 3.</strong> Layer-wise SAE L0 (seed 42). The α = 1.0 curve diverges from baseline
          at layer 3 and remains below through layer 10; α = 0.1 tracks baseline. Layer 6 (used for
          main-text H2/H3 analyses) lies in the middle of the effect band.
        </figcaption>
      </figure>
    </div>
  </section>

  <!-- ── 6 · H1 ────────────────────────────────────────────── -->
  <section id="h1" aria-labelledby="h1-heading">
    <div class="container">
      <div class="result-header">
        <h2 id="h1-heading">H1 — Neurons don't change</h2>
        <span class="verdict verdict-null">NULL</span>
      </div>
      <p>
        Entropy-based selectivity
        <abbr class="tooltip" data-tip="Normalised negative entropy of a neuron's class-activation distribution; 1 = perfectly selective.">Mᵤ</abbr>
        is <strong>flat across α</strong>: 0.234 ± 0.002 (α = 0) vs 0.236 ± 0.002
        (α = 1.0); Δ = 0.0019, not significant. The SAE-feature-level Mᵤ is also null,
        including at a larger 25k-image evaluation.
      </p>
      <p>
        <strong>Interpretation:</strong> Monosemanticity measures how one neuron spreads
        across classes — a marginal statistic insensitive to the <em>joint</em> arrangement
        of many neurons. TopoLoss reorganises neurons into functionally coherent spatial
        clusters (H3) without changing per-neuron selectivity (H1). Consistent with the
        superposition hypothesis: spatial co-location, not per-neuron selectivity, is the
        relevant unit of circuit purity.
      </p>
    </div>
  </section>

  <!-- ── 7 · TAKEAWAY ─────────────────────────────────────── -->
  <section id="takeaway" class="section-alt" aria-labelledby="takeaway-heading">
    <div class="container container-prose">
      <h2 id="takeaway-heading">Takeaway</h2>
      <p>
        A penalty of fewer than 0.1k parameters, added to the training loss,
        systematically shifts circuit <em>geometry</em> in a way existing tools can
        measure (d = 0.95 for H3) — at a modest cost (~1.6 pp accuracy at α = 1.0)
        and without disentangling individual neurons. This positions cheap architectural
        priors as a viable <strong>training-time complement to post-hoc tooling</strong>
        like SAEs, and suggests current neuron-level monosemanticity metrics are blind
        to a class of real interpretability gains.
      </p>
      <p>
        <strong>Scale caveat:</strong> In a 4-layer TinyViT the H2 trend replicates
        (L0 −2.1% at α = 1.0) but H3 saturates at just 1.42×, far below the ViT-S
        result of 2.79× — consistent with a depth/scale requirement for circuit
        formation. Limitations: single dataset (ImageNet-100), partial accuracy
        confound on H2, single seed for TinyViT.
      </p>
    </div>
  </section>

  <!-- ── 8 · VIDEO ─────────────────────────────────────────── -->
  <section id="video" aria-labelledby="video-heading">
    <div class="container">
      <h2 id="video-heading">Video Walkthrough</h2>
      <div class="video-wrap">
        <video controls preload="none" poster="assets/video_poster.jpg"
               aria-label="Animated walkthrough of TopoLoss and the H1/H2/H3 results">
          <source src="assets/toposae_walkthrough.mp4" type="video/mp4">
          <p>Your browser does not support HTML video.
             <a href="assets/toposae_walkthrough.mp4">Download the video</a>.</p>
        </video>
      </div>
    </div>
  </section>

  <!-- ── 9 · BIBTEX ────────────────────────────────────────── -->
  <section id="cite" class="section-alt" aria-labelledby="cite-heading">
    <div class="container">
      <h2 id="cite-heading">Citation</h2>
      <div class="bibtex-wrap">
        <button class="copy-btn" aria-label="Copy BibTeX to clipboard">Copy</button>
        <pre class="bibtex"><code>@inproceedings{ranka2026topographic,
  title     = {Topographic Training Concentrates Causal Circuits
               Without Improving Neuron Monosemanticity},
  author    = {Ranka, Gautam and Pandere, Shubham and D'souza, Aiden Ross},
  booktitle = {Mechanistic Interpretability Workshop at the 43rd
               International Conference on Machine Learning (ICML)},
  year      = {2026}
}</code></pre>
      </div>
    </div>
  </section>

</main>

<!-- ── FOOTER ────────────────────────────────────────────── -->
<footer class="site-footer">
  <div class="container">
    <p>
      Gautam Ranka · Shubham Pandere · Aiden Ross D'souza<br>
      IvLabs, VNIT Nagpur, India &nbsp;·&nbsp;
      <a href="https://github.com/IvLabs/toposae" target="_blank" rel="noopener">GitHub</a>
      &nbsp;·&nbsp; 2026
    </p>
  </div>
</footer>

<script src="main.js"></script>
</body>
</html>
```

- [ ] **Step 2.2: Open in browser and verify structure**
```bash
python3 -m http.server 8000 --directory docs &
# open http://localhost:8000 in browser
# verify: all sections visible, no broken images yet (assets task must run first)
# kill server after check: kill %1
```

- [ ] **Step 2.3: Commit**
```bash
git add docs/index.html
git commit -m "feat(website): add HTML skeleton with all sections and landmarks"
```

---

## Chunk 2: CSS

### Task 3: Write style.css

**Files:**
- Create: `docs/style.css`

Write the full stylesheet following `website/design-system.md`. All styles via CSS custom properties. Mobile-first.

- [ ] **Step 3.1: Create `docs/style.css`**

```css
/* ── CUSTOM PROPERTIES ──────────────────────────────────── */
:root {
  --bg:            #ffffff;
  --bg-alt:        #f7f7f9;
  --text:          #1a1a1e;
  --text-muted:    #5b5b66;
  --accent:        #4f46e5;
  --accent-weak:   #eef0fd;
  --border:        #e5e5ea;
  --verdict-null:  #6b7280;
  --verdict-warn:  #b45309;
  --verdict-pos:   #15803d;

  --pad:           1.25rem;
  --radius:        10px;
  --radius-sm:     6px;
  --nav-h:         56px;
  --shadow-sm:     0 1px 3px rgba(0,0,0,.08);
}

/* ── RESET & BASE ───────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html {
  scroll-behavior: smooth;
  scroll-padding-top: calc(var(--nav-h) + 1rem);
}

@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}

body {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: clamp(1rem, 1.5vw + .5rem, 1.125rem);
  line-height: 1.65;
  color: var(--text);
  background: var(--bg);
}

/* ── TYPOGRAPHY ─────────────────────────────────────────── */
h1, h2, h3 {
  font-family: "Georgia", "Source Serif 4", serif;
  line-height: 1.25;
  color: var(--text);
}
h1 { font-size: clamp(1.75rem, 3vw + 1rem, 2.75rem); margin-bottom: 1.25rem; }
h2 { font-size: clamp(1.3rem, 2vw + .75rem, 1.9rem); margin-bottom: 1rem; }
h3 { font-size: clamp(1.05rem, 1vw + .5rem, 1.3rem); margin-bottom: .5rem; }

p { max-width: 72ch; margin-bottom: 1rem; }
p:last-child { margin-bottom: 0; }

a { color: var(--accent); text-decoration: underline; }
a:hover { opacity: .8; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }

code { font-family: ui-monospace, "JetBrains Mono", monospace; font-size: .9em; background: var(--bg-alt); padding: .1em .35em; border-radius: 3px; }
abbr[title], abbr.tooltip { text-decoration: underline dotted var(--text-muted); cursor: help; }

/* ── LAYOUT ─────────────────────────────────────────────── */
.container {
  width: 100%;
  max-width: 820px;
  margin-inline: auto;
  padding-inline: var(--pad);
}
.container-prose { max-width: 700px; }

/* ── NAV ────────────────────────────────────────────────── */
.site-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  height: var(--nav-h);
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  transition: box-shadow .2s;
}
.site-nav.scrolled { box-shadow: var(--shadow-sm); }

.nav-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1100px;
  margin-inline: auto;
  padding-inline: var(--pad);
}

.nav-wordmark {
  font-family: "Georgia", serif;
  font-weight: bold;
  font-size: 1.1rem;
  color: var(--text);
  text-decoration: none;
}

.site-nav nav {
  display: flex;
  gap: 1.25rem;
  align-items: center;
}
.site-nav nav a {
  font-size: .9rem;
  color: var(--text-muted);
  text-decoration: none;
  padding: .25rem .1rem;
  border-bottom: 2px solid transparent;
  transition: color .15s, border-color .15s;
}
.site-nav nav a:hover,
.site-nav nav a.active { color: var(--accent); border-bottom-color: var(--accent); }
.nav-github { display: flex; align-items: center; color: var(--text-muted); }
.nav-github:hover { color: var(--accent); }

@media (max-width: 640px) {
  .site-nav nav { gap: .75rem; overflow-x: auto; }
  .site-nav nav a { white-space: nowrap; }
  .nav-github svg { width: 18px; height: 18px; }
}

/* ── SECTIONS ───────────────────────────────────────────── */
section { padding-block: 4.5rem; }
.section-alt { background: var(--bg-alt); }

/* ── HERO ───────────────────────────────────────────────── */
.section-hero { text-align: center; padding-block: 5rem 4rem; }
.section-hero .container { display: flex; flex-direction: column; align-items: center; gap: 1.25rem; }
.section-hero h1 { max-width: 780px; }

.authors { font-size: 1.05rem; color: var(--text-muted); }
.affiliation, .venue { font-size: .9rem; color: var(--text-muted); }
.venue { font-style: italic; }

.hero-buttons { display: flex; gap: .75rem; flex-wrap: wrap; justify-content: center; }

.btn {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  padding: .55rem 1.25rem;
  border-radius: var(--radius);
  font-size: .95rem;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  transition: opacity .15s, background .15s;
  border: 2px solid transparent;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { opacity: .88; color: #fff; }
.btn-secondary { border-color: var(--accent); color: var(--accent); background: transparent; }
.btn-secondary:hover { background: var(--accent-weak); }

.tldr {
  max-width: 680px;
  font-size: 1rem;
  background: var(--accent-weak);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius-sm);
  padding: .9rem 1.1rem;
  text-align: left;
  margin-bottom: 0;
}

/* stat cards */
.stat-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  width: 100%;
  max-width: 640px;
}
.stat-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: .35rem;
  box-shadow: var(--shadow-sm);
}
.stat-number {
  font-family: "Georgia", serif;
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: bold;
  color: var(--accent);
  line-height: 1;
}
.stat-label { font-size: .8rem; color: var(--text-muted); text-align: center; line-height: 1.35; }

@media (max-width: 560px) {
  .stat-cards { grid-template-columns: 1fr; max-width: 300px; }
}

/* ── EQUATION ───────────────────────────────────────────── */
.equation-block {
  margin-block: 1.75rem;
  overflow-x: auto;
  text-align: center;
}

/* ── METHOD GLOSS ───────────────────────────────────────── */
.method-gloss { margin-bottom: 2rem; }
.method-setup {
  font-size: .92rem;
  color: var(--text-muted);
  background: var(--bg-alt);
  padding: .75rem 1rem;
  border-radius: var(--radius-sm);
  margin-top: .5rem;
}

/* ── FIGURES ────────────────────────────────────────────── */
.fig-block {
  margin-block: 2rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  background: var(--bg);
}
.fig-block img {
  width: 100%;
  height: auto;
  display: block;
}
.fig-block a { display: block; }
.fig-block a:hover img { opacity: .92; }
figcaption {
  padding: .9rem 1.1rem;
  font-size: .875rem;
  color: var(--text-muted);
  font-style: italic;
  border-top: 1px solid var(--border);
  line-height: 1.5;
}
.fig-wide { max-width: 960px; margin-inline: auto; }

.fig-placeholder {
  background: var(--bg-alt);
  padding: 3rem 2rem;
  text-align: center;
  color: var(--text-muted);
  font-style: italic;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── HYPOTHESIS CARDS ───────────────────────────────────── */
.hypothesis-cards {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
  margin-top: 1.5rem;
}
.h-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: .5rem;
  box-shadow: var(--shadow-sm);
}
.h-tag {
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.h-metric { font-size: .85rem; color: var(--text-muted); margin-bottom: .5rem; }

@media (max-width: 700px) {
  .hypothesis-cards { grid-template-columns: 1fr; }
}

/* ── VERDICT CHIPS ──────────────────────────────────────── */
.verdict {
  display: inline-block;
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
  padding: .25rem .65rem;
  border-radius: 999px;
  border: 1.5px solid currentColor;
  margin-top: auto;
  width: fit-content;
}
.verdict-null { color: var(--verdict-null); background: #f3f4f6; }
.verdict-warn { color: var(--verdict-warn); background: #fef3c7; }
.verdict-pos  { color: var(--verdict-pos);  background: #dcfce7; }

/* ── RESULT HEADER ──────────────────────────────────────── */
.result-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}
.result-header h2 { margin-bottom: 0; }

/* ── ALPHA TOGGLE ───────────────────────────────────────── */
.alpha-toggle {
  display: inline-flex;
  border: 2px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-block: 1.25rem .75rem;
}
.alpha-btn {
  background: var(--bg);
  border: none;
  border-right: 1px solid var(--border);
  padding: .5rem 1.1rem;
  font-size: .9rem;
  cursor: pointer;
  color: var(--text-muted);
  transition: background .15s, color .15s;
}
.alpha-btn:last-child { border-right: none; }
.alpha-btn.active,
.alpha-btn:hover { background: var(--accent); color: #fff; }
.alpha-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }

.alpha-readout {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 1.5rem;
  padding: .75rem 1rem;
  background: var(--accent-weak);
  border-radius: var(--radius-sm);
  border-left: 3px solid var(--accent);
}
.alpha-ratio {
  font-family: "Georgia", serif;
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--accent);
  white-space: nowrap;
}
.alpha-caption { font-size: .9rem; color: var(--text-muted); }

/* ── CAVEAT BOX ─────────────────────────────────────────── */
.caveat-box {
  border-left: 4px solid var(--verdict-warn);
  background: #fffbeb;
  padding: 1rem 1.25rem;
  border-radius: var(--radius-sm);
  font-size: .92rem;
  margin-block: 1.25rem;
  color: var(--text);
  max-width: 72ch;
}

/* ── SUPPORTING LINE ────────────────────────────────────── */
.supporting {
  font-size: .92rem;
  color: var(--text-muted);
  margin-top: .5rem;
}

/* ── VIDEO ──────────────────────────────────────────────── */
.video-wrap {
  margin-top: 1.5rem;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  background: #000;
}
.video-wrap video {
  width: 100%;
  display: block;
  max-height: 70vh;
}

/* ── BIBTEX ─────────────────────────────────────────────── */
.bibtex-wrap {
  position: relative;
  margin-top: 1rem;
}
.bibtex {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1rem;
  font-size: .85rem;
  overflow-x: auto;
  line-height: 1.7;
}
.copy-btn {
  position: absolute;
  top: .75rem;
  right: .75rem;
  padding: .3rem .8rem;
  font-size: .8rem;
  font-weight: 600;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: opacity .15s;
}
.copy-btn:hover { opacity: .85; }
.copy-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ── TOOLTIP POPOVER ─────────────────────────────────────── */
.tooltip-popup {
  position: absolute;
  z-index: 200;
  background: var(--text);
  color: #fff;
  font-size: .8rem;
  padding: .4rem .7rem;
  border-radius: var(--radius-sm);
  max-width: 240px;
  line-height: 1.45;
  pointer-events: none;
  white-space: normal;
  box-shadow: 0 2px 8px rgba(0,0,0,.2);
}

/* ── FOOTER ─────────────────────────────────────────────── */
.site-footer {
  border-top: 1px solid var(--border);
  padding-block: 2rem;
  text-align: center;
  font-size: .875rem;
  color: var(--text-muted);
}
.site-footer a { color: var(--text-muted); }
.site-footer a:hover { color: var(--accent); }

/* ── RESULTS INTRO ──────────────────────────────────────── */
.results-intro { color: var(--text-muted); margin-bottom: .5rem; }
```

- [ ] **Step 3.2: Reload browser, verify visual pass**

Open `http://localhost:8000`. Check:
- Nav is sticky, wordmark visible, links in nav.
- Hero title renders in serif, buttons present.
- Stat cards lay out in 3-column on desktop.
- Sections alternate white/gray.
- Verdict chips colored correctly (gray/amber/green).
- No horizontal overflow at 360px viewport width (DevTools mobile sim).

- [ ] **Step 3.3: Commit**
```bash
git add docs/style.css
git commit -m "feat(website): add full CSS with design-system tokens"
```

---

## Chunk 3: JavaScript

### Task 4: Write main.js

**Files:**
- Create: `docs/main.js`

Five behaviors, all progressive enhancement. With JS disabled the page is complete.

- [ ] **Step 4.1: Create `docs/main.js`**

```js
/* TopoSAE project page — vanilla JS progressive enhancements */

/* ── 1. NAV SCROLL SHADOW ─────────────────────────────── */
(function navShadow() {
  const nav = document.querySelector('.site-nav');
  if (!nav) return;
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 10);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

/* ── 2. SCROLL-SPY (active nav link) ─────────────────── */
(function scrollSpy() {
  const links = document.querySelectorAll('.site-nav nav a[href^="#"]');
  if (!links.length) return;

  const sections = [...links].map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      // find the topmost intersecting section
      const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      const id = visible[0].target.id;
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${id}`));
    },
    { rootMargin: '-56px 0px -60% 0px', threshold: 0 }
  );

  sections.forEach(s => observer.observe(s));
})();

/* ── 3. COUNT-UP STAT CARDS ───────────────────────────── */
(function countUp() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const cards = document.querySelectorAll('.stat-number[data-target]');
  if (!cards.length) return;

  function animateCard(el) {
    const target  = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals ?? '0', 10);
    const prefix  = el.dataset.prefix ?? '';
    const suffix  = el.dataset.suffix ?? '';
    const duration = 1200; // ms
    const start = performance.now();

    function frame(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value = (target * ease).toFixed(decimals);
      el.textContent = prefix + value + suffix;
      if (progress < 1) requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          animateCard(e.target);
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.6 }
  );

  cards.forEach(c => observer.observe(c));
})();

/* ── 4. ALPHA TOGGLE ──────────────────────────────────── */
(function alphaToggle() {
  const buttons = document.querySelectorAll('.alpha-btn');
  const ratioEl = document.querySelector('.alpha-ratio');
  const captionEl = document.querySelector('.alpha-caption');
  if (!buttons.length || !ratioEl || !captionEl) return;

  const data = {
    '0.0': { ratio: '1.68 ± 0.42', caption: 'Baseline — near the random line' },
    '0.1': { ratio: '2.27 ± 0.93', caption: 'Weak topography (+0.59 vs baseline)' },
    '1.0': { ratio: '2.79 ± 1.24', caption: 'Strong topography (+1.12 vs baseline, p < 0.0001)' },
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const alpha = btn.dataset.alpha;
      if (data[alpha]) {
        ratioEl.textContent = data[alpha].ratio;
        captionEl.textContent = data[alpha].caption;
      }
    });
  });
})();

/* ── 5. BIBTEX COPY ───────────────────────────────────── */
(function copyBibtex() {
  const btn = document.querySelector('.copy-btn');
  const pre = document.querySelector('.bibtex code');
  if (!btn || !pre) return;

  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = orig; }, 2000);
    } catch {
      // silently degrade — user can still select text
    }
  });
})();

/* ── 6. TOOLTIP POPOVER ───────────────────────────────── */
(function tooltips() {
  const abbrs = document.querySelectorAll('abbr.tooltip[data-tip]');
  if (!abbrs.length) return;

  let popup = null;

  function showTip(abbr) {
    popup = document.createElement('div');
    popup.className = 'tooltip-popup';
    popup.textContent = abbr.dataset.tip;
    document.body.appendChild(popup);

    const rect = abbr.getBoundingClientRect();
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;

    const top = rect.top + scrollY - popup.offsetHeight - 8;
    const left = Math.min(
      rect.left + scrollX + rect.width / 2 - popup.offsetWidth / 2,
      window.innerWidth - popup.offsetWidth - 16
    );
    popup.style.top  = `${Math.max(scrollY + 8, top)}px`;
    popup.style.left = `${Math.max(8, left)}px`;
  }

  function hideTip() {
    if (popup) { popup.remove(); popup = null; }
  }

  abbrs.forEach(abbr => {
    abbr.addEventListener('mouseenter', () => showTip(abbr));
    abbr.addEventListener('mouseleave', hideTip);
    abbr.addEventListener('focusin',   () => showTip(abbr));
    abbr.addEventListener('focusout',  hideTip);
  });
})();
```

- [ ] **Step 4.2: Test all five behaviors**

Open `http://localhost:8000`. Verify:

1. **Nav shadow:** scroll down — thin shadow appears on nav bar.
2. **Scroll-spy:** scroll slowly — the correct nav link lights up in accent color.
3. **Count-up:** on first scroll into hero stat cards, numbers animate up to `2.79×`, `d = 0.95`, `<0.1k`.
4. **α-toggle:** click `α = 0.1` — readout updates to `2.27 ± 0.93` and caption changes. Click `α = 1.0` — updates to `2.79 ± 1.24`.
5. **Copy BibTeX:** click Copy — button reads "Copied!" for 2 seconds.
6. **Tooltips:** hover over `Mᵤ`, `L0`, or `d` — small dark popover appears. Focus via Tab also shows it.

- [ ] **Step 4.3: Test with JS disabled**

In DevTools → Settings → Debugger → Disable JavaScript. Reload. Verify:
- All sections and content visible.
- Stat cards show final values (not 0).
- α-toggle buttons visible (inert — clicking does nothing but page is complete).
- BibTeX block readable (Copy button visible but does nothing).
- Tooltips fall back to native `<abbr title>` on hover.

Re-enable JS.

- [ ] **Step 4.4: Commit**
```bash
git add docs/main.js
git commit -m "feat(website): add JS progressive enhancements (count-up, alpha-toggle, tooltips, copy)"
```

---

## Chunk 4: Polish + QA

### Task 5: Responsive + accessibility check

**Files:**
- Modify: `docs/style.css` (tweaks only, no new features)
- Modify: `docs/index.html` (alt text review)

- [ ] **Step 5.1: Mobile test at 360px**

DevTools → Dimensions → 360×800. Check:
- Nav links don't overflow (allowed to horizontally scroll as chips).
- Stat cards stack vertically.
- Hypothesis cards stack vertically.
- Buttons wrap but don't overflow.
- Hero title doesn't overflow horizontally.
- Fix any overflow found.

- [ ] **Step 5.2: Tablet test at 768px**

DevTools → 768×1024. Check:
- Stat cards: confirm 3-column still readable at this width.
- Hypothesis cards: confirm 3-column not too narrow (adjust breakpoint in CSS if needed).

- [ ] **Step 5.3: Keyboard navigation**

Tab through the entire page from top to bottom. Every interactive element
(nav links, buttons, α-toggle buttons, copy button, figure links, video controls)
must show a visible focus ring. Fix any missing focus styles.

- [ ] **Step 5.4: Alt text audit**

Read the `alt` attribute on every `<img>`. Each should describe the *result shown*,
not just "figure". The HTML template already has descriptive alt — confirm it
matches the actual figure content once assets are in place.

- [ ] **Step 5.5: KaTeX render check**

Reload with network access. Confirm the two equations in §Method render:
- Block equation: `L = L_CE + L_topo` and the sum formula.
- Inline: `√d × √d`.

If they show raw LaTeX, check browser console for KaTeX errors. Common fix: ensure
KaTeX CDN scripts loaded (`<script defer>` and `onload` attribute intact).

- [ ] **Step 5.6: Contrast spot-check**

Use DevTools accessibility panel or browser extension to check:
- Body text on `--bg`: must be ≥ 4.5:1 (#1a1a1e on white — passes).
- `--accent` on white (`#4f46e5` on `#fff`) — check; if borderline add font-weight.
- Verdict chips: check colored text on light background.

- [ ] **Step 5.7: Commit any fixes**
```bash
git add docs/index.html docs/style.css
git commit -m "fix(website): responsive and accessibility polish"
```

---

### Task 6: Fig 1 placeholder resolution

**Files:**
- Modify: `docs/index.html` (when composite figure available)

This task is decoupled from the rest — the site ships without Fig 1; this fills it in.

- [ ] **Step 6.1: Check for a composite Fig 1 PNG**
```bash
ls results/figures/ | grep -iE "overview|fig1|figure1"
```
If nothing matches, ask the paper author to export or compose the 3-panel Figure 1 (TopoLoss schematic + H2 panel + H3 panel) as a single PNG.

- [ ] **Step 6.2: When available, copy it**
```bash
cp <source_file> docs/assets/fig1_overview.png
```

- [ ] **Step 6.3: Replace the placeholder in index.html**

Find the `<div class="fig-placeholder">` block in the §Method section and replace it with:
```html
<img src="assets/fig1_overview.png"
     alt="Three-panel overview. Left: TopoLoss penalises weight differences between adjacent units on a topographic grid. Centre: SAE L0 decreases monotonically with α (blue line) while dead-feature fraction rises (red dashed). Right: per-class causal patching ratios for Baseline, TopoWeak, TopoStrong — mean rises from 1.68 to 2.79."
     loading="lazy">
```

- [ ] **Step 6.4: Commit**
```bash
git add docs/assets/fig1_overview.png docs/index.html
git commit -m "feat(website): add Fig 1 overview composite"
```

---

### Task 7: GitHub Pages deploy

- [ ] **Step 7.1: Verify docs/ is complete**
```bash
ls -la docs/
# must have: index.html  style.css  main.js  .nojekyll  assets/
ls docs/assets/
# must have: toposae.pdf  fig2_h3_patching.png  fig2_h3_patching.pdf
#            fig3_layerwise_sae.png  fig3_layerwise_sae.pdf
#            spatial_coherence.png  video_poster.jpg  toposae_walkthrough.mp4
```

- [ ] **Step 7.2: Final local test from the exact docs/ root**
```bash
python3 -m http.server 8000 --directory docs
# open http://localhost:8000
```
Walk through every section. Confirm:
- All figures load (no broken image icons).
- Video loads and plays.
- PDF link opens in new tab.
- GitHub link opens in new tab.
- Video anchor (#video) in hero button scrolls to video section.
- BibTeX copy works.

- [ ] **Step 7.3: Commit final state**
```bash
git add docs/
git commit -m "feat(website): production-ready TopoSAE project page"
```

- [ ] **Step 7.4: Enable GitHub Pages**

In the GitHub repo (`https://github.com/IvLabs/toposae`):
- Settings → Pages → Source: **Deploy from a branch**
- Branch: `master` (or `main`) · Folder: `/docs`
- Save.

After ~1 minute: `https://ivlabs.github.io/toposae/` should serve the page.

- [ ] **Step 7.5: Verify live URL**

Open `https://ivlabs.github.io/toposae/`. Repeat the checks from Step 7.2.
Confirm KaTeX equations render (CDN must be reachable). If assets 404, confirm
paths are relative and the `.nojekyll` file is present.

---

## Summary

| Chunk | Tasks | Deliverable |
|-------|-------|-------------|
| 1     | 1–2   | Assets in `docs/assets/` + validated HTML scaffold |
| 2     | 3     | Full CSS with design-system tokens |
| 3     | 4     | Vanilla JS with 5 progressive enhancements |
| 4     | 5–7   | Polish, Fig 1, deploy |

**BibTeX note:** Confirm the exact workshop `booktitle` and add DOI/URL once ICML 2026 workshop proceedings are published.

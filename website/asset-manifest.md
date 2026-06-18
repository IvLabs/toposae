# Asset Manifest

Which assets the page uses, where they live now, and what to copy into the
published site's `assets/`. **Copy** assets into the site dir — do not hot-link
`../results/...` or `../media/...`, so the GitHub Pages deploy is self-contained.

Source paths are relative to repo root (`/home/ashed/Documents/projects/toposae`).

## Figures

| Site slot | Source file | Notes |
|-----------|-------------|-------|
| **Fig 1 — Method/Overview** (§Method) | `results/figures/` — *3-panel overview (TopoLoss schematic + H2 + H3)* | Paper's Figure 1. If no single composite PNG exists, compose from the panels below or export it; flagged for the author to confirm the exact source file. |
| **Fig 2 — H3 causal patching** (§H3) | `results/figures/exp016_h3_multiclass.png` (vector: `exp016_h3_multiclass.pdf`) | Paper's Figure 2 — per-class patch ratios + per-variant bars. Primary result figure. |
| **Fig 3 — Layer-wise SAE L0** (§H2) | `results/figures/layerwise_sae_l0.png` (vector: `layerwise_sae_l0.pdf`) | Paper's Figure 3 — L0 + dead features per layer. |
| Spatial coherence (optional support, §H3) | `results/figures/spatial_coherence.png` | Optional inline support for the coherence stat. |
| Monosemanticity summary (optional, §H1) | `results/figures/monosemanticity_summary.png` or `imagenet100_h1_summary.png` | Optional visual for the H1 null. |

Other figures available if a section wants more (`alpha_sweep.png`,
`alpha_scaling.png`, `patching_comparison.png`, `sae_comparison.png`,
`exp015_paired_stats.png`, `vit_s16_h1_summary.png`) — not required by the spec.

**Web prep:** export/compress PNGs to web resolution (≤ ~1600px wide, optimized);
keep the `.pdf` vectors as the click-through "full res" target. Provide descriptive
`alt` text per `design-system.md`.

## Paper PDF

| Slot | Source | Notes |
|------|--------|-------|
| Hero "Paper" button | `drafts/new_paper/draft_extracted/paper.pdf` | Copy the final published PDF into `assets/` (e.g. `assets/toposae.pdf`), or link to the arXiv/OpenReview URL once available. |

## Video

| Slot | Source | Notes |
|------|--------|-------|
| §Video embed | `media/videos/manim_animation/1080p60/TopoSAEPresentation.mp4` | **CONFIRMED FINAL.** Full manim walkthrough at 1080p60. |
| Video poster | extract first frame from the MP4 | Used as `poster=` attribute on the `<video>` element. |

**Web prep commands (run from repo root):**
```bash
ffmpeg -i media/videos/manim_animation/1080p60/TopoSAEPresentation.mp4 \
  -vf scale=1280:-2 -c:v libx264 -crf 22 -preset slow \
  -movflags +faststart docs/assets/toposae_walkthrough.mp4

ffmpeg -i media/videos/manim_animation/1080p60/TopoSAEPresentation.mp4 \
  -vframes 1 -q:v 2 docs/assets/video_poster.jpg
```

## External links

| Link | Target |
|------|--------|
| Code | https://github.com/IvLabs/toposae |
| Repo (footer / nav icon) | https://github.com/IvLabs/toposae |
| Paper | `assets/toposae.pdf` (or arXiv/OpenReview when available) |

## Suggested published-site layout (`/docs`)

```
docs/
├── index.html
├── style.css
├── main.js
├── .nojekyll
└── assets/
    ├── toposae.pdf
    ├── fig1_overview.png
    ├── fig2_h3_patching.png   (+ .pdf)
    ├── fig3_layerwise_sae.png (+ .pdf)
    ├── spatial_coherence.png
    ├── video_poster.png
    └── toposae_walkthrough.mp4   (added when finalized)
```

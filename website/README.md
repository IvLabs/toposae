# TopoSAE — Project Page Spec

This folder is the **specification** for the paper's demo website. It contains no
site code — only the design, content, and asset decisions an implementer (human or
agent) needs to build the page.

**Paper:** *Topographic Training Concentrates Causal Circuits Without Improving
Neuron Monosemanticity* — Ranka, Pandere, D'souza (IvLabs, VNIT Nagpur).
Mechanistic Interpretability Workshop @ ICML 2026.

**Repo / host:** https://github.com/IvLabs/toposae (GitHub Pages).

## Decisions (locked)

| Axis          | Choice                                                              |
|---------------|---------------------------------------------------------------------|
| Structure     | Single-page vertical scroll, sticky anchor-nav (academic genre)     |
| Aesthetic     | Clean academic — white base, one accent color, figures carry the page |
| Interactivity | Static core + light accents (α-toggle, count-up stats, metric tooltips) |
| Hosting       | Static `index.html` + `assets/`, served by GitHub Pages             |

## Files in this spec

| File                  | What it covers                                              |
|-----------------------|-------------------------------------------------------------|
| `SPEC.md`             | Site structure, every section, layout, interactive behavior, hosting |
| `design-system.md`    | Colors, typography, spacing, components, responsive rules    |
| `content.md`          | The actual copy for every section, drawn from the paper      |
| `asset-manifest.md`   | Figure→section mapping, source paths, video, what to copy where |

## Implementer's quick start

1. Read `SPEC.md` top to bottom.
2. Pull verbatim copy from `content.md`; do not paraphrase the statistics.
3. Copy assets per `asset-manifest.md` into the published site's `assets/`.
4. Apply tokens from `design-system.md`.
5. Keep it dependency-light: one HTML file, one CSS file, one small JS file. No build step.

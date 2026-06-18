# TopoSAE Project Page — Build Spec

Single-page, statically-served project page for the paper. No framework, no build
step: `index.html` + `style.css` + `main.js` + `assets/`. Targets GitHub Pages.

---

## 1. Page anatomy (top → bottom)

```
┌──────────────────────────────────────────────────────────┐
│ STICKY NAV   TopoSAE · Abstract Method Results Video Cite  │  ← always visible
├──────────────────────────────────────────────────────────┤
│ 0. HERO        title · authors · affiliation · venue       │
│                [ Paper ] [ Code ] [ Video ]  buttons       │
│                TL;DR sentence                              │
│                3 headline stat cards (count-up)            │
├──────────────────────────────────────────────────────────┤
│ 1. ABSTRACT    paper abstract, lightly formatted           │
├──────────────────────────────────────────────────────────┤
│ 2. METHOD      TopoLoss equation + plain-language gloss    │
│                overview figure (Fig 1)                     │
├──────────────────────────────────────────────────────────┤
│ 3. THREE QUESTIONS  H1/H2/H3 framing card row              │
├──────────────────────────────────────────────────────────┤
│ 4. H3 RESULT   "Circuits concentrate" — Fig 2 + α-toggle   │  ← lead result
├──────────────────────────────────────────────────────────┤
│ 5. H2 RESULT   "Superposition drops (with caveat)" Fig 3   │
├──────────────────────────────────────────────────────────┤
│ 6. H1 RESULT   "Neurons unchanged" — the null              │
├──────────────────────────────────────────────────────────┤
│ 7. TAKEAWAY    one-paragraph synthesis + scale caveat      │
├──────────────────────────────────────────────────────────┤
│ 8. VIDEO       embedded manim walkthrough (or placeholder) │
├──────────────────────────────────────────────────────────┤
│ 9. BIBTEX      copy-to-clipboard citation block            │
├──────────────────────────────────────────────────────────┤
│ FOOTER         authors · IvLabs · repo link · license      │
└──────────────────────────────────────────────────────────┘
```

Order rationale: lead with **H3** (the strong, positive, headline result —
2.79×, d=0.95), then H2 (supported-with-caveat), then H1 (the null). This is the
paper's own "H3 but not H1" narrative arc, not the H1→H2→H3 enumeration order.

---

## 2. Section specs

All copy is in `content.md`. All figures are in `asset-manifest.md`. Section IDs
below are the nav anchor targets.

### NAV (`<header>`, sticky)
- Left: wordmark `TopoSAE`. Right: anchor links to `#abstract #method #results
  #video #cite` + a small GitHub icon link.
- Condenses/elevates shadow on scroll. Collapses to a hamburger or horizontal
  scroll below 640px. Smooth-scroll to anchors with offset for the sticky bar.

### 0 · HERO (`#top`)
- Title (full paper title), then author line with `*` equal-contribution note and
  affiliation, then venue line.
- Three pill buttons: **Paper** (PDF), **Code** (GitHub), **Video** (scrolls to
  `#video`). Buttons are the primary accent-colored elements on the page.
- TL;DR: one bold sentence (see `content.md`).
- **Three headline stat cards** in a row — the page's signature element:
  `2.79×` causal concentration · `d = 0.95` effect size · `<0.1k` added params.
  Numbers **count up** on first scroll-into-view (see §3).

### 1 · ABSTRACT (`#abstract`)
- The paper abstract as a single readable column (max ~70ch). No figure.

### 2 · METHOD (`#method`)
- TopoLoss equation rendered with KaTeX (CDN, no build): `L = L_CE + L_topo`,
  `L_topo = α Σ_{(i,j)∈E} ‖w_i − w_j‖²`.
- 2–3 sentence plain gloss: what TopoLoss does, where it's applied
  (attn.proj of all 12 ViT-S blocks, units on a √d×√d grid), and the cost
  (<0.1k params, ~1.6pp accuracy at α=1.0).
- **Fig 1 overview** (TopoLoss schematic + H2 + H3 panels), full-width, captioned.

### 3 · THREE QUESTIONS (`#results` anchor starts here)
- Intro line: "We audit TopoLoss on three interpretability axes."
- Three cards (H1, H2, H3): each shows the question, the metric, and a verdict
  chip — `NULL` (H1, gray), `SUPPORTED *` (H2, amber, asterisk=caveat),
  `SUPPORTED` (H3, green). Chips set reader expectations before the detail.

### 4 · H3 — Causal circuits concentrate (`#h3`)
- Heading + verdict chip (green). Lead result.
- **Fig 2** (per-class patch ratios + per-variant bars).
- **α-toggle** (see §3): 3-state segmented control `α=0.0 / 0.1 / 1.0` that updates
  a live readout — the patch ratio (`1.68 → 2.27 → 2.79`) and its caption — and
  highlights the matching variant. Figure itself stays static behind the readout.
- Supporting line: spatial coherence (top-k units 6.1% more concentrated, p=0.0005).

### 5 · H2 — Superposition drops, with a caveat (`#h2`)
- Heading + amber chip. **Fig 3** (layer-wise SAE L0).
- Stats: L0 −11% (p=0.007, d=7.06), dead features ↑19-fold.
- **Caveat callout box** (visually distinct, e.g. left amber border): the accuracy
  confound — α=1.0 costs −1.6pp; OLS R²=0.72 of L0 on accuracy. State it plainly.

### 6 · H1 — Neurons don't change (`#h1`)
- Heading + gray `NULL` chip. This is a *negative* result presented as a finding,
  not a failure.
- Stat: entropy selectivity M_u flat (0.234 → 0.236, n.s.); SAE-feature M_u also null.
- One framing sentence: superposition redistributes info across neurons, so
  spatial co-location — not per-neuron selectivity — is the unit of circuit purity.

### 7 · TAKEAWAY (`#takeaway`)
- One synthesis paragraph: a cheap training-time prior shifts circuit *geometry*
  measurably (d=0.95) without disentangling individual neurons — a complement to
  post-hoc SAEs, not a replacement.
- Short scale caveat: TinyViT (4-layer) H3 saturates at 1.42×, suggesting a
  depth/scale requirement; single dataset (ImageNet-100), single seed per TinyViT.

### 8 · VIDEO (`#video`)
- `<video controls poster=…>` embedding the manim walkthrough.
- Until the final render is dropped in, show a **placeholder card**: thumbnail/title
  frame + "Video walkthrough coming soon." The implementer wires the real `src`
  when the file lands (see `asset-manifest.md`).

### 9 · BIBTEX (`#cite`)
- `<pre>` block with the BibTeX entry (in `content.md`) + a **Copy** button
  (clipboard API, "Copied!" confirmation).

### FOOTER
- Authors + correspondence emails, IvLabs / VNIT Nagpur, repo link, year, and a
  license line for the site content.

---

## 3. Interactive accents (the only JS)

Keep JS to one small vanilla file. Each accent must degrade gracefully — with JS
disabled the page is still complete and readable.

1. **Count-up stat cards** — `IntersectionObserver` animates the three hero numbers
   from 0 to target on first view. No-JS fallback: numbers render at final value.
2. **α-toggle (H3)** — segmented control over `{0.0, 0.1, 1.0}`. Updates a text
   readout (ratio + caption) and a highlight state. Pure DOM text swap; data table
   lives inline in `main.js`. No-JS fallback: shows the α=1.0 (headline) value as
   static text.
3. **Metric tooltips** — hover/focus glossary on first use of `L0`, `Mᵤ`, `d`,
   `α`, `patching ratio`. `<abbr title>` or a small popover. Definitions in
   `content.md` §Glossary. Keyboard-accessible.
4. **Smooth-scroll + scroll-spy** — nav anchors smooth-scroll; active section link
   is highlighted. Respect `prefers-reduced-motion` (disable smooth scroll + count-up).
5. **Copy BibTeX** — clipboard write + transient confirmation.

No carousels, no sliders that fetch, no analytics, no external trackers.

---

## 4. Hosting (GitHub Pages)

- The **published site is self-contained**: `index.html`, `style.css`, `main.js`,
  and an `assets/` dir holding copies of the figures/PDF/video it references.
  Do **not** hot-link `../results/figures/...` or `../media/...` from the page —
  copy what's needed so the Pages deploy doesn't depend on repo layout.
- Recommended source: **`/docs` on the default branch** (Settings → Pages → Source:
  `main` /docs). Lowest-friction, no separate branch. The built site files (the
  contents described by this spec) go in `docs/`. Alternative: a `gh-pages` branch
  if the team prefers keeping `docs/` for other documentation.
- Use **relative asset paths** so the project-page URL
  `https://ivlabs.github.io/toposae/` resolves correctly.
- Add a `.nojekyll` file if any asset path starts with `_` (none currently expected,
  but cheap insurance).

---

## 5. Quality bar / acceptance

- Loads with no console errors; works with JS disabled (degraded but complete).
- Lighthouse: a11y ≥ 95, no layout shift on the stat cards.
- All statistics on the page match `content.md` exactly (which matches the paper).
- Mobile (360px) through desktop (1440px+) all read well; figures never overflow.
- All three external links (Paper PDF, GitHub, Video) resolve.
- Total page weight reasonable: compress figures for web (see `asset-manifest.md`).

## 6. Explicitly out of scope (YAGNI)

- No framework/bundler, no CMS, no dark-mode toggle, no i18n, no comment system,
  no live demo/inference, no data-fetching. If the team later wants an interactive
  patching demo, that is a separate spec.

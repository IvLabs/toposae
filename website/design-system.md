# Design System — Clean Academic

White-base academic project page. The figures (on white grounds) are the visual
payload; the chrome stays quiet. One accent color, generous whitespace, strong
typographic hierarchy. Reference genre: arXiv/NeRF-style project pages.

## Color tokens

| Token              | Value      | Use                                            |
|--------------------|------------|------------------------------------------------|
| `--bg`             | `#ffffff`  | Page background                                |
| `--bg-alt`         | `#f7f7f9`  | Alternating section bands, stat cards          |
| `--text`           | `#1a1a1e`  | Body + headings                                |
| `--text-muted`     | `#5b5b66`  | Captions, meta, footer                         |
| `--accent`         | `#4f46e5`  | Indigo — buttons, links, active nav, headline numbers |
| `--accent-weak`    | `#eef0fd`  | Button hover, tooltip bg, accent fills         |
| `--border`         | `#e5e5ea`  | Hairlines, figure frames, card borders         |
| `--verdict-null`   | `#6b7280`  | H1 chip (gray)                                 |
| `--verdict-warn`   | `#b45309`  | H2 chip + caveat box (amber)                   |
| `--verdict-pos`    | `#15803d`  | H3 chip (green)                                |

One accent only (indigo). Verdict colors are used *solely* on the H1/H2/H3 chips
and the H2 caveat box — they are semantic, not decorative. Do not introduce more
hues elsewhere.

Optional, subtle: a very faint topographic-grid motif (light `--border` squares)
behind the hero only, at low opacity, as a nod to Fig 1a. Keep it barely-there;
if it competes with the text, drop it. Default is plain `--bg`.

## Typography

- **Headings:** a clean serif (e.g. *Source Serif 4* / *Charter*) — academic feel.
- **Body:** a humanist sans (e.g. *Inter* / system UI stack) at ~18px, line-height
  1.65, measure capped at ~70ch for prose blocks.
- **Mono:** for metrics, equations-in-text, BibTeX (e.g. *JetBrains Mono* / ui-monospace).
- Headline stat numbers: serif or mono, large (clamp ~2.5–4rem), `--accent`.
- Load at most two webfont families (one serif, one sans); mono can be system.
  Self-host or use a single CDN; respect performance budget.

Scale (suggested, fluid via `clamp`):
`h1` 2.4–3.2rem · `h2` 1.6–2rem · `h3` 1.2–1.4rem · body 1.0–1.125rem · small 0.875rem.

## Spacing & layout

- Content column: `max-width: 760px`, centered, `--pad` 1.25rem gutters.
- Full-width figures may break out to `max-width: 1000px` within their section.
- Vertical rhythm: sections separated by ~5–6rem; alternate `--bg` / `--bg-alt`
  bands to delineate without rules.
- 8px spacing base. Border-radius: 10px cards/buttons, 6px chips/inputs.
- Shadows: at most one soft elevation (`0 1px 3px rgba(0,0,0,.08)`) on cards and the
  scrolled nav. Avoid heavy/neon shadows (that's the dark-theme look we did not pick).

## Components

- **Button (primary):** accent fill, white text, 10px radius; hover → slightly
  darker / `--accent-weak` ring. **Secondary:** outline in `--accent`.
- **Stat card:** `--bg-alt` fill, big accent number + small `--text-muted` label;
  3-up grid that stacks on mobile.
- **Verdict chip:** small pill, semantic color at low-alpha bg + solid text/border.
- **Figure block:** `--border` frame, rounded; `<figcaption>` in `--text-muted`
  small, italic figure number prefix (e.g. *Figure 2.*). Click figure → open
  full-res (PDF/PNG) in new tab.
- **Caveat box:** `--bg-alt` with a 4px left border in `--verdict-warn`, label
  "Caveat".
- **Equation block:** centered, KaTeX, generous vertical margin.
- **Segmented control (α-toggle):** three buttons in a bordered group; active =
  accent fill.
- **Code/BibTeX block:** mono on `--bg-alt`, copy button top-right.

## Responsive

- Breakpoints: 640px (mobile↔tablet), 1024px (tablet↔desktop).
- Nav collapses < 640px (hamburger or horizontal-scroll chips).
- Stat cards: 3-up ≥ 768px, stacked below.
- Figures: full container width on mobile, never horizontal-scroll.
- Tap targets ≥ 44px.

## Accessibility

- Contrast ≥ WCAG AA throughout (verify accent-on-white and chips).
- Visible focus rings (accent) on all interactive elements; full keyboard nav.
- `prefers-reduced-motion`: disable smooth-scroll, count-up, any transition.
- Real `alt` text on every figure (describe the result, not just "figure").
- Semantic landmarks: `header nav main section footer`, one `h1`, ordered headings.

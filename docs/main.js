/* TopoSAE project page — vanilla JS progressive enhancements
 * Degrades gracefully: page is complete and readable with JS disabled.
 */

/* ── 1. NAV SCROLL SHADOW ─────────────────────────────────── */
(function navShadow() {
  const nav = document.querySelector('.site-nav');
  if (!nav) return;
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 10);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

/* ── 2. SCROLL-SPY (active nav link) ─────────────────────── */
(function scrollSpy() {
  const links = document.querySelectorAll('.site-nav nav a[href^="#"]');
  if (!links.length) return;

  const sections = [...links]
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter(e => e.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      const id = visible[0].target.id;
      links.forEach(a => {
        a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
      });
    },
    { rootMargin: '-56px 0px -60% 0px', threshold: 0 }
  );

  sections.forEach(s => observer.observe(s));
})();

/* ── 3. COUNT-UP STAT CARDS ───────────────────────────────── */
(function countUp() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const cards = document.querySelectorAll('.stat-number[data-target]');
  if (!cards.length) return;

  function animateCard(el) {
    const target   = parseFloat(el.dataset.target);
    const decimals = parseInt(el.dataset.decimals ?? '0', 10);
    const prefix   = el.dataset.prefix ?? '';
    const suffix   = el.dataset.suffix ?? '';
    const duration = 1200;
    const start    = performance.now();

    function frame(now) {
      const progress = Math.min((now - start) / duration, 1);
      const ease     = 1 - Math.pow(1 - progress, 3);
      const value    = (target * ease).toFixed(decimals);
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

/* ── 4. ALPHA TOGGLE ──────────────────────────────────────── */
(function alphaToggle() {
  const buttons   = document.querySelectorAll('.alpha-btn');
  const ratioEl   = document.querySelector('.alpha-ratio');
  const captionEl = document.querySelector('.alpha-caption');
  if (!buttons.length || !ratioEl || !captionEl) return;

  const data = {
    '0.0': {
      ratio:   '1.68 ± 0.42',
      caption: 'Baseline — near the random line',
    },
    '0.1': {
      ratio:   '2.27 ± 0.93',
      caption: 'Weak topography (+0.59 vs baseline)',
    },
    '1.0': {
      ratio:   '2.79 ± 1.24',
      caption: 'Strong topography (+1.12 vs baseline, p < 0.0001)',
    },
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const alpha = btn.dataset.alpha;
      if (data[alpha]) {
        ratioEl.textContent   = data[alpha].ratio;
        captionEl.textContent = data[alpha].caption;
      }
    });
  });
})();

/* ── 5. BIBTEX COPY ───────────────────────────────────────── */
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
      /* silently degrade — user can still select text manually */
    }
  });
})();

/* ── 6. TOOLTIP POPOVER ───────────────────────────────────── */
(function tooltips() {
  const abbrs = document.querySelectorAll('abbr.tooltip[data-tip]');
  if (!abbrs.length) return;

  let popup = null;

  function showTip(abbr) {
    popup = document.createElement('div');
    popup.className    = 'tooltip-popup';
    popup.textContent  = abbr.dataset.tip;
    popup.setAttribute('role', 'tooltip');
    document.body.appendChild(popup);

    const rect    = abbr.getBoundingClientRect();
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;
    const popW    = popup.offsetWidth;
    const popH    = popup.offsetHeight;

    const top  = rect.top + scrollY - popH - 8;
    const left = Math.min(
      rect.left + scrollX + rect.width / 2 - popW / 2,
      window.innerWidth + scrollX - popW - 16
    );
    popup.style.top  = `${Math.max(scrollY + 8, top)}px`;
    popup.style.left = `${Math.max(8, left)}px`;
  }

  function hideTip() {
    if (popup) { popup.remove(); popup = null; }
  }

  abbrs.forEach(abbr => {
    abbr.setAttribute('tabindex', '0');
    abbr.addEventListener('mouseenter', () => showTip(abbr));
    abbr.addEventListener('mouseleave', hideTip);
    abbr.addEventListener('focusin',    () => showTip(abbr));
    abbr.addEventListener('focusout',   hideTip);
  });
})();

/* Generic UI pieces — shared by every site this server hosts.
 *
 * Nothing here knows what an NPC is. Site-specific vocabulary (layer
 * colours, interaction modes, state glyphs) lives in that site's own
 * lib/ui.js, which re-exports this module so pages import one name. */

import { h, mount, fmtNum, ago, worldTime, truncId } from './dom.js';
import { link } from './router.js';

export function bar(frac, color) {
  return h('div', { class: 'bar' },
    h('i', { style: `width:${Math.max(0, Math.min(1, frac)) * 100}%;background:${color || 'var(--accent)'}` }));
}

export function kv(pairs) {
  const dl = h('dl', { class: 'kv' });
  for (const [k, v] of pairs) {
    if (v == null) continue;
    dl.appendChild(h('dt', {}, k));
    dl.appendChild(h('dd', {}, v instanceof Node ? v : String(v)));
  }
  return dl;
}

export function idBadge(id) {
  const el = h('code', { class: 'mono tiny dim', title: 'click to copy · ' + id, style: 'cursor:pointer' }, truncId(id));
  el.addEventListener('click', () => {
    navigator.clipboard?.writeText(String(id));
    toast('id copied', 'ok');
  });
  return el;
}

export function section(title, ...kids) {
  return h('div', {}, h('h2', {}, title), ...kids);
}

export function empty(glyph, title, detail, action) {
  return h('div', { class: 'empty' },
    h('div', { class: 'big' }, glyph),
    h('div', { style: 'font-weight:700;color:var(--ink-dim);margin-bottom:6px' }, title),
    detail ? h('div', { class: 'tiny' }, detail) : null,
    action ? h('div', { style: 'margin-top:16px' }, action) : null);
}

export function toast(text, kind) {
  const host = document.getElementById('toasts');
  if (!host) return;
  const el = h('div', { class: 'toast' + (kind ? ' ' + kind : '') }, text);
  host.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .2s, transform .2s';
    el.style.opacity = '0';
    el.style.transform = 'translateX(18px)';
    setTimeout(() => el.remove(), 220);
  }, 2600);
}

/** A modal. Returns { el, close }. Esc and scrim click both close. */
export function modal({ title, body, footer, wide }) {
  const close = () => el.remove();
  const box = h('div', { class: 'modal', style: wide ? 'width:min(960px,100%)' : '' },
    h('div', { class: 'modal-hd' },
      h('div', { style: 'font-weight:700' }, title),
      h('span', { class: 'spacer', style: 'flex:1' }),
      h('button', { class: 'btn ghost sm', onClick: close }, '✕')),
    h('div', { class: 'modal-bd' }, body),
    footer ? h('div', { class: 'modal-ft' }, footer) : null);
  const el = h('div', { class: 'scrim', onClick: (e) => { if (e.target === el) close(); } }, box);
  document.addEventListener('keydown', function esc(e) {
    if (e.key === 'Escape') { close(); document.removeEventListener('keydown', esc); }
  });
  document.body.appendChild(el);
  return { el, close };
}

export function confirmDialog({ title, message, confirmText, danger, requireText, onConfirm }) {
  let input = null;
  const go = () => {
    if (requireText && (!input || input.value !== requireText)) return;
    m.close(); onConfirm();
  };
  const btn = h('button', { class: 'btn ' + (danger ? 'danger' : 'primary'), onClick: go },
    confirmText || 'Confirm');
  if (requireText) btn.setAttribute('disabled', '');
  const body = h('div', {},
    h('p', { style: 'color:var(--ink-soft)' }, message),
    requireText
      ? h('label', { class: 'field' },
        h('span', {}, `type “${requireText}” to confirm`),
        (input = h('input', {
          class: 'input', placeholder: requireText,
          onInput: () => { if (input.value === requireText) btn.removeAttribute('disabled'); else btn.setAttribute('disabled', ''); },
        })))
      : null);
  const m = modal({
    title, body,
    footer: [h('button', { class: 'btn ghost', onClick: () => m.close() }, 'Cancel'), btn],
  });
  if (input) input.focus();
  return m;
}

/** Sparkline / line chart as inline SVG — no library, theme-aware via currentColor. */
export function lineChart(points, opts = {}) {
  const w = opts.width || 800, hh = opts.height || 240, pad = opts.pad || 26;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = opts.min != null ? opts.min : 0;
  const y1 = opts.max != null ? opts.max : Math.max(...ys) * 1.15 || 1;
  const px = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (w - pad * 2);
  const py = (y) => hh - pad - ((y - y0) / (y1 - y0 || 1)) * (hh - pad * 2);
  const d = points.map((p, i) => (i ? 'L' : 'M') + px(p.x).toFixed(1) + ' ' + py(p.y).toFixed(1)).join(' ');
  const area = d + ` L${px(x1).toFixed(1)} ${hh - pad} L${px(x0).toFixed(1)} ${hh - pad} Z`;

  const bands = (opts.bands || []).map((b) =>
    `<rect x="${pad}" y="${py(b.to)}" width="${w - pad * 2}" height="${Math.max(0, py(b.from) - py(b.to))}"
       fill="${b.color}" opacity="0.10"/>
     <line x1="${pad}" x2="${w - pad}" y1="${py(b.from)}" y2="${py(b.from)}"
       stroke="${b.color}" stroke-dasharray="3 4" opacity="0.6"/>
     <text x="${w - pad}" y="${py(b.from) - 5}" text-anchor="end" font-size="10"
       fill="${b.color}" font-family="var(--mono)">${b.label}</text>`).join('');

  const ticks = [y0, (y0 + y1) / 2, y1].map((v) =>
    `<line x1="${pad}" x2="${w - pad}" y1="${py(v)}" y2="${py(v)}" stroke="var(--line)"/>
     <text x="4" y="${py(v) + 3}" font-size="10" fill="var(--ink-ghost)" font-family="var(--mono)">${v.toFixed(2)}</text>`).join('');

  const svg = `<svg viewBox="0 0 ${w} ${hh}" preserveAspectRatio="none">
    ${ticks}${bands}
    <path d="${area}" fill="${opts.color || 'var(--accent)'}" opacity="0.12"/>
    <path d="${d}" fill="none" stroke="${opts.color || 'var(--accent)'}" stroke-width="1.8"
      stroke-linejoin="round" stroke-linecap="round"/>
  </svg>`;
  return h('div', { class: 'chart', html: svg });
}


export { h, mount, fmtNum, ago, worldTime, truncId, link };

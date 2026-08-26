/* Canvas charts, ported from zend's perf.html.
 *
 * Four disciplines that page paid for:
 *
 *   1. Create the chart ONCE. `optsFn` is re-invoked per refresh to pull fresh
 *      series, so nothing recreates DOM or re-adds listeners. (zend's earlier
 *      version appended a new canvas per refresh — the growing-page bug.)
 *   2. Store the logical height in a data-attribute. Drawing overwrites the
 *      height *attribute* with the dpr-scaled backing size, so re-reading it
 *      later doubles the canvas every refresh.
 *   3. Coalesce redraws to one per animation frame. Without this, every
 *      mousemove pixel redrew every series — the main cause of sluggish hover.
 *   4. Time runs backwards: the right edge is "now", labels are ages.
 */

const cssvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtN = (n, d = 0) => (isFinite(n) ? n : 0).toLocaleString(undefined,
  { maximumFractionDigits: d, minimumFractionDigits: d });

function schedule(spec) {
  if (spec._raf) return;
  spec._raf = requestAnimationFrame(() => { spec._raf = 0; spec.draw(); });
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

/**
 * makeChart(canvas, optsFn) → { refresh, destroy }
 *
 * optsFn() returns:
 *   { xs, series:[{data, color, type:'line'|'area'|'bar', axis:'l'|'r', dash, lw, baseData}],
 *     ymax?, y2?, yFmt?, y2Fmt?, markers:[{y,color}], xFmt?, tip(i) → [{t, c?}] }
 */
export function makeChart(cv, optsFn) {
  if (!cv) return null;
  const logical = cv.dataset.h ? +cv.dataset.h : (+cv.getAttribute('height') || 200);
  cv.dataset.h = logical;                 // never re-read the mutated attribute
  cv.style.height = logical + 'px';

  const spec = { cv, optsFn, opts: optsFn(), h: logical, _mx: null, _raf: 0 };
  spec.draw = () => draw(spec);
  spec.refresh = () => { spec.opts = spec.optsFn(); spec.draw(); };

  const ro = new ResizeObserver(() => schedule(spec));
  ro.observe(cv);
  cv.onmousemove = (e) => { spec._mx = e.clientX - cv.getBoundingClientRect().left; schedule(spec); };
  cv.onmouseleave = () => { spec._mx = null; schedule(spec); };

  spec.destroy = () => { ro.disconnect(); cv.onmousemove = cv.onmouseleave = null; };
  spec.draw();
  return spec;
}

function draw(spec) {
  const { cv, opts } = spec;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = cv.clientWidth || 600, cssH = spec.h;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  const xs = opts.xs;
  const ink = cssvar('--ink'), faint = cssvar('--ink-ghost'), grid = cssvar('--line');
  if (!xs || !xs.length) {
    g.fillStyle = faint; g.font = '11px ' + cssvar('--mono');
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(opts.emptyText || 'no samples yet', cssW / 2, cssH / 2);
    return;
  }

  const padL = opts.padL != null ? opts.padL : 46;
  const padR = opts.y2 ? 44 : 14, padT = 10, padB = 22;
  const plotW = cssW - padL - padR, plotH = cssH - padT - padB;

  const xmin = xs[0], xmax = (opts.xnow != null ? opts.xnow : xs[xs.length - 1]) || 1;
  const X = (t) => padL + (xmax === xmin ? 1 : (t - xmin) / (xmax - xmin)) * plotW;

  const lSeries = opts.series.filter((s) => (s.axis || 'l') === 'l');
  const rSeries = opts.series.filter((s) => s.axis === 'r');
  const ymax = opts.ymax != null ? opts.ymax : Math.max(1e-6, ...lSeries.flatMap((s) => s.data));
  const y2max = opts.y2 ? Math.max(1e-6, ...rSeries.flatMap((s) => s.data)) : 1;
  const Y = (v) => padT + plotH - (v / ymax) * plotH;
  const Y2 = (v) => padT + plotH - (v / y2max) * plotH;

  g.font = '10px ' + cssvar('--mono');
  g.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const yv = ymax * i / 4, y = Y(yv);
    g.strokeStyle = grid; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
    g.fillStyle = faint; g.textAlign = 'right';
    g.fillText(opts.yFmt ? opts.yFmt(yv) : fmtN(yv), padL - 7, y);
  }
  if (opts.y2) {
    g.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const yv = y2max * i / 4;
      g.fillStyle = opts.y2Color || faint;
      g.fillText(opts.y2Fmt ? opts.y2Fmt(yv) : fmtN(yv), padL + plotW + 7, Y2(yv));
    }
  }

  g.textAlign = 'center'; g.textBaseline = 'top';
  for (let i = 0; i <= 5; i++) {
    const tv = xmin + (xmax - xmin) * i / 5;
    g.fillStyle = faint;
    g.fillText(opts.xFmt ? opts.xFmt(tv, xmax) : String(Math.round(tv)), X(tv), padT + plotH + 6);
  }

  (opts.markers || []).forEach((m) => {
    if (m.y > ymax) return;
    g.strokeStyle = m.color; g.lineWidth = 1.2; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(padL, Y(m.y)); g.lineTo(padL + plotW, Y(m.y)); g.stroke();
    g.setLineDash([]);
  });

  opts.series.filter((s) => s.type === 'bar').forEach((s) => {
    const yf = s.axis === 'r' ? Y2 : Y;
    const bw = Math.max(1.5, plotW / xs.length * 0.7);
    g.fillStyle = s.color; g.globalAlpha = 0.85;
    for (let i = 0; i < xs.length; i++) {
      const x = X(xs[i]);
      g.fillRect(x - bw / 2, yf(s.data[i]), bw, (s.axis === 'r' ? Y2(0) : Y(0)) - yf(s.data[i]));
    }
    g.globalAlpha = 1;
  });

  opts.series.filter((s) => s.type !== 'bar').forEach((s) => {
    const yf = s.axis === 'r' ? Y2 : Y;
    if (s.type === 'area') {
      g.beginPath(); g.moveTo(X(xs[0]), yf(s.data[0]));
      for (let i = 1; i < xs.length; i++) g.lineTo(X(xs[i]), yf(s.data[i]));
      if (s.baseData) { for (let i = xs.length - 1; i >= 0; i--) g.lineTo(X(xs[i]), yf(s.baseData[i])); }
      else { g.lineTo(X(xs[xs.length - 1]), yf(0)); g.lineTo(X(xs[0]), yf(0)); }
      g.closePath();
      const grad = g.createLinearGradient(0, padT, 0, padT + plotH);
      grad.addColorStop(0, s.color + '55'); grad.addColorStop(1, s.color + '0a');
      g.fillStyle = grad; g.fill();
    }
    g.beginPath(); g.moveTo(X(xs[0]), yf(s.data[0]));
    for (let i = 1; i < xs.length; i++) g.lineTo(X(xs[i]), yf(s.data[i]));
    g.strokeStyle = s.color; g.lineWidth = s.lw || 1.8; g.lineJoin = 'round';
    if (s.dash) g.setLineDash([5, 4]);
    g.stroke(); g.setLineDash([]);
  });

  if (spec._mx != null && spec._mx >= padL && spec._mx <= padL + plotW && opts.tip) {
    let idx = 0, best = 1e9;
    for (let i = 0; i < xs.length; i++) {
      const d = Math.abs(X(xs[i]) - spec._mx);
      if (d < best) { best = d; idx = i; }
    }
    const hx = X(xs[idx]);
    g.strokeStyle = cssvar('--line-2'); g.lineWidth = 1;
    g.beginPath(); g.moveTo(hx, padT); g.lineTo(hx, padT + plotH); g.stroke();

    const rows = opts.tip(idx) || [];
    g.font = '10.5px ' + cssvar('--mono');
    const tw = Math.max(...rows.map((r) => g.measureText(r.t).width)) + 20;
    const th = rows.length * 15 + 8;
    let tx = hx + 11; if (tx + tw > cssW - 2) tx = hx - tw - 11; if (tx < 2) tx = 2;
    const ty = padT + 6;
    g.fillStyle = cssvar('--panel-2'); g.strokeStyle = cssvar('--line-2'); g.lineWidth = 1;
    roundRect(g, tx, ty, tw, th, 7); g.fill(); g.stroke();
    g.textBaseline = 'middle'; g.textAlign = 'left';
    rows.forEach((r, i) => {
      const yy = ty + 11 + i * 15;
      if (r.c) { g.fillStyle = r.c; g.fillRect(tx + 9, yy - 3, 7, 3); }
      g.fillStyle = r.c || ink;
      g.fillText(r.t, tx + (r.c ? 22 : 9), yy);
    });
    opts.series.forEach((s) => {
      const yf = s.axis === 'r' ? Y2 : Y;
      g.fillStyle = s.color;
      g.beginPath(); g.arc(hx, yf(s.data[idx]), 3, 0, 7); g.fill();
    });
  }
}

/** Stacked bars where height is a duration and colour is a category. */
export function makeStackChart(cv, optsFn) {
  if (!cv) return null;
  const logical = cv.dataset.h ? +cv.dataset.h : (+cv.getAttribute('height') || 200);
  cv.dataset.h = logical;
  cv.style.height = logical + 'px';

  const spec = { cv, optsFn, opts: optsFn(), h: logical, _mx: null, _raf: 0 };
  spec.draw = () => drawStack(spec);
  spec.refresh = () => { spec.opts = spec.optsFn(); spec.draw(); };
  const ro = new ResizeObserver(() => schedule(spec));
  ro.observe(cv);
  cv.onmousemove = (e) => { spec._mx = e.clientX - cv.getBoundingClientRect().left; schedule(spec); };
  cv.onmouseleave = () => { spec._mx = null; schedule(spec); };
  spec.destroy = () => { ro.disconnect(); cv.onmousemove = cv.onmouseleave = null; };
  spec.draw();
  return spec;
}

function drawStack(spec) {
  const { cv, opts } = spec;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = cv.clientWidth || 600, cssH = spec.h;
  cv.width = cssW * dpr; cv.height = cssH * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  const bars = opts.bars || [];
  const faint = cssvar('--ink-ghost'), grid = cssvar('--line');
  if (!bars.length) {
    g.fillStyle = faint; g.font = '11px ' + cssvar('--mono');
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText(opts.emptyText || 'waiting for data…', cssW / 2, cssH / 2);
    return;
  }

  const padL = 50, padR = 14, padT = 10, padB = 22;
  const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
  const keys = opts.keys;
  const total = (b) => keys.reduce((a, k) => a + (b.parts[k] || 0), 0);
  // Scale to VISIBLE bars only — an off-screen sliver must not shrink everything.
  const ymax = Math.max(1, ...bars.filter((b) => b.visible !== false).map(total)) * 1.05;
  const y0 = padT + plotH;

  g.font = '10px ' + cssvar('--mono'); g.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const yv = ymax * i / 4, y = y0 - (yv / ymax) * plotH;
    g.strokeStyle = grid; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + plotW, y); g.stroke();
    g.fillStyle = faint; g.textAlign = 'right';
    g.fillText(opts.yFmt ? opts.yFmt(yv) : fmtN(yv), padL - 7, y);
  }

  const n = bars.length;
  const bw = Math.max(2, (plotW / n) * 0.82);
  const xOf = (i) => padL + (i + 0.5) * (plotW / n);

  bars.forEach((b, i) => {
    let acc = 0;
    keys.forEach((k) => {
      const v = b.parts[k] || 0;
      if (!v) return;
      const hgt = v / ymax * plotH;
      g.fillStyle = opts.color(k); g.globalAlpha = 0.92;
      g.fillRect(xOf(i) - bw / 2, y0 - acc - hgt, bw, hgt);
      g.globalAlpha = 1;
      acc += hgt;
    });
  });

  g.textAlign = 'center'; g.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const bi = Math.round((n - 1) * i / 4);
    g.fillStyle = faint;
    g.fillText(opts.xLabel ? opts.xLabel(bars[bi], bi) : String(bi), xOf(bi), y0 + 6);
  }

  if (spec._mx != null && opts.tip) {
    let idx = -1, best = 1e9;
    bars.forEach((_, i) => { const d = Math.abs(xOf(i) - spec._mx); if (d < best) { best = d; idx = i; } });
    if (idx >= 0 && best < (plotW / n) * 0.9) {
      const hx = xOf(idx);
      g.strokeStyle = cssvar('--line-2'); g.lineWidth = 1;
      g.beginPath(); g.moveTo(hx, padT); g.lineTo(hx, y0); g.stroke();
      const rows = opts.tip(bars[idx], idx) || [];
      g.font = '10.5px ' + cssvar('--mono');
      const tw = Math.max(...rows.map((r) => g.measureText(r.t).width)) + 20;
      const th = rows.length * 15 + 8;
      let tx = hx + 12; if (tx + tw > cssW - 2) tx = hx - tw - 12; if (tx < 2) tx = 2;
      const ty = padT + 6;
      g.fillStyle = cssvar('--panel-2'); g.strokeStyle = cssvar('--line-2'); g.lineWidth = 1;
      roundRect(g, tx, ty, tw, th, 7); g.fill(); g.stroke();
      g.textBaseline = 'middle'; g.textAlign = 'left';
      rows.forEach((r, i) => {
        const yy = ty + 11 + i * 15;
        if (r.c) { g.fillStyle = r.c; g.fillRect(tx + 9, yy - 3, 7, 3); }
        g.fillStyle = r.c || cssvar('--ink');
        g.fillText(r.t, tx + (r.c ? 22 : 9), yy);
      });
    }
  }
}

export const ageFmt = (t, now) => ((now - t) / 60).toFixed(1) + 'm';

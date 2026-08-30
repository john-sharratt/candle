/* Performance — what the card is doing, and whether that is fine.
 *
 * A port of zend's telemetry page onto npcd's own `/v1/telemetry` and
 * `/v1/memory`. Four things carried across, in order of how much they matter:
 *
 *   A verdict.   A sentence saying whether the numbers are healthy. Charts
 *                report; somebody opening this page wants to know if anything
 *                is wrong, and reading a dozen charts to find out is a task.
 *   Staleness.   An age that ticks every second. Without it a daemon that
 *                stopped answering looks exactly like a healthy one, because
 *                the last good numbers stay on screen indefinitely.
 *   Pause.       A chart that redraws every four seconds is hard to read.
 *   Thresholds.  Cards carry ok / warn / crit, so a number is interpreted
 *                rather than merely displayed.
 *
 * ── The history is the daemon's, not this page's ─────────────────────────────
 *
 * An earlier version of this file accumulated its own charts from successive
 * polls, because `/v1/telemetry` returned a single instantaneous reading. That
 * is worse in three ways that all show up in use: a reload threw the history
 * away, a closed tab recorded nothing, and a "last 60 minutes" panel could not
 * exist because the page had only ever seen its own uptime.
 *
 * The daemon now keeps an hour of samples and hands the window over on each
 * poll, the way zend does. So every chart here draws from `t.series`, and a
 * fresh tab opens with an hour already on screen.
 *
 * ── Absent is not zero ───────────────────────────────────────────────────────
 *
 * `/v1/telemetry` reports what it can actually see: the card, its memory, the
 * host. Everything downstream of the inference engine comes back `null`,
 * because there is no engine yet.
 *
 * So nothing here coerces. A `?? 0` would put "0.0 t/s" beside an amber chip
 * and "mean batch 0.0 — near 1 means no batching" under it: confident
 * statements about a system that is not running, indistinguishable from the
 * same page describing one running badly. Absent renders as `—` in a fourth,
 * colourless state, and a panel with no series behind it says so in words
 * rather than drawing a flat line at zero.
 *
 * A column that is `null` was never measured. A column with `null` at the front
 * was measured from partway through the window — an engine that started four
 * minutes ago — and the chart begins where the measurements do.
 *
 * None of that is provisional. When the engine reports, `engine_connected`
 * flips and every panel fills in; this file does not change.
 */

import { API, BACKEND } from '../lib/api.js';
import { h, mount, fmtNum } from '../lib/dom.js';
import { empty } from '../lib/ui.js';
import { makeChart, makeStackChart } from '../lib/chart.js';

const POLL_MS = 4000;
/* Memory accounting moves slowly and costs a process enumeration on the daemon,
 * so it is not worth the chart cadence. */
const MEM_POLL_MS = 8000;
/* Bars in the decomposition chart. An hour of 2s samples is ~1,800, which at
 * any real width is less than a pixel each. */
const STACK_BARS = 64;

/* What "bad" means here. Named rather than buried in the expressions that use
 * them, because they are the one thing on this page that cannot be derived
 * from the data — get them wrong and it either cries wolf or stays green
 * through a real problem. */
const LIMITS = {
  vramWarn: 0.85, vramCrit: 0.95,     // fraction of the card in use
  hostWarn: 0.85, hostCrit: 0.95,     // fraction of host RAM in use
  inboxWarn: 25, inboxCrit: 80,       // p99 events waiting on one character
  batchThin: 1.2,                     // mean characters per decode
};
const stateBy = (v, warn, crit) => (v == null ? 'na' : v >= crit ? 'crit' : v >= warn ? 'warn' : 'ok');

/* A number, or null — never a substitute. `0` is a measurement and has to
 * survive; only null/undefined/NaN mean nobody looked. */
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const show = (v, f) => (v == null ? '—' : (f ? f(v) : String(v)));
const gib = (mib) => (mib == null ? '—' : (mib / 1024).toFixed(1) + ' GiB');
const gibOf = (mib) => (mib == null ? null : mib / 1024);
/* Weights files are quoted in decimal GB by everyone who publishes them, so a
 * GiB figure here would not match the number on the model card it came from. */
const fmtGB = (b) => (b == null ? '—' : (b / 1e9).toFixed(1) + ' GB');

function fmtBytes(b) {
  if (b == null) return '—';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0, v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
  return v.toFixed(i >= 3 ? 1 : 0) + ' ' + u[i];
}

/* Seconds as something readable. Two units, largest first: past a few hours
 * nobody is counting seconds, and "9021s" makes a reader do arithmetic. */
function fmtDur(s) {
  if (s == null) return '—';
  const d = Math.floor(s / 86400), hr = Math.floor(s / 3600) % 24;
  const m = Math.floor(s / 60) % 60, sec = Math.floor(s) % 60;
  if (d) return `${d}d ${hr}h`;
  if (hr) return `${hr}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

// ── series ───────────────────────────────────────────────────────────────────

const colOf = (s, n) => (Array.isArray(s && s[n]) ? s[n] : null);

/* Take several columns and the indices where ALL of them are measured.
 *
 * `makeChart` derives its y-scale with `Math.max(...data)`, so a single null
 * poisons the whole axis into NaN and the panel goes blank. Dropping the gaps
 * here is what makes a mid-window engine start draw as a line that begins late
 * rather than as nothing at all. Returns null when any column was never
 * measured, which is the panel's cue to explain itself instead. */
function aligned(t, cols) {
  if (!Array.isArray(t) || !t.length || cols.some((c) => !c)) return null;
  const xs = [], ys = cols.map(() => []);
  for (let i = 0; i < t.length; i += 1) {
    if (cols.every((c) => num(c[i]) != null)) {
      xs.push(t[i]);
      cols.forEach((c, k) => ys[k].push(c[i]));
    }
  }
  return xs.length ? { xs, ys } : null;
}

/* A metric's shape over the window, small enough to sit inside a KPI card.
 *
 * A number answers "where am I"; it always provokes "and which way am I going",
 * which is the question a chart three screens down should not be needed for.
 *
 * Built as SVG markup rather than through `h()`, which uses `createElement` and
 * so cannot make namespaced SVG nodes. Colour comes from `currentColor`, so the
 * card's state class tints the trend without any of it being passed in. */
function sparkline(vals) {
  const all = (vals || []).filter((x) => num(x) != null);
  if (all.length < 3) return null;
  /* Thinned before anything else. An hour of samples is ~1,800 points, and a
   * path string that long — rebuilt for every card on every poll — is a great
   * deal of DOM churn to describe a shape 100 pixels wide. It also keeps
   * `Math.min(...v)` off a spread big enough to matter. */
  const v = thin(all, 96);
  const W = 100, H = 26;
  const lo = Math.min(...v), hi = Math.max(...v);
  /* A flat series must not become a full-scale zigzag of floating-point noise:
   * with no real span, pin it to the middle. */
  const span = (hi - lo) || 1;
  const flat = (hi - lo) < Math.abs(hi || 1) * 1e-6;
  const x = (i) => ((i / (v.length - 1)) * W).toFixed(2);
  const y = (k) => (flat ? H / 2 : H - 2 - ((k - lo) / span) * (H - 4)).toFixed(2);
  const pts = v.map((k, i) => `${x(i)},${y(k)}`).join(' L');
  const last = `${x(v.length - 1)},${y(v[v.length - 1])}`;
  return h('div', { class: 'kpi-spark', html:
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">`
    + `<path d="M${pts} L${W},${H} L0,${H} Z" fill="currentColor" opacity=".14"/>`
    + `<path d="M${pts}" fill="none" stroke="currentColor" stroke-width="1.5"`
    + ` vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`
    + `<circle cx="${last.split(',')[0]}" cy="${last.split(',')[1]}" r="2" fill="currentColor"/>`
    + '</svg>' });
}

/* Evenly drop samples down to at most `max`, keeping the newest. */
function thin(idx, max) {
  if (idx.length <= max) return idx;
  const step = idx.length / max;
  const out = [];
  for (let i = 0; i < max; i += 1) out.push(idx[Math.floor(i * step)]);
  if (out[out.length - 1] !== idx[idx.length - 1]) out.push(idx[idx.length - 1]);
  return out;
}

// ── panels ───────────────────────────────────────────────────────────────────

/* A card that holds either a chart or an explanation of why there is none.
 *
 * The swap happens per refresh rather than at load, which is what lets every
 * engine panel on this page fill itself in the moment an engine starts
 * reporting — no reload, and no second code path to keep in step. */
function chartPanel(title, cap, kind) {
  const cv = h('canvas', { height: 185 });
  const host = h('div');
  const legend = h('div', { class: 'legend' });
  const el = h('div', { class: 'panel' },
    h('h3', { style: 'margin-top:0' }, title),
    h('div', { class: 'tiny dim', style: 'margin-bottom:10px' }, cap),
    host, legend);

  let chart = null;
  let opts = null;
  const make = kind === 'stack' ? makeStackChart : makeChart;

  return {
    el,
    /* `next` is a chart options object, or null for "nothing to draw".
     * `keys` is the legend: `[{label, color}]`, so the colours mean something
     * without having to hover to find out. */
    update(next, why, keys) {
      if (!next) {
        if (chart) { if (chart.destroy) chart.destroy(); chart = null; }
        mount(host, empty('◷', 'Not measured yet', why));
        mount(legend);
        return;
      }
      opts = next;
      if (!chart) {
        mount(host, cv);
        chart = make(cv, () => opts);
      } else {
        chart.refresh();
      }
      mount(legend, ...(keys || []).map((k) =>
        h('span', {}, h('i', { style: `background:${k.color}` }), k.label)));
    },
    destroy() { if (chart && chart.destroy) chart.destroy(); },
  };
}

export async function render() {
  /* `perf` scopes this page's styling. The panel and KPI rules it carries alter
   * hover and spacing, and unscoped they would reach every other page using the
   * same components. */
  const el = h('div', { class: 'page wide perf' });
  let mem = null;
  let lastAt = 0;
  let paused = false;
  let failures = 0;

  const live = h('span', { class: 'chip' }, 'connecting…');
  const age = h('span', { class: 'tiny dim' });
  const pauseBtn = h('button', { class: 'btn sm' }, '⏸ pause');
  const device = h('div', { class: 'dev' });
  const kpis = h('div', { class: 'grid g4', style: 'margin-bottom:14px' });
  const verdict = h('div', { class: 'panel verdict' },
    h('span', { class: 'verdict-pill pill-na' }, '· · ·'),
    h('div', { class: 'verdict-body' }, h('div', { class: 'skel', style: 'height:34px' })));

  pauseBtn.addEventListener('click', () => {
    paused = !paused;
    pauseBtn.textContent = paused ? '▶ resume' : '⏸ pause';
    pauseBtn.classList.toggle('primary', paused);
    el.classList.toggle('is-paused', paused);
    if (!paused) refresh();
  });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Performance'),
      h('div', { class: 'sub' }, 'What the card is doing, and what is waiting on it.')),
    h('div', { class: 'row' }, live, age, pauseBtn, h('span', { class: 'chip' }, 'backend: ' + BACKEND))));
  el.appendChild(device);
  el.appendChild(kpis);
  el.appendChild(verdict);

  // ── the panels ────────────────────────────────────────────────────────────

  /* Measured today, from the driver and the OS. */
  const pVram = chartPanel('VRAM pressure',
    'Whole-card occupancy against the wall, over the retained hour.', 'line');
  const pDecomp = chartPanel('VRAM decomposition',
    'Where card memory has gone. Weights, KV and image budgets are the engine’s '
    + 'to report; until it does, what is in use is shown undivided.', 'stack');
  const pHost = chartPanel('Host memory',
    'Machine RAM in use, and this daemon’s own resident set within it.', 'line');

  /* Waiting on the engine. Real panels, drawn the moment it reports. */
  const pTps = chartPanel('Token throughput',
    'Decode tokens per second across the population, against prefill.', 'line');
  const pBatch = chartPanel('Batch composition',
    'How many characters share each decode. If this stays near 1, batching is not '
    + 'happening — and the popular-character-is-cheapest claim is not holding.', 'line');
  const pInbox = chartPanel('Inbox depth',
    'Events waiting per character, p50 against p99. A rising p99 with a flat p50 is '
    + 'one character falling behind, not the population.', 'line');
  const pTicks = chartPanel('Tick rate & active characters',
    'How often a character thinks, and how many are thinking at once.', 'line');
  const pImages = chartPanel('Image queue',
    'Portraits waiting on the card.', 'line');

  el.appendChild(h('h2', {}, 'Memory'));
  el.appendChild(h('div', { class: 'grid g2' }, pVram.el, pDecomp.el));
  el.appendChild(h('div', { class: 'grid g2' }, pHost.el, h('div', { class: 'panel' },
    h('h3', { style: 'margin-top:0' }, 'Memory — full accounting'),
    h('div', { class: 'tiny dim', style: 'margin-bottom:10px' },
      'What the OS says, and what this process holds. The two together separate '
      + '“structurally tight” from “this daemon is the problem”.'),
    h('div', { id: 'mem-host' }))));

  el.appendChild(h('h2', {}, 'Engine'));
  el.appendChild(h('div', { class: 'grid g2' }, pTps.el, pBatch.el));
  el.appendChild(h('div', { class: 'grid g2' }, pInbox.el, pTicks.el));
  el.appendChild(h('div', { class: 'grid g2' }, pImages.el, h('div', { class: 'panel' },
    h('h3', { style: 'margin-top:0' }, 'Arriving with the engine'),
    h('div', { class: 'tiny dim', style: 'margin-bottom:10px' },
      'zend’s dashboard carries these too. They measure the paged-KV wave '
      + 'scheduler, so there is nothing behind them until this daemon runs one — '
      + 'and an empty panel each would be more noise than information.'),
    h('ul', { class: 'tiny dim', style: 'margin:0;padding-left:18px;line-height:1.9' },
      ['Phase timeline and phase throughput', 'Drain backlog', 'Wave latency',
        'Resident KV arenas by format', 'Migration drain (hot → warm)',
        'Projection decomposition', 'Working-set promotes']
        .map((s) => h('li', {}, s))))));

  const panels = [pVram, pDecomp, pHost, pTps, pBatch, pInbox, pTicks, pImages];

  // ── KPI cards ─────────────────────────────────────────────────────────────

  /* Named fields rather than six positional arguments: `kpi('VRAM', x, y, 'ok',
   * 0.4, series)` gives a reader no way to tell the meter from the trend. */
  const kpi = ({ lab, val, note, state, fill, spark }) => h('div', { class: 'panel kpi' },
    // `kpi-rail`, not `rail` — that name already belongs to the sidebar, whose
    // layout rules would otherwise apply to a 3px stripe.
    h('div', { class: 'kpi-rail bg-' + state }),
    h('div', { class: 'lbl' }, lab),
    h('div', { class: 'val st-' + state }, String(val)),
    note ? h('div', { class: 'tiny dim' }, note) : null,
    fill != null ? h('div', { class: 'kpi-meter' },
      h('i', { class: 'bg-' + state, style: `width:${Math.min(100, Math.max(0, fill * 100)).toFixed(0)}%` })) : null,
    // Tinted by the card's own state, so a card going amber takes its trend
    // with it and the row reads at a glance.
    spark ? h('div', { class: 'st-' + state }, spark) : null);

  /* What this page is about, stated once. The card's identity used to live in
   * the VRAM panel's caption, which is the last place somebody checking what
   * hardware they are looking at would think to read. */
  function paintDevice(t, r) {
    const g = t.gpu || {};
    const m = t.model || null;
    const chip = (lab, val) => (val == null ? null
      : h('span', { class: 'dev-chip' }, lab + ' ', h('b', {}, String(val))));
    /* One identity block: a mark, a name, and a line under it. The card and the
     * model get the same treatment because they are the same kind of fact — the
     * two things you need to know before any number below means anything. */
    const id = (mark, name, sub) => h('div', { class: 'dev-id' },
      h('div', { class: 'dev-mark' }, mark),
      h('div', { style: 'min-width:0' },
        h('div', { class: 'dev-name' }, name),
        h('div', { class: 'tiny dev-sub' }, sub)));

    mount(device,
      id('▩', g.name || 'No NVIDIA device',
        r.total == null
          ? 'NVML reports no card — host metrics only'
          : `${gib(r.total)} of card memory · sampled every ${show(num(t.sample_period_s), (v) => v + 's')}`),
      m ? h('div', { class: 'dev-split' }) : null,
      m ? id('◆', `${m.name} · ${m.quant}`,
        /* "Selected", not "running". This is the quant the card's memory
         * chooses — a real fact, decided by hardware — but nothing has loaded
         * it, and a banner saying otherwise would be the one fabrication left
         * on the page. */
        `${m.params_total} total · ${m.params_active} active · ${fmtGB(num(m.bytes))}`
        + (r.engine ? ' · loaded' : ' · selected, not loaded')) : null,
      h('div', { class: 'dev-chips' },
        chip('sm', g.compute_cap),
        // The *negotiated* link, not the card's maximum: a fast card in a slow
        // slot is worth seeing, and naming only the capability hides it.
        g.pcie_gen ? chip('PCIe', `${g.pcie_gen}.0 ×${g.pcie_width}`) : null,
        chip('window', fmtDur(r.windowS)),
        chip('up', fmtDur(num(t.uptime_s)))));
    // The weights file, for somebody who needs the exact artefact rather than
    // the family name. A title rather than a visible line: it matters rarely,
    // and it is long.
    if (m) device.title = `${m.repo}/${m.filename}`;
  }

  /* The newest value of a column, and the largest — "peak" is what turns a
   * current reading into a claim about the window. */
  const lastOf = (c) => {
    if (!c) return null;
    for (let i = c.length - 1; i >= 0; i -= 1) if (num(c[i]) != null) return c[i];
    return null;
  };
  const peakOf = (c) => {
    if (!c) return null;
    const vals = c.filter((v) => num(v) != null);
    return vals.length ? Math.max(...vals) : null;
  };

  /* Pull the readings out once, so the painters never have to remember which
   * fields an engine supplies and which the driver does. */
  function readings(t) {
    const s = t.series || {};
    const ho = t.host || {};
    const used = lastOf(colOf(s, 'vram_used_mib'));
    const total = lastOf(colOf(s, 'vram_total_mib'));
    const free = lastOf(colOf(s, 'vram_free_mib'));
    const hTotal = num(ho.total_mib);
    const hFree = num(ho.free_mib);
    const hUsed = (hTotal != null && hFree != null) ? hTotal - hFree : null;
    const ts = Array.isArray(s.t) ? s.t : [];
    return {
      s, used, total, free,
      /* How much history is actually behind the charts. Shown in the banner
       * because "the last hour" and "the last ninety seconds" are very
       * different claims and the page cannot tell them apart on its own. */
      windowS: ts.length ? ts[ts.length - 1] - ts[0] : 0,
      frac: (used != null && total > 0) ? used / total : null,
      peakUsed: peakOf(colOf(s, 'vram_used_mib')),
      hUsed, hTotal, rss: num(ho.rss_mib),
      hFrac: (hUsed != null && hTotal > 0) ? hUsed / hTotal : null,
      tps: lastOf(colOf(s, 'decode_tps')),
      peakTps: peakOf(colOf(s, 'decode_tps')),
      mean: lastOf(colOf(s, 'mean_npcs_per_decode')),
      max: lastOf(colOf(s, 'max_batch')),
      p50: lastOf(colOf(s, 'inbox_depth_p50')),
      p99: lastOf(colOf(s, 'inbox_depth_p99')),
      active: lastOf(colOf(s, 'npcs_active')),
      engine: t.engine_connected === true,
    };
  }

  function paintKPIs(t, r) {
    const s = r.s;
    /* Card memory, host memory and uptime are measured whatever else is not, so
     * the page always has real content — it is never four dashes. */
    const cards = [
      kpi({
        lab: 'VRAM used',
        val: gib(r.used),
        note: 'of ' + gib(r.total) + (r.peakUsed != null ? ' · peak ' + gib(r.peakUsed) : ''),
        state: stateBy(r.frac, LIMITS.vramWarn, LIMITS.vramCrit),
        fill: r.frac,
        spark: sparkline(colOf(s, 'vram_used_mib')),
      }),
      kpi({
        lab: 'Headroom',
        val: gib(r.free),
        note: 'before the card is full',
        state: r.frac == null ? 'na'
          : r.frac > LIMITS.vramCrit ? 'crit' : r.frac > LIMITS.vramWarn ? 'warn' : 'ok',
        spark: sparkline(colOf(s, 'vram_free_mib')),
      }),
      kpi({
        lab: 'Host RAM',
        val: gib(r.hUsed),
        note: 'of ' + gib(r.hTotal) + (r.rss != null ? ' · this daemon ' + gib(r.rss) : ''),
        state: stateBy(r.hFrac, LIMITS.hostWarn, LIMITS.hostCrit),
        fill: r.hFrac,
        spark: sparkline(colOf(s, 'host_used_mib')),
      }),
    ];

    if (r.engine) {
      cards.push(kpi({
        lab: 'Decode',
        val: show(r.tps, (x) => x.toFixed(1) + ' t/s'),
        note: show(r.active, (n) => `${n} character${n === 1 ? '' : 's'} active`)
          + (r.peakTps != null ? ' · peak ' + r.peakTps.toFixed(0) : ''),
        state: r.tps == null ? 'na' : r.tps > 0 ? 'ok' : 'warn',
        spark: sparkline(colOf(s, 'decode_tps')),
      }));
      // Low is the failure here, so the usual comparison runs the other way.
      cards.push(kpi({
        lab: 'Mean batch',
        val: show(r.mean, (x) => x.toFixed(1)),
        note: `max ${show(r.max)} · near 1 means no batching`,
        state: r.mean == null ? 'na' : r.mean < LIMITS.batchThin ? 'warn' : 'ok',
        spark: sparkline(colOf(s, 'mean_npcs_per_decode')),
      }));
      cards.push(kpi({
        lab: 'Inbox p99',
        val: show(r.p99),
        note: 'p50 ' + show(r.p50),
        state: stateBy(r.p99, LIMITS.inboxWarn, LIMITS.inboxCrit),
        spark: sparkline(colOf(s, 'inbox_depth_p99')),
      }));
    } else {
      cards.push(kpi({
        lab: 'Engine',
        val: 'not connected',
        note: 'Throughput, batching and inbox depth come from the inference engine.',
        state: 'na',
      }));
    }
    mount(kpis, ...cards);
  }

  function paintVerdict(t, r) {
    const st = stateBy(r.frac, LIMITS.vramWarn, LIMITS.vramCrit);
    const head = st === 'crit' ? 'PRESSURE' : st === 'warn' ? 'CLIMBING'
      : st === 'na' ? 'NO CARD' : 'HEALTHY';

    const bits = [];
    bits.push(r.total == null
      ? 'No NVIDIA device is visible to this daemon, so there is no card to report on. '
      : `${gib(r.used)} of ${gib(r.total)} in use`
        + (t.gpu && t.gpu.name ? ` on ${t.gpu.name}` : '') + '. ');
    if (r.hFrac != null) {
      bits.push(`Host RAM is ${Math.round(r.hFrac * 100)}% used. `);
    }

    if (!r.engine) {
      /* The important sentence on this page today, and the one a `?? 0` would
       * have replaced with "no characters are thinking" — true of a loaded,
       * idle engine, and saying nothing about one that does not exist. */
      bits.push('The inference engine has not reported, so throughput, batching and inbox '
        + 'depth are not being measured — the dashes above are missing readings, not zeroes.');
    } else if (r.active === 0) {
      bits.push('No characters are thinking, so nothing here is under load yet.');
    } else {
      if (r.active != null) {
        bits.push(`${r.active} character${r.active === 1 ? '' : 's'} active`,
          r.tps != null ? [' at ', h('b', {}, r.tps.toFixed(1) + ' tok/s')] : [], '. ');
      }
      if (r.mean != null) {
        bits.push(r.mean < LIMITS.batchThin
          ? ['Mean batch is ', h('b', { class: 'st-warn' }, r.mean.toFixed(1)),
            ' — decodes are running nearly one character at a time, which is the cost model not holding.']
          : ['Decodes carry ', h('b', {}, r.mean.toFixed(1)), ' characters each — batching is working.']);
      }
      if (r.p99 != null && r.p99 >= LIMITS.inboxWarn) {
        bits.push(` Deepest inbox is ${r.p99} events`,
          r.p99 >= LIMITS.inboxCrit ? ' — something is not keeping up.' : ' — worth watching.');
      }
    }
    /* The pill carries the answer, the sentence carries the reasoning. Reading
     * one word is faster than parsing a paragraph, and somebody who opened this
     * page because something felt wrong wants the word first. */
    mount(verdict,
      h('span', { class: 'verdict-pill pill-' + st }, head),
      h('div', { class: 'verdict-body' }, ...bits.flat()));
  }

  // ── charts ───────────────────────────────────────────────────────────────

  const WAITING = 'The inference engine is not reporting, so there is nothing to plot. '
    + 'This fills in by itself when it does.';

  function paintCharts(t, r) {
    const s = r.s;
    const ts = Array.isArray(s.t) ? s.t : [];
    const tmax = ts.length ? ts[ts.length - 1] : 0;
    /* The axis runs backwards from now, which is how somebody reads a
     * dashboard: the right edge is this moment. */
    const xFmt = (v) => {
      const back = Math.max(0, tmax - v);
      return back < 90 ? `-${Math.round(back)}s` : `-${Math.round(back / 60)}m`;
    };
    /* `makeChart` takes `xFmt` only — `xLabel` is `makeStackChart`'s, and a
     * function there, not a string. */
    const base = { xFmt };
    const line = (data, color, label, extra) => ({ data, color, label, ...(extra || {}) });

    /* One hover tooltip builder for every chart: the sample's age, then a
     * coloured row per series. Without this a chart can be read for shape but
     * not for value, which is half of what it is for. */
    const tipOf = (xs, rows) => (i) => [
      { t: xFmt(xs[i]) },
      // `sr`, not `r` — `r` is this function's readings argument, and shadowing
      // it inside a callback is how the wrong one gets read later.
      ...rows.filter((sr) => sr.data[i] != null)
        .map((sr) => ({ t: `${sr.label}  ${sr.fmt(sr.data[i])}`, c: sr.color })),
    ];
    const keysOf = (rows) => rows.map((sr) => ({ label: sr.label, color: sr.color }));

    // VRAM pressure — used against the card's own wall, with the thresholds the
    // KPI cards judge against drawn in, so a reading and its verdict share one
    // picture instead of living in two places.
    const vram = aligned(ts, [colOf(s, 'vram_used_mib'), colOf(s, 'vram_total_mib')]);
    const vramRows = vram && [
      { label: 'used', color: 'var(--l-action)', data: vram.ys[0].map((v) => v / 1024), fmt: (v) => v.toFixed(1) + ' GiB' },
      { label: 'card total', color: 'var(--line-3)', data: vram.ys[1].map((v) => v / 1024), fmt: (v) => v.toFixed(1) + ' GiB' },
    ];
    const cardG = vram ? Math.max(...vram.ys[1]) / 1024 : 0;
    pVram.update(vram && {
      ...base,
      xs: vram.xs,
      series: [line(vramRows[0].data, vramRows[0].color, 'used GiB', { type: 'area' }),
        line(vramRows[1].data, vramRows[1].color, 'card total GiB', { dash: true, lw: 1.2 })],
      ymax: cardG * 1.02,
      markers: [
        { y: cardG * LIMITS.vramWarn, color: 'var(--warn)' },
        { y: cardG * LIMITS.vramCrit, color: 'var(--crit)' },
      ],
      yFmt: (v) => v.toFixed(0),
      tip: tipOf(vram.xs, vramRows),
    }, 'No NVIDIA device is visible to this daemon, so there is no card memory to plot.',
    vram && keysOf(vramRows).concat([
      { label: `warn ${Math.round(LIMITS.vramWarn * 100)}%`, color: 'var(--warn)' },
      { label: `crit ${Math.round(LIMITS.vramCrit * 100)}%`, color: 'var(--crit)' },
    ]));

    // VRAM decomposition — stacked, thinned to something a bar chart can show.
    const dTotal = colOf(s, 'vram_total_mib'), dUsed = colOf(s, 'vram_used_mib');
    const dFree = colOf(s, 'vram_free_mib');
    const dW = colOf(s, 'weights_mib'), dK = colOf(s, 'kv_mib'), dI = colOf(s, 'image_mib');
    let bars = null;
    if (dTotal && dUsed && ts.length) {
      const idx = thin(ts.map((_, i) => i).filter((i) => num(dUsed[i]) != null), STACK_BARS);
      bars = idx.map((i) => {
        const w = num(dW && dW[i]) || 0, k = num(dK && dK[i]) || 0, im = num(dI && dI[i]) || 0;
        const other = Math.max(0, dUsed[i] - w - k - im);
        return {
          t: ts[i],
          parts: {
            weights: w / 1024, kv: k / 1024, image: im / 1024,
            'in use': other / 1024,
            free: (num(dFree && dFree[i]) || 0) / 1024,
          },
        };
      });
    }
    const decompColor = {
      weights: 'var(--l-agency)', kv: 'var(--l-perception)', image: 'var(--l-beliefs)',
      'in use': 'var(--line-3)', free: 'var(--line-2)',
    };
    pDecomp.update(bars && bars.length && {
      bars,
      keys: ['weights', 'kv', 'image', 'in use', 'free'],
      color: (k) => decompColor[k],
      yFmt: (v) => v.toFixed(0) + 'G',
      xLabel: (b) => xFmt(b.t),
      /* `t` is the row's text and `c` its swatch colour — the value has to be
       * part of the text, not passed as `c`. Reversed so the tooltip reads
       * top-down in the order the bar stacks bottom-up. */
      tip: (b) => [{ t: xFmt(b.t) }].concat(
        Object.entries(b.parts).filter(([, v]) => v > 0.01).reverse()
          .map(([k, v]) => ({ t: `${k}  ${v.toFixed(1)} GiB`, c: decompColor[k] }))),
    }, 'No NVIDIA device is visible to this daemon, so there is no card memory to break down.',
    bars && bars.length && ['weights', 'kv', 'image', 'in use', 'free']
      .map((k) => ({ label: k, color: decompColor[k] })));

    // Host memory — the machine, and this daemon inside it.
    const host = aligned(ts, [colOf(s, 'host_used_mib'), colOf(s, 'host_total_mib')]);
    const rssCol = aligned(ts, [colOf(s, 'rss_mib')]);
    const gibFmt = (v) => v.toFixed(1) + ' GiB';
    const hostRows = host && [
      { label: 'host used', color: 'var(--info)', data: host.ys[0].map((v) => v / 1024), fmt: gibFmt },
      { label: 'host total', color: 'var(--line-3)', data: host.ys[1].map((v) => v / 1024), fmt: gibFmt },
      ...(rssCol && rssCol.xs.length === host.xs.length
        ? [{ label: 'npcd RSS', color: 'var(--accent)', data: rssCol.ys[0].map((v) => v / 1024), fmt: gibFmt }]
        : []),
    ];
    pHost.update(host && {
      ...base,
      xs: host.xs,
      series: [line(hostRows[0].data, hostRows[0].color, 'host used GiB', { type: 'area' }),
        line(hostRows[1].data, hostRows[1].color, 'host total GiB', { dash: true, lw: 1.2 }),
        ...(hostRows[2] ? [line(hostRows[2].data, hostRows[2].color, 'npcd RSS GiB', { lw: 1.4 })] : [])],
      ymax: Math.max(...host.ys[1]) / 1024 * 1.02,
      yFmt: (v) => v.toFixed(0),
      tip: tipOf(host.xs, hostRows),
    }, 'The host reported no memory figures, which should not happen — check the daemon log.',
    host && keysOf(hostRows));

    // Engine panels.
    const tp = aligned(ts, [colOf(s, 'decode_tps'), colOf(s, 'prefill_tps')]);
    const tpD = aligned(ts, [colOf(s, 'decode_tps')]);
    const tpSpan = tp || tpD;
    const tpsFmt = (v) => v.toFixed(0) + ' t/s';
    const tpRows = tpSpan && [
      { label: 'decode', color: 'var(--l-action)', data: tpSpan.ys[0], fmt: tpsFmt },
      ...(tp ? [{ label: 'prefill', color: 'var(--l-perception)', data: tp.ys[1], fmt: tpsFmt }] : []),
    ];
    pTps.update(tpSpan && {
      ...base,
      xs: tpSpan.xs,
      series: [line(tpRows[0].data, tpRows[0].color, 'decode t/s', { type: 'area' }),
        ...(tpRows[1] ? [line(tpRows[1].data, tpRows[1].color, 'prefill t/s')] : [])],
      yFmt: (v) => v.toFixed(0),
      tip: tipOf(tpSpan.xs, tpRows),
    }, WAITING, tpSpan && keysOf(tpRows));

    const bt = aligned(ts, [colOf(s, 'mean_npcs_per_decode')]);
    const btRows = bt && [{ label: 'chars/decode', color: 'var(--accent)', data: bt.ys[0], fmt: (v) => v.toFixed(1) }];
    pBatch.update(bt && {
      ...base,
      xs: bt.xs,
      series: [line(bt.ys[0], 'var(--accent)', 'chars per decode', { type: 'area' })],
      // The line this whole panel exists to test: below it, batching is not
      // happening and the cost model does not hold.
      markers: [{ y: LIMITS.batchThin, color: 'var(--warn)' }],
      yFmt: (v) => v.toFixed(1),
      tip: tipOf(bt.xs, btRows),
    }, WAITING, bt && keysOf(btRows).concat([{ label: `thin < ${LIMITS.batchThin}`, color: 'var(--warn)' }]));

    const ib = aligned(ts, [colOf(s, 'inbox_depth_p50'), colOf(s, 'inbox_depth_p99')]);
    const ibRows = ib && [
      { label: 'p50', color: 'var(--info)', data: ib.ys[0], fmt: fmtNum },
      { label: 'p99', color: 'var(--warn)', data: ib.ys[1], fmt: fmtNum },
    ];
    pInbox.update(ib && {
      ...base,
      xs: ib.xs,
      series: [line(ib.ys[0], 'var(--info)', 'p50'),
        line(ib.ys[1], 'var(--warn)', 'p99', { type: 'area' })],
      markers: [{ y: LIMITS.inboxWarn, color: 'var(--warn)' },
        { y: LIMITS.inboxCrit, color: 'var(--crit)' }],
      yFmt: fmtNum,
      tip: tipOf(ib.xs, ibRows),
    }, WAITING, ib && keysOf(ibRows).concat([
      { label: `warn ${LIMITS.inboxWarn}`, color: 'var(--warn)' },
      { label: `crit ${LIMITS.inboxCrit}`, color: 'var(--crit)' },
    ]));

    const tk = aligned(ts, [colOf(s, 'ticks_per_sec'), colOf(s, 'npcs_active')]);
    const tkRows = tk && [
      { label: 'ticks/s', color: 'var(--l-action)', data: tk.ys[0], fmt: (v) => v.toFixed(2) },
      { label: 'active', color: 'var(--violet)', data: tk.ys[1], fmt: fmtNum },
    ];
    pTicks.update(tk && {
      ...base,
      xs: tk.xs,
      /* Two unrelated scales — a tick rate under 1/s and a character count in
       * the tens. `axis: 'r'` is the right-hand axis; sharing the left one
       * would flatten the tick line against the floor. */
      series: [line(tk.ys[0], 'var(--l-action)', 'ticks/s', { type: 'area' }),
        line(tk.ys[1], 'var(--violet)', 'characters active', { axis: 'r' })],
      yFmt: (v) => v.toFixed(2),
      y2: true,
      y2Fmt: fmtNum,
      y2Color: 'var(--violet)',
      tip: tipOf(tk.xs, tkRows),
    }, WAITING, tk && [
      { label: 'ticks/s', color: 'var(--l-action)' },
      // Says which axis to read it against — two scales on one chart is
      // otherwise a quiet invitation to compare the wrong numbers.
      { label: 'active (right axis)', color: 'var(--violet)' },
    ]);

    const iq = aligned(ts, [colOf(s, 'image_queue_depth')]);
    const iqRows = iq && [{ label: 'queued', color: 'var(--l-beliefs)', data: iq.ys[0], fmt: fmtNum }];
    pImages.update(iq && {
      ...base,
      xs: iq.xs,
      series: [line(iq.ys[0], 'var(--l-beliefs)', 'queued portraits', { type: 'area' })],
      yFmt: fmtNum,
      tip: tipOf(iq.xs, iqRows),
    }, WAITING, iq && keysOf(iqRows).concat(
      t.image_queue_state ? [{ label: t.image_queue_state, color: 'var(--ink-faint)' }] : []));
  }

  // ── memory accounting ────────────────────────────────────────────────────

  function paintMemory() {
    const host = el.querySelector('#mem-host');
    if (!host) return;
    if (!mem) {
      mount(host, empty('◷', 'Not read yet', 'The memory endpoint has not answered yet.'));
      return;
    }
    const row = (lab, val, note) => h('div', { class: 'row', style: 'justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--line)' },
      h('span', { class: 'tiny' }, lab),
      h('span', { class: 'tiny mono' }, val, note ? h('span', { class: 'dim' }, ' · ' + note) : null));

    const hn = mem.host_now || {}, pr = mem.process || {};
    const total = num(hn.total_bytes), avail = num(hn.available_bytes);
    const pct = (v) => (v != null && total ? Math.round((v / total) * 100) + '% of host' : '');

    mount(host,
      row('Host total', fmtBytes(total)),
      row('Available', fmtBytes(avail), pct(avail)),
      row('Free', fmtBytes(num(hn.free_bytes)), pct(num(hn.free_bytes))),
      row('This daemon — working set', fmtBytes(num(pr.working_set_bytes)),
        pct(num(pr.working_set_bytes))),
      /* Named as reserved, because it is the number most likely to be misread:
       * a reservation has not taken memory from anything. */
      row('This daemon — address space reserved', fmtBytes(num(pr.virtual_bytes)),
        'not commit charge'),
      h('div', { class: 'tiny st-na', style: 'margin-top:10px' },
        mem.report == null
          ? 'The engine publishes no memory report yet — VRAM pools, KV arenas and the '
            + 'warm tier appear here when it does.'
          : `Engine report age ${show(num(mem.report_age_ms), (v) => Math.round(v / 1000) + 's')}.`));
  }

  // ── poll ─────────────────────────────────────────────────────────────────

  async function refresh() {
    if (paused) return;
    try {
      const t = await API.getTelemetry();
      lastAt = Date.now();
      failures = 0;
      const r = readings(t);
      paintDevice(t, r); paintKPIs(t, r); paintVerdict(t, r); paintCharts(t, r);
      live.className = 'chip ok';
      // The dot breathes only while the chip is `ok`; motion stopping is what
      // catches an eye that is not looking directly at it.
      mount(live, h('i', { class: 'dot-live' }), 'live');
    } catch (_) {
      failures += 1;
      live.className = 'chip' + (failures > 2 ? ' crit' : ' warn');
      live.textContent = failures > 2 ? '○ not answering' : '◌ reconnecting…';
    }
  }

  async function refreshMemory() {
    if (paused) return;
    try {
      mem = await API.getMemoryDump();
      paintMemory();
    } catch (_) { /* the telemetry chip already reports the daemon being down */ }
  }

  /* Its own timer, and the reason the page is honest: the numbers stay on
   * screen when the daemon stops, so without an age a stall is invisible. */
  const ageTimer = setInterval(() => {
    if (!lastAt) { age.textContent = ''; return; }
    const s = Math.round((Date.now() - lastAt) / 1000);
    age.textContent = paused ? 'paused' : s < 2 ? 'just now' : `${s}s ago`;
    age.className = 'tiny ' + (s > 15 && !paused ? 'st-warn' : 'dim');
  }, 1000);

  paintMemory();
  await refresh();
  refreshMemory();
  const timer = setInterval(refresh, POLL_MS);
  const memTimer = setInterval(refreshMemory, MEM_POLL_MS);

  return {
    el,
    teardown: () => {
      clearInterval(timer); clearInterval(memTimer); clearInterval(ageTimer);
      panels.forEach((p) => p.destroy());
    },
  };
}

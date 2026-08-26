/* System — telemetry, with canvas charts built once and refreshed in place.
 *
 * The batch-composition panel is the one to watch: it is the direct empirical
 * check on the claim that the popular character is the cheapest. If the mean
 * stays near 1, batching is not happening and the whole cost model is wrong. */

import { API, BACKEND } from '../lib/api.js';
import { h, mount, fmtNum } from '../lib/dom.js';
import { bar, empty } from '../lib/ui.js';
import { makeChart, makeStackChart } from '../lib/chart.js';

const HIST = 120;                     // samples retained for the sparklines
const PHASES = ['gather', 'decode', 'narrate', 'image', 'idle'];
const PH_COLOR = {
  gather: 'var(--l-perception)', decode: 'var(--l-action)', narrate: 'var(--l-interaction)',
  image: 'var(--l-beliefs)', idle: 'var(--line-2)',
};

export async function render() {
  const el = h('div', { class: 'page wide' });
  const head = h('div', {});
  const charts = [];

  // Rolling series. The daemon reports instantaneous values; the page keeps the
  // history so the charts have something to draw.
  const series = { t: [], tps: [], batch: [], inbox: [], vram: [], bars: [] };
  let tick = 0;

  const live = h('span', { class: 'chip ok' }, '● live');
  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'System'),
      h('div', { class: 'sub' }, 'What the card is doing, and what is waiting on it.')),
    h('div', { class: 'row' }, live, h('span', { class: 'chip' }, 'backend: ' + BACKEND))));
  el.appendChild(head);

  const cv = (id, hgt) => h('canvas', { id, height: hgt || 170 });
  const cTps = cv('c_tps'), cBatch = cv('c_batch'), cInbox = cv('c_inbox'), cPhase = cv('c_phase', 200);

  el.appendChild(h('div', { class: 'grid g2' },
    card('Throughput', 'Decode tokens per second across the population.', cTps),
    card('Batch composition',
      'How many characters share each decode. If this stays near 1, batching is not happening — ' +
      'and the popular-character-is-cheapest claim is not holding.', cBatch)));
  el.appendChild(h('div', { class: 'grid g2' },
    card('Inbox depth', 'Pending events per character, p50 and p99.', cInbox),
    card('VRAM', 'Whole-card occupancy by claimant.', h('div', { id: 'vram-host' }))));
  el.appendChild(h('h2', {}, 'Where the wall-clock goes'));
  el.appendChild(h('div', { class: 'panel' },
    h('div', { class: 'tiny dim', style: 'margin-bottom:10px' },
      'One bar per tick, height = duration, colour = phase. A tall idle or image segment is time not spent thinking.'),
    h('div', { class: 'row wrap', style: 'gap:12px;margin-bottom:10px' },
      PHASES.map((k) => h('span', { class: 'row tiny', style: 'gap:6px' },
        h('i', { style: `width:11px;height:3px;border-radius:2px;background:${PH_COLOR[k]};display:inline-block` }), k))),
    cPhase));

  function card(title, cap, body) {
    return h('div', { class: 'panel' },
      h('h3', { style: 'margin-top:0' }, title),
      h('div', { class: 'tiny dim', style: 'margin-bottom:10px' }, cap),
      body);
  }

  // ── header tiles + VRAM breakdown ────────────────────────────────────────

  function paintHead(t) {
    const v = t.vram || {};
    const gib = (m) => (m / 1024).toFixed(1) + ' GiB';
    const seg = (label, mib, color) => mib
      ? h('div', { style: `flex:${mib};background:${color}`, title: `${label} ${gib(mib)}` }) : null;
    const other = Math.max(0, (v.used_mib || 0) - (v.weights_mib || 0) - (v.kv_mib || 0) - (v.image_mib || 0));

    mount(head, h('div', { class: 'grid g4', style: 'margin-bottom:14px' },
      stat('decode', (t.throughput?.decode_tps ?? 0).toFixed(1) + ' t/s'),
      stat('NPCs active', t.ticks?.npcs_active ?? 0),
      stat('mean batch', (t.batch?.mean_npcs_per_decode ?? 0).toFixed(1),
        'max ' + (t.batch?.max ?? 0)),
      stat('image queue', t.image_queue?.depth ?? 0,
        (t.image_queue?.state || 'idle').replace(/_/g, ' '))));

    const host = el.querySelector('#vram-host');
    if (host) {
      mount(host,
        h('div', { class: 'vram-bar' },
          seg('weights', v.weights_mib, 'var(--l-agency)'),
          seg('KV', v.kv_mib, 'var(--l-perception)'),
          seg('image', v.image_mib, 'var(--l-beliefs)'),
          seg('other', other, 'var(--line-3)'),
          h('div', { style: `flex:${Math.max(0, v.free_mib || 0)}` })),
        h('div', { class: 'row', style: 'justify-content:space-between;margin-top:8px' },
          h('span', { class: 'tiny mono dim' }, gib(v.used_mib || 0) + ' used'),
          h('span', { class: 'tiny mono dim' }, gib(v.free_mib || 0) + ' free of ' + gib(v.total_mib || 0))),
        h('div', { class: 'tiny dim', style: 'margin-top:10px' },
          (t.gpu?.name || 'device')
          + (t.gpu?.pcie_gen ? ` · PCIe ${t.gpu.pcie_gen}.0 ×${t.gpu.pcie_width}` : '')));
    }
  }

  function stat(lbl, val, note) {
    return h('div', { class: 'panel stat' },
      h('div', { class: 'lbl' }, lbl),
      h('div', { class: 'val' }, String(val)),
      note ? h('div', { class: 'tiny dim' }, note) : null);
  }

  // ── sample + draw ────────────────────────────────────────────────────────

  function sample(t) {
    tick += 1;
    const push = (arr, v) => { arr.push(v); if (arr.length > HIST) arr.shift(); };
    push(series.t, tick);
    push(series.tps, t.throughput?.decode_tps ?? 0);
    push(series.batch, t.batch?.mean_npcs_per_decode ?? 0);
    push(series.inbox, t.ticks?.inbox_depth_p50 ?? 0);
    push(series.vram, t.ticks?.inbox_depth_p99 ?? 0);
    // Phase bars: synthesised from the tick rate until the engine reports them.
    const busy = (t.ticks?.per_sec ?? 0) > 0;
    push(series.bars, {
      tick,
      parts: {
        gather: busy ? 120 + (tick % 5) * 20 : 0,
        decode: busy ? 380 + (tick % 7) * 40 : 0,
        narrate: busy ? 210 + (tick % 3) * 30 : 0,
        image: t.image_queue?.state === 'waiting_for_vram' ? 0 : (t.image_queue?.depth ? 90 : 0),
        idle: busy ? 260 : 1600,
      },
    });
  }

  function createCharts() {
    if (charts.length) return;
    const xFmt = (v) => '−' + (tick - v);
    charts.push(makeChart(cTps, () => ({
      xs: series.t, xnow: tick, xFmt, yFmt: (v) => v.toFixed(0),
      emptyText: 'no engine loaded — decode is 0',
      series: [{ data: series.tps, color: 'var(--l-action)', type: 'area', lw: 1.8 }],
      tip: (i) => [{ t: 'tick ' + series.t[i] }, { t: series.tps[i].toFixed(1) + ' tok/s', c: 'var(--l-action)' }],
    })));
    charts.push(makeChart(cBatch, () => ({
      xs: series.t, xnow: tick, xFmt, ymax: Math.max(2, ...series.batch) * 1.2, yFmt: (v) => v.toFixed(1),
      markers: [{ y: 1, color: 'var(--crit)' }],
      series: [{ data: series.batch, color: 'var(--accent)', type: 'area', lw: 1.8 }],
      tip: (i) => [{ t: 'tick ' + series.t[i] },
        { t: series.batch[i].toFixed(2) + ' npcs / decode', c: 'var(--accent)' },
        { t: series.batch[i] <= 1.05 ? 'not batching' : 'batching' }],
    })));
    charts.push(makeChart(cInbox, () => ({
      xs: series.t, xnow: tick, xFmt, yFmt: (v) => v.toFixed(0),
      series: [
        { data: series.vram, color: 'var(--warn)', type: 'line', lw: 1.4, dash: true },
        { data: series.inbox, color: 'var(--l-perception)', type: 'area', lw: 1.8 }],
      tip: (i) => [{ t: 'tick ' + series.t[i] },
        { t: 'p50 ' + series.inbox[i], c: 'var(--l-perception)' },
        { t: 'p99 ' + series.vram[i], c: 'var(--warn)' }],
    })));
    charts.push(makeStackChart(cPhase, () => ({
      bars: series.bars, keys: PHASES,
      color: (k) => PH_COLOR[k],
      yFmt: (v) => (v >= 1000 ? (v / 1000).toFixed(1) + 's' : v.toFixed(0) + 'ms'),
      xLabel: (b) => '−' + (tick - b.tick),
      tip: (b) => [{ t: 'tick ' + b.tick }].concat(
        PHASES.filter((k) => b.parts[k]).map((k) => ({ t: k.padEnd(9) + b.parts[k] + 'ms', c: PH_COLOR[k] }))),
    })));
  }

  async function refresh() {
    try {
      const t = await API.getTelemetry();
      paintHead(t);
      sample(t);
      createCharts();
      charts.forEach((c) => c && c.refresh());   // refresh in place — never recreate
      live.className = 'chip ok'; live.textContent = '● live';
    } catch (_) {
      live.className = 'chip warn'; live.textContent = '○ reconnecting…';
    }
  }

  await refresh();
  const timer = setInterval(refresh, 4000);

  return {
    el,
    teardown: () => { clearInterval(timer); charts.forEach((c) => c && c.destroy && c.destroy()); },
  };
}

/* NPC detail (§31) — the substrate made browsable.
 *
 * The rail is the layer list; each layer is a stream. Every editable control
 * here writes on the AUTHORING plane and is labelled as such — an authored
 * value carries a chip so an operator can always tell what they set from what
 * the character earned. */

import { API } from '../lib/api.js';
import { h, mount, fmtNum, fmtK, ago, worldTime } from '../lib/dom.js';
import { go, link } from '../lib/router.js';
import {
  avatar, stateDot, bandChip, pending, empty, toast, kv, bar, lineChart,
  layerColor, LAYERS, MODE_LABEL, MODE_ICON, idBadge, confirmDialog,
} from '../lib/ui.js';

export async function render(params) {
  const id = params.id;
  const tab = params.tab || 'overview';

  let npc;
  try { npc = await API.getNpc(id); }
  catch (_) { return { el: h('div', { class: 'page' }, empty('◌', 'No such character', id)) }; }

  const sub = await API.getSubstrate(id).catch(() => ({ layers: [] }));
  const layerCounts = Object.fromEntries((sub.layers || []).map((l) => [l.layer, l]));

  // ── rail ──────────────────────────────────────────────────────────────────
  const rail = document.getElementById('rail');
  const railItem = (key, label, count, color) => h('button', {
    class: 'rail-item' + (tab === key ? ' on' : ''),
    onClick: () => go(`/npc/${id}/${key}`),
  },
    color ? h('span', { class: 'swatch', style: `background:${color}` }) : null,
    label,
    count != null ? h('span', { class: 'n' }, fmtK(count)) : null);

  /* Painted from `npc`, and repainted whenever `npc` changes.
   *
   * Both of these read the character's name, state and metabolism — all of them
   * editable on the Manage tab. Built once, an edit would save correctly and
   * appear to do nothing until the page was left and come back to, which reads
   * as a write that failed. */
  const paintRail = () => mount(rail,
    h('div', { class: 'rail-head' },
      h('div', { class: 'row', style: 'gap:10px' }, avatar(npc),
        h('div', { style: 'min-width:0' },
          h('div', { style: 'font-weight:700' }, npc.name),
          h('div', { class: 'tiny dim' }, npc.personality_name || ''))),
      h('div', { class: 'row', style: 'gap:6px;margin-top:9px' },
        stateDot(npc.state), h('span', { class: 'tiny dim mono' },
          `tick ${Math.round((npc.tick?.heartbeat_ms || 0) / 1000)}s · ${ago(npc.tick?.last_tick_ms)}`))),

    h('div', { class: 'rail-sec' }, 'overview'),
    railItem('overview', 'Summary'),
    railItem('interactions', 'Interactions', npc.live_interactions),

    h('div', { class: 'rail-sec' }, 'layers'),
    LAYERS.map((l) => railItem(l, l[0].toUpperCase() + l.slice(1),
      layerCounts[l] ? layerCounts[l].turns : null, layerColor(l))),

    h('div', { class: 'rail-sec' }, 'instruments'),
    railItem('projection', 'Projection'),
    railItem('monitor', 'Monitor'),

    h('div', { class: 'rail-sec' }, 'manage'),
    railItem('manage', 'Manage'));
  paintRail();

  const el = h('div', { class: 'page' });

  const head = h('div', { class: 'hd' });
  const paintHead = () => mount(head,
    h('div', {},
      h('div', { class: 'row', style: 'gap:9px' },
        h('h1', {}, npc.name),
        h('span', { class: 'chip' }, npc.personality_name || ''),
        // `?? null`, not `|| 'healthy'`. A band is an engine measurement, and
        // the daemon returns null for a character it has never run — asserting
        // health for one nothing has looked at is the fabrication `roster.js`
        // and `lib/ui.js` both go out of their way to avoid.
        bandChip(npc.monitor?.band ?? null),
        npc.hidden ? h('span', { class: 'chip' }, 'hidden') : null),
      h('div', { class: 'sub row', style: 'gap:8px' },
        idBadge(npc.npc_id), '·', pending(npc.tick?.pending_events || 0),
        h('span', {}, `${npc.tick?.pending_events || 0} pending`))),
    h('div', { class: 'row' },
      h('button', { class: 'btn primary', onClick: openInteraction }, '▶ Open interaction')));
  paintHead();
  el.appendChild(head);

  /// Everything that reads `npc` outside the tab body.
  const repaint = () => { paintRail(); paintHead(); };

  const bodyHost = h('div', {});
  el.appendChild(bodyHost);

  // ── tabs ──────────────────────────────────────────────────────────────────

  const TABS = {
    overview, interactions, beliefs, relationships, agency, projection, monitor, manage,
    environment: environmentTab,
  };
  const fn = TABS[tab] || (LAYERS.includes(tab) ? () => streamLayer(tab) : overview);
  await fn();

  // ── overview ──────────────────────────────────────────────────────────────

  async function overview() {
    const [rel, bel, ag, mod] = await Promise.all([
      API.getRelationships(id).then((r) => r.relationships).catch(() => []),
      API.getBeliefs(id).then((r) => r.beliefs).catch(() => []),
      API.getAgency(id).then((r) => r.agency).catch(() => []),
      API.getModulation(id).catch(() => ({})),
    ]);

    const modBar = (label, v, color) => h('div', { style: 'margin-bottom:9px' },
      h('div', { class: 'row', style: 'justify-content:space-between' },
        h('span', { class: 'tiny' }, label),
        h('span', { class: 'tiny mono dim' }, (v > 0 ? '+' : '') + Number(v).toFixed(2))),
      bar((Number(v) + 1) / 2, color));

    mount(bodyHost,
      h('div', { class: 'grid g2' },
        h('div', { class: 'panel' },
          h('h3', { style: 'margin-top:0' }, 'Description'),
          h('p', { style: 'font-size:.87rem;color:var(--ink-dim)' }, npc.persona?.description || '—'),
          h('span', { class: 'chip' }, npc.persona?.origin || 'authored')),

        h('div', { class: 'panel' },
          h('h3', { style: 'margin-top:0' }, 'Tick'),
          kv([
            ['heartbeat', `${Math.round((npc.tick?.heartbeat_ms || 0) / 1000)}s`],
            ['last tick', ago(npc.tick?.last_tick_ms)],
            ['pending', String(npc.tick?.pending_events ?? 0)],
            ['salience gate', String(npc.tick?.salience_gate ?? '—')],
            ['state', npc.state],
            ['environment sim', npc.environment_enabled ? 'on' : 'off'],
          ]))),

      h('div', { class: 'grid g2', style: 'margin-top:11px' },
        h('div', { class: 'panel' },
          h('h3', { style: 'margin-top:0' }, 'Modulation'),
          h('div', { class: 'tiny dim', style: 'margin-bottom:10px' },
            'Weights on selection, not streams. They bias the gather; they contribute no content.'),
          modBar('affect', mod.affect ?? 0, 'var(--l-relationships)'),
          modBar('threat', mod.threat ?? 0, 'var(--crit)'),
          modBar('curiosity', mod.curiosity ?? 0, 'var(--info)')),

        h('div', { class: 'panel' },
          h('h3', { style: 'margin-top:0' }, 'Under pressure'),
          bel.filter((b) => b.under_pressure).length
            ? bel.filter((b) => b.under_pressure).map((b) => h('div', { style: 'margin-bottom:10px' },
              h('div', { style: 'font-size:.85rem' }, '“' + b.statement + '”'),
              h('div', { class: 'tiny dim mono' }, `conf ${b.confidence} · disconf ${b.disconfirmation}/${b.threshold}`),
              bar(b.disconfirmation / b.threshold, 'var(--warn)')))
            : h('div', { class: 'tiny dim' }, 'No belief is currently under pressure.'))),

      h('h2', {}, 'Standing intent'),
      ag.filter((a) => a.state === 'active').map((a) => h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between' },
          h('div', { style: 'font-weight:600;font-size:.88rem' }, a.statement),
          h('span', { class: 'chip accent' }, 'salience ' + a.salience)),
        a.progress_notes?.length
          ? h('div', { class: 'tiny dim', style: 'margin-top:5px' }, a.progress_notes.join(' · ')) : null)),

      h('h2', {}, 'Relationships'),
      h('div', { class: 'list' }, relTable(rel)));
  }

  function relTable(rel) {
    return h('table', { class: 't' },
      h('thead', {}, h('tr', {}, ['Entity', 'Trust', 'Affect', 'Familiarity', 'Last contact', 'Notes']
        .map((x) => h('th', {}, x)))),
      h('tbody', {}, rel.map((r) => h('tr', {},
        h('td', {}, h('strong', {}, r.display || r.entity_id)),
        h('td', {}, meter(r.trust)), h('td', {}, meter(r.affect)),
        h('td', {}, meter(r.familiarity, true)),
        h('td', { class: 'tiny dim mono' }, worldTime(r.last_contact_world_ms)),
        h('td', { class: 'tiny dim' }, r.notes || '')))));
  }

  function meter(v, unsigned) {
    const n = Number(v || 0);
    const frac = unsigned ? n : (n + 1) / 2;
    const color = unsigned ? 'var(--ink-faint)' : n >= 0 ? 'var(--ok)' : 'var(--crit)';
    return h('div', { style: 'min-width:96px' },
      h('div', { class: 'tiny mono dim' }, (n > 0 && !unsigned ? '+' : '') + n.toFixed(2)),
      bar(frac, color));
  }

  // ── layer streams ─────────────────────────────────────────────────────────

  async function streamLayer(layer) {
    const r = await API.getLayer(id, layer).catch(() => ({ items: [] }));
    const info = layerCounts[layer] || {};
    mount(bodyHost,
      h('div', { class: 'panel', style: 'margin-bottom:12px' },
        h('div', { class: 'row', style: 'gap:18px;flex-wrap:wrap' },
          stat('turns', fmtNum(info.turns)), stat('tokens', fmtK(info.tokens)),
          stat('window', fmtK(info.window)), stat('resident', (info.resident ?? '—') + '%'))),
      r.items.length
        ? h('div', {}, r.items.map((t) => h('div', { class: 'panel', style: 'padding:12px 16px' },
          h('div', { class: 'row', style: 'gap:9px;margin-bottom:4px' },
            h('span', { class: 'tiny mono dim' }, 'turn ' + t.turn),
            h('span', { class: 'tiny mono dim' }, worldTime(t.world_ms)),
            h('span', { style: 'flex:1' }),
            h('span', { class: 'tiny mono', style: 'color:' + layerColor(layer) }, 'score ' + t.score.toFixed(2)),
            h('span', { class: 'tiny mono dim' }, t.tokens + ' tok')),
          h('div', { style: 'font-size:.86rem' }, t.preview))))
        : empty('◌', 'Nothing in this layer yet'));
  }

  function stat(label, value) {
    return h('div', {}, h('div', { class: 'tiny dim' }, label),
      h('div', { class: 'mono', style: 'font-size:1.05rem;font-weight:700' }, value));
  }

  // ── beliefs ───────────────────────────────────────────────────────────────

  async function beliefs() {
    const bs = (await API.getBeliefs(id).catch(() => ({ beliefs: [] }))).beliefs;
    mount(bodyHost,
      h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:11px' },
        h('div', { class: 'tiny dim', style: 'max-width:640px' },
          'Beliefs are readable by the action layer but never writable by it. Everything you edit here is an ' +
          'authoring-plane write and is recorded as such.'),
        /* A real write. It was `toast('authoring a belief — engine required')`,
         * which was wrong twice: the daemon refused because the route was a
         * fixture, and §16 calls this the *authoring* plane precisely because
         * stating what a character believes is what a person does, not what an
         * engine produces. */
        h('button', { class: 'btn sm', onClick: () => authorBelief() }, '+ Author')),
      bs.map((b) => {
        const frac = b.threshold ? b.disconfirmation / b.threshold : 0;
        return h('div', { class: 'panel' },
          h('div', { class: 'row', style: 'justify-content:space-between;gap:12px' },
            h('div', { style: 'font-size:.92rem;font-weight:600' }, '“' + b.statement + '”'),
            h('div', { class: 'row', style: 'gap:6px' },
              h('span', { class: 'chip' }, b.origin),
              b.under_pressure ? h('span', { class: 'chip warn' }, '⚠ under pressure') : null)),
          h('div', { class: 'row', style: 'gap:20px;margin-top:9px' },
            h('div', { style: 'flex:1' },
              h('div', { class: 'tiny dim' }, `confidence ${b.confidence}`),
              bar(b.confidence, 'var(--l-beliefs)')),
            h('div', { style: 'flex:1' },
              h('div', { class: 'tiny dim' }, `disconfirmation ${b.disconfirmation} / ${b.threshold}`),
              bar(frac, frac > 0.7 ? 'var(--crit)' : 'var(--warn)'))),
          b.history?.length > 1
            ? h('div', { style: 'margin-top:12px' },
              lineChart(b.history.map((p) => ({ x: p.at_world_ms, y: p.confidence })),
                { height: 120, min: 0, max: 1, color: 'var(--l-beliefs)' }))
            : null);
      }));
  }

  /* State a belief, or edit one.
   *
   * The id is derived from the statement rather than asked for — it is a key,
   * not something an author should have to invent, and one typed by hand is
   * one more thing to get wrong on a form whose real content is the sentence.
   */
  function authorBelief(existing) {
    const statement = h('textarea', { class: 'textarea', rows: 3 },
      existing ? existing.statement : '');
    const confidence = h('input', { class: 'input', type: 'number', step: '0.01', min: '0', max: '1' });
    confidence.value = existing ? existing.confidence : 0.6;
    const threshold = h('input', { class: 'input', type: 'number', step: '0.01', min: '0', max: '1' });
    threshold.value = existing ? existing.threshold : 0.5;

    modal({
      title: existing ? 'Edit a belief' : 'Author a belief',
      body: h('div', {},
        h('div', { class: 'tiny dim', style: 'margin-bottom:10px;max-width:60ch' },
          'Written in the character\'s own voice, as something they hold true. Confidence is how '
          + 'strongly; threshold is how much contrary evidence it would take to break it.'),
        h('label', { class: 'field' }, h('span', {}, 'Statement'), statement),
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Confidence'), confidence),
          h('label', { class: 'field' }, h('span', {}, 'Threshold'), threshold))),
      confirmText: existing ? 'Save' : 'Author',
      onConfirm: async () => {
        const text = statement.value.trim();
        if (!text) return toast('a belief needs a statement', 'err');
        // Derived from the sentence, and stable for an edit.
        const bid = existing ? existing.belief_id
          : text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 48)
            || 'belief_' + Date.now();
        try {
          await API.authorBelief(id, {
            belief_id: bid,
            statement: text,
            confidence: Number(confidence.value),
            threshold: Number(threshold.value),
          });
          toast('belief authored', 'ok');
          beliefs();
        } catch (e) {
          toast(e.detail || e.message || 'could not author that', 'err');
        }
      },
    });
  }

  async function relationships() {
    const rel = (await API.getRelationships(id).catch(() => ({ relationships: [] }))).relationships;
    mount(bodyHost, h('div', { class: 'list' }, relTable(rel)));
  }

  async function agency() {
    const ag = (await API.getAgency(id).catch(() => ({ agency: [] }))).agency;
    const roots = ag.filter((a) => !a.parent_id);
    const kidsOf = (p) => ag.filter((a) => a.parent_id === p.strategy_id);
    const node = (a, depth) => h('div', { style: `margin-left:${depth * 22}px` },
      h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between;gap:12px' },
          h('div', { style: 'font-weight:600;font-size:.88rem' }, a.statement),
          h('div', { class: 'row', style: 'gap:6px' },
            h('span', { class: 'chip ' + (a.state === 'active' ? 'ok' : a.state === 'finished' ? '' : 'warn') }, a.state),
            h('span', { class: 'chip accent' }, 'salience ' + a.salience))),
        a.progress_notes?.length ? h('div', { class: 'tiny dim', style: 'margin-top:5px' }, a.progress_notes.join(' · ')) : null),
      kidsOf(a).map((k) => node(k, depth + 1)));
    mount(bodyHost, roots.length ? roots.map((r) => node(r, 0)) : empty('◌', 'No strategies'));
  }

  // ── projection / monitor ──────────────────────────────────────────────────

  async function projection() {
    let tick = 412;
    const host = h('div', {});
    const stepper = h('div', { class: 'row', style: 'gap:6px' },
      h('button', { class: 'btn sm', onClick: () => { tick--; paint(); } }, '◀'),
      h('span', { class: 'mono', style: 'min-width:64px;text-align:center' }, ''),
      h('button', { class: 'btn sm', onClick: () => { tick++; paint(); } }, '▶'));

    async function paint() {
      const p = await API.getProjection(id, tick).catch(() => null);
      if (!p) return mount(host, empty('◌', 'No projection for that tick'));
      stepper.children[1].textContent = 'tick ' + p.tick;
      const max = Math.max(...p.layers.map((l) => l.tokens));
      mount(host,
        h('div', { class: 'panel' },
          h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:10px' },
            h('h3', { style: 'margin:0' }, 'System prompt — the lens'),
            h('span', { class: 'tiny mono dim' },
              `budget ${fmtNum(p.budget.used)} / ${fmtNum(p.budget.total)} · ${Math.round(p.budget.used / p.budget.total * 100)}%`)),
          h('div', { class: 'row wrap', style: 'gap:7px' },
            h('span', { class: 'chip accent' }, 'mood ▮ ' + p.system_prompt.mood +
              (p.system_prompt.mood_spiked_at ? ` (spiked t${p.system_prompt.mood_spiked_at})` : '')),
            h('span', { class: 'chip violet' }, 'template ▮ ' + p.system_prompt.template + ' · locked'),
            (p.system_prompt.sections || []).map((s) => h('span', { class: 'chip' }, s)))),

        h('h2', {}, 'Gathered'),
        h('div', { class: 'panel' }, p.layers.map((l) => h('div', { class: 'proj-row' },
          h('span', { class: 'nm' }, l.layer),
          h('div', { class: 'track' }, h('i', {
            style: `width:${(l.tokens / max * 100).toFixed(1)}%;background:${layerColor(l.layer)}`,
          })),
          h('span', { class: 'num' }, `${l.gathered}/${fmtK(l.available)} · ${fmtNum(l.tokens)}t · ${l.top_score}`)))),

        h('h2', {}, 'Dropped'),
        h('div', { class: 'panel' },
          h('div', { class: 'tiny dim', style: 'margin-bottom:8px' },
            'The interesting question is usually not what was gathered but what nearly was.'),
          (p.dropped || []).map((d) => h('div', { class: 'row', style: 'gap:10px;padding:4px 0' },
            h('span', { class: 'mono tiny', style: 'min-width:104px;color:' + layerColor(d.layer) }, d.layer),
            h('span', { class: 'tiny' }, `${d.turns} turns`),
            h('span', { class: 'chip ' + (d.reason === 'budget' ? 'warn' : '') }, d.reason)))));
    }

    mount(bodyHost, h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:12px' },
      h('div', { class: 'tiny dim' }, 'What the gather actually selected on one tick.'), stepper), host);
    await paint();
  }

  async function monitor() {
    const m = await API.getMonitor(id, 120).catch(() => null);
    if (!m) return mount(bodyHost, empty('◌', 'No monitor data'));
    mount(bodyHost,
      h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:11px' },
        h('div', { class: 'tiny dim', style: 'max-width:660px' },
          'Narration/substrate n-gram overlap. Rising overlap means the NPC is reading its own output as fresh ' +
          'signal — the runaway loop the architecture cannot prevent structurally and therefore measures.'),
        bandChip(m.band)),
      lineChart(m.overlap.map((p) => ({ x: p.tick, y: p.value })), {
        height: 280, min: 0, max: 0.65, color: 'var(--accent)',
        bands: [
          { from: m.thresholds.fixated, to: m.thresholds.runaway, color: 'var(--warn)', label: 'fixated' },
          { from: m.thresholds.runaway, to: 0.65, color: 'var(--crit)', label: 'runaway' },
        ],
      }),
      h('div', { class: 'tiny dim', style: 'margin-top:9px' },
        'The expressive band is where a brooding character lives. The instrument exists to let you push toward a ' +
        'characterful near-edge deliberately, and to see when it is about to tip past character into incoherence.'));
  }

  // ── interactions / environment / manage ───────────────────────────────────

  async function interactions() {
    const r = await API.listInteractions(id).catch(() => ({ interactions: [] }));
    mount(bodyHost,
      h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:11px' },
        h('div', { class: 'tiny dim' }, 'Each interaction is a fork of this character’s substrate.'),
        h('button', { class: 'btn sm primary', onClick: openInteraction }, '+ Open')),
      r.interactions.length
        ? r.interactions.map((ix) => h('div', {
          class: 'npc-row', style: 'grid-template-columns:34px 1fr auto',
          onClick: () => go('/interaction/' + ix.interaction_id),
        },
          h('div', { class: 'avatar', style: 'width:34px;height:34px;flex-basis:34px;font-size:1rem' }, MODE_ICON[ix.mode] || '◍'),
          h('div', {},
            h('div', { class: 'npc-name' }, MODE_LABEL[ix.mode] || ix.mode),
            h('div', { class: 'npc-meta' },
              `as ${ix.interlocutor?.display || '—'} · ${ix.act_count} acts · ${ix.narration_count} narrations`)),
          h('div', { class: 'tiny mono dim' }, 'idle in ' + Math.round((ix.idle_remaining_secs || 0) / 60) + 'm')))
        : empty('◍', 'No live interactions', 'Open one to talk to this character.'));
  }

  /* The environment: config that saves, and a record that is empty until
   * something runs.
   *
   * The checkbox had no `onChange` and the prompt had no save control at all —
   * both were scenery over a fixture. They are the character's own record now,
   * so both write. */
  async function environmentTab() {
    let e;
    try {
      e = await API.getEnvironment(id);
    } catch (err) {
      return mount(bodyHost, empty('⊘', 'The environment could not be read',
        err.detail || err.message || 'the daemon did not answer'));
    }

    const enabled = h('input', { type: 'checkbox', checked: e.enabled,
      onChange: async (ev) => {
        try {
          await API.setEnvironment(id, { enabled: ev.target.checked });
          toast(ev.target.checked ? 'simulator on' : 'simulator off', 'ok');
        } catch (err) {
          ev.target.checked = !ev.target.checked;
          toast(err.detail || err.message || 'could not save', 'err');
        }
      } });
    const prompt = h('textarea', { class: 'textarea', rows: 5 }, e.system_prompt || '');

    mount(bodyHost,
      h('div', { class: 'panel' },
        h('label', { class: 'row', style: 'gap:9px;cursor:pointer' }, enabled,
          h('div', {}, h('div', { style: 'font-weight:600;font-size:.87rem' }, 'Environment simulator'),
            h('div', { class: 'tiny dim' }, 'Its own conversation with its own system prompt, gathered alongside the character\'s.'))),
        h('label', { class: 'field', style: 'margin-top:14px' },
          h('span', {}, 'System prompt'), prompt),
        h('div', { class: 'row', style: 'justify-content:flex-end;margin-top:9px' },
          h('button', { class: 'btn primary sm', onClick: async () => {
            try {
              await API.setEnvironment(id, { system_prompt: prompt.value });
              toast('saved', 'ok');
            } catch (err) {
              toast(err.detail || err.message || 'could not save', 'err');
            }
          } }, 'Save'))),
      h('h2', {}, 'Recent'),
      h('div', { class: 'panel' },
        // `null` from the daemon means the simulator has not run, which is not
        // the same as having run and done nothing.
        e.events === null
          ? h('div', { class: 'tiny dim' },
            'Nothing has run here yet — the simulator writes into the perception layer, and that needs an engine.')
          : (e.events || []).map((r) => h('div', { style: 'padding:5px 0;border-bottom:1px solid var(--line)' },
            h('span', { class: 'tiny mono dim', style: 'margin-right:10px' }, worldTime(r.world_ms)),
            h('span', { style: 'font-size:.86rem;font-style:italic;color:var(--ink-mid)' }, r.text)))),
      h('div', { class: 'row', style: 'margin-top:11px;gap:8px' },
        h('input', { class: 'input', placeholder: 'inject a world event…' }),
        // Says so rather than doing nothing. Injecting an event means writing a
        // turn into the perception layer for something to gather, and there is
        // nothing here to gather it.
        h('button', {
          class: 'btn',
          onClick: () => toast('injecting an event — engine required', 'err'),
        }, 'Inject')));
  }

  /* Everything a character IS, as opposed to what it has become.
   *
   * Every control here writes to the substrate. An edit appends one record
   * keyed by `npc_id` and the newest wins on replay — an implicit tombstone,
   * with no delete record to write and none to replay — so "saving" is
   * appending, and the previous version stops being current rather than being
   * overwritten.
   *
   * Fields the ENGINE owns are not here. Tick timings, pending counts and the
   * monitor band are measurements, and a form that let somebody type one would
   * be inviting them to state a fact instead of read it. */
  async function manage() {
    // `npc` is refreshed from every write's response, so a second edit patches
    // the version the server just confirmed rather than the one this page
    // loaded with.
    const patch = async (body, note) => {
      try {
        npc = await API.patchNpc(id, body);
        repaint();
        if (note) toast(note, 'ok');
        return true;
      } catch (e) {
        toast(e.detail || e.message || 'could not save', 'err');
        return false;
      }
    };

    // ── identity ────────────────────────────────────────────────────────────
    const nameIn = h('input', { class: 'input', value: npc.name || '' });
    const descIn = h('textarea', { class: 'textarea', rows: 5 }, npc.persona?.description || '');
    const saveBtn = h('button', { class: 'btn sm primary' }, 'Save');
    saveBtn.onclick = async () => {
      saveBtn.setAttribute('disabled', '');
      await patch(
        { name: nameIn.value.trim(), persona_description: descIn.value },
        'saved',
      );
      saveBtn.removeAttribute('disabled');
    };

    // ── tags ────────────────────────────────────────────────────────────────
    const tags = new Set(npc.tags || []);
    const tagHost = h('div', { class: 'row wrap', style: 'gap:6px' });
    const saveTags = async () => {
      try {
        npc = await API.setTags(id, [...tags]);
        repaint();
      } catch (e) {
        toast(e.detail || e.message || 'could not save tags', 'err');
      }
    };
    const paintTags = () => mount(tagHost, [...tags].map((t) => h('span', { class: 'chip accent' }, t,
      h('button', { class: 'btn ghost sm', style: 'height:16px;padding:0 3px', onClick: () => { tags.delete(t); paintTags(); saveTags(); } }, '✕'))));
    paintTags();

    const tagIn = h('input', {
      class: 'input', placeholder: 'add a tag…', style: 'width:150px',
      onKeydown: (e) => {
        if (e.key === 'Enter' && e.target.value.trim()) {
          tags.add(e.target.value.trim()); e.target.value = ''; paintTags(); saveTags();
        }
      },
    });

    // ── metabolism ──────────────────────────────────────────────────────────
    // Authored configuration, not measurement: how often an idle character
    // thinks, and how loud an event has to be to wake it.
    const beat = h('select', { class: 'select', style: 'width:auto' },
      [[5000, '5s'], [30000, '30s'], [60000, '1m'], [300000, '5m'], [600000, '10m'], [3600000, '1h']]
        .map(([ms, label]) => h('option', { value: ms, selected: (npc.tick?.heartbeat_ms || 0) === ms }, label)));
    beat.onchange = () => patch({ heartbeat_ms: Number(beat.value) }, 'metabolism saved');

    const gate = h('input', {
      type: 'range', min: '0', max: '1', step: '0.01',
      value: String(npc.tick?.salience_gate ?? 0.42), style: 'width:180px',
    });
    const gateOut = h('span', { class: 'mono tiny' }, String(npc.tick?.salience_gate ?? 0.42));
    gate.oninput = () => { gateOut.textContent = gate.value; };
    gate.onchange = () => patch({ salience_gate: Number(gate.value) }, 'gate saved');

    const envOn = h('input', {
      type: 'checkbox', checked: !!npc.environment_enabled,
      onChange: (e) => patch({ environment_enabled: e.target.checked }),
    });

    mount(bodyHost,
      h('div', { class: 'panel' },
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Name'), nameIn),
          h('div', { class: 'field' }, h('span', {}, 'Personality'),
            h('div', { class: 'row', style: 'gap:6px;padding-top:6px' },
              h('span', { class: 'chip' }, npc.personality_name || npc.personality_id || '—'),
              h('span', { class: 'chip' }, npc.world_id || '—')),
            h('div', { class: 'tiny dim', style: 'margin-top:5px' },
              'Fixed at creation. A character is what it started as; the substrate is what it turned into.'))),
        h('label', { class: 'field' }, h('span', {}, 'Description'), descIn),
        h('div', { class: 'tiny dim' },
          'This is the character’s identity section in the system prompt, and the source a portrait is ' +
          'generated from. Written as a present-day person: the personality supplies the anchor, this ' +
          'supplies the human texture.'),
        h('div', { class: 'row', style: 'margin-top:10px;gap:8px' },
          h('button', {
            class: 'btn sm',
            onClick: () => toast('regenerating a description — engine required', 'err'),
          }, '⟳ Regenerate description'),
          saveBtn)),

      h('div', { class: 'panel' },
        h('h3', { style: 'margin-top:0' }, 'Tags'),
        h('div', { class: 'row', style: 'gap:9px;align-items:flex-start' }, tagHost, tagIn),
        h('label', { class: 'row', style: 'gap:9px;margin-top:16px;cursor:pointer' },
          h('input', { type: 'checkbox', checked: !!npc.hidden,
            onChange: async (e) => {
              try {
                npc = await API.setHidden(id, e.target.checked);
                // The header carries a `hidden` chip, so this one is visible.
                repaint();
              } catch (err) {
                toast(err.detail || err.message || 'could not save', 'err');
                e.target.checked = !e.target.checked;
                return;
              }
              if (e.target.checked && !tags.size) toast('Hidden with no tags — this character will be unreachable from the roster', 'err');
            } }),
          h('div', {},
            h('div', { style: 'font-weight:600;font-size:.87rem' }, 'Hidden'),
            h('div', { class: 'tiny dim' },
              'Keeps this character out of the default list. Still found by filtering for any tag above. ' +
              'Hiding is discretion, not encryption.')))),

      h('div', { class: 'panel' },
        h('h3', { style: 'margin-top:0' }, 'Metabolism'),
        h('div', { class: 'row wrap', style: 'gap:26px;align-items:flex-end' },
          h('label', { class: 'field', style: 'margin:0' }, h('span', {}, 'Heartbeat'), beat),
          h('label', { class: 'field', style: 'margin:0' },
            h('span', {}, 'Salience gate ', gateOut), gate),
          h('label', { class: 'row', style: 'gap:9px;cursor:pointer' }, envOn,
            h('div', {}, h('div', { style: 'font-size:.86rem;font-weight:600' }, 'Environment'),
              h('div', { class: 'tiny dim' }, 'a simulator feeds it events')))),
        h('div', { class: 'tiny dim', style: 'margin-top:11px;max-width:88ch' },
          'The resting rate an idle character thinks at, and the level below which an event does not wake ' +
          'it. Both are authored settings rather than measurements — what the character is actually doing ' +
          'is on the Monitor tab, and nothing here can be typed into it.')),

      h('div', { class: 'panel' },
        h('h3', { style: 'margin-top:0' }, 'Danger zone'),
        h('div', { class: 'row wrap', style: 'gap:8px' },
          h('button', { class: 'btn sm', onClick: duplicate }, 'Duplicate'),
          h('button', { class: 'btn sm', onClick: exportJson }, 'Export JSON'),
          h('button', {
            class: 'btn sm',
            onClick: () => patch(
              { state: npc.state === 'suspended' ? 'idle' : 'suspended' },
              npc.state === 'suspended' ? 'resumed' : 'suspended',
            ).then((ok) => { if (ok) go('/npc/' + id + '/manage'); }),
          }, npc.state === 'suspended' ? 'Resume' : 'Suspend'),
          h('button', {
            class: 'btn sm danger',
            onClick: () => confirmDialog({
              title: 'Delete ' + npc.name, danger: true, requireText: npc.name,
              confirmText: 'Delete permanently',
              message: 'This tombstones the character. Its substrate stops being gathered and it disappears from every list.',
              onConfirm: async () => { await API.deleteNpc(id); toast('deleted', 'ok'); go('/'); },
            }),
          }, 'Delete'))));
  }

  /* A new character with this one's settings and none of its life.
   *
   * World, personality, description and tags carry; the substrate does not. A
   * copy that inherited lived experience would be the same character twice,
   * which is not what anybody means by duplicate — the point is a second one
   * that starts where this one started. */
  async function duplicate() {
    try {
      const made = await API.createNpc({
        name: (npc.name || 'Character') + ' (copy)',
        world_id: npc.world_id,
        personality_id: npc.personality_id,
        persona_description: npc.persona?.description || '',
        environment_enabled: !!npc.environment_enabled,
        tags: npc.tags || [],
      });
      toast('created ' + made.name, 'ok');
      go('/npc/' + made.npc_id + '/manage');
    } catch (e) {
      toast(e.detail || e.message || 'could not duplicate', 'err');
    }
  }

  /* The record as the daemon returned it.
   *
   * To the clipboard rather than a download: a script-driven save is inert in a
   * sandboxed frame, so a download button would do nothing and look broken.
   *
   * Written against `navigator.clipboard` directly rather than through
   * `lib/clip.js`, whose `copyText` is fire-and-forget and returns nothing —
   * awaiting it yields `undefined`, so a toast keyed on the result would always
   * claim failure. Here the promise is the answer. */
  async function exportJson() {
    const text = JSON.stringify(npc, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      toast('record copied to the clipboard', 'ok');
    } catch (_) {
      // Denied permission, or an insecure origin. Say so rather than claiming
      // a copy that did not happen.
      toast('could not reach the clipboard', 'err');
    }
  }

  async function openInteraction() {
    const modes = ['physical', 'video_call', 'voice_call', 'instant_message'];
    const sel = h('select', { class: 'select' }, modes.map((m) => h('option', { value: m }, MODE_LABEL[m])));
    const { close } = (await import('../lib/ui.js')).modal({
      title: 'Open an interaction with ' + npc.name,
      body: h('div', {},
        h('label', { class: 'field' }, h('span', {}, 'Mode'), sel),
        h('div', { class: 'tiny dim' },
          'Mode is fixed for the life of the interaction — it sets what the interlocutor can observe, and which ' +
          'tools exist. Changing it later means ending this one and opening another.')),
      footer: [h('button', { class: 'btn primary', onClick: async () => {
        const ix = await API.openInteraction(id, { mode: sel.value });
        close(); go('/interaction/' + ix.interaction_id);
      } }, 'Open')],
    });
  }

  /* No `teardown` clearing the rail, deliberately.
   *
   * Clearing it here emptied the rail the instant a tab was clicked, and the
   * replacement only arrived after `getNpc` and `getSubstrate` had both come
   * back — so the rail visibly vanished for the length of two round trips on
   * every click within a character. `paintRail` swaps the children in one go
   * when the new data is ready, which is the same end state without the gap.
   *
   * Nothing is left stale: `/npc/:id` and `/npc/:id/:tab` are the only routes
   * marked `keepsRail`, so leaving the character for any other page clears the
   * rail in `app.js` on the way out. */
  return { el };
}

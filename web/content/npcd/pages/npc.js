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
  // `modal` was missing, and `authorBelief` calls it — so "Author a belief"
  // threw a ReferenceError instead of opening. It went unnoticed because the
  // only other caller on this page reached it through a dynamic import.
  layerColor, idBadge, confirmDialog, modal,
} from '../lib/ui.js';
import * as sessions from '../lib/sessions.js';
import { scene, actParts } from '../lib/scene.js';

export async function render(params) {
  const id = params.id;
  const tab = params.tab || 'overview';
  /* Anything a tab starts that outlives its render — currently the Pulse tab's
   * poll. Run on the way out; see the note beside the return. */
  const teardowns = [];

  let npc;
  try { npc = await API.getNpc(id); }
  catch (_) { return { el: h('div', { class: 'page' }, empty('◌', 'No such character', id)) }; }

  const sub = await API.getSubstrate(id).catch(() => ({ layers: [] }));
  const layerCounts = Object.fromEntries((sub.layers || []).map((l) => [l.layer, l]));
  /* The mind's own layers, in the order its projection declares them — never a
   * list this page keeps, so a layer added to `projection.yaml` appears here
   * without anybody touching the console. See `npcd::engine::layers`. */
  const declared = (sub.layers || []).map((l) => l.layer);

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
    // Directly under Summary, because "what is it doing right now" is the second
    // question anybody has about a character and there was previously nowhere on
    // this page to answer it — the loop was only visible from the global view.
    railItem('pulse', 'Pulse', npc.tick?.ticks),
    /* The two ways to be present to a character, side by side, because the
     * choice between them is one question: are you standing with them, or
     * reaching them from somewhere else. Both used to sit behind an "Open
     * interaction" button that navigated away to a console of its own. */
    railItem('messages', 'Messages'),
    railItem('presence', 'In the room', npc.live_interactions),

    h('div', { class: 'rail-sec' }, 'layers'),
    // Counted in conversations — a day, a dream, a file — because that is what
    // the layer view lists.
    declared.map((l) => railItem(l, l[0].toUpperCase() + l.slice(1),
      layerCounts[l].conversations, layerColor(l))),

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
    /* Talking to a character is two rail tabs, not a button here. It used to be
     * "▶ Open interaction", which asked for a mode in a modal and then
     * navigated away to a console — three decisions and a page change to do the
     * one thing this page is for. */
    h('div', { class: 'row', style: 'gap:8px' },
      link(`/npc/${id}/messages`, { class: 'btn' }, '▤ Message'),
      link(`/npc/${id}/presence`, { class: 'btn primary' }, '◍ In the room')));
  paintHead();
  el.appendChild(head);

  /// Everything that reads `npc` outside the tab body.
  const repaint = () => { paintRail(); paintHead(); };

  const bodyHost = h('div', {});
  el.appendChild(bodyHost);

  /// The Messages tab's poll, declared here because the tab switch below clears
  /// it and a `let` beside the function it belongs to would be in the temporal
  /// dead zone when that runs.
  let messagePoll = null;

  // ── tabs ──────────────────────────────────────────────────────────────────

  const TABS = {
    overview, messages, presence, beliefs, relationships, agency, projection, monitor, manage,
    environment: environmentTab,
    pulse: pulseTab,
  };
  const fn = TABS[tab] || (declared.includes(tab) ? () => streamLayer(tab) : overview);
  // Any tab other than Messages stops its poll — otherwise the timer runs
  // against a detached node for as long as the console is open.
  if (tab !== 'messages') clearInterval(messagePoll);
  await fn();

  // ── pulse ─────────────────────────────────────────────────────────────────

  /* This character's own loop: what it perceived, what it did, and what it is
   * still holding.
   *
   * The same three panels the global Pulse view shows, scoped to one character
   * and with the window added — which only makes sense for one character at a
   * time, and is the panel that separates "it forgot" from "it never perceived
   * that". A turn that has faded out of the window is the first; a turn that
   * never reached the feed is the second, and they need different fixes.
   *
   * Reuses `.tick`, `.act` and `.perceive` from the global view rather than
   * restyling them, so a tick reads identically wherever it is seen. */
  async function pulseTab() {
    const stream = h('div', { class: 'pulse-stream' });
    const windowHost = h('div', { class: 'npc-window' });
    let stop = false;
    let seen = -1;

    const input = h('input', {
      class: 'input mono', style: 'flex:1',
      placeholder: 'say something, or /act shakes your hand',
      onKeydown: (e) => { if (e.key === 'Enter') send(); },
    });

    async function send() {
      const line = input.value.trim();
      if (!line) return;
      try {
        const r = await API.pulseInject(id, line);
        input.value = '';
        toast((r.preempts ? 'Preempt · ' : 'Delivered · ') + r.prose, 'ok');
        tick();
      } catch (e) { toast(e.detail || e.message || 'refused', 'err'); }
    }

    async function tick() {
      if (stop) return;
      try {
        const [feed, win] = await Promise.all([
          API.pulse({ limit: 40, npc_id: id }),
          API.npcWindow(id).catch(() => null),
        ]);
        const ticks = (feed.ticks || []).slice().reverse();
        if (!ticks.length) {
          mount(stream, h('div', { class: 'pulse-empty' },
            h('p', {}, feed.ready
              ? 'This character has not thought yet. It wakes on its own heartbeat — ' +
                'or send it something above.'
              : 'The engine is still loading.')));
        } else {
          const newest = ticks[0].tick;
          mount(stream, ...ticks.map((t) => {
            const c = { blocked: 'quiet', pending: 'batch', preempted: 'preempt' }[t.cause] || t.cause;
            return h('article', {
              class: 'tick is-' + t.cause + (t.tick > seen ? ' is-new' : ''),
              style: '--hue:' + (npc.hue != null ? npc.hue : 32),
            },
              h('div', { class: 'tick-spine' }),
              h('header', { class: 'tick-head' },
                h('span', { class: 'tick-cause' }, c),
                h('span', { class: 'tick-spacer' }),
                h('span', { class: 'tick-beat' }, '♥ ' + Math.round(t.heartbeat_ms / 1000) + 's')),
              h('div', { class: 'tick-perceived' },
                ...t.perceived.map((p) => h('p', { class: 'perceive' }, p))),
              t.acts && t.acts.length
                ? h('div', { class: 'tick-acts' }, ...t.acts.map((a) => {
                    const i = a.indexOf(' — ');
                    return h('div', { class: 'act' },
                      h('span', { class: 'act-tool' }, i === -1 ? a : a.slice(0, i)),
                      i === -1 ? null : h('span', { class: 'act-intent' }, a.slice(i + 3)));
                  }))
                : h('div', { class: 'tick-silent' }, 'nothing came of it'));
          }));
          seen = newest;
        }

        /* The verbatim tail. Deliberately below the feed and visually quieter:
         * it is state, not events, and reading it top-down would suggest the
         * character perceives it fresh each tick, which is the opposite of what
         * a window is. */
        if (win && win.turns) {
          mount(windowHost,
            h('div', { class: 'npc-window-hd' },
              h('span', {}, 'Carried into the next decode'),
              h('span', { class: 'tiny dim mono' },
                `${win.turns.length}/${win.cap} turns · ${fmtK(win.faded)} faded`)),
            win.turns.length
              ? h('div', { class: 'npc-window-turns' },
                  ...win.turns.map((t) => h('div', {
                    class: 'win-turn is-' + t.speaker,
                  }, t.text)))
              : h('p', { class: 'tiny dim', style: 'margin:0' },
                  'Nothing held — this character has not perceived anything yet today.'));
        }
      } catch (e) {
        mount(stream, h('div', { class: 'pulse-empty' },
          h('p', {}, e.error === 'no_engine' || e.status === 503
            ? 'The engine is still loading.'
            : 'Could not read this character’s pulse — ' + (e.detail || e.message))));
      }
    }

    mount(bodyHost,
      h('div', { class: 'page' },
        h('div', { class: 'row', style: 'gap:9px;margin-bottom:18px' },
          input,
          h('button', { class: 'btn primary', onClick: send }, 'Send')),
        stream,
        windowHost));

    await tick();
    const t = setInterval(tick, 2000);
    /* The rail persists across tabs, so leaving this one has to stop the poll —
     * otherwise every visit leaves an interval running against a detached DOM
     * for the life of the session. */
    teardowns.push(() => { stop = true; clearInterval(t); });
  }

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

  /* One layer, as this character can read it: the conversations in it — a day,
   * a dream, an ingested file — newest first, each with what it was written
   * with and its turns. Nothing here knows which layer it is showing. */
  async function streamLayer(layer) {
    const r = await API.getLayer(id, layer).catch(() => ({ conversations: [] }));
    const info = layerCounts[layer] || {};
    const convs = r.conversations || [];
    mount(bodyHost,
      h('div', { class: 'panel', style: 'margin-bottom:12px' },
        h('div', { class: 'row', style: 'gap:18px;flex-wrap:wrap' },
          stat('conversations', fmtNum(info.conversations)), stat('turns', fmtNum(info.turns)),
          stat('window', fmtK(info.window)))),
      convs.length
        ? h('div', {}, convs.map((c) => conversationPanel(layer, c)),
          r.more ? h('div', { class: 'tiny dim', style: 'margin-top:6px' },
            `The newest ${convs.length} of ${fmtNum(info.conversations)}.`) : null)
        : empty('◌', r.engine_connected === false ? 'The engine is not running' : 'Nothing in this layer yet'));
  }

  /* A user half every turn shares — the label a dream's lines are written
   * under, say — is said once above them rather than above every line. */
  function conversationPanel(layer, c) {
    const turns = c.turns || [];
    const shared = turns.length > 1 && turns.every((t) => t.user === turns[0].user) ? turns[0].user : null;
    const meta = Object.entries(c.metadata || {});
    return h('div', { class: 'panel', style: 'padding:12px 16px' },
      h('div', { class: 'row', style: 'gap:9px;margin-bottom:6px;flex-wrap:wrap' },
        h('span', { class: 'mono', style: 'font-weight:700;color:' + layerColor(layer) },
          c.name || 'conversation ' + c.timeline),
        h('span', { style: 'flex:1' }),
        h('span', { class: 'tiny mono dim' }, turns.length + (turns.length === 1 ? ' turn' : ' turns'))),
      meta.length
        ? h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap;margin-bottom:6px' },
          meta.map(([k, v]) => h('span', { class: 'chip', title: k }, `${k}: ${v}`)))
        : null,
      shared ? h('div', { class: 'tiny dim', style: 'margin-bottom:4px' }, shared) : null,
      turns.map((t) => h('div', { style: 'font-size:.86rem;margin:4px 0' },
        shared ? null : h('div', { class: 'tiny dim' }, t.user),
        h('div', {}, t.assistant))));
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

  /* Messaging the character on its handset.
   *
   * **The same threads the characters use between themselves.** What is sent
   * here goes into the world rather than into a private channel: the character
   * is told about it by the ordinary perception sweep and answers with the
   * ordinary `message` act, from wherever it is standing. That is what makes
   * the reply worth having — it is the character speaking from inside the
   * world, and everything else in the world can see the conversation happened.
   *
   * Polled rather than streamed, and the shape follows from that: a reply
   * arrives on the character's own schedule, when its next turn comes round,
   * because it is deciding to answer rather than being queried. There is no
   * event to stream — the thread simply has one more line on it.
   */
  /* **The frame is built once and only its contents change.**
   *
   * It used to rebuild everything on each repaint — including the composer —
   * and a repaint happens on every reply. So sending a message destroyed the
   * box you had just typed into and took the caret with it: you sent one line
   * and then had to click back into the field to send another. Removing an
   * element from the document blurs it, so keeping the same node and re-mounting
   * it would not have helped either; the fix is not to touch it at all.
   *
   * Only `thread` and `actsInner` are repainted now, and neither can hold
   * focus. */
  async function messages() {
    // The lane is the scroller, not the thread inside it — see `.msg-thread`.
    const toTheEnd = () => {
      const lane = thread.parentElement;
      if (lane) lane.scrollTop = lane.scrollHeight;
    };
    /* **What it does, beside what reached you** — the front page's sample, and
     * on a thread it is the only way to see it at all.
     *
     * A message reaches somebody who is nowhere near, so the character goes on
     * living the whole time you are texting it: walking somewhere, reading a
     * bench, deciding something. None of that arrives on the thread, which is
     * correct — you cannot see it — and it left this tab a chat window with a
     * character at the other end doing nothing visible.
     *
     * Polled off the character's own tick feed rather than streamed, for the
     * reason the thread is: there is no session here, and a reply lands when
     * its next turn comes round. */
    const actsInner = h('div', { class: 'acts-inner' });
    const thread = h('div', { class: 'msg-thread' });

    /* What the lane last showed.
     *
     * **Repainting an unchanged lane is visible**, because `.act-item` animates
     * in — so a poll that found nothing new replayed the rise on every row and
     * the whole pane blinked every four seconds, hardest for a character that
     * was doing nothing at all. Nothing here diffs rows, so the cheap and
     * complete answer is not to touch the DOM when the answer has not moved. */
    let shown = '';

    async function paintActs() {
      const p = await API.pulse({ limit: 40, npc_id: id }).catch(() => null);
      if (!p) return;
      const mine = (p.ticks || []).filter((t) => String(t.npc_id) === String(id));
      /* `\0` as an escape rather than a literal NUL byte. The value is
       * identical; what changes is that this file is no longer *binary* to
       * ripgrep, which was silently excluding all 61KB of it from every
       * content search in the repo. */
      const sig = mine.map((t) => t.tick + ':' + (t.acts || []).join('\0')).join('|');
      if (sig === shown) return;
      shown = sig;
      mount(actsInner, mine.length
        ? mine.flatMap((t) => (t.acts || []).map((a) => {
          // Same split the scene uses, so an act reads identically in both.
          const p = actParts(a, '');
          return h('div', { class: 'act-item' },
            h('div', { class: 'row' },
              h('span', { class: 'tk' }, 't' + t.tick),
              h('span', { class: 'tool' }, p.tool),
              h('span', { class: 'intent' }, p.intent)));
        }))
        : h('div', { class: 'tiny dim' }, 'nothing yet'));
      // The lane is the scroller. Guarded because this also runs from the poll,
      // and a throw inside an interval callback would kill the loop silently.
      const lane = actsInner.parentElement;
      if (lane) lane.scrollTop = lane.scrollHeight;
    }

    /* The thread, repainted whole. It holds no focus and no scroll of its own,
     * so rebuilding it costs nothing a reader can feel. */
    const paintThread = (r) => {
      mount(thread, (r.messages || []).length
        /* The words in their own node with the speaker under them — the same
         * bubble the scene uses, so a message from a character looks the same
         * wherever you read it. The name used to sit above the text as a
         * `.npc-meta` heading, which read as a label introducing a block rather
         * than as somebody having said something. */
        ? (r.messages || []).map((m) => h('div', {
          class: 'msg' + (m.from === r.as ? ' mine' : ''),
        },
          h('div', { class: 'words' }, m.text),
          h('div', { class: 'who' }, m.from)))
        : h('div', { class: 'tiny dim' }, 'Nothing said yet. Send something and it reaches their handset.'));
      toTheEnd();
    };

    // A conversation tab fills the page rather than sitting in a box on it —
    // see `.talk-host`. `bodyHost` is rebuilt per render, so this needs no
    // clearing when another tab takes over.
    bodyHost.className = 'talk-host';

    const first = await API.getMessages(id).catch(() => ({ messages: [], in_a_world: false }));
    let count = (first.messages || []).length;

    const box = h('input', {
      // `in` is not a class anybody defined, so this was a bare browser input
      // beside a styled button — the stray underline.
      class: 'input', style: 'flex:1',
      placeholder: 'Say something to ' + (first.with || 'them') + '…',
    });

    async function say(e) {
      e.preventDefault();
      const text = (box.value || '').trim();
      if (!text) return;
      box.value = '';
      // Focus stays where it was — nothing here replaces the field.
      await API.sendMessage(id, text).catch(() => {});
      const next = await API.getMessages(id).catch(() => null);
      if (!next) return;
      count = (next.messages || []).length;
      paintThread(next);
    }

    mount(bodyHost,
      h('div', { class: 'tiny dim', style: 'margin-bottom:11px' },
        first.in_a_world
          ? `On ${first.with}’s handset, as ${first.as}. It answers on its own schedule — when its next turn comes round.`
          : 'This character has no body in a world, so there is nothing to reach it on.'),
      h('div', { class: 'lane-frame' },
        h('div', { class: 'lane-body' },
          h('div', { class: 'lane' }, h('div', { class: 'pane-hd' }, 'the conversation'), thread),
          h('div', { class: 'lane acts' }, h('div', { class: 'pane-hd' }, 'what they do'), actsInner)),
        first.in_a_world
          ? h('form', { class: 'composer', onSubmit: say },
            h('div', { class: 'composer-inner row' },
              box,
              h('button', { class: 'btn primary', type: 'submit' }, 'Send')))
          : null));

    paintThread(first);
    paintActs();
    // After the outlet has the page — `focus` on a detached input does nothing.
    if (first.in_a_world) requestAnimationFrame(() => box.focus());

    // Poll while this tab is the one showing. Cleared by the tab switch, so
    // leaving stops the timer rather than leaving it running against a detached
    // node for as long as the console is open.
    clearInterval(messagePoll);
    if (first.in_a_world) {
      messagePoll = setInterval(async () => {
        // The act lane moves whether or not anything was said to you — that is
        // the point of it — so it repaints every pass.
        paintActs();
        const next = await API.getMessages(id).catch(() => null);
        if (!next) return;
        const n = (next.messages || []).length;
        if (n === count) return;
        count = n;
        paintThread(next);
      }, 4000);
    }
  }

  // ── interactions / environment / manage ───────────────────────────────────

  /* ── being in the room ─────────────────────────────────────────────────────
   *
   * **You are somewhere, and it is not a page you are on.** Either your body is
   * standing with this character — in which case it follows them about, they
   * are told they have company, and you can speak — or you are away, in which
   * case there is nothing to say anything *through*. That is the whole state,
   * and this tab is the two halves of it plus the control that crosses between.
   *
   * It replaced a list of "interactions" behind an Open button that navigated
   * to a console of its own. That put a session — a thing the world knows about
   * — behind an object the operator had to manage, when the only question
   * anybody has is "am I with them or not".
   *
   * A conversation is company, not a copy: what is said here goes through the
   * same door everything else does, the character remembers it afterwards, and
   * the rest of the world can see that it happened.
   */
  async function presence() {
    const commands = (await API.listCommands().catch(() => ({ commands: [] }))).commands;

    // Not `bar` — that is the progress-bar helper imported at the top of this
    // file, and shadowing it here would be a trap for the next edit.
    const control = h('div', { class: 'presence-bar' });
    const body = h('div', { class: 'presence-body' });
    let view = null;
    let busy = false;

    // Detaching the view does not leave the room — see `lib/scene.js`. This is
    // only the stream being let go when the tab is left.
    const drop = () => { if (view) { view.teardown(); view = null; } };
    teardowns.push(drop);

    async function walkIn() {
      if (busy) return;
      busy = true;
      paintControl(null);
      try {
        await sessions.open(id, 'physical');
        await paint();
      } catch (e) {
        /* The one refusal worth naming: a character with no body cannot be
         * stood beside, and the answer is the message thread rather than a
         * failure the reader has to interpret. */
        toast(e.error === 'not_in_a_world'
          ? `${npc.name} has no body in a world — message them instead`
          : (e.detail || e.message || 'could not walk in'), 'err');
        busy = false;
        await paint();
      }
    }

    async function leave(session) {
      if (busy) return;
      busy = true;
      paintControl(null);
      drop();
      await sessions.close(session.ix);
      await paint();
    }

    /** Where you are, and the control that changes it. */
    function paintControl(session) {
      if (busy) {
        return mount(control, h('span', { class: 'tiny dim' }, 'walking…'));
      }
      if (!session) {
        return mount(control,
          h('div', { class: 'presence-where' },
            h('span', { class: 'dot idle' }),
            h('span', {}, 'You are not in the world.'),
            h('span', { class: 'tiny dim' },
              'Nothing you type reaches them from here.')),
          h('button', { class: 'btn primary', onClick: walkIn },
            '→ Go to ' + npc.name));
      }
      mount(control,
        h('div', { class: 'presence-where' },
          h('span', { class: 'dot active' }),
          h('span', {}, 'You are with ' + npc.name + '.'),
          h('span', { class: 'tiny dim' },
            'You follow them wherever they go, and they know you are there.')),
        h('button', {
          class: 'btn danger', onClick: () => leave(session),
        }, 'Leave the room'));
    }

    async function paint() {
      drop();
      busy = false;
      const session = await sessions.where(id, npc);
      paintControl(session);
      if (!session) {
        return mount(body, empty('◍', 'You are somewhere else',
          `Walk in and ${npc.name} is told they have company. Until then there is no room to speak into — `
          + 'the message thread is the way to reach somebody you are not standing next to.'));
      }
      view = scene({
        npc,
        session,
        who: session.interlocutor?.display || 'You',
        commands,
        // The daemon ended it — it went quiet, or another tab walked out. The
        // scene says so itself; this puts the control back to "not here".
        onEnded: () => paintControl(null),
      });
      mount(body, view.el);
    }

    // The control last, so it sits at the bottom of the tab: where you are is
    // the thing you check after reading, and the button that changes it belongs
    // under the room rather than over it. `talk-host` makes the room above it
    // fill the page — see `.talk-host`.
    bodyHost.className = 'talk-host';
    mount(bodyHost, body, control);
    await paint();
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
          /* **Real, and streamed into the field.** It writes the same
           * generation the create step does, against the character's own world
           * and personality, and lands in the textarea so the existing Save
           * decides whether it is kept. */
          h('button', {
            class: 'btn sm',
            onClick: async (e) => {
              // Disabled while it runs; the label stays put. The text arriving
              // in the field below is the progress indicator.
              const b = e.currentTarget;
              b.disabled = true;
              const before = descIn.value;
              descIn.value = '';
              descIn.placeholder = 'writing…';
              try {
                const r = await API.generateDescriptionStream(
                  { world_id: npc.world_id, personality_id: npc.personality_id },
                  (ev) => {
                    if (ev.event === 'token') {
                      descIn.value += ev.text;
                      descIn.scrollTop = descIn.scrollHeight;
                    }
                  });
                descIn.value = r.description;
                /* Not saved here. The field is now dirty and Save is what
                 * commits it, so a draft you dislike is discarded by leaving. */
                toast('description written — Save to keep it', 'ok');
              } catch (err) {
                descIn.value = before;
                toast(err.error === 'engine_unavailable'
                  ? 'the engine is still loading — try again in a moment'
                  : (err.detail || err.message || 'could not write a description'), 'err');
              } finally {
                b.disabled = false;
                descIn.placeholder = '';
              }
            },
          }, '⟳ Regenerate description'),
          /* **The portrait, drawn from the description above it.**
           *
           * Here rather than only on the create step because this is where the
           * description is edited: a portrait is generated *from* it, so the
           * button belongs beside the thing it reads. The daemon answers with
           * the whole record, so the header's avatar updates from the same
           * response rather than from a second fetch.
           *
           * It blocks — the drain stops every character thinking while it runs
           * — so the button says so and disables itself rather than letting an
           * impatient second press queue a second stop-the-world job. */
          h('button', {
            class: 'btn sm',
            onClick: async (e) => {
              // Disabled while it runs; the label stays put. A toast opens the
              // wait and the portrait itself closes it.
              const b = e.currentTarget;
              b.disabled = true;
              toast('drawing the portrait — the cast is paused', 'ok');
              try {
                npc = await API.generatePortrait(id);
                repaint();
                toast('portrait drawn', 'ok');
              } catch (err) {
                /* `no_image_model` is a deployment fact, not a fault: this
                 * daemon has no image guest configured. Saying which it is
                 * stops somebody debugging a model that was never there. */
                toast(err.error === 'no_image_model'
                  ? 'no image model is configured on this daemon'
                  : (err.detail || err.message || 'could not draw a portrait'), 'err');
              } finally {
                b.disabled = false;
              }
            },
          }, '⟳ Draw portrait'),
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
            h('span', {}, 'Salience gate ', gateOut), gate)),
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

  /* There is no `openInteraction` any more, and its absence is the change.
   *
   * It asked which mode you wanted in a modal, opened a session, and navigated
   * to a console — so being with a character was an object you created and
   * managed rather than a thing that was true of you. The two modes are now the
   * two tabs above: `messages` reaches somebody who is nowhere near, and
   * `presence` is standing in the room, with the walking in and out done by the
   * control at the bottom of it. */

  /* The `teardown` does NOT clear the rail, deliberately.
   *
   * Clearing it emptied the rail the instant a tab was clicked, and the
   * replacement only arrived after `getNpc` and `getSubstrate` had both come
   * back — so the rail visibly vanished for the length of two round trips on
   * every click within a character. `paintRail` swaps the children in one go
   * when the new data is ready, which is the same end state without the gap.
   *
   * Nothing is left stale: `/npc/:id` and `/npc/:id/:tab` are the only routes
   * marked `keepsRail`, so leaving the character for any other page clears the
   * rail in `app.js` on the way out.
   *
   * What it does clear is anything a tab left running. The Pulse tab polls, and
   * a poll that survives the page runs against a detached DOM for the rest of
   * the session — one more request every two seconds per visit, for ever. */
  return { el, teardown: () => { for (const f of teardowns) f(); } };
}

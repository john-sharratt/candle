/* Interaction console (§32–35).
 *
 * The two-latency stream is the whole design: `act` frames land immediately as
 * they commit; `narration` assembles at tick close, grouped under the tick that
 * bounds it. Watching the right column move while the left is still empty is
 * the operator seeing the mind act before it explains itself.
 *
 * Messaging modes render bubbles that wait for `rendered` (the NPC emits intent;
 * the narrator writes the words). Physical mode renders a colour-railed scene
 * where every entry is a typed turn. */

import { API } from '../lib/api.js';
import { h, mount, worldTime, ago } from '../lib/dom.js';
import { go, link } from '../lib/router.js';
import { toast, MODE_LABEL, MODE_ICON, empty } from '../lib/ui.js';
import { parseLine, filterCommands, toLine } from '../lib/cmd.js';
import { sticky, selectionInside, throttled } from '../lib/live.js';
import { state as vp, onBreakpoint } from '../lib/viewport.js';

const MESSAGING = new Set(['instant_message', 'video_call', 'voice_call']);

export async function render(params) {
  const ix = params.ix;
  const info = await API.getInteraction(ix).catch(() => null);
  if (!info) return { el: h('div', { class: 'page' }, empty('◌', 'No such interaction', ix)) };

  const npc = await API.getNpc(info.npc_id).catch(() => ({ name: 'NPC', npc_id: info.npc_id }));
  const commands = (await API.listCommands().catch(() => ({ commands: [] }))).commands;

  const messaging = MESSAGING.has(info.mode);
  let operatorView = true;

  const stageInner = h('div', { class: 'stage-inner' });
  const stage = h('div', { class: 'stage' }, stageInner);
  const actsPane = h('div', { class: 'acts' });
  const actIndex = new Map();     // act_id -> { data, bubbleBody, actEl }
  const ticks = new Map();        // tick -> stage container

  // ── header ────────────────────────────────────────────────────────────────

  const idleEl = h('span', { class: 'tiny mono dim' }, '');
  let idleLeft = info.idle_remaining_secs || 0;
  const idleTimer = setInterval(() => {
    idleLeft = Math.max(0, idleLeft - 1);
    const m = String(Math.floor(idleLeft / 60)).padStart(2, '0');
    const s = String(idleLeft % 60).padStart(2, '0');
    idleEl.textContent = `idle ${m}:${s}`;
  }, 1000);

  // On a wide screen the act stream is a docked column and "ops" toggles it.
  // On narrow it becomes a bottom sheet: the two-lane reading is impossible
  // side-by-side at phone width, so acts move behind a pull-up instead of
  // competing with the narration for the same 380 pixels.
  const applyActsVisibility = () => {
    if (vp.narrow) {
      actsPane.classList.remove('hidden');
      actsPane.classList.toggle('sheet-open', operatorView);
    } else {
      actsPane.classList.remove('sheet-open');
      actsPane.classList.toggle('hidden', !operatorView);
    }
    opsBtn.classList.toggle('primary', operatorView);
    opsBtn.textContent = vp.narrow ? (operatorView ? '▾ acts' : '▴ acts') : '◫ ops';
    repaintObservability();
  };

  const opsBtn = h('button', {
    class: 'btn sm', title: 'Operator view — shows intents and acts the interlocutor cannot observe',
    onClick: () => { operatorView = !operatorView; applyActsVisibility(); },
  }, '◫ ops');

  // Acts start closed on narrow (the narration is the point of the page there)
  // and open on wide (the two-lane split is the point of the page here).
  operatorView = !vp.narrow;
  const unwatchVp = onBreakpoint(() => { operatorView = !vp.narrow; applyActsVisibility(); });

  const head = h('div', { class: 'console-hd' },
    link('/npc/' + npc.npc_id, { class: 'btn ghost sm' }, '←'),
    h('span', { style: 'font-size:1.1rem' }, MODE_ICON[info.mode] || '◍'),
    h('strong', {}, npc.name),
    h('span', { class: 'chip' }, MODE_LABEL[info.mode] || info.mode),
    h('span', { class: 'tiny dim' }, 'as ' + (info.interlocutor?.display || '—')),
    h('span', { style: 'flex:1' }),
    idleEl, opsBtn,
    h('button', {
      class: 'btn sm danger',
      onClick: async () => { await API.endInteraction(ix); toast('interaction ended', 'ok'); go('/npc/' + npc.npc_id); },
    }, 'End'));

  // ── composer + slash palette ──────────────────────────────────────────────

  const input = h('textarea', {
    class: 'textarea', rows: 1, placeholder: messaging ? 'Message…  ( / for commands )' : 'Say something, or / for commands',
    onInput: onType,
    onKeydown: onKey,
  });
  const palette = h('div', { class: 'palette', hidden: true });
  let pal = { open: false, sel: 0, matches: [], parsed: null };

  const composer = h('div', { class: 'composer' }, palette,
    h('div', { class: 'composer-inner row' }, input,
      h('button', { class: 'btn primary', onClick: send }, 'Send')));

  function onType() {
    input.style.height = 'auto';
    input.style.height = Math.min(150, input.scrollHeight) + 'px';
    const line = input.value;
    if (!line.startsWith('/')) return closePalette();
    const parsed = parseLine(line, commands);
    pal.parsed = parsed;
    if (parsed.command) {
      pal.matches = [parsed.command];
      paintParams(parsed);
    } else {
      pal.matches = parsed.matches || filterCommands(commands, parsed.term || '');
      pal.sel = Math.min(pal.sel, Math.max(0, pal.matches.length - 1));
      paintList();
    }
    palette.hidden = false;
    pal.open = true;
  }

  function closePalette() { palette.hidden = true; pal.open = false; pal.sel = 0; }

  function paintList() {
    const groups = new Map();
    pal.matches.forEach((c) => {
      if (!groups.has(c.group)) groups.set(c.group, []);
      groups.get(c.group).push(c);
    });
    let idx = 0;
    mount(palette, [...groups.entries()].map(([g, cs]) => h('div', {},
      h('div', { class: 'group' }, g),
      cs.map((c) => {
        const i = idx++;
        return h('div', {
          class: 'opt' + (i === pal.sel ? ' on' : ''),
          onMouseenter: () => { pal.sel = i; paintList(); },
          onClick: () => accept(c),
        }, h('span', { class: 'nm' }, '/' + c.name), h('span', { class: 'ds' }, c.summary));
      }))));
    if (!pal.matches.length) mount(palette, h('div', { class: 'opt' }, h('span', { class: 'ds' }, 'no command matches')));
  }

  function paintParams(p) {
    const c = p.command;
    mount(palette,
      h('div', { class: 'group', style: 'display:flex;gap:10px;align-items:baseline' },
        h('span', { style: 'color:var(--accent)' }, '/' + c.name),
        h('span', { style: 'text-transform:none;letter-spacing:0;font-weight:400' }, c.summary),
        h('span', { style: 'flex:1' }),
        h('span', { style: 'text-transform:none' }, '→ ' + c.emits)),
      h('div', { class: 'params' },
        p.fields.length ? p.fields.map((f) => h('div', { class: 'param ' + f.state },
          h('span', { class: 'pn' }, f.name),
          h('span', { class: 'pt' }, typeLabel(f.schema) + (f.required ? '' : ' ?')),
          h('span', { class: 'pd' }, f.error || f.schema.description ||
            (f.schema.enum ? f.schema.enum.join(' · ') : '')),
          h('span', { class: 'pv' }, f.state === 'satisfied' ? '✓ ' + fmtVal(f.value)
            : f.schema.default !== undefined ? '(default ' + f.schema.default + ')' : '')))
          : h('div', { class: 'tiny dim' }, 'no parameters'),
        h('div', { class: 'tiny dim', style: 'margin-top:9px' },
          p.complete ? '⏎ send' : '⏎ blocked — ' + (p.missing || []).join(', ') + ' required',
          '   ·   ⇥ next field   ·   esc cancel')));
  }

  const typeLabel = (s) => s.enum ? 'enum' : (s.type || 'string') +
    (s.minimum != null || s.maximum != null ? ` ${s.minimum ?? ''}..${s.maximum ?? ''}` : '');
  const fmtVal = (v) => typeof v === 'string' && /\s/.test(v) ? JSON.stringify(v) : String(v);

  function accept(c) {
    input.value = '/' + c.name + ' ';
    input.focus();
    onType();
  }

  function onKey(e) {
    if (pal.open && !pal.parsed?.command) {
      if (e.key === 'ArrowDown') { e.preventDefault(); pal.sel = Math.min(pal.sel + 1, pal.matches.length - 1); return paintList(); }
      if (e.key === 'ArrowUp') { e.preventDefault(); pal.sel = Math.max(0, pal.sel - 1); return paintList(); }
      if (e.key === 'Tab' || (e.key === 'Enter' && pal.matches.length)) {
        e.preventDefault(); return accept(pal.matches[pal.sel]);
      }
    }
    if (e.key === 'Escape') return closePalette();
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  }

  /* One injection, and whether it landed.
   *
   * The caller paints the line onto the stage only on `true`, so a refusal is
   * visible as the message not appearing, with the reason in a toast. */
  async function deliver(payload) {
    try {
      await API.inject(ix, payload);
      return true;
    } catch (e) {
      toast(e.detail || e.message || 'the daemon did not take that', 'err');
      return false;
    }
  }

  async function send() {
    const line = input.value.trim();
    if (!line) return;
    if (line.startsWith('/')) {
      const p = parseLine(line, commands);
      if (!p.command) return toast('Unknown command', 'err');
      if (!p.complete) return toast('Missing: ' + (p.missing || []).join(', '), 'err');
      /* Painted only once it was accepted.
       *
       * The error used to be swallowed and the line added regardless, so a
       * message the daemon never took looked exactly like one it did — the
       * operator reads their own words on the stage and believes the character
       * heard them. */
      if (!(await deliver({ command: p.command.name, args: p.args }))) return;
      addLocal(p.command.name === 'say' ? p.args.text : '/' + p.command.name,
        p.command.name === 'say' ? 'say' : p.command.name === 'beat' ? 'beat_' : 'cue', true);
    } else {
      if (!(await deliver({ text: line }))) return;
      addLocal(line, 'say', true);
    }
    input.value = '';
    input.style.height = 'auto';
    closePalette();
  }

  // ── stage rendering ───────────────────────────────────────────────────────

  function tickBox(t) {
    if (ticks.has(t)) return ticks.get(t);
    const box = h('div', {});
    stageInner.appendChild(box);
    ticks.set(t, box);
    return box;
  }

  // Follow the tail, but yield the instant the operator scrolls up — and never
  // let a scroll we caused read as them taking over (lib/live.js).
  const scroller = sticky(stage);
  function scrollDown() { scroller.follow(); }

  function addLocal(text, kind, mine) {
    if (messaging) {
      stageInner.appendChild(h('div', { class: 'bubble' + (mine ? ' me' : '') },
        h('div', { class: 'body' }, text),
        h('div', { class: 'who' }, (mine ? info.interlocutor?.display || 'You' : npc.name))));
    } else {
      stageInner.appendChild(h('div', { class: 'beat ' + kind },
        h('div', { class: 'rail-mark' }),
        h('div', { class: 'txt' }, text),
        h('div', { class: 'tag' }, mine ? 'you' : kind)));
    }
    scrollDown();
  }

  function onAct(a) {
    // right column — always, with intent (operator view)
    const observable = !a.observable_in || a.observable_in.includes(info.mode);
    const actEl = h('div', { class: 'act-item' + (observable ? '' : ' unobs') },
      h('div', { class: 'top' },
        h('span', { class: 'tk' }, 't' + a.tick),
        h('span', { class: 'tool' }, a.tool),
        !observable ? h('span', { class: 'tiny dim' }, '⊘ not observable') : null),
      a.intent ? h('div', { class: 'intent' }, '→ ' + a.intent) : null,
      h('div', { class: 'tiny dim mono', 'data-rendered': a.act_id }, observable ? 'rendering…' : ''));
    actsPane.appendChild(actEl);
    actsPane.scrollTop = actsPane.scrollHeight;

    // left column — only what this vantage can observe
    let bubbleBody = null;
    if (observable) {
      if (messaging) {
        if (a.tool === 'speak' || a.tool === 'send_image') {
          bubbleBody = h('div', { class: 'body' },
            h('span', { class: 'typing' }, h('i'), h('i'), h('i')));
          stageInner.appendChild(h('div', { class: 'bubble' }, bubbleBody, h('div', { class: 'who' }, npc.name)));
          scrollDown();
        }
      } else {
        const kind = a.tool === 'speak' ? 'say' : 'act';
        bubbleBody = h('div', { class: 'txt dim' }, '…');
        tickBox(a.tick).appendChild(h('div', { class: 'beat ' + kind },
          h('div', { class: 'rail-mark' }), bubbleBody, h('div', { class: 'tag' }, kind)));
        scrollDown();
      }
    }
    actIndex.set(a.act_id, { data: a, bubbleBody, actEl, observable });
  }

  // Rendered prose REPLACES a placeholder, so it is the one path here that can
  // destroy DOM the operator is mid-selection on. Queue it and flush only while
  // no selection lives in the stage — nothing is lost, the surface just pauses
  // while you copy (zend's index.html arrived at the same rule).
  const renderQueue = new Map();
  const flushRendered = throttled(() => {
    if (!renderQueue.size) return;
    if (selectionInside(stage)) return;   // retry on the next frame
    for (const [actId, text] of renderQueue) {
      const rec = actIndex.get(actId);
      if (!rec) continue;
      const slot = rec.actEl.querySelector(`[data-rendered="${actId}"]`);
      if (slot) slot.textContent = rec.observable ? '✓ ' + text : '';
      if (rec.bubbleBody) {
        rec.bubbleBody.classList.remove('dim');
        mount(rec.bubbleBody, text);
      }
    }
    renderQueue.clear();
    scrollDown();
  });

  function onActRendered(r) {
    if (!actIndex.has(r.act_id)) return;
    renderQueue.set(r.act_id, r.rendered?.text || '');
    flushRendered();
  }

  // A selection that ends while frames are queued must not strand them.
  document.addEventListener('selectionchange', flushRendered);

  function onTick(t) {
    tickBox(t.tick).appendChild(h('div', { class: 'tickline' }, `tick ${t.tick} · ${t.acts} acts`));
    scrollDown();
  }

  function onNarration(n) {
    const box = tickBox(n.tick);
    if (messaging) {
      // messaging narration is the caption for the window, shown quietly
      box.appendChild(h('div', { class: 'tiny dim', style: 'margin:4px 0 14px;font-style:italic' }, n.text));
    } else {
      box.appendChild(h('div', { class: 'beat say' },
        h('div', { class: 'rail-mark' }),
        h('div', { class: 'txt' }, n.text),
        h('div', { class: 'tag' }, 'narration')));
    }
    scrollDown();
  }

  function onSceneImage(s) {
    if (messaging) return;
    stageInner.appendChild(h('div', { class: 'scene-img' },
      h('div', { class: 'ph' }, '⛰  ' + (s.prompt || 'the scene, from where you are standing')),
      h('div', { class: 'tiny dim', style: 'padding:7px 10px;border-top:1px solid var(--line-2)' },
        'the scene · from your vantage')));
    scrollDown();
  }

  function repaintObservability() {
    // Participant view hides acts the interlocutor could not observe.
    actsPane.querySelectorAll('.act-item.unobs').forEach((n) => { n.style.display = operatorView ? '' : 'none'; });
    stageInner.querySelectorAll('.beat.beat_').forEach((n) => { n.style.display = operatorView ? '' : 'none'; });
  }

  // ── assemble ──────────────────────────────────────────────────────────────

  const el = h('div', { class: 'page flush' },
    h('div', { class: 'console' }, head,
      h('div', { class: 'console-body' }, stage, actsPane),
      composer));

  if (!messaging) {
    stageInner.appendChild(h('div', { class: 'tiny dim', style: 'margin-bottom:14px' },
      'Every line below is a turn on the interaction layer. Colour and rail come from turn metadata, not a second log.'));
  }

  applyActsVisibility();

  const stream = API.streamInteraction(ix, {
    onAct, onActRendered, onTick, onNarration, onSceneImage,
    onError: () => toast('stream closed', 'err'),
  });

  return {
    el,
    teardown: () => {
      stream.cancel();
      clearInterval(idleTimer);
      document.removeEventListener('selectionchange', flushRendered);
      unwatchVp();
    },
  };
}

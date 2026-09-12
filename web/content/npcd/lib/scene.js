/* Standing in a room with a character: what you see, and what you say.
 *
 * # Why this is a component and not a page
 *
 * Being present to a character is not somewhere you go, it is something that is
 * true of you — your body is in their room and it follows them about. So it
 * belongs on the character's own page beside everything else about them, next
 * to the message thread, rather than behind a button that navigates away to a
 * console of its own. This is that surface, with no chrome of its own: the
 * caller owns the header, the presence control and the page.
 *
 * # Two lanes, because that is the thing worth seeing
 *
 * The same split the front page's sample makes: what it *does* beside what you
 * see. They are not the same list. The scene is scoped to your vantage, so an
 * act with no observable trace — the character thinking, deciding, weighing
 * something — never appears in it; you are standing in a room, not in its head.
 * The right lane has every act regardless, which is what makes the two disagree
 * and what makes the disagreement legible.
 *
 * I dropped this lane when the console moved onto the character's page, on the
 * reasoning that the Pulse tab already shows the loop. That was wrong: Pulse is
 * the instrument you go to, and this is the thing you are looking at while you
 * talk. Losing it made the page a chat window.
 *
 * An act arrives as the scene it makes: what was said, what was done, in the
 * order it happened. The text is the act's own — this daemon renders an act *as*
 * its own line and never sends a separate rendering, so waiting for one leaves
 * an ellipsis on the page for ever.
 */

import { API } from './api.js';
import { h, mount } from './dom.js';
import { toast } from './ui.js';
import { parseLine, filterCommands, FREE_TEXT } from './cmd.js';
import { sticky, selectionInside, throttled } from './live.js';
import * as sessions from './sessions.js';

/* Tools whose observable trace is somebody speaking. */
const SPOKEN = new Set(['speak', 'say', 'tell', 'ask', 'answer', 'reply', 'send_image']);

/* Tools that land in the room without a word: `act` is one body doing something
 * to another — steadying them, taking a weapon off them, putting them down —
 * and `gesture` is shown to everyone and lands on nobody.
 *
 * **These belong on the stage, and nothing else does.** You are standing in the
 * room: somebody putting a hand on you is not a line in a log, it is the thing
 * that just happened to you. But mirroring *every* act here is what the right
 * lane was built to stop — walking, reading a bench, deciding something is a
 * character's own business and would bury the exchange under its errands. So
 * the rule is what a body in the room would register as an event: words, and
 * hands. */
const IN_THE_ROOM = new Set(['act', 'gesture']);

/**
 * One act, split into the label and the thing worth reading.
 *
 * `Act::summary` renders `tool — args`, and the act lane puts the first half in
 * a narrow monospace column and the second in an italic one. **But not every
 * line has that shape.** An act the world answered — `file_read`, `read`,
 * `scan` — comes back as the world's own sentence with no separator in it
 * ("You read the appraisal bench. It stands at `reading`."), and putting that
 * in the 92-pixel tool column wraps a sentence down the side of the pane.
 *
 * So a line with no separator is not a tool call being named, it is something
 * that happened, and it goes in the column for things that happened.
 */
export function actParts(tool, intent) {
  if (intent) return { tool, intent };
  const cut = (tool || '').indexOf(' — ');
  if (cut >= 0) return { tool: tool.slice(0, cut), intent: tool.slice(cut + 3) };
  // No separator and nothing beside it: a sentence, not a name.
  return /\s/.test(tool || '') ? { tool: '', intent: tool } : { tool, intent: '' };
}

/**
 * The scene, live.
 *
 * `session` is a `lib/sessions.js` record — it owns the transcript, so leaving
 * this tab and coming back shows the exchange you left rather than an empty
 * room. `onEnded` fires when the daemon says the session is over (it went
 * quiet, or somebody ended it elsewhere), so the caller can put its presence
 * control back to "not here".
 */
export function scene({ npc, session, who, commands, onEnded }) {
  const stageInner = h('div', { class: 'stage-inner' });
  const stage = h('div', { class: 'lane' },
    h('div', { class: 'pane-hd' }, 'what you see'), stageInner);
  const actsInner = h('div', { class: 'acts-inner' });
  const actsPane = h('div', { class: 'lane acts' },
    h('div', { class: 'pane-hd' }, 'what they do'), actsInner);
  let ended = false;

  // Follow the tail, but yield the instant the reader scrolls up — and never
  // let a scroll we caused read as them taking over (lib/live.js).
  const scroller = sticky(stage);
  const scrollDown = () => scroller.follow();

  /* act_id → the nodes that act put on the page, so a rendering can replace its
   * text in place if one is ever sent. */
  const actNodes = new Map();

  // ── composer ──────────────────────────────────────────────────────────────

  const input = h('textarea', {
    class: 'textarea', rows: 1,
    placeholder: 'Say something, or / for commands',
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
    if (!pal.matches.length) {
      mount(palette, h('div', { class: 'opt' }, h('span', { class: 'ds' }, 'no command matches')));
    }
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

  /* One injection, and whether it landed. The line is painted only on `true`,
   * so a refusal is visible as the message not appearing rather than as words
   * on the stage the character never heard. */
  async function deliver(payload) {
    try {
      await API.inject(session.ix, payload);
      return true;
    } catch (e) {
      toast(e.detail || e.message || 'the daemon did not take that', 'err');
      return false;
    }
  }

  /* **The raw line goes to the daemon, whatever it is.**
   *
   * It used to post `{command, args}` for a slash line — which `inject` does
   * not read at all; it reads `line`/`text` and parses the grammar itself. So
   * every slash command came back `empty_line` and nothing reached the
   * character. The parse here is for the palette, the hints and the validation;
   * `engine::slash` is the authority on what a command *means*, and there is
   * no second grammar for the two to disagree about.
   */
  async function send() {
    const line = input.value.trim();
    if (!line || ended) return;

    let mine = { t: 'mine', text: line, kind: 'said' };
    if (line.startsWith('/')) {
      const p = parseLine(line, commands);
      if (!p.command) return toast('Unknown command', 'err');
      if (!p.complete) {
        return toast('/' + p.command.name + ' needs ' + p.command.argument, 'err');
      }
      /* Your own words on the stage, not the command you typed them behind —
       * and an act of yours reads the way the character's does, because it is
       * the same thing happening in the same room from the other side. */
      const words = p.args[FREE_TEXT] || '';
      const kind = { say: 'said', act: 'contact' }[p.command.name];
      mine = {
        t: 'mine',
        text: kind ? words : line,
        kind: kind || 'cue',
      };
    }

    if (!(await deliver({ text: line }))) return;
    add(mine);
    input.value = '';
    input.style.height = 'auto';
    closePalette();
  }

  // ── the transcript ────────────────────────────────────────────────────────

  /* Every line goes into the session's log first and is painted from it, so the
   * page can be rebuilt from the log alone — which is what makes leaving this
   * tab and coming back show the exchange you left. */
  function add(entry) {
    sessions.append(session, entry);
    paint(entry);
    scrollDown();
  }

  /* Somebody said something: a railed turn, the speaker named on the line. The
   * words go in their own span so a rendering can replace them without taking
   * the speaker's name with it. */
  const said = (speaker, text, cls) => h('div', { class: 'beat said ' + (cls || '') },
    h('div', { class: 'rail-mark' }),
    h('div', { class: 'txt' },
      h('span', { class: 'speaker' }, speaker),
      h('span', { class: 'words' }, text)),
    h('div', { class: 'tag' }, cls === 'me' ? 'you' : 'said'));

  function paint(e) {
    if (e.t === 'mine') {
      if (e.kind === 'said') {
        stageInner.appendChild(said(who, e.text, 'me'));
      } else if (e.kind === 'contact') {
        stageInner.appendChild(h('div', { class: 'beat contact me' },
          h('div', { class: 'rail-mark' }),
          h('div', { class: 'txt' },
            h('span', { class: 'speaker' }, who),
            h('span', { class: 'words' }, e.text)),
          h('div', { class: 'tag' }, 'you')));
      } else {
        stageInner.appendChild(h('div', { class: 'beat ' + e.kind },
          h('div', { class: 'rail-mark' }),
          h('div', { class: 'txt' }, e.text),
          h('div', { class: 'tag' }, 'you')));
      }
      return;
    }
    if (e.t === 'act') return paintAct(e);
    if (e.t === 'ended') {
      stageInner.appendChild(h('div', { class: 'scene-note' }, e.text));
    }
  }

  function paintAct(e) {
    // Right lane: every act, one line — tick, tool, intent — the way the front
    // page's sample renders them. An act with no observable trace says so in
    // the intent slot rather than adding a fourth element, because that *is*
    // what it did.
    const parts = actParts(e.tool, e.intent);
    actsInner.appendChild(h('div', { class: 'act-item' + (e.observable ? '' : ' unobs') },
      h('div', { class: 'row' },
        h('span', { class: 'tk' }, 't' + e.tick),
        h('span', { class: 'tool' }, parts.tool),
        h('span', { class: 'intent' },
          e.observable ? parts.intent : 'no observable trace'))));
    actsPane.scrollTop = actsPane.scrollHeight;

    /* Left lane: **what a body in the room would register.**
     *
     * Words and hands, and nothing else. It used to carry every act as a railed
     * beat with the tool as its tag, which is the right lane again one column
     * over in a different shape — two renderings of one list is the same view
     * twice, and it buried the exchange under the character's errands.
     *
     * But cutting it back to speech alone went too far the other way: somebody
     * taking hold of you is not a line in a log, it is the thing that just
     * happened to you, and it was appearing only on the right.
     *
     * An act with no observable trace is never here either way — that is the
     * character thinking, and you are standing in a room, not in its head. */
    if (!e.observable) return;
    const text = e.rendered || e.intent || e.tool;
    let node = null;
    if (SPOKEN.has(e.tool)) {
      node = said(npc.name, text, '');
    } else if (IN_THE_ROOM.has(e.tool)) {
      node = h('div', { class: 'beat contact' },
        h('div', { class: 'rail-mark' }),
        h('div', { class: 'txt' },
          h('span', { class: 'speaker' }, npc.name),
          h('span', { class: 'words' }, text)),
        h('div', { class: 'tag' }, e.tool));
    } else {
      return;
    }
    actNodes.set(e.act_id, { entry: e, words: node.querySelector('.words') });
    stageInner.appendChild(node);
  }

  // ── the stream ────────────────────────────────────────────────────────────

  /* **What happened before this page was watching.**
   *
   * The transcript is held by the browser, so a reload empties it — and the
   * conversation is not over, it is standing in a room somewhere with a body in
   * it. Coming back to a blank stage after an exchange reads as the exchange
   * having been lost, when the daemon has it the whole time.
   *
   * So an empty transcript is filled from the character's own tick feed before
   * the stream attaches. `act_id` is minted the way the daemon mints it, so the
   * live stream recognises anything it hands back rather than painting it
   * twice, and `onAct` sets `lastTick` — which is what the stream then resumes
   * from, so the two meet exactly.
   *
   * **Only the character's side comes back.** Your own lines were injected
   * events rather than acts, and nothing structured records them; they survive
   * in what the character perceived, as prose. Reconstructing them by matching
   * on that prose would be inventing a transcript rather than replaying one, so
   * the stage says what it is showing instead.
   */
  async function backfill() {
    if (session.log.length) return;
    const p = await API.pulse({ limit: 40, npc_id: npc.npc_id }).catch(() => null);
    const ticks = (p?.ticks || []).filter((t) => String(t.npc_id) === String(npc.npc_id));
    const before = session.log.length;
    for (const t of ticks) {
      (t.acts || []).forEach((a, n) => {
        const parts = actParts(a, '');
        onAct({
          act_id: `${t.tick}-${n}`,
          tick: t.tick,
          tool: parts.tool || a,
          intent: parts.intent,
        });
      });
    }
    if (session.log.length > before) {
      stageInner.insertBefore(
        h('div', { class: 'scene-note' }, 'what they did before you were watching'),
        stageInner.firstChild,
      );
    }
  }

  function onAct(a) {
    // The stream resumes one tick behind what this page already read, so the
    // tail of that tick comes back and has to be recognised rather than painted
    // twice.
    if (session.seen.has(a.act_id)) return;
    session.seen.add(a.act_id);
    session.lastTick = Math.max(session.lastTick, a.tick - 1);
    add({
      t: 'act',
      act_id: a.act_id,
      tick: a.tick,
      tool: a.tool,
      intent: a.intent,
      observable: !a.observable_in || a.observable_in.includes('physical'),
      rendered: '',
    });
  }

  // A rendering REPLACES text already on the page, so it is the one path here
  // that can destroy DOM the reader is mid-selection on. Queue it and flush only
  // while no selection lives in the stage — nothing is lost, the surface just
  // pauses while you copy.
  const renderQueue = new Map();
  const flushRendered = throttled(() => {
    if (!renderQueue.size) return;
    if (selectionInside(stage)) return;
    for (const [actId, text] of renderQueue) {
      const rec = actNodes.get(actId);
      if (!rec) continue;
      rec.entry.rendered = text;      // so a repaint keeps it
      if (rec.words) mount(rec.words, text);
    }
    renderQueue.clear();
    scrollDown();
  });

  function onActRendered(r) {
    if (!actNodes.has(r.act_id)) return;
    renderQueue.set(r.act_id, r.rendered?.text || '');
    flushRendered();
  }

  document.addEventListener('selectionchange', flushRendered);

  /* No `onTick`. The stream sends one at the close of every tick and it used to
   * put a `tick 412 · 1 acts` rule across the scene — the machine's own beat,
   * interleaved with what people said, on the one surface that is meant to read
   * as a room. Which tick an act belongs to is on its row in the right lane,
   * where a tick number is a thing somebody might want. */

  /* The daemon says it is over — it went quiet, or somebody ended it elsewhere.
   * Say so on the stage, stop taking lines, and tell the caller so its presence
   * control stops claiming you are in the room. */
  function onState(s) {
    if (s.state !== 'ended' || ended) return;
    ended = true;
    sessions.forget(session.ix);
    add({ t: 'ended', text: 'you are no longer in the room' });
    input.disabled = true;
    input.placeholder = 'you have left';
    if (onEnded) onEnded();
  }

  // ── assemble ──────────────────────────────────────────────────────────────

  // Everything this browser already had, before anything is fetched.
  for (const e of session.log) paint(e);

  const el = h('div', { class: 'lane-frame' },
    h('div', { class: 'lane-body' }, stage, actsPane),
    composer);

  /* The stream attaches *after* the backfill, so it resumes from the last tick
   * the backfill brought in rather than from wherever the character has got to
   * — otherwise the two would meet with a hole between them. A conversation
   * this browser already has a transcript for skips the fetch entirely. */
  let stream = null;
  let gone = false;
  backfill().finally(() => {
    if (!session.log.length) {
      stageInner.appendChild(h('div', { class: 'tiny dim', style: 'padding:6px 0' },
        `You are standing with ${npc.name}. Say something, and it answers on its own turn.`));
    }
    // Cancelled before it opened: the tab was left while the backfill was in
    // flight, and an unattended stream holds the session alive.
    if (gone) return;
    stream = API.streamInteraction(session.ix, {
      onAct, onActRendered, onState,
      onError: () => toast('stream closed', 'err'),
    }, session.lastTick || null);
    scroller.follow();
  });

  requestAnimationFrame(() => { scroller.follow(); input.focus(); });

  return {
    el,
    /* Detaching the view does NOT leave the room. Presence is world state — the
     * caller's Leave control is what walks you out, and the daemon's idle
     * timeout is the backstop for a browser that stopped watching. */
    teardown: () => {
      gone = true;
      if (stream) stream.cancel();
      document.removeEventListener('selectionchange', flushRendered);
    },
  };
}

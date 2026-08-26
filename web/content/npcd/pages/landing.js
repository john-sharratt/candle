/* The public front page (§24).
 *
 * The demo is the pitch. A screenshot of a chat window looks like every other
 * product; an NPC acting *before* it explains itself — the act stream moving
 * while narration is still assembling — cannot be faked by a wrapper around
 * someone else's API, and it is visible in about four seconds. */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';

const MARK = `<svg viewBox="0 0 32 32" width="100%" height="100%">
  <circle cx="16" cy="7.4" r="3.1" fill="currentColor"/>
  <rect x="9.5" y="13.4" width="13" height="3.1" rx="1.55" fill="currentColor" opacity=".92"/>
  <rect x="6.6" y="18.6" width="18.8" height="3.1" rx="1.55" fill="currentColor" opacity=".62"/>
  <rect x="4" y="23.8" width="24" height="3.1" rx="1.55" fill="currentColor" opacity=".34"/>
</svg>`;

export async function render() {
  const el = h('div', { class: 'landing' });
  const providers = (await API.getProviders().catch(() => ({ providers: [] }))).providers || [];

  // ── hero ──────────────────────────────────────────────────────────────────

  el.appendChild(h('section', { class: 'hero' },
    h('div', { class: 'hero-mark', html: MARK }),
    h('h1', {}, 'NPCs that ', h('em', {}, 'remember'), ' you.'),
    h('p', { class: 'lede' },
      'A mind per character. Years of lived history, convictions that hold under pressure, and a hundred of ' +
      'them thinking at once — on a single graphics card.'),
    h('div', { class: 'cta' },
      providers.length
        ? providers.map((p) => h('button', { class: 'btn primary lg', onClick: () => window.__npcdSignIn() },
          'Continue with ' + p.display))
        : h('button', { class: 'btn primary lg', onClick: () => window.__npcdSignIn() }, 'Get started'),
      h('a', { class: 'btn lg', href: '#demo' }, 'Watch one think ↓')),
    h('div', { class: 'tiny dim', style: 'margin-top:16px' },
      'Free while in preview · your characters stay yours')));

  // ── live demo ─────────────────────────────────────────────────────────────

  const acts = h('div', { class: 'demo-acts' });
  const narration = h('div', { class: 'demo-narration' },
    h('span', { class: 'dim tiny' }, 'waiting for the tick to close…'));

  el.appendChild(h('section', { class: 'demo', id: 'demo' },
    h('div', { class: 'demo-label' },
      h('span', { class: 'dot ticking' }), 'live — not a recording'),
    h('div', { class: 'demo-frame' },
      h('div', { class: 'demo-hd' },
        h('div', { class: 'avatar' }, 'V'),
        h('div', {},
          h('div', { style: 'font-weight:700' }, 'Varek'),
          h('div', { class: 'tiny dim' }, 'Loyal Soldier · eastern ridge, dusk')),
        h('span', { style: 'flex:1' }),
        h('span', { class: 'chip accent' }, 'physical encounter')),
      h('div', { class: 'demo-body' },
        h('div', { class: 'demo-col' },
          h('div', { class: 'demo-col-hd' }, 'what he does'),
          acts),
        h('div', { class: 'demo-col narrate' },
          h('div', { class: 'demo-col-hd' }, 'what you see'),
          narration))),
    h('p', { class: 'demo-caption' },
      'The left column is the character acting. The right is the narrator explaining it, one beat later. ',
      h('strong', {}, 'He moves before he can tell you why'), ' — because the mind decides, and the surface ' +
      'only ever reports what it actually did.')));

  // ── the three claims ──────────────────────────────────────────────────────

  el.appendChild(h('section', { class: 'feat' },
    feature('Unbounded memory',
      'Not a bigger context window — a different shape. Provenance-selected attention over a three-tier paged ' +
      'substrate, so error per step stays flat whether the character is an hour old or a year old.',
      'O(1) error at any depth'),
    feature('Convictions that hold',
      'A belief moves only when disconfirming evidence accumulates past a threshold. Your character cannot be ' +
      'argued out of who it is in a single clever message, because the action layer physically cannot write there.',
      'evidence, not persuasion'),
    feature('One card, many minds',
      'Characters share an immutable prefix and diverge only at the suffix. The popular one is the cheapest, ' +
      'not the most expensive — 64 concurrent sessions on 16 GB.',
      '2,446 tok/s aggregate')));

  // ── how it works ──────────────────────────────────────────────────────────

  el.appendChild(h('section', { class: 'how' },
    h('h2', { class: 'section-title' }, 'How a character thinks'),
    h('div', { class: 'how-grid' },
      step('1', 'Perceive', 'The world pushes batched events — descriptions, ascii maps at four zoom bands, ' +
        'entity sightings. Cheap, because perception is prefill.'),
      step('2', 'Gather', 'Under a token budget, provenance pulls the currently-relevant blocks from nine ' +
        'layers: perception, action, agency, relationships, beliefs, memory, world.'),
      step('3', 'Act', 'One cognitive step. The character emits *intent* — never words — as tool calls that ' +
        'commit to the act stream.'),
      step('4', 'Narrate', 'A separate pass renders those acts into prose, scoped to what your vantage can ' +
        'actually observe. A voice call cannot see him turn his head.'))));

  // ── what you get ──────────────────────────────────────────────────────────

  el.appendChild(h('section', { class: 'how' },
    h('h2', { class: 'section-title' }, 'Built to be inspected'),
    h('div', { class: 'how-grid' },
      step('◱', 'Projection inspector', 'For any tick: what was gathered, what was dropped, and why. Dropped ' +
        'turns are first-class — the interesting question is usually what nearly made it.'),
      step('◲', 'Belief pressure', 'Watch a conviction come under strain in real time, with its ' +
        'disconfirmation bar filling toward the threshold that will rewrite it.'),
      step('◳', 'Metacognition monitor', 'Overlap between what a character says and what it holds. Rising ' +
        'overlap means it is reading its own output as fresh signal — visible before it becomes incoherence.'),
      step('◰', 'Full API', 'Everything the interface can do, the API can do. The GUI is just a client — ' +
        'there is no privileged path.'))));

  // ── closing ───────────────────────────────────────────────────────────────

  el.appendChild(h('section', { class: 'closer' },
    h('h2', {}, 'Give a character a year of memory.'),
    h('p', { class: 'lede' }, 'Then see whether it still recognises you.'),
    h('div', { class: 'cta' },
      h('button', { class: 'btn primary lg', onClick: () => window.__npcdSignIn() }, 'Start building')),
    h('div', { class: 'foot tiny dim' },
      'npcd · the NPC engine behind Battle Cities · runs headless on your own hardware')));

  // ── drive the demo ────────────────────────────────────────────────────────

  const stream = API.streamInteraction('demo', {
    onAct: (a) => {
      const hidden = a.observable_in && a.observable_in.length === 0;
      acts.appendChild(h('div', { class: 'demo-act' + (hidden ? ' hidden-act' : '') },
        h('span', { class: 'tk' }, 't' + a.tick),
        h('span', { class: 'tool' }, a.tool),
        h('span', { class: 'intent' }, hidden ? 'no observable trace' : (a.intent || ''))));
      while (acts.children.length > 7) acts.removeChild(acts.firstChild);
    },
    onNarration: (n) => {
      mount(narration, n.text);
      narration.animate?.([{ opacity: 0.25 }, { opacity: 1 }], { duration: 420, easing: 'ease-out' });
    },
    onTick: () => {},
  });

  return { el, teardown: () => stream.cancel() };
}

function feature(title, body, stat) {
  return h('div', { class: 'feat-card' },
    h('div', { class: 'feat-stat' }, stat),
    h('h3', {}, title),
    h('p', {}, body));
}

function step(n, title, body) {
  return h('div', { class: 'how-step' },
    h('div', { class: 'how-n' }, n),
    h('div', {}, h('h3', {}, title), h('p', {}, body)));
}

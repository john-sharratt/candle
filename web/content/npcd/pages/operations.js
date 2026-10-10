/* Operations — the objectives the command table holds.
 *
 * Each operation is one piece of the world's storyline carried through a
 * workflow from the mind's missions.yaml: steps taken by Makers and by the
 * table, each step's outcome choosing the next, until one leads to done or
 * failed. Nothing stands in the record until its workflow says it does.
 *
 * Running operations are shown open, with where each stands in its workflow
 * and who is carrying it; finished ones fold away below. An operation can be
 * renamed, its objective restated, the brief of its waiting mission rewritten,
 * sent to any step of its workflow, or called off. The page polls, and holds
 * its paint while you are editing. */

import { API } from '../lib/api.js';
import { h, mount, fmtNum } from '../lib/dom.js';
import { disclosure } from '../lib/lazy.js';

/* How an operation ended. */
const ENDS = {
  succeeded: ['Passed', 'ok'],
  failed: ['Failed', 'warn'],
  cancelled: ['Called off', ''],
};

export async function render(_params, _q) {
  const el = h('div', { class: 'page wide ops' });
  const kpiHost = h('div', { class: 'grid g4', style: 'margin-bottom:16px' });
  const activeHost = h('div', {});
  const doneHost = h('div', { style: 'margin-top:20px' });
  const note = h('span', { class: 'tiny dim' });
  let editing = 0; // forms open; the poll does not repaint under them
  /* What the reader has opened — finished cards and the sections inside any
   * card — by operation, so the poll's repaint keeps them open rather than
   * folding them shut every five seconds while they are being read. */
  const opened = new Set();
  const keyOf = (world, op) => `${world}#${op.id}`;

  /* A <details> section that stays as the reader left it across repaints. */
  function section(key, title, text) {
    const d = h('details', { open: opened.has(key) || null },
      h('summary', { class: 'tiny' }, title),
      h('pre', { class: 'disc-pre' }, text));
    d.addEventListener('toggle', () => (d.open ? opened.add(key) : opened.delete(key)));
    return d;
  }

  const reviewPath = h('input', {
    class: 'input', placeholder: 'layers/stories/the-ledger.md', style: 'width:22em',
    'aria-label': 'document to put through review',
  });
  const reviewBtn = h('button', {
    class: 'btn sm',
    onClick: async () => {
      const path = reviewPath.value.trim();
      if (!path) return;
      try {
        const r = await API.reviewDocument(path);
        say(`${path}: ${(r.opened || []).length} operation(s) opened.`);
        reviewPath.value = '';
        paint();
      } catch (e) { say(e.message || String(e), true); }
    },
  }, 'Put through review');
  const openBtn = h('button', {
    class: 'btn sm ghost',
    onClick: async () => {
      try {
        const r = await API.commandTable(true);
        say(`The table is open; ${r.called} called to it.`);
      } catch (e) { say(e.message || String(e), true); }
    },
  }, 'Open the table');

  function say(text, bad) {
    note.textContent = text;
    note.className = 'tiny ' + (bad ? 'st-warn' : 'dim');
  }

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Operations'),
      h('div', { class: 'sub' },
        'What the command table is having written. Each operation runs a workflow from the '
        + "mind's missions.yaml — written, read by the table, reviewed and checked by other "
        + 'Makers — before it stands in the record.')),
    h('div', { class: 'row wrap', style: 'gap:6px' }, openBtn, reviewPath, reviewBtn)));
  el.appendChild(h('div', { style: 'margin:-6px 0 12px' }, note));
  el.appendChild(kpiHost);
  el.appendChild(activeHost);
  el.appendChild(doneHost);

  function stat(lbl, val) {
    return h('div', { class: 'panel stat' },
      h('div', { class: 'lbl' }, lbl),
      h('div', { class: 'val' }, fmtNum(val)));
  }

  /* Where the operation stands: its workflow's steps left to right, those
   * taken this far marked, the one it waits on lit. */
  function chain(op) {
    const taken = new Set((op.history || []).map((t) => t.step));
    const steps = (op.steps || []).map((s) => {
      const cls = s.name === op.step ? 'chip accent' : taken.has(s.name) ? 'chip ok' : 'chip';
      return h('span', { class: cls, title: s.table ? 'the table takes it' : 'a Maker takes it' }, s.name);
    });
    if (op.finished) {
      const [label, cls] = ENDS[op.state] || [op.state, ''];
      steps.push(h('span', { class: 'chip ' + cls }, label));
    }
    return h('div', { class: 'row wrap', style: 'gap:4px' },
      h('span', { class: 'tiny dim mono', style: 'margin-right:4px' }, op.workflow),
      ...steps.flatMap((s, i) => (i ? [h('span', { class: 'tiny dim' }, '→'), s] : [s])));
  }

  function who(op) {
    const name = (n, id) => n || id || '—';
    // Each Maker who took a step, once, in the order they came to it.
    const makers = [];
    for (const t of op.history || []) {
      if (t.by !== 'table' && !makers.some((m) => m.by === t.by)) makers.push(t);
    }
    return h('div', { class: 'row wrap tiny', style: 'gap:12px;margin-top:6px' },
      ...makers.map((t) => h('span', {}, `${t.step} by `, h('b', {}, name(t.by_name, t.by)))),
      op.round ? h('span', { class: 'dim' }, `round ${op.round + 1} · sent back ${op.send_backs}×`) : null,
      op.carrying ? h('span', { class: 'chip accent' }, 'carried now by ' + name(op.carrying_name, op.carrying)) : null,
      op.waiting_brief ? h('span', { class: 'chip' }, 'waiting at the table') : null);
  }

  /* The document an operation wrote, read from wherever it is now — the record,
   * or where its rejection moved it. Fetched once when first opened and kept,
   * so the poll's repaint shows it again without asking for it again. */
  const docs = new Map();
  function docView(world, op) {
    const key = `${keyOf(world, op)}:doc`;
    const panel = h('div', { style: 'margin:6px 0' });
    const show = (d) => mount(panel,
      d.error
        ? h('div', { class: 'tiny st-warn' }, d.error)
        : h('div', {},
          h('div', { class: 'tiny dim', style: 'margin-bottom:4px' },
            `${d.where === 'rejected' ? 'Rejected — ' : 'On the record — '}${d.path} · ${fmtNum(d.words)} words `,
            h('button', {
              class: 'btn sm ghost',
              onClick: () => { docs.delete(key); load(); },
            }, 'Reload')),
          h('pre', { class: 'disc-pre', style: 'white-space:pre-wrap;max-height:32em;overflow:auto' }, d.text)));
    const load = async () => {
      mount(panel, h('div', { class: 'tiny dim' }, 'Reading…'));
      let d;
      try {
        d = await API.operationDocument(world, op.id);
      } catch (e) {
        d = { error: e.message || String(e) };
      }
      docs.set(key, d);
      if (opened.has(key)) show(d);
    };
    const toggle = h('button', { class: 'btn sm ghost' }, opened.has(key) ? 'Hide' : 'View');
    toggle.addEventListener('click', () => {
      if (opened.has(key)) {
        opened.delete(key);
        toggle.textContent = 'View';
        mount(panel);
      } else {
        opened.add(key);
        toggle.textContent = 'Hide';
        if (docs.has(key)) show(docs.get(key)); else load();
      }
    });
    if (opened.has(key)) {
      if (docs.has(key)) show(docs.get(key)); else load();
    }
    return { toggle, panel };
  }

  function details(world, op) {
    const log = (op.history || []).map((t) => h('li', {},
      h('span', { class: 'mono tiny' },
        `${t.step} · ${t.by_name || t.by}${t.outcome ? ' · ' + t.outcome : ''} → ${t.to}`),
      t.notes ? h('div', { class: 'tiny dim', style: 'white-space:pre-wrap' }, t.notes) : null));
    const doc = op.document ? docView(world, op) : null;
    return h('div', {},
      h('div', { class: 'tiny row wrap', style: 'margin:6px 0;gap:6px;align-items:center' }, 'Document ',
        h('code', { class: 'mono' }, op.document || '—'),
        doc ? doc.toggle : null,
        h('span', { class: 'tiny dim' }, ` · ${op.generator} · ${op.target}`)),
      doc ? doc.panel : null,
      op.why ? h('div', { class: 'tiny st-warn', style: 'margin:6px 0' }, op.why) : null,
      op.waiting_brief
        ? section(`${keyOf(world, op)}:brief`, 'The waiting brief', op.waiting_brief)
        : null,
      log.length ? h('ol', { class: 'ops-log', style: 'margin:8px 0;padding-left:18px' }, ...log) : null,
      actions(world, op));
  }

  /* Send it to a step of its workflow — read again by the table, checked
   * again against the storyline — while nobody carries its step. */
  function sendTo(world, op) {
    if (op.carrying || !(op.steps || []).length) return null;
    const pick = h('select', { class: 'input', 'aria-label': 'step to send it to' },
      ...op.steps.map((s) => h('option', { value: s.name, selected: s.name === op.step || null }, s.name)));
    return h('span', { class: 'row', style: 'gap:4px' },
      pick,
      h('button', {
        class: 'btn sm ghost',
        title: 'Send it to this step of its workflow, in a new round',
        onClick: async () => {
          try {
            await API.stepOperation(world, op.id, pick.value);
            say(`${op.name} goes to ${pick.value}.`);
            paint();
          } catch (e) { say(e.message || String(e), true); }
        },
      }, 'Send to step'));
  }

  /* Edit and call off — the form replaces the buttons while it is open. */
  function actions(world, op) {
    const host = h('div', { class: 'row wrap', style: 'gap:6px;margin-top:8px' });
    const buttons = () => mount(host,
      h('button', { class: 'btn sm ghost', onClick: form }, 'Edit'),
      sendTo(world, op),
      op.finished ? null : h('button', {
        class: 'btn sm ghost',
        onClick: async () => {
          try {
            await API.cancelOperation(world, op.id);
            say(`${op.name} called off.`);
            paint();
          } catch (e) { say(e.message || String(e), true); }
        },
      }, 'Call off'));
    const form = () => {
      editing += 1;
      const name = h('input', { class: 'input', value: op.name, 'aria-label': 'name' });
      const objective = h('input', { class: 'input', value: op.objective, 'aria-label': 'objective', style: 'width:100%' });
      const brief = op.waiting_brief != null
        ? h('textarea', { class: 'input mono', rows: 8, style: 'width:100%', 'aria-label': 'brief' }, op.waiting_brief)
        : null;
      const done = () => { editing = Math.max(0, editing - 1); buttons(); };
      mount(host, h('div', { style: 'display:grid;gap:6px;width:100%' },
        h('label', { class: 'tiny dim' }, 'Name'), name,
        h('label', { class: 'tiny dim' }, 'Objective'), objective,
        brief ? h('label', { class: 'tiny dim' }, 'Brief of the mission waiting at the table') : null, brief,
        h('div', { class: 'row', style: 'gap:6px' },
          h('button', {
            class: 'btn sm',
            onClick: async () => {
              const body = {};
              if (name.value.trim() !== op.name) body.name = name.value;
              if (objective.value.trim() !== op.objective) body.objective = objective.value;
              if (brief && brief.value !== op.waiting_brief) body.brief = brief.value;
              try {
                if (Object.keys(body).length) await API.editOperation(world, op.id, body);
                done();
                say(`${body.name || op.name} saved.`);
                paint();
              } catch (e) { say(e.message || String(e), true); }
            },
          }, 'Save'),
          h('button', { class: 'btn sm ghost', onClick: done }, 'Cancel'))));
    };
    buttons();
    return host;
  }

  function activeCard(world, op) {
    return h('div', { class: 'panel', style: 'margin-bottom:12px' },
      h('div', { class: 'row', style: 'justify-content:space-between;align-items:baseline;gap:8px' },
        h('h3', { style: 'margin:0' }, op.name),
        h('span', { class: 'tiny dim mono' }, `${world} · #${op.id}`)),
      h('div', { style: 'margin:4px 0 8px' }, op.objective),
      chain(op),
      who(op),
      details(world, op));
  }

  function finishedCard(world, op) {
    const [label, cls] = ENDS[op.state] || [op.state, ''];
    const key = keyOf(world, op);
    const card = disclosure({
      dense: true,
      open: opened.has(key),
      head: [
        h('span', { class: 'disc-title' }, op.name),
        h('div', { class: 'disc-meta' },
          h('span', { class: 'chip ' + cls }, label),
          h('span', { class: 'tiny dim' }, op.objective)),
      ],
      body: (host) => mount(host, chain(op), who(op), details(world, op)),
    });
    // After the disclosure's own toggle, which is registered first: record
    // where the reader left it.
    card.querySelector('.disc-hd').addEventListener('click', () =>
      (card.hasAttribute('data-open') ? opened.add(key) : opened.delete(key)));
    return card;
  }

  async function paint() {
    let data;
    try {
      data = await API.operations();
    } catch (e) {
      mount(activeHost, h('div', { class: 'lazy-err' }, 'Could not read the operations: ' + (e.message || e)));
      return;
    }
    const all = Object.entries(data.worlds || {}).flatMap(([w, ops]) => ops.map((op) => [w, op]));
    const active = all.filter(([, o]) => !o.finished);
    const done = all.filter(([, o]) => o.finished);
    const count = (state) => done.filter(([, o]) => o.state === state).length;
    mount(kpiHost,
      stat('running', active.length),
      stat('passed', count('succeeded')),
      stat('failed', count('failed')),
      stat('called off', count('cancelled')));
    mount(activeHost,
      h('h2', {}, 'Running'),
      active.length
        ? active.map(([w, o]) => activeCard(w, o))
        : h('div', { class: 'tiny dim' }, 'Nothing is running. Open the table and the generator will set work.'));
    mount(doneHost,
      h('h2', {}, 'Finished'),
      done.length
        ? done.map(([w, o]) => finishedCard(w, o))
        : h('div', { class: 'tiny dim' }, 'No operation has finished yet.'));
  }

  await paint();
  const timer = setInterval(() => { if (!editing) paint(); }, 5000);
  return { el, teardown: () => clearInterval(timer) };
}

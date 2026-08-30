/* Tools (§38). Calibration state is surfaced honestly — a silently
 * uncalibrated tool is a mysteriously bad NPC. */

import { API } from '../lib/api.js';
import { h } from '../lib/dom.js';
import { toast, modal, only, empty } from '../lib/ui.js';

const MODE_SHORT = { physical: 'phys', video_call: 'video', voice_call: 'voice', instant_message: 'im' };

export async function render() {
  const el = h('div', { class: 'page', style: 'max-width:1100px' });

  /* A failure is reported, not defaulted.
   *
   * This was `.catch(() => ({ tools: [], uncalibrated: 0 }))`, and zero
   * uncalibrated tools is exactly the value that renders a green **all
   * calibrated** chip. So a daemon that could not be reached at all came back
   * as the reassuring answer, over an empty table. */
  let r;
  try {
    r = await API.listTools();
  } catch (e) {
    el.appendChild(h('div', { class: 'hd' }, h('div', {}, h('h1', {}, 'Tools'))));
    el.appendChild(h('div', { class: 'panel' },
      empty('⊘', 'The tool catalog could not be read',
        e.detail || e.message || 'the daemon did not answer')));
    return el;
  }

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Tools'),
      h('div', { class: 'sub' },
        'The act vocabulary. Every tool carries intent, not output — the narrator renders the words.')),
    h('div', { class: 'row' },
      // Reading the catalog is every signed-in operator's; running a
      // calibration pass is not. It is a daemon-wide side effect that changes
      // how every character on this machine selects a tool — the only write on
      // this page that is not scoped to the caller's own characters, which is
      // what puts it with the admin controls rather than beside them.
      /* `null` means nothing has counted, and there is no tool registry to
       * count — the engine registers tools with the layers each may write. Only
       * a real number gets a chip; the empty-state panel below says why the
       * table is empty. */
      r.uncalibrated == null
        ? null
        : r.uncalibrated
        ? (only('admin', () => h('button', {
          class: 'btn primary',
          onClick: async () => {
            // Awaited and reported. It used to toast success whatever came
            // back, which is the same failure as the chip above.
            try {
              await API.calibrateTools();
              toast('calibration pass queued', 'ok');
            } catch (e) {
              toast(e.detail || e.message || 'could not queue a calibration pass', 'err');
            }
          },
        }, `Calibrate ${r.uncalibrated} tool${r.uncalibrated === 1 ? '' : 's'}`))
          || h('span', { class: 'chip warn', title: 'calibration is an admin’s to run' },
            `${r.uncalibrated} uncalibrated`))
        : h('span', { class: 'chip ok' }, 'all calibrated'))));

  const groups = new Map();
  for (const t of r.tools) {
    if (!groups.has(t.category)) groups.set(t.category, []);
    groups.get(t.category).push(t);
  }

  // An empty catalog with a reason. Without this the page is a heading over
  // nothing, which reads as a daemon that lost its tools rather than one that
  // has never had any.
  if (!r.tools.length) {
    el.appendChild(h('div', { class: 'panel' },
      empty('◌', 'No tools registered',
        r.engine_connected === false
          ? 'Tools are registered by the engine, with the layers each one may write. This daemon '
            + 'is not running one, so the catalog is empty and there is nothing to calibrate.'
          : 'This daemon has registered no tools.')));
  }

  for (const [cat, ts] of groups) {
    el.appendChild(h('h2', {}, cat));
    el.appendChild(h('div', { class: 'list' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          ['Tool', 'Description', 'Modes', 'Source', 'Calibrated'].map((x) => h('th', {}, x)))),
        h('tbody', {}, ts.map((t) => h('tr', {
          class: 'click',
          onClick: () => modal({
            title: '/' + t.name,
            body: h('div', {},
              h('p', { style: 'color:var(--ink-soft)' }, t.description),
              h('h3', {}, 'Parameters — the schema the model actually sees'),
              h('pre', {
                class: 'mono',
                style: 'background:var(--bg-deep);border:1px solid var(--line);border-radius:8px;' +
                  'padding:11px;overflow:auto;font-size:.75rem',
              }, JSON.stringify(t.parameters || { type: 'object', properties: {} }, null, 2)),
              h('div', { class: 'tiny dim' },
                'Derived from the Rust request type by schemars — never hand-written, so the prompt and the parser cannot disagree.')),
          }),
        },
          h('td', {}, h('code', { class: 'mono', style: 'color:var(--accent)' }, t.name)),
          h('td', { class: 'tiny', style: 'color:var(--ink-soft)' }, t.description),
          h('td', {}, h('div', { class: 'row wrap', style: 'gap:4px' },
            (t.modes || []).length === 4
              ? h('span', { class: 'chip' }, 'all')
              : (t.modes || []).map((m) => h('span', { class: 'chip accent' }, MODE_SHORT[m] || m)))),
          h('td', {}, h('span', { class: 'chip' + (t.source === 'extension' ? ' violet' : '') }, t.source)),
          h('td', {}, t.calibrated
            ? h('span', { class: 'chip ok' }, 'yes')
            : h('span', { class: 'chip warn' }, 'uncalibrated'))))))));
  }

  el.appendChild(h('div', { class: 'panel', style: 'margin-top:18px' },
    h('h3', { style: 'margin-top:0' }, 'Why calibration matters'),
    h('div', { class: 'tiny dim' },
      'Tool selection quality comes from calibration examples prefilled into a reserved layer at startup. ' +
      'A tool registered while the engine is live is usable but selects worse until the next pass. ' +
      'Extension tools are registered through the crate — a tool is a Rust closure with typed parameters and cannot be posted as JSON.')));

  return { el };
}

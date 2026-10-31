/* Performance — the engine's wave and pressure telemetry.
 *
 * The shared performance page (`common/lib/perf.js`), the one zend's console
 * mounts as Telemetry: there is one engine under every tenant, so there is one
 * page that reads it, drawn here in this console's theme. */

import { h } from '/common/lib/dom.js';
import { mountPerf } from '/common/lib/perf.js';

export async function render() {
  const el = h('div', { class: 'page wide' });
  const perf = mountPerf(el, { title: 'Performance' });
  return { el, teardown: perf.stop };
}

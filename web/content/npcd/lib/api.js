/* ============================================================================
 * npc-api — the single seam the UI talks to. It never calls fetch directly.
 *
 *   live (default) : the embedded daemon, whose /v1/* is itself mocked for now.
 *   mock           : ?mock=1 — a pure client-side store, so the GUI can be
 *                    developed and Playwright-driven with no daemon at all.
 *
 * Both satisfy the identical method contract (docs/npc_api_gui_design.md §41).
 * ========================================================================== */

import { LiveAPI } from './api.live.js';

let useMock = false;
try {
  useMock = new URLSearchParams(location.search).has('mock') || window.NPC_BACKEND === 'mock';
} catch (_) { /* non-browser */ }

/* The mock is fetched only when it is the one being used.
 *
 * It is the largest file the console has — fifty-seven kilobytes of fixtures,
 * more than twice the shell — and a static `import` of it meant every real page
 * load downloaded and parsed the whole thing to reach a ternary that then chose
 * the live client. Nothing else changes: `?mock=1` is known before the first
 * request is made, so the decision is not deferred, only the download.
 *
 * Top-level `await` in a module, which is why this is a `.js` served as a
 * module and not a script. It suspends only in mock mode; the live path never
 * touches the promise. */
export const API = useMock ? (await import('./api.mock.js')).MockAPI : LiveAPI;
export const BACKEND = useMock ? 'mock' : 'live';
window.NpcAPI = API;
window.NPC_BACKEND_ACTIVE = BACKEND;

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
import { MockAPI } from './api.mock.js';

let useMock = false;
try {
  useMock = new URLSearchParams(location.search).has('mock') || window.NPC_BACKEND === 'mock';
} catch (_) { /* non-browser */ }

export const API = useMock ? MockAPI : LiveAPI;
export const BACKEND = useMock ? 'mock' : 'live';
window.NpcAPI = API;
window.NPC_BACKEND_ACTIVE = BACKEND;

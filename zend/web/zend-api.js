/* ============================================================================
 * zend-api.js — backend selector
 * ----------------------------------------------------------------------------
 * Picks the ZendAPI implementation at load time and exposes it as
 * window.ZendAPI — the single seam the UI talks to (it never calls fetch
 * directly). See docs/zend_ui_redesign.md §3.
 *
 *  - live  (default): the embedded daemon build.
 *  - mock           : ?mock=1 in the URL, or window.ZEND_BACKEND === 'mock'.
 *                     Used for standalone design iteration and the Phase-1
 *                     Playwright suite.
 *
 * Both implementations satisfy the identical method/event contract (§4).
 * ========================================================================== */
(function () {
  'use strict';

  let useMock = false;
  try {
    useMock = new URLSearchParams(location.search).has('mock')
           || window.ZEND_BACKEND === 'mock';
  } catch (_) { /* non-browser context — default to live */ }

  const impl = useMock ? window.ZendMockAPI : window.ZendLiveAPI;
  if (!impl) {
    throw new Error('zend-api.js: ' + (useMock ? 'ZendMockAPI' : 'ZendLiveAPI') +
      ' not loaded — check script order in index.html');
  }
  window.ZendAPI = impl;
  window.ZEND_BACKEND_ACTIVE = useMock ? 'mock' : 'live';
})();

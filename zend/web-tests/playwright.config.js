// Playwright config for the Zend UI suite.
// Phase 1: runs against the mock (?mock=1) served by server.js — no daemon.
// Phase 2: point BASE_URL at a live/stub daemon and drop ?mock=1 to re-run the
//          same specs as the contract-parity gate (docs/zend_ui_redesign.md §7).
const { defineConfig, devices } = require('@playwright/test');

const PORT = process.env.ZEND_WEB_PORT || 4321;

module.exports = defineConfig({
  testDir: '.',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: true,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: process.env.ZEND_BASE_URL || ('http://localhost:' + PORT),
    actionTimeout: 5000,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: process.env.ZEND_BASE_URL ? undefined : {
    command: 'node server.js',
    url: 'http://localhost:' + PORT + '/index.html',
    reuseExistingServer: true,
    timeout: 15000,
  },
});

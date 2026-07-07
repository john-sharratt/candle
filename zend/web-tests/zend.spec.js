// Zend UI acceptance suite (docs/zend_ui_redesign.md §6).
// Phase 1: runs against the mock (?mock=1). Phase 2: same specs re-run against a
// live/stub daemon (set ZEND_BASE_URL, drop ?mock) as the contract-parity gate.
const { test, expect } = require('@playwright/test');

const URL = '/index.html?mock=1';

// Boot the app. By design the home page no longer auto-selects a conversation;
// pass `{ conv: id }` to open one via the URL hash (the same path a reload uses to
// restore the chat the user had open).
async function boot(page, opts) {
  const conv = opts && opts.conv;
  await page.goto(conv ? (URL + '#conv=' + encodeURIComponent(conv)) : URL);
  await page.waitForFunction(() => window.__ZEND_READY__ === true);
}

// helper: synthesize a file drop onto the conversation drop zone
async function dropFiles(page, files) {
  await page.evaluate((files) => {
    const scroller = document.querySelector('.z-chatscroll');
    const zone = scroller ? scroller.parentElement : document.querySelector('#app');
    const dt = new DataTransfer();
    files.forEach((f) => {
      dt.items.add(new File([f.content || 'x'], f.name, { type: f.type || 'text/plain' }));
    });
    const fire = (type) => {
      const e = new DragEvent(type, { bubbles: true, cancelable: true });
      Object.defineProperty(e, 'dataTransfer', { value: dt });
      zone.dispatchEvent(e);
    };
    fire('dragenter'); fire('dragover'); fire('drop');
  }, files);
}

test.describe('1.0 smoke', () => {
  test('boots with mock backend and no console errors', async ({ page }) => {
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(String(e)));
    await boot(page);
    expect(await page.evaluate(() => window.ZEND_BACKEND_ACTIVE)).toBe('mock');
    expect(errors).toEqual([]);
  });
});

test.describe('1.0b home page on load', () => {
  test('reload with nothing open lands on the home page, not a chat', async ({ page }) => {
    await boot(page); // no hash → nothing auto-selected
    await expect(page.getByRole('heading', { name: /Good (Morning|Afternoon|Evening)\./ })).toBeVisible();
    await expect(page.locator('.zmd')).toHaveCount(0);
  });

  test('reload restores the conversation carried in the URL hash', async ({ page }) => {
    await boot(page, { conv: '1' }); // hash → restore that chat
    await expect(page.locator('.code-block').first()).toBeVisible();
    await expect(page.locator('.zmd').first()).toBeVisible();
  });
});

test.describe('1.1 sidebar', () => {
  test('collapsed rail by default; expands and collapses', async ({ page }) => {
    await boot(page);
    expect(await page.locator('.z-rail').count()).toBe(1);
    await page.getByTitle('Expand sidebar').click();
    await expect(page.locator('.z-sb')).toBeVisible();
    await page.getByTitle('Collapse sidebar').click();
    await expect(page.locator('.z-rail')).toBeVisible();
  });

  test('show-archived reveals archived rows; archive then restore', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Expand sidebar').click();
    // archived conv (#5) hidden by default
    await expect(page.getByText('Scratch notes on WS reconnect backoff')).toHaveCount(0);
    await page.getByText('Show archived').click();
    await expect(page.getByText('Scratch notes on WS reconnect backoff')).toBeVisible();
    // restore it
    await page.getByTitle('Restore conversation').first().click();
    // archive a live one
    await page.getByTitle('Archive conversation').first().click();
  });

  test('selecting a conversation hydrates it', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Expand sidebar').click();
    await page.getByText('Explain the tokenizer ChatML decoder').click();
    // lazy hydrate fills an assistant bubble
    await expect(page.locator('.zmd').first()).toBeVisible();
  });
});

test.describe('1.2 chat rendering', () => {
  test('renders seeded markdown, code block, think + tool card', async ({ page }) => {
    await boot(page, { conv: '1' });
    await expect(page.locator('.code-block').first()).toBeVisible();
    await expect(page.locator('.code-lang').first()).toHaveText(/RUST/i);
    await expect(page.locator('.tool-call-card').first()).toBeVisible();
    await expect(page.locator('details.think-block').first()).toBeVisible();
  });
  test('code copy button copies', async ({ page }) => {
    await boot(page, { conv: '1' });
    await page.locator('.copy-btn').first().click();
    await expect(page.locator('.copy-btn').first()).toHaveText(/Copied/);
  });
});

test.describe('1.3 streaming', () => {
  test('sending streams tokens into a new assistant bubble', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Expand sidebar').click();
    await page.getByText('Why is decode latency spiking under load?').click();
    const ta = page.locator('#zend-prompt');
    await ta.fill('Give me the short version');
    await ta.press('Enter');
    const last = page.locator('[data-msg]').last().locator('.zmd');
    await expect(last).toContainText('redo log', { timeout: 10000 });
  });
});

test.describe('1.4 thinking block', () => {
  test('think block is collapsed by default', async ({ page }) => {
    await boot(page, { conv: '1' });
    const det = page.locator('details.think-block').first();
    await expect(det).toBeVisible();
    expect(await det.evaluate((el) => el.open)).toBe(false);
  });
});

test.describe('1.5 composer dials', () => {
  test('effort Off suppresses the think block', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Thinking effort').click();
    await page.getByRole('button', { name: /Off/ }).click();
    // new conversation, send
    await page.getByTitle('Expand sidebar').click();
    await page.getByTitle('New conversation').first().click();
    const ta = page.locator('#zend-prompt');
    await ta.fill('walk me through the request lifecycle');
    await ta.press('Enter');
    await page.waitForTimeout(2500);
    await expect(page.locator('[data-msg]').last().locator('details.think-block')).toHaveCount(0);
  });

  test('lower verbosity yields a shorter answer than higher', async ({ page }) => {
    await boot(page);
    const ask = async (verb) => {
      await page.getByTitle('New conversation').first().click().catch(() => {});
      await page.getByTitle('Response length').click();
      await page.getByRole('button', { name: new RegExp(verb) }).click();
      const ta = page.locator('#zend-prompt');
      await ta.fill('give me the lifecycle overview');
      await ta.press('Enter');
      await page.waitForTimeout(3500);
      return (await page.locator('[data-msg]').last().locator('.zmd').innerText()).length;
    };
    const terse = await ask('Terse');
    const comp = await ask('Comprehensive');
    expect(comp).toBeGreaterThan(terse);
  });
});

test.describe('1.6 projection timeline', () => {
  test('active conversation seeds dots; hover shows popover; click opens substrate', async ({ page }) => {
    await boot(page, { conv: '1' });
    await expect(page.locator('.z-dot')).toHaveCount(7);
    await page.locator('.z-dot').first().hover();
    await expect(page.getByText('unbounded context')).toBeVisible();
    await page.locator('.z-dot').first().click();
    await expect(page.getByText('Windowed substrate')).toBeVisible();
  });
});

test.describe('1.7 windowed substrate', () => {
  test('opens with sections; assistant section expanded by default; copy all', async ({ page }) => {
    await boot(page, { conv: '1' });
    await page.locator('.z-dot').last().click();
    await expect(page.getByText('Windowed substrate')).toBeVisible();
    await expect(page.getByText('kv-inject').first()).toBeVisible();
    await page.getByText('Copy all').click();
    await expect(page.getByText('Copied')).toBeVisible();
  });
});

test.describe('1.8 files + upload', () => {
  test('files pane lists seeded files; viewer opens', async ({ page }) => {
    await boot(page, { conv: '1' });
    await page.getByTitle('Conversation files').click();
    await expect(page.locator('.z-file-row')).toHaveCount(4);
    await page.locator('.z-file-row').first().click();
    await expect(page.getByText('Download')).toBeVisible();
  });

  test('drag-drop shows the upload modal with a per-part bar, then completes', async ({ page }) => {
    await boot(page, { conv: '1' });
    await dropFiles(page, [{ name: 'notes.py', content: 'print(1)\n'.repeat(400) }]);
    await expect(page.getByText(/Uploading|prefilling/)).toBeVisible();
    await expect(page.locator('.z-upbar')).toBeVisible();
    // completes -> modal closes, files pane opens
    await expect(page.locator('.z-upbar')).toHaveCount(0, { timeout: 10000 });
    await expect(page.getByText('notes.py')).toBeVisible();
  });
});

test.describe('1.9 logs', () => {
  test('logs pane toggles, seeds, and clears', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Show logs').click();
    await expect(page.locator('.z-logs')).toBeVisible();
    expect(await page.locator('.z-logs').getByText('zend::', { exact: false }).count()).toBeGreaterThan(0);
    await page.getByText('Clear', { exact: true }).click();
    await page.getByTitle('Hide logs').click();
    await expect(page.locator('.z-logs')).toHaveCount(0);
  });
});

test.describe('1.10 cross-cutting', () => {
  test('theme persists across reload', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Appearance').click();
    await page.getByRole('button', { name: 'Vivid' }).click();
    await page.reload();
    await page.waitForFunction(() => window.__ZEND_READY__ === true);
    expect(await page.evaluate(() => localStorage.getItem('zend.theme'))).toBe('vivid');
  });

  test('Escape closes overlays in priority order', async ({ page }) => {
    await boot(page, { conv: '1' });
    await page.locator('.z-dot').first().click();
    await expect(page.getByText('Windowed substrate')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByText('Windowed substrate')).toHaveCount(0);
  });

  test('composer keeps focus when clicking a non-input element', async ({ page }) => {
    await boot(page, { conv: '1' });
    await page.locator('#zend-prompt').click();
    await page.locator('.z-chatscroll').click({ position: { x: 200, y: 50 } }).catch(() => {});
    expect(await page.evaluate(() => document.activeElement && document.activeElement.id)).toBe('zend-prompt');
  });
});

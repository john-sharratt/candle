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
    // Dispatch ON the element that carries the drop handlers (`.z-dropzone`), not
    // an ancestor — DragEvents bubble upward, so dispatching on `#app` (an
    // ancestor) would never reach the handler and the assertion would pass
    // vacuously. Fall back to the old resolution only if the class is absent.
    const scroller = document.querySelector('.z-chatscroll');
    const zone = document.querySelector('.z-dropzone')
      || (scroller ? scroller.parentElement : document.querySelector('#app'));
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

// Open a fresh conversation so file drag-and-drop is enabled — uploads target an
// OPEN chat, never the home/greeting screen. Uses the "Read some files" starter
// card, now the entry point for a read-these-files session (clicking it sends a
// priming message, so the chat view renders and drops are accepted).
async function openConvForUpload(page) {
  await page.getByText('Read some files').click();
  await expect(page.locator('.z-chatscroll')).toBeVisible();
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
  test('per-turn copy button copies the whole turn', async ({ page }) => {
    await boot(page);
    const btn = page.locator('.z-turn-copy').first();
    await btn.click();
    await expect(btn.locator('span')).toHaveText('Copied');
  });
  test('chat turns are text-selectable (not preventDefault-blocked)', async ({ page }) => {
    await boot(page);
    // The global mousedown handler must not cancel selection inside a turn.
    const blocked = await page.evaluate(() => {
      const turn = document.querySelector('.z-turn.z-selectable');
      if (!turn) return 'no turn';
      const ev = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
      turn.dispatchEvent(ev);
      return ev.defaultPrevented;   // false => selection is allowed
    });
    expect(blocked).toBe(false);
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
    await expect(page.locator('.z-file-row')).toHaveCount(6);
    await page.locator('.z-file-row').first().click();
    await expect(page.getByText('Download')).toBeVisible();
  });

  test('markdown file renders as markdown; csv as a table', async ({ page }) => {
    await boot(page);
    await page.getByTitle('Conversation files').click();
    await page.getByText('substrate-schema.md').click();
    // The .md body renders through the markdown pipeline (heading element),
    // not as a raw <pre>.
    await expect(page.locator('.zfile-render.zmd h1')).toBeVisible();
    await page.keyboard.press('Escape');
    await page.getByText('bench.csv').click();
    await expect(page.locator('table.zfile-table thead th').first()).toHaveText('shape');
    // The quoted field with an embedded comma stays one cell.
    await expect(page.locator('table.zfile-table td', { hasText: 'q512_f16, prefix4k' })).toBeVisible();
  });

  test('executable uploads are blocked with a reason, never stored', async ({ page }) => {
    await boot(page);
    await openConvForUpload(page);
    await dropFiles(page, [{ name: 'malware.exe', content: 'MZ...' }]);
    // The modal shows the rejected row + reason; the file is never admitted.
    await expect(page.getByText(/executable, installer, and auto-run/)).toBeVisible();
    await expect(page.locator('.z-upstat', { hasText: 'rejected' })).toBeVisible();
    await page.getByRole('button', { name: 'Close' }).click();
    await page.getByTitle('Conversation files').click();
    await expect(page.getByText('malware.exe')).toHaveCount(0);
  });

  test('uploaded file appears inline in history and opens on click', async ({ page }) => {
    await boot(page);
    await openConvForUpload(page);
    await dropFiles(page, [{ name: 'inline-demo.py', content: 'print("hi")\n' }]);
    // Wait out the modal (upload + phases), then the inline tile shows in chat.
    await expect(page.locator('.z-upbar')).toHaveCount(0, { timeout: 10000 });
    const tile = page.locator('.z-uptile', { hasText: 'inline-demo.py' });
    await expect(tile).toBeVisible();
    // The persisted throughput renders as an inline stat line under the tiles.
    await expect(page.locator('.z-uptile-stat').filter({ hasText: 't/s' })).toBeVisible();
    // Clicking the tile opens the document viewer directly (files pane stays
    // closed); the viewer places the measured upload time in its header.
    await tile.click();
    await expect(page.getByText('Download')).toBeVisible();
    await expect(page.getByText(/Uploaded in/)).toBeVisible();
  });

  test('drag-drop runs the three stages (upload, read, summarize), then completes', async ({ page }) => {
    await boot(page);
    await openConvForUpload(page);
    await dropFiles(page, [{ name: 'notes.py', content: 'print(1)\n'.repeat(400) }]);
    // The modal is titled by the file; the pipeline shows one row per stage.
    await expect(page.locator('.z-uptitle', { hasText: 'notes.py' })).toBeVisible();
    await expect(page.getByText('Upload', { exact: true })).toBeVisible();
    await expect(page.locator('.z-upbar')).toBeVisible();
    // Stage 2 (Read) appears as its own row once the engine phase runs.
    // Summarisation is a background task and is deliberately NOT shown.
    await expect(page.getByText('Read', { exact: true })).toBeVisible();
    await expect(page.getByText('Summarize', { exact: true })).toHaveCount(0);
    // Each stage's metric rides inline on its row: upload MB/s, and prefill
    // token count with t/s for the read stage (measured live).
    await expect(page.locator('.z-upmetric').filter({ hasText: 'MB/s' })).toBeVisible();
    await expect(page.locator('.z-upmetric').filter({ hasText: 'tokens' }).first()).toBeVisible();
    await expect(page.locator('.z-upmetric').filter({ hasText: 't/s' }).first()).toBeVisible();
    // completes -> running bars gone, modal closes, tile shows in chat
    await expect(page.locator('.z-upbar')).toHaveCount(0, { timeout: 10000 });
    await expect(page.getByText('notes.py')).toBeVisible();
  });

  test('home screen ignores file drops; the "Read some files" card opens a conversation', async ({ page }) => {
    await boot(page);
    await page.getByTitle('New conversation').first().click();
    // On the home / greeting screen there is no chat scroller and dropping a file
    // does nothing — no upload modal appears (the home-drop path was removed).
    await expect(page.locator('.z-chatscroll')).toHaveCount(0);
    await dropFiles(page, [{ name: 'ignored.py', content: 'print(1)\n' }]);
    await expect(page.locator('.z-upbar')).toHaveCount(0);
    // The starter card opens a conversation (its priming message is sent), which
    // is the supported way to begin a read-these-files session.
    await page.getByText('Read some files').click();
    await expect(page.locator('.z-chatscroll')).toBeVisible();
    // ...and now a drop IS accepted into that open conversation.
    await dropFiles(page, [{ name: 'now-ok.py', content: 'print(2)\n' }]);
    await expect(page.locator('.z-upbar')).toBeVisible();
  });

  test('a follow-up message after an upload still sends and gets a reply', async ({ page }) => {
    await boot(page);
    await openConvForUpload(page);
    await dropFiles(page, [{ name: 'ctx.py', content: 'print(1)\n' }]);
    // Let the upload finish (modal closes) — history now holds an inline `upload`
    // event whose role is NOT a valid chat role.
    await expect(page.locator('.z-upbar')).toHaveCount(0, { timeout: 10000 });
    await expect(page.locator('.z-uptile', { hasText: 'ctx.py' })).toBeVisible();
    // The card's own reply must have finished (composer back to Send) so the
    // follow-up isn't dropped by the still-streaming guard in send().
    await expect(page.locator('button[aria-label="Send"]')).toBeVisible({ timeout: 10000 });
    // Sending must filter that event out of the request `messages`; leaving it in
    // makes the daemon reject the whole request (422) so the send silently no-ops
    // and the user turn flashes then vanishes. Assert it persists AND draws a reply.
    const ta = page.locator('#zend-prompt');
    await ta.fill('what does this file do?');
    await ta.press('Enter');
    await expect(page.getByText('what does this file do?')).toBeVisible();
    await page.waitForTimeout(2500);
    await expect(page.getByText('what does this file do?')).toBeVisible(); // did not vanish
    await expect(page.locator('[data-msg]').last().locator('.zmd')).not.toBeEmpty();
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

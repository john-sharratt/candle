// Job blocks and streaming replies redraw only what moved
// (docs/performance_patch.md G1, G2, G3, G6, G7). Against the mock (?mock=1).
const { test, expect } = require('@playwright/test');

const URL = '/index.html?mock=1';

async function boot(page, conv) {
  await page.goto('about:blank');
  await page.goto(conv ? (URL + '#conv=' + encodeURIComponent(conv)) : URL);
  await page.waitForFunction(() => window.__ZEND_READY__ === true);
}

const BUILD = 'build:76c3a0f4d39bd9db:XzBY8ehGFz0';
const TASK = 'task:review/verify';

// Every chat row marked, so a test can see whether it is the same node later.
const markRows = (page) => page.evaluate(() => {
  document.querySelectorAll('[data-msg]').forEach((el, i) => { el.__mark = i + 1; });
  return document.querySelectorAll('[data-msg]').length;
});
const rowsKept = (page) => page.evaluate(() =>
  Array.from(document.querySelectorAll('[data-msg]')).every((el) => el.__mark));
const seconds = async (page) => parseInt(await page.locator('.job-block[data-job="' + BUILD + '"] .job-time').textContent(), 10);

// Pause the page's clock where it stands, so time moves only as a test says.
async function pause(page) {
  const now = await page.evaluate(() => Date.now());
  await page.clock.pauseAt(now + 50);
}

// Count the page's view calls for `handle`, with the page's own clock.
const countViews = (page, handle) => page.evaluate((handle) => {
  window.__views = [];
  const api = window.ZendAPI;
  const view = api.jobView.bind(api);
  api.jobView = (h) => { if (h === handle) window.__views.push(performance.now()); return view(h); };
}, handle);

test.describe('G1 a job redraws only its own blocks', () => {
  test('a poll that changes nothing keeps every row and moves the clock', async ({ page }) => {
    await page.addInitScript(() => { window.__ZEND_MOCK_JOB_STILL__ = true; });
    await page.clock.install();
    await boot(page, '7');
    await expect(page.locator('.job-block[data-job="' + BUILD + '"] .job-line')).toHaveText('running — compiling zend');
    await pause(page);
    expect(await markRows(page)).toBeGreaterThan(1);
    const before = await seconds(page);
    await page.clock.runFor(2000);
    await page.clock.runFor(2000);
    expect(await rowsKept(page)).toBe(true);
    expect(await seconds(page)).toBe(before + 4);
  });

  test('a job that moves leaves the bubbles that do not draw it alone', async ({ page }) => {
    await page.clock.install();
    await boot(page, '7');
    await expect(page.locator('.job-block[data-job="' + BUILD + '"] .job-line')).toHaveText('running — compiling zend');
    await pause(page);
    await markRows(page);
    const tail = await page.locator('.job-wait[data-job-open="' + BUILD + '"] .job-wait-tail').textContent();
    await page.clock.runFor(2100);
    await expect(page.locator('.job-wait[data-job-open="' + BUILD + '"] .job-wait-tail')).not.toHaveText(tail);
    const kept = await page.evaluate(() => {
      const row = Array.from(document.querySelectorAll('[data-msg]')).find((el) => /The check is running/.test(el.textContent));
      return !!(row && row.__mark);
    });
    expect(kept).toBe(true);
  });

  test('a sub-agent thought opened, redrawn and closed stays closed', async ({ page }) => {
    await page.addInitScript(() => { window.__ZEND_MOCK_SUBAGENT_PARTS__ = true; });
    await boot(page, '7');
    const block = page.locator('.job-block[data-job="' + TASK + '"]');
    await block.locator('.job-head').click();
    const think = block.locator('details.think-block').first();
    await expect(think).toBeVisible();
    await think.locator('summary').click();
    await expect(think).toHaveAttribute('open', '');
    // The task runs on, so its block is redrawn on the next poll.
    await page.waitForTimeout(2500);
    await expect(block.locator('details.think-block').first()).toHaveAttribute('open', '');
    await block.locator('details.think-block summary').first().click();
    await page.waitForTimeout(2500);
    await expect(block.locator('details.think-block').first()).not.toHaveAttribute('open', '');
  });
});

test.describe('G2 a streaming reply redraws only its live part', () => {
  test('a closed call is drawn once while the reasoning after it streams', async ({ page }) => {
    await page.addInitScript(() => { window.__ZEND_MOCK_PARTS_STREAM__ = true; });
    await boot(page);
    await page.getByText('Why is decode latency spiking under load?').click();
    await expect(page.locator('.zmd').first()).toBeVisible();
    const ta = page.locator('#zend-prompt');
    await ta.fill('Read three files');
    await ta.press('Enter');
    const last = page.locator('[data-msg]').last();
    await expect(last.locator('.tool-call-card')).toHaveCount(3);
    // Well over ten frames of the reasoning streaming in.
    await page.waitForTimeout(1500);
    await expect(last.locator('details.think-block')).toHaveCount(1);
    expect(await page.evaluate(() => window.__ZEND_RESULT_JSON__)).toEqual([1, 1, 1]);
  });

  test('opening one thought while the reply decodes leaves the other shut', async ({ page }) => {
    await page.addInitScript(() => { window.__ZEND_MOCK_TWO_THINKS__ = true; });
    await boot(page);
    await page.getByText('Why is decode latency spiking under load?').click();
    await expect(page.locator('.zmd').first()).toBeVisible();
    const ta = page.locator('#zend-prompt');
    await ta.fill('Think twice');
    await ta.press('Enter');
    const thinks = page.locator('[data-msg]').last().locator('details.think-block');
    await expect(thinks).toHaveCount(2);
    await thinks.nth(0).locator('summary').click();
    // Many frames of the second thought streaming in, each a redraw.
    await page.waitForTimeout(1500);
    await expect(thinks.nth(0)).toHaveAttribute('open', '');
    await expect(thinks.nth(1)).not.toHaveAttribute('open', '');
    await thinks.nth(1).locator('summary').click();
    await page.waitForTimeout(500);
    await expect(thinks.nth(1)).toHaveAttribute('open', '');
    await thinks.nth(0).locator('summary').click();
    await page.waitForTimeout(500);
    await expect(thinks.nth(0)).not.toHaveAttribute('open', '');
    await expect(thinks.nth(1)).toHaveAttribute('open', '');
  });
});

test.describe('G3 the job poll', () => {
  test('with nothing clicked, a job is viewed every two seconds', async ({ page }) => {
    await page.clock.install();
    await boot(page, '7');
    await expect(page.locator('.job-block[data-job="' + BUILD + '"] .job-line')).toHaveText('running — compiling zend');
    await pause(page);
    await countViews(page, BUILD);
    await page.clock.runFor(4200);
    const at = await page.evaluate(() => window.__views);
    expect(at.length).toBe(2);
    const gap = at[1] - at[0];
    expect(gap).toBeGreaterThanOrEqual(2000);
    expect(gap).toBeLessThan(2200);
  });

  test('clicks while a poll is in flight make one more view when it settles', async ({ page }) => {
    await boot(page, '7');
    await expect(page.locator('.job-block[data-job="' + BUILD + '"] .job-line')).toHaveText('running — compiling zend');
    await page.evaluate((handle) => {
      window.__views = [];
      let release;
      window.__gate = new Promise((r) => { release = r; });
      window.__release = release;
      const api = window.ZendAPI;
      const view = api.jobView.bind(api);
      api.jobView = (h) => {
        if (h === handle) window.__views.push(performance.now());
        return window.__gate.then(() => view(h));
      };
    }, BUILD);
    await expect.poll(() => page.evaluate(() => window.__views.length), { timeout: 5000 }).toBe(1);
    const head = page.locator('.job-block[data-job="' + BUILD + '"] .job-head');
    await head.click();
    await head.click();
    await head.click();
    expect(await page.evaluate(() => window.__views.length)).toBe(1);
    await page.evaluate(() => window.__release());
    await expect.poll(() => page.evaluate(() => window.__views.length), { timeout: 1500 }).toBe(2);
    await page.waitForTimeout(800);
    expect(await page.evaluate(() => window.__views.length)).toBe(2);
  });

  test('a hidden page polls nothing, and polls at once when shown', async ({ page }) => {
    await page.clock.install();
    await page.addInitScript(() => {
      window.__hidden = false;
      Object.defineProperty(document, 'hidden', { configurable: true, get: () => window.__hidden });
    });
    await boot(page, '7');
    await expect(page.locator('.job-block[data-job="' + BUILD + '"] .job-line')).toHaveText('running — compiling zend');
    await pause(page);
    await page.evaluate(() => { window.__hidden = true; document.dispatchEvent(new Event('visibilitychange')); });
    await countViews(page, BUILD);
    await page.clock.runFor(6000);
    expect(await page.evaluate(() => window.__views.length)).toBe(0);
    await page.evaluate(() => { window.__hidden = false; document.dispatchEvent(new Event('visibilitychange')); });
    await page.clock.runFor(10);
    expect(await page.evaluate(() => window.__views.length)).toBe(1);
  });
});

test.describe('G6 the mock loads only when asked for', () => {
  test('a boot without ?mock never asks for the mock', async ({ page }) => {
    const asked = [];
    page.on('request', (r) => asked.push(r.url()));
    await page.goto('/index.html');
    await page.waitForLoadState('load');
    expect(asked.filter((u) => u.includes('zend-api.mock.js'))).toEqual([]);
    expect(asked.some((u) => u.includes('zend-api.live.js'))).toBe(true);
  });
});

test.describe('G7 a sub-agent is drawn from its parts', () => {
  test('its cards show their results and no tool-result turn is drawn as text', async ({ page }) => {
    await page.addInitScript(() => { window.__ZEND_MOCK_SUBAGENT_PARTS__ = true; });
    await boot(page, '7');
    const block = page.locator('.job-block[data-job="' + TASK + '"]');
    await block.locator('.job-head').click();
    const card = block.locator('.tool-call-card').first();
    await expect(card).toBeVisible();
    await expect(card.locator('.tool-call-badge')).toHaveText('result');
    await expect(block.locator('.job-msg.user', { hasText: 'tool_response' })).toHaveCount(0);
    await expect(block.locator('.job-msg.user')).toHaveCount(1);
  });
});

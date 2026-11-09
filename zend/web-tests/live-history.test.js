// Unit test for zend/web/zend-api.live.js's history mapping: an assistant
// bubble keeps the parts the daemon split it into. Plain Node
// (`node --test live-history.test.js`).
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { historyOf } = require('../web/zend-api.live.js');

test('the live history keeps each bubble\'s parts', () => {
  const parts = [{ kind: 'prose', text: 'Reading.' }, { kind: 'call', tool: 'file_read', args: [['path', 'a.rs']], result: { ok: true } }];
  const got = historyOf([
    { role: 'assistant', content: 'Reading.<tool_call>…</tool_call>', parts },
    { role: 'user', content: '<tool_response>{"ok": true}</tool_response>', tool_tokens: [4] },
  ]);
  assert.equal(got[0].parts, parts);
  assert.equal(got[1].parts, null, 'a bubble with no parts keeps none');
  assert.deepEqual(got[1].tool_tokens, [4]);
});

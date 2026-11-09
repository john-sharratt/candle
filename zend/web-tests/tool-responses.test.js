// Unit tests for zend/web/tool-responses.js — a tool-result turn parsed once.
// Plain Node (`node --test tool-responses.test.js`); no browser.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { responsesOf } = require('../web/tool-responses.js');

const RESULTS = '<tool_response>\n{"a": 1}\n</tool_response>\n<tool_response>\n{"b": 2}\n</tool_response>';

test('a tool-result message answers the same array while its content holds', () => {
  const m = { role: 'user', content: RESULTS };
  const first = responsesOf(m);
  assert.deepEqual(first, [
    '<tool_response>\n{"a": 1}\n</tool_response>',
    '<tool_response>\n{"b": 2}\n</tool_response>',
  ]);
  assert.equal(responsesOf(m), first);
});

test('a message whose content changed is parsed again', () => {
  const m = { role: 'user', content: RESULTS };
  const first = responsesOf(m);
  m.content = '<tool_response>\n{"c": 3}\n</tool_response>';
  const second = responsesOf(m);
  assert.notEqual(second, first);
  assert.deepEqual(second, ['<tool_response>\n{"c": 3}\n</tool_response>']);
});

test('a message that is no tool result answers null, both times', () => {
  const m = { role: 'user', content: 'a question' };
  assert.equal(responsesOf(m), null);
  assert.equal(responsesOf(m), null);
  assert.equal(responsesOf({ role: 'assistant', content: RESULTS }), null);
});

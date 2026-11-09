// Unit tests for zend/web/conversation-merge.js — the sidebar's rows merged
// with the server's list. Plain Node (`node --test conversation-merge.test.js`).
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { mergeConversations } = require('../web/conversation-merge.js');

test('a list naming a new id twice adds it once', () => {
  const got = mergeConversations([], [{ id: 'n', title: 'New' }, { id: 'n', title: 'New' }], null);
  assert.deepEqual(got, { convs: [{ id: 'n', title: 'New', history: [] }], dropped: [], mutated: true });
});

test('a known row takes a new title and rank', () => {
  const known = { id: 'k', title: 'Old', updated_ms: 1, history: [] };
  const got = mergeConversations([known], [{ id: 'k', title: 'Titled', updated_ms: 5 }], null);
  assert.deepEqual(got, { convs: [{ id: 'k', title: 'Titled', updated_ms: 5, history: [] }], dropped: [], mutated: true });
  assert.equal(got.convs[0], known, 'updated in place');
});

test('a row the server does not list is dropped, unless it is open or streaming', () => {
  const gone = { id: 'g', title: 'Gone', history: [] };
  const open = { id: 'o', title: 'Draft', history: [] };
  const streaming = { id: 's', title: 'Live', streaming: true, history: [] };
  const got = mergeConversations([gone, open, streaming], [], 'o');
  assert.deepEqual(got, { convs: [open, streaming], dropped: ['g'], mutated: true });
});

test('a list that changes nothing changes nothing', () => {
  const known = { id: 1, title: 'Same', updated_ms: 2, archived: false, history: [] };
  const got = mergeConversations([known], [{ id: '1', title: 'Same', updated_ms: 2, archived: false }], null);
  assert.deepEqual(got, { convs: [known], dropped: [], mutated: false });
});

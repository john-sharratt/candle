// Unit tests for zend/web/job-poll.js — what a job poll redraws and reads.
// Plain Node (`node --test job-poll.test.js`); no browser.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { visiblyChanged, reads } = require('../web/job-poll.js');

const job = (over) => Object.assign({
  kind: 'build', ended: false, line: 'build: 3 s', open: false,
  lines: [{ n: 1, text: 'a' }], tail: [{ n: 1, text: 'a' }, { n: 2, text: 'b' }, { n: 3, text: 'c' }],
  messages: null,
}, over);

test('a new tail array holding the same lines draws nothing new', () => {
  const prev = job();
  const next = job({ tail: prev.tail.map((l) => ({ n: l.n, text: l.text })) });
  assert.equal(visiblyChanged(prev, next), false);
});

test('a changed middle tail line is drawn', () => {
  const prev = job();
  const next = job({ tail: [{ n: 1, text: 'a' }, { n: 2, text: 'B' }, { n: 3, text: 'c' }] });
  assert.equal(visiblyChanged(prev, next), true);
});

test('what is read alone draws nothing new', () => {
  const prev = job();
  assert.equal(visiblyChanged(prev, Object.assign({}, prev, { fresh: false, total: 9, size: 90 })), false);
});

test('a job seen for the first time is drawn', () => {
  assert.equal(visiblyChanged(undefined, job()), true);
});

const view = (over) => Object.assign({ kind: 'build', ended: false, total: 10, size: 100, turns: undefined }, over);

test('an unchanged build view reads nothing', () => {
  assert.deepEqual(reads({ kind: 'build', ended: false, total: 10, size: 100, open: false }, view()), []);
});

test('an open build that grew reads its tail and on from the last line held', () => {
  assert.deepEqual(reads({ kind: 'build', ended: false, total: 10, size: 100, open: true }, view({ total: 12, size: 130 })), ['tail', 'from']);
});

test('an open, unread build that ended reads its tail and its log, though no line was added', () => {
  assert.deepEqual(reads({ kind: 'build', ended: false, total: 10, size: 100, open: true, fresh: true }, view({ ended: true })), ['tail', 'from']);
});

test('a running task reads its tail every tick', () => {
  assert.deepEqual(reads({ kind: 'task', ended: false, total: 4, size: 40, open: false }, view({ kind: 'task', total: 4, size: 40 })), ['tail']);
});

test('an open task whose conversation gained a turn reads it again', () => {
  assert.deepEqual(
    reads({ kind: 'task', ended: false, open: true, turns: 3 }, view({ kind: 'task', turns: 4 })),
    ['tail', 'conversation'],
  );
});

test('an ended task not open reads nothing', () => {
  assert.deepEqual(reads({ kind: 'task', ended: true, open: false }, view({ kind: 'task', ended: true })), []);
});

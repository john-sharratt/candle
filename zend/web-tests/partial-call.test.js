// Unit tests for zend/web/partial-call.js — the reader behind the GUI's live
// writing box. Plain Node (`node --test partial-call.test.js`); no browser.
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { parse } = require('../web/partial-call.js');

test('a content value mid-decode is reported as the one being written', () => {
  const c = parse('\n{"name": "write", "arguments": {"path": "docs/a.md", "content": "# Title\\n\\nFirst par');
  assert.equal(c.name, 'write');
  assert.equal(c.args.path, 'docs/a.md');
  assert.equal(c.args.content, '# Title\n\nFirst par');
  assert.equal(c.writing, 'content');
});

test('a finished call has nothing being written', () => {
  const c = parse('{"name": "write", "arguments": {"path": "a", "content": "x"}}');
  assert.deepEqual(c, { name: 'write', args: { path: 'a', content: 'x' }, writing: null });
});

test('an escape cut in half is left out, not shown half-decoded', () => {
  assert.equal(parse('{"name": "write", "arguments": {"content": "a\\').args.content, 'a');
  assert.equal(parse('{"name": "write", "arguments": {"content": "a\\u00').args.content, 'a');
  assert.equal(parse('{"name": "write", "arguments": {"content": "a\\u00e9').args.content, 'aé');
  assert.equal(parse('{"name": "write", "arguments": {"content": "say \\"hi\\"').args.content, 'say "hi"');
});

test('braces and quotes inside a value are content, not structure', () => {
  const c = parse('{"name": "write", "arguments": {"path": "a.rs", "content": "fn f() { \\"}\\" }\\n');
  assert.equal(c.args.content, 'fn f() { "}" }\n');
  assert.equal(c.writing, 'content');
});

test('nested values are read past, and the arguments after them still found', () => {
  const c = parse('{"name": "t", "arguments": {"opts": {"x": "deep", "l": [1, "s"]}, "n": 3, "path": "p", "body": "b');
  assert.equal(c.args.path, 'p');
  assert.equal(c.args.body, 'b');
  assert.equal(c.args.x, undefined, 'a nested string is not an argument');
  assert.equal(c.writing, 'body');
});

test('a call cut before any argument names only its tool', () => {
  assert.deepEqual(parse('{"name": "wri'), { name: null, args: {}, writing: null });
  assert.deepEqual(parse('{"name": "write", "argu'), { name: 'write', args: {}, writing: null });
});

test('the function-block syntax reads the same way', () => {
  const c = parse('\n<function=write>\n<parameter=path>\ndocs/a.md\n</parameter>\n<parameter=content>\nline one\nline two');
  assert.equal(c.name, 'write');
  assert.equal(c.args.path, 'docs/a.md');
  assert.equal(c.args.content, 'line one\nline two');
  assert.equal(c.writing, 'content');
  const done = parse('<function=write>\n<parameter=path>\na\n</parameter>\n</function>');
  assert.equal(done.writing, null);
});

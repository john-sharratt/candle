/* ============================================================================
 * partial-call.js — read a tool call while it is still being written
 * ----------------------------------------------------------------------------
 * The daemon streams a `<tool_call>` block token by token, so for the whole
 * time a large argument decodes — a file's content, thousands of tokens — the
 * page holds a call that is not yet valid in any syntax. This reads what is
 * there so far: the tool's name, every string argument written (the last one
 * possibly cut mid-value), and which argument is still being written.
 *
 * Both call syntaxes the daemon's models write:
 *  - JSON (Qwen3):            {"name": "write", "arguments": {"path": "a", "content": "…
 *  - function block (Qwen3.5): <function=write>\n<parameter=path>\na\n</parameter>\n…
 *
 * Exposed as window.ZendPartialCall for the page, and as a CommonJS module for
 * the Node test (zend/web-tests/partial-call.test.js).
 * ========================================================================== */
(function (root) {
  'use strict';

  const ESCAPES = { n: '\n', t: '\t', r: '\r', b: '\b', f: '\f', '"': '"', '\\': '\\', '/': '/' };

  // A JSON string starting at `src[i]` (the opening quote), decoded. `done` is
  // false when the text ends first — including partway through an escape,
  // which is left out rather than shown half-decoded.
  function readString(src, i) {
    let s = '';
    let j = i + 1;
    while (j < src.length) {
      const c = src[j];
      if (c === '"') return { s, done: true, end: j + 1 };
      if (c !== '\\') { s += c; j++; continue; }
      if (j + 1 >= src.length) break;
      const e = src[j + 1];
      if (e === 'u') {
        const hex = src.slice(j + 2, j + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) break;
        s += String.fromCharCode(parseInt(hex, 16));
        j += 6;
        continue;
      }
      s += e in ESCAPES ? ESCAPES[e] : e;
      j += 2;
    }
    return { s, done: false, end: src.length };
  }

  // The JSON call syntax. A stack of open containers tracks where each string
  // sits: a value of `name` in the outer object is the tool, and a value in
  // the object under `arguments` (or `parameters`) is an argument. Strings
  // anywhere deeper are read past and not reported.
  function parseJson(src) {
    const out = { name: null, args: {}, writing: null };
    const stack = [];
    let i = 0;
    while (i < src.length) {
      const c = src[i];
      const top = stack[stack.length - 1];
      if (c === '{' || c === '[') {
        if (top && top.type === 'obj' && top.expect === 'value') top.expect = 'after';
        stack.push({ type: c === '{' ? 'obj' : 'arr', key: null, expect: c === '{' ? 'key' : 'value' });
        i++;
      } else if (c === '}' || c === ']') {
        stack.pop();
        i++;
      } else if (c === ',') {
        if (top && top.type === 'obj') top.expect = 'key';
        i++;
      } else if (c === ':') {
        if (top && top.type === 'obj') top.expect = 'value';
        i++;
      } else if (c === '"') {
        const r = readString(src, i);
        if (top && top.type === 'obj' && top.expect === 'key') {
          if (!r.done) break;
          top.key = r.s;
        } else if (top && top.type === 'obj' && top.expect === 'value') {
          const depth = stack.length;
          const parent = stack[depth - 2];
          // A name is only a name once complete: `wri` is not a tool.
          if (depth === 1 && top.key === 'name' && r.done) out.name = r.s;
          if (depth === 2 && parent && (parent.key === 'arguments' || parent.key === 'parameters')) {
            out.args[top.key] = r.s;
            if (!r.done) out.writing = top.key;
          }
          top.expect = 'after';
        }
        if (!r.done) break;
        i = r.end;
      } else {
        i++;
      }
    }
    return out;
  }

  // The function-block syntax: one element per argument, its value raw text.
  function parseFunctionBlock(src) {
    const out = { name: null, args: {}, writing: null };
    const head = /<function=([^>\s]+)>/.exec(src);
    if (head) out.name = head[1];
    const param = /<parameter=([^>]+)>\n?([\s\S]*?)(\n?<\/parameter>|$)/g;
    let m;
    while ((m = param.exec(src)) !== null) {
      out.args[m[1]] = m[2];
      if (!m[3]) out.writing = m[1];
      if (m[0].length === 0) break;
    }
    return out;
  }

  // `src` is everything after `<tool_call>` so far.
  function parse(src) {
    const text = String(src || '');
    return /^\s*<function=/.test(text) ? parseFunctionBlock(text) : parseJson(text);
  }

  const api = { parse };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ZendPartialCall = api;
})(typeof window !== 'undefined' ? window : this);

/* ============================================================================
 * tool-responses.js — a tool-result turn's blocks, parsed once
 * ----------------------------------------------------------------------------
 * A user turn that is purely tool results — `<tool_response>…</tool_response>`
 * blocks, one per call — is folded into the calls it answers. Parsed on every
 * render, each turn's blocks were fresh strings every time, and the cards'
 * memo compared them character by character. So the parse is kept on the
 * message while its content is the same string, and the same array comes back.
 *
 * Exposed as window.ZendToolResponses for the page, and as a CommonJS module
 * for the Node test (zend/web-tests/tool-responses.test.js).
 * ========================================================================== */
(function (root) {
  'use strict';

  /** Whether `content` is a tool-result turn. */
  function isToolResponse(content) {
    const t = (content || '').trim();
    return t.indexOf('<tool_response>') !== -1 && t.indexOf('</tool_response>') !== -1;
  }

  /** The `<tool_response>…</tool_response>` blocks of a tool-result turn, in order. */
  function toolResponseBlocks(content) {
    return String(content || '').match(/<tool_response>[\s\S]*?<\/tool_response>/g) || [content];
  }

  /**
   * Message `m`'s tool-result blocks, or `null` for a message that is not a
   * tool result — kept on `m._resp` and reused while `m.content` is the same.
   */
  function responsesOf(m) {
    const kept = m._resp;
    if (kept && kept.content === m.content) return kept.blocks;
    const blocks = m.role === 'user' && isToolResponse(m.content) ? toolResponseBlocks(m.content) : null;
    m._resp = { content: m.content, blocks };
    return blocks;
  }

  const api = { isToolResponse, toolResponseBlocks, responsesOf };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ZendToolResponses = api;
})(typeof window !== 'undefined' ? window : this);

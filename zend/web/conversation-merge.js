/* ============================================================================
 * conversation-merge.js — the sidebar's rows, merged with the server's list
 * ----------------------------------------------------------------------------
 * The server is authoritative on which conversations exist, their titles,
 * whether they are archived and their rank. A row it no longer lists is
 * dropped — but never the one being streamed or the one open, which may be a
 * draft the server has not seen yet. A row it lists that the page does not
 * hold is added once, however many times the list names it.
 *
 * Matched through a Map by id, so a list of N against M rows costs N + M
 * rather than N × M.
 *
 * Exposed as window.ZendConversationMerge for the page, and as a CommonJS
 * module for the Node test (zend/web-tests/conversation-merge.test.js).
 * ========================================================================== */
(function (root) {
  'use strict';

  /**
   * `convs` merged with the server's `list`, with `activeId` open. Answers
   * `{ convs, dropped, mutated }`: the rows, the ids dropped, and whether
   * anything changed. A known row is updated in place.
   */
  function mergeConversations(convs, list, activeId) {
    const listed = new Set(list.map((e) => String(e.id)));
    const dropped = [];
    let mutated = false;
    const kept = convs.filter((c) => {
      if (c.streaming || c.id === activeId) return true;
      if (!listed.has(String(c.id))) { dropped.push(c.id); mutated = true; return false; }
      return true;
    });
    const byId = new Map(kept.map((c) => [String(c.id), c]));
    const added = [];
    for (const e of list) {
      const ex = byId.get(String(e.id));
      if (ex) {
        // The titler is authoritative.
        if (e.title && e.title !== ex.title) { ex.title = e.title; mutated = true; }
        // Archiving is one-way, so the sync only ever sets the flag: a list
        // served before an archive lands still says "not archived".
        if (e.archived && !ex.archived) { ex.archived = true; mutated = true; }
        // The rank moves whenever any conversation is used, here or elsewhere.
        if (e.updated_ms && e.updated_ms !== ex.updated_ms) { ex.updated_ms = e.updated_ms; mutated = true; }
      } else {
        const row = Object.assign({}, e, { history: e.history || [] });
        byId.set(String(e.id), row);
        added.push(row);
        mutated = true;
      }
    }
    return { convs: added.length ? kept.concat(added) : kept, dropped, mutated };
  }

  const api = { mergeConversations };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ZendConversationMerge = api;
})(typeof window !== 'undefined' ? window : this);

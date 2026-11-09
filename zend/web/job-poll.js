/* ============================================================================
 * job-poll.js — what a job poll redraws, and what it reads
 * ----------------------------------------------------------------------------
 * The console follows every job it draws: a build's or deploy's log, a task's
 * own conversation. Two decisions sit under that poll, kept here so they are
 * tested apart from the page:
 *
 *  - `visiblyChanged(prev, next)`: whether a job's new record changes anything
 *    its block or wait window draws — its kind, whether it ended, its line,
 *    whether it is open, its log's length and last line, its tail line by
 *    line (a task's output is not append-only), and its conversation. Only
 *    then does the job's `rev` move and the messages that draw it redraw.
 *  - `reads(job, view)`: which reads a poll makes after a job's view — its
 *    tail, its log from the line after the last held, or a task's
 *    conversation — so a tick that changes nothing reads nothing.
 *
 * Exposed as window.ZendJobPoll for the page, and as a CommonJS module for the
 * Node test (zend/web-tests/job-poll.test.js).
 * ========================================================================== */
(function (root) {
  'use strict';

  function lastText(lines) {
    return lines && lines.length ? lines[lines.length - 1].text : null;
  }

  function sameTail(a, b) {
    const x = a || [];
    const y = b || [];
    if (x.length !== y.length) return false;
    for (let i = 0; i < x.length; i++) if (x[i].text !== y[i].text) return false;
    return true;
  }

  /** Whether `next` draws differently from `prev`. */
  function visiblyChanged(prev, next) {
    if (!prev) return true;
    return prev.kind !== next.kind ||
      prev.ended !== next.ended ||
      prev.line !== next.line ||
      prev.open !== next.open ||
      (prev.lines || []).length !== (next.lines || []).length ||
      lastText(prev.lines) !== lastText(next.lines) ||
      !sameTail(prev.tail, next.tail) ||
      prev.messages !== next.messages;
  }

  /**
   * The reads a poll makes after `view` of `job`: `'tail'`, `'from'` (an open
   * log, from the line after the last held) and `'conversation'` (an open
   * task's), in that order.
   *
   * `job` holds what the page last saw — `ended`, `total` and `size` of the
   * last view, the `turns` of the conversation it last fetched — whether it is
   * `open`, and `fresh` when it has not been read since it was opened.
   *
   * A build's, deploy's or maintenance job's tail is read when its output
   * moved or it ended; a running task's every tick, its turn report carrying a
   * clock. Keyed on `ended`, never on whether a turn is live: that is visible
   * only on the member running it.
   */
  function reads(job, view) {
    const out = [];
    const endedNow = view.ended !== job.ended;
    if (view.kind === 'task') {
      if (!view.ended || endedNow) out.push('tail');
      if (job.open && (view.turns !== job.turns || endedNow || job.fresh)) out.push('conversation');
      return out;
    }
    const moved = view.total !== job.total || view.size !== job.size;
    if (moved || endedNow) out.push('tail');
    if (job.open && ((view.total || 0) > (job.total || 0) || endedNow || job.fresh)) out.push('from');
    return out;
  }

  const api = { visiblyChanged, reads };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ZendJobPoll = api;
})(typeof window !== 'undefined' ? window : this);

/* The two controls above a paper that need script.
 *
 * `Download PDF` is deliberately not here — it is a plain anchor in the
 * markup, so the one control an archivist actually needs survives with script
 * disabled. These two degrade to nothing, which is the right failure: a reader
 * without script still gets the whole document and the browser's own print.
 */

import { copyText } from './clip.js';

const tools = document.querySelector('.paper-tools');
if (tools) {
  const citation = tools.dataset.citation || '';
  tools.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    if (btn.dataset.act === 'print') {
      window.print();
      return;
    }
    if (btn.dataset.act === 'copy' && citation) {
      // `copyText` flashes the label only on a *confirmed* write — the
      // clipboard API rejects asynchronously, so a bare try/catch would let the
      // button claim success while nothing reached the clipboard.
      copyText(citation, btn, 'Copied', 'Copy citation');
    }
  });
}

/* Clipboard with an honest fallback.
 *
 * zend's lesson: `navigator.clipboard.writeText` rejects ASYNCHRONOUSLY, so a
 * bare try/catch never fires and the button cheerfully says "Copied" when
 * nothing was copied. Await it, fall back to the hidden-textarea path, and
 * flash the label only on real success.
 */

export function flash(btn, on, off) {
  if (!btn) return;
  const prev = off || btn.textContent;
  btn.textContent = on || 'Copied';
  setTimeout(() => { btn.textContent = prev; }, 1400);
}

function legacyCopy(text) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '-1000px';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch (_) { return false; }
}

export function copyText(text, btn, onLabel, offLabel) {
  const done = (ok) => { if (ok) flash(btn, onLabel, offLabel); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => done(true), () => done(legacyCopy(text)));
  } else {
    done(legacyCopy(text));
  }
}

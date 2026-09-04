/* Getting a generated image out of the page — to the clipboard, or to disk.
 *
 * The daemon answers a draw with base64 in a JSON line rather than `image/png`
 * bytes, because the seed has to ride back with the picture (see
 * `npcd::guest_routes`). So every consumer starts from a base64 string and has
 * to turn it into something a browser will treat as a file.
 */

/** Base64 PNG → a `Blob`.
 *
 * `atob` rather than `fetch("data:image/png;base64,…")`. A 512×512 portrait is
 * ~700 KB of base64, and handing that to `fetch` makes the whole image a URL —
 * a request the browser is entitled to treat like any other, subject to
 * whatever policy the page is under. `atob` is synchronous, has no such
 * surface, and is what the bytes needed in the first place.
 */
export function pngBlob(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: 'image/png' });
}

/** Put an image on the clipboard. Resolves to whether it worked.
 *
 * **The blob is passed as a promise, not as a value**, and that is a Safari
 * requirement rather than a style choice: Safari only allows a clipboard write
 * inside the user gesture that triggered it, and awaiting anything before
 * `write` loses the gesture. Handing `ClipboardItem` a promise lets the write
 * start synchronously and settle afterwards. Chrome accepts the same form.
 *
 * PNG is the only image type clipboards broadly accept, which is what the
 * daemon produces anyway.
 *
 * Returns `false` rather than throwing on the browsers that have no
 * `ClipboardItem` at all — Firefox needed a flag for this until recently, and a
 * button that reported success there would be lying.
 */
export async function copyImage(blob) {
  if (!navigator.clipboard || typeof ClipboardItem === 'undefined') return false;
  try {
    await navigator.clipboard.write([
      new ClipboardItem({ [blob.type || 'image/png']: Promise.resolve(blob) }),
    ]);
    return true;
  } catch (_) {
    return false;
  }
}

/** Save a blob as `filename`.
 *
 * An **object URL**, not a `data:` URL. The href used to carry the whole image
 * as base64, which is ~600 KB sitting in a DOM attribute and is the form mobile
 * Safari handles worst — it commonly navigates to the image instead of saving
 * it, and large `data:` URLs can be refused outright.
 *
 * The URL is revoked on the next tick rather than immediately: revoking inside
 * the same frame as the click races the browser's own fetch of it, and the
 * download silently does nothing.
 *
 * **On iOS this may still open the image rather than save it.** `download` is
 * advisory there, and the platform's answer is long-press → Save to Photos.
 * Nothing here can change that, which is why the button says "Download" and not
 * "Save to your device".
 */
export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

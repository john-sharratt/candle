/* Images — a prompt box over the co-resident image guest.
 *
 * The portrait route (`/v1/npc/:id/portrait`) draws *for a character*: it builds
 * the prompt from that character's description and writes the result back onto
 * the record. This page is the same guest with none of that — you write the
 * prompt, you get the picture, and nothing is stored. It exists so the estate's
 * own generator can be used for artwork that is not a portrait.
 *
 * ── Why there is no negative prompt ─────────────────────────────────────────
 *
 * Z-Image-Turbo is guidance-distilled: the model card runs it at
 * `guidance_scale=0.0`, so a step is ONE forward with no unconditioned branch
 * beside it. A negative prompt works by pushing away from that branch, and with
 * no branch there is nothing to push away from — the field would be a box that
 * accepts text and changes nothing, which is worse than its absence. The
 * daemon's `ImageRequest` has no such field for the same reason.
 *
 * The framing that a negative prompt used to carry is stated positively
 * instead: see the style presets below, which is where "head and shoulders"
 * belongs anyway.
 *
 * ── Why the sizes are the ones offered ──────────────────────────────────────
 *
 * Every side is a multiple of 16 because the daemon refuses anything else, and
 * it is right to: the latent is the image over 8 and the transformer patches
 * that by 2, so a remainder is rounded away somewhere in the middle rather than
 * refused, and the caller gets back a size it did not ask for.
 *
 * The sizes above 512 are the model's own resolution buckets. Both sides stay
 * inside the 768–1280 band it was trained on, which is why there is no 16:9
 * here: 1024×576 looks reasonable and puts the short side well below the grid.
 *
 * **What used to bound this was the decoder, and no longer does.** Its working
 * set grew with the image's AREA and came from the CUDA pool, which a
 * co-resident guest gets little of — so the first large draw of a session died
 * out of memory and the retry succeeded. The decode is tiled now
 * (`guest::tiled`), so its peak is one 512×512 tile whatever the output
 * is, and a bigger image costs proportionally more time instead of failing.
 *
 * What still grows with area is the DRAIN — every second of it is the whole
 * estate not thinking — so the ceiling here is courtesy, not capacity.
 *
 * ── What a reference picture is, and is not ─────────────────────────────────
 *
 * Dropping a picture in makes the draw START from it instead of from noise: it
 * is encoded, noise is mixed in at the strength the Reference dial names, and
 * the denoise finishes from there. So it carries composition, pose, palette and
 * framing across, and the prompt still decides what the picture is OF.
 *
 * It is NOT instruction editing. "Give him a hat" is not a thing that can be
 * asked here, because the model is never told what changed — it is handed a
 * partly dissolved picture and asked to complete it. That capability wants
 * reference conditioning and weights this deployment does not have, so the copy
 * below says "start from" and never "edit".
 *
 * ── Why the dials are these two ─────────────────────────────────────────────
 *
 * Reference and Structure are the sampler's real levers and there are no others
 * to offer. In particular there is deliberately NO guidance / prompt-strength
 * dial: Turbo runs at `guidance_scale = 0.0` with no unconditioned branch, so a
 * slider weighing the prompt against one would be weighing against something
 * that is never computed — a control that moves and changes nothing, which is
 * the same bug as the negative prompt field above.
 */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { can } from '../lib/router.js';
import { empty, toast } from '../lib/ui.js';
import { copyText, flash } from '../lib/clip.js';
import { copyImage, download, pngBlob } from '../lib/img.js';

/* Offered sizes. Square first because it is what most artwork here wants, then
 * the two aspects, all multiples of 16. */
const SIZES = [
  { label: '512 × 512', w: 512, h: 512, note: 'square' },
  { label: '1024 × 1024', w: 1024, h: 1024, note: 'square, the model’s native size' },
  { label: '1248 × 832', w: 1248, h: 832, note: 'wide 3:2 — the model’s own bucket' },
  { label: '832 × 1248', w: 832, h: 1248, note: 'tall 2:3 — the model’s own bucket' },
  { label: '384 × 384', w: 384, h: 384, note: 'square, quicker' },
  { label: '640 × 384', w: 640, h: 384, note: 'landscape' },
  { label: '384 × 640', w: 384, h: 640, note: 'portrait' },
  { label: '256 × 256', w: 256, h: 256, note: 'thumbnail' },
];

/* Framings worth having as one click, stated positively.
 *
 * Appended to the prompt rather than replacing it, and shown in the box so what
 * is sent is what you can see — a preset that edited the request invisibly
 * would make a bad result impossible to attribute. */
const PRESETS = [
  { label: 'Portrait', add: 'character portrait, head and shoulders, centred, detailed face' },
  { label: 'Full figure', add: 'full body character, standing, plain background' },
  { label: 'Environment', add: 'wide establishing shot, atmospheric lighting' },
  { label: 'Item', add: 'a single object, centred, plain background, product lighting' },
  { label: 'Painterly', add: 'painterly, oil on canvas, visible brushwork' },
  { label: 'Photographic', add: 'photographic, natural lighting, shallow depth of field' },
];

/* Turbo's schedule is trained at eight steps; the default here is twenty-four.
 *
 * **Steps buy detail, not different people.** Measured: one prompt at three
 * seeds returns the same face at eight steps and at twelve alike, and a given
 * seed's face does not change between the two. The identity comes from the
 * prompt — the same seed with a described subject returns someone else entirely.
 * So if every draw is giving you the same person, the number to change is not
 * this one; it is the words. Name an age, a build, a nose.
 *
 * What twenty-four buys is finish, and it is a deliberate trade rather than a
 * free one: it is three times the trained count and roughly three times the
 * denoise, and every second of it is time the whole estate stops thinking. This
 * page is for artwork somebody is going to keep, so it spends that. The portrait
 * route does not — it draws for a record, per character, and stays at twelve. */
const STEPS = [8, 12, 16, 24];

/* The dials' defaults and ranges, mirroring the daemon's own bounds
 * (`guest::work`'s `MAX_REFERENCE_HOLD`, `MIN_SHIFT`, `MAX_SHIFT`).
 *
 * `hold` stops at 0.95 rather than 1 because the daemon refuses a full hold:
 * there would be no distance left to walk, so the draw would cost the whole
 * estate a drain to hand back the picture that was uploaded.
 *
 * `shift` is the schedule's own, and 3.0 is what the model's step count is
 * distilled against — so the page sends it ONLY once it has been moved, and a
 * deployment that set its own in `guests.yaml` keeps it until then. */
const HOLD = { min: 0, max: 0.95, step: 0.05, def: 0.25 };
const SHIFT = { min: 1, max: 8, step: 0.5, def: 3 };

/* What a reference upload may be, matching `refimage`'s own limits — checked
 * here so a file that was never going to work says so instantly instead of
 * after a megabyte has crossed the wire. */
const REF_TYPES = ['image/png', 'image/jpeg'];
const REF_MAX_BYTES = 8 * 1024 * 1024;

/* What the box starts with, so the page is one press from a picture.
 *
 * A *filled* box rather than a placeholder, which is the whole point: a
 * placeholder demonstrates nothing until you have already written something, and
 * the first thing anyone wants from a generator is to see what it does. Select
 * all and type over it to start your own.
 *
 * It is also the house style argued by example. Everything that makes this model
 * behave is in here — a described subject rather than a role, the setting, the
 * light, the lens — which is the difference between the same face every seed and
 * a character. See the note under the box, and `npcd::guest_routes` for why the
 * description and not the seed is what decides who you get. */
const DEFAULT_PROMPT =
  'a Vietnamese female warrior in a modern battlefield in 2300 with spider drones ' +
  'fighting humans everywhere and a live weapon fire in the background of dune ' +
  'desert.  the Asian warrior has tight fitting armor with a long heavily armored ' +
  'traditional Vietnamese dress and cloak that is practical in battle but keeps ' +
  'culture, she holds a futuristic glowing power sword in one hand and a laser ' +
  'pistol in the other, on her belt to the side is shield emitter with the glow of ' +
  'its field emanating around her, photographic, natural lighting, shallow depth ' +
  'of field';

export async function render() {
  const el = h('div', { class: 'page wide' });

  let busy = false;
  let size = SIZES[0];
  let steps = 24;
  let seed = null; // null → the daemon draws one and reports it back
  let restricted = false; // admin-only; see the checkbox at the panel's foot
  let last = null; // { png_base64, width, height, seed, prompt, ms, cutout }
  let showingCut = false; // whether the stage is showing the lifted version
  let reference = null; // { base64, url, name } — the picture a draw starts from
  let hold = HOLD.def;
  let shift = SHIFT.def;
  let shiftMoved = false; // until then the deployment's own schedule stands

  const prompt = h('textarea', {
    class: 'textarea',
    rows: 7,
    // Shown if the box is ever emptied — the shorter shape of the same advice.
    placeholder:
      'A weathered ship captain in his fifties, scarred left hand, salt-bleached coat…',
    onKeyDown: (e) => {
      // Ctrl/⌘-Enter generates, which is what a prompt box is expected to do.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); go(); }
    },
  });
  prompt.value = DEFAULT_PROMPT;

  const seedInput = h('input', {
    class: 'input mono', type: 'text', placeholder: 'random',
    style: 'width:11ch',
    onInput: (e) => {
      const v = e.target.value.trim();
      seed = v === '' ? null : (Number.isFinite(+v) ? +v : null);
    },
  });

  /* ── the reference ──────────────────────────────────────────────────────
   *
   * A `<label>` around a hidden file input, so a click opens the picker and the
   * keyboard reaches it, with the drag handlers on the same element. Building
   * it out of a `<div>` and a click handler would have cost both of those and
   * bought nothing. */
  const refInput = h('input', {
    type: 'file',
    accept: REF_TYPES.join(','),
    class: 'ref-file',
    onChange: (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) takeReference(f);
      // Cleared so choosing the *same* file twice still fires a change.
      e.target.value = '';
    },
  });
  const refBody = h('div', { class: 'ref-body' });
  const refZone = h('label', { class: 'ref-zone' }, refInput, refBody);

  /* Drag over the whole zone rather than the thumbnail, so there is a target
   * worth hitting on a laptop trackpad. `preventDefault` on dragover is what
   * makes the element a drop target at all — without it the browser navigates
   * to the file instead. */
  refZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    refZone.classList.add('over');
  });
  refZone.addEventListener('dragleave', () => refZone.classList.remove('over'));
  refZone.addEventListener('drop', (e) => {
    e.preventDefault();
    refZone.classList.remove('over');
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) takeReference(f);
  });

  function takeReference(file) {
    // Checked here as well as at the daemon, because a file that was never
    // going to work should say so before a megabyte crosses the wire.
    if (!REF_TYPES.includes(file.type)) {
      toast('A reference has to be a PNG or a JPEG.', 'warn');
      return;
    }
    if (file.size > REF_MAX_BYTES) {
      toast(`That file is ${(file.size / 1e6).toFixed(1)} MB and the limit is 8 MB.`, 'warn');
      return;
    }
    const fr = new FileReader();
    fr.onerror = () => toast('That file could not be read.', 'warn');
    fr.onload = () => {
      const url = String(fr.result || '');
      // `data:image/png;base64,AAAA…` — the daemon wants the payload alone.
      const comma = url.indexOf(',');
      if (comma < 0) { toast('That file could not be read.', 'warn'); return; }
      reference = { base64: url.slice(comma + 1), url, name: file.name };
      renderReference();
    };
    fr.readAsDataURL(file);
  }

  function clearReference() {
    reference = null;
    renderReference();
  }

  function renderReference() {
    if (!reference) {
      mount(refBody,
        h('div', { class: 'ref-empty' },
          h('div', { class: 'ref-icon' }, '⇪'),
          h('div', {}, h('strong', {}, 'Drop a picture'), ' or click to choose'),
          h('div', { class: 'dim' }, 'The draw starts from it instead of from noise')));
    } else {
      mount(refBody,
        h('img', { class: 'ref-thumb', src: reference.url, alt: reference.name }),
        h('div', { class: 'ref-meta' },
          h('div', { class: 'ref-name' }, reference.name),
          h('div', { class: 'dim' }, 'Cropped to the size above'),
          h('button', {
            class: 'btn sm ghost',
            // Inside a `<label>`, so a plain click would re-open the picker on
            // the way up. This is the one place that has to be stopped.
            onClick: (e) => { e.preventDefault(); e.stopPropagation(); clearReference(); },
          }, 'Remove')));
    }
    holdDial.setEnabled(!!reference);
  }

  /* A labelled slider with a live readout.
   *
   * Two of these, so it is a function. The readout is the point: a bare range
   * input tells you where the handle is and not what that means, and both of
   * these dials are quantities somebody needs to be able to state. */
  function dial({ label, cfg, hint, fmt, onInput }) {
    const value = h('span', { class: 'dial-val mono' }, fmt(cfg.def));
    const input = h('input', {
      class: 'dial', type: 'range',
      min: cfg.min, max: cfg.max, step: cfg.step, value: cfg.def,
      // The state is updated before the readout is painted, so a `fmt` that
      // reads that state — Structure's does, to say whether it is still the
      // deployment's — describes where the dial now is rather than where it was.
      onInput: (e) => { const v = +e.target.value; onInput(v); mount(value, fmt(v)); },
    });
    const row = h('div', { class: 'dial-row' },
      h('div', { class: 'dial-head' }, h('label', { class: 'lbl' }, label), value),
      input,
      h('div', { class: 'dim dial-hint' }, hint));
    return {
      row,
      setEnabled(on) {
        input.disabled = !on;
        row.classList.toggle('off', !on);
      },
    };
  }

  const holdDial = dial({
    label: 'Reference',
    cfg: HOLD,
    hint: 'How much of the dropped picture survives. Low re-imagines it; high keeps it and only refines.',
    fmt: (v) => `${Math.round(v * 100)}%`,
    onInput: (v) => { hold = v; },
  });

  const shiftDial = dial({
    label: 'Structure',
    cfg: SHIFT,
    hint: 'Where the steps are spent. Higher decides the composition; lower resolves the detail.',
    fmt: (v) => (shiftMoved || v !== SHIFT.def ? v.toFixed(1) : `${v.toFixed(1)} · default`),
    onInput: (v) => { shift = v; shiftMoved = true; },
  });

  const out = h('div', { class: 'img-out' });
  const status = h('div', { class: 'dim' }, 'Ready.');
  const goBtn = h('button', { class: 'btn primary', onClick: () => go() }, 'Generate');
  // The keyboard hint is hidden on a touch device, where there is no Ctrl key
  // and the words are just clutter in a row that has to fit a phone.
  const hint = h('span', { class: 'dim img-hint' }, 'Ctrl+Enter');

  /* The bar. Built once and kept, because `setProgress` runs per step and
   * rebuilding the node each time would restart the width transition. */
  const bar = h('div', { class: 'bar-fill' });
  const barLabel = h('div', { class: 'bar-label dim mono' });
  const barBox = h('div', { class: 'bar' }, bar);

  /* What the daemon is doing, as a fraction and a phrase.
   *
   * `total` is the guest's own — denoise steps plus the decode — so the bar
   * never reaches the end while there is still work. The decode is the last
   * unit and is announced when it STARTS, so the bar rests just short of full
   * while it runs; that is the honest place for it, since the event which
   * completes it is the finished picture. */
  const setProgress = (ev) => {
    if (!ev) { bar.style.width = '0%'; mount(barLabel, 'Waiting for the engine…'); return; }
    if (ev.event === 'loading') {
      // No fraction to show: this is the model crossing the link, and its
      // length depends on what has to be evicted first. An indeterminate stripe
      // rather than a number nobody could compute.
      barBox.classList.add('idle');
      bar.style.width = '100%';
      mount(barLabel, 'Loading the model…');
      return;
    }
    if (ev.event !== 'step' || !ev.total) return;
    barBox.classList.remove('idle');
    const frac = Math.max(0, Math.min(1, ev.done / ev.total));
    bar.style.width = `${(frac * 100).toFixed(1)}%`;
    // One word for both phases. The stream distinguishes `denoising` from
    // `decoding` because they are different work and a reader of the API should
    // be able to tell them apart — but they are the daemon's vocabulary, not the
    // reader's, and somebody watching a picture appear does not need to be told
    // which half of the pipeline is running.
    // The step count only while there are steps being counted. The other
    // phases — reading a reference, decoding — are single long operations with
    // no interior number, and "step 0 of 7" for one of them reads as stuck.
    mount(barLabel, ev.what === 'denoising'
      ? `Generating · step ${ev.done} of ${ev.total - 1}`
      : 'Generating…');
  };

  const setBusy = (b, msg) => {
    busy = b;
    goBtn.disabled = b;
    // The label does not change — a control whose name moves under the pointer
    // is its own bug. Progress belongs in the status line and the frame.
    mount(status, msg || (b ? 'Generating…' : 'Ready.'));
    if (b) {
      setProgress(null);
      mount(out, h('div', { class: 'img-frame busy' },
        h('div', { class: 'bar-wrap' }, barBox, barLabel)));
    }
  };

  async function go() {
    const text = prompt.value.trim();
    if (!text) { toast('A prompt is the one thing with no default.', 'warn'); return; }
    if (busy) return;
    setBusy(true);
    const t0 = performance.now();
    try {
      const body = { prompt: text, width: size.w, height: size.h, steps };
      if (seed != null) body.seed = seed;
      // Sent only when set: the daemon honours it for an admin and refuses it
      // for anyone else, so this is a request, not a grant — the same shape as
      // `?reveal=1`. The default omits the field and draws exactly as before.
      if (restricted) body.lora = 'restricted';
      if (reference) {
        body.reference = reference.base64;
        body.reference_hold = hold;
      }
      // Only once it has been moved, so a deployment that set its own shift in
      // `guests.yaml` keeps it until somebody actually asks for another.
      if (shiftMoved) body.shift = shift;
      const r = await API.generateImage(body, setProgress);
      last = { ...r, prompt: text, ms: Math.round(performance.now() - t0), cutout: null };
      showingCut = false;
      setBusy(false, `Drawn in ${(last.ms / 1000).toFixed(1)}s · seed ${r.seed}`);
      showResult();
    } catch (e) {
      setBusy(false, 'Failed.');
      showError(e);
    }
  }

  function showError(e) {
    const msg = String((e && (e.detail || e.error || e.message)) || e);
    /* The two failures a caller can actually act on, named. Everything else is
     * shown verbatim rather than dressed up as one of them.
     *
     * A draw can fail on either side of the stream opening, and the two carry
     * their reason differently — that is a property of streaming, not an
     * inconsistency to paper over. Before the first line it is an ordinary
     * status with a body. After it the status is long gone, so the daemon says
     * why in a terminal `error` line and the code arrives as `e.error`. Both
     * are checked here; testing only one silently loses half the failures.
     *
     * Prose is the last resort and only for the driver's own out-of-memory,
     * which has no code of its own — everything else is matched on a code or a
     * status, so rewording a message cannot quietly stop this from matching. */
    const code = e && e.error;
    const oom = code === 'no_room' || /out of memory/i.test(msg);
    const noGuest = code === 'no_guest' || (e && e.status === 501);
    // The content check. Its own case because it is the one refusal the writer
    // can act on directly, and because reading it as "the draw failed" would
    // send someone looking for a fault in the daemon.
    const declined = code === 'prompt_declined' || (e && e.status === 403);
    // "The checker could not run" is a 503 like a loading engine, and says so
    // itself, so it falls through to the generic case with its own message
    // rather than being dressed up as a missing guest.
    if (declined) {
      mount(out, empty('⊘', 'That prompt was not drawn', msg));
      return;
    }
    mount(out, empty(
      oom ? '▨' : '⚠',
      oom ? 'Not enough room for that size' : noGuest ? 'No generator to borrow' : 'The draw failed',
      oom
        ? 'The decoder’s working set grows with the image’s area and comes from ' +
          'what the engine’s reservation leaves behind. Try a smaller size.'
        : noGuest
          ? msg + ' — a daemon serves images between its engine’s waves, so it ' +
            'needs both an engine that has finished loading and an `image:` ' +
            'section in its `guests.yaml`.'
          : msg));
  }

  /* What the buttons act on: the lifted version when it is the one on screen,
   * the drawn one otherwise. Copying or downloading something other than what
   * is being looked at is the kind of bug nobody reports and everybody hits. */
  const shownPng = () => (showingCut && last && last.cutout ? last.cutout : last && last.png_base64);

  function showResult() {
    if (!last) return;
    const src = `data:image/png;base64,${shownPng()}`;
    mount(out,
      // The checkerboard only when a background has actually been lifted —
      // behind an opaque picture it would just be an odd border.
      h('div', { class: 'img-frame' + (showingCut ? ' alpha' : '') },
        h('img', { src, alt: last.prompt, class: 'img-result' })),
      // `img-meta` owns its own flex and wrapping — it holds seven children,
      // three of them buttons, and on a phone they cannot sit on one line. The
      // facts and the actions are two groups so a wrap breaks between them
      // rather than mid-caption.
      h('div', { class: 'img-meta' },
        h('div', { class: 'img-facts' },
          h('span', { class: 'dim mono' }, `${last.width}×${last.height}`),
          h('span', { class: 'dim mono' }, `seed ${last.seed}`),
          h('span', { class: 'dim mono' }, `${(last.ms / 1000).toFixed(1)}s`),
          showingCut ? h('span', { class: 'chip' }, 'background removed') : null),
        h('div', { class: 'img-acts' },
          // The two that act on the picture come first and are not ghosts: they
          // are what somebody who liked a draw reaches for, and the rest of the
          // row is bookkeeping.
          h('button', {
            class: 'btn sm',
            title: 'Put the picture on the clipboard, to paste straight into something else',
            onClick: (e) => copyPicture(e.currentTarget),
          }, 'Copy image'),
          h('button', {
            class: 'btn sm',
            title: 'Save the picture as a PNG',
            // A button rather than an `<a download>` carrying the whole image as
            // a data URL — see `lib/img.js` for why that shape fails on mobile.
            onClick: () => {
              const tag = showingCut ? '-cutout' : '';
              download(pngBlob(shownPng()), `${slug(last.prompt)}-${last.seed}${tag}.png`);
            },
          }, 'Download'),
          // Third because it is an action *on* the picture like the two above,
          // and before the bookkeeping. Its label is the state it moves to.
          h('button', {
            class: 'btn sm',
            title: showingCut
              ? 'Put the background back'
              : 'Separate the subject and leave the background transparent',
            onClick: (e) => toggleCutout(e.currentTarget),
          }, showingCut ? 'Restore background' : 'Remove background'),
          // The seed is the only way back to an image you liked, so it is one
          // click to pin rather than something to retype from the caption.
          h('button', {
            class: 'btn sm ghost',
            title: 'Put this seed in the box, so the next draw repeats it',
            onClick: () => { seedInput.value = String(last.seed); seed = last.seed; toast('Seed pinned.'); },
          }, 'Reuse seed'),
          h('button', {
            class: 'btn sm ghost',
            // `copyText` takes the button and flashes it itself; it returns
            // nothing. Calling `.then` on it — which this did — throws, so the
            // button reported nothing and copied nothing.
            onClick: (e) => copyText(last.prompt, e.currentTarget, 'Copied', 'Copy prompt'),
          }, 'Copy prompt'))));
  }

  /* Copy the picture itself, and say honestly when the browser will not.
   *
   * `copyImage` resolves to whether it worked rather than throwing, because the
   * failure is a platform one — Firefox wanted a flag for `ClipboardItem` until
   * recently — and a button that flashed "Copied" over an empty clipboard is
   * worse than one that admits it. The fallback offered is the one that always
   * works: the file. */
  /* Lift the background, or put it back.
   *
   * The lifted version is kept on `last` once it has been fetched, so toggling
   * back and forth is free and the daemon is asked exactly once per picture.
   *
   * The two refusals are told apart deliberately. A 422 is a fact about the
   * picture — its edges are not one colour, so there is no background to lift —
   * and saying "failed" for it would send somebody looking for a fault. Its own
   * text is the honest thing to show, so it is shown. */
  /* Fetch the lifted version once and keep it on `last`, so toggling back and
   * forth is free and the daemon is asked exactly once per picture. Resolves
   * to the failure text, or null when it worked. */
  async function fetchCutout() {
    if (!last || last.cutout) return null;
    try {
      const r = await API.cutout({ png_base64: last.png_base64 });
      last.cutout = r.png_base64;
      last.lifted = r.lifted;
      return null;
    } catch (e) {
      return String((e && (e.detail || e.error || e.message)) || e);
    }
  }

  async function toggleCutout(btn) {
    if (!last || busy) return;
    if (showingCut) {
      showingCut = false;
      showResult();
      return;
    }
    if (last.cutout) {
      showingCut = true;
      showResult();
      return;
    }
    const label = btn.textContent;
    btn.disabled = true;
    mount(btn, 'Lifting…');
    const failure = await fetchCutout();
    if (failure) {
      btn.disabled = false;
      mount(btn, label);
      toast(failure, 'warn');
      return;
    }
    showingCut = true;
    showResult();
    // Said out loud, because a picture whose background was already nearly
    // nothing looks unchanged and the button would seem not to have worked.
    if (typeof last.lifted === 'number' && last.lifted < 0.02) {
      toast('There was almost nothing to lift — this picture is nearly all subject.');
    }
  }

  async function copyPicture(btn) {
    if (!last) return;
    const ok = await copyImage(pngBlob(shownPng()));
    if (ok) {
      flash(btn, 'Copied', 'Copy image');
      return;
    }
    toast('This browser will not copy an image — use Download instead.', 'warn');
  }

  const sizeRow = h('div', { class: 'row wrap' },
    ...SIZES.map((s) => {
      const b = h('button', {
        class: 'btn sm' + (s === size ? ' primary' : ' ghost'),
        title: s.note,
        onClick: () => {
          size = s;
          [...sizeRow.children].forEach((c, i) =>
            (c.className = 'btn sm' + (SIZES[i] === size ? ' primary' : ' ghost')));
        },
      }, s.label);
      return b;
    }));

  const stepRow = h('div', { class: 'row wrap' },
    ...STEPS.map((n) => {
      const b = h('button', {
        class: 'btn sm' + (n === steps ? ' primary' : ' ghost'),
        title: n === 8
          ? 'What Turbo’s schedule is trained at — quickest'
          : n === 24
            ? 'The default — the most finish, and the longest drain'
            : 'Fewer steps, quicker. It will not change who you get.',
        onClick: () => {
          steps = n;
          [...stepRow.children].forEach((c, i) =>
            (c.className = 'btn sm' + (STEPS[i] === steps ? ' primary' : ' ghost')));
        },
      }, String(n));
      return b;
    }));

  mount(el,
    h('div', { class: 'hd' },
      h('div', {}, h('h1', {}, 'Images'),
        h('div', { class: 'sub' },
          'Draws with the co-resident image guest — the same generator the ' +
          'portrait button uses. Nothing here is saved to a character.'))),
    h('div', { class: 'img-page' },
      h('div', { class: 'img-controls card' },
        h('label', { class: 'lbl' }, 'Prompt'),
        prompt,
        // Where the variety actually comes from, said once, next to the box it
        // applies to. The model's face is decided by the description and barely
        // at all by the seed, so "generate again" on a vague prompt returns the
        // same person — which reads as a broken seed unless the page says this.
        h('div', { class: 'dim' },
          'Describe the person, not just the role — age, build, a feature. ' +
          'Re-rolling the seed changes the setting, not the face.'),
        h('div', { class: 'row wrap' },
          ...PRESETS.map((p) => h('button', {
            class: 'btn sm ghost',
            title: `Append: ${p.add}`,
            onClick: () => {
              const cur = prompt.value.trim();
              prompt.value = cur ? `${cur}, ${p.add}` : p.add;
              prompt.focus();
            },
          }, p.label))),
        h('label', { class: 'lbl' }, 'Size'),
        sizeRow,
        // The reference and its dial together, because the dial means nothing
        // without the picture and is disabled until there is one.
        h('label', { class: 'lbl' }, 'Reference picture'),
        refZone,
        holdDial.row,
        // Steps and Seed side by side, and stacked once there is no room for
        // two. A flex row would keep them side by side and squeeze the seed box
        // to nothing; `auto-fit` drops to one column instead.
        h('div', { class: 'img-pair' },
          h('div', {},
            h('label', { class: 'lbl' }, 'Steps'),
            stepRow),
          h('div', {},
            h('label', { class: 'lbl' }, 'Seed'),
            seedInput)),
        shiftDial.row,
        h('div', { class: 'img-go' }, goBtn, hint, status),
        /* Admin only, and visibility is discretion rather than access control —
         * the daemon re-checks the role on the request and refuses the flag
         * from anyone else, exactly like the hidden-document reveal. Drawing
         * restricted routes to the deployment's alternative checkpoint and
         * stands on the admin's own judgement in place of the prompt check. */
        can('admin')
          ? h('label', { class: 'img-restricted dim' },
              h('input', {
                type: 'checkbox',
                onChange: (e) => { restricted = e.target.checked; },
              }),
              h('span', {}, 'Restricted'))
          : null),
      h('div', { class: 'img-stage' }, out)));

  // The zone starts in its empty state and the hold dial starts disabled —
  // painted here rather than in the tree above, so the two states are built by
  // exactly one function and cannot drift apart.
  renderReference();

  mount(out, empty('▨', 'Nothing drawn yet',
    'Write a prompt and press Generate. The first draw of a session also loads ' +
    'the model, so it takes several seconds longer than the ones after it.'));

  return el;
}

/* A filename from the prompt: lowercase words, hyphenated, short enough to read
 * in a downloads list. */
function slug(s) {
  return String(s || 'image')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 48) || 'image';
}

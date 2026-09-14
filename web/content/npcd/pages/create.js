/* Creating an NPC (§29) — three steps, each with a working default.
 *
 * The description IS the character: it becomes the identity section in the
 * system prompt. The **portrait** is drawn from a prompt, which is a different
 * sentence — a description is written to be read, a prompt to be drawn — and
 * the face step shows that prompt so it can be adjusted.
 *
 * Where the prompt starts from, and the picture in the frame, both come from
 * the personality when it was authored with them (`portrait:` in its YAML; see
 * `npcd::personality_portrait`). So choosing Keeper opens the face step already
 * looking like Keeper, with the words that drew it — and Generate makes another
 * one from those words rather than from a blank field.
 *
 * Visibility and tags are deliberately absent; they belong to an existing
 * character (§30). */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { pngBlob } from '../lib/img.js';
import { disclosure } from '../lib/lazy.js';
import { can, go } from '../lib/router.js';
import { onReveal, revealing } from '../lib/reveal.js';
import { toast, empty } from '../lib/ui.js';

/* Stable ids, so the selector that was open can be found again after a repaint
 * and re-opened over the new list — see the `onReveal` handler at the foot. */
const SEL_WORLD = 'create-world';
const SEL_PERSONALITY = 'create-personality';

export async function render() {
  const el = h('div', { class: 'page', style: 'max-width:900px' });

  const draft = {
    // Both references start empty and are filled from the first listed
    // document. A hardcoded default would name a personality this daemon may
    // not have — they are files in the mind, not fixed rows.
    name: '', world_id: '', personality_id: '',
    /* Whether each field is the model's work or the author's. The description
     * has tracked this all along; the name now does too, because changing world
     * or personality recasts what was *generated* and must leave what was
     * *typed* alone — and without an origin for the name there is no way to
     * tell a name somebody chose from one that was invented for them. */
    name_origin: 'generated',
    description: '', description_origin: 'generated',
    portrait: null, portrait_origin: null,
    /* The words the portrait is drawn from, and whether a person has touched
     * them. Held on the draft so stepping back to the description and forward
     * again does not discard an edit — and `portrait_prompt_touched` is what
     * stops the personality's authored prompt overwriting one. */
    portrait_prompt: '', portrait_prompt_touched: false,
    /* The words that drew the portrait now in the frame. The face step compares
     * the current prompt against this to decide whether arriving there owes a
     * fresh draw — so rewriting the description and coming back redraws, and
     * merely passing through does not. */
    portrait_drawn_from: '',
    /* Which personality's authored portrait is currently in the frame, so
     * changing the personality can replace a default nobody chose while leaving
     * a picture somebody did choose alone. */
    portrait_from_personality: '',
    // The file to upload, or the intent to draw. Never both — choosing either
    // in the face step clears the other, because an upload outranks the
    // generator permanently and a draft holding both would have to pick.
    portrait_file: null, portrait_generate: false,
  };
  let step = 1;

  /* Fetched **once**, hidden documents and all when the viewer is an admin.
   *
   * Holding RIGHT ALT then only changes which of them are rendered, so the
   * reveal is instant and needs no round trip — an earlier version refetched on
   * every press and release, which made the key feel broken while the request
   * was in flight. A non-admin gets the discreet list from the daemon whatever
   * this asks for, so there is nothing here for one to hold.
   *
   * The hidden entries do sit in an admin's page memory unrevealed. That is the
   * same bargain the flag already makes: this is discretion — keeping them out
   * of a dropdown on a screen share — and never secrecy, since the same admin
   * can reveal them with a keypress and fetch any of them by id regardless. */
  const [worlds, personalities] = await Promise.all([
    API.listWorlds('', can('admin')).then((r) => r.worlds || []).catch(() => []),
    API.listPersonalities('', can('admin')).then((r) => r.personalities || []).catch(() => []),
  ]);

  /* Which **worlds** to render, given whether the key is down.
   *
   * A hidden world that is *already selected* stays in the list either way.
   * Releasing the key must not silently undo a choice the admin deliberately
   * made — the selection would fall back to the first entry and the form would
   * quietly point at a different world than the one on screen a moment ago. */
  const shownWorlds = () =>
    worlds.filter((w) => revealing() || !w.hidden || w.world_id === draft.world_id);

  /* Which **personalities** the selected world casts.
   *
   * The world names its cast in `personalities:`, the same place and the same
   * shape as `selects` — a world is a filter, and what it admits is written on
   * the world. Changing the world therefore changes the cast entirely rather
   * than narrowing it.
   *
   * Not `hidden`, deliberately: that flag answers "should this appear in a
   * listing", a question about screen shares, revealed by a keypress. This
   * answers which world a character is *of*, which no keypress should change —
   * Cindy is Earth's whether or not a key is held.
   *
   * A world that names no cast admits everyone. That is the standing default,
   * and it is what keeps adding the key to one world from stranding every
   * other: a world nobody has cast yet still offers a full list rather than an
   * empty one. */
  const hostable = (p, worldId) => {
    const world = worlds.find((w) => w.world_id === worldId);
    const cast = world && world.personalities;
    return !Array.isArray(cast) || cast.includes(p.personality_id);
  };
  /* Cast, **and** hidden, on the same terms the world list uses.
   *
   * `hostable` above answers which world a character is of, and deliberately
   * ignores `hidden`. That left nothing else asking, so a hidden personality
   * was on screen for any admin whether or not a key was held — the one thing
   * the flag exists to prevent, and the opposite of how the world selector two
   * lines up behaves. A hidden entry that is *already selected* stays, for the
   * reason `shownWorlds` gives: releasing the key must not silently move the
   * form to a different character than the one on screen a moment ago. */
  const shownPersonalities = () =>
    personalities.filter(
      (p) =>
        hostable(p, draft.world_id)
        && (revealing() || !p.hidden || p.personality_id === draft.personality_id),
    );
  /* The default is the first **visible** entry, never a hidden one.
   *
   * `worlds` now holds the hidden entries too, so taking `[0]` would pick one
   * whenever a hidden id sorts first — and a form that opens with `earth`
   * already chosen defeats the flag entirely, since the name would sit in the
   * closed selector with nobody holding a key. Only if every entry is hidden
   * does the first of those stand, which is an authored state, not an
   * accident. */
  const firstWorld = worlds.find((w) => !w.hidden) || worlds[0];
  if (firstWorld) draft.world_id = firstWorld.world_id;
  // The personality default follows the world, by the same rule the list does:
  // opening on a pairing the daemon would refuse is a form that starts wrong.
  // Visible first, for the reason the world default is: a form that opens with
  // a hidden name already in the closed selector defeats the flag entirely.
  const firstPersonality = personalities.find((p) => hostable(p, draft.world_id) && !p.hidden)
    || personalities.find((p) => hostable(p, draft.world_id))
    || personalities[0];
  if (firstPersonality) draft.personality_id = firstPersonality.personality_id;

  const body = h('div', {});
  const foot = h('div', { class: 'row', style: 'justify-content:flex-end;gap:9px;margin-top:22px' });

  /* The title carries the name once there is one, so the page says who it is
   * about rather than only what kind of page it is. The name is generated on
   * arrival and editable throughout, so this is a node that gets rewritten
   * rather than a string built once — see `redrawTitle`. */
  const title = h('h1', {});

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, title,
      h('div', { class: 'sub' }, 'Two steps. Both have a default, so Next then Create is a valid character.')),
    h('div', { class: 'steps' },
      ['Identity', 'Face'].map((s, i) =>
        h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
          (step > i + 1 ? '✓' : '①②'[i]) + ' ' + s)))));
  el.appendChild(body);
  el.appendChild(foot);

  /* "New character · Varek", or just "New character" until there is a name.
   *
   * The separator is only there to join two things, so it goes with the second
   * of them — a title ending in a dangling "·" while somebody clears the field
   * to retype it looks broken for exactly as long as they are looking at it.
   */
  const redrawTitle = () => {
    const name = draft.name.trim();
    title.textContent = name ? `New character · ${name}` : 'New character';
  };
  redrawTitle();

  /* **Is a portrait being drawn right now**, held for the whole page rather
   * than on the button that started it.
   *
   * A draw outlives the step that launched it. `stepFace` rebuilds its buttons
   * every time you arrive, so a flag living on the Regenerate button is a flag
   * the *next* set of buttons knows nothing about: stepping Back and forward
   * mid-draw produced a fresh, enabled button and a second concurrent draw,
   * racing the first to write into `draft.portrait_file`.
   *
   * Pressing Create mid-draw is the worse half of the same problem. The
   * character is written with whatever portrait the draft holds at that instant
   * — the old one, or none — and the draw that was still running lands in a
   * draft nobody will submit again, so the picture is silently discarded after
   * the cast has already been paused to make it. */
  let drawing = false;

  const redrawSteps = () => {
    const host = el.querySelector('.steps');
    mount(host, ['Identity', 'Face'].map((s, i) =>
      h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
        (step > i + 1 ? '✓' : '①②'[i]) + ' ' + s)));
  };

  // ── step 1 ────────────────────────────────────────────────────────────────

  async function stepIdentity() {
    const nameIn = h('input', { class: 'input', placeholder: 'Varek', value: draft.name,
      onInput: (e) => {
        draft.name = e.target.value;
        draft.name_origin = 'authored';
        redrawTitle();
      } });
    const desc = h('textarea', {
      class: 'textarea', rows: 6, placeholder: 'generating…',
      onInput: (e) => { draft.description = e.target.value; draft.description_origin = 'authored'; markOrigin(); },
    });
    const originChip = h('span', { class: 'chip' }, 'generated');
    const markOrigin = () => { originChip.textContent = draft.description_origin; };

    const regen = h('button', { class: 'btn sm', onClick: gen }, '⟳ Regenerate');
    /* The button is disabled while this runs, because a second click would queue
     * a second job that stops the cast again for no extra answer.
     *
     * An ordinary wait, deliberately. The daemon does swap a second model onto
     * the card to answer this, but the swap is ~0.3s and the whole request is a
     * second or two, so naming the machinery would be explaining an interval
     * nobody is left waiting through. The text arriving is what says it works.
     *
     * The failure is reported in the field rather than as a toast, because the
     * field is where the author is looking and what they have to do about it
     * (write one themselves) happens there. */
    /* Name the character, then describe *that* person.
     *
     * Two calls rather than one because they are two things an author judges
     * separately: liking the name and not the prose used to mean regenerating
     * both and losing the name. The name is also twenty tokens, so it lands
     * almost immediately and the form stops looking empty while the description
     * is still being written.
     *
     * Both run on the resident model beside the cast, so neither pays a model
     * load. */
    async function nameThenDescribe() {
      if (!draft.world_id) return;
      nameIn.setAttribute('disabled', '');
      nameIn.placeholder = 'naming…';
      try {
        const r = await API.generateName({
          world_id: draft.world_id,
          personality_id: draft.personality_id,
        });
        draft.name = r.name;
        draft.name_origin = 'generated';
        nameIn.value = r.name;
        redrawTitle();
      } catch (_) {
        // A name that could not be written is not a reason to skip the
        // description: the field is editable and an author can type one.
        nameIn.placeholder = 'Varek';
      } finally {
        nameIn.removeAttribute('disabled');
      }
      await gen();
    }

    async function gen() {
      // Disabled while it runs; the label stays put. The text appearing in the
      // field is the progress indicator, and it is where the reader is looking.
      regen.setAttribute('disabled', '');
      desc.value = '';
      desc.placeholder = 'writing…';
      try {
        /* Streamed into the field as it is written, at about reading speed.
         *
         * The textarea is scrolled to the bottom on each fragment so a
         * description longer than the box keeps its newest words in view —
         * without it the text grows out of sight after three lines. */
        const r = await API.generateDescriptionStream(
          {
            personality_id: draft.personality_id,
            world_id: draft.world_id,
            // The subject, so the prose is about the person in the name field
            // rather than a second one it invents. Empty means "invent one",
            // which is what a cleared name field should do.
            name: draft.name || '',
          },
          (ev) => {
            if (ev.event === 'token') {
              desc.placeholder = '';
              desc.value += ev.text;
              desc.scrollTop = desc.scrollHeight;
            }
          });
        /* The final text replaces the preview rather than being appended to it.
         * The daemon's fragments can end a character or two off where tokenizer
         * cleanup revised something already sent, and `done` is authoritative. */
        draft.description = r.description;
        draft.description_origin = 'generated';
        desc.value = r.description;
        markOrigin();
      } catch (e) {
        desc.value = '';
        /* A loading engine is the one refusal that resolves on its own, so it
         * says so rather than only asking for a description. */
        desc.placeholder = e && e.error === 'engine_unavailable'
          ? 'the engine is still loading — try again in a moment, or write one yourself'
          : 'generation unavailable — write one yourself';
      }
      regen.removeAttribute('disabled');
    }

    /* **Change the world or the personality and the character is recast.**
     *
     * Both are inputs to the name and the description — the world supplies the
     * setting, the personality supplies the anchor — so leaving a name and a
     * paragraph written for the *previous* pairing in the fields is leaving
     * something that no longer belongs to the character being made. A
     * quartermaster from a world you have just navigated away from is worse than
     * an empty field, because it looks like an answer.
     *
     * **What was typed is never recast.** Clearing is keyed on the origin of
     * each field independently, so an author who wrote their own name and let
     * the description generate keeps the name and gets a new paragraph. That is
     * the same rule the step's opening already follows — it fills empty fields
     * and does not overwrite work — and clearing is precisely how this hands the
     * job back to it rather than growing a second path that regenerates.
     */
    const recast = () => {
      if (draft.name_origin === 'generated') draft.name = '';
      if (draft.description_origin === 'generated') draft.description = '';
      draw();
    };

    const worldSel = worlds.length
      ? h('select', {
        class: 'select',
        id: SEL_WORLD,
        /* Redraws, because the personality list is a function of this. Changing
         * to a world that cannot host the chosen personality has to drop her
         * from the list *and* move the selection off her — leaving a submitted
         * pairing the world refuses would fail at the daemon with a message
         * about categories, for a choice the form had already shown as made. */
        onChange: (e) => {
          draft.world_id = e.target.value;
          const ok = shownPersonalities();
          if (!ok.some((p) => p.personality_id === draft.personality_id)) {
            draft.personality_id = ok[0] ? ok[0].personality_id : '';
          }
          recast();
        },
      },
        shownWorlds().map((w) => h('option', { value: w.world_id, selected: w.world_id === draft.world_id }, w.name)))
      : h('div', { class: 'tiny dim' }, 'no worlds — point the daemon at a mind');

    mount(body,
      h('div', { class: 'panel' },
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Name'), nameIn),
          h('div', {},
            h('label', { class: 'field' }, h('span', {}, 'World'), worldSel),
            h('label', { class: 'field' }, h('span', {}, 'Personality'),
              personalities.length
                ? h('select', {
                  class: 'select',
                  id: SEL_PERSONALITY,
                  onChange: (e) => { draft.personality_id = e.target.value; recast(); },
                },
                  shownPersonalities().map((a) => h('option', {
                    value: a.personality_id, selected: a.personality_id === draft.personality_id,
                  }, a.name || a.personality_id.split('-').map((w) => w.replace(/^./, (c) => c.toUpperCase())).join(' '))))
                // No "+" beside either selector: worlds and personalities are
                // files an author writes into the mind, so a button here would
                // make the console and the mind disagree about what exists.
                : h('div', { class: 'tiny dim' }, 'no personalities — point the daemon at a mind')))),

        h('label', { class: 'field', style: 'margin-top:6px' },
          h('span', {}, h('span', {}, 'Description — who this character is '), originChip),
          desc),
        h('div', { class: 'row', style: 'justify-content:space-between' },
          h('div', { class: 'tiny dim', style: 'max-width:620px' },
            'This becomes the character’s identity in the system prompt, and the portrait is generated from it. ' +
            'Written as a present-day person: the personality supplies the anchor and the traits, this supplies ' +
            'the human texture.'),
          regen))
    );

    /* On opening: name first, then a description of that person. An author who
     * has already typed either keeps both — this fills an empty form, it does
     * not overwrite work. */
    if (!draft.description) {
      if (!draft.name) nameThenDescribe();
      else gen();
    } else { desc.value = draft.description; markOrigin(); }

    mount(foot,
      h('button', { class: 'btn ghost', onClick: () => go('/') }, 'Cancel'),
      h('button', { class: 'btn primary', onClick: () => { step = 2; draw(); } }, 'Next →'));
  }

  // ── step 2 ────────────────────────────────────────────────────────────────

  async function stepFace() {
    const models = (await API.listImageModels().catch(() => ({ models: [] }))).models || [];
    const prog = h('i', { style: 'width:0%' });
    /* **The bar exists only while a draw is running.**
     *
     * It used to be rendered permanently and left wherever the last thing to
     * touch it put it — which meant the step opened showing a *full* bar,
     * because loading the personality's portrait and dropping a file both wrote
     * 100% into it. A progress bar sitting at 100% over nothing is worse than
     * no bar: it reports a job that is not running and did not just finish.
     *
     * So it starts hidden, appears when the button is pressed, and goes away
     * when the picture lands or the draw fails. `.bar` sets no `display`, so
     * clearing the inline value returns it to its natural block. */
    const progWrap = h('div', { class: 'bar', style: 'margin:10px 0 6px;display:none' }, prog);

    /* **One bar across both phases, and it never goes backwards.**
     *
     * A draw is two pieces of work with no common unit: the guest loads the
     * model, then it denoises. Only the second reports a fraction. Filling the
     * bar during the load and then starting the steps from zero — which is what
     * this did — shows a bar completing and restarting inside one draw, so the
     * first 100% is a lie and the drop back reads as a failure.
     *
     * The fix is one scale with the load given a reserved head: it climbs to
     * LOAD_BAND while the model comes across the link and the steps map onto
     * what is left. `advance` clamps to the maximum reached, so no event can
     * move it back — a late-arriving `loading` line after the steps have begun
     * cannot undo them.
     *
     * The load holds at LOAD_BAND rather than creeping through it. There is no
     * fraction to creep on — its length depends on what has to be evicted first
     * — and inventing one is the `fakeProgress()` this file already deleted
     * once. The label says which phase is running; the bar says how far. */
    const LOAD_BAND = 15;
    let reached = 0;
    const advance = (pct) => {
      reached = Math.max(reached, Math.min(100, pct));
      prog.style.width = `${reached.toFixed(1)}%`;
    };
    const showProgress = (on) => {
      progWrap.style.display = on ? '' : 'none';
      if (!on) {
        reached = 0;
        prog.style.width = '0%';
      }
    };
    /* The status line, and it is **empty until it has something to say**. It
     * opened on "queued" over nothing queued, then rested on "ready" over a
     * button that visibly works. It now carries progress while a draw runs and
     * the reason when one fails, and is silent the rest of the time. */
    const label = h('div', { class: 'tiny dim' });

    const initial = (draft.name || '?')[0] || '?';

    /* **Sized in viewport units, not pixels.** It was a fixed 170px square,
     * which is small on the desktop this is authored on and still 170px on a
     * phone where it is most of the width. `clamp` gives it a floor, a ceiling
     * and a share of the viewport in between, and `aspect-ratio` keeps it square
     * without a second number to keep in step with the first.
     *
     * `--art` carries the side so the initial letter can scale with the frame
     * rather than sitting small in the middle of a large one. */
    const art = h('div', {
      style: '--art:clamp(170px,26vw,300px);width:var(--art);aspect-ratio:1;'
        + 'border-radius:12px;display:grid;place-items:center;'
        + 'background:linear-gradient(145deg,var(--panel-3),var(--bg-deep));border:1px solid var(--line-2);'
        + 'font-size:calc(var(--art)/4);color:var(--accent)',
      onClick: () => enlarge(),
    }, initial);

    /* Filling the frame, and saying whether it can be clicked.
     *
     * Three places put a picture here and two put the letter back, and the
     * enlarge affordance has to follow: `zoom-in` over an initial that opens
     * nothing is a promise the frame does not keep. Kept together so the cursor
     * cannot drift from the contents. */
    const showPortrait = (blob) => {
      const url = URL.createObjectURL(blob);
      mount(art, h('img', {
        src: url,
        style: 'width:100%;height:100%;object-fit:cover;border-radius:12px',
        // Released once the browser has decoded it; the preview keeps painting
        // from the decoded image and the blob does not sit in memory until the
        // page is closed. `enlarge` makes its own URL from the file.
        onLoad: () => URL.revokeObjectURL(url),
      }));
      art.style.cursor = 'zoom-in';
      art.title = 'Click to see it full size';
    };
    const showInitial = () => {
      mount(art, initial);
      art.style.cursor = 'default';
      art.title = '';
    };

    /* The portrait, full size, on the house scrim.
     *
     * A fresh object URL from the file rather than the one the thumbnail used:
     * that one is revoked the moment the browser has decoded it, so the picture
     * on screen keeps painting but the URL is dead and cannot be given to a
     * second `<img>`. The file is the durable thing, so the enlarged view is
     * made from it and revoked again on close.
     *
     * Escape closes it as well as a click, and the listener is removed with the
     * scrim — a page that accumulates key handlers every time you look at a
     * picture is a page that gets slower the longer you use it.
     */
    function enlarge() {
      if (!draft.portrait_file) return;
      const url = URL.createObjectURL(draft.portrait_file);
      const close = () => {
        scrim.remove();
        URL.revokeObjectURL(url);
        document.removeEventListener('keydown', onKey);
      };
      const onKey = (e) => { if (e.key === 'Escape') close(); };
      const scrim = h('div', { class: 'scrim', style: 'cursor:zoom-out', onClick: close },
        h('img', {
          src: url,
          alt: draft.name ? `A portrait of ${draft.name}` : 'A portrait',
          style: 'max-width:min(880px,92vw);max-height:88vh;border-radius:12px;'
            + 'box-shadow:var(--shadow-lg)',
        }));
      document.addEventListener('keydown', onKey);
      document.body.appendChild(scrim);
    }

    /* The frame says what the frame is doing.
     *
     * A portrait takes ten seconds or so, and the state used to live in a line
     * of small text above the button — far from the empty square everybody is
     * actually watching. Put it where the result appears and there is nothing
     * to hunt for. The child overrides the frame's 2.6rem initial-letter size. */
    const artSays = (text) => mount(art, h('div', {
      // `pre-line` so a newline in the message is a line break: a text node's
      // whitespace collapses otherwise, and the two halves run together.
      style: 'font-size:.78rem;line-height:1.45;color:var(--ink-faint);text-align:center;'
        + 'padding:0 14px;white-space:pre-line',
    }, text));

    /* There was a `fakeProgress()` here: a bar that crept to 100% on a timer
     * and then set `portrait_origin = 'generated'`, having generated nothing.
     * It is gone. No image model is loaded, so the honest states are "uploaded"
     * and "none" — and a progress bar that completes over a thing that never
     * ran is the same lie as a fixture standing in for a library. */

    /* **The portrait is the drop target, so there is no drop target.**
     *
     * There was a dashed box here — first a wide bar under the frame, then a
     * square beside the Regenerate button — and both looked bolted on, because
     * they were: a second empty square, at a size that matched nothing, sitting
     * next to a small button in a wide column. The composition had two things
     * competing to be the picture.
     *
     * Dropping onto the picture you are replacing needs no box at all. The frame
     * takes the drag, borrows the accent border while a file is over it, and the
     * two things you can actually *press* become a matched pair of buttons —
     * which is a shape that never looks strange.
     */
    const dragOver = (on) => {
      art.style.borderColor = on ? 'var(--accent)' : 'var(--line-2)';
      art.style.borderStyle = on ? 'dashed' : 'solid';
    };
    art.addEventListener('dragover', (e) => { e.preventDefault(); dragOver(true); });
    art.addEventListener('dragleave', () => dragOver(false));
    art.addEventListener('drop', (e) => {
      e.preventDefault();
      dragOver(false);
      useFile(e.dataTransfer.files[0]);
    });

    const uploadBtn = h('button', {
      class: 'btn sm ghost',
      style: 'flex:0 0 auto;white-space:nowrap',
      onClick: () => file.click(),
    }, '⤒ Upload');

    const file = h('input', { type: 'file', accept: 'image/*', style: 'display:none',
      onChange: (e) => useFile(e.target.files[0]) });

    /* Held as the FILE, not as an object URL.
     *
     * It used to keep `URL.createObjectURL(f)` in `draft.portrait` and call
     * that "uploaded" — but `create()` never sent it, and an object URL is a
     * handle to a blob in this tab that goes away with the tab. The image was
     * discarded on submit, every time, silently.
     *
     * The upload is a second request after the character exists, because it is
     * addressed to one: `PUT /v1/npc/:id/portrait`. So it happens in `create()`
     * once there is an id, and this step only holds the bytes and the preview.
     */
    function useFile(f) {
      if (!f) return;
      draft.portrait_file = f;
      // An upload outranks the generator permanently, so choosing a file
      // cancels a pending draw rather than queueing both.
      draft.portrait_generate = false;
      draft.portrait_origin = 'uploaded';
      // Chosen by a person, so changing personality must not replace it.
      draft.portrait_from_personality = '';
      showPortrait(f);
      // No progress bar for an upload: there is no job to report on, and a full
      // one left behind reads as a draw that finished. Nothing to say either —
      // the picture in the frame is the confirmation.
      showProgress(false);
      label.textContent = '';
    }

    /* What the chosen personality was authored with — the prompt, and the
     * portrait already picked for it. Absent for a personality that carries
     * neither, which is every one of them until somebody writes a `portrait:`
     * block, so everything below has to read as a fallback rather than a
     * requirement. */
    const authored = () => {
      const p = personalities.find((x) => x.personality_id === draft.personality_id);
      return (p && p.portrait) || {};
    };

    /* The prompt box opens holding whatever would have been used anyway: the
     * personality's authored prompt, or the description. Sending it back
     * unchanged is the same request as sending nothing, which is what makes the
     * box safe to show — it explains the draw rather than adding a step to it.
     *
     * **An edit wins over both, and only an edit.** Keyed on
     * `portrait_prompt_touched` rather than on the box being non-empty: the box
     * is *always* non-empty after the first visit, so a plain `||` on the
     * stored value would freeze the first personality's prompt in place and
     * changing personality would silently keep drawing the old character. */
    const startingPrompt = () =>
      (draft.portrait_prompt_touched && draft.portrait_prompt)
      || authored().prompt
      || (draft.description || '').trim();

    /* **Eight rows, and the stored prompt carries no hard newlines.**
     *
     * A prompt is one stream of clauses, not lines. Stored as a YAML literal
     * block (`|`) it arrived with a newline every ~85 characters — the width the
     * *file* was wrapped to — and this box is narrower than that, so every
     * authored line wrapped a second time and the text came out as ragged
     * half-lines. The personality stores it folded (`>-`) instead, so the value
     * is one line and wraps to whatever width the box happens to be. */
    const promptBox = h('textarea', {
      class: 'textarea', rows: 8,
      // The string, not the boolean: `h` drops a `false` value entirely, so
      // `spellcheck: false` would leave checking on — and a prompt is a run of
      // comma-separated fragments that a spell checker underlines throughout.
      spellcheck: 'false',
      style: 'font-size:.82rem;line-height:1.5;resize:vertical',
      placeholder: 'the words this portrait is drawn from',
      onInput: (e) => {
        draft.portrait_prompt = e.target.value;
        // Touched, so switching personality no longer replaces it.
        draft.portrait_prompt_touched = true;
      },
    });
    promptBox.value = startingPrompt();
    draft.portrait_prompt = promptBox.value;

    /* Where the words came from, said plainly — the box is prefilled from three
     * different places and which one it was changes what editing it means. */
    const promptSource = h('div', { class: 'tiny dim', style: 'margin-top:8px' },
      authored().prompt
        ? 'Authored on this personality. Editing it here changes this character’s portrait only — '
          + 'the personality keeps its own.'
        : 'From the description. Edit it to draw something the description does not say.');

    /* **Folded away, because it is a thing you check rather than a thing you
     * fill in.**
     *
     * Eight rows of prose is the tallest element on the step and the one least
     * often touched: the prompt arrives correct — from the personality or from
     * the description — and the ordinary path is to look at the portrait and
     * press Regenerate, never opening this at all. Left expanded it pushed the
     * upload target and the environment of the page down past a screenful of
     * text nobody was reading.
     *
     * **The nodes are held, not rebuilt.** `disclosure` empties its body on
     * collapse — it was written for a lazily-loaded tree where that is the point
     * — so a body that *constructed* a textarea would hand back a blank one, and
     * an edit made before collapsing would be gone. Re-mounting the same two
     * nodes keeps the value, because the value lives on the element and the
     * element survives being detached.
     */
    /* **It tells you what opening it lets you do.**
     *
     * Two earlier goes at this heading missed in opposite directions. "The
     * prompt this portrait is drawn from" described the field's plumbing in a
     * full sentence. "Art direction" was the right length but borrowed a term of
     * art — it is what the thing *is* to somebody who already knows, and means
     * nothing to somebody who does not, which on a closed panel is the whole
     * job. A collapsed row has to earn the click.
     *
     * So: what happens if you open it, in words anybody has. */
    const promptPanel = disclosure({
      dense: true,
      head: h('span', { class: 'disc-title' }, 'Change how they look'),
      body: (el) => {
        mount(el, promptBox, promptSource);
      },
    });

    /* Show the portrait this personality was authored with.
     *
     * The bytes are fetched rather than merely displayed, because `create()`
     * attaches a *file* — the same upload path a dropped image takes. The store
     * is content-addressed, so re-uploading a picture it already holds is one
     * more reference to one file, not a copy.
     *
     * Marked `generated`, not `uploaded`: an upload outranks the generator
     * permanently, and a default nobody chose must not be the thing that stops
     * a later draw from replacing it.
     */
    async function useAuthoredPortrait() {
      const id = authored().image_id;
      if (!id) return false;
      try {
        const res = await fetch(API.imageUrl(id));
        if (!res.ok) throw new Error(`the daemon answered ${res.status}`);
        const blob = await res.blob();
        draft.portrait_file = new File([blob], 'portrait.png', { type: blob.type || 'image/png' });
        draft.portrait_generate = false;
        draft.portrait_origin = 'generated';
        draft.portrait_from_personality = draft.personality_id;
        showPortrait(blob);
        // Nothing was generated — the picture was already there. See `progWrap`.
        showProgress(false);
        return true;
      } catch (_) {
        /* A portrait that will not load is not a reason to block the step —
         * the character can still be created, drawn or uploaded into. The
         * frame stays on the initial and says nothing, because there is
         * nothing here for an author to fix. */
        return false;
      }
    }

    // `loaded`, not `length`. The catalog lists what this daemon *could* run;
    // whether any of it is resident is the only thing that decides whether a
    // portrait can be made. Keyed on `length`, the step offered a model picker
    // and a progress bar and then refused — which is the contradiction the fake
    // progress bar used to paper over.
    const canGenerate = models.some((m) => m.loaded);

    /* **It draws, now, and shows you the portrait.**
     *
     * It used to only mark the draft — the actual draw was deferred to
     * `create()`, because `POST /v1/npc/:id/portrait/generate` is addressed to a
     * character and there is not one yet. That is true of the *route*, but it
     * was never true of the model: `/v1/image/generate` takes a prompt and gives
     * back bytes, and the box on this page is the prompt. So the deferral bought
     * nothing and cost the button its meaning — you pressed it and no portrait
     * appeared.
     *
     * The bytes are kept and attached once the character exists, marked
     * `generated` so a later draw may still replace them.
     *
     * **It says Regenerate.** There is already a portrait in the frame the
     * moment the step opens — the personality's, or one the step drew on
     * arrival — so "Generate" named an action that had visibly already
     * happened. "Regenerate" reads as what it is: the same face again,
     * differently. */
    const regenBtn = h('button', {
      class: 'btn sm',
      // Sized to its label, so it and Upload sit as a pair rather than
      // stretching to split the row between them.
      style: 'flex:0 0 auto;white-space:nowrap',
      disabled: !canGenerate,
      onClick: () => drawPortrait({ announce: true }),
    }, '⟳ Regenerate');

    /* Draw the portrait from whatever the prompt box currently holds.
     *
     * Shared by the button and by the step's own opening, which draws without
     * being asked — see the foot of this function. `announce` is the difference
     * between the two: a draw somebody pressed for should say so when it fails,
     * and one the page started on its own should not raise a toast over a
     * failure nobody caused.
     */
    async function drawPortrait({ announce }) {
      if (!canGenerate || drawing) return;
      /* The box is the prompt, and it opened holding the description when
       * there was nothing else — so an empty box now means both are empty,
       * and there is genuinely nothing to draw from. */
      const words = (promptBox.value || '').trim();
      if (!words) {
        const why = 'write a description or a prompt first — a portrait is drawn from words';
        label.textContent = why;
        if (announce) toast(why, 'err');
        return;
      }
      /* The controls are disabled while it runs and no label changes. A control
       * that renames itself mid-action is its own puzzle: the state goes in the
       * frame, which is the thing being watched and where the portrait lands. */
      setDrawing(true);
      artSays('◍ drawing…\nthe cast is paused');
      label.textContent = '';
      showProgress(true);
      try {
        /* **The bar follows the draw, and it finishes.**
         *
         * `/v1/image/generate` is NDJSON: it streams `step` lines carrying
         * `{done, total, what}` — the guest's own units, denoise steps plus the
         * decode — and delivers the picture whole on the terminal line. Without
         * an `onEvent` this read like the blocking call it replaced and the bar
         * sat at 0% through the entire draw, then stayed there afterwards,
         * which is the opposite of what a progress bar is for.
         *
         * The decode is the last unit and is announced when it STARTS, so the
         * stream leaves the bar just short of full. The event that completes it
         * is the finished picture, so the 100% below is written on success
         * rather than by the stream — and it is the honest place for it. */
        const img = await API.generateImage({ prompt: words }, (ev) => {
          if (!ev) return;
          if (ev.event === 'loading') {
            // The head of the same scale, not a bar of its own.
            advance(LOAD_BAND);
            label.textContent = 'loading the model…';
            return;
          }
          if (ev.event !== 'step' || !ev.total) return;
          const frac = Math.max(0, Math.min(1, ev.done / ev.total));
          advance(LOAD_BAND + (100 - LOAD_BAND) * frac);
          // The step count only while there are steps being counted. The decode
          // is one long operation with no interior number, and "step 0 of 7"
          // for it reads as stuck.
          label.textContent = ev.what === 'denoising'
            ? `drawing · step ${ev.done} of ${ev.total - 1}`
            : 'drawing…';
        });
        if (!img || !img.png_base64) {
          throw new Error('the daemon returned no image');
        }
        const blob = pngBlob(img.png_base64);
        draft.portrait_file = new File([blob], 'portrait.png', { type: 'image/png' });
        // Real bytes now, so there is nothing left to defer to create.
        draft.portrait_generate = false;
        draft.portrait_origin = 'generated';
        // A drawn portrait is this author's, not the personality's default — so
        // changing personality must not replace it.
        draft.portrait_from_personality = '';
        // The words that produced what is now in the frame. The step's opening
        // compares against this to decide whether it owes a draw.
        draft.portrait_drawn_from = words;
        showPortrait(blob);
        /* The picture is the completion. The bar goes away rather than resting
         * at full, and the status line goes quiet rather than announcing
         * "drawn" over a portrait you are looking at and telling you to press a
         * button you can see. It speaks again only when something is wrong. */
        showProgress(false);
        label.textContent = '';
      } catch (err) {
        const why = err.error === 'no_image_model'
          ? 'no image model is configured on this daemon'
          : (err.detail || err.message || 'could not draw a portrait');
        // Gone: a bar left part-full over a draw that stopped claims progress
        // toward a picture that is not coming.
        showProgress(false);
        /* The frame goes back to resting rather than being left saying
         * "drawing…" over a draw that has stopped. The reason goes to the label
         * and to a toast — a failure that only appeared in small grey text read
         * as nothing having happened, which is how this was reported. */
        showInitial();
        label.textContent = why;
        if (announce) toast(why, 'err');
      } finally {
        setDrawing(false);
      }
    }

    mount(body, h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'align-items:flex-start;gap:20px' },
        // The picture, and nothing else. Everything that acts on it is beside
        // it, where there is room for the controls to sit at a readable width.
        // No width here — `art` sizes itself, and a second figure to keep in
        // step with it is a second figure to get wrong.
        h('div', { style: 'flex:0 0 auto' }, art),
        h('div', { style: 'flex:1' },
          /* **Named for the character, not for where the picture came from.**
           *
           * This said "A portrait, from the personality" or "…from these words"
           * depending on which source had supplied it — a distinction about the
           * console's plumbing, on the one heading that should be about *whom*
           * you are looking at. Where the words came from is still said, in the
           * line under the prompt box where it is actionable.
           *
           * "of Varek" rather than "Varek's": a possessive has to decide what to
           * do with a name that already ends in s, and getting that wrong on
           * somebody's character reads as carelessness. */
          h('div', { style: 'font-weight:700;margin-bottom:2px' },
            draft.name.trim() ? `A portrait of ${draft.name.trim()}` : 'A portrait'),
          h('div', { class: 'tiny dim' },
            authored().prompt
              ? 'This personality was authored with a portrait and the prompt that drew it. Edit the '
                + 'words and press Regenerate for a different one.'
              : 'Drawn from the description unless you change the words below.'),
          canGenerate ? progWrap : null, label,
          /* **A picker only when there is something to pick.**
           *
           * A deployment configures one image guest, so the list is almost
           * always a single entry — and a dropdown holding one option is a
           * control that looks like a decision and cannot be one. It renders
           * from two models up, where choosing actually does something.
           *
           * `vram_gib` is null for a co-resident guest: it stands in ground
           * claimed from the KV side for the length of a drain, so it has no
           * standing footprint to quote. Rendered unconditionally it read
           * "· null GiB". */
          canGenerate && models.length > 1
            ? h('div', { class: 'row', style: 'margin-top:12px;gap:8px' },
              h('select', { class: 'select', style: 'width:auto' },
                models.map((m) => h('option', { value: m.id, selected: m.default },
                  m.vram_gib ? `${m.display} · ${m.vram_gib} GiB` : m.display))))
            : null,
          /* **Said only when it changes what you can do.**
           *
           * There was a paragraph here describing how long a draw takes, that
           * pressing again gives a different portrait, and that an upload
           * outranks the generator — three things the button, the progress bar
           * and the act of uploading already demonstrate. Explaining a control
           * that is visible and works is noise around it.
           *
           * What survives is the case where the controls are NOT self-evident:
           * no model loaded, so Regenerate is dead and the only route to a
           * portrait is Upload. That has to be said, because nothing on screen
           * says it. */
          canGenerate
            ? null
            : h('div', { class: 'tiny dim', style: 'margin-top:10px;max-width:70ch' },
              'Generation needs an image model on this daemon, and there is none. Uploading works '
              + 'now. A character with no portrait shows its initial.'),
          /* The two things you can press, side by side and the same size, with
           * the third way in named underneath rather than drawn as a box. */
          h('div', { class: 'row', style: 'margin-top:12px;gap:8px;align-items:center' },
            regenBtn, uploadBtn, file),
          h('div', { class: 'tiny dim', style: 'margin-top:6px' },
            'or drag an image onto the portrait'),
          h('div', { style: 'margin-top:14px' }, promptPanel)))));

    /* Held, because a draw disables them and re-enables them — and because they
     * are rebuilt on every arrival at this step, a draw that is *already*
     * running when the step opens has to find them disabled from the start. */
    const backBtn = h('button', {
      class: 'btn ghost', disabled: drawing,
      onClick: () => { step = 1; draw(); },
    }, '← Back');
    const createBtn = h('button', {
      class: 'btn primary', disabled: drawing,
      onClick: create,
    }, 'Create');
    mount(foot, backBtn, createBtn);

    /* Every control a draw takes away, in one place.
     *
     * Regenerate, because a second draw would race the first. Create, because
     * the character would be written without the portrait being made for it.
     * Back, because leaving the step rebuilds these buttons and the guard would
     * go with them. */
    const setDrawing = (on) => {
      drawing = on;
      regenBtn.disabled = on || !canGenerate;
      uploadBtn.disabled = on;
      backBtn.disabled = on;
      createBtn.disabled = on;
    };

    /* Open showing the personality's own portrait.
     *
     * Three cases, and only the first two touch the frame:
     *
     *  - nothing in the frame → show the personality's, if it has one;
     *  - what is in the frame came from a personality that is no longer the
     *    selected one → replace it, and clear it if the new personality has
     *    none, so the frame never shows the last character's face;
     *  - somebody drew or dropped it (`portrait_from_personality` empty) → it
     *    is theirs, and nothing here replaces it.
     *
     * The fetch is after `mount` so the step paints immediately and the picture
     * arrives into it, rather than the whole step waiting on an image.
     */
    const stale = draft.portrait_from_personality
      && draft.portrait_from_personality !== draft.personality_id;
    if (!draft.portrait_file || stale) {
      useAuthoredPortrait().then((got) => {
        if (!got && stale) {
          // The new personality has no portrait of its own, so the old one's
          // must go rather than stand in for it.
          draft.portrait_file = null;
          draft.portrait_origin = null;
          draft.portrait_from_personality = '';
          showInitial();
          showProgress(false);
        }
        openingDraw();
      });
    } else {
      openingDraw();
    }

    /* **The step draws on arrival, so you are looking at this character.**
     *
     * Landing on an empty square — or on the personality's stock face — and
     * having to press a button to see the character you just described made the
     * draw feel like an extra step rather than the point of the page. So it
     * starts one itself.
     *
     * It does not draw every time you arrive, and the guard is the prompt rather
     * than a visit count: `portrait_drawn_from` records the words that produced
     * what is in the frame, so stepping Back, rewriting the description and
     * coming forward redraws — the description changed, the face should — while
     * Back and straight forward again does not. That matters because a draw
     * pauses every character in the cast for its duration; spending that on a
     * picture already drawn from the same words buys nothing.
     *
     * An upload is never overwritten. It outranks the generator everywhere else
     * and it would be a poor place to start making an exception, least of all
     * silently and without being asked.
     */
    function openingDraw() {
      if (!canGenerate) return;
      if (draft.portrait_origin === 'uploaded') return;
      const words = (promptBox.value || '').trim();
      if (!words || draft.portrait_drawn_from === words) return;
      drawPortrait({ announce: false });
    }
  }


  /* Write the character.
   *
   * One record appended to the substrate, keyed by a freshly minted `npc_id`,
   * flushed and fsynced before this returns — so a character the page says was
   * created is one that survives the daemon being killed a second later.
   *
   * What lands is the record's own fields, and they are all this wizard now
   * collects. There used to be a third step offering beliefs, relationships
   * and goals to pick from: those are substrate *layer* content — turns, not
   * columns — so none of them were ever written, and the ones on offer came
   * from a fixture that returned the same three for every character whatever
   * was typed. A step whose Regenerate button could not regenerate, over
   * choices that could not be saved, is worse than no step. It is gone, and it
   * comes back when there is an engine to gather them. */
  async function create() {
    /* The disabled button is the visible half; this is the half that holds. A
     * portrait still being drawn is one the character would be written without,
     * and the draw is already costing the whole cast its tick — so the answer
     * is "wait", not "create and discard it". */
    if (drawing) return toast('The portrait is still being drawn', 'warn');
    if (!draft.name.trim()) return toast('Give the character a name', 'err');
    try {
      const npc = await API.createNpc({
        name: draft.name, world_id: draft.world_id, personality_id: draft.personality_id,
        // `persona_description`, the record's own field name. It was
        // `description`, which the daemon does not read — a character created
        // through this page arrived with an empty persona and no error to say
        // why, because an absent persona is legal.
        persona_description: draft.description,
      });
      /* The portrait, now that there is a character to attach it to.
       *
       * Its failure does not fail the create: the character exists, and losing
       * it over a picture would be the wrong trade. The toast says which
       * happened rather than reporting plain success over a portrait that did
       * not land — which is the mistake this whole step is a fix for.
       */
      if (draft.portrait_file) {
        try {
          /* The origin travels with the bytes. A portrait drawn on this page is
           * `generated` even though it arrives through the upload route — filed
           * as `uploaded` it could never be redrawn, because the generator
           * refuses to replace an uploaded portrait. */
          await API.putPortrait(npc.npc_id, draft.portrait_file, draft.portrait_origin);
          toast(`${draft.name} created`, 'ok');
        } catch (e) {
          toast(`${draft.name} created, but the portrait did not upload: `
            + (e.detail || e.message || 'unknown error'), 'err');
        }
      } else if (draft.portrait_generate) {
        /* The draw, now that there is a character to address it to. Same trade
         * as the upload above: its failure does not fail the create, and the
         * toast says which happened rather than reporting plain success over a
         * portrait that never landed.
         *
         * It blocks for the length of a drain — every character stops thinking
         * while it runs — so the toast before it exists to explain a pause that
         * would otherwise look like a hang. */
        toast(`${draft.name} created — drawing the portrait, the cast is paused`, 'ok');
        try {
          await API.generatePortrait(npc.npc_id);
          toast('portrait drawn', 'ok');
        } catch (e) {
          toast(`${draft.name} created, but the portrait was not drawn: `
            + (e.error === 'no_image_model'
              ? 'no image model is configured on this daemon'
              : (e.detail || e.message || 'unknown error')), 'err');
        }
      } else {
        toast(`${draft.name} created`, 'ok');
      }
      go('/npc/' + npc.npc_id);
    } catch (e) { toast(e.detail || e.message || 'could not create', 'err'); }
  }

  function draw() {
    redrawSteps();
    ({ 1: stepIdentity, 2: stepFace }[step])();
  }

  /* Repaint on every press and release of RIGHT ALT — and **only** repaint.
   *
   * The key changes what is in the world list. It does not open the list, does
   * not move focus, and does not decide anything on the reader's behalf: an
   * earlier version re-opened the selector so a press with the popup already
   * open would show the new options, and the cure was worse than the disease —
   * holding the key made a dropdown appear out of nowhere.
   *
   * The cost is that a popup already open keeps showing the options it opened
   * with, because a native `<select>` does not re-render one and there is no
   * way to make it. Closing and opening it shows the new list. That is a
   * smaller surprise than a control that opens itself. */
  const stopReveal = onReveal(() => draw());

  draw();
  return { el, teardown: stopReveal };
}

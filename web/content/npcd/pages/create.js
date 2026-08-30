/* Creating an NPC (§29) — three steps, each with a working default.
 *
 * The description IS the character: it becomes the identity section in the
 * system prompt, and the portrait is generated from it. There is no separate
 * image prompt — a second place to say who the character is would guarantee
 * drift. Visibility and tags are deliberately absent; they belong to an
 * existing character (§30). */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
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
    description: '', description_origin: 'generated',
    portrait: null, portrait_origin: null,
    environment_enabled: true,
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
  const shownPersonalities = () => personalities.filter((p) => hostable(p, draft.world_id));
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
  const firstPersonality = personalities.find((p) => hostable(p, draft.world_id)) || personalities[0];
  if (firstPersonality) draft.personality_id = firstPersonality.personality_id;

  const body = h('div', {});
  const foot = h('div', { class: 'row', style: 'justify-content:flex-end;gap:9px;margin-top:22px' });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'New character'),
      h('div', { class: 'sub' }, 'Two steps. Both have a default, so Next then Create is a valid character.')),
    h('div', { class: 'steps' },
      ['Identity', 'Face'].map((s, i) =>
        h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
          (step > i + 1 ? '✓' : '①②'[i]) + ' ' + s)))));
  el.appendChild(body);
  el.appendChild(foot);

  const redrawSteps = () => {
    const host = el.querySelector('.steps');
    mount(host, ['Identity', 'Face'].map((s, i) =>
      h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
        (step > i + 1 ? '✓' : '①②'[i]) + ' ' + s)));
  };

  // ── step 1 ────────────────────────────────────────────────────────────────

  async function stepIdentity() {
    const nameIn = h('input', { class: 'input', placeholder: 'Varek', value: draft.name,
      onInput: (e) => { draft.name = e.target.value; } });
    const desc = h('textarea', {
      class: 'textarea', rows: 6, placeholder: 'generating…',
      onInput: (e) => { draft.description = e.target.value; draft.description_origin = 'authored'; markOrigin(); },
    });
    const originChip = h('span', { class: 'chip' }, 'generated');
    const markOrigin = () => { originChip.textContent = draft.description_origin; };

    const regen = h('button', { class: 'btn sm', onClick: gen }, '⟳ Regenerate');
    async function gen() {
      regen.setAttribute('disabled', '');
      desc.value = '';
      desc.placeholder = 'generating…';
      try {
        const r = await API.generateDescription({ personality_id: draft.personality_id, world_id: draft.world_id });
        draft.description = r.description;
        draft.description_origin = 'generated';
        desc.value = r.description;
        markOrigin();
      } catch (_) { desc.placeholder = 'generation unavailable — write one yourself'; }
      regen.removeAttribute('disabled');
    }

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
          draw();
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
                ? h('select', { class: 'select', id: SEL_PERSONALITY, onChange: (e) => { draft.personality_id = e.target.value; } },
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

    if (!draft.description) gen();
    else { desc.value = draft.description; markOrigin(); }

    mount(foot,
      h('button', { class: 'btn ghost', onClick: () => go('/') }, 'Cancel'),
      h('button', { class: 'btn primary', onClick: () => { step = 2; draw(); } }, 'Next →'));
  }

  // ── step 2 ────────────────────────────────────────────────────────────────

  async function stepFace() {
    const models = (await API.listImageModels().catch(() => ({ models: [] }))).models || [];
    const prog = h('i', { style: 'width:0%' });
    const progWrap = h('div', { class: 'bar', style: 'margin:10px 0 6px' }, prog);
    const label = h('div', { class: 'tiny dim' }, 'queued');

    const art = h('div', {
      style: 'width:170px;height:170px;border-radius:12px;display:grid;place-items:center;' +
        'background:linear-gradient(145deg,var(--panel-3),var(--bg-deep));border:1px solid var(--line-2);' +
        'font-size:2.6rem;color:var(--accent)',
    }, (draft.name || '?')[0] || '?');

    /* There was a `fakeProgress()` here: a bar that crept to 100% on a timer
     * and then set `portrait_origin = 'generated'`, having generated nothing.
     * It is gone. No image model is loaded, so the honest states are "uploaded"
     * and "none" — and a progress bar that completes over a thing that never
     * ran is the same lie as a fixture standing in for a library. */

    const drop = h('div', {
      style: 'border:1px dashed var(--line-2);border-radius:10px;padding:14px;text-align:center;color:var(--ink-faint);font-size:.82rem;cursor:pointer',
      onClick: () => file.click(),
      onDragover: (e) => { e.preventDefault(); drop.style.borderColor = 'var(--accent)'; },
      onDragleave: () => { drop.style.borderColor = 'var(--line-2)'; },
      onDrop: (e) => { e.preventDefault(); drop.style.borderColor = 'var(--line-2)'; useFile(e.dataTransfer.files[0]); },
    }, 'or drop an image here · ', h('span', { style: 'color:var(--accent)' }, 'upload a portrait'));

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
      draft.portrait_origin = 'uploaded';
      const url = URL.createObjectURL(f);
      mount(art, h('img', {
        src: url,
        style: 'width:100%;height:100%;object-fit:cover;border-radius:12px',
        // Released once the browser has decoded it; the preview keeps painting
        // from the decoded image and the blob does not sit in memory until the
        // page is closed.
        onLoad: () => URL.revokeObjectURL(url),
      }));
      label.textContent = 'uploaded — this outranks the generator permanently';
      prog.style.width = '100%';
    }

    // `loaded`, not `length`. The catalog lists what this daemon *could* run;
    // whether any of it is resident is the only thing that decides whether a
    // portrait can be made. Keyed on `length`, the step offered a model picker
    // and a progress bar and then refused — which is the contradiction the fake
    // progress bar used to paper over.
    const canGenerate = models.some((m) => m.loaded);
    label.textContent = canGenerate ? 'ready' : 'no image model is loaded';

    mount(body, h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'align-items:flex-start;gap:20px' },
        h('div', {}, art),
        h('div', { style: 'flex:1' },
          h('div', { style: 'font-weight:700;margin-bottom:2px' }, 'A portrait, from the description'),
          h('div', { class: 'tiny dim' },
            'There is no prompt field — the portrait derives from the description, so there is nowhere for '
            + 'the two to drift apart.'),
          canGenerate ? progWrap : null, label,
          h('div', { class: 'row', style: 'margin-top:12px;gap:8px' },
            canGenerate
              ? h('select', { class: 'select', style: 'width:auto' },
                models.map((m) => h('option', { value: m.id, selected: m.default }, `${m.display} · ${m.vram_gib} GiB`)))
              : null,
            h('button', {
              class: 'btn sm',
              onClick: () => toast('generating a portrait — image model required', 'err'),
            }, '⟳ Generate')),
          h('div', { class: 'tiny dim', style: 'margin-top:10px;max-width:70ch' },
            'Generation needs an image model on this daemon, and there is none. Uploading works now and '
            + 'outranks the generator permanently, so a portrait you choose is never replaced by one it '
            + 'invents. A character with no portrait shows its initial.'),
          h('div', { style: 'margin-top:14px' }, drop, file))),
      /* The environment simulator lives here because it is a field on the
       * record, and this is now the last step that has any. It sat on a third
       * step alongside generated beliefs, relationships and goals — none of
       * which were ever written — so removing those would have taken this real
       * setting with them. */
      h('label', { class: 'row', style: 'gap:9px;margin-top:16px;cursor:pointer' },
        h('input', {
          type: 'checkbox',
          checked: draft.environment_enabled,
          onChange: (e) => { draft.environment_enabled = e.target.checked; },
        }),
        h('div', {},
          h('div', { style: 'font-size:.85rem;font-weight:600' }, 'Environment simulator'),
          h('div', { class: 'tiny dim' },
            'No world simulation is attached, so this generates what happens around the character. '
            + 'Turn it off if your own game drives events.')))));

    mount(foot,
      h('button', { class: 'btn ghost', onClick: () => { step = 1; draw(); } }, '← Back'),
      h('button', { class: 'btn primary', onClick: create }, 'Create'));
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
    if (!draft.name.trim()) return toast('Give the character a name', 'err');
    try {
      const npc = await API.createNpc({
        name: draft.name, world_id: draft.world_id, personality_id: draft.personality_id,
        // `persona_description`, the record's own field name. It was
        // `description`, which the daemon does not read — a character created
        // through this page arrived with an empty persona and no error to say
        // why, because an absent persona is legal.
        persona_description: draft.description,
        environment_enabled: draft.environment_enabled,
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
          await API.putPortrait(npc.npc_id, draft.portrait_file);
          toast(`${draft.name} created`, 'ok');
        } catch (e) {
          toast(`${draft.name} created, but the portrait did not upload: `
            + (e.detail || e.message || 'unknown error'), 'err');
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

/* Creating an NPC (§29) — three steps, each with a working default.
 *
 * The description IS the character: it becomes the identity section in the
 * system prompt, and the portrait is generated from it. There is no separate
 * image prompt — a second place to say who the character is would guarantee
 * drift. Visibility and tags are deliberately absent; they belong to an
 * existing character (§30). */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { go } from '../lib/router.js';
import { toast, empty } from '../lib/ui.js';

export async function render() {
  const el = h('div', { class: 'page', style: 'max-width:900px' });

  const draft = {
    name: '', world_id: '', archetype_id: '1',
    description: '', description_origin: 'generated',
    portrait: null, portrait_origin: null,
    environment_enabled: true,
    beliefs: [], relationships: [], agency: [],
    picked: { beliefs: new Set(), relationships: new Set(), agency: new Set() },
  };
  let step = 1;

  const [worlds, archetypes] = await Promise.all([
    API.listWorlds().then((r) => r.worlds || []).catch(() => []),
    API.listArchetypes().then((r) => r.archetypes || []).catch(() => []),
  ]);
  if (worlds[0]) draft.world_id = worlds[0].world_id;

  const body = h('div', {});
  const foot = h('div', { class: 'row', style: 'justify-content:flex-end;gap:9px;margin-top:22px' });

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'New character'),
      h('div', { class: 'sub' }, 'Three steps. Every one has a default, so Next three times is a valid character.')),
    h('div', { class: 'steps' },
      ['Identity', 'Face', 'Inner life'].map((s, i) =>
        h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
          (step > i + 1 ? '✓' : '①②③'[i]) + ' ' + s)))));
  el.appendChild(body);
  el.appendChild(foot);

  const redrawSteps = () => {
    const host = el.querySelector('.steps');
    mount(host, ['Identity', 'Face', 'Inner life'].map((s, i) =>
      h('span', { class: 'step' + (step === i + 1 ? ' on' : step > i + 1 ? ' done' : '') },
        (step > i + 1 ? '✓' : '①②③'[i]) + ' ' + s)));
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
        const r = await API.generateDescription({ archetype_id: draft.archetype_id, world_id: draft.world_id });
        draft.description = r.description;
        draft.description_origin = 'generated';
        desc.value = r.description;
        markOrigin();
      } catch (_) { desc.placeholder = 'generation unavailable — write one yourself'; }
      regen.removeAttribute('disabled');
    }

    const worldSel = worlds.length
      ? h('select', { class: 'select', onChange: (e) => { draft.world_id = e.target.value; } },
        worlds.map((w) => h('option', { value: w.world_id, selected: w.world_id === draft.world_id }, w.name)))
      : h('div', { class: 'tiny dim' }, 'no worlds yet');

    mount(body,
      h('div', { class: 'panel' },
        h('div', { class: 'grid g2' },
          h('label', { class: 'field' }, h('span', {}, 'Name'), nameIn),
          h('div', {},
            h('label', { class: 'field' }, h('span', {}, 'World'),
              h('div', { class: 'row' }, worldSel,
                h('button', { class: 'btn sm', title: 'Create a world', onClick: () => go('/worlds') }, '+'))),
            h('label', { class: 'field' }, h('span', {}, 'Archetype'),
              h('div', { class: 'row' },
                h('select', { class: 'select', onChange: (e) => { draft.archetype_id = e.target.value; } },
                  archetypes.map((a) => h('option', { value: a.archetype_id }, a.name))),
                h('button', { class: 'btn sm', onClick: () => go('/archetypes') }, '+'))))),

        h('label', { class: 'field', style: 'margin-top:6px' },
          h('span', {}, h('span', {}, 'Description — who this character is '), originChip),
          desc),
        h('div', { class: 'row', style: 'justify-content:space-between' },
          h('div', { class: 'tiny dim', style: 'max-width:620px' },
            'This becomes the character’s identity in the system prompt, and the portrait is generated from it. ' +
            'Written as a present-day person: the archetype supplies the fantasy framing, this supplies the human texture.'),
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

    let timer = null;
    function fakeProgress() {
      let p = 0;
      timer = setInterval(() => {
        p = Math.min(1, p + 0.04 + Math.random() * 0.03);
        prog.style.width = (p * 100).toFixed(0) + '%';
        label.textContent = p < 0.15 ? 'waiting for the wave boundary'
          : p < 1 ? 'generating · ' + Math.round(p * 100) + '%' : 'done';
        if (p >= 1) { clearInterval(timer); timer = null; draft.portrait_origin = 'generated'; }
      }, 260);
    }

    const drop = h('div', {
      style: 'border:1px dashed var(--line-2);border-radius:10px;padding:14px;text-align:center;color:var(--ink-faint);font-size:.82rem;cursor:pointer',
      onClick: () => file.click(),
      onDragover: (e) => { e.preventDefault(); drop.style.borderColor = 'var(--accent)'; },
      onDragleave: () => { drop.style.borderColor = 'var(--line-2)'; },
      onDrop: (e) => { e.preventDefault(); drop.style.borderColor = 'var(--line-2)'; useFile(e.dataTransfer.files[0]); },
    }, 'or drop an image here · ', h('span', { style: 'color:var(--accent)' }, 'upload a portrait'));

    const file = h('input', { type: 'file', accept: 'image/*', style: 'display:none',
      onChange: (e) => useFile(e.target.files[0]) });

    function useFile(f) {
      if (!f) return;
      if (timer) { clearInterval(timer); timer = null; }
      const url = URL.createObjectURL(f);
      mount(art, h('img', { src: url, style: 'width:100%;height:100%;object-fit:cover;border-radius:12px' }));
      draft.portrait = url;
      draft.portrait_origin = 'uploaded';
      label.textContent = 'uploaded — this outranks the generator permanently';
      prog.style.width = '100%';
    }

    mount(body, h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'align-items:flex-start;gap:20px' },
        h('div', {}, art),
        h('div', { style: 'flex:1' },
          h('div', { style: 'font-weight:700;margin-bottom:2px' }, 'Generating from the description'),
          h('div', { class: 'tiny dim' }, 'There is no prompt field — the portrait derives from the description, so there is nowhere for the two to drift apart.'),
          progWrap, label,
          h('div', { class: 'row', style: 'margin-top:12px;gap:8px' },
            h('select', { class: 'select', style: 'width:auto' },
              models.map((m) => h('option', { value: m.id, selected: m.default }, `${m.display} · ${m.vram_gib} GiB`))),
            h('button', { class: 'btn sm', onClick: () => { prog.style.width = '0%'; fakeProgress(); } }, '⟳ seed')),
          h('div', { style: 'margin-top:14px' }, drop, file)))));

    fakeProgress();

    mount(foot,
      h('button', { class: 'btn ghost', onClick: () => { step = 1; draw(); } }, '← Back'),
      h('button', { class: 'btn', onClick: () => { step = 3; draw(); } }, 'Skip'),
      h('button', { class: 'btn primary', onClick: () => { step = 3; draw(); } }, 'Next →'));
  }

  // ── step 3 ────────────────────────────────────────────────────────────────

  async function stepInner() {
    const host = h('div', {});
    mount(body, h('div', { class: 'panel' },
      h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:12px' },
        h('div', { style: 'font-weight:700' }, 'Inner life'),
        h('button', { class: 'btn sm', onClick: load }, '⟳ Regenerate')),
      host));

    async function load() {
      mount(host, h('div', { class: 'tiny dim' }, 'generating beliefs, relationships and goals…'));
      const r = await API.generateAttributes({ description: draft.description, archetype_id: draft.archetype_id });
      draft.beliefs = r.beliefs || []; draft.relationships = r.relationships || []; draft.agency = r.agency || [];
      draft.picked.beliefs = new Set(draft.beliefs.slice(0, 2).map((b) => b.belief_id));
      draft.picked.relationships = new Set(draft.relationships.map((x) => x.entity_id));
      draft.picked.agency = new Set(draft.agency.filter((a) => a.state === 'active').map((a) => a.strategy_id));
      paint();
    }

    function group(title, items, keyOf, labelOf, metaOf, set) {
      return h('div', { style: 'margin-bottom:16px' },
        h('div', { class: 'row', style: 'gap:8px;margin-bottom:6px' },
          h('h3', { style: 'margin:0' }, title), h('span', { class: 'chip' }, 'generated')),
        items.length ? items.map((it) => {
          const k = keyOf(it);
          const cb = h('input', { type: 'checkbox', checked: set.has(k),
            onChange: (e) => { e.target.checked ? set.add(k) : set.delete(k); } });
          return h('label', {
            class: 'row', style: 'gap:9px;padding:6px 8px;border-bottom:1px solid var(--line);align-items:flex-start;cursor:pointer',
          }, cb,
            h('div', { style: 'flex:1;min-width:0' },
              h('div', { style: 'font-size:.85rem' }, labelOf(it)),
              h('div', { class: 'tiny dim mono' }, metaOf(it))));
        }) : h('div', { class: 'tiny dim' }, 'none'));
    }

    function paint() {
      mount(host,
        group('Beliefs', draft.beliefs, (b) => b.belief_id, (b) => b.statement,
          (b) => `conf ${b.confidence} · threshold ${b.threshold}`, draft.picked.beliefs),
        group('Relationships', draft.relationships, (r) => r.entity_id, (r) => r.display,
          (r) => `trust ${r.trust} · affect ${r.affect}`, draft.picked.relationships),
        group('Goals', draft.agency, (a) => a.strategy_id, (a) => a.statement,
          (a) => `${a.state} · salience ${a.salience}`, draft.picked.agency),
        h('label', { class: 'row', style: 'gap:9px;margin-top:12px;cursor:pointer' },
          h('input', { type: 'checkbox', checked: draft.environment_enabled,
            onChange: (e) => { draft.environment_enabled = e.target.checked; } }),
          h('div', {},
            h('div', { style: 'font-size:.85rem;font-weight:600' }, 'Environment simulator'),
            h('div', { class: 'tiny dim' },
              'No world simulation is attached, so this generates what happens around the character. Turn it off if your own game drives events.'))));
    }

    mount(foot,
      h('button', { class: 'btn ghost', onClick: () => { step = 2; draw(); } }, '← Back'),
      h('button', { class: 'btn primary', onClick: create }, 'Create'));

    await load();
  }

  async function create() {
    if (!draft.name.trim()) return toast('Give the character a name', 'err');
    try {
      const npc = await API.createNpc({
        name: draft.name, world_id: draft.world_id, archetype_id: draft.archetype_id,
        description: draft.description, environment_enabled: draft.environment_enabled,
        seed: {
          beliefs: draft.beliefs.filter((b) => draft.picked.beliefs.has(b.belief_id)),
          relationships: draft.relationships.filter((r) => draft.picked.relationships.has(r.entity_id)),
          agency: draft.agency.filter((a) => draft.picked.agency.has(a.strategy_id)),
        },
      });
      toast(draft.name + ' created', 'ok');
      go('/npc/' + npc.npc_id);
    } catch (e) { toast(e.detail || e.message || 'could not create', 'err'); }
  }

  function draw() {
    redrawSteps();
    ({ 1: stepIdentity, 2: stepFace, 3: stepInner }[step])();
  }

  draw();
  return { el };
}

/* The mind — browsing and editing the authored corpus.
 *
 * Everything a person wrote is here: the world's canon, the craft libraries the
 * prompt is assembled from, the characters, the settings. Until this existed
 * the only way to change any of it was a text editor on the machine the daemon
 * runs on.
 *
 * # Nothing on this page is a file
 *
 * The console addresses things by what they ARE — `canon/ammo/bolt` — and never
 * learns where the mind keeps them, which extension a section uses, or that
 * there is a filesystem behind it. That is not tidiness: an interface that said
 * `layers/world/ammo/bolt.md` would have promised that canon lives under
 * `layers/` and that prose is markdown, and would break the day either changed.
 * The daemon owns all of it.
 *
 * So there are no folders here, only **collections** and **entries**. A
 * collection can also have text of its own — a topic with an overview — and
 * when it does, opening it shows both what it says and what it holds.
 *
 * # Two panes, one address
 *
 * The left pane is a breadcrumb and one level. Not a whole tree: the canon
 * alone runs to eighteen hundred entries, and drawing all of them to render a
 * sidebar would be slower than reading the corpus. One level is also what makes
 * the world filter cheap — the daemon applies it to the rows it is about to
 * send.
 *
 * The right pane is the open text. No preview: this is prose being corrected,
 * not composed, and a preview would take half the width to show what the author
 * is already looking at.
 *
 * # The world is a lens, not a container
 *
 * A world narrows what is listed — its `selects` gate the canon topics, its
 * `excludes` gate the section categories, its cast gates the characters — but
 * the corpus is one thing underneath. So the selector says "as seen by", and
 * clearing it shows everything rather than nothing.
 */

import { API } from '../lib/api.js';
import { h, mount, fmtK } from '../lib/dom.js';
import { fieldsForm } from '../lib/fields.js';
import { query } from '../lib/router.js';
import { toast, empty, confirmDialog, modal, mayEdit, roNote } from '../lib/ui.js';

/* A count for a collection, a length for an entry. One column, because the two
 * are never shown together and a header saying "size / items" would be
 * describing the table rather than the corpus. */
function measure(n) {
  if (n.kind === 'collection') {
    const inside = n.count || 0;
    const said = n.has_text ? ' · has text' : '';
    return `${fmtK(inside)} inside${said}`;
  }
  const c = n.chars || 0;
  return c < 1000 ? `${c} characters` : `${(c / 1000).toFixed(1)}k characters`;
}

export async function render() {
  const el = h('div', { class: 'page' });
  const q = query();
  let id = q.id || '';
  let world = q.world || '';
  let open = null; // { id, title, text } — what is in the editor
  let dirty = false;

  const listPane = h('div', {});
  const editPane = h('div', {});
  const crumbBar = h('div', { class: 'row', style: 'gap:2px;flex-wrap:wrap;align-items:center' });
  const worldSel = h('select', { class: 'select', style: 'width:auto' });
  const scopeNote = h('span', { class: 'tiny dim' });
  const addBtn = h('span', {});

  /* The URL carries both, so a reload or a shared link reopens the same place
   * through the same lens. `replaceState` rather than a route change, because
   * navigating would rebuild the page and take the editor — and anything
   * unsaved in it — with it. */
  const remember = () => {
    const u = new URLSearchParams();
    if (id) u.set('id', id);
    if (world) u.set('world', world);
    const s = u.toString();
    history.replaceState(null, '', '#/mind' + (s ? '?' + s : ''));
  };

  /* Leaving edited text is the one place this page can lose work, so it is the
   * one place it asks. `confirm` rather than the app's own dialog because it
   * has to answer before navigation continues, and a modal cannot. */
  const confirmLeave = () =>
    !dirty || window.confirm('There are unsaved changes here. Leave without saving?');

  async function goTo(next) {
    if (!confirmLeave()) return;
    id = next;
    open = null;
    dirty = false;
    remember();
    await paintList();
    paintEditor();
  }

  /* Opened as fields where the document has any, and as text where it does not.
   *
   * A section is a set of keys with a conversation in it, and asking somebody
   * to edit that as YAML is asking them to proof-read a serialisation format.
   * A canon page is prose from its first byte to its last, and putting prose in
   * a form would be inventing a structure it does not have.
   *
   * So the daemon is asked for fields first and answers `not_fields` when there
   * are none — a fact about the document rather than a failure, and the text
   * editor is the right surface for those. */
  async function openEntry(target) {
    if (!confirmLeave()) return;
    try {
      const fields = await API.mindFields(target, world || undefined);
      open = { ...fields, mode: 'fields' };
      dirty = false;
      paintEditor();
      return;
    } catch (e) {
      if (e.error !== 'not_fields') {
        toast(e.detail || e.message || 'could not open that', 'err');
        return;
      }
    }
    try {
      open = { ...(await API.mindEntry(target, world || undefined)), mode: 'text' };
      dirty = false;
      paintEditor();
    } catch (e) {
      toast(e.detail || e.message || 'could not open that', 'err');
    }
  }

  // ── left: one level ───────────────────────────────────────────────────────

  /* Built from the address rather than remembered, so a pasted link lands with
   * a working breadcrumb. The titles come from the daemon as it walks down, and
   * the last one is the place itself. */
  function crumbs(place) {
    const out = [h('button', { class: 'btn ghost sm', onClick: () => goTo('') }, 'the mind')];
    if (!place.id) return out;
    const parts = place.id.split('/');
    let acc = '';
    parts.forEach((part, i) => {
      acc = acc ? `${acc}/${part}` : part;
      const here = acc;
      const last = i === parts.length - 1;
      out.push(h('span', { class: 'tiny dim' }, '›'));
      out.push(h('button', {
        class: 'btn ghost sm',
        onClick: () => goTo(here),
      }, last ? place.title : titleOf(part)));
    });
    return out;
  }

  /* A crumb above the current place has no title on the wire — only the place
   * itself does — so the id is made readable the same way the daemon makes one.
   * Display only: the id is never derived from this. */
  const titleOf = (name) => {
    const s = name.replace(/[_-]/g, ' ');
    return s.charAt(0).toUpperCase() + s.slice(1);
  };

  async function paintList() {
    mount(listPane, h('div', { class: 'tiny dim' }, 'reading…'));
    let res;
    try {
      res = await API.mindList(id, world || undefined);
    } catch (e) {
      mount(crumbBar, crumbs({ id, title: titleOf(id.split('/').pop() || '') }));
      mount(
        listPane,
        h('div', { class: 'panel' },
          empty(
            e.error === 'no_mind' ? '◌' : '⊘',
            e.error === 'no_mind' ? 'No mind directory' : 'Not here',
            e.detail || e.message || 'could not read that',
            e.error === 'out_of_scope'
              ? h('button', { class: 'btn sm', onClick: () => setWorld('') }, 'Show the whole mind')
              : null,
          )),
      );
      mount(addBtn);
      return;
    }

    mount(crumbBar, crumbs(res));
    scopeNote.textContent = res.scoped
      ? 'as this world sees it'
      : world
        ? 'this world narrows nothing'
        : 'the whole mind';

    // A place with text of its own is openable as well as openable-into. The
    // button is here rather than in the list because it is about the place the
    // breadcrumb already names.
    const self = res.has_text && res.id
      ? h('button', {
        class: 'btn sm' + (open && open.id === res.id ? ' primary' : ''),
        onClick: () => openEntry(res.id),
      }, 'Open ' + res.title)
      : null;

    // Adding is only possible inside something that can hold entries.
    mount(
      addBtn,
      mayEdit('admin') && res.id
        ? h('button', { class: 'btn sm', onClick: addEntry }, '+ Add to ' + res.title)
        : null,
    );

    /* One row shape for everything — a section, a topic, an entry — so the eye
     * learns it once. The icon carries the kind, the second line carries
     * whatever this row has to say for itself, and the measure is right-aligned
     * in a fixed column so the numbers line up down the pane. */
    const rows = (res.children || []).map((n) =>
      h('button', {
        class: 'mind-row' + (open && open.id === n.id ? ' on' : ''),
        title: n.id,
        onClick: () => (n.kind === 'collection' ? goTo(n.id) : openEntry(n.id)),
      },
        h('span', { class: 'ic' }, n.kind === 'collection' ? '▸' : '◇'),
        h('div', { style: 'flex:1;min-width:0' },
          h('div', { class: 'nm' }, n.title),
          n.blurb ? h('div', { class: 'sub' }, n.blurb) : null),
        h('span', { class: 'meta' }, measure(n))),
    );

    mount(
      listPane,
      self ? h('div', { style: 'margin-bottom:9px' }, self) : null,
      h('div', { class: 'panel', style: 'padding:0;overflow:hidden' },
        rows.length
          ? rows
          : h('div', { class: 'tiny dim', style: 'padding:16px' },
            res.scoped ? 'nothing here that this world includes' : 'nothing here yet')),
    );
  }

  // ── right: the open text ──────────────────────────────────────────────────

  function paintEditor() {
    if (!open) {
      mount(
        editPane,
        h('div', { class: 'panel' },
          empty('◇', 'Nothing open',
            'Pick something on the left. A collection opens into what it holds; '
            + 'anything with text opens here.')),
      );
      return;
    }

    const state = h('span', { class: 'chip' }, 'saved');
    const touched = () => {
      dirty = true;
      state.textContent = 'unsaved';
      state.className = 'chip warn';
    };
    const settled = () => {
      dirty = false;
      state.textContent = 'saved';
      state.className = 'chip';
    };

    // The two bodies. A document with fields gets the form; prose gets the
    // text. `body` is what goes in the frame either way, so the header, the
    // save and the delete are written once.
    let body;
    let save;
    // Shown only where there is somewhere to switch to. A document with fields
    // can always be opened as its text; prose has no second view, so the button
    // is absent rather than dead.
    let swap = null;

    if (open.mode === 'fields') {
      const form = fieldsForm(open.fields, touched, mayEdit('admin'));
      body = form.el;
      swap = h('button', {
        class: 'btn sm ghost',
        title: 'Edit the file itself',
        onClick: async () => {
          if (!confirmLeave()) return;
          try {
            open = {
              ...(await API.mindEntry(open.id, world || undefined)),
              mode: 'text',
              // Remembered, so the way back is offered on exactly the documents
              // that have somewhere to go back to.
              hasFields: true,
            };
            dirty = false;
            paintEditor();
          } catch (e) {
            toast(e.detail || e.message || 'could not open that', 'err');
          }
        },
      }, 'Text');
      save = async () => {
        try {
          await API.saveMindFields(open.id, form.read(), world || undefined);
          settled();
          toast('saved', 'ok');
          paintList();
        } catch (e) {
          // `cannot_patch` is the one refusal worth its own words: it means the
          // document could not be edited without rewriting it whole, which
          // would cost it its comments. The daemon refuses rather than doing
          // that quietly, and the text editor is the way through.
          toast(
            e.error === 'cannot_patch'
              ? 'this one cannot be saved field by field — open it as text'
              : e.detail || e.message || 'could not save',
            'err',
          );
        }
      };
    } else {
      const area = h('textarea', {
        class: 'textarea mono',
        rows: 24,
        style: 'width:100%;font-size:.82rem;line-height:1.5',
        spellcheck: 'false',
        onInput: () => {
          if (area.value !== open.text) touched();
          else settled();
        },
      });
      area.value = open.text;
      if (!mayEdit('admin')) area.setAttribute('readonly', '');
      body = area;
      if (open.hasFields) {
        swap = h('button', {
          class: 'btn sm ghost',
          title: 'Back to the fields',
          onClick: () => openEntry(open.id),
        }, 'Fields');
      }
      save = async () => {
        try {
          await API.saveMindEntry(open.id, area.value, world || undefined, false);
          open.text = area.value;
          settled();
          toast('saved', 'ok');
          // The length beside it in the list is now wrong, and a stale number
          // next to something just edited reads as a save that did not land.
          paintList();
        } catch (e) {
          toast(e.detail || e.message || 'could not save', 'err');
        }
      };
    }

    const remove = () =>
      confirmDialog({
        title: 'Delete ' + open.title,
        danger: true,
        requireText: open.title,
        confirmText: 'Delete permanently',
        message:
          'This removes it from the mind on disk. The mind is not under version '
          + 'control, so there is nothing to restore it from. Anything inside it is kept.',
        onConfirm: async () => {
          try {
            await API.deleteMindEntry(open.id, world || undefined);
            toast('deleted', 'ok');
            open = null;
            dirty = false;
            await paintList();
            paintEditor();
          } catch (e) {
            toast(e.detail || e.message || 'could not delete', 'err');
          }
        },
      });

    mount(
      editPane,
      h('div', { class: 'panel' },
        h('div', { class: 'row', style: 'justify-content:space-between;margin-bottom:8px;gap:9px' },
          h('div', { style: 'min-width:0' },
            h('div', { style: 'font-weight:700' }, open.title),
            h('div', { class: 'tiny dim mono' }, open.id)),
          h('div', { class: 'row', style: 'gap:8px;align-items:center' },
            state,
            swap,
            mayEdit('admin')
              ? h('button', { class: 'btn sm primary', onClick: save }, 'Save')
              : null,
            mayEdit('admin')
              ? h('button', { class: 'btn sm danger ghost', onClick: remove }, 'Delete')
              : null)),
        body,
        mayEdit('admin') ? null : roNote('this')),
    );
  }

  // ── adding ────────────────────────────────────────────────────────────────

  /* A new entry goes inside whatever is open, which is the only place it could
   * sensibly go — the button says so by name.
   *
   * The form asks for a NAME and nothing else. No extension, because the
   * section decides how it is stored; no location, because the breadcrumb
   * already answered that. */
  function addEntry() {
    const input = h('input', { class: 'input', placeholder: 'new-entry' });
    const create = async () => {
      const name = input.value.trim();
      if (!name) return toast('give it a name', 'err');
      const target = `${id}/${name}`;
      try {
        await API.saveMindEntry(target, '', world || undefined, true);
        toast('created', 'ok');
        await paintList();
        await openEntry(target);
      } catch (e) {
        toast(e.detail || e.message || 'could not create', 'err');
      }
    };

    const m = modal({
      title: 'Add to ' + (crumbBar.lastChild ? crumbBar.lastChild.textContent : 'the mind'),
      body: h('div', {},
        h('p', { class: 'tiny dim' },
          'A short name. It becomes part of the address, so letters, digits, '
          + 'hyphens and underscores travel best.'),
        h('label', { class: 'field' }, h('span', {}, 'Name'), input)),
      footer: [
        h('button', { class: 'btn ghost', onClick: () => m.close() }, 'Cancel'),
        h('button', { class: 'btn primary', onClick: () => { m.close(); create(); } }, 'Create'),
      ],
    });
    input.focus();
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); m.close(); create(); }
    });
  }

  // ── the world lens ────────────────────────────────────────────────────────

  async function setWorld(next) {
    if (!confirmLeave()) return;
    world = next;
    open = null;
    dirty = false;
    remember();
    await paintList();
    paintEditor();
  }

  worldSel.addEventListener('change', (e) => setWorld(e.target.value));

  // ── frame ─────────────────────────────────────────────────────────────────

  el.appendChild(
    h('div', { class: 'hd' },
      h('div', {},
        h('h1', {}, 'Mind'),
        h('div', { class: 'sub' },
          'The authored corpus — canon, craft, characters and settings. Saving writes it.')),
      h('div', { class: 'row', style: 'gap:9px;align-items:center' }, worldSel, scopeNote)),
  );

  el.appendChild(
    h('div', { class: 'row', style: 'gap:9px;margin-bottom:10px;align-items:center;flex-wrap:wrap' },
      crumbBar,
      h('span', { class: 'spacer', style: 'flex:1' }),
      addBtn),
  );

  el.appendChild(
    h('div', { class: 'grid', style: 'grid-template-columns:minmax(260px,1fr) 2fr;gap:14px;align-items:start' },
      listPane,
      editPane),
  );

  // An empty option is a real choice rather than a placeholder: it means the
  // whole corpus.
  try {
    const { worlds = [] } = await API.listWorlds();
    mount(worldSel,
      h('option', { value: '', selected: !world }, 'the whole mind'),
      worlds.map((w) =>
        h('option', { value: w.world_id, selected: w.world_id === world }, 'as seen by ' + w.name)));
  } catch (_) {
    mount(worldSel, h('option', { value: '' }, 'the whole mind'));
  }

  /* An address arriving in the URL may name either kind, and a link from
   * elsewhere usually names an entry — a section tile on the world page sends
   * `responses/blush_then_own`, which is a thing to read, not a place to be in.
   *
   * So it is tried as a place first and, failing that, opened as a thing with
   * its parent listed around it. Doing it here rather than making callers say
   * which they meant keeps one kind of link: an address is an address. */
  async function openAddress() {
    if (!id) return paintList();
    try {
      await API.mindList(id, world || undefined);
      return paintList();
    } catch (e) {
      if (e.error !== 'not_found') return paintList();
    }
    const target = id;
    const parent = id.includes('/') ? id.slice(0, id.lastIndexOf('/')) : '';
    id = parent;
    remember();
    await paintList();
    await openEntry(target);
  }

  await openAddress();
  if (!open) paintEditor();

  return { el };
}

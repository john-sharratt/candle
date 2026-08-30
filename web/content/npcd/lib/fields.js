/* Editing a document as fields, and its examples as conversations.
 *
 * The daemon hands over a document as a list of fields — a label, a kind, a
 * value, and the author's own comment from above that key. This turns each kind
 * into a control and hands back the values. Nothing here knows what YAML is,
 * which is the point: the person editing a response's wording should not have
 * to be a serialisation format's proof-reader.
 *
 * # Kinds
 *
 *   line           a short string          input
 *   text           prose                   auto-growing textarea
 *   number         a number                numeric input, a number on the way back
 *   bool           true or false           checkbox
 *   choice         one of a fixed set      select
 *   list           short strings           chips, add and remove
 *   conversations  the `examples:` shape    the editor below
 *   group          a mapping               these same controls, indented
 *   rows           a list of mappings       one titled card each
 *   raw            anything not modelled    a YAML box for THAT VALUE only
 *
 * `group` and `rows` make this recursive, which is what a projection layer
 * needs: its `budget` is numbers nested inside numbers, and its `groups` are a
 * list of mappings with a `selection` inside each. Flat they are two YAML
 * boxes; as themselves they are a dozen inputs with names on them.
 *
 * `raw` is the honest escape hatch. A form that silently dropped the half of a
 * document it did not understand would be worse than one that admits it — so an
 * unmodelled shape is shown as its own YAML and round-trips untouched, and the
 * daemon refuses a malformed one by name rather than writing it out as a string.
 *
 * # Why the conversation editor exists
 *
 * An example is four turns that PRODUCE a reply — the provenance a section is
 * selected by. Read as YAML it is indentation and block scalars; read as a
 * conversation it is two people talking. The second is the thing being authored,
 * so it is the thing shown.
 */

import { h, mount } from './dom.js';

/* Grows to its content, so a long turn is read rather than scrolled. Capped so
 * one enormous example cannot push everything else off the screen. */
function autosize(ta) {
  const fit = () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight + 2, 520) + 'px';
  };
  ta.addEventListener('input', fit);
  // After layout, or `scrollHeight` is measured against a box with no width.
  requestAnimationFrame(fit);
  return ta;
}

const ROLES = ['user', 'assistant', 'system'];

/* One turn. The role decides the colour and the side, so the shape of an
 * exchange is legible before a word of it is read.
 *
 * `content` and `thinking` are both optional and both shown: the final
 * assistant turn of a lead-in is the decode point and carries thinking with no
 * content, which is a shape the editor has to be able to express or it cannot
 * represent the corpus it is editing. */
function turnEditor(turn, onChange, onRemove, canEdit) {
  const role = h('select', {
    class: 'select',
    style: 'width:auto;font-size:.72rem;padding:1px 6px',
    onChange: (e) => { turn.role = e.target.value; onChange(); box.className = 'turn ' + turn.role; },
  }, ROLES.map((r) => h('option', { value: r, selected: r === turn.role }, r)));
  if (!canEdit) role.setAttribute('disabled', '');

  const content = autosize(h('textarea', {
    rows: 2,
    placeholder: 'what is said',
    onInput: (e) => { turn.content = e.target.value; onChange(); },
  }));
  content.value = turn.content || '';

  const thinking = autosize(h('textarea', {
    class: 'think',
    rows: 2,
    placeholder: 'thinking (optional)',
    onInput: (e) => { turn.thinking = e.target.value; onChange(); },
  }));
  thinking.value = turn.thinking || '';

  if (!canEdit) {
    content.setAttribute('readonly', '');
    thinking.setAttribute('readonly', '');
  }

  const box = h('div', { class: 'turn ' + (turn.role || 'user') },
    h('div', { class: 'row', style: 'gap:7px;align-items:center' },
      role,
      h('span', { style: 'flex:1' }),
      canEdit
        ? h('button', {
          class: 'btn ghost sm',
          style: 'padding:0 6px',
          title: 'Remove this turn',
          onClick: onRemove,
        }, '✕')
        : null),
    content,
    thinking);
  return box;
}

/* The gap between two turns, and the place a new one goes.
 *
 * Every gap has one, including the ends, so a turn can be added *anywhere* —
 * which is what the shape needs: a lead-in is user → assistant → user →
 * assistant, and fixing one usually means inserting in the middle rather than
 * appending. Hidden until hovered, so a conversation reads as a conversation. */
function inserter(onAdd) {
  return h('div', { class: 'tween' },
    h('button', { onClick: onAdd, title: 'Add a turn here' }, '+ turn'));
}

/* One example: a note and its turns. */
function conversationEditor(convo, onChange, onRemove, canEdit) {
  const turns = h('div', {});

  const paint = () => {
    const kids = [];
    const list = convo.turns || (convo.turns = []);
    // A gap before each turn and one after the last, so every position is
    // reachable — including an empty conversation, which would otherwise have
    // nowhere to put a first turn.
    list.forEach((t, i) => {
      if (canEdit) kids.push(inserter(() => { list.splice(i, 0, newTurn(list, i)); onChange(); paint(); }));
      kids.push(turnEditor(t, onChange, () => { list.splice(i, 1); onChange(); paint(); }, canEdit));
    });
    if (canEdit) {
      kids.push(inserter(() => { list.push(newTurn(list, list.length)); onChange(); paint(); }));
    }
    mount(turns, kids);
  };

  const note = h('input', {
    placeholder: 'what this example shows',
    onInput: (e) => { convo.note = e.target.value; onChange(); },
  });
  note.value = convo.note || '';
  if (!canEdit) note.setAttribute('readonly', '');

  paint();
  return h('div', { class: 'convo' },
    h('div', { class: 'hd' },
      h('span', { class: 'tiny dim' }, '◇'),
      note,
      canEdit
        ? h('button', { class: 'btn ghost sm', title: 'Remove this example', onClick: onRemove }, '✕')
        : null),
    turns);
}

/* What to call one row of a `rows` field.
 *
 * Its own `id` or `name` if it has one, because that is what the author calls
 * it — a layer's selection groups are `canon`, `held`, `scene`. Only when there
 * is nothing to go on does it fall back to a number, which is a label that
 * tells the reader nothing except where to count from. */
function rowTitle(row, i) {
  const named = (row || []).find((f) => f.key === 'id' || f.key === 'name');
  return named && named.value ? String(named.value) : `#${i + 1}`;
}

/* A new turn alternates from the one before it, because that is what the shape
 * is: user → assistant → user → assistant. Guessing right most of the time
 * beats making somebody set it every time. */
function newTurn(list, at) {
  const before = list[at - 1];
  const role = before && before.role === 'user' ? 'assistant' : 'user';
  return { role, content: '' };
}

/* One field, by kind. Returns { el, read } — `read` gives the value back. */
function fieldEditor(f, onChange, canEdit) {
  const editable = canEdit && !f.readonly;

  if (f.kind === 'group') {
    // A mapping is its own set of fields, indented under this one. A layer's
    // budget is two numbers and an optional pair inside that, which is four
    // inputs — not a YAML box with four numbers hidden in it.
    const form = fieldsForm(f.fields || [], onChange, editable);
    return { el: h('div', { class: 'sub' }, form.el), read: form.read };
  }

  if (f.kind === 'rows') {
    // A list of mappings — a layer's selection groups. Each row is its own set
    // of fields under its own heading, so which one you are editing is never a
    // question of counting brackets.
    const forms = (f.rows || []).map((row) => fieldsForm(row, onChange, editable));
    return {
      el: h('div', { class: 'rows' },
        forms.length
          ? forms.map((form, i) => h('div', { class: 'row-card' },
            h('div', { class: 'hd' }, rowTitle(f.rows[i], i)),
            form.el))
          : h('div', { class: 'tiny dim' }, 'none')),
      read: () => forms.map((form) => form.read()),
    };
  }

  if (f.kind === 'choice') {
    const sel = h('select', { class: 'select', onChange },
      (f.choices || []).map((c) =>
        h('option', { value: c, selected: c === f.value }, c)));
    if (!editable) sel.setAttribute('disabled', '');
    return { el: sel, read: () => sel.value };
  }

  if (f.kind === 'number') {
    // `step: any` so a threshold of 0.3 is typeable, and the value goes back as
    // a NUMBER — a window returned as the string "8000" is a different document
    // that looks identical.
    const input = h('input', { class: 'input', type: 'number', step: 'any' });
    input.value = f.value == null ? '' : String(f.value);
    input.addEventListener('input', onChange);
    if (!editable) input.setAttribute('disabled', '');
    return {
      el: input,
      read: () => {
        // An empty box is not zero. Left blank, the value it had is what the
        // document keeps saying.
        if (input.value.trim() === '') return f.value;
        const n = Number(input.value);
        return Number.isFinite(n) ? n : f.value;
      },
    };
  }

  if (f.kind === 'bool') {
    const box = h('input', { type: 'checkbox', onChange });
    box.checked = f.value === true;
    if (!editable) box.setAttribute('disabled', '');
    return { el: h('label', { class: 'chk' }, box, f.value === true ? ' true' : ' false'),
      read: () => box.checked };
  }

  if (f.kind === 'conversations') {
    const list = Array.isArray(f.value) ? f.value.map((c) => ({ ...c })) : [];
    const host = h('div', {});
    const paint = () => mount(host,
      list.length
        ? list.map((c, i) => conversationEditor(c, onChange, () => { list.splice(i, 1); onChange(); paint(); }, editable))
        : h('div', { class: 'tiny dim', style: 'padding:6px 0' }, 'none yet'),
      editable
        ? h('button', {
          class: 'btn sm',
          style: 'margin-top:4px',
          onClick: () => { list.push({ note: '', turns: [] }); onChange(); paint(); },
        }, '+ Add an example')
        : null);
    paint();
    return { el: host, read: () => list };
  }

  if (f.kind === 'list') {
    const items = Array.isArray(f.value) ? f.value.slice() : [];
    const host = h('div', {});
    const paint = () => mount(host,
      h('div', { class: 'row', style: 'gap:6px;flex-wrap:wrap' },
        items.map((it, i) => h('span', { class: 'chip' }, it,
          editable
            ? h('button', {
              class: 'btn ghost sm',
              style: 'padding:0 4px;margin-left:4px',
              onClick: () => { items.splice(i, 1); onChange(); paint(); },
            }, '✕')
            : null)),
        editable
          ? h('button', {
            class: 'btn sm ghost',
            onClick: () => {
              const v = window.prompt('Add to ' + f.label);
              if (v && v.trim()) { items.push(v.trim()); onChange(); paint(); }
            },
          }, '+')
          : null),
      items.length ? null : h('span', { class: 'tiny dim' }, 'empty'));
    paint();
    return { el: host, read: () => items };
  }

  if (f.kind === 'raw') {
    // The value as YAML, and only this value. The daemon parses it back and
    // refuses a malformed one by name.
    const ta = autosize(h('textarea', { class: 'textarea mono', rows: 4, spellcheck: 'false' }));
    ta.value = f.yaml || '';
    ta.addEventListener('input', onChange);
    if (!editable) ta.setAttribute('readonly', '');
    return { el: ta, read: () => ({ __yaml: ta.value }) };
  }

  if (f.kind === 'text') {
    const ta = autosize(h('textarea', { class: 'textarea', rows: 3 }));
    ta.value = f.value == null ? '' : String(f.value);
    ta.addEventListener('input', onChange);
    if (!editable) ta.setAttribute('readonly', '');
    return { el: ta, read: () => ta.value };
  }

  const input = h('input', { class: 'input' });
  input.value = f.value == null ? '' : String(f.value);
  input.addEventListener('input', onChange);
  if (!editable) input.setAttribute('disabled', '');
  return { el: input, read: () => input.value };
}

/* The whole form. Returns { el, read } — `read` gives back the values map the
 * daemon patches into the document. */
export function fieldsForm(fields, onChange, canEdit) {
  const readers = [];
  const el = h('div', {});
  mount(el, fields.map((f) => {
    const ed = fieldEditor(f, onChange, canEdit);
    readers.push([f.key, ed.read]);
    return h('div', { class: 'fld' },
      h('div', { class: 'lb' }, f.label),
      // The author's own comment, shown where the editing happens rather than
      // only in a file nobody opens. Their paragraphs come through as
      // paragraphs — the note above `examples:` is two of them and eight lines,
      // and as one run it is a wall nobody reads.
      f.note
        ? h('div', { class: 'nt' }, f.note.split('\n\n').map((p) => h('p', {}, p)))
        : null,
      ed.el);
  }));

  return {
    el,
    read: () => {
      const out = {};
      for (const [key, read] of readers) out[key] = read();
      return out;
    },
  };
}

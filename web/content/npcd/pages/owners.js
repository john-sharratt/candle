/* Owners — who each character belongs to, and handing one to somebody else.
 *
 * Ownership is what every other page is checked against: a character shows on
 * "My NPCs", answers its pulse and takes its authoring only for its owner. A
 * character made under one account — a test harness's, a colleague's — is
 * invisible to everyone else, so an admin hands it over here, to any account
 * that has signed in. Admin, like `/v1/npcs/owners` and `/v1/npc/:nid/owner`:
 * this is the one write that crosses ownership. */

import { API } from '../lib/api.js';
import { h, mount } from '/common/lib/dom.js';
import { toast } from '../lib/ui.js';

/* An account as a person would recognise it: chosen name, then display name,
 * then email, with the public id last so two alike can still be told apart. */
function accountLabel(a) {
  const name = (a.unique_name || '').trim();
  const display = (a.display || '').trim();
  const email = (a.email || '').trim();
  const lead = display && display !== name ? `${name || display} (${display})` : name || display || email || a.user_id;
  return email && lead !== email ? `${lead} · ${email}` : lead;
}

export async function render(_params, _q) {
  const el = h('div', { class: 'page wide' });
  const tableHost = h('div', {});
  const note = h('span', { class: 'tiny dim' });
  let data = { me: null, accounts: [], npcs: [] };
  let busy = false;

  const say = (text, bad) => {
    note.textContent = text;
    note.className = 'tiny ' + (bad ? 'st-warn' : 'dim');
  };

  const byId = () => new Map(data.accounts.map((a) => [a.user_id, a]));
  const ownerName = (id) => {
    const a = byId().get(id);
    return a ? accountLabel(a) : `${id} (no account)`;
  };

  async function load() {
    data = await API.getOwners();
    paint();
  }

  async function hand(npc, to) {
    if (busy || npc.owner_id === to) return;
    busy = true;
    try {
      await API.setOwner(npc.npc_id, to);
      toast(`${npc.name} now belongs to ${ownerName(to)}`, 'ok');
      await load();
    } catch (e) {
      say(e.detail || e.message || String(e), true);
    }
    busy = false;
  }

  async function takeAll() {
    const mine = data.me;
    const theirs = data.npcs.filter((n) => n.owner_id !== mine);
    if (!theirs.length || busy) return;
    busy = true;
    let moved = 0;
    for (const n of theirs) {
      try {
        await API.setOwner(n.npc_id, mine);
        moved++;
      } catch (e) {
        say(`${n.name}: ${e.detail || e.message || String(e)}`, true);
      }
    }
    toast(`${moved} character${moved === 1 ? '' : 's'} assigned to you`, 'ok');
    busy = false;
    await load();
  }

  const takeAllBtn = h('button', { class: 'btn sm', onClick: takeAll }, 'Assign every character to me');

  function paint() {
    const accounts = data.accounts;
    takeAllBtn.disabled = !data.npcs.some((n) => n.owner_id !== data.me);
    if (!data.npcs.length) {
      return mount(tableHost, h('div', { class: 'tiny dim' }, 'No characters yet.'));
    }
    mount(tableHost, h('div', { class: 'list' },
      h('table', { class: 't' },
        h('thead', {}, h('tr', {},
          ['Character', 'World', 'Owner', 'Hand to', ''].map((x) => h('th', {}, x)))),
        h('tbody', {}, data.npcs.map((n) => {
          const id = `owner-${n.npc_id}`;
          const pick = h('select', {
            class: 'select', id, 'aria-label': `new owner for ${n.name}`, style: 'width:auto;max-width:28em',
            onChange: (e) => hand(n, e.target.value),
          }, accounts.map((a) => h('option', { value: a.user_id, selected: a.user_id === n.owner_id || null },
            accountLabel(a) + (a.user_id === data.me ? ' — you' : ''))),
          // An owner with no account still has to show as selected, or the
          // picker would claim the character is somebody else's.
          byId().has(n.owner_id) ? null : h('option', { value: n.owner_id, selected: true, disabled: true },
            `${n.owner_id} (no account)`));
          const mineAlready = n.owner_id === data.me;
          return h('tr', {},
            h('td', {}, h('a', { href: `#/npc/${n.npc_id}` }, n.name)),
            h('td', { class: 'tiny' }, n.world_id || '—'),
            h('td', { class: 'tiny' }, mineAlready ? 'you' : ownerName(n.owner_id)),
            h('td', {}, pick),
            h('td', {}, mineAlready ? null : h('button', {
              class: 'btn sm ghost', onClick: () => hand(n, data.me),
            }, 'Assign to me')));
        })))));
  }

  el.appendChild(h('div', { class: 'hd' },
    h('div', {}, h('h1', {}, 'Owners'),
      h('div', { class: 'sub' },
        'Who each character belongs to. A character shows on My NPCs, and takes its owner\'s '
        + 'authoring, only for the account that owns it. Hand one to anybody who has signed in.')),
    h('div', { class: 'row wrap', style: 'gap:6px' }, takeAllBtn)));
  el.appendChild(h('div', { style: 'margin:-6px 0 12px' }, note));
  el.appendChild(tableHost);

  try {
    await load();
  } catch (e) {
    say(e.detail || e.message || String(e), true);
  }
  return { el };
}

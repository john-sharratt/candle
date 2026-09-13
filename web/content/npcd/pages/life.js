/* A personality's authored life.
 *
 * The ladder is four rungs — story, years, months, days — and each is built
 * from the ones above it. That is why they are tabs rather than one long page:
 * the order is a dependency, and a tab strip makes it obvious that you cannot
 * usefully write the months before the years exist.
 *
 * WHAT THIS PAGE IS FOR, AND WHAT IT IS NOT
 *
 * It is a review tool. The model writes; a person reads it, corrects what is
 * wrong, and regenerates what is not worth correcting. Everything here exists
 * to make that loop fast: regenerating one month is one fork off a prefix the
 * daemon has already primed, so it comes back in seconds.
 *
 * THE STICKY RULE, VISIBLE
 *
 * Saving a stratum marks it EDITED, and a generation never overwrites an edited
 * node — only an explicit "regenerate this one" does. That is why an edited
 * card says so, and why the regenerate button on it warns. A tool that eats
 * your corrections when you fix a typo upstream is one you stop using for
 * anything you care about.
 *
 * Editing a stratum also marks everything below it STALE: those documents no
 * longer follow from their parent. Stale is an invitation, not an action — the
 * console offers to bring the subtree back into line and never does it
 * unasked.
 *
 * CONSEQUENCES ARE AN INPUT, NOT AN OUTPUT
 *
 * The belief a day produces is authored HERE, on the day, and generation
 * injects it into the document as a `<tool_call>`. It never round-trips
 * through the model, which is why there is no class of malformed or
 * mis-attributed calls to review. The prompt inverts with it: not "write a day
 * and tell me what it produced" but "write a day that earns these". */

import { API } from '../lib/api.js';
import { h, mount } from '../lib/dom.js';
import { toast, mayEdit, roNote, empty, confirmDialog } from '../lib/ui.js';

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'];
const pad = (n, w = 2) => String(n).padStart(w, '0');
const monthKey = (y, m) => `${pad(y, 4)}-${pad(m)}`;
const dayKey = (y, m, d) => `${pad(y, 4)}-${pad(m)}-${pad(d)}`;

/* Page state. Held in a module-level object rather than threaded through every
 * renderer: the page is one editor over one plan, and passing it down six
 * levels would be ceremony around a single subject. */
const S = { who: null, plan: null, catalog: null, job: null, tab: 'story', year: null, month: null, poll: 0, writtenOnly: false };

/* Whether a stratum has anything in it yet.
 *
 * Title OR text: a day the months phase named has a title and no prose, and it
 * is emphatically not empty — it is the one kind of unwritten node that already
 * says something. */
const written = (node) => Boolean(node && ((node.text || '').trim() || (node.title || '').trim()));

/* An index down the left, one editor on the right.
 *
 * # Why a pane rather than a list of cards
 *
 * These tabs used to render a full editor — title, textarea, Save, Regenerate —
 * for every slot, and a life has as many slots as it has time. Yen is 362
 * years, two of them written, so the tab meant to show a life showed two
 * paragraphs and three hundred and sixty empty textareas and the written ones
 * could not be found by scrolling.
 *
 * The slots are all real: they are what "Generate what is missing" fills, and
 * any of them can be written by hand. So none is hidden. What changes is that
 * the *list* is a list — a scannable column where a year is one row and its
 * state is a colour — and the editor is the one thing you are actually editing.
 *
 * The index carries the same three states the cards did, because they are how
 * you decide what to look at next: written, edited by hand (sticky, nothing
 * regenerates over it), and out of date (a parent moved under it).
 */
function pane(host, items, opts) {
  const list = S.writtenOnly ? items.filter(written) : items;
  const gaps = items.length - items.filter(written).length;
  const sel = list.find(opts.selected) || list[0];
  return h('div', { class: 'life-split' },
    h('nav', { class: 'life-index' },
      h('div', { class: 'life-index-head' },
        h('span', {}, `${items.length - gaps}/${items.length} written`),
        !gaps ? null : h('button', {
          class: 'life-index-filter' + (S.writtenOnly ? ' on' : ''),
          title: 'Show only the ones with something in them',
          onClick: () => { S.writtenOnly = !S.writtenOnly; draw(host); },
        }, S.writtenOnly ? 'showing written' : 'show written only')),
      !list.length
        ? h('p', { class: 'note' }, 'Nothing written at this level yet.')
        : h('div', { class: 'life-index-rows' }, ...list.map((it) => h('button', {
          class: 'life-index-row'
            + (sel && opts.key(it) === opts.key(sel) ? ' on' : '')
            + (written(it) ? '' : ' blank')
            + (it.edited ? ' edited' : '') + (it.stale ? ' stale' : ''),
          onClick: () => { opts.pick(it); draw(host); },
        },
        h('span', { class: 'k' }, opts.label(it)),
        h('span', { class: 'v' }, opts.meta ? opts.meta(it) : (it.title || ''))))) ),
    h('div', { class: 'life-pane' },
      sel ? opts.render(sel)
        : h('p', { class: 'note' }, 'Nothing to edit at this level yet.')));
}

export async function render(params, q) {
  const who = params.aid || q.a;
  /* The page's state outlives one render, so a different character must clear
   * the year and month with it. Carried over, they select a year the new life
   * does not have — Cindy's 1998 on a character born in 2010 — and the pickers
   * come up empty with nothing on screen saying why. */
  if (who !== S.who) { S.year = null; S.month = null; S.plan = null; S.job = null; }
  S.who = who;
  S.tab = params.tab || q.tab || 'story';
  const host = h('div', { class: 'life' });
  if (!S.who) {
    mount(host, empty('○', 'No personality', 'Open a personality to write their life.'));
    return host;
  }

  /* **A failed catalog fetch is not cached.** Assigning the fallback made
   * `S.catalog` truthy, so a single failed read — a daemon still loading, one
   * dropped request — left the console with no phases, no cadences and no tools
   * for the life of the tab, and every later visit skipped the retry. Only a
   * successful read is kept; a failure falls back for this render and is asked
   * again on the next. */
  if (!S.catalog) {
    S.catalog = await API.getLifeCatalog().catch(() => null);
  }
  await reload();
  draw(host);
  /* The poll is the page's only timer and it belongs to the page. Left running,
   * the console would keep asking the daemon about a job nobody is watching for
   * as long as the tab stayed open. */
  return { el: host, teardown: () => { if (S.poll) { clearInterval(S.poll); S.poll = 0; } } };
}

async function reload() {
  /* A read that FAILED is not a character without a life, and the difference is
   * destructive. Both render `plan: null`, but the empty state offers "Create
   * the life", and that PUTs a seed — whose carry-over is
   * `if let Ok(Some(old)) = plan::load(...)`, so the `Err` arm falls through and
   * a fresh empty plan is written over the authored story, years and months. A
   * 500 from an unparseable plan file is exactly when the existing life is most
   * worth keeping, and exactly when this offered to replace it.
   *
   * So the error is kept and rendered as an error. `plan: undefined` — not
   * `null` — is what distinguishes "we do not know" from "there is none". */
  const r = await API.getLife(S.who).catch((e) => ({ error: e }));
  S.error = r.error || null;
  S.plan = r.plan || null;
  S.job = r.job || null;
  S.counts = r.counts || {};
  /* Re-seeding can move a life's dates, so a selection that was valid a moment
   * ago may name a year the plan no longer has. Checked against the plan rather
   * than merely defaulted when unset. */
  if (S.plan && !yearOf(S.year)) S.year = defaultYear();
  if (!monthOf(S.year, S.month)) S.month = firstMonth();
  /* Also on load, not only on a tab click: arriving straight at
   * `/personality/:id/life/days` from a link is exactly the case where the
   * selection has never been focused for the tab being shown. */
  if (S.plan) focus(S.tab);
}

/* Which year to open on.
 *
 * **Not `years[0]`.** A life is laid out across its whole span before a word of
 * it is written, so the first year is the year of somebody's birth and is
 * almost always blank — landing there showed an empty Months tab for a life
 * with written years in it, which reads as a broken page rather than an
 * unwritten decade.
 *
 * The selection only drives the Months and Days tabs, so it is ordered by what
 * those tabs can actually show. A year's grain decides whether it has months at
 * all: a century of coarse vigil has none, so the written year is not always a
 * year the months tab can render, and preferring prose alone lands on a year
 * with nothing below it. */
function defaultYear() {
  const ys = (S.plan && S.plan.years) || [];
  const months = (y) => (y.months || []).length;
  const pick =
    // A month with something in it — the tab opens on real content.
    ys.find((y) => (y.months || []).some(written))
    // Failing that, somewhere with months to write into.
    || ys.find(months)
    // Failing that, a written year, so the picker at least names the part of
    // the life that exists.
    || ys.find(written)
    || ys[0];
  return (pick || {}).year ?? null;
}

/* The catalog, or an empty one when the read failed.
 *
 * An accessor rather than a cached fallback: a failure must not be *kept*, or a
 * single dropped request leaves the console with no phases for the life of the
 * tab. The empty shape is supplied here, per read, so the next render asks
 * again. */
const cat = () => S.catalog || { tools: [], cadences: [], phases: [] };

const yearOf = (y) => (S.plan ? S.plan.years.find((x) => x.year === y) : null);
/* The month to open on, written by preference — same reason as [`defaultYear`].
 * A year whose months are all blank still opens on its first. */
const firstMonth = () => {
  const y = yearOf(S.year);
  if (!y || !y.months.length) return null;
  return (y.months.find(written) || y.months[0]).month;
};
const monthOf = (y, m) => { const yy = yearOf(y); return yy ? yy.months.find((x) => x.month === m) : null; };

/* ---- rendering ------------------------------------------------------- */

function draw(host) {
  mount(host,
    header(host),
    /* A stable slot rather than the overlay itself. The progress poll ticks
     * every 1.2 s while a generation runs, and `mount` is `replaceChildren` —
     * so redrawing the whole page on each tick tore out whatever the operator
     * was typing, once per second and a bit, in the editor right beside the
     * bar they were watching. `tick` below refills only this. */
    h('div', { class: 'life-overlay' }, S.job && !S.job.outcome ? overlay(host) : null),
    !S.plan ? seedTab(host) : tabs(host),
    !S.plan ? null : body(host));
}

/* Refresh the progress overlay in place, leaving the rest of the page — and
 * every input in it — untouched. */
function tick(host) {
  const slot = host.querySelector(':scope > .life-overlay');
  if (!slot) return;
  mount(slot, S.job && !S.job.outcome ? overlay(host) : null);
}

const redraw = (host) => reload().then(() => draw(host));

function header(host) {
  const c = S.counts || {};
  const n = (k) => (c[k] ? `${c[k][0]}/${c[k][1]}` : '—');
  const running = S.job && !S.job.outcome;
  return h('header', { class: 'life-head' },
    h('h1', {}, S.who),
    S.plan ? h('div', { class: 'life-counts' },
      h('span', {}, `story ${n('story')}`),
      h('span', {}, `years ${n('years')}`),
      h('span', {}, `months ${n('months')}`),
      h('span', {}, `days ${n('days')}`)) : null,
    S.plan && mayEdit() ? h('div', { class: 'life-actions' },
      running
        ? h('button', { class: 'btn danger', onClick: () => API.cancelLife(S.who).then(() => redraw(host)) }, 'Stop')
        : h('button', {
            class: 'btn primary',
            /* Everything not yet written, top to bottom. The daemon puts the
             * phases in ladder order whatever we ask for, so this cannot
             * generate months against an outline nothing expanded. */
            onClick: () => start(host, []),
          }, 'Generate everything missing')) : null);
}

/* The overlay, and why it reports two stages.
 *
 * A forked wave finishes in clumps, so a bar that sits at 0/12 and jumps to
 * 12/12 says nothing during exactly the seconds somebody is wondering whether
 * it has hung. Priming is its own stage with no counter — honest, because there
 * is nothing to count — and the fan-out carries both a done count and how many
 * forks are in flight. */
function overlay(host) {
  const j = S.job;
  const pct = Math.round((j.progress || 0) * 100);
  if (!S.poll) S.poll = setInterval(async () => {
    S.job = await API.getLifeJob(S.who).catch(() => null);
    /* Finished: the plan has changed underneath, so the whole page is redrawn.
     * Still running: only the overlay, because the rest of the page may hold an
     * editor with unsaved text in it. */
    if (!S.job || S.job.outcome) { clearInterval(S.poll); S.poll = 0; redraw(host); }
    else tick(host);
  }, 1200);
  return h('div', { class: 'life-progress' },
    h('div', { class: 'life-phases' },
      ...(cat().phases || []).map((p) => h('span', {
        class: 'life-phase' + (p.value === j.phase ? ' now' : (j.completed || []).includes(p.value) ? ' done' : ''),
      }, p.label))),
    h('div', { class: 'bar' }, h('div', { class: 'fill', style: `width:${pct}%` })),
    h('p', { class: 'life-detail' },
      j.stage_label,
      j.total ? ` — ${j.done} of ${j.total} ${j.unit}` : '',
      j.in_flight ? ` · ${j.in_flight} in flight` : '',
      j.detail ? ` · ${j.detail}` : ''));
}

/* Move the year/month selection to where the tab being opened has something.
 *
 * The two tabs share one selection and want different things from it: Months
 * wants a year with months, Days wants a month with days, and a life can easily
 * have those in different years — Yen's months are a two-year window in the
 * 2770s and her notable days are in 2775 and 3086. Opening Days on the month
 * Months happened to leave selected showed "no days marked" for a life with
 * days in it.
 *
 * Only ever moves off a selection that has nothing to show, so it cannot pull
 * somebody away from the year they were reading. */
function focus(tab) {
  const ys = (S.plan && S.plan.years) || [];
  if (tab === 'years') {
    /* The selection is shared, and the other two tabs want a year with months
     * under it — which for a coarse-grained span is not a year anybody has
     * written. Opening the editor on a blank year while the written ones sit in
     * the index is the same complaint this pane was built to answer. */
    if (written(yearOf(S.year))) return;
    const w = ys.find(written);
    if (w) S.year = w.year;
    return;
  }
  if (tab !== 'days') return;
  if ((monthOf(S.year, S.month) || {}).days?.length) return;
  for (const y of ys) {
    const m = (y.months || []).find((x) => x.days.length);
    if (m) { S.year = y.year; S.month = m.month; return; }
  }
}

function tabs(host) {
  const t = (id, label) => h('button', {
    class: 'life-tab' + (S.tab === id ? ' on' : ''),
    onClick: () => { S.tab = id; focus(id); draw(host); },
  }, label);
  return h('nav', { class: 'life-tabs' },
    t('story', 'Life story'), t('years', 'Years'),
    t('months', 'Months'), t('days', 'Notable days'), t('seed', 'Seed'));
}

function body(host) {
  switch (S.tab) {
    case 'seed': return seedTab(host);
    case 'years': return yearsTab(host);
    case 'months': return monthsTab(host);
    case 'days': return daysTab(host);
    default: return storyTab(host);
  }
}

/* ---- seed ------------------------------------------------------------ */

/* The seed is the reusable half, not the life. Two characters from one seed
 * must get two different lives, which is why nothing generated is stored in
 * it and why `cadence` is here: left alone every authored life bends toward
 * the same arc, and a city whose whole cast shares it reads as one person. */
function seedTab(host) {
  const s = (S.plan && S.plan.seed) || {};
  const f = {};
  const field = (k, label, value, type = 'text') =>
    h('label', { class: 'field' }, h('span', {}, label),
      (f[k] = h('input', { class: 'input', type, value: value ?? '', disabled: !mayEdit() })));
  const area = (k, label, value) =>
    h('label', { class: 'field wide' }, h('span', {}, label),
      (f[k] = h('textarea', { class: 'textarea', rows: 4, disabled: !mayEdit() }, value ?? '')));

  const cadence = h('select', { class: 'select', disabled: !mayEdit() },
    ...(cat().cadences || []).map((c) => h('option', {
      value: c.value, selected: c.value === (s.cadence || 'even'),
    }, c.label)));
  f.cadence = cadence;

  return h('section', { class: 'life-seed' },
    /* Three states, not two: a life, no life, or no answer. The last one must
     * not read as the middle one — see `reload`. */
    S.error ? h('p', { class: 'note error' },
      'Could not read this life: ' + (S.error.message || S.error)
      + '. Saving a seed now would replace whatever is on disk, so it is disabled '
      + 'until the read succeeds.') : null,
    !S.plan && !S.error ? h('p', { class: 'note' },
      'This character has no life yet. Give them a seed and the ladder can start — the '
      + 'story, years, months and days tabs appear once there is one to hang them on.') : null,
    h('div', { class: 'grid' },
      field('display', 'Name', s.display || S.who),
      field('born', 'Born', s.born, 'date'),
      /* Where the AUTHORED life stops and the LIVED one begins. Everything
       * after this date is what the character accumulates in play, and
       * regenerating must never reach it. */
      field('through', 'Authored through', s.through, 'date'),
      field('place', 'Place', s.place),
      field('role', 'Became', s.role),
      h('label', { class: 'field' }, h('span', {}, 'Cadence'), cadence)),
    area('facts', 'Things that must be true (one per line)', (s.facts || []).join('\n')),
    area('world', 'World events — `YYYY[-MM[-DD]] | what happened` (one per line)',
      (s.world || []).map((w) => `${w.date} | ${w.what}`).join('\n')),
    area('cast', 'People who already exist — `entity-id | Name | who they are` (one per line)',
      (s.cast || []).map((c) => `${c.entity_id} | ${c.display} | ${c.what || ''}`).join('\n')),
    /* Disabled while the read is unresolved: a seed save is a full rewrite, and
     * the daemon's carry-over silently skips a plan it could not load. */
    mayEdit() ? h('button', {
      class: 'btn primary',
      disabled: !!S.error,
      onClick: async () => {
        const rows = (k) => f[k].value.split('\n').map((l) => l.trim()).filter(Boolean);
        const seed = {
          who: S.who,
          display: f.display.value.trim(),
          born: f.born.value.trim(),
          through: f.through.value.trim(),
          place: f.place.value.trim(),
          role: f.role.value.trim(),
          cadence: f.cadence.value,
          facts: rows('facts'),
          world: rows('world').map((l) => { const p = l.split('|'); return { date: (p[0] || '').trim(), what: (p[1] || '').trim() }; }),
          cast: rows('cast').map((l) => {
            const p = l.split('|');
            return { entity_id: (p[0] || '').trim(), display: (p[1] || '').trim(), what: (p[2] || '').trim() };
          }),
        };
        try {
          await API.setLifeSeed(S.who, seed);
          toast('Seed saved');
          S.year = null; S.month = null;
          await redraw(host);
        } catch (e) { problems(e); }
      },
    }, S.plan ? 'Save seed' : 'Create the life') : roNote('the seed'));
}

/* The daemon reports every problem with a seed at once, so the toast does too —
 * a validator that stops at the first fault turns one correction into five
 * round trips, and a UI that shows only the first undoes that. */
function problems(e) {
  const list = (e && e.problems) || [];
  toast(list.length ? list.map((p) => p.message).join(' · ') : (e.detail || e.message || 'Failed'), 'error');
}

/* ---- story ----------------------------------------------------------- */

function storyTab(host) {
  const st = S.plan.story || {};
  return h('section', { class: 'life-story' },
    stratum(host, 'story', st, 'The arc',
      'Four to eight paragraphs. Everything below expands what this decides, so anything left out will not appear later.'),
    h('h3', {}, 'The cast'),
    h('p', { class: 'note' },
      'Fixed here and inherited by every stratum below. Nothing lower invents a person, which is what stops one professor becoming four different ids.'),
    // Wrapped so the table scrolls rather than the page: four columns of names
    // and prose cannot reflow, and a body that scrolls sideways on a phone is
    // the one layout failure that makes everything else feel broken too.
    (st.cast || []).length ? h('div', { class: 'life-tables' }, h('table', { class: 'life-cast' },
      h('thead', {}, h('tr', {}, h('th', {}, 'id'), h('th', {}, 'Name'), h('th', {}, 'Who they are'), h('th', {}, ''))),
      h('tbody', {}, ...(st.cast || []).map((c) => h('tr', {},
        h('td', {}, h('code', {}, c.entity_id)),
        h('td', {}, c.display),
        h('td', {}, c.what || ''),
        h('td', {}, c.from_seed ? h('span', { class: 'chip' }, 'seeded') : null,
          c.npc_id != null ? h('span', { class: 'chip' }, 'NPC ' + c.npc_id) : null)))))) : h('p', { class: 'note' }, 'Nobody named yet.'),
    h('h3', {}, 'The years'),
    h('p', { class: 'note' },
      'What each year is FOR. The years tab expands these; it does not choose them.'),
    (st.outline || []).length ? h('div', { class: 'life-tables' }, h('table', { class: 'life-outline' },
      h('tbody', {}, ...(st.outline || []).map((b) => h('tr', {},
        h('td', {}, h('strong', {}, b.year)),
        h('td', {}, b.title),
        h('td', {}, b.premise)))))) : h('p', { class: 'note' }, 'No outline yet.'));
}

/* ---- years / months / days ------------------------------------------- */

function yearsTab(host) {
  return h('section', {},
    phaseBar(host, 'years', 'Write every year the story outlined.'),
    pane(host, S.plan.years, {
      key: (y) => y.year,
      label: (y) => String(y.year),
      /* The outline's own one-liner where the year has no title yet: it is what
       * the story phase decided this year is *for*, so it is the most useful
       * thing to scan when choosing what to write next. */
      meta: (y) => y.title || (beat(y.year) ? beat(y.year).title : ''),
      selected: (y) => y.year === S.year,
      pick: (y) => { S.year = y.year; S.month = firstMonth(); },
      render: (y) => stratum(host, String(y.year), y,
        `${y.year}${beat(y.year) ? ' — ' + beat(y.year).title : ''}`,
        beat(y.year) ? beat(y.year).premise : ''),
    }));
}

const beat = (year) => ((S.plan.story.outline || []).find((b) => b.year === year));

function monthsTab(host) {
  const y = yearOf(S.year);
  return h('section', {},
    phaseBar(host, 'months', 'Every month is written — a quiet month is written as a quiet month, never skipped.'),
    yearPicker(host),
    !y ? empty('○', 'No year', 'This life has no years yet.') :
      /* A year with no months is not a gap — it is a year the seed gave a
       * coarse grain, and no amount of generating will put months in it. Said
       * plainly, because an empty list here otherwise reads as a failure. */
      !y.months.length ? h('p', { class: 'note' },
        `${y.year} is written at year grain, so it has no months. `
        + 'Set that span\'s grain to months on the Seed tab if you want them.') :
      pane(host, y.months, {
        key: (m) => m.month,
        label: (m) => MONTHS[m.month - 1],
        /* The day count, because that is what the months phase decides and what
         * the Days tab has to work with — a month with none is a month that
         * named nothing worth remembering. */
        meta: (m) => m.title
          || (m.days.length ? `${m.days.length} day${m.days.length > 1 ? 's' : ''}` : ''),
        selected: (m) => m.month === S.month,
        pick: (m) => { S.month = m.month; },
        render: (m) => stratum(host, monthKey(y.year, m.month), m,
          `${MONTHS[m.month - 1]} ${y.year}`, '',
          () => { S.tab = 'days'; S.month = m.month; draw(host); },
          m.days.length ? `${m.days.length} notable day${m.days.length > 1 ? 's' : ''}` : 'no notable days'),
      }));
}

function daysTab(host) {
  const m = monthOf(S.year, S.month);
  return h('section', {},
    phaseBar(host, 'days', 'Each day is written to EARN the consequences you set on it.'),
    yearPicker(host), monthPicker(host),
    !m ? empty('○', 'No month', 'Pick a month with a written life.') :
      h('div', {},
        m.days.length ? h('div', {}, ...m.days.map((d) => dayCard(host, d))) :
          h('p', { class: 'note' }, 'No days marked. The month names its own; you can add one below.'),
        mayEdit() ? addDay(host) : null));
}

function yearPicker(host) {
  return h('label', { class: 'field inline' }, h('span', {}, 'Year'),
    h('select', {
      class: 'select',
      onChange: (e) => { S.year = Number(e.target.value); S.month = firstMonth(); draw(host); },
    }, ...S.plan.years.map((y) => h('option', { value: y.year, selected: y.year === S.year },
      /* A year with blank prose but written months is not unwritten — labelling
       * it so sent somebody past the only year that had anything in it. */
      `${y.year}${written(y) || (y.months || []).some(written) ? '' : ' (unwritten)'}`))));
}

function monthPicker(host) {
  const y = yearOf(S.year);
  if (!y) return null;
  return h('label', { class: 'field inline' }, h('span', {}, 'Month'),
    h('select', { class: 'select', onChange: (e) => { S.month = Number(e.target.value); draw(host); } },
      ...y.months.map((m) => h('option', { value: m.month, selected: m.month === S.month },
        `${MONTHS[m.month - 1]}${m.days.length ? ` · ${m.days.length}` : ''}`
        + (written(m) ? '' : ' (unwritten)')))));
}

function phaseBar(host, phase, note) {
  return h('div', { class: 'life-phasebar' },
    h('p', { class: 'note' }, note),
    mayEdit() ? h('button', { class: 'btn', onClick: () => start(host, [phase]) }, 'Generate what is missing') : null);
}

async function start(host, phases, redo) {
  try {
    S.job = await API.generateLife(S.who, { phases, redo: redo || [] });
    draw(host);
  } catch (e) {
    toast(e.detail || e.message || 'Could not start', 'error');
  }
}

/* One editable stratum: title, prose, and what has happened to it. */
function stratum(host, key, node, heading, hint, open, aside) {
  const edit = mayEdit();
  const title = h('input', { class: 'input', value: node.title || '', disabled: !edit, placeholder: 'Title' });
  const text = h('textarea', { class: 'textarea', rows: node.text ? 12 : 3, disabled: !edit, placeholder: hint || '' }, node.text || '');
  return h('article', { class: 'life-card' + (node.stale ? ' stale' : '') + (node.edited ? ' edited' : '') },
    h('header', {},
      h('h3', { onClick: open || null, class: open ? 'link' : '' }, heading),
      node.edited ? h('span', { class: 'chip edited' }, 'edited by hand') : null,
      /* Stale is an invitation, never an action: these documents no longer
       * follow from a parent that changed, and the console says so rather than
       * quietly regenerating over somebody's afternoon. */
      node.stale ? h('span', { class: 'chip stale', title: 'A parent changed after this was written' }, 'out of date') : null,
      aside ? h('span', { class: 'chip' }, aside) : null),
    h('div', { class: 'life-body' }, title, text),
    edit ? h('div', { class: 'life-rowactions' },
      h('button', {
        class: 'btn primary sm',
        onClick: async () => {
          try {
            const r = await API.setLifeNode(S.who, key, { title: title.value, text: text.value });
            toast(r.stale_below ? `Saved — ${r.stale_below} below are now out of date` : 'Saved');
            await redraw(host);
          } catch (e) { problems(e); }
        },
      }, 'Save'),
      h('button', {
        class: 'btn sm',
        /* The ONLY way to overwrite an edited node, which is what makes the
         * stickiness real rather than advisory. */
        onClick: () => {
          /* Only the node — the daemon derives which rung writes it. A second
           * copy of that mapping in JavaScript would be a second place for it to
           * be wrong, and a mismatch between the two regenerates nothing at all,
           * silently. */
          const go = () => start(host, [], [key]);
          if (!node.edited) return go();
          confirmDialog({
            title: 'Regenerate over your edit?',
            message: 'You wrote this by hand. Regenerating replaces it with the model\'s version.',
            confirmText: 'Regenerate',
            danger: true,
            onConfirm: go,
          });
        },
      }, 'Regenerate'),
      /* **Narrate — this one stratum, in the narrator's voice.**
       *
       * Beside Regenerate rather than replacing it, because they are different
       * things. Regenerate queues the ladder: it may run several rungs, it fans
       * out, and it comes back as a job you watch. This asks for exactly this
       * node and returns the prose — one prompt, one answer, no job to poll.
       *
       * It disables itself while it runs rather than letting a second press
       * queue a second rewrite of the same node. */
      h('button', {
        class: 'btn sm',
        title: 'Rewrite just this stratum in the narrator\'s voice.',
        onClick: async (e) => {
          const b = e.target;
          const run = async () => {
            const was = b.textContent;
            b.disabled = true;
            b.textContent = '◍ narrating…';
            try {
              const r = await API.narrateLifeNode(S.who, key, { force: node.edited });
              toast(`Narrated — ${r.tokens} tokens`
                + (r.stale_below ? `, ${r.stale_below} below are now out of date` : ''), 'ok');
              await redraw(host);
            } catch (err) {
              problems(err);
            } finally {
              b.disabled = false;
              b.textContent = was;
            }
          };
          if (!node.edited) return run();
          confirmDialog({
            title: 'Narrate over your edit?',
            message: 'You wrote this by hand. Narrating replaces it with the narrator\'s version.',
            confirmText: 'Narrate',
            danger: true,
            onConfirm: run,
          });
        },
      }, 'Narrate')) : null);
}

/* ---- a day, and the consequences it must earn ------------------------ */

function dayCard(host, d) {
  const key = dayKey(S.year, S.month, d.day);
  return h('article', { class: 'life-card day' + (d.stale ? ' stale' : '') + (d.edited ? ' edited' : '') },
    h('header', {},
      h('h3', {}, `${MONTHS[S.month - 1]} ${d.day}`),
      h('span', { class: 'chip' }, d.title || 'untitled'),
      d.edited ? h('span', { class: 'chip edited' }, 'edited by hand') : null,
      d.stale ? h('span', { class: 'chip stale' }, 'out of date') : null,
      mayEdit() ? h('button', {
        class: 'btn ghost sm',
        onClick: () => confirmDialog({
          title: `Forget ${key}?`,
          message: 'The day stops being one worth remembering and its document is removed, so the character stops remembering it.',
          confirmText: 'Forget it',
          danger: true,
          onConfirm: async () => {
            await API.removeLifeDay(S.who, key).catch(problems);
            await redraw(host);
          },
        }),
      }, 'Forget') : null),
    d.text ? h('p', { class: 'life-prose' }, d.text) : h('p', { class: 'note' }, 'Not written yet.'),
    consequences(host, key, d.consequences || []),
    mayEdit() ? h('div', { class: 'life-rowactions' },
      h('button', { class: 'btn sm', onClick: () => start(host, [], [key]) },
        d.text ? 'Rewrite this day' : 'Write this day')) : null);
}

/* The consequence editor.
 *
 * Built from the daemon's own authoring catalog, so a tool added in Rust
 * appears here without a second edit — and so this cannot offer a tool the
 * executor does not have. Entities come from the story's cast for the same
 * reason ids are fixed up there: a typed one would be a second person who never
 * appears again. */
function consequences(host, key, list) {
  const cast = S.plan.story.cast || [];
  const tools = cat().tools || [];
  const rows = h('div', { class: 'life-cons' });
  let draft = list.map((c) => ({ tool: c.tool, args: { ...c.args } }));

  const paint = () => mount(rows, ...draft.map((c, i) => consRow(c, i)),
    mayEdit() ? h('div', { class: 'life-consadd' },
      ...tools.map((t) => h('button', {
        class: 'btn ghost sm',
        onClick: () => { draft.push({ tool: t.name, args: {} }); paint(); },
      }, '+ ' + t.name.replace(/_/g, ' ')))) : null,
    mayEdit() ? h('button', {
      class: 'btn primary sm',
      onClick: async () => {
        try {
          await API.setLifeConsequences(S.who, key, draft.filter((c) => c.tool));
          toast('Consequences saved');
          await redraw(host);
        } catch (e) { problems(e); }
      },
    }, 'Save consequences') : null);

  const consRow = (c, i) => {
    const tool = tools.find((t) => t.name === c.tool);
    if (!tool) return h('p', { class: 'note' }, `Unknown tool ${c.tool}`);
    return h('div', { class: 'life-consrow' },
      h('strong', {}, tool.name.replace(/_/g, ' ')),
      h('span', { class: 'note', title: tool.description }, `→ ${tool.writes}`),
      ...tool.params.map((p) => argField(c, p, tool, cast)),
      mayEdit() ? h('button', {
        class: 'btn ghost sm', onClick: () => { draft.splice(i, 1); paint(); },
      }, 'remove') : null);
  };
  paint();
  return h('div', {}, h('h4', {}, 'This day must produce'),
    h('p', { class: 'note' },
      'You author these; the model never writes one. They are injected into the document, and the prose is asked to earn them.'),
    rows);
}

/* Dials are sliders because they have real ranges with real meaning, and
 * familiarity's is different from the others: you cannot know somebody less
 * than not at all, so it has no negative half. */
const RANGE = { trust: [-1, 1], affect: [-1, 1], familiarity: [0, 1], confidence: [0, 1], threshold: [0, 1] };

function argField(c, p, tool, cast) {
  const req = (tool.required || []).includes(p);
  const r = RANGE[p];
  if (p === 'entity_id') {
    return h('label', { class: 'field inline' }, h('span', {}, p),
      h('select', {
        class: 'select',
        disabled: !mayEdit(),
        onChange: (e) => { c.args[p] = e.target.value; },
      }, h('option', { value: '' }, '—'),
        ...cast.map((m) => h('option', { value: m.entity_id, selected: m.entity_id === c.args[p] },
          `${m.display} (${m.entity_id})`))));
  }
  if (r) {
    /* An UNSET dial and a dial set to zero are different things, and the
     * difference is load-bearing: `revise_relationship` moves only the dials it
     * names, so leaving trust alone and setting it to exactly 0.0 produce
     * different characters. A slider has to sit somewhere, so the label is
     * dimmed and the readout says `—` until somebody actually moves it — the
     * position alone would read as a value nobody chose. */
    const set = c.args[p] != null;
    const out = h('output', {}, set ? c.args[p] : '—');
    const label = h('label', { class: 'field inline' + (set ? '' : ' unset') }, h('span', {}, p),
      h('input', {
        type: 'range', min: r[0], max: r[1], step: 0.05,
        value: set ? c.args[p] : (r[0] + r[1]) / 2,
        disabled: !mayEdit(),
        onInput: (e) => {
          c.args[p] = Number(e.target.value);
          out.textContent = e.target.value;
          label.classList.remove('unset');
        },
      }), out,
      /* And a way back to unset, or a dial touched by accident could never be
       * un-set without reloading the day. */
      mayEdit() ? h('button', {
        class: 'btn ghost sm', title: 'leave this dial alone',
        onClick: () => { delete c.args[p]; out.textContent = '—'; label.classList.add('unset'); },
      }, '×') : null);
    return label;
  }
  return h('label', { class: 'field inline' }, h('span', {}, p + (req ? ' *' : '')),
    h('input', {
      class: 'input', value: c.args[p] ?? '', disabled: !mayEdit(),
      onInput: (e) => { c.args[p] = e.target.value; },
    }));
}

function addDay(host) {
  const y = S.year, m = S.month;
  const last = new Date(y, m, 0).getDate();
  const day = h('select', { class: 'select' }, ...Array.from({ length: last }, (_, i) =>
    h('option', { value: i + 1 }, i + 1)));
  const title = h('input', { class: 'input', placeholder: 'What this day was' });
  return h('div', { class: 'life-addday' },
    h('span', { class: 'note' }, 'Mark another day worth remembering:'), day, title,
    h('button', {
      class: 'btn sm',
      onClick: async () => {
        try {
          await API.addLifeDay(S.who, { date: dayKey(y, m, Number(day.value)), title: title.value.trim() });
          await redraw(host);
        } catch (e) { problems(e); }
      },
    }, 'Add day'));
}


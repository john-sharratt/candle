# Tool Decisions

*Every item left under review or in dispute on tool shape, decided. The test:
**does it materially change what a Maker does socially, or what tasks it can
perform?** No — drop it. Yes — refactor it into the thing that does.*

Closes the review list in
[`tool_interaction_reconciliation.md`](tool_interaction_reconciliation.md) §4.
Store-shaped disputes (git, eras, the corpus index, document metadata, the image
store) are untouched here — they are questions about state, not about whether a
tool is the right tool.

**129 → 93.** The vocabulary did not shrink. It moved off the tool axis and onto
the argument axis, where the world can enumerate it.

---

## The move that does most of the work

Twenty-nine tools were named `*_read_*`, and six were named `*_take_*`. Each set
is one act performed against a different thing.

The catalogue already knows what to do with that, and has done since `move_to`:

> A closed set over an argument the character *means* would be the machinery
> writing its lines; a closed set over one that merely names something present is
> the machinery declining to let it name what is not.

Reading the era you are holding and reading the dispatch board are not two acts.
They are one act against two things, and **which things** is a fact about where
you are standing — exactly what `Choices` exists to express.

So `read` and `claim` become **body tools with world-enumerated arguments**, and
a part stops declaring what can be read at it. It declares only what can be
*changed* there. A part that holds something readable makes it readable by
existing.

This is better for tests, not merely smaller: one property — *everything the world
says is readable here is readable here, and nothing else is* — replaces
twenty-nine near-identical files, and the calibration examples now teach the
decision that actually matters, which is **which** thing to read rather than which
tool name to spell.

---

## The decisions

### 1. `character_write_beliefs` — **drop the tool, keep the task**

Dispute **E**, decided against the tool. §16 of the API contract is categorical:
a tool declaring `beliefs` in `writes_layers` is rejected at registration, and
reaching the belief endpoints from a tool context is `422`.

The task survives, and lands somewhere better. *Making* has "give somebody a
belief the world will not reward them for holding" — that is authoring a character
sheet, `Section::Characters`, which is who a character is **before they have lived
anything**. It goes through `character_write_identity`. The protected layer is
`layers/beliefs/`, which holds what a character has since *earned*, and a Maker
granting one of those directly would manufacture conviction without evidence —
bypassing the single process that makes a belief mean anything.

A Maker writes what happened. The sleep clock decides what it proves.

### 2. Twenty-nine reading tools — **refactor to one `read`**

No material change: the task is *"read far enough either side of your gap that you
could be contradicted and would know it"*, and it is the same task whether the
call is `chronicle_read_era` or `read(what: "the era I am holding")`.

But reads are material — *Grounding* is a whole cluster and every task in it is a
read — so this is a refactor, not a drop.

```
read { what }        // `what` bound to what is readable from where you stand
```

Dropped names: `chronicle_read_era` `read_any_page` `read_density` `read_conflicts`
· `record_read_description` `read_condition` · `story_read_ledger` `read_around_gap`
`read_filed` · `portrait_read_background` `read_hung` `read_palette` `read_unmade`
· `character_read` · `place_read_index` `read_entry` · `map_read` · `orders_read` ·
`dispatch_read` `read_holds` `read_wake` · `roster_read` · `cast_read_all` ·
`plant_read_panel` · `plan_read` · `trial_read_failures` · `standard_read` ·
`gather_read_standing` · `enquiry_read_history`

`read` does **not** absorb two neighbours, and the boundaries are worth stating
because they will be tested:

- **`observe` stays.** It attends to the physical situation — a room, a person, a
  thing — and returns a percept. `read` attends to recorded content. The world can
  represent that distinction: one goes through the map, the other through the
  corpus.
- **`file_read` stays.** It addresses a document by path for editing. `read` shows
  you what a station holds. One is the editing surface, the other is the situation.

### 3. Six claiming tools — **refactor to one `claim`**

`story_take_next_silence` `place_take_next_unwritten` `roster_take_unheld`
`orders_take` `portrait_take_faceless` `room_sit`

All six are `World::take(id, subject)` — the machinery that is already built,
already refuses with `AlreadyHeld { subject, by }` across the whole building, and
already releases on `set_off`. Six names for one call.

```
claim { what }       // bound to what is claimable from where you stand; None for a seat
release { }          // give back what you hold, without walking away
```

`room_sit` is the degenerate case — a seat claims no subject, and produces the
`TookStation { subject: None }` the witness already renders as *"sat down to
work"*. That is material: *Idle company*'s opener is "sit down near somebody
working and get on with your own thing beside them", and being seated is the
signal that you are staying.

`release` absorbs `orders_give_back`, and generalises it — *Claiming* has "give
back something you could finish, because somebody else needs it more" for every
kind of held thing, not just orders.

### 4. `follow.distance` — **drop the parameter**

`close | at a distance | out of sight`, against a world where `Where` is a node
and nothing models proximity within one. Two bodies in a room are in the room.

No task in the repertoire shadows anybody. The parameter reads as mechanical, is
not, and its own example leans on it — *"following close would satisfy the verb
and fail the order"* — which is the tell: it describes an effect the world cannot
produce. How you follow is narration, and belongs in an intent if it belongs
anywhere.

`follow` itself stays. *Arrivals & departures* has "take a newly arrived Maker
round the building", and somebody has to be the one following.

### 5. `story_read_aloud`, `story_hear_draft`, `story_give_opinion` — **drop two, refactor one**

Dispute **G**, decided.

- **`story_read_aloud` — drop.** It is `say` with a draft as its subject. Reading
  aloud records nothing; what matters is that people were there and heard it,
  which is presence plus speech.
- **`story_hear_draft` — drop.** Hearing is perception. You do not spend an act to
  receive one, and no other tool in the catalogue does.
- **`story_give_opinion` — refactor to `record_verdict`.** This one is material,
  and sharply so: `verdict` is a currency three clusters consume, and *Filing*'s
  signature is `accord, verdict → filed`. A verdict that is only something
  somebody said cannot be consumed by anything. It has to attach to the draft and
  persist.

```
record_verdict { on, judgement, what_would_change_it? }
```

`what_would_change_it` is optional and carries *Review & sign-off*'s "refuse to
pass a thing, and say exactly what would change that" — the difference between a
refusal and a verdict is that a refusal names its remedy.

### 6. The spatial verbs — **drop one, rename two**

- **`map_redraw_coast` — drop.** Redrawing a coast against markdown is editing a
  document, which `file_edit` and `place_write_entry` already do. It names no
  distinct act.
- **`map_drown_place` → `map_remove_place`.** The task is real — *Change & its
  wake* has "take a place out of the world, and reckon with everything written
  about it" — and the honest name is the one that says a place is being removed.
- **`map_move_border` → `map_settle_border`, and it takes two.** The task is
  "move the boundary you think is wrong, and go and warn everyone whose work it
  breaks", and a border between two regions belongs to both holders — the same
  reason the concordance table exists for eras. Moving it alone was never right.
  It joins the *Settling* family beside `place_settle_route`,
  `character_settle_relation` and `portrait_settle_likeness`.

### 7. `record_trace_custody` — **drop, use `bench_blame`**

*Custody*'s task is "trace a thing back through everyone who has held it, and find
the point where the chain goes quiet". Once the mind folder is a repository that
is `git blame`, exactly and literally, with a timestamp on every link — the
hierarchy design's §5.2 made real. A second custody mechanism beside it would be a
worse copy that can disagree.

This is conditional on dispute **A**. If the repository is not created, this comes
back.

### 8. Two clocks — **not a tool change; one invariant**

`wait_for` carries its own patience; an interaction carries a per-mode idle
timeout of 5 min / 10 min / 24 h. Both are material and both stay.

The rule, which is one test rather than a redesign: **the interaction's timeout is
a ceiling and the wait is a floor inside it.** A wait may not outlive the
interaction it is waiting inside. Nothing currently says so.

### 9. Two naming schemes — **refactor: one address per person**

Material, and the most dangerous item on the list. `tell` and `ask` bind `to`
against `Within::company` — the names the world writes down — while an
interaction's interlocutor carries a `display` name and §12 makes the *unique
name* the address.

This is the failure `Choices::Company` was built to prevent, recreated one level
up. The original: a character "asked for 'Perrin' when the world had written down
'Perrin Vastwood', was refused, and asked the same way on every tick for as long
as it stood there."

**The unique name is the address, everywhere.** `Within::company` carries unique
names; `Choices::Interlocutor` carries the interlocutor's unique name; the display
name is the narrator's business and never appears in an argument.

### 10. The two defects — **confirmed, both must land before tests**

- **`Mode` carries four values**, not two. The contract gives `send_image` to
  `video_call` and `instant_message` and withholds it from `voice_call`; the
  two-value collapse hands an NPC on a phone call a way to text a photo down it.
- **`send_image` binds `to` to `Choices::Interlocutor`**, not `Choices::Company`.
  In a messaging interaction the room's occupants are precisely the wrong set.

---

## The result

| | Was | Now |
|---|---|---|
| Total | 129 | **93** |
| Body tools *(argument-bound, available anywhere)* | 9 | **16** |
| Part tools *(acts that change content)* | 79 | 61 |
| `bench_` / `file_` | 0 / 0 | 11 / 5 |

**The body catalogue** — `say` `tell` `ask` `gesture` `move_to` `follow`
`observe` `wait_for` `send_image` `promise` `remind` `sign_off` `reach_out`
`read` `claim` `release`

Thirteen tools dropped outright, twenty-eight folded into `read`, five into
`claim`, and four renamed. Every task in the repertoire still has an act that
performs it; what changed is that the world now enumerates the objects instead of
the catalogue enumerating the verbs.

**What a part declares changed too.** A part used to list everything you could do
near it, reads included. It now lists only what can be *changed* there — and being
readable, claimable and sittable follows from the part existing. Shorter files,
and no way for the map to disagree with the grammar about what can be read.

---

## What this does not decide

Untouched, because they are questions about state rather than tool shape, and the
instruction was to leave implementation alone:

**A** no repository · **B** no eras · **C** no corpus index · **D** the remaining
stores — promises, orders, appointments, trials, standards, judgements, and what a
vault of prose keeps on a rack · **H** no document metadata · **I** no image store

One of them now blocks a decision made here: **§7 depends on A.** If the mind
folder does not become a repository, `record_trace_custody` returns and custody
needs a field of its own.

# The Tool Surface — Audit and Reconciliation

*What a Maker can actually do, checked three ways: against the engine that
implements it, against the vault that offers it, and against the 480 tasks that
are supposed to spend it.*

The premise of [`asynchronous_mind_hierarchy.md`](asynchronous_mind_hierarchy.md)
is that **tools are fixed at the boundary with reality** and everything above
invents against them. That only holds if the boundary is real. This document is
the check on the boundary: every tool that exists, every tool the world
advertises, every task that needs one, and the differences between the three.

---

## 1. The headline: the world's tool surface is not connected to anything

There are two catalogues, and only one of them is a catalogue.

| | Where | Count | Implemented | Reaches the model |
|---|---|---|---|---|
| The body catalogue | `npcd/src/engine/tools.rs` — `CATALOG` | 9 | 9 | as a compiled grammar |
| The world's part tools | 34 part files, `tools:` | 79 | **0** | **not at all** |

The second row is the finding. `reach::in_world` computes what a body can do
because of where it is standing, and `reach::tools` / `reach::can_reach` are
written and tested — but the only consumer is `reach::line`, which renders
`Within reach: a character terminal.` into the situation. It carries the part's
*name*. The tool identifiers, and the `about` text that explains them, are
computed and discarded.

Two consequences worth stating plainly, because both were believed otherwise:

- **A Maker cannot see these tools, so it cannot try them.**
  [`maker_repertoire.md`](maker_repertoire.md) says the 24 newest identifiers
  "appear in a Maker's reach and refuse". They do not appear and there is nothing
  to refuse them — they are absent from the grammar, which is the one place
  availability is decided. The line is wrong and is corrected below.
- **Every station description in the vault is currently fiction.** The chronicle
  terminal's own text says sitting at one "claims a single era… so nobody else
  can rewrite it while you are working". Nothing claims, nothing locks, nothing
  writes.

This is not a small gap to be closed by wiring `reach::tools` into
`specs_within`. Those 79 identifiers have no parameters, no argument binding, no
handlers and no plane. `reach.rs` says so itself — *"they arrive as identifiers
with a sentence rather than as parameter schemas… what the call looks like is
the business of whatever implements it."* Nothing implements it. **Building the
foundation is building all 79, plus what this audit adds.**

---

## 2. Before — everything that exists

### 2.1 The body catalogue — 9 tools, armed

| Tool | Plane | Availability | Bound arguments |
|---|---|---|---|
| `tell` | Speech | Nearby | `to` → company (heard by the room) |
| `whisper` | Speech | AmongOthers (two or more others here) | `to` → company (heard by the addressee only) |
| `shout` | Speech | Always | — (heard in this room and every room in sight) |
| `ask` | Speech | Nearby | `to` → company |
| `gesture` | Speech | Nearby | `to` → company *(optional; drops when alone)* |
| `move_to` | World | Always | `destination` → reachable |
| `follow` | World | Nearby | — |
| `observe` | Internal | Always | — |
| `wait_for` | Meta | Nearby | `who` → waitable, `for` → wait kinds |
| `send_image` | Speech | MessagingOnly | `to` → company |

### 2.2 The world's part tools — 79 identifiers, 17 namespaces, none implemented

| Namespace | n | Identifiers | Standing on |
|---|---|---|---|
| `record.` | 12 | `appraise` `record_reason` `let_go` `accession` `write_provenance` `trace_custody` `describe` `read_description` `arrange` `mend` `mark_repair` `read_condition` | appraisal bench, accession desk, catalogue, mending bench |
| `story.` | 9 | `read_ledger` `take_next_silence` `read_around_gap` `draft` `file` `read_filed` `read_aloud` `hear_draft` `give_opinion` | gap ledger, story desk, filed stories, reading table |
| `portrait.` | 9 | `read_background` `draw` `redraw` `read_hung` `read_house_style` `file_plate` `read_unmade` `take_faceless` `settle_likeness` | easel, hung faces, house palette, plate rack, likeness table |
| `chronicle.` | 8 | `read_era` `rewrite_page` `add_entry` `retire_entry` `read_any_page` `settle_boundary` `read_density` `read_conflicts` | chronicle terminal, archive, concordance table, timeline wall |
| `character.` | 6 | `read` `write_identity` `write_wants` `write_beliefs` `write_memories` `settle_relation` | character terminal, relations table |
| `place.` | 6 | `read_index` `take_next_unwritten` `settle_route` `read_entry` `write_entry` `write_local_history` | place index, road table, survey desk |
| `map.` | 5 | `read` `move_border` `add_place` `drown_place` `redraw_coast` | map table, gallery rail |
| `orders.` | 4 | `read` `take` `give_back` `report_done` | order table |
| `enquiry.` | 3 | `take_question` `answer_from_record` `name_the_gap` | enquiry desk |
| `plant.` | 3 | `read_panel` `note_drift` `raise_fault` | plant panel |
| `stores.` | 3 | `put_back` `take_out` `walk_the_racks` | stores |
| `structure.` | 3 | `lay_out_scenes` `test_the_want` `find_the_slack` | structure board |
| `roster.` | 2 | `read` `take_unheld` | roster |
| `cast.` | 2 | `read_all` `report_disagreement` | watch desk |
| `room.` | 2 | `sit` `talk` | seat, hung faces, gallery rail |
| `dispatch.` | 1 | `read` | dispatch board |
| `creator.` | 1 | `present` | creator's chair |

Both map copies — `npc-map/maps/` and `<mind>/map/battle-cities/` — are byte
identical, so every change below applies twice.

---

## 3. What does not reconcile

### 3.1 A tool that duplicates an implemented one — remove

**`room.talk`**, on the seat, the hung faces and the gallery rail.

`shout` is `Availability::Always`; `tell` and `whisper` are offered wherever there is company. `tell` is the ordinary addressed voice; `whisper` is described as the exception — for keeping something from somebody else who is present — because offered as "for when it is only for them" it became the cast's default in two-person conversations.
Speech is a body act, and hanging it on furniture asserts that a Maker can talk
near a seat, a portrait wall or a map rail and not elsewhere — which is false,
and which puts the map into direct disagreement with the grammar. It is the only
part tool that names an act the body catalogue already performs.

*Removed from three parts; the identifier disappears.*

### 3.2 A queue with four consumers and no producer

The order table carries `orders.read`, `orders.take`, `orders.give_back` and
`orders.report_done`. Its own description says *"orders change while Makers are
away working"* — but nothing in the world changes them. Every tool on the table
either reads the queue or removes something from it.

This is the mechanism behind engine note 18: *Seeds* tasks produce `finding`, and
nothing turns a finding into a claim held by a named Maker. **Delegation &
authority** and **Teaching** both depend on giving somebody work, and both are
unexecutable for the same missing verb.

*Adds `orders.set` and `orders.hand_to`.*

### 3.3 Destructive and mutating tools on unclaimed fixtures

Every other part that writes is a `station` with a `binds` line, so working at it
takes an exclusive claim. Three of the newest are not:

| Part | Kind | Mutating tools | Claim |
|---|---|---|---|
| appraisal bench | fixture | `record.let_go` *(destroys)*, `record.appraise`, `record.record_reason` | none |
| mending bench | fixture | `record.mend`, `record.mark_repair` | none |
| catalogue | fixture | `record.describe`, `record.arrange` | none |

`record.let_go` is the sharp one: a Maker can discard a record it never claimed,
concurrently with another Maker doing the same thing to the same record. The
chronicle terminal refuses exactly this and says so at length.

Rather than reclassify — the catalogue *should* be readable by anyone while being
arranged by one — this is what the mode mechanism in §5 is for: reading is always
available, and entering the writing mode takes the claim.

### 3.4 A cross-discipline concern scoped to one discipline

`portrait.read_house_style` stands on the house palette, in the portraits level.
The task it serves — *"read the house style before making anything that has to
sit beside other people's work"* — is in **Grounding**, which every discipline
passes through, and the whole of **Craft & standard** is about a standard the
building holds, not one the painters hold.

*Becomes `standard.read` on a standards board reachable from every level, and
the house palette keeps its palette.*

---

## 4. What is missing — clusters no tool can execute

A task is executable when the act at its centre has a tool that performs it.
Speech tasks are executable now; the body catalogue covers them. These are the
clusters whose central act nothing performs.

| Cluster | Tasks | What has no tool | Fix |
|---|---|---|---|
| **Planning & sequencing** | 9 | Nothing writes a plan, orders its pieces, or reads it back. The repertoire notes it: *"No station stands behind this stage."* | planning board, `plan.*` |
| **Revision** | 8 | Nothing reopens a draft. `story.draft` is a single call; the ordered passes need a draft to be edited repeatedly before it is filed. | terminal modes, `bench.*` |
| **Gatherings** | 7 | The long table exists with ten seats; nothing calls a gathering or reads what is standing. | `gather.*` on the reading table |
| **Promises & coordination** | 7 | A promise is `tell` with an intent, so it leaves nothing behind. The only cluster that creates obligation outliving a conversation. | body `promise`, `remind` |
| **Change & its wake** | 6 | `map.move_border` makes the change. Nothing reads what depends on it, and nothing carries the wake to whoever it broke. | `dispatch.post_wake`, `read_wake` |
| **Findability** | 6 | No cross-reference, no note for whoever comes next, no way to tidy a drifted index. | `record.cross_reference`, `leave_note`, `tidy_index` |
| **Experiment** | 5 | *"Keep the failed attempt where somebody could learn from it"* has nowhere to keep it. | trials shelf, `trial.*` |
| **Craft & standard** | 5 | See §3.4. Nothing reads the standard generally and nothing changes it. | standards board, `standard.*` |

**Partly served, one verb short each:**

- **Custody** — `record.hand_on` is missing. *"Hand something on properly: what
  it is, what you did to it, and what you did not."*
- **Service** — `enquiry.read_history` (*"find out what people keep asking for
  and never find"*) and `enquiry.raise_work` (*"turn a question you could not
  answer into a piece of work somebody could take"*, which needs §3.2's producer).
- **Delegation & authority**, **Teaching** — both blocked on §3.2.

**Deliberately unchanged:**

- `send_image` is spent by no task in the repertoire. It serves messaging mode,
  which is a different surface — a player texting a character, not a Maker
  working. It stays, and the mismatch is expected rather than a defect.
- *"Let a silence run rather than filling it"* looked like a missing act. It is
  `wait_for(who, someone_speaks)`, which is exactly what it means.

---

## 5. Terminals have modes, and the mode is where the repository lives

A terminal's tools depend on being within reach of it **and on what mode it is
in**. This is what makes the lowest level an ordinary agentic coding loop —
branch, edit, diff, commit, roll back — against the mind folder as a git
repository, which is §5 of the hierarchy design.

### 5.1 The schema

`Part` grows one optional field. Parts without it behave exactly as they do now.

```yaml
id: chronicle-terminal
kind: station
binds: one era
tools:                                   # always, once within reach
  - chronicle.read_era
  - chronicle.read_any_page
modes:
  working:                               # entered by bench.open_working
    claims: true                         # takes the part's `binds` exclusively
    tools:
      - chronicle.rewrite_page
      - chronicle.add_entry
      - chronicle.retire_entry
  offered:                               # entered by bench.offer
    claims: true
    tools: []
```

The mode is **per body, not per part** — which Maker is in which mode at which
part is engine state, so two Makers at two chronicle terminals in the same room
are in different modes and see different grammars. The map declares what the
modes *are*; the engine holds who is in one.

`bench.*` is mixed into every part declaring `modes:`, so one implementation
serves every editing station rather than each namespace growing its own commit.

### 5.2 The `bench_` namespace — 11 tools, named after git

Every verb here is one the model has seen millions of times, for the reason §5.4
gives. The mode column is what the engine tracks; the name is what git calls it.

| Tool | From → to | What it is |
|---|---|---|
| `bench_branch` | reading → working | Start a change. Branches; takes the claim. |
| `bench_stash` | working → reading | Keep the change without offering it. |
| `bench_stash_pop` | reading → working | Pick your own stashed work back up. |
| `bench_diff` | working, offered | What you have altered against what stands. |
| `bench_restore` | working → reading | Discard your changes. |
| `bench_stage` | working → offered | Put it up as done. |
| `bench_unstage` | offered → working | Take it back for more work. |
| `bench_commit` | offered → reading | Merge it into what stands. **May fail.** |
| `bench_status` | any | What is open, and what a failed commit collides with. |
| `bench_blame` | any | Who last changed this, and when. |
| `bench_log` | any | What has been done to this thing. |

Three of these are load-bearing for the hierarchy design rather than merely
convenient:

- **`bench_commit` may fail with a conflict**, and the conflict names the other
  party. That is §5.1 of the hierarchy doc: parallel work manufactures social
  work as a side effect, and the conflict is the trigger condition for the
  **Settling** cluster. No idle pump can produce this, because an idle pump has
  no reason to prefer one interlocutor over another.
- **`bench_blame` is the custody chain.** The *Custody* task *"trace a thing back
  through everyone who has held it, and find the point where the chain goes
  quiet"* becomes a literal implementation.
- **Branch-merged is not negotiable by the agent that produced it.** This is what
  keeps the archive honest when `task_complete` is self-reported (§7.3 there).

### 5.3 The file tools are zend-tools', not new ones

**Revision** and **Making** need a draft edited repeatedly before it is filed,
which is ordinary file editing. `zend-tools` already implements it —
`file_read`, `file_edit`, `file_write`, `file_list`, `file_delete` over a session
overlay that copies a workspace file up on first edit and records a whiteout on
delete. Its own documentation states the alignment that matters here: `file_edit`
replaces `old_str` only where it appears exactly once, which *"matches Claude
Code's `str_replace` semantics and forces the model to provide enough context to
identify a single edit site."*

It is a workspace member with no candle dependency, so `npcd` can depend on it
directly. Reusing it costs one dependency and buys names the model already has a
strong prior over, semantics that are already debugged, and one implementation
serving both products.

*Adds no new identifiers — five reused ones.*

### 5.4 Which parts get modes

The nine that write content: chronicle terminal, character terminal, story desk,
easel, survey desk, map table, accession desk, catalogue, mending bench. The
appraisal bench too — `record_let_go` belongs in a working mode that had to be
entered deliberately, which is §3.3's fix.

Fixtures that only read (archive, timeline wall, filed stories, gap ledger, place
index, hung faces, house palette, plate rack, gallery rail, roster, dispatch
board, watch desk, stores, plant panel, structure board, seat) and the four
settling tables keep a flat `tools:` list.

---

## 5A. Naming — `namespace_verb`, and verbs the model has already seen

Two separate questions, measured rather than assumed, because they have different
answers.

### 5A.1 The envelope is correct

`<tool_call>` and `</tool_call>` are dedicated special tokens in the Qwen3.8
vocabulary, and `ToolCallEnvelope::qwen3()` reproduces the trained format exactly:

```
<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>
```

Nothing to change.

### 5A.2 The separator: the dot is not a tokenisation problem, and is still wrong

Checked against the shipped BPE vocabularies. A lower id is an earlier merge, so
a more frequent sequence in the pre-training corpus:

| verb | dotted | underscored |
|---|---|---|
| read | `.read` **3989** | `_read` 6241 |
| write | `.write` **3708** | `_write` 8892 |
| edit | `.edit` **12352** | `_edit` 12774 |
| commit | `.commit` **15267** | `_commit` 35145 |
| merge | `.merge` 24210 | `_merge` **20250** |
| diff | `.diff` 39335 | `_diff` **15384** |
| branch | `.branch` 50789 | `_branch` **27124** |

The dot is *cheaper* for accessor verbs — the corpus is full of `obj.read()` —
and dearer for the git verbs, which appear in code as identifiers. Tokenisation
does not decide this.

`Qwen3.5-0.8B` and `Qwen3.6-35B-A3B` ship a byte-identical `tokenizer.json`, and
`Qwen3.8-27B` carries the same ids (its 1.3 KB difference is added special tokens,
not vocabulary). **A naming decision here is stable across all three.**

Three things do decide it, and all point the same way:

- **The house convention has no dots.** `zend-tools` — the working tool surface of
  the sibling product — ships `file_read`, `file_edit`, `file_write`, `file_list`,
  `file_delete`, `code_session_exec`, `connect_ssh`: `namespace_verb`, underscore.
  The tool-call compiler's own documentation names `ssh_session_exec` versus
  `ssh_session_exec_async` as the prefix case its trie was built against. The
  dotted part tools are this codebase's outlier, not its convention.
- **Both major function-calling schemas forbid the dot.** OpenAI and Anthropic
  constrain a tool name to `^[a-zA-Z0-9_-]{1,64}$`. Qwen's function-calling
  fine-tune is built on that data, so the distribution in the `"name"` slot
  specifically is snake_case. Cheap tokenisation and a familiar *name shape* are
  different properties, and the second is the one that drives selection.
- **Some namespace words are not tokens.** `record` (8286), `dispatch` (17730),
  `story` (25638) and `bench` (26157) are single tokens; `chronicle`, `enquiry`
  and `roster` are not. Eight `chronicle_*` names carry the extra tokens in every
  turn's grammar.

**All 113 identifiers become `namespace_verb` snake_case** — `chronicle_read_era`,
`record_let_go`, `bench_commit`.

### 5A.3 The verbs: name the act after the thing the model has done a million times

The separator is cosmetic beside this. A coined verb recruits no prior, and the
first draft of `bench_` was entirely coinages:

| Coined | Replaced by | Prior it recruits |
|---|---|---|
| `open_working` | `bench_branch` | `git branch`, `checkout -b` |
| `show_changes` | `bench_diff` | `git diff` |
| `set_aside` / `resume` | `bench_stash` / `bench_stash_pop` | `git stash` |
| `roll_back` | `bench_restore` | `git restore` |
| `offer` / `withdraw` | `bench_stage` / `bench_unstage` | `git add`, `restore --staged` |
| `read_conflict` | `bench_status` | `git status` |

This puts machine vocabulary inside a fiction, which is correct rather than a
compromise, and the hierarchy design already argues it: the design *"does not ask
a model to be a character in a simulation; it asks it to do work of a kind it has
seen an enormous amount of, and lets character fall out of the choices it makes
while doing it."* The part's `short` and `long` prose stays in the vault's voice;
the identifier is git. It is the same split as `say` taking an intent while the
narrator renders the words — the surface carries the fiction, the mechanism is
plain.

The rule generalises past `bench_`: **prefer the word the craft already uses.**
`record_accession`, `record_appraise` and `record_arrange` are archival terms of
art and stay; `plan_set_scope` should be `plan_scope`; `trial_set_side_by_side`
should be `trial_compare`.

---

## 6. After — the reconciled surface

### 6.1 Removed

| Identifier | From | Why |
|---|---|---|
| `room.talk` | seat, hung faces, gallery rail | `say`/`tell` already do it, everywhere |

### 6.1a Renamed — all 78 survivors

Every dotted identifier becomes `namespace_verb`, per §5A.2. Mechanical, and the
one change that touches every part file: `chronicle.read_era` →
`chronicle_read_era`, `record.let_go` → `record_let_go`, `room.sit` → `room_sit`.

### 6.2 Added

**Body catalogue — 2**

| Tool | Plane | Availability | Bound arguments |
|---|---|---|---|
| `promise` | World | Nearby | `to` → company |
| `remind` | Speech | Nearby | `who` → company, `which` → **that person's outstanding promises** |

`remind`'s second argument is the audit's own discipline applied to a new tool:
bound to the promises that actually stand, so reminding somebody of a thing they
never promised is not a mistake available to it.

**New parts — 3 fixtures, placed in rooms that already exist**

| Part | Tools | Serves |
|---|---|---|
| planning board | `plan_break_down` `plan_order` `plan_reorder` `plan_scope` `plan_read` | Planning & sequencing |
| trials shelf | `trial_keep` `trial_read_failures` `trial_compare` | Experiment |
| standards board | `standard_read` `standard_propose` `standard_settle` | Craft & standard, Norms |

**New namespace on the editing stations — 11**

`bench_` — §5.2.

**Reused from `zend-tools` — 5**

`file_read` `file_edit` `file_write` `file_list` `file_delete` — §5.3. Not new
identifiers, and not new implementations.

**Extended parts — 6, adding 14**

| Part | Added | Serves |
|---|---|---|
| order table | `orders_set` `orders_hand_to` | Delegation & authority, Teaching, Service |
| dispatch board | `dispatch_read_holds` `dispatch_post_wake` `dispatch_read_wake` | Change & its wake, Noticing |
| catalogue | `record_cross_reference` `record_leave_note` `record_tidy_index` | Findability |
| reading table | `gather_call` `gather_read_standing` | Gatherings |
| enquiry desk | `enquiry_read_history` `enquiry_raise_work` | Service |
| accession desk | `record_hand_on` | Custody |

### 6.3 The counts

| | Before | After |
|---|---|---|
| Body tools | 9 | 11 |
| Part tools | 79 | 113 |
| Reused from `zend-tools` | 0 | 5 |
| Namespaces | 17 | 23 |
| **Declared total** | **88** | **129** |
| **Implemented** | **9** | **14** *(9 + 5 reused)* |
| Clusters with no executable act | 8 *(53 tasks)* | 0 |

The implemented row is the point of the exercise. Reusing the file tools moves it
by five without writing anything; the gap that remains — **115 tools** — is the
foundation the hierarchy is waiting on.

### 6.4 Build order

`bench_*` first, and not because it is the largest. It is the only group that
serves eight clusters at once, it is what turns the lowest level into the
agentic-coding loop the models are best at, and it supplies the two things the
archive cannot be trusted without: a merge that can fail, and a history the agent
did not write about itself. It also arrives beside five file tools that already
work, so the whole editing surface lands together.

After it, the order is by how many tasks unblock per tool: `orders_set` and
`orders_hand_to` (two verbs, two whole clusters), then `plan_*`, then the reading
tools each station needs before its writing tools mean anything.

The rename in §6.1a should land **before** any of it — it touches every part file
and every future reference, and it is free now and expensive once handlers and
tests name the old form.

---

## 7. Corrections to the documents this contradicts

- [`maker_repertoire.md`](maker_repertoire.md) — *"They are authored into the
  parts, so they appear in a Maker's reach and refuse."* They do not appear.
  Nothing in the grammar offers them and nothing refuses them. The sentence
  should say that they are authored and unreachable, which is a stronger form of
  invariant 8 rather than a weaker one.
- [`vault_world.md`](vault_world.md) — every station description asserting that
  sitting claims a thing and locks it describes the design, not the engine. The
  claim is real once §5 is built; until then the descriptions run ahead of it.

# Tools Against World State

*Where each of the 129 tools reads and writes, which stores already hold what it
needs, and which tools have nothing to act on.*

Companion to [`tool_surface_audit.md`](tool_surface_audit.md), which settled what
the tools *are*. This settles what they would *touch*. A tool with no store is
not a tool that needs writing — it is a decision that has not been made, and
those are collected in §4 rather than guessed at.

---

## 1. The stores that exist

Four, and they are more capable than the audit assumed.

### 1.1 The mind corpus — `npcd::mind`

Read, write, add, remove, list. Atomic, addressed, scoped by world, already
serving the console. **Nothing outside the module knows there are files.**

| Section | On disk | Holds |
|---|---|---|
| `canon` | `layers/world/` | 1,268 markdown documents — `history/`, `geography/`, `locations/`, `factions/`, `events/`, `creatures/`, `identity/`, `relationships/` … |
| `agency` | `layers/agency/` | what characters want, and how they pursue it |
| `beliefs` | `layers/beliefs/` | what each character holds true |
| `memory` | `layers/memory/` | what each character remembers |
| `characters` | `personalities/` | who a character is before living anything |
| `worlds` | `worlds/` | settings and their filters |
| `responses`, `moods`, `settings` | — | craft libraries and configuration |

Three granularities, all built: [`doc`] whole documents, [`section`] a document
as fields, [`parts`] one item inside a document. `layers/life/<character>/` holds
dated episodes — `2004-11-03 The Year You Stopped Performing.md` — which matters
in §4.2, because it proves dated documents are already a pattern here.

### 1.2 The live world — `npc_map::world::World`

Who is where, what they are holding, and what happened.

**The claim mechanism is built.** `Actor.hold: Option<Hold>` carries
`{ subject, part, since }`; `World::take` refuses with `AlreadyHeld { subject, by }`
checking *every* hold in the building rather than the room; `World::set_off`
releases on leaving; `TookStation` / `LeftStation` are witnessed and rendered.

This corrects the audit. It said *"nothing claims, nothing locks."* Wrong — the
machinery is complete and **called only from `npc-map`'s own tests**, because no
tool exists that takes a station. `room_sit` is one call away from working.

### 1.3 The substrate — `candle-conversation::persistence`

`AuthoredBelief`, `AuthoredRelationship`, `Modulation`, tombstones, snapshots —
and **`AuthoredStrategy { strategy_id, statement, parent_id, state }`**, which
already models mission-to-task decomposition and which nothing writes. The whole
`plan_` namespace is a consumer for a store that is sitting there.

### 1.4 The guests — image and prose

Z-Image-Turbo is co-resident with the engine, guidance-distilled, twelve steps.
Its own documentation uses the example that matters here: *"a portrait is a couple
of seconds of drain rather than a minute of it."* `portrait_draw` has a real
pipeline behind it, not a placeholder.

---

## 2. What is not there

- **The mind folder is not a git repository.** `git rev-parse` inside it returns
  `fatal: not a git repository`. §5 of the hierarchy design — branch at a bench,
  merge on finish, blame as custody, the repository as oracle — rests on a
  repository that has never been initialised.
- **No index over the corpus.** Nothing knows which places are named but not
  described, which entries contradict, or where the record is thin. Every
  "notice what is missing" tool needs one.
- **No document metadata.** Canon files are bare markdown: no provenance, no
  condition, no accession date, no keep/discard judgement.
- **No queues, obligations or appointments.** No orders, no promises, no
  gatherings, no enquiries arriving from outside.
- **No image store.** The pipeline can make a portrait; nothing holds one.

---

## 3. The mapping

**✓ clean** — the store exists and the tool is writable against it today.
**✗ disputed** — see §4, by letter.

| Namespace | Store | Clean | Disputed |
|---|---|---|---|
| body | `World`, scheduler | 9 — `say` `tell` `ask` `gesture` `move_to` `follow` `observe` `wait_for` `send_image` | 2 — `promise` `remind` **(D)** |
| `bench_` | *git* | 0 | 11 — all **(A)** |
| `file_` | `mind::doc`, `mind::catalog` | 5 — all | 0 |
| `chronicle_` | canon `history/` | 1 — `read_any_page` | 7 — `read_era` `rewrite_page` `add_entry` `retire_entry` `settle_boundary` **(B)**; `read_density` `read_conflicts` **(C)** |
| `record_` | canon, doc metadata | 5 — `describe` `read_description` `cross_reference` `leave_note` `let_go` | 11 — `tidy_index` `arrange` **(C)**; `appraise` `write_reason` **(D)**; `mend` `mark_repair` `read_condition` `accession` `write_provenance` `trace_custody` `hand_on` **(H)** |
| `story_` | canon | 4 — `read_around_gap` `draft` `file` `read_filed` | 5 — `read_ledger` `take_next_silence` **(C)**; `read_aloud` `hear_draft` `give_opinion` **(G)** |
| `portrait_` | image guest, canon | 5 — `draw` `redraw` `read_background` `read_palette` `settle_likeness` | 4 — `file_plate` `read_hung` `read_unmade` `take_faceless` **(I)** |
| `character_` | agency, memory, personalities, `AuthoredRelationship` | 5 — `read` `write_identity` `write_wants` `write_memories` `settle_relation` | 1 — `write_beliefs` **(E)** |
| `place_` | canon `locations/`, `geography/` | 4 — `read_entry` `write_entry` `write_local_history` `settle_route` | 2 — `read_index` `take_next_unwritten` **(C)** |
| `map_` | canon `geography/` | 2 — `read` `add_place` | 3 — `move_border` `drown_place` `redraw_coast` **(F)** |
| `orders_` | — | 0 | 6 — all **(D)** |
| `dispatch_` | — | 0 | 4 — all **(D)** |
| `roster_` | personalities, `World` holds | 2 — both | 0 |
| `cast_` | personalities | 1 — `read_all` | 1 — `report_disagreement` **(C)** |
| `room_` | `World::take` | 1 — `sit` | 0 |
| `creator_` | — | 0 | 1 — `present` **(D)** |
| `enquiry_` | — | 0 | 5 — all **(D)** |
| `plant_` | `npcd::telemetry` | 3 — all | 0 |
| `stores_` | — | 0 | 3 — all **(D)** |
| `structure_` | canon | 3 — all | 0 |
| `plan_` | `AuthoredStrategy` | 5 — all | 0 |
| `trial_` | — | 0 | 3 — all **(D)** |
| `standard_` | — | 0 | 3 — all **(D)** |
| `gather_` | — | 0 | 2 — all **(D)** |
| | | **55** | **74** |

Two resolutions worth naming, because they turn apparent gaps into existing
capability:

- **`plant_` reads the daemon's own telemetry.** The *Rounds* cluster wants
  *"the panel of what the place does for itself"* and *"the figure that has been
  slightly wrong for a while"*. `npcd::telemetry` holds exactly that — engine
  rate, VRAM, ring-buffered series. The plant room is the daemon's own vitals,
  which is truer than any invented gauge.
- **`plan_` writes `AuthoredStrategy`.** The store is built, carries `parent_id`,
  and models mission-to-task decomposition. It is the joint between this document
  and the hierarchy design.

---

## 4. Under dispute

### A. There is no repository — 11 tools

`bench_branch` `bench_stash` `bench_stash_pop` `bench_diff` `bench_restore`
`bench_stage` `bench_unstage` `bench_commit` `bench_status` `bench_blame`
`bench_log`

The whole editing surface, and the hierarchy design's §5, assume the mind folder
is a git repository. It is not one.

Resolving it is one command, but it is a real decision rather than a formality:
1,268 canon documents plus every character layer come under version control, and
a live daemon writes into the working tree while Makers hold branches. The
questions are whether the daemon commits as itself or as the Maker, what happens
to a branch a Maker abandons by walking away, and whether `.substrate/` is
ignored (it must be).

**Resolving A also resolves part of H** — `bench_blame` *is* the custody chain,
so `record_trace_custody` stops needing a provenance field of its own.

### B. There are no eras — 5 tools

`chronicle_read_era` `chronicle_rewrite_page` `chronicle_add_entry`
`chronicle_retire_entry` `chronicle_settle_boundary`

The chronicle terminal `binds: one era` and the concordance table settles where
two eras meet. `layers/world/history/` holds fifteen **topic** files —
`great_war.md`, `golden_age.md`, `genocide.md`, `escape.md`, `vault_creation.md` —
with no date ranges and no boundaries. Two topics do not have a boundary to
settle, so `settle_boundary` has nothing to operate on and the concordance table
is furniture.

Either eras become real (front matter with a span, which `layers/life/` already
demonstrates by naming its documents with dates), or the chronicle rebinds to
*one topic* and `settle_boundary` becomes `settle_overlap` — a weaker but honest
act over two documents that both cover the same ground.

### C. Computed views nothing computes — 9 tools

`chronicle_read_density` `chronicle_read_conflicts` `story_read_ledger`
`story_take_next_silence` `place_read_index` `place_take_next_unwritten`
`cast_report_disagreement` `record_tidy_index` `record_arrange`

Each needs an answer over the whole corpus that nothing derives: where the record
is thin, which entries contradict, which places are named but never described.

**This is the most load-bearing dispute in the document.** The gap ledger is the
vault's primary source of work — *Noticing* opens nearly every mission, and
`story_take_next_silence` is step one of the first worked mission in the
repertoire. Without it a Maker has nothing to claim and the entire chain from
`finding` onward never starts.

Two shapes, and they trade differently:

| | Maintained index | Live pass |
|---|---|---|
| Cost | cheap to read, must be kept current | expensive per call |
| Staleness | goes stale silently after every write | never stale |
| Honesty | asserts a gap that may have been filled | sees what is there now |

A third is available here and probably right: **the provenance gallery already
scans the corpus by attentional similarity.** "Which entries contradict" is a
similarity query, not a keyword one, and the machinery is built.

### D. No store, and the store is a design decision — 31 tools

`promise` `remind` · `orders_*` (6) · `dispatch_*` (4) · `enquiry_*` (5) ·
`stores_*` (3) · `trial_*` (3) · `standard_*` (3) · `gather_*` (2) ·
`record_appraise` `record_write_reason` · `creator_present`

These split into three unlike groups and should not be resolved together.

**Cheap and well-defined — a table each.** Promises with a due time, orders with
a holder, appointments with an hour, kept failures, agreed standards, keep/discard
judgements. Each is a small durable record; each closes a whole cluster. These
are engine notes 13, 14, 16, 18 made concrete.

**Undefined — what is the thing?** `stores_*` asks a vault of documents what it
keeps on a rack. *Care of the place* and *The set-aside day* want *"put back the
thing somebody left out"* and *"walk the racks against the list"* — but the vault
holds prose, and prose does not go back on a shelf. Either stores becomes
something real (drafts checked out and not returned? plates? claimed subjects left
held?) or the three tools go and those tasks move to `record_` and `bench_`.

**Needs an outside — who asks?** `enquiry_*` needs questions arriving from beyond
the vault (engine note 22) and `creator_present` needs somebody to present to
(engine note 21). Both name a boundary the world does not have. The obvious
candidate is the player: an enquiry is a real question from a real user, and the
creator's chair is where a Maker faces one. That would make *Service* the one
cluster that touches the product's actual users.

### E. A tool that contradicts a standing invariant — 1 tool

`character_write_beliefs`

`tools.rs` holds `tests::no_tool_writes_beliefs` as a build failure, and its
module documentation is explicit: beliefs move only through the evidence-threshold
process on the sleep clock; an operator authoring a character can write them
*"through the authoring API; that plane is separate and is not reachable from
here."*

The character terminal proposes to make it reachable. The question is whether a
Maker sitting at one is an **author** (writing another character's beliefs as
content, which the invariant permits) or a **character** (writing beliefs, which
it forbids). I read it as the former and the invariant as intact — a Maker never
writes its *own* beliefs — but it must be ruled on rather than assumed, because
the test will fail and the temptation will be to weaken the test.

### F. Spatial verbs over prose — 3 tools

`map_move_border` `map_drown_place` `map_redraw_coast`

`layers/world/geography/` and `map.md` are markdown. There are no borders, no
coastlines, no coordinates — so these are document edits wearing spatial names,
and *"agree with somebody how two places connect, so neither entry lies about the
journey"* has no journey to lie about.

Either the world grows a real geography (a node graph, as the vault itself has),
or the names say what they do: `map_rewrite_region`, `map_remove_place`. The
second is cheap and honest; the first is a much larger world-model decision.

### G. Speech acts with furniture attached — 3 tools

`story_read_aloud` `story_hear_draft` `story_give_opinion`

Reading aloud is `say` with a draft as its subject. Hearing one is perception.
Giving an opinion is `tell`. This is `room.talk`'s defect one level subtler: the
only thing these add over the body catalogue is *which draft*, and that could be
an argument.

They earn their place only if a reading is real state — a session with listeners,
and verdicts recorded against the draft, which is what *Testing it* and *Review &
sign-off* need in order to produce `verdict` at all. Otherwise they collapse into
`say` and `tell` and the reading table keeps only `gather_*`.

### H. No document metadata — 7 tools

`record_accession` `record_write_provenance` `record_trace_custody`
`record_hand_on` `record_mend` `record_mark_repair` `record_read_condition`

Canon documents are bare markdown. Nothing records where a thing came from, who
has held it, what was done to it, or how well it has survived — engine note 23,
and the *Custody* cluster's own premise that *"a thing whose origin was never
written down cannot be trusted afterwards."*

The resolution is front matter, on 1,268 files. That is mechanical but it is a
corpus-wide change and it interacts with A: once the repository exists, `git log`
answers "what was done to it" and `git blame` answers "who has held it", leaving
only *condition* — a judgement, which genuinely needs a field.

### I. No image store — 4 tools

`portrait_file_plate` `portrait_read_hung` `portrait_read_unmade`
`portrait_take_faceless`

The pipeline can make a portrait; nothing holds one. The mind corpus is markdown
and YAML, and `address::Format` knows two formats, neither of them an image.

`portrait_draw` and `portrait_redraw` are clean because they produce an image
in-band. The four above all assume a plate rack that persists — so this is one
decision, not four: does the corpus gain a binary section, or do plates live
beside the daemon's own data?

---

## 5. Where this leaves the build

**55 tools are writable against stores that exist.** That is more than the audit
implied, and it is enough to run a whole mission end to end — notice is the gap,
but claiming, grounding, planning, making, settling, filing and maintaining all
have somewhere to put their results.

**74 are disputed, but not evenly.** Three decisions unblock 51 of them:

| Decision | Unblocks |
|---|---|
| `git init` the mind folder, and settle the branch-per-Maker questions | 11 **(A)**, most of 7 **(H)** |
| Build the corpus index — or route it through the provenance gallery | 9 **(C)** |
| Add the small durable tables: promises, orders, appointments, trials, standards, judgements | ~22 of 31 **(D)** |

The rest are genuine design questions rather than missing work: what a document
vault stores on a rack, who asks it questions, whether the world's geography is
prose or a graph, and whether a reading is an event or a sentence.

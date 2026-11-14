# Hit-driven projection

A dialogue's slot is the model's native window, on the rotation it was trained
with, holding the K/V of whatever the model's own indexer keeps asking for. A
pool up to four times that size stands behind it, its index rows on the device
and its K/V in the warm tier. The indexer scores the whole pool as it prefills
and decodes; what it would have read, had the K/V been there, is loaded at the
next projection, and what it stops reading leaves.

Admission runs in two phases. The pool admits whole conversations — a file's
reading, a guide, the dialogue itself — by provenance's score and their hits.
The slot admits rounds of those conversations: a request and its answer, a
call and its result, each carved where a real conversation could have been
cut, so a large file is read a few pages at a time.

This replaces the minimum YaRN factor and the dialogue's fixed budgets as its
admission rule: the `working_set` shares in `zend/src/prompts/projection.yaml`
stay only as caps inside the score-ordered slot fill (§7.2). It is written
for Qwen3.8-Flash-Next and every model whose attention selects; a model whose
attention reads every position takes none of it.

## 1. Why

The findings this answers are in
`docs/results/model_not_seeing_context_2026-11-12.md`,
`docs/results/theorem_depth_tests_2026-11-10.md` and
`docs/qsa_stratified_selection.md`. In short:

- **Nothing is guaranteed a seat.** Each window of a query's selection spends
  the checkpoint's budget of 2,051 cells on the blocks its indexer ranks
  highest. The system prompt is attended only when its blocks rank, and at
  230K tokens they get 4–16 of their 69 blocks.
- **Production runs off the trained rotation.** `rope: min_yarn_factor: 2` puts
  every sequence on the ×2 rung, though the checkpoint was trained to 262,144
  positions at ×1 and its vendor advises a YaRN factor only when a context
  needs one.
- **The slot is larger than the trained window.** The dialogue's budgets sum to
  about 428K tokens, so the selection chooses among far more than the indexer
  was trained to choose among, cut into windows the checkpoint never saw.
- **Relevance is decided before the model reads.** Provenance fills the slot
  from its own scores. What the model's indexer then reads, and what it would
  have read, is not fed back.

## 2. The design in one page

| Part | What it is |
|---|---|
| **Rotation** | Every sequence on rung 1, the trained RoPE. The slot stays inside the trained window unless the user's locks alone pass it. |
| **Pinned ranges** | Declared ranges of the system prompt that every attention query attends, outside the selection budget, in their declared place. |
| **Pool** | Whole conversations as candidates — the dialogue, its inherited lineage, and every reading provenance proposes — up to `pool_tokens` (1M / 512K / 256K by card). Index rows on the device at the sampled layers; K/V warm, cold where warm runs short. |
| **Rounds** | The carving unit: a request and its answer, or a call and its result. Every slot of rounds reads as a conversation that could have happened. |
| **Slot** | The pool's best rounds, K/V loaded, at most the trained window (262,144 positions) less one append's reserve, seated at compact positions and selected as one window. |
| **Hits** | Per round, how often the indexer selected its blocks (loaded) or would have (unloaded, the shadow scan), on a sample of the selecting layers, over every prefilled and decoded query. |
| **Openings** | A turn's opening is prefilled once, in chunks of at most `reserve_tokens` with a boundary between them; its hits move the slot at the next boundary. |
| **Projection** | At each projection boundary, hits move each round's score, the pool is re-chosen by conversation, and the slot by round: locks, pins, floors, then score. |
| **Journals** | What leaves the pool is carried by journal notes. |
| **Kept objects** | What the substrate holds that no model computed — accounts, threads, tokens, world state, journal text — is exported as text before the model changes and imported after (§13.4). |

## 3. Terms

- **Piece**: one conversation the pool admits or drops whole: a file's reading,
  a folder summary, a guide or journal note, one link of the inherited lineage,
  or the dialogue itself.
- **Round**: one unit the slot places or removes whole, carved from a piece by
  the round grammar (§8.1): its opening request, one tool call with its result,
  or its closing answer.
- **Half**: one of a turn's two records, the user record or the assistant
  record (`code_read::couple_turn`), sealed on its own (§8.1).
- **Cell, block, chunk**: a cell is one position. A block is `ratio` cells
  (four on Flash-Next), one row of the index and the unit the selection ranks
  (`qsa_select.rs`); the checkpoint's budget is 2,051 cells, about 512 blocks.
  A chunk is the 32-token unit K/V is stored and quantized in, and plays no
  part in selection or hits. Hits are counted per block.
- **Pool**: the pieces whose sampled index rows are resident for this sequence.
- **Slot**: the rounds of the pool whose K/V is loaded. Only the slot is
  attended.
- **Selecting layers**: the layers that have a QSA index, in layer order
  (twelve on Flash-Next: layers 3, 7, … 47).
- **Sampled layers**: the selecting layers at the configured percentiles
  (§6.1).
- **Hit**: one block chosen by one query at one sampled layer, by the real
  selection or the shadow scan (§6).
- **Boundary**: a point where the projection runs. A reply reaches one at a
  line feed past the content gate, every `reproject_every_n_tokens` (64), and
  at `<tool_call>` (`provenance::boundaries`, `scheduler/decode.rs`); a turn's
  opening and each prefill chunk end at one (§7.4).
- **Stride**: the queries between two boundaries.

## 4. Rotation

The slot holds at most `L0` positions, the schedule's trained window
(`qwen4exp::rope::FLASH_NEXT_L0`, 262,144). A sequence's rung is a function of
its reach (`docs/progressive_yarn.md` §8), so a slot inside `L0` is on rung 1
at every wave: the frequencies and the temperature the checkpoint was trained
with.

- **The minimum factor is removed.** `rope: min_yarn_factor` leaves
  `projection.yaml`, and the schema field, its error
  (`projection/error.rs`) and the minimum-factor argument of `rope_key`
  (`candle-conversation/src/models/identity/rope_key.rs`) go with it. No
  deployment sets a floor.
- **The substrate starts empty.** The minimum factor is part of the model's
  identity key (`models/builder.rs`), so removing it makes every sealed block
  another model's, and the model binding refuses the old substrate
  (`persistence/model_binding.rs`). That is what it is: past the first
  attention layer, every block's K and V were computed by layers that attended
  under ×2. No model-derived record is carried across. zend's and npcd's
  substrates are moved aside on every member when the build rolls out
  (§13.4); the priming chain is primed again, code readings come back as they
  are asked for, and zend's conversations and npcd's NPC memory start empty.
  What no model computed — the custom objects and the blobs they name, and
  npcd's cast — is exported as text first and imported into the new
  substrates. Journals are git repositories outside the substrate and are
  untouched.
- **The reserve is one append, not a reply.** The slot's pieces fill
  `L0 − reserve_tokens` (default 16,384). The slot is re-chosen at every
  boundary (§7.2), so it only has to hold what can arrive between two of them:
  one cadence of decoded tokens, or one prefill chunk (§7.4). A reply of any
  length ends inside rung 1, because each boundary sheds what the next stride
  needs, down to the floors of §8.5; what gives way when even those pass the
  slot is in §7.2.
- **The higher rungs stay for callers that drive a sequence past `L0`
  themselves** (the YaRN gates, a plain-prompt conversation), and for one case
  in zend's dialogue: locks the user asked for that alone pass the slot (§7.2).
  Locks are placed whole, and a slot they carry past `L0` moves to the rung its
  reach needs.
- **Re-seating is free.** Keys and index rows are stored un-rotated
  (`progressive_yarn.md` I1) and rotated as they load, so seating a piece at a
  new compact position rewrites nothing.

## 5. Pinned ranges

A pinned range is a span of the system prompt every attention query attends,
at every selecting layer, whatever its score.

- **Declared in the schema.** A system-prompt item carries `pin: true` — a
  section, a section-tree node (every branch's variant of it) or a collection
  (every member, as it stands at each projection) — and `system_prompt:
  pin_tokens:` bounds them. zend flags `persona`, `journal_intro` and
  `journal`; the tool catalog is not pinned, and is selected by score as it
  is today. Projection marks each emitted section with its pin
  (`ResolvedSection::pin`, `projection/pins.rs`).
- **Pinned in place.** A pin marks where an item already is; it never moves it.
  System-prompt sections are sealed after the sections above them, so the slot
  seats the system prompt in its declared order and the pinned ranges are
  wherever their items fall.
- **Outside the budget.** Pinned blocks are attended in addition to the
  selection's `top_k + ratio − 1` cells, as the forced recent span is
  (`Recent::Forced`, `qsa_select.rs`). They never enter the ranking, so they
  take no candidate's seat.
- **Carried as ranges.** Beside the prompt span (a single leading length,
  `IndexCache::set_prompt_end`, which a stratified selection ranks in every
  window), the index cache holds up to four `[start, end)` position ranges
  (`IndexCache::set_pins`, `qsa_select::MAX_PINS`), declared after each
  projection walk from the pinned sections' places (`pin_ranges_of` in
  `candle-conversation/src/scheduler/projection_assembler.rs`) through
  `ManagedBatchedModel::set_selection_pins`. They move with every splice and
  cut, as the prompt span does, and reach the selection kernel as each row's
  block ranges (`PinRow`).
- **Bounded by `pin_tokens`.** The journal section grows by a line per
  stretch of the conversation, so the pinned total is not known at load.
  `pin_tokens` (zend: 1,024) bounds it: every whole pin that fits — the
  persona, the journal's introduction — then the journal's entries newest
  first until one does not fit (`within_budget`); older entries stay in the
  slot, selected by score. The engine sizes every selection row and its union
  buffer for twice `pin_tokens` in blocks — every pinned section is its own
  sealed page and can end on a short block — plus one straddling block per range
  (`StrataTokens::pinned`, `Strata::pin_blocks`, `max_entries_for`); a row
  whose pins cover more — ragged section pages can — gives up the front of
  its last pinned range, the oldest journal entries, and is never refused
  (it is built inside a forward). The engine's bound is the largest any
  tenant asks; a sequence that declares pins on an engine bounded at zero is
  warned of at the declaration.
- **At most `MAX_PINS` ranges, the growing one last.** The pinned items' places
  are fixed by the schema's declared order, and a collection's members sit
  together, so the number of disjoint ranges is known at construction:
  adjacent pinned items make one range, and a schema whose pins fall into more
  than `MAX_PINS` (4) ranges is refused (`InvalidPins`). A pinned collection
  is the one pin that grows, so it must be the last pinned item, or the schema
  is refused; that is what makes "the front of the last range" its oldest
  members. zend's pins are two ranges: the persona, and the journal's
  introduction with the journal behind it.
- **The slot is the model's trained window.** There is no `slot_tokens`: the
  slot is `L0` as the engine reports it for its model, so the two cannot
  disagree. The working set's shares and `reserve_tokens` are checked against
  it when the engine is built; the shares and the conversation's `window` are
  caps (§7.2). The system prompt is measured as it is emitted and the fill
  (§7.2) places everything else in what it leaves, so the system prompt needs
  no margin and a slot never leaves the trained window on its own content.
  Only locks, or floors that alone pass the slot, carry it out (§7.2), and the
  assembler warns when a placement does.
- **Cost.** Every pinned cell is read by every query at every selecting layer:
  at `pin_tokens`, half the checkpoint's own read. Forcing the 8K recent span
  cost +93% decode and confused reasoning after a tool call
  (`qsa_stratified_selection.md`), which is why the catalog is not pinned by
  default.

Pinning makes a range attended. The recurrent layers do not read the slot, so
it does not put the range into their state.

## 6. Hits

### 6.1 The sampled layers

`hits.layers` is a list of percentiles over the selecting layers, default
`[40, 65, 90]`. Percentile `p` over `n` selecting layers is the layer at index
`round(p / 100 · (n − 1))`; two percentiles that land on one layer use it
once. On Flash-Next that is the 5th, 8th and 11th selecting layers: model
layers 19, 31 and 43.

Only the sampled layers hold index rows for unloaded pieces, and only they
count hits. The slot holds every selecting layer's rows, as it must: the real
selection runs at all of them.

### 6.2 The real selection

The slot is one window. Unless locks alone carry it further (§7.2), it never
exceeds `L0`, the length the indexer was trained to rank, so the strata over a dialogue's slot are `Strata::WHOLE`: one
ranking over every candidate, the checkpoint's selection exactly, plus the
pinned ranges forced and the query's tail. Nothing without K/V is ever named
in a selection. The stratified windows (`StrataTokens::DEFAULT`) remain for a
sequence that runs past `L0` outside zend's dialogue.

Prefill rows and decode rows are selected alike: the wave's selection table
holds one row per query, decode slots first, then prefill rows, and every row
ranks the history it can see.

At a sampled layer, each selected block of a loaded round is one hit for that
round. The lowest score the selection kept is that query's **cut** at that
layer. A row whose visible cells fit inside the budget attends all of them and
ranks nothing: it has no cut, counts no hits, and keeps no query for the
shadow scan.

### 6.3 The shadow scan

The real selection only sees the slot, so on its own it can keep a round and
never bring one in. The shadow scan asks, for what is pooled and not loaded,
what the indexer would have selected had it been there. A turn-by-turn carving
of large readings depends on it: without it, a round outside the slot has no
way back but its base score.

At each sampled layer, the stride's queries are scored against the unloaded
rounds' index rows.

- **In windows ending at the query.** The unloaded rows are cut into windows
  in pool order, each as long as the query's own position `p`. Each window is
  seated directly behind the query: its rows rotate at positions `0 .. p`,
  through the scorer's per-page signed offset, so the query is the one the
  real selection used, rotated where it already is. The indexer ranks a
  context of the length the slot gives it, never longer than it was trained
  on, and its scores are on the scale of the real cut.
- **Windows hold unloaded rows only.** The slot's prompt and pinned ranges are
  already attended and are not repeated in any window. The work is bounded by
  the pool, not by the window count: every unloaded row is scored once per
  query whatever `p` is, and a window's top-k is linear in its length, so a
  small `p` early in a dialogue makes many short windows for the same total.
  A window no longer than the budget keeps every block, and the cut alone
  decides its hits.
- **A hit is a block that would have been read.** Each window ranks its blocks
  as the real selection ranks the slot's: the checkpoint's budget, one
  window. An unloaded block is one hit for its round when it is in its
  window's selection **and** scores above the query's cut. The first condition
  keeps it to what the indexer would choose from a context that size; the
  second keeps a window of nothing relevant from handing out a full budget of
  hits.
- **Weighed as real hits.** A shadow hit counts into the same counter, at the
  same weight, as a hit on a loaded round. Above the cut it would have
  displaced a block the selection did read, so it is evidence of the same
  strength, and an unloaded round can outscore a loaded one on what the
  indexer asks for.
- **The same three layers.** The shadow scan runs at the sampled layers
  (§6.1) and nowhere else; those are the only layers whose index rows the
  pool holds for unloaded rounds.
- **Once per boundary, in one launch per sampled layer.** The stride's index
  queries on the sampled layers are kept beside the stride (four heads of 128
  F32 per token per layer: about 0.4 MB for 64 decoded tokens on three
  layers), each with its cut record — the lowest score the real selection
  kept, the query's position, its candidate blocks and its tail cells, which
  the selection kernel writes per row. The hit count's own launch copies them
  into slot `position mod 64` of the sequence's kept-query buffer, so decode
  and prefill launch nothing for it. A boundary's read of the counters
  (`take_hits`) asks for a scan; the first wave after it scores the kept
  queries against every unloaded window at each sampled layer, in one launch
  over the wave's asking sequences, before that layer's count writes the next
  stride's queries over them.
- **A prefill keeps its last 64 rows.** A prefilled opening or chunk keeps the
  queries of its last 64 positions, as many as a decoded stride, and only
  those enter the shadow scan. They are the queries nearest the reply, whose
  question has been fully read; the opening's other rows still count real hits
  on the slot.
- **Nothing is attended.** The shadow scan writes counters and nothing else.

### 6.4 Counting

Counters are per round per sequence, on the device (`qwen4exp/hits.rs`). A
device map takes each block to its round, loaded or pooled, rebuilt when a
projection declares its rounds (`ManagedBatchedModel::set_hit_rounds`); after
the selection at a sampled layer one batched launch adds each chosen block
into its round's counter (`qsa_hits.cu`, a span table naming each sequence's
rows, map and counters), and each row that selected adds one query. The
counters live on the reservation's held ground and are cumulative over a
declaration. After the wave's last sampled layer a second launch publishes
each sequence's counters into a mapped pinned block (`qwen4exp/hit_summary.rs`)
— a sequence word, the counts, the word again — as MoE routing publishes its
summary. The reprojection reads the newest finished block at the boundary
(`take_hits`) and takes the difference from its last read; nothing on either
side waits, and what is still in flight is in the next read. A row inside the
budget made no selection and counts nothing.

A counter's identity is the round's: (piece id, round index), stable across
projections (§12 item 8), so the hook that declares them is
`set_hit_rounds`. Counters are kept per sequence, and per round within it.

A round's **hit density** over a stride is its hits divided by its blocks, by
the number of sampled layers, and by the queries that could have hit it. A
round is loaded or unloaded for a whole stride, since membership changes only
at a boundary, so that count is one of two: for a loaded round, the stride's
rows that made a selection (every such row of a prefill, every decoded token);
for an unloaded round, the rows kept for the shadow scan (at most 64). Each
query is then one unit of evidence whichever way it was counted. Density, not
count, so a long round does not outrank a short one by size. A piece's
density is its rounds' hits over its blocks, on the same denominators. A decoded stride is at most 64 queries, so one
stride's density is noisy; the momentum (§7.1) is what the score reads, and at
a leak of 0.2 it carries about the last five strides.

## 7. Projection

### 7.1 Score

Each round, and each piece, carries a base score and a hit momentum.

- **Base** is what the projection gives it today, on the one band a
  reprojection scores on, `0 ..= 1,000`, where 1,000 is one full-strength hit
  (`LINEAGE_BASE`): provenance's normalized score for a retrieved piece and
  each of its rounds, the seed momentum for a seeded one (normalized to the
  same band), the turn's own belief score for a round of the dialogue, as the
  history is scored today, and `lineage_base` for a link of the inherited
  lineage (§8.4). There is no distance base: the dialogue's recent exchanges
  hold their seats as floors (§8.5), and older rounds compete on evidence like
  everything else. Ties break oldest first.
- **Hit momentum** `h` follows the stride's density `d`:
  `h ← (1 − β) · h + d`, with `β` the working set's leak (0.2).
- **Score** is `base · (1 + gain · h)`. A piece the indexer keeps reading holds
  its score against a falling base; one it never reads falls with it.

`hits.gain` (default 1) and `hits.beta` (default 0.2) are schema fields beside
`hits.layers`. The lift applies twice over: to a working-set member's strength
— its momentum times `1 + gain · h`, wherever the working set admits, keeps or
dislodges (`WorkingSet::score_of`) — and to each placed turn's score in the
reprojection that follows (`ProjectionScores::lift_turns`), so a turn the
model keeps reading outranks the history it would otherwise age out with.

### 7.2 Choosing the pool and the slot

At each boundary:

1. **Pool, by piece.** The pieces holding a floor (§8.5), then pieces by
   score — provenance's base lifted by the piece's hit momentum — until the
   pool holds `pool_tokens`. A piece enters or leaves the pool whole: every
   round of a file's reading is a candidate for the slot, or none is.
   `pool_tokens` is per card, by the card's size and nothing else:
   1,000,000 on the 72 GB card, 512,000 on the 24 GB card, 256,000 on the
   16 GB card (`scheduler::pool::pool_tokens_for`). No configuration
   overrides it, in `projection.yaml` or the machine config. The slot is the
   same on all three. The pool is per sequence: each dialogue chooses its own.
   On the 16 GB card it is barely larger than the slot and the shadow scan has
   little to find; that is accepted.
2. **Shared index pages.** A piece's sampled-layer index pages are placed on
   the device once per card, whichever sequences pool it, and refcounted: a
   sequence pooling a piece takes a reference, one dropping it gives the
   reference back, and the pages are released when the last reference goes.
   Many pool pieces are file readings every dialogue shares, so pool memory
   grows with the distinct pieces pooled, not with the dialogues open. Hit
   counters stay per sequence: the shared page says nothing about who read it.
   The pages are read-only once placed, so a sequence's selection, the shadow
   scan and the hit map all address them through the span's descriptor tables
   (AGENTS invariant 2b) without a copy. They live in the span's KV-regions
   tenancy, beside the slot's index pages, allocated from the region pool.
   Index pages are sealed and persisted with a record's K/V (warm, and in the
   redo log), so a piece's rows come from wherever its K/V is: a cold piece's
   rows are read from the log on admission and it is scored from the boundary
   after they land.
3. **Pooled K/V.** A piece entering the pool has its K/V brought to the warm
   tier, so a round chosen for the slot loads at PCIe speed. The warm tier is
   the substrate's, one copy of a sealed record however many sequences pool
   it, so warm K/V is shared per piece as the index pages are, and bounded by
   the warm tier's own capacity, not by `pool_tokens`. Where the warm tier has
   no room for all of it, the pool takes what fits and leaves the rest cold; a
   cold round chosen for the slot loads from the redo log, one boundary later,
   and holds no room while it is in flight: the slot is filled without it, and
   it is placed at the next boundary if it still ranks. A round placed in the
   slot is hot.
4. **Slot, by round.** One fill replaces the budget passes of the working set
   and `select_conversation`, and `slot_room` with them. Its room is
   `L0 − reserve_tokens`, less the system prompt as emitted (pinned ranges
   included) and what the turn has already appended. Rounds in this order:
   1. **Locks**, whole. A read the user asked for is placed entire, whatever
      it costs. If the locks alone pass the slot they are placed anyway, and
      the slot runs past `L0` on the rung its reach needs (§4).
   2. **Floors** (§8.5), in their own order: the current exchange (its
      request, its newest tool round and the reply as it grows), the
      dialogue's first request, then the recent exchanges newest first.
   3. **Everything else by score**, highest first, whichever piece it comes
      from, each round admitted only while its kind is under its cap. A
      round of a piece not yet placed brings the piece's request and answer
      with it (its floor, §8.1), and is admitted only if all three fit.

   Every other group of the projection, on the dialogue's layer or any
   other, is part of the last tier: each exchange it selects is a unit, whole,
   at its selection score, under its layer's `window` as cap (the dialogue's
   `window` for a group on the dialogue's layer). On a selecting engine
   nothing is flexed; the flexbox sizes the layers only for a model that does
   not select.

   **A working set that cannot fit `L0` fills `2·L0`.** When `lock_tokens`
   and `reserve_tokens` together pass the trained window, or the reserve
   alone leaves nothing of it, the slot's window is the rung above, YaRN ×2,
   and the assembler warns once; the fill runs as before against that room.

   Each tier takes its room from the tiers below it, so as locks grow,
   provenance's rounds and the lower-scored history of the dialogue shrink
   with them, never the locks themselves.

   **When the floors pass the room, they give way from the bottom.** A recent
   exchange that does not fit sheds its tool rounds, oldest first, and leaves
   whole once it has none left; the recent exchanges go oldest first. If the
   current exchange alone still passes the room, its own older tool rounds go,
   oldest first; its request, its newest tool round and the reply stay. Only
   when those and the first request alone pass the room (`L0 −
   reserve_tokens − system prompt − appended`) does the slot run past `L0`,
   as locks carry it, and the assembler warns. A
   reply therefore stays on rung 1 unless one exchange's request, newest
   result and reply alone outgrow the trained window.

   **The working set's shares are caps.** They bound how much of each kind the
   slot holds from the last tier, and never reserve room:

   | Cap | Bounds |
   |---|---|
   | `budget` (64,000) | every working-set round in the slot together: readings, folder summaries, guides |
   | `folder` (10,000) | folder-summary rounds |
   | `guidance` (16,000) | guide and journal-note rounds |
   | `max_file` (48,000) | one reading's rounds, its request and answer included |
   | `window` (180,000) | the dialogue's own rounds and the inherited lineage, its floors included |

   A round whose kind is at its cap is skipped and the fill goes on to the
   next by score, so a capped kind leaves its room to the others. Floors count
   toward their kind's cap but are never skipped for it. Locks are bounded by
   `lock_tokens` alone. The pool has no caps but `pool_tokens`. The caps may
   sum past the room, and do: `budget` and `window` together are about 244K,
   so a slot whose working set is thin is filled by the dialogue, and one
   whose dialogue is short by the working set. The caps keep either from
   taking the whole slot; they never leave it underfilled while either kind
   has rounds to give. `window` is raised from 96,000 to 180,000 for that,
   in the dialogue layer of `projection.yaml`.
5. **Rounds in and out.** A round entering the slot has its K/V and its
   remaining layers' index rows brought to the device; one leaving keeps its
   sampled rows and its warm K/V while its piece stays pooled.
6. **Seating.** Slot rounds are seated at compact positions: the system
   prompt in its declared order, then the lineage, then each placed piece's
   rounds in the piece's own order, then the dialogue's rounds as the
   projection orders them today.

### 7.3 Membership and positions

The slot's membership changes only at a boundary, and a reply crosses many of
them. Between two boundaries the slot's positions are fixed; at a boundary the
slot may be renumbered under a reply in progress, as reprojection does today,
and the reply's own tokens move with it.

A piece chosen at a boundary is loaded through the deferred placement the
scheduler already runs (`scheduler/decode.rs`, the deferred glue fire): its
K/V comes up from the warm or cold tier while the sequence's next wave is
built, and the sequence decodes against the slot once the placement lands. A
piece leaving the slot is released at the same placement.

### 7.4 Openings

A turn's opening — the user's turn, or a tool result — is prefilled once,
against the slot the last boundary chose. Its prefill rows are selected
(§6.2), so the sampled layers count real hits and its last 64 rows are kept
for the shadow scan, exactly as decoded queries are. The boundary at its end
re-chooses the slot from those hits, and the reply decodes against that slot
(`wants_early_reprojection`, which fires at the first decoded token).

An opening longer than `reserve_tokens` is cut into chunks of that length
with a boundary between them, so a long tool result moves the slot as it
arrives and the slot never passes `L0`. Each chunk is prefilled once, against
the slot the boundary before it chose.

The prefill group carries an opening no further than its next multiple of
`reserve_tokens` (`ActivePrefill::stops_every`). Where a chunk lands on one
with tokens still to come, the opening waits (`awaiting`), rides no group,
and its reprojection is queued like a decode's. The reprojection holds the
turn as an opening rather than a decode (`scheduler/held_turn.rs`): the same
swap, the opening's prefilled part captured as the turn's tail and counted
against the slot's room, and on completion the opening goes back to the
prefill group on the carved view to prefill its next chunk. A reprojection
that does not run — a selection the slot already holds, or no ground for
the scan — lets it go on against the slot it has. A staged or captured
calibration prefill is not cut: its projection points and recurrent arms are
its own.

An opening is not prefilled twice. The failure that would have justified it
was attention never reaching a seat the slot held (§1), which the one window
over a trained-length slot and the pinned ranges answer directly; a second
prefill of every tool result is a cost with nothing left to buy.

### 7.5 What provenance does now

Provenance proposes pieces for the pool and gives them their base score. It no
longer decides what is attended: with only index rows to pay for, it can
propose four times what the slot holds, and the indexer's hits decide which of
that is loaded.

## 8. The conversation

### 8.1 The round grammar

Every piece placed in the slot reads as a conversation that could have
happened: the user speaks, the assistant answers it, a tool's result follows
the call that asked for it. No carving ever seats two user turns or two
assistant turns side by side.

A conversation of tool use has one shape:

```
U request · (A call · U result)* · A answer
```

- the **request**: the user's turn that opens it;
- each **tool round**: the assistant's call and the result returned for it;
- the **answer**: the assistant's closing reply.

These are the rounds the slot places. The request and the answer are the
piece's floor: when any round of a piece is placed, its request and answer
are placed with it. A carving keeps the grammar by construction. Dropping any
tool round leaves a request, rounds that each answer the turn before them, and
a closing answer to the request.

Tool rounds cross turns. The turn store keeps a turn as a user record and an
assistant record (`code_read::couple_turn`): turn `n` is the result of call
`n − 1` and the reply that makes call `n`. A round's call is one turn's
assistant record and its result is the next turn's user record, so a round is
two whole records from two adjacent turns, never a record split at the call.
Those records are the **halves**: each is sealed on its own, with its own
index page.

- **Sealed per half.** Every turn is sealed as its two records, each with
  K/V and index rows of its own. A half placed alone
  (`inject_sealed_turn_half`) carries its index page, so the indexer scores it
  and the hit map counts it as a block range of its round. A half placed
  without an index page is not a valid placement: the selection could not
  rank it, and it would be attended only through the query's tail. The
  substrate starts empty under the new key (§4), so every record is sealed
  per half from its first write.
- **Ragged, not padded.** Each half ends on its own partial chunk and partial
  block, as every system-prompt section already does (§5). Seated halves take
  consecutive positions; the short tail is a short page in storage, never dead
  positions in the slot.
- **Seated by the call that produced it.** A round's call and result keep
  their order and sit between the request and the answer. Rounds keep the
  order they were read in, so the pages they hold read in file order.

### 8.2 A file's reading is carved

A code reading (`zend/src/code_read/`) is one piece. Its request asks for the
file, its tool rounds each read one range of it, and its answer summarises it.
The pool holds the whole reading; the slot holds its request and answer and
the pages the indexer asks for. Placing a 48K-token file in a dialogue does
not cost 48K, only the pages the question reads.

- **One read per reply.** The reading asks for one `file_read` per reply, one
  range of at most `PAGE_LINES` (200) lines each, so each tool round is one page
  and carves at that page's cost. A reply that batched `MAX_TOOL_CALLS_PER_TURN`
  (12) reads would make a round of twelve pages, placed or dropped together.
  Code-reading ingest bounds its own replies to one call
  (`opening_prompt`, the reading's tool grammar), and its round limit is the
  file's page count, not `MAX_FILE_READ_ROUNDS`, so a file of any length reads
  to its end in rounds of one page.
- **Scored per round.** Each page has its own hit counter and base. The base
  is provenance's score for the reading, the same for every round of it; hits
  decide which pages of it the slot holds.
- **The reading's own projection is not carved.** While a reading is being
  ingested its conversation is its own target and selected whole
  (`target_is_ingest_self`), as it is today.

### 8.3 The dialogue

The dialogue is carved by the same grammar, and each exchange is a piece of it
for the floor rule: the user's turn, its tool rounds and the assistant's
answer to it, never the user's turn without its answer.

- **Recent exchanges are floors, whole.** The current exchange and the
  `recent` (16) before it hold every round they have, tool rounds included,
  and give way only as §7.2 says when the floors pass the room: tool rounds
  oldest first, then the exchange whole.
- **Older exchanges compete by score.** An older exchange's request and
  answer are one round of the fill, the **pair**, scored from their turns'
  base and lifted by their hits; a plain question and answer is that pair
  alone. Each of its tool rounds is a round of its own, scored the same way,
  and is admitted only with the pair: a tool round that ranks before its pair
  brings the pair with it and enters only if both fit. So the history is
  score-ordered, and what remains of any exchange still reads as a complete
  exchange. A reading's request and answer are not a pair of this kind: they
  carry nothing without a page, so they come only with its first placed
  round (§8.1).

### 8.4 The inherited lineage

Every dialogue descends from the priming chain (`branch_ingest/prime.rs`): a
repository's root listing, then its README, ARCHITECTURE, AGENTS and CLAUDE,
about 40K tokens and capped by `Substrate::INHERITED_CHAIN_TOKEN_CAP`. Today
the lineage was inviolate in a dialogue. Here each turn of it is scored like the
conversation's history from a base of its own — `lineage_base` on the
conversation group's selection, default one full-strength hit (1,000,
`LINEAGE_BASE`), with the ancestor whisker that makes the oldest ancestor go
first and whole — and competes in the slot fill's score tier (§7.2) as a
round of the dialogue's kind, counted against `window`. It holds its seat
against everything below its base and gives way to a turn that outscores it:
one the indexer keeps reading, whose score its hits have lifted (§7.1).
Seated, the lineage keeps its order, after the system prompt and before
everything else.

### 8.5 Floors

Always in the pool and the slot, whatever their score, in the order of §7.2:

- reads the user asked for, which the working set locks, placed whole. Locks
  come first; when they pass the slot the slot runs past `L0` (§4), and
  everything below them gives up its room. `lock_tokens` still bounds what
  can be locked: a lock past it is refused and the read runs for real
  (`Refusal::Full`), its result in the dialogue's own rounds.
- the system prompt, its pinned ranges with it (§5);
- the current exchange: its request, its newest tool round and the reply as
  it grows, and its older tool rounds until §7.2 sheds them;
- the dialogue's first user turn;
- the dialogue's most recent exchanges (`recent`, 16), whole, newest first;
- each placed piece's request and answer (§8.1), which come with the piece's
  first placed round rather than on their own.

Everything else is scored round by round. When provenance scores are low the
dialogue fills the slot; when they are high the oldest unread rounds give way.

### 8.6 Journals

An exchange that has left the pool is no longer attended and cannot be scored.
What it decided is carried by the journal, whose notes are pieces provenance
proposes like any other. Notes are written as notes about the work, never in
the assistant's first person: a compressed turn written as if the assistant
had said it is read back as something it said
(`projection.yaml`, the dormant `summary:` block).

## 9. What changes where

| Piece | File |
|---|---|
| `rope: min_yarn_factor` removed with its schema field, error and identity argument; the substrate moved aside under the new key, its kept objects exported and imported | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{schema,error,builder}.rs`, `candle-conversation/src/models/{builder.rs,identity/rope_key.rs}` |
| `hits:` block; `pin: true` on system-prompt items and `pin_tokens`; `reserve_tokens`, `lock_tokens`; `lineage_base` on the conversation rule; the `working_set` shares and `window` (180,000) as caps on the slot fill; `slot_tokens` removed, the slot read from the engine's `L0` | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{yaml,schema}.rs` |
| Pinned ranges in place of the prompt span; forced in every row; one window over the slot | `candle-transformers/src/models/qwen4exp/{indexer,qsa_select,wave}.rs`, `candle-kernels/src/simple/qsa_topk.cu` |
| Sampled layers from percentiles | `candle-transformers/src/models/selection_strata.rs` beside `StrataTokens` |
| Index rows resident without K/V; the stride's kept queries and cuts; the shadow scan and its windows | `candle-transformers/src/models/qwen4exp/{resident_page,indexer,hits,shadow}.rs`, `candle-kernels/src/simple/{qsa_topk,qsa_hits,qsa_shadow}.cu` |
| The block-to-piece table, per-piece counters and the mapped summary | `candle-transformers/src/models/{hit_layers.rs,qwen4exp/hits.rs,qwen4exp/hit_summary.rs}`, read in `candle-conversation/src/scheduler/{projection_assembler,hit_density}.rs` |
| Openings chunked at `reserve_tokens` with a boundary between chunks | `candle-conversation/src/scheduler/{prefill,decode,held_turn,reproject_abandon}.rs` |
| Text export and import of every kept custom object and the blobs it names | `zend/src/api/kept_objects.rs` and `npcd/src/api.rs`, over `candle-conversation/src/persistence/kept_objects/` |
| Scores, the pool, the slot, floors, the lineage, exchanges and tool rounds as units | `candle-conversation/src/projection/{fill,dialogue_fill,rounds,project}.rs`, `candle-conversation/src/scheduler/projection_assembler.rs` |
| `ManagedBatchedModel` hooks for pinned ranges, hit rounds and summaries, and the trained window (no-ops for a model that does not select) | `candle-transformers/src/models/batched_inference.rs` |
| `pool_tokens` per card size; pooled K/V warmed, cold where warm runs short | `candle-conversation/src/scheduler/{pool,pool_refresh,pooled_rounds}.rs`, `candle-conversation/src/persistence/warm_asks.rs`, `candle-conversation/src/projection/cold_to_warm.rs` |
| Rounds carved by the grammar; slot order locks, pins, floors, score | `candle-conversation/src/projection/{rounds,fill,dialogue_fill}.rs`, `candle-conversation/src/scheduler/projection_assembler.rs` |
| Every turn record placed alone, each with its index pages | `candle-conversation/src/{turn_layout,index_pages,substrate}.rs`, `candle-conversation/src/scheduler/projection_assembler.rs` |
| One `file_read` per reply in code-reading ingest, the round limit the file's page count | `zend/src/code_read/mod.rs` |
| Shared sampled-layer index pages per card | `candle-transformers/src/models/qwen4exp/{resident_page,wave}.rs`, the span's region pool |

## 10. Cost

- **Index memory.** One pooled key is 128 F32 per four tokens per layer. The
  unloaded pool on three sampled layers is at most about 290 MB at a 1M pool
  (about 750K unloaded), about 100 MB at 512K and about 4 MB at 256K, where
  the pool is barely larger than the slot. Pages are shared per card, so
  dialogues pooling the same readings pay for them once; the slot's twelve
  layers are about 400 MB a sequence, as today at that size.
- **Warm K/V.** A pooled piece's K/V is held in the warm tier. Where the warm
  tier is short, cold rounds cost a redo-log read and arrive a boundary late.
- **Shadow scan.** One launch per sampled layer on the first wave after a
  boundary: up to 64 queries per layer
  against the unloaded keys, about 290 MB a sequence at a 1M pool. A stride
  reads those keys once for all its queries, against once per token if it ran
  in the forward.
- **Ingest.** One read per reply makes a reading take one reply per page:
  more, shorter turns than batched reads, and every turn record sealed in two
  halves.
- **K/V.** At most the trained window, against about 428K of budget today.
- **Decode.** One window instead of two halves the cells a query reads at a
  full slot. The pinned ranges add their cells to every query's read.

## 11. What this does not do

- **A piece computed in another context is still read as one.** A reading
  sealed in its own conversation carries that context in its K/V; placed in a
  dialogue it can be attended and not used (`theorem_depth_tests_2026-11-10.md`).
  This design loads what the indexer asks for. It does not change what a loaded
  piece's K/V says.
- **The recurrent layers never see the slot.** Thirty-six of Flash-Next's
  forty-eight layers carry a state that ran over the conversation's own stream
  and nothing placed beside it, pinned ranges included.

## 12. Built

Everything this document describes is built. Each item names the section it
answers to and where it lives.

1. **Rotation (§4).** The minimum factor is gone from the schema, the engine
   asks, the model builder, the rung selection (`RungSelect` has no floor) and
   the identity key (`rope_key` names the schedule alone).
2. **One window over the slot (§6.2).** `StrataTokens::DEFAULT` windows are
   the trained window, 262,144 positions, so a slot inside it is one ranking.
3. **Pinned ranges (§5).** `pin: true` on system-prompt items and
   `pin_tokens` in the schema, projection marks, assembler ranges, the index
   cache, the selection definition (`selection_entries`) and the selection
   kernel (`qsa_topk.cu`), which the kernel tests pin against the
   definition. Pins past `MAX_PINS` ranges, or a pinned collection that is
   not the last pin, are refused at construction.
4. **Sampled layers and counting on the real selection (§6.1, §6.2, §6.4).**
   Percentiles to layers (`hit_layers.rs`), the per-sequence map and
   counters, the batched count after each sampled layer's selection
   (`qsa_hits.cu`).
5. **The counters off the blocking path (§6.4).** Counters stand on held
   ground in the span. After its last sampled layer the device publishes
   them into a mapped pinned summary behind a seqlock (`hit_summary.rs`:
   begin, token, generation, end, then `queries`, `shadow_queries`, then the
   hits), and a reprojection reads the summary without waiting on the
   device, as MoE routing's summary is read (AGENTS invariant 3). Queries are
   counted on the device.
6. **Per-record sealing (§8.1).** A turn's user and assistant records are
   placed alone, each with the index pages inside it, and a record with none
   is refused at placement; the seal warns when its pages miss the edge
   between the records (`turn_layout.rs`, `index_pages.rs`).
7. **One read per reply (§8.2).** Code-reading ingest bounds a reply to one
   `file_read` call; the reading's grammar is a loop of one call, its round
   limit the file's page count (`zend/src/code_read/mod.rs`).
8. **Rounds (§8.1, §8.3).** `projection/rounds.rs` carves every placed piece
   and the dialogue into rounds by the grammar; hits are declared and counted
   per round of each exchange, and a round's momentum lifts the turns it
   covers. A round is named by its piece and index, so its counter and
   momentum follow it across projections.
9. **Score (§7.1).** Hit momentum per working-set member and per round,
   folded at every reprojection, lifting working-set strength and turn
   scores. Density reads each round against the denominator it was counted
   under: `queries` for a loaded round, `shadow_queries` for a pooled one
   (`scheduler/hit_density.rs`).
10. **Shared index pages (§7.2).** Index pages are resident per layer
    (`qwen4exp/resident_page.rs`). A sequence's pool holds a piece's sampled
    layers only, shared per card with every slot that places the rest, and
    the pool's rows are addressed by descriptor table.
11. **The pool (§7.2).** `pool_tokens` by card size (`scheduler/pool.rs`); a
    dialogue's pool is its floors, then provenance's candidates lifted by
    their rounds' hits, up to the card's `pool_tokens`; each pooled
    conversation's pages are held at the sampled layers
    (`scheduler/pool_refresh.rs`). At each reprojection a pooled
    conversation's K/V is asked into the warm tier, and a cold-only turn is
    rebuilt in host RAM from its log image within the substrate's share of
    the warm budget (`PersistenceTrigger::warm_turns`,
    `ChunkedKvBacking::warm_from_cold`). The hit map covers loaded rounds,
    then pooled ones (`PooledRound`).
12. **The slot fill (§7.2).** `projection/fill.rs`: one fill after the
    system prompt — locks, floors that give way from the bottom, then units
    by score under the caps (budget, folders, guidance, `max_file_tokens`,
    the target's window, each other layer's window), each round bringing its
    piece's floor, every unit placed once. A dialogue's lineage, working set
    and exchanges are one fill, carved back into whole turns and single
    records in seating order (`dialogue_fill.rs`). On a selecting engine
    every group of the projection enters the fill, so nothing is flexed.
    `slot_tokens` and `slot_room` are gone: the slot is the engine's `L0`
    less `reserve_tokens`, and a working set whose `lock_tokens` and
    `reserve_tokens` pass it fills `2·L0` (YaRN ×2) with one warning
    (`SlotWindow`). The dialogue's window is 180,000.
13. **The lineage (§8.4)** is scored from `lineage_base`, ancestors ranked so
    a tight window gives up the oldest first and whole.
14. **Hits across views.** A view carved from a slot counts into the slot's
    counters while the slot keeps the declaration it was carved under; a
    fork that projects for itself counts into counters of its own and starts
    with no pool.
15. **The shadow scan (§6.3).** The selection writes each row's cut (cut,
    query position, candidates, tail) beside its rotated query; the count
    kernel keeps the last 64 rows' of each sampled layer in slot `pos % 64`.
    On the first wave after a boundary, `qsa_shadow.cu` scores the kept
    queries against the pooled unloaded rounds, one launch per sampled layer
    over the wave's sequences, its descriptors built at scan time because
    compaction moves pages (`qwen4exp/shadow.rs`). Windows are seated behind
    each query; a hit is in its window's top `keep` and above the cut, and is
    counted into the same counters with `shadow_queries` as its denominator.
16. **Openings (§7.4).** An opening's rows are selected and counted like
    decoded rows, and the turn's first reprojection fires at its first
    decoded token (`wants_early_reprojection`). On a selecting engine an
    opening longer than `reserve_tokens` is prefilled in strides of that
    length: each stride stops at its boundary and waits while a reprojection
    rechooses the slot from the hits its rows ran up, holding the turn as an
    opening (`scheduler/held_turn.rs`) and handing it back to the prefill
    group on the carved view. A reprojection that does not take the turn
    (an unchanged selection, refused ground) lets the opening prefill on
    against the slot it has. Staged and captured calibration prefills are
    never cut.
17. **Kept objects, exported and imported (§13.4).** zend and npcd each
    answer `GET /v1/substrate/{zend,npcd}/objects/export` and `POST
    /v1/substrate/{zend,npcd}/objects/import` (admin, as `/v1/npcs/export`
    is; under each daemon's own substrate path, as its other substrate views
    are, so the two tenants' routes never collide on one host). The library
    is `candle-conversation/src/persistence/kept_objects/`. The export is
    text, JSON Lines: a header line `{"format": "substrate-objects",
    "version": 1, "daemon": "zend" | "npcd", "taken_at_ms": …}`, then one
    line per kept custom object, `{"object": <CustomObjectPayload>,
    "stamp": [gen, epoch, hlc, origin]}` with its key, metadata and JSON value
    as stored and its record's original stamp, in the order of those stamps,
    then one line per blob a kept live object names, `{"blob": "<sha256
    hex>", "bytes_b64": "…"}`. An object names a blob by spelling its SHA-256
    in lowercase hex anywhere in its metadata or its value, the one spelling
    every object that names a blob writes; a spelling no blob answers names
    nothing.
    - **What is kept.** Every live custom object except those whose meaning
      is a model-derived record, which the move-aside destroys. One list in
      the code names the left-out prefixes: `cluster/` (the cluster's own
      state, rebuilt when the app starts over), `recurrent/` (recurrent reply
      rows, model state), `working_set:` (marks on turns that no longer
      exist; the locks they rebuild go with the dialogue) and `channels/`
      (sections npcd submits to a character's conversation, submitted again
      when the character's conversation is rebuilt). Every other key is kept:
      the prefixes are declared across both daemons and the host with no
      registry to walk, so the classification is the left-out list alone
      (`kept_objects::LEFT_OUT`), and a new prefix is kept unless it is added
      there. Kept objects that name a conversation, thread, turn or journal
      by id (`mcp/conversation/`, `mcp/chat/`, `journal/turns/`, MCP threads)
      keep the reference: a reader already resolves an id with no record
      behind it as absent, as it does a tombstoned one, and none refuses on
      it.
    - **The import** writes each blob, then each object, under the importing
      member's node with fresh stamps, in the export's order, so whatever is
      ordered by stamp comes back in its original order and replicates to
      the others like any write. A key already live is left as it is and
      named in the import's answer, with counts of what was written. It
      takes several exports at once — one from each member, one after
      another in the body, each starting at its header line — and writes the
      union, the latest original stamp winning per key, so an object a
      member held that had not reached the others yet is not lost.
    - **Tombstones travel too.** The export also carries every ceased key it
      holds, as an object line with `ceased: true`, in the same stamp order.
      In the union a tombstone competes with a live object on its stamp like
      any write, so a key deleted on one member before its tombstone reached
      another stays deleted. A key whose winner is a tombstone is not written
      at all: the new substrate never held it.

## 13. Handover

### 13.1 Rules

AGENTS.md binds every change: no env-var flags, no compatibility paths, no
TODOs or stubs, files changed with the file tools only, tests beside the code
(kernel and codec tests against raw expected bytes), present-tense comments,
the hot-path invariants. No AI attribution in commits.

### 13.2 Tests

Every item of §12 is unit-tested beside its code. Beyond them:

- **Kernels.** The shadow scan against a CPU definition of the same windows,
  top-k and cut, byte for byte on the counters; the hit count with pooled
  blocks in the map.
- **Grammar.** A property test over random pieces and slot sizes: every
  carving reads `U (A U)* A`, never two of one role adjacent, and every placed
  piece holds its request and answer.
- **Fill.** Locks past the slot are placed and the slot leaves rung 1; a kind
  at its cap is skipped and its room goes to the next; floors count toward
  caps and are never skipped; floors past the room give way in the order of
  §7.2; a thin working set leaves its room to the dialogue.
- **Density.** A loaded and an unloaded round hit by every query they could
  have been hit by score the same density, whatever the stride's prefill and
  shadow row counts.
- **Refcount.** Two sequences pooling one reading share its pages; the pages
  outlive the first release and go with the last.
- **Kept objects.** An export imported into an empty substrate exports again
  line for line, but for the header's time and the stamps the import mints
  fresh, and with no tombstone the old one carried; `cluster/` objects are
  not in it; a live key is left as it was; the union keeps a key one member
  deleted after another last wrote it deleted.
- **Gates.** `/fast-test` and `/sweep` against a local, unclustered motherd
  (AGENTS "Test locally, not clustered"), and the MCP latency gate.

### 13.3 Builds

Push to `main` and build and test on BW through the MCP (`repo_job_start`),
not on another machine.

### 13.4 Rollout

The new build's identity key is another model's as far as the cluster is
concerned: its substrate refuses the old one, and the app's model lock closes
it 4412 at the handshake, so it is never a candidate, never certified and
never distributed (`replicated_substrate.md` §3.6). It goes out by that
section's operator procedure, which stops every member, moves each substrate
aside and starts the members on the new build with empty substrates. The
Zen MCP is served by the members it stops, so everything that needs it is
done before step 3.

1. **Build on every member's machine.** Push to `main`, then on each machine
   that runs a member build `cargo build --release -p motherd -p web` at that
   commit through `repo_job_start` pinned to its node, so each holds the new
   `target\release\motherd.exe` before anything stops. Builds only; nothing
   is staged. Every member's build must report the same identity — the
   commit, with no local digest — or the members do not share one build
   count. Compare them before step 3. A build that carries a local digest
   names the uncommitted input behind it: an input that is not motherd's
   (keep's own `keep.yaml`, a machine config) is taken out of motherd's
   build inputs in this change; one that is goes into the commit or is
   reverted by whoever owns it. Nothing stops until all match.
2. **Export.** From every member — the objects replicate, but one not yet
   carried to the others would be lost from a single export — save `GET
   /v1/substrate/zend/objects/export` and `GET
   /v1/substrate/npcd/objects/export` (§12 item 17), and once
   `GET /v1/npcs/export` (the cast), to files outside the data folders. Read
   each back to confirm its line count.
   They hold everything the substrates keep that no model computed:
   accounts, MCP threads, workers and OAuth tokens, journal text, Discord
   links, world state, kept images.
3. **Stop every member.** `curl -X POST http://127.0.0.1:8092/stop/motherd`
   on each machine's keep.
4. **Move the substrates aside.** On each member, rename the zend and npcd
   substrate folders its `motherd.yaml` names to `substrate.pre-hit` beside
   them. Nothing is deleted: restoring them with the old build is the rollback
   (`replicated_substrate.md` §15.1).
5. **Restart the gateways.** Each gateway through its own machine's keep, so
   it forgets its floors (§3.5) and the first member to join on the new build
   sets them, with no certification against members that no longer exist.
6. **Start the new build.** On each member, POST its `motherd.exe` to keep
   (`keep/stage/motherd.exe`, AGENTS "Manual release"); keep trials it on the
   empty substrates, and the first member admitted sets the app's new model
   lock. Confirm every member reaches `state: ready`.
7. **Import, first.** As soon as the members are ready and before any
   reading is asked for, on one member, `POST
   /v1/substrate/zend/objects/import` and `POST
   /v1/substrate/npcd/objects/import` with every member's exports, then `POST /v1/npcs/import`
   with the cast's; restart motherd through keep once so the characters
   spawn. These are admin HTTP calls to the member's own port, so they need
   none of the OAuth tokens they restore — only an admin's identity, from a
   peer the member believes (its gateway's address). Each route reads a body
   of up to `IMPORT_BODY_LIMIT` (8 GiB): one member's zend export alone was
   634 MB. Read each import's answer: every
   key written, none left as it was. A key left as it was means something was
   written between the start and the import (a sign-in, a thread) and is
   reported for a decision.
8. **Ingest.** The priming chain is built at boot, before a member reports
   `ready` (`zend/src/session.rs`, `prime`), so it is already primed when
   step 7 runs and writes nothing the export holds. The code readings come
   back as they are asked for, after the import.
9. **Check.** On each card, a dialogue places its pins, carves a reading by
   round and stays on rung 1; every character is back with its owner; Claude's
   Zen connector answers without signing in again.

Journals, keep's stable binaries, machine configs and the moved-aside
substrates are untouched.

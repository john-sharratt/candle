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
| **Scoring prefill** | A turn's opening is prefilled once to count its hits, rewound, projected from them, and prefilled again against the slot it chose. |
| **Projection** | At each projection boundary, hits move each round's score, the pool is re-chosen by conversation, and the slot by round: locks, pins, floors, then score. |
| **Journals** | What leaves the pool is carried by journal notes. |

## 3. Terms

- **Piece**: one conversation the pool admits or drops whole: a file's reading,
  a folder summary, a guide or journal note, one link of the inherited lineage,
  or the dialogue itself.
- **Round**: one unit the slot places or removes whole, carved from a piece by
  the round grammar (§8.1): its opening request, one tool call with its result,
  or its closing answer.
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
  under ×2. Nothing is carried across. zend's and npcd's substrates are wiped
  on every member when the build rolls out (§14.4); the priming chain is
  primed again, code readings come back as they are asked for, and zend's
  conversations and npcd's NPC memory start empty. Journals are git
  repositories outside the substrate and are untouched.
- **The reserve is one append, not a reply.** The slot's pieces fill
  `L0 − reserve_tokens` (default 16,384). The slot is re-chosen at every
  boundary (§7.2), so it only has to hold what can arrive between two of them:
  one cadence of decoded tokens, or one prefill chunk (§7.4). A reply of any
  length ends inside rung 1, because each boundary sheds what the next stride
  needs.
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
- **The slot is fitted at every projection.** The working set's shares and
  the reserve are checked against `slot_tokens` at parse; the shares and the
  conversation's `window` are caps (§7.2). Each projection measures the system prompt and the
  working set as it emits them and gives the conversation what the slot has
  left after them and the reserve (`slot_room`), so the system prompt needs
  no margin and a slot never leaves the trained window on its own content.
  Only shares and a prompt that alone pass the slot could carry it out, and
  the assembler warns when a placement does.
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
layer.

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
- **Once per boundary, in one launch.** The stride's index queries on the
  sampled layers are kept beside the stride (four heads of 128 F32 per token
  per layer: about 0.4 MB for 64 decoded tokens on three layers), with each
  query's cut. At the boundary they are scored against every unloaded window
  in one launch over the wave's sequences. Decode and prefill launch nothing
  for it.
- **A prefill keeps its last 64 rows.** A prefilled opening or chunk keeps the
  queries of its last 64 positions, as many as a decoded stride, and only
  those enter the shadow scan. They are the queries nearest the reply, whose
  question has been fully read; the opening's other rows still count real hits
  on the slot.
- **Nothing is attended.** The shadow scan writes counters and nothing else.

### 6.4 Counting

Counters are per round per sequence, on the device (`qwen4exp/hits.rs`). A
device map takes each block to its round, loaded or pooled, rebuilt when a
projection declares its rounds (`ManagedBatchedModel::set_hit_pieces`); after
the selection at a sampled layer one batched launch adds each chosen block
into its piece's counter (`qsa_hits.cu`, a span table naming each sequence's
rows, map and counters). Nothing is read inside a forward: the reprojection
reads and zeroes the counts at the boundary (`take_hits`), where it already
stands between forwards. A row inside the budget made no selection and counts
nothing.

A round's **hit density** over a stride is its hits divided by its blocks and
by the queries the stride held. Density, not count, so a long round does not
outrank a short one by size. A piece's density is its rounds' hits over its
blocks. A decoded stride is at most 64 queries, so one
stride's density is noisy; the momentum (§7.1) is what the score reads, and at
a leak of 0.2 it carries about the last five strides.

## 7. Projection

### 7.1 Score

Each round, and each piece, carries a base score and a hit momentum.

- **Base** is what the projection gives it today: provenance's score for a
  retrieved piece and each of its rounds, the seed momentum for a seeded one,
  distance from the cursor for a round of the dialogue (nearer is higher), and
  the lineage's base for a link of the inherited lineage (§8.4).
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
   `pool_tokens` is per card, as the deployment's machine config sets it:
   1,000,000 on the 72 GB card, 512,000 on the 24 GB card, 256,000 on the
   16 GB card. The slot is the same on all three.
2. **Shared index pages.** A piece's sampled-layer index pages are placed on
   the device once per card, whichever sequences pool it, and refcounted: a
   sequence pooling a piece takes a reference, one dropping it gives the
   reference back, and the pages are released when the last reference goes.
   Many pool pieces are file readings every dialogue shares, so pool memory
   grows with the distinct pieces pooled, not with the dialogues open. Hit
   counters stay per sequence: the shared page says nothing about who read it.
   The pages are read-only once placed, so a sequence's selection, the shadow
   scan and the hit map all address them through the span's descriptor tables
   (AGENTS invariant 2b) without a copy.
3. **Pooled K/V.** A piece entering the pool has its K/V brought to the warm
   tier, so a
   round chosen for the slot loads at PCIe speed. Where the warm tier has no
   room for all of it, the pool takes what fits and leaves the rest cold; a
   cold round chosen for the slot loads from the redo log, one boundary later.
   A round placed in the slot is hot.
4. **Slot, by round.** Rounds in this order, until the slot holds
   `L0 − reserve_tokens` less what the turn has already appended:
   1. **Locks**, whole. A read the user asked for is placed entire, whatever
      it costs. If the locks alone pass the slot they are placed anyway, and
      the slot runs past `L0` on the rung its reach needs (§4).
   2. **Pins** (§5).
   3. **Floors** (§8.5): a reading's request and answer when any round of it
      is placed, the dialogue's first request and its recent rounds.
   4. **Everything else by score**, highest first, whichever piece it comes
      from, each round admitted only while its kind is under its cap.

   Each tier takes its room from the tiers below it, so as locks grow,
   provenance's rounds and the lower-scored history of the dialogue shrink
   with them, never the locks themselves.

   **The working set's shares are caps.** They bound how much of each kind the
   slot holds from the last tier, and never reserve room:

   | Cap | Bounds |
   |---|---|
   | `budget` (64,000) | every working-set round in the slot together: readings, folder summaries, guides |
   | `folder` (10,000) | folder-summary rounds |
   | `guidance` (16,000) | guide and journal-note rounds |
   | `max_file` (48,000) | one reading's rounds, its request and answer included |
   | `window` (96,000) | the dialogue's own rounds, its floors included |

   A round whose kind is at its cap is skipped and the fill goes on to the
   next by score, so a capped kind leaves its room to the others. Floors count
   toward their kind's cap but are never skipped for it. Locks are bounded by
   `lock_tokens` alone. The pool has no caps but `pool_tokens`.
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

### 7.4 The scoring prefill

A turn's opening projection runs before the turn is prefilled, so without
more it would be chosen from the previous turn's hits and none of the new
question's. The opening is therefore prefilled twice:

1. **Snapshot.** The sequence's recurrent state and its index's open rows are
   snapshot through the rewind buffers speculative decoding already uses
   (`IndexCache::snapshot` / `restore`, `Qwen4ExpBatched::restore_recurrent`).
2. **Score.** The opening — the user's turn, or a tool result — is prefilled
   against the current slot. Its prefill rows are selected (§6.2), so the
   sampled layers count real hits and keep their queries for the shadow scan,
   exactly as decoded queries do.
3. **Rewind and project.** The snapshot is restored, the shadow scan runs, and
   the projection re-chooses the slot from the opening's hits.
4. **Prefill.** The opening is prefilled again against the slot it chose. Its
   K/V is computed by layers that attended what it asked for, and decoding
   starts from there.

An opening longer than `reserve_tokens` is cut into chunks of that length with
a boundary between them; each chunk is scored, rewound and prefilled as
above, so a long tool result moves the slot as it arrives and the slot never
passes `L0`. A user's turn is usually a few hundred tokens, so its second
prefill costs little against the reply it opens.

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

Tool rounds cross turn records. The turn store keeps a turn as a user record
and an assistant record (`code_read::couple_turn`). A round's call is the tail
of one assistant record and its result is the head of the next user record,
so a round spans two records. A round is carved from halves of records, not
whole records, and each half carries its own index page.

- **Sealed per half.** Every turn record is sealed as two halves, each with
  K/V and index rows of its own. A half placed alone
  (`inject_sealed_turn_half`) carries its index page, so the indexer scores it
  and the hit map counts it as a block range of its round. A half placed
  without an index page is not a valid placement: the selection could not
  rank it, and it would be attended only through the query's tail. The
  substrate starts empty under the new key (§4), so every record is sealed
  per half from its first write.
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

The dialogue is carved by the same grammar. One exchange is the user's turn,
its tool rounds and the assistant's answer to it, never the user's turn
without its answer. When the dialogue must shrink, its tool rounds go first,
oldest first, and the user's question and the assistant's answer stay. What
remains reads as a complete exchange. An exchange leaves whole once it has no
rounds left to shed.

### 8.4 The inherited lineage

Every dialogue descends from the priming chain (`branch_ingest/prime.rs`): a
repository's root listing, then its README, ARCHITECTURE, AGENTS and CLAUDE,
about 40K tokens and capped by `Substrate::INHERITED_CHAIN_TOKEN_CAP`. Today
the lineage was inviolate in a dialogue. Here each turn of it is scored like the
conversation's history from a base of its own — `lineage_base` on the
conversation group's selection, default one full-strength hit (1,000,
`LINEAGE_BASE`) — and competes with the history in the window's budget pass,
lowest first (`select_conversation`). It holds its seat against everything
below its base and gives way to a turn that outscores it: one the indexer keeps
reading, whose score its hits have lifted (§7.1). Seated, the lineage keeps
its order, after the system prompt and before everything else.

### 8.5 Floors

Always in the pool and the slot, whatever their score, in the order of §7.2:

- reads the user asked for, which the working set locks, placed whole. Locks
  come first; when they pass the slot the slot runs past `L0` (§4), and
  everything below them gives up its room. `lock_tokens` still bounds what
  can be locked: a lock past it is refused and the read runs for real
  (`Refusal::Full`), its result in the dialogue's own rounds.
- the pinned ranges;
- each placed piece's request and answer (§8.1);
- the dialogue's first user turn;
- the dialogue's most recent exchanges (`recent`, 16).

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
| `rope: min_yarn_factor` removed with its schema field, error and identity argument; the substrate wiped under the new key | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{schema,error,builder}.rs`, `candle-conversation/src/models/{builder.rs,identity/rope_key.rs}` |
| `hits:` block; `pin: true` on system-prompt items and `pin_tokens`; `slot_tokens`, `reserve_tokens`, `pool_tokens`, `lock_tokens`; `lineage_base` on the conversation rule; the `working_set` shares and `window` as caps on the slot fill | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{yaml,schema}.rs` |
| Pinned ranges in place of the prompt span; forced in every row; one window over the slot | `candle-transformers/src/models/qwen4exp/{indexer,qsa_select,wave}.rs`, `candle-kernels/src/simple/qsa_topk.cu` |
| Sampled layers from percentiles | `candle-transformers/src/models/selection_strata.rs` beside `StrataTokens` |
| Index rows resident without K/V; the stride's kept queries and cuts; the shadow scan and its windows | `candle-transformers/src/models/qwen4exp/{paged_index,place,indexer}.rs`, the paged scorer |
| The block-to-piece table, per-piece counters and the mapped summary | `candle-transformers/src/models/qwen4exp/indexer.rs`, read in `candle-conversation/src/scheduler/` |
| The scoring prefill: snapshot, score, rewind, project, prefill; chunked openings | `candle-conversation/src/scheduler/{prefill,decode}.rs`, `candle-transformers/src/models/qwen4exp/wave.rs` |
| Scores, the pool, the slot, floors, the lineage, exchanges and tool rounds as units | `candle-conversation/src/projection/` (resolver, working set, `select_conversation`), `candle-conversation/src/scheduler/projection_assembler.rs` |
| `ManagedBatchedModel` hooks for pinned ranges, hit summaries and the scoring rewind (no-ops for a model that does not select) | `candle-transformers/src/models/batched_inference.rs` |
| `pool_tokens` per card; pooled K/V warmed, cold where warm runs short | the machine config, `candle-conversation/src/scheduler/projection_assembler.rs`, the tier migration (`persistence/`) |
| Rounds carved by the grammar; slot order locks, pins, floors, score | `candle-conversation/src/projection/` (`select_conversation`, the working set), `candle-conversation/src/scheduler/projection_assembler.rs` |
| Every turn record sealed per half, each half with its index page | `candle-conversation/src/scheduler/` (`inject_sealed_turn_half`), `candle-conversation/src/persistence/` |
| One `file_read` per reply in code-reading ingest, the round limit the file's page count | `zend/src/code_read/mod.rs` |
| Shared, refcounted sampled-layer index pages per card | `candle-transformers/src/models/qwen4exp/{paged_index,place}.rs`, the span's region pool |

## 10. Cost

- **Index memory.** One pooled key is 128 F32 per four tokens per layer. The
  unloaded pool on three sampled layers is at most about 290 MB at a 1M pool
  (about 750K unloaded), about 100 MB at 512K and about 4 MB at 256K, where
  the pool is barely larger than the slot. Pages are shared per card, so
  dialogues pooling the same readings pay for them once; the slot's twelve
  layers are about 400 MB a sequence, as today at that size.
- **Warm K/V.** A pooled piece's K/V is held in the warm tier. Where the warm
  tier is short, cold rounds cost a redo-log read and arrive a boundary late.
- **Shadow scan.** One launch per boundary: up to 64 queries per layer
  against the unloaded keys, about 290 MB a sequence at a 1M pool. A stride
  reads those keys once for all its queries, against once per token if it ran
  in the forward.
- **Ingest.** One read per reply makes a reading take one reply per page:
  more, shorter turns than batched reads, and every turn record sealed in two
  halves.
- **Scoring prefill.** One extra prefill of each opening, and the snapshot
  copy of the recurrent state and the index's open rows.
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

1. **Rotation (§4).** The minimum factor is gone from the schema, the engine
   asks, the model builder, the rung selection (`RungSelect` has no floor) and
   the identity key (`rope_key` names the schedule alone).
2. **One window over the slot (§6.2).** `StrataTokens::DEFAULT` windows are
   the trained window, 262,144 positions, so a slot inside it is one ranking.
3. **Pinned ranges (§5).** Schema names and bound, projection marks, assembler
   ranges, the index cache, the selection definition (`selection_entries`)
   and the selection kernel (`qsa_topk.cu`), which the kernel tests pin
   against the definition.
4. **Sampled layers and counting on the real selection (§6.1, §6.2, §6.4).**
   Percentiles to layers (`hit_layers.rs`), the per-sequence map and counters,
   the batched count after each sampled layer's selection, the read at the
   reprojection.
5. **Score (§7.1).** Hit momentum per working-set member and per placed turn,
   folded at every reprojection, lifting working-set strength and turn scores.
6. **The opening's hits (§7.4, first part).** An opening's prefill rows are
   selected and counted like decoded rows, and the turn's first reprojection
   fires at its first decoded token (`wants_early_reprojection`), so the slot
   the reply decodes against is chosen from the opening's own hits.
7. **The slot's size (§4, §7.2).** `slot_tokens` and `reserve_tokens` in the
   working set; a schema whose working-set shares and reserve pass the slot is
   refused, and every projection fits the conversation to what the slot has
   left after the system prompt and the working set as emitted
   (`slot_room`).
8. **The lineage (§8.4)** is scored from `lineage_base`, ancestors ranked so
   a tight window gives up the oldest first and whole.
9. **Hits across views.** A view carved from a slot counts into the slot's
   counters while the slot keeps the declaration it was carved under; a fork
   that projects for itself counts into counters of its own.

Items 3 (as the `pin: true` flag) and 7 are written and compile-unchecked:
they were changed after the last green run. The rest passed clippy
(`-D warnings`) and the suites of candle-kernels, candle-transformers (GPU
kernels included), candle-conversation and zend on the 3090.

Known costs in what is built, both fixed in §13:

- `take_hits` reads the counters back with a blocking copy at each
  reprojection.
- Per-sequence counters are allocated outside the span's arenas.

## 13. To build

Everything below is one change; it ships when all of it is done. The order is
the order of dependence.

1. **Land the baseline.** Build and test what §12 lists, fix what fails, so
   every later step starts green.
2. **The counters off the blocking path.** `take_hits` reads the counters
   through a mapped pinned summary the device writes after its last sampled
   layer, as MoE routing's summary is read (AGENTS invariant 3): no host call
   waits on it. Counters live in the span's persist tenancy, not in loose
   allocations.
3. **Per-half sealing (§8.1).** Every turn record sealed as two halves, each
   with K/V and index rows; `inject_sealed_turn_half` places the half's index
   page; a half with no page is refused at placement.
4. **One read per reply (§8.2).** Code-reading ingest bounds its replies to
   one `file_read` call; the round limit is the file's page count.
5. **Rounds (§8.1, §8.3).** The projection carves every placed piece and the
   dialogue into rounds by the grammar; request and answer are a piece's
   floor; the dialogue sheds tool rounds oldest first. A round's identity is
   stable across projections (piece id, round index) so its counter and
   momentum follow it.
6. **Shared index pages (§7.2).** Sampled-layer index pages placed once per
   card and refcounted; the pool's rows addressed by descriptor table.
7. **The pool (§7.2).** `pool_tokens` per card from the machine config; pool
   admission by piece; pooled K/V warmed, cold where warm runs short; the hit
   map covers loaded and pooled rounds.
8. **The slot fill (§7.2).** Locks, pins, floors, score; the shares as caps;
   locks past the slot run it past `L0`. This replaces today's budget passes in
   the working set and `select_conversation`; the old passes are removed, not
   kept beside it.
9. **The shadow scan (§6.3).** The stride's kept queries and cuts on the
   sampled layers, the last 64 rows of a prefill; one launch per boundary over
   the wave's sequences; windows seated behind the query; a hit is in its
   window's top-k and above the cut; counted into the same counters.
10. **The scoring prefill (§7.4).** Snapshot, score, rewind, project, prefill
    again; openings longer than `reserve_tokens` chunked with a boundary
    between chunks. This replaces `wants_early_reprojection` (§12 item 6).
11. **Docs.** This document's §12 rewritten as everything built; §13 removed;
    `docs/zend_working_set.md`, `docs/guidance_layer.md`,
    `docs/shared_model_host.md` and `docs/performance.md` updated to match.

## 14. Handover

### 14.1 Rules

AGENTS.md binds every step: no env-var flags, no compatibility paths, no
TODOs or stubs, files changed with the file tools only, tests beside the code
(kernel and codec tests against raw expected bytes), present-tense comments,
the hot-path invariants. No AI attribution in commits.

### 14.2 Tests

Each step lands with its unit tests. Beyond them:

- **Kernels.** The shadow scan against a CPU definition of the same windows,
  top-k and cut, byte for byte on the counters; the hit count with pooled
  blocks in the map.
- **Grammar.** A property test over random pieces and slot sizes: every
  carving reads `U (A U)* A`, never two of one role adjacent, and every placed
  piece holds its request and answer.
- **Fill.** Locks past the slot are placed and the slot leaves rung 1; a kind
  at its cap is skipped and its room goes to the next; floors count toward
  caps and are never skipped.
- **Refcount.** Two sequences pooling one reading share its pages; the pages
  outlive the first release and go with the last.
- **Scoring prefill.** The second prefill's K/V matches a prefill run directly
  against the chosen slot.
- **Gates.** `/fast-test` and `/sweep` against a local, unclustered motherd
  (AGENTS "Test locally, not clustered"), and the MCP latency gate.

### 14.3 Builds

Push to `main` and build and test on BW through the MCP (`repo_job_start`),
not on another machine.

### 14.4 Rollout

The new build's identity key refuses every existing substrate, so the
rollout wipes them on every member at once. A member left with an old
substrate would refuse to open, and a wiped member catching up from an
old-keyed peer would fetch what it then refuses (`replicated_substrate.md`
§7.3), so no member serves on the old substrate once the wipe starts.

1. **Export the cast.** `GET /v1/npcs/export` (admin) saves every
   character's record: who exists, who owns them, where they stand. The cast
   is not NPC memory and survives the wipe; its conversations, beliefs and
   dreams do not.
2. **Stop every member.** `curl -X POST http://127.0.0.1:8092/stop/motherd`
   on each machine's keep.
3. **Wipe.** motherd takes no one-shot flags (`motherd.yaml` cannot carry
   them, `motherd/src/main.rs`). zend's `--wipe-substrate` deletes the
   substrate before it opens (`zend/src/cli.rs`), so on each member run the
   new build's `zend` binary once with it against the zend data folder that
   member's `motherd.yaml` names. npcd's `--wipe-conversations` opens the
   substrate first, which the new key refuses, so npcd's substrate folder is
   removed outright; the cast export holds what must come back. Confirm both
   folders are gone or empty before moving on.
4. **Deploy.** Roll the new build out through the `zend-deploy` skill; start
   each member on it.
5. **Restore the cast.** `POST /v1/npcs/import` (admin) with the export's
   body writes the records verbatim (ids, owners, places); the characters
   spawn on the next start, so restart motherd through keep once after the
   import.
6. **Check.** Prime the chain, then confirm on each card that a dialogue
   places its pins, carves a reading by round, and stays on rung 1, and that
   every character is back with its owner.

Journals, keep's stable binaries and machine configs are untouched.

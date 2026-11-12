# Hit-driven projection

A dialogue's slot is the model's native window, on the rotation it was trained
with, holding the K/V of whatever the model's own indexer keeps asking for. A
pool four times that size stands behind it as index rows only. The indexer
scores the whole pool as it prefills and decodes; what it would have read, had
the K/V been there, is loaded at the next projection, and what it stops reading
leaves.

This replaces the dialogue's fixed budgets (`working_set` and `window` in
`zend/src/prompts/projection.yaml`) and the minimum YaRN factor. It is written
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
| **Rotation** | Every sequence on rung 1, the trained RoPE. The slot never exceeds the trained window, so no rung above it is reached. |
| **Pinned ranges** | Declared ranges of the system prompt that every attention query attends, outside the selection budget, in their declared place. |
| **Pool** | Up to 1,000,000 tokens of candidates: the whole conversation, its inherited lineage, and every piece provenance proposes. Index rows only. |
| **Slot** | The pool's best pieces, K/V loaded, at most the trained window (262,144 positions) less one append's reserve, seated at compact positions and selected as one window. |
| **Hits** | Per piece, how often the indexer selected its blocks (loaded) or would have (unloaded), on a sample of the selecting layers, over every prefilled and decoded query. |
| **Scoring prefill** | A turn's opening is prefilled once to count its hits, rewound, projected from them, and prefilled again against the slot it chose. |
| **Projection** | At each projection boundary, hits move each piece's score, and the slot is re-chosen from the pool by score. |
| **Turns** | Conversation turns are pieces under the same rule, with floors. |
| **Journals** | What leaves the pool is carried by journal notes. |

## 3. Terms

- **Piece**: one unit the projection places or removes whole: a file's reading,
  a folder summary, a guide or journal note, one link of the inherited lineage,
  or one exchange of the conversation (§8).
- **Pool**: the pieces whose index rows are resident for this sequence.
- **Slot**: the pieces of the pool whose K/V is loaded. Only the slot is
  attended.
- **Selecting layers**: the layers that have a QSA index, in layer order
  (twelve on Flash-Next: layers 3, 7, … 47).
- **Sampled layers**: the selecting layers at the configured percentiles
  (§6.1).
- **Hit**: one block chosen by one query at one sampled layer (§6).
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
- **Everything sealed is sealed again.** The minimum factor is part of the
  model's identity key (`models/builder.rs`), so removing it makes every
  sealed block another model's. That is what it is: past the first attention
  layer, every block's K and V were computed by layers that attended under ×2.
  The substrate re-ingests under the new key: the priming chain, the code
  readings and the conversations' own turns.
- **The reserve is one append, not a reply.** The slot's pieces fill
  `L0 − reserve_tokens` (default 16,384). The slot is re-chosen at every
  boundary (§7.2), so it only has to hold what can arrive between two of them:
  one cadence of decoded tokens, or one prefill chunk (§7.4). A reply of any
  length ends inside rung 1, because each boundary sheds what the next stride
  needs.
- **The higher rungs stay for callers that drive a sequence past `L0`
  themselves** (the YaRN gates, a plain-prompt conversation). zend's dialogue
  never reaches them.
- **Re-seating is free.** Keys and index rows are stored un-rotated
  (`progressive_yarn.md` I1) and rotated as they load, so seating a piece at a
  new compact position rewrites nothing.

## 5. Pinned ranges

A pinned range is a span of the system prompt every attention query attends,
at every selecting layer, whatever its score.

- **Declared in the schema.** A `system_prompt` item carries `pin: true`. The
  persona and the journal section are pinned; the tool catalog is not, and is
  selected by score as it is today. Any item may be pinned.
- **Pinned in place.** A pin marks where an item already is; it never moves it.
  System-prompt sections are sealed after the sections above them, so the slot
  seats the system prompt in its declared order and the pinned ranges are
  wherever their items fall.
- **Outside the budget.** Pinned blocks are attended in addition to the
  selection's `top_k + ratio − 1` cells, as the forced recent span is
  (`Recent::Forced`, `qsa_select.rs`). They never enter the ranking, so they
  take no candidate's seat.
- **Carried as ranges, not one prefix.** The prompt span the engine holds today
  is a single leading length (`IndexCache::set_prompt_end`). It becomes a short
  list of `[start, end)` position ranges, declared after each projection walk
  from the pinned items' places (`prompt_end_of` in
  `candle-conversation/src/scheduler/projection_assembler.rs`).
- **Bounded by `pin_tokens`.** The journal section grows by a line per
  stretch of the conversation, so the pinned total is not known at load.
  `pin_tokens` (default 1,024) bounds it: the persona and the journal's
  introduction are pinned whole, and the journal's entries are pinned newest
  first until the bound is reached; older entries stay in the slot, selected
  by score. At load, `pin_tokens` in blocks is added to the bound the strata
  are checked against (`max_entries_for`), and a schema whose bound would pass
  the kernel's union buffer is refused there.
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

The slot is one window. It never exceeds `L0`, the length the indexer was
trained to rank, so the strata over a dialogue's slot are `Strata::WHOLE`: one
ranking over every candidate, the checkpoint's selection exactly, plus the
pinned ranges forced and the query's tail. Nothing without K/V is ever named
in a selection. The stratified windows (`StrataTokens::DEFAULT`) remain for a
sequence that runs past `L0` outside zend's dialogue.

Prefill rows and decode rows are selected alike: the wave's selection table
holds one row per query, decode slots first, then prefill rows, and every row
ranks the history it can see.

At a sampled layer, each selected block of a loaded piece is one hit for that
piece. The lowest score the selection kept is that query's **cut** at that
layer.

### 6.3 The shadow scan

At a sampled layer, the stride's queries are scored against the unloaded
pieces' index rows.

- **In windows ending at the query.** The unloaded rows are cut into windows
  in pool order, each as long as the query's own position `p`. Each window is
  seated directly behind the query: its rows rotate at positions `0 .. p`,
  through the scorer's per-page signed offset, so the query is the one the
  real selection used, rotated where it already is. The indexer ranks a
  context of the length the slot gives it, never longer than it was trained
  on, and its scores are on the scale of the real cut.
- **A hit is a block that would have been read.** Each window ranks its blocks
  as the real selection ranks the slot's: the checkpoint's budget, one
  window. An unloaded block is one hit for its piece when it is in its
  window's selection and scores above the query's cut. The first condition
  keeps it to what the indexer would choose from a context that size; the
  second keeps a window of nothing relevant from handing out a full budget of
  hits.
- **Once per stride, in one launch.** The stride's index queries on the
  sampled layers are kept beside the stride (four heads of 128 F32 per token
  per layer: about 0.4 MB for 64 decoded tokens on three layers), with each
  query's cut. At the boundary they are scored against every unloaded window
  in one launch over the wave's sequences. Decode and prefill launch nothing
  for it.
- **Nothing is attended.** The shadow scan writes counters and nothing else.

### 6.4 Counting

Counters are per piece per sequence, on the device. A device table maps each
block of the pool to its piece, rebuilt when the pool changes; the selection
kernel at a sampled layer and the shadow scan add into the counters through
it. The host reads them through a mapped summary at the boundary, off the
forward path, as the MoE routing summary is read (`AGENTS.md`, hot-path
invariant 3). No launch waits on the host.

A piece's **hit density** over a stride is its hits divided by its blocks and
by the queries the stride held. Density, not count, so a long file does not
outrank a short turn by size. A decoded stride is at most 64 queries, so one
stride's density is noisy; the momentum (§7.1) is what the score reads, and at
a leak of 0.2 it carries about the last five strides.

## 7. Projection

### 7.1 Score

Each piece carries a base score and a hit momentum.

- **Base** is what the projection gives it today: provenance's score for a
  retrieved piece, the seed momentum for a seeded one, distance from the cursor
  for a turn of the conversation (nearer is higher), and the lineage's base
  for a link of the inherited lineage (§8.2).
- **Hit momentum** `h` follows the stride's density `d`:
  `h ← (1 − β) · h + d`, with `β` the working set's leak (0.2).
- **Score** is `base · (1 + gain · h)`. A piece the indexer keeps reading holds
  its score against a falling base; one it never reads falls with it.

`hits.gain` and `hits.beta` are schema fields beside `hits.layers`.

### 7.2 Choosing the pool and the slot

At each boundary:

1. **Pool.** The floors (§8.3), then pieces by score, until the pool holds
   `pool_tokens` (1,000,000). A piece entering the pool has its sampled
   layers' index rows placed.
2. **Slot.** The floors, then pool pieces by score, until the slot holds
   `L0 − reserve_tokens` less what the turn has already appended. A piece
   entering the slot has its K/V and its remaining layers' index rows brought
   to the device; one leaving keeps its sampled rows.
3. **Seating.** Slot pieces are seated in pool order at compact positions: the
   system prompt in its declared order, then the lineage, then the pieces and
   the exchanges as the projection orders them today.

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

### 8.1 Turns are pieces

One exchange is one piece: the user's turn and the assistant's reply to it,
never one without the other. Within an exchange, a tool round is a unit: a call
and its result are placed or removed together.

When an exchange must shrink before it leaves, its tool rounds go first,
oldest first, and the user's question and the assistant's final answer stay.
What remains reads as a complete exchange.

### 8.2 The inherited lineage

Every dialogue descends from the priming chain (`branch_ingest/prime.rs`): a
repository's root listing, then its README, ARCHITECTURE, AGENTS and CLAUDE,
about 40K tokens and capped by `Substrate::INHERITED_CHAIN_TOKEN_CAP`. Today
the lineage is inviolate in a dialogue (`select_conversation`, reason `seed`).
Here each link of it is a piece in the pool with a base score of its own,
`lineage_base` in the schema, set above a seed's; it holds its seat while the
indexer reads it and gives way like anything else when it does not. Seated, the
lineage keeps its order, after the system prompt and before everything else.

A code-reading conversation's own group is not a dialogue's slot, and stays as
it is: selected whole by `target_is_ingest_self`.

### 8.3 Floors

Always in the pool and the slot, whatever their score:

- the pinned ranges;
- the conversation's first user turn;
- the most recent exchanges (`recent`, 16);
- reads the user asked for, which the working set locks today, up to
  `lock_tokens` (default 64,000, inside the slot). A lock past the bound
  releases the oldest lock, which stays in the pool and is scored like any
  other piece.

Everything else in the conversation is scored like any other piece. When
provenance scores are low the conversation fills the slot; when they are high
the oldest unread exchanges give way.

### 8.4 Journals

An exchange that has left the pool is no longer attended and cannot be scored.
What it decided is carried by the journal, whose notes are pieces provenance
proposes like any other. Notes are written as notes about the work, never in
the assistant's first person: a compressed turn written as if the assistant
had said it is read back as something it said
(`projection.yaml`, the dormant `summary:` block).

## 9. What changes where

| Piece | File |
|---|---|
| `rope: min_yarn_factor` removed with its schema field, error and identity argument; the substrate re-ingested under the new key | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{schema,error,builder}.rs`, `candle-conversation/src/models/{builder.rs,identity/rope_key.rs}` |
| `hits:` block; `pin:` on system-prompt items; `pin_tokens`, `pool_tokens`, `reserve_tokens`, `lock_tokens`, `lineage_base`; the `working_set` budgets and `window` replaced by the slot and pool | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{yaml,schema}.rs` |
| Pinned ranges in place of the prompt span; forced in every row; one window over the slot | `candle-transformers/src/models/qwen4exp/{indexer,qsa_select,wave}.rs`, `candle-kernels/src/simple/qsa_topk.cu` |
| Sampled layers from percentiles | `candle-transformers/src/models/selection_strata.rs` beside `StrataTokens` |
| Index rows resident without K/V; the stride's kept queries and cuts; the shadow scan and its windows | `candle-transformers/src/models/qwen4exp/{paged_index,place,indexer}.rs`, the paged scorer |
| The block-to-piece table, per-piece counters and the mapped summary | `candle-transformers/src/models/qwen4exp/indexer.rs`, read in `candle-conversation/src/scheduler/` |
| The scoring prefill: snapshot, score, rewind, project, prefill; chunked openings | `candle-conversation/src/scheduler/{prefill,decode}.rs`, `candle-transformers/src/models/qwen4exp/wave.rs` |
| Scores, the pool, the slot, floors, the lineage, exchanges and tool rounds as units | `candle-conversation/src/projection/` (resolver, working set, `select_conversation`), `candle-conversation/src/scheduler/projection_assembler.rs` |
| `ManagedBatchedModel` hooks for pinned ranges, hit summaries and the scoring rewind (no-ops for a model that does not select) | `candle-transformers/src/models/batched_inference.rs` |

## 10. Cost

- **Index memory.** One pooled key is 128 F32 per four tokens per layer. The
  unloaded pool (about 750K tokens) on three sampled layers is about 290 MB a
  sequence; the slot's twelve layers are about 400 MB, as today at that size.
- **Shadow scan.** One launch per boundary: the stride's queries against about
  290 MB of unloaded keys a sequence. A decoded stride reads those keys once
  for up to 64 queries, against once per token if it ran in the forward.
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

## 12. Open points

1. **Shared index pages.** Many pool pieces are file readings every dialogue
   shares. Whether their placed index rows can be shared between sequences
   decides the pool's memory with several dialogues open.
2. **The pool on a small card.** 1,000,000 tokens of index is sized for the
   72 GB card. `pool_tokens` is per deployment; the 16 GB and 24 GB cards take
   a smaller pool and the same slot.

# Hit-driven projection

A dialogue's slot is the model's native window, on the rotation it was trained
with, holding the K/V of whatever the model's own indexer keeps asking for. A
pool four times that size stands behind it as index rows only. The indexer
scores the whole pool while it decodes; what it would have read, had the K/V
been there, is loaded at the next projection, and what it stops reading leaves.

This replaces the dialogue's fixed budgets (`working_set` and `window` in
`zend/src/prompts/projection.yaml`) and the minimum YaRN factor. It is written
for Qwen3.8-Flash-Next and every model whose attention selects; a model whose
attention reads every position takes none of it.

## 1. Why

The findings this answers are in
`docs/results/model_not_seeing_context_2026-11-12.md`,
`docs/results/theorem_depth_tests_2026-11-10.md` and
`docs/qsa_stratified_selection.md`. In short:

- **Nothing is guaranteed a seat.** A query attends the 2,051 cells its indexer
  ranks highest, plus its own partial block. The system prompt is attended only
  when its blocks rank, and at 230K tokens they get 4–16 of their 69 blocks.
- **Production runs off the trained rotation.** `rope: min_yarn_factor: 2` puts
  every sequence on the ×2 rung, though the checkpoint was trained to 262,144
  positions at ×1 and its vendor advises a YaRN factor only when a context
  needs one.
- **The slot is larger than the trained window.** The dialogue's budgets sum to
  about 428K tokens, so the selection chooses among far more than the indexer
  was trained to choose among.
- **Relevance is decided before the model reads.** Provenance fills the slot
  from its own scores. What the model's indexer then reads, and what it would
  have read, is not fed back.

## 2. The design in one page

| Part | What it is |
|---|---|
| **Rotation** | Every sequence on rung 1, the trained RoPE. The slot never exceeds the trained window, so no rung above it is reached. |
| **Pinned ranges** | Declared ranges of the system prompt that every attention query attends, outside the selection budget. |
| **Pool** | Up to 1,000,000 tokens of candidates: the whole conversation, and every piece provenance proposes. Index rows only. |
| **Slot** | The pool's best pieces, K/V loaded, at most the trained window (262,144 positions) less the reply's reserve, seated at compact positions. |
| **Hits** | Per piece, how often the indexer selected its blocks (loaded) or would have (unloaded), on a sample of the selecting layers. |
| **Projection** | At each projection boundary, hits move each piece's score, and the slot is re-chosen from the pool by score. |
| **Turns** | Conversation turns are pieces under the same rule, with floors. |
| **Journals** | What leaves the pool is carried by journal notes. |

## 3. Terms

- **Piece**: one unit the projection places or removes whole: a file's reading,
  a folder summary, a guide or journal note, or one exchange of the conversation
  (§8).
- **Pool**: the pieces whose index rows are resident for this sequence.
- **Slot**: the pieces of the pool whose K/V is loaded. Only the slot is
  attended.
- **Selecting layers**: the layers that have a QSA index, in layer order
  (twelve on Flash-Next: layers 3, 7, … 47).
- **Sampled layers**: the selecting layers at the configured percentiles
  (§6.1).
- **Hit**: one block chosen by one query at one sampled layer (§6).

## 4. Rotation

The slot holds at most `L0` positions, the schedule's trained window
(`qwen4exp::rope::FLASH_NEXT_L0`, 262,144). A sequence's rung is a function of
its reach (`docs/progressive_yarn.md` §8), so a slot inside `L0` is on rung 1
at every wave: the frequencies and the temperature the checkpoint was trained
with.

- `rope: min_yarn_factor` is removed from `projection.yaml`. The schema field
  stays for a deployment that wants a floor; zend sets none.
- The reply's reserve is taken out of the slot, not added to it: the slot's
  pieces fill `L0 − max_response_tokens`, so a full reply still ends inside
  rung 1.
- The schedule's higher rungs remain for callers that drive a sequence past
  `L0` themselves (the YaRN gates, a plain-prompt conversation). zend's
  dialogue never reaches them.
- Keys and index rows are stored un-rotated (`progressive_yarn.md` I1), so
  re-seating a piece at a new compact position costs nothing.

## 5. Pinned ranges

A pinned range is a span of the system prompt every attention query attends,
at every selecting layer, whatever its score.

- **Declared in the schema.** A `system_prompt` item carries `pin: true`. The
  persona and the journal section are pinned; the tool catalog is not, and is
  selected by score as it is today. Any item may be pinned.
- **Outside the budget.** Pinned blocks are attended in addition to the
  selection's `top_k + ratio − 1` cells, as the forced recent span is
  (`Recent::Forced`, `qsa_select.rs`). They never enter a window's ranking, so
  they take no candidate's seat.
- **Carried as ranges, not one prefix.** The prompt span the engine holds today
  is a single leading length (`IndexCache::set_prompt_end`). It becomes a short
  list of `[start, end)` position ranges, declared after each projection walk
  from the pinned items' places (`prompt_end_of` in
  `candle-conversation/src/scheduler/projection_assembler.rs`).
- **Checked at load.** The pinned total, in blocks, is added to the bound the
  strata are checked against (`max_entries_for`), and a schema whose pinned
  ranges would pass the kernel's union buffer is refused there.
- **Cost.** Every pinned cell is read by every query at every selecting layer.
  The persona and journal are about 500 tokens, a quarter of the checkpoint's
  own read. Forcing the 8K recent span cost +93% decode and confused reasoning
  after a tool call (`qsa_stratified_selection.md`), which is why the catalog
  is not pinned by default.

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

The slot is selected exactly as the engine selects today: the strata over the
slot's candidates, the pinned ranges forced, the query's tail. Nothing without
K/V is ever named in a selection.

At a sampled layer, each selected block of a loaded piece is one hit for that
piece.

### 6.3 The shadow scan

At a sampled layer, the same queries are scored against the unloaded pieces'
index rows.

- **In windows of `L0`.** The unloaded rows are cut into windows of `L0`
  positions in pool order. Each window is seated directly behind the query:
  its rows rotate at positions `0 .. L0` and the query at the position just
  past them, through the scorer's per-page signed offset. The indexer is then
  ranking a context of the length it was trained on.
- **A hit is a block that would have been read.** The real selection's cut
  score at this layer for this query is the lowest score it kept. An unloaded
  block whose score is above that cut is one hit for its piece.
- **Nothing is attended.** The shadow scan writes counters and nothing else.

### 6.4 Counting

Counters are per piece per sequence, on the device. The selection and shadow
kernels add into them; the host reads them through a mapped summary at the
projection boundary, off the forward path, as the MoE routing summary is read
(`AGENTS.md`, hot-path invariant 3). No launch waits on the host.

A piece's **hit density** over a stride is its hits divided by its blocks and
by the queries the stride decoded. Density, not count, so a long file does not
outrank a short turn by size.

## 7. Projection

### 7.1 Score

Each piece carries a base score and a hit momentum.

- **Base** is what the projection gives it today: provenance's score for a
  retrieved piece, the seed momentum for a seeded one, distance from the cursor
  for a turn of the conversation (nearer is higher).
- **Hit momentum** `h` follows the stride's density `d`:
  `h ← (1 − β) · h + d`, with `β` the working set's leak (0.2).
- **Score** is `base · (1 + gain · h)`. A piece the indexer keeps reading holds
  its score against a falling base; one it never reads falls with it.

`hits.gain` and `hits.beta` are schema fields beside `hits.layers`.

### 7.2 Choosing the pool and the slot

At each projection boundary:

1. **Pool.** The floors (§8.2), then pieces by score, until the pool holds
   `pool_tokens` (1,000,000). A piece entering the pool has its sampled
   layers' index rows placed.
2. **Slot.** The floors, then pool pieces by score, until the slot holds
   `L0 − max_response_tokens`. A piece entering the slot has its K/V and its
   remaining layers' index rows brought to the device; one leaving keeps its
   sampled rows.
3. **Seating.** Slot pieces are seated in pool order at compact positions, the
   pinned ranges first.

The slot's membership changes only here. Between boundaries its positions are
fixed, so a reply never has its context renumbered beneath it.

### 7.3 What provenance does now

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

### 8.2 Floors

Always in the pool and the slot, whatever their score:

- the pinned ranges;
- the conversation's first user turn;
- the most recent exchanges (`recent`, 16);
- reads the user asked for, which the working set locks today.

Everything else in the conversation is scored like any other piece. When
provenance scores are low the conversation fills the slot; when they are high
the oldest unread exchanges give way.

### 8.3 Journals

An exchange that has left the pool is no longer attended and cannot be scored.
What it decided is carried by the journal, whose notes are pieces provenance
proposes like any other. Notes are written as notes about the work, never in
the assistant's first person: a compressed turn written as if the assistant
had said it is read back as something it said
(`projection.yaml`, the dormant `summary:` block).

## 9. What changes where

| Piece | File |
|---|---|
| `rope: min_yarn_factor` removed; `hits:` block; `pin:` on system-prompt items; `pool_tokens`; the `working_set` budgets and `window` replaced by the slot and pool | `zend/src/prompts/projection.yaml`, `candle-conversation/src/projection/{yaml,schema}.rs` |
| Pinned ranges in place of the prompt span; forced in every row | `candle-transformers/src/models/qwen4exp/{indexer,qsa_select,wave}.rs`, `candle-kernels/src/simple/qsa_topk.cu` |
| Sampled layers from percentiles | `candle-transformers/src/models/selection_strata.rs` beside `StrataTokens` |
| Index rows resident without K/V; the shadow scan and its windows | `candle-transformers/src/models/qwen4exp/{paged_index,place,indexer}.rs`, the paged scorer |
| Per-piece counters and the mapped summary | `candle-transformers/src/models/qwen4exp/indexer.rs`, read in `candle-conversation/src/scheduler/` |
| Scores, the pool, the slot, floors, exchanges and tool rounds as units | `candle-conversation/src/projection/` (resolver, working set), `candle-conversation/src/scheduler/projection_assembler.rs` |
| `ManagedBatchedModel` hooks for pinned ranges and hit summaries (no-ops for a model that does not select) | `candle-transformers/src/models/batched_inference.rs` |

## 10. Cost

- **Index memory.** One pooled key is 128 F32 per four tokens per layer. The
  unloaded pool (about 750K tokens) on three sampled layers is about 290 MB a
  sequence; the slot's twelve layers are about 400 MB, as today at that size.
- **Shadow scan.** About 290 MB of key reads per decoded token on three layers.
  Where that is too much for a wave of many sequences, the stride's queries
  are kept and scored against the unloaded rows in one launch at the
  projection boundary; the counters are the same either way.
- **K/V.** At most the trained window, against about 428K of budget today.
- **Decode.** The real selection is unchanged. The pinned ranges add their
  cells to every query's read.

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

1. **Locked reads.** §8.2 keeps the reads a user asked for as a floor. If they
   are instead meant to be the first thing given up, they leave the floors and
   take the lowest base score.
2. **Re-keying.** The minimum factor is part of the model's RoPE key
   (`candle-conversation/src/models/identity/rope_key.rs`). Removing it changes
   the key; what that makes zend read again is to be confirmed before the
   schema changes.
3. **Shared index pages.** Many pool pieces are file readings every dialogue
   shares. Whether their placed index rows can be shared between sequences
   decides the pool's memory with several dialogues open.
4. **The pool on a small card.** 1,000,000 tokens of index is sized for the
   72 GB card. `pool_tokens` is per deployment; the 16 GB and 24 GB cards take
   a smaller pool and the same slot.

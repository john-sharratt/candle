# Thinking-Span Projection

**Status:** implemented, authoritative. §8.1–§8.9 are all landed; the clean-turn
re-prefill (`enqueue_clean_turn_reprefill`) and the `keep_reasoning` tool-call
carve-out are gone.

Two things this document records as *settled by measurement*, not by argument,
because both were reasoned to the wrong answer first:

- **§8.7's coverage checks warn; they do not refuse.** Promoting them to `Err`
  was tried against a live model and killed 120 of 355 ingests and every
  conversation — the path already diverged by a few tokens for reasons of its
  own, so refusing there kills conversations over a fault that is not theirs.
- **§8.2a's unit boundary is taken by `Scheduler::begin_unit`, at the K/V
  anchor.** Three other placements were tried and are wrong; each is recorded
  there with why it looked right.

---

## 1. The rule

> **A turn seals with everything it decoded. At projection, the most recent
> turn is injected whole; every older turn is injected with its
> `<think>…</think>` span windowed out of the K/V.**

That is the entire design. It is positional — a function of a turn's distance
from the end of the conversation — and nothing else. No mode, no flag, no tool
state, no conversation type.

A turn's reasoning is therefore visible in exactly one subsequent projection
and never again.

---

## 2. What problem it solves

Three problems, which until now had three different mechanisms:

1. **A turn must not attend its own reasoning.** Models are trained with prior
   turns' thinking stripped; feeding it back is off-distribution and inflates
   context. Previously solved by re-prefilling the turn reasoning-free before
   sealing it.

2. **A tool call needs its reasoning to survive the round trip.** The model
   thinks, emits a call, receives a result in a *new turn*, and answers. The
   reasoning that motivated the call must still be attendable when the result
   arrives. Previously solved by `keep_reasoning`, a flag threaded from
   `ToolMode` through `TurnOptions` → `SubmitTurn` → `PrefillWork` →
   `DecodeState` to suppress the re-prefill.

3. **Sealed bytes are the long-term record.** Whatever is sealed is what every
   future projection, summary, and fork sees.

The rule solves all three at once. (2) falls out for free: the turn carrying
the tool call *is* the previous turn when the result is decoded, so its
reasoning is projected without anything having to know a tool was involved.

---

## 3. Why it costs nothing

**The slot is stateless between turns.** After a seal, the scheduler truncates
the slot to zero and the next turn's `apply_projection` rebuilds it from the
substrate:

> *"Drop the slot's chunks now the residence owns them (the next projection
> rebuilds from the substrate)"* — `scheduler/mod.rs:5353`
>
> *"The next turn's `apply_projection` rebuilds the slot from substrate anyway,
> so holding onto these `Arc<ChunkGid>`s between turns just pins arena slots
> that nothing reads."* — `scheduler/mod.rs:6736`

So the rebuild already happens, every turn, unconditionally. This design does
not add an operation anywhere. It changes *which windows* an existing rebuild
injects. There is no live slot to repair, no rolling strip, no deferred
cleanup, and no drift between a resident slot and its record.

This is the load-bearing fact. Every earlier design in this area — seal-twice,
tombstone-and-reseal, park-and-defer, clean re-prefill — was paying to keep a
persistent slot honest. The slot is not persistent.

---

## 4. The mechanisms it rests on

Every one already exists. Nothing here is new machinery; the design is a new
*composition* of primitives the engine already ships.

### 4.1 A sealed chunk is a window, not a chunk

```rust
pub struct SealedChunk {
    pub gids: HeadGids,
    /// Start position within the physical chunk where this window begins.
    pub offset: u16,
    /// Number of valid tokens in this window (from `offset`).
    pub token_count: u16,
    ...
}
```
— `candle-nn/src/kv_cache/chunked/types.rs`

`offset`/`token_count` make a sealed chunk an arbitrary sub-range of a physical
chunk. Two `SealedChunk`s may therefore reference the **same** physical chunk
with disjoint windows, sharing one refcounted `HeadGids`. When a thinking block
begins and ends inside a single chunk, the pre-span and post-span windows are
both views of that chunk — no copy, no split allocation, no boundary alignment.

**Consequence:** thinking spans need no alignment to chunk boundaries. A block
may be projected partially, and the same block may be projected twice.

### 4.2 `window_sealed_tokens` already does the windowing

`candle-conversation/src/conversation.rs:96` takes a per-layer sealing and a
token range `[start, end)` and returns a zero-copy view: whole chunks cloned
as-is, boundary chunks cloned with `offset += overlap_start - chunk_start` and
`token_count = overlap_len`, chunks outside the range dropped.

It is already in production for `turn_user_sealed_half`
(`substrate.rs:4177`), which windows a turn down to its user-message body for
the compression assembler. This design calls it twice per turn and
concatenates the two chunk lists.

### 4.3 Positions recompact themselves

The reason a hole is safe, and the fact that decides the whole design:

> *"K bytes in the cache are stored **un-rotated** by the prefill kernel. RoPE
> rotation is applied at the latest responsible moment — inside the attention
> kernel at read time, against `slice_rope(...)` from the per-chunk `ChunkMeta`
> buffer, which is rebuilt from cumulative usage of preceding blocks in the
> current slot's layout each time the decode/prefill metadata is synced.*
>
> *Consequence: a `SealedChunk` (and its underlying K/V bytes) can be injected
> at **any absolute position** in any sequence and the kernel will apply the
> correct RoPE for that new position. No re-rotation, no byte copy, no CoW
> required."* — `candle-nn/src/kv_cache/chunked/types.rs:135`

Because position is derived from cumulative usage of the layout as injected,
omitting a span does not leave a positional gap — it **compacts**. The answer
lands at exactly the position a clean re-prefill would have given it. This is
not an approximation of the re-prefill's positional result; it is the identical
result, reached without a forward.

### 4.4 Injection accepts any chunk list

`inject_arc_sealed` (`scheduler/projection_assembler.rs:1463`) copies the
per-layer chunk vectors and calls `inject_sealed_at_tail`. It asserts the layer
count and nothing about contiguity. A holed sequence needs no change here.

### 4.5 The layout already records the span

```rust
Thinking {
    text: String,
    kv: Option<KvSpan>,   // Some ⇒ REAL (K/V stored); None ⇒ ETHEREAL
},
```
— `candle-conversation/src/turn_layout.rs`

and the module doc states the intent outright:

> *"ethereal `TurnSegment::Thinking` — reasoning prose kept, its K/V dropped"*
>
> *"The `offset` inside a `KvSpan` is redundant with the running length sum …
> but kept **so a sub-range of a turn can be projected directly** — e.g. inject
> only the answer, or window to the user half."*

`build_turn_layout(…, ethereal_thinking: false)` — the path a turn already
takes when it seals with its reasoning — emits exactly this: a real `Thinking`
segment whose `KvSpan` names where the reasoning sits in the sealed grid. The
projection needs nothing more.

### 4.6 The index is a list of pages, and a page carries its own width

The QSA index is coarser than the K/V: one row per `ratio` tokens, keyed off
hidden states and unrecoverable from the K/V. That granularity would be a real
obstacle except for two properties that together make a hole in it expressible.

**A page records how wide its last row is.** A sequence's index is a list of
`IndexPage`s followed by a live tail. Each page carries `last_cells` — the
tokens its final row covers, `1..=ratio` — because a piece ends where its tokens
end:

> *"The flushed block is a summary of fewer than `ratio` tokens and is therefore
> NOT what a continuous run would have produced for that span. That is the
> deliberate trade: **the scorer carries each page's width and derives the
> candidate prefix from it, so a short block is expressible**; a block pooled
> from another turn's tokens is not correctable at all."*
> — `qwen4exp/indexer.rs:236` (`flush_open_block`)

**And the scorer walks those widths rather than dividing by `ratio`:**

> *"Per row, the candidate blocks are those wholly below its tail. With injected
> pages ahead of the live tail this is a walk over their widths rather than a
> division — see `candidates_at`."* — `indexer.rs:577`

**Consequence:** dropping a whole page renumbers everything downstream by
construction, with no rounding in either direction. Provided the reasoning is a
page of its own, its rows are omitted exactly.

That is the one thing this design must add: **the reasoning must be its own
page.** A page is made by closing the live tail, which is the composition of two
methods that already exist —

```rust
/// Close the live tail into a page, so what follows starts a new one.
pub fn close_tail_into_page(&mut self, …) -> Result<()> {
    let cells = self.flush_open_block(…)?.unwrap_or(ratio);
    if self.n_blocks == 0 {
        return Ok(());
    }
    let rows = self.live_rows()?.to_owned_tensor()?;
    self.n_blocks = 0;
    self.n_open = 0;
    self.push_page(IndexPage::new(rows, 0, cells), ratio)
}
```

— `flush_open_block` pooling the carried rows into one short block, and
`push_page` doing the prefix-sum bookkeeping. Note that the reset *satisfies*
`push_page`'s own guard ("injected prefixes must precede anything this sequence
forwarded") rather than bypassing it: after the tail has been lifted out, there
is nothing forwarded left to precede.

#### Why a flush alone is not enough

`flush_open_block` on its own is the obvious move, and it is wrong — quietly.
The live tail's arithmetic is a **uniform division**, in three places:

| | |
|---|---|
| `candidates_at` | `page_row_span() + (limit − span) / ratio` |
| `block_start` | `page_token_span() + (block − rows) · ratio` |
| `indexed_tokens` | `page_token_span() + n_blocks · ratio + n_open` |

A flush advances `n_blocks` by one while consuming fewer than `ratio` tokens. A
short block left *inside* the tail therefore puts every later position on the
wrong block, and makes `indexed_tokens` over-report by `ratio − cells` — for the
rest of the sequence, internally consistent and wrong against the K/V. The
coverage gate (§8.7) would fire on every thinking turn, forever.

This is why every existing caller flushes on a **fork** (`wave.rs:415`,
`wave.rs:519`) or on a throwaway ingest slot, and in the same breath lifts the
rows into a page whose `last_cells` records the short width. **A page is the
only place a ragged width can live**; closing the tail is what puts it there.

---

## 5. Removal inventory

The design's value is as much in what it deletes as in what it adds. A whole
deferred-seal subsystem exists only to make the re-prefill possible, and it
disappears entire. This section is the working list; it names what goes, what
stays, and where the two are easy to confuse.

### 5.1 The deferred-seal subsystem

The re-prefill cannot seal at decode time — it must wait for its own prefill
unit to ride the shared wave — so a turn's seal and its `Done` event are parked
and replayed a wave later. That parking machinery is the bulk of the removal.

| Symbol | Where | Notes |
|---|---|---|
| `enqueue_clean_turn_reprefill` | `scheduler/mod.rs:5142–5258` | Builds the clean grid, stashes the pending seal, enqueues the unit. |
| `complete_turn_reprefill` | `scheduler/mod.rs:5264–5376` | Snapshots the clean K/V, seals, fires the deferred `Done`. |
| `PendingTurnSeal` | `scheduler/mod.rs:1348–1396` | The parked seal: `parent_id`, `seal_block_from`, `seal_pos_from`, `layout`, `token_ids`, tags, `event_tx`, stats, and the private `_sink_rx`. |
| `Scheduler::pending_turn_seals` | `scheduler/mod.rs:2617` | `HashMap<u64, PendingTurnSeal>`; init at 2874. |
| `Scheduler::next_turn_seal_id` | `scheduler/mod.rs:2620` | Monotonic id source; init at 2875. |
| `SealAction::TurnReprefill { pending_id }` | `scheduler/mod.rs:1456–1465` | The enum variant and its doc comment. |
| Its promote-loop arm | `scheduler/prefill.rs:2502–2525` | Error path (surface on caller channel + truncate) and success path. |
| Its unreachable arm | `scheduler/mod.rs:7779–7783` | In `perform_seal_and_write`'s exhaustive match. |
| The go/no-go branch | `scheduler/mod.rs:6525–6603` | The `truncate_sequence_to_blocks` + `seal_writer_boundary` gate, the DISCARD arm that releases the view's recurrent state, and the fall-through to the immediate seal. |

Its telemetry goes with it, and it is spread across two files: `PromoteStep::Reprefill`
and `PROMOTE_REPREFILL_US` (`run.rs:21, 26, 34`), `note_reprefill_split` /
`take_reprefill_split` with `REPREFILL_WRITE_US` / `REPREFILL_TRUNC_US`
(`run.rs:39–57`), the call site (`mod.rs:5368`), and three fields of the
housekeeping log (`mod.rs:2307, 2310, 2311`). `take_promote_split` returns a
triple and becomes a pair, which the compiler will point at.

The DISCARD/MOVE split at `6563`/`6609` collapses: with no re-prefill to
re-advance the state, every turn takes **MOVE** (`move_recurrent`). Nothing is
released, because nothing can be rebuilt.

**One thing in here is not removable, and it has to move first.**
`complete_turn_reprefill` also seals the turn's QSA index page
(`mod.rs:5319`), and that is the **only** call to `seal_positional_range` in the
scheduler — `perform_seal_and_write` does not seal one. Deleting §5.1 before
relocating it would take out the sole producer of a turn's index rows, leaving
every turn sealed as K/V that nothing indexes. §8.3 moves it onto the immediate
path, which is a prerequisite for this section rather than part of it.

### 5.2 The reasoning-stripping token surgery

| Symbol | Where | Notes |
|---|---|---|
| `strip_think_from_tokens_keep_layout` | `scheduler/mod.rs:4769–4788` | Detokenise → strip → retokenise, to build the clean replay tokens. Its only caller is `enqueue_clean_turn_reprefill`, so it goes with it. |

This is the operation the design replaces: a text round trip through the
tokeniser, at seal time, to produce a token stream the model must then be run
over again. The windowing in §4.2 achieves the same visibility outcome with no
tokeniser involvement and no forward.

**But `think_strip::strip_think_blocks_keep_layout` STAYS** — only the scheduler
wrapper above goes. The underlying function has a second caller,
`zend/src/tools.rs:274`, which uses it so that a `<tool_call>` emitted *inside* a
reasoning block is not dispatched. That is a parsing rule about text, unrelated
to K/V, and deleting it would start executing tool calls the model only thought
about. It belongs on §5.5's list.

### 5.3 The `keep_reasoning` carve-out

`keep_reasoning` marks "this turn's result may arrive as a follow-up turn, so do
not strip its reasoning". With nothing stripping, it has no meaning. It is
threaded through five types and set from `ToolMode`, so removal touches the
whole submit path:

- `TurnOptions::keep_reasoning` — `turn.rs:96`
- `SchedulerRequest::SubmitTurn::keep_reasoning` — `scheduler/mod.rs:230`
- `PrefillWork::keep_reasoning` — `scheduler/mod.rs:1547`
- `DecodeState::keep_reasoning` — `scheduler/mod.rs:1144`
- Plumbing (read and pass through) — `scheduler/mod.rs:3049, 3522`;
  `prefill.rs:2709, 2808`; `conversation.rs:1876, 2046, 2122`
- Construction sites passing `false` — `conversation.rs:4227`;
  `engine.rs:1317`; `tree/summarize.rs:255`; `scheduler/mod.rs:4631, 4949, 5252`
- The one site that passes `true` — `zend/src/session.rs:2554`
  (`keep_reasoning: tools_mode != ToolMode::None`)

That last line is the only place the sealing layer has ever known what a tool
is. Deleting it is the point of the design: **after this change, no type
between `TurnOptions` and the substrate carries tool state.**

Its comment is worth reading before deleting it, because it already describes
this design and asserts a property the code does not have: *"a turn that calls
nothing keeps its reasoning for the one turn, which the next turn's projection
then drops anyway."* The projection drops nothing today — `inject_sealed_turn`
injects `turn_sealed_of`, the whole turn, on every projection forever. So a
`keep_reasoning` turn's thinking is permanently attendable, which is further
from the chat template's `last_query_index` than the one-turn rule is, not
closer. The comment is not wrong about the intent; it is a year early. §8.6 is
what makes it true.

### 5.4 The seal-time visibility decision

| Symbol | Where | Notes |
|---|---|---|
| `build_turn_layout`'s `ethereal_thinking` parameter | `scheduler/mod.rs:4982` | Chose between a real and an ethereal `Thinking` segment. Every turn now seals real, so the parameter and its `true` call site (`mod.rs:5189`) go. |

`TurnSegment::Thinking::kv` stays `Option<KvSpan>`. The type still needs to
express an ethereal block — a `/no_think` turn's collapsed `<think></think>`
has no K/V — and the projection reads `Some(span)` as "there is a span to
window out" and `None` as "nothing to do". The option is a genuine hit/miss,
not a feature flag.

### 5.5 What must NOT be removed

Each of these looks like part of the machinery above and is not:

- **`think_strip::strip_think_blocks`** (`scheduler/mod.rs:4762`,
  `tree/summarize.rs:354`) — strips reasoning from *display and summary text*.
  Unrelated to K/V; stays. So does its caller `strip_think_from_tokens`
  (`mod.rs:4754`), which the compression path uses at `mod.rs:4813–4814`.
- **`think_strip::strip_think_blocks_keep_layout`** — §5.2. The scheduler's
  wrapper goes; the function itself is load-bearing in `zend/src/tools.rs`.
- **`think_strip::strip_empty_think_blocks`** (`scheduler/mod.rs:4994`) — drops
  an empty `<think></think>` from display text so a `/no_think` turn does not
  surface one. Stays.
- **`SealAction::CompressionPass` / `CompressionTurn` and
  `pending_compression_seals`** — a *separate* deferred-seal subsystem, for
  summarisation, with near-identical shape. It has its own reason to exist (the
  compressed turn is re-prefilled for role coherence, not to strip reasoning)
  and is untouched. This is the single easiest thing to delete by accident.
- **The `no_think` dial** and the `/no_think` glue in the turn grid. A
  submit-time effort control, not a seal-time one.
- **`seal_writer_boundary`** — used by the section-ingest path too.

### 5.6 Two consequences to settle during removal

**Provenance signatures now cover reasoning tokens.** `PendingTurnSeal`
documents that wide-Q sigs are deliberately *not* carried across the
re-prefill: the seal re-gathers them with `gather_wide_sigs`
(`scheduler/mod.rs:8113`) from the reasoning-free grid "so they match the
sealed K/V". Under this design the sealed grid contains the reasoning, so a
whole-turn gather would index it, and BDP recall could select a block that the
projection then windows out — a retrieval that returns nothing usable.

**Gather sigs over the turn minus its reasoning span.** The sigs describe what
is *retrievable*, and the reasoning is retrievable exactly once, from the
turn immediately following it, by position rather than by recall. Restricting
the gather keeps recall and projection agreeing on what exists.

**The block-index / position reconciliation carries forward.** `seal_pos_from`
exists because "the K/V boundary is a BLOCK index and the index cache is
addressed by POSITION", and a page starting one row early carries a block
belonging to the previous turn — which a projection borrowing both would then
hold twice, at two different positions. The field goes with `PendingTurnSeal`,
and so does the reconciliation it performed: §8.3 seals by **width**
(`seal_positional_tail_span`) rather than from a position, which is the same
formulation the reprojection path already uses and needs no anchor.

The hazard itself does not go away — the same alignment must now hold between a
windowed turn's chunks and its retained pages. Two things make it hold rather
than one: both sides are derived from a single pair of grid indices (§8.1), and
the page boundary IS the window boundary by construction (§8.2, §8.5). §8.7's
coverage gate is what proves it.

### 5.7 Net effect

Dialogue turns, tool turns, summaries and `code_read` scopes seal through one
path. A turn's seal fires at decode time, unconditionally, with no parked state
and no second wave — which also unconditions the coupling that blocked the
earlier park-and-defer design, where the tool loop pairs a call turn to its
follow-up by `SealResult::turn_index` and there was no other window in which
that index was knowable.

---

## 6. Reasoning behind the trade-offs

### 6.1 Why not keep the re-prefill

The re-prefill rebuilds `[user][clean answer][tail]` on the slot and produces
three things: reasoning-free sealed bytes, a reasoning-free live slot, and a
recurrent state that never folded the reasoning.

The second is moot — §3, the slot does not survive the turn. The first is what
this design does for free. The third is the only real product, and §6.2 argues
it is not worth a full turn's prefill per turn.

### 6.2 The recurrent state keeps the reasoning, permanently

`full_attention_interval = 4`, and `kv_layers()` counts only
`LayerKind::Attention` (`qwen4exp/config.rs:146,407`). **Three layers in four
have no K/V at all.** For those, the GDN state is not auxiliary memory beside a
cache — it is the entire memory:

> *"The GDN state is an accumulated sum with no per-token decomposition, and the
> PLE conv history is likewise irrecoverable from the KV."*
> — `qwen4exp/wave.rs:1181` (`carries_recurrent_state`)

Windowing is a metadata edit on an addressable store. The GDN and PLE states are
folds. No projection rule can reach them, and they are carried across turns
(`fork_recurrent`/`move_recurrent`) precisely because they cannot be rebuilt
without a forward.

So "the reasoning is gone" is only ever true of the 1-in-4 attention layers.
State this plainly wherever it matters rather than implying a stronger property.

Accepting it is the right call:

- **The alternative costs a full turn's prefill, every turn** — roughly doubling
  prefill work — to correct a fold that is *gated*, and therefore decaying,
  rather than accumulating without bound.
- **It is already accepted in production.** Tool turns take `keep_reasoning` today,
  which skips the re-prefill entirely and leaves precisely this state.
- **The behaviour being imitated does not have the problem to solve.**
  Interleaved thinking in a pure transformer is a context-assembly rule; there
  is no hidden state to correct. Stripping the K/V is the closest achievable
  analogue, and chasing the fold pursues a fidelity the reference never had.

### 6.3 Why one turn, and not "since the last user query"

The Qwen chat template's `last_query_index` keeps thinking for every message
after the last genuine user query, and a `<tool_response>`-wrapped user message
does not count as one. That preserves reasoning across an entire multi-step tool
loop.

One-turn-back reproduces this **exactly** for the common shape —
`think → call → result → answer` — because the answer turn's predecessor is the
call turn. At three or more tool steps it loses the oldest step's reasoning
while retaining that step's prose and every intermediate result. That is a
graceful degradation, not a failure.

Making it exact would mean the rule reading "keep thinking for turns after the
last non-tool user message" — which puts tool awareness back into the projection
layer to buy fidelity in a case that degrades gracefully anyway. The trade is
not worth the coupling. **This is the design's central simplicity decision and
the one most likely to be revisited under pressure; revisit it only with a
measured failure, never with a hypothetical one.**

#### The second known degradation: a summary in the window

An asynchronous summary turn can be appended between a call turn's seal and its
result turn's submit, and it takes the highest turn index when it does. §8.6's
rule then windows the call turn, and the result decodes without the reasoning
that motivated the call — keeping the `<tool_call>` and the `<tool_response>`,
losing the motivation.

This is not hypothetical, and it is not new to this design: the tool loop
already defends against exactly it, and says so —

> *"Couple the call turn by its OWN sealed index (`resp.seal.turn_index`), never
> by 'the last turn': the async summariser can append a summary turn in this
> window, so the newest index may not be the call turn."*
> — `zend/src/session.rs:2831`

It is recorded rather than fixed, for the reason above. Every repair is a second
clause: "highest index that is not a summary" needs the projection to ask what
kind a turn is, and no reformulation avoids it, because the summary genuinely
*was* appended between the two turns. It also self-limits — a summary only lands
here in a session already under compression pressure, and the loss is one tool
call's motivation. **Do not fix this without a measured failure**; note it here
so the next reader knows it was seen, not missed.

### 6.4 Why not seal twice, or tombstone and reseal

Both were considered and are superseded. Sealing a turn twice (once with
reasoning, once without) doubles the write and the storage, and requires the
projection to choose between two records — a decision point that must be given
tool state to make. Tombstoning the reasoning-bearing seal after the fact
requires a second pass over a sealed turn and makes the record temporarily
wrong. Windowing at read time needs neither: one record, one decision, made
from position alone.

### 6.5 The storage property

Every turn now seals its reasoning K/V, and that K/V is read by exactly one
subsequent projection. Reasoning bytes therefore accumulate in the substrate and
are, after one turn, cold. This is correct and cheap — sealed K/V is compressed
and tiered to NVMe — and it is what makes the one-turn window possible at all.
It is a property of the design, not a leak.

### 6.6 Why the cut is exact, and not rounded to row boundaries

There is a tempting cheaper design that needs no live change at all: keep the
turn's single index page, and at projection time split it on **row** boundaries
— drop the rows lying wholly inside the reasoning, window the K/V to exactly
those rows' token range. Both halves stay expressible as ordinary `IndexPage`s
(page A's rows are all full; page B keeps the original `last_cells`), so there
is no new method, no decode-time hook, and no change to the turn record.

It is rejected because neither rounding is safe, and the unsafety is silent:

- **Round inward** (drop only rows wholly inside the span) and up to `ratio − 1`
  reasoning tokens survive at each edge. At `ratio = 4` that is `<think>` plus a
  couple of tokens, then a jump to the last few, then the answer — which
  compacts into a *fabricated short thought* on every historical turn. It reads
  perfectly and is not what the model said.
- **Round outward** (drop every row the span touches) and up to `ratio − 1`
  tokens on each side go with it. On the left those are the tail of
  `<|im_start|>assistant\n`, so the role marker is corrupted; on the right they
  are the answer's first tokens.

There is no third rounding, and no repair available: **this model cannot
gap-fill** (`can_gap_fill = !carries_recurrent_state`, and the stack is
recurrent), so a projection can inject or omit and can never recompute anything
mid-sequence. Whatever the window drops is simply gone.

That leaves exactness, and exactness requires the boundary to be a page
boundary at the moment the rows are pooled — §4.6. The price is one 8-line
method built from two existing ones. Worth recording rather than re-deriving:
the row-aligned version will look cheaper every time someone reads §8.2.

---

## 7. Principles

These are why the design is small, and the rules that keep it small.

1. **Record everything; decide late.** Sealing is a faithful record of what was
   decoded. Visibility is a projection-time policy. Never destroy information at
   write time to implement a read-time rule — that is what forced the re-prefill,
   the `keep_reasoning` carve-out, and the seal-twice proposal in turn.

2. **The projection is the only authority on what the model sees.** One decision
   point, consulted on every rebuild. If a second place starts deciding
   visibility, they will disagree, and the disagreement will be silent.

3. **Positional, not stateful.** The rule is a function of a turn's index. A
   rule that needs to know *why* a turn exists needs that reason threaded
   through every layer that touches a turn — which is exactly the
   `keep_reasoning` plumbing this design removes.

4. **Know what can be windowed and what cannot.** K/V is an addressable store:
   windowable, at token granularity, for free. The GDN state, the PLE conv
   history, and anything else folded are not. Never write a comment or a commit
   message implying reasoning was removed from the model when it was removed
   from a quarter of its layers.

5. **Both halves of a piece travel together.** K/V and its index page are one
   object. A span injected without its rows is silent below the QSA identity
   threshold and a hard refusal above it — the failure mode that consumed a full
   session. Any code path that windows one must window the other in the same
   step.

6. **No new mechanism where composition will do.** Every primitive here already
   existed and most already had a production caller. The one addition —
   `close_tail_into_page` — introduces no new *kind* of thing: it is a page,
   which the index already has, made by two methods it already has. That is the
   test to apply to any extension: not "is this small?" but "does this add a
   concept, or arrange existing ones?" A page the scorer already walks costs
   nothing to reason about; a fourth kind of index region would cost everything.

### How to keep it simple

The pressure on this design will be to add a condition. Each of these is a
reintroduction of the state the design exists to remove:

- *"Keep thinking for two turns when …"* — the rule stops being positional.
- *"Except for tool turns"* — the tool flag comes back, through every layer.
- *"Seal without thinking when we know we won't need it"* — write-time decisions
  return, and now two records disagree.
- *"Skip the page close when the block is short"* — the index stops being
  page-exact and the coverage gate starts warning instead of asserting.
- *"Round the cut to the nearest row and skip the decode-time work"* — §6.6, the
  one that will look cheapest and is the only one that silently changes what the
  model reads.

If the rule ever needs a second clause, that is a signal to re-derive it, not to
extend it.

---

## 8. Implementation

**8.1–8.4** make a turn's record complete and exact, **8.5–8.8** make the
projection read it, **8.9** deletes §5. §8.10 says which of them can land
separately, which is fewer than the numbering suggests.

### 8.1 One index, recorded as the token goes past — **LANDED**

`</think>` is a single token id, resolved by name at scheduler construction
(matching `Engine::compile_think_steering`, `engine.rs:902`, which gates the same
way). `DecodeState::think_close_at` records the index into the generated run of
the first one decoded, and everything derives from that: the layout's `Thinking`
span (8.4), the index cuts (8.2), and the projection's window (8.5).

**One position, not two.** The span is
`[assistant_content_start, think_close_at + 1)` — the assistant body up to and
including the marker — so the open marker is never needed. That is not merely
cheaper, it is *more* correct: any preamble the model emits before `<think>`
lands inside the reasoning region, which is where it belongs, instead of being
stranded in the answer by a span that starts at the open marker.

This **replaces** the previous derivation, which decoded the assistant text,
string-searched for `<think>`, and re-tokenised the block to measure it. That was
an approximation — its own comment conceded the answer span absorbed the
round-trip remainder — and it was harmless only because the ethereal split
dropped the whole span regardless. The text search survives for the segment's
display prose, which is what it was always right for.

**Every token enters the generated run through `DecodeState::push_generated`,**
because they arrive by four routes and the boundary has to be seen on all of
them. The two spellings say when the token is forwarded relative to the call,
which is what decides whether a cut is possible (§8.2b):

| Route | Where | Enters by |
|---|---|---|
| the sampler's own commit | `decode.rs`, `commit_decoded_tokens` | `commit_generated` |
| the turn's first token | the three `DecodeState` constructions | `commit_generated` |
| a stencil's static run | `decode.rs`, the stencil prefill | `push_forwarded` |
| an assistant prefill prefix | `decode.rs`, the heal path | `push_forwarded` |

The second row is the one that would have been missed. A think-steer tree
**always suppresses the model's own `</think>` and injects the closing tag as a
static run instead** — the comment at that site says so — so recording at the
sampler alone would leave every steered turn with no span at all, which is most
of them. The single funnel is what makes that structural rather than remembered.

**Decoded positions only, which needs no special case for any turn shape.**

The live submit path never bakes a reasoning marker into the prefill: it builds
the assistant head as `user_message ++ user_end ++ assistant_start` and stops
(`conversation.rs:1916`), and `DecodeState::prefill_tokens` says so outright —
*"The think block is NEVER baked here … either way it lands in
`generated_tokens`."* So both markers are always decoded, and the recorded pair
is available wherever there is one to record.

> **Do not reason about the submit path from `dialect.rs`.**
> `Dialect::no_think_block` (`"<think>\n\n</think>\n\n"`, documented as
> "prefilled after `assistant_start`") is a real capability of this family and
> is **not** on the live path: its only reader is `thinking_suppression`, whose
> only caller is `batch_test/utils.rs`. The dialect describes what the family
> supports, not what the daemon sends.

Two turn shapes then fall out with no clause of their own:

- **A prefilled assistant body** (repo_map, a `code_read` scope) decodes
  nothing, so it records no pair, has no real `Thinking` segment, and is
  injected whole. A tool-exchange turn additionally returns from
  `build_turn_layout` before the think split (`mod.rs:5027`), so it is covered
  twice over.
- **A turn whose reasoning is empty** — one decoding a collapsed
  `<think></think>` — records a pair like any other and gets a span of a token
  or two, a one-row page, and the ordinary windowing. Nothing about it is a
  special case; the span is simply small.

A segment's *text* stays independent of its span, which is what keeps this from
disturbing the display: `strip_empty_think_blocks` (§5.5) still keeps an empty
block out of the rendered text while the span describes the K/V those tokens
occupy.

### 8.2 Close the index page at each boundary — **LANDED**

There are **two** kinds of boundary, they are not variations of one thing, and
conflating them cost three wrong placements before the distinction was drawn:

| | where a **unit** begins | where a **region** of a turn begins |
|---|---|---|
| separates | this unit's rows from the prefix ahead of them | the turn's reasoning from its answer |
| decided by | the K/V anchor the seal will read back | the token being committed |
| known to | the scheduler, at three points | `DecodeState`, at every commit |
| funnel | `Scheduler::begin_unit` | `commit_generated` |

#### 8.2a The unit boundary — one call with the K/V anchor

**"This unit's own tokens start here" is one fact recorded for two stores**, and
the seal reads them back together: `[anchor, block_count)` of the K/V, and the
matching token count from the index (§8.3). Nothing in either store makes them
agree. The prefix ahead of a unit is *forwarded* on the slot, so its rows land in
the same live index tail the unit is about to extend, and the seal then asks for
"the last N tokens" of a run that begins before the unit does.

So the cut is taken **by `Scheduler::begin_unit` (or the free
`close_unit_boundary` it wraps), and nowhere else.** There are four such
instants, because *where a turn's own tokens begin depends on how its user half
got onto the slot*:

| Site | Unit begins because | Unit |
|---|---|---|
| `apply_segments_finish`, **before** the `deferred_user` prefill | the projection is about to prefill the turn's user message onto the parent | a turn's user half |
| `create_view`, **before** `create_view_sequence` | the slot forwarded nothing during projection, so the turn starts on the view | a turn |
| `reproject_view`, after `apply_projection_finish` | the same turn's anchor moved to the rebuilt prefix's end | a re-anchored turn |
| `prepare_section_ingest` | the prefix is injected and the section's own tokens follow | a section |

The first two are **not** alternatives to be chosen between, and that is the
subtlety: a projection carrying an in-flight user message prefills it onto the
parent *before* the carve, while a slot on the `skip_projection` ingest path
reaches the carve having forwarded nothing. Both fire; closing an already-empty
tail is a no-op, so whichever is the real boundary is the one that closes
anything.

`begin_unit` takes no anchor argument, because the four do not share a
coordinate system — a windowed borrow counts view blocks, an ingest counts the
slot's own — while the *cut* is the same operation on the same slot in all four.

**In `create_view` the cut is on the PARENT and it is before the carve.** Both
halves are load-bearing. `fork_recurrent` copies the index cache wholesale, live
tail included, so a boundary taken on the view afterwards leaves the parent's
tail open — it survives into the view as rows the turn extends, and the parent
re-accumulates the same prefix for the next turn to close again, giving a page
list that grows quadratically in turn count and page edges that move under turns
already sealed against them.

**In `reproject_view` the cut is what makes the tail pages installable at all.**
The rebuild re-supplies the turn's user half by *prefilling* it (`deferred_user`),
which leaves rows in the live tail; `push_positional_state` refuses a page onto a
non-empty tail, so without the cut every tail page is rejected and the slot holds
the turn's K/V with nothing indexing it.

> **Three placements were tried and are wrong. Recorded because each looked
> right.**
>
> - **End of `apply_segments_finish`** — reads like the boundary (everything above
>   it *is* prefix), but the deferred user message prefills onto the same slot
>   *after* that point, and the whole path is skipped for a slot that accumulates
>   across turns (`skip_projection`) — which is the `code_read` ingest, i.e. the
>   case that was failing. Numbers came back byte-identical between rebuilds.
> - **`prepare_section_ingest` alone** — correct for sections, and a repo_map turn
>   takes neither this path nor the one above (`insert_turn_inner` →
>   `submit_prefill_unit` → `SubmitTurn`).
> - **Prefill admission** — right in effect, wrong in place: on the view, after
>   the fork, so the parent's tail stayed open and grew. Admission is when work
>   leaves the queue, which happens once per unit but says nothing about where
>   that unit's tokens start.
>
> Measured before the boundary existed: one page 376 tokens wide holding a
> 323-token projected prefix and a 53-token turn prefill, so a 55-token turn
> sealed rows spanning 378 — over by exactly the prefix. Refusing the over-wide
> page instead (the walk's own guard, §4.6) turned the same cause into the same
> turn sealing 2 tokens of rows. One cause, both symptoms.

`tail_span_pages`' refusal of an over-wide page is therefore an **alarm, not a
mode**: reaching it means some path opened a unit without passing `begin_unit`.

#### 8.2b The reasoning boundary — one funnel at commit

Call `close_tail_into_page` (§4.6) twice per thinking turn, so its index becomes
three pages — `[pre]`, `[reasoning]`, `[answer]`.

**The timing needs stating once because it is off by one from where it reads.** A
token committed at step *t* is *forwarded* at step *t+1*, and a token's index row
is appended by the forward that carries it. So a cut made at commit time lands
before the next step's append, which gives:

| Close when… | so that… |
|---|---|
| this is the turn's **first** committed token | the reasoning page opens where the assistant body does |
| the token committed *before* this one was `</think>` | `</think>` closes it |

Neither condition mentions `<think>`, for the same reason §8.1 does not: the
reasoning region starts at the assistant body, not at the open marker. The first
cut is therefore nearly free — it falls on the prefill/decode boundary, which is
already a wave boundary — and it needs no token test at all, only "is
`generated_tokens` empty". The second is keyed off the recorded
`think_close_at`, not the raw previous token, which is what bounds it to exactly
one cut.

**The push and the cut are one operation, so they are one function.** Only the
push knows a region has opened, and the cut has to land in the window between the
commit and the forward — a window nothing else runs in. `commit_generated` does
both. Handing the caller a `bool` to act on made that a convention rather than a
mechanism: four sites construct or advance a `DecodeState`, each was free to
answer differently, and the compiler had nothing to say about one that forgot.

Two tokens arrive **already forwarded** — a stencil's static run, and the
byte-heal path — and both record their boundary through `push_forwarded`, which
deliberately cannot cut: the rows are already pooled, so a cut there would leave
them on the page the new region was meant to start after. `run_prefill` has
already split that forward at the boundary, which is the only moment the cut
could land, and it does so in the one function that forwards an arbitrary token
span for a single sequence — so a third caller inherits it.

One invariant this needs: **a wave must not forward tokens from both sides of a
boundary for the same sequence**, or the straddling block is pooled from both and
cannot be un-pooled afterwards. Plain decode satisfies it for free — one token
per sequence per wave. Speculative decode does *not* get it from
`commit_decoded_tokens` running per accepted position, which serialises the
bookkeeping but not the forward: an accepted block still appends as one span.

**Speculation is LIVE on this checkpoint, and this section previously said it was
not.** The merged engine artifact folds the MTP head in as `blk.{num_layers}`, so
`self.model.mtp` is `Some` and `draft_budget` returns the ladder's 4 for any
narrow wave — every decode step drafts. The old claim ("carries no MTP tensors …
a constraint to honour when the head lands") cost real time: it was used to rule
speculation *out* while hunting a surplus that speculation was producing.

#### 8.2d A cut may not land inside a speculative rewind's window

`IndexCache::snapshot` captures the live tail — `n_blocks`, the open rows,
`n_open` — and **not the pages**. Within one decode step the order is: snapshot at
wave entry → sample → `commit_decoded_tokens` → `truncate_sequences`. A cut taken
in that window moves the tail's rows *into* a page and zeroes the tail; the
rewind then restores the tail from a snapshot that still contains those rows, and
they are counted twice. The surplus is exactly one drafted block, which is why it
appeared on a full accept as readily as a partial one.

Measured before the fix: the index ran a constant **+5 against a 5-token block**
on every speculative step, uniformly across all 13 KV layers, and rode into the
seal as pages covering tokens the turn does not hold — 10 of 12 turns in a
six-turn conversation, while every answer was still correct.

The rollback cannot move: it runs after the commit loop because a stash span is
good for exactly one step, and truncating twice leaves the second call with no
rewind point (`decode.rs`, "One call, not two"). So the **cut** moves.
`commit_generated` records `DecodeState::pending_page_cut`; `flush_page_cut`
takes it once the K/V is final — after the rollback on the decode path,
immediately on the prefill and summarise paths, where nothing speculates. That
point is also the only one at which the live tail is the accepted prefix rather
than the drafted block, so the page a cut produces there describes the right
tokens as well as the right count.

#### 8.2c What pins all three

Unit tests on the CPU double (`scheduler::tests`), which needed the double to
model index pages at all — a `ToyPages` of closed widths plus a live tail, forked
wholesale by `fork_recurrent` and moved by `move_recurrent`, exactly as the real
cache is. Three fail against every one of the three wrong boundary placements
above; `a_commit_marks_the_cut_and_the_flush_takes_it` pins §8.2d's separation,
including that a flush with nothing pending invents no page. The boundary
arithmetic itself is pinned separately and device-free by
`paged_index::tail_span_pages`' own tests (§4.6).

End to end, `candle-conversation/tests/projection_identity.rs` runs one script
twice on the same engine — the **prefill** path (`disable_reprojection`, the slot
seeded once and appended to) against the **projection** path (reset and rebuilt
from sealed substrate K/V) — and asserts three things, in order: no glue island
was reserved, no turn sealed an index that fails to cover it, and the two arms
reached the same answer on every turn. The order matters: the answer comparison
alone passed a run that mis-sealed ten turns of twelve, because a six-turn
conversation never re-reads its own pages. It is `#[ignore]`d and needs
`--release`.

### 8.3 Seal the pages the turn actually has, from the immediate path

**Store the list.** `seal_positional_tail_span(seq, tokens)` (`wave.rs:452`)
already walks back over whole pages until the turn's width is covered and returns
**one blob per page**. Store that list on the turn record in place of the single
blob `seal_positional_range` returns, and hand it back a blob at a time —
`push_positional_state` already takes them one at a time, and the reprojection
path already produces and consumes the list (`mod.rs:9160`, `mod.rs:9333`). This
removes an impedance mismatch rather than adding one: the record holds one page
today only because the seal it is fed from returns one.

**Every page carries its token width, and the reasoning is found by POSITION.**
The seal already knows each width (`page_at` returns it), so carrying it costs
nothing — and the alternative does not work. The page **count is not stable**: a
mid-decode reprojection closes an extra page, and reprojection is the common
case, so selecting the reasoning by ordinal ("the second of three") would window
only the turns that never reprojected and silently leave the rest whole. Widths
make the selection exact at any split, including a reprojection that lands
*inside* the reasoning and splits it across two pages.

`index_pages::without_span` returns `None` when the span does not fall on page
boundaries, and the caller then injects the turn whole — see §8.5.

**The persistence side does not change.** `RecordType::TurnIndexPage` is
last-writer-wins per turn stream id with `chunk_index` pinned at 0, and five
sites depend on that shape — replay (`substrate.rs:3678`), compaction
(`compaction.rs:465`), relocation (`maintenance.rs:358`), accounting
(`accounting.rs:59`), and `is_metadata_record` (`persistence/mod.rs:205`). Keep
**one record** and make its payload a length-prefixed sequence of the existing
per-page blobs. All five sites are untouched and `entry.index_page:
Option<Vec<u8>>` stays; only the encode and decode move. Emitting one record per
page instead would key supersession on `(stream_id, chunk_index)` and break LWW
at every one of those sites — do not.

**Move the seal onto the immediate path.** This is the prerequisite for 8.9
(§5.1). `seal_positional_range` is called from exactly one place in the scheduler
— `complete_turn_reprefill` — so the removal would otherwise take the only
producer of a turn's index rows with it. `perform_seal_and_write` already
computes the sealed range's token count, which is the width
`seal_positional_tail_span` takes; sealing by width also retires `seal_pos_from`
(§5.6).

That gap is neither hypothetical nor new: a turn taking the immediate path today
— every `keep_reasoning` tool turn — seals with K/V and **no index rows at all**.
Relocating the seal fixes that on the way past.

### 8.4 Seal the reasoning as a REAL span

`build_turn_layout`'s `ethereal_thinking` parameter goes (§5.4); every turn seals
through `with_thinking_split(text, len, false)`, where `len` is 8.1's exact span.

### 8.5 `Substrate::turn_sealed_without_thinking(timeline, index)`

Sibling of `turn_user_sealed_half`. Read the turn's `TurnLayout`, take the real
`Thinking` segment's `KvSpan`, and return `window_sealed_tokens(full, 0,
span.offset)` concatenated per layer with `window_sealed_tokens(full, span.end(),
total)` — **together with** the turn's page list minus its reasoning page.

**Both halves or neither, enforced rather than intended.** The reasoning is the
second of the turn's three pages. A turn whose list is not those three was sealed
by a path that made no cuts, or by an earlier build, so its rows describe the
whole turn and cannot be filtered to match a window — and injecting windowed K/V
beside whole-turn rows is exactly the divergence the pairing exists to prevent.
Such a turn is handed over **whole**: visibly imperfect rather than quietly
inconsistent. A turn
with no real `Thinking` segment returns the full sealing and the full list,
unchanged.

One call returns both halves, so they cannot be requested separately (principle
5). They also agree by construction rather than by two computations happening to
match: the page boundary and the `KvSpan` are the same grid position, from 8.1.

### 8.6 The projection rule

In the segment walk, a `SealedKind::Turn` injects whole when it is the highest
turn index on the slot's own target timeline, and windowed otherwise.

**By turn index, not by position in the piece list.** Groups emit in score order
with the highest last (`project.rs:1682`), so a tool-scope or repo_map group can
emit turns after the conversation group's — the last `Turn` piece in the walk is
not reliably the most recent turn. The index comparison always finds it, because
the conversation group's recent window is inviolate (`selection.rs:238`) and so
the predecessor is always selected.

The one case where the highest index is not the turn you want — an async summary
appended between a tool call and its result — is a known, accepted degradation.
Read §6.3 before adding a clause for it.

### 8.7 The coverage gate

Assert that the slot's `positional_coverage` equals its K/V width. The two sites
that report this (`apply_segments_finish`, `reproject_view_complete`) already
check both directions precisely.

**They report. Refusing was tried, measured, and reverted — and the measurement
found a regression in §8.3, not a pre-existing fault.**

A live run on a freshly wiped substrate promoted both to `Err`. It aborted 120 of
355 repo_map directory ingests and killed every conversation. The first reading
was that the divergence predated this work. **That reading was wrong**, and an
A/B against the committed baseline on the same wiped substrate says so:

| | occurrences | amounts |
|---|---|---|
| Baseline | 48 | all `3 past` — one mode |
| With §8.1–§8.9 | 25 | `323 past` ×21, `323 short of` ×3, `2066 short of` ×1 |

Same slots, same `held` values. There is a small pre-existing 3-token divergence;
this work adds ~320 on top — **exactly one repo_map ingest turn's width** — and
adds a `short` direction the baseline never shows.

The suspect is §8.3's switch from position-based sealing
(`seal_positional_range`, exactly the turn's own rows) to width-based
(`seal_positional_tail_span`, which walks back over **whole pages**). A turn
whose rows begin partway into a page takes that page entire, so the stored set
covers more than the turn and every projection borrowing it carries an index
wider than its K/V. The `short` direction additionally implicates the new
multi-page inject loop, which `break`s mid-list on a refused page where the old
single-blob push could not.

The seal now measures this directly — it sums the page widths against
`turn_token_count` and logs the per-page widths on a mismatch — because three
attempts to pin the mechanism by reading the call graph reached three different
answers. **Fix that before promoting these checks**, and before trusting any
behavioural result: the gates in §9 have not been run.

This is what proves 8.2 and 8.5 agree. A windowed turn contributes exactly
`page[pre] + page[answer]` tokens to coverage and exactly the same count to the
slot, so a divergence in either the cut or the window surfaces here as a number
instead of as a wrong answer many turns later.

### 8.8 Restrict the sig gather

The seal drops the reasoning span from the turn's signature run
(`sigs_without_span`) before encoding it. The sigs describe what is
*retrievable*, and the reasoning is retrievable exactly once — from the turn
immediately following it, by position — so leaving it in lets the belief scan
score a turn on content every later projection removes.

**The index shift is safe here, and the reason is worth keeping** because it
would not be safe everywhere. The run is 1:1 with the turn's grid, so removing a
middle range shifts every index past the hole. Exactly one consumer indexes it
positionally — `Substrate::user_sig_span`, which reads the layout's `User`
segments — and on a dialogue turn the user body lies wholly before the assistant
body, hence wholly before the reasoning, so the hole is past everything it
addresses. The one shape with a `User` segment *after* the assistant body is a
code_read tool exchange, and that returns from `build_turn_layout` before the
thinking split: no `Thinking` segment, so the filter is never invoked for it. The
other consumer, the belief gallery, scores each signature independently and reads
no index at all.

What remains is 1:1 with the **windowed** turn — which is what a projection of
that turn actually injects.

### 8.9 Remove §5, in one change

**This is the step that cannot be got wrong quietly.** The re-prefill is what
makes sealed bytes reasoning-free today, so deleting it while the projection
still injects turns whole leaves every turn attending its own thoughts — a
regression that is fluent, silent, and invisible to any gate that does not
specifically probe for it. 8.3 is a hard prerequisite, not an ordering
preference (§5.1); 8.6 must be live in the same landing (§8.10).

Delete §5.1–§5.4 together rather than piecemeal. They are one subsystem:
`keep_reasoning` is meaningless without the strip, the strip is unreachable
without `enqueue_clean_turn_reprefill`, and the parked seal exists only to carry
the re-prefill's result. Removing them separately leaves intermediate states in
which a turn can be sealed by two paths, which is precisely the
two-records-disagreeing failure §6.4 rejects.

Work outward from `SealAction::TurnReprefill`: deleting the variant makes the
compiler name every site that produced or consumed it, and the exhaustive matches
on `SealAction` mean nothing is missed silently. Do the same for
`keep_reasoning` by deleting the `TurnOptions` field first. Check §5.5 before
removing anything whose name contains `think` or `pending`.

The DISCARD/MOVE split collapses to MOVE, which already exists, and the
post-decode tail stops having two handlings — the clean grid replayed it inline,
the MOVE path calls `run_prefill` — leaving only the second, which is already
what a `keep_reasoning` turn does today.

### 8.10 Landing plan

**The re-prefill masks its own replacements.** It replays a grid whose `<think>`
tokens are already gone, so the page hook (8.2) never fires on the grid that is
actually sealed — its pages are built on the decode slot and thrown away — and
the windowing (8.6) is exercised only by `keep_reasoning` turns, because every
re-prefilled turn seals `Thinking { kv: None }` and is correctly injected whole.
Dialogue turns, which are the traffic, exercise none of it until the re-prefill
is gone. Staging finely therefore buys much less than the numbering implies: most
intermediate states are inert rather than incrementally verifiable.

Two pieces are genuinely independent, and both are bug fixes on their own merits:

1. **The exact span (8.1).** ✅ **Landed.** `Scheduler::think_close`,
   `DecodeState::think_close_at`, `DecodeState::push_generated`, and
   `build_turn_layout` taking the recorded index; the tokeniser round trip is
   gone. Pinned by `thinking_span_covers_the_body_through_the_close_marker` and
   `no_recorded_boundary_leaves_the_body_whole` in `turn_layout.rs`, which assert
   against a hand-built grid rather than against re-tokenised prose — a test that
   re-derived the length from the text would have agreed with the bug.
2. **The seal relocation (8.3).** Verifiable against current behaviour: every
   turn sealed through the immediate path gains index rows, which fixes the live
   gap where `keep_reasoning` turns have none.

Everything else — 8.2, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9 — becomes observable only
together, and lands together. **Three landings**, with the throughput baseline
(§9) taken before the first and after the last.

**No staging flag.** The tempting middle step is to pin `keep_reasoning` true for
one landing so the windowing can bake while the re-prefill still stands as a
fallback. That is a dual code path kept alive by a flag — prohibited outright,
and precisely the thing this design exists to remove. The safety net is §9's
behavioural gates, not a dial. If those gates cannot be written, that is the
signal to stop rather than to add a toggle.

### 8.11 What this costs, in total

Worth stating plainly, because the design's claim is that it is smaller than what
it replaces.

**Added, ~100 lines:** one 8-line `IndexCache` method composed of two existing
ones; the trait method that reaches it; a two-condition hook in the decode step;
two `usize` fields on `DecodeState`; one substrate accessor beside an existing
sibling; one `if` on a turn index in the walk. The turn record holds a list where
it held a single element.

**Deleted, ~475 lines:** `enqueue_clean_turn_reprefill` (~117),
`complete_turn_reprefill` (~113), `PendingTurnSeal` (~49), the go/no-go with its
DISCARD arm (~79), the promote and unreachable `SealAction` arms (~29),
`strip_think_from_tokens_keep_layout` (~20), the `SealAction` variant (~10), the
two scheduler fields (~10), `ethereal_thinking` and its call site (~20), the four
telemetry counters and three log fields (~35), and `keep_reasoning` across 18
sites (~30).

**Relocated, not deleted, ~35 lines:** the index-page seal (§8.3). This is the
whole of the difference between "delete §5.1" and "delete §5.1 safely".

**And per turn:** one full re-forward of the turn, one wave of `Done` latency,
and one tokeniser round trip.

---

## 9. Gates

- **Windowing is exact.** Unit tests over synthetic sealings: a reasoning span
  wholly inside one chunk (the same physical chunk emitted twice, sharing
  gids); spanning a chunk boundary; abutting the start; abutting the end; empty.
  Assert chunk `offset`/`token_count` against expected values, and assert the
  gid refcounts show sharing rather than copying.
- **Index and K/V agree.** For each of the above, assert
  `indexed_tokens == injected width` exactly.
- **A closed page leaves the tail's arithmetic intact.** The trap §4.6 names,
  as a direct test: close a tail mid-sequence with a short final block, then
  assert `candidates_at`, `block_start`, and `indexed_tokens` against a
  hand-computed walk for every position across the boundary. A bare
  `flush_open_block` at the same point must fail this test — if it passes, the
  test is not measuring what it claims.
- **Boundary tokens are found where the layout says.** The recorded pair (8.1)
  indexes `<think>` and `</think>` in the turn's own token grid. Assert against
  the grid, not against re-tokenised text — the point of the change is that the
  two can disagree.
- **Every sealed turn has rows.** A turn sealed through the immediate path
  carries a page list covering its full width. This is the gate that would have
  caught today's `keep_reasoning` turns sealing with none (8.3), so it is worth
  writing before the change that fixes them.
- **Positions compact.** A turn projected with its reasoning windowed out yields
  the same absolute positions for the answer tokens as the same turn prefilled
  reasoning-free. This is the claim of §4.3 and it should be measured, not
  assumed.
- **The tool round trip carries its reasoning.** A `think → call → result →
  answer` exchange where the answer depends on a fact stated only inside the
  call turn's reasoning.
- **The turn after that does not.** The same conversation one turn later:
  the fact is no longer attendable.
- **Recall of the fold.** A probe asking about something stated *only* inside a
  windowed-out reasoning block, several turns later. The GDN state saw it and
  cannot forget it (§6.2), so a positive result here is expected and is not a
  bug — the gate exists to measure how strongly the fold retains it, which is
  the evidence for or against §6.2's trade.
- **The removed forward, measured.** The throughput baseline §8.10 brackets:
  the ladder run before the first landing and after the last, several runs each,
  with the commit recorded against every figure. One run is not evidence on this
  branch — the same build has swung 161→247 t/s between runs — and a table
  without its commit cannot be re-derived later.

  The re-prefill's own cost is not separable from the existing counters: the
  housekeeping split measures its *seal* (`reprefill_write_ms` /
  `reprefill_truncate_ms`), which the immediate path performs too, while the
  extra forward lands in the prefill band undistinguished. So the before/after
  ladder is the measurement, and the enqueue trace (`mod.rs:5194`, which already
  logs the clean grid's token count) is what says how much forward went away.

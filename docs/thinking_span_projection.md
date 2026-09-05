# Thinking-Span Projection

**Status:** design, authoritative. Supersedes the clean-turn re-prefill
(`enqueue_clean_turn_reprefill`) and the `hold_seal` tool-call carve-out.

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
   arrives. Previously solved by `hold_seal`, a flag threaded from
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
> rebuilds from the substrate)"* — `scheduler/mod.rs:5352`
>
> *"The next turn's `apply_projection` rebuilds the slot from substrate anyway,
> so holding onto these `Arc<ChunkGid>`s between turns just pins arena slots
> that nothing reads."* — `scheduler/mod.rs:6732`

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

`inject_arc_sealed` (`scheduler/projection_assembler.rs:1465`) copies the
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

### 4.6 The index can end a block early, and pages are ragged

The QSA index is coarser than the K/V: one row per `ratio` tokens, keyed off
hidden states and unrecoverable from the K/V. That granularity would be a real
obstacle except that the index has the same early-termination property the K/V
has:

> *"The flushed block is a summary of fewer than `ratio` tokens and is therefore
> NOT what a continuous run would have produced for that span. That is the
> deliberate trade: **the scorer carries each page's width and derives the
> candidate prefix from it, so a short block is expressible**; a block pooled
> from another turn's tokens is not correctable at all."*
> — `qwen4exp/indexer.rs:236` (`flush_open_block`)

And the scorer does not divide by `ratio` to find a query's candidate prefix —
it **walks page widths**:

> *"Per row, the candidate blocks are those wholly below its tail. With injected
> pages ahead of the live tail this is a walk over their widths rather than a
> division — see `candidates_at`."* — `indexer.rs:575`

**Consequence:** dropping a whole page renumbers everything downstream by
construction. Provided the reasoning occupies *whole pages*, its index rows can
be omitted exactly, with no rounding in either direction.

That is the one thing this design must add: an index flush at the `<think>` and
`</think>` boundaries, so the reasoning starts and ends on a page boundary. It
uses `flush_open_block`, which `seal_positional_tail_span` already calls for the
live tail.

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
| `enqueue_clean_turn_reprefill` | `scheduler/mod.rs:5141` | Builds the clean grid, stashes the pending seal, enqueues the unit. |
| `complete_turn_reprefill` | `scheduler/mod.rs:5263` | Snapshots the clean K/V, seals, fires the deferred `Done`. |
| `PendingTurnSeal` | `scheduler/mod.rs:1354` | The parked seal: `parent_id`, `seal_block_from`, `seal_pos_from`, `layout`, `token_ids`, tags, `event_tx`, stats. |
| `Scheduler::pending_turn_seals` | `scheduler/mod.rs:2616` | `HashMap<u64, PendingTurnSeal>`; init at 2873. |
| `Scheduler::next_turn_seal_id` | `scheduler/mod.rs:2619` | Monotonic id source; init at 2874. |
| `SealAction::TurnReprefill { pending_id }` | `scheduler/mod.rs:1464` | The enum variant and its doc comment. |
| Its promote-loop arm | `scheduler/prefill.rs:2506–2524` | Error path (surface on caller channel + truncate) and success path. |
| `PromoteStep::Reprefill` | `scheduler/run.rs` | Telemetry step; check the perf page for a reference before removing. |
| The go/no-go branch | `scheduler/mod.rs:6549–6602` | The `truncate_sequence_to_blocks` + `seal_writer_boundary` gate, the DISCARD arm that releases the view's recurrent state, and the fall-through to the immediate seal. |

The DISCARD/MOVE split at `6563`/`6609` collapses: with no re-prefill to
re-advance the state, every turn takes **MOVE** (`move_recurrent`). Nothing is
released, because nothing can be rebuilt.

### 5.2 The reasoning-stripping token surgery

| Symbol | Where | Notes |
|---|---|---|
| `strip_think_from_tokens_keep_layout` | `scheduler/mod.rs:4774` | Detokenise → strip → retokenise, to build the clean replay tokens. |
| `think_strip::strip_think_blocks_keep_layout` | `think_strip.rs` | Its only caller is the above; becomes dead. |

This is the operation the design replaces: a text round trip through the
tokeniser, at seal time, to produce a token stream the model must then be run
over again. The windowing in §4.2 achieves the same visibility outcome with no
tokeniser involvement and no forward.

### 5.3 The `hold_seal` carve-out

`hold_seal` marks "this turn's continuation is already scheduled, so do not
strip its reasoning". With nothing stripping, it has no meaning. It is threaded
through five types and set from `ToolMode`, so removal touches the whole
submit path:

- `TurnOptions::hold_seal` — `turn.rs:95`
- `SchedulerRequest::SubmitTurn::hold_seal` — `scheduler/mod.rs:229`
- `PrefillWork::hold_seal` — `scheduler/mod.rs:1546`
- `DecodeState::hold_seal` — `scheduler/mod.rs:1143`
- Plumbing (read and pass through) — `scheduler/mod.rs:3048, 3521`;
  `prefill.rs:2709, 2808`; `conversation.rs:1876, 2046, 2122`
- Construction sites passing `false` — `conversation.rs:4227`;
  `engine.rs:1317`; `tree/summarize.rs:255`; `scheduler/mod.rs:4630, 4948, 5251`
- The one site that passes `true` — `zend/src/session.rs:2554`
  (`hold_seal: tools_mode != ToolMode::None`)

That last line is the only place the sealing layer has ever known what a tool
is. Deleting it is the point of the design: **after this change, no type
between `TurnOptions` and the substrate carries tool state.**

### 5.4 The seal-time visibility decision

| Symbol | Where | Notes |
|---|---|---|
| `build_turn_layout`'s `ethereal_thinking` parameter | `scheduler/mod.rs:4981` | Chose between a real and an ethereal `Thinking` segment. Every turn now seals real, so the parameter and its `true` call site go. |

`TurnSegment::Thinking::kv` stays `Option<KvSpan>`. The type still needs to
express an ethereal block — a `/no_think` turn's collapsed `<think></think>`
has no K/V — and the projection reads `Some(span)` as "there is a span to
window out" and `None` as "nothing to do". The option is a genuine hit/miss,
not a feature flag.

### 5.5 What must NOT be removed

Each of these looks like part of the machinery above and is not:

- **`think_strip::strip_think_blocks`** (`scheduler/mod.rs:4761`,
  `tree/summarize.rs:354`) — strips reasoning from *display and summary text*.
  Unrelated to K/V; stays.
- **`think_strip::strip_empty_think_blocks`** (`scheduler/mod.rs:4993`) — drops
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
(`scheduler/mod.rs:8115`) from the reasoning-free grid "so they match the
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
but the hazard does not: the same alignment must hold between a windowed turn's
chunks and its filtered pages. §8's index flush is what makes it hold, and §9's
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
> — `qwen4exp/wave.rs:1184` (`carries_recurrent_state`)

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
- **It is already accepted in production.** Tool turns take `hold_seal` today,
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

---

## 7. Principles

These are why the design is small, and the rules that keep it small.

1. **Record everything; decide late.** Sealing is a faithful record of what was
   decoded. Visibility is a projection-time policy. Never destroy information at
   write time to implement a read-time rule — that is what forced the re-prefill,
   the `hold_seal` carve-out, and the seal-twice proposal in turn.

2. **The projection is the only authority on what the model sees.** One decision
   point, consulted on every rebuild. If a second place starts deciding
   visibility, they will disagree, and the disagreement will be silent.

3. **Positional, not stateful.** The rule is a function of a turn's index. A
   rule that needs to know *why* a turn exists needs that reason threaded
   through every layer that touches a turn — which is exactly the `hold_seal`
   plumbing this design removes.

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
   existed and most already had a production caller. That is the reason the
   change is additive rather than architectural, and it is worth preserving:
   if an extension seems to need a new kind of thing, first check whether an
   existing one already has the property.

### How to keep it simple

The pressure on this design will be to add a condition. Each of these is a
reintroduction of the state the design exists to remove:

- *"Keep thinking for two turns when …"* — the rule stops being positional.
- *"Except for tool turns"* — the tool flag comes back, through every layer.
- *"Seal without thinking when we know we won't need it"* — write-time decisions
  return, and now two records disagree.
- *"Skip the flush when the block is short"* — the index stops being page-exact
  and the coverage gate starts warning instead of asserting.

If the rule ever needs a second clause, that is a signal to re-derive it, not to
extend it.

---

## 8. Implementation

1. **Index flush at the reasoning boundaries.** During decode, call
   `IndexCache::flush_open_block` when `<think>` opens and when `</think>`
   closes, so the reasoning occupies whole index pages. Uses the existing flush
   path; the ragged-page contract (§4.6) already supports a short block.

2. **`Substrate::turn_sealed_without_thinking(timeline, index)`.** Sibling of
   `turn_user_sealed_half`: read the turn's `TurnLayout`, find the real
   `Thinking` segment's `KvSpan`, and return
   `window_sealed_tokens(full, 0, span.offset)` concatenated per layer with
   `window_sealed_tokens(full, span.end(), total)`. A turn with no real
   `Thinking` segment returns the full sealing unchanged.

3. **Page selection to match.** The turn's stored index pages are filtered to
   those outside the reasoning span, in the same call that produces the windowed
   K/V, so the two cannot be requested separately.

4. **The projection rule.** In the segment walk, a `SealedKind::Turn` that is
   not the last turn injects the windowed sealing; the last turn injects whole.

5. **The coverage gate.** After each turn is injected, assert that the index
   tokens the slot now holds equal the injected K/V width — the same accounting
   `score_rows` reports on a shortfall (`page_token_span() + n_blocks * ratio +
   n_open`). This is an assertion, not a warning.

6. **Restrict the sig gather** to the turn minus its reasoning span (§5.6).

7. **Remove §5, in one change.** Order matters: steps 1–6 must be landed and
   green first, because the re-prefill is what currently makes sealed bytes
   reasoning-free, and deleting it before the projection windows correctly
   leaves every turn attending its own thoughts — a regression that is fluent,
   silent, and therefore not caught by any behavioural gate that does not
   specifically probe for it.

   Then delete §5.1–§5.4 together rather than piecemeal. They are one subsystem:
   `hold_seal` is meaningless without the strip, the strip is unreachable
   without `enqueue_clean_turn_reprefill`, and the parked seal exists only to
   carry the re-prefill's result. Removing them separately leaves intermediate
   states in which a turn can be sealed by two paths, which is precisely the
   two-records-disagreeing failure §6.4 rejects.

   Work outward from `SealAction::TurnReprefill`: deleting the variant makes the
   compiler name every site that produced or consumed it, and the exhaustive
   matches on `SealAction` mean nothing is missed silently. Do the same for
   `hold_seal` by deleting the `TurnOptions` field first. Check §5.5 before
   removing anything whose name contains `think` or `pending`.

---

## 9. Gates

- **Windowing is exact.** Unit tests over synthetic sealings: a reasoning span
  wholly inside one chunk (the same physical chunk emitted twice, sharing
  gids); spanning a chunk boundary; abutting the start; abutting the end; empty.
  Assert chunk `offset`/`token_count` against expected values, and assert the
  gid refcounts show sharing rather than copying.
- **Index and K/V agree.** For each of the above, assert
  `indexed_tokens == injected width` exactly.
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

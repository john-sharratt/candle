# The re-feed — every projection ends by reading its own last tokens again

**Status:** design (2026-11-20). Replaces the request's two readings
(`scheduler/first_reading.rs`, `docs/hit_driven_projection.md` §7.4). Builds on
the slot, the pool, the fill and the hit counters of
`docs/hit_driven_projection.md`, which stay.

## 1. Why

A placed file is only usable once a prefill has run over it. Measured on
BlackWorld (2026-11-20, `docs/results/dependency_battery_2026-11-20_b.md`):

- The battery's direct questions rose 38.5% → 47.9% when a request's question
  was prefilled again after its files were placed — the one change that put a
  prefill *after* a projection.
- A file placed by a mid-decode reprojection, with nothing prefilled over it,
  was ignored even while it led the slot: `recorded_reply.rs` entered at reply
  token 64 or 129, scored highest of everything placed, and the model said it
  had read nothing holding the symbol. Placed at token 8, it was used.
- On 11-10 (`provenance_depth.rs`) sealed K/V read under a context other than
  the one it was sealed in failed, and the same turns forwarded fresh recalled.

A reprojection swaps K/V in, and on Flash-Next 36 of 48 layers are recurrent:
the swapped-in content reaches them only through the tokens forwarded after it.
The two readings put one such prefill at the start of a request and nothing at
any other projection.

## 2. The design

**At every projection that changes the slot — a turn's opening boundary and
every mid-decode reprojection — the turn's last tokens are forwarded again over
the rebuilt slot.** At most `REFEED` = 8 of them: the recurrent layers' short
span (`delta_net::timeline::TIMELINE_ROWS`), the most one decode wave writes
for a row.

1. **The rows ride the decode wave.** A re-feed is a row of the next decode
   step's wave, as a stencil's static run is (`scheduler/stencil_wave.rs`): no
   forward of its own. Its last row is scored, and the token after it is
   sampled from that row.
2. **The attention layers' K/V and index rows are overwritten.** The re-fed
   tokens keep their positions: the view's K/V tail and its QSA index tail are
   moved back over them and written again, so each position is attended once,
   computed in the new context.
3. **The recurrent state is not rewound.** The re-fed tokens are folded into the
   DeltaNet state on top of what it holds. The state is an associative memory
   written by the delta rule — each token writes `v − S·k`, the error against
   what its key already holds — so a re-fed token whose key the state holds
   replaces that association with the value the new context gives it rather
   than storing it twice. The cost: the state ages by the re-fed tokens once
   more (the gate applies per token), and the short convolution's first rows
   read the original last tokens as their predecessors. Nothing in the engine
   rolls a committed recurrent token back (`state_store.rs`: the timeline is
   the undo, within one step), and the re-feed needs none.
4. **No re-feed when nothing changed.** A reprojection that keeps the selection
   the slot already holds (`ReprojectOutcome::Kept`) does not rebuild, and the
   last tokens were computed against the context they still have.
5. **A re-feed never crosses a closed index page.** The index closes a page at
   the turn's start, at the prefill→decode boundary (the reply's first token
   opens a page), at `<think>`, at `</think>` and at the turn closer
   (`page_break_tokens`). A re-feed reaches back over the tokens since the
   newest closed page and no further: `min(REFEED, tokens since the last
   cut)`. A request's opening boundary fires just after `<think>`, so its
   re-feed is the token or two after that cut; a mid-decode reprojection inside
   the reasoning re-feeds 8.
6. **At a tool call, only at its open tag.** The `<tool_call>` lock-in boundary
   re-feeds; no boundary fires inside the call body, so nothing re-feeds while
   a stencil drives it.

### 2.1 The rebuilt view owns the turn

A turn decodes on a view of its conversation's parent slot: the view borrows
the parent's chunks and owns only what the turn writes, so a faulted turn is
discarded with the parent untouched and a finished one transfers its blocks at
finalize (`carve_view`, `ViewState`). A reprojection rebuilds the **parent**
and, today, injects the turn's captured tail onto the parent too
(`inject_sealed_at_tail(parent_id, …)`) and carves a view borrowing all of it.
After its first reprojection a turn's own tokens therefore sit below the view's
`writer_start` — borrowed, though nothing else holds them — and
`truncate_sequence_to_tokens` refuses to cut there (`sequence_ops.rs`).

**The rebuild carves the new view over the rebuilt prefix only and injects the
turn's tail, and its index pages, onto the view.** The turn's tokens stay the
view's writes until the turn finalizes, exactly as on a turn that never
reprojected; the parent holds only the projected context. The turn's recurrent
state moves onto the view as it does now (`move_recurrent`).

### 2.2 The tail stays native where a re-feed can reach

Completed chunks of a live turn are quantized in the gap between forwards
(`run.rs`, `gap/table.rs`). The gap takes a writer chunk only when neither it
nor the chunk after it holds a position a re-feed can reach: the view's current
and previous chunks stay native, since a re-feed of at most 8 tokens crosses at
most one 32-token chunk boundary. A turn seal quantizes everything, as now.

### 2.3 What the re-feed's queries score

The re-feed's index queries join the stride's kept queries (up to 64 decoded or
prefilled rows, `docs/hit_driven_projection.md` §6.3) and all of them score the
pool's unloaded rounds at the sampled layers, into the same hit counters and
momentum: the next projection's fill ranks by what the indexer reads from the
context it now has.

**Scoring runs at the rung that covers the card's pool**, apart from the
attention's: ×8 for the 72 GB card's 2M tokens, ×4 for the 24 GB card's 1M, ×2
for the 16 GB card's 512K (`scheduler::pool::pool_tokens_for`). The pool is
scored as one context at that rung rather than in windows no longer than the
query's position. The attention stays at the sequence's own rung — rung 0, ×1,
inside the slot. Index keys are stored un-rotated and rotated at read
(`indexer.rs`), so the scoring rung is a rotation table carried by the scoring
job (`ShadowJob`), not different stored keys. The schedule gains a ×8 rung
reaching 2,097,152 positions (`qwen4exp/rope.rs`: ×1 to 262,144, ×2 to
524,288, ×4 to 1,000,000 today).

### 2.4 Further back

Every projection re-feeds at most 8 tokens. A re-feed that reaches back in
multiples of 8 — several waves, 8 a wave, until the recurrent state and the
recent K/V are carried by the new context — is the next step once the battery
has measured this one.

## 3. What it replaces

| Removed | Why it goes |
|---|---|
| The request's first reading (`scheduler/first_reading.rs`, `PrefillWork::rereads`, `ActivePrefill`'s stop one token short, `Scheduler::first_readings`) | Every projection now re-reads; the request's opening is one of them. |
| The second reading's skipped boundary (`PrefillWork::second_reading`) | The opening's boundary runs and ends with a re-feed. |
| Seeds held out of a first reading (`WorkingSet::holds_out`, `SlotBoundary::question_pending`) | There is one reading; the seeds are in it, as background. |
| The shadow scan's windows no longer than the query's position | One context at the pool's rung. |

What stays: the pool and its scan leaders, the fill and its caps, the hit
counters and momentum, the early boundaries at 8, 16 and 32 reasoning tokens,
the first question's opening, the seeds, and the fill report in every
projection record.

## 4. Build

1. Remove the request's two readings (§3).
2. The rebuilt view owns the turn (§2.1): carve over the rebuilt prefix, inject
   the tail and its index pages onto the view.
3. The gap holds the view's current and previous chunks native (§2.2).
4. `IndexCache::rewind_rows(n)` — the live tail's last `n` rows, refused across
   a closed page; with the K/V tail's `truncate_sequence_to_tokens`, one
   `rewind_tail(seq, n)` on the model.
5. The re-feed row: after a projection rebuilds a turn's view, its last
   `min(8, tokens since the last cut)` tokens are rewound and queued as a row of
   the next decode wave; the token after it is sampled from its last row. None
   on a `Kept` reprojection; at a tool call only at the open tag.
6. The re-feed's queries join the stride's for the scoring job; the scoring
   job's rung is the pool's; the ×8 rung added to the schedule.
7. The battery against 62.2% (`dependency_battery_2026-11-20_b.md`).

Each step lands with its unit tests: the view's ownership after a rebuild
against its `writer_start` and chunk counts, the gap's hold against the chunks
it takes, the rewind against raw rows and offsets and a closed page, the
re-feed row's tokens and positions, the scoring rung's rotation table against
the rung's own.

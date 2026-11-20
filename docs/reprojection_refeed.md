# The re-feed — every projection ends by reading its own last tokens again

**Status:** built (2026-11-20). Replaces the request's two readings
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

**At every projection that changes the slot — a boundary between a long
opening's chunks and every mid-decode reprojection — the turn's last tokens are
forwarded again over the rebuilt slot.** At most `REFEED` = 8 of them (`scheduler/refeed.rs`, equal
to `HELD_TAIL_TOKENS`): the recurrent layers' short span
(`delta_net::timeline::TIMELINE_ROWS`), the most one decode wave writes for a
row.

1. **The rows ride the decode wave.** A decoding turn's re-feed is a row of its
   next decode wave, ahead of its pending token, as a stencil's static run is
   (`scheduler/stencil_wave.rs`, `Scheduler::play_refeeds`): no forward of its
   own, and the token after the row is sampled from its last position. A run a
   stencil played the same iteration takes the re-feed at its front. An opening
   stopped at a boundary instead steps its prefill offset back over the re-fed
   tokens and prefills them again in its next stride.
2. **The attention layers' K/V and index rows are rewritten.** The re-fed
   tokens keep their positions: they are left out of the rebuild (§2.1) and
   written again by the re-feed, so each position is attended once, computed in
   the new context.
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
5. **A re-feed never crosses a closed index page, and starts on an index
   block.** The index closes a page at the turn's start, at the prefill→decode
   boundary (the reply's first token opens a page), at `<think>`, at
   `</think>` and at the turn closer (`page_break_tokens`); its live tail holds
   `n_blocks` completed blocks of `ratio` = 4 tokens and the open rows after
   them. A re-feed rewinds the live tail only, and only to a block boundary
   (`indexer::refeed_cut`, `IndexCache::rewind_live`): the whole live tail when
   it holds 8 tokens or fewer, otherwise back to the newest block boundary
   within 8 — 5 to 8 tokens. The layers with an index agree on the span
   (`refeed_span`); a span longer than what the turn itself has forwarded, or a
   turn with a stencil row still being written, re-feeds nothing.
6. **At a tool call, only at its open tag.** The `<tool_call>` lock-in boundary
   re-feeds; no boundary fires inside the call body, so nothing re-feeds while
   a stencil drives it.

### 2.1 The re-fed tokens are left out of the rebuild

A turn decodes on a view of its conversation's parent slot: the view borrows
the parent's chunks and owns only what the turn writes, so a faulted turn is
discarded with the parent untouched and a finished one transfers its blocks at
finalize (`carve_view`, `ViewState`). A reprojection captures the turn's tail
(`snapshot_sequence_tail`), rebuilds the parent, injects the captured tail onto
it (`inject_sealed_at_tail`) and carves a view borrowing all of it. After a
reprojection the turn's earlier tokens sit below the new view's `writer_start`,
and `truncate_sequence_to_tokens` refuses to cut there (`sequence_ops.rs`) — so
the re-feed does not cut the rebuilt slot. It cuts what is about to go into it:

- **The captured K/V tail is trimmed** before it is injected
  (`SealedSequence::trim_tail`). The capture is the scheduler's own copy, so
  the cut touches nothing live; the chunks it cuts are native (§2.2), and a
  quantized chunk is refused rather than cut.
- **The old view's index live tail is rewound** before its positional tail is
  sealed for the rebuild (`ManagedBatchedModel::rewind_positional_tail`), so
  the rebuilt slot's index ends where its K/V does.

The rebuilt slot then ends `n` tokens short of the turn, and the re-feed writes
those `n` positions as the new view's own writes. The view ownership is
otherwise unchanged: the parent holds the projected context and the turn's
earlier tail, the view writes from the re-feed on, and the turn's recurrent
state moves onto the view as before (`move_recurrent`).

### 2.2 The tail stays native where a re-feed can reach

Completed chunks of a live turn are quantized in the gap between forwards
(`run.rs`, `gap/table.rs`). The gap takes no chunk holding one of the writer's
last `HELD_TAIL_TOKENS` = 8 tokens (`LayerShape::held_from`): a re-feed reaches
at most 8 tokens back, crossing at most one 32-token chunk boundary, so the
current chunk and, when the last 8 tokens straddle a boundary, the previous one
stay native. A turn seal quantizes everything, as before.

### 2.3 What the re-feed's queries score

The re-feed's index queries join the stride's kept queries (up to 64 decoded or
prefilled rows, `docs/hit_driven_projection.md` §6.3) and all of them score the
pool's unloaded rounds at the sampled layers, into the same hit counters and
momentum: the next projection's fill ranks by what the indexer reads from the
context it now has.

**The pool is scored as one context, seated behind each query, at the rung that
covers it** (`qsa_shadow.cu`). A kept query with `C` candidate blocks scores the
`L` unloaded rows over `S = max(C, L)` blocks: row `j` at `(S − L + j) · ratio`,
the query at `S · ratio + tail − 1`. The query was kept rotated at its own
position on the sequence's rung; the kernel rotates it back on that rung's
table (`ShadowJob::q_tab`) and forward to its seat on the table of the rung
covering the seats (`ShadowJob::tab`, `max(reach, L · ratio) + ratio`,
`hits.rs`). For a pool of 2M tokens that is ×8, for 1M ×4, for 512K ×2
(`scheduler::pool::pool_tokens_for`). The attention stays at the sequence's own
rung — rung 0, ×1, inside the slot. Index keys are stored un-rotated and
rotated at read (`indexer.rs`), so the scoring rung is a rotation table the job
carries, not different stored keys. The schedule has a ×8 rung reaching
2,097,152 positions, the factored table's reach (`qwen4exp/rope.rs`: ×1 to
262,144, ×2 to 524,288, ×4 to 1,000,000, ×8 to 2,097,152).

### 2.4 No boundary at the opening's end

A request's opening ends on `<think>` and the newline after it, and the index
closes a page at `<think>` and at the prefill→decode boundary, so a boundary at
the opening's end could re-feed a token or two and never the question: on
2026-11-20 (c) the reply then reasoned over a slot nothing had been forwarded
over, and the direct questions fell from 47.9% to 27.1%. No boundary fires
there (`scheduler/prefill.rs`, the promotion to decode). The reply decodes on
the slot its question was prefilled against, and the first boundary is the
reasoning's first early one at decoded token 9 (`scheduler/early_boundaries.rs`):
the reply's first token opens a page, 8 tokens have been forwarded and the 9th
is pending, so the re-feed is the whole of the reasoning so far — the first
tokens the model reads the placed files with are its own reasoning about the
question. A first-token tool call likewise decodes on the opening's slot; no
boundary fires inside its body.

Measured against the boundary at the opening's end
(`docs/results/dependency_battery_2026-11-20_d.md`): 53.8% against 52.8%, no
measurable change, and still below the 62.2% of the question read again over
its placed files.

### 2.5 Further back

Every projection re-feeds at most 8 tokens. A re-feed that reaches back in
multiples of 8 — several waves, 8 a wave, until the recurrent state and the
recent K/V are carried by the new context — is the next step once the battery
has measured this one.

## 3. What it replaces

| Removed | Why it goes |
|---|---|
| The request's first reading (`scheduler/first_reading.rs`, `PrefillWork::rereads`, `ActivePrefill`'s stop one token short, `Scheduler::first_readings`) | Every projection now re-reads; the request's opening is one of them. |
| The second reading's skipped boundary (`PrefillWork::second_reading`) | No boundary fires at the opening's end (§2.4). |
| The boundary at the opening's end (`wants_early_reprojection`, `ReprojectionPolicy::has_belief_collections`) | The first early boundary, at decoded token 9, takes its place (§2.4). |
| Seeds held out of a first reading (`WorkingSet::holds_out`, `SlotBoundary::question_pending`) | There is one reading; the seeds are in it, as background. |
| The shadow scan's windows no longer than the query's position | One context at the rung covering it. |

What stays: the pool and its scan leaders, the fill and its caps, the hit
counters and momentum, the early boundaries at 9, 16 and 32 reasoning tokens,
the first question's opening, the seeds, and the fill report in every
projection record.

## 4. Build

| Step | Where | Tested by |
|---|---|---|
| The two readings removed (§3) | `scheduler/` | the conversation suite |
| The gap holds the last 8 tokens' chunks native (§2.2) | `gap/table.rs` | `the_chunks_holding_the_last_eight_tokens_stay_native` |
| The captured tail trimmed (§2.1) | `SealedSequence::trim_tail` | `a_trim_cuts_the_last_tokens_across_a_chunk_boundary`, `a_trim_refuses_a_quantized_chunk_and_an_overlong_cut` |
| The index rewound to a block boundary (§2 item 5) | `indexer::refeed_cut`, `IndexCache::rewind_live`, `wave.rs` | `a_refeed_rewinds_to_the_newest_block_boundary_within_its_reach`, `a_rewound_tail_keeps_its_first_blocks` |
| The re-feed row (§2 items 1, 5) | `scheduler/refeed.rs`, `reproject_view_prepare`, `play_stencil_runs` | `the_last_tokens_come_off_the_end`, `a_refeed_row_ends_on_the_pending_token` |
| No boundary at the opening's end; the first early boundary at 9 (§2.4) | `scheduler/prefill.rs`, `scheduler/early_boundaries.rs` | `a_reasoning_reply_takes_its_early_boundaries`, `the_first_boundary_refeeds_the_reasoning_so_far` |
| One context at the covering rung, the ×8 rung (§2.3) | `qsa_shadow.cu`, `shadow.rs`, `hits.rs`, `rope.rs` | `the_shadow_scan_counts_what_its_definition_counts`, `a_span_is_the_kernels_layout`, `flash_next_takes_the_reference_angle` |

Measured against 62.2% (`dependency_battery_2026-11-20_b.md`) by the battery.

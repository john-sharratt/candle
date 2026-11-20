# The re-feed — every projection ends by reading its own last tokens again

**Status:** design (2026-11-20). Replaces the request's two readings
(`scheduler/first_reading.rs`, `docs/hit_driven_projection.md` §7.4) and the
boundary at the end of an opening. Builds on the slot, the pool and the hit
counters of `docs/hit_driven_projection.md`, which stay.

## 1. Why

A placed file is only usable once a prefill has run over it. Measured on
BlackWorld (2026-11-20, `docs/results/dependency_battery_2026-11-20_b.md`):

- The battery's direct questions rose 38.5% → 47.9% when a request's question
  was prefilled again after its files were placed — the only change that put a
  prefill *after* a projection.
- A file placed by a mid-decode reprojection, with nothing prefilled over it,
  was ignored even while it led the slot: `recorded_reply.rs` entered at reply
  token 64 or 129, scored highest of everything placed, and the model said it
  had read nothing holding the symbol. Placed at token 8, it was used.
- On 11-10 (`provenance_depth.rs`) sealed K/V read under a context other than
  the one it was sealed in failed, and the same turns forwarded fresh recalled.

A reprojection swaps K/V in, and on Flash-Next 36 of 48 layers are recurrent:
the swapped-in content reaches them only through the tokens that are forwarded
after it. The two readings put one such prefill at the start of a request, and
nothing at any other projection.

## 2. The design

**At every projection — a request's opening, a tool result's, every
mid-decode reprojection — the turn's last `REFEED` tokens are forwarded again
over the rebuilt slot.** `REFEED` is 8: the recurrent layers' short span
(`delta_net::timeline::TIMELINE_ROWS`), the most one decode wave writes for a
row.

1. **The rows ride the decode wave.** A re-feed is a row of the next decode
   step's wave, as a stencil's static run is (`scheduler/stencil_wave.rs`):
   no forward of its own. Its last row is scored, and the token after the
   re-feed is sampled from it.
2. **The attention layers' K/V and index rows are overwritten.** The re-fed
   tokens keep their positions: the sequence's K/V tail and its QSA index tail
   are moved back `REFEED` tokens and written again, so the attention layers
   read each position once, computed in the new context.
3. **The recurrent state is not rewound.** The re-fed tokens are folded into the
   DeltaNet state on top of what it holds. The state is an associative memory
   written by the delta rule — each token writes `v − S·k`, the error against
   what its key already holds — so a re-fed token whose key the state holds
   replaces that association with the value the new context gives it, rather
   than storing it twice. What it costs: the state ages `REFEED` steps more per
   projection (the gate applies per token), and the short convolution's first
   rows read the original last tokens as their predecessors. Nothing in the
   engine can roll a committed recurrent token back (`state_store.rs`: the
   timeline is the undo, within one step), and the re-feed needs none.
4. **The re-feed's queries score the pool.** The re-fed rows' index queries are
   kept as a stride's are and scored against the pool's unloaded rounds at the
   sampled layers (`docs/hit_driven_projection.md` §6.3), into the same hit
   counters: the next projection's fill ranks by what the indexer reads from the
   context it now has.
5. **Scoring runs at a forced rung.** The pool is up to 2M tokens; the re-feed's
   queries score it as one context at YaRN ×4 or ×8 — a rung the scoring job
   carries apart from the attention's, which stays at the sequence's own (rung
   0, ×1, inside the slot). Index keys are stored un-rotated and rotated at
   read (`indexer.rs`), so a forced rung is a different rotation table on the
   scoring job, not different stored keys. The schedule gains a ×8 rung
   (`qwen4exp/rope.rs`: ×1 to 262,144, ×2 to 524,288, ×4 to 1,000,000 today).
6. **A re-feed may reach further back, in multiples of 8.** `REFEED · n`, the
   rows written 8 a wave over `n` waves, until the recurrent state and the
   recent K/V are carried by the new context. A request's last 8 tokens are
   mostly its template's closing — `<|im_end|>`, the assistant header,
   `<think>` — so a request is where a longer re-feed is expected to matter.

### 2.1 What the K/V tail must hold to be re-fed

- **Native until it cannot be re-fed.** Completed chunks of a live turn are
  quantized in the gap between forwards (`run.rs`, `gap/table.rs`). The gap
  takes a writer chunk only when neither it nor the chunk after it holds a
  position a re-feed can reach: the current chunk and the one before it stay
  native, since `REFEED · n ≤ 32` crosses at most one chunk boundary. A turn
  seal quantizes everything, as now.
- **No closed page is re-fed.** The index closes a page at the reasoning
  boundary and after a break token (`close_positional_page`); a re-feed reaches
  back no further than the newest closed page, so it rewinds only the live
  tail. `IndexCache::truncate_to` cuts at page boundaries and clears the whole
  tail; the re-feed needs a rewind of the tail's last rows
  (`IndexCache::rewind_rows`).

## 3. What it replaces

| Removed | Why it goes |
|---|---|
| The request's first reading (`scheduler/first_reading.rs`, `PrefillWork::rereads`, `ActivePrefill`'s stop one token short, `Scheduler::first_readings`) | Every projection now re-reads; the request's opening is one of them. |
| The second reading's skipped boundary (`PrefillWork::second_reading`) | The opening's boundary runs and ends with a re-feed. |
| Seeds held out of a first reading (`WorkingSet::holds_out`, `question_pending`) | There is one reading; the seeds are in it, as background. |
| The shadow scan's windows no longer than the query's position | One context at the forced rung. |

What stays: the pool and its scan leaders, the fill and its caps, the hit
counters and momentum, the early boundaries at 8, 16 and 32 reasoning tokens,
the first question's opening, the seeds, and the fill report in every
projection record.

## 4. Build

1. Remove the request's two readings (§3).
2. `IndexCache::rewind_rows(n)` — the live tail's last `n` rows, refused across
   a closed page; with the K/V tail's `truncate_sequence_to_tokens`, one
   `rewind_tail(seq, n)` on the model.
3. The gap holds the current and previous writer chunks native.
4. The re-feed row: after a projection rebuilds a turn's view, its last
   `REFEED · n` tokens are rewound and queued as a row of the next decode wave
   (`stencil_wave`), 8 a wave; the token after it is sampled from its last row.
5. The re-feed's queries kept for the scoring job; the scoring job's rung forced
   (`ShadowJob`), the ×8 rung added to the schedule.
6. The battery against 62.2% (`dependency_battery_2026-11-20_b.md`).

Each step lands with its unit tests: the rewind against raw rows and offsets,
the gap's hold against the chunks it takes, the re-feed row's tokens and
positions, the forced rung's rotation table against the rung's own.

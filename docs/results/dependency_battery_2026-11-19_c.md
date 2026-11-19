# The §9.12 dependency battery — production, 2026-11-19 (third run)

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run a third time on BlackWorld, on `46c0e6456`: the afternoon's build (`dependency_battery_2026-11-19_b.md`)
plus two fixes to the file corpus. `b490eec04` stops the file reads queued behind a crawling creep
cohort from being given up after 60 s, so every file the day's commits changed was re-read (the
`code_reading` plan read `queued=0` before the run); `46c0e6456` keeps a file whose new version is
not yet read in scope at its path's latest reading (`docs/zend_branch_ingest.md` §8.2). The second
run's d1 was declined because `conversation.rs`, changed that afternoon, was out of every scope.
WhiteWorld was not serving.

## How it was run

As on every production run: through the connector's private chats, one fresh chat per question,
one at a time, closed (deleted) once answered, every question prefixed *"Answer from your memory of
the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`): an item
counts when any of its spellings is a case-insensitive substring of the text after the reasoning
block.

## Results

| Category | Mean recall | Full marks | 11-19 (afternoon) | 11-19 (morning) | 11-18 | 11-11 | 11-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 41.7% | 2 of 8 | 39.6%, 2 | 35.4%, 1 | 24.0%, 1 | 27.1%, 0 | 33.3%, 1 |
| Transitive | 52.1% | 2 of 8 | 56.3%, 2 | 44.8%, 2 | 45.8%, 1 | 37.5%, 1 | 54.2%, 2 |
| Architectural | 85.4% | 5 of 8 | 72.9%, 4 | 72.9%, 5 | 68.8%, 4 | 68.8%, 5 | 85.4%, 5 |
| **All 24** | **59.7%** | **9** | 56.3%, 8 | 51.0%, 8 | 46.2%, 6 | 44.4%, 6 | 57.6%, 8 |

| Id | Recall | Afternoon | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | declined; no part of `recorded_reply.rs` or `conversation.rs` placed |
| d2 | 0/2 | 1/2 | `region_pool::tier_fits` from `AGENTS.md`, never `region_pool.rs` |
| d3 | 2/2 | 2/2 | `persistence/elevate.rs` from cross-references; the scheduler as caller |
| d4 | 1/6 | 1/6 | declined; `resolver.rs` in a list of the projection's files |
| d5 | 0/4 | 0/4 | declined; `bucketize` from `AGENTS.md`, no files |
| d6 | 2/2 | 1/2 | definition in `expert_lre/cache.rs`, only a test caller seen; `pipeline.rs` in a list of files not read |
| d7 | 2/3 | 3/3 | `= 2`, `cache.rs`; no loader named |
| d8 | 1/2 | 0/2 | declined; the scheduler imports and uses `FoldParams` |
| t1 | 1/4 | 0/4 | declined; `FoldParams` named as where to look |
| t2 | 2/4 | 2/4 | `replay.rs`, `recorded_turn`; stopped before the consumer |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager word → cold wait, with `dispatch.rs` lines |
| t5 | 2/3 | 1/3 | `PINNED_LAYERS`, `batch_victims` (not `best_victim`), `startup.rs` named but not read |
| t6 | 0/4 | 1/4 | declined |
| t7 | 0/3 | 3/3 | declined; `projection_assembler.rs` not read |
| t8 | 3/3 | 2/3 | scheduler's tier prefetch → `elevate_to_hot` in `persistence/elevate.rs` |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 2/3 | 2/3 | **declined**; "recurrent" and "gap-fill" in its account of what it had read |
| a4 | 2/2 | 0/2 | stored un-rotated, RoPE applied at read |
| a5 | 2/2 | 2/2 | invariant 3, `bucketize` |
| a6 | 1/2 | 1/2 | refused via `decode_wide_sigs_checked`; "fold parameters", not `FoldParams` |
| a7 | 2/3 | 2/3 | **declined**; "prefill" and "think" in what it had read; `TurnLayout` does not match `turn_layout` |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**59.7% and nine full marks, the best run since the battery moved to production — but inside the
noise of the afternoon's 56.3%.** Scored strictly — the two declined architectural answers (a3, a7)
at zero and d6's `pipeline.rs` dropped — All 24 reads 52.1%, against about 50.7% on the same basis in
the afternoon. Questions moved both ways again: a4, t8, t5, d6, d8 gained; t7, t6, d2 lost.

**The corpus fix shows in what could be placed, not in d1.** Every file changed by the day's commits
was re-read before the run, so no answer was out of scope for that reason; d1 was declined because
the projection placed neither of its two files, the selection limit
(`code-retrieval-is-vote-starved`), not the scope. Run straight after the fix, before the re-reads
had finished being the reason, the same question placed `conversation.rs` and scored 2/2.

**The model stayed honest.** Every miss was a decline or an adjacent answer that named what it had
read; no answer invented a path. a7's reply carried part of its reasoning ahead of its `</think>` —
the reasoning block's opening was lost from the stream — and is scored, as the rule says, on the
text after it.

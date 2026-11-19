# The §9.12 dependency battery — production, 2026-11-19 (second run)

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run again on the day's later build: BlackWorld on `0aacc85f3`, which adds shared journal sections
and their pre-warmer (74a835cb5, `docs/shared_journal_sections.md`) and the per-job titler
(0aacc85f3) to the build of the morning run (`dependency_battery_2026-11-19.md`). WhiteWorld was
not serving; every chat was served by BlackWorld.

## How it was run

As on every production run: through the connector's private chats, one fresh chat per question,
one at a time, closed (deleted) once answered, every question prefixed *"Answer from your memory of
the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`): an item
counts when any of its spellings is a case-insensitive substring of the text after the reasoning
block.

## Results

| Category | Mean recall | Full marks | 11-19 (morning) | 11-18 | 11-11 | 11-10 |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 39.6% | 2 of 8 | 35.4%, 1 | 24.0%, 1 | 27.1%, 0 | 33.3%, 1 |
| Transitive | 56.3% | 2 of 8 | 44.8%, 2 | 45.8%, 1 | 37.5%, 1 | 54.2%, 2 |
| Architectural | 72.9% | 4 of 8 | 72.9%, 5 | 68.8%, 4 | 68.8%, 5 | 85.4%, 5 |
| **All 24** | **56.3%** | **8** | 51.0%, 8 | 46.2%, 6 | 44.4%, 6 | 57.6%, 8 |

| Id | Recall | Morning | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 0/2 | 2/2 | declined |
| d2 | 1/2 | 1/2 | `region_pool.rs`, from `AGENTS.md`; not `span_geometry.rs` |
| d3 | 2/2 | 1/2 | `persistence/elevate.rs`, the scheduler as caller, from `ARCHITECTURE.md` |
| d4 | 1/6 | 1/6 | `resolver.rs` only |
| d5 | 0/4 | 0/4 | declined; named `bucketize` from `AGENTS.md`, no files |
| d6 | 1/2 | 0/2 | had read `expert_lre/cache.rs`, found no `set_eviction_costs`; declined |
| d7 | 3/3 | 2/3 | `= 2`, `cache.rs`, `layer_loader.rs` |
| d8 | 0/2 | 0/2 | declined |
| t1 | 0/4 | 0/4 | the capture path and kernels from `ARCHITECTURE.md`; none of the four items |
| t2 | 2/4 | 1/4 | `replay.rs`, `recorded_turn`; stopped before the consumer |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager word → cold wait |
| t5 | 1/3 | 1/3 | `PINNED_LAYERS`, read `cache.rs` in detail; named `batch_victims`, not `best_victim`; no `startup.rs` |
| t6 | 1/4 | 1/4 | `REPROJECT_EVERY_N_TOKENS` from a guide; declined the chain |
| t7 | 3/3 | 0/3 | `SlotState.glue_islands`, keyed by a content hash, `GLUE_ISLAND_RETAIN_GENERATIONS = 4` — offered with a doubt that it was what was asked |
| t8 | 2/3 | 3/3 | `elevate_to_hot` in `persistence/elevate.rs`; said "the projection path", not the scheduler |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 2/3 | 1/3 | **declined**; "recurrent" and "gap-fill" appear in its account of what it had not read |
| a4 | 0/2 | 2/2 | answered with chunk ids and arena indirection, not un-rotated keys |
| a5 | 2/2 | 2/2 | invariant 3, `bucketize` |
| a6 | 1/2 | 1/2 | refused via `decode_wide_sigs_checked`, "fold parameters", not `FoldParams` |
| a7 | 2/3 | 0/3 | **declined**; "prefill" and "think" appear in what it had read |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**56.3% against 51.0% in the morning, eight full marks both times — inside the noise.** Two
architectural items (a3, a7) score 2/3 on answers that declined, because the rule matches words in
the model's account of what it had not read; without them All 24 would read about 50.7%. Questions
moved both ways again: d3, d7, t2, t7 gained; d1, a4 and t8 lost.

**The model stayed honest.** Every miss was a decline or an adjacent answer that said what it had
read; no answer invented a path. The answers that scored were read from `AGENTS.md`,
`ARCHITECTURE.md` and the few source files the working set placed (`expert_lre/cache.rs`,
`resolver.rs`, `session/replay.rs`); the Direct misses are still the files those hub texts do not
hold.

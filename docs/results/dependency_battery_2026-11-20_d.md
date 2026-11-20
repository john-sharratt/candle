# The §9.12 dependency battery — production, 2026-11-20 (d): no boundary at the opening's end

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `0ffde9265`: (c)'s re-feed, with the boundary at the end of a request's
opening removed and the reasoning's first early boundary moved from decoded token 8 to 9
(`docs/reprojection_refeed.md` §2.4) — the reply decodes on the slot its question was
prefilled against, and the first projection re-feeds the reasoning's first 8 tokens over the
slot it chooses. One change against (c).

WhiteWorld was not serving.

## How it was run

As on every production run: one fresh connector chat per question, one at a time, deleted once
answered, every question — in the battery file's exact wording — prefixed *"Answer from your memory
of the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`:
case-insensitive substrings, so a name mentioned in a decline counts).

## Results

| Category | Mean recall | Full marks | 11-20 (c) | 11-20 (b) | 11-20 (a) |
|---|---:|---:|---:|---:|---:|
| Direct | 31.3% | 1 of 8 | 27.1%, 1 | 47.9%, 2 | 38.5%, 0 |
| Transitive | 51.0% | 1 of 8 | 43.8%, 2 | 51.0%, 1 | 50.0%, 3 |
| Architectural | 79.2% | 5 of 8 | 87.5%, 6 | 87.5%, 6 | 83.3%, 5 |
| **All 24** | **53.8%** | **7** | 52.8%, 9 | 62.2%, 9 | 57.3%, 8 |

| Id | Recall | (c) | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 1/2 | 0/2 | read `recorded_reply.rs`; no caller |
| d2 | 0/2 | 2/2 | declined; `region_pool::tier_fits` from AGENTS.md only |
| d3 | 2/2 | 0/2 | `persistence/elevate.rs`; "the scheduler's elevate step" from journals |
| d4 | 0/6 | 2/6 | declined; had read one journal note about npcd |
| d5 | 0/4 | 0/4 | declined; `bucketize` only from AGENTS.md |
| d6 | 1/2 | 1/2 | read `cache.rs`, the definition; no caller |
| d7 | 0/3 | 1/3 | declined |
| d8 | 1/2 | 0/2 | read `wide_sig.rs`, the definition and its tests; no caller |
| t1 | 1/4 | 0/4 | declined; `FoldParams` from a journal note, ARCHITECTURE.md, AGENTS.md |
| t2 | 2/4 | 2/4 | read `replay.rs`: `recorded_turn`; not `conversation.rs` |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager → cold, from AGENTS.md |
| t5 | 2/3 | 0/3 | read `cache.rs`: `PINNED_LAYERS = 2`, `batch_victims`; `startup.rs` guessed |
| t6 | 1/4 | 1/4 | read `scheduler/mod.rs`: `ReprojectionPolicy`, `BoundaryRule`; not the config fields |
| t7 | 0/3 | 0/3 | declined; had read `scheduler/mod.rs` |
| t8 | 2/3 | 3/3 | `elevate_to_hot` in `persistence/elevate.rs`; named the substrate, not the scheduler |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 3/3 | 2/3 | recurrent, gap-fill, `reserve_glue_island` |
| a4 | 1/2 | 2/2 | "pre-rotation", RoPE at read — not a spelling the rule accepts |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 1/2 | 2/2 | refused; "fold parameters", never `FoldParams` |
| a7 | 1/3 | 1/3 | declined; "think" incidental |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**53.8% with 7 full marks against (c)'s 52.8% with 9: no measurable change.** Direct and
transitive rose a few points (27.1% → 31.3%, 43.8% → 51.0%) and the architectural set fell
(87.5% → 79.2%). Two of those architectural losses are spelling: a4 said "pre-rotation" and a6
"fold parameters", correct answers the substring rule does not accept. Run to run, single
questions move a full mark in either direction (d2, d3 and t5 each swapped), the spread these
runs have always had.

**The declines still name journals, guides and AGENTS.md as what was in context.** d2, d4, d5,
d7, t1 and t7 each say so. More answers name a source file than in (c): d1, d6, d8, t2, t5 and t6
each read one, but it is the definition's file and never the caller's. Moving the first projection
from the opening's end to reasoning token 9 removes the boundary that re-feeds nothing, and does
not recover (b)'s 62.2%. In (b) the question itself was prefilled again over the slot chosen for
it; here the re-feed is the reasoning's first 8 tokens, and the question is still never read over
the files placed for it.

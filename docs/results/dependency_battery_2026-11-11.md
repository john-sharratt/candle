# The §9.12 dependency battery — production, 2026-11-11

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run again after the replication repair: both members on build `4813-cb6116ba`, joined and
in sync — WhiteWorld holding 167 of BlackWorld's turns pending, down from 81,517.

## How it was run

As on 2026-11-10 (`docs/results/dependency_battery_2026-11-10.md`): through the connector's
private chats, one fresh chat per question, closed (deleted) once answered, every question
prefixed *"Answer from your memory of the repository only — do not call any tool, do not
read or search any file. If you do not remember, say so."* No answer called a tool. Scored
by the battery's rule (`score.rs`).

## Results

| Category | Mean recall | Full marks | 2026-11-10 |
|---|---:|---:|---:|
| Direct | 27.1% | 0 of 8 | 33.3%, 1 |
| Transitive | 37.5% | 1 of 8 | 54.2%, 2 |
| Architectural | 68.8% | 5 of 8 | 85.4%, 5 |
| **All 24** | **44.4%** | **6** | 57.6%, 8 |

| Id | Recall | 11-10 | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 0/2 | 1/2 | `recorded_reply` as a zend module; "only the zend crate's module list" |
| d2 | 1/2 | 0/2 | `region_pool`, from `CLAUDE.md` invariant 7 |
| d3 | 1/2 | 1/2 | `elevate.rs` from `docs/architecture.md`; caller given as `persistence/transfer.rs` |
| d4 | 4/6 | 1/6 | `scan.rs`, `gpu.rs`, `packed.rs`, `resolver.rs`, from the provenance design doc |
| d5 | 0/4 | 2/4 | nothing named; "nothing in this conversation has named `moe_bucketize`" |
| d6 | 1/2 | 2/2 | `pipeline.rs`, from the module map; declined the definition |
| d7 | 0/3 | 0/3 | declined; had read `zend/src/lib.rs` |
| d8 | 0/2 | 0/2 | declined; had read `docs/performance_series.md` |
| t1 | 0/4 | 2/4 | `wide_sig.rs`, the BDP kernels; no fold group names |
| t2 | 1/4 | 1/4 | `recorded_reply`; had read `zenj/src/lib.rs` and an archived plan |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; CUDA side in `arena_table.cuh` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager word → cold wait |
| t5 | 1/3 | 1/3 | `PINNED_LAYERS`, from an archived results doc |
| t6 | 0/4 | 2/4 | declined; had read `zend/src/lib.rs` and a directory listing |
| t7 | 0/3 | 0/3 | declined; named other caches from `docs/gallery_dedup.md` |
| t8 | 2/3 | 3/3 | scheduler → `elevate_to_hot`; not `elevate.rs` |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `tier_fits` |
| a3 | 0/3 | 2/3 | declined; had read `CLAUDE.md` and two **yenbot** files |
| a4 | 0/2 | 2/2 | declined; had read `host/src/download.rs` and `docs/zend_working_set.md` |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 1/2 | 1/2 | "mismatch" only; the fold described from `zend_working_set.md` |
| a7 | 3/3 | 2/3 | **leaked**: the items came from the 2026-11-10 results line for a7, which it read |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

The model is honest about what it holds: no answer invented a path it then asserted, and
each declined where it had not seen the source. What it holds is the problem. Answer after
answer names the same few pieces as the whole of its context — `zend/src/lib.rs` and
`docs/zend_working_set.md` in eight of the 24, regardless of the question — and others drew
pieces unrelated to the question (`docs/performance_series.md` for `FoldParams`, two
`yenbot` files for a projection question, `host/src/download.rs` for key storage). The
code-reading layer's file turns that answer the direct questions were not in view.

So retrieval is up but not selecting: a few hub pieces win every projection, and the
answers follow what was projected. The battery's own results file being retrievable is a
second defect — a7's full marks are read off the previous run's score line.

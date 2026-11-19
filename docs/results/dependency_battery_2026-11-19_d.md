# The §9.12 dependency battery — production, 2026-11-19 (fourth run, recalibrated answer files)

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `14867aadf`, after the calibrations of its 33 answer files were retired and
written again. Every calibration drafted from 2026-11-15 ~21:30 UTC had been written blind — the
draft, an append-only fork of the ingest base parented on the file's reading, was framed by the
system prompt alone (1b403a77a), so it was asked for a file's questions knowing only its name.
`14867aadf` makes an append-only conversation's frame carry its lineage (`conversation::Frame`),
and every draft now logs what of its reading it holds (`calibration draft projected`): of the 33,
every reading up to 44K tokens was held whole (`held` = `reading`), 26 calibrations were sealed and
spot-checked against their source, and the readings over `INHERITED_CHAIN_TOKEN_CAP` (64,000 tokens)
held nothing and were refused — `quantized/cuda.rs` (144K), `region_pool.rs` (97K), and with them
`conversation.rs`, `resolver.rs`, `scheduler/mod.rs` and `projection_assembler.rs`, which are left
with no calibration. WhiteWorld was not serving.

## How it was run

As on every production run: one fresh connector chat per question, one at a time, deleted once
answered, every question prefixed *"Answer from your memory of the repository only — do not call any
tool, do not read or search any file. If you do not remember, say so."* No answer called a tool.
Scored by the battery's rule (`score.rs`).

## Results

| Category | Mean recall | Full marks | 11-19 (c) | 11-19 (b) | 11-19 (a) | 11-18 | 11-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 31.3% | 1 of 8 | 41.7%, 2 | 39.6%, 2 | 35.4%, 1 | 24.0%, 1 | 33.3%, 1 |
| Transitive | 41.7% | 2 of 8 | 52.1%, 2 | 56.3%, 2 | 44.8%, 2 | 45.8%, 1 | 54.2%, 2 |
| Architectural | 77.1% | 5 of 8 | 85.4%, 5 | 72.9%, 4 | 72.9%, 5 | 68.8%, 4 | 85.4%, 5 |
| **All 24** | **50.0%** | **8** | 59.7%, 9 | 56.3%, 8 | 51.0%, 8 | 46.2%, 6 | 57.6%, 8 |

| Id | Recall | What the answer named, and what it said it had read |
|---|---:|---|
| d1 | 1/2 | had read `recorded_reply.rs` whole; no caller |
| d2 | 0/2 | `region_pool::tier_fits` from `AGENTS.md` |
| d3 | 2/2 | `persistence/elevate.rs`; the scheduler as caller |
| d4 | 0/6 | declined |
| d5 | 0/4 | declined; `bucketize` from `AGENTS.md` |
| d6 | 1/2 | declined; `cache.rs` in a list of files not read |
| d7 | 0/3 | had read `expert_lre/eval.rs`, which imports `PINNED_LAYERS`; declined its value |
| d8 | 1/2 | declined; the scheduler imports `FoldParams` |
| t1 | 0/4 | the band thirds and `fold_provenance` from a journal note and `ARCHITECTURE.md` |
| t2 | 1/4 | declined; `recorded_reply` named |
| t3 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | read `dispatch.rs`: bucketize → mapped ring → stager word → cold wait |
| t5 | 1/3 | declined; `PINNED_LAYERS` from a journal note |
| t6 | 0/4 | declined |
| t7 | 0/3 | declined; had read lines 1–2621 of `scheduler/mod.rs`, not `projection_assembler.rs` |
| t8 | 3/3 | `elevate_to_hot` in `persistence/elevate.rs`, the scheduler's elevation |
| a1 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 2/3 | **declined**; "recurrent", "gap-fill" incidental |
| a4 | 2/2 | un-rotated keys, RoPE at read |
| a5 | 2/2 | invariant 3, `bucketize` |
| a6 | 1/2 | refusal via `decode_wide_sigs_checked`, not `FoldParams` |
| a7 | 0/3 | declined |
| a8 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**Recalibrating the answer files moved nothing measurable: 50.0% against 59.7% and 56.3% on the
blind calibrations, the spread of the earlier runs.** The calibrations are right now and the scores
did not follow, so what holds the battery back is not what calibration supplies. Two things recur in
the answers. The code share is still filled by a file's opening pages taken in order — t7 read lines
1–2621 of `scheduler/mod.rs` — rather than by the pages a question points at. And six of the
battery's answer files are past the 64,000-token lineage cap, so their drafts cannot see them: they
have no calibration at all.

**The model stayed honest.** Every miss was a decline or a nearby answer naming what it had read.

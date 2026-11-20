# The §9.12 dependency battery — production, 2026-11-20 (h): the request read twice, on the min/max fill

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `b616c4d78`, against (f) — one change, the request's two readings back
(`docs/hit_driven_projection.md` §7.4, `scheduler/first_reading.rs`):

- the question is prefilled first against the projection with the seeds held out, scored like a
  boundary and discarded;
- the turn is submitted again, its opening ranked by the first reading's scores — the seeds and
  the question's files in one ranking — and the question is prefilled the second time over them
  before the reply decodes.

Everything else as in (f): the reading's minimum and maximum in the fill, the cadence alone
counted from the last boundary, no boundary inside a tool call or at the opening's end, the
re-feed at every later projection (the re-decode of (g) reverted, `2918def1a`). WhiteWorld was not
serving. The chats were kept open for review.

## Results

| Category | Mean recall | Full marks | 11-20 (g) | 11-20 (f) | 11-20 (b) |
|---|---:|---:|---:|---:|---:|
| Direct | 37.5% | 1 of 8 | 22.9%, 0 | 25.0%, 1 | 47.9%, 2 |
| Transitive | 66.7% | 3 of 8 | 50.0%, 3 | 59.4%, 2 | 51.0%, 1 |
| Architectural | 68.8% | 4 of 8 | 64.6%, 4 | 68.8%, 4 | 87.5%, 6 |
| **All 24** | **57.6%** | **8** | 45.8%, 7 | 51.0%, 7 | 62.2%, 9 |

| Id | Recall | (f) | What the answer said |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | `recorded_reply`'s signature quoted, given to `scheduler/mod.rs`; no caller |
| d2 | 0/2 | 0/2 | `region_pool` from AGENTS.md; declined the path |
| d3 | 1/2 | 1/2 | `elevate_to_hot`'s signature and header quoted, given to `scheduler/mod.rs` |
| d4 | 1/6 | 0/6 | `resolver.rs` imports and calls it; definition not read |
| d5 | 2/4 | 0/4 | `quantized/cuda.rs`, `moe_bucketize.cu`, with lines; `expert_lre/` for the caller |
| d6 | 1/2 | 1/2 | `cache.rs`, the method and its passes; no caller |
| d7 | 1/3 | 3/3 | `cache.rs` imports it; value not read |
| d8 | 2/2 | 0/2 | `wide_sig.rs`, the definition and its tests; caller "the scheduler" by guess |
| t1 | 3/4 | 2/4 | `FoldParams`, `FOLD_GROUPS`, `heads_per_group` in a name list that loops |
| t2 | 2/4 | 2/4 | `replay.rs` → `TurnOptions::recorded_turn`; the forcing function not read |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table.cuh` and the tables; no `blocks.cuh` |
| t4 | 4/4 | 4/4 | the whole chain, in detail |
| t5 | 1/3 | 1/3 | `PINNED_LAYERS = 2`, `rank_victims`, the floor; startup not read |
| t6 | 4/4 | 4/4 | both config fields, `boundary_rule`, `ReprojectionPolicy` |
| t7 | 0/3 | 0/3 | declined |
| t8 | 3/3 | 2/3 | `elevate_to_hot` in `elevate.rs`, `lift_warm_to_gpu`, `install_promoted` |
| a1–a8 | 3,2,2,0,2,1,1,2 | 3,2,2,0,2,1,1,2 | unchanged |

## What the answers show

**57.6% with 8 full marks, against (f)'s 51.0% with 7** — the second-best run after (b)'s 62.2%.
Direct rose (25.0% → 37.5%: d4, d5, d8), transitive rose again (59.4% → 66.7%: t1, t8).

**The answers read the files they were given.** Where (f)'s answers declined on an inventory of
seeds ("none of these files mention `FoldParams::derive`"), (h)'s open on the answer file and
quote it: `derive`'s location and its three tests in `wide_sig.rs` (d8), `moe_bucketize`'s wrapper
and kernel with their lines (d5), `set_eviction_costs`'s passes in `cache.rs` (d6),
`PINNED_LAYERS = 2` and the victim filter (t5). With the question prefilled over its files, the
reasoning's first inventory includes them.

**What is left is the second hop and the file's name.** The answers decline the caller — the
file that calls the definition — which is a second file the question does not name (d1, d2, d6,
d7, t2, t5). And three answers give the definition's content to a file beside it: d1's
`recorded_reply` and d3's `elevate_to_hot` to `scheduler/mod.rs`, which was placed with them.

**t1 loops on its list of names**, as in (g), here without a re-decode — `fold_provenance_fitted`
repeated a few hundred times. (g)'s seams (8 answers with a broken name, here 1) were the
re-decode's; this loop is not.

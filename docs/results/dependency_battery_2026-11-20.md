# The §9.12 dependency battery — production, 2026-11-20 (after the full recalibration)

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `60f7a9998` (the commits after it are documentation), after the whole
substrate was recalibrated overnight. Three changes stand between this run and 11-19 (d):

- **Hits count across a reprojection** (`dace103fb`). A reprojected slot moved its recurrent state
  to a new holder and left the hit counters behind, so every piece's hit momentum read zero after
  the first boundary; the counters now travel with the slot (`hits::move_holder`) and a fork shares
  its parent's (`hits::fork_holder`).
- **A reading past the lineage cap is drafted over windows** (`2480a1974`). The six answer files
  over 64,000 tokens — `conversation.rs`, `resolver.rs`, `scheduler/mod.rs`,
  `projection_assembler.rs`, `quantized/cuda.rs`, `region_pool.rs` — had no calibration on 11-19
  (d); each is now calibrated, `projection_assembler.rs`'s spot-checked at 36 of 37 identifiers
  real.
- **The whole substrate recalibrated.** The 1,227 calibrations drafted blind (2026-11-15 to
  11-19) were retired and drafted again: 782 sealed and 20 failed on the first pass, 258 sealed
  and 5 failed on the second.

WhiteWorld was not serving.

## How it was run

As on every production run: one fresh connector chat per question, one at a time, deleted once
answered, every question prefixed *"Answer from your memory of the repository only — do not call any
tool, do not read or search any file. If you do not remember, say so."* No answer called a tool.
Scored by the battery's rule (`score.rs`).

## Results

| Category | Mean recall | Full marks | 11-19 (d) | 11-19 (c) | 11-19 (b) | 11-19 (a) | 11-18 | 11-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 38.5% | 0 of 8 | 31.3%, 1 | 41.7%, 2 | 39.6%, 2 | 35.4%, 1 | 24.0%, 1 | 33.3%, 1 |
| Transitive | 50.0% | 3 of 8 | 41.7%, 2 | 52.1%, 2 | 56.3%, 2 | 44.8%, 2 | 45.8%, 1 | 54.2%, 2 |
| Architectural | 83.3% | 5 of 8 | 77.1%, 5 | 85.4%, 5 | 72.9%, 4 | 72.9%, 5 | 68.8%, 4 | 85.4%, 5 |
| **All 24** | **57.3%** | **8** | 50.0%, 8 | 59.7%, 9 | 56.3%, 8 | 51.0%, 8 | 46.2%, 6 | 57.6%, 8 |

| Id | Recall | What the answer named, and what it said it had read |
|---|---:|---|
| d1 | 0/2 | declined |
| d2 | 1/2 | `region_pool.rs` only in a list of files not read |
| d3 | 1/2 | `elevate.rs`; no caller |
| d4 | 1/6 | read `resolver.rs`, which imports the function |
| d5 | 1/4 | **read `dispatch.rs`**, the caller, and traced the import to `candle::quantized::cuda` — 0/4 on every earlier run |
| d6 | 1/2 | `cache.rs` with the exact definition; no caller |
| d7 | 2/3 | `= 2`, `cache.rs`; not `layer_loader.rs` |
| d8 | 1/2 | the scheduler; not the defining file |
| t1 | 1/4 | `FoldParams` |
| t2 | 0/4 | declined |
| t3 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | bucketize → mapped ring → stager word → cold wait |
| t5 | **3/3** | **read `cache.rs` whole** — the constant, the victim filter and the startup fill with line references; 1/3 on every earlier run |
| t6 | 0/4 | the question's words drew Battle Cities' documents |
| t7 | 0/3 | `projection_assembler.rs` is calibrated now but was not placed |
| t8 | 3/3 | `elevate_to_hot` in `persistence/elevate.rs`, the scheduler's elevation |
| a1 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | **3/3** | **answered** — `reserve_glue_island`, `can_gap_fill`; declined on every 11-19 run |
| a4 | 1/2 | RoPE at read; never that keys are stored un-rotated |
| a5 | 2/2 | invariant 3, `bucketize` |
| a6 | 1/2 | the refusal; not `FoldParams` |
| a7 | 2/3 | `think`, `build_turn_layout`; not the re-prefill |
| a8 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**57.3% against 50.0% the day before, inside the spread of the earlier runs (50.0–59.7%) — but the
gains are of a different kind.** Three answers that had never been right were right because the
projection placed the file the question was about and the model read it: d5 (`dispatch.rs`), t5
(`cache.rs`, whole, with line references) and a3 (the glue-island answer, declined on every 11-19
run). That is what the hit fix is for: a file's read pages now keep their momentum past the first
reprojection boundary. The losses are on questions whose answer file was not placed at all (d1, t2,
t6, t7), which the scores of the placed files cannot reach.

**What still holds the battery back is placement, not calibration.** t7 is the clearest case: its
answer file is calibrated now, and the projection still did not choose it. t6's words matched Battle
Cities' documents instead of the code. Both are the down-select — which pieces enter the pool before
the fill scores them — and not the scoring formulas.

**The model stayed honest.** Every miss was a decline or a nearby answer naming what it had read.

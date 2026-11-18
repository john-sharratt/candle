# The §9.12 dependency battery — production, 2026-11-19

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run again after the pool-first projection revision went to production (8088a2457,
`docs/hit_driven_projection.md` §12 items 18–25): the working set is the pool, chosen at every
observation by strength within the card's tokens; caps are shares of the room (readings 22.5%,
folders 2%, guides 5%, journals 10.5%, dialogue 60%); rounds score on the band; a third of each
cap may enter per boundary. The lock tier had gone the day before (1f5ad2746), and the reasoning
window became the request's newest eight rounds (e74ade355). BlackWorld on build
`5211-8088a2457`; WhiteWorld on `5207-e74ade355` and held out of serving, so the chats were
served by BlackWorld.

## How it was run

As on 2026-11-10, 11-11 and 11-18: through the connector's private chats, one fresh chat per
question, closed (deleted) once answered, every question prefixed *"Answer from your memory of
the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`): an item
counts when any of its spellings is a case-insensitive substring of the text after the reasoning
block.

## Results

| Category | Mean recall | Full marks | 2026-11-18 | 2026-11-11 | 2026-11-10 |
|---|---:|---:|---:|---:|---:|
| Direct | 35.4% | 1 of 8 | 24.0%, 1 | 27.1%, 0 | 33.3%, 1 |
| Transitive | 44.8% | 2 of 8 | 45.8%, 1 | 37.5%, 1 | 54.2%, 2 |
| Architectural | 72.9% | 5 of 8 | 68.8%, 4 | 68.8%, 5 | 85.4%, 5 |
| **All 24** | **51.0%** | **8** | 46.2%, 6 | 44.4%, 6 | 57.6%, 8 |

| Id | Recall | 11-18 | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 2/2 | 0/2 | both items |
| d2 | 1/2 | 0/2 | `region_pool.rs`; not the second file |
| d3 | 1/2 | 0/2 | `elevate.rs`; named the persistence thread, not the scheduler |
| d4 | 1/6 | 2/6 | `resolver.rs` only |
| d5 | 0/4 | 1/4 | declined |
| d6 | 0/2 | 2/2 | declined |
| d7 | 2/3 | 1/3 | the value `2`, `cache.rs`; no loader |
| d8 | 0/2 | 0/2 | declined |
| t1 | 0/4 | 3/4 | declined |
| t2 | 1/4 | 2/4 | `recorded_turn`; not the harness |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager word → cold wait |
| t5 | 1/3 | 0/3 | `PINNED_LAYERS`; **read `cache.rs` in detail** but named neither `best_victim` nor `startup.rs` |
| t6 | 1/4 | 0/4 | `ReprojectionPolicy` via `build_reprojection_policy` → `submit_turn_on` → `submit_prefill_unit`, `disable_reprojection`; the cadence from a guide's prose, not its fields |
| t7 | 0/3 | 0/3 | declined; had read a journal note and `README.md` / `ARCHITECTURE.md` |
| t8 | 3/3 | 2/3 | scheduler loop notices, `elevate_to_hot` in `persistence/elevate.rs`, from `ARCHITECTURE.md` and `AGENTS.md` |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 1/3 | 2/3 | gap-fill; **read `wave_plan.rs`** and reasoned from `HyperWidths` (no single-stream seam) rather than recurrence; declined the refusal |
| a4 | 2/2 | 0/2 | un-rotated K, re-RoPE'd at read from the chunk's `rope_base` |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 1/2 | 1/2 | refused with `warn_fold_mismatch` through `decode_wide_sigs_checked`, group sizes only — complete and correct, says "fold parameters", not `FoldParams` |
| a7 | 0/3 | 1/3 | declined; had read a guide and `battle-cities` files |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**The pool-first revision moved the battery up, inside the noise.** 51.0% against 46.2% the day
before, eight full marks against six: Direct rose most (24.0% → 35.4%), Architectural a little,
Transitive held. On 24 questions a swing of this size is within what two runs of the same build
could show, and individual questions moved both ways — d6 and t1, which read their files on
11-18, declined today, while d1, a4 and t8 came up to full marks.

**The answers lean on wider reading.** Where an answer drew on source, it was often a large file
read in detail — `cache.rs` (t5), `wave_plan.rs` (a3), `wide_sig.rs`'s fold check (a6) — rather
than the hub documents alone. The live log of the same build shows why: the pool ranks pieces by
strength summed over their pages, so a large file collects more strength than a small one that
answers the question better. `resolver.rs` (264 pages, 132K tokens) ranked second among readings
in one observation. The fill's new per-cap log (`fill: each cap's use`) shows the readings cap at
50,788 of 52,978 tokens, so one such file takes a large share of the room it competes for.

**The hub documents still carry the architectural answers.** `AGENTS.md` and `ARCHITECTURE.md`
answered t4, t8, a2, a5 and a8, which is why Architectural stays the strongest category; the
Direct and Transitive questions need a specific file that the hub texts do not hold, and most of
their misses are honest declines.

The model stayed honest throughout: every miss was a decline or an adjacent answer that said what
it had and had not read, and no answer invented a path.

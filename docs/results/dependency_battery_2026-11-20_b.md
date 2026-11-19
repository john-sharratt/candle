# The §9.12 dependency battery — production, 2026-11-20 (b): a request read twice

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `2f8c622ca`, after the day's projection changes on top of the overnight
recalibration measured in `dependency_battery_2026-11-20.md`:

- **The fill in the record** (`8d80eea20`): every projection record carries each offered piece's
  scores, what it placed and its standing in the pool, and the strongest scanned pieces the pool
  does not hold (`projection/fill_report.rs`).
- **The scan's leaders first** (`8d80eea20`): each kind's pool minimum is filled by the boundary's
  scan leaders before hit-lifted strength.
- **Early boundaries** (`8d80eea20`): a reply still reasoning reprojects at 8, 16 and 32 tokens.
- **The seeds kept, and a first question opens its slot** (`3dee9051d`): the seeds stay as
  background; a dialogue's first question takes its opening whether or not its set exists yet.
- **A request is read twice** (`123ca356c`, `40d0d2bdb`, `2f8c622ca`): the question is first
  prefilled to its last token against the projection with the seeds held out and scored; that
  reading is discarded and the turn submitted again, its opening ranked by the first reading's
  scores beside the seeds, and it only prefills before decoding
  (`docs/hit_driven_projection.md` §7.4).

WhiteWorld was not serving.

## How it was run

As on every production run: one fresh connector chat per question, one at a time, deleted once
answered, every question — in the battery file's exact wording — prefixed *"Answer from your memory
of the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`).

## Results

| Category | Mean recall | Full marks | 11-20 (a) | 11-19 (d) | 11-19 (c) | 11-19 (b) | 11-19 (a) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct | 47.9% | 2 of 8 | 38.5%, 0 | 31.3%, 1 | 41.7%, 2 | 39.6%, 2 | 35.4%, 1 |
| Transitive | 51.0% | 1 of 8 | 50.0%, 3 | 41.7%, 2 | 52.1%, 2 | 56.3%, 2 | 44.8%, 2 |
| Architectural | 87.5% | 6 of 8 | 83.3%, 5 | 77.1%, 5 | 85.4%, 5 | 72.9%, 4 | 72.9%, 5 |
| **All 24** | **62.2%** | **9** | 57.3%, 8 | 50.0%, 8 | 59.7%, 9 | 56.3%, 8 | 51.0%, 8 |

| Id | Recall | What the answer named, and what it said it had read |
|---|---:|---|
| d1 | 0/2 | declined; had read a guide, the Battle Cities README and journal files |
| d2 | 2/2 | `region_pool.rs` read, `tier_fits` imported from `span_geometry` |
| d3 | 2/2 | `persistence/elevate.rs` from `substrate.rs`'s cross-references; the scheduler |
| d4 | 1/6 | read `resolver.rs`, which imports and calls `heads_per_group` |
| d5 | 0/4 | declined; `moe_route` from a test's import |
| d6 | 1/2 | read `cache.rs`, the definition; no caller |
| d7 | 2/3 | `= 2`, `cache.rs`; no loader |
| d8 | 1/2 | declined; `wide_sig.rs` named for `FoldParams` |
| t1 | 1/4 | `FoldParams::derive` in the scheduler, the fold gate; no group constant |
| t2 | 2/4 | `replay.rs`, `recorded_turn`; read `Replay` and `replayed_step` but named their file `replay.rs` |
| t3 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table`; not `blocks.cuh` |
| t4 | 4/4 | read `dispatch.rs`: bucketize → mapped ring → stager word → cold spin |
| t5 | 2/3 | `PINNED_LAYERS`, `startup.rs`; `batch_victims` for `best_victim` |
| t6 | 1/4 | `ReprojectionPolicy`, `opening_chunk`; not the config fields |
| t7 | 0/3 | declined |
| t8 | 2/3 | the scheduler, `elevate_to_hot`; the module path, not `elevate.rs` |
| a1 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | `weight_floor`, `tier_fits` in `try_place` |
| a3 | 2/3 | **declined**; "recurrent", "gap-fill" incidental |
| a4 | 2/2 | un-rotated K bytes, RoPE re-derived at read |
| a5 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 2/2 | `FoldParams`, `decode_wide_sigs_checked` refuses a mismatch — `FoldParams` named for the first time |
| a7 | 1/3 | declined; "think" incidental |
| a8 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**62.2% with 9 full marks — the highest mean this battery has measured, against 57.3% this
morning and 50.0–59.7% on 11-19.** Direct rose most (38.5% → 47.9%): d2 and d3 are full, and the
answers read their files rather than recalling a guide. a6 names `FoldParams` for the first time.

**Placement is still the binding limit, and still uneven.** d1, d5 and t7 declined with files the
question does not concern in their context. On d1, retried alone at the same build, the first
reading ranked `recorded_reply.rs` third (1,345) behind `first_reading.rs` — whose doc comments
name `recorded_reply` — and `conversation.rs`; it entered the slot at token 129 and led from there.
The first reading's own score of the answer file varies with what the question is read against
(722, 1,345, 2,243 and 3,224 across today's runs of d1), so its ranking is only as stable as that
context.

**t2 shows a placed file read under the wrong name**: the answer quoted `Replay` and
`replayed_step` from `recorded_reply.rs` and attributed them to a `replay.rs`.

**The model stayed honest.** Every miss was a decline or a nearby answer naming what it had read.

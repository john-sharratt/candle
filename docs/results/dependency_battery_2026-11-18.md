# The §9.12 dependency battery — production, 2026-11-18

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run again after the scoring lab's formula went to production on the code-reading groups
(010405b50, 2026-11-16: each probe token votes for its best 8 files, `kind_z`, the priming chain
left out; `docs/results/provenance_scoring_lab_2026-11-16.md`). BlackWorld on build
`5184-e59070a50`; WhiteWorld on `5180-46ce83ab1` and loading, so the chats were served by
BlackWorld.

## How it was run

As on 2026-11-10 and 2026-11-11: through the connector's private chats, one fresh chat per
question, closed (deleted) once answered, every question prefixed *"Answer from your memory of
the repository only — do not call any tool, do not read or search any file. If you do not
remember, say so."* No answer called a tool. Scored by the battery's rule (`score.rs`): an item
counts when any of its spellings is a case-insensitive substring of the text after the reasoning
block.

## Results

| Category | Mean recall | Full marks | 2026-11-11 | 2026-11-10 |
|---|---:|---:|---:|---:|
| Direct | 24.0% | 1 of 8 | 27.1%, 0 | 33.3%, 1 |
| Transitive | 45.8% | 1 of 8 | 37.5%, 1 | 54.2%, 2 |
| Architectural | 68.8% | 4 of 8 | 68.8%, 5 | 85.4%, 5 |
| **All 24** | **46.2%** | **6** | 44.4%, 6 | 57.6%, 8 |

| Id | Recall | 11-11 | What the answer named, and what it said it had read |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | declined; held "the archived guide" to the crate's architecture |
| d2 | 0/2 | 1/2 | `region_pool::tier_fits` from `AGENTS.md` invariant 7, not `region_pool.rs`; declined the files |
| d3 | 0/2 | 1/2 | declined; had read `mcp/src/guides/verify.md`, `conversation_tests.rs`, the README |
| d4 | 2/6 | 4/6 | `scan.rs`, `resolver.rs` from a directory map, not as callers; declined |
| d5 | 1/4 | 0/4 | `cuda.rs` offered as a "likely" place; declined; had read `verify.md` and `AGENTS.md` |
| d6 | 2/2 | 1/2 | **read `cache.rs`**: the definition and its signature exactly; `pipeline.rs` from a module list |
| d7 | 1/3 | 0/3 | `cache.rs`; had read `working_set_tests.rs`; no value, no loader |
| d8 | 0/2 | 0/2 | declined; had read `story-rewrite/src/fixtures.rs` and directory listings |
| t1 | 3/4 | 0/4 | **read `wide_sig.rs`** by part: `FoldParams::derive` → `[n−2,1,1]`, `FOLD_GROUPS`, `heads_per_group`; not `window_score` |
| t2 | 2/4 | 1/4 | **read `recorded_reply.rs`**: the signature, the test; `recorded_turn`; not the harness |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE`, `types.rs`, `arena_table` from `ARCHITECTURE.md` and `AGENTS.md` |
| t4 | 4/4 | 4/4 | bucketize → mapped pinned ring → stager word → cold wait, from `AGENTS.md` |
| t5 | 0/3 | 1/3 | **off-question**: the whole answer summarised an unnamed guide file |
| t6 | 0/4 | 0/4 | declined; had read `index_pages.rs` |
| t7 | 0/3 | 0/3 | declined; had read `candle-conversation/src/lib.rs` |
| t8 | 2/3 | 2/3 | `elevate_to_hot` in `persistence/elevate.rs`, from `ARCHITECTURE.md` and `migrate.rs`; "scheduler" not named |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `tier_fits`; placed `tier_fits` in a `guard.rs` that does not hold it |
| a3 | 2/3 | 0/3 | DeltaNet and gap-fill from adjacent text; declined the rule and `reserve_glue_island` |
| a4 | 0/2 | 0/2 | declined; had read `lib.rs`, `journals/README.md` |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize` |
| a6 | 1/2 | 1/2 | **read `wide_sig.rs`** by part: `fold_admits`, `decode_wide_sigs_checked`, the refusal, why only group sizes are checked — a correct, complete answer that does not spell `FoldParams` |
| a7 | 1/3 | 3/3 | declined; had read a guide; no leak of the earlier results this time |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## What the answers show

**The formula change did not move the battery.** 46.2% against 44.4% on 2026-11-11 is inside
the noise of 24 questions, and six full marks as before. The lab measured the formula on the
ranking of answer files (MRR 0.258 → 0.701 on the 23 file-answered questions); the battery
measures what a chat answers from what its projection placed, and the two did not move together.

**Projections are still hub-dominated.** Most answers name what they had read, and what they
name is mostly not the question's files: an `mcp` guide (d3, d4, d5), a `README` (d3, d4, t1, a3,
a4), `AGENTS.md` or `ARCHITECTURE.md` (d2, d5, t3, t4, t8, a3, a5), a crate root (`lib.rs` in t7
and a4, `models/mod.rs` in a3), and pieces with no bearing on the question
(`story-rewrite/src/fixtures.rs` in d8, `journals/README.md` in a4). The code-reading file turns
reached the answer in five — `cache.rs` (d6), `working_set_tests.rs` (d7), `wide_sig.rs` (t1, a6),
`recorded_reply.rs` (t2) — and where one did, the answer was specific and correct.

**Two faults beyond retrieval.**

- *t5 answered a different question.* Its whole answer is a one-sentence summary of a guide
  file, the form of a calibration reading's answer, not the battery's question.
- *Pieces are named under the wrong path.* a6 attributes `wide_sig.rs` to `mcp/src/guides/`,
  and a7 cites a `candle-conversation/src/guides/bot-watch.md` that does not exist: the label a
  projected piece carries does not always match its file.

The model stayed honest throughout: it declined where it had not read the source, and the
answers that did name paths it had not read (a2's `guard.rs`, d5's "likely" `cuda.rs`) said so or
were adjacent misses rather than inventions.

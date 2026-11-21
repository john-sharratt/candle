# The §9.12 dependency battery — production, 2026-11-22 (i): after the file splits

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `02491d7ab`. Against (h), this run changes only the repository: every Rust
file and CUDA kernel over 60 KB was split into a folder of files, one concern per file
(`6a72fbc12` and the commits after it). The battery's expectations and `files` were moved to
the new paths with the code (`t3` expects `types/mod.rs`, `d6` expects `cache/bookkeeping.rs`,
`d7` and `t5` expect `cache/policy.rs`).

Every question went into a fresh private chat, one at a time, behind the prefix "Answer from
your memory of the repository only — do not call any tool, do not read or search any file. If
you do not remember, say so." Each was scored with `score.rs`'s case-insensitive substring
rules. WhiteWorld was not serving, so every turn ran on BlackWorld. Model-turn calls are now
placed by the machine table (`02491d7ab`), so that would have held with both members up.

## Results

| Category | Mean recall | Full marks | 11-20 (h) | 11-20 (b) |
|---|---:|---:|---:|---:|
| Direct | 18.8% | 0 of 8 | 37.5%, 1 | 47.9%, 2 |
| Transitive | 67.7% | 2 of 8 | 66.7%, 3 | 51.0%, 1 |
| Architectural | 72.9% | 4 of 8 | 68.8%, 4 | 87.5%, 6 |
| **All 24** | **53.1%** | **6** | 57.6%, 8 | 62.2%, 9 |

| Id | Recall | (h) | What the answer said |
|---|---:|---:|---|
| d1 | 0/2 | 0/2 | declined; saw `recorded_reply` only as a field in `scheduler/prefill/first_samples.rs` |
| d2 | 0/2 | 0/2 | declined; "only a journal entry mentioned it" |
| d3 | 1/2 | 1/2 | "the scheduler calls it", from `substrate/tiers.rs`' doc comments; definition not read |
| d4 | 2/6 | 1/6 | `scan.rs` from `gpu.rs`'s import; no other caller |
| d5 | 0/4 | 2/4 | declined; saw `run_moe_bucketize` in `candle-kernels/src/lib.rs` only |
| d6 | 0/2 | 1/2 | wrong: `chunked/boundary.rs` calling `ExpertCacheInner::set_eviction_costs` |
| d7 | 2/3 | 1/3 | `= 2`, `qwen35/layer_loader.rs`; defined "in `expert_lre/`", file not named |
| d8 | 0/2 | 2/2 | declined; a test file named it in prose only |
| t1 | 3/4 | 3/4 | `FoldParams::derive`, `FOLD_GROUPS`, `heads_per_group`, the late-fusion scorer; no `window_score` |
| t2 | 2/4 | 2/4 | `replay.rs` → `TurnOptions::recorded_turn`; the forcing function not read |
| t3 | 2/4 | 3/4 | `CHUNK_SIZE`, `arena_table.cuh` and its tables; old path `types.rs`, no `blocks.cuh` |
| t4 | 4/4 | 4/4 | the whole chain, live table to stager publish |
| t5 | 2/3 | 1/3 | `PINNED_LAYERS` "in `cache.rs`", `startup.rs`; the victim search not read |
| t6 | 4/4 | 4/4 | the two fields, `boundary_rule`, `build_reprojection_policy` → `ReprojectionPolicy` |
| t7 | 1/3 | 0/3 | declined; "hash" only from an unrelated cache |
| t8 | 2/3 | 3/3 | the scheduler, `elevate_to_hot`; given to `substrate/eviction.rs` |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `tier_fits`, from `span_geometry.rs` |
| a3 | 2/3 | 2/3 | recurrent state, `reserve_glue_island`; spelled `can_gap_fill`, which the gap-fill rule does not take |
| a4 | 0/2 | 0/2 | GIDs resolved at read; no RoPE |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize`'s summary in a mapped ring |
| a6 | 1/2 | 1/2 | the fold refusal and why; "fold parameters", never `FoldParams` |
| a7 | 2/3 | 1/3 | declined; named `turn_layout` and thinking in passing |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## Reading

- **Direct fell; transitive and architectural held.** Every direct question asks for a file
  path. Six of the eight answers say the only code they saw was some *other* file that mentions
  the symbol (an import, a doc comment, a field), and they decline the definition. The answers
  do name the split files (`span_geometry.rs` in a2, `turn_layout/tests_thinking.rs` in a7), so
  the readings of the split files are there. What the opening places is a neighbour of the
  answer file, not the answer file itself.
- **Old paths still in the answers.** t3 gives `chunked/types.rs` and t5 gives `PINNED_LAYERS`
  "in `cache.rs`", both paths from before the split. Those come from ARCHITECTURE.md and journal
  entries, which still name the old layout, not from code readings.
- **The first d1 was sent outside private mode by mistake.** Its answer (0/2, "the only file I
  read was a guide") is not counted; d1 was asked again privately, and that answer is the one
  scored above.
- Compared with (h), the splits did not help the direct category on their own. Item 1 of the
  standing list (pull in the caller file through a symbol→reference index) and item 2 (name the
  file on every carved turn) are still the levers for the direct questions.

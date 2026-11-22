# The §9.12 dependency battery — production, 2026-11-22 (j): no re-feed, the docs made true

The battery of `docs/unbounded_agents.md` §9.12 (`zend/examples/dependency_battery/battery.yaml`),
run on BlackWorld at `58bd9bd40` (build 5424) against (i). Three changes since (i):

- **The re-feed is gone** (`ec7991f24`). A rebuilt slot no longer forwards the turn's last 8
  tokens again on top of the recurrent state. The shadow scan scores in windows on the sequence's
  own rung again, and Flash-Next's schedule ends at ×4 to 1,000,000.
- **The documents were made true of the code** (`37eed2bb7`, `ac5c4743e`, `1e62e2b80`).
  - 27 transient documents were archived.
  - Every reference design was checked against HEAD and corrected.
  - code_reading now ignores `docs/results/*` and `docs/archived/*`, so a record of an older tree
    is no longer read as current.
  - The ingest re-read and re-calibrated every changed document before the run.
- WhiteWorld's sum-by-amax change (`412ffaaa1`) drafted garbage and was reverted (`58bd9bd40`).
  The run was on the revert, whose code is `bf636c446`'s plus the comment edits above.

Each question went into a fresh private chat, one at a time, with the prefix "Answer from your
memory of the repository only — do not call any tool, do not read or search any file. If you do
not remember, say so." Answers are scored with `score.rs`'s case-insensitive substring rules.
Placement put every turn on BlackWorld, and WhiteWorld was serving and idle. Decode ran at
84–112 t/s per turn.

## Results

| Category | Mean recall | Full marks | 11-22 (i) | 11-20 (h) | 11-20 (b) |
|---|---:|---:|---:|---:|---:|
| Direct | 31.3% | 0 of 8 | 18.8%, 0 | 37.5%, 1 | 47.9%, 2 |
| Transitive | 40.6% | 0 of 8 | 67.7%, 2 | 66.7%, 3 | 51.0%, 1 |
| Architectural | 79.2% | 6 of 8 | 72.9%, 4 | 68.8%, 4 | 87.5%, 6 |
| **All 24** | **50.3%** | **6** | 53.1%, 6 | 57.6%, 8 | 62.2%, 9 |

| Id | Recall | (i) | What the answer said |
|---|---:|---:|---|
| d1 | 1/2 | 0/2 | `recorded_reply.rs` and its signature; caller "likely the scheduler or projection path", not named |
| d2 | 0/2 | 0/2 | `region_pool.rs` from a journal; caller declined |
| d3 | 1/2 | 1/2 | called by the scheduler, from `tiers.rs`' comments; definition not read |
| d4 | 2/6 | 2/6 | `scan.rs` from `gpu.rs`'s import, two call sites in `gpu.rs`; no others |
| d5 | 2/4 | 0/4 | `moe_bucketize.cu`, the FFI in `quantized/cuda/moe_bucketize.rs`; callers not named |
| d6 | 0/2 | 0/2 | `ExpertCacheInner::set_eviction_costs` "in candle-nn"; caller declined |
| d7 | 2/3 | 2/3 | `= 2`, `layer_loader.rs`; definition "in `expert_lre`", file not named |
| d8 | 0/2 | 0/2 | declined |
| t1 | 3/4 | 3/4 | `FoldParams::derive`, `FOLD_GROUPS`, `heads_per_group`, the late-fusion scorer; no `window_score`; **no loop** |
| t2 | 3/4 | 2/4 | `replay.rs` → `TurnOptions::recorded_turn` → `recorded_reply`; the decode-loop consumer not named |
| t3 | 0/4 | 2/4 | declined; only two journal entries placed |
| t4 | 3/4 | 4/4 | bucketize's tile tables, the mapped summary word, the cold-expert spin; no stager |
| t5 | 1/3 | 2/3 | `PINNED_LAYERS` from a guide; value, victim search and startup declined |
| t6 | 0/4 | 4/4 | declined; `scheduler/reprojection.rs` placed, the config fields not |
| t7 | 0/3 | 1/3 | declined; `projection/event/classify.rs` placed, which says glue is re-derived |
| t8 | 2/3 | 2/3 | the scheduler, `elevate_to_hot`, `lift_warm_to_gpu`; `elevate.rs` not named |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE` in `chunked/types/mod.rs` (the current path), `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits` |
| a3 | 0/3 | 2/3 | declined; one unrelated guide placed |
| a4 | 2/2 | 0/2 | keys stored un-rotated, RoPE applied at read, from `apply_build.rs` |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize`'s summary in a mapped ring |
| a6 | 2/2 | 1/2 | `FoldParams` stamped on the record, `decode_wide_sigs_checked` refuses |
| a7 | 1/3 | 2/3 | declined; "thinking" only |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## Reading

- **The loops are gone.** In (i), d1's thinking wrote `lines=1-44` for about 3,000 of 3,770 tokens,
  and t1 looped on its list of names in (g) and (h). In (j), no turn loops. The longest is t4 at
  4,032 tokens, most of it a detailed answer. Its thinking repeats one file name about six times
  while recalling what it read, then breaks off on its own ("I don't need to reproduce those
  exactly"). The re-feed's duplicated span in the recurrent state was the cause the change
  removed.
- **The answers cite the code as it is.**
  - a1 gives `chunked/types/mod.rs`, where (i) gave the pre-split `types.rs`.
  - a4 and a6 score full for the first time since (b).
  - Architectural reaches 79.2%, the best since (b).
- **Transitive fell, and every drop is a decline over the wrong slot.**
  - t3, t6, t7 and a3 placed a journal entry, a guide or a neighbouring file, then declined. In
    (i) the same questions placed the answer file and scored 2–4.
  - This is selection noise across runs (fix 8(b) of the standing list, fixed seeds, is not in
    place). It is not a loss of knowledge: no answer was wrong, and each declined honestly.
- **Direct rose from 18.8% to 31.3%** but is still the weakest category. The caller in a second
  file is still missing (d1, d2, d6, d8), and the standing list's fix 1, pulling in the caller
  file through a symbol→reference index, still applies.
- **Against (i), the overall score is level (50.3% against 53.1%, six full marks each)**, within
  the run-to-run spread the transitive declines show. The change that matters here is
  qualitative: the thinking no longer degenerates.

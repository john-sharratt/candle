# The §9.12 dependency battery — production, 2026-11-23 (l) and (m): the probe's reach and reprojection in thinking

Two runs on BlackWorld, by the method of (j): each question in a fresh private chat, one at a
time, with the prefix "Answer from your memory of the repository only — do not call any tool, do
not read or search any file. If you do not remember, say so.", scored by `score.rs`'s
case-insensitive substring rules.

- **(l)** at `98d8f646c`: the probe reaches 256 tokens, and a boundary inside the think block
  reprojects, as every build before it.
- **(m)** at `e6ebbb188` (build 5449), after the load's normalization replay had drained:
  `reproject_probe_tokens: 128` and `reproject_in_thinking: false` in BlackWorld's
  `motherd-local.yaml`. The think block decodes on the projection it opened on, and the committed
  `</think>` reprojects once.

## Results

| Category | (l) recall | (l) full | (m) recall | (m) full |
|---|---:|---:|---:|---:|
| Direct | 25.0% | 0 of 8 | 20.8% | 0 of 8 |
| Transitive | 56.3% | 2 of 8 | 51.0% | 1 of 8 |
| Architectural | 66.7% | 5 of 8 | 47.9% | 3 of 8 |
| **All 24** | **49.3%** | **7** | **39.9%** | **4** |

| Id | (l) | (m) | (m)'s answer |
|---|---:|---:|---|
| d1 | 0/2 | 1/2 | `recorded_reply.rs` named in passing; caller declined |
| d2 | 0/2 | 0/2 | `region_pool::tier_fits` from a journal; caller declined |
| d3 | 1/2 | 1/2 | "the scheduler"; definition credited to `substrate/mod.rs` |
| d4 | 2/6 | 2/6 | `scan.rs` from `window_score.rs`'s import; other callers declined |
| d5 | 0/4 | 0/4 | declined |
| d6 | 0/2 | 0/2 | declined |
| d7 | 2/3 | 1/3 | `layer_loader.rs`; value and definition declined |
| d8 | 1/2 | 0/2 | declined: nothing placed names `FoldParams` |
| t1 | 1/4 | 3/4 | `FoldParams::derive`, `FOLD_GROUPS`, `heads_per_group`; no `window_score` |
| t2 | 2/4 | 1/4 | `TurnOptions::recorded_turn`, from `commit.rs`; `replay.rs` not named |
| t3 | 3/4 | 3/4 | `CHUNK_SIZE` in `types/mod.rs` and `arena_table.cuh`; no `blocks.cuh` |
| t4 | 4/4 | 4/4 | bucketize, the mapped ring, the stager-published word, the cold spin |
| t5 | 1/3 | 2/3 | `PINNED_LAYERS`, the pad's `victims`, `startup.rs` |
| t6 | 4/4 | 0/4 | declined; a journal naming the new `ReprojectionSettings` placed |
| t7 | 0/3 | 0/3 | declined |
| t8 | 2/3 | 2/3 | the scheduler, `elevate_to_hot`, `lift_warm_to_gpu`; `elevate.rs` not named |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE`, `candle-kernels` |
| a2 | 2/2 | 0/2 | declined: two unrelated journals placed |
| a3 | 0/3 | 0/3 | declined |
| a4 | 0/2 | 0/2 | declined |
| a5 | 2/2 | 2/2 | invariant 3, `moe_bucketize`'s mapped ring |
| a6 | 2/2 | 1/2 | refused at `fold_admits`; "fold parameters", `FoldParams` not spelled |
| a7 | 1/3 | 1/3 | declined |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## Reading

- **(m) is 9.4 points below (l), with three fewer full marks.** Every drop is a decline over a
  slot that held the wrong piece: t6, a2 and d8 placed journals or guides and declined, where
  (l) placed the answer files. No answer was wrong, none looped, and every one ended normally.
- **Gains and losses cross.** t1 rose 1 → 3 and t5 1 → 2, while t6 fell 4 → 0 and a2 2 → 0.
  That is the per-question selection noise (j) and (k) recorded, where identical configurations
  moved categories by ±25 points.
- **The overall spread is the question.** The five runs before these, (h) to (l), span 46.9% to
  57.6%. (m)'s 39.9% is below that range, but one run per arm cannot separate the setting from
  the noise. Both settings changed at once, so a real drop is not attributed to either.

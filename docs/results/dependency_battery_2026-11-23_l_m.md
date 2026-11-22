# The §9.12 dependency battery — production, 2026-11-23 (l)–(o): the probe's reach and reprojection in thinking

Four runs on BlackWorld, by the method of (j): each question in a fresh private chat, one at a
time, with the prefix "Answer from your memory of the repository only — do not call any tool, do
not read or search any file. If you do not remember, say so.", scored by `score.rs`'s
case-insensitive substring rules.

- **(l)** at `98d8f646c`: the probe reaches 256 tokens, and a boundary inside the think block
  reprojects, as every build before it.
- **(m)** at `e6ebbb188` (build 5449), after the load's normalization replay had drained:
  `reproject_probe_tokens: 128` and `reproject_in_thinking: false` in BlackWorld's
  `motherd-local.yaml`. The think block decodes on the projection it opened on, and the committed
  `</think>` reprojects once.
- **(n)** on (m)'s build, restarted through keep with `reproject_in_thinking: true` and the probe
  still at 128, one minute after the load's normalization replay drained: the probe's reach
  alone changed against (l).
- **(o)** is (m) repeated: the same build, restarted through keep with `reproject_in_thinking:
  false` and the probe at 128, one minute after the normalization replay drained.

## Results

| Category | (l) 256, in thinking | (m) 128, held | (n) 128, in thinking | (o) 128, held |
|---|---:|---:|---:|---:|
| Direct | 25.0%, 0 full | 20.8%, 0 | 26.0%, 0 | 14.6%, 1 |
| Transitive | 56.3%, 2 full | 51.0%, 1 | 64.6%, 2 | 50.0%, 2 |
| Architectural | 66.7%, 5 full | 47.9%, 3 | 70.8%, 5 | 47.9%, 3 |
| **All 24** | **49.3%, 7** | **39.9%, 4** | **53.8%, 7** | **37.5%, 6** |

(o) by question: d1 0/2, d2 0/2, d3 2/2, d4 1/6, d5 0/4, d6 0/2, d7 0/3, d8 0/2; t1 2/4,
t2 2/4, t3 4/4, t4 4/4, t5 2/3, t6 0/4, t7 0/3, t8 1/3; a1 3/3, a2 2/2, a3 0/3, a4 0/2, a5 2/2,
a6 1/2, a7 1/3, a8 0/2. t3 reached full marks with `blocks.cuh` for the first time. a8, full in
every earlier run, declined: the answer said both of its placed reads "came back empty".

(n) by question: d1 1/2, d2 0/2, d3 1/2, d4 1/6, d5 1/4, d6 0/2, d7 2/3, d8 0/2; t1 3/4,
t2 2/4, t3 3/4, t4 2/4, t5 2/3, t6 0/4, t7 3/3, t8 3/3; a1 3/3, a2 2/2, a3 1/3, a4 0/2, a5 2/2,
a6 2/2, a7 1/3, a8 2/2. t7 reached full marks for the first time (`glue_islands`, the content
hash, `GLUE_ISLAND_RETAIN_GENERATIONS = 4`), and t8 named `elevate.rs`.

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
- **The overall spread is the question.** The seven runs that reproject in thinking, (h) to (l)
  and (n), span 46.9% to 57.6%. Both runs that hold the think block, (m) at 39.9% and (o) at
  37.5%, fall below that whole range.
- **(n) separates the two settings.** With the probe at 128 and the think block reprojecting,
  the battery scored 53.8% with seven full marks, level with or above (l) at 256. The 128-token
  probe costs nothing measurable here.
- **(o) repeats (m), and the hold's cost stands.** At the same probe, the two held runs average
  38.7% against (n)'s 53.8%, a drop of about 15 points. It falls on direct (20.8%, 14.6% against
  26.0%) and architectural (47.9% twice against 70.8%). The answers decline rather than err: a
  think block held on its opening projection reasons over what the question alone selected, and
  the reprojection at `</think>` comes after the reasoning that needed the answer files.
  Reprojecting during thinking is what brings the files in for the thought that uses them.

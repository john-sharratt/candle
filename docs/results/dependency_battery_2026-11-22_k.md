# The §9.12 dependency battery — production, 2026-11-22 (k): the corrected sum-by-amax fix

Battery (j) repeated on BlackWorld at `f96877cf5` (build 5427): WhiteWorld's corrected NaN fix
merged on top of (j)'s `03ebb5b52`.

- qwen4exp's int8 activations and every expert SwiGLU now store their per-128 block sums by amax
  (`SumScale::ByAmax`). A block whose raw Σx would pass f16's 65504 no longer turns the experts'
  outputs into NaN.
- The MoE gather carries its source's convention (`fused_moe_gather_q8a128`).
- No q8a128 operand constructor defaults the convention, so every producer and every copy names
  it. The first attempt (`412ffaaa1`, reverted in `58bd9bd40`) left the gather labelling by-amax
  bytes as raw, and every calibration draft came out as digit runs.

The method is as in (j). Every turn ran on BlackWorld; the candidate was certified and WhiteWorld
took the same build during the run. Decode ran at 100–118 t/s per turn, against (j)'s 84–112.

**The gate:**
- **Output is sane:** every answer is coherent prose and ends normally.
- **Nothing tripped:** BlackWorld logged no NaN refusal and no degenerate decode in the run.
- **No turn loops:** the longest are t1 (3,597 tokens) and d1 (3,202), both full answers.

## Results

| Category | Mean recall | Full marks | (j) |
|---|---:|---:|---:|
| Direct | 22.9% | 0 of 8 | 31.3%, 0 |
| Transitive | 63.5% | 3 of 8 | 40.6%, 0 |
| Architectural | 54.2% | 3 of 8 | 79.2%, 6 |
| **All 24** | **46.9%** | **6** | 50.3%, 6 |

| Id | Recall | (j) | What the answer said |
|---|---:|---:|---|
| d1 | 1/2 | 1/2 | `recorded_reply.rs`, its definition; no non-test caller |
| d2 | 0/2 | 0/2 | `region_pool::tier_fits` from a journal; declined the path |
| d3 | 1/2 | 1/2 | the call in `scheduler/submit_admit.rs`; definition not read |
| d4 | 1/6 | 2/6 | `scan.rs`, but the import credited to `projection.yaml` |
| d5 | 0/4 | 2/4 | declined; ARCHITECTURE.md's account only |
| d6 | 0/2 | 0/2 | declined |
| d7 | 2/3 | 2/3 | `= 2`, `qwen35/layer_loader.rs`; the definition credited to a wrong `mod.rs` |
| d8 | 0/2 | 0/2 | declined |
| t1 | 4/4 | 3/4 | `FoldParams::derive`, `FOLD_GROUPS`, `heads_per_group`, the scorer, `window_score.rs` |
| t2 | 2/4 | 3/4 | `replay.rs` → `TurnOptions::recorded_turn`; the consumer not read |
| t3 | 4/4 | 0/4 | `CHUNK_SIZE` in `types/mod.rs` and `arena_table.cuh`, `blocks.cuh`, the arena tables |
| t4 | 4/4 | 3/4 | bucketize's summary, the mapped ring, the stager-published word, the cold wait |
| t5 | 2/3 | 1/3 | `PINNED_LAYERS`, `victims()` skipping resident slots, the startup fill (`startup.rs`) |
| t6 | 1/4 | 0/4 | declined; named `ReprojectionPolicy` as where to look |
| t7 | 0/3 | 0/3 | declined |
| t8 | 2/3 | 2/3 | the scheduler, `elevate_to_hot` in `eviction.rs`, `lift_warm_to_gpu` |
| a1 | 3/3 | 3/3 | 32, `CHUNK_SIZE` in `chunked/types/mod.rs`, `candle-kernels` |
| a2 | 2/2 | 2/2 | `weight_floor`, `region_pool::tier_fits`, from `span_geometry.rs` |
| a3 | 0/3 | 0/3 | declined |
| a4 | 0/2 | 2/2 | declined; GID addressing only |
| a5 | 1/2 | 2/2 | declined; "readback" only |
| a6 | 1/2 | 2/2 | the refusal and why, from `wide_sig.rs`; `FoldParams` not named |
| a7 | 1/3 | 1/3 | declined; "thinking" only |
| a8 | 2/2 | 2/2 | `substrate/substrate.log`, cold |

## Reading

- **The fix does not degrade the model.** The output is coherent, the throughput is higher than
  (j)'s, and no turn loops or degenerates.
- **The overall score is level with (j)** (46.9% with 6 full marks, against 50.3% with 6), but
  the categories traded places:
  - transitive rose 40.6% → 63.5%, with t1, t3 and t4 at full marks for the first time together;
  - architectural fell 79.2% → 54.2%;
  - direct fell 31.3% → 22.9%.
- **Every drop is a decline over a slot that held the wrong piece.** That is the same
  run-to-run selection noise (j) recorded, and it shows these per-category figures move by
  ±25 points between identical configurations. A change smaller than that needs fixed sampling
  seeds (fix 8(b)) or several runs to measure.
- **A wrong-file attribution recurred.** d4 credited `scan.rs`'s import to `projection.yaml`, and
  d7 credited the definition to the wrong `mod.rs`. Fix 2 (name the file on every carved turn)
  addresses this.

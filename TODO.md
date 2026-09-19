# hunt-snapshot — where this branch stands

Snapshot as of 2026-09-19. Branch `hunt-snapshot`, measured on the RTX 4090 Mobile (16 GB),
Qwen3.6-35B-A3B, zend running `-v --disable-layer code_reading --host 127.0.0.1 --port 8081`.

## Direction

The goal is **repo_map ingest throughput on this branch at least equal to `main`** (which
does not show the problems below), without redesigning rate or admission:

- `rate.rs` and `admit::fill` are close to where we want them. They may be *tuned*, but
  **no new throttles, no choking the load going in**. The wave regulates itself; the
  workers do not. It has to take whatever is thrown at it and get the best throughput the
  VRAM allows.
- Fix forward. The regressions below were introduced by this branch (main is clean), so the
  job is to find the causes and fix them, not to work around them.
- The measure that counts is **directories ingested per minute**, not tok/s alone.

## Done on this branch

| commit | what |
|---|---|
| `8861aa6bd` | Every target compiles again. The probe layer is wired in. A submission that is both a prefill and a decode is now judged by both formulas at admit time: `Cost::decodes_after` and `WaveRate::decode_would_carry` (aligned with the target). The repo_map pool spawns at `REPO_MAP_PARALLELISM` (96), gated by `admits_now`. |
| `1ae4b529a` | web roles tests: a dead upstream holds its port for the whole test. |
| `6eaaa6153` | A short verify head is an error that names itself (`Scheduler::head_logits`) rather than a silently shorter vector. seg2 now fails its whole group atomically, and a `verify row set replaced` trace was added. |
| this commit | The three changes below. |

### In this commit

1. **Batched rebuild lift.** In `scheduler/prefill.rs`, `prelift_demoted_working_sets` groups
   the demoted turns of one admission by substrate (`Conversation::same_substrate`) and runs
   **one** `elevate_projection_working_set` per group over the union of their working sets
   (`projection_assembler::union_working_sets`). The per-turn elevate is kept as the safety
   net. Measured: 3 batches covering 15 turns in 0.1 s; rebuild p50 went from 29 ms to 8 ms.
2. **Sealed-segment reader.** New file `persistence/sealed_reader.rs`, with changes in
   `segmented_log.rs` and `resolver.rs`. `read_recurrent_snapshot` now reads records in
   *sealed* (immutable) segments straight from disk, without the persistence mutex, so a
   promote no longer waits out a compaction for that read. The active segment is published
   through an atomic (`ActiveSegment`). An unlinked segment re-resolves the record's
   location, and anything else falls back to the locked read.
3. **Stampede removed from the repo_map KV gate.** In `zend/src/repo_scan/mod.rs`, the 20 s
   `SCAN_POOL_WAIT_CAP` bypass is gone. Combined with a 96-worker pool, the bypass admitted
   128 slots / 7.2 GB of KV and ingested 0 directories. A full pool now admits only as it
   drains, and the cap is logged every time it changes.

Tests: the unit tests for all three are green (sealed_reader 6, recurrent_snapshot_read 4,
union_working_set 4, repo_scan gate tests). `cargo fmt` is clean and the release build is
green. `hybrid_recurrent_state` passed 5/5 (2,454 s).

## Measurements

| | repro2 (before this commit) | run5 (this commit, first 12 dirs) |
|---|---:|---:|
| s/dir | 12.95 | 27.7 (Q1–Q2 4–5 s, Q3–Q4 34–37 s) |
| tok/s | 35.9 | 17.2 |
| promote stalls ≥ 5 s | 112, holding 1,485 s, max 48 s | 5, holding 69 s, max 29 s |
| compaction hold (persistence thread) | mean 16.6 s | mean 14.2 s, max 20.9 s |
| rebuilds | 1,656, p50 29 ms | 22, p50 8 ms |
| weights_mib mean | 6,518 | 7,667 |
| failed dirs / verify diagnostics | 0 / — | 0 / 0 |

run5 was stopped after 12 directories, which is too small a sample to judge by.

## Next — in order

1. **Find the remaining promote stall.** Stalls up to 29 s survive the sealed reader, and
   they still line up with segment compaction. Something else in `promote_new_prefills`
   waits on the persistence mutex while `maintenance` runs its execute phase. Correlate
   each stall ≥ 5 s with the compaction windows in the log, then trace which lock
   `promote` takes (candidates: store writes, `insert_turn_staged`, recurrent snapshot
   *writes*, tier migration). Give that path the same treatment: take it off the lock, or
   keep the lock out of the compaction's execute phase.
2. **Compaction's own cost.** 67 of 79 ops were a full resident re-emit (~11.5k records),
   with ~22k chunk relocations and an fsync under the lock. Make the re-emit incremental
   or move it off the lock, so a compaction stops holding the persistence thread for 15–20 s.
3. **Gate cap flapping.** `max_live_conversations` swings between `Some(1)` and `Some(28)`
   every 10–30 s while 28 conversations are live. Find which input to
   `scan_width_from_governor` / the memory report is transient (likely a mid-wave snapshot
   rather than the reservation, per invariant 7) and read the stable figure instead. This is
   a tune, not a new throttle.
4. **Verify-row regression (branch-introduced; main is clean).** The head sometimes
   recognises only a subset of the verify members (seen as 3 of 10) through the
   `verify_row_seqs` state. It is now an error that names itself, and the
   `verify row set replaced` trace is at `debug` under the module path. On the next failing
   run, read the trace to find the writer that replaced the set mid-wave, then fix it.
   Bisecting against main is allowed if the trace isn't enough.
5. **Full wiped-substrate measurement.** Run with `--wipe-substrate`, measured with the
   scratchpad `measure.sh` (dirs/min, s/dir quartiles, phase totals, promote distribution,
   compaction holds, rebuilds, weights_mib), and compare against repro2 and main.
6. **`wave_decode_set` is never cleared.** Noted and deliberately left alone. Confirm whether
   it matters once 1–4 are done.
7. **The remaining /bed tail.** Full `/fast-test` chain, `/sweep`, then the report.

(The dead `windowed_ingest_ranges_impl` that failed clippy was removed: its only caller
had been deleted and the bounded ingest is done by per-unit conversations, per
`docs/unified_wave_inference_engine.md` §7 step 0.)

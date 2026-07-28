# Wave Blocking & Sync — Measured Decomposition, Incident Learnings, and the Safe Mitigation Plan

Audit of the scheduler-thread blocking and GPU synchronization that dominates
wave-latency variance. Measured against two live `code_read` workspace-ingest
runs on the RTX PRO 5000 (Qwen3-30B-A3B):

- **Jul 25 soak** (3h07m, 12:38–15:48): the `ba11d654` build — copy-stream
  migrate + compaction batching + proactive demote. Source of the stall
  decomposition in §2 — and of a decode-corruption regression (§1).
- **Jul 26 run** (post-revert, tree == `2b10fd57`): decode quality fully
  recovered (0/12,612 degenerate); compaction stalls present again in
  single-shot form (54 ops, 3.3–8.0 s lock holds at fresh-store scale, growing
  with the store).

Four deep audits feed this document (compaction/churn, drain/tail attribution,
GPU-sync inventory, structural options), plus the corruption forensics.

## 1. Incident learnings (Jul 25–26) — why the plan below avoids GPU streams

The `ba11d654` set shipped three changes in one commit; the soak showed a severe
silent decode regression, isolated by revert (`91a2d277`) and a clean re-run:

- **Symptom:** ~39% of decode steps emitted persistent token-0 (`!`) runs —
  specific slots in the 25–46-wide co-batch born-broken from their first decoded
  token while sibling slots in the *same forward* decoded normally. Zero
  WARN/ERROR anywhere: completely silent corruption. Garbage summaries were
  durably stored in the substrate.
- **Isolation:** the pre-`ba11d654` baseline (Jul 23–24) and the post-revert
  run are both exactly **0** degenerate. Everything in `2b10fd57`'s lineage —
  the CFW wave unification, parallel fork+splice scope ingest, the 2 s
  time-slice, mid-wave admission, deferred-glue ordering — is **exonerated**.
  Within `ba11d654`: the proactive demote never fired (VRAM unpressured all
  soak) and corruption began ~5 s after daemon start (before any in-run
  batched relocation). The **copy-stream migrate is the dominant suspect**
  (candidate mechanisms: the `PalHeadDesc` raw-device-pointer window between
  primary-stream resolution and copy-stream consumption; stream-ordered
  allocator reuse against in-flight copy-stream reads; warm-copy DtoH
  integrity).
- **Hard lessons, now standing policy for this subsystem:**
  1. A single-threaded byte-parity test is **insufficient** for cross-stream
     changes — `convert_on_copy_stream_matches_primary_bytes` passed while the
     live path corrupted. Any future stream change must be gated on a
     **concurrent-decode corruption test** plus the live tripwire below.
  2. Silent-corruption tripwire (keep running during any risky soak):
     `grep -a "decode_step batch" .substrate/zend.log | grep -acE '(\|!){6,}'`
     — healthy baseline is a hard 0.
  3. Ship stream changes **alone**, never bundled with unrelated safe fixes —
     the bundling is why the safe compaction work got reverted with the
     culprit.
- `ba11d654` remains in history; its CPU-side pieces are re-landable (§3).

## 2. Measured decomposition — where the stalled time goes

| Staller | Magnitude (measured) | Root cause (verified in source) |
|---|---|---|
| Segment-maintenance lock holds | Jul 25: **19.3% duty cycle**, 163 ops, `max_hold ≈ exec` ≈ 14.4 s/op, 1:1 with every 2–13.8 s stalled wave. Jul 26 (fresh store, single-shot): 54 ops, 3.3–8.0 s holds — regrows with store size | The op re-emits the **entire resident metadata set** — ~150 k streams × up to 4 records × 4 KB sector padding ≈ **2 GB + ~15 embedded fsyncs per op** — under one persistence-lock hold. The `resident_reemit_floor` skip never fires during ingest (ops always target the newest eligible segment, which postdates the last re-emission). **Batching the relocations alone was proven insufficient** on Jul 25: with 128-record relocation batches the hold *stayed* ~14.4 s because the first batch's resident emission is the hold. |
| Compaction churn loop | ~25 MB/s continuous write bandwidth is pure re-emission churn; ~190 GB of half-dead emission segments accumulated | Self-amplifying: each op's ~2 GB re-emission fills half a 4 GiB segment; the next op supersedes it (`metadata_locs` is last-writer-wins) → every emission segment is ~50% dead within one cycle → permanent candidate supply. Maintenance compacts its own exhaust. |
| Scheduler coupling to the lock | 9–10 s unaccounted waves during holds | Direct: cold→hot elevation, `recover_turn_chunks`, `tombstone_timeline`/`distill_timeline`/`couple_turn` (the code_read eviction calls), bg-quantizer commits — all take the persistence Mutex. Transitive: `SubstrateWriter::enqueue` backpressure (16 384-event cap) when the writer parks behind the lock. |
| Drain spikes | 1.0–3.9 s, ~12/25 min; also stolen mid-quantum | Unbounded `drain_submissions` loop × (per-probe synchronous `run_prefill` forwards ~70–100 ms + untimed pageable-PCIe elevate lifts + double `materialize` tokenizer-decode per submit). Cold→hot holds the persistence Mutex across whole NVMe batches. |
| Seal-heavy wave tails | 100–460 ms, scaling with `seal_count` | `promote_finished_prefills_to_decodes` runs the whole seal machinery untimed: a sync forward per completed scope, per-prefill `sample_single` D2H, `quantize_pending_sections` device-sync on the seal path, demote→`cuMemPoolTrimTo` chains, the 2 s telemetry/reclaim block. |
| Decode step floor | ~74 ms/token WDDM launch floor | 48 MoE routing DtoH round-trips per step — the structural blocker for launch-ahead/graphs. |

## 3. Compaction-stall mitigation — the safe path (no GPU streams touched)

Every item here is CPU-side: lock discipline, planning policy, and file I/O.
None of it can corrupt KV. Ordered as a concrete execution plan; the first
three together remove the stall class at current scale.

### Step 1 — Ingest-aware trigger gating (hours, config-level)
While ingesting (the `pending_warm_bytes` signal and code_read session state
already exist): raise `SEGMENT_COMPACT_MIN_DEAD` 0.10 → 0.40–0.50, `MIN_AGE`
60 → 300 s, `MAINTENANCE_INTERVAL` 15 → 60 s; restore on idle. Cuts op
frequency 5–10× immediately and pushes the work to idle time where lock holds
are harmless. Cost: temporary disk bloat — irrelevant at the 16 TB target.

### Step 2 — Re-land relocation batching + slice the resident emission (the pair, ~2 days)
Re-land the audited compaction batching from `ba11d654` (cherry-pick the
`maintenance.rs`/`resolver.rs` hunks and their two unit tests —
`chunked_execute_maintenance_batch_compacts_correctly`,
`take_install_fence_is_one_shot` excluded as substrate-side) **together with**
slicing the resident re-emission itself: turn `emit_resident: bool` into a
range like the relocations (~512 records / ~2 MB per lock hold), releasing the
lock between slices.

The pairing is mandatory — the Jul 25 forensics proved relocation batching
alone leaves `max_hold ≈ 14.4 s` because the unbatched first-batch resident
emission *is* the hold. Keep `ba11d654`'s final fsync discipline: batches
append-only, **one** `commit()` after the loop, sources unlinked only in
`finish_maintenance` afterward (durable-before-unlink preserved; no
fsync-per-batch). Expected result: worst per-hold ~50 ms–1 s (a slice crossing
an index-flush fsync), i.e. the 3–14 s stall class is gone even though op
exec time is unchanged.

### Step 3 — Drop-over-Compact for tombstone-dead segments (1–2 days)
`segment_liveness` already classifies tombstoned streams; code_read kills
whole timelines (scope forks, replaced files), so many fresh segments trend to
100% dead where maintenance is an O(1) unlink. Skip Compact when
tombstoned-dead dominates a young segment and let it ripen into a Drop.
Converts most ingest-time relocation monsters (1 300 batches × 128 records)
into unlinks — eliminates most relocation I/O and shrinks Step 2's remaining
work.

### Step 4 — Delta re-emission (2–4 days) — kills the churn loop itself
Re-emit only records whose current durable copy lives in the target segments.
`metadata_locs` already maps `(RecordType, stream_id) → RecordLoc` for the
four per-stream resident types; extend it to the per-timeline types
(Label/ConvState/TreeMetadata/DebugId/Distilled/Tombstone). Re-emission
shrinks ~500 k records → the few thousand physically in the target: removes
~95–99% of the emission work **and** the ~2 GB/op write amplification **and**
extinguishes the churn loop's fuel (no more instantly-superseded bulk copies).
The floor mechanism becomes dead code. Risk is medium and purely CPU-side:
correctness rests on complete location tracking per resident type — the
wholesale re-emit is today's guarantee that a dropped segment can't take the
only tombstone marker with it; land with recovery-walk tests for each type.

### Step 5 — Decouple the scheduler from the lock (S–M each, still CPU-only)
- **Cold reads off the Mutex**: sealed segments are immutable post-fsync; give
  `SealedPool` independent read-only handles so `cold_load_turns_into_hot` /
  `read_stream_chunks_batched` need the Mutex only for the offset lookup, not
  the NVMe I/O. Severs elevate↔compaction coupling entirely.
- **Non-blocking metadata enqueue** in `SubstrateWriter`: metadata jobs spill
  to an unbounded side-buffer instead of blocking the scheduler at the event
  cap — severs the transitive seal→compaction stall.
- **`segment_liveness` memoization**: the planner scan is O(all live chunk
  locs — tens of millions) under the persistence lock + substrate read lock
  every eligible 15 s pass, unmeasured today. Incrementalize or cache.

### Step 6 — Structural end-state (1–2+ wk each, when justified)
- Bulk cold KV out of the compactable log: per-file-timeline extent/blob store
  (write-once, delete-as-a-unit — dropping the file drops the extent; no
  compaction ever). Removes the compaction *cause* for ingest.
- Resident metadata out of segments: atomically-rewritten manifest + small
  delta log; re-emission, the floor, and `metadata_locs` all disappear.

**Sequencing note:** Step 1 is deployable today. Steps 2+3 remove the stall
class; Step 4 removes the waste; Step 5 removes the coupling. Each is
independently landable and testable on the existing CPU tmpdir harness
(`SubstratePersistence::open_in_with_substrate` — no GPU, no model).

## 4. Non-compaction options (unchanged from the audits, safe subset first)

| # | Option | Impact | Effort | Risk |
|---|---|---|---|---|
| 4a | Exclude `SubmitTurn` from `mid_wave_admission` (defer heavy requests to loop top; the 2 s clip bounds latency) | Stops multi-second mid-quantum decode theft | Trivial | None (CPU scheduling) |
| 4b | Time-box/slice the drain (≤K submissions or ≤T ms per wave) | 1.6–3.9 s spikes → bounded ~100–300 ms | Low | Low |
| 4c | Attribution: `WavePhase::Finalize` + `WavePhase::Relief` + drain lift/forward sub-timers | Makes the remaining tails measurable | Trivial | None |
| 4d | Ride probe/scope `assistant_start` prefills on the wave cohort instead of synchronous `run_prefill` | ~70–100 ms × probes per drain + one forward per scope off the seal tail | Medium | Medium (CPU scheduling; also fixes the glue-ordering quirk) |
| 4e | Pre-projection off-thread + warm-prefetch (pairs with Step 5 cold-read de-Mutexing) | Drain → tens of ms | Medium | Medium (staleness already tolerated by the reproject discipline) |
| 4f | Lazy `materialize_conversation` (GUI-subscriber-gated) | Small–moderate CPU per submit | Low | Low |
| 4g | Trim/demote gating; batch `sample_single` + `gather_wide_sigs` per wave; GpuChunks high-water reuse | Tens of ms each, additive | Low each | Low |
| 4h | **TCC/MCDM driver-mode experiment** (`nvidia-smi -dm 1`; nothing in-repo requires WDDM) | Potentially ~27 → 100+ steps/quantum | Config-only | Low (reversible driver setting) |

## 5. Deferred: GPU-stream work (HIGH RISK — gated)

These were the `ba11d654` overlap program and the sync-inventory conversions.
All are **deferred** until the Jul 25 corruption mechanism is found and fixed,
and every item is individually gated on: (a) a **concurrent-decode corruption
test** (co-batched decode hammering while the change runs, byte-verifying
attended KV), (b) the §1 tripwire running through a full ingest soak, (c)
landing **alone** in its own commit.

- Copy-stream hot→warm migrate (the reverted Phase 2) — re-land last, only
  with the mechanism understood.
- Event-install conversions for the scheduler-thread quantize passes; the
  warm→cold gather onto a copy stream; batched warm→hot elevate on a side
  stream; async install (the Phase-3 sync removal + migrate-epoch fence).
- The decode-floor program (speculative expert dispatch → device-chained
  sampling → CUDA graphs) — separate track, same gating discipline for any
  stream/graph change.

Rationale: the corruption was silent, durable (garbage summaries persisted to
the substrate), and invisible to single-threaded verification. The safe plan
in §3 removes the dominant stall source without touching any of this; the
overlap program should resume only from a position where §3 has already made
waves consistent and any regression is instantly visible via the tripwire.

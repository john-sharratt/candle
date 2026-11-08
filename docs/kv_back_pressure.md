# KV back pressure: producers wait on the engine, not on their own residues

**Problem.** A producer that starts conversations faster than the engine retires
them fills the KV reservation until no wave can be placed. Read-first ingest is
the case that broke: 16 `code_read` workers each read a file, flag it
`EVICT_WHEN_COLD`, drop it, and start the next. Every KV region filled with
finished readings, and the pass died at its failure cap:

- "no region is claimable … every one of the KV reservation's 3720 regions is
  occupied";
- "wave too wide for the partition … conceded 0 B".

This document replaces the stopgap that paced it, `code_read::wait_evicted`.
That function has each worker poll its own timeline's hot residency for up to
120 s. It is removed by the change that lands this design.

## 1. Why the engine did not push back

Four gaps, each enough on its own:

1. **Admission does not see the debt.** Admission (`scheduler::admit`,
   `promote_new_prefills`) judges an offer by throughput. The rule that forces
   the queue head in whenever nothing is running (`force_queue_head`, and the
   empty-wave head waiver in `admit::fill`) bypasses every floor. A pool that
   submits each next file the moment the last one drops creates those "nothing
   running" moments over and over.
2. **Retiring a reading is lazy and can starve.** `evict_ingest_timeline` only
   marks the residence and triggers the persistence thread. The hot copy of a
   just-finished reading is native R16/F16 with no lower copy, so it cannot be
   dropped until a pass quantizes it to warm. That pass:
   - wakes on a trigger or every 5 s;
   - needs **fresh GPU regions** for its quantize destination;
   - may not create arenas while a forward is open (`is_wave_deferral`).

   Under a full card with back-to-back waves, the one thing that frees regions
   needs regions and gaps it cannot get.
3. **A placement failure retries blind.** The tier-refusal path (`WaveRetry`,
   `retry_step`) backs off and retries 1024 times. It never runs the relief
   ladder (`relieve_vram_pressure`) or asks for a drain.
4. **Producers have nothing to wait on.** `ConversationEngine` exposes no
   back-pressure surface:
   - `BacklogGauge`, `on_backlog_fall` and `region_stats` are crate-private;
   - `Filled::stopped_on_weights` reaches only a debug log;
   - so the pool's width is a constant (`CODE_READ_PARALLELISM = 16`) with no
     engine feedback.

## 2. The design

### 2.1 Retire debt is a counted quantity

**Retire debt** is the hot bytes of residences flagged `EVICT_WHEN_COLD` that
have no warm or cold copy yet: KV that is finished and must still be drained
before it frees anything.

- The substrate keeps it as a counter beside the region pool's own:
  - incremented at `mark_timeline_evict_when_cold`;
  - decremented where the hot copy is dropped (`install_migrated`,
    `demote_idle_hot`) or the residence goes.
- It is exact, not a scan, and is read in `O(1)`.

### 2.2 The engine grants room; producers block on the grant

`ConversationEngine::room(class)` returns a **`Room`** guard. It blocks until
starting one more conversation of `class` fits:

```
free_regions − setpoint − retire_debt_regions ≥ class.reserve
```

- **`class.reserve`** is the regions a conversation of that class can reach at
  its peak. For `code_read` that is one file's reading at `max_file_tokens`.
- **Waiting is a `Condvar`, woken by the events that move the inequality:**
  - a region released (`release_region`, an arena sweep);
  - retire debt falling (the gauge's `on_backlog_fall`, now a fan-out rather
    than a single `OnceLock` slot);
  - a conversation ending.

  No poll and no timeout: the condition is what changes. Cancellation is a
  wake too (`ingest_cancel`).
- **The `Room` is held for the conversation's life** and dropped with it.
  Dropping only lets the next producer re-check: the inequality, not a count,
  is the gate, so a held `Room` reserves nothing on its own.

The ingest pool takes a `Room` before each file, in place of `wait_evicted`.
`CODE_READ_PARALLELISM` becomes the pool's upper bound, and the engine decides
how much of it runs at once.

### 2.3 Admission sees the same inequality, the forced head included

`promote_new_prefills` refuses a producer-class offer that fails §2.2's
inequality, **including the forced queue head**.

- This keeps deadlock-freedom. The forced head exists so the queue always
  makes progress. Retire debt drains without any new admission (§2.4), so
  refusing a producer while debt is outstanding cannot wedge the queue.
- **Dialogue and live turns are not producer-class** and keep today's rules.
  Back pressure never stands a person's reply behind an ingest backlog.

### 2.4 Draining makes progress under pressure

A debt only works as a gate if it is guaranteed to drain:

1. **A quantize reserve.** The persistence migrate gets a small, fixed region
   reserve (`migrate_reserve`, sized for one group's Q destination). Neither
   the scheduler nor the weight broker can spend it. Retiring a reading never
   needs to buy ground.
2. **A migration window.** When retire debt is non-zero and no pass has run
   for `MIGRATE_STARVE` waves, the wave loop closes a gap at a wave boundary
   for the pass's arena work. Today's `is_wave_deferral` can refuse that work
   indefinitely under back-to-back waves.
3. **Debt-driven wakes.** Marking a residence adds debt and triggers a pass
   (it already does). The 5 s tick stays only as the idle fallback.

### 2.5 A refused placement relieves before it retries

The tier-refusal retry (`run.rs` `WaveRetry` path, `decode::retry_step`) runs
the relief ladder before retrying:

- rung 4, `evict_cold_tail` plus `flush_at_least`, now including the retire
  debt;
- then the weight-side concession.

It fails a wave only when **no rung freed anything** across `K` consecutive
attempts. That is a measure of progress, not a count of 1024 blind tries, so a
recoverable shortage never fails a file.

## 3. What this removes

- **`code_read::wait_evicted`**, its `EVICT_WAIT`, and the per-worker 50 ms
  poll.
- **`settle_evicted_timeline`'s residency poll.** It becomes a wait on
  §2.1's debt for that timeline.
- **The stale claim at `scheduler/run.rs:1606`** that the backlog cuts
  `admit_budget`.

## 4. Tests

- **Retire debt is exact.** Mark, migrate and drop a residence, and assert the
  counter's bytes at each step.
- **A `Room` blocks and wakes.** A pool of 16 producers against a reservation
  sized for 4 runs at most 4 at once, and each waiter wakes on a release with
  no sleep in the test.
- **The forced head is refused under debt, and the queue still drains.**
  Simulate a full reservation with outstanding debt and no wave running, then
  assert the producer is held and the debt clears.
- **The migrate never needs ground.** With every non-reserve region live, a
  pass still lands one group in warm.
- **A refused wave relieves, then succeeds.** Fill the KV with retire debt,
  then show a decode wave places after one relief cycle and is never failed.
- **A dialogue turn is never held behind ingest debt.**

## 5. Rollout

A single change, per the no-dual-path rule:

- the debt counter;
- `Room` and its wakes;
- admission's producer-class check;
- the migrate reserve and window;
- the relieving retry;
- deleting `wait_evicted`.

Measured against the night it broke:

- the re-ingest on BlackWorld at 16 workers runs with no "wave too wide" and no
  failed files;
- "files/h" holds at least what `wait_evicted` gave;
- retire debt, `Room` waits and admission refusals are logged once per pass.

# The substrate without a waiting writer

Every reader and writer of a substrate shares one lock (`SubstrateLock`,
`candle-conversation/src/substrate/lock.rs`). This document is how the threads
that must never stall stop waiting for it — the scheduler first — and how the
long holds that make them wait go away, in three steps.

## 1. What the lock costs

`SubstrateLock` logs every hold and every wait of 250 ms or more with its call
site. Ten minutes of BlackWorld under ingest (build 4029, 2026-10-29):

| Who | Site | n | max | total |
|---|---|---|---|---|
| maintenance's liveness scan (read) | `resolver/maintenance_op.rs:92` | 51 | 715 ms | 29.8 s |
| maintenance's plan gather (read) | `resolver/maintenance_op.rs:110` | 12 | 1,010 ms | 9.9 s |
| seal / metadata / tiering writers (wait) | `resolver.rs` `record_turn`, `couple_turn`, `rewrite_for_compaction`, `mark_timeline_transient`, `mint_timeline`, … | ~25 | 808 ms | ~11 s |
| the scheduler's placement ranges (wait) | `scheduler/pending_ranges.rs:27` | 5 | 458 ms | 1.9 s |

Build 4027, before the demote change below: the scheduler's ingest demote
(`scheduler/prefill.rs`, `demote_ingest_once`) waited 56 times, up to 715 ms,
24.9 s in ten minutes — the scheduler thread stalled, every forward behind it.

Two facts shape the design:

- **The writers are short and the readers are long.** Every write measured is
  milliseconds once it holds the lock; every slow wait is behind a read that
  walks the whole substrate — maintenance's two scans, up to a second each.
- **On Windows a waiting writer queues new readers behind it.** One long read
  and one waiting writer stall every reader that arrives after them, the
  scheduler's projection reads included.

Passing over a held substrate (`try_write`, skip this wave) is not an answer:
the long reads are frequent enough that the skip is taken nearly every wave,
and what it was to relieve — the ingest demote — never runs while the ingest
backs up behind it.

## 2. Writes as messages (step 1)

A mutation whose caller does not read its result back is **sent**, not
applied: `SubstrateLock::enqueue` puts it on a lock-free queue and returns at
once. It is applied by whoever next holds the write lock:

- **Every write acquisition drains the queue first**, in order, before its own
  mutation. A synchronous writer therefore always lands after everything sent
  before it — nothing queued is overtaken, and a write that must be seen at
  once (`record_turn`, `restore_section`, an eviction whose freed bytes the
  caller prices) stays synchronous and correct.
- **An applier thread** — one per substrate, started with its first message —
  takes the write lock whenever the queue is non-empty, so a message is applied
  within one long hold of being sent even when no other writer comes. The
  applier waits; the sender never does.

Nothing is skipped: every message is applied, at most one long read after it is
sent. A reader that runs before then reads the state as it was before the
message — which is why only mutations nobody reads back go through the queue.

**A message runs on whatever thread drains it, so it must not free device
memory.** A chunk freed off the scheduler thread mid-compaction re-enters the
free list as a destination the pass's own claims can take — the reason every
off-thread drop takes `migrate_flight::try_hold_chunk_locations` and defers
when refused, which for a demote is a skip again. A message that takes
device chunks out of the substrate hands them back to the thread that owns
their release — over a channel, dropped there.

### The ingest demote, first

`demote_cold_ingest` is the first mutation sent. It decides which warm-backed
ingest residences to shed and takes their hot copies out of the substrate; the
copies go back to the scheduler, which drops them on its own thread at its next
wave and releases the emptied arenas, as it did when the demote was inline.

- **One demote in flight per substrate.** While one is queued the scheduler
  sends no other: each would shed toward the same target, computed from the
  same pressure, and together they would overshoot it. This is coalescing, not
  skipping — the one in flight is always applied.
- **What it shed is reported when it is applied**, with its target: the wave
  stats' eviction row, and the persistence nudge when it shed less than its
  target (the warm-starved case), both from the returned batch.
- Each substrate's demote gets the whole target. Ingest timelines are one
  tenant's, so in practice one substrate is shed.

## 3. Liveness without a scan (step 2)

Maintenance's liveness scan — every stream, every chunk, summed per segment —
is the longest and most frequent hold (§1). The substrate sees every change
that makes a record live or dead: an append, a supersession, a tombstone, a
distill, an orphaning. Per-segment live bytes kept up to date at those points
make the plan read counters, not walk the substrate.

The liveness rules (`visit_counted_live_records`: tombstoned, distilled,
orphaned, carried signatures) are subtle and have churned segments when one
was missed. The full scan stays, run as an audit against the counters off the
planning path, and a disagreement is logged with the segment and the record
classes that differ.

## 4. Reads from a snapshot (step 3)

With the writers off the lock (step 1) and the longest reader gone (step 2),
the remaining contention is projection reads against seals. If the lock log
still shows it, readers move to an immutable published snapshot
(`arc-swap`-style), written by the one applier: the large indexes (`streams`,
`residence`, the tombstone sets) become structurally shared maps so publishing
a version is O(changed), and a reader never waits at all. It is the largest
change — `substrate.rs` is ~12k lines and every map operation gets dearer — so
it is taken only on the evidence of the lock log after steps 1 and 2.

## 5. Measuring each step

The lock log is the instrument: the same ten-minute table as §1 after each
step, on the same machine under ingest. Step 1 is done when no site on the
scheduler thread waits (`scheduler/*`); step 2 when the
`maintenance_op.rs` holds are gone from the table.

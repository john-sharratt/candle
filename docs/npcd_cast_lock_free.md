# The cast without a lock

npcd's cast — every character it knows (`npcd/src/npcs.rs`) — is read by the
console, the turn loop, the pulse, the effector and the load walk, and written
by the console's edits, the effector's agency acts, a character naming its
mood, and the replication refresh. It was one `tokio::sync::RwLock<Npcs>`
around all of it. This document is the design that replaced that lock; it
follows `docs/substrate_lock_free.md`, whose persistence owner it builds on.

## 1. What the lock cost

Every write held the cast's **write** lock across a call to the persistence
owner that appended the record and fsynced it. The owner is the one thread
that serves the whole redo log, so the write lock was held for the fsync *and*
for however long the owner was busy with the engine's own appends ahead of it.

| Writer | Where | How often |
|---|---|---|
| an agency act (`plan_*`, `orders_*`) | `effector/plan.rs` | every act a character takes in a turn |
| a mood | `tenant.rs` (`remember_mood`) | whenever a character names a new register |
| the replication refresh | `tenant.rs`, `authority/cast.rs` | every pass that takes in another member's writes — and it walked every character's record location inside the same owner call |
| a console edit | `api.rs`, `portrait.rs` | a person pressing save |

`tokio::sync::RwLock` is fair: a writer waiting holds back every reader that
arrives after it. So each of those writes stalled every read behind it — the
console's roster, the pulse, the plan reads, and the turn loop's persona read,
which the next character's turn waits on. One character's fsync held up
another character's turn.

## 2. The design

**Readers load a snapshot. Writers run on the log's own thread.**

- The cast is an immutable value, `Cast` (`npcs/cast.rs`): the characters in
  an `imbl::OrdMap` keyed by id — the same order the `BTreeMap` it replaced
  iterated in, which `cast()`, `export_all()` and the roster rely on — and the
  record each was last read from. `Npcs` holds it in an `ArcSwap`.
- **A read** is `Npcs::view()`: one atomic load, an `Arc<Cast>` that stays
  whole for as long as the reader holds it. It never waits, and nothing a
  writer does changes a snapshot a reader already has.
- **A write** is one call to the persistence owner. Inside that call it loads
  the current cast, computes the records it supersedes, appends them, fsyncs,
  and only then publishes the cast with them in. Every cast write passes
  through that one thread, so writes are totally ordered with each other and
  with their appends — there is no compare-and-swap to lose, and no second
  writer between the read a write is computed from and the record it lands.

The ordering is the one `Npcs` always kept, now enforced by where it runs
rather than by a guard: **the record is durable before memory says so, and
memory says whatever is durable.** A refused edit publishes nothing. An append
that fails partway through a batch (`import`, `repoint_portraits`) commits and
publishes the records staged before it and returns the error, so the daemon's
view never disagrees with the log. Replay keeps the newest record per
character, and the newest record is always the one the newest published cast
holds, because the cast is published in the same step that appends it. Each
published record is also the one its character is *read as*, so a refresh does
not read the daemon's own writes again.

A writer waits on the owner for its own append and fsync. Called from an async
task on a multi-threaded runtime, that wait is taken off the worker
(`block_in_place`), so writes from many tasks at once never park every worker
the readers need.

### 2.1 A write computed from what it writes

`break_down` names a new child strategy `<parent>-<n>` from the children the
parent already has. It used to compute that id from a read and then write it,
holding the write lock across both. The id is now part of the write: the
target of a strategy write is `StrategyTarget::Id` or
`StrategyTarget::ChildOf`, and a `ChildOf` is resolved against the record the
write supersedes, inside the owner call. Two break-downs in a row see each
other's children because the second is computed on the owner's thread after
the first was published there.

### 2.2 The refresh

`refresh` takes in records another member wrote. It compares each
character's current record identity against the one the cast was read from,
reads the ones that moved, and publishes a cast with them — all inside one
owner call, so a write cannot land between the identities it read and the
cast it publishes.

### 2.3 What a write must not do

A write's computation runs on the persistence owner's thread. It must be pure
over the `Cast` it is handed: no call back into the owner (that would wait on
itself), no await, no I/O. Every write in `npcs.rs` is validation and a record
built from the snapshot.

## 3. What remains

| What | Why it is not a critical section |
|---|---|
| the persistence owner's queue | a write waits for its own append and fsync, which it always did; no reader waits on it |
| `Npcs` itself | `Arc<Npcs>`, shared; nothing in it is locked |

At shutdown nothing waits for the cast: a write in flight is a call queued on
the owner, and the owner serves every queued call before it stops
(`docs/substrate_lock_free.md` §9.1).

## 4. Tests

| Invariant | Test |
|---|---|
| a refused write publishes nothing and appends nothing | `npcs::tests::a_refused_write_publishes_nothing` |
| a reader's snapshot is unchanged by a later write | `npcs::tests::a_snapshot_is_not_moved_by_a_write` |
| concurrent writes to one character lose nothing | `npcs::tests::concurrent_writes_to_one_character_all_land` |
| `break_down`s in a row never share a child id | `effector::plan::tests::break_downs_in_a_row_each_mint_their_own_child` |
| the refresh takes in another member's records | the existing replication tests (`authority/cast.rs`) |

# The substrate without locks

Every reader and writer of a substrate, of its redo log, and of zend's engine
waited on one of a handful of critical sections. This document is the audit of
all of them — what each guarded, who took it, and the races that shape allowed
— and the design that removed them. It replaces
`docs/substrate_mutation_queue.md`, whose queue moved some writes off the
scheduler but left every reader blocking and every queued write invisible
until applied.

**The goal, stated as properties:**

1. **No reader waits.** A projection, a scan, a GUI view, a status poll
   never blocks on another thread, however long that thread's work is.
2. **No writer waits on a reader.** A write never queues behind a scan.
3. **Every write is visible at once.** When a write returns, every later read
   on any thread sees it (no "applied at the next writer").
4. **No race.** Every decision a thread makes from shared state is either
   made and applied in one atomic step, or re-validated atomically at the
   step that acts on it. No check-then-act across two holds.
5. **Device memory is freed on the thread that owns it**, whoever decides to
   free it.

The one deliberate wait that remains is **flow control**: a producer facing a
full writer queue under overload waits for room (§9.4). That is not a critical
section — nothing is held while it waits.

## 1. Why this is possible: the substrate is already a CRDT

The replicated substrate (`docs/replicated_substrate.md` §6.5, §6.9) defines
the durable state as an **order-independent fold of stamped records**:

| Record kind | Merge rule | Concurrency consequence |
|---|---|---|
| Keyed values (`StreamDecl`, chunk/tokens locations, `Snapshot`, `ConvState`, `TreeMetadata`, `DebugId`, `CustomObject`, `ProjectionEvents`, `WideQSig`, `TurnIndexPage`, …) | greatest version stamp wins | a **max-register**: commutative |
| `Label` | per field, greatest stamp | one max-register per field |
| `Commit` | `committed_through = max(…)` | commutative |
| couplings, VFS kills, applied-set gaps | set insert | insert-only set |
| tombstones (timeline, turn, section generation) | dominate | insert-only set, checked at read and at apply |
| a timeline's turns | ordered by decl stamp, gaps held | one immutable entry per timeline |

Every apply commutes. **Commuting operations need no mutual exclusion — only
atomic publication.** The global `RwLock` serialised operations the data model
already proved independent; it bought nothing an atomic swap of an immutable
version does not, and cost every stall measured (§2).

The transient state — KV tiers, pins, working sets, scopes, caches — is not
replicated and not a CRDT, but it is per-residence or per-timeline, and each
piece has a natural atomic representation (§5).

## 2. The audit

Four read-only passes mapped every lock in `candle-conversation`, zend and the
persistence layer (2026-10-29, build 4042), and the findings below were
checked against the code. This section is the record of what the design
replaced; §3 onward describes what replaced it.

### 2.1 The locks

| Lock | Where | Guards | Taken by | Worst holds |
|---|---|---|---|---|
| `SubstrateLock` (`RwLock<Substrate>` + mutation queue) | `substrate/lock.rs`, `Conversation::inner` | all 49 fields of `Substrate` (§2.2) | every thread: scheduler, persistence, writer, applier, replication, zend handlers and ingest, summariser, maintenance | maintenance reads 0.5–1.0 s every ~15 s; replication batches 0.25–1.16 s; projection reads |
| `PersistenceLock` (`Mutex<SubstratePersistence>` + waiter count) | `persistence/priority_lock.rs` | the log (segments, staging, handles), manifest, accounting, header index, location maps, relocation watch, replica/rebase, minter handle | writer thread (one hold per job), maintenance batches, cold reads, elevate, replication, custom objects, model binding | relocation batches ~150 ms; compaction 16–31 s; I/O and fsync under it |
| `Mutex<ConversationEngine>` | zend `state.engine`, ~155 sites | **nothing of its own** — every engine method takes `&self`, and npcd shared the same type without a mutex | every zend path | held across scheduler round-trips, prefills, git lookups, VFS restores, full metadata scans (15.6 s measured before a fork, 2026-10-29) |
| `conversations` map (`Mutex<HashMap<String, Arc<ConvLock>>>`) | `zend/src/session.rs` | which conversations are resident | turn path, sweep, deletes, file changes | held across `fork_resuming` — a scheduler round-trip |
| `base_conv: Mutex<Sequence>`, ingest bases | `session.rs` | the dialogue base, each ingest layer's base | forks, history reads | across `fork_resuming` |
| `ConvLock<ConvState>` (tokio mutex) | per conversation | one turn in flight per conversation | turn path | the whole turn — **by design** (§8.4) |
| `events_cache: Mutex` (inside the substrate) | `substrate.rs` | decoded events memo | belief scans under the substrate read | decode under the inner lock |
| `Conversation` side locks | `resolver.rs` | `normalization`, `maintenance` (status), `branch_checkpoint`, `gallery_cache`, `held_index_pages`, `conv_state_writes` | scheduler, persistence, API | `gallery_cache` taken inside a substrate read |
| GalleryArena `residency`, `index_cache`, `zero_page`, dedup `book`/`masks` | `provenance/gallery_arena/` | resident pages, indices, codebook, masks | scheduler | `residency` across device uploads |
| Writer `Backpressure` / `Progress` / handle | `persistence/writer.rs` | queue depth, write tickets | producers, the writer | flow control |
| Minter | `persistence/identity/minter.rs` | origin sequence, section generations | every minting path | leaf |
| `CHUNK_LOCATIONS` | `migrate_flight.rs` | chunk addresses during GPU moves | migrates, off-thread frees, compaction (all `try_`) | held across GPU work |
| Inherited-log cache and file | `persistence/inherit.rs` | loaded inherited substrates; the seek cursor of each | cold loads | a load under the cache lock's check |
| Engine internals | `engine.rs` | `plain_frames`, `attachments`, `scheduler_handle` | engine methods | short |
| Sampler, telemetry, statics | `batched_sampler.rs`, `phase_ring.rs`, `memory_report.rs`, `relief_trace.rs` | scheduler-owned buffers, rings | scheduler (+ status reads) | `LEARNED_STORE` across `fs::write` once a minute on the scheduler |

### 2.2 What the substrate lock protected

The 49 fields fall into six groups.

| Group | Fields | Durable? | Who writes |
|---|---|---|---|
| **Residences and tiers** | `residence` slab (hot / warm / cold payloads, flags, `last_used_epoch`, `byte_size`), `hot_lru`, `warm_lru`, `working_set_pins`, `wave_epoch` | cold refs re-derived; rest transient | scheduler (install, evict, demote, lift), persistence thread (hot→warm, cold land, idle demote, purge), writer (cold install), API threads (tombstone, distill — **freeing device memory**), maintenance (cold-ref refresh) |
| **Timelines** | `timelines` (turns, tree meta, label, conv_id, custom, gaps, summariser queue), `timelines_by_group`, `timeline_token_totals`, `calibrations_of`, `timeline_by_debug_id`, `pending`, `conv_order_counter`, `conv_active_counter`, `reload_due` | yes (derived indexes rebuilt) | scheduler (seal, ranges, last selection), zend (labels, state, metadata), summariser, replication, walker |
| **Streams** | `streams` (`StreamIndex`), `tagged_turns`, `gallery_epoch`, `events_cache` | yes | seal, writer (locations), maintenance (relocation), replication, walker |
| **Death** | `tombstoned_timelines`, `tombstoned_turns`, `ended_sections`, `deleted`, `distilled_timelines`, `retired_snapshots`, `tombstone_epoch` | yes | zend, retention, replication, walker, compaction |
| **Keyed singletons** | `recurrent_snapshots` + `snapshot_ids`, `branch_checkpoints` + ids, `custom_objects` + ids + version, `sections` + `section_token_total`, `owned_sections` | yes (sections restored at triage) | writer, scheduler (sections), zend |
| **Configuration and runtime sets** | `timeline_compression`, `scoped_groups`, `working_set_shares`, `retrieval_scopes`, `working_sets`, `transient_*`, `unresumed_timelines`, `splice_source_timelines`, `layer_corrupt_turn`, `disabled_layers`, `append_only_layers`, `ingest_sees`, `chunk_exclusion` | transient | setup, zend, scheduler |

### 2.3 The races that shape allowed

Wrong results or unsafe frees, each reachable under the locked shape. §10
maps each to the mechanism that closes it.

| # | Race | Effect |
|---|---|---|
| R1 | **A seal was two writes.** `record_turn` appended the turn under one write and applied its stream declaration under a second. | a reader between them saw a turn with no declaration: no tags, no gallery-epoch bump, writer 0 in its tombstone mark |
| R2 | **A lift installed without re-validating.** `elevate_to_hot` classified under a read and `install_promoted` installed later with no check that the residence was live or not already hot. | hot K/V put back on a tombstoned residence; duplicate LRU entries |
| R3 | **Tombstones freed device memory off the scheduler thread, unguarded.** | a chunk freed mid-compaction re-entered the free list as a destination the pass claimed; a forward attending a pinned turn lost it |
| R4 | **The pre-lift eviction protected only its own projection**, over an LRU spanning every conversation; `demote_turns_to_warm` checked neither pins nor a keep-list. | another live slot's turns demoted under it |
| R5 | **Two lifts published no pins and stamped no use.** Priming and section-ingest prefixes lifted without a pin; `install_section_hot` stamped no epoch. | the idle demote dropped a freshly lifted section before it was read back |
| R6 | **Hot→warm and quantize swaps installed over changed state.** `install_warm_and_hot` and `replace_section_hot` replaced `hot` unconditionally. | a tombstoned or re-lifted residence got an old copy installed |
| R7 | **Cold install used a stale residence**, installed by the writer long after `mark_cold_pending`. | cold refs installed on a residence whose content changed |
| R8 | **The branch-checkpoint read had no moved-location retry.** | "indexed but unreadable" after a relocation landed between the location read and the payload read |
| R9 | **Appended but not yet indexed, at maintenance planning.** Writer and direct paths appended under persistence, released it, then indexed under the substrate write. | a record rolled back on reload; the substrate pointing at a deleted file |
| R10 | **Own-written high water skipped gaps.** Runtime paths minted, wrote RAM, then enqueued or appended directly, so oseqs reached the log out of order. | a peer told an oseq was written whose predecessor was still queued |
| R11 | **Pins were replaced wholesale and owed.** One scheduler's view replaced the set; a busy lock made the republish owed. | pins dropped between elevate and apply; a second scheduler on the substrate would erase the first's |
| R12 | **zend read-modify-writes across two engine holds** — `delete_upload`, uploads appends, `set_forked_from`'s cycle check, `holder()`→`repoint()`. | lost updates; a lineage cycle; a link repointed to a tombstoned holder |
| R13 | **The ingest demote ignored pins**, protecting only the keep-list captured when it was sent. | another queued sender would have evicted a live working set |

## 3. The shape

The substrate is **one versioned root** — an immutable `Substrate` built of
persistent collections, published by compare-and-swap — beside **one slab of
residences** that changes by atomic steps and belongs to no version.

```text
SubstrateLock  (ArcSwap<Substrate> — substrate/lock.rs)
 └─ Substrate  (Clone in O(1): imbl maps and vectors — substrate/pcoll.rs —
     │           with each timeline's entry behind its own Arc)
     ├─ timelines, streams, death sets, keyed values, derived indexes,
     │  configuration and runtime sets, epochs           — all in the version
     └─ residences: Arc<ResidenceSlab>                     — shared by every version
```

| Operation | Implementation | Waits on |
|---|---|---|
| **Read** | `SubstrateLock::read` loads the latest `Arc<Substrate>` (`SubstrateReadGuard`) | nothing |
| **Edit** | `SubstrateLock::update(FnMut(&mut Substrate) -> R)`: clone the version (O(1)), run the edit, compare-and-swap it in; a lost swap runs the edit again on the newer version | nothing — a lost race is a retry |
| **Tier transition** | an atomic step on the residence (§5) | nothing |
| **Log write** | queued to the persistence owner (§9) | flow control only |

An edit runs again whenever another writer published first, so **an edit is a
pure function of the version it is handed**: it reads only that version and
changes nothing outside it — no I/O, no channel sends, no minting. Work with an
effect outside the substrate is done by the caller after `update` returns. The
one exception is residence allocation: an edit allocates the residences it
names, and the slab tracks each attempt's allocations
(`residence::begin_attempt` / `end_attempt`), so a lost attempt's residences are
discarded rather than left holding memory no version names. Edits do not nest:
an edit that would edit again (directly, or by a call that waits on another
thread's edit) retries for ever, and `begin_attempt` asserts against it.

An edit that is slow or retried often is logged with its call site
(`lock::SLOW`, `lock::RETRIES_NOTED`): split it, or move work out of it.

**What "consistent" means.** A reader holds one whole version: every timeline,
stream and index in it agree, whatever commits after. A turn is in the version
with its declaration, tags and layout, or not at all (R1). Two reads at
different instants are two valid states — the CRDT property (§1) makes every
set of applied records a valid state, which is exactly what a replica sees.

**Read-your-writes** holds trivially: a write is complete when its swap
returns, and nothing is queued.

## 4. Ordering rules

1. **An epoch moves in the same version as the change it announces.**
   `gallery_epoch`, `tombstone_epoch` and `custom_objects_version` are fields of
   the substrate, so a reader that sees the new epoch sees the change, and one
   that sees the change sees the epoch. Caches keyed on an epoch (the gallery
   cache, the events memo) are filled from the version that carried it.
2. **A tombstone is in the version before the residence is killed** (§5.2): a
   reader that finds a live residence of a dead timeline filters it at the
   dominance check, and one that finds a dead residence never lifts it.
3. **A conversation-state change is published, then minted, then stamped**
   (§6.2), so the greatest identity carries every change published before it.

## 5. Residences and tiers

### 5.1 The residence

The slab is append-only (`boxcar`): an index, once handed out, names the same
`Residence` for the substrate's life, and pushing never moves an existing one.
Each residence (`substrate/residence.rs`):

```text
Residence {
    hot:   ArcSwapOption<HotKv>                 // VRAM chunks
    warm:  ArcSwapOption<Vec<SealedSequence>>   // RAM copy
    cold:  ArcSwapOption<Vec<StoredSequence>>   // redo-log references
    flags: AtomicU32     // PENDING_QUANTIZE EVICT_WHEN_COLD COLD_PENDING
                         // NO_COLD_PERSIST SPLICE_SOURCE DEAD SECTION
    pins:  AtomicU32     // working sets holding it (§5.3)
    byte_size, last_used_epoch, touched: AtomicU64
    stream_id, compression                      // immutable, set at alloc
}
```

The slab also keeps lock-free sets of the residences whose hot and warm cells
may hold a payload (papaya), the idle demote's epoch, and the recency clock.

### 5.2 Transitions

Every transition is a step on the residence's own cells, re-validated in the
step that acts:

| Transition | Step | Closes |
|---|---|---|
| install hot (seal, lift) | refuse a `DEAD` residence; store; re-read `DEAD` and take the payload back if a kill landed between | R2 |
| lift | `install_hot_if_absent`: compare-and-swap from empty, so two lifts never stack and a lift never lands on a hot residence | R2 |
| migrate / quantize swap | `replace_hot(expected)`: compare-and-swap from the exact copy the migrate read; a residence killed or re-lifted since takes nothing (`install_migrated`, `replace_section_hot`) | R6 |
| evict hot | taken only from a residence no publisher pins and no splice holds, with a warm or cold copy behind it | R3, R4, R13 |
| cold land | `install_cold` stores the references and clears `COLD_PENDING`; turn content is immutable per residence, so the references describe what the residence holds | R7 |
| kill (tombstone, distill, retire) | set `DEAD`, then take hot (and warm when no cold copy will come) | R3 |

**The kill/install pair** is sequentially consistent: a kill sets `DEAD` and
then takes the payloads; an install stores its payload and then reads `DEAD`,
taking its own payload back when it is set. One of the two always sees the
other, so a dead residence never keeps a payload. **Membership is re-checked
after a removal**: a residence leaves the hot or warm set only when its cell is
empty, and is put back when an install landed between the swap and the
removal.

**An install is a use.** Every hot install stamps `last_used_epoch` with the
current epoch, so the idle demote leaves a freshly sealed or lifted residence
for its whole grace window, however long ago it was last read (R5). A residence
is also born at the current epoch, never at zero.

### 5.3 Pins

Pins are **counts per publisher** (`ResidenceSlab::publish_pins(owner, set)`):
each publisher — the scheduler is `SCHEDULER_PINS` — replaces its own set, and
the slab adjusts each residence's count by the difference. Two schedulers on one
substrate pin independently; their counts add (R11). A publish also stamps
every residence it holds as used this epoch. `release_all_pins` is the
shutdown drain's.

### 5.4 Device memory: the reclaim queue

A hot payload is a `HotKv`. Whoever takes it out of a residence — evictor,
killer, demote — drops its `Arc`; when the last holder drops it on any thread
but the device's own, the chunks go to a lock-free reclaim queue instead of
being freed there. The scheduler thread marks itself the device's owner
(`own_device_frees`) and drains the queue at each wave boundary and before a KV
compaction plans (`drain_reclaimed`), so a pass never runs with frees pending
from an earlier wave, and no chunk is freed under a pass (R3).

A payload in the queue is still a holder: its chunks stay allocated until the
next drain. A hot copy dropped off-thread while a pass runs lands in the queue
after its residence stopped naming it, so the pass's sweep visits the queue as
well, after every residence (`rewrite_reclaimed`). A copy dropped before its
residence was visited is rewritten there; one dropped after is already the
rewritten copy. Without it, the pass's copies of those slots are named by no
holder it reached: measured as 160 of 57,985 relocated slots in one pass on the
local zend run, reported by the pass's `unwitnessed` count.

A detached tenant's substrate is a holder in the same way. A detach takes it out
of the scheduler's attached set, but its residences keep their resident copies
until its last handle goes — a tenant's shutdown drops those handles after the
detach, not at it. So the sweep visits every detached substrate still alive,
before the reclaim queue; one that has gone holds nothing, its dropped copies
being in the queue. Each is upgraded from the weak reference the scheduler
already keeps of it once, before the pass (`Detached::alive`), and let go after
the pass returns: upgraded inside the pass, a tenant dropping its last handle
meanwhile would leave the pass's upgrade as the last, and the substrate would
free its chunks into the pool mid-relocation. Without it, a pass that ran in
that window found most of the arena unnamed: measured on WhiteWorld at a member
shutdown, 39,321 of 94,936 and then 55,981 of 137,287 relocated slots, in the
two passes that ran between npcd's detach and the engine's shutdown.

### 5.5 Recency without lists

`hot_lru` and `warm_lru` are gone. Each install and each use stamps the
residence from the slab's clock (`touched`); a walk that needs oldest-first
order (`hot_turns_oldest_first`, `warm_oldest_first`) reads the membership set
and sorts by the stamp. Sections are in the hot set and never in the turns'
order: no recency-ranked eviction weighs them against turns, and only the idle
demote takes their hot copy.

### 5.6 Chunk locations under a compaction

A KV compaction moves chunks between slots of a live arena, so a migrate that
captured chunk addresses must not run across one. `CHUNK_LOCATIONS`
(`candle-nn/.../migrate_flight.rs`) is one atomic word — a holder count and a
frozen bit — and every step on it is one compare-and-swap that refuses rather
than waits: a migrate or an off-thread snapshot holds it shared, a compaction
freezes it, and whichever is refused comes back on its next pass.

## 6. Timelines and streams

### 6.1 One edit, one decision

Timelines, streams, death sets, keyed values and the derived indexes are all
fields of the version, so every multi-field decision is one edit:

- **The seal is one edit** (R1): the turn's entry, its declaration, tags and
  layout enter the version together (`record_turn`, `record_summary_turn`).
- **A new conversation claims its timeline in one edit** (`new_conversation_on`).
- A record for a timeline not yet registered is held in the version's pending
  store and applied by the edit that registers the timeline.

### 6.2 Conversation state

A conversation's whole state (`ConvState`: archive flag, branches, activity) is
one record whose greatest identity supersedes the rest. A change is published in
one edit, then the identity is minted, then a second edit stamps it and reads
the state to write (`update_conv_state`). Whichever writer mints last reads a
state holding every change published before its mint, and every change's own
mint follows its publication — so the record with the greatest identity holds
them all, with no lock between the two edits.

### 6.3 Metadata read-modify-write

`Conversation::update_conversation_metadata(timeline, key, next)` rewrites one
`custom` key from the value it holds, as one step (R12). `next` is handed the
version and the current value, so a decision it makes from anything else in the
substrate — a lineage cycle, a tombstone — is part of the same edit. The write's
identity is minted before the edit, so an edit that read a value set by a later
identity than its own would lose to it field by field: it is refused
(`Substrate::custom_field_takes`) and the step starts again under a fresh
identity. An identity that writes nothing is queued as `WriteJob::Abandoned`,
marked never-coming in write order, so it holds back no peer (§9.3).

### 6.4 Memos

The decoded events digest is a `OnceLock` on the stream's runtime entry, reset
when the events blob is written or pruned; nothing locks around the decode.

## 7. Caches beside the version

The `Conversation` handle's side state is lock-free:

| State | Representation |
|---|---|
| normalization levels | papaya map of `ArcSwap<ScopeState>`; `observe` is a read-copy-update |
| maintenance status | `ArcSwap` |
| branch checkpoint | `ArcSwapOption` |
| gallery cache | papaya map of `Arc<(epoch, gallery)>`; a fill keeps the newer epoch |
| held index pages | papaya map with an atomic sweep threshold |
| inherited-log cache | papaya map; inherited reads are positioned reads of a file nothing writes |
| minter | `ArcSwap<Origination>` read-copy-update; section generations in a papaya map |
| learned rates | `ArcSwapOption`; the store files are written on a thread of their own |

## 8. zend

### 8.1 The engine is shared bare

`Mutex<ConversationEngine>` protected no engine state (§2.1). zend holds the
engine directly and every path takes `&ConversationEngine`. So do the dialogue
base and every ingest base: every fork and every read on a `Sequence` takes
`&self`.

### 8.2 The resident conversations

`zend::resident::Resident` holds the resident conversations by `conv_id`
without a lock. Each key owns a cell: the first request for a key inserts it and
forks into it, every other request for the same key awaits that cell, and a
request for a different key waits on nothing. The fork runs outside any lock;
it takes the conversation's own turn lock before the cell publishes the value,
so a second request parks on it until the fork has restored the conversation's
files.

The idle sweep releases only a conversation nothing holds. Its "nothing holds
it" and a request's "it is still resident" are one handshake: a request
announces itself (acquiring, or by the handle it took), then confirms the entry
is resident and not retiring; the sweep marks the entry retiring, counts handles
and acquirers, and either clears the mark or removes the entry. Both fence
between their step and their check, so a request that kept its handle was
counted, a removed entry has no request holding it, and no second fork starts
under the key while a request holds the first.

A removed entry is freed by the map only in a batch, once enough entries have
been retired, so the entry's value is a slot (`ArcSwapOption`) emptied as the
entry leaves the map — by `remove` and by the sweep. The released conversation,
with its scheduler slot and resident KV, goes when its last holder lets it go;
the map's retired copy holds nothing. npcd's live characters
(`engine/mind.rs`) are held the same way.

### 8.3 Read-modify-writes

| Sequence | Implementation |
|---|---|
| uploads appended, deleted, re-pointed, given stats | `edit_upload_records` → one `update_conversation_metadata` step |
| `set_forked_from` cycle check | the check runs inside the step that writes the parent |
| `holder()` → `repoint()` | `repoint` refuses, in the same step, a holder tombstoned since it was found, and the caller reads the unit again |
| loaded state and loading substrate | `ArcSwapOption` / `OnceLock` |

### 8.4 What stays

`ConvLock` — one turn in flight per conversation — is not a critical section
over shared data; it is the conversation's turn order, and a second submit for
the same conversation is meant to wait its turn. It stays, held across the
turn, and nothing else is ever taken under it that another conversation waits
for.

## 9. The redo log

### 9.1 One owner

One thread owns a substrate's `SubstratePersistence`
(`persistence/owner.rs`); no other thread touches it. Every other thread hands
it work over three lock-free queues:

1. **Calls** — a closure run on the log with its result handed back
   (`PersistenceOwner::call`): a keyed write and its commit, a cold-read plan,
   a reload's recovery reads. Served between any two writes, so a call waits for
   at most one write job.
2. **Writes** — the appends a producer queues and forgets (`WriteJob`), taken in
   submission order, up to 512 per group commit. Each is appended and then
   indexed in the substrate before the next job, so a record is indexed before
   its segment can seal (R9). `call_after_writes` runs a closure in this queue's
   order — what a full rewrite of the log uses, since it must see every record
   whose in-memory change came first.
3. **Background** — maintenance, one batch per job (`call_behind`), run only
   while nothing else waits.

A call from the owner's own thread would wait for itself and panics. A call's
closure borrows from its caller's frame; the caller waits for the owner to run
and drop it before the frame returns.

**A call is short.** The owner is one thread, so nothing on the substrate moves
while a call runs: not writes, not replication, not the calls behind it. Work
as long as the disk is slow gets what only the owner can give it — a plan,
locations, handles — and does the reading on its own thread. A liveness audit
that walked every segment on the owner stopped BlackWorld's zend for fifteen
minutes on 2026-10-31: `AuditPlan::walk` now runs on the caller. A call that
holds the owner 250 ms or more is named in the log by its caller.

**A status read is not a call.** What may be stale without harm is published as
`LogFacts`, atomics any thread reads: the record types held, the space latch,
the last dead ratio. `own_sendable_through` stays a call, since a lease's next
holder waits for that `oseq` and a stale, low answer would let it serve before
it had what the last holder wrote; the applied set and the deferred records,
which the same protocol paths read, stay with it.

A shutdown serves every call already queued before it commits the tail and
acknowledges: each is a caller blocked on its answer, and the owner's wait,
picking among ready queues, may take the shutdown first. A call sent after the
owner has stopped panics its caller, as it always did.

### 9.2 Readers

A durable record is read without the owner at all: `SealedReader` reads sealed
segments and the committed prefix of the active one positionally, published by
the log as atomics. A location that moved between the read of the location and
the read of the record is followed (`read_located`, R8). A cold load asks the
owner for its plan and a set of `Arc<DirectFile>` handles (`cold_read` →
`ColdSource`) and does the read and the GPU upload on its own thread.

### 9.3 Replication's high water

The own-written high water a member offers its peers is the **gapless prefix**
of oseqs written (`own_sendable_through`): it stops at the first gap above the
open top that no never-coming mark covers. A failed append marks its identity
never-coming, and so does an identity minted for a write that was not made
(`WriteJob::Abandoned`), so neither holds the prefix back (R10).

### 9.4 Flow control

The write queue is bounded by events (16,384) and by bytes (512 MiB, bulk jobs
only). A producer facing a full queue backs off — holding nothing — until the
owner drains it. A single in-flight job always makes progress, however large.
`wait_written(ticket)` returns once every job at or below the ticket is
appended.

### 9.5 Compaction and maintenance

A full rewrite (`compact_shared`) runs on the owner after every queued write.
The rewrite returns what it carried (`Rewritten`), which is re-applied to the
live version — idempotently, since the fold commutes — in batches of edits;
`drop_locations_into` then drops every location still into a replaced segment
(a record the rewrite did not carry), and `forget_dropped_stream` forgets a
stream the rewrite dropped wholesale unless a record the rewrite never saw has
declared it again. The replaced segments are unlinked only after the re-apply.

A segment maintenance op plans on a version, runs its batches as background
calls on the owner, and applies its result in one edit.

### 9.6 Replication apply

A remote batch is decided on a private clone of the version (dominance,
outranks, adoption plans), appended on the owner, and published in one edit, so
no reader sees half a batch and nothing waits on the batch.

## 10. Every race, closed

| # | Closed by |
|---|---|
| R1 seal in two writes | one edit (§6.1) |
| R2 lift onto a dead or hot residence | `install_hot_if_absent` and the kill/install pair (§5.2) |
| R3 unguarded off-thread free | the reclaim queue (§5.4); pins and splice sources in the evict step |
| R4 eviction of another slot's turns | per-residence pins honoured by every evict and demote (§5.2–5.3) |
| R5 lifts without a use | an install is a use (§5.2) |
| R6 swap over changed state | `replace_hot(expected)` (§5.2) |
| R7 stale cold install | immutable turn content per residence; `DEAD` residences never lifted (§5.2) |
| R8 checkpoint read after a move | `read_located` (§9.2) |
| R9 appended-not-indexed | append then index on the owner, before the next job (§9.1) |
| R10 high water over a gap | gapless prefix with never-coming marks (§9.3) |
| R11 wholesale pins | per-publisher counts (§5.3) |
| R12 zend multi-call RMWs | one-step metadata edits, the resident map (§6.3, §8) |
| R13 ingest demote ignores pins | it evicts through the same step (§5.2) |

## 11. Proving it

| Property | Test |
|---|---|
| a reader keeps its version; concurrent edits all land; a lost attempt's residences are discarded; a panicking edit publishes nothing | `substrate/lock.rs` |
| a dead residence keeps no payload; lifts never stack; a replace expects its copy; membership follows the cells; pins count per publisher; a lift is a use | `substrate/residence.rs` |
| a migrate over a residence that moved on installs nothing | `substrate.rs` |
| calls return and borrow; a call's panic unwinds into its caller; writes land in order; calls from many threads are each served; calls go ahead of background work; shutdown drains, including behind a burst; a call queued beside a shutdown is answered | `persistence/owner.rs` |
| concurrent mints are unique and ordered | `persistence/identity/minter.rs` |
| every permutation of a record set yields the same substrate | `substrate::order_independence_tests` |
| one fork per key; a failed fork is retried; the sweep releases only what nothing holds; no request keeps a released value; a lock taken in the fork is held before anyone sees the value; a released value is dropped with its last handle | `zend/src/resident.rs` |
| chunk locations: migrates compose, a compaction excludes them, a refused pass is waited for | `migrate_flight.rs` |
| a queued reclaim payload and a detached substrate still alive are holders the sweep rewrites | `substrate/residence.rs`, `scheduler/detached.rs` |

## 12. What remains a lock, and why

| Lock | Why it is not a critical section |
|---|---|
| `ConvLock` | a conversation's turn order (§8.4) |
| Scheduler-owned buffers — the gallery arena's residency, index cache and zero page, the dedup codebook and masks, the sampler's penalty tables | taken by the scheduler thread alone; nothing else contends for them |
| Telemetry rings (`phase_ring`, `relief_trace`, `memory_report`) | the scheduler appends; a status read copies the ring out |
| Join handles and one-shot setup cells | taken once, at start or shutdown |

New dependencies: `arc-swap`, `imbl` (persistent collections), `papaya`
(lock-free maps and sets), `boxcar` (append-only slab).

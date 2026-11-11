# KV drains under pressure: a quant reserve, and compaction that never blocks migration

> **Superseded in part by `docs/kv_quantize_in_the_gap.md`.** That design
> quantizes K/V on the scheduler thread in every gap and reduces persistence to
> a copy. The hot→warm quantize, the backlog-sized reserve (§2a), the group cap
> (§2c) and the quantize side of §3 are replaced by it. Until it is built, this
> document describes the code.

**What broke** (BlackWorld, 2026-11-08 02:11–02:16; every count is from its
log). All 3,720 KV regions were live and the hot→warm migration barely ran: 93
passes, 51 doing nothing and one moving data. Three things held it, each
waiting on the one thing that would have freed it:

| Log line | Count |
|---|---|
| "hot→warm quantize deferred — a wave holds the partition and the destinations need a new arena" | 106 |
| "hot→warm cross-layer quantize failed … every one of the KV reservation's 3720 regions is occupied" | 28 |
| "kv compaction declined: MigrateInFlight" | 586 |
| "hot→warm groups deferred: a KV compaction is relocating chunks" | 30 |
| "section persist deferred: KV compaction relocating chunks" | 258 |

1. **The quantize needed a new arena it could not get.** Each layer's Q-format
   destination is allocated from its size class's arenas, and those were full.
   A new arena cannot be created while a wave is open
   (`enter_arena_window` → `KV_ARENA_MID_WAVE`). When a gap came, the KV side
   had no free region. The migrate that frees regions needed a free region
   first.
2. **An arena made for it could be swept away before it was used.**
   `create_deferred_arenas` stamps an owed arena and finishes its creation at
   once (`alloc.rs:473`). An empty, finished, unprotected arena is exactly
   what `release_empty_arenas` reclaims on the next sweep. The gap runs that
   sweep through compaction (`run.rs:1380`), relief, `claim_region` and
   `settle_device`.
3. **Compaction and migration exclude each other through one process-global
   word** (`migrate_flight.rs`: `CHUNK_LOCATIONS` with a `FROZEN` bit, and
   `COMPACTION_WAITING`).
   - Compaction freezes every chunk location in the process for its whole
     pass, or is refused while any migrate holds them.
   - A refused compaction raises `COMPACTION_WAITING`, and from then on every
     migrate steps aside: `continue 'policies` skips a whole compression class
     for the pass, not one group.
   - Each defers on the other's mere existence, across every pool, class and
     tenant, though any single pass touches only a few pools.

Producer throttling only kept this cycle from closing. This design breaks the
cycle in the engine, and every throttle goes (§1).

## 1. Throttles removed

- **`code_read::wait_evicted`** (17a446a83) and `EVICT_WAIT` are deleted. The
  ingest pool reads at its full width, `CODE_READ_PARALLELISM`.
- **The back-pressure design (`docs/kv_back_pressure.md`) is withdrawn.** Its
  producer "room" is a throttle under another name, so the file is deleted.

## 2. The quant reserve: one arena's worth free per reachable class, always there

**Built** as a headroom rule, not a marked arena:
- Each reserve class (`GidPool::keep_reserve`, registered by
  `ChunkedKvBacking::keep_reserves` in place of `warm_protected_arenas`) keeps
  at least one arena's worth of **free slots**.
- The empty sweep (`ArenaPool::try_tombstone`) releases an empty arena of a
  reserve class only while the class keeps a whole arena free without it, and
  takes the highest-index empty first, so the lowest stands.

Because the reserve is free capacity rather than a particular arena,
allocation and compaction need no special case. Allocation's lock-free path
is unchanged. Compaction packs within a class without changing its free total,
so it cannot consume a reserve. The detail below is the original design; where
it names a marked arena, the headroom rule is what was built.

**What it holds.** For every size class the active `CompressionPolicy` can
quantize into, the KV side keeps exactly **one empty reserve arena**. It also
keeps one for each native hot class (F16/R16) that sealing writes into.

- A class is reachable when some candidate of some level maps to it
  (`size_class` ladder, `arena_key_for`). Those are a subset of the 15 classes
  (320 … 16,384 B), not all 15.
- It replaces `warm_protected_arenas` (`alloc.rs:149-192`), which protects one
  arena per key for a fixed list. There is one mechanism, not two.

**How it is placed.**
- **Left-most.** A reserve arena takes the lowest free region (`try_claim`
  already pops the lowest free index). It never takes a frontier advance while
  a lower hole exists, so the reserve sits as low in the span as the KV side
  allows.
- **Fixed.** Compaction packs other arenas around a reserve's region and never
  relocates it upward.
- **Moved down, never kept above a hole.** The reserve an empty sweep keeps is
  whatever empty arenas its class has, and compaction can leave one of those on
  the top region. On WhiteWorld (2026-11-10) a 1,024 B class's reserve stood on
  region 487 with ~430 free regions (6.7 GB) below it. The frontier could not
  come down, and the expert zone sat 1 GB under its floor. So the sweep releases
  an empty arena the reserve would keep when the lowest free region lies below
  it (`try_tombstone`'s `lowest_hole`, read from `region_pool::lowest_free_region`).
  The class then falls short, the next gap owes it an arena, and the claim takes
  the lowest free region. The re-made arenas fill the lowest holes in order, so
  one round settles it.

**When it is used.** A reserve arena is allocated from only when its class has
no free slot. Its first allocation makes it an ordinary arena, and its class is
then **owed a new reserve**. A migrate mid-wave therefore always has a
destination. It never needs a new arena inside a wave, so "deferred: needs a
new arena" cannot recur while the reserve stands.

**When it is replenished** (`BackingInner::owe_reserves`, at the top of every
`create_deferred_arenas`), at the moments arena creation is legal:
- the wave-loop gap, which now creates arenas **ahead of** compaction, so a
  sweep cannot see the new reserve before it is protected;
- the persistence pass top.

The gap runs on every loop iteration, so admission needs no third site.

Replenishing one reserve costs one region claim per class consumed.

**Never thrashed:**
1. A reserve is in `protected_arenas`, which every empty sweep honours:
   - `release_empty_arenas`;
   - `global_release_empty_arenas` (including from `claim_region`);
   - relief rung 1;
   - `settle_device`.
2. The weight side cannot buy a reserve region. `reclaim_spare_ground` and
   `buy_ground` skip it.
3. An owed arena created for a pass's retry stays protected until the demand
   it was owed for is consumed or abandoned. `finish_creation` moves from
   creation time to that point, which fixes cause 2.
4. Hysteresis: a class that is no longer reachable (the policy changed) gives
   its reserve back only after one full persistence pass with no demand for it.

**Accounting.** Reserve arenas are KV ground and are never free regions:
- the weight side buys only free regions, so it cannot take one;
- they count against the KV side of `weight_floor`, adding at most one region
  per reachable class.

**Arenas made ahead of demand** are protected until the demand fills them or
[`FRESH_GAPS`] gaps pass (`BackingInner::fresh_arenas`, `settle_fresh`).

## 2a. The reserve grows with the R16 backlog

A fixed reserve of one arena per class can be smaller than one migrate group
needs: a group is up to `MAX_GROUP_RESIDENCES` finished readings across every
layer. So the reserve is sized from the backlog it serves.

**R16 is the backlog.** Hot KV is written native (R16/F16), and the hot→warm
migrate turns each native chunk slot into **at least one** destination slot,
by a factor that depends on the class:
- a mixed-format chunk takes a slot **per band**, in its band's class;
- every quantized chunk adds metadata **record** slots, in the record arenas
  (`kind = Record`).

So the live native slots are the demand's measure, and the factor per class is
learned from what migrates actually allocate rather than assumed.

**Target per class, read from what VRAM holds now.** The quant slots live on
the card are mostly what migrates produced, so their mix is the factor. Band
and Record kinds are kept apart:

```
pending     = live native (R16 + F16) slots                     (per backing)
band(c)     = live slots of quant band class c
record(c)   = live slots of record class c
factor(c)   = band(c)   / Σ band               for a quant band class
            = record(c) / Σ band               for a record class
target(c)   = clamp( ceil(pending × factor(c)),
                     floor = one arena of c,
                     cap   = RESERVE_CAP × KV regions, split by factor )
```

- **Why that is the factor.** A native slot becomes one band slot, in the
  class its block was chosen for. So each band class's share of the live quant
  band slots is its per-native-slot factor, and those shares sum to 1. Records
  ride in proportion to bands, so a record class's live slots over all quant
  band slots is its factor.
- **Every term is a counter the pools keep** (`kv_gpu_class_stats`). No scan
  and no learning state.
- **Where it errs.** Quant KV that stays hot is weighted toward active
  conversations and galleries, so when chat rather than ingest dominates, the
  split can differ from a burst of file readings. The total is unaffected,
  and the one-arena floor per class absorbs a small split error.
- **Cold start.** With no quant slots live, every class sits at its floor,
  which is today's reserve.
- **`pending`** is the native classes' live slot counts, which the pools
  already hold. One counter read, no scan. A hot copy kept beside its warm
  copy (a conversation still active) counts too, so `pending` can overstate
  the backlog. `RESERVE_CAP` bounds what that costs.
- **`RESERVE_CAP_PERCENT`** = 25% of the KV regions (`region_stats().total`,
  the ground below `weight_floor`), so a burst never hands the whole card to
  free slots.
- **The record class** is the backing's one record key
  (`record_layout().key`). Every quantized chunk takes one record, so the
  records' share is their live slots over the live band slots, as above.
- **The destinations are transient.** The migrate quantizes into GPU Q arenas,
  copies them to warm, and the install drops the hot copy. So one pass needs
  about one group's worth of destination slots, while `pending × factor`
  sizes for the whole backlog. The cap bounds the difference, and what is
  over-reserved is returned by the sweep as the backlog drains.

**Where it lives.** The arithmetic is `chunked/reserve_target.rs`
(`reserve_targets`), and it is pure and tested on its own. The pool holds each
class's target (`ChunkGidPool::set_reserve_target`), and
`BackingInner::owe_reserves` sets the targets and owes the shortfall at the
top of every `create_deferred_arenas`. The native classes keep their one-arena
floor. The migrate does write one of them: R16 is a K candidate, the fallback
for a block no smaller format holds, so a group's quantize can need R16 slots
too. But the native slots live in that class outnumber the migrate's share of
it, so no factor read from it would size the migrate's demand, and the ingest's
own R16 writes take the class's free slots as fast as the gap makes them. §2c
is what keeps the migrate moving there.

**Grows at the gap, shrinks under the sweep, never thrashes.**
- **Grow.** `owe_reserves` owes class c an arena whenever
  `free(c) < target(c)`, up to the shortfall in whole arenas (as
  `owe_capacity` does), and the gap creates them. Every arena still starts
  lowest-free.
- **Shrink.** The empty sweep releases an arena of class c only while
  `free(c) − arena(c) ≥ target(c) + arena(c)`. That is a one-arena hysteresis
  band, so a target oscillating by a few slots neither creates nor releases.
- **Knocked off by persistence.** As the migrate drains, `pending` falls, then
  each `target` falls, and the sweep then releases what the band allows.

**What it replaces.** The fixed one-arena-per-class headroom of §2 becomes this
target's floor. It is one rule, not two.

## 2c. A refused group shrinks before it waits

A pass snapshots the oldest pending residences first, up to
`MAX_GROUP_RESIDENCES`, and a quantize short of destination slots refuses the
whole group. Refused, the group used to wait for a gap and then end the pass.
The next pass took the same residences, so a group too big for what its
classes keep free stood at the head of the queue for good. Measured on
BlackWorld on 4801 (2026-11-10, a full re-read of `code_reading`):
- the same three groups of eight were refused short of 8,349, 8,857 and 12,685
  R16 slots for over forty minutes;
- every pass line read `moved_residences=0`;
- 45 GB of native KV stood on the card, and the expert zone sat on its floor
  at a 6–17% hit rate.

Two things kept it there.

**A refusal owed one layer, and a group is every layer.** The quantize
allocates layer by layer, and the layers share one pool per class. A refusal
owed what the refused layer needed. The gap made that, and the retry got past
that layer to be refused at the next one, so a group of one residence needed
thirteen gaps for thirteen layers of ~4,200 R16 slots each. Meanwhile the
ingest's own R16 writes filled each gap's arenas before the retry. So a
refused layer owes its own demand together with every layer after it
(`quantize_layers_deferred_within`, `add_layer_demand`), and one gap makes
everything the group still needs. The layers before it give back what they
claimed when the group drops, and the retry claims it again.

**The group was as large as the backlog allowed.** So the group size is a cap
the pass adjusts (`persistence/group_cap.rs`, `GroupCap`):
- **Halved** by every refusal, and the smaller group is tried at once, before
  any gap wait. One residence needs a few hundred to a few thousand slots of a
  class, which the class almost always has free.
- **Doubled back** towards `MAX_GROUP_RESIDENCES` by every group that moves.
- **One residence refused** waits for the gap as before (`await_owed_arenas`,
  `MAX_GAP_RETRIES`).

## 2b. Logging for the next attempt

Everything below goes to `candle_conversation::kv_reserve` at INFO, and at
WARN on the event marked:

1. **`kv_reserve: pass`**, once per persistence pass that moved something or
   left a class owed (`persistence/reserve_log.rs`). It carries:
   - `pending_slots` and `pending_mib`;
   - `classes`, one field per class: `label:target/free+owed@factor`, with a
     `!` when the class is held at its cap.

   With that line the factor is a number in the log, not an assumption, and
   the design can be checked against it.
2. **Grow events** (`kv_reserve: grow <class> by N arena(s)`, from
   `owe_reserves`) fire at INFO when a class is owed arenas. Each names the
   class, the arenas owed, the target, the free slots, the factor, and the
   reason: `backlog` or `capped`. A target crossing an arena boundary with no
   arena owed (`kv_reserve: target`, reason `drained` on the way down) is
   DEBUG. It moves with every seal and drain, many times a second (measured
   on BW on 4662: 15 lines in 3 s at INFO), and the pass line carries every
   target.
3. **The quantize deferral** (`kv_reserve: quantize deferred short of …`, from
   `owe_capacity`). It names the class, the slots the layer needed, the slots
   free, the class's target and the arenas owed. It is a WARN when the need
   was inside the class's reserve target: the reserve was sized to cover it
   and did not, so that is a sizing or replenishing bug. A need past the target
   is demand beyond the cap, and is logged at INFO.
4. **Migrate progress**, on the pass line: `moved_residences`, `moved_mib` and
   `pass_ms`. The drain per class is the change in `free` between two pass
   lines.
5. **Region accounting**, on the pass line: `reserve_free_mib` against
   `cap_mib` and `kv_mib`, so a reserve crowding the card is visible.
6. **Ingest rate:** `code_read: N file(s) read in S s`
   (`zend::code_read::ingest`, `code_read/ingest_rate.rs`). It is written once
   per 5-minute window and carries `files_per_min`. A window closes at the
   first commit past its length, so a quiet spell is one long window.

## 3. Compaction and migration: held off only while the migrate writes

**The contention.** A migrate group took `CHUNK_LOCATIONS` shared for its whole
length, and a compaction pass needs it exclusive. On BW (2026-11-08, 02:11–02:16)
that refused 586 passes. The group's time is almost all DtoH: the quantize and
convert were 0–42 ms of groups lasting 20–922 ms.

**What the exclusion actually protects.** A pass copies each source slot as it
stands at the copy, and correctness does not depend on the pass reaching every
holder: nothing frees a slot anyone names, so a holder it misses keeps reading
the source, unchanged. What is unsafe is copying a slot **while it is being
written** — a migrate's quantize destination before its convert lands. The
holder the sweep rewrites to that copy reads unfinished bytes as its KV.

**What was rejected.**
- *Per-pool holds.* A migrate learns its destination classes only inside the
  quantize, and it reads the native pools, which are where compaction is most
  needed.
- *Skipping slots whose refcount shows an extra holder.* `HeadGids` is a shared
  `Arc<Vec<ChunkGid>>`, so six holders can read `strong_count() == 1`
  (`compact.rs` header). The refcount is not a holder count.
- *Pinning planned sources.* A free slot's refcount word holds its recycle-list
  link, so a "retain if live" from a raw id can succeed on a freed slot.

**What was built.**
1. **The hold spans only the writes.** `try_migrate_flight` is taken before the
   group's snapshot, as before. After the quantize (claims, records) and the
   convert, `migrate_group_hot_to_warm` synchronises `copy_stream` and calls
   `MigrateFlight::settle`, which drops the location hold. The in-flight count
   (`migrate_in_flight`) stays until the marker drops.
2. **The pass steps around what the group holds** (`HeldChunks`,
   `migrate_flight.rs`). The group's native chunks are registered before the
   migrate starts, and its quantized copies after the fence, both before
   `settle`. So any pass overlapping the DtoH and install sees all of them.
   `compact_with` drops every planned move whose source is registered
   (`retain_unheld`) and reports the count as `held`. A plan that is all held
   refuses `MigrateInFlight`. Without this the pass would spend claims on chunks
   the group's handles keep alive, which the sweep reports as `unwitnessed`.
3. **The install lands over a relocated copy.** A pass that relocates a
   residence's hot copy replaces its `Arc<HotKv>`, so the install's `Arc::ptr_eq`
   would refuse the group. `HotKv` now carries a lineage: a compaction rewrite
   keeps it (`relocate_hot`), and every other copy starts its own. The install
   and the section quantize compare lineages (`replace_hot`).
4. **Every device handle the group drops goes through the reclaim queue.** The
   group's clones are dropped before the `HotKv`s they share chunks with. Refused
   installs, the evict-when-cold path and a failed group's quantized copies are
   dropped as a `HotKv`. So no slot a pass planned is freed mid-pass off the
   compacting thread.

The fairness flag (`COMPACTION_WAITING`) and section persist's
`try_hold_chunk_locations` are unchanged.

## 4. A refused wave relieves before it retries

The tier-refusal retry (`WaveRetry` and `GIVE_UP_AFTER = 1024` in
`wave_retry.rs`; `decode::retry_step`) today only lowers the frontier and backs
off. It is changed to run the relief ladder before retrying, with eviction of
what has a warm copy, and to fail a wave only when a full relief cycle frees
nothing. With §2 and §3 in place, warm copies exist under pressure, so the
ladder's evict rung has something to evict.

## 5. Tests

Assertions are on exact counts and indices:

1. **The reserve stands.** After each sweep kind, every reachable class has
   exactly one empty, protected arena, at the lowest index its class can take.
2. **No thrash.** Across 1,000 alternating compaction and empty sweeps, a
   reserve's region index does not change, and no reserve is released and
   recreated.
3. **Consumed, then replenished at the gap.** A class whose reserve takes its
   first slot mid-wave owes one; the next gap creates it, protected, before
   compaction runs.
4. **A migrate under a full reservation.** With every non-reserve region live
   and a wave open, a hot→warm pass quantizes into the reserve and lands in
   warm, with no arena created mid-wave and no deferral.
5. **The hold spans only the writes** (`migrate_flight` tests). A pass is
   refused while a migrate is unsettled and admitted once it settles; the
   migrate is still in flight; a new migrate waits for the pass.
6. **A pass steps around held chunks**, counting a slot two groups share until
   both let go (`held_chunks_are_skipped_until_every_holder_lets_go`). An install
   lands over a relocation of the copy it read and over nothing newer
   (`a_replace_lands_over_a_relocation_of_the_copy_it_read`).
7. **A refused wave relieves.** A decode wave refused for its tier places
   after one relief cycle that evicts warm-backed KV, and is never failed.

## 6. Order of work, all in one change (no dual path)

1. Delete `wait_evicted` and `docs/kv_back_pressure.md`.
2. The reserve, replacing `warm_protected_arenas`:
   - the protection set;
   - left-most claim;
   - weight-side exclusion;
   - `finish_creation` moved to consumption;
   - replenishment at the gap (before compaction), at admission relief and at
     the pass top.
3. The migrate's hold narrowed to its writes, held chunks stepped around, and
   installs by lineage (§3).
4. The relieving retry.
5. Logs, once per pass: reserve per class (standing, owed), pools compaction
   skipped as held, groups a migrate skipped.

**Measured against the night it broke**, on BlackWorld at 16 workers:
- zero "needs a new arena" deferrals while the reserve stands;
- "compaction declined: MigrateInFlight" only during a group's quantize and
  convert, and `unwitnessed` no higher than before;
- zero "wave too wide" failures;
- files per hour at least what the throttle gave (about 270/h).

# KV drains under pressure: a quant reserve, and compaction that never blocks migration

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
- **`RESERVE_CAP`** = 25% of the KV reservation, so a burst never hands the
  whole card to free slots.

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

## 2b. Logging for the next attempt

Everything below goes to `candle_conversation::kv_reserve` at INFO, once per
persistence pass, and at WARN on the events marked:

1. **Reserve line, per backing:** `pending` native slots and MiB, and per class
   `target`, `free`, `owed` and the measured `factor`. With that line the
   factor is a number in the log, not an assumption, and the design can be
   checked against it.
2. **Grow and shrink events:** the class, how many arenas, the target before
   and after, and the reason (`backlog`, `drained`, `capped`).
3. **The quantize deferral** (`quantize deferred`), now with the class, the
   slots the group needed, the slots free, and whether the reserve was at its
   cap. WARN if the reserve was not at its cap: the reserve should have
   covered it, so that is a sizing bug.
4. **Migrate progress:** slots moved hot→warm this pass, per class, plus pass
   time, so the drain rate is a number rather than an inference.
5. **Region accounting:** reserve arenas and slots against the KV reservation,
   with the cap, so a reserve crowding the card is visible.
6. **Ingest rate:** `code_read` logs files completed per 5-minute window,
   once per window. A rate then needs no grep over per-file DEBUG lines.

## 3. Compaction and migration: pool-grained, and neither waits

> **Revised on reading the code: per-pool holds do not reach the contention.**
>
> - A migrate learns its destination classes only inside the quantize, block
>   by block, so it would have to hold every reachable quant pool shared.
> - It reads native hot pools, which are exactly where finished readings
>   fragment and compaction is most needed.
>
> Per-pool holds would starve compaction on the pools that matter, as the
> global word does now.
>
> The design that reaches it is per **chunk**: compaction never relocates a
> slot that a holder it cannot sweep has a reference to. A migrate's snapshot
> holds one, so its refcount says so. The planner skips such slots instead of
> refusing the whole pass. That is a change to compaction's planner (census and
> sweep), and is proposed for a decision before it is built.
>
> The "skip a whole class" refusal in `run_pass` is not a defect. It is taken
> before the group is snapshotted, when every group of that class would be
> refused at the same instant.
>
> The text below is the original per-pool design, kept for the record.

**Replacing the global word.** `CHUNK_LOCATIONS` / `COMPACTION_WAITING` give
way to a **per-pool gate**, one atomic word per `ArenaKey`
`(class, location, kind)`. Compaction already plans per pool (`plan_pool`).

- **Migrate.** A group takes a shared hold on exactly the pools it reads (its
  source chunks' pools) and writes (its Q destination pools). The group
  records them as it builds its copy table, and releases them when the group
  installs, as today.
- **Compaction.** Each pass freezes only the pools it is about to move, one at
  a time, with `compare_exchange(0, FROZEN)` per pool. A pool held shared is
  **skipped this pass, not waited on**, and compaction moves on to the next
  pool. There is no `COMPACTION_WAITING`: compaction never asks a migrate to
  step aside.
- **A migrate meeting a frozen pool** waits at most for that one pool's move,
  which is bounded by the per-pool share of the 80 ms budget. A refusal skips
  **that group** only, never `continue 'policies`.
- **Section persist** (`try_hold_chunk_locations`, `drain_in_held_groups`)
  takes the same per-pool shared holds.

Correctness is unchanged. Today's word protects a migrate's copy table from a
chunk being relocated under it, and protects compaction from installs carrying
pre-compaction gids. Both races are within a pool: a relocation moves a chunk
between slots of the same pool. Holding exactly the pools a migrate touches
gives the same guarantee where the work actually is. Pools nobody touches
compact freely, and migrations into untouched pools never wait.

`compaction_epoch()` exists (`compact.rs:312`), but nothing in the migrate path
consults it. It becomes a debug assertion at install: a group's pools' epochs
are unchanged since its holds were taken.

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
5. **Pool grain.**
   - A migrate holding pool A does not stop compaction packing pool B.
   - Compaction holding pool B does not stop a migrate into pool A.
   - A migrate whose pool is frozen skips only its group.
6. **The relocation race is still closed.** A compaction cannot relocate a
   chunk in a pool a migrate holds. A migrate's install finds its pools'
   epochs unchanged.
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
3. The per-pool gate, replacing `migrate_flight.rs`'s global word; the group
   skip; section persist on the same holds.
4. The relieving retry.
5. Logs, once per pass: reserve per class (standing, owed), pools compaction
   skipped as held, groups a migrate skipped.

**Measured against the night it broke**, on BlackWorld at 16 workers:
- zero "needs a new arena" deferrals while the reserve stands;
- zero "compaction declined: MigrateInFlight";
- zero "wave too wide" failures;
- files per hour at least what the throttle gave (about 270/h).

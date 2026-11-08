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

## 2. The quant reserve: one empty arena per reachable class, always there

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

**When it is replenished**, at the moments arena creation is legal:
- the wave-loop gap (`run.rs:1383`, `create_deferred_arenas`), moved **ahead
  of** compaction so a sweep cannot see the new reserve before it is
  protected;
- admission relief (`promote_new_prefills` → `relieve_vram_pressure`, after
  `end_wave_transient`);
- the persistence pass top (`thread.rs:1100`).

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

**Accounting.** Reserve regions are KV ground.
- They count against the KV side of `weight_floor`, which grows by the reserve
  set (one region per reachable class) at most.
- `region_stats` reports them separately (`reserve`), so a check that reads
  free regions does not count a reserve as headroom (invariant 7).

## 3. Compaction and migration: pool-grained, and neither waits

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

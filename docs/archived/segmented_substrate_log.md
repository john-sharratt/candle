# Segmented Substrate Log — Design

Status: **implemented (rev 5). All phases landed on branch `unified-wave-engine`,
CPU-tested (162 persistence + 44 substrate tests green); GPU cold-load segment
routing is daemon-validated.**

Implementation notes vs. this design:
- **Incremental drop/compact/combine (§6) is implemented** in
  `persistence/maintenance.rs` (`run_maintenance` → `pick_maintenance_op` →
  `apply_maintenance_op`). Safety rests on re-emitting the full **resident**
  record set (incl. `Tombstone` markers) before any drop, so a dropped segment
  never takes the only copy of live metadata and a tombstoned timeline can't
  resurrect (see the module doc). Per-segment dead accounting is derived from the
  substrate index (`segment_liveness`), not a sharded `RecordAccounting` — the
  resident set is re-emitted wholesale each op, so a segment's `dead = total −
  live_readback` slightly over-counts, which only over-eagerly compacts (safe).
- The **whole-store** compactor (`compact`) is **kept** (segment-aware:
  `collect_live_records` routes reads per segment, `SegmentedLog::adopt_compacted`
  swaps in a single fresh segment) as a manual/emergency op; `write_compacted_log`
  is **not** deleted (repurposed). It is no longer on any automatic path.
- Load-path compaction is **removed** (§9): the `--no-compact-substrate` flag,
  the `substrate_has_reclaimable` gate, and the `Compacting` load step are gone;
  reclaim is fully background.
Scope: the on-disk redo log under `candle-conversation/src/persistence/` (`.substrate/`)

---

## 1. Problem

The substrate is one **monolithic append-only file** (`.substrate/substrate.log`)
compacted by a **whole-file rewrite**: `compaction::write_compacted_log` walks
*every* live record, buffers them all into a `Vec<(header, payload)>` in RAM,
writes a brand-new file, and swaps it in. No segment/rotation concept exists.

The functional reclaim is correct (distilled timelines' chunks are dropped —
`keep_chunks = distill.is_none()`). The problems are non-functional:

| Symptom | Cause |
|---|---|
| **Too slow** | Reclaiming any dead byte rewrites the entire surviving set — O(total live), spikes RAM (whole live set buffered). |
| **Runs almost every time** | Startup gate `has_reclaimable_records()` = "any tombstoned *or* distilled timeline." Calibration distills ~744 timelines/run → ~always true. |
| **Very large file** | Append-only, no rotation. |
| **Can't be split** | No segment boundary → no unit smaller than the whole log. |

## 2. Goals / non-goals

**Goals**
- Split the log into **~4 GB segments**, all living directly in `.substrate/`.
- **Whole-segment drop** (`unlink`) when a sealed segment is fully dead — the
  calibration/29 GB case becomes deletes, not a rewrite.
- **Incremental compact** and **combine** (defrag small adjacent segments), both
  O(one/two segments), bounded RAM.
- **Fully background** — no compaction in the load/startup path at all.
- **Trigger on real savings + settle time**, not "a marker exists."
- **Preserve the exact block-read I/O pattern** (`O_DIRECT` concurrent stripes).
- **GUI visibility** — surface an in-progress compaction with clickable progress.
- **Derive segment state from the directory** — no manifest file.

**Non-goals**
- No change to record-format semantics, the tier model, or distill/tombstone
  reclaim rules.
- Single writer only (the persistence thread). No concurrent writers.
- Not a leveled LSM — a flat, id-ordered segment set.

## 3. I/O model — what must not regress

The redo log is **not** memory-mapped. It uses:
- `LogFile` — a buffered `File` handle for **writes** (group-commit staging +
  fsync) and small reads (superblock, single-record lookups).
- `DirectFile` — **`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`**, opening
  `MAX_CONCURRENT_READS = 16` independent handles on the same file.
  `read_stripes_concurrent` coalesces adjacent chunk records into stripes and
  issues **one positioned `pread64`/`ReadFile+OVERLAPPED` per stripe across the
  16 handles**. This bypasses the page cache and lets NVMe DMA into
  sector-aligned scratch — the "really good" cold-load read path.

**Constraint:** segmentation must keep this pattern *per segment*, byte-for-byte.
A segment reuses `LogFile` + `DirectFile` unchanged; the only difference is which
file a `(segment, offset)` read routes to. Stripe coalescing runs **within a
segment** (a turn spanning a seal splits into one stripe batch per segment, each
still 16-way concurrent). No read path changes shape.

### 3.1 The real "too many files" problem: handles, not mmaps

Each open `DirectFile` = 16 handles. With N segments, eagerly opening all of them
is 16·N handles — at ~60 segments that's ~960, brushing Linux's default
`RLIMIT_NOFILE` soft cap (1024). (Windows tolerates far more but there's no
reason to.) Memory-map limits — Linux `vm.max_map_count` (default 65530), Windows
address space — **do not apply** to the redo log because it isn't mapped.

**Solution — a bounded LRU of open sealed segments.** Keep the active segment's
`LogFile`/`DirectFile` always open, plus an LRU of `OPEN_SEALED_SEGMENTS` (e.g.
**8**) sealed `DirectFile`s. A read to a not-open sealed segment opens it
(evicting the LRU tail). Steady state: (1 + 8)·16 = 144 handles regardless of
segment count. Sealed segments are immutable, so reopening is a plain `open` with
no recovery. Cold-load of a working set touches few segments at once, so the LRU
hit rate is high and the stripe-concurrency per segment is unchanged.

> Note: `provenance/raw_store.rs` *is* mmap-backed and separate from the redo
> log. It is out of scope here; if it later needs segmenting, `vm.max_map_count`
> and per-mapping address space (segment_size · N) become the relevant limits —
> flagged for a future pass, not solved now.

## 4. Segment model + directory (no manifest)

Segments live in `.substrate/` with an id encoded in the name:

```
.substrate/
  seg-0000000001.log      ← sealed (immutable, complete)
  seg-0000000002.log      ← sealed
  seg-0000000003.log      ← active: the highest-id file, the ONE append target
```

- **Single namespace.** Every segment is a `seg-<id>.log` file. There is **no
  `.active` extension**: the active is simply the **highest-id** file, and a
  segment is sealed by a higher-id one existing. (Sealing needs no rename —
  which removes the two-active crash window; a legacy `seg-<id>.active` from
  before this simplification is adopted → renamed to `.log` → on open.)
- **Active**: the highest-id segment — the current append target. All writes
  (fresh appends *and* relocated records from compact/combine — §6) go here.
- **Sealed**: every lower-id segment, immutable.
- **Next id**: `max(existing ids) + 1`, derived from the directory.
- Each segment file is exactly today's `LogFile` layout (superblock + 4 KB
  records + `HeaderIndex` chain). No new on-disk record types.

### 4.1 Why no manifest — deriving everything

The segment set, order, and active/sealed split are all derivable:
- **Set + order** = the `seg-<id>.log` directory listing, sorted by id.
- **Active** = the highest-id file.
- **Recency** = **append order = id order**, *because relocation always appends
  to the active (highest id)* (§6). So a key's live version is always its
  highest-id occurrence.

This is the invariant that lets us drop the manifest: **the active segment always
holds the highest id, so id-order is a valid total recency order.** Recovery
replays segments by ascending id (active last); later overwrites earlier in the
in-RAM index, landing on the highest-id (newest) record per key — exactly what
the single-file offset order does today.

A manifest would only add explicit set-membership + orphan flags; §7 shows those
are handled by id-order-wins + self-healing dead-drop instead.

## 5. Addressing change (the core plumbing)

Today the in-RAM stream index stores a bare offset (`ChunkLocation { offset,
record_size }`) and reads via `read_record_at(&log, offset, size)`. It becomes:

```rust
struct RecordLoc { segment: SegmentId, offset: u64, record_size: u64 }
```

`SegmentId` threads through: the stream index, chunk-plan stripe coalescing in
`mod.rs`, `cold_load`/`direct_io` stripe reads, `inherit.rs`, `header_index`, and
recovery. Reads route to the segment's `LogFile`/`DirectFile` via the LRU pool.
This is the widest-touching change; **§9 is the full audit checklist** to do it
without regressions.

### 5.1 How the segment id reaches the in-RAM index during recovery

The manifest holds only three singleton `RecordLoc`s; every per-stream `ChunkLoc`
lives on the `Substrate`, stamped by `Substrate::apply_walker_entry`. Both consume
a `WalkEntry`. So the single threading mechanism is a **`segment: SegmentId` field
on `WalkEntry`**: the walker/recovery functions take the segment they are walking
and stamp every entry they emit; `manifest.ingest` and `apply_walker_entry` read
`entry.segment` instead of the `FIRST_SEGMENT` placeholder. This keeps the sink
signature (`FnMut(&WalkEntry)`) unchanged — only the entry-construction sites in
`walker.rs` / `recovery.rs` / `header_index.rs` gain the field.

Multi-segment recovery (`SegmentedLog::open`) walks the segments in **ascending
id** (each sealed segment via its own `HeaderIndex` chain / full-walk fallback,
active last with torn-tail truncation), feeding all entries into **one** shared
`Substrate` + manifest through the same sink. Because a later (higher-id) segment's
`apply_walker_entry` overwrites an earlier one for the same key, the in-RAM index
lands on the highest-id (newest) record per key — the id-order-wins recency model
of §4.1. The three manifest singletons resolve the same way.

## 6. Background maintenance — drop / compact / combine

One shared mechanism — *relocate live records to the active, then unlink the
source* — with three triggers. All run on the **persistence thread's background
pass, at most one op per pass**; **none run at startup**.

1. **Drop (O(1))** — a sealed segment with `live_bytes == 0`
   (all records superseded/tombstoned/distilled-away): `unlink`. The
   calibration/29 GB case — reclaim by deleting whole files.
2. **Compact (O(one segment))** — a sealed segment past the savings + settle
   thresholds (§8): read only its **live** records, append them to the active,
   fsync the active, update their `RecordLoc`s to the active, then unlink the
   source.
3. **Combine (O(two segments))** — two **adjacent** sealed segments each below
   `COMBINE_SEGMENT_BYTES` (e.g. **2 GB** live): relocate both into the active,
   drop both. Reduces segment count (and open-handle pressure) after compaction
   has shrunk neighbours. Runs asynchronously like the others.

Because relocation always targets the active (highest id), a relocated copy wins
over its source by id-order, so a crash mid-relocation self-heals (§7).

> Combine consolidates *into the active* rather than minting a mid-id merged
> segment — that keeps "active = highest id" true, which is what makes the
> manifest-free recency model sound. Merged data packs into 4 GB segments
> naturally as the active seals.

## 7. Crash safety (single writer, no manifest)

Ordering rules and their crash windows:

- **Append/seal**: seal = fsync active's index chain → create
  `seg-(max+1).log` → close the old segment's write handle. **No rename.** A
  crash at any point leaves the old (durable) segment plus possibly an empty new
  one; the highest id is unambiguously the active on the next open, so there is
  **no two-active state to heal**. The active's tail is re-derived
  (`set_write_offset`) on recovery. Consistent.
- **Relocate (compact/combine)**: write live copies to active → **fsync active**
  → unlink source(s). Crash before fsync → copies not durable, source intact →
  op re-runs. Crash after fsync, before unlink → copies live in the active
  (win by id-order), source's relocated records now superseded → source
  `live_bytes` drops to 0 → dropped next pass. A partially-written copy in the
  active tail is discarded by tail recovery (only committed records survive).
- **Drop**: `unlink` (last step). Crash before → segment still present but fully
  dead → re-dropped. Crash after → gone.

**Invariant:** every `seg-<id>.log` file on disk is authoritative; recency is
id-order; a fully-dead segment is reclaimable; a crashed relocation leaves the
winning copy in the higher-id active. No manifest, no orphan list needed —
orphans are simply dead segments that the next background pass drops. (A stray
`*.tmp` from an interrupted relocation write is ignored by recovery and deleted
on open.)

## 8. Trigger policy

A sealed segment becomes a **compact** candidate only when **both** hold
(interpretation of "> 1 minute and > 10% saving per segment" — confirm):

- **Savings**: `dead_bytes(segment) / segment_bytes ≥ SEGMENT_COMPACT_MIN_DEAD`
  (**10%**), i.e. relocating is worth the I/O.
- **Settle/rate-limit**: at least `SEGMENT_COMPACT_MIN_AGE` (**60 s**) since the
  segment was sealed or last compacted — so a hot segment isn't churned and a
  just-sealed one has time to accumulate deaths.

**Drop** is unconditional at `live_bytes == 0` (no thresholds — a fully-dead
segment is always worth deleting). **Combine** triggers on adjacency + the 2 GB
size threshold (independent of the dead ratio). Per-segment dead/live bytes come
from a **sharded `RecordAccounting`** (§5): the same O(1)/append last-writer-wins
keying, but the dead debit lands on the segment that physically held the
superseded record (cross-segment supersession debits the *older* segment;
distill/tombstone debits every segment holding the timeline's records).

## 9. Regression audit checklist (rotation, pointers, handles)

Every site that today assumes a single log or a bare offset must be found and
converted. Concrete list to audit (from this pass's tracing):

- `persistence/mod.rs` — `read_record_at` call sites (≈lines 231/235/240/549/559,
  785); the chunk-plan stripe builder (`build_stripes`, `file_offset` fields
  ≈610–635, 696–708) — coalesce **per segment**; `should_compact`/`dead_ratio`
  (925–941) — per-segment; `commit`/`stage`/`write_offset` — route to active.
- `persistence/log_file.rs` — one `LogFile` per segment; `set_write_offset` only
  on the active; superblock per segment.
- `persistence/direct_io.rs` — the 16-handle `DirectFile` becomes a per-segment
  resource behind the LRU pool; `read_stripes_concurrent` unchanged per segment.
- `persistence/cold_load.rs` — stripe reads keyed by `(segment, offset)`.
- `persistence/chunk_plan.rs` — `SourceLog` gains a **`Sealed(SegmentId)`** variant
  (alongside `Active` / `Inherited(i)`); `plan_chunked_read` partitions a stream's
  chunks **by segment** (grouping on `ChunkLoc.segment`) before stripe coalescing,
  because a live timeline's chunks now span the sealed segments **and** the active.
  Each source's stripes stay within one file — the coalescing invariant is
  preserved per segment.
- `persistence/pipeline.rs` — the reader pool's `match work.source` resolves
  `SourceLog::Sealed(id)` to that segment's pooled `DirectFile` (`n_handles()` must
  match the segment actually being read). This is the **GPU cold-load** path
  (`kv_migrate` scatter); validated on the live daemon, not CPU tests.
- `persistence/recovery.rs` / `header_index.rs` — per-segment `HeaderIndex` chain;
  multi-segment replay ordering (ascending id, active last); `WalkEntry.segment`
  stamped per walk (§5.1).
- `persistence/inherit.rs` — inherited-log reads (fork parents) become
  segment-addressed.
- `persistence/compaction.rs` — delete `write_compacted_log`/whole-file rewrite;
  replace with drop/compact/combine.
- `persistence/accounting.rs` — shard per segment.
- Loader (`zend/src/session.rs`) — **remove** the `substrate_has_reclaimable`
  compaction+reload step from the load path.
- `substrate_inspect.rs` — take the `.substrate` **directory**, enumerate
  segments, and aggregate (§10).

**Rotation invariant to verify:** sealing never moves an existing record, so all
outstanding `RecordLoc`s stay valid across a rotate; only the *append target*
changes. The audit must confirm nothing caches "the log file" by identity across
a seal, and that the handle pool re-resolves segments by id.

## 10. Tooling — `substrate_inspect`

Change `--log <file>` to a `--dir <path>` defaulting to `.substrate/`. The tool
enumerates `seg-*.{log,active}` in id order, opens each (read-only), and
aggregates the summary/dump across segments (per-segment live/dead + totals; a
`--segment <id>` filter for drilling in). Same record decoders, just iterated.

## 11. GUI — compaction visibility

The persistence thread publishes a small status the engine exposes over the API:

```rust
struct CompactionStatus {
    active: bool,
    op: Option<CompactionOp>,   // Drop | Compact | Combine
    segment: Option<SegmentId>,
    bytes_done: u64, bytes_total: u64,
    started_ms: u64,
}
```

Updated as each background op progresses (bytes relocated / to relocate). The GUI
shows an unobtrusive indicator when `active`, and clicking it opens a small panel
with the current op, segment, and a progress bar (`bytes_done/bytes_total`).
Mirrors how the load-progress overlay already streams from the daemon.

## 12. Config / constants

```rust
const SEGMENT_TARGET_BYTES: u64      = 4 * 1024 * 1024 * 1024; // ≈4 GB seal point
const COMBINE_SEGMENT_BYTES: u64     = 2 * 1024 * 1024 * 1024; // combine below this
const SEGMENT_COMPACT_MIN_DEAD: f32  = 0.10;                   // ≥10% savings
const SEGMENT_COMPACT_MIN_AGE_S: u64 = 60;                     // settle/rate-limit
const OPEN_SEALED_SEGMENTS: usize    = 8;                      // LRU of DirectFiles
```

## 13. Migration — auto-split on open

On open, the segment layer **auto-migrates** a legacy monolithic
`.substrate/substrate.log` into the segmented layout. This is O(1) (two renames /
one create — never a byte rewrite) and preserves the existing store, so a running
deployment upgrades in place with no rebuild.

**Algorithm** (`SegmentedLog::open`, before the recovery walk):

1. Scan `.substrate/` for `seg-*.log` (and legacy `seg-*.active`). If **any**
   segment file already exists, the store is already segmented → skip migration.
2. Otherwise, if a bare `substrate.log` exists, migrate it:
   - **Rename `substrate.log` → `seg-0000000001.log`** (it becomes the oldest
     **sealed** segment) and **create a fresh empty `seg-0000000002.log`** (the
     active, since it is now the highest id). All new writes land in the small
     active; the big legacy blob is read-only sealed state.
3. If neither exists, this is a fresh store → create `seg-0000000001.log`.

Separately, on **every** open a leftover legacy `seg-<id>.active` (from before
the `.active` extension was retired) is **adopted** — renamed to `seg-<id>.log`
— so the single-namespace scan picks it up as the highest-id active.

The migrated `seg-0000000001.log` may carry a **torn tail** (the legacy file
crashed mid-append). Sealed-segment recovery (§7, the ascending-id walk) already
detects a torn tail via the walker and **truncates that segment file once**, in
place — so no dedicated pre-pass or rewrite is needed. A legacy file with no
`HeaderIndex` chain simply takes the full-walk recovery path for that one
segment, exactly as the monolithic open does today.

The 29 GB / calibration bloat is **not** rewritten at migration time: it becomes
one sealed segment whose dead records are reclaimed by the ordinary background
**compact/combine/drop** passes (§6) over the following minutes — relocating its
live records into the active and then `unlink`-ing it. Migration cost is two
directory operations regardless of legacy file size.

`FILE_FORMAT_VERSION` is **not** bumped for segmentation: the per-segment
on-disk record/superblock format is byte-identical to the monolithic format
(segmentation is a file-set/addressing change, not a wire-format change), so a
sealed `seg-*.log` is exactly a legacy `substrate.log` under a new name.

## 14. Phased rollout

1. **Addressing** *(done)* — `SegmentId` + `RecordLoc`/`ChunkLoc` threaded through
   the stream index, still one segment (`FIRST_SEGMENT`). Pure plumbing; reads +
   recovery identical.
2. **Seal/rotate + handle pool + migration** — the `SegmentedLog` abstraction:
   directory scan (single `seg-<id>.log` namespace — active = highest id, no
   `.active` extension or rename-on-seal), auto-split migration (§13), sealing at
   `SEGMENT_TARGET_BYTES` (create-next), the LRU `DirectFile` pool,
   `WalkEntry.segment` threading (§5.1), and multi-segment ascending-id recovery.
3. **Per-segment accounting** — shard `RecordAccounting`; cross-segment
   supersession + distill/tombstone debits.
4. **Background drop/compact/combine** — the persistence-thread pass + trigger
   policy; remove the startup compaction gate.
5. **Delete** `write_compacted_log` and the whole-file compactor.
6. **Tooling + GUI** — `substrate_inspect --dir`, the `CompactionStatus` API +
   GUI indicator.

Phases 1–4 are mostly CPU-testable: auto-split migration, segment sizing,
id-order recovery, per-segment accounting, and **crash-point simulations**
(truncate/kill between each fsync in §7) to prove the manifest-free model. The
per-segment cold-load routing (`chunk_plan` `SourceLog::Sealed(id)` + `pipeline`
handle selection) is exercised on the live GPU daemon, since it drives the
`kv_migrate` scatter.

## 15. Resolved decisions

1. **Relocation target = the active segment.** Compact/combine append relocated
   live records to the active (highest-id) segment, then drop the sources. This
   is the invariant that keeps the store **manifest-free** (id-order = recency).
   *No LSN/generation state is introduced.*
2. **Compact trigger = `dead_ratio ≥ 10%` AND `age_since_seal ≥ 60 s`** (settle +
   rate-limit). Drop is unconditional at `live_bytes == 0`; combine on adjacency
   + the 2 GB size threshold.
3. **Turns may span a seal.** Seal on any record boundary at ~4 GB; a turn's
   chunks may cross into the next segment (each chunk is independently
   `(segment, offset)`-addressed). Only that one turn's cold-load splits into a
   per-segment stripe batch.
4. **Write-avoidance is a separate follow-up.** Land segmentation first (it fully
   resolves reclaim + speed). "Skip cold-persisting distilled-timeline KV" is a
   later, independent change; segments just fill slightly faster until then.

Remaining defaults (not blockers — adjustable during implementation):
- `OPEN_SEALED_SEGMENTS = 8` (LRU of open sealed `DirectFile`s). Revisit only if
  `RLIMIT_NOFILE` proves tight.
- `provenance/raw_store.rs` stays single-file, out of scope.

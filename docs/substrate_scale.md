# Substrate cost proportional to what is used, not to what is stored

The conversation substrate is an append-only redo log with an in-RAM index.
Opening it is meant to read record headers — through the `HeaderIndex` chain —
and keep locations, reading a payload only when something uses it. Serving is
meant to cost what a request touches. A long-lived store breaks both: on
WhiteWorld (2026-11-22; zend's store: ~500k streams, 19.95M chunk records,
478k wide-Q signature records, 21.6M accounting entries) the open left ~36 GB
of heap, most of it never touched again, and several background passes walk the
whole index every few seconds.

This document is the audit of every such cost and the plan that removes each.
Each item names where the cost is, what it scales with, and the change.
Sizes marked *est.* are from type layouts, not measured.

## 1. What the open reads and keeps

`payload_needed` (`persistence/recovery.rs`) and the walker
(`persistence/walker.rs`) skip the payloads of `Chunk`, `Tokens`, `Snapshot`,
`BranchCheckpoint`, `TurnIndexPage`, `VfsEvent`, `Blob` and the singletons.
Everything else is read whole.

| # | Cost | Where | Scales with | Change |
|---|---|---|---|---|
| O1 | `WideQSig` and `ProjectionEvents` payloads read and held whole per stream: **16.0 GB measured** (`held_payload_bytes`) | `substrate/loading.rs` → `apply_wide_q_sigs` / `apply_projection_events`; `StreamRuntime.{wide_q_sigs, projection_events}` | per token × heads (sigs); per projection (events) | Located, not held, as `TurnIndexPage` is: the substrate keeps the record's location and the facts a scan reads every pass — a window's token count, width and fold, parsed from the record's head; an events digest decoded on first use. Bytes are read from the log when an upload, a host scan or a reader needs them. A blob sealed in this process is held until it is durable, then settles to its location. |
| O2 | The scheduler's replay reads **every turn's `Tokens` record** from disk, one random read a turn, and keeps a `Vec<u32>` a turn for the process | `projection/resolver/log_reconstruct.rs`, `persistence/resume/recover.rs` `recover_turn_meta` | turns (reads); tokens (RAM) | Token ids are read when a turn is projected or cold-loaded, from the location the stream already holds. |
| O3 | Turn text held twice: the decoded `TurnDecl` in `StreamRuntime.decl`, and a clone moved into `TurnPart.layout` | `persistence/resume/recover.rs` `recovered_turn_decls`; `log_reconstruct.rs` | text bytes | One shared `Arc<[TurnSegment]>`. |
| O4 | The chain path fetches every metadata payload of a segment into one `HashMap` before replaying any | `recovery.rs` `recover_with_sink`, `fetch_records_coalesced` | metadata bytes per segment | Fetch and replay in bounded batches, in append order. |
| O5 | A compaction or adoption at open keeps a second copy of every in-RAM payload in `Rewritten.walked` | `persistence/compaction/rewritten.rs`, `compacting.rs` | metadata bytes | Keep locations; with O1 the large payloads are no longer in RAM to copy. |
| O6 | Custom objects held decoded as `serde_json::Value` (journal text, npcd accounts, world state) | `substrate/apply_records.rs` `apply_custom_object` | JSON size × `Value` overhead | Hold the encoded bytes; decode on read. |
| O7 | npcd decodes every `Npc` record, superseded ones included, then re-reads every live character | `npcd/src/npcs/methods.rs` `load_reporting`, `caught_up` | characters × versions | Decode only the record `npc_locs` names, once. |

### 1.1 O1 in detail — located signatures and events

- **The walk.** `payload_needed` skips `WideQSig` and `ProjectionEvents`;
  `Rewritten::keep` strips them. `apply_walker_entry` records each as
  `Located::Stored(RecordLoc)` — at open, after a compaction's re-apply, and for
  a record replicated in (its local location). Applying the record the stream
  already holds under the same identity only moves its location: no gallery or
  content epoch, no new fingerprint, nothing to re-upload.
- **This process's own writes** are `Located::Held` from the apply until the
  owner's append returns, then settle to the location (`settle_located`), as
  index pages do.
- **Reading.** `Substrate::payloads()` is a `PayloadSource` every version of the
  substrate shares: the store's `SealedReader`, set when the store opens, and a
  weak handle to its `PersistenceOwner`, set when the owner spawns. A read of a
  stored record goes lock-free to the durable copy at its location and checks
  the header names the same stream and identity; otherwise it reads the record
  by identity through the owner, which finds it wherever maintenance moved it.
  A read never runs on the owner's thread or inside a substrate edit.
- **What scans read without the bytes.** A signature window's token count,
  width and fold come from the record's first bytes, read once and memoized on
  the stream (`sig_head`); its fingerprint is the stamp the substrate gave the
  value when it was written, so a window is re-uploaded exactly when the value
  changed. The events digest is decoded on first use and memoized, as before.
- **Maintenance and compaction** relocate a stored record raw, by location —
  the index-page stage generalised to every located per-stream record — and
  re-emit only what is held. `drop_segments` forgets a stored location into a
  dropped segment.
- **`replicated_state`** renders both by identity and length, since one member
  may hold a value's bytes while another holds only where they are.

## 2. What the in-RAM index costs

The persistent maps (`imbl`) allocate a full node — 64 keys, 65 children — for
a map of any size, including an empty one (`pcoll.rs`). Every chunk location is
also held about five times.

| # | Cost | Where | *est.* | Change |
|---|---|---|---|---|
| M1 | Per chunk: `StreamRuntime.chunks` (`POrdMap<u64, ChunkLoc>`), `StreamIds.chunks` (`POrdMap<u64, RecordIdentity>`), `RecordAccounting.live_sizes` + `owned`, a `LocationIndex` slot, and a residence's `StoredChunk` cold copy — ~470–510 B a chunk | `substrate/entries.rs`, `substrate/stamps.rs`, `persistence/accounting.rs`, `substrate/residence.rs` | ~10 GB at 20M chunks | One dense per-stream table of packed locations (chunk indices are dense); chunk identities range-encoded; accounting stops keying chunks (dead bytes counted when a chunk is superseded); the cold copy dropped — cold reads plan from the stream. |
| M2 | `RecordAccounting.live_sizes`: 21.6M entries in 33.5M 80-byte buckets — **the 2.72 GB single allocation** (exact) | `persistence/accounting.rs` | 2.72 GB | With M1, chunks leave it; what remains is per metadata record. |
| M3 | Empty `imbl` roots: ~30 KB per timeline before any content; one 1.6 KB inner node per stream in `StreamIndex.streams` | `TimelineEntry::new`, `TimelineIds`, `stream_index.rs` | T × 30 KB + ~0.9 GB | Small maps that are usually empty become absent until written; one writer per stream is held without a map. |
| M4 | `metadata_locs` / `resident_locs` / `marker_ids` as `PMap`s, ~1.2 KB a key | `persistence/stamped_locs.rs`, `identity/marker_ids.rs` | 1.6–3 GB | Locations move into the stream's own slots, or a flat hash map. |
| M5 | `LocationIndex` slots repeat the stream and chunk index the substrate already holds | `persistence/replica/location_index.rs` | ≥1 GB | Slot keeps the packed location; the header is listed from the substrate. |
| M6 | A 2.1 GB single allocation in the loader scope, unattributed | — | 2.1 GB | Attribute first (scoped census at open). |

## 3. What serving does over the whole store

| # | Cost | Trigger | Scales with | Change |
|---|---|---|---|---|
| R1 | A replicated declaration that displaces a turn reloads **every** turn — one `Tokens` read each — on the scheduler thread | `apply_records.rs` `reload_due` → `ReconstructSubstrate` | turns, random reads | Reload only the timelines whose positions moved; no token reads (O2). |
| R2 | Maintenance walks every stream, chunk and metadata location every 15 s, and gates on age and dead ratio only after the walk | `persistence/maintenance/worklist.rs` `segment_liveness`, `plan.rs` | chunks + streams | Live bytes per segment kept current on append and supersede; the plan reads them. |
| R3 | Planning an op JSON-encodes every resident record to keep the few in the target segment | `maintenance/resident.rs` `gather_resident_set` | streams | Records indexed by segment; the plan visits the target's alone. |
| R4 | While the disk is full, R2's walk runs every second | `cluster/src/replicate/fault.rs` | chunks + streams | Falls out of R2. |
| R5 | `held_payload_bytes` iterates every stream on every persistence pass | `persistence/thread/pass.rs`, `substrate/tiers.rs` | streams | A running total kept by the applicators. |
| R6 | Belief gallery reassembly keyed on the global `gallery_epoch`: any write anywhere re-walks the collection's whole scope | `projection/resolver/belief_gallery.rs` | turns in scope, per reprojection | Keyed on the collection's own epoch. |
| R7 | Each open tab's sidebar poll lists every timeline twice | `zend/src/session/conversations.rs` | timelines, per poll | An index from conversation id and metadata key to timelines. |
| R8 | Other per-request timeline scans: `IngestIndex::read`, `has_live_intact_metadata` (per repo-scan job), npcd's conv-id prefix lookups | `zend/src/retrieval_scope/index.rs`, `substrate/metadata.rs` | timelines | The index of R7; ordered for prefix lookups. |
| R9 | `pinned_hot_by_timeline` walks every turn once a minute | `substrate/eviction.rs` | turns | From the pinned residences. |
| R10 | Warm replay decodes every dialogue turn's signature to keep 512; collection normalization walks every stream once a collection | `warm_replay.rs`, `warm_collection.rs` | turns | Order first, decode the kept; use the tagged-turn index. |
| R11 | Substrate overview, npcd document retirement, COORD tombstone counts, section persist re-snapshotting, calibration cache key | see each | streams / timelines | Counters kept on write; indexed lookups. |

## 4. Order

Largest first, each landed whole with its tests:

1. **O1** — 16 GB of heap and most of the open's payload reads.
2. **O2, O3, R1** — the replay's per-turn reads and the second text copy.
3. **M1, M2** — the per-chunk copies and the accounting table.
4. **R2–R5** — the 15 s maintenance walk and the per-pass scan.
5. **M3–M6, O4–O7** — the remaining index and open costs.
6. **R6–R11** — the per-request scans.

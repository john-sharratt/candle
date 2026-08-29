//! Background maintenance for the segmented redo log — drop / compact /
//! combine (`docs/segmented_substrate_log.md` §6).
//!
//! One shared mechanism — **relocate a sealed segment's live records into the
//! active, then unlink the segment** — with three triggers:
//!
//! - **Drop (O(1))** — a sealed segment with no live records: unlink it. The
//!   calibration / 29 GB case becomes a delete, not a rewrite.
//! - **Compact (O(one segment))** — a sealed segment past the savings + settle
//!   thresholds (§8): relocate its live records, then drop it.
//! - **Combine (O(two segments))** — two adjacent small sealed segments:
//!   relocate both, drop both. Reduces segment count / open-handle pressure.
//!
//! Runs on the persistence thread's background pass, **one op per pass**, and
//! **never at startup**.
//!
//! ## Phased locking (no inference starvation)
//!
//! The op is split so the slow relocation I/O never holds the substrate write
//! lock — mirroring the persistence thread's hot→warm / warm→cold discipline
//! (snapshot locked, I/O unlocked, install locked):
//!
//! 1. [`SubstratePersistence::plan_maintenance`] — pick the op + snapshot the
//!    resident set and relocation worklist (brief **read** + persistence lock).
//! 2. [`SubstratePersistence::execute_maintenance`] — append the resident set,
//!    relocate the targets' live records, fsync (**persistence lock only** — the
//!    substrate lock is released, so decode's in-RAM projection proceeds).
//! 3. [`MaintenanceResult::apply_to_substrate`] — repoint the index at the
//!    relocated records (brief **write** lock), with a supersession guard.
//! 4. [`SubstratePersistence::finish_maintenance`] — unlink the drained sources
//!    (persistence lock).
//!
//! Maintenance runs only on the single persistence thread, but the substrate
//! writer thread can still append between phases (it takes the persistence
//! lock per job) — so a record planned for relocation may be superseded before
//! execute. Two guards cover that window: execute re-validates each planned
//! `Snapshot` against the persistence-side live-tail map (`snapshot_locs`) and
//! skips superseded ones, and [`MaintenanceResult::apply_to_substrate`] only
//! repoints index entries that still match their planned old location. Reads
//! always see a consistent index (the sources stay live until step 4).
//!
//! ## Why relocation is safe (no resurrection, no lost metadata)
//!
//! Records fall into two classes. **Read-back** records (`Chunk`, `Tokens`,
//! `ModelSpec`/`Template`/`Tokenizer`) are indexed by `(segment, offset)`; a
//! relocation re-appends them to the active and repoints the in-RAM index. Their
//! per-segment live set is [`SubstratePersistence::segment_liveness`], with the
//! same distill/tombstone filter [`super::compaction::collect_live_records`]
//! uses. **Resident** records (`StreamDecl`, `Commit`, `ProjectionEvents`,
//! `WideQSig`, `Label`, `ConvState`, `TreeMetadata`, `DebugId`, `Distilled`,
//! `Tombstone`) carry no location — the substrate holds their decoded state — so
//! before any drop the whole resident set is re-emitted into the active
//! ([`SubstratePersistence::re_emit_resident_set`]). That guarantees a dropped
//! segment never takes the only on-disk copy of live metadata with it, and the
//! re-emitted **tombstone markers** keep tombstoned timelines dead even while
//! their (excluded) records still physically linger in undropped segments.
//!
//! Crash safety follows the write→fsync→unlink order: the relocated copies are
//! committed to the active before any source is unlinked, and until the unlink
//! the source's higher-id-losing copies are simply superseded by id order.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::SystemTime;

use super::manifest::{
    encode_conv_state_payload, encode_label_payload, ChunkLoc, ConvState, RecordLoc,
};
use super::record::{
    DebugIdPayload, DistillMode, DistillPayload, RecordHeader, RecordType, TombstonePayload,
};
use super::segment::SegmentId;
use super::streams::{StreamDecl, StreamId};
use super::{Result, SubstratePersistence};
use crate::substrate::Substrate;

/// Compact trigger: a sealed segment must be at least this fraction dead
/// weight before relocating it is worth the I/O (§8).
pub const SEGMENT_COMPACT_MIN_DEAD: f64 = 0.10;

/// Compact trigger: a sealed segment must have settled at least this long
/// since it was sealed (or last compacted) — a rate-limit so a just-sealed
/// segment accumulates deaths before it is churned (§8).
pub const SEGMENT_COMPACT_MIN_AGE_SECS: u64 = 60;

/// Combine trigger: two adjacent sealed segments each below this live size
/// merge into the active, cutting the segment count after compaction has
/// shrunk neighbours (§6).
pub const COMBINE_SEGMENT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One sealed segment's stats, fed to [`pick_maintenance_op`].
#[derive(Clone, Copy, Debug)]
pub struct SegmentStat {
    pub id: SegmentId,
    /// Record bytes on disk (excluding the superblock).
    pub total_bytes: u64,
    /// Live read-back record bytes (superseded / tombstoned / distilled
    /// excluded).
    pub live_bytes: u64,
    /// Seconds since the segment was sealed (file mtime).
    pub age_secs: u64,
}

/// The maintenance op chosen for one background pass (§6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaintenanceOp {
    /// Drop a fully-dead sealed segment (no live read-back records).
    Drop(SegmentId),
    /// Relocate one sealed segment's live records into the active, then drop it.
    Compact(SegmentId),
    /// Relocate two adjacent small sealed segments, then drop both.
    Combine(SegmentId, SegmentId),
}

impl MaintenanceOp {
    /// The segment(s) this op relocates-then-drops.
    pub fn targets(&self) -> Vec<SegmentId> {
        match *self {
            MaintenanceOp::Drop(a) | MaintenanceOp::Compact(a) => vec![a],
            MaintenanceOp::Combine(a, b) => vec![a, b],
        }
    }

    /// A short human-readable label for the daemon status / GUI indicator.
    pub fn label(&self) -> String {
        match *self {
            MaintenanceOp::Drop(a) => format!("dropped segment {a}"),
            MaintenanceOp::Compact(a) => format!("compacted segment {a}"),
            MaintenanceOp::Combine(a, b) => format!("combined segments {a}+{b}"),
        }
    }
}

/// Choose at most one maintenance op from per-segment stats, in priority
/// order: drop a fully-dead segment (cheapest, biggest win); else compact the
/// segment that most exceeds the savings + settle thresholds; else combine the
/// first adjacent pair of small segments.
///
/// Pure — the caller supplies the stats (ages included) so the policy is
/// unit-testable without a clock or disk. `stats` is assumed ascending by id,
/// so `windows(2)` are the adjacency candidates for combine.
pub fn pick_maintenance_op(stats: &[SegmentStat], force: bool) -> Option<MaintenanceOp> {
    // 1. Drop: any fully-dead segment (unconditional — always worth deleting).
    if let Some(s) = stats.iter().find(|s| s.live_bytes == 0) {
        return Some(MaintenanceOp::Drop(s.id));
    }
    // 2. Compact the deadest segment past the ratio + settle thresholds. When
    //    `force` is set (the manual `POST /v1/debug/maintenance` trigger) the age
    //    and ratio gates are waived — any segment carrying *some* dead weight
    //    qualifies — so a just-sealed segment holding a freshly-archived
    //    conversation's now-dead records is compacted immediately.
    let mut best: Option<(SegmentId, f64)> = None;
    for s in stats {
        if s.total_bytes == 0 {
            continue;
        }
        if !force && s.age_secs < SEGMENT_COMPACT_MIN_AGE_SECS {
            continue;
        }
        let dead = s.total_bytes.saturating_sub(s.live_bytes);
        let ratio = dead as f64 / s.total_bytes as f64;
        let qualifies = if force {
            dead > 0
        } else {
            ratio >= SEGMENT_COMPACT_MIN_DEAD
        };
        if qualifies && best.is_none_or(|(_, r)| ratio > r) {
            best = Some((s.id, ratio));
        }
    }
    if let Some((id, _)) = best {
        return Some(MaintenanceOp::Compact(id));
    }
    // 3. Combine: first adjacent pair both below the size threshold.
    for w in stats.windows(2) {
        if w[0].live_bytes < COMBINE_SEGMENT_BYTES && w[1].live_bytes < COMBINE_SEGMENT_BYTES {
            return Some(MaintenanceOp::Combine(w[0].id, w[1].id));
        }
    }
    None
}

/// The distill/tombstone classification of a stream, matching
/// [`super::compaction::collect_live_records`].
fn classify(
    entry_decl: &Option<StreamDecl>,
    tombstoned: &HashSet<u64>,
    distilled: &HashMap<u64, DistillMode>,
) -> (bool, Option<DistillMode>) {
    if let Some(StreamDecl::Turn(t)) = entry_decl {
        (
            tombstoned.contains(&t.timeline_id),
            distilled.get(&t.timeline_id).copied(),
        )
    } else {
        (false, None)
    }
}

/// A resident record snapshotted from the substrate for re-emission — carries
/// no on-disk location (the substrate holds its decoded state), so it is
/// rebuilt from live state rather than read back.
struct Resident {
    rt: RecordType,
    stream_id: u64,
    chunk_index: u64,
    payload: Vec<u8>,
}

/// One already-encoded record to relocate verbatim from a source segment: its
/// on-disk `(offset, record_size)` plus a synthesized header for the accounting
/// / header-index bookkeeping (no decode needed).
struct RawReloc {
    offset: u64,
    record_size: u64,
    header: RecordHeader,
}

/// A planned maintenance op with everything the execute phase needs, gathered
/// from the substrate under a **brief** read lock — so the slow relocation I/O
/// runs without holding the substrate lock (mirrors the persistence thread's
/// hot→warm / warm→cold discipline: snapshot locked, I/O unlocked, install
/// locked). See [`SubstratePersistence::plan_maintenance`].
pub struct MaintenancePlan {
    op: MaintenanceOp,
    /// The full resident set to re-emit (the drop-safety net, §6 module doc).
    resident: Vec<Resident>,
    /// `(stream, chunk_index, source_loc)` chunks to relocate off the targets.
    chunk_relocs: Vec<(StreamId, u64, ChunkLoc)>,
    /// `(stream, source_loc)` `Tokens` records to relocate.
    token_relocs: Vec<(StreamId, RecordLoc)>,
    /// `(snapshot stream, source_loc)` recurrent-state snapshots to relocate.
    snapshot_relocs: Vec<(StreamId, RecordLoc)>,
    /// `(type, source_loc)` singleton records to relocate.
    singleton_relocs: Vec<(RecordType, RecordLoc)>,
}

impl MaintenancePlan {
    /// The op this plan executes.
    pub fn op(&self) -> MaintenanceOp {
        self.op
    }
}

/// The relocated records' new locations, produced by
/// [`SubstratePersistence::execute_maintenance`] and applied to the substrate
/// index under a **brief** write lock ([`Self::apply_to_substrate`]).
pub struct MaintenanceResult {
    /// `(stream, chunk_index, old_loc, new_loc)`.
    chunk_updates: Vec<(StreamId, u64, ChunkLoc, ChunkLoc)>,
    /// `(stream, old_loc, new_loc)`.
    token_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
    /// `(snapshot stream, old_loc, new_loc)`.
    snapshot_updates: Vec<(StreamId, RecordLoc, RecordLoc)>,
}

impl MaintenanceResult {
    /// Repoint the substrate index at the relocated records, then refresh cold
    /// residence refs. **Supersession guard:** a record whose index entry no
    /// longer points at `old_loc` was rewritten since the plan was made, so its
    /// relocated copy is dead (reclaimed by a later pass) and skipped — the
    /// newer copy wins. Held under a brief substrate write lock only.
    pub fn apply_to_substrate(&self, substrate: &mut Substrate) {
        for (sid, idx, old, new) in &self.chunk_updates {
            if substrate.stream_of(*sid).and_then(|s| s.chunks.get(idx)) == Some(old) {
                substrate.apply_chunk_loc(*sid, *idx, *new);
            }
        }
        for (sid, old, new) in &self.token_updates {
            if substrate.stream_of(*sid).and_then(|s| s.tokens.as_ref()) == Some(old) {
                substrate.apply_tokens_loc(*sid, *new);
            }
        }
        for (sid, old, new) in &self.snapshot_updates {
            // Same supersession guard: a snapshot rewritten (a newer seal)
            // since the plan was made keeps its newer location; the relocated
            // copy of the old one is dead and reclaimed by a later pass.
            if substrate.recurrent_snapshot_loc(*sid) == Some(*old) {
                substrate.apply_snapshot_loc(*sid, *new);
            }
        }
        substrate.refresh_cold_refs();
    }
}

/// Snapshot every **resident** record from the substrate's live state — the
/// drop-safety net (re-emitted so no dropped segment holds the only copy of
/// live metadata), including a `Tombstone` marker per tombstoned timeline and a
/// `Distilled` marker per distilled timeline. Pure read; no I/O.
fn gather_resident_set(substrate: &Substrate) -> Vec<Resident> {
    let tombstoned: HashSet<u64> = substrate
        .tombstoned_timelines()
        .iter()
        .map(|t| t.raw())
        .collect();
    let distilled: HashMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();

    let mut out: Vec<Resident> = Vec::new();
    for (stream_id, entry) in substrate.all_streams() {
        let (is_tomb, distill) = classify(&entry.decl, &tombstoned, &distilled);
        // Tombstoned AND undistilled goes; tombstoned-but-distilled is the
        // provenance corpus and is retained by its mode. See the same gate in
        // `compaction::collect_live_records`.
        if is_tomb && distill.is_none() {
            continue;
        }
        // Orphan (no decl): re-emit nothing — its StreamDecl is gone, so its
        // sig/projection records are unreachable dead weight. Re-emitting them
        // would keep the orphan's metadata immortal on the incremental path.
        // (The `if let Some(decl)` below already skips the StreamDecl; this also
        // skips the sig/projection re-emit.) Mirrors the other orphan gates.
        if entry.decl.is_none() {
            continue;
        }
        let keep_sig = distill != Some(DistillMode::TextOnly);
        if let Some(decl) = &entry.decl {
            out.push(Resident {
                rt: RecordType::StreamDecl,
                stream_id: stream_id.0,
                chunk_index: 0,
                payload: decl.encode(),
            });
        }
        if keep_sig {
            if let Some(p) = &entry.projection_events {
                out.push(Resident {
                    rt: RecordType::ProjectionEvents,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    payload: p.clone(),
                });
            }
            if let Some(p) = &entry.wide_q_sigs {
                out.push(Resident {
                    rt: RecordType::WideQSig,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    payload: p.clone(),
                });
            }
        }
        if let Some(through) = entry.committed_through {
            out.push(Resident {
                rt: RecordType::Commit,
                stream_id: stream_id.0,
                chunk_index: through,
                payload: Vec::new(),
            });
        }
    }
    for (tl, conv, label, archived, custom) in substrate.live_conv_meta() {
        // A tombstoned timeline that is ALSO distilled is the provenance corpus
        // (calibration exemplars: archived, distilled, then tombstoned out of the
        // live gather while their signatures keep answering the belief scan).
        // Its marker/metadata must survive, or the next reload sees a
        // content-declaring turn with no distill exemption and condemns it.
        // Only an undistilled tombstone is a retired conversation.
        if tombstoned.contains(&tl) && !distilled.contains_key(&tl) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::Label,
            stream_id: 0,
            chunk_index: 0,
            payload: encode_label_payload(tl, &conv, &label, &custom),
        });
        if archived {
            out.push(Resident {
                rt: RecordType::ConvState,
                stream_id: 0,
                chunk_index: 0,
                payload: encode_conv_state_payload(tl, ConvState { archived: true }),
            });
        }
    }
    for p in substrate.live_tree_metadata_payloads() {
        if tombstoned.contains(&p.timeline_id) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::TreeMetadata,
            stream_id: 0,
            chunk_index: 0,
            payload: p.encode(),
        });
    }
    for (tl, id) in substrate.live_debug_ids() {
        if tombstoned.contains(&tl) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::DebugId,
            stream_id: 0,
            chunk_index: 0,
            payload: DebugIdPayload {
                timeline_id: tl,
                debug_id: id,
            }
            .encode(),
        });
    }
    // Re-emitted for tombstoned timelines too: the marker is what exempts a
    // distilled turn from the integrity verdict, and a distilled turn has
    // legitimately shed its tokens/KV. Losing it condemns the provenance corpus
    // as "corrupt" on the next reload.
    for (&tl, &mode) in &distilled {
        out.push(Resident {
            rt: RecordType::Distilled,
            stream_id: 0,
            chunk_index: 0,
            payload: DistillPayload {
                timeline_id: tl,
                mode,
            }
            .encode(),
        });
    }
    for t in substrate.tombstoned_timelines() {
        out.push(Resident {
            rt: RecordType::Tombstone,
            stream_id: 0,
            chunk_index: 0,
            payload: TombstonePayload {
                timeline_id: t.raw(),
                turn_index: None,
                reason: None,
            }
            .encode(),
        });
    }
    out
}

impl SubstratePersistence {
    /// Run one maintenance op end to end (plan → execute → apply → drop) under
    /// the caller's already-held substrate lock. Convenience for single-threaded
    /// callers (tests); the daemon uses the **phased** API — [`Self::plan_maintenance`]
    /// → [`Self::execute_maintenance`] → [`MaintenanceResult::apply_to_substrate`]
    /// → [`Self::finish_maintenance`] — so the slow relocation I/O never holds
    /// the substrate write lock. Returns the op that ran, or `None` for a no-op.
    pub fn run_maintenance(&mut self, substrate: &mut Substrate) -> Result<Option<MaintenanceOp>> {
        let Some(plan) = self.plan_maintenance(substrate, false)? else {
            return Ok(None);
        };
        let op = plan.op;
        let result = self.execute_maintenance(&plan)?;
        result.apply_to_substrate(substrate);
        self.finish_maintenance(&plan)?;
        Ok(Some(op))
    }

    /// Count of sealed segments plus the active — surfaced in the daemon's
    /// status for the GUI's compaction indicator.
    pub fn segment_count(&self) -> usize {
        self.segments.sealed_ids().len() + 1
    }

    /// **Phase 1** — pick an op and gather everything needed to execute it,
    /// reading the substrate + manifest. Runs under a brief read lock; returns
    /// `None` for a no-op pass. The gathered plan (resident snapshot +
    /// relocation worklist) lets the slow I/O phase run without the substrate
    /// lock.
    pub fn plan_maintenance(
        &self,
        substrate: &Substrate,
        force: bool,
    ) -> Result<Option<MaintenancePlan>> {
        let sealed = self.segments.sealed_ids().to_vec();
        if sealed.is_empty() {
            return Ok(None);
        }
        let liveness = self.segment_liveness(substrate);
        let now = SystemTime::now();
        let mut stats = Vec::with_capacity(sealed.len());
        for &id in &sealed {
            stats.push(SegmentStat {
                id,
                total_bytes: self.segments.sealed_record_bytes(id)?,
                live_bytes: liveness.get(&id).copied().unwrap_or(0),
                age_secs: self.segments.sealed_age_secs(id, now)?,
            });
        }
        let Some(op) = pick_maintenance_op(&stats, force) else {
            return Ok(None);
        };
        // Re-emit the resident (metadata) set only when this op could otherwise
        // lose a unique metadata record — i.e. it targets a segment not covered by
        // the last re-emission. Skipping it for older-only targets is what breaks
        // the re-emit→looks-dead→compact churn (see `need_resident_reemit`).
        let resident = if self.need_resident_reemit(&op.targets()) {
            gather_resident_set(substrate)
        } else {
            Vec::new()
        };
        let (chunk_relocs, token_relocs, snapshot_relocs, singleton_relocs) =
            self.gather_relocations(substrate, &op.targets());
        Ok(Some(MaintenancePlan {
            op,
            resident,
            chunk_relocs,
            token_relocs,
            snapshot_relocs,
            singleton_relocs,
        }))
    }

    /// Whether a maintenance op targeting `targets` must re-emit the whole
    /// resident (metadata) set as a drop-safety net, or can safely skip it.
    ///
    /// Safe to skip iff every target segment is strictly older than
    /// [`resident_reemit_floor`](SubstratePersistence::resident_reemit_floor):
    /// segment ids are append-ordered, so the last re-emission duplicated every
    /// then-existing metadata record into segments `>= floor`; a target `<
    /// floor` therefore holds only records that predate the re-emission, each
    /// with a surviving copy at `>= floor`. Dropping it loses nothing. A `None`
    /// floor (no durable snapshot yet) or any target `>= floor` (which may hold
    /// metadata written after the last re-emission) forces a re-emit.
    fn need_resident_reemit(&self, targets: &[SegmentId]) -> bool {
        // Exact guard for per-stream metadata (`StreamDecl` / `WideQSig` /
        // `ProjectionEvents` / `Commit`): `metadata_locs` holds ONLY the current
        // (last-writer-wins) copy of each, so if any target segment holds one, it
        // is the sole durable copy of a LIVE record — dropping without re-emitting
        // would delete a live turn's decl outright (the silent-loss bug: turns
        // vanish on reload, their KV orphaned). The floor heuristic alone
        // mis-skipped this whenever the current copy sat below the floor (e.g. a
        // decl sealed in the plan→execute window, which lands under the
        // execute-time floor without a re-emitted copy). A segment holding only
        // SUPERSEDED metadata has no `metadata_locs` entry pointing at it, so this
        // still skips it — no re-emit→looks-dead→compact churn.
        if self
            .metadata_locs
            .values()
            .any(|loc| targets.contains(&loc.segment))
        {
            return true;
        }
        // Timeline-keyed resident metadata (`Label` / `ConvState` / `TreeMetadata`
        // / `DebugId` / `Tombstone` / `Distilled`) carries no tracked location, so
        // the append-ordered floor invariant guards it: a target strictly below
        // the last re-emission's floor holds only records already duplicated at
        // `>= floor`.
        match self.resident_reemit_floor {
            None => true,
            Some(floor) => targets.iter().any(|t| *t >= floor),
        }
    }

    /// **Phase 2** — the slow I/O: append the resident set, then relocate each
    /// target segment's live records into the active, then commit. Holds only
    /// the persistence lock; the caller does **not** hold the substrate lock
    /// here, so decode's in-RAM projection reads/writes proceed. The relocated
    /// copies are durable (fsynced) before any source is unlinked. Returns the
    /// new locations for [`MaintenanceResult::apply_to_substrate`].
    pub fn execute_maintenance(&mut self, plan: &MaintenancePlan) -> Result<MaintenanceResult> {
        // Resident set — re-emitted from in-RAM state (not read from disk), so
        // the normal encoding append. When re-emitted, record the active segment
        // it lands into (captured BEFORE the appends, the floor of the segments
        // the re-emission writes to) as the new drop-safety floor, so subsequent
        // older-only ops can skip the re-emission (see `need_resident_reemit`). An
        // empty resident set means the planner chose to skip — leave the floor.
        if !plan.resident.is_empty() {
            let reemit_floor = self.segments.active_id();
            for r in &plan.resident {
                self.append_record(r.rt, 0, r.stream_id, r.chunk_index, 0, 0, &r.payload)?;
            }
            self.resident_reemit_floor = Some(reemit_floor);
        }

        // Chunks — the relocation bulk. Group by source segment and move each
        // group with **coalesced reads + verbatim staging** (no decode / CRC /
        // re-encode) so the fast block-read path is used, not a syscall per
        // record.
        let mut chunk_updates = Vec::with_capacity(plan.chunk_relocs.len());
        let mut by_seg: BTreeMap<SegmentId, Vec<(StreamId, u64, ChunkLoc)>> = BTreeMap::new();
        for &(sid, idx, old) in &plan.chunk_relocs {
            by_seg.entry(old.segment).or_default().push((sid, idx, old));
        }
        for (source, recs) in by_seg {
            let items: Vec<RawReloc> = recs
                .iter()
                .map(|&(sid, idx, old)| RawReloc {
                    offset: old.offset,
                    record_size: old.record_size,
                    header: RecordHeader {
                        record_type: RecordType::Chunk,
                        format: old.format,
                        payload_len: old.payload_len,
                        crc: 0,
                        stream_id: sid.0,
                        chunk_index: idx,
                        token_count: old.token_count,
                    },
                })
                .collect();
            let new = self.relocate_raw_from_segment(source, &items)?;
            for ((sid, idx, old), (segment, offset, record_size)) in recs.into_iter().zip(new) {
                chunk_updates.push((
                    sid,
                    idx,
                    old,
                    ChunkLoc {
                        segment,
                        offset,
                        payload_len: old.payload_len,
                        record_size,
                        token_count: old.token_count,
                        format: old.format,
                    },
                ));
            }
        }

        // Tokens — same coalesced verbatim path.
        let token_updates = self.relocate_stream_records(RecordType::Tokens, &plan.token_relocs)?;

        // Snapshots — the same verbatim path, behind a supersession check.
        // Snapshot records are last-writer-wins by append order (keyed
        // `(Snapshot, stream_id)`), and the seal thread's writer can append a
        // NEWER tail for a planned stream between plan and execute (the plan
        // holds no lock across that window). Relocating the planned — now
        // stale — copy would append it physically AFTER the newer record, so
        // the next reload's walk would install the stale copy as the tail
        // (rolled-back recurrent state), and the accounting would credit the
        // newer LIVE record's bytes as dead. So a planned snapshot whose
        // location is no longer the stream's live tail (per `snapshot_locs`,
        // the persistence-side twin of the substrate index — current under
        // this thread's persistence lock, and also emptied by a timeline
        // tombstone) is skipped entirely: it is dead in its source segment,
        // produces no update for `apply_to_substrate`, and the segment drop
        // reclaims it.
        let live_snapshots: Vec<(StreamId, RecordLoc)> = plan
            .snapshot_relocs
            .iter()
            .filter(|(sid, old)| self.snapshot_locs.get(&sid.0) == Some(old))
            .copied()
            .collect();
        let snapshot_updates =
            self.relocate_stream_records(RecordType::Snapshot, &live_snapshots)?;

        // Singletons — few (≤3); the encoding append repoints them via
        // `manifest.ingest`.
        for &(rt, old) in &plan.singleton_relocs {
            let rec = self
                .segments
                .read_record_at(old.segment, old.offset, old.record_size)?;
            self.append_record(rt, 0, 0, 0, 0, 0, &rec.payload)?;
        }

        // Durability barrier: relocated copies are fsynced before any source is
        // unlinked (in `finish_maintenance`).
        self.commit()?;
        Ok(MaintenanceResult {
            chunk_updates,
            token_updates,
            snapshot_updates,
        })
    }

    /// Relocate a set of already-encoded records from `source` into the active
    /// via **coalesced stripe reads + verbatim staging**. Records are sorted by
    /// offset and adjacent ones read in a single I/O; each is staged byte-for-
    /// byte (its CRC preserved — no re-encode). Returns the new
    /// `(segment, offset, size)` per item, in input order.
    fn relocate_raw_from_segment(
        &mut self,
        source: SegmentId,
        items: &[RawReloc],
    ) -> Result<Vec<(SegmentId, u64, u64)>> {
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_unstable_by_key(|&i| items[i].offset);
        let mut new_locs = vec![(source, 0u64, 0u64); items.len()];
        let mut buf: Vec<u8> = Vec::new();
        let mut i = 0;
        while i < order.len() {
            // Coalesce a contiguous run of records into one stripe read.
            let start = items[order[i]].offset;
            let mut end = start + items[order[i]].record_size;
            let mut j = i + 1;
            while j < order.len() && items[order[j]].offset == end {
                end += items[order[j]].record_size;
                j += 1;
            }
            let len = (end - start) as usize;
            if buf.len() < len {
                buf.resize(len, 0);
            }
            self.segments.read_into(source, start, &mut buf[..len])?;
            for &k in &order[i..j] {
                let it = &items[k];
                let within = (it.offset - start) as usize;
                let raw = &buf[within..within + it.record_size as usize];
                new_locs[k] = self.append_raw_record(&it.header, raw)?;
            }
            i = j;
        }
        Ok(new_locs)
    }

    /// Relocate per-stream `RecordLoc`-shaped records (`Tokens` / `Snapshot`)
    /// verbatim into the active segment — grouped by source segment for
    /// coalesced stripe reads — returning `(stream, old_loc, new_loc)` per
    /// input record. The two types share the same header shape (format 0,
    /// chunk_index 0, token_count 0); only `record_type` differs.
    fn relocate_stream_records(
        &mut self,
        record_type: RecordType,
        relocs: &[(StreamId, RecordLoc)],
    ) -> Result<Vec<(StreamId, RecordLoc, RecordLoc)>> {
        let mut updates = Vec::with_capacity(relocs.len());
        let mut by_seg: BTreeMap<SegmentId, Vec<(StreamId, RecordLoc)>> = BTreeMap::new();
        for &(sid, old) in relocs {
            by_seg.entry(old.segment).or_default().push((sid, old));
        }
        for (source, recs) in by_seg {
            let items: Vec<RawReloc> = recs
                .iter()
                .map(|&(sid, old)| RawReloc {
                    offset: old.offset,
                    record_size: old.record_size,
                    header: RecordHeader {
                        record_type,
                        format: 0,
                        payload_len: old.payload_len,
                        crc: 0,
                        stream_id: sid.0,
                        chunk_index: 0,
                        token_count: 0,
                    },
                })
                .collect();
            let new = self.relocate_raw_from_segment(source, &items)?;
            for ((sid, old), (segment, offset, record_size)) in recs.into_iter().zip(new) {
                updates.push((
                    sid,
                    old,
                    RecordLoc {
                        segment,
                        offset,
                        payload_len: old.payload_len,
                        record_size,
                    },
                ));
            }
        }
        Ok(updates)
    }

    /// **Phase 4** — unlink the drained source segments and record the op for
    /// the status indicator. Runs under the persistence lock, after the index
    /// apply. Safe because the index no longer points at any source (Phase 3
    /// repointed the live records; the rest were dead).
    pub fn finish_maintenance(&mut self, plan: &MaintenancePlan) -> Result<()> {
        let targets = plan.op.targets();
        for &t in &targets {
            self.segments.drop_sealed(t)?;
        }
        // Drop metadata-location entries that pointed into the now-unlinked
        // segments. Live metadata for non-tombstoned streams was already
        // re-pointed forward by the resident re-emission (or never lived here, per
        // the reemit-floor invariant); any residue is a tombstoned stream's
        // metadata, which is genuinely gone with its segment. Leaving stale
        // entries would mis-attribute live bytes to a segment that no longer exists.
        self.metadata_locs
            .retain(|_, loc| !targets.contains(&loc.segment));
        let unix = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_maintenance = Some((plan.op.label(), unix));
        tracing::trace!(op = ?plan.op, "substrate maintenance op applied");
        Ok(())
    }

    /// Test/forced-op convenience: run all phases inline for a specific op under
    /// the caller's already-held substrate lock.
    pub fn apply_maintenance_op(
        &mut self,
        substrate: &mut Substrate,
        op: &MaintenanceOp,
    ) -> Result<()> {
        let resident = if self.need_resident_reemit(&op.targets()) {
            gather_resident_set(substrate)
        } else {
            Vec::new()
        };
        let (chunk_relocs, token_relocs, snapshot_relocs, singleton_relocs) =
            self.gather_relocations(substrate, &op.targets());
        let plan = MaintenancePlan {
            op: *op,
            resident,
            chunk_relocs,
            token_relocs,
            snapshot_relocs,
            singleton_relocs,
        };
        let result = self.execute_maintenance(&plan)?;
        result.apply_to_substrate(substrate);
        self.finish_maintenance(&plan)
    }

    /// Gather the relocation worklist for `targets` — the live read-back records
    /// (`Chunk` / `Tokens` / singletons) physically in those segments, with the
    /// same distill/tombstone filter as [`Self::segment_liveness`]. Read-only.
    fn gather_relocations(
        &self,
        substrate: &Substrate,
        targets: &[SegmentId],
    ) -> (
        Vec<(StreamId, u64, ChunkLoc)>,
        Vec<(StreamId, RecordLoc)>,
        Vec<(StreamId, RecordLoc)>,
        Vec<(RecordType, RecordLoc)>,
    ) {
        let tombstoned: HashSet<u64> = substrate
            .tombstoned_timelines()
            .iter()
            .map(|t| t.raw())
            .collect();
        let distilled: HashMap<u64, DistillMode> = substrate
            .distilled_timelines()
            .iter()
            .map(|(t, m)| (t.raw(), *m))
            .collect();
        let in_target = |seg: SegmentId| targets.contains(&seg);

        let mut chunks: Vec<(StreamId, u64, ChunkLoc)> = Vec::new();
        let mut tokens: Vec<(StreamId, RecordLoc)> = Vec::new();
        for (stream_id, entry) in substrate.all_streams() {
            let (is_tomb, distill) = classify(&entry.decl, &tombstoned, &distilled);
            // Same rule as the re-emit path: a tombstoned-but-DISTILLED timeline
            // is the provenance corpus, so it is relocated by its mode (the
            // per-mode gates below already withhold its chunks/tokens) rather
            // than abandoned. Only an undistilled tombstone is skipped outright.
            if is_tomb && distill.is_none() {
                continue;
            }
            // Orphan (no decl): its records are unreachable on reload, so DON'T
            // relocate them — segment_liveness excludes them from live weight, so
            // relocating (instead of dropping) would move the orphan KV forward
            // every compact while the dead-ratio never improves: perpetual churn
            // and the bloat is never reclaimed. Dropping the source segment
            // reclaims it. Mirrors the collect_live_records / segment_liveness gate.
            if entry.decl.is_none() {
                continue;
            }
            if distill.is_none() {
                for (&idx, loc) in &entry.chunks {
                    if in_target(loc.segment) {
                        chunks.push((stream_id, idx, *loc));
                    }
                }
            }
            if distill != Some(DistillMode::ProvenanceOnly) {
                if let Some(loc) = entry.tokens {
                    if in_target(loc.segment) {
                        tokens.push((stream_id, loc));
                    }
                }
            }
        }
        // Live snapshot tails physically inside a target segment. No
        // tombstone/distill gate: the substrate map only ever holds live
        // conversations' tails. Execute re-validates each entry against the
        // persistence-side `snapshot_locs` right before relocating, so a tail
        // superseded (or tombstoned) after this plan is skipped, not copied.
        let mut snapshots: Vec<(StreamId, RecordLoc)> = Vec::new();
        for (sid, loc) in substrate.recurrent_snapshot_entries() {
            if in_target(loc.segment) {
                snapshots.push((sid, loc));
            }
        }
        let mut singletons: Vec<(RecordType, RecordLoc)> = Vec::new();
        for (rt, loc) in [
            (RecordType::ModelSpec, self.manifest.model_spec),
            (RecordType::Template, self.manifest.template),
            (RecordType::Tokenizer, self.manifest.tokenizer),
        ] {
            if let Some(loc) = loc {
                if in_target(loc.segment) {
                    singletons.push((rt, loc));
                }
            }
        }
        (chunks, tokens, snapshots, singletons)
    }

    /// Live read-back record bytes per segment — `Chunk` / `Tokens` /
    /// singleton records the substrate index still references, with the same
    /// distill/tombstone filter [`super::compaction::collect_live_records`]
    /// applies. Resident records are excluded (they are re-emitted wholesale
    /// on every op), so a segment's `dead = total - live` slightly over-counts
    /// dead weight — safe, since the trigger only over-eagerly compacts and
    /// every op preserves the resident set.
    pub fn segment_liveness(&self, substrate: &Substrate) -> HashMap<SegmentId, u64> {
        let tombstoned: HashSet<u64> = substrate
            .tombstoned_timelines()
            .iter()
            .map(|t| t.raw())
            .collect();
        let distilled: HashMap<u64, DistillMode> = substrate
            .distilled_timelines()
            .iter()
            .map(|(t, m)| (t.raw(), *m))
            .collect();

        let mut live: HashMap<SegmentId, u64> = HashMap::new();
        for loc in [
            self.manifest.model_spec,
            self.manifest.template,
            self.manifest.tokenizer,
        ]
        .into_iter()
        .flatten()
        {
            *live.entry(loc.segment).or_default() += loc.record_size;
        }
        // Stream ids whose timeline is tombstoned — their records (chunks, tokens,
        // AND metadata) are all dead weight the maintenance reclaims, so exclude
        // their metadata from the live count below (else the tombstoned segment
        // never looks reclaimable).
        let mut tombstoned_streams: HashSet<u64> = HashSet::new();
        // Streams with a live `StreamDecl` — the reconstructible ones. A stream
        // WITHOUT a decl is an orphan (its decl was lost in a prior generation):
        // its turn can never be rebuilt on reload, so none of its records count
        // as live weight — mirroring the `collect_live_records` orphan gate — and
        // the segments holding only its records become reclaimable instead of
        // pinned forever.
        let mut live_streams: HashSet<u64> = HashSet::new();
        for (sid, entry) in substrate.all_streams() {
            let (is_tomb, distill) = classify(&entry.decl, &tombstoned, &distilled);
            if is_tomb {
                tombstoned_streams.insert(sid.0);
                continue;
            }
            if entry.decl.is_none() {
                continue;
            }
            live_streams.insert(sid.0);
            if distill.is_none() {
                for loc in entry.chunks.values() {
                    *live.entry(loc.segment).or_default() += loc.record_size;
                }
            }
            if distill != Some(DistillMode::ProvenanceOnly) {
                if let Some(loc) = entry.tokens {
                    *live.entry(loc.segment).or_default() += loc.record_size;
                }
            }
        }
        // Per-stream metadata records (`StreamDecl` / `ProjectionEvents` /
        // `WideQSig` / `Commit`) carry no location in the substrate index, so they
        // are counted from the persistence-side `metadata_locs` map instead —
        // last-writer-wins, so only the CURRENT copy of each is here; superseded
        // copies are absent and correctly read as dead. Without this the segment
        // holding a stream's live metadata reads as reclaimable and gets
        // re-emitted-forward every maintenance pass (the periodic-compaction churn).
        // Tombstoned streams and orphans (no live decl) are skipped so their
        // residual metadata never pins a segment that should be reclaimed.
        for ((_rt, stream_id), loc) in &self.metadata_locs {
            if tombstoned_streams.contains(stream_id) || !live_streams.contains(stream_id) {
                continue;
            }
            *live.entry(loc.segment).or_default() += loc.record_size;
        }
        // Recurrent-state snapshots: one live tail per conversation, tracked in
        // the substrate index (the `Tokens` shape — no `StreamDecl` of their
        // own, so the live_streams gate does not apply). The map holds only
        // live conversations' tails — a timeline tombstone removes its entry —
        // so every entry counts.
        for (_sid, loc) in substrate.recurrent_snapshot_entries() {
            *live.entry(loc.segment).or_default() += loc.record_size;
        }
        // Characters. Tracked persistence-side rather than on the substrate
        // index — the substrate holds no opinion about an NPC — and the map is
        // last-writer-wins, so only each character's current record is here and
        // every superseded one correctly reads as dead. Counting these is what
        // stops the segment holding the live cast from looking reclaimable:
        // without it the dead ratio is overstated, maintenance re-emits the
        // whole cast forward on every pass, and the log churns.
        for loc in self.npc_locs.values() {
            *live.entry(loc.segment).or_default() += loc.record_size;
        }
        live
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::ChunkPayload;
    use crate::persistence::streams::StreamId;
    use crate::persistence::{SubstratePersistence, SUBSTRATE_DIR};
    use crate::substrate::Substrate;
    use std::path::PathBuf;

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("maint_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn chunk_payload(seed: u32) -> ChunkPayload {
        ChunkPayload {
            offset: seed as u16,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
            k_pal: vec![seed as u8; 4],
            v_pal: vec![(seed + 1) as u8; 2],
            k_scale: vec![seed as f32, seed as f32 * 0.5],
            v_scale: vec![seed as f32 + 0.25],
            kv_bytes: (0..512u32).map(|i| ((i + seed * 13) % 256) as u8).collect(),
        }
    }

    /// The single-tail contract end to end: a newer snapshot supersedes the
    /// older one across reload, its dead bytes are credited, segment
    /// maintenance relocates exactly the tail, and a timeline tombstone kills
    /// the entry.
    #[test]
    fn snapshot_single_tail_survives_reload_and_relocation() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};

        let dir = tmp_dir("snapshot_tail");
        let timeline: u64 = 42;
        let sid = snapshot_stream_id(timeline);
        let payload = |turn: u32, fill: u8| {
            SnapshotPayload {
                timeline_id: timeline,
                turn_index: turn,
                schedule_hash: 0xA5A5,
                layers: vec![SnapshotLayer {
                    layer_index: 0,
                    n_v_heads: 1,
                    d_v: 4,
                    d_k: 4,
                    state: vec![fill; 64],
                    conv_channels: 2,
                    conv_tail_cols: 2,
                    conv_tail: vec![fill; 16],
                }],
            }
            .encode()
        };

        // Write two snapshots for the same conversation; the second wins.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let dead_before = sp.accounting.dead_bytes();
            let loc_a = sp.write_snapshot(sid, &payload(1, 0x11)).unwrap();
            substrate.apply_snapshot_loc(sid, loc_a);
            let loc_b = sp.write_snapshot(sid, &payload(2, 0x22)).unwrap();
            substrate.apply_snapshot_loc(sid, loc_b);
            assert!(
                sp.accounting.dead_bytes() > dead_before,
                "the superseded snapshot must be credited as dead bytes"
            );
            sp.commit().unwrap();
        }

        // Reload: the walker replays both records in append order; the index
        // must hold exactly the newer one, and its payload must decode to
        // turn 2.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail snapshot survives reload");
        let bytes = sp.read_record_payload(&tail).unwrap();
        let decoded = SnapshotPayload::decode(&bytes).unwrap();
        assert_eq!(decoded.turn_index, 2, "the newer snapshot is the tail");
        assert_eq!(decoded.layers[0].state[0], 0x22);

        // Relocation worklist contains exactly the tail (the dead copy is
        // invisible to the index).
        let (_, _, snapshots, _) = sp.gather_relocations(&substrate, &[SegmentId(1)]);
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].0, sid);

        // A timeline tombstone kills the entry.
        substrate.apply_tombstone(&crate::persistence::record::TombstonePayload {
            timeline_id: timeline,
            turn_index: None,
            reason: None,
        });
        assert!(substrate.recurrent_snapshot_loc(sid).is_none());
        let (_, _, snapshots, _) = sp.gather_relocations(&substrate, &[SegmentId(1)]);
        assert!(
            snapshots.is_empty(),
            "tombstoned snapshot must not relocate"
        );
    }

    fn sealed_log(dir: &std::path::Path, id: u64) -> PathBuf {
        dir.join(SUBSTRATE_DIR).join(format!("seg-{id:010}.log"))
    }

    /// Encoded snapshot payload for `timeline` at `turn` with a recognisable
    /// `fill` byte — the fixture shared by the snapshot-relocation tests.
    fn snapshot_payload(timeline: u64, turn: u32, fill: u8) -> Vec<u8> {
        use crate::persistence::record::{SnapshotLayer, SnapshotPayload};
        SnapshotPayload {
            timeline_id: timeline,
            turn_index: turn,
            schedule_hash: 0xA5A5,
            layers: vec![SnapshotLayer {
                layer_index: 0,
                n_v_heads: 1,
                d_v: 4,
                d_k: 4,
                state: vec![fill; 64],
                conv_channels: 2,
                conv_tail_cols: 2,
                conv_tail: vec![fill; 16],
            }],
        }
        .encode()
    }

    /// Relocating a LIVE snapshot credits exactly the OLD copy's on-disk
    /// bytes as dead (the relocated copy supersedes it in the accounting
    /// map), repoints both indexes at the new location, and the relocated
    /// tail decodes correctly across a reload.
    #[test]
    fn snapshot_relocation_credits_exact_dead_bytes() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_acct");
        let timeline: u64 = 91;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        // Two tails in seg 1: turn 1 is superseded (the segment's dead
        // weight), turn 2 is the live tail the compact must carry forward.
        let loc_a = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 1, 0x11))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a);
        let loc_b = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 2, 0x22))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b);
        sp.commit().unwrap();
        sp.seal_active().unwrap(); // seg 1 sealed; active is seg 2
        assert_eq!(
            sp.accounting.dead_bytes(),
            loc_a.record_size,
            "the superseded turn-1 copy is the only dead weight so far"
        );

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("seg 1 carries dead weight, force-compact qualifies");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));

        let result = sp.execute_maintenance(&plan).unwrap();
        assert_eq!(result.snapshot_updates.len(), 1);
        let (usid, old, new) = result.snapshot_updates[0];
        assert_eq!(usid, sid);
        assert_eq!(old, loc_b);
        assert_eq!(new.segment, SegmentId(2), "relocated into the active");
        assert_eq!(
            new.record_size, loc_b.record_size,
            "verbatim copy, same size"
        );
        assert_eq!(
            sp.accounting.dead_bytes(),
            loc_a.record_size + loc_b.record_size,
            "relocation credits exactly the OLD live copy's bytes as dead"
        );

        result.apply_to_substrate(&mut substrate);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(new));
        sp.finish_maintenance(&plan).unwrap();
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&new).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        drop(sp);

        // Reload: the relocated record is the tail the walk installs.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail survives reload");
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(decoded.turn_index, 2);
        assert_eq!(decoded.layers[0].state[0], 0x22);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The plan→execute supersession race: after a maintenance plan targets a
    /// stream's snapshot tail, the seal thread's writer appends a NEWER tail
    /// for the same stream. Execute must skip the planned (now stale) copy
    /// entirely — no append, no accounting movement, no index update — and a
    /// reload must still see the newer record as the tail.
    #[test]
    fn superseded_snapshot_is_not_relocated() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::SnapshotPayload;

        let dir = tmp_dir("snap_reloc_race");
        let timeline: u64 = 92;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let loc_a = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 1, 0x11))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a);
        let loc_b = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 2, 0x22))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b);
        sp.commit().unwrap();
        sp.seal_active().unwrap(); // seg 1 sealed (turn-1 dead, turn-2 the tail)

        // Phase 1 plans the relocation of the turn-2 tail out of seg 1.
        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");
        assert_eq!(plan.op(), MaintenanceOp::Compact(SegmentId(1)));

        // RACE: before execute, the writer appends a newer turn-3 tail into
        // the active segment and installs it in the substrate index.
        let loc_c = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 3, 0x33))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_c);
        sp.commit().unwrap();
        assert_eq!(
            loc_c.segment,
            SegmentId(2),
            "the newer tail lands in the active"
        );
        let dead_before = sp.accounting.dead_bytes();
        assert_eq!(
            dead_before,
            loc_a.record_size + loc_b.record_size,
            "turn-1 and turn-2 are both dead the moment turn-3 lands"
        );
        let offset_before = sp.write_offset();

        // Phase 2: the planned copy is stale — it must NOT be appended.
        let result = sp.execute_maintenance(&plan).unwrap();
        assert!(
            result.snapshot_updates.is_empty(),
            "a superseded snapshot produces no update"
        );
        assert_eq!(
            sp.write_offset(),
            offset_before,
            "no bytes were appended for the stale copy"
        );
        assert_eq!(
            sp.accounting.dead_bytes(),
            dead_before,
            "the live turn-3 record was not credited dead"
        );

        // Phase 3 + 4: the index keeps the newer tail; seg 1 still drops
        // (the stale copy is dead there).
        result.apply_to_substrate(&mut substrate);
        assert_eq!(substrate.recurrent_snapshot_loc(sid), Some(loc_c));
        sp.finish_maintenance(&plan).unwrap();
        assert!(
            !sealed_log(&dir, 1).exists(),
            "seg 1 dropped; the stale copy died with it"
        );
        drop(sp);

        // Reload: the walk replays only the surviving turn-3 record — the
        // tail is the newer state, not a rolled-back turn 2.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let tail = substrate
            .recurrent_snapshot_loc(sid)
            .expect("tail survives reload");
        assert_eq!(tail, loc_c);
        assert_eq!(
            sp.snapshot_locs.get(&sid.0),
            Some(&loc_c),
            "the persistence-side live-tail map is rebuilt by the walk"
        );
        let decoded = SnapshotPayload::decode(&sp.read_record_payload(&tail).unwrap()).unwrap();
        assert_eq!(
            decoded.turn_index, 3,
            "reload sees the NEWER snapshot as the tail"
        );
        assert_eq!(decoded.layers[0].state[0], 0x33);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A timeline tombstone landing between plan and execute empties the
    /// stream's live-tail entry, so the planned relocation is skipped and the
    /// tombstoned snapshot dies with its segment instead of being appended
    /// after the tombstone record (which a reload would resurrect as a tail).
    #[test]
    fn tombstoned_snapshot_is_not_relocated() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::TombstonePayload;

        let dir = tmp_dir("snap_reloc_tomb");
        let timeline: u64 = 93;
        let sid = snapshot_stream_id(timeline);

        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let loc_a = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 1, 0x11))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_a);
        let loc_b = sp
            .write_snapshot(sid, &snapshot_payload(timeline, 2, 0x22))
            .unwrap();
        substrate.apply_snapshot_loc(sid, loc_b);
        sp.commit().unwrap();
        sp.seal_active().unwrap();

        let plan = sp
            .plan_maintenance(&substrate, true)
            .unwrap()
            .expect("compact planned");

        // RACE: the timeline is tombstoned between plan and execute — the
        // durable record lands in the active segment, and both the substrate
        // index and the persistence-side map drop their entries.
        sp.write_tombstone(timeline, None).unwrap();
        substrate.apply_tombstone(&TombstonePayload {
            timeline_id: timeline,
            turn_index: None,
            reason: None,
        });
        sp.commit().unwrap();
        let dead_before = sp.accounting.dead_bytes();
        let offset_before = sp.write_offset();

        let result = sp.execute_maintenance(&plan).unwrap();
        assert!(
            result.snapshot_updates.is_empty(),
            "a tombstoned snapshot produces no update"
        );
        assert_eq!(sp.write_offset(), offset_before);
        assert_eq!(sp.accounting.dead_bytes(), dead_before);
        result.apply_to_substrate(&mut substrate);
        sp.finish_maintenance(&plan).unwrap();
        assert!(!sealed_log(&dir, 1).exists());
        drop(sp);

        // Reload: the tombstone record survives in the active segment and no
        // snapshot record outlived seg 1 — nothing resurrects a tail.
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate.recurrent_snapshot_loc(sid).is_none(),
            "no tail resurrected after the tombstone"
        );
        assert!(!sp.snapshot_locs.contains_key(&sid.0));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A chunk payload with a caller-sized `kv_bytes` (constant fill — cheap to
    /// build) for the perf benchmark.
    fn filled_payload(seed: u32, kv_len: usize) -> ChunkPayload {
        let mut p = chunk_payload(seed);
        p.kv_bytes = vec![(seed & 0xff) as u8; kv_len];
        p
    }

    fn mb_per_s(bytes: u64, dur: std::time::Duration) -> f64 {
        let secs = dur.as_secs_f64().max(1e-9);
        (bytes as f64 / (1024.0 * 1024.0)) / secs
    }

    /// Performance benchmark for the incremental **compact** — the expensive
    /// maintenance op (relocation I/O). Builds a realistically-sized sealed
    /// segment (half live, half superseded), then times each phase of relocating
    /// its live half into the active. Warm-cache numbers, so they isolate the
    /// *code* cost (per-record syscalls + decode/re-encode + append), which is
    /// what "as fast as possible" targets — cold-disk throughput is disk-bound.
    ///
    /// Run: `cargo test -p candle-conversation --release --lib
    /// maintenance_perf_compact -- --ignored --nocapture`
    #[test]
    #[ignore = "perf benchmark — run explicitly with --ignored --nocapture (prefer --release)"]
    fn maintenance_perf_compact() {
        use std::time::Instant;

        // Scale knobs — ~16 KB payloads (representative KV chunk), half dead.
        const LIVE: u64 = 8_000;
        const DEAD: u64 = 8_000;
        const KV_BYTES: usize = 16 * 1024;

        let dir = tmp_dir("perf_compact");
        let sid = StreamId(1);

        let t_build = Instant::now();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // Seg 1: LIVE + DEAD chunks.
            for i in 0..(LIVE + DEAD) {
                sp.write_chunk(sid, i, 32, 4, None, &filled_payload(i as u32, KV_BYTES))
                    .unwrap();
            }
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed
                                       // Active (seg 2) supersedes the DEAD half.
            for i in LIVE..(LIVE + DEAD) {
                sp.write_chunk(sid, i, 32, 4, None, &filled_payload(i as u32 + 1, KV_BYTES))
                    .unwrap();
            }
            sp.commit().unwrap();
        }
        let seg1_total = std::fs::metadata(sealed_log(&dir, 1)).unwrap().len();
        eprintln!(
            "\n── compact benchmark ─────────────────────────────────\nbuild: {:?}  |  seg-1 on disk: {} MB ({} chunks, {}/{} live)",
            t_build.elapsed(),
            seg1_total >> 20,
            LIVE + DEAD,
            LIVE,
            LIVE + DEAD,
        );

        // Reopen to rebuild the segmented index.
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let relocated = seg1_total / 2; // ≈ the live half

        // Phase 1 — plan (snapshot).
        let t = Instant::now();
        let (chunk_relocs, _, _, _) = sp.gather_relocations(&substrate, &[SegmentId(1)]);
        let plan_el = t.elapsed();

        // Reference: the naive per-record read path (`read_chunk` decodes +
        // CRC-verifies each record). The compact below uses coalesced reads +
        // verbatim staging instead, so it should beat even this read-only loop.
        let t = Instant::now();
        let mut read_bytes = 0u64;
        for &(_, idx, _) in &chunk_relocs {
            let p = sp.read_chunk(&substrate, sid, idx).unwrap();
            read_bytes += p.kv_bytes.len() as u64;
        }
        let read_el = t.elapsed();

        // Full compact (execute I/O + index apply + drop).
        let t = Instant::now();
        sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
            .unwrap();
        let compact_el = t.elapsed();

        eprintln!(
            "plan (gather {} relocs): {plan_el:?}\n\
             per-record read reference ({} live chunks, {} MB): {read_el:?}  →  {:.0} MB/s\n\
             full compact (coalesced read + verbatim stage + fsync + drop, ~{} MB): {compact_el:?}  →  {:.0} MB/s",
            chunk_relocs.len(),
            chunk_relocs.len(),
            read_bytes >> 20,
            mb_per_s(read_bytes, read_el),
            relocated >> 20,
            mb_per_s(relocated, compact_el),
        );
        assert!(!sealed_log(&dir, 1).exists(), "seg 1 compacted away");
        // Sanity: the relocated live chunks read back correctly.
        assert_eq!(
            sp.read_chunk(&substrate, sid, 0).unwrap().kv_bytes.len(),
            KV_BYTES
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A sealed segment whose only read-back record is superseded by a later
    /// segment is fully dead — maintenance drops it (O(1) unlink), and the live
    /// version still reads correctly and survives a reload.
    #[test]
    fn maintenance_drops_a_fully_superseded_segment() {
        let dir = tmp_dir("drop");
        let sid = StreamId(101);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed, holds chunk-0 v1
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(2))
                .unwrap(); // seg 2 supersedes
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(2),
                "the higher-id (newer) copy wins"
            );
            assert!(sealed_log(&dir, 1).exists());
            assert!(
                sp.run_maintenance(&mut substrate).unwrap().is_some(),
                "seg 1 is fully dead"
            );
            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was dropped");
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(2));
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(2));
            assert_eq!(substrate.live_chunk_count(), 1);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stream's CURRENT per-stream metadata (here a `WideQSig`) counts as LIVE
    /// weight in `segment_liveness`, so a segment holding only live metadata is
    /// not seen as reclaimable dead — the fix that stops such a segment from being
    /// re-emitted-forward on every maintenance pass. Also survives a reload (the
    /// location map is rebuilt from the walk).
    #[test]
    fn current_metadata_counts_as_live_weight() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("meta_live");
        // A LIVE stream: it needs a StreamDecl to be reconstructible, otherwise
        // it is an orphan whose residual metadata is (correctly) dead weight.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 777,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            substrate.apply_stream_decl(sid, decl.clone());
            sp.append_wide_q_sigs(sid, b"wide-q-signature-bytes")
                .unwrap();
            sp.commit().unwrap();
            let live: u64 = sp.segment_liveness(&substrate).values().sum();
            assert!(
                live > 0,
                "a stream's current WideQSig metadata must count as live weight"
            );
            // Superseding it leaves exactly one live copy (LWW) — liveness stays
            // bounded, it does not accumulate per re-write.
            sp.append_wide_q_sigs(sid, b"newer-wide-q-signature")
                .unwrap();
            sp.commit().unwrap();
            assert_eq!(
                sp.metadata_locs
                    .keys()
                    .filter(|(rt, s)| *rt == RecordType::WideQSig && *s == sid.0)
                    .count(),
                1,
                "last-writer-wins: one current metadata location per (type, stream)"
            );
        }
        // Rebuilt from the walk on reload.
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let live: u64 = sp.segment_liveness(&substrate).values().sum();
            assert!(live > 0, "metadata liveness must survive a reload");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// An orphan stream (KV chunks on disk, NO `StreamDecl`) is neither relocated
    /// nor re-emitted by the incremental maintenance path — it is left to be
    /// reclaimed when its source segment drops. Regression for the churn bug:
    /// `segment_liveness` excludes orphan chunks from live weight, so if
    /// `gather_relocations` still relocated them, a mixed segment would be
    /// compacted forever (dead-ratio never improving) and the orphan KV would
    /// migrate segment-to-segment, never reclaimed.
    #[test]
    fn orphan_stream_is_not_relocated_or_re_emitted() {
        let dir = tmp_dir("orphan_reloc");
        let sid = StreamId(4242); // no decl declared → orphan
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 holds the orphan chunk
        }
        // Re-walk so the substrate sees the chunk with no owning decl.
        let mut substrate = Substrate::new();
        let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate
                .stream_of(sid)
                .is_some_and(|s| s.decl.is_none() && !s.chunks.is_empty()),
            "the stream is an orphan: chunk present, no decl",
        );
        // gather_relocations must NOT relocate the orphan's chunk.
        let (chunks, _tokens, _snapshots, _singletons) =
            sp.gather_relocations(&substrate, &[SegmentId(1)]);
        assert!(
            chunks.is_empty(),
            "orphan chunks must not be relocated (would perpetuate the bloat + churn)",
        );
        // gather_resident_set must not re-emit any record for the orphan stream.
        let residents = gather_resident_set(&substrate);
        assert!(
            residents.iter().all(|r| r.stream_id != sid.0),
            "orphan metadata must not be re-emitted",
        );
        // And it contributes zero live weight, so its segment is reclaimable.
        let live: u64 = sp.segment_liveness(&substrate).values().sum();
        assert_eq!(live, 0, "an orphan chunk is not live weight");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The re-emit skip guard (`need_resident_reemit`) is what breaks the
    /// compaction churn: once the resident (metadata) set is durable at `floor`,
    /// ops targeting only older segments skip the wholesale re-emission — so they
    /// stop generating fresh uncounted-dead metadata that would trigger the next
    /// compaction — while any target at/after `floor` (which may hold metadata
    /// written after the snapshot) forces a re-emit.
    #[test]
    fn need_resident_reemit_skips_old_only_targets() {
        let dir = tmp_dir("reemit_floor");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();

        // No durable snapshot yet → always re-emit.
        assert!(sp.need_resident_reemit(&[SegmentId(1)]));

        // A durable snapshot at floor = seg 10 duplicated every then-existing
        // metadata record into segments >= 10.
        sp.resident_reemit_floor = Some(SegmentId(10));
        // Targets strictly older than the floor are all duplicated at >= floor → skip.
        assert!(!sp.need_resident_reemit(&[SegmentId(5)]));
        assert!(!sp.need_resident_reemit(&[SegmentId(9), SegmentId(1)]));
        // A target at/after the floor may hold post-snapshot metadata → re-emit.
        assert!(sp.need_resident_reemit(&[SegmentId(10)]));
        assert!(sp.need_resident_reemit(&[SegmentId(12)]));
        // Any target at/after the floor in a mixed set forces a re-emit.
        assert!(sp.need_resident_reemit(&[SegmentId(5), SegmentId(11)]));

        // Exact guard: a target holding the CURRENT copy of a per-stream metadata
        // record (here a StreamDecl in seg 5, BELOW the floor) must force a
        // re-emit — the floor heuristic alone would wrongly skip it and the drop
        // would delete the live turn's decl (the silent-loss bug).
        sp.metadata_locs.insert(
            (RecordType::StreamDecl, 42),
            RecordLoc {
                segment: SegmentId(5),
                offset: 0,
                payload_len: 0,
                record_size: 4096,
            },
        );
        assert!(
            sp.need_resident_reemit(&[SegmentId(5)]),
            "a target holding a current StreamDecl must force a re-emit even below the floor",
        );
        // A below-floor target with no current metadata still skips (no churn).
        assert!(!sp.need_resident_reemit(&[SegmentId(4)]));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A partially-dead sealed segment is compacted: its live record is
    /// relocated into the active (the index repointed), then it is dropped.
    /// Both chunks read correctly afterwards and survive a reload.
    #[test]
    fn maintenance_compacts_and_relocates_a_live_record() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact");
        // The stream needs a decl to be live — a decl-less stream is an orphan
        // whose chunks are (correctly) reclaimed rather than relocated.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 202,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 2,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(10))
                .unwrap(); // seg 1, live
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(11))
                .unwrap(); // seg 1, will be superseded
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 1, 32, 4, None, &chunk_payload(12))
                .unwrap(); // seg 2 supersedes chunk-1
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(1)
            );
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&1].segment,
                SegmentId(2)
            );

            sp.apply_maintenance_op(&mut substrate, &MaintenanceOp::Compact(SegmentId(1)))
                .unwrap();

            assert!(!sealed_log(&dir, 1).exists(), "seg 1 was compacted away");
            assert_eq!(
                substrate.stream_of(sid).unwrap().chunks[&0].segment,
                SegmentId(2),
                "chunk-0 was relocated into the active segment"
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 0).unwrap(),
                chunk_payload(10)
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 1).unwrap(),
                chunk_payload(12)
            );
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(
                sp.read_chunk(&substrate, sid, 0).unwrap(),
                chunk_payload(10)
            );
            assert_eq!(
                sp.read_chunk(&substrate, sid, 1).unwrap(),
                chunk_payload(12)
            );
            assert_eq!(substrate.live_chunk_count(), 2);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tombstoned timeline whose decl + chunk live only in a dropped segment
    /// must stay tombstoned after the drop — the re-emitted `Tombstone` marker
    /// keeps it dead so a reload can't resurrect it.
    #[test]
    fn tombstone_marker_survives_a_segment_drop() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        use crate::projection::TimelineId;

        let dir = tmp_dir("tomb");
        let tid = 777u64;
        let turn = StreamDecl::Turn(TurnDecl {
            timeline_id: tid,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 16,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = turn.stream_id();
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&turn).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(1))
                .unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1: the tombstoned-to-be turn's decl + chunk
            sp.write_tombstone(tid, None).unwrap(); // marker lands in seg 2 (active)
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let tl = TimelineId::from_raw(tid).unwrap();
            assert!(substrate.is_tombstoned(tl));
            // seg 1 holds only the tombstoned turn's records → fully dead → drop.
            assert!(sp.run_maintenance(&mut substrate).unwrap().is_some());
            assert!(!sealed_log(&dir, 1).exists());
        }
        {
            let mut substrate = Substrate::new();
            let _sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                substrate.is_tombstoned(TimelineId::from_raw(tid).unwrap()),
                "the tombstone survived the drop that removed the timeline's records"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn stat(id: u64, total: u64, live: u64, age: u64) -> SegmentStat {
        SegmentStat {
            id: SegmentId(id),
            total_bytes: total,
            live_bytes: live,
            age_secs: age,
        }
    }

    #[test]
    fn drop_wins_over_everything() {
        // A fully-dead segment is dropped even if a compact candidate exists.
        let stats = [stat(1, 1000, 0, 0), stat(2, 1000, 100, 999)];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Drop(SegmentId(1)))
        );
    }

    #[test]
    fn compact_needs_dead_ratio_and_settle() {
        // Dead enough but not settled → no op.
        let young = [stat(1, 1000, 100, 10)];
        assert_eq!(pick_maintenance_op(&young, false), None);
        // Settled but not dead enough (5% < 10%) → no op.
        let clean = [stat(1, 1000, 950, 999)];
        assert_eq!(pick_maintenance_op(&clean, false), None);
        // Settled and 90% dead → compact.
        let dead = [stat(1, 1000, 100, 999)];
        assert_eq!(
            pick_maintenance_op(&dead, false),
            Some(MaintenanceOp::Compact(SegmentId(1)))
        );
    }

    #[test]
    fn force_waives_age_and_ratio_gates() {
        // Young (age 10 < 60) and only 5% dead — no op on a normal pass.
        let s = [stat(1, 1000, 950, 10)];
        assert_eq!(pick_maintenance_op(&s, false), None);
        // A forced pass compacts it: any dead weight qualifies, age ignored.
        assert_eq!(
            pick_maintenance_op(&s, true),
            Some(MaintenanceOp::Compact(SegmentId(1)))
        );
        // A fully-live segment is never compacted, even forced (nothing to shed).
        let live = [stat(1, 1000, 1000, 0)];
        assert_eq!(pick_maintenance_op(&live, true), None);
    }

    #[test]
    fn compact_prefers_the_deadest_settled_segment() {
        let stats = [
            stat(1, 1000, 500, 999), // 50% dead
            stat(2, 1000, 100, 999), // 90% dead — deadest
            stat(3, 1000, 800, 999), // 20% dead
        ];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Compact(SegmentId(2)))
        );
    }

    #[test]
    fn combine_pairs_adjacent_small_segments() {
        // No drop, no compact candidate (all clean + settled), two small
        // adjacent segments → combine.
        let stats = [stat(1, 1000, 1000, 999), stat(2, 1000, 1000, 999)];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Combine(SegmentId(1), SegmentId(2)))
        );
    }

    #[test]
    fn no_op_when_nothing_qualifies() {
        // One big fully-live settled segment: nothing to drop/compact, and a
        // single segment has no adjacent pair to combine.
        let big = [stat(
            1,
            COMBINE_SEGMENT_BYTES + 1,
            COMBINE_SEGMENT_BYTES + 1,
            999,
        )];
        assert_eq!(pick_maintenance_op(&big, false), None);
    }

    #[test]
    fn op_targets_enumerate_segments() {
        assert_eq!(
            MaintenanceOp::Drop(SegmentId(3)).targets(),
            vec![SegmentId(3)]
        );
        assert_eq!(
            MaintenanceOp::Combine(SegmentId(1), SegmentId(2)).targets(),
            vec![SegmentId(1), SegmentId(2)]
        );
    }
}

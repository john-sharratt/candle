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
//! Because maintenance runs only on the single persistence thread, no concurrent
//! writer can append between phases; only reads interleave, and they always see
//! a consistent index (the sources stay live until step 4).
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
        if qualifies && best.map_or(true, |(_, r)| ratio > r) {
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
        if is_tomb {
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
        if tombstoned.contains(&tl) {
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
    for (&tl, &mode) in &distilled {
        if tombstoned.contains(&tl) {
            continue;
        }
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
        let resident = gather_resident_set(substrate);
        let (chunk_relocs, token_relocs, singleton_relocs) =
            self.gather_relocations(substrate, &op.targets());
        Ok(Some(MaintenancePlan {
            op,
            resident,
            chunk_relocs,
            token_relocs,
            singleton_relocs,
        }))
    }

    /// **Phase 2** — the slow I/O: append the resident set, then relocate each
    /// target segment's live records into the active, then commit. Holds only
    /// the persistence lock; the caller does **not** hold the substrate lock
    /// here, so decode's in-RAM projection reads/writes proceed. The relocated
    /// copies are durable (fsynced) before any source is unlinked. Returns the
    /// new locations for [`MaintenanceResult::apply_to_substrate`].
    pub fn execute_maintenance(&mut self, plan: &MaintenancePlan) -> Result<MaintenanceResult> {
        // Resident set — re-emitted from in-RAM state (not read from disk), so
        // the normal encoding append.
        for r in &plan.resident {
            self.append_record(r.rt, 0, r.stream_id, r.chunk_index, 0, &r.payload)?;
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
        let mut token_updates = Vec::with_capacity(plan.token_relocs.len());
        let mut tok_by_seg: BTreeMap<SegmentId, Vec<(StreamId, RecordLoc)>> = BTreeMap::new();
        for &(sid, old) in &plan.token_relocs {
            tok_by_seg.entry(old.segment).or_default().push((sid, old));
        }
        for (source, recs) in tok_by_seg {
            let items: Vec<RawReloc> = recs
                .iter()
                .map(|&(sid, old)| RawReloc {
                    offset: old.offset,
                    record_size: old.record_size,
                    header: RecordHeader {
                        record_type: RecordType::Tokens,
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
                token_updates.push((
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

        // Singletons — few (≤3); the encoding append repoints them via
        // `manifest.ingest`.
        for &(rt, old) in &plan.singleton_relocs {
            let rec = self
                .segments
                .read_record_at(old.segment, old.offset, old.record_size)?;
            self.append_record(rt, 0, 0, 0, 0, &rec.payload)?;
        }

        // Durability barrier: relocated copies are fsynced before any source is
        // unlinked (in `finish_maintenance`).
        self.commit()?;
        Ok(MaintenanceResult {
            chunk_updates,
            token_updates,
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

    /// **Phase 4** — unlink the drained source segments and record the op for
    /// the status indicator. Runs under the persistence lock, after the index
    /// apply. Safe because the index no longer points at any source (Phase 3
    /// repointed the live records; the rest were dead).
    pub fn finish_maintenance(&mut self, plan: &MaintenancePlan) -> Result<()> {
        for t in plan.op.targets() {
            self.segments.drop_sealed(t)?;
        }
        let unix = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_maintenance = Some((plan.op.label(), unix));
        tracing::debug!(op = ?plan.op, "substrate maintenance op applied");
        Ok(())
    }

    /// Test/forced-op convenience: run all phases inline for a specific op under
    /// the caller's already-held substrate lock.
    pub fn apply_maintenance_op(
        &mut self,
        substrate: &mut Substrate,
        op: &MaintenanceOp,
    ) -> Result<()> {
        let resident = gather_resident_set(substrate);
        let (chunk_relocs, token_relocs, singleton_relocs) =
            self.gather_relocations(substrate, &op.targets());
        let plan = MaintenancePlan {
            op: *op,
            resident,
            chunk_relocs,
            token_relocs,
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
            if is_tomb {
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
        (chunks, tokens, singletons)
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
        for (_sid, entry) in substrate.all_streams() {
            let (is_tomb, distill) = classify(&entry.decl, &tombstoned, &distilled);
            if is_tomb {
                continue;
            }
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

    fn sealed_log(dir: &std::path::Path, id: u64) -> PathBuf {
        dir.join(SUBSTRATE_DIR).join(format!("seg-{id:010}.log"))
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
                sp.write_chunk(sid, i, 32, 4, &filled_payload(i as u32, KV_BYTES))
                    .unwrap();
            }
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed
                                       // Active (seg 2) supersedes the DEAD half.
            for i in LIVE..(LIVE + DEAD) {
                sp.write_chunk(sid, i, 32, 4, &filled_payload(i as u32 + 1, KV_BYTES))
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
        let (chunk_relocs, _, _) = sp.gather_relocations(&substrate, &[SegmentId(1)]);
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
            sp.write_chunk(sid, 0, 32, 4, &chunk_payload(1)).unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1 sealed, holds chunk-0 v1
            sp.write_chunk(sid, 0, 32, 4, &chunk_payload(2)).unwrap(); // seg 2 supersedes
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

    /// A partially-dead sealed segment is compacted: its live record is
    /// relocated into the active (the index repointed), then it is dropped.
    /// Both chunks read correctly afterwards and survive a reload.
    #[test]
    fn maintenance_compacts_and_relocates_a_live_record() {
        let dir = tmp_dir("compact");
        let sid = StreamId(202);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, &chunk_payload(10)).unwrap(); // seg 1, live
            sp.write_chunk(sid, 1, 32, 4, &chunk_payload(11)).unwrap(); // seg 1, will be superseded
            sp.commit().unwrap();
            sp.seal_active().unwrap();
            sp.write_chunk(sid, 1, 32, 4, &chunk_payload(12)).unwrap(); // seg 2 supersedes chunk-1
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
            sp.write_chunk(sid, 0, 32, 4, &chunk_payload(1)).unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap(); // seg 1: the tombstoned-to-be turn's decl + chunk
            sp.write_tombstone(tid).unwrap(); // marker lands in seg 2 (active)
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

//! Store compaction — reclaim dead weight by rewriting only the **live**
//! records into a fresh segment (`docs/archived/segmented_substrate_log.md`).
//!
//! An append-only store only grows: every superseded partial-tail snapshot,
//! every stale `ModelKey` / `Template`, every chunk of a deleted stream is
//! dead weight the skip-load walk still steps over. Compaction keeps only the
//! live records — the winners under last-writer-wins that the manifest and the
//! substrate index already resolve.
//!
//! [`collect_live_records`] plans the live set in dependency order (`ModelKey`,
//! `Template`, then per stream its `StreamDecl`, `Chunk`s, `Tokens`, `Commit`)
//! **without any disk reads** — read-back records (`Chunk` / `Tokens` /
//! singletons) carry their source location, resident metadata carries its
//! freshly-encoded bytes. [`write_compacted_log`] then reads the read-back
//! records off their source segments in **coalesced stripes** and stages them
//! **verbatim** (no decode / CRC re-verify / re-encode — the same fast path the
//! incremental maintenance relocation uses), interleaving a fresh `HeaderIndex`
//! chain. The orchestration that adopts the result as the sole segment and drops
//! the rest is [`SubstratePersistence::compact`] (via
//! [`super::segmented_log::SegmentedLog::adopt_compacted`]).
//!
//! [`SubstratePersistence::compact`]: super::SubstratePersistence::compact
//! [`SubstratePersistence::should_compact`]: super::SubstratePersistence::should_compact

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::adoption::Adoption;
use super::blob::BlobIndex;
use super::content_hash::snapshot_stream_id;
use super::deleted::DeletedPayload;
use super::header_index::{encode_index_payload, IndexEntry, INDEX_FLUSH_ENTRIES};
use super::identity::MarkerIds;
use super::log_file::LogFile;
use super::manifest::{encode_conv_state_payload, Manifest, RecordLoc};
use super::rebase::RebaseMap;
use super::record::{
    decode_header, decode_record, encode_record, encode_record_written, DebugIdPayload,
    DistillMode, DistillPayload, RecordHeader, RecordType, SectionTombstonePayload,
    TombstonePayload, TurnCouplingPayload,
};
use super::replica::ReplicaPayload;
use super::segment::{SegmentId, FIRST_SEGMENT};
use super::stamped_locs::StampedLocs;
use super::streams::{StreamDecl, StreamId};
use super::stripes::{stripes, MAX_STRIPE_BYTES};
use super::tombstone_expiry::TombstoneExpiry;
use super::vfs::{carried, VfsIndex};
use super::Result;
use crate::projection::TimelineId;
use crate::substrate::{IndexPage, Substrate};

/// Bytes staged into the compacted log before an incremental flush to disk —
/// bounds the write buffer so a whole-store rewrite never holds the entire new
/// log in RAM at once.
const COMPACT_FLUSH_BYTES: usize = 256 * 1024 * 1024;

/// One record destined for the compacted log. `Raw` records (`Chunk` / `Tokens`
/// / the singletons) are staged **verbatim** from their source segment — no
/// decode, no CRC re-verify, no re-encode — via coalesced stripe reads; their
/// header is synthesized from the substrate/manifest index (never read from
/// disk) for the header-index digest + singleton manifest. `Synth` records are
/// the resident metadata, encoded fresh from in-RAM state — or, for the blobs
/// the substrate already holds shared (index pages, wide-Q signatures,
/// projection events), the substrate's own bytes. Those run to a megabyte a
/// turn and the whole live list exists before the first byte is written, so a
/// copy here is a second image of every live turn's metadata held for the
/// length of the rewrite.
pub enum CompactItem {
    Raw {
        header: RecordHeader,
        segment: SegmentId,
        offset: u64,
        record_size: u64,
    },
    Synth {
        header: RecordHeader,
        payload: Arc<[u8]>,
    },
}

impl CompactItem {
    fn raw(header: RecordHeader, segment: SegmentId, offset: u64, record_size: u64) -> CompactItem {
        CompactItem::Raw {
            header,
            segment,
            offset,
            record_size,
        }
    }

    fn synth(header: RecordHeader, payload: impl Into<Arc<[u8]>>) -> CompactItem {
        CompactItem::Synth {
            header,
            payload: payload.into(),
        }
    }

    /// The record's header — synthesized for `Raw`, authored for `Synth`.
    pub fn header(&self) -> &RecordHeader {
        match self {
            CompactItem::Raw { header, .. } | CompactItem::Synth { header, .. } => header,
        }
    }

    /// A `Synth` item's payload, or `None` for a `Raw` (read-back) item.
    #[cfg(test)]
    pub fn synth_payload(&self) -> Option<&[u8]> {
        match self {
            CompactItem::Synth { payload, .. } => Some(&payload[..]),
            CompactItem::Raw { .. } => None,
        }
    }
}

/// Header for a singleton (`ModelKey` / `Template` / `Tokenizer`) staged
/// verbatim — `payload_len` from its manifest location, every other field zero.
fn singleton_header(rt: RecordType, payload_len: u64) -> RecordHeader {
    RecordHeader {
        record_type: rt,
        format: 0,
        payload_len,
        crc: 0,
        stream_id: 0,
        chunk_index: 0,
        token_count: 0,
        ..RecordHeader::NONE
    }
}

/// `(offset, record_size)` of a `Raw` item — used by tests to read a chosen
/// source location back and confirm which record was selected.
#[cfg(test)]
fn raw_loc(it: &CompactItem) -> (u64, u64) {
    match it {
        CompactItem::Raw {
            offset,
            record_size,
            ..
        } => (*offset, *record_size),
        CompactItem::Synth { .. } => unreachable!("raw_loc on a Synth item"),
    }
}

/// How many prompt branch checkpoints survive a compaction.
///
/// A bound rather than a policy: they are a cache, they are ~63 MiB each, and
/// nothing else reclaims them. Sized so a workspace that switches between a
/// handful of selector assignments never loses one it is actually using, while
/// a schema edited many times cannot grow the log without limit.
pub(super) const MAX_BRANCH_CHECKPOINTS: usize = 8;

/// The branch checkpoints a compaction carries forward — the newest
/// [`MAX_BRANCH_CHECKPOINTS`] by write position, and the ids of those dropped.
///
/// "Newest by write position" is the closest thing to recently-used available
/// from the index, which holds locations and no timestamps. A checkpoint that is
/// re-read but not rewritten does not refresh, so an actively-used branch can in
/// principle age out — it costs one prefill and is written back, which is the
/// whole reason a cache can be evicted on a rough signal.
pub(super) fn partition_branch_checkpoints(
    substrate: &Substrate,
) -> (Vec<(StreamId, RecordLoc)>, Vec<StreamId>) {
    let mut all: Vec<(StreamId, RecordLoc)> = substrate.branch_checkpoint_entries().collect();
    // Newest last in write order, so sort ascending and keep the tail.
    all.sort_by_key(|(_, loc)| (loc.segment, loc.offset));
    let dropped_n = all.len().saturating_sub(MAX_BRANCH_CHECKPOINTS);
    let dropped: Vec<StreamId> = all[..dropped_n].iter().map(|(sid, _)| *sid).collect();
    (all.split_off(dropped_n), dropped)
}

fn branch_checkpoints_to_keep(substrate: &Substrate) -> Vec<(StreamId, RecordLoc)> {
    partition_branch_checkpoints(substrate).0
}

/// The records a member keeps of its own and never replicates, which a rewrite
/// carries whole: what it has applied of other origins and deferred — the
/// records it covers that a rewrite drops are known gone from then on — and the
/// generations fixed for what it wrote cut off (§6.7).
#[derive(Clone, Debug, Default)]
pub struct Housekeeping {
    pub replica: ReplicaPayload,
    pub rebases: RebaseMap,
}

/// A housekeeping record — no identity — holding `payload`.
fn local_record(record_type: RecordType, payload: Vec<u8>) -> CompactItem {
    CompactItem::synth(
        RecordHeader {
            record_type,
            payload_len: payload.len() as u64,
            ..RecordHeader::NONE
        },
        payload,
    )
}

/// Collect every **live** record, in dependency order, as [`CompactItem`]s —
/// planning only, **no disk reads**. `ModelKey` / `Template` / `Tokenizer` /
/// `Chunk` / `Tokens` become `Raw` items carrying their source `(segment,
/// offset, record_size)` and a header synthesized from the substrate/manifest
/// index; every resident record (`StreamDecl`, `Commit`, `Label`, `ConvState`,
/// `ProjectionEvents`, `WideQSig`, `TreeMetadata`, `TurnCoupling`, `DebugId`,
/// `Distilled`) is a `Synth` item carrying its freshly-encoded payload.
/// [`write_compacted_log`] reads the `Raw` bytes back coalesced and stages them
/// verbatim.
///
/// **Every record class must appear here or be deliberately reclaimed.** There
/// is no fallback that carries an unhandled type forward and nothing warns when
/// one is missing — the record simply stops existing at the next compaction.
/// `Npc` and `TurnCoupling` were each lost exactly that way. The exhaustive
/// match in [`super::accounting::RecordAccounting::record`] is the checklist:
/// a header-keyed type needs a `Raw` carry-forward from its location map, and a
/// payload-keyed one needs a `Synth` re-emit from substrate RAM.
///
/// A tombstone is carried until `expiry` drops it (§6.8): this compaction runs
/// past every tombstone it carries, and says so in the `Deleted` record it
/// writes. The member's own [`Housekeeping`] is written whole.
#[allow(clippy::too_many_arguments)]
pub fn collect_live_records(
    manifest: &Manifest,
    substrate: &Substrate,
    npc_locs: &StampedLocs<u64>,
    vfs_index: &VfsIndex,
    blob_index: &BlobIndex,
    marker_ids: &MarkerIds,
    node: u64,
    expiry: TombstoneExpiry,
    housekeeping: &Housekeeping,
) -> Vec<CompactItem> {
    let mut out: Vec<CompactItem> = Vec::new();

    if !housekeeping.replica.is_empty() {
        out.push(local_record(
            RecordType::Replica,
            housekeeping.replica.encode(),
        ));
    }
    if !housekeeping.rebases.is_empty() {
        out.push(local_record(
            RecordType::Rebase,
            housekeeping.rebases.encode(),
        ));
    }

    // A conversation's file events, staged verbatim: their bodies belong to
    // the daemon above, so like characters they are carried by location.
    // Only the live events of live timelines — a retired, distilled or
    // unregistered conversation's are orphans and stay behind. A tombstone
    // stays behind too when the events it killed were all this member's: they
    // are not carried, so nothing is left for it to kill. One that killed
    // another member's event is carried, because that event can still arrive
    // and must still die (`VfsIndex`, §6.5).
    for (timeline, tl) in carried(substrate, vfs_index) {
        let events = tl
            .events()
            .iter()
            .map(|(id, loc)| (RecordType::VfsEvent, id, loc));
        let tombstones = tl
            .kept_tombstones(node)
            .map(|(id, loc)| (RecordType::VfsTombstone, id, loc));
        for (record_type, id, loc) in events.chain(tombstones) {
            out.push(CompactItem::raw(
                RecordHeader {
                    record_type,
                    format: 0,
                    payload_len: loc.payload_len,
                    crc: 0,
                    stream_id: timeline,
                    chunk_index: id.seq,
                    token_count: 0,
                    origin: id.origin,
                    ..RecordHeader::NONE
                },
                loc.segment,
                loc.offset,
                loc.record_size,
            ));
        }
    }

    // Blobs, staged verbatim by location: their bytes are never in RAM, and
    // nothing supersedes one — the first record of each hash is carried.
    for (key, loc) in blob_index.blobs() {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Blob,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: key.hi,
                chunk_index: key.lo,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Characters, staged verbatim from wherever they physically live.
    //
    // `Raw`, not `Synth`, and this is the one record class where that is forced
    // rather than chosen: every other payload-keyed record on this page is
    // re-encoded from the substrate's RAM, but an NPC's payload is not in this
    // process's RAM at all — the registry that owns it lives in the daemon
    // above. So the winner is carried across byte-for-byte, exactly as
    // `Snapshot` is.
    //
    // Omitting this loop does not fail a test or log a warning. It deletes
    // every character in the store, at the next compaction, silently.
    //
    // Tombstoned characters are carried too: a deleted character's id must stay
    // taken, because the acts it already committed still name it.
    let mut npcs: Vec<(&u64, &RecordLoc)> = npc_locs.iter().collect();
    // Deterministic order — a compaction that shuffles the cast produces a
    // different file from the same inputs, which makes the output impossible to
    // diff between runs.
    npcs.sort_unstable_by_key(|(id, _)| **id);
    for (npc_id, loc) in npcs {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Npc,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: *npc_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Singletons — staged verbatim from wherever they physically live.
    for (rt, loc) in [
        (RecordType::ModelKey, manifest.model_key),
        (RecordType::Template, manifest.template),
        (RecordType::Tokenizer, manifest.tokenizer),
    ] {
        if let Some(loc) = loc {
            out.push(CompactItem::raw(
                singleton_header(rt, loc.payload_len),
                loc.segment,
                loc.offset,
                loc.record_size,
            ));
        }
    }

    // Tombstoned timelines drop out of the compacted log entirely
    // — their records are physically gone, not merely hidden.  This
    // is what reclaims disk after a refresh cycle replaces a
    // timeline.
    let tombstoned: std::collections::HashSet<u64> = substrate
        .tombstoned_timelines()
        .iter()
        .map(|t| t.raw())
        .collect();
    let distilled: HashMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();
    // An ended section generation needs no gate here: its `StreamDecl` /
    // `Chunk` / `Tokens` records are the corrupted or stale bytes the tombstone
    // exists to keep unread, and the tombstone already took them out of the
    // index (`Substrate::end_section`), so the walk below never meets them.

    // Recurrent-state snapshots: one live tail per conversation, staged
    // verbatim (`Raw`, the `Tokens` shape — the payload is a multi-MB state
    // blob nothing holds in RAM).
    //
    // **Distilled timelines shed theirs.** Distillation keeps the provenance
    // signatures and drops the content; the recurrent state IS content —
    // derived from the token stream, and useless without the K/V that was shed
    // alongside it. Without this gate a distilled timeline sheds megabytes of
    // K/V and carries ~63 MiB of state forward on every compaction pass,
    // forever, for a conversation that is never resumed. That a distilled
    // conversation becomes unresumable is correct: distillation is for
    // calibration exemplars.
    //
    // Tombstoned timelines are already handled on apply (the tombstone removes
    // the map entry), but a timeline that is tombstoned AND distilled
    // deliberately escapes the wholesale drop below, so the same set is checked
    // here rather than assumed.
    let snapshot_drop: std::collections::HashSet<u64> = distilled
        .keys()
        .chain(tombstoned.iter())
        .map(|&t| snapshot_stream_id(t).0)
        .collect();
    for (stream_id, loc) in substrate.recurrent_snapshot_entries() {
        if snapshot_drop.contains(&stream_id.0) {
            continue;
        }
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Snapshot,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }
    // A stream whose current record is a retraction keeps it: the retraction
    // is the value, and it is what refuses an older snapshot arriving later.
    for (stream_id, id) in substrate.retracted_snapshots() {
        if snapshot_drop.contains(&stream_id.0) {
            continue;
        }
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Snapshot,
                format: 0,
                payload_len: 0,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            Vec::new(),
        ));
    }

    // Prompt branch checkpoints, **capped**.
    //
    // The only record here that is a cache rather than durable state: a
    // checkpoint is a pure function of a prompt that is still on disk, so
    // dropping one costs a single prefill on next use, and keeping one costs
    // ~63 MiB in every future segment. Nothing else reclaims them — they are
    // keyed by content, so a prompt edit supersedes nothing and orphans the old
    // branch, and no timeline tombstone names them. Without this the log grows
    // by one orphan per prompt edit, forever.
    //
    // Newest-first by write position, which is the closest thing to
    // recently-used the compactor can see. A workspace exercises few branches at
    // a time, so the cap is generous relative to real use and still bounds the
    // worst case (the live schema's tree has 200) at a fixed size.
    let kept = branch_checkpoints_to_keep(substrate);
    for (stream_id, loc) in kept {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::BranchCheckpoint,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Per-stream live records — sourced from the substrate's in-RAM
    // stream index, the authoritative source. Every writer's stream is
    // carried: two members' turn N share a stream id and are both kept.
    //
    // In a fixed order — each timeline's turns in turn order, then every
    // other stream by id — so two compactions of one store write the same
    // file, and an adoption (`super::adoption`), which stamps records in the
    // order they are written, stamps a timeline's declarations rising with
    // the index they were written at: every turn keeps its position.
    let mut streams: Vec<_> = substrate.all_physical_streams().collect();
    streams.sort_by_key(|(key, entry)| match &entry.decl {
        Some(StreamDecl::Turn(t)) => (0, t.timeline_id, t.turn_index, *key),
        _ => (1, 0, 0, *key),
    });
    for (key, entry) in streams {
        let stream_id = key.stream_id;
        // Distillation degree for this turn's timeline (None = not distilled).
        // Both modes drop the KV chunks (the bulk of the on-disk cost);
        // ProvenanceOnly also drops tokens (keep sig + projections — the belief
        // gallery reads only the sig), TextOnly also drops sig + projections
        // (keep tokens — a plain read-only text record). Tombstoned timelines
        // drop entirely, as before.
        // An ORPHAN stream — no StreamDecl, and none coming — cannot have its
        // turn/section reconstructed on reload (reload rebuilds turns from
        // their decl), so every record it still owns — chunks, tokens, sig,
        // projections — is unreachable dead weight. Drop the whole stream so
        // compaction reclaims the orphaned KV instead of copying it forward on
        // every pass forever. Without this gate a decl lost in a prior
        // generation leaves its (huge) KV chunks immortal on disk. A stream
        // another origin wrote is not an orphan for want of a decl: its decl
        // is still on its way (`StreamRuntime::is_orphan`).
        if entry.is_orphan(node) {
            continue;
        }
        let mut turn_dead = false;
        let distill: Option<DistillMode> = if let Some(StreamDecl::Turn(t)) = &entry.decl {
            // A tombstoned timeline's records go wholesale — UNLESS it is
            // also distilled. That pair is the provenance corpus: calibration
            // exemplars are archived, distilled `ProvenanceOnly`, and then
            // tombstoned so they leave the live gather, while their
            // `WideQSig`s and projection events keep answering the belief
            // scan. Dropping them here erased the very records the tombstone
            // exists to preserve, and — because the `Distilled` marker went
            // with them — the next reload saw a content-declaring turn with
            // no KV and no distill exemption, i.e. `TurnIntegrity::MissingKv`.
            // Integrity repair then tombstoned it "for corruption", the
            // calibration resume filter could no longer see it, and the case
            // regenerated: ~196 exemplars rebuilt per boot, forever.
            //
            // So distilled timelines fall through to the per-mode retention
            // below whether or not they are tombstoned; only an undistilled
            // tombstone is dropped outright.
            if tombstoned.contains(&t.timeline_id) && !distilled.contains_key(&t.timeline_id) {
                continue;
            }
            // A turn-scoped tombstone on a LIVE timeline sheds the turn's
            // bulk records (chunks/tokens/sig/projections) but keeps its
            // StreamDecl: the reload restores the turn as an empty
            // placeholder from the decl, preserving the timeline's turn
            // indexing (a fully-dropped decl would renumber every later
            // turn away from its persisted stream ids).
            turn_dead = substrate.is_turn_stream_retired(key, t);
            distilled.get(&t.timeline_id).copied()
        } else {
            None
        };
        let keep_chunks = !turn_dead && distill.is_none();
        let keep_tokens = !turn_dead && distill != Some(DistillMode::ProvenanceOnly);
        let keep_sig = !turn_dead && distill != Some(DistillMode::TextOnly);
        if let Some(decl) = &entry.decl {
            let payload = decl.encode();
            out.push(CompactItem::synth(
                RecordHeader {
                    record_type: RecordType::StreamDecl,
                    format: 0,
                    payload_len: payload.len() as u64,
                    crc: 0,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    token_count: 0,
                    ..RecordHeader::NONE
                }
                .with_identity(entry.ids.decl),
                payload,
            ));
        }
        if keep_chunks {
            for (&idx, loc) in &entry.chunks {
                out.push(CompactItem::raw(
                    RecordHeader {
                        record_type: RecordType::Chunk,
                        format: loc.format,
                        payload_len: loc.payload_len,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: idx,
                        token_count: loc.token_count,
                        ..RecordHeader::NONE
                    },
                    loc.segment,
                    loc.offset,
                    loc.record_size,
                ));
            }
        }
        if keep_tokens {
            if let Some(loc) = entry.tokens {
                out.push(CompactItem::raw(
                    RecordHeader {
                        record_type: RecordType::Tokens,
                        format: 0,
                        payload_len: loc.payload_len,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    },
                    loc.segment,
                    loc.offset,
                    loc.record_size,
                ));
            }
        }
        // Per-turn projection-event timeline — re-emitted from the resident
        // bytes rather than read back from disk, so the GUI dots survive a
        // compaction pass. Dropped for TextOnly (`keep_sig` false).
        if keep_sig {
            if let Some(payload) = &entry.projection_events {
                out.push(CompactItem::synth(
                    RecordHeader {
                        record_type: RecordType::ProjectionEvents,
                        format: 0,
                        payload_len: payload.len() as u64,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    }
                    .with_identity(entry.ids.projection_events),
                    payload.clone(),
                ));
            }
        }
        // Per-turn wide-Q signature window — same resident-bytes re-emit so the
        // decode→decode consensus substrate survives a compaction pass. Dropped
        // for TextOnly (`keep_sig` false).
        if keep_sig {
            if let Some(payload) = &entry.wide_q_sigs {
                out.push(CompactItem::synth(
                    RecordHeader {
                        record_type: RecordType::WideQSig,
                        format: 0,
                        payload_len: payload.len() as u64,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    }
                    .with_identity(entry.ids.wide_q_sigs),
                    payload.clone(),
                ));
            }
        }
        // The turn's QSA index page, re-emitted with its K/V for the same
        // reason: a compaction that carries a turn's chunks forward but drops
        // its page leaves a turn that can be borrowed and cannot be indexed.
        // Unlike the signature this is not sig-gated — it belongs to the K/V,
        // and a `TextOnly` turn that keeps its chunks keeps its page.
        // A page held in RAM is written from its bytes; one left in the log is
        // carried by location, as the turn's K/V is.
        let page_header = |payload_len: u64| {
            RecordHeader {
                record_type: RecordType::TurnIndexPage,
                format: 0,
                payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(entry.ids.index_page)
        };
        match &entry.index_page {
            Some(IndexPage::Resident(payload)) => out.push(CompactItem::synth(
                page_header(payload.bytes().len() as u64),
                payload.shared_bytes(),
            )),
            Some(IndexPage::Stored(loc)) => out.push(CompactItem::raw(
                page_header(loc.payload_len),
                loc.segment,
                loc.offset,
                loc.record_size,
            )),
            None => {}
        }
        if let Some(through) = entry.committed_through {
            out.push(CompactItem::synth(
                RecordHeader {
                    record_type: RecordType::Commit,
                    format: 0,
                    payload_len: 0,
                    crc: 0,
                    stream_id: stream_id.0,
                    chunk_index: through,
                    token_count: 0,
                    ..RecordHeader::NONE
                }
                .with_identity(entry.ids.commit),
                Vec::new(),
            ));
        }
    }
    // Timeline tombstones are kept although their records vanish wholesale
    // above: a record replicated late — by a member that was cut off — must
    // still find the tombstone and die. Each stays until it is old and this is
    // not the first complete compaction to run past it, when it is dropped and
    // leaves nothing behind (a member away longer drops what it did not author
    // before it syncs, `docs/replicated_substrate.md` §7.8); this compaction
    // has run past every one it carries, and the `Deleted` record it writes
    // says so.
    let expired = expiry.expired(substrate, marker_ids);
    let mut dead: Vec<TimelineId> = substrate
        .tombstoned_timelines()
        .iter()
        .copied()
        .filter(|tl| !expired.timelines.contains(tl))
        .collect();
    dead.sort_unstable();
    let mut swept: Vec<u64> = Vec::new();
    for tl in dead {
        let payload = TombstonePayload {
            timeline_id: tl.raw(),
            turn_index: None,
            turn_writer: 0,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::Tombstone, 0, &payload);
        swept.push(tl.raw());
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Tombstone,
                payload_len: payload.len() as u64,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // A keyed value's tombstone expires the same way: once a complete
    // compaction has run past it and it is old, it is dropped. The custom
    // objects below skip what expires here.
    let mut swept_keys: Vec<u64> = substrate
        .held_custom_objects()
        .into_iter()
        .filter(|(obj, _)| obj.ceased && !expired.keys.contains(&obj.stream_id()))
        .map(|(obj, _)| obj.stream_id())
        .collect();
    swept_keys.sort_unstable();
    if !swept.is_empty() || !swept_keys.is_empty() {
        let payload = DeletedPayload { swept, swept_keys }.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Deleted,
                payload_len: payload.len() as u64,
                ..RecordHeader::NONE
            },
            payload,
        ));
    }
    // Turn-scoped tombstones on LIVE timelines are re-emitted: the compacted
    // log keeps the dead turn's StreamDecl (index preservation) but sheds its
    // bulk records, and the reload's placeholder logic keys off this tombstone
    // to restore the turn as an empty hole instead of re-hitting the (now
    // absent) content.
    let mut marks: Vec<_> = substrate.tombstoned_turns().iter().copied().collect();
    marks.sort_unstable();
    for mark in marks {
        if tombstoned.contains(&mark.timeline.raw()) {
            continue;
        }
        let payload = TombstonePayload {
            timeline_id: mark.timeline.raw(),
            turn_index: Some(mark.turn_index),
            turn_writer: mark.writer,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::Tombstone, 0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Tombstone,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Tombstoned section streams — re-emitted the same way turn-scoped
    // tombstones are: the marker itself is what makes the drop permanent
    // across reloads, so it must survive the rewrite that drops the
    // section's now-dead `Chunk` / `Tokens` / `StreamDecl` records (those
    // simply never appear in the live set above, since the tombstone took them
    // out of the index). Losing this marker would resurrect the corrupted
    // chunks on the next boot that restores from an older segment still
    // holding them. One per ended generation, naming it; one per unstamped
    // stream a tombstone written before generations ended.
    let ended = substrate
        .ended_section_generations()
        .map(|(sid, g)| (sid, Some(g)))
        .chain(substrate.ended_unstamped_sections().map(|sid| (sid, None)));
    for (stream_id, generation) in ended {
        let payload = SectionTombstonePayload {
            generation,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::SectionTombstone, stream_id.0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::SectionTombstone,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline Label / ConvState records — synthesised from
    // the substrate's live state, one `Label` per record that holds a field.
    // Tombstoned timelines are skipped so retired conversations don't leave
    // dangling sidebar entries on disk.
    for (timeline_id, meta, id) in substrate.live_label_records() {
        // A tombstoned timeline that is ALSO distilled is the provenance corpus
        // (calibration exemplars: archived, distilled, then tombstoned out of the
        // live gather while their signatures keep answering the belief scan).
        // Its marker/metadata must survive, or the next reload sees a
        // content-declaring turn with no distill exemption and condemns it.
        // Only an undistilled tombstone is a retired conversation.
        if tombstoned.contains(&timeline_id) && !distilled.contains_key(&timeline_id) {
            continue;
        }
        let payload = super::manifest::encode_label_payload(
            timeline_id,
            &meta.conv_id,
            &meta.label,
            &meta.custom,
        );
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Label,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline conversation state — the whole state, archived flag and
    // branches alike, keyed in the header by timeline like every other
    // `ConvState` record. Same retirement rule as the labels above.
    for (timeline_id, state, id) in substrate.live_conv_states() {
        if tombstoned.contains(&timeline_id) && !distilled.contains_key(&timeline_id) {
            continue;
        }
        let payload = encode_conv_state_payload(timeline_id, &state);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::ConvState,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: timeline_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-(timeline, turn) summary-tree metadata — emit one record
    // per live tree node directly from substrate state.
    for (payload, id) in substrate.live_tree_metadata_payloads() {
        if tombstoned.contains(&payload.timeline_id) {
            continue;
        }
        let bytes = payload.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::TreeMetadata,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Records about a timeline not registered yet, held until its first turn
    // declaration arrives. They are durable state like any other: dropping one
    // here would lose it for good, since its `oseq` was already applied.
    for (timeline_id, rt, payload, id) in substrate.pending_records() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let stream_id = if rt == RecordType::ConvState {
            timeline_id
        } else {
            0
        };
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-(timeline, turn) tool round-trip couplings.
    //
    // Resident and payload-keyed, like `Label` and `DebugId` — nothing
    // supersedes an old copy and `accounting` never credits one as dead, so the
    // only way one reaches the compacted log is a re-emit from RAM. Dropping it
    // deletes the only statement that a call turn and its response are one
    // exchange: `summary_tree::exchange` groups an exchange as the maximal run
    // of coupled turns, so every compaction began splitting the two halves of a
    // round-trip apart. Nothing failed — a timeline with no couplings is a
    // legitimate state (nothing ever called a tool there) and reads as intact,
    // which is exactly why this went unnoticed.
    for (timeline_id, from_turn, id) in substrate.live_couplings() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = TurnCouplingPayload {
            timeline_id,
            from_turn,
        }
        .encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::TurnCoupling,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline debug_id.
    for (timeline_id, debug_id, id) in substrate.live_debug_ids() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = DebugIdPayload {
            timeline_id,
            debug_id,
        };
        let bytes = payload.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::DebugId,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Generic keyed custom objects — re-emitted from the held in-RAM set so
    // the winning object outlives the compaction pass, keyed on the same
    // stable key-hash `stream_id` it was written under; a tombstone among
    // them is carried until it expired above.
    for (obj, id) in substrate.held_custom_objects() {
        if expired.keys.contains(&obj.stream_id()) {
            continue;
        }
        let bytes = obj.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::CustomObject,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: obj.stream_id(),
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Per-distilled-timeline marker — re-emitted (with mode) so a distilled
    // timeline stays distilled across the compaction it just shed through.
    // Without this the marker is lost and, e.g., a text-only archived
    // conversation would read back as un-distilled.
    //
    // Re-emitted for TOMBSTONED timelines too. The marker is what exempts a
    // distilled turn from the `MissingKv` integrity verdict (`classify_turn`),
    // and a distilled turn has legitimately shed its KV — so losing the marker
    // condemns the very corpus the distillation created. The provenance corpus
    // is tombstoned by design (out of the live gather, still answering the
    // belief scan by signature), so "tombstoned" cannot mean "forget it was
    // distilled".
    for (&timeline_id, &mode) in &distilled {
        let payload = DistillPayload { timeline_id, mode }.encode();
        let id = marker_ids.identity_for(RecordType::Distilled, 0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Distilled,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Every re-encoded record above kept the identity of the record that holds
    // its value now — the same origin and `oseq`, so every member still
    // recognises it, and the same stamp, so it still ranks where it did. A
    // read-back record keeps its own: `write_compacted_log` indexes it from its
    // copied header.
    out
}

/// Write the live set to a fresh log at `path`, interleaving a fresh
/// `HeaderIndex` chain and publishing its head in the superblock — a
/// just-compacted log recovers through the chain like any other.
///
/// `Raw` items are read back from their source segment through `read_into` in
/// **coalesced stripes** (a contiguous run of records → one sequential read)
/// and staged **byte-for-byte** — no decode, no CRC re-verify, no re-encode.
/// `Synth` items are encoded from their in-RAM payload. The write buffer is
/// flushed incrementally so the whole new log is never resident at once.
/// Returns the open [`LogFile`] and its (singleton-only) manifest. `path` is
/// removed first if it already exists.
///
/// With `adoption`, a record with no identity is stamped as it is written
/// ([`Adoption::restamp`]) — a read-back one decoded and encoded again under
/// its new header — and every other record is written as without.
///
/// `written` is told how many of `items` are staged, after each record or
/// stripe: a whole-store rewrite runs for minutes, and this is its counter.
pub fn write_compacted_log(
    path: &Path,
    items: &[CompactItem],
    read_into: &mut dyn FnMut(SegmentId, u64, &mut [u8]) -> Result<()>,
    mut adoption: Option<&mut Adoption>,
    written: &mut dyn FnMut(usize),
) -> Result<(LogFile, Manifest)> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut log = LogFile::create(path)?;
    let mut manifest = Manifest::new();

    let mut pending: Vec<IndexEntry> = Vec::new();
    let mut last_index: (u64, u64) = (0, 0);
    let flush_index =
        |log: &mut LogFile, pending: &mut Vec<IndexEntry>, last_index: &mut (u64, u64)| {
            let payload = encode_index_payload(*last_index, pending);
            let h = RecordHeader {
                record_type: RecordType::HeaderIndex,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            };
            let bytes = encode_record(&h, &payload);
            let offset = log.stage(&bytes);
            *last_index = (offset, bytes.len() as u64);
            pending.clear();
        };

    // Single streaming pass in dependency order. `Synth` items encode from their
    // in-RAM payload; a `Raw` item begins a **run** — the maximal span of
    // consecutive `Raw` items that is also physically contiguous in one source
    // segment (a stream's chunk/token run: adjacent on disk and adjacent in the
    // list). The run is read in **coalesced stripes** of at most
    // `MAX_STRIPE_BYTES` and each record staged straight from the read buffer:
    // one copy per record, no per-record allocation, and no buffer — read or
    // write — larger than a stripe plus `COMPACT_FLUSH_BYTES`.
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            CompactItem::Synth { header, payload } => {
                let mut h = *header;
                h.payload_len = payload.len() as u64;
                let stamped = match adoption.as_deref_mut() {
                    Some(a) => a.restamp(&h, payload)?,
                    None => None,
                };
                let (written_header, bytes) = match &stamped {
                    Some((h, payload)) => encode_record_written(h, payload),
                    None => encode_record_written(&h, payload),
                };
                let offset = log.stage(&bytes);
                pending.push(IndexEntry::from_header(
                    &written_header,
                    offset,
                    bytes.len() as u64,
                ));
                i += 1;
                written(i);
            }
            CompactItem::Raw {
                segment,
                offset,
                record_size,
                ..
            } => {
                let seg = *segment;
                let start = *offset;
                let mut end = start + record_size;
                let mut j = i + 1;
                while let Some(CompactItem::Raw {
                    segment: s2,
                    offset: o2,
                    record_size: rs2,
                    ..
                }) = items.get(j)
                {
                    if *s2 != seg || *o2 != end {
                        break;
                    }
                    end += rs2;
                    j += 1;
                }
                // The run is read a bounded stripe at a time, and the write
                // buffer flushed between stripes: a segment of back-to-back
                // live chunks is one run of gigabytes (see `stripes`).
                let run = &items[i..j];
                let spans: Vec<(u64, u64)> = run
                    .iter()
                    .map(|it| match it {
                        CompactItem::Raw {
                            offset,
                            record_size,
                            ..
                        } => (*offset, *record_size),
                        CompactItem::Synth { .. } => unreachable!("a Raw run holds only Raw items"),
                    })
                    .collect();
                for (a, b) in stripes(&spans, MAX_STRIPE_BYTES) {
                    let from = spans[a].0;
                    let len = (spans[b - 1].0 + spans[b - 1].1 - from) as usize;
                    if buf.len() < len {
                        buf.resize(len, 0);
                    }
                    read_into(seg, from, &mut buf[..len])?;
                    let mut within = 0usize;
                    for it in &run[a..b] {
                        let CompactItem::Raw { record_size, .. } = it else {
                            unreachable!("a Raw run holds only Raw items");
                        };
                        let read = *record_size as usize;
                        let raw = &buf[within..within + read];
                        within += read;
                        // Indexed from the copied record's own header, so the copy
                        // keeps the identity and digest it was written with — or,
                        // adopted, the identity it is stamped with here.
                        let (header, _) = decode_header(raw)?;
                        let stamped = match adoption.as_deref_mut() {
                            Some(a) if header.identity().is_none() => {
                                let (_, payload, _) = decode_record(raw)?;
                                a.restamp(&header, payload)?
                                    .map(|(h, p)| encode_record_written(&h, &p))
                            }
                            _ => None,
                        };
                        let (header, staged, sz) = match &stamped {
                            Some((written, bytes)) => (*written, log.stage(bytes), bytes.len()),
                            None => (header, log.stage(raw), read),
                        };
                        let header = &header;
                        // Singletons resolve from the manifest (the substrate
                        // replay never rebuilds them); `Chunk` / `Tokens` are
                        // substrate-indexed and never enter the manifest.
                        let loc = RecordLoc {
                            segment: FIRST_SEGMENT,
                            offset: staged,
                            payload_len: header.payload_len,
                            record_size: sz as u64,
                        };
                        match header.record_type {
                            RecordType::ModelKey => manifest.model_key = Some(loc),
                            RecordType::Template => manifest.template = Some(loc),
                            RecordType::Tokenizer => manifest.tokenizer = Some(loc),
                            _ => {}
                        }
                        pending.push(IndexEntry::from_header(header, staged, sz as u64));
                        if pending.len() >= INDEX_FLUSH_ENTRIES {
                            flush_index(&mut log, &mut pending, &mut last_index);
                        }
                    }
                    if log.pending_len() >= COMPACT_FLUSH_BYTES {
                        log.flush()?;
                    }
                    written(i + b);
                }
                i = j;
            }
        }
        if pending.len() >= INDEX_FLUSH_ENTRIES {
            flush_index(&mut log, &mut pending, &mut last_index);
        }
        // Bound the in-RAM write buffer for a whole-store rewrite.
        if log.pending_len() >= COMPACT_FLUSH_BYTES {
            log.flush()?;
        }
    }
    if !pending.is_empty() {
        flush_index(&mut log, &mut pending, &mut last_index);
    }
    log.commit()?;
    if last_index != (0, 0) {
        log.set_last_index(last_index)?;
    }

    Ok((log, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::log_file::{read_record_at, LogSource, MemLog, SUPERBLOCK_SIZE};
    use crate::persistence::manifest::ConvState;
    use crate::persistence::rebase::RebaseEntry;
    use crate::persistence::record::CustomObjectPayload;
    use crate::persistence::record::{encode_record, SectionGeneration};
    use crate::persistence::tombstone_expiry::DEFAULT_TOMBSTONE_RETENTION;
    use crate::persistence::vfs::VfsEventId;
    use crate::substrate::file_store_key;

    /// A clock at the epoch: no tombstone is ever old enough to shrink.
    const NO_EXPIRY: TombstoneExpiry = TombstoneExpiry {
        now_ms: 0,
        retention: DEFAULT_TOMBSTONE_RETENTION,
    };

    /// True iff any record of type `rt` carries `payload` (only `Synth` items
    /// hold a payload; `Raw` read-back records don't).
    fn has_synth(live: &[CompactItem], rt: RecordType, payload: &[u8]) -> bool {
        live.iter()
            .any(|it| it.header().record_type == rt && it.synth_payload() == Some(payload))
    }

    /// True iff any record of type `rt` is present (by header, `Raw` or `Synth`).
    fn has_type(live: &[CompactItem], rt: RecordType) -> bool {
        live.iter().any(|it| it.header().record_type == rt)
    }

    /// **The live list shares a turn's metadata blobs; it does not copy them.**
    /// The whole list exists before the first byte is written, so a copy is a
    /// second image of every live turn's index page and wide-Q signature held
    /// for the length of the rewrite.
    #[test]
    fn the_live_list_shares_the_substrates_blobs() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        use crate::substrate::HeldPayloadBytes;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 6,
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
        let mut mem =
            MemLog::with_records(&record(RecordType::StreamDecl, sid.0, 0, &decl.encode()));
        let (manifest, mut substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let none = RecordIdentity::NONE;
        substrate.apply_projection_events(sid, vec![1, 2, 3], none);
        substrate.apply_wide_q_sigs(sid, vec![4, 5, 6, 7], none);
        substrate.apply_index_page(sid, vec![8, 9], none);
        // The payloads held whole are what the memory report counts.
        assert_eq!(
            substrate.held_payload_bytes(),
            HeldPayloadBytes {
                wide_q_sigs: 4,
                projection_events: 3,
            }
        );

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let entry = substrate.stream_of(sid).unwrap();
        let payload_of = |rt: RecordType| {
            live.iter()
                .find_map(|it| match it {
                    CompactItem::Synth { header, payload }
                        if header.record_type == rt && header.stream_id == sid.0 =>
                    {
                        Some(payload)
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{rt:?} is missing from the live list"))
        };

        let events = payload_of(RecordType::ProjectionEvents);
        assert_eq!(&events[..], &[1, 2, 3]);
        assert!(Arc::ptr_eq(
            events,
            entry.projection_events.as_ref().unwrap()
        ));

        let sigs = payload_of(RecordType::WideQSig);
        assert_eq!(&sigs[..], &[4, 5, 6, 7]);
        assert!(Arc::ptr_eq(sigs, entry.wide_q_sigs.as_ref().unwrap()));

        let page = payload_of(RecordType::TurnIndexPage);
        assert_eq!(&page[..], &[8, 9]);
        assert!(Arc::ptr_eq(
            page,
            &entry
                .index_page
                .as_ref()
                .unwrap()
                .resident()
                .unwrap()
                .shared_bytes()
        ));
    }

    fn record(rt: RecordType, stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index,
                token_count: if rt == RecordType::Chunk { 32 } else { 0 },
                ..RecordHeader::NONE
            },
            payload,
        )
    }

    // A valid `Chunk` record whose KV bytes carry `content`, with the golden set
    // (Fletcher-32 over the KV bytes) — so `read_record_at`'s chunk verification
    // passes. Used where a test reads a chunk's payload back; the header-only
    // liveness tests can keep the opaque `record(RecordType::Chunk, …)`.
    fn chunk_record(stream_id: u64, chunk_index: u64, content: &[u8]) -> Vec<u8> {
        use crate::persistence::record::ChunkPayload;
        let cp = ChunkPayload {
            offset: 0,
            k_formats: vec![4],
            v_formats: vec![5],
            k_pal: vec![],
            v_pal: vec![],
            k_scale: vec![],
            v_scale: vec![],
            kv_bytes: content.to_vec(),
        };
        let payload = cp.encode();
        encode_record(
            &RecordHeader {
                record_type: RecordType::Chunk,
                format: 4,
                payload_len: payload.len() as u64,
                crc: candle::fletcher::fletcher32(&cp.kv_bytes),
                stream_id,
                chunk_index,
                token_count: 32,
                ..RecordHeader::NONE
            },
            &payload,
        )
    }

    #[test]
    fn collect_keeps_only_the_live_winners() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let mut blob = Vec::new();
        // A stale ModelKey, then the live one.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model-v1-stale"));
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model-v2-live"));
        // The stream's decl — without it the chunk is an orphan and is reclaimed.
        let decl5 = StreamDecl::Turn(TurnDecl {
            timeline_id: 5,
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
        blob.extend_from_slice(&record(RecordType::StreamDecl, 5, 0, &decl5.encode()));
        // A stream: a 20-token partial chunk, then the sealed winner.
        blob.extend_from_slice(&chunk_record(5, 0, b"partial-20tok-dead"));
        blob.extend_from_slice(&chunk_record(5, 0, b"sealed-final-live"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        // Exactly: 1 ModelKey + 1 StreamDecl + 1 Chunk = 3 (LWW keeps one of each).
        assert_eq!(live.len(), 3);
        // Each read-back item points at the live winner's on-disk location — the
        // manifest / substrate index already resolved last-writer-wins — so the
        // stale ModelKey and the dead partial chunk are never staged. Read the
        // chosen source bytes back to prove which record was selected.
        let model = live
            .iter()
            .find(|it| it.header().record_type == RecordType::ModelKey)
            .unwrap();
        let (off, size) = raw_loc(model);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"model-v2-live",
            "the stale ModelKey is dropped"
        );
        let chunk = live
            .iter()
            .find(|it| it.header().record_type == RecordType::Chunk)
            .unwrap();
        let (off, size) = raw_loc(chunk);
        let chunk_rec = read_record_at(&mut mem, off, size).unwrap();
        let chunk_cp =
            crate::persistence::record::ChunkPayload::decode(&chunk_rec.payload).unwrap();
        assert_eq!(
            chunk_cp.kv_bytes, b"sealed-final-live",
            "the dead partial chunk is dropped"
        );
    }

    /// An orphaned stream — KV chunks (and tokens) on disk but NO `StreamDecl` —
    /// is reclaimed by compaction, not copied forward. Regression for the
    /// silent-loss bloat bug: a decl lost in a prior generation must not leave
    /// its (huge) KV immortal on disk. The turn is already unreloadable without
    /// its decl, so dropping the chunks loses nothing recoverable and reclaims
    /// the space.
    #[test]
    fn orphan_stream_without_decl_is_reclaimed() {
        let mut blob = Vec::new();
        // Two orphan streams: one with a chunk, one with a chunk + tokens. No
        // StreamDecl for either. Plus a legit ModelKey singleton to prove only
        // the orphans are dropped.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model"));
        blob.extend_from_slice(&chunk_record(9001, 0, b"orphan-kv-a"));
        blob.extend_from_slice(&chunk_record(9002, 0, b"orphan-kv-b"));
        blob.extend_from_slice(&record(RecordType::Tokens, 9002, 0, b"orphan-tokens"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            !has_type(&live, RecordType::Chunk),
            "orphan chunks (no StreamDecl) must be reclaimed, not kept forever",
        );
        assert!(
            !has_type(&live, RecordType::Tokens),
            "an orphan stream's tokens are reclaimed with it",
        );
        // The legit singleton survives — only orphan streams are dropped.
        assert!(has_type(&live, RecordType::ModelKey));
    }

    #[test]
    fn compacted_log_recovers_to_an_identical_live_set() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"m1"));
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"m2"));
        // The stream's decl — without it the chunks are orphans and are reclaimed.
        let decl7 = StreamDecl::Turn(TurnDecl {
            timeline_id: 7,
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
        blob.extend_from_slice(&record(RecordType::StreamDecl, 7, 0, &decl7.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-old"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-new"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 1, b"c1"));
        blob.extend_from_slice(&record(RecordType::Tokens, 7, 0, b"tokens"));
        let mut mem = MemLog::with_records(&blob);
        let (before, before_sub, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(
            &before,
            &before_sub,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let path = std::env::temp_dir().join(format!(
            "kvtier_compact_{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut counted = Vec::new();
        let (mut new_log, _after_manifest) = write_compacted_log(
            &path,
            &live,
            &mut |_s, off, dest| mem.read_into(off, dest),
            None,
            &mut |n| counted.push(n),
        )
        .unwrap();
        // The counter only rises, and ends on the whole live set.
        assert!(counted.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(counted.last().copied(), Some(live.len()));

        // The compacted log re-walked into a substrate has the same
        // live streams + chunks as the source.
        let hint = new_log.superblock().last_index;
        assert_ne!(hint, (0, 0), "compacted log must carry an index chain");
        let mut after_sub = Substrate::new();
        let recovered = super::super::recovery::recover_with_sink(
            &mut new_log,
            super::super::segment::FIRST_SEGMENT,
            hint,
            |e| after_sub.apply_walker_entry(e),
        )
        .unwrap();
        assert_eq!(
            recovered.last_index,
            Some(hint),
            "recovery of a compacted log must take the chain path"
        );
        assert_eq!(before_sub.live_chunk_count(), after_sub.live_chunk_count());
        assert_eq!(after_sub.live_chunk_count(), 2);
        std::fs::remove_file(&path).ok();
    }

    /// Compactor drops every record bound to a tombstoned timeline.
    /// The simulation: write a `Label` for timeline X plus a stream
    /// decl/chunk for X, then a `Tombstone` for X.  The
    /// compactor's `collect_live_records` output must contain
    /// neither the Label nor the chunk.
    /// A tombstoned timeline that is ALSO distilled is the provenance corpus —
    /// calibration exemplars, archived then distilled `ProvenanceOnly` then
    /// tombstoned so they leave the live gather while still answering the belief
    /// scan by signature. Compaction must retain them by their MODE, and must
    /// keep re-emitting the `Distilled` marker.
    ///
    /// Losing the marker is what made this self-sustaining: the next reload saw
    /// a content-declaring turn with no KV and no distill exemption
    /// (`TurnIntegrity::MissingKv`), integrity repair tombstoned it "for
    /// corruption", and the calibration resume filter regenerated the case —
    /// ~196 exemplars rebuilt every boot.
    #[test]
    fn distilled_tombstoned_timeline_keeps_its_marker_and_sigs() {
        use crate::persistence::record::{DistillPayload, TombstonePayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let corpus_tl: u64 = 4242;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: corpus_tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, 300, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 300, 0, b"corpus-kv"));
        blob.extend_from_slice(&record(RecordType::WideQSig, 300, 0, b"corpus-sig"));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: corpus_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: corpus_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The marker survives, at its mode — without it the exemption is lost.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::ProvenanceOnly)),
            "distilled+tombstoned timeline must keep its Distilled marker",
        );
        // ProvenanceOnly retention still applies: the signature is what the
        // belief scan reads, so it stays; the KV is shed as usual.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::WideQSig
                    && it.header().stream_id == 300),
            "the provenance corpus keeps its WideQSig",
        );
        assert!(
            !live
                .iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 300),
            "ProvenanceOnly still sheds the KV",
        );
    }

    /// The corpus's Label must survive its tombstone too.
    ///
    /// The Label carries `custom` — which holds the calibration marker the
    /// resume filter matches on — and the archived flag it then tests. Dropping
    /// it leaves an exemplar that cannot be recognised as done even once the
    /// distill marker is found, so the case regenerates anyway.
    #[test]
    fn distilled_tombstoned_timeline_keeps_its_label_and_marker() {
        use crate::persistence::record::{DistillPayload, TombstonePayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let corpus_tl: u64 = 777;
        let retired_tl: u64 = 888;
        let mut custom = std::collections::BTreeMap::new();
        custom.insert("calib_marker".to_string(), "file_read|abc".to_string());

        // Both need a turn stream, or the timeline is never registered and
        // `live_conv_meta` (which the Label re-emit walks) never sees it.
        let turn_decl = |tl: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
                role: 1,
                block_start: 0,
                block_end: 1,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            400,
            0,
            &turn_decl(corpus_tl).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            500,
            0,
            &turn_decl(retired_tl).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(
                corpus_tl,
                "corpus-conv",
                "Corpus",
                &custom,
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(
                retired_tl,
                "retired-conv",
                "Retired",
                &custom,
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: corpus_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        for tl in [corpus_tl, retired_tl] {
            blob.extend_from_slice(&record(
                RecordType::Tombstone,
                0,
                0,
                &TombstonePayload {
                    timeline_id: tl,
                    turn_index: None,
                    turn_writer: 0,
                    reason: None,
                }
                .encode(),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let has_label = |needle: &str| {
            live.iter().any(|it| {
                it.header().record_type == RecordType::Label
                    && it
                        .synth_payload()
                        .and_then(|p| std::str::from_utf8(p).ok())
                        .map(|s| s.contains(needle))
                        .unwrap_or(false)
            })
        };
        assert!(
            has_label("corpus-conv"),
            "distilled+tombstoned keeps its Label (marker + archived flag)"
        );
        assert!(
            !has_label("retired-conv"),
            "an ordinary tombstone still drops its Label"
        );
    }

    #[test]
    fn tombstoned_timeline_records_drop_during_compaction() {
        use crate::persistence::record::TombstonePayload;
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let dead_tl: u64 = 12345;
        let alive_tl: u64 = 67890;

        // Build a redo log that contains:
        //   - a Label record for the dead timeline
        //   - a Label record for the alive timeline
        //   - a TurnDecl stream for the dead timeline + one Chunk
        //   - a TurnDecl stream for the alive timeline + one Chunk
        //   - the Tombstone naming the dead timeline
        let turn_decl = |tl: u64| TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        };
        let dead_decl = StreamDecl::Turn(turn_decl(dead_tl));
        let alive_decl = StreamDecl::Turn(turn_decl(alive_tl));
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(
                dead_tl,
                "dead-conv",
                "Dead",
                &std::collections::BTreeMap::new(),
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(
                alive_tl,
                "alive-conv",
                "Alive",
                &std::collections::BTreeMap::new(),
            ),
        ));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 100, 0, &dead_decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 100, 0, b"dead-chunk-payload"));
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            200,
            0,
            &alive_decl.encode(),
        ));
        blob.extend_from_slice(&record(RecordType::Chunk, 200, 0, b"alive-chunk-payload"));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The dead timeline's records are physically gone from the live set.
        // Chunks are read-back (`Raw`) items keyed by stream id; the dead
        // stream (100) must contribute none.
        assert!(
            !live
                .iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 100),
            "tombstoned timeline's chunk must be dropped during compaction",
        );
        assert!(
            !live.iter().any(|it| it
                .synth_payload()
                .and_then(|p| std::str::from_utf8(p).ok())
                .map(|s| s.contains("dead-conv"))
                .unwrap_or(false)),
            "tombstoned timeline's Label must be dropped during compaction",
        );
        // The alive timeline's records survive: its chunk (stream 200) present,
        // its Label carrying "alive-conv" present.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Chunk
                    && it.header().stream_id == 200),
            "alive timeline's chunk must survive compaction",
        );
        assert!(
            live.iter().any(|it| it
                .synth_payload()
                .and_then(|p| std::str::from_utf8(p).ok())
                .map(|s| s.contains("alive-conv"))
                .unwrap_or(false)),
            "alive timeline's Label must survive compaction",
        );
    }

    #[test]
    fn projection_events_record_survives_compaction() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 42,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 4242u64; // header stream id ties the two records to one stream
        let proj_payload = br#"[{"start_token":0,"seconds":3.0,"buckets":[]}]"#.to_vec();

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::ProjectionEvents, sid, 0, &proj_payload));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_synth(&live, RecordType::ProjectionEvents, &proj_payload),
            "ProjectionEvents record must survive compaction with its payload intact",
        );
    }

    #[test]
    fn wide_q_sigs_record_survives_compaction() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 43,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 4343u64; // header stream id ties the two records to one stream
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "WideQSig record must survive compaction with its payload intact",
        );
    }

    #[test]
    fn distilled_turn_keeps_sig_drops_content() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 77u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            // A calibration turn: gather-scope tags (unchanged by distillation).
            tags: vec!["tool".to_string(), "calculator".to_string()],
        });
        let sid = 7777u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::Tokens, sid, 0, b"token-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        // The distillation marker for the timeline (provenance-only).
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        // The declaration and the sig survive — the belief gallery still finds it.
        assert!(
            has_type(&live, RecordType::StreamDecl),
            "provenance-only turn keeps its StreamDecl (tags + gallery scope)",
        );
        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "provenance-only turn keeps its WideQSig",
        );
        // The content is reclaimed.
        assert!(
            !has_type(&live, RecordType::Chunk),
            "provenance-only turn drops its KV chunks",
        );
        assert!(
            !has_type(&live, RecordType::Tokens),
            "provenance-only turn drops its tokens",
        );
        // The marker survives the compaction it just shed through, at its mode.
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::ProvenanceOnly)),
            "distilled marker re-emitted with its mode",
        );
    }

    /// Encoded snapshot payload for `timeline` at `turn` with a recognisable
    /// `fill` byte — the same fixture shape the maintenance relocation tests
    /// use, kept local because that module's test helpers are private to it.
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
            // Filled, not empty: a relocation that carried the layers and
            // dropped the model's own state would still pass every assertion
            // that only inspects the layers.
            aux: vec![fill; 24],
        }
        .encode()
    }

    /// **A distilled timeline sheds its recurrent snapshot; its sig survives.**
    ///
    /// The inverse of the test above, on the axis that was missing a gate.
    /// Distillation keeps the provenance signatures and drops the content, and
    /// the recurrent state is content — derived from the token stream, useless
    /// without the K/V shed alongside it. Ungated, a distilled timeline sheds
    /// megabytes of K/V and carries ~63 MiB of state forward on every pass, for
    /// a conversation that is never resumed.
    #[test]
    fn a_distilled_timelines_recurrent_snapshot_is_dropped() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 131u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: vec!["tool".to_string()],
        });
        let sid = 13131u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);
        let snap = snapshot_payload(tl, 0, 0xAB);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a distilled timeline kept its recurrent snapshot — megabytes of \
             state carried forward on every compaction pass, for a conversation \
             that can never be resumed because its K/V is gone"
        );
        assert!(
            has_synth(&live, RecordType::WideQSig, &wide_payload),
            "the provenance signature must survive — that is what distillation \
             is for"
        );
    }

    /// A tombstoned timeline's snapshot is dropped too. The tombstone already
    /// removes the map entry on apply, but a timeline that is tombstoned AND
    /// distilled deliberately escapes the wholesale drop, so the snapshot gate
    /// checks the set rather than assuming the entry is gone.
    #[test]
    fn a_tombstoned_timelines_recurrent_snapshot_is_dropped() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::TombstonePayload;

        let tl = 132u64;
        let snap = snapshot_payload(tl, 3, 0xCD);
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a tombstoned timeline's recurrent snapshot survived compaction"
        );
    }

    /// **A distilled timeline's recurrent tail is retired at the INDEX**, so
    /// every rewrite path sheds it without having to consult the distilled set.
    ///
    /// The gate used to live only in `collect_live_records`. Incremental
    /// maintenance relocated the same tails forward on every pass, on the stated
    /// reasoning that "the substrate map only ever holds live conversations'
    /// tails" — true of a tombstone, which removes the entry, and false of a
    /// distill, which did not. Measured on a production store: 766 distilled
    /// timelines × ~116 MiB, about 89 GB of a 120 GB log, none of it resumable.
    ///
    /// Asserting on the index (`recurrent_snapshot_entries`) rather than on one
    /// path's output is the point — that is the property both paths read.
    #[test]
    fn a_distilled_timelines_recurrent_snapshot_is_retired_from_the_index() {
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{DistillMode, DistillPayload};

        let tl = 155u64;
        let snap = snapshot_payload(tl, 3, 0xAB);
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snap,
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        // The index itself — what maintenance's relocation worklist walks.
        assert_eq!(
            substrate.recurrent_snapshot_entries().count(),
            0,
            "a distilled timeline's tail is still live in the substrate index, so \
             incremental maintenance will relocate it forward forever",
        );
        // And therefore also absent from the full-compaction live set.
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            !has_type(&live, RecordType::Snapshot),
            "a distilled timeline's recurrent snapshot survived compaction",
        );
    }

    /// A LIVE timeline keeps its tail — the retirement must key on the marker,
    /// not fire for every snapshot that shares the walk.
    #[test]
    fn an_undistilled_timelines_recurrent_snapshot_is_kept() {
        use crate::persistence::content_hash::snapshot_stream_id;

        let tl = 156u64;
        let snap = snapshot_payload(tl, 3, 0xAB);
        let blob = record(RecordType::Snapshot, snapshot_stream_id(tl).0, 0, &snap);

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(substrate.recurrent_snapshot_entries().count(), 1);
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(has_type(&live, RecordType::Snapshot));
    }

    /// **Branch checkpoints are capped; conversation snapshots are not.**
    ///
    /// The one record here that is a cache: a checkpoint is a pure function of a
    /// prompt still on disk, so dropping one costs a prefill on next use, while
    /// keeping one costs ~63 MiB in every future segment. Nothing else reclaims
    /// them — they are keyed by *content*, so editing the prompt supersedes
    /// nothing and orphans the old branch, and no timeline tombstone names it.
    /// Without the cap the log grows by one orphan per prompt edit, forever.
    ///
    /// The newest survive: the drop is oldest-first by write position.
    #[test]
    fn branch_checkpoints_are_capped_and_the_newest_survive() {
        use crate::persistence::content_hash::{branch_checkpoint_stream_id, ContentHash};

        let n = MAX_BRANCH_CHECKPOINTS + 3;
        let mut blob = Vec::new();
        let mut ids = Vec::new();
        for i in 0..n {
            let sid = branch_checkpoint_stream_id(ContentHash {
                lo: 1000 + i as u64,
                hi: 0,
            });
            ids.push(sid.0);
            // Payload contents are irrelevant to the cap — it reads the index.
            blob.extend_from_slice(&record(
                RecordType::BranchCheckpoint,
                sid.0,
                0,
                &snapshot_payload(0, 0, i as u8),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let kept: Vec<u64> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::BranchCheckpoint)
            .map(|it| it.header().stream_id)
            .collect();
        assert_eq!(
            kept.len(),
            MAX_BRANCH_CHECKPOINTS,
            "{n} checkpoints compacted to {} — the cap did not apply, and the log \
             grows without bound as the prompt is edited",
            kept.len()
        );
        // The three oldest are the ones dropped.
        for old in &ids[..3] {
            assert!(
                !kept.contains(old),
                "an older checkpoint survived while a newer one was dropped"
            );
        }
        for recent in &ids[3..] {
            assert!(
                kept.contains(recent),
                "a recent checkpoint was dropped while older ones survived — the \
                 branch a conversation is most likely to be using is the one that \
                 would have to be recomputed"
            );
        }
    }

    /// A checkpoint is not a conversation snapshot: distilling or tombstoning a
    /// timeline must not touch one, because it belongs to the *prompt* and
    /// outlives every conversation that used it.
    #[test]
    fn a_branch_checkpoint_survives_a_tombstoned_timeline() {
        use crate::persistence::content_hash::{
            branch_checkpoint_stream_id, snapshot_stream_id, ContentHash,
        };
        use crate::persistence::record::TombstonePayload;

        let tl = 141u64;
        let branch = branch_checkpoint_stream_id(ContentHash { lo: 77, hi: 0 });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(tl).0,
            0,
            &snapshot_payload(tl, 1, 0xAB),
        ));
        blob.extend_from_slice(&record(
            RecordType::BranchCheckpoint,
            branch.0,
            0,
            &snapshot_payload(0, 0, 0xCD),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            !has_type(&live, RecordType::Snapshot),
            "the tombstoned conversation's snapshot should be gone"
        );
        assert!(
            has_type(&live, RecordType::BranchCheckpoint),
            "the prompt's branch checkpoint was dropped with the conversation — \
             it belongs to the prompt, and every other conversation on that \
             branch would have to recompute it"
        );
    }

    /// A live timeline keeps exactly one snapshot — the newest. The header key
    /// is a synthetic per-timeline stream id, so the last writer wins by append
    /// order and the accounting *is* the tombstone; this pins that the surviving
    /// bytes are the SECOND write, not merely that one survived.
    #[test]
    fn a_live_timeline_keeps_exactly_its_newest_snapshot() {
        use crate::persistence::content_hash::snapshot_stream_id;

        let tl = 133u64;
        let first = snapshot_payload(tl, 0, 0x11);
        let second = snapshot_payload(tl, 1, 0x22);
        let sid = snapshot_stream_id(tl).0;

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Snapshot, sid, 0, &first));
        blob.extend_from_slice(&record(RecordType::Snapshot, sid, 0, &second));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        let snaps: Vec<_> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::Snapshot)
            .collect();
        assert_eq!(snaps.len(), 1, "exactly one snapshot survives per timeline");
        assert_eq!(
            snaps[0].header().payload_len as usize,
            second.len(),
            "the surviving snapshot must be the newest write"
        );
    }

    /// The text-only degree (archived conversations): keep the StreamDecl +
    /// tokens as a plain read-only record; drop the KV chunks, the WideQSig, and
    /// the projection events.
    #[test]
    fn text_only_distill_keeps_tokens_drops_sig_and_kv() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let tl = 91u64;
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: tl,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = 9191u64;
        let wide_payload =
            crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]);

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, sid, 0, b"kv-chunk-content"));
        blob.extend_from_slice(&record(RecordType::Tokens, sid, 0, b"token-content"));
        blob.extend_from_slice(&record(RecordType::WideQSig, sid, 0, &wide_payload));
        blob.extend_from_slice(&record(
            RecordType::ProjectionEvents,
            sid,
            0,
            b"proj-events",
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: tl,
                mode: DistillMode::TextOnly,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );

        assert!(
            has_type(&live, RecordType::StreamDecl),
            "text-only keeps its StreamDecl"
        );
        // Tokens are kept as a read-back item; read its source bytes to confirm
        // the readable record survives.
        let tokens = live
            .iter()
            .find(|it| it.header().record_type == RecordType::Tokens)
            .expect("text-only keeps its tokens (the readable record)");
        let (off, size) = raw_loc(tokens);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"token-content",
            "text-only keeps its tokens (the readable record)",
        );
        assert!(
            !has_type(&live, RecordType::Chunk),
            "text-only drops its KV chunks"
        );
        assert!(
            !has_type(&live, RecordType::WideQSig),
            "text-only drops its signature"
        );
        assert!(
            !has_type(&live, RecordType::ProjectionEvents),
            "text-only drops its projection events",
        );
        assert!(
            live.iter()
                .any(|it| it.header().record_type == RecordType::Distilled
                    && it
                        .synth_payload()
                        .and_then(|p| DistillPayload::decode(p).ok())
                        .map(|d| d.mode)
                        == Some(DistillMode::TextOnly)),
            "text-only marker re-emitted with its mode",
        );
    }

    /// The failure this guards against is silent and total: NPC payloads are
    /// not in this process's RAM, so nothing can re-synthesise them, and a
    /// compaction that does not carry them forward deletes every character in
    /// the store without an error, a warning, or a failing test anywhere else.
    /// An empty store, for tests whose subject is the NPC map rather than the
    /// log. `collect_live_records` is planning-only and reads no bytes, so the
    /// locations it echoes need not point at anything real.
    fn empty_store() -> (Manifest, Substrate) {
        let mut mem = MemLog::with_records(&[]);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        (manifest, substrate)
    }

    fn loc_at(offset: u64, size: u64) -> RecordLoc {
        RecordLoc {
            segment: FIRST_SEGMENT,
            offset,
            payload_len: size - 64,
            record_size: size,
        }
    }

    #[test]
    fn collect_carries_the_live_cast_forward() {
        let (manifest, substrate) = empty_store();
        // What the persistence layer tracks after a walk: one location per
        // character, already resolved to its greatest stamp.
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(701, Some(loc_at(4_096, 256)), RecordIdentity::NONE);
        npc_locs.offer(700, Some(loc_at(8_192, 320)), RecordIdentity::NONE);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let npcs: Vec<&CompactItem> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::Npc)
            .collect();

        assert_eq!(npcs.len(), 2, "both characters survive, neither duplicated");
        // Sorted by id despite the reverse insertion order, so the same inputs
        // always produce the same file.
        assert_eq!(npcs[0].header().stream_id, 700);
        assert_eq!(npcs[1].header().stream_id, 701);
        // Each stages the exact bytes the map named.
        assert_eq!(raw_loc(npcs[0]), (8_192, 320));
        assert_eq!(raw_loc(npcs[1]), (4_096, 256));
    }

    /// With no characters, nothing is emitted — the loop must not invent an
    /// empty record or disturb an existing store.
    #[test]
    fn collect_emits_nothing_for_an_empty_cast() {
        let (manifest, substrate) = empty_store();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(!has_type(&live, RecordType::Npc));
    }

    /// A deleted character keeps its record. The id must stay taken — acts it
    /// already committed still name it — so tombstoning is a superseding
    /// record, never a removal from the map.
    #[test]
    fn collect_carries_a_tombstoned_character_forward_too() {
        let (manifest, substrate) = empty_store();
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(900, Some(loc_at(4_096, 192)), RecordIdentity::NONE);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            live.iter().any(
                |it| it.header().record_type == RecordType::Npc && it.header().stream_id == 900
            ),
            "a tombstoned character is still a record",
        );
    }

    /// **A conversation's file events are carried while it lives, and only
    /// then**: a tombstoned, a distilled and an unregistered conversation's
    /// events are orphans and stay behind, and no tombstone is carried — what
    /// it killed is not carried either.
    #[test]
    fn collect_carries_only_live_conversations_file_events() {
        use crate::persistence::record::{DistillMode, DistillPayload};
        use crate::persistence::streams::{StreamDecl, TurnDecl};

        let decl = |timeline_id: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id,
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
            })
            .encode()
        };
        let (live_tl, dead_tl, distilled_tl, never_tl) = (11u64, 12u64, 13u64, 14u64);
        let mut blob = Vec::new();
        for (sid, tl) in [(801u64, live_tl), (802, dead_tl), (803, distilled_tl)] {
            blob.extend_from_slice(&record(RecordType::StreamDecl, sid, 0, &decl(tl)));
        }
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: distilled_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let ev = |seq| VfsEventId { origin: 0, seq };
        let mut vfs = VfsIndex::new();
        for tl in [live_tl, dead_tl, distilled_tl, never_tl] {
            vfs.record_event(tl, ev(0), loc_at(4_096 * tl, 128));
            vfs.record_event(tl, ev(1), loc_at(4_096 * tl + 1_024, 128));
        }
        vfs.record_tombstone(live_tl, ev(2), loc_at(90_000, 64), &[ev(0)]);

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &vfs,
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let events: Vec<(u64, u64, (u64, u64))> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::VfsEvent)
            .map(|it| (it.header().stream_id, it.header().chunk_index, raw_loc(it)))
            .collect();
        assert_eq!(events, vec![(live_tl, 1, (4_096 * live_tl + 1_024, 128))]);
        assert!(!has_type(&live, RecordType::VfsTombstone));
    }

    /// **A file store's events are carried though no conversation is
    /// registered on it** — declared by a record read back with the log, as
    /// every member reads it before its maintenance starts — and a declared
    /// timeline that is tombstoned, or an undeclared one, is left behind.
    #[test]
    fn collect_carries_a_declared_file_stores_events() {
        let (store_tl, dead_tl, undeclared_tl) = (21u64, 22u64, 23u64);
        let mut blob = Vec::new();
        for tl in [store_tl, dead_tl] {
            let declared = CustomObjectPayload::new(
                file_store_key(TimelineId::from_raw(tl).unwrap()),
                serde_json::Value::Null,
            );
            blob.extend_from_slice(&record(
                RecordType::CustomObject,
                declared.stream_id(),
                0,
                &declared.encode(),
            ));
        }
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let mut vfs = VfsIndex::new();
        for tl in [store_tl, dead_tl, undeclared_tl] {
            vfs.record_event(
                tl,
                VfsEventId { origin: 0, seq: 0 },
                loc_at(4_096 * tl, 128),
            );
        }
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &vfs,
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        let carried: Vec<u64> = live
            .iter()
            .filter(|it| it.header().record_type == RecordType::VfsEvent)
            .map(|it| it.header().stream_id)
            .collect();
        assert_eq!(carried, vec![store_tl]);
    }

    /// A tool round-trip's coupling survives compaction.
    ///
    /// This is the gate whose absence let every coupling in the store be
    /// deleted. `TurnCoupling` is payload-keyed, so `accounting` never credits a
    /// copy as dead and no location map relocates it: re-emitting from the
    /// substrate's live set is the *only* thing carrying it. And the loss is
    /// invisible — a timeline with no couplings is a legitimate state (nothing
    /// there ever called a tool), so the store reads as intact while every
    /// exchange has quietly come apart into independent turns.
    #[test]
    fn collect_carries_turn_couplings_forward() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let tl = 77u64;
        let decl = |turn_index: u32, seq: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index,
                turn_id_day: 0,
                turn_id_seq: seq,
                role: 2,
                block_start: 0,
                block_end: 2,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        // A three-turn repo_map chain (list → read → summarise): the first two
        // couple, joining all three into one exchange.
        let mut blob = Vec::new();
        for (i, sid) in [(0u32, 600u64), (1, 601), (2, 602)] {
            blob.extend_from_slice(&record(
                RecordType::StreamDecl,
                sid,
                0,
                &decl(i, sid as u32).encode(),
            ));
        }
        for from_turn in [0u32, 1] {
            blob.extend_from_slice(&record(
                RecordType::TurnCoupling,
                0,
                0,
                &TurnCouplingPayload {
                    timeline_id: tl,
                    from_turn,
                }
                .encode(),
            ));
        }

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Precondition: the walk really installed them, so a failure below is
        // compaction dropping them rather than replay never loading them.
        assert_eq!(
            substrate
                .live_couplings()
                .into_iter()
                .map(|(t, from, _)| (t, from))
                .collect::<Vec<_>>(),
            vec![(tl, 0), (tl, 1)],
            "replay must install both couplings",
        );

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        for from_turn in [0u32, 1] {
            assert!(
                has_synth(
                    &live,
                    RecordType::TurnCoupling,
                    &TurnCouplingPayload {
                        timeline_id: tl,
                        from_turn,
                    }
                    .encode(),
                ),
                "coupling {from_turn} must survive compaction — without it the \
                 call turn and its response become independent exchanges",
            );
        }
    }

    /// A tombstoned timeline's couplings go with it. They describe turns that no
    /// longer exist, so carrying them would make the marker set immortal.
    #[test]
    fn collect_drops_couplings_of_a_tombstoned_timeline() {
        let tl = 88u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::TurnCoupling,
            0,
            0,
            &TurnCouplingPayload {
                timeline_id: tl,
                from_turn: 0,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(!has_type(&live, RecordType::TurnCoupling));
    }

    /// **The guard that makes [`survival`] load-bearing.**
    ///
    /// Builds a store holding one record of every written type, compacts it, and
    /// requires that each type whose contract is `Relocated` or `Resident` comes
    /// out the other side. A type added to `RecordType` and to `survival` but not
    /// to `collect_live_records` fails here rather than deleting user data in
    /// production — which is exactly how `TurnCoupling`, `Npc`, and turn-scoped
    /// `Tombstone`s were lost: each compiled clean, passed every test, and the
    /// accounting classified it as neither live nor dead so nothing counted the
    /// loss either.
    ///
    /// [`survival`]: crate::persistence::survival::survival
    #[test]
    fn every_record_type_survives_compaction() {
        use crate::persistence::blob::BlobKey;
        use crate::persistence::content_hash::snapshot_stream_id;
        use crate::persistence::record::{
            CustomObjectPayload, DistillMode, DistillPayload, TreeMetadataPayload,
        };
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        use crate::persistence::survival::{survival, Survival, WRITTEN_RECORD_TYPES};
        use std::collections::BTreeMap;

        let live_tl = 1001u64;
        let distilled_tl = 1002u64;
        let custom: BTreeMap<String, String> = BTreeMap::new();
        let decl = |timeline_id: u64, turn_index: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id,
                turn_index,
                turn_id_day: 0,
                turn_id_seq: turn_index + 1,
                role: 2,
                block_start: 0,
                block_end: 2,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        let live_sid = 7001u64;
        let distilled_sid = 7002u64;

        let mut blob = Vec::new();
        // Singletons.
        blob.extend_from_slice(&record(RecordType::ModelKey, 0, 0, b"model"));
        blob.extend_from_slice(&record(RecordType::Template, 0, 0, b"template"));
        blob.extend_from_slice(&record(RecordType::Tokenizer, 0, 0, b"tokenizer"));
        // A live turn, with every per-stream record it can carry.
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            live_sid,
            0,
            &decl(live_tl, 0).encode(),
        ));
        blob.extend_from_slice(&record(RecordType::Chunk, live_sid, 0, b"kv"));
        blob.extend_from_slice(&record(RecordType::Tokens, live_sid, 0, b"tokens"));
        blob.extend_from_slice(&record(RecordType::Commit, live_sid, 0, b""));
        blob.extend_from_slice(&record(
            RecordType::ProjectionEvents,
            live_sid,
            0,
            b"events",
        ));
        blob.extend_from_slice(&record(RecordType::TurnIndexPage, live_sid, 0, b"page"));
        blob.extend_from_slice(&record(
            RecordType::WideQSig,
            live_sid,
            0,
            &crate::provenance::encode_wide_sigs(&[crate::provenance::WideQSig::from_band(
                &vec![1.0f32; 4 * 128],
                128,
            )]),
        ));
        // Per-timeline metadata. `ConvState` is emitted only for a state that
        // differs from an untouched one, so archive this one and give it a
        // branch.
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(live_tl, "conv", "Live", &custom),
        ));
        blob.extend_from_slice(&record(
            RecordType::ConvState,
            live_tl,
            0,
            &encode_conv_state_payload(
                live_tl,
                &ConvState {
                    archived: true,
                    branches: [("candle".to_string(), "zen/work".to_string())].into(),
                    active: 3,
                },
            ),
        ));
        blob.extend_from_slice(&record(
            RecordType::TreeMetadata,
            0,
            0,
            &TreeMetadataPayload {
                timeline_id: live_tl,
                turn_index: 0,
                kind: 0,
                tree_height: 0,
                children: Vec::new(),
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::DebugId,
            0,
            0,
            &DebugIdPayload {
                timeline_id: live_tl,
                debug_id: "resume-key".to_string(),
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::TurnCoupling,
            0,
            0,
            &TurnCouplingPayload {
                timeline_id: live_tl,
                from_turn: 0,
            }
            .encode(),
        ));
        // A turn-scoped tombstone — on a turn of the LIVE timeline, so it is a
        // dead turn inside a living conversation rather than a dead timeline.
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: live_tl,
                turn_index: Some(1),
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        // Recurrent state: a conversation snapshot and a prompt-branch checkpoint.
        blob.extend_from_slice(&record(
            RecordType::Snapshot,
            snapshot_stream_id(live_tl).0,
            0,
            b"snapshot",
        ));
        blob.extend_from_slice(&record(RecordType::BranchCheckpoint, 9_001, 0, b"branch"));
        // A separate distilled timeline, so its marker is emitted without
        // shedding the live turn's content.
        blob.extend_from_slice(&record(
            RecordType::StreamDecl,
            distilled_sid,
            0,
            &decl(distilled_tl, 0).encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Distilled,
            0,
            0,
            &DistillPayload {
                timeline_id: distilled_tl,
                mode: DistillMode::ProvenanceOnly,
            }
            .encode(),
        ));
        // A host-defined keyed object — the key's hash rides in the header
        // `stream_id`, exactly as `put_custom_object` writes it.
        let custom_obj = CustomObjectPayload::new(
            "command-table:vault",
            serde_json::json!({
                "open": true
            }),
        );
        blob.extend_from_slice(&record(
            RecordType::CustomObject,
            custom_obj.stream_id(),
            0,
            &custom_obj.encode(),
        ));
        // A tombstoned section stream — the section counterpart of the
        // turn-scoped `Tombstone` above.
        blob.extend_from_slice(&record(
            RecordType::SectionTombstone,
            9_501,
            0,
            &SectionTombstonePayload {
                generation: None,
                reason: None,
            }
            .encode(),
        ));
        // A dead timeline's tombstone, and the swept set naming it — a
        // compaction carries both while it is young.
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: 1003,
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
        blob.extend_from_slice(&record(
            RecordType::Deleted,
            0,
            0,
            &DeletedPayload {
                swept: vec![1003],
                ..DeletedPayload::default()
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Characters are tracked persistence-side, not in the substrate.
        let mut npc_locs = StampedLocs::new();
        npc_locs.offer(8812, Some(loc_at(4_096, 256)), RecordIdentity::NONE);
        // So are a conversation's file events and their tombstones.
        let mut vfs = VfsIndex::new();
        // The tombstone kills another member's event, so it is carried.
        vfs.record_event(
            live_tl,
            VfsEventId { origin: 0, seq: 0 },
            loc_at(8_192, 128),
        );
        vfs.record_tombstone(
            live_tl,
            VfsEventId { origin: 0, seq: 1 },
            loc_at(12_288, 64),
            &[VfsEventId {
                origin: 0xB,
                seq: 0,
            }],
        );

        // And blobs, by location.
        let mut blobs = BlobIndex::new();
        blobs.record(BlobKey { hi: 3, lo: 4 }, loc_at(16_384, 512));

        // What the member has applied of another origin, and the generation
        // fixed for what it wrote cut off — held persistence-side.
        let mut housekeeping = Housekeeping::default();
        housekeeping.replica.applied.insert(0xB, 1);
        housekeeping.rebases.add(RebaseEntry {
            origin: 0xA,
            from: 4,
            through: 9,
            gen: 2,
        });
        let live = collect_live_records(
            &manifest,
            &substrate,
            &npc_locs,
            &vfs,
            &blobs,
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &housekeeping,
        );

        for &rt in WRITTEN_RECORD_TYPES {
            match survival(rt) {
                Survival::Relocated | Survival::Resident => assert!(
                    has_type(&live, rt),
                    "{rt:?} declares survival::{:?} but compaction emitted no such \
                     record — it would be deleted from the store on the first pass",
                    survival(rt),
                ),
                // Derived: every copy is dropped and the writer rebuilds the
                // chain in the new file, so its absence here is correct.
                Survival::Regenerated => assert!(
                    !has_type(&live, rt),
                    "{rt:?} is regenerated by the writer; carrying a stale copy \
                     forward would corrupt the new chain",
                ),
                Survival::NeverWritten => unreachable!("{rt:?} is in WRITTEN_RECORD_TYPES"),
            }
        }

        // The conversation state comes through whole — its branches too, not
        // just the archived flag — keyed in the header by its timeline.
        let state = ConvState {
            archived: true,
            branches: [("candle".to_string(), "zen/work".to_string())].into(),
            active: 3,
        };
        assert!(has_synth(
            &live,
            RecordType::ConvState,
            &encode_conv_state_payload(live_tl, &state)
        ));
        assert!(live.iter().any(|it| {
            it.header().record_type == RecordType::ConvState && it.header().stream_id == live_tl
        }));
    }

    /// Turn-scoped tombstones survive. The reload's placeholder logic keys off
    /// the marker to restore the dead turn as an empty hole; without it the turn
    /// reads as content-declaring with no content.
    #[test]
    fn collect_carries_turn_scoped_tombstones_forward() {
        let tl = 99u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: tl,
                turn_index: Some(3),
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        assert!(
            has_synth(
                &live,
                RecordType::Tombstone,
                &TombstonePayload {
                    timeline_id: tl,
                    turn_index: Some(3),
                    turn_writer: 0,
                    reason: None,
                }
                .encode(),
            ),
            "a turn-scoped tombstone must survive, or the dead turn returns",
        );
    }

    /// A section tombstone survives compaction — the section counterpart of
    /// [`collect_carries_turn_scoped_tombstones_forward`]. Losing it would
    /// resurrect the corrupted chunks the reactive repair tombstoned.
    #[test]
    fn collect_carries_section_tombstones_forward() {
        let stream_id = 55_001u64;
        let generation = SectionGeneration { writer: 4, oseq: 9 };
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::SectionTombstone,
            stream_id,
            0,
            &SectionTombstonePayload {
                generation: Some(generation),
                reason: Some(
                    "chunk 4: (offset 0, 32 tokens) on layer(s) [0]; (offset 0, 0 \
                              tokens) on layer(s) [1..47]"
                        .to_string(),
                ),
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(
            substrate.is_section_tombstoned(StreamId(stream_id)),
            "replay must mark the stream tombstoned",
        );

        let live = collect_live_records(
            &manifest,
            &substrate,
            &StampedLocs::new(),
            &VfsIndex::new(),
            &BlobIndex::new(),
            &MarkerIds::new(),
            0,
            NO_EXPIRY,
            &Housekeeping::default(),
        );
        // The reason is diagnostic only and is not re-derived at compaction
        // time — the same rule the turn-scoped tombstone's re-emit follows —
        // so only the marker's presence (not its original reason text) is
        // pinned here.
        assert!(
            has_synth(
                &live,
                RecordType::SectionTombstone,
                &SectionTombstonePayload {
                    generation: Some(generation),
                    reason: None,
                }
                .encode(),
            ),
            "a section tombstone must survive, or the corrupted chunks return",
        );
    }
}

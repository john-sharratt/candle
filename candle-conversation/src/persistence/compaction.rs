//! Store compaction — reclaim dead weight by rewriting only the **live**
//! records into a fresh segment (`docs/segmented_substrate_log.md`).
//!
//! An append-only store only grows: every superseded partial-tail snapshot,
//! every stale `ModelSpec` / `Template`, every chunk of a deleted stream is
//! dead weight the skip-load walk still steps over. Compaction keeps only the
//! live records — the winners under last-writer-wins that the manifest and the
//! substrate index already resolve.
//!
//! [`collect_live_records`] plans the live set in dependency order (`ModelSpec`,
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

use std::path::Path;

use super::header_index::{encode_index_payload, IndexEntry, INDEX_FLUSH_ENTRIES};
use super::log_file::LogFile;
use super::manifest::{encode_conv_state_payload, ConvState, Manifest, RecordLoc};
use super::record::{
    encode_record, DebugIdPayload, DistillMode, DistillPayload, RecordHeader, RecordType,
};
use super::segment::{SegmentId, FIRST_SEGMENT};
use super::Result;
use crate::substrate::Substrate;

/// Bytes staged into the compacted log before an incremental flush to disk —
/// bounds the write buffer so a whole-store rewrite never holds the entire new
/// log in RAM at once.
const COMPACT_FLUSH_BYTES: usize = 256 * 1024 * 1024;

/// One record destined for the compacted log. `Raw` records (`Chunk` / `Tokens`
/// / the singletons) are staged **verbatim** from their source segment — no
/// decode, no CRC re-verify, no re-encode — via coalesced stripe reads; their
/// header is synthesized from the substrate/manifest index (never read from
/// disk) for the header-index digest + singleton manifest. `Synth` records are
/// the small resident metadata, encoded fresh from in-RAM state.
pub enum CompactItem {
    Raw {
        header: RecordHeader,
        segment: SegmentId,
        offset: u64,
        record_size: u64,
    },
    Synth {
        header: RecordHeader,
        payload: Vec<u8>,
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

    fn synth(header: RecordHeader, payload: Vec<u8>) -> CompactItem {
        CompactItem::Synth { header, payload }
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
            CompactItem::Synth { payload, .. } => Some(payload),
            CompactItem::Raw { .. } => None,
        }
    }
}

/// Header for a singleton (`ModelSpec` / `Template` / `Tokenizer`) staged
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

/// Collect every **live** record, in dependency order, as [`CompactItem`]s —
/// planning only, **no disk reads**. `ModelSpec` / `Template` / `Tokenizer` /
/// `Chunk` / `Tokens` become `Raw` items carrying their source `(segment,
/// offset, record_size)` and a header synthesized from the substrate/manifest
/// index; every resident record (`StreamDecl`, `Commit`, `Label`, `ConvState`,
/// `ProjectionEvents`, `WideQSig`, `TreeMetadata`, `DebugId`, `Distilled`) is a
/// `Synth` item carrying its freshly-encoded payload. [`write_compacted_log`]
/// reads the `Raw` bytes back coalesced and stages them verbatim.
pub fn collect_live_records(manifest: &Manifest, substrate: &Substrate) -> Vec<CompactItem> {
    let mut out: Vec<CompactItem> = Vec::new();

    // Singletons — staged verbatim from wherever they physically live.
    for (rt, loc) in [
        (RecordType::ModelSpec, manifest.model_spec),
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
    let distilled: std::collections::HashMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();

    // Per-stream live records — sourced from the substrate's in-RAM
    // stream index, the authoritative source.
    for (stream_id, entry) in substrate.all_streams() {
        // Distillation degree for this turn's timeline (None = not distilled).
        // Both modes drop the KV chunks (the bulk of the on-disk cost);
        // ProvenanceOnly also drops tokens (keep sig + projections — the belief
        // gallery reads only the sig), TextOnly also drops sig + projections
        // (keep tokens — a plain read-only text record). Tombstoned timelines
        // drop entirely, as before.
        let distill: Option<DistillMode> =
            if let Some(crate::persistence::streams::StreamDecl::Turn(t)) = &entry.decl {
                if tombstoned.contains(&t.timeline_id) {
                    continue;
                }
                distilled.get(&t.timeline_id).copied()
            } else {
                None
            };
        let keep_chunks = distill.is_none();
        let keep_tokens = distill != Some(DistillMode::ProvenanceOnly);
        let keep_sig = distill != Some(DistillMode::TextOnly);
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
                },
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
                    },
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
                    },
                    payload.clone(),
                ));
            }
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
                },
                Vec::new(),
            ));
        }
    }
    // Per-timeline Label / ConvState records — synthesised from
    // the substrate's live state.  Tombstoned timelines are
    // skipped so retired conversations don't leave dangling
    // sidebar entries on disk.
    for (timeline_id, conv_id, label, archived, custom) in substrate.live_conv_meta() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = super::manifest::encode_label_payload(timeline_id, &conv_id, &label, &custom);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Label,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
            },
            payload,
        ));
        if archived {
            let cs_payload = encode_conv_state_payload(timeline_id, ConvState { archived: true });
            out.push(CompactItem::synth(
                RecordHeader {
                    record_type: RecordType::ConvState,
                    format: 0,
                    payload_len: cs_payload.len() as u64,
                    crc: 0,
                    stream_id: 0,
                    chunk_index: 0,
                    token_count: 0,
                },
                cs_payload,
            ));
        }
    }
    // Per-(timeline, turn) summary-tree metadata — emit one record
    // per live tree node directly from substrate state.
    for payload in substrate.live_tree_metadata_payloads() {
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
            },
            bytes,
        ));
    }
    // Per-timeline debug_id.
    for (timeline_id, id) in substrate.live_debug_ids() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = DebugIdPayload {
            timeline_id,
            debug_id: id,
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
            },
            bytes,
        ));
    }
    // Per-distilled-timeline marker — re-emitted (with mode) so a distilled
    // timeline stays distilled across the compaction it just shed through.
    // Without this the marker is lost and, e.g., a text-only archived
    // conversation would read back as un-distilled. Tombstoned timelines are
    // gone, so skip them.
    for (&timeline_id, &mode) in &distilled {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = DistillPayload { timeline_id, mode }.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Distilled,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
            },
            payload,
        ));
    }
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
pub fn write_compacted_log(
    path: &Path,
    items: &[CompactItem],
    read_into: &mut dyn FnMut(SegmentId, u64, &mut [u8]) -> Result<()>,
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
    // list). The whole run is read in **one coalesced stripe** and each record
    // staged straight from the read buffer: one copy per record, no per-record
    // allocation, no whole-store intermediate buffer.
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0;
    while i < items.len() {
        match &items[i] {
            CompactItem::Synth { header, payload } => {
                let mut h = *header;
                h.payload_len = payload.len() as u64;
                let bytes = encode_record(&h, payload);
                let offset = log.stage(&bytes);
                pending.push(IndexEntry::from_header(&h, offset, bytes.len() as u64));
                i += 1;
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
                let len = (end - start) as usize;
                if buf.len() < len {
                    buf.resize(len, 0);
                }
                read_into(seg, start, &mut buf[..len])?;
                let mut within = 0usize;
                for it in &items[i..j] {
                    let CompactItem::Raw {
                        header,
                        record_size,
                        ..
                    } = it
                    else {
                        unreachable!("a Raw run holds only Raw items");
                    };
                    let sz = *record_size as usize;
                    let staged = log.stage(&buf[within..within + sz]);
                    within += sz;
                    // Singletons resolve from the manifest (the substrate replay
                    // never rebuilds them); `Chunk` / `Tokens` are substrate-
                    // indexed and never enter the manifest.
                    let loc = RecordLoc {
                        segment: FIRST_SEGMENT,
                        offset: staged,
                        payload_len: header.payload_len,
                        record_size: sz as u64,
                    };
                    match header.record_type {
                        RecordType::ModelSpec => manifest.model_spec = Some(loc),
                        RecordType::Template => manifest.template = Some(loc),
                        RecordType::Tokenizer => manifest.tokenizer = Some(loc),
                        _ => {}
                    }
                    pending.push(IndexEntry::from_header(header, staged, sz as u64));
                    if pending.len() >= INDEX_FLUSH_ENTRIES {
                        flush_index(&mut log, &mut pending, &mut last_index);
                    }
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
    use crate::persistence::log_file::{read_record_at, LogSource, MemLog, SUPERBLOCK_SIZE};
    use crate::persistence::record::encode_record;

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
            },
            payload,
        )
    }

    #[test]
    fn collect_keeps_only_the_live_winners() {
        let mut blob = Vec::new();
        // A stale ModelSpec, then the live one.
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"model-v1-stale"));
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"model-v2-live"));
        // A stream: a 20-token partial chunk, then the sealed winner.
        blob.extend_from_slice(&record(RecordType::Chunk, 5, 0, b"partial-20tok-dead"));
        blob.extend_from_slice(&record(RecordType::Chunk, 5, 0, b"sealed-final-live"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(&manifest, &substrate);
        // Exactly: 1 ModelSpec + 1 Chunk = 2 (no StreamDecl/Commit/Tokens here).
        assert_eq!(live.len(), 2);
        // Each read-back item points at the live winner's on-disk location — the
        // manifest / substrate index already resolved last-writer-wins — so the
        // stale ModelSpec and the dead partial chunk are never staged. Read the
        // chosen source bytes back to prove which record was selected.
        let model = live
            .iter()
            .find(|it| it.header().record_type == RecordType::ModelSpec)
            .unwrap();
        let (off, size) = raw_loc(model);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"model-v2-live",
            "the stale ModelSpec is dropped"
        );
        let chunk = live
            .iter()
            .find(|it| it.header().record_type == RecordType::Chunk)
            .unwrap();
        let (off, size) = raw_loc(chunk);
        assert_eq!(
            read_record_at(&mut mem, off, size).unwrap().payload,
            b"sealed-final-live",
            "the dead partial chunk is dropped"
        );
    }

    #[test]
    fn compacted_log_recovers_to_an_identical_live_set() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"m1"));
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"m2"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-old"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"c0-new"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 1, b"c1"));
        blob.extend_from_slice(&record(RecordType::Tokens, 7, 0, b"tokens"));
        let mut mem = MemLog::with_records(&blob);
        let (before, before_sub, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(&before, &before_sub);
        let path = std::env::temp_dir().join(format!(
            "kvtier_compact_{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (mut new_log, _after_manifest) =
            write_compacted_log(&path, &live, &mut |_s, off, dest| mem.read_into(off, dest))
                .unwrap();

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
                reason: None,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(&manifest, &substrate);

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
        let live = collect_live_records(&manifest, &substrate);

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
        let live = collect_live_records(&manifest, &substrate);

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
        let live = collect_live_records(&manifest, &substrate);

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
        let live = collect_live_records(&manifest, &substrate);

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
}

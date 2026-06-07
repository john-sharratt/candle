//! Log compaction — the whole-file dead-record rewrite (§5.8 of
//! `docs/kv_tier_migration.md`).
//!
//! An append-only log only grows: every superseded partial-tail snapshot,
//! every stale `ModelSpec` / `Template`, every chunk of a deleted stream is
//! dead weight the skip-load walk still steps over. Compaction rebuilds the
//! log keeping only the **live** records — the winners under last-writer-wins
//! that the manifest already resolved.
//!
//! The live set is streamed into a fresh file in dependency order
//! (`ModelSpec`, `Template`, then per stream its `StreamDecl`, `Chunk`s,
//! `Tokens`, `Signatures`, `Commit`), closed with a fresh `Checkpoint`. The
//! orchestration that swaps the new file in is [`SubstratePersistence::compact`].

use std::path::Path;

use super::checkpoint;
use super::log_file::{read_record_at, LogFile, LogSource, SUPERBLOCK_SIZE};
use super::manifest::{encode_conv_state_payload, ConvState, Manifest};
use super::record::{encode_record, DebugIdPayload, Record, RecordHeader, RecordType};
use super::walker::{self, WalkEntry};
use super::Result;
use crate::substrate::Substrate;

/// Collect every **live** record from `log`, in dependency order, as
/// `(header, payload)` pairs ready to re-encode.
///
/// `ModelSpec` / `Template` / `Chunk` / `Tokens` / `Signatures` are read
/// back from the log at their manifest offsets; `StreamDecl` and `Commit`
/// are synthesised from the manifest (the manifest holds the decoded
/// declaration and the durable-through index directly).
pub fn collect_live_records(
    log: &mut dyn LogSource,
    manifest: &Manifest,
    substrate: &Substrate,
) -> Result<Vec<(RecordHeader, Vec<u8>)>> {
    let mut out: Vec<(RecordHeader, Vec<u8>)> = Vec::new();

    if let Some(loc) = manifest.model_spec {
        let r = read_record_at(log, loc.offset, loc.record_size)?;
        out.push((r.header, r.payload));
    }
    if let Some(loc) = manifest.template {
        let r = read_record_at(log, loc.offset, loc.record_size)?;
        out.push((r.header, r.payload));
    }
    if let Some(loc) = manifest.tokenizer {
        let r = read_record_at(log, loc.offset, loc.record_size)?;
        out.push((r.header, r.payload));
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

    // Per-stream live records — sourced from the substrate's in-RAM
    // stream index, the authoritative source.
    for (stream_id, entry) in substrate.all_streams() {
        if let Some(crate::persistence::streams::StreamDecl::Turn(t)) = &entry.decl {
            if tombstoned.contains(&t.timeline_id) {
                continue;
            }
        }
        if let Some(decl) = &entry.decl {
            let payload = decl.encode();
            out.push((
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
        for loc in entry.chunks.values() {
            let r = read_record_at(log, loc.offset, loc.record_size)?;
            out.push((r.header, r.payload));
        }
        if let Some(loc) = entry.tokens {
            let r = read_record_at(log, loc.offset, loc.record_size)?;
            out.push((r.header, r.payload));
        }
        if let Some(loc) = entry.signatures {
            let r = read_record_at(log, loc.offset, loc.record_size)?;
            out.push((r.header, r.payload));
        }
        if let Some(through) = entry.committed_through {
            out.push((
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
    for (timeline_id, conv_id, label, archived) in substrate.live_conv_meta() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = super::manifest::encode_label_payload(timeline_id, &conv_id, &label);
        out.push((
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
            let cs_payload = encode_conv_state_payload(
                timeline_id,
                ConvState { archived: true },
            );
            out.push((
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
        out.push((
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
        out.push((
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
    Ok(out)
}

/// Write the live records to a fresh log at `path`, closed with a
/// `Checkpoint`. Returns the open [`LogFile`] and its manifest. `path` is
/// removed first if it already exists.
pub fn write_compacted_log(
    path: &Path,
    live: &[(RecordHeader, Vec<u8>)],
) -> Result<(LogFile, Manifest)> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    let mut log = LogFile::create(path)?;
    let mut manifest = Manifest::new();

    for (header, payload) in live {
        let mut h = *header;
        h.payload_len = payload.len() as u64;
        let bytes = encode_record(&h, payload);
        let offset = log.stage(&bytes);
        manifest.ingest(&WalkEntry {
            offset,
            record: Record {
                header: h,
                payload: payload.clone(),
            },
            size: bytes.len() as u64,
        })?;
    }
    log.commit()?;

    // Close with a fresh Checkpoint over the just-built manifest.
    let ckpt_payload = checkpoint::encode_checkpoint(&manifest);
    let ckpt_header = RecordHeader {
        record_type: RecordType::Checkpoint,
        format: 0,
        payload_len: ckpt_payload.len() as u64,
        crc: 0,
        stream_id: 0,
        chunk_index: 0,
        token_count: 0,
    };
    let ckpt_offset = log.stage(&encode_record(&ckpt_header, &ckpt_payload));
    log.commit()?;
    log.set_latest_checkpoint(ckpt_offset)?;
    manifest.last_checkpoint_offset = Some(ckpt_offset);

    Ok((log, manifest))
}

/// Fraction of the log's records that are dead weight — the heuristic that
/// drives the automatic compaction trigger. `0.0` = nothing to reclaim.
pub fn dead_record_ratio(
    log: &mut dyn LogSource,
    manifest: &Manifest,
    substrate: &Substrate,
) -> Result<f32> {
    let (entries, _) = walker::collect(log, SUPERBLOCK_SIZE)?;
    let total = entries.len();
    if total == 0 {
        return Ok(0.0);
    }
    // Live on-disk records: read-back records (model/template/chunk/
    // tokens/signatures) — synthesised entries (StreamDecl, Commit,
    // Label, ConvState, TreeMetadata, DebugId) are not on disk in the
    // active log; they're re-emitted from substrate state during
    // compaction.
    let live_on_disk = collect_live_records(log, manifest, substrate)?
        .iter()
        .filter(|(h, _)| {
            !matches!(
                h.record_type,
                RecordType::StreamDecl
                    | RecordType::Commit
                    | RecordType::Label
                    | RecordType::ConvState
                    | RecordType::TreeMetadata
                    | RecordType::DebugId
            )
        })
        .count()
        + substrate
            .all_streams()
            .filter(|(_, s)| s.decl.is_some())
            .count()
        + substrate
            .all_streams()
            .filter(|(_, s)| s.committed_through.is_some())
            .count()
        + substrate.live_conv_meta().len()
        + substrate.live_tree_metadata_payloads().len()
        + substrate.live_debug_ids().len();
    let dead = total.saturating_sub(live_on_disk.min(total));
    Ok(dead as f32 / total as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::MemLog;
    use crate::persistence::record::encode_record;

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

        let live = collect_live_records(&mut mem, &manifest, &substrate).unwrap();
        // Exactly: 1 ModelSpec + 1 Chunk = 2 (no StreamDecl/Commit/Tokens here).
        assert_eq!(live.len(), 2);
        let model = live
            .iter()
            .find(|(h, _)| h.record_type == RecordType::ModelSpec)
            .unwrap();
        assert_eq!(model.1, b"model-v2-live", "the stale ModelSpec is dropped");
        let chunk = live
            .iter()
            .find(|(h, _)| h.record_type == RecordType::Chunk)
            .unwrap();
        assert_eq!(
            chunk.1, b"sealed-final-live",
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

        let live = collect_live_records(&mut mem, &before, &before_sub).unwrap();
        let path = std::env::temp_dir().join(format!(
            "kvtier_compact_{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (mut new_log, _after_manifest) = write_compacted_log(&path, &live).unwrap();

        // The compacted log re-walked into a substrate has the same
        // live streams + chunks as the source.
        let hint = new_log.superblock().latest_checkpoint_offset;
        let mut after_sub = Substrate::new();
        let _ = checkpoint::recover_with_sink(&mut new_log, hint, |e| {
            after_sub.apply_walker_entry(e)
        })
        .unwrap();
        assert_eq!(before_sub.live_chunk_count(), after_sub.live_chunk_count());
        assert_eq!(after_sub.live_chunk_count(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dead_ratio_is_zero_for_a_clean_log() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"only"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(
            dead_record_ratio(&mut mem, &manifest, &substrate).unwrap(),
            0.0
        );
    }

    #[test]
    fn dead_ratio_rises_with_superseded_records() {
        let mut blob = Vec::new();
        for i in 0..8u32 {
            // Eight writes to the same key — seven are dead.
            blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, format!("v{i}").as_bytes()));
        }
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let ratio = dead_record_ratio(&mut mem, &manifest, &substrate).unwrap();
        assert!(ratio > 0.8, "7 of 8 records are dead, got ratio {ratio}");
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
            scores: super::super::streams::PerDepthScores::default(),
            user_chunk_count: 0,
            user_token_count: 0,
            user_sig_count: 0,
            user_text: String::new(),
            assistant_text: String::new(),
        };
        let dead_decl = StreamDecl::Turn(turn_decl(dead_tl));
        let alive_decl = StreamDecl::Turn(turn_decl(alive_tl));
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(dead_tl, "dead-conv", "Dead"),
        ));
        blob.extend_from_slice(&record(
            RecordType::Label,
            0,
            0,
            &super::super::manifest::encode_label_payload(alive_tl, "alive-conv", "Alive"),
        ));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 100, 0, &dead_decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 100, 0, b"dead-chunk-payload"));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 200, 0, &alive_decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 200, 0, b"alive-chunk-payload"));
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload {
                timeline_id: dead_tl,
            }
            .encode(),
        ));

        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let live = collect_live_records(&mut mem, &manifest, &substrate).unwrap();

        // The dead timeline's records are physically gone from the
        // live set.
        assert!(
            !live.iter().any(|(_, p)| p == b"dead-chunk-payload"),
            "tombstoned timeline's chunk must be dropped during compaction",
        );
        assert!(
            !live.iter().any(|(_, p)| {
                std::str::from_utf8(p)
                    .map(|s| s.contains("dead-conv"))
                    .unwrap_or(false)
            }),
            "tombstoned timeline's Label must be dropped during compaction",
        );
        // The alive timeline's records survive intact.
        assert!(
            live.iter().any(|(_, p)| p == b"alive-chunk-payload"),
            "alive timeline's chunk must survive compaction",
        );
        assert!(
            live.iter().any(|(_, p)| {
                std::str::from_utf8(p)
                    .map(|s| s.contains("alive-conv"))
                    .unwrap_or(false)
            }),
            "alive timeline's Label must survive compaction",
        );
    }
}

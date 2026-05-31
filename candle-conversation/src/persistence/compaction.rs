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
use super::manifest::Manifest;
use super::record::{encode_record, Record, RecordHeader, RecordType};
use super::walker::{self, WalkEntry};
use super::Result;

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

    for (stream_id, entry) in &manifest.streams {
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
    // Synthesised — per-timeline conv metadata the manifest holds decoded.
    // One record per surviving entry; last-write-wins semantics mean the
    // manifest already holds the canonical winner per timeline.
    for (timeline_id, meta) in &manifest.labels {
        let payload =
            super::manifest::encode_label_payload(*timeline_id, &meta.conv_id, &meta.label);
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
    }
    // Per-timeline lifecycle state — same last-write-wins shape as
    // Label. One record per timeline whose archive state has ever
    // been touched; default-false states aren't written (the
    // `write_conv_state` no-op gate plus the default `archived=false`
    // mean an untouched timeline carries no ConvState record at all).
    for (timeline_id, state) in &manifest.conv_states {
        let payload = super::manifest::encode_conv_state_payload(*timeline_id, *state);
        out.push((
            RecordHeader {
                record_type: RecordType::ConvState,
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
pub fn dead_record_ratio(log: &mut dyn LogSource, manifest: &Manifest) -> Result<f32> {
    let (entries, _) = walker::collect(log, SUPERBLOCK_SIZE)?;
    let total = entries.len();
    if total == 0 {
        return Ok(0.0);
    }
    // Live on-disk records: read-back records (model/template/chunk/tokens/
    // signatures) — the StreamDecl/Commit/Label entries collect_live_records
    // synthesises are excluded from the "live on disk" count.
    let live_on_disk = collect_live_records(log, manifest)?
        .iter()
        .filter(|(h, _)| {
            !matches!(
                h.record_type,
                RecordType::StreamDecl
                    | RecordType::Commit
                    | RecordType::Label
                    | RecordType::ConvState
            )
        })
        .count()
        + manifest
            .streams
            .values()
            .filter(|s| s.decl.is_some())
            .count()
        + manifest
            .streams
            .values()
            .filter(|s| s.committed_through.is_some())
            .count()
        + manifest.labels.len()
        + manifest.conv_states.len();
    let dead = total.saturating_sub(live_on_disk.min(total));
    Ok(dead as f32 / total as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::MemLog;
    use crate::persistence::record::encode_record;
    use crate::persistence::streams::StreamId;

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
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(&mut mem, &manifest).unwrap();
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
        let (before, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let live = collect_live_records(&mut mem, &before).unwrap();
        let path = std::env::temp_dir().join(format!(
            "kvtier_compact_{}.log",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (mut new_log, after) = write_compacted_log(&path, &live).unwrap();

        // The compacted manifest has the same live streams + chunks.
        assert_eq!(before.streams.len(), after.streams.len());
        assert_eq!(
            before.streams[&StreamId(7)].chunks.len(),
            after.streams[&StreamId(7)].chunks.len(),
        );
        // And it recovers cleanly from disk.
        let hint = new_log.superblock().latest_checkpoint_offset;
        let recovered = checkpoint::recover(&mut new_log, hint).unwrap();
        assert_eq!(recovered.manifest.streams.len(), 1);
        assert_eq!(recovered.manifest.live_chunk_count(), 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dead_ratio_is_zero_for_a_clean_log() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"only"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(dead_record_ratio(&mut mem, &manifest).unwrap(), 0.0);
    }

    #[test]
    fn dead_ratio_rises_with_superseded_records() {
        let mut blob = Vec::new();
        for i in 0..8u32 {
            // Eight writes to the same key — seven are dead.
            blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, format!("v{i}").as_bytes()));
        }
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let ratio = dead_record_ratio(&mut mem, &manifest).unwrap();
        assert!(ratio > 0.8, "7 of 8 records are dead, got ratio {ratio}");
    }
}

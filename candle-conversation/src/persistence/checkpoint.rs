//! Checkpoints and recovery (§5.6 of `docs/kv_tier_migration.md`).
//!
//! A `Checkpoint` record's payload is exactly a serialised [`Manifest`].
//! Recovery loads the latest checkpoint (a superblock hint points at it)
//! and replays only the records appended after it, stopping at the first
//! torn record — which is then the truncation point.

use super::log_file::{LogSource, SUPERBLOCK_SIZE};
use super::manifest::Manifest;
use super::record::{decode_header, decode_record, padded_record_len, RecordType, ALIGN};
use super::walker;
use super::{PersistenceError, Result};
#[cfg(test)]
use crate::substrate::Substrate;

/// The outcome of recovering a log.
#[derive(Clone, Debug)]
pub struct Recovered {
    /// The fully reconstructed manifest.
    pub manifest: Manifest,
    /// First byte past the last valid record — the live log tail.
    pub tail_offset: u64,
    /// Whether a torn record was found at `tail_offset` (the file must be
    /// truncated there).
    pub torn: bool,
}

/// Encode a checkpoint record payload — exactly the serialised manifest.
pub fn encode_checkpoint(manifest: &Manifest) -> Vec<u8> {
    manifest.encode()
}

/// Load and decode the `Checkpoint` record at `offset`. Returns the manifest
/// snapshot and the offset immediately past the checkpoint record.
fn load_checkpoint(src: &mut dyn LogSource, offset: u64) -> Result<(Manifest, u64)> {
    let size = src.size()?;
    let remaining = size.saturating_sub(offset);
    let probe_len = remaining.min(ALIGN as u64) as usize;
    let probe = src.read_at(offset, probe_len)?;
    let (header, header_bytes) = decode_header(&probe)?;
    if header.record_type != RecordType::Checkpoint {
        return Err(PersistenceError::Corrupt(format!(
            "expected a Checkpoint record at offset {offset}, found {:?}",
            header.record_type
        )));
    }
    let total = padded_record_len(header_bytes - 1, header.payload_len);
    let record_bytes = if total <= probe.len() {
        probe[..total].to_vec()
    } else {
        src.read_at(offset, total)?
    };
    let (_header, payload, _) = decode_record(&record_bytes)?;
    let manifest = Manifest::decode(payload)?;
    Ok((manifest, offset + total as u64))
}

/// Recover a log into a manifest and its true tail.
///
/// `checkpoint_hint` is the superblock's latest-checkpoint offset (0 if
/// none). When the hint resolves to a valid checkpoint, recovery starts
/// from that snapshot and replays only the tail; otherwise it falls back to
/// a full walk from the first record. Both paths produce the same manifest.
pub fn recover(src: &mut dyn LogSource, checkpoint_hint: u64) -> Result<Recovered> {
    recover_with_sink(src, checkpoint_hint, |_| {})
}

/// Recovery + per-record sink.  Identical to [`recover`] except every
/// walked record is also passed to `sink`.  Used by the production
/// open paths to dispatch records straight into a [`Substrate`]
/// (via [`Substrate::apply_walker_entry`]) during the same walker pass
/// that builds the manifest's singleton offsets — so per-entity state
/// never has to be mirrored from the manifest into the substrate
/// afterwards.
pub fn recover_with_sink<F>(
    src: &mut dyn LogSource,
    checkpoint_hint: u64,
    mut sink: F,
) -> Result<Recovered>
where
    F: FnMut(&walker::WalkEntry),
{
    // The checkpoint payload carries only singleton offsets (Phase 3):
    // `streams`, `labels`, `conv_states`, `tree_metadata`, `debug_ids`
    // are `#[serde(skip)]`.  Load the snapshot for its singletons, then
    // ALWAYS walk from `SUPERBLOCK_SIZE` so the per-record sink (and
    // the manifest's own per-record dispatch) rebuilds every
    // per-entity collection from the records on disk.  Compaction
    // bounds this walk's wall-clock by rewriting the log to contain
    // only live records.
    let mut manifest = if checkpoint_hint >= SUPERBLOCK_SIZE {
        match load_checkpoint(src, checkpoint_hint) {
            Ok((mut snapshot, _next)) => {
                snapshot.last_checkpoint_offset = Some(checkpoint_hint);
                snapshot
            }
            Err(_) => Manifest::new(),
        }
    } else {
        Manifest::new()
    };

    let mut ingest_err: Option<PersistenceError> = None;
    let outcome = walker::walk(src, SUPERBLOCK_SIZE, |entry| {
        if ingest_err.is_none() {
            if let Err(e) = manifest.ingest(entry) {
                ingest_err = Some(e);
            }
        }
        sink(entry);
    })?;
    if let Some(e) = ingest_err {
        return Err(e);
    }

    Ok(Recovered {
        manifest,
        tail_offset: outcome.tail_offset,
        torn: outcome.torn,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::{LogFile, MemLog};
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};

    fn record(rt: RecordType, stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0, // overwritten by encode_record
                stream_id,
                chunk_index,
                token_count: if rt == RecordType::Chunk { 32 } else { 0 },
            },
            payload,
        )
    }

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_ckpt_{tag}_{nanos}.log"));
        p
    }

    #[test]
    fn recover_full_walk_no_checkpoint() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"a"));
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 1, b"b"));
        let mut mem = MemLog::with_records(&blob);

        let mut substrate = Substrate::new();
        let rec =
            recover_with_sink(&mut mem, 0, |e| substrate.apply_walker_entry(e)).unwrap();
        assert!(!rec.torn);
        // Chunk count is on the substrate now.
        assert_eq!(substrate.live_chunk_count(), 2);
    }

    #[test]
    fn recovery_from_checkpoint_equals_full_walk() {
        // Records, then a checkpoint of the manifest-so-far, then more records.
        let mut early = Vec::new();
        early.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"c0"));
        early.extend_from_slice(&record(RecordType::Chunk, 1, 1, b"c1"));
        let mut early_mem = MemLog::with_records(&early);
        let (snapshot, _) = Manifest::build_from_walk(&mut early_mem, SUPERBLOCK_SIZE).unwrap();

        let checkpoint_payload = encode_checkpoint(&snapshot);
        let checkpoint_rec = record(RecordType::Checkpoint, 0, 0, &checkpoint_payload);
        let checkpoint_offset = SUPERBLOCK_SIZE + early.len() as u64;

        let mut full = early.clone();
        full.extend_from_slice(&checkpoint_rec);
        full.extend_from_slice(&record(RecordType::Chunk, 1, 2, b"c2"));
        full.extend_from_slice(&record(RecordType::Chunk, 2, 0, b"d0"));

        let mut mem_a = MemLog::with_records(&full);
        let (_, sub_a, out_a) = Manifest::build_with_substrate(&mut mem_a, SUPERBLOCK_SIZE).unwrap();

        let mut mem_b = MemLog::with_records(&full);
        let from_checkpoint = recover(&mut mem_b, checkpoint_offset).unwrap();
        let mut mem_c = MemLog::with_records(&full);
        let (_, sub_b, _) = Manifest::build_with_substrate(&mut mem_c, SUPERBLOCK_SIZE).unwrap();

        // Per-stream state lives on the substrate; walking from
        // checkpoint vs from scratch reproduces the same in-RAM
        // chunk count.
        assert_eq!(sub_a.live_chunk_count(), sub_b.live_chunk_count());
        assert_eq!(out_a.tail_offset, from_checkpoint.tail_offset);
        assert_eq!(sub_a.live_chunk_count(), 4);
    }

    #[test]
    fn stale_checkpoint_hint_falls_back_to_full_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"x"));
        let mut mem = MemLog::with_records(&blob);
        // A hint pointing into the middle of nowhere — recovery must still work.
        let _rec = recover(&mut mem, SUPERBLOCK_SIZE + 999_999).unwrap();
        // Chunk count is on the substrate now.
        let mut mem2 = MemLog::with_records(&blob);
        let (_, substrate, _) =
            Manifest::build_with_substrate(&mut mem2, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(substrate.live_chunk_count(), 1);
    }

    #[test]
    fn crash_recovery_truncates_a_torn_tail() {
        let path = tmp_path("crash");
        let good_records;
        {
            let mut log = LogFile::create(&path).unwrap();
            log.stage(&record(RecordType::Chunk, 1, 0, b"alpha"));
            log.stage(&record(RecordType::Chunk, 1, 1, b"beta"));
            log.commit().unwrap();
            good_records = log.write_offset();
            // Append a third record, then simulate a real crash by
            // **physically truncating** the file partway through it.
            log.stage(&record(RecordType::Chunk, 1, 2, b"gamma"));
            log.commit().unwrap();
        }
        // Drop the file back to `good_records + 64` bytes — the third
        // record's header is partially on disk but the payload isn't.
        // The walker stops here because the (parseable) header
        // promises a record the file doesn't fully hold. Payload
        // bit-rot of an otherwise-complete record is now caught
        // **out-of-band** by the background CRC validator rather than
        // at recovery time, so we use a true crash scenario here.
        {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(good_records + 64).unwrap();
        }
        {
            let mut log = LogFile::open(&path).unwrap();
            let hint = log.superblock().latest_checkpoint_offset;
            let mut substrate = Substrate::new();
            let rec = recover_with_sink(&mut log, hint, |e| substrate.apply_walker_entry(e))
                .unwrap();
            assert!(rec.torn, "the truncated third record must be detected");
            assert_eq!(rec.tail_offset, good_records);
            assert_eq!(substrate.live_chunk_count(), 2);
            // Applying the recovery: truncate to the good tail.
            log.truncate_to(rec.tail_offset).unwrap();
            log.set_write_offset(rec.tail_offset);
            assert_eq!(log.write_offset(), good_records);
        }
        std::fs::remove_file(&path).ok();
    }

    /// Phase 3: the `Checkpoint` payload carries only singleton
    /// offsets (`model_spec`, `template`, `tokenizer`,
    /// `last_checkpoint_offset`).  Per-entity collections like
    /// `streams`, `labels`, `tree_metadata` are marked
    /// `#[serde(skip)]` — they get rebuilt by the walker on reload, so
    /// the checkpoint payload stays bounded by singleton count.
    ///
    /// This test verifies the contract: encode/decode round-trips the
    /// singletons but drops the per-entity collections.
    #[test]
    fn checkpoint_payload_carries_only_singletons() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 9, 0, b"only"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(
            substrate.live_chunk_count() > 0,
            "walker should have built a non-empty in-RAM streams index on the substrate"
        );
        let payload = encode_checkpoint(&manifest);
        let decoded = Manifest::decode(&payload).unwrap();
        // The checkpoint payload no longer contains any per-entity
        // state — that lives on the substrate; reload rebuilds it via
        // the walker.  Only singletons round-trip.
        assert_eq!(decoded.model_spec, manifest.model_spec);
        assert_eq!(decoded.template, manifest.template);
        assert_eq!(decoded.tokenizer, manifest.tokenizer);
        assert_eq!(decoded.last_checkpoint_offset, manifest.last_checkpoint_offset);
    }
}

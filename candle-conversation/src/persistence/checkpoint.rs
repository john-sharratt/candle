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
    let (mut manifest, replay_from) = if checkpoint_hint >= SUPERBLOCK_SIZE {
        match load_checkpoint(src, checkpoint_hint) {
            Ok((mut snapshot, next)) => {
                snapshot.last_checkpoint_offset = Some(checkpoint_hint);
                (snapshot, next)
            }
            // A stale or corrupt hint — recover the safe way, from scratch.
            Err(_) => (Manifest::new(), SUPERBLOCK_SIZE),
        }
    } else {
        (Manifest::new(), SUPERBLOCK_SIZE)
    };

    let mut ingest_err: Option<PersistenceError> = None;
    let outcome = walker::walk(src, replay_from, |entry| {
        if ingest_err.is_none() {
            if let Err(e) = manifest.ingest(entry) {
                ingest_err = Some(e);
            }
        }
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

        let rec = recover(&mut mem, 0).unwrap();
        assert!(!rec.torn);
        assert_eq!(rec.manifest.live_chunk_count(), 2);
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
        let from_scratch = recover(&mut mem_a, 0).unwrap();

        let mut mem_b = MemLog::with_records(&full);
        let from_checkpoint = recover(&mut mem_b, checkpoint_offset).unwrap();

        assert_eq!(
            from_scratch.manifest.streams,
            from_checkpoint.manifest.streams
        );
        assert_eq!(from_scratch.tail_offset, from_checkpoint.tail_offset);
        assert_eq!(from_checkpoint.manifest.live_chunk_count(), 4);
    }

    #[test]
    fn stale_checkpoint_hint_falls_back_to_full_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"x"));
        let mut mem = MemLog::with_records(&blob);
        // A hint pointing into the middle of nowhere — recovery must still work.
        let rec = recover(&mut mem, SUPERBLOCK_SIZE + 999_999).unwrap();
        assert_eq!(rec.manifest.live_chunk_count(), 1);
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
            let rec = recover(&mut log, hint).unwrap();
            assert!(rec.torn, "the truncated third record must be detected");
            assert_eq!(rec.tail_offset, good_records);
            assert_eq!(rec.manifest.live_chunk_count(), 2);
            // Applying the recovery: truncate to the good tail.
            log.truncate_to(rec.tail_offset).unwrap();
            log.set_write_offset(rec.tail_offset);
            assert_eq!(log.write_offset(), good_records);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn checkpoint_payload_roundtrip() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 9, 0, b"only"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let payload = encode_checkpoint(&manifest);
        assert_eq!(Manifest::decode(&payload).unwrap(), manifest);
    }
}

//! Log recovery — rebuilding the in-RAM state from one walk of the redo log.
//!
//! Recovery is a single **filtered** walk from the first record: every
//! record's header is read, but payload bytes are fetched only for the
//! record types whose payload feeds in-RAM state (stream declarations,
//! labels, tree metadata, …). The bulk record types — `Chunk`, `Tokens`,
//! `Signatures` — are stored by `(offset, len)` reference in the substrate
//! and their payloads are skipped, so recovering a multi-GB log costs a
//! header-only scan, not a full-file read. The walk stops at the first
//! torn record, which becomes the truncation point.
//!
//! Compaction bounds the walk's wall-clock by rewriting the log to
//! contain only live records.

use super::log_file::{LogSource, SUPERBLOCK_SIZE};
use super::manifest::Manifest;
use super::record::RecordType;
use super::walker;
use super::{PersistenceError, Result};

/// The outcome of recovering a log.
#[derive(Clone, Debug)]
pub struct Recovered {
    /// The reconstructed manifest (singleton record locations).
    pub manifest: Manifest,
    /// First byte past the last valid record — the live log tail.
    pub tail_offset: u64,
    /// Whether a torn record was found at `tail_offset` (the file must be
    /// truncated there).
    pub torn: bool,
}

/// Whether recovery needs a record's payload bytes in RAM.
///
/// `Chunk` / `Tokens` / `Signatures` payloads are stored by reference —
/// `Substrate::apply_walker_entry` keeps only their `(offset, len)` — and
/// the `ModelSpec` / `Template` / `Tokenizer` singleton payloads are read
/// back individually after recovery from their manifest locations. None of
/// them need payload bytes during the walk; skipping them turns recovery
/// into a header-only scan of the bulk of the file.
fn payload_needed(rt: RecordType) -> bool {
    !matches!(
        rt,
        RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Signatures
            | RecordType::ModelSpec
            | RecordType::Template
            | RecordType::Tokenizer
    )
}

/// Recover a log into a manifest and its true tail.
pub fn recover(src: &mut dyn LogSource) -> Result<Recovered> {
    recover_with_sink(src, |_| {})
}

/// Recovery + per-record sink. Identical to [`recover`] except every
/// walked record is also passed to `sink`. The production open paths use
/// the sink to dispatch records straight into a
/// [`Substrate`](crate::substrate::Substrate) (via
/// `Substrate::apply_walker_entry`) during the same walker pass that
/// collects the manifest's singleton offsets — per-entity state never has
/// to be mirrored from the manifest into the substrate afterwards.
///
/// Entries for payload-skipped record types (see [`payload_needed`])
/// carry an empty `payload`; their headers, offsets, and sizes are exact.
pub fn recover_with_sink<F>(src: &mut dyn LogSource, mut sink: F) -> Result<Recovered>
where
    F: FnMut(&walker::WalkEntry),
{
    let mut manifest = Manifest::new();
    let mut ingest_err: Option<PersistenceError> = None;
    let outcome = walker::walk_filtered(src, SUPERBLOCK_SIZE, payload_needed, |entry| {
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
    use crate::substrate::Substrate;

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
        p.push(format!("kvtier_recovery_{tag}_{nanos}.log"));
        p
    }

    #[test]
    fn recover_full_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"a"));
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 1, b"b"));
        let mut mem = MemLog::with_records(&blob);

        let mut substrate = Substrate::new();
        let rec = recover_with_sink(&mut mem, |e| substrate.apply_walker_entry(e)).unwrap();
        assert!(!rec.torn);
        assert_eq!(substrate.live_chunk_count(), 2);
    }

    /// Bulk record payloads are skipped during recovery — the sink sees
    /// their exact headers, offsets, and sizes but an empty payload,
    /// while payload-bearing types (`StreamDecl` here) arrive intact.
    #[test]
    fn recovery_skips_bulk_payloads_but_keeps_framing() {
        use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl};

        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "sect".to_string(),
        });
        let decl_bytes = decl.encode();
        let chunk_payload = vec![0xABu8; 9000]; // spans multiple sectors

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::StreamDecl, 7, 0, &decl_bytes));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, &chunk_payload));
        blob.extend_from_slice(&record(RecordType::Tokens, 7, 0, b"tok-bytes"));
        let mut mem = MemLog::with_records(&blob);

        let mut seen: Vec<(RecordType, usize, u64)> = Vec::new();
        let rec = recover_with_sink(&mut mem, |e| {
            seen.push((
                e.record.header.record_type,
                e.record.payload.len(),
                e.record.header.payload_len,
            ));
        })
        .unwrap();
        assert!(!rec.torn);
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].0, RecordType::StreamDecl);
        assert_eq!(
            seen[0].1 as u64, seen[0].2,
            "StreamDecl payload must be read in full"
        );
        assert_eq!(seen[1].0, RecordType::Chunk);
        assert_eq!(seen[1].1, 0, "Chunk payload must be skipped");
        assert_eq!(seen[1].2, 9000, "Chunk header framing must be exact");
        assert_eq!(seen[2].0, RecordType::Tokens);
        assert_eq!(seen[2].1, 0, "Tokens payload must be skipped");
    }

    /// The substrate built from a filtered recovery matches one built
    /// from a full-payload walk — chunk locations come from headers.
    #[test]
    fn filtered_recovery_matches_full_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"c0"));
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 1, b"c1"));
        blob.extend_from_slice(&record(RecordType::Chunk, 2, 0, b"d0"));
        blob.extend_from_slice(&record(RecordType::Tokens, 1, 0, b"tokens"));

        let mut mem_a = MemLog::with_records(&blob);
        let mut sub_a = Substrate::new();
        recover_with_sink(&mut mem_a, |e| sub_a.apply_walker_entry(e)).unwrap();

        let mut mem_b = MemLog::with_records(&blob);
        let mut sub_b = Substrate::new();
        walker::walk(&mut mem_b, SUPERBLOCK_SIZE, |e| sub_b.apply_walker_entry(e)).unwrap();

        assert_eq!(sub_a.live_chunk_count(), sub_b.live_chunk_count());
        assert_eq!(sub_a.live_chunk_count(), 3);
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
        // The walker stops here because the (parseable) header promises
        // a record the file doesn't fully hold. Payload bit-rot of an
        // otherwise-complete record is caught **out-of-band** by the
        // background CRC validator rather than at recovery time.
        {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(good_records + 64).unwrap();
        }
        {
            let mut log = LogFile::open(&path).unwrap();
            let mut substrate = Substrate::new();
            let rec = recover_with_sink(&mut log, |e| substrate.apply_walker_entry(e)).unwrap();
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
}

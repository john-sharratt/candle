//! Background CRC validator — catches latent bit-rot without
//! blocking the recovery / cold-load hot paths.
//!
//! [`decode_record`](super::record::decode_record) is intentionally
//! **not** CRC-verifying anymore: torn writes show up at the header /
//! length level (the walker stops on those) and the cold-load
//! allocator does not need a CRC walk over every payload to make
//! forward progress. What CRC mismatches actually catch is **bit
//! rot** — flipped bits in already-durable bytes that the OS faithfully
//! returns on every read. That class of failure has no urgency at
//! recovery time: the user is willing to wait for the validator to
//! find it asynchronously, log a warning, and skip the affected
//! chunk on subsequent loads.
//!
//! ## Lifecycle
//!
//! A validator thread is spawned by [`SubstratePersistence`] when the
//! substrate opens. It walks every record in the active log and every
//! inherited log once, in order, verifying CRCs against the
//! corresponding [`RecordHeader::crc`]. Mismatches are accumulated in
//! a shared [`BadChunkRegistry`] keyed by `(StreamId, chunk_index)`
//! and a `tracing::warn!` is emitted for each.
//!
//! The thread exits as soon as the walk completes — there is no
//! periodic re-scan. A shutdown flag is honoured if the substrate is
//! dropped before the walk finishes.
//!
//! ## Reader integration
//!
//! [`SubstratePersistence::plan_chunked_read`] consults the registry
//! to drop bad chunks from the cold-load plan; the projection layer
//! ignores any chunk listed in the registry when materialising a
//! turn. Bad chunks are leaked on disk (compaction skips them on the
//! next rewrite) — the on-disk file is never truncated to remove
//! them.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;

use super::log_file::LogFile;
use super::record::{verify_record_crc, RecordType};
use super::streams::StreamId;
use super::walker;

/// Identifier for a chunk record on disk — uniquely names a chunk
/// across the active + inherited logs (the manifest's last-writer-wins
/// rule still applies: only the winning record is ever read, so the
/// `(StreamId, chunk_index)` pair is the right granularity to filter
/// at the cold-load planner).
pub type BadChunkKey = (StreamId, u64);

/// A thread-safe set of `(stream_id, chunk_index)` pairs whose
/// payload CRC failed verification. Shared between the validator
/// thread (writer) and the cold-load planner (reader).
#[derive(Clone, Default, Debug)]
pub struct BadChunkRegistry {
    inner: Arc<RwLock<HashSet<BadChunkKey>>>,
}

impl BadChunkRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Mark `(stream_id, chunk_index)` as known-bad. Idempotent.
    pub fn mark_bad(&self, key: BadChunkKey) {
        if let Ok(mut g) = self.inner.write() {
            g.insert(key);
        }
    }

    /// `true` if `(stream_id, chunk_index)` has been flagged.
    pub fn is_bad(&self, key: BadChunkKey) -> bool {
        self.inner.read().map(|g| g.contains(&key)).unwrap_or(false)
    }

    /// Snapshot of every flagged chunk. Used by diagnostics and tests.
    pub fn snapshot(&self) -> Vec<BadChunkKey> {
        self.inner
            .read()
            .map(|g| g.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Number of flagged chunks.
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.len()).unwrap_or(0)
    }

    /// `true` if no chunks have been flagged.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Handle on the running validator thread — `drop`s the join handle
/// after signalling shutdown so the substrate can close without
/// waiting on the validator to finish its sweep.
pub struct CrcValidator {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl CrcValidator {
    /// Spawn a validator thread that walks `active_path` and every
    /// `inherited_paths` log once, in order, verifying CRCs and
    /// reporting failures into `registry`.
    ///
    /// The thread opens its own [`LogFile`] read handles so it does
    /// not contend with the substrate's read path. Errors opening a
    /// log surface as a single `tracing::warn!` and that log is
    /// skipped — the validator never panics.
    pub fn spawn(
        active_path: PathBuf,
        inherited_paths: Vec<PathBuf>,
        registry: BadChunkRegistry,
    ) -> CrcValidator {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("substrate-crc-validator".into())
            .spawn(move || {
                run_validator(active_path, inherited_paths, registry, stop_clone);
            })
            .expect("spawning a single std thread should always succeed");
        CrcValidator {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for CrcValidator {
    fn drop(&mut self) {
        // Signal stop and let the join run silently — best-effort.
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn run_validator(
    active_path: PathBuf,
    inherited_paths: Vec<PathBuf>,
    registry: BadChunkRegistry,
    stop: Arc<AtomicBool>,
) {
    let t_start = std::time::Instant::now();
    let mut total_records: usize = 0;
    let mut bad_records: usize = 0;

    let mut all_paths: Vec<PathBuf> = inherited_paths;
    all_paths.push(active_path);

    for path in &all_paths {
        if stop.load(Ordering::Acquire) {
            break;
        }
        match LogFile::open(path) {
            Ok(mut log) => {
                let start = super::log_file::SUPERBLOCK_SIZE;
                let (records, bad) = validate_log(&mut log, start, &registry, &stop);
                total_records += records;
                bad_records += bad;
            }
            Err(e) => {
                tracing::warn!(
                    target: "candle_conversation::persistence::crc_validator",
                    path = %path.display(),
                    error = %e,
                    "crc validator could not open log; skipping",
                );
            }
        }
    }

    let elapsed_ms = t_start.elapsed().as_millis();
    if bad_records == 0 {
        tracing::debug!(
            target: "candle_conversation::persistence::crc_validator",
            total_records,
            elapsed_ms,
            "crc validator complete — no bit-rot detected",
        );
    } else {
        tracing::warn!(
            target: "candle_conversation::persistence::crc_validator",
            total_records,
            bad_records,
            elapsed_ms,
            "crc validator complete — flagged bad chunks will be skipped on subsequent loads",
        );
    }
}

/// Walk one log file, returning `(records_seen, records_failed_crc)`.
/// Stops early if `stop` is set.
fn validate_log(
    log: &mut LogFile,
    start: u64,
    registry: &BadChunkRegistry,
    stop: &AtomicBool,
) -> (usize, usize) {
    let mut total = 0usize;
    let mut bad = 0usize;
    // Re-walk records with the shared walker, calling verify_record_crc
    // inside the visitor. We can't bail out of `walker::walk` mid-call
    // — its visitor returns nothing — so the stop flag is checked
    // here and the walker just keeps going if it fires mid-sweep
    // (best-effort: a quick exit is more important than a perfectly
    // partial sweep).
    let _ = walker::walk(log, start, |entry| {
        if stop.load(Ordering::Acquire) {
            return;
        }
        total += 1;
        let header = &entry.record.header;
        if let Err(e) = verify_record_crc(header, &entry.record.payload) {
            bad += 1;
            if header.record_type == RecordType::Chunk {
                let key: BadChunkKey = (StreamId(header.stream_id), header.chunk_index);
                registry.mark_bad(key);
                tracing::warn!(
                    target: "candle_conversation::persistence::crc_validator",
                    stream_id = header.stream_id,
                    chunk_index = header.chunk_index,
                    error = %e,
                    "chunk failed CRC verification — marking invalid",
                );
            } else {
                tracing::warn!(
                    target: "candle_conversation::persistence::crc_validator",
                    record_type = ?header.record_type,
                    stream_id = header.stream_id,
                    error = %e,
                    "non-chunk record failed CRC verification (left in place)",
                );
            }
        }
    });
    (total, bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::SUPERBLOCK_SIZE;
    use crate::persistence::record::{crc32, encode_record, RecordHeader, RecordType};

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("crc_validator_{tag}_{nanos}.log"));
        p
    }

    fn chunk_record(stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: RecordType::Chunk,
                format: 0,
                payload_len: payload.len() as u64,
                crc: crc32(payload),
                stream_id,
                chunk_index,
                token_count: 32,
            },
            payload,
        )
    }

    /// Bad chunk registry stores and queries `(stream, idx)` pairs.
    #[test]
    fn registry_marks_and_queries() {
        let r = BadChunkRegistry::new();
        assert!(r.is_empty());
        r.mark_bad((StreamId(7), 3));
        r.mark_bad((StreamId(7), 5));
        r.mark_bad((StreamId(7), 3)); // idempotent
        assert_eq!(r.len(), 2);
        assert!(r.is_bad((StreamId(7), 3)));
        assert!(r.is_bad((StreamId(7), 5)));
        assert!(!r.is_bad((StreamId(7), 4)));
        assert!(!r.is_bad((StreamId(8), 3)));
    }

    /// End-to-end: write a log with one good chunk and one whose
    /// payload byte has been flipped on disk, run the validator
    /// against it, and assert that the bad chunk shows up in the
    /// registry while the good one does not.
    #[test]
    fn validator_flags_corrupt_chunk() {
        let path = tmp_path("flag");
        // Stage two records via the real LogFile (so the superblock
        // is well-formed), then close the file and flip one payload
        // byte on disk to simulate latent bit-rot.
        let bad_offset_first_payload_byte: u64;
        {
            let mut log = LogFile::create(&path).unwrap();
            log.stage(&chunk_record(1, 0, b"hello-world"));
            let off = log.stage(&chunk_record(1, 1, b"corrupt-me"));
            log.commit().unwrap();
            bad_offset_first_payload_byte = off;
        }
        // Find the second record's header newline so we can flip
        // the first byte AFTER it (i.e. inside the payload, not the
        // header — the header must still parse for the validator to
        // CRC the payload).
        {
            use std::io::{Read, Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .unwrap();
            f.seek(SeekFrom::Start(bad_offset_first_payload_byte))
                .unwrap();
            let mut sector = [0u8; 4096];
            f.read_exact(&mut sector).unwrap();
            let newline_pos = sector.iter().position(|&b| b == b'\n').unwrap();
            let payload_byte_pos = bad_offset_first_payload_byte + newline_pos as u64 + 1;
            f.seek(SeekFrom::Start(payload_byte_pos)).unwrap();
            f.write_all(&[0xFFu8]).unwrap();
            f.sync_all().unwrap();
        }
        let registry = BadChunkRegistry::new();
        let mut log = LogFile::open(&path).unwrap();
        let stop = AtomicBool::new(false);
        let (total, bad) = validate_log(&mut log, SUPERBLOCK_SIZE, &registry, &stop);
        assert_eq!(total, 2, "walker visited both records");
        assert_eq!(bad, 1, "exactly one CRC failure");
        assert!(registry.is_bad((StreamId(1), 1)));
        assert!(!registry.is_bad((StreamId(1), 0)));
        let _ = std::fs::remove_file(&path);
    }
}

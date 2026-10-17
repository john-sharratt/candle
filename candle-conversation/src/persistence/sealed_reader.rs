//! Reading records from **sealed** segments without the persistence mutex.
//!
//! Every read through [`super::SubstratePersistence`] takes its mutex, and
//! background maintenance holds that mutex across its relocation I/O — the
//! resident re-emit, every live chunk relocated, and the fsync, one critical
//! section. Measured on a whole-workspace ingest: a mean of 16.5 s per op and up
//! to 31 s. A short read that only wants a record that has not moved queues
//! behind all of it, and the one that mattered is the recurrent-state snapshot
//! read at every admission of a hybrid-model turn: the scheduler thread stalled
//! on it inside `promote_new_prefills`, with no decode running, for as long as
//! the compaction took.
//!
//! A sealed segment needs no mutex to read. Sealing happens only after the
//! segment's records were flushed and fsynced by a commit; the file is then
//! truncated to its logical end and its write handle dropped, and nothing ever
//! writes to it again — maintenance relocates records *out* of a sealed segment
//! and eventually unlinks it, but never edits it. So a reader that knows which
//! segments are sealed can open one read-only and read a record out of it
//! concurrently with anything the persistence thread is doing.
//!
//! What the reader needs to know that is otherwise behind the mutex is which
//! segment is active: the active one still has records staged in RAM that may
//! not be on disk, so it is declined and left to the locked path. The log
//! publishes its active id into [`ActiveSegment`] after every seal, and ids only
//! ever grow, so `segment < active` is a sealed segment that is already durable.
//!
//! **Two races, both closed by construction.** An unlink by maintenance cannot
//! pull a file out from under an open read: Rust opens every file here with
//! `FILE_SHARE_DELETE`, so the unlink succeeds and the file lingers until the
//! handle closes, and segment ids are never reused, so the lingering name
//! cannot collide. And a read that arrives *after* the unlink fails — with
//! `NotFound`, or on a Windows volume without POSIX delete semantics with
//! `PermissionDenied` while the name is still pending deletion — but
//! maintenance repoints the index at the relocated copy before it unlinks the
//! source, so a caller that re-reads its location after **any** failed read
//! gets one that exists. What the error says is not what the caller acts on;
//! whether the index moved is.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::manifest::RecordLoc;
use super::record::{
    decode_header, decode_record, padded_record_len, verify_record_crc, HEADER_PROBE_SIZE,
};
use super::segment::SegmentId;
use super::segmented_log::segment_path;
use super::{PersistenceError, Result};
use candle::direct_io::AlignedScratch;
use candle::direct_write::{Access, UnbufferedFile};

/// The active segment's id, published by the log and read without a lock.
///
/// Written with `Release` after the previous active is sealed — flushed,
/// fsynced, truncated, and its write handle dropped — and read with `Acquire`,
/// so a reader that sees the new id also sees the sealed file complete.
#[derive(Debug, Clone)]
pub struct ActiveSegment(Arc<AtomicU64>);

impl ActiveSegment {
    pub(super) fn new(id: SegmentId) -> ActiveSegment {
        ActiveSegment(Arc::new(AtomicU64::new(id.raw())))
    }

    /// Publish the new active id. Call only once the previous active is sealed.
    pub(super) fn publish(&self, id: SegmentId) {
        self.0.store(id.raw(), Ordering::Release);
    }

    /// Whether `segment` is sealed as of the latest publish.
    pub fn is_sealed(&self, segment: SegmentId) -> bool {
        segment.raw() < self.0.load(Ordering::Acquire)
    }
}

/// Reads records out of sealed segments by path, without the persistence
/// mutex. Cheap to clone; one per store handle.
#[derive(Debug, Clone)]
pub struct SealedReader {
    dir: PathBuf,
    active: ActiveSegment,
}

impl SealedReader {
    pub(super) fn new(dir: PathBuf, active: ActiveSegment) -> SealedReader {
        SealedReader { dir, active }
    }

    /// The payload of the record at `loc`, or `None` when `loc` is in the
    /// active segment — whose records may still be staged in RAM, and which
    /// only the locked path can read.
    ///
    /// One read-only unbuffered open and one read of the record's whole
    /// sectors — the page cache keeps nothing of it, as it keeps nothing of
    /// the log (`log_file`) — then the same decode and CRC check the log's own
    /// read applies. Not a [`super::log_file::LogFile`]: that opens a pool of
    /// direct-I/O handles and reads the superblock, none of which a single
    /// record read uses. It also leaves the log's own handle pool untouched:
    /// the pool is `&mut` state behind the mutex this exists to avoid.
    pub fn read_record_payload(&self, loc: &RecordLoc) -> Option<Result<Vec<u8>>> {
        if !self.active.is_sealed(loc.segment) {
            return None;
        }
        Some(self.read_sealed(loc))
    }

    fn read_sealed(&self, loc: &RecordLoc) -> Result<Vec<u8>> {
        let file = UnbufferedFile::open(&segment_path(&self.dir, loc.segment), Access::ReadOnly)?;
        let len = loc.record_size as usize;
        let mut bytes = vec![0u8; len];
        let mut scratch = AlignedScratch::new();
        file.read_span(loc.offset, &mut bytes, &mut scratch)
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    PersistenceError::Truncated { need: len, have: 0 }
                } else {
                    PersistenceError::Io(e)
                }
            })?;
        let (header, payload, _) = decode_record(&bytes)?;
        verify_record_crc(&header, payload)?;
        Ok(payload.to_vec())
    }

    /// The whole record at `offset` of `segment` — header line, payload and
    /// padding, as the log's own `read_event` reads it — or `None` when the
    /// segment is the active one. Its size is not known ahead, so a header
    /// probe comes first and then the record; both positioned reads of a file
    /// nothing writes. The caller checks the header names what it asked for:
    /// a location taken before a relocation may name a file since unlinked.
    pub fn read_raw(&self, segment: SegmentId, offset: u64) -> Option<Result<Vec<u8>>> {
        if !self.active.is_sealed(segment) {
            return None;
        }
        Some(self.read_raw_sealed(segment, offset))
    }

    fn read_raw_sealed(&self, segment: SegmentId, offset: u64) -> Result<Vec<u8>> {
        let mut file = File::open(segment_path(&self.dir, segment))?;
        let mut probe = vec![0u8; HEADER_PROBE_SIZE];
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut probe)?;
        let (header, header_bytes) = decode_header(&probe)?;
        let len = padded_record_len(header_bytes - 1, header.payload_len);
        let mut bytes = vec![0u8; len];
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut bytes).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                PersistenceError::Truncated { need: len, have: 0 }
            } else {
                PersistenceError::Io(e)
            }
        })?;
        Ok(bytes)
    }
}

/// Reads of one record, each after re-reading its location, before a caller
/// reports the record unreadable.
///
/// A retry is taken only when the index names a different location than the
/// read that failed — a relocation that completed in between — so a second
/// attempt nearly always lands, and a record that is simply unreadable where
/// the index still points is reported on the first failure. Three bounds a
/// store compacted repeatedly under one read.
pub const SEALED_READ_ATTEMPTS: usize = 3;

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};
    use crate::persistence::segmented_log::SegmentedLog;

    /// A record with a payload the test can recognise, and its padded size.
    fn record(stream_id: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: RecordType::Tokens,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            payload,
        )
    }

    /// Stage one record into the active segment and commit it, returning where
    /// it landed.
    fn write(log: &mut SegmentedLog, stream_id: u64, payload: &[u8]) -> RecordLoc {
        let bytes = record(stream_id, payload);
        let (segment, offset) = log.stage(&bytes);
        log.commit().expect("commit");
        RecordLoc {
            segment,
            offset,
            payload_len: payload.len() as u64,
            record_size: bytes.len() as u64,
        }
    }

    fn open(dir: &std::path::Path) -> SegmentedLog {
        SegmentedLog::open_with_sink(dir, |_| {})
            .expect("open")
            .segments
    }

    /// **A sealed record is read with no log at all.** The log is dropped
    /// before the read, which is the strongest form of "needs no lock on it":
    /// there is nothing left to lock.
    #[test]
    fn a_sealed_record_reads_without_the_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (reader, loc) = {
            let mut log = open(dir.path());
            let loc = write(&mut log, 7, b"sealed and immutable");
            log.seal_and_rotate().expect("seal");
            (log.sealed_reader(), loc)
        };
        let payload = reader
            .read_record_payload(&loc)
            .expect("a sealed segment is read")
            .expect("and the read succeeds");
        assert_eq!(payload, b"sealed and immutable");
    }

    /// The active segment is declined, never read: its records may still be
    /// staged in RAM, and only the log holding them can see them.
    #[test]
    fn the_active_segment_is_declined() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 1, b"still active");
        assert!(log.sealed_reader().read_record_payload(&loc).is_none());
    }

    /// **A reader taken before a seal sees it after.** A conversation takes its
    /// reader once, at construction, so the seal has to reach readers that
    /// already exist — through the published active id they share — or a
    /// long-lived handle would decline every segment sealed since it opened.
    #[test]
    fn a_reader_taken_earlier_sees_later_seals() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let reader = log.sealed_reader();
        let loc = write(&mut log, 3, b"sealed after the reader was taken");
        assert!(
            reader.read_record_payload(&loc).is_none(),
            "still the active segment when first asked"
        );
        log.seal_and_rotate().expect("seal");
        let payload = reader
            .read_record_payload(&loc)
            .expect("sealed now")
            .expect("read");
        assert_eq!(payload, b"sealed after the reader was taken");
        // And the new active is declined in turn.
        let next = write(&mut log, 4, b"the new active");
        assert!(reader.read_record_payload(&next).is_none());
    }

    /// A segment maintenance unlinked is a failed read, not a panic and not an
    /// empty payload — the caller's cue to re-read the location, which
    /// maintenance repointed before the unlink.
    #[test]
    fn an_unlinked_segment_is_a_failed_read() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 5, b"relocated then dropped");
        log.seal_and_rotate().expect("seal");
        log.drop_sealed(loc.segment).expect("drop");
        log.sealed_reader()
            .read_record_payload(&loc)
            .expect("still counts as sealed")
            .expect_err("the file is gone");
    }

    /// A location that does not start a record fails the decode rather than
    /// returning some other record's bytes.
    #[test]
    fn a_misaligned_location_fails_the_decode() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 6, b"a record the read will overrun");
        log.seal_and_rotate().expect("seal");
        let torn = RecordLoc {
            offset: loc.offset + 1,
            ..loc
        };
        log.sealed_reader()
            .read_record_payload(&torn)
            .expect("sealed")
            .expect_err("misaligned");
    }

    /// **A record whose bytes changed on disk is refused by its CRC**, exactly
    /// as the log's own read refuses it: the lock-free path does not skip the
    /// integrity check the locked one applies.
    #[test]
    fn a_corrupted_payload_fails_its_crc() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let payload = b"the bytes the crc covers";
        let loc = write(&mut log, 8, payload);
        log.seal_and_rotate().expect("seal");
        let reader = log.sealed_reader();
        assert_eq!(
            reader
                .read_record_payload(&loc)
                .expect("sealed")
                .expect("intact"),
            payload
        );

        // Flip one payload byte in the sealed file.
        let path = segment_path(dir.path(), loc.segment);
        let mut bytes = std::fs::read(&path).expect("read segment");
        let at = bytes
            .windows(payload.len())
            .position(|w| w == payload)
            .expect("the payload is on disk");
        bytes[at] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("rewrite segment");

        reader
            .read_record_payload(&loc)
            .expect("sealed")
            .expect_err("a changed payload fails its crc");
    }

    /// **The read does not wait for whoever holds the log.** This is the
    /// property the reader exists for: a compaction holds the persistence
    /// mutex across seconds of relocation I/O, and a sealed read must complete
    /// underneath it. The log is held behind a mutex for the whole test, and
    /// the read runs on another thread against a deadline.
    #[test]
    fn a_read_completes_while_the_log_is_held() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 9, b"read under a held lock");
        log.seal_and_rotate().expect("seal");
        let reader = log.sealed_reader();
        let held = Mutex::new(log);
        let _guard = held.lock().expect("hold the log");

        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let read = reader.read_record_payload(&loc).map(|r| r.ok());
            let _ = tx.send(read);
        });
        let got = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the read finished while the log was held");
        worker.join().expect("worker");
        assert_eq!(got, Some(Some(b"read under a held lock".to_vec())));
    }

    /// **A whole record reads back as the log wrote it**, header and padding
    /// included, sized from its own header — what a peer's fetch sends without
    /// the persistence lock. The active segment is declined, and an unlinked one
    /// fails, so the caller falls back to the locked read.
    #[test]
    fn a_raw_read_returns_the_whole_record_and_declines_what_it_cannot_vouch_for() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let written = record(9, b"fetched by a peer catching up");
        let loc = write(&mut log, 9, b"fetched by a peer catching up");
        assert!(log
            .sealed_reader()
            .read_raw(loc.segment, loc.offset)
            .is_none());
        log.seal_and_rotate().expect("seal");
        let raw = log
            .sealed_reader()
            .read_raw(loc.segment, loc.offset)
            .expect("sealed")
            .expect("read");
        assert_eq!(raw, written);
        log.drop_sealed(loc.segment).expect("drop");
        log.sealed_reader()
            .read_raw(loc.segment, loc.offset)
            .expect("still counts as sealed")
            .expect_err("the file is gone");
    }
}

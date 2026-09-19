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
//! cannot collide. And a read that arrives *after* the unlink sees
//! [`std::io::ErrorKind::NotFound`] — but maintenance repoints the index at the
//! relocated copy before it unlinks the source, so a caller that re-reads its
//! location gets one that exists.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::log_file::{read_record_at, LogFile};
use super::manifest::RecordLoc;
use super::segment::SegmentId;
use super::segmented_log::segment_path;
use super::{PersistenceError, Result};

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
    /// Opens the segment read-only for this one read. That costs an open and a
    /// superblock read, which is nothing beside the wait it replaces, and it
    /// leaves the log's own handle pool untouched — the pool is `&mut` state
    /// behind the mutex this exists to avoid.
    pub fn read_record_payload(&self, loc: &RecordLoc) -> Option<Result<Vec<u8>>> {
        if !self.active.is_sealed(loc.segment) {
            return None;
        }
        let path = segment_path(&self.dir, loc.segment);
        Some(
            LogFile::open_read_only(&path)
                .and_then(|mut log| read_record_at(&mut log, loc.offset, loc.record_size))
                .map(|record| record.payload),
        )
    }
}

/// Lock-free attempts at one record before a caller falls back to the locked
/// read.
///
/// Each retry follows a relocation that completed — maintenance repointed the
/// index, then unlinked the file the previous location named — so a second
/// attempt nearly always lands. Three bounds a store compacted repeatedly under
/// one read, and the locked read the caller falls back to is always correct.
pub const SEALED_READ_ATTEMPTS: usize = 3;

/// Whether a sealed read failed because the segment is gone — unlinked by
/// maintenance after the record was relocated, which the caller answers by
/// re-reading the record's location rather than by failing.
pub fn is_unlinked(err: &PersistenceError) -> bool {
    matches!(err, PersistenceError::Io(e) if e.kind() == std::io::ErrorKind::NotFound)
}

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

    /// A segment maintenance unlinked reports itself as such, so the caller
    /// knows to re-read the location — which maintenance repointed before the
    /// unlink — rather than fail.
    #[test]
    fn an_unlinked_segment_is_reported_as_unlinked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 5, b"relocated then dropped");
        log.seal_and_rotate().expect("seal");
        log.drop_sealed(loc.segment).expect("drop");
        let err = log
            .sealed_reader()
            .read_record_payload(&loc)
            .expect("still counts as sealed")
            .expect_err("the file is gone");
        assert!(is_unlinked(&err), "reported as unlinked: {err}");
    }

    /// Any other failure is not mistaken for an unlink: a caller must not
    /// chase a moved record when the record is simply unreadable.
    #[test]
    fn a_torn_record_is_not_mistaken_for_an_unlink() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let loc = write(&mut log, 6, b"a record the read will overrun");
        log.seal_and_rotate().expect("seal");
        let torn = RecordLoc {
            offset: loc.offset + 1,
            ..loc
        };
        let err = log
            .sealed_reader()
            .read_record_payload(&torn)
            .expect("sealed")
            .expect_err("misaligned");
        assert!(!is_unlinked(&err), "a bad record is not a moved one: {err}");
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
}

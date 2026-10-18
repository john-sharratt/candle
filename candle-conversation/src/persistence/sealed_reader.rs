//! Reading **durable** records — any in a sealed segment, and those below the
//! active segment's last commit — without the persistence mutex.
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
//! The active segment's **committed prefix** needs none either: a commit
//! fsyncs everything below the logical end, and the file only grows, so a
//! record wholly below it is as final as a sealed one. What lies past it may
//! still be staged in RAM, so it is declined and left to the locked path. A
//! record written moments ago — the recurrent snapshot a turn's seal just
//! enqueued, which the next admission reads back — is readable here as soon
//! as the commit that covered it returns.
//!
//! What the reader needs to know that is otherwise behind the mutex is which
//! segment is active and how far it is committed. The log publishes both into
//! [`ActiveSegment`] — the id after every seal, the end after every commit —
//! and ids only ever grow, so `segment < active` is a sealed segment that is
//! already durable.
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

/// The active segment's id and how much of it is durable, published by the log
/// and read without a lock.
///
/// The id is written with `Release` after the previous active is sealed —
/// flushed, fsynced, truncated, and its write handle dropped — and read with
/// `Acquire`, so a reader that sees the new id also sees the sealed file
/// complete.
///
/// The durable prefix is the active's logical end as of its last successful
/// commit: everything below it was fsynced and, the file being append-only,
/// is never written again. It is published as a segment and an end, the end
/// stored first, and read segment, end, segment: a reader accepts the end
/// only when both segment reads agree on the segment it wants. An end
/// belonging to a later segment is stored after that segment's id, so a
/// reader that sees it sees the later id too and declines; a fresh active
/// starts at an end of 0, which accepts nothing.
#[derive(Debug, Clone)]
pub struct ActiveSegment(Arc<Published>);

#[derive(Debug)]
struct Published {
    active: AtomicU64,
    committed_segment: AtomicU64,
    committed_end: AtomicU64,
}

impl ActiveSegment {
    /// `id` is the active, durable through `committed_end` — what a recovery
    /// walk found on disk, or nothing yet for a new segment.
    pub(super) fn new(id: SegmentId, committed_end: u64) -> ActiveSegment {
        ActiveSegment(Arc::new(Published {
            active: AtomicU64::new(id.raw()),
            committed_segment: AtomicU64::new(id.raw()),
            committed_end: AtomicU64::new(committed_end),
        }))
    }

    /// Publish the new active id. Call only once the previous active is sealed.
    /// The new active has nothing durable yet.
    pub(super) fn publish(&self, id: SegmentId) {
        self.0.committed_end.store(0, Ordering::Release);
        self.0.committed_segment.store(id.raw(), Ordering::Release);
        self.0.active.store(id.raw(), Ordering::Release);
    }

    /// Publish that the active `id` is durable through `end`. Call only after
    /// a commit that fsynced it.
    pub(super) fn committed(&self, id: SegmentId, end: u64) {
        self.0.committed_end.store(end, Ordering::Release);
        self.0.committed_segment.store(id.raw(), Ordering::Release);
    }

    /// Whether `segment` is sealed as of the latest publish.
    pub fn is_sealed(&self, segment: SegmentId) -> bool {
        segment.raw() < self.0.active.load(Ordering::Acquire)
    }

    /// Whether the record at `loc` is durable and will never be written again:
    /// in a sealed segment, or wholly inside the active's committed prefix.
    pub fn is_durable(&self, loc: &RecordLoc) -> bool {
        if self.is_sealed(loc.segment) {
            return true;
        }
        let before = self.0.committed_segment.load(Ordering::Acquire);
        let end = self.0.committed_end.load(Ordering::Acquire);
        let after = self.0.committed_segment.load(Ordering::Acquire);
        before == after
            && before == loc.segment.raw()
            && loc.offset.saturating_add(loc.record_size) <= end
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

    /// Whether the record at `loc` is durable — readable here, and never to
    /// change — as of the latest publish ([`ActiveSegment::is_durable`]).
    pub fn is_durable(&self, loc: &RecordLoc) -> bool {
        self.active.is_durable(loc)
    }

    /// The payload of the record at `loc`, or `None` when `loc` is not yet
    /// durable — past the active segment's last commit, where it may still be
    /// staged in RAM and only the locked path can read it.
    ///
    /// One read-only unbuffered open and one read of the record's whole
    /// sectors — the page cache keeps nothing of it, as it keeps nothing of
    /// the log (`log_file`) — then the same decode and CRC check the log's own
    /// read applies. Not a [`super::log_file::LogFile`]: that opens a pool of
    /// direct-I/O handles and reads the superblock, none of which a single
    /// record read uses. It also leaves the log's own handle pool untouched:
    /// the pool is `&mut` state behind the mutex this exists to avoid.
    pub fn read_record_payload(&self, loc: &RecordLoc) -> Option<Result<Vec<u8>>> {
        if !self.active.is_durable(loc) {
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

    /// Stage one record into the active segment without committing it.
    fn stage(log: &mut SegmentedLog, stream_id: u64, payload: &[u8]) -> RecordLoc {
        let bytes = record(stream_id, payload);
        let (segment, offset) = log.stage(&bytes);
        RecordLoc {
            segment,
            offset,
            payload_len: payload.len() as u64,
            record_size: bytes.len() as u64,
        }
    }

    /// **The active segment is read through its last commit, and declined
    /// past it.** A staged record may exist only in RAM, so it is declined;
    /// once a commit has fsynced it, it reads here like a sealed one — the
    /// record a seal wrote moments ago, without the log's lock.
    #[test]
    fn the_active_is_read_through_its_last_commit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let reader = log.sealed_reader();
        let loc = stage(&mut log, 1, b"staged, then committed");
        assert!(!reader.is_durable(&loc));
        assert!(
            reader.read_record_payload(&loc).is_none(),
            "a staged record is declined"
        );
        log.commit().expect("commit");
        assert!(reader.is_durable(&loc));
        assert_eq!(
            reader
                .read_record_payload(&loc)
                .expect("committed")
                .expect("read"),
            b"staged, then committed"
        );
        // A later record past the commit is declined while the earlier one
        // still reads.
        let next = stage(&mut log, 2, b"not committed yet");
        assert!(reader.read_record_payload(&next).is_none());
        assert!(reader.is_durable(&loc));
    }

    /// **A reader taken earlier sees later commits and seals.** A conversation
    /// takes its reader once, at construction, so both have to reach readers
    /// that already exist — through the published state they share — or a
    /// long-lived handle would decline everything written since it opened. A
    /// rotation leaves the new active with nothing durable, and the old one
    /// sealed.
    #[test]
    fn a_reader_taken_earlier_sees_later_commits_and_seals() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = open(dir.path());
        let reader = log.sealed_reader();
        let loc = stage(&mut log, 3, b"written after the reader was taken");
        assert!(reader.read_record_payload(&loc).is_none(), "staged");
        log.commit().expect("commit");
        log.seal_and_rotate().expect("seal");
        let payload = reader
            .read_record_payload(&loc)
            .expect("sealed now")
            .expect("read");
        assert_eq!(payload, b"written after the reader was taken");
        let next = stage(&mut log, 4, b"the new active");
        assert!(reader.read_record_payload(&next).is_none());
        log.commit().expect("commit");
        assert_eq!(
            reader
                .read_record_payload(&next)
                .expect("committed")
                .expect("read"),
            b"the new active"
        );
    }

    /// **A durable end is never taken for the wrong segment.** The published
    /// end is the active's; a location in a different, unsealed segment id at
    /// an offset below it is declined, and a fresh active starts with nothing
    /// durable.
    #[test]
    fn a_durable_end_belongs_to_its_own_segment() {
        let active = ActiveSegment::new(SegmentId(5), 4096);
        let at = |segment: u64, offset: u64| RecordLoc {
            segment: SegmentId(segment),
            offset,
            payload_len: 8,
            record_size: 64,
        };
        assert!(active.is_durable(&at(5, 1024)));
        assert!(!active.is_durable(&at(5, 4096 - 63)), "past the end");
        assert!(active.is_durable(&at(4, 1 << 30)), "sealed");
        assert!(!active.is_durable(&at(6, 0)), "not the committed segment");
        active.publish(SegmentId(6));
        assert!(
            !active.is_durable(&at(6, 0)),
            "a fresh active holds nothing"
        );
        assert!(
            active.is_durable(&at(5, 1 << 30)),
            "the old active is sealed"
        );
        active.committed(SegmentId(6), 128);
        assert!(active.is_durable(&at(6, 64)));
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

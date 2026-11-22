//! Reading one record back by location — whole, or only the first bytes of its
//! payload — from whatever holds the log.
//!
//! The substrate keeps a located record's place, not its bytes
//! (`substrate::Located`), and reads it back through a [`RecordReader`] when
//! something uses it. The store's [`SealedReader`] is one: it reads any durable
//! record without the persistence mutex. A tool that holds a log some other way
//! — an inspector over single-file logs — supplies its own.
//!
//! A head read is what lets a scan size a signature window without its tokens:
//! the window's shape is in the payload's first bytes, so one sector is read
//! rather than the record.

use std::sync::Mutex;

use candle::direct_io::AlignedScratch;
use candle::direct_write::{Access, UnbufferedFile};

use super::log_file::{read_record_at, LogFile};
use super::manifest::RecordLoc;
use super::record::{decode_header, RecordHeader, HEADER_PROBE_SIZE};
use super::sealed_reader::SealedReader;
use super::segmented_log::segment_path;
use super::{PersistenceError, Result};

/// Reads a record back by its location.
pub trait RecordReader: Send + Sync {
    /// The record at `loc`: its header, and its payload — whole for `head`
    /// `None`, else at most its first `head` bytes. `None` when this reader
    /// cannot read `loc`: a record not yet durable, which only the log's owner
    /// can read.
    ///
    /// A whole read is checked against the record's CRC; a head read is not —
    /// the CRC covers the whole payload — so a head is only ever a shape a
    /// later whole read verifies.
    fn read(&self, loc: &RecordLoc, head: Option<usize>)
        -> Option<Result<(RecordHeader, Vec<u8>)>>;
}

impl RecordReader for SealedReader {
    fn read(
        &self,
        loc: &RecordLoc,
        head: Option<usize>,
    ) -> Option<Result<(RecordHeader, Vec<u8>)>> {
        match head {
            None => self.read_records(std::slice::from_ref(loc)).map(|read| {
                read.map(|mut r| {
                    let record = r.remove(0);
                    (record.header, record.payload)
                })
            }),
            Some(n) => self.is_durable(loc).then(|| self.read_head(loc, n)),
        }
    }
}

/// Reads records out of one single-file log — what a tool that walked a
/// `LogFile` reads stored bytes back with. Every location names this file,
/// whatever segment it carries.
pub struct LogFileReader(Mutex<LogFile>);

impl LogFileReader {
    /// A reader over `log`.
    pub fn new(log: LogFile) -> Self {
        Self(Mutex::new(log))
    }
}

impl RecordReader for LogFileReader {
    fn read(
        &self,
        loc: &RecordLoc,
        head: Option<usize>,
    ) -> Option<Result<(RecordHeader, Vec<u8>)>> {
        let mut log = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Some(
            read_record_at(&mut *log, loc.offset, loc.record_size).map(|mut r| {
                if let Some(n) = head {
                    r.payload.truncate(n);
                }
                (r.header, r.payload)
            }),
        )
    }
}

impl SealedReader {
    /// The header of the durable record at `loc` and the first `n` bytes of
    /// its payload, reading only the sectors that hold them.
    fn read_head(&self, loc: &RecordLoc, n: usize) -> Result<(RecordHeader, Vec<u8>)> {
        let file = UnbufferedFile::open(&segment_path(self.dir(), loc.segment), Access::ReadOnly)?;
        let mut scratch = AlignedScratch::new();
        let whole = loc.record_size as usize;
        let mut len = whole.min(HEADER_PROBE_SIZE);
        loop {
            let mut bytes = vec![0u8; len];
            file.read_span(loc.offset, &mut bytes, &mut scratch)?;
            let (header, header_bytes) = decode_header(&bytes)?;
            let want = header_bytes + n.min(header.payload_len as usize);
            if want <= len {
                return Ok((header, bytes[header_bytes..want].to_vec()));
            }
            if len == whole {
                return Err(PersistenceError::Truncated {
                    need: want,
                    have: len,
                });
            }
            len = want.min(whole);
        }
    }
}

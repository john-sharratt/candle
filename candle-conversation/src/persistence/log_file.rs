//! The append-only redo-log file (§5.1 of `docs/kv_tier_migration.md`).
//!
//! A single pre-grown file: a 4 KB superblock followed by 4 KB-aligned
//! records. Writes are buffered into a group-commit staging buffer and
//! flushed as one sequential write; `commit` additionally `fsync`s for
//! durability. Records are never rewritten in place — only the
//! superblock is (file identity, format version, and the advisory
//! `HeaderIndex` hint recovery anchors its backward chain on).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::direct_io::DirectFile;
use super::record::{crc32, decode_record, verify_record_crc, Record, ALIGN};
use super::{PersistenceError, Result};

/// Size of the file superblock — the first block, holding file identity,
/// the format version, and the latest `HeaderIndex` hint.
pub const SUPERBLOCK_SIZE: u64 = 4096;

/// File-level magic — ASCII `"SLOG"`, little-endian. Distinct from the
/// per-record magic.
pub const FILE_MAGIC: u32 = 0x474f_4c53;

/// On-disk file-format version.
///
/// Bumped to `2` for the per-half turn split — `TurnDecl` gained the
/// `user_content_start` / `user_content_end` / `assistant_content_start`
/// content boundaries and reload semantics changed to window the turn's
/// content halves out of the in-memory `TurnEntryData`.  Old (v1) logs
/// are rejected on open, forcing a clean rebuild rather than silently
/// loading pre-split records into the new substrate shape.
pub const FILE_FORMAT_VERSION: u32 = 2;

/// The file is grown ahead in extents of this size so appends write into
/// already-allocated space.
const GROW_EXTENT: u64 = 1 << 20;

/// Decoded superblock contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Superblock {
    pub format_version: u32,
    /// `(offset, padded_size)` of the newest committed `HeaderIndex`
    /// record — the anchor of recovery's backward digest chain (§5.6).
    /// `(0, 0)` means no index has been written. A **hint**, not
    /// authoritative: recovery validates the record it points at and
    /// falls back to the full forward walk on any mismatch, so a stale
    /// or garbage value (including the retired checkpoint offset that
    /// logs written by earlier builds carry in these bytes, with a
    /// zero size alongside it) costs a fallback, never corruption.
    pub last_index: (u64, u64),
}

impl Superblock {
    fn encode(&self) -> [u8; SUPERBLOCK_SIZE as usize] {
        let mut b = [0u8; SUPERBLOCK_SIZE as usize];
        b[0..4].copy_from_slice(&FILE_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&self.format_version.to_le_bytes());
        b[8..16].copy_from_slice(&self.last_index.0.to_le_bytes());
        b[16..24].copy_from_slice(&self.last_index.1.to_le_bytes());
        // Bytes 24..SUPERBLOCK_SIZE-4 are reserved (zero). They are
        // covered by the CRC, so a reader validates whatever a writer
        // put there without interpreting it.
        let crc = crc32(&b[0..SUPERBLOCK_SIZE as usize - 4]);
        b[SUPERBLOCK_SIZE as usize - 4..].copy_from_slice(&crc.to_le_bytes());
        b
    }

    fn decode(b: &[u8]) -> Result<Superblock> {
        if b.len() < SUPERBLOCK_SIZE as usize {
            return Err(PersistenceError::Truncated {
                need: SUPERBLOCK_SIZE as usize,
                have: b.len(),
            });
        }
        let magic = u32::from_le_bytes(b[0..4].try_into().unwrap());
        if magic != FILE_MAGIC {
            return Err(PersistenceError::BadMagic {
                expected: FILE_MAGIC,
                found: magic,
            });
        }
        let stored = u32::from_le_bytes(
            b[SUPERBLOCK_SIZE as usize - 4..SUPERBLOCK_SIZE as usize]
                .try_into()
                .unwrap(),
        );
        let computed = crc32(&b[0..SUPERBLOCK_SIZE as usize - 4]);
        if stored != computed {
            return Err(PersistenceError::BadChecksum {
                header: stored,
                computed,
            });
        }
        let format_version = u32::from_le_bytes(b[4..8].try_into().unwrap());
        if format_version != FILE_FORMAT_VERSION {
            return Err(PersistenceError::Corrupt(format!(
                "unsupported log format version {format_version}"
            )));
        }
        let last_index = (
            u64::from_le_bytes(b[8..16].try_into().unwrap()),
            u64::from_le_bytes(b[16..24].try_into().unwrap()),
        );
        Ok(Superblock {
            format_version,
            last_index,
        })
    }
}

/// An open append-only log file.
pub struct LogFile {
    file: File,
    /// A second open handle on the same file, opened with `O_DIRECT` /
    /// `FILE_FLAG_NO_BUFFERING`. Used exclusively for the cold-load fast
    /// path ([`LogFile::direct_file`]) so bulk reads bypass the OS page
    /// cache and let the NVMe controller DMA straight into sector-aligned
    /// host scratch. The buffered `file` handle still owns every write
    /// and every non-aligned small read (superblock, single-record
    /// lookups, etc.).
    direct: DirectFile,
    superblock: Superblock,
    /// Durable logical end — offset where the next record will land.
    write_offset: u64,
    /// Physical file length (pre-grown; always `>= write_offset`).
    allocated: u64,
    /// Group-commit staging buffer — appended records not yet flushed.
    pending: Vec<u8>,
}

impl LogFile {
    /// Create a fresh log file, writing the superblock. Fails if the path
    /// already exists.
    pub fn create(path: &Path) -> Result<LogFile> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        let superblock = Superblock {
            format_version: FILE_FORMAT_VERSION,
            last_index: (0, 0),
        };
        let direct = DirectFile::open(path)?;
        let mut log = LogFile {
            file,
            direct,
            superblock,
            write_offset: SUPERBLOCK_SIZE,
            allocated: 0,
            pending: Vec::new(),
        };
        log.grow_to(SUPERBLOCK_SIZE)?;
        log.write_superblock()?;
        log.file.sync_all()?;
        Ok(log)
    }

    /// Open an existing log file and validate its superblock. The write
    /// offset is left at the start of the record region; a recovery pass
    /// must call [`LogFile::set_write_offset`] once the true tail is known.
    ///
    /// The superblock is rewritten on every index flush, so a crash can
    /// tear it. A torn superblock (valid magic, bad CRC) is **self-healed**:
    /// it holds nothing authoritative — only the format version and the
    /// advisory index hint — so it is rewritten fresh with a zero hint and
    /// recovery takes the full-walk path. A wrong magic stays a hard error
    /// (the file isn't a substrate log), and a valid-CRC superblock with an
    /// unsupported version stays a hard error (a genuinely old format).
    pub fn open(path: &Path) -> Result<LogFile> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let allocated = file.metadata()?.len();
        let direct = DirectFile::open(path)?;
        let mut log = LogFile {
            file,
            direct,
            superblock: Superblock {
                format_version: FILE_FORMAT_VERSION,
                last_index: (0, 0),
            },
            write_offset: SUPERBLOCK_SIZE,
            allocated,
            pending: Vec::new(),
        };
        let head = log.read_at(0, SUPERBLOCK_SIZE as usize)?;
        match Superblock::decode(&head) {
            Ok(sb) => log.superblock = sb,
            Err(PersistenceError::BadChecksum { .. }) => {
                tracing::warn!(
                    "torn superblock (magic valid, CRC mismatch) — rewriting it; \
                     recovery falls back to the full walk"
                );
                log.write_superblock()?;
                log.file.sync_data()?;
            }
            Err(e) => return Err(e),
        }
        Ok(log)
    }

    /// The cache-bypassing read handle on this log. Used by the
    /// cold-load fast path to submit stripe reads in parallel via
    /// [`DirectFile::read_stripes_concurrent`].
    pub fn direct_file(&self) -> &DirectFile {
        &self.direct
    }

    /// The decoded superblock.
    pub fn superblock(&self) -> Superblock {
        self.superblock
    }

    /// The durable logical end of the log — offset of the next append.
    pub fn write_offset(&self) -> u64 {
        self.write_offset
    }

    /// Physical (pre-grown) file length.
    pub fn allocated_len(&self) -> u64 {
        self.allocated
    }

    /// Set the write offset — used by recovery once the true tail is known.
    /// Any pre-grown bytes beyond it are free space, overwritten by appends.
    pub fn set_write_offset(&mut self, offset: u64) {
        assert!(
            offset >= SUPERBLOCK_SIZE,
            "write offset is inside the superblock"
        );
        assert!(
            self.pending.is_empty(),
            "set_write_offset with un-flushed records"
        );
        self.write_offset = offset;
    }

    /// Read `len` bytes at `offset`. Errors if the file does not hold that
    /// many bytes there.
    pub fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                PersistenceError::Truncated { need: len, have: 0 }
            } else {
                PersistenceError::Io(e)
            }
        })?;
        Ok(buf)
    }

    /// Stage one already-encoded, 4 KB-aligned record into the group-commit
    /// buffer. Returns the file offset the record will occupy once flushed.
    pub fn stage(&mut self, record_bytes: &[u8]) -> u64 {
        assert!(
            record_bytes.len() % ALIGN == 0,
            "record bytes must be 4 KB-aligned"
        );
        let offset = self.write_offset + self.pending.len() as u64;
        self.pending.extend_from_slice(record_bytes);
        offset
    }

    /// Bytes currently staged but not yet flushed.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Flush the group-commit buffer to the file as one sequential write.
    /// Does not `fsync` — see [`LogFile::commit`].
    pub fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let end = self.write_offset + self.pending.len() as u64;
        self.grow_to(end)?;
        self.file.seek(SeekFrom::Start(self.write_offset))?;
        self.file.write_all(&self.pending)?;
        self.write_offset = end;
        self.pending.clear();
        Ok(())
    }

    /// Flush and `fsync` — the group-commit durability boundary.
    pub fn commit(&mut self) -> Result<()> {
        self.flush()?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Record the newest committed `HeaderIndex` location in the
    /// superblock and persist it. Called after the index record's bytes
    /// are durably committed, so the hint never points past the flushed
    /// region on an ordered-write device; recovery tolerates a stale or
    /// dangling hint either way (full-walk fallback).
    pub fn set_last_index(&mut self, last_index: (u64, u64)) -> Result<()> {
        self.superblock.last_index = last_index;
        self.write_superblock()?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Physically truncate the file to `len` (used by recovery after a torn
    /// tail, and by compaction).
    pub fn truncate_to(&mut self, len: u64) -> Result<()> {
        self.file.set_len(len)?;
        self.allocated = len;
        if self.write_offset > len {
            self.write_offset = len.max(SUPERBLOCK_SIZE);
        }
        Ok(())
    }

    fn write_superblock(&mut self) -> Result<()> {
        let bytes = self.superblock.encode();
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&bytes)?;
        Ok(())
    }

    /// Grow the physical file so it holds at least `end` bytes, rounding up
    /// to the next [`GROW_EXTENT`].
    fn grow_to(&mut self, end: u64) -> Result<()> {
        if end <= self.allocated {
            return Ok(());
        }
        let target = end.div_ceil(GROW_EXTENT) * GROW_EXTENT;
        self.file.set_len(target)?;
        self.allocated = target;
        Ok(())
    }
}

/// Read-only random access over a log's bytes — implemented by [`LogFile`]
/// and by an in-memory `&[u8]` so the walker and recovery can be tested
/// without touching disk.
pub trait LogSource {
    /// Read `len` bytes at `offset`, or an error if unavailable.
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>>;
    /// Read `dest.len()` bytes at `offset` directly into `dest` — the
    /// batched cold-read path calls this once per stripe so a single
    /// reusable scratch buffer absorbs the whole turn's records.
    fn read_into(&mut self, offset: u64, dest: &mut [u8]) -> Result<()>;
    /// Total readable length.
    fn size(&mut self) -> Result<u64>;
}

impl LogSource for LogFile {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        LogFile::read_at(self, offset, len)
    }

    fn read_into(&mut self, offset: u64, dest: &mut [u8]) -> Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(dest).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                PersistenceError::Truncated {
                    need: dest.len(),
                    have: 0,
                }
            } else {
                PersistenceError::Io(e)
            }
        })?;
        Ok(())
    }

    fn size(&mut self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }
}

/// An in-memory log image — a superblock-prefixed byte buffer — for tests.
pub struct MemLog {
    pub bytes: Vec<u8>,
}

impl MemLog {
    /// A `MemLog` whose superblock is valid and whose record region is `records`.
    pub fn with_records(records: &[u8]) -> MemLog {
        let sb = Superblock {
            format_version: FILE_FORMAT_VERSION,
            last_index: (0, 0),
        };
        let mut bytes = sb.encode().to_vec();
        bytes.extend_from_slice(records);
        MemLog { bytes }
    }
}

impl LogSource for MemLog {
    fn read_at(&mut self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let start = offset as usize;
        let end = start + len;
        if end > self.bytes.len() {
            return Err(PersistenceError::Truncated {
                need: len,
                have: self.bytes.len().saturating_sub(start),
            });
        }
        Ok(self.bytes[start..end].to_vec())
    }

    fn read_into(&mut self, offset: u64, dest: &mut [u8]) -> Result<()> {
        let start = offset as usize;
        let end = start + dest.len();
        if end > self.bytes.len() {
            return Err(PersistenceError::Truncated {
                need: dest.len(),
                have: self.bytes.len().saturating_sub(start),
            });
        }
        dest.copy_from_slice(&self.bytes[start..end]);
        Ok(())
    }

    fn size(&mut self) -> Result<u64> {
        Ok(self.bytes.len() as u64)
    }
}

/// Read, decode, and CRC-verify the whole record at `offset`.
/// `record_size` is the exact padded on-disk size from a `RecordLoc` /
/// `ChunkLoc` — captured at walk time or write time. Single read, no
/// probe. Bit-rot in the payload surfaces here as `BadChecksum` — the
/// random-access reads this serves (singletons, tokens, signatures,
/// compaction's live-set copy) all consume the payload immediately.
///
/// Returns an owned [`Record`] — the on-disk bytes go out of scope at
/// return time. Callers on the hot path that want to read the payload
/// in place should use the borrowed [`decode_record`] against their
/// own buffer instead of going through this convenience wrapper.
pub fn read_record_at(src: &mut dyn LogSource, offset: u64, record_size: u64) -> Result<Record> {
    let bytes = src.read_at(offset, record_size as usize)?;
    let (header, payload, _) = decode_record(&bytes)?;
    verify_record_crc(&header, payload)?;
    Ok(Record {
        header,
        payload: payload.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};

    fn tmp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_logfile_{tag}_{nanos}.log"));
        p
    }

    fn rec(stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        let header = RecordHeader {
            record_type: RecordType::Chunk,
            format: 0,
            payload_len: payload.len() as u64,
            crc: 0, // overwritten by encode_record
            stream_id,
            chunk_index,
            token_count: 32,
        };
        encode_record(&header, payload)
    }

    #[test]
    fn create_open_roundtrip() {
        let path = tmp_path("create_open");
        {
            let log = LogFile::create(&path).unwrap();
            assert_eq!(log.write_offset(), SUPERBLOCK_SIZE);
            assert_eq!(log.superblock().format_version, FILE_FORMAT_VERSION);
        }
        {
            let log = LogFile::open(&path).unwrap();
            assert_eq!(log.superblock().format_version, FILE_FORMAT_VERSION);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stage_flush_commit_and_read_back() {
        let path = tmp_path("stage_flush");
        let r0 = rec(1, 0, b"first");
        let r1 = rec(1, 1, b"second-record-payload");
        {
            let mut log = LogFile::create(&path).unwrap();
            let off0 = log.stage(&r0);
            let off1 = log.stage(&r1);
            assert_eq!(off0, SUPERBLOCK_SIZE);
            assert_eq!(off1, SUPERBLOCK_SIZE + r0.len() as u64);
            assert_eq!(log.pending_len(), r0.len() + r1.len());
            log.commit().unwrap();
            assert_eq!(log.pending_len(), 0);
            assert_eq!(
                log.write_offset(),
                SUPERBLOCK_SIZE + (r0.len() + r1.len()) as u64
            );
        }
        {
            let mut log = LogFile::open(&path).unwrap();
            let back0 = log.read_at(SUPERBLOCK_SIZE, r0.len()).unwrap();
            assert_eq!(back0, r0);
            let back1 = log
                .read_at(SUPERBLOCK_SIZE + r0.len() as u64, r1.len())
                .unwrap();
            assert_eq!(back1, r1);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn pre_grow_keeps_logical_length_correct() {
        let path = tmp_path("pregrow");
        let r = rec(2, 0, b"x");
        {
            let mut log = LogFile::create(&path).unwrap();
            // Physical file is grown to a 1 MiB extent.
            assert!(log.allocated_len() >= GROW_EXTENT);
            log.stage(&r);
            log.commit().unwrap();
            // Logical end is exact, regardless of the pre-grown physical size.
            assert_eq!(log.write_offset(), SUPERBLOCK_SIZE + r.len() as u64);
            assert!(log.allocated_len() >= log.write_offset());
        }
        std::fs::remove_file(&path).ok();
    }

    /// Superblocks written by earlier builds carry the retired
    /// checkpoint offset in bytes 8..16 (now the index hint's offset
    /// half) and zeros in 16..24 (its size half). Decode must accept
    /// them — the bogus hint has a zero size, which recovery rejects
    /// into the full-walk fallback.
    #[test]
    fn legacy_checkpoint_bytes_decode_as_a_rejectable_hint() {
        let path = tmp_path("reserved_bytes");
        {
            LogFile::create(&path).unwrap();
        }
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut b = vec![0u8; SUPERBLOCK_SIZE as usize];
            b[0..4].copy_from_slice(&FILE_MAGIC.to_le_bytes());
            b[4..8].copy_from_slice(&FILE_FORMAT_VERSION.to_le_bytes());
            b[8..16].copy_from_slice(&123_456u64.to_le_bytes());
            let crc = crc32(&b[0..SUPERBLOCK_SIZE as usize - 4]);
            b[SUPERBLOCK_SIZE as usize - 4..].copy_from_slice(&crc.to_le_bytes());
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&b).unwrap();
        }
        {
            let log = LogFile::open(&path).unwrap();
            assert_eq!(log.superblock().format_version, FILE_FORMAT_VERSION);
            assert_eq!(log.superblock().last_index, (123_456, 0));
        }
        std::fs::remove_file(&path).ok();
    }

    /// A torn superblock (valid magic, bad CRC — the crash window of the
    /// per-index-flush superblock rewrite) self-heals on open instead of
    /// bricking the log: the hint zeroes out and the records survive.
    #[test]
    fn torn_superblock_self_heals_on_open() {
        let path = tmp_path("torn_sb");
        let r = rec(3, 0, b"survives");
        {
            let mut log = LogFile::create(&path).unwrap();
            log.stage(&r);
            log.commit().unwrap();
            log.set_last_index((SUPERBLOCK_SIZE, r.len() as u64))
                .unwrap();
        }
        // Tear the superblock: flip a byte in the CRC-covered reserved
        // region, leaving the magic + version intact.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(100)).unwrap();
            f.write_all(&[0xFF]).unwrap();
        }
        {
            let mut log = LogFile::open(&path).unwrap();
            assert_eq!(
                log.superblock().last_index,
                (0, 0),
                "healed superblock starts with a zero hint"
            );
            // The record region is untouched.
            let back = log.read_at(SUPERBLOCK_SIZE, r.len()).unwrap();
            assert_eq!(back, r);
            // The healed superblock is durable and writable again.
            log.set_last_index((SUPERBLOCK_SIZE, r.len() as u64))
                .unwrap();
        }
        {
            let log = LogFile::open(&path).unwrap();
            assert_eq!(
                log.superblock().last_index,
                (SUPERBLOCK_SIZE, r.len() as u64)
            );
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn last_index_hint_persists_across_reopen() {
        let path = tmp_path("last_index");
        {
            let mut log = LogFile::create(&path).unwrap();
            assert_eq!(log.superblock().last_index, (0, 0));
            log.set_last_index((45_056, 8192)).unwrap();
        }
        {
            let log = LogFile::open(&path).unwrap();
            assert_eq!(log.superblock().last_index, (45_056, 8192));
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn corrupt_superblock_rejected() {
        let path = tmp_path("bad_sb");
        {
            LogFile::create(&path).unwrap();
        }
        // Smash the file magic.
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
        }
        assert!(LogFile::open(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn memlog_source_reads_records() {
        let r0 = rec(5, 0, b"alpha");
        let mut blob = Vec::new();
        blob.extend_from_slice(&r0);
        let mut mem = MemLog::with_records(&blob);
        let record = read_record_at(&mut mem, SUPERBLOCK_SIZE, r0.len() as u64).unwrap();
        assert_eq!(record.header.stream_id, 5);
        assert_eq!(record.header.record_type, RecordType::Chunk);
        assert_eq!(record.payload, b"alpha");
    }

    /// A flipped payload byte surfaces as `BadChecksum` at the
    /// consumption point — `read_record_at` CRC-verifies every payload
    /// it returns.
    #[test]
    fn read_record_at_rejects_payload_bit_rot() {
        let mut r0 = rec(5, 0, b"alpha-payload");
        let newline = r0.iter().position(|&b| b == b'\n').unwrap();
        r0[newline + 1] ^= 0x5A;
        let mut mem = MemLog::with_records(&r0);
        let err = read_record_at(&mut mem, SUPERBLOCK_SIZE, r0.len() as u64).unwrap_err();
        assert!(
            matches!(err, PersistenceError::BadChecksum { .. }),
            "expected BadChecksum, got {err:?}"
        );
    }
}

//! The skip-load walk (§5.4 of `docs/kv_tier_migration.md`).
//!
//! A walk steps record by record from a start offset, framing each record
//! by its header `length` and verifying its CRC. It stops cleanly at the
//! end of valid data (the zero-filled pre-grown tail, or EOF) and reports a
//! torn record if it meets one — the recovery truncation point.

use super::log_file::LogSource;
use super::record::{decode_header, decode_record, padded_record_len, Record, HEADER_SIZE};
use super::Result;

/// One record encountered by a walk.
#[derive(Clone, Debug)]
pub struct WalkEntry {
    /// File offset of the record.
    pub offset: u64,
    /// The decoded, checksum-verified record.
    pub record: Record,
    /// Padded on-disk size of the record.
    pub size: u64,
}

/// The outcome of a walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkOutcome {
    /// Number of valid records visited.
    pub records: usize,
    /// First byte past the last valid record — the recovered log tail.
    pub tail_offset: u64,
    /// Whether the walk stopped on a torn / corrupt record (`true`) rather
    /// than a clean end (`false`). Recovery truncates the file to
    /// `tail_offset` when this is `true`.
    pub torn: bool,
}

fn all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Walk records from `start`, invoking `visit` for each valid record in
/// file order. Returns where and how the walk ended.
pub fn walk(
    src: &mut dyn LogSource,
    start: u64,
    mut visit: impl FnMut(&WalkEntry),
) -> Result<WalkOutcome> {
    let size = src.size()?;
    let mut offset = start;
    let mut records = 0usize;

    loop {
        if offset + HEADER_SIZE as u64 > size {
            // Not enough room for another header — clean end.
            return Ok(WalkOutcome {
                records,
                tail_offset: offset,
                torn: false,
            });
        }
        let header_bytes = src.read_at(offset, HEADER_SIZE)?;
        if all_zero(&header_bytes) {
            // Zero-filled pre-grown region — clean end.
            return Ok(WalkOutcome {
                records,
                tail_offset: offset,
                torn: false,
            });
        }
        let header = match decode_header(&header_bytes) {
            Ok(h) => h,
            // A non-zero, non-decodable header is a torn write.
            Err(_) => {
                return Ok(WalkOutcome {
                    records,
                    tail_offset: offset,
                    torn: true,
                })
            }
        };
        let total = padded_record_len(header.payload_len) as u64;
        if offset + total > size {
            // Header promises a record the file does not fully hold.
            return Ok(WalkOutcome {
                records,
                tail_offset: offset,
                torn: true,
            });
        }
        let record_bytes = src.read_at(offset, total as usize)?;
        let (record, consumed) = match decode_record(&record_bytes) {
            Ok(decoded) => decoded,
            // Header framed it but the checksum failed — torn.
            Err(_) => {
                return Ok(WalkOutcome {
                    records,
                    tail_offset: offset,
                    torn: true,
                })
            }
        };
        debug_assert_eq!(consumed as u64, total);
        let entry = WalkEntry {
            offset,
            record,
            size: total,
        };
        visit(&entry);
        records += 1;
        offset += total;
    }
}

/// Walk and collect every entry into a `Vec` — convenience for tests and
/// small logs.
pub fn collect(src: &mut dyn LogSource, start: u64) -> Result<(Vec<WalkEntry>, WalkOutcome)> {
    let mut entries = Vec::new();
    let outcome = walk(src, start, |e| entries.push(e.clone()))?;
    Ok((entries, outcome))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};
    use crate::persistence::record::{encode_record, RecordHeader, RecordType};

    fn chunk(stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: RecordType::Chunk,
                format: 0,
                payload_len: payload.len() as u64,
                stream_id,
                chunk_index,
                token_count: 32,
            },
            payload,
        )
    }

    #[test]
    fn walks_every_record_in_order() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"aaa"));
        blob.extend_from_slice(&chunk(1, 1, b"bbbbbb"));
        blob.extend_from_slice(&chunk(2, 0, b"c"));
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(outcome.records, 3);
        assert!(!outcome.torn);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].record.header.stream_id, 1);
        assert_eq!(entries[0].record.header.chunk_index, 0);
        assert_eq!(entries[1].record.header.chunk_index, 1);
        assert_eq!(entries[2].record.header.stream_id, 2);
        assert_eq!(entries[0].record.payload, b"aaa");
    }

    #[test]
    fn empty_log_walk_is_clean() {
        let mut mem = MemLog::with_records(&[]);
        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(entries.is_empty());
        assert_eq!(outcome.records, 0);
        assert!(!outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE);
    }

    #[test]
    fn zero_filled_tail_is_a_clean_end() {
        let mut blob = chunk(1, 0, b"hi");
        let real_len = blob.len();
        blob.extend_from_slice(&[0u8; 8192]); // simulate pre-grown region
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(!outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE + real_len as u64);
    }

    #[test]
    fn torn_tail_record_stops_the_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"good"));
        let good_len = blob.len();
        // A second record with a corrupted payload byte (the payload sits at
        // [64, 74); byte 66 is inside it and inside the checksummed range).
        let mut bad = chunk(1, 1, b"corrupt-me");
        bad[66] ^= 0x5A;
        blob.extend_from_slice(&bad);
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE + good_len as u64);
    }

    #[test]
    fn physically_truncated_tail_is_torn() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"good"));
        let good_len = blob.len();
        // Only the first half of the second record reached disk — the
        // header frames a 4 KB record the file does not fully hold.
        let second = chunk(1, 1, b"second");
        blob.extend_from_slice(&second[..2048]);
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE + good_len as u64);
    }
}

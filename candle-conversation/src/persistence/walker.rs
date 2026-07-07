//! The skip-load walk.
//!
//! A walk steps record by record from a start offset, framing each
//! record by the `payload_len` declared in its JSON header and
//! verifying the payload CRC. It stops cleanly at the zero-filled
//! pre-grown tail (or EOF) and reports a torn record at the first
//! unparseable / truncated / bad-CRC record — the recovery
//! truncation point.
//!
//! Records whose header carries an unrecognised `type` are silently
//! skipped (their padded size is still known from the header's
//! `payload_len`, so the walker advances past them and continues).
//! This is the forward-compatibility lever that lets newer writers
//! add record kinds without making older readers fail to recover.

use super::log_file::LogSource;
use super::record::{decode_header, decode_record, padded_record_len, Record, RecordType, ALIGN};
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
    /// Number of valid, known-type records visited.
    pub records: usize,
    /// Number of records with an unknown record-type tag that were
    /// silently skipped. Diagnostic only — these are forward-compat
    /// hits, not corruption.
    pub unknown_records: usize,
    /// First byte past the last valid record — the recovered log tail.
    pub tail_offset: u64,
    /// Whether the walk stopped on a torn / corrupt record (`true`)
    /// rather than a clean end (`false`). Recovery truncates the file
    /// to `tail_offset` when this is `true`.
    pub torn: bool,
}

fn all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Walk records from `start`, invoking `visit` for each valid,
/// **known-type** record in file order. Records of unknown type are
/// counted but skipped silently.
pub fn walk(
    src: &mut dyn LogSource,
    start: u64,
    mut visit: impl FnMut(&WalkEntry),
) -> Result<WalkOutcome> {
    let size = src.size()?;
    let mut offset = start;
    let mut records = 0usize;
    let mut unknown_records = 0usize;

    loop {
        let remaining = size.saturating_sub(offset);
        if remaining == 0 {
            return Ok(WalkOutcome {
                records,
                unknown_records,
                tail_offset: offset,
                torn: false,
            });
        }
        // Probe the first sector to learn the record's framing. The
        // JSON header is guaranteed to fit within one sector.
        let probe_len = remaining.min(ALIGN as u64) as usize;
        let probe = src.read_at(offset, probe_len)?;
        if all_zero(&probe) {
            // Pre-grown zero-filled tail — clean end.
            return Ok(WalkOutcome {
                records,
                unknown_records,
                tail_offset: offset,
                torn: false,
            });
        }
        let (header, header_bytes) = match decode_header(&probe) {
            Ok(h) => h,
            Err(_) => {
                return Ok(WalkOutcome {
                    records,
                    unknown_records,
                    tail_offset: offset,
                    torn: true,
                })
            }
        };
        let total = padded_record_len(header_bytes - 1, header.payload_len) as u64;
        if offset + total > size {
            // The header promises a record the file doesn't fully hold.
            return Ok(WalkOutcome {
                records,
                unknown_records,
                tail_offset: offset,
                torn: true,
            });
        }
        // Skip-unknown: we know the padded size without touching the
        // payload, so advance past the record without invoking
        // `decode_record` (which would needlessly construct a Record
        // we'd just discard) and without visiting.
        if header.record_type == RecordType::Unknown {
            unknown_records += 1;
            offset += total;
            continue;
        }
        // Location-only records — `Chunk` (KV), `Tokens` — carry the
        // overwhelming bulk of a large log's bytes, but the consumer indexes them by
        // `(offset, payload_len, record_size)` and reads the payload lazily by offset
        // later: `apply_walker_entry` never reads their `payload`. So skip the second
        // read of the whole record (and the payload copy) and hand back just the header
        // we already parsed from the probe sector. This is what makes a multi-GB walk
        // cheap — without it the walk re-reads and copies the entire log into RAM. The
        // walk's torn-record detection is header-framed (size/zero-fill), so skipping
        // the payload read changes nothing there.
        if matches!(header.record_type, RecordType::Chunk | RecordType::Tokens) {
            let entry = WalkEntry {
                offset,
                record: Record {
                    header,
                    payload: Vec::new(),
                },
                size: total,
            };
            visit(&entry);
            records += 1;
            offset += total;
            continue;
        }
        let record_bytes = src.read_at(offset, total as usize)?;
        let (header, payload, consumed) = match decode_record(&record_bytes) {
            Ok(decoded) => decoded,
            Err(_) => {
                return Ok(WalkOutcome {
                    records,
                    unknown_records,
                    tail_offset: offset,
                    torn: true,
                })
            }
        };
        debug_assert_eq!(consumed as u64, total);
        let entry = WalkEntry {
            offset,
            record: Record {
                header,
                payload: payload.to_vec(),
            },
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
                crc: 0, // overwritten by encode_record
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
        assert_eq!(outcome.unknown_records, 0);
        assert!(!outcome.torn);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].record.header.stream_id, 1);
        assert_eq!(entries[0].record.header.chunk_index, 0);
        assert_eq!(entries[1].record.header.chunk_index, 1);
        assert_eq!(entries[2].record.header.stream_id, 2);
        // Chunk is location-only: its payload is skip-loaded (the consumer reads KV
        // lazily by offset). The header — including payload_len — is preserved.
        assert!(entries[0].record.payload.is_empty());
        assert_eq!(entries[0].record.header.payload_len, 3);
    }

    fn small_record(rt: RecordType, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
            },
            payload,
        )
    }

    #[test]
    fn location_records_skip_payload_other_records_retain_it() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"chunk-kv-bytes"));
        blob.extend_from_slice(&small_record(RecordType::Commit, b"commit-payload"));
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(outcome.records, 2);
        // Location-only record: payload skipped, header (incl. len) intact.
        assert!(entries[0].record.payload.is_empty());
        assert_eq!(
            entries[0].record.header.payload_len,
            b"chunk-kv-bytes".len() as u64
        );
        // Non-location record: payload fully read.
        assert_eq!(entries[1].record.payload, b"commit-payload");
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

    /// Header corruption stops the walk as torn — the new boundary
    /// for synchronous recovery now that walker no longer CRCs
    /// payloads inline. (Payload bit-rot is caught out-of-band by the
    /// background CRC validator.)
    #[test]
    fn torn_tail_header_stops_the_walk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"good"));
        let good_len = blob.len();
        let mut bad = chunk(1, 1, b"second");
        // Replace the first byte (the opening `{`) so JSON parse fails
        // immediately — `decode_header` rejects anything not starting
        // with `{`.
        bad[0] = b'X';
        blob.extend_from_slice(&bad);
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE + good_len as u64);
    }

    /// Payload bit-rot is **not** a synchronous torn write — the walker
    /// passes it through. The background CRC validator surfaces it
    /// later via [`verify_record_crc`].
    #[test]
    fn payload_bit_rot_passes_walker_silently() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"good"));
        let mut bad = chunk(1, 1, b"corrupt-me");
        let newline_pos = bad.iter().position(|&b| b == b'\n').unwrap();
        bad[newline_pos + 1] ^= 0x5A;
        blob.extend_from_slice(&bad);
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 2, "walker accepts both records");
        assert!(
            !outcome.torn,
            "walker does not stop on payload CRC mismatch"
        );
    }

    #[test]
    fn physically_truncated_tail_is_torn() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"good"));
        let good_len = blob.len();
        // Only the first half of the second record reached disk — the
        // header frames a record the file does not fully hold.
        let second = chunk(1, 1, b"second");
        blob.extend_from_slice(&second[..2048]);
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(outcome.torn);
        assert_eq!(outcome.tail_offset, SUPERBLOCK_SIZE + good_len as u64);
    }

    /// A record whose header carries a `type` this version doesn't
    /// recognise must be **silently skipped** — the walk continues
    /// past it and surfaces it as an `unknown_records` count.
    /// Forward compatibility with new record kinds.
    #[test]
    fn unknown_type_records_are_skipped_without_torn() {
        // Build a record with an unknown type by hand.
        let payload = b"future-bytes";
        let crc = super::super::record::crc32(payload);
        let header_line = format!(
            "{{\"type\":\"future_kind\",\"payload_len\":{},\"crc\":{}}}",
            payload.len(),
            crc
        );
        let header_len = header_line.len();
        let total = padded_record_len(header_len, payload.len() as u64);
        let mut unknown = vec![0u8; total];
        unknown[..header_len].copy_from_slice(header_line.as_bytes());
        unknown[header_len] = b'\n';
        unknown[header_len + 1..header_len + 1 + payload.len()].copy_from_slice(payload);

        let mut blob = Vec::new();
        blob.extend_from_slice(&chunk(1, 0, b"first"));
        blob.extend_from_slice(&unknown);
        blob.extend_from_slice(&chunk(2, 0, b"third"));
        let mut mem = MemLog::with_records(&blob);

        let (entries, outcome) = collect(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(!outcome.torn, "unknown type is not a torn record");
        assert_eq!(outcome.records, 2, "the two known records are visited");
        assert_eq!(outcome.unknown_records, 1, "the unknown record is counted");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].record.header.stream_id, 1);
        assert_eq!(entries[1].record.header.stream_id, 2);
    }
}

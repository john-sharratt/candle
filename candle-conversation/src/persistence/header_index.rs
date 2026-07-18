//! The header-index chain — batched record digests for fast recovery.
//!
//! Recovery's cost driver is that record offsets are discovered one
//! serial read at a time: each header read reveals where the next
//! record starts, so a multi-GB log walk degenerates into hundreds of
//! thousands of queue-depth-1 sector reads. A `HeaderIndex` record
//! collapses that chain: its payload carries a fixed-width **digest**
//! of every record appended since the previous index, plus the file
//! location of that previous index. Recovery follows the backward
//! chain from the superblock hint (a handful of reads for the whole
//! log), replays the digests in append order, batch-fetches the few
//! payload-bearing metadata records whose offsets the digests expose,
//! and forward-walks only the un-indexed tail.
//!
//! Index records are **derived data**: any inconsistency (stale hint,
//! torn index, unknown version) makes recovery fall back to the plain
//! forward walk, and compaction drops every index record and the
//! writer regenerates the chain in the new file.
//!
//! ## Payload wire format (little-endian)
//!
//! ```text
//! u32  version            (INDEX_PAYLOAD_VERSION)
//! u64  prev_offset        (file offset of the previous HeaderIndex; 0 = chain start)
//! u64  prev_size          (padded on-disk size of that record; 0 = chain start)
//! u32  n_entries
//! n × 38-byte entry:
//!   u8   record_type tag  (RecordType::tag)
//!   u8   format
//!   u32  token_count
//!   u64  stream_id
//!   u64  chunk_index
//!   u64  offset           (file offset of the digested record)
//!   u32  record_size      (padded on-disk size)
//!   u32  payload_len      (unpadded payload length)
//! ```

use super::record::{Record, RecordHeader, RecordType};
use super::segment::SegmentId;
use super::walker::WalkEntry;
use super::{PersistenceError, Result};

/// Version stamp of the `HeaderIndex` payload encoding. A reader that
/// sees a different version falls back to the forward walk rather
/// than guessing at the layout.
pub const INDEX_PAYLOAD_VERSION: u32 = 1;

/// Byte size of one encoded digest entry.
pub const INDEX_ENTRY_BYTES: usize = 38;

/// Number of digests the writer accumulates before flushing a
/// `HeaderIndex` record (≈ 152 KB of payload). Recovery forward-walks
/// at most this many un-indexed tail records, so the constant bounds
/// the slow part of a restart.
pub const INDEX_FLUSH_ENTRIES: usize = 4096;

/// One record's header digest — everything recovery needs to replay
/// the record without touching it on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    pub record_type: RecordType,
    pub format: u8,
    pub token_count: u32,
    pub stream_id: u64,
    pub chunk_index: u64,
    /// File offset of the digested record.
    pub offset: u64,
    /// Padded on-disk size of the digested record.
    pub record_size: u32,
    /// Unpadded payload length of the digested record.
    pub payload_len: u32,
}

impl IndexEntry {
    /// Digest a just-appended (or just-walked) record.
    pub fn from_header(header: &RecordHeader, offset: u64, record_size: u64) -> IndexEntry {
        IndexEntry {
            record_type: header.record_type,
            format: header.format,
            token_count: header.token_count as u32,
            stream_id: header.stream_id,
            chunk_index: header.chunk_index,
            offset,
            record_size: record_size as u32,
            payload_len: header.payload_len as u32,
        }
    }

    /// Synthesize the payload-less [`WalkEntry`] this digest stands in
    /// for — identical to what a payload-skipping forward walk would
    /// have visited for the same record (the header's `crc` is not
    /// carried by the digest; entries synthesized here never have
    /// their payload consumed, so nothing reads it).
    pub fn to_walk_entry(self, segment: SegmentId) -> WalkEntry {
        WalkEntry {
            segment,
            offset: self.offset,
            record: Record {
                header: RecordHeader {
                    record_type: self.record_type,
                    format: self.format,
                    payload_len: self.payload_len as u64,
                    crc: 0,
                    stream_id: self.stream_id,
                    chunk_index: self.chunk_index,
                    token_count: self.token_count as u64,
                },
                payload: Vec::new(),
            },
            size: self.record_size as u64,
        }
    }
}

/// Encode a `HeaderIndex` record payload: the digests appended since
/// the previous index, chained to `prev = (offset, padded_size)`
/// (`(0, 0)` starts a chain).
pub fn encode_index_payload(prev: (u64, u64), entries: &[IndexEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 8 + 8 + 4 + entries.len() * INDEX_ENTRY_BYTES);
    out.extend_from_slice(&INDEX_PAYLOAD_VERSION.to_le_bytes());
    out.extend_from_slice(&prev.0.to_le_bytes());
    out.extend_from_slice(&prev.1.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.push(e.record_type.tag());
        out.push(e.format);
        out.extend_from_slice(&e.token_count.to_le_bytes());
        out.extend_from_slice(&e.stream_id.to_le_bytes());
        out.extend_from_slice(&e.chunk_index.to_le_bytes());
        out.extend_from_slice(&e.offset.to_le_bytes());
        out.extend_from_slice(&e.record_size.to_le_bytes());
        out.extend_from_slice(&e.payload_len.to_le_bytes());
    }
    out
}

/// Decode a `HeaderIndex` record payload — the inverse of
/// [`encode_index_payload`]. Returns `(prev, entries)`.
pub fn decode_index_payload(payload: &[u8]) -> Result<((u64, u64), Vec<IndexEntry>)> {
    let mut pos = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8]> {
        if *p + n > payload.len() {
            return Err(PersistenceError::Truncated {
                need: n,
                have: payload.len().saturating_sub(*p),
            });
        }
        let s = &payload[*p..*p + n];
        *p += n;
        Ok(s)
    };
    let version = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap());
    if version != INDEX_PAYLOAD_VERSION {
        return Err(PersistenceError::Corrupt(format!(
            "HeaderIndex payload version {version}, expected {INDEX_PAYLOAD_VERSION}"
        )));
    }
    let prev_offset = u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap());
    let prev_size = u64::from_le_bytes(take(&mut pos, 8)?.try_into().unwrap());
    let n = u32::from_le_bytes(take(&mut pos, 4)?.try_into().unwrap()) as usize;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let e = take(&mut pos, INDEX_ENTRY_BYTES)?;
        entries.push(IndexEntry {
            record_type: RecordType::from_tag(e[0]),
            format: e[1],
            token_count: u32::from_le_bytes(e[2..6].try_into().unwrap()),
            stream_id: u64::from_le_bytes(e[6..14].try_into().unwrap()),
            chunk_index: u64::from_le_bytes(e[14..22].try_into().unwrap()),
            offset: u64::from_le_bytes(e[22..30].try_into().unwrap()),
            record_size: u32::from_le_bytes(e[30..34].try_into().unwrap()),
            payload_len: u32::from_le_bytes(e[34..38].try_into().unwrap()),
        });
    }
    if pos != payload.len() {
        return Err(PersistenceError::Corrupt(format!(
            "HeaderIndex payload has {} trailing bytes",
            payload.len() - pos
        )));
    }
    Ok(((prev_offset, prev_size), entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seed: u64) -> IndexEntry {
        IndexEntry {
            record_type: RecordType::Chunk,
            format: 4,
            token_count: 32,
            stream_id: seed,
            chunk_index: seed * 7,
            offset: 4096 + seed * 8192,
            record_size: 8192,
            payload_len: 5000,
        }
    }

    /// Raw expected bytes for a one-entry payload — the wire format is
    /// pinned byte-for-byte, not round-trip-only.
    #[test]
    fn encode_matches_raw_expected_bytes() {
        let e = IndexEntry {
            record_type: RecordType::Tokens, // tag 5
            format: 0x0A,
            token_count: 0x0102_0304,
            stream_id: 0x1112_1314_1516_1718,
            chunk_index: 0x2122_2324_2526_2728,
            offset: 0x3132_3334_3536_3738,
            record_size: 0x4142_4344,
            payload_len: 0x5152_5354,
        };
        let bytes = encode_index_payload((0xAABB, 0x1000), &[e]);
        let mut expected = Vec::new();
        expected.extend_from_slice(&1u32.to_le_bytes()); // version
        expected.extend_from_slice(&0xAABBu64.to_le_bytes()); // prev_offset
        expected.extend_from_slice(&0x1000u64.to_le_bytes()); // prev_size
        expected.extend_from_slice(&1u32.to_le_bytes()); // n_entries
        expected.push(5); // Tokens tag
        expected.push(0x0A); // format
        expected.extend_from_slice(&0x0102_0304u32.to_le_bytes());
        expected.extend_from_slice(&0x1112_1314_1516_1718u64.to_le_bytes());
        expected.extend_from_slice(&0x2122_2324_2526_2728u64.to_le_bytes());
        expected.extend_from_slice(&0x3132_3334_3536_3738u64.to_le_bytes());
        expected.extend_from_slice(&0x4142_4344u32.to_le_bytes());
        expected.extend_from_slice(&0x5152_5354u32.to_le_bytes());
        assert_eq!(bytes, expected);
        assert_eq!(bytes.len(), 4 + 8 + 8 + 4 + INDEX_ENTRY_BYTES);
    }

    #[test]
    fn round_trips_many_entries() {
        let entries: Vec<IndexEntry> = (0..1000).map(entry).collect();
        let bytes = encode_index_payload((77_824, 4096), &entries);
        let (prev, decoded) = decode_index_payload(&bytes).unwrap();
        assert_eq!(prev, (77_824, 4096));
        assert_eq!(decoded, entries);
    }

    #[test]
    fn chain_start_is_zero_zero() {
        let bytes = encode_index_payload((0, 0), &[]);
        let (prev, decoded) = decode_index_payload(&bytes).unwrap();
        assert_eq!(prev, (0, 0));
        assert!(decoded.is_empty());
    }

    #[test]
    fn unknown_version_is_rejected() {
        let mut bytes = encode_index_payload((0, 0), &[entry(1)]);
        bytes[0] = 99;
        assert!(matches!(
            decode_index_payload(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }

    #[test]
    fn truncated_and_trailing_bytes_are_rejected() {
        let bytes = encode_index_payload((0, 0), &[entry(1), entry(2)]);
        assert!(decode_index_payload(&bytes[..bytes.len() - 5]).is_err());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(decode_index_payload(&padded).is_err());
    }

    /// A digest replays as exactly the WalkEntry a payload-skipping
    /// walk would have produced for the same record.
    #[test]
    fn digest_round_trips_through_walk_entry() {
        let header = RecordHeader {
            record_type: RecordType::Commit,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id: 42,
            chunk_index: 191, // through_index
            token_count: 0,
        };
        let e = IndexEntry::from_header(&header, 20_480, 4096);
        let w = e.to_walk_entry(crate::persistence::segment::FIRST_SEGMENT);
        assert_eq!(w.offset, 20_480);
        assert_eq!(w.size, 4096);
        assert_eq!(w.record.header, header);
        assert!(w.record.payload.is_empty());
    }
}

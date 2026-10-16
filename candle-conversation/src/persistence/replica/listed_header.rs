//! What a catch-up's headers say of one held record (`docs/replicated_substrate.md`
//! §7.2), kept beside its location so a listing reads nothing off disk.
//!
//! **The length is the one field not read back exactly.** A record travels as
//! its header line, newline and payload, unpadded, and the line spells `crc` in
//! decimal. Every other field of the line is in memory however the record was
//! found — appended, relocated, walked, or replayed from a header-index digest —
//! but a digest does not carry `crc`. So the length counts the line with `crc`
//! at its widest: never under what travels, and at most nine bytes over. A
//! catch-up uses it to budget its fetch window and to show progress against.

use std::io::{self, Write};

use super::super::record::RecordHeader;

/// See the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListedHeader {
    /// The record's type tag.
    pub record_type: u8,
    pub stream_id: u64,
    pub chunk_index: u64,
    /// The record as it travels, its `crc` counted at its widest.
    pub len: u32,
    pub digest: [u8; 16],
}

impl ListedHeader {
    /// Nothing listed: the value of an index slot no record holds.
    pub const NONE: ListedHeader = ListedHeader {
        record_type: 0,
        stream_id: 0,
        chunk_index: 0,
        len: 0,
        digest: [0; 16],
    };

    pub fn of(header: &RecordHeader) -> ListedHeader {
        ListedHeader {
            record_type: header.record_type.tag(),
            stream_id: header.stream_id,
            chunk_index: header.chunk_index,
            len: travelling_len(header),
            digest: header.digest.to_bytes(),
        }
    }
}

/// Bytes written, and nothing kept.
struct Counted(usize);

impl Write for Counted {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `header`'s record as it travels — its header line, the newline and the
/// payload — with the line's `crc` at its widest.
fn travelling_len(header: &RecordHeader) -> u32 {
    let widest = RecordHeader {
        crc: u32::MAX,
        ..*header
    };
    let mut line = Counted(0);
    serde_json::to_writer(&mut line, &widest)
        .expect("RecordHeader serialization is infallible for the supported field set");
    (line.0 as u64 + 1 + header.payload_len) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::ContentHash;
    use crate::persistence::record::{encode_record_written, RecordType};

    fn header(record_type: RecordType, payload_len: u64) -> RecordHeader {
        RecordHeader {
            record_type,
            payload_len,
            stream_id: 0x1234,
            chunk_index: 7,
            origin: 0xABCD,
            oseq: 42,
            gen: 3,
            hlc: 99 << 16,
            ..RecordHeader::NONE
        }
    }

    /// The bytes `read_event` hands out for a record: its header line,
    /// newline and payload, unpadded.
    fn travelling(header: &RecordHeader, payload: &[u8]) -> (RecordHeader, usize) {
        let (written, image) = encode_record_written(header, payload);
        let line = image.iter().position(|b| *b == b'\n').unwrap() + 1;
        (written, line + payload.len())
    }

    /// The line spelled out: with `crc` at its widest, the listed length is
    /// exactly the line's bytes plus the newline and payload.
    #[test]
    fn the_length_is_the_line_at_its_widest_crc_and_the_payload() {
        let h = RecordHeader {
            record_type: RecordType::Tokens,
            payload_len: 3,
            stream_id: 9,
            chunk_index: 1,
            origin: 5,
            oseq: 2,
            ..RecordHeader::NONE
        };
        let line = "{\"type\":\"tokens\",\"payload_len\":3,\"crc\":4294967295,\"format\":0,\
                    \"stream_id\":9,\"chunk_index\":1,\"token_count\":0,\"origin\":5,\"oseq\":2}";
        assert_eq!(travelling_len(&h), line.len() as u32 + 1 + 3);
    }

    /// Against what really travels: never under it, at most nine bytes over,
    /// and exact when the written `crc` is ten digits wide.
    #[test]
    fn the_length_bounds_what_travels_within_nine_bytes() {
        for (record_type, payload) in [
            (RecordType::Tokens, vec![1u8, 2, 3]),
            (RecordType::Commit, Vec::new()),
            (RecordType::CustomObject, vec![7u8; 5000]),
        ] {
            let h = header(record_type, payload.len() as u64);
            let (written, exact) = travelling(&h, &payload);
            let listed = ListedHeader::of(&written).len as usize;
            let crc_digits = written.crc.to_string().len();
            assert_eq!(listed - exact, 10 - crc_digits, "{record_type:?}");
        }
    }

    /// A `Chunk` carries its golden in `crc`, set by the caller and written as
    /// given: a zero golden is the widest gap, nine bytes.
    #[test]
    fn a_chunk_with_a_one_digit_golden_is_nine_bytes_over() {
        let h = RecordHeader {
            format: 4,
            token_count: 32,
            ..header(RecordType::Chunk, 64)
        };
        let (written, exact) = travelling(&h, &[0u8; 64]);
        assert_eq!(written.crc, 0);
        assert_eq!(ListedHeader::of(&written).len as usize, exact + 9);
    }

    /// The other fields are the header's, the digest as its bytes.
    #[test]
    fn the_fields_are_the_headers() {
        let h = RecordHeader {
            digest: ContentHash {
                lo: 0x0807_0605_0403_0201,
                hi: 0x100F_0E0D_0C0B_0A09,
            },
            ..header(RecordType::Commit, 0)
        };
        let listed = ListedHeader::of(&h);
        assert_eq!(
            (listed.record_type, listed.stream_id, listed.chunk_index),
            (RecordType::Commit.tag(), 0x1234, 7)
        );
        assert_eq!(listed.digest, h.digest.to_bytes());
    }
}

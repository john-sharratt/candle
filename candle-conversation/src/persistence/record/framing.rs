//! Frames a record as a JSON header line, a payload, and zero padding to a sector.
//!
//! Every record is padded to a sector boundary so it can be read unbuffered, and
//! the header must fit in the first sector so one read learns the framing. The
//! writer rejects oversized headers and unknown types when it encodes. A
//! host-authored record gets a CRC over its payload. A chunk keeps the GPU golden
//! its caller already set, and the digest is always the payload's own.

use super::*;

/// Record / sector alignment. Every encoded record is padded to a
/// multiple of this so it begins on a sector boundary for unbuffered
/// I/O. The header line is required to fit within the first
/// [`ALIGN`] bytes of the record so a single sector read is enough
/// to learn the record's framing.
pub const ALIGN: usize = 4096;

/// Initial bytes read when probing a record's header. Sized to one
/// sector — the header line is guaranteed to fit within this.
pub const HEADER_PROBE_SIZE: usize = ALIGN;

/// Hard upper bound on the JSON header line, exclusive of the trailing
/// newline. The header must fit in one sector with at least one byte
/// left for the newline; pinning the bound here makes the writer
/// reject pathologically large headers (e.g. malformed manifests
/// trying to stream into the header) at encode time.
pub const MAX_HEADER_LINE: usize = ALIGN - 1;

/// Total on-disk size of a record whose JSON header line is
/// `header_line_len` bytes long (without the newline) and whose payload
/// is `payload_len` bytes, padded up to the [`ALIGN`] boundary.
pub fn padded_record_len(header_line_len: usize, payload_len: u64) -> usize {
    // +1 for the newline that ends the header.
    let raw = header_line_len + 1 + payload_len as usize;
    raw.div_ceil(ALIGN) * ALIGN
}

/// A record's framing, as [`record_head`] builds it.
pub struct RecordHead {
    /// The header as written: the caller's, with `crc` and `digest` filled in
    /// from the payload.
    pub header: RecordHeader,
    /// The header line with its trailing newline.
    pub line: Vec<u8>,
    /// The count of zero bytes that pad the record to its [`ALIGN`]ed span.
    pub pad: usize,
}

/// A record's framing around a payload given as `parts`. The on-disk record is
/// `line`, then the parts in order, then `pad` zero bytes — [`encode_record`]
/// lays exactly that into one buffer, and a large record is written from the
/// parts directly.
///
/// Panics if `header.payload_len` disagrees with the parts' total length, if
/// the resulting JSON header would exceed [`MAX_HEADER_LINE`] bytes, or if
/// `header.record_type` is [`RecordType::Unknown`].
pub fn record_head(header: &RecordHeader, parts: &[&[u8]]) -> RecordHead {
    let payload_len: usize = parts.iter().map(|p| p.len()).sum();
    assert_eq!(
        header.payload_len as usize, payload_len,
        "RecordHeader.payload_len must match the payload"
    );
    assert!(
        header.record_type != RecordType::Unknown,
        "encode_record refuses to write RecordType::Unknown — \
         that variant is the reader's catch-all for tags it does \
         not recognise, not a writable type"
    );
    let mut effective = *header;
    // `Chunk` records carry the GPU-computed golden (Fletcher-32 over the arena
    // bytes, taken before the device→host copy) in `crc`; the caller has already
    // set it. Recomputing host-side here would checksum the post-copy bytes and
    // reintroduce the very blind spot the golden exists to close, so leave it be.
    // Every other record type is host-authored — crc32 over the payload, chained
    // across its parts.
    if header.record_type != RecordType::Chunk {
        let crc = parts
            .iter()
            .fold(crc32_init(), |crc, part| crc32_update(crc, part));
        effective.crc = crc32_finish(crc);
    }
    // The digest is always the payload's, whatever the caller passed: a copy
    // carrying another record's digest would read as content-equal to it.
    let mut hasher = ContentHasher::new();
    for part in parts {
        hasher.update(part);
    }
    effective.digest = hasher.finish();

    let mut head = serde_json::to_string(&effective)
        .expect("RecordHeader serialization is infallible for the supported field set")
        .into_bytes();
    assert!(
        head.len() <= MAX_HEADER_LINE,
        "JSON record header is {} bytes, exceeds the {}-byte cap",
        head.len(),
        MAX_HEADER_LINE,
    );
    let total = padded_record_len(head.len(), effective.payload_len);
    head.push(b'\n');
    let pad = total - head.len() - payload_len;
    RecordHead {
        header: effective,
        line: head,
        pad,
    }
}

/// Encode a record into its padded on-disk byte image.
///
/// Panics as [`record_head`] does.
pub fn encode_record(header: &RecordHeader, payload: &[u8]) -> Vec<u8> {
    encode_record_written(header, payload).1
}

/// [`encode_record`], with the header as written beside the image — `crc` and
/// `digest` filled in — for the bookkeeping that indexes the record.
pub fn encode_record_written(header: &RecordHeader, payload: &[u8]) -> (RecordHeader, Vec<u8>) {
    let head = record_head(header, &[payload]);
    // Built by appending, not by zero-filling and overwriting: only the sector
    // padding tail is genuinely zero-valued, and `resize` writes just that.
    let total = head.line.len() + payload.len() + head.pad;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&head.line);
    out.extend_from_slice(payload);
    out.resize(total, 0);
    (head.header, out)
}

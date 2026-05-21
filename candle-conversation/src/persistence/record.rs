//! Redo-log record codec — the on-disk wire format (§5.3 of
//! `docs/kv_tier_migration.md`).
//!
//! Every record is a fixed 64-byte header followed by a variable payload,
//! the whole thing zero-padded to a 4 KB boundary so unbuffered, sector
//! aligned I/O can read any record. The header is self-describing: a reader
//! can skip a record knowing only its header (`length` → padded size).
//!
//! This module owns the framing codec and the little-endian byte
//! primitives the rest of the persistence layer encodes with. Payload
//! *content* (model spec, stream declarations, …) is encoded by the
//! modules that own those types.

use super::{PersistenceError, Result};

/// Record-boundary magic — ASCII `"SBL1"`, little-endian.
pub const RECORD_MAGIC: u32 = 0x314c_4253;

/// On-disk record-header layout version.
pub const HEADER_VERSION: u16 = 1;

/// Fixed record-header size in bytes.
pub const HEADER_SIZE: usize = 64;

/// Record / sector alignment. Every encoded record is padded to a multiple
/// of this so it begins on a sector boundary for unbuffered I/O.
pub const ALIGN: usize = 4096;

/// The eight record types (§5.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum RecordType {
    ModelSpec = 1,
    Template = 2,
    StreamDecl = 3,
    Chunk = 4,
    Tokens = 5,
    Signatures = 6,
    Commit = 7,
    Checkpoint = 8,
}

impl RecordType {
    fn from_tag(tag: u8) -> Result<RecordType> {
        Ok(match tag {
            1 => RecordType::ModelSpec,
            2 => RecordType::Template,
            3 => RecordType::StreamDecl,
            4 => RecordType::Chunk,
            5 => RecordType::Tokens,
            6 => RecordType::Signatures,
            7 => RecordType::Commit,
            8 => RecordType::Checkpoint,
            other => return Err(PersistenceError::UnknownRecordType(other)),
        })
    }
}

/// The fixed 64-byte record header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordHeader {
    pub record_type: RecordType,
    /// `Chunk` only — the `KvFormat` tag; 0 for every other record type.
    pub format: u8,
    /// Unpadded payload byte length.
    pub payload_len: u64,
    /// Owning stream, or 0 when not stream-scoped (`ModelSpec` / `Template`
    /// / `Checkpoint`).
    pub stream_id: u64,
    /// Position in the stream's local chunk grid (`Chunk` only).
    pub chunk_index: u64,
    /// Token count of a `Chunk` (`32` sealed, `<32` partial tail).
    pub token_count: u64,
}

/// A fully decoded record — its header and owned payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub header: RecordHeader,
    pub payload: Vec<u8>,
}

/// Total on-disk size of a record with `payload_len` payload bytes, padded
/// up to the [`ALIGN`] boundary.
pub fn padded_record_len(payload_len: u64) -> usize {
    let raw = HEADER_SIZE + payload_len as usize;
    raw.div_ceil(ALIGN) * ALIGN
}

/// Encode a record into its padded on-disk byte image.
pub fn encode_record(header: &RecordHeader, payload: &[u8]) -> Vec<u8> {
    assert_eq!(
        header.payload_len as usize,
        payload.len(),
        "RecordHeader.payload_len must match the payload slice"
    );
    let total = padded_record_len(header.payload_len);
    let mut out = vec![0u8; total];

    out[0..4].copy_from_slice(&RECORD_MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&HEADER_VERSION.to_le_bytes());
    out[6] = header.record_type as u8;
    out[7] = header.format;
    out[8..16].copy_from_slice(&header.payload_len.to_le_bytes());
    out[16..24].copy_from_slice(&header.stream_id.to_le_bytes());
    out[24..32].copy_from_slice(&header.chunk_index.to_le_bytes());
    out[32..40].copy_from_slice(&header.token_count.to_le_bytes());
    // bytes [40, 60) reserved — left zero.
    out[64..64 + payload.len()].copy_from_slice(payload);

    // Checksum covers header bytes [0, 60) and the unpadded payload.
    let crc = {
        let c = crc32_update(crc32_init(), &out[0..60]);
        crc32_finish(crc32_update(c, payload))
    };
    out[60..64].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Decode the header at the start of `buf` without copying the payload.
/// Validates magic, version, type, and that the buffer holds the whole
/// padded record. Does *not* verify the checksum (that needs the payload —
/// see [`decode_record`]).
pub fn decode_header(buf: &[u8]) -> Result<RecordHeader> {
    if buf.len() < HEADER_SIZE {
        return Err(PersistenceError::Truncated {
            need: HEADER_SIZE,
            have: buf.len(),
        });
    }
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != RECORD_MAGIC {
        return Err(PersistenceError::BadMagic {
            expected: RECORD_MAGIC,
            found: magic,
        });
    }
    let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
    if version != HEADER_VERSION {
        return Err(PersistenceError::Corrupt(format!(
            "unsupported header version {version}"
        )));
    }
    let record_type = RecordType::from_tag(buf[6])?;
    let format = buf[7];
    let payload_len = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let stream_id = u64::from_le_bytes(buf[16..24].try_into().unwrap());
    let chunk_index = u64::from_le_bytes(buf[24..32].try_into().unwrap());
    let token_count = u64::from_le_bytes(buf[32..40].try_into().unwrap());
    Ok(RecordHeader {
        record_type,
        format,
        payload_len,
        stream_id,
        chunk_index,
        token_count,
    })
}

/// Decode one full record from the start of `buf`. Returns the record and
/// the number of bytes it occupies on disk (its padded length, so a walker
/// can advance). Verifies the checksum.
pub fn decode_record(buf: &[u8]) -> Result<(Record, usize)> {
    let header = decode_header(buf)?;
    let total = padded_record_len(header.payload_len);
    if buf.len() < total {
        return Err(PersistenceError::Truncated {
            need: total,
            have: buf.len(),
        });
    }
    let payload = &buf[HEADER_SIZE..HEADER_SIZE + header.payload_len as usize];
    let stored_crc = u32::from_le_bytes(buf[60..64].try_into().unwrap());
    let computed = {
        let c = crc32_update(crc32_init(), &buf[0..60]);
        crc32_finish(crc32_update(c, payload))
    };
    if stored_crc != computed {
        return Err(PersistenceError::BadChecksum {
            header: stored_crc,
            computed,
        });
    }
    Ok((
        Record {
            header,
            payload: payload.to_vec(),
        },
        total,
    ))
}

/// The payload of a `Chunk` record (§5.3 of `docs/kv_tier_migration.md`):
/// a sealed or partial KV chunk's host-side quantization metadata followed
/// by its gathered arena bytes.
///
/// The metadata prefix is **mandatory** — the arena blob alone is not
/// self-describing, and the quantized KV cannot be dequantised without
/// `k_pal` / `v_pal` / `k_scale` / `v_scale`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPayload {
    /// The `SealedChunk` window skip-count (start of valid data).
    pub offset: u16,
    /// `KvFormat` tag of the K side — adaptive quantization picks K and V
    /// formats independently per block, so both must be persisted.
    pub k_format: u8,
    /// `KvFormat` tag of the V side.
    pub v_format: u8,
    /// Packed K palette maps.
    pub k_pal: Vec<u8>,
    /// Packed V palette maps.
    pub v_pal: Vec<u8>,
    /// Outer K scales.
    pub k_scale: Vec<f32>,
    /// Outer V scales.
    pub v_scale: Vec<f32>,
    /// The gathered arena KV blob for this chunk.
    pub kv_bytes: Vec<u8>,
}

impl ChunkPayload {
    /// Encode to the `Chunk` record payload bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.put_u16(self.offset);
        w.put_u8(self.k_format);
        w.put_u8(self.v_format);
        w.put_blob(&self.k_pal);
        w.put_blob(&self.v_pal);
        w.put_u32(self.k_scale.len() as u32);
        for &s in &self.k_scale {
            w.put_f32(s);
        }
        w.put_u32(self.v_scale.len() as u32);
        for &s in &self.v_scale {
            w.put_f32(s);
        }
        w.put_blob(&self.kv_bytes);
        w.into_bytes()
    }

    /// Decode from the `Chunk` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<ChunkPayload> {
        let mut r = ByteReader::new(payload);
        let offset = r.get_u16()?;
        let k_format = r.get_u8()?;
        let v_format = r.get_u8()?;
        let k_pal = r.get_blob()?.to_vec();
        let v_pal = r.get_blob()?.to_vec();
        let n_k = r.get_u32()? as usize;
        let mut k_scale = Vec::with_capacity(n_k);
        for _ in 0..n_k {
            k_scale.push(r.get_f32()?);
        }
        let n_v = r.get_u32()? as usize;
        let mut v_scale = Vec::with_capacity(n_v);
        for _ in 0..n_v {
            v_scale.push(r.get_f32()?);
        }
        let kv_bytes = r.get_blob()?.to_vec();
        if !r.is_done() {
            return Err(PersistenceError::Corrupt(format!(
                "ChunkPayload has {} trailing bytes",
                r.remaining()
            )));
        }
        Ok(ChunkPayload {
            offset,
            k_format,
            v_format,
            k_pal,
            v_pal,
            k_scale,
            v_scale,
            kv_bytes,
        })
    }
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
// ---------------------------------------------------------------------------

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

/// Initial CRC-32 accumulator.
pub fn crc32_init() -> u32 {
    0xFFFF_FFFF
}

/// Fold `data` into a running CRC-32 accumulator.
pub fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc = CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc
}

/// Finalize a CRC-32 accumulator.
pub fn crc32_finish(crc: u32) -> u32 {
    !crc
}

/// One-shot CRC-32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    crc32_finish(crc32_update(crc32_init(), data))
}

// ---------------------------------------------------------------------------
// Little-endian byte primitives — the encoding used by every payload codec.
// ---------------------------------------------------------------------------

/// Append-only little-endian byte writer.
#[derive(Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    /// A fresh, empty writer.
    pub fn new() -> ByteWriter {
        ByteWriter { buf: Vec::new() }
    }

    /// Bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether nothing has been written.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn put_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn put_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn put_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Raw bytes, no length prefix.
    pub fn put_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// A length-prefixed (`u32`) byte blob.
    pub fn put_blob(&mut self, data: &[u8]) {
        self.put_u32(data.len() as u32);
        self.buf.extend_from_slice(data);
    }

    /// A length-prefixed UTF-8 string.
    pub fn put_str(&mut self, s: &str) {
        self.put_blob(s.as_bytes());
    }

    /// Consume the writer, yielding the buffer.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Cursor-based little-endian byte reader.
pub struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    /// A reader positioned at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> ByteReader<'a> {
        ByteReader { buf, pos: 0 }
    }

    /// Bytes not yet consumed.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Whether every byte has been consumed.
    pub fn is_done(&self) -> bool {
        self.pos == self.buf.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(PersistenceError::Truncated {
                need: n,
                have: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn get_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn get_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn get_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn get_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn get_f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    /// Read `n` raw bytes, no length prefix.
    pub fn get_raw(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    /// Read a length-prefixed (`u32`) byte blob.
    pub fn get_blob(&mut self) -> Result<&'a [u8]> {
        let n = self.get_u32()? as usize;
        self.take(n)
    }

    /// Read a length-prefixed UTF-8 string.
    pub fn get_str(&mut self) -> Result<String> {
        let bytes = self.get_blob()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| PersistenceError::Corrupt(format!("invalid utf-8: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // The canonical CRC-32 check value for the ASCII string "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn padded_len_rounds_to_4k() {
        assert_eq!(padded_record_len(0), 4096);
        assert_eq!(padded_record_len(1), 4096);
        assert_eq!(padded_record_len(4096 - 64), 4096);
        assert_eq!(padded_record_len(4096 - 64 + 1), 8192);
        assert_eq!(padded_record_len(100_000), 102_400);
    }

    fn sample_header(payload_len: u64) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Chunk,
            format: 7,
            payload_len,
            stream_id: 0xDEAD_BEEF_0000_0001,
            chunk_index: 5,
            token_count: 32,
        }
    }

    #[test]
    fn encode_layout_is_exact() {
        let payload = [0xAAu8, 0xBB, 0xCC, 0xDD];
        let header = sample_header(payload.len() as u64);
        let bytes = encode_record(&header, &payload);

        assert_eq!(bytes.len(), 4096);
        // magic "SBL1"
        assert_eq!(&bytes[0..4], &[0x53, 0x42, 0x4c, 0x31]);
        // header version 1
        assert_eq!(&bytes[4..6], &1u16.to_le_bytes());
        // record type Chunk = 4, format = 7
        assert_eq!(bytes[6], 4);
        assert_eq!(bytes[7], 7);
        // payload_len
        assert_eq!(&bytes[8..16], &4u64.to_le_bytes());
        // stream_id
        assert_eq!(&bytes[16..24], &0xDEAD_BEEF_0000_0001u64.to_le_bytes());
        // chunk_index, token_count
        assert_eq!(&bytes[24..32], &5u64.to_le_bytes());
        assert_eq!(&bytes[32..40], &32u64.to_le_bytes());
        // reserved bytes [40, 60) are zero
        assert!(bytes[40..60].iter().all(|&b| b == 0));
        // payload sits immediately after the 64-byte header
        assert_eq!(&bytes[64..68], &payload);
        // padding tail is zero
        assert!(bytes[68..].iter().all(|&b| b == 0));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let payload: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
        let header = sample_header(payload.len() as u64);
        let bytes = encode_record(&header, &payload);
        assert_eq!(bytes.len(), padded_record_len(payload.len() as u64));

        let (record, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(record.header, header);
        assert_eq!(record.payload, payload);
    }

    #[test]
    fn empty_payload_roundtrips() {
        let header = RecordHeader {
            record_type: RecordType::Commit,
            format: 0,
            payload_len: 0,
            stream_id: 42,
            chunk_index: 0,
            token_count: 0,
        };
        let bytes = encode_record(&header, &[]);
        let (record, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, 4096);
        assert_eq!(record.header, header);
        assert!(record.payload.is_empty());
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = encode_record(&sample_header(0), &[]);
        bytes[0] ^= 0xFF;
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::BadMagic { .. })
        ));
    }

    #[test]
    fn flipped_payload_byte_caught_by_checksum() {
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut bytes = encode_record(&sample_header(payload.len() as u64), &payload);
        bytes[70] ^= 0x01;
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::BadChecksum { .. })
        ));
    }

    #[test]
    fn flipped_header_byte_caught_by_checksum() {
        let mut bytes = encode_record(&sample_header(0), &[]);
        bytes[24] ^= 0x01; // mutate chunk_index inside the checksummed range
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::BadChecksum { .. })
        ));
    }

    #[test]
    fn truncated_record_rejected() {
        let bytes = encode_record(&sample_header(0), &[]);
        assert!(matches!(
            decode_record(&bytes[..100]),
            Err(PersistenceError::Truncated { .. })
        ));
        assert!(matches!(
            decode_header(&bytes[..10]),
            Err(PersistenceError::Truncated { .. })
        ));
    }

    #[test]
    fn unknown_record_type_rejected() {
        let mut bytes = encode_record(&sample_header(0), &[]);
        bytes[6] = 99;
        // Re-checksum so the type error is what surfaces, not the checksum.
        let crc = crc32(&bytes[0..60]);
        bytes[60..64].copy_from_slice(&crc.to_le_bytes());
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::UnknownRecordType(99))
        ));
    }

    #[test]
    fn byte_primitives_roundtrip() {
        let mut w = ByteWriter::new();
        w.put_u8(0xAB);
        w.put_u16(0x1234);
        w.put_u32(0xDEAD_BEEF);
        w.put_u64(0x0123_4567_89AB_CDEF);
        w.put_i32(-12345);
        w.put_f32(3.5);
        w.put_blob(&[9, 8, 7]);
        w.put_str("héllo");
        let bytes = w.into_bytes();

        let mut r = ByteReader::new(&bytes);
        assert_eq!(r.get_u8().unwrap(), 0xAB);
        assert_eq!(r.get_u16().unwrap(), 0x1234);
        assert_eq!(r.get_u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(r.get_u64().unwrap(), 0x0123_4567_89AB_CDEF);
        assert_eq!(r.get_i32().unwrap(), -12345);
        assert_eq!(r.get_f32().unwrap(), 3.5);
        assert_eq!(r.get_blob().unwrap(), &[9, 8, 7]);
        assert_eq!(r.get_str().unwrap(), "héllo");
        assert!(r.is_done());
    }

    #[test]
    fn byte_reader_truncation_is_an_error() {
        let mut r = ByteReader::new(&[1, 2, 3]);
        assert!(r.get_u64().is_err());
    }

    #[test]
    fn chunk_payload_roundtrip() {
        let payload = ChunkPayload {
            offset: 4,
            k_format: 2,
            v_format: 7,
            k_pal: vec![1, 2, 3, 4, 5],
            v_pal: vec![9, 8, 7],
            k_scale: vec![0.5, 1.0, 2.0],
            v_scale: vec![3.25],
            kv_bytes: (0..1000u32).map(|i| (i % 256) as u8).collect(),
        };
        let bytes = payload.encode();
        assert_eq!(ChunkPayload::decode(&bytes).unwrap(), payload);
    }

    #[test]
    fn chunk_payload_empty_metadata_roundtrip() {
        // A float partial chunk: identity palette, no outer scales.
        let payload = ChunkPayload {
            offset: 0,
            k_format: 0,
            v_format: 0,
            k_pal: Vec::new(),
            v_pal: Vec::new(),
            k_scale: Vec::new(),
            v_scale: Vec::new(),
            kv_bytes: vec![0xAB; 64],
        };
        assert_eq!(ChunkPayload::decode(&payload.encode()).unwrap(), payload);
    }

    #[test]
    fn chunk_payload_rejects_trailing_bytes() {
        let payload = ChunkPayload {
            offset: 1,
            k_format: 1,
            v_format: 1,
            k_pal: vec![1],
            v_pal: vec![2],
            k_scale: vec![1.0],
            v_scale: vec![1.0],
            kv_bytes: vec![3],
        };
        let mut bytes = payload.encode();
        bytes.push(0);
        assert!(matches!(
            ChunkPayload::decode(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }
}

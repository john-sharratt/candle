//! Redo-log record codec — the on-disk wire format.
//!
//! Each record is framed as:
//!
//! ```text
//! <ndjson-header>\n<payload-bytes>[zero-padding to 4 KB]
//! ```
//!
//! The header is a single line of UTF-8 JSON terminated by a newline.
//! The payload is `payload_len` raw bytes, contents type-dependent —
//! either binary (chunks, tokens, model spec, tokenizer hash) or
//! JSON (label, conv state, stream decl, manifest checkpoint).
//!
//! Records are zero-padded out to a 4 KB sector boundary so the
//! walker can do unbuffered, sector-aligned reads and the zero tail
//! of a pre-grown file marks the clean log end.
//!
//! Forward compatibility:
//! - **Adding a record type** — old readers see [`RecordType::Unknown`]
//!   and skip the record (the walker advances past it without
//!   visiting it; no ingest error).
//! - **Adding a header field** — old readers ignore unknown JSON
//!   keys (serde's default behaviour) and use defaults for keys
//!   the new writer omitted.
//! - **Adding a payload field (when the payload is JSON)** — same
//!   shape: old readers ignore unknown keys, new readers use
//!   `#[serde(default)]` on every field so an older writer's
//!   missing field decodes to its default.
//!
//! Torn-write detection:
//! - Header parse failure (non-JSON, missing newline, header line
//!   exceeds [`ALIGN`]) → torn.
//! - Payload CRC mismatch → torn. The CRC is over the payload
//!   bytes only; the header carries the expected value.
//! - File ends before the record's padded size → torn.
//!
//! Module layout:
//! - [`RecordType`] / [`RecordHeader`] — the typed framing.
//! - [`encode_record`] / [`decode_record`] / [`decode_header`] —
//!   the codec.
//! - [`ChunkPayload`] — the one structured **binary** payload that
//!   carries KV chunk metadata + arena bytes (kept binary because
//!   JSON would balloon size and parse cost for a hot path).
//! - [`ByteWriter`] / [`ByteReader`] — little-endian primitives,
//!   used by `ChunkPayload` and anyone else still encoding binary
//!   structured payloads.
//! - `crc32*` — the CRC primitives, also used by [`super::log_file`]
//!   for the superblock checksum.

use serde::{Deserialize, Serialize};

use super::{PersistenceError, Result};

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

/// The redo-log record types.
///
/// `Unknown` is the sentinel for tags this version doesn't recognise —
/// produced by `#[serde(other)]` on the deserialize side. The walker
/// skips records that decode to `Unknown` instead of erroring, which is
/// the forward-compatibility property the on-disk format buys us.
///
/// `Unknown` must never be written: [`encode_record`] panics if it sees
/// one (it's a programming error to construct a header with `Unknown`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordType {
    ModelSpec = 1,
    Template = 2,
    StreamDecl = 3,
    Chunk = 4,
    Tokens = 5,
    Signatures = 6,
    Commit = 7,
    Checkpoint = 8,
    /// The model's `tokenizer.json` digest — a workspace singleton.
    Tokenizer = 9,
    /// Per-timeline conversation display label + conv_id (sidebar
    /// title). Last-writer-wins on replay.
    Label = 10,
    /// Per-timeline lifecycle state (archived flag, future flags…).
    /// Last-writer-wins on replay.
    ConvState = 11,
    /// Summary-tree node metadata for one turn — kind / children /
    /// height / dirty.  JSON payload — see
    /// [`TreeMetadataPayload`].  Reload constructs the per-timeline
    /// summary tree by replaying these records in order (the latest
    /// record per `(timeline, turn_index)` wins).  Last-writer-wins.
    TreeMetadata = 12,
    /// Per-timeline substrate-side resume key (`debug_id`).  JSON
    /// payload `{ debug_id: String }`.  Last-writer-wins.  Allows
    /// `find_or_create(debug_id)` to recover a workspace timeline
    /// after restart.
    DebugId = 13,
    /// Per-timeline tombstone — marks every turn, section, and
    /// stream record bound to the named timeline as logically
    /// deleted.  JSON payload [`TombstonePayload`].  Replay applies
    /// the tombstone so the old turns vanish from the substrate's
    /// view; the compactor physically drops the matching records
    /// on the next compaction pass.
    Tombstone = 14,
    /// Per-turn projection-event timeline — the materialized-context
    /// composition + decode throughput the GUI draws as timeline dots.
    /// JSON payload: a `Vec<ProjectionEvent>` for one turn, keyed by the
    /// turn's `stream_id`. Last-writer-wins on replay.
    ProjectionEvents = 15,
    /// Cached summary of a section collection (today: the tool catalog),
    /// keyed by a hash of the ordered collection content. A workspace
    /// singleton — last-writer-wins on replay, and the compactor keeps only
    /// the latest, dropping every superseded copy. JSON payload
    /// [`ToolSummaryPayload`]. Regenerated only when the catalog hash changes.
    ToolSummary = 16,
    /// Catch-all for record-type tags this version doesn't recognise.
    /// Records that deserialize as `Unknown` are skipped by the walker.
    #[serde(other)]
    Unknown,
}

/// The decoded record header — the wire fields we carry per record.
/// Defaults make every field (except `type` / `payload_len` / `crc`)
/// optional in the JSON, so older writers that omit a field decode
/// cleanly and newer writers can introduce optional fields without
/// breaking older readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordHeader {
    #[serde(rename = "type")]
    pub record_type: RecordType,
    /// Unpadded payload byte length.
    pub payload_len: u64,
    /// CRC-32 over the payload bytes.
    pub crc: u32,
    /// `Chunk` only — the `KvFormat` tag; 0 for every other record type.
    #[serde(default)]
    pub format: u8,
    /// Owning stream, or 0 when not stream-scoped.
    #[serde(default)]
    pub stream_id: u64,
    /// Position in the stream's local chunk grid (`Chunk`), or
    /// `through_index` (`Commit`); 0 otherwise.
    #[serde(default)]
    pub chunk_index: u64,
    /// Token count of a `Chunk` (`32` sealed, `<32` partial tail).
    #[serde(default)]
    pub token_count: u64,
}

/// A fully decoded record — header and owned payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub header: RecordHeader,
    pub payload: Vec<u8>,
}

/// Total on-disk size of a record whose JSON header line is
/// `header_line_len` bytes long (without the newline) and whose payload
/// is `payload_len` bytes, padded up to the [`ALIGN`] boundary.
pub fn padded_record_len(header_line_len: usize, payload_len: u64) -> usize {
    // +1 for the newline that ends the header.
    let raw = header_line_len + 1 + payload_len as usize;
    raw.div_ceil(ALIGN) * ALIGN
}

/// Encode a record into its padded on-disk byte image.
///
/// Panics if `header.payload_len` disagrees with `payload.len()`, if
/// the resulting JSON header would exceed [`MAX_HEADER_LINE`] bytes,
/// or if `header.record_type` is [`RecordType::Unknown`].
pub fn encode_record(header: &RecordHeader, payload: &[u8]) -> Vec<u8> {
    assert_eq!(
        header.payload_len as usize,
        payload.len(),
        "RecordHeader.payload_len must match the payload slice"
    );
    assert!(
        header.record_type != RecordType::Unknown,
        "encode_record refuses to write RecordType::Unknown — \
         that variant is the reader's catch-all for tags it does \
         not recognise, not a writable type"
    );
    let mut effective = *header;
    effective.crc = crc32(payload);

    let header_line = serde_json::to_string(&effective)
        .expect("RecordHeader serialization is infallible for the supported field set");
    assert!(
        header_line.len() <= MAX_HEADER_LINE,
        "JSON record header is {} bytes, exceeds the {}-byte cap",
        header_line.len(),
        MAX_HEADER_LINE,
    );

    let total = padded_record_len(header_line.len(), effective.payload_len);
    let mut out = vec![0u8; total];
    out[..header_line.len()].copy_from_slice(header_line.as_bytes());
    out[header_line.len()] = b'\n';
    let payload_start = header_line.len() + 1;
    out[payload_start..payload_start + payload.len()].copy_from_slice(payload);
    // The remaining bytes stay zero — the sector padding tail.
    out
}

/// Decode the header at the start of `probe` without touching the
/// payload. Returns the decoded header and the number of bytes the
/// header occupies on disk **including** the trailing newline.
///
/// `probe` must contain the first [`HEADER_PROBE_SIZE`] bytes of the
/// record (or fewer if EOF arrives first — in which case the parse
/// fails clean).
pub fn decode_header(probe: &[u8]) -> Result<(RecordHeader, usize)> {
    if probe.is_empty() {
        return Err(PersistenceError::Truncated { need: 1, have: 0 });
    }
    if probe[0] != b'{' {
        return Err(PersistenceError::Corrupt(format!(
            "expected JSON header starting with '{{', found 0x{:02x}",
            probe[0]
        )));
    }
    // Cap the scan window: a JSON header is required to fit in one
    // sector. Bounding the search keeps a malformed record from
    // scanning arbitrarily far for a newline.
    let scan_len = probe.len().min(ALIGN);
    let newline_pos = probe[..scan_len]
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| {
            PersistenceError::Corrupt(
                "no newline found within the header probe — header missing or too large"
                    .to_string(),
            )
        })?;
    let header_line = std::str::from_utf8(&probe[..newline_pos])
        .map_err(|e| PersistenceError::Corrupt(format!("record header is not valid UTF-8: {e}")))?;
    let header: RecordHeader = serde_json::from_str(header_line)
        .map_err(|e| PersistenceError::Corrupt(format!("record header JSON parse: {e}")))?;
    Ok((header, newline_pos + 1))
}

/// Decode one full record from the start of `buf` — header + payload
/// slice + padded total. Returns `(header, payload, total_bytes)`
/// where `payload` is a **borrowed** view into `buf` (zero-copy) and
/// `total_bytes` is the record's padded on-disk size so a walker can
/// advance.
///
/// **Does not CRC-verify the payload.** Verification is a separate
/// concern handled by [`verify_record_crc`] — the cold-load hot path
/// skips it entirely (the background validator catches latent bit rot
/// out-of-band), and the walker only verifies on demand for torn-write
/// detection at recovery time.
///
/// Cold-load uses the payload slice in place against the pinned
/// scratch buffer; callers that need owned bytes — typically
/// random-access reads like [`super::log_file::read_record_at`] —
/// `.to_vec()` at the boundary where ownership starts.
pub fn decode_record(buf: &[u8]) -> Result<(RecordHeader, &[u8], usize)> {
    let (header, header_bytes) = decode_header(buf)?;
    let total = padded_record_len(header_bytes - 1, header.payload_len);
    if buf.len() < total {
        return Err(PersistenceError::Truncated {
            need: total,
            have: buf.len(),
        });
    }
    let payload_end = header_bytes + header.payload_len as usize;
    let payload = &buf[header_bytes..payload_end];
    Ok((header, payload, total))
}

/// CRC-verify the payload of a record against the header's stored CRC.
///
/// Used by the **background validator thread** to catch latent bit rot
/// after startup, and by the walker only when explicitly asked (e.g.
/// torn-write probes during recovery). `RecordType::Unknown` records
/// are skipped — the payload format may not be readable in our world
/// view and the walker has already decided to advance past them.
pub fn verify_record_crc(header: &RecordHeader, payload: &[u8]) -> Result<()> {
    if header.record_type == RecordType::Unknown {
        return Ok(());
    }
    let computed = crc32(payload);
    if computed != header.crc {
        return Err(PersistenceError::BadChecksum {
            header: header.crc,
            computed,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `ChunkPayload` — the one structured binary payload. Kept binary
// because chunks are the hot path; JSON-encoding KV bytes would
// inflate size and parse time for no benefit.
// ---------------------------------------------------------------------------

/// The payload of a `Chunk` record: per-`(head, palette)` quantization
/// metadata followed by the gathered arena bytes.
///
/// The metadata prefix is **mandatory** — the arena blob alone is not
/// self-describing, and the quantized KV cannot be dequantised without
/// `k_pal` / `v_pal` / `k_scale` / `v_scale`.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPayload {
    /// The `SealedChunk` window skip-count (start of valid data).
    pub offset: u16,
    /// `KvFormat` tag of every K palette sub-band, in `[h*N_PALETTE + p]`
    /// order (`n_kv_head × N_PALETTE` entries).
    pub k_formats: Vec<u8>,
    /// `KvFormat` tag of every V palette sub-band, same layout.
    pub v_formats: Vec<u8>,
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

/// Metadata-only view of a `ChunkPayload` — everything except `kv_bytes`,
/// plus the length of `kv_bytes` that was peeked from the on-disk record.
/// Produced by [`ChunkPayload::decode_with_kv_range`] for the cold-load
/// fast path: the `kv_bytes` slice itself lives in a caller-owned buffer
/// (typically pinned host memory), so we never allocate a fresh `Vec` for
/// it during cold load.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkPayloadMeta {
    pub offset: u16,
    pub k_formats: Vec<u8>,
    pub v_formats: Vec<u8>,
    pub k_pal: Vec<u8>,
    pub v_pal: Vec<u8>,
    pub k_scale: Vec<f32>,
    pub v_scale: Vec<f32>,
    /// Length of the `kv_bytes` blob in the source record — convenience
    /// for callers that already have the slice range from
    /// `decode_with_kv_range`.
    pub kv_bytes_len: usize,
}

impl ChunkPayload {
    /// Encode to the `Chunk` record payload bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.put_u16(self.offset);
        w.put_blob(&self.k_formats);
        w.put_blob(&self.v_formats);
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

    /// Cold-load fast path: decode every field **except** `kv_bytes`,
    /// returning the metadata alongside the byte range where `kv_bytes`
    /// data lives in `payload` (the slice *after* its u32 length
    /// prefix). The caller can then leave `kv_bytes` in the read
    /// buffer (typically pinned host memory) and source the HtoD
    /// directly from there — no per-chunk Vec allocation, no extra
    /// host-to-host memcpy. Used by
    /// [`super::SubstratePersistence::read_stream_records_into_pinned`].
    pub fn decode_with_kv_range(
        payload: &[u8],
    ) -> Result<(ChunkPayloadMeta, std::ops::Range<usize>)> {
        let mut r = ByteReader::new(payload);
        let offset = r.get_u16()?;
        let k_formats = r.get_blob()?.to_vec();
        let v_formats = r.get_blob()?.to_vec();
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
        let kv_len = r.get_u32()? as usize;
        let kv_start = r.position();
        if kv_start + kv_len > payload.len() {
            return Err(PersistenceError::Truncated {
                need: kv_start + kv_len,
                have: payload.len(),
            });
        }
        // Note: we do NOT advance the reader past kv_bytes here — we are
        // intentionally not consuming those bytes. The caller has the
        // range and reads from the (pinned) buffer directly.
        Ok((
            ChunkPayloadMeta {
                offset,
                k_formats,
                v_formats,
                k_pal,
                v_pal,
                k_scale,
                v_scale,
                kv_bytes_len: kv_len,
            },
            kv_start..(kv_start + kv_len),
        ))
    }

    /// Decode from the `Chunk` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<ChunkPayload> {
        let mut r = ByteReader::new(payload);
        let offset = r.get_u16()?;
        let k_formats = r.get_blob()?.to_vec();
        let v_formats = r.get_blob()?.to_vec();
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
            k_formats,
            v_formats,
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
// Little-endian byte primitives — the encoding used by `ChunkPayload`
// and any other still-binary structured payload (kept public for
// re-use; metadata payloads are now JSON via serde).
// ---------------------------------------------------------------------------

/// Append-only little-endian byte writer.
#[derive(Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn new() -> ByteWriter {
        ByteWriter { buf: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

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

    pub fn put_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    pub fn put_blob(&mut self, data: &[u8]) {
        self.put_u32(data.len() as u32);
        self.buf.extend_from_slice(data);
    }

    pub fn put_str(&mut self, s: &str) {
        self.put_blob(s.as_bytes());
    }

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
    pub fn new(buf: &'a [u8]) -> ByteReader<'a> {
        ByteReader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn is_done(&self) -> bool {
        self.pos == self.buf.len()
    }

    /// Current cursor position within the underlying buffer. Used by the
    /// cold-load batched-read path to capture the `kv_bytes` slice range
    /// without copying it out.
    pub fn position(&self) -> usize {
        self.pos
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

    pub fn get_raw(&mut self, n: usize) -> Result<&'a [u8]> {
        self.take(n)
    }

    pub fn get_blob(&mut self) -> Result<&'a [u8]> {
        let n = self.get_u32()? as usize;
        self.take(n)
    }

    pub fn get_str(&mut self) -> Result<String> {
        let bytes = self.get_blob()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|e| PersistenceError::Corrupt(format!("invalid utf-8: {e}")))
    }
}

// ── TreeMetadata + DebugId structured payloads ──────────────────────────────

/// JSON payload for a [`RecordType::TreeMetadata`] record — one entry
/// per `(timeline, turn_index)`.  Last-writer-wins on reload: a later
/// record for the same key supersedes the earlier one.
///
/// Stored as JSON because it's small (a handful of integer fields plus a
/// children list bounded by `MERGE_FANOUT`) and because the redo-log already
/// has a JSON-header culture; binary would buy nothing here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeMetadataPayload {
    /// `TimelineId.raw()` — opaque u64 key.
    pub timeline_id: u64,
    /// `TurnIndex.0` — the turn this metadata belongs to.
    pub turn_index: u32,
    /// Discriminant matching `summary_tree::TurnKind`:
    ///   0 = Normal, 1 = SummaryOfTurns, 2 = SummaryOfSummaries.
    pub kind: u8,
    /// Forest level: `SummaryOfTurns` = 1, a `SummaryOfSummaries` over level-`h`
    /// children = `h + 1`; 0 for Normal sub-leaves.
    pub tree_height: u8,
    /// For `SummaryOfTurns`: Normal-child indices in chronological order.  For
    /// `SummaryOfSummaries`: exactly `MERGE_FANOUT` same-level child indices.
    /// For `Normal`: empty.
    pub children: Vec<u32>,
}

impl TreeMetadataPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TreeMetadataPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("TreeMetadata JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::DebugId`] record.  One entry per
/// `(timeline)`, last-writer-wins on replay.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebugIdPayload {
    pub timeline_id: u64,
    pub debug_id: String,
}

impl DebugIdPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("DebugIdPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("DebugId JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::Tombstone`] record.  Naming
/// `timeline_id` marks it as logically deleted: every
/// `StreamDecl::Turn`, `Chunk`, `Tokens`, `Signatures`, `Commit`,
/// `TreeMetadata`, `Label`, `ConvState`, and `DebugId` record bound
/// to that timeline becomes inert on replay, and the compactor
/// drops them from disk on the next compaction pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstonePayload {
    pub timeline_id: u64,
}

impl TombstonePayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TombstonePayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("Tombstone JSON parse: {e}")))
    }
}

/// One cached tool-catalog summary: `catalog_hash` is a 128-bit hash of the
/// ordered collection content the summary was generated from, and `summary` is
/// the generated text. The caller hashes the freshly-injected catalog and
/// compares to `catalog_hash`; an equal hash is a cache hit (no regeneration).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSummaryEntry {
    pub catalog_hash: u128,
    pub summary: String,
}

/// JSON payload for a [`RecordType::ToolSummary`] record. A workspace singleton
/// (last-writer-wins) holding both tool-mode summaries in one record so the
/// existing single-slot manifest/compaction logic is unchanged: `comprehensive`
/// is the overview of the full catalog, `restricted` the overview of the safe
/// (non-high-risk) subset. Either may be `None` when its generation was absent
/// or failed. A mismatch on either entry's `catalog_hash` triggers a regenerate
/// + rewrite of the whole record; the compactor reclaims the superseded copy.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSummaryPayload {
    pub comprehensive: Option<ToolSummaryEntry>,
    pub restricted: Option<ToolSummaryEntry>,
}

impl ToolSummaryPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ToolSummaryPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("ToolSummary JSON parse: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn tree_metadata_payload_round_trips() {
        let p = TreeMetadataPayload {
            timeline_id: 42,
            turn_index: 17,
            kind: 1, // SummaryOfTurns
            tree_height: 1,
            children: vec![10],
        };
        let bytes = p.encode();
        let back = TreeMetadataPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);

        // Through a real record frame.
        let header = RecordHeader {
            record_type: RecordType::TreeMetadata,
            format: 0,
            payload_len: bytes.len() as u64,
            crc: crc32(&bytes),
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
        };
        let frame = encode_record(&header, &bytes);
        let (hdr2, payload2, _total) = decode_record(&frame).unwrap();
        assert_eq!(hdr2.record_type, RecordType::TreeMetadata);
        let back2 = TreeMetadataPayload::decode(payload2).unwrap();
        assert_eq!(back2, p);
    }

    #[test]
    fn tree_metadata_sos_round_trips() {
        let p = TreeMetadataPayload {
            timeline_id: 7,
            turn_index: 99,
            kind: 2, // SummaryOfSummaries
            tree_height: 4,
            children: vec![50, 75, 88],
        };
        let bytes = p.encode();
        let back = TreeMetadataPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn tool_summary_payload_round_trips_and_bytes() {
        let p = ToolSummaryPayload {
            comprehensive: Some(ToolSummaryEntry {
                catalog_hash: 1u128 << 64, // 18446744073709551616 — exercises the high word
                summary: "## A\n  x, y".to_string(),
            }),
            restricted: None,
        };
        // Exact wire bytes: serde_json emits the struct fields in order, the
        // u128 as a decimal number and the string with its newline escaped.
        let bytes = p.encode();
        let expected = "{\"comprehensive\":{\"catalog_hash\":18446744073709551616,\"summary\":\"## A\\n  x, y\"},\"restricted\":null}";
        assert_eq!(bytes, expected.as_bytes());
        let back = ToolSummaryPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);

        // Through a real record frame.
        let header = RecordHeader {
            record_type: RecordType::ToolSummary,
            format: 0,
            payload_len: bytes.len() as u64,
            crc: crc32(&bytes),
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
        };
        let frame = encode_record(&header, &bytes);
        let (hdr2, payload2, _total) = decode_record(&frame).unwrap();
        assert_eq!(hdr2.record_type, RecordType::ToolSummary);
        assert_eq!(ToolSummaryPayload::decode(payload2).unwrap(), p);
    }

    #[test]
    fn debug_id_payload_round_trips() {
        let p = DebugIdPayload {
            timeline_id: 123,
            debug_id: "coherent-50".to_string(),
        };
        let bytes = p.encode();
        let back = DebugIdPayload::decode(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn record_header_serialises_tree_metadata_variant() {
        // The new variants must serialise as JSON strings the decoder
        // can round-trip — protects against typo'd `rename_all` collisions.
        let h = RecordHeader {
            record_type: RecordType::TreeMetadata,
            format: 0,
            payload_len: 0,
            crc: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
        };
        let line = serde_json::to_string(&h).unwrap();
        let back: RecordHeader = serde_json::from_str(&line).unwrap();
        assert_eq!(back.record_type, RecordType::TreeMetadata);

        let h2 = RecordHeader {
            record_type: RecordType::DebugId,
            ..h
        };
        let line2 = serde_json::to_string(&h2).unwrap();
        let back2: RecordHeader = serde_json::from_str(&line2).unwrap();
        assert_eq!(back2.record_type, RecordType::DebugId);
    }

    #[test]
    fn padded_len_rounds_to_4k() {
        // Header line of 100 bytes + 1 newline + 0 payload = 101 → 4096.
        assert_eq!(padded_record_len(100, 0), 4096);
        // Header of 100 bytes + 1 newline + 3995 payload = 4096 → 4096.
        assert_eq!(padded_record_len(100, 3995), 4096);
        // One more byte pushes us to the next sector.
        assert_eq!(padded_record_len(100, 3996), 8192);
        // Large payload spans many sectors.
        assert_eq!(padded_record_len(100, 100_000), 102_400);
    }

    fn sample_header(payload_len: u64, payload: &[u8]) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Chunk,
            format: 7,
            payload_len,
            crc: crc32(payload),
            stream_id: 0xDEAD_BEEF_0000_0001,
            chunk_index: 5,
            token_count: 32,
        }
    }

    #[test]
    fn encode_layout_starts_with_json_header_and_newline() {
        let payload = [0xAAu8, 0xBB, 0xCC, 0xDD];
        let header = sample_header(payload.len() as u64, &payload);
        let bytes = encode_record(&header, &payload);

        // Padded out to one full sector.
        assert_eq!(bytes.len(), ALIGN);

        // The header is JSON terminated by a single newline.
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        let header_str = std::str::from_utf8(&bytes[..newline_pos]).unwrap();
        let parsed: RecordHeader = serde_json::from_str(header_str).unwrap();
        assert_eq!(parsed.record_type, RecordType::Chunk);
        assert_eq!(parsed.format, 7);
        assert_eq!(parsed.payload_len, 4);
        assert_eq!(parsed.stream_id, 0xDEAD_BEEF_0000_0001);
        assert_eq!(parsed.chunk_index, 5);
        assert_eq!(parsed.token_count, 32);
        assert_eq!(parsed.crc, crc32(&payload));

        // Payload sits immediately after the newline.
        let payload_start = newline_pos + 1;
        assert_eq!(
            &bytes[payload_start..payload_start + payload.len()],
            &payload
        );

        // The remainder of the sector is zero padding.
        assert!(bytes[payload_start + payload.len()..]
            .iter()
            .all(|&b| b == 0));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let payload: Vec<u8> = (0..5000u32).map(|i| (i % 256) as u8).collect();
        let header = sample_header(payload.len() as u64, &payload);
        let bytes = encode_record(&header, &payload);

        let (got_header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(got_header, header);
        assert_eq!(got_payload, payload.as_slice());
    }

    #[test]
    fn empty_payload_roundtrips() {
        let header = RecordHeader {
            record_type: RecordType::Commit,
            payload_len: 0,
            crc: crc32(&[]),
            format: 0,
            stream_id: 42,
            chunk_index: 0,
            token_count: 0,
        };
        let bytes = encode_record(&header, &[]);
        let (got_header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, ALIGN);
        assert_eq!(got_header, header);
        assert!(got_payload.is_empty());
    }

    #[test]
    fn unknown_record_type_decodes_to_unknown_variant_without_error() {
        // Hand-roll a record whose `type` is a tag this version
        // doesn't know about ("future_kind"). The reader must
        // surface it as `RecordType::Unknown` so the walker can
        // skip past it.
        let payload = b"future-payload";
        let crc = crc32(payload);
        let header_line = format!(
            "{{\"type\":\"future_kind\",\"payload_len\":{},\"crc\":{}}}",
            payload.len(),
            crc
        );
        let header_len = header_line.len();
        let total = padded_record_len(header_len, payload.len() as u64);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';
        bytes[header_len + 1..header_len + 1 + payload.len()].copy_from_slice(payload);

        let (header, got_payload, consumed) = decode_record(&bytes).unwrap();
        assert_eq!(consumed, total);
        assert_eq!(header.record_type, RecordType::Unknown);
        // The payload survived even though we don't know how to
        // interpret it — useful for diagnostics and round-trip in
        // tests.
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn extra_header_field_is_ignored() {
        // A newer writer adds a key we don't model — we must
        // silently ignore it and parse the rest normally. This is
        // the field-level forward-compat property.
        let payload = b"hi";
        let crc = crc32(payload);
        let header_line = format!(
            "{{\"type\":\"commit\",\"payload_len\":{},\"crc\":{},\"future_field\":\"opaque\"}}",
            payload.len(),
            crc
        );
        let header_len = header_line.len();
        let total = padded_record_len(header_len, payload.len() as u64);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';
        bytes[header_len + 1..header_len + 1 + payload.len()].copy_from_slice(payload);

        let (header, got_payload, _) = decode_record(&bytes).unwrap();
        assert_eq!(header.record_type, RecordType::Commit);
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn missing_optional_field_defaults_to_zero() {
        // Some older writer omitted `format` / `stream_id` /
        // `chunk_index` / `token_count` — those should default to
        // zero rather than fail.
        let payload = b"";
        let header_line = "{\"type\":\"commit\",\"payload_len\":0,\"crc\":0}";
        let header_len = header_line.len();
        let total = padded_record_len(header_len, 0);
        let mut bytes = vec![0u8; total];
        bytes[..header_len].copy_from_slice(header_line.as_bytes());
        bytes[header_len] = b'\n';

        let (header, got_payload, _) = decode_record(&bytes).unwrap();
        assert_eq!(header.record_type, RecordType::Commit);
        assert_eq!(header.format, 0);
        assert_eq!(header.stream_id, 0);
        assert_eq!(header.chunk_index, 0);
        assert_eq!(header.token_count, 0);
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn bad_header_first_byte_rejected() {
        // The first byte isn't `{` — corruption (or a torn record).
        let mut bytes = encode_record(&sample_header(0, &[]), &[]);
        bytes[0] = b'X';
        assert!(matches!(
            decode_record(&bytes),
            Err(PersistenceError::Corrupt(_))
        ));
    }

    #[test]
    fn flipped_payload_byte_caught_by_verify_record_crc() {
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut bytes = encode_record(&sample_header(payload.len() as u64, &payload), &payload);
        // Flip a byte inside the payload — find it just after the
        // header newline.
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        bytes[newline_pos + 2] ^= 0x01;
        // decode_record itself no longer CRC-verifies; the parse succeeds.
        let (header, payload_slice, _) = decode_record(&bytes).unwrap();
        // The dedicated verifier surfaces the bit rot.
        assert!(matches!(
            verify_record_crc(&header, payload_slice),
            Err(PersistenceError::BadChecksum { .. })
        ));
    }

    #[test]
    fn flipped_header_byte_caught_by_json_or_crc() {
        // Mutating a byte inside the header line either breaks JSON
        // parsing or — if the JSON still parses — leaves a stale
        // `crc` value in the header that no longer matches the
        // payload. The fast path catches the first case; the
        // background validator catches the second.
        let payload = [1u8, 2, 3, 4];
        let mut bytes = encode_record(&sample_header(payload.len() as u64, &payload), &payload);
        let newline_pos = bytes.iter().position(|&b| b == b'\n').unwrap();
        let target = (0..newline_pos)
            .find(|&i| bytes[i].is_ascii_digit())
            .expect("the encoded header contains at least one digit");
        bytes[target] = if bytes[target] == b'0' { b'9' } else { b'0' };

        let caught = match decode_record(&bytes) {
            Err(_) => true,
            Ok((header, payload_slice, _)) => verify_record_crc(&header, payload_slice).is_err(),
        };
        assert!(
            caught,
            "either parse failure or CRC mismatch must surface the header bit-flip"
        );
    }

    #[test]
    fn truncated_record_rejected() {
        let bytes = encode_record(&sample_header(0, &[]), &[]);
        // Decode against a buffer that's far too short.
        assert!(decode_record(&bytes[..10]).is_err());
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
            k_formats: vec![2, 7, 2, 7],
            v_formats: vec![7, 7, 7, 7],
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
        let payload = ChunkPayload {
            offset: 0,
            k_formats: Vec::new(),
            v_formats: Vec::new(),
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
            k_formats: vec![1, 1, 1, 1],
            v_formats: vec![1, 1, 1, 1],
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

    #[test]
    #[should_panic(expected = "RecordType::Unknown")]
    fn encode_refuses_unknown_variant() {
        let header = RecordHeader {
            record_type: RecordType::Unknown,
            payload_len: 0,
            crc: 0,
            format: 0,
            stream_id: 0,
            chunk_index: 0,
            token_count: 0,
        };
        encode_record(&header, &[]);
    }

    #[test]
    fn header_line_too_large_panics() {
        // A pathologically large "extra" payload smuggled into a
        // payload of huge size — the header itself is always tiny,
        // so this should always fit. But we can still verify the
        // bound exists: pin MAX_HEADER_LINE against ALIGN so a
        // future tweak doesn't accidentally allow multi-sector
        // headers.
        const _: () = assert!(MAX_HEADER_LINE < ALIGN);
    }
}

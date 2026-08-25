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
//! JSON (label, conv state, stream decl, tree metadata).
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

use super::content_hash::ContentHash;
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
    Commit = 7,
    // Tag 8 is retired (it was the manifest checkpoint snapshot —
    // recovery is a pure walk now, and old logs' checkpoint records
    // decode as `Unknown` and are skipped by the walker).
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
    /// Per-timeline distillation marker — the timeline's turns keep their
    /// `StreamDecl` + `WideQSig` (the belief gallery reads only the sig) but shed
    /// their content (`Tokens` + KV `Chunk`s) on the next compaction pass. Used to
    /// collapse the calibration corpus to sig-only. JSON payload
    /// [`DistillPayload`]. Idempotent — duplicate markers replay identically. The
    /// marker itself is **consumed** by that compaction (not re-emitted), so the
    /// next reload finds none and doesn't re-trigger — see `docs/tool_provenance_distillation.md`.
    Distilled = 16,
    /// A turn's wide per-token `sign(Q)` window (all heads, all layers) — the decode→decode
    /// (`Q·Q`) consensus substrate. Opaque payload encoded by `provenance::wide_sig`, keyed
    /// by the turn's stream id, last-writer-wins (each (re)projection overwrites the window).
    WideQSig = 17,
    /// A batch of header digests for the records appended since the
    /// previous `HeaderIndex`, plus a link to that previous index — the
    /// backward chain recovery follows instead of probing every record
    /// header (§5.6). Binary payload — see
    /// [`header_index`](super::header_index). Derived data: the
    /// compactor drops every copy and the writer regenerates the chain
    /// in the new file.
    HeaderIndex = 18,
    /// Couples a turn to the tool response that follows it — the two halves of
    /// one tool round-trip. JSON payload [`TurnCouplingPayload`]. Idempotent;
    /// replay collects them into a per-timeline set.
    ///
    /// Written by the caller (not the seal) in the one window where the fact is
    /// certain: after the tools have actually returned output, and *before* the
    /// response turn is submitted. So a coupling exists iff the round-trip really
    /// happened — capture mode and malformed calls emit none — and it is always
    /// durable before the turn it points at exists, which is what lets the
    /// summariser group an exchange without ever racing its own record.
    TurnCoupling = 19,
    /// A recurrent-state snapshot — the Gated DeltaNet matrices and conv tails
    /// of every recurrent layer (hybrid Qwen3.5/3.6/3.8 models;
    /// `docs/qwen35_qwen38_models.md` §5).
    ///
    /// **Two kinds, told apart by stream id, never by inspection.** A
    /// *conversation's* snapshot ([`SnapshotPayload`]) is taken at a turn seal
    /// and keyed by `snapshot_stream_id(timeline)`. A *prompt branch's*
    /// checkpoint ([`BranchCheckpointPayload`]) is computed at build time for
    /// one selector assignment and keyed by `branch_checkpoint_stream_id(prefix)`.
    /// Everything below — the accounting supersede key, the recovery walk's
    /// location map, the compactor's carry-forward — reads `header.stream_id`
    /// and never the payload, so both kinds ride the identical single-tail
    /// machinery. A reader computes the stream id before asking, so it always
    /// knows which shape it will get; the payloads carry distinct leading magic
    /// so a future mistake is an error rather than a plausible state.
    ///
    /// **Single tail per conversation**: keyed in the *header* by a synthetic
    /// per-timeline stream id, so the newest snapshot supersedes every
    /// previous one mechanically — [`super::accounting`] credits the old copy
    /// as dead bytes and `snapshot_locs` keeps exactly the tail alive through
    /// segment maintenance. No explicit tombstone is written for supersede;
    /// the last-writer-wins accounting *is* the tombstone. Binary payload
    /// [`SnapshotPayload`]; the header CRC covers it like any metadata record.
    Snapshot = 20,
    /// A **prompt branch's** recurrent checkpoint — the state after a whole
    /// system prompt for one selector assignment, keyed by the branch's content
    /// prefix ([`BranchCheckpointPayload`]).
    ///
    /// Structurally identical to [`Self::Snapshot`] — same single-tail rule,
    /// same accounting, same relocation — and a separate type for one reason:
    /// **a branch checkpoint is a cache and a conversation snapshot is not.**
    /// Losing a snapshot loses history that cannot be recomputed. Losing a
    /// checkpoint costs one prefill, because the prompt that produced it is
    /// still on disk. Compaction has to treat them differently, and it reads
    /// the header rather than the payload, so the difference has to live in the
    /// type.
    ///
    /// That difference is load-bearing: checkpoints are keyed by content, so
    /// nothing supersedes one when the prompt changes and no timeline tombstone
    /// ever names it. Without a bound they accumulate one orphan per prompt
    /// edit, ~63 MiB each, carried forward verbatim by every compaction.
    BranchCheckpoint = 21,
    /// Catch-all for record-type tags this version doesn't recognise.
    /// Records that deserialize as `Unknown` are skipped by the walker.
    #[serde(other)]
    Unknown,
}

impl RecordType {
    /// The numeric wire tag used inside `HeaderIndex` digests (the
    /// enum's explicit discriminant). The JSON header keeps its
    /// snake_case string form; this compact form exists only for the
    /// fixed-width digest entries.
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Inverse of [`RecordType::tag`]. Unrecognised tags decode to
    /// [`RecordType::Unknown`] — the same forward-compatibility rule as
    /// the JSON header's `#[serde(other)]`.
    pub fn from_tag(tag: u8) -> RecordType {
        match tag {
            1 => RecordType::ModelSpec,
            2 => RecordType::Template,
            3 => RecordType::StreamDecl,
            4 => RecordType::Chunk,
            5 => RecordType::Tokens,
            7 => RecordType::Commit,
            9 => RecordType::Tokenizer,
            10 => RecordType::Label,
            11 => RecordType::ConvState,
            12 => RecordType::TreeMetadata,
            13 => RecordType::DebugId,
            14 => RecordType::Tombstone,
            15 => RecordType::ProjectionEvents,
            16 => RecordType::Distilled,
            17 => RecordType::WideQSig,
            18 => RecordType::HeaderIndex,
            19 => RecordType::TurnCoupling,
            20 => RecordType::Snapshot,
            21 => RecordType::BranchCheckpoint,
            _ => RecordType::Unknown,
        }
    }
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
    // `Chunk` records carry the GPU-computed golden (Fletcher-32 over the arena
    // bytes, taken before the device→host copy) in `crc`; the caller has already
    // set it. Recomputing host-side here would checksum the post-copy bytes and
    // reintroduce the very blind spot the golden exists to close, so leave it be.
    // Every other record type is host-authored — crc32 over the payload.
    if header.record_type != RecordType::Chunk {
        effective.crc = crc32(payload);
    }

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
/// concern handled by [`verify_record_crc`] at the payload's
/// consumption point (cold-load decode, `read_record_at`); the
/// recovery walk skips payload bytes entirely for the bulk record
/// types and never pays for verification it can't use.
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

/// CRC-verify the payload of a metadata record against the header's stored CRC.
///
/// Called at every payload **consumption point** — the pipelined
/// cold-load decode, the batched stream read, and `read_record_at` —
/// so latent bit rot fails loudly exactly where the bytes are about to
/// be used, at the cost of one checksum pass over bytes just read from
/// disk. `RecordType::Unknown` records are skipped — the payload
/// format may not be readable in our world view and the walker has
/// already decided to advance past them.
///
/// `Chunk` records are **skipped here**. Their `header.crc` holds the GPU golden
/// (a Fletcher-32 over the arena bytes taken *before* the device→host copy, see
/// candle-kernels `simple/fletcher32.cu`), not a crc32 over the payload — so this
/// crc32 comparison never applies. Chunk integrity is checked against that golden
/// separately, with the severity each site warrants: a hard error at the write
/// boundary ([`SubstratePersistence::append_record`]) and a non-fatal warning
/// when the bytes are read into RAM/VRAM (cold-load / elevation). Keeping chunks
/// off this path also keeps the restart/recovery read — which only verifies
/// metadata — from ever putting chunk bytes on the golden path, protecting
/// restart time.
pub fn verify_record_crc(header: &RecordHeader, payload: &[u8]) -> Result<()> {
    if header.record_type == RecordType::Unknown || header.record_type == RecordType::Chunk {
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

/// Recompute a `Chunk` record's golden — Fletcher-32 over the KV-bytes slice of
/// its payload — to compare against the golden stored in `RecordHeader.crc`
/// (computed on the GPU at seal, before the DtoH copy). The metadata prefix
/// (`offset` / formats / palettes / scales) is host-authored and deliberately
/// not covered. Errors only if `payload` isn't a decodable `ChunkPayload`.
pub fn recompute_chunk_golden(payload: &[u8]) -> Result<u32> {
    let (_, kv_range) = ChunkPayload::decode_with_kv_range(payload)?;
    Ok(candle::fletcher::fletcher32(&payload[kv_range]))
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
/// `StreamDecl::Turn`, `Chunk`, `Tokens`, `Commit`,
/// `TreeMetadata`, `Label`, `ConvState`, and `DebugId` record bound
/// to that timeline becomes inert on replay, and the compactor
/// drops them from disk on the next compaction pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TombstonePayload {
    pub timeline_id: u64,
    /// When `Some`, this tombstone kills only ONE TURN — the
    /// `(timeline_id, turn_index)` pair — instead of the whole timeline. Used by
    /// the per-layer `drop_turn` corrupt-turn policy so a single unrecoverable
    /// turn (e.g. a partial write) doesn't take its whole conversation with it.
    /// `None` tombstones the entire timeline (the default, and every pre-existing
    /// tombstone). Skipped from the serialized record when `None`, so a
    /// timeline-level tombstone stays byte-identical to the pre-`turn_index` format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<u32>,
    /// Why the timeline was tombstoned, when known — e.g.
    /// `"corrupt reload (turn N): <detail>"` for a turn dropped during substrate
    /// reconstruction because its persisted state was inconsistent. Diagnostic
    /// only: the runtime treats any tombstone as dead regardless. `None` for the
    /// ordinary deletions (file removed, superseded generation, spliced fork).
    /// Skipped from the serialized record when `None`, so a reason-less tombstone
    /// stays byte-identical to the pre-reason format on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
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

#[cfg(test)]
mod tombstone_payload_tests {
    use super::TombstonePayload;

    #[test]
    fn reasonless_tombstone_is_byte_identical_to_pre_reason_format() {
        // A None reason must serialise to exactly `{"timeline_id":N}` so existing
        // on-disk tombstone records (written before the reason field) read back
        // identically and new reason-less tombstones don't change the byte layout.
        let p = TombstonePayload {
            timeline_id: 77,
            turn_index: None,
            reason: None,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":77}"#.to_vec());
    }

    #[test]
    fn reason_is_serialised_when_present() {
        let p = TombstonePayload {
            timeline_id: 42,
            turn_index: None,
            reason: Some("corrupt reload (turn 1): chunk mismatch".to_string()),
        };
        assert_eq!(
            p.encode(),
            br#"{"timeline_id":42,"reason":"corrupt reload (turn 1): chunk mismatch"}"#.to_vec()
        );
    }

    #[test]
    fn old_record_without_reason_decodes_to_none() {
        // A record persisted before the field existed still decodes cleanly.
        let decoded = TombstonePayload::decode(br#"{"timeline_id":9}"#).unwrap();
        assert_eq!(decoded.timeline_id, 9);
        assert_eq!(decoded.reason, None);
    }

    #[test]
    fn round_trips_with_reason() {
        let p = TombstonePayload {
            timeline_id: 5,
            turn_index: None,
            reason: Some("corrupt reload".to_string()),
        };
        assert_eq!(TombstonePayload::decode(&p.encode()).unwrap(), p);
    }

    #[test]
    fn turn_scoped_tombstone_bytes_are_pinned() {
        // The turn-scoped form is the reload's ONLY signal to hole-restore the
        // turn instead of renumbering the timeline — pin its exact on-disk key
        // and layout so a serde tweak can't silently demote it to a
        // timeline-level tombstone (which drops the whole conversation).
        let p = TombstonePayload {
            timeline_id: 77,
            turn_index: Some(2),
            reason: None,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":77,"turn_index":2}"#.to_vec());
        assert_eq!(TombstonePayload::decode(&p.encode()).unwrap(), p);
        // And a reasoned turn tombstone keeps field order stable.
        let q = TombstonePayload {
            timeline_id: 3,
            turn_index: Some(9),
            reason: Some("corrupt reload (turn 9): chunk mismatch".to_string()),
        };
        assert_eq!(
            q.encode(),
            br#"{"timeline_id":3,"turn_index":9,"reason":"corrupt reload (turn 9): chunk mismatch"}"#
                .to_vec()
        );
        assert_eq!(TombstonePayload::decode(&q.encode()).unwrap(), q);
    }
}

/// The degree to which a distilled timeline's turns shed content at compaction.
/// Both modes drop the KV chunks (the bulk of the on-disk cost); they differ on
/// what else survives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistillMode {
    /// Keep `StreamDecl` + `WideQSig` + projection events; drop tokens + KV
    /// chunks. The provenance corpus (calibration exemplars): retrievable by
    /// signature, no verbatim text, not resumable. The default so pre-`mode`
    /// on-disk `Distilled` records decode to the original behaviour.
    #[default]
    ProvenanceOnly,
    /// Keep `StreamDecl` + tokens; drop KV chunks + `WideQSig` + projection
    /// events. Archived conversations: a plain read-only text record — not
    /// retrievable by provenance, not resumable.
    TextOnly,
}

/// JSON payload for a [`RecordType::TurnCoupling`] record — joins turn
/// `from_turn` to the tool response that follows it.
///
/// Only `from_turn` is needed: a tool response is always the immediately
/// following turn, so the record can be written before that turn exists (and
/// therefore before the summariser can observe it). `from_turn ∈ set` reads as
/// "turn `from_turn + 1` is the tool response to turn `from_turn`".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnCouplingPayload {
    pub timeline_id: u64,
    pub from_turn: u32,
}

impl TurnCouplingPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("TurnCouplingPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("TurnCoupling JSON parse: {e}")))
    }
}

/// JSON payload for a [`RecordType::Distilled`] record — names the timeline whose
/// turns should shed content at compaction, and the [`DistillMode`] degree.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillPayload {
    pub timeline_id: u64,
    /// Absent on pre-`mode` records → [`DistillMode::ProvenanceOnly`].
    #[serde(default)]
    pub mode: DistillMode,
}

impl DistillPayload {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("DistillPayload serialise infallible")
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        serde_json::from_slice(buf)
            .map_err(|e| PersistenceError::Corrupt(format!("Distill JSON parse: {e}")))
    }
}

/// One recurrent layer's state inside a [`SnapshotPayload`].
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotLayer {
    /// Trunk layer index this state belongs to.
    pub layer_index: u32,
    /// `[n_v_heads × d_v × d_k]` F32 delta-rule matrices, LE bytes.
    pub n_v_heads: u32,
    pub d_v: u32,
    pub d_k: u32,
    pub state: Vec<u8>,
    /// `[conv_channels × conv_tail_cols]` F32 conv tail, LE bytes.
    pub conv_channels: u32,
    pub conv_tail_cols: u32,
    pub conv_tail: Vec<u8>,
}

impl From<candle_transformers::models::delta_net::ExportedLayerState> for SnapshotLayer {
    /// The model's export rows are field-for-field this record's layer rows —
    /// `candle-conversation` depends on `candle-transformers` and not the
    /// reverse, so the model hands back its own type and the record shape is
    /// built here.
    fn from(l: candle_transformers::models::delta_net::ExportedLayerState) -> Self {
        Self {
            layer_index: l.layer_index,
            n_v_heads: l.n_v_heads,
            d_v: l.d_v,
            d_k: l.d_k,
            state: l.state,
            conv_channels: l.conv_channels,
            conv_tail_cols: l.conv_tail_cols,
            conv_tail: l.conv_tail,
        }
    }
}

/// Binary payload for a [`RecordType::Snapshot`] record — a conversation's
/// full recurrent state at the seal of `turn_index`.
///
/// `schedule_hash` fingerprints the model's layer schedule + DeltaNet dims;
/// restore refuses a snapshot whose hash disagrees with the loaded model
/// (a resumed conversation must recompute instead of scattering a foreign
/// layout into the state arena). `turn_index` binds the snapshot to the turn
/// whose seal produced it: reload discards a snapshot newer than the last
/// recovered turn (a torn shutdown between the turn's records and this one)
/// and falls back to recompute.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotPayload {
    pub timeline_id: u64,
    pub turn_index: u32,
    pub schedule_hash: u64,
    pub layers: Vec<SnapshotLayer>,
}

/// Wire version of [`SnapshotPayload`]; bump on layout change, keep decode
/// for every version ever written.
const SNAPSHOT_PAYLOAD_VERSION: u32 = 1;

impl SnapshotPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        w.put_u32(SNAPSHOT_PAYLOAD_VERSION);
        w.put_u64(self.timeline_id);
        w.put_u32(self.turn_index);
        w.put_u64(self.schedule_hash);
        w.put_u32(self.layers.len() as u32);
        for l in &self.layers {
            encode_snapshot_layer(&mut w, l);
        }
        w.into_bytes()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(buf);
        let version = r.get_u32()?;
        if version != SNAPSHOT_PAYLOAD_VERSION {
            return Err(PersistenceError::Corrupt(format!(
                "snapshot payload version {version} unknown (this build reads \
                 {SNAPSHOT_PAYLOAD_VERSION})"
            )));
        }
        let timeline_id = r.get_u64()?;
        let turn_index = r.get_u32()?;
        let schedule_hash = r.get_u64()?;
        let n_layers = r.get_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(decode_snapshot_layer(&mut r)?);
        }
        Ok(Self {
            timeline_id,
            turn_index,
            schedule_hash,
            layers,
        })
    }
}

/// Binary payload for a [`RecordType::Snapshot`] record written under a
/// **branch checkpoint** stream — the recurrent state after a whole system
/// prompt, for one selector assignment.
///
/// A new conversation does not prefill its system prompt: it Arc-injects sealed
/// section K/V, so the wave never sees those tokens and the recurrent state
/// would enter the first user turn at zero while the attention layers hold the
/// entire prompt. This record is what removes that asymmetry, and it is the
/// only place a conversation needs a state it did not compute
/// (`docs/deltanet_state_persistence.md` §4.6).
///
/// **Two payload shapes share `RecordType::Snapshot`, distinguished by stream
/// id.** That is not an overload: every piece of the single-tail machinery —
/// the accounting supersede key, the recovery walk's location map, the
/// compactor's carry-forward — keys on `header.stream_id` and never decodes the
/// payload. The stream id *is* the identity of what the state belongs to, and a
/// reader computes it before asking, so it always knows which shape it will get.
/// A conversation's snapshot is keyed by `snapshot_stream_id(timeline)`, a
/// branch's by [`branch_checkpoint_stream_id`](super::content_hash::branch_checkpoint_stream_id).
///
/// `prefix_hash` is stored as well as keyed on, so a read can confirm it got
/// the branch it asked for rather than trusting a 64-bit stream id not to
/// collide. `schedule_hash` fingerprints the model's layer schedule + DeltaNet
/// dims exactly as it does for a conversation snapshot: a checkpoint computed
/// under one geometry must never be scattered into another.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchCheckpointPayload {
    /// The branch's cumulative content prefix — its identity.
    pub prefix_hash: ContentHash,
    pub schedule_hash: u64,
    pub layers: Vec<SnapshotLayer>,
}

/// Leading magic, so the two payloads sharing [`RecordType::Snapshot`] can
/// never be mistaken for one another on the wire.
///
/// Without it they are not merely similar, they are **compatible**: a
/// conversation snapshot decodes as a branch checkpoint without error, because
/// `(version, timeline, turn, schedule, n_layers)` and
/// `(version, prefix_lo, prefix_hi, schedule, n_layers)` are the same widths in
/// the same order, so every field lands somewhere plausible. A misrouted read
/// would then produce a *state*, not an error — the failure mode this whole
/// area exists to remove. `SnapshotPayload`'s own version field is `1`, which
/// this magic cannot collide with, so the refusal runs in both directions.
const BRANCH_CHECKPOINT_MAGIC: &[u8; 4] = b"BRCK";

/// Wire version of [`BranchCheckpointPayload`].
const BRANCH_CHECKPOINT_VERSION: u32 = 1;

impl BranchCheckpointPayload {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        for &b in BRANCH_CHECKPOINT_MAGIC {
            w.put_u8(b);
        }
        w.put_u32(BRANCH_CHECKPOINT_VERSION);
        w.put_u64(self.prefix_hash.lo);
        w.put_u64(self.prefix_hash.hi);
        w.put_u64(self.schedule_hash);
        w.put_u32(self.layers.len() as u32);
        for l in &self.layers {
            encode_snapshot_layer(&mut w, l);
        }
        w.into_bytes()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(buf);
        for (i, &want) in BRANCH_CHECKPOINT_MAGIC.iter().enumerate() {
            let got = r.get_u8()?;
            if got != want {
                return Err(PersistenceError::Corrupt(format!(
                    "not a branch checkpoint: magic byte {i} is {got:#04x}, expected \
                     {want:#04x}. A conversation snapshot has the same field widths \
                     in the same order and would otherwise decode as a plausible \
                     branch state."
                )));
            }
        }
        let version = r.get_u32()?;
        if version != BRANCH_CHECKPOINT_VERSION {
            return Err(PersistenceError::Corrupt(format!(
                "branch checkpoint payload version {version} unknown (this build \
                 reads {BRANCH_CHECKPOINT_VERSION})"
            )));
        }
        let lo = r.get_u64()?;
        let hi = r.get_u64()?;
        let schedule_hash = r.get_u64()?;
        let n_layers = r.get_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(decode_snapshot_layer(&mut r)?);
        }
        Ok(Self {
            prefix_hash: ContentHash { lo, hi },
            schedule_hash,
            layers,
        })
    }
}

/// One layer's state, in the shared on-wire form both recurrent payloads use.
fn encode_snapshot_layer(w: &mut ByteWriter, l: &SnapshotLayer) {
    w.put_u32(l.layer_index);
    w.put_u8(0); // dtype tag: 0 = F32 (the only state dtype)
    w.put_u32(l.n_v_heads);
    w.put_u32(l.d_v);
    w.put_u32(l.d_k);
    w.put_blob(&l.state);
    w.put_u32(l.conv_channels);
    w.put_u32(l.conv_tail_cols);
    w.put_blob(&l.conv_tail);
}

fn decode_snapshot_layer(r: &mut ByteReader<'_>) -> Result<SnapshotLayer> {
    let layer_index = r.get_u32()?;
    let dtype = r.get_u8()?;
    if dtype != 0 {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: unknown state dtype tag {dtype}"
        )));
    }
    let n_v_heads = r.get_u32()?;
    let d_v = r.get_u32()?;
    let d_k = r.get_u32()?;
    let state = r.get_blob()?.to_vec();
    let expect = n_v_heads as usize * d_v as usize * d_k as usize * 4;
    if state.len() != expect {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: state blob {} bytes, dims say {expect}",
            state.len()
        )));
    }
    let conv_channels = r.get_u32()?;
    let conv_tail_cols = r.get_u32()?;
    let conv_tail = r.get_blob()?.to_vec();
    let expect_tail = conv_channels as usize * conv_tail_cols as usize * 4;
    if conv_tail.len() != expect_tail {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: conv tail {} bytes, dims say {expect_tail}",
            conv_tail.len()
        )));
    }
    Ok(SnapshotLayer {
        layer_index,
        n_v_heads,
        d_v,
        d_k,
        state,
        conv_channels,
        conv_tail_cols,
        conv_tail,
    })
}

#[cfg(test)]
mod snapshot_payload_tests {
    use super::*;

    fn tiny() -> SnapshotPayload {
        SnapshotPayload {
            timeline_id: 7,
            turn_index: 3,
            schedule_hash: 0xDEAD_BEEF_CAFE_F00D,
            layers: vec![SnapshotLayer {
                layer_index: 1,
                n_v_heads: 1,
                d_v: 2,
                d_k: 2,
                state: vec![0u8; 16],
                conv_channels: 3,
                conv_tail_cols: 1,
                conv_tail: vec![1u8; 12],
            }],
        }
    }

    /// Byte-exact pin: durable state must decode identically forever.
    #[test]
    fn encode_is_byte_stable() {
        let bytes = tiny().encode();
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(&1u32.to_le_bytes()); // version
        expect.extend_from_slice(&7u64.to_le_bytes()); // timeline
        expect.extend_from_slice(&3u32.to_le_bytes()); // turn
        expect.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes());
        expect.extend_from_slice(&1u32.to_le_bytes()); // n_layers
        expect.extend_from_slice(&1u32.to_le_bytes()); // layer_index
        expect.push(0); // dtype
        expect.extend_from_slice(&1u32.to_le_bytes()); // n_v_heads
        expect.extend_from_slice(&2u32.to_le_bytes()); // d_v
        expect.extend_from_slice(&2u32.to_le_bytes()); // d_k
        expect.extend_from_slice(&16u32.to_le_bytes()); // state blob len
        expect.extend_from_slice(&[0u8; 16]);
        expect.extend_from_slice(&3u32.to_le_bytes()); // conv_channels
        expect.extend_from_slice(&1u32.to_le_bytes()); // conv_tail_cols
        expect.extend_from_slice(&12u32.to_le_bytes()); // tail blob len
        expect.extend_from_slice(&[1u8; 12]);
        assert_eq!(bytes, expect);
    }

    #[test]
    fn roundtrip_and_dim_validation() {
        let p = tiny();
        let back = SnapshotPayload::decode(&p.encode()).unwrap();
        assert_eq!(p, back);

        // A state blob that disagrees with its dims is corrupt, not clamped.
        let mut bad = tiny();
        bad.layers[0].state.pop();
        let err = SnapshotPayload::decode(&bad.encode()).unwrap_err();
        assert!(err.to_string().contains("state blob"));
    }

    #[test]
    fn wire_tag_is_twenty() {
        assert_eq!(RecordType::Snapshot as u8, 20);
        assert_eq!(RecordType::from_tag(20), RecordType::Snapshot);
    }

    /// The branch checkpoint's own tag. Durable, so it is pinned like every
    /// other: a tag that moved would make old records decode as a different
    /// kind of state.
    #[test]
    fn branch_checkpoint_wire_tag_is_twenty_one() {
        assert_eq!(RecordType::BranchCheckpoint as u8, 21);
        assert_eq!(RecordType::from_tag(21), RecordType::BranchCheckpoint);
        assert_ne!(RecordType::BranchCheckpoint, RecordType::Snapshot);
    }
}

#[cfg(test)]
mod branch_checkpoint_tests {
    use super::*;

    /// One layer at the 35B's real DeltaNet geometry — 32 V heads, d_v 128,
    /// d_k 128, conv_dim 8192 with a 3-column tail. 2 MiB of state and 96 KiB of
    /// tail per layer, which is what makes the whole-stack figure ~63 MiB.
    fn real_layer(idx: u32) -> SnapshotLayer {
        SnapshotLayer {
            layer_index: idx,
            n_v_heads: 32,
            d_v: 128,
            d_k: 128,
            state: vec![idx as u8; 32 * 128 * 128 * 4],
            conv_channels: 8192,
            conv_tail_cols: 3,
            conv_tail: vec![!(idx as u8); 8192 * 3 * 4],
        }
    }

    fn real_checkpoint() -> BranchCheckpointPayload {
        BranchCheckpointPayload {
            prefix_hash: ContentHash {
                lo: 0x0123_4567_89AB_CDEF,
                hi: 0xFEDC_BA98_7654_3210,
            },
            schedule_hash: 0xDEAD_BEEF_CAFE_F00D,
            layers: (0..30).map(real_layer).collect(),
        }
    }

    /// Byte-exact header pin. The body reuses the layer encoding
    /// [`SnapshotPayload`] already pins byte-for-byte, so only the fields this
    /// payload adds need their own assertion — and they need it for the same
    /// reason: this is durable state, and a silently-shifted field decodes as a
    /// plausible state rather than as an error.
    #[test]
    fn header_encodes_to_exact_bytes() {
        let p = real_checkpoint();
        let bytes = p.encode();
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(b"BRCK"); // magic
        expect.extend_from_slice(&1u32.to_le_bytes()); // version
        expect.extend_from_slice(&0x0123_4567_89AB_CDEFu64.to_le_bytes()); // prefix lo
        expect.extend_from_slice(&0xFEDC_BA98_7654_3210u64.to_le_bytes()); // prefix hi
        expect.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes()); // schedule
        expect.extend_from_slice(&30u32.to_le_bytes()); // n_layers
        assert_eq!(&bytes[..expect.len()], &expect[..]);
    }

    /// Round-trip at the real geometry, byte-identical. Not a tolerance: the
    /// state is copied, never recomputed, so any difference is a layout bug.
    #[test]
    fn round_trips_at_real_35b_geometry() {
        let p = real_checkpoint();
        let back = BranchCheckpointPayload::decode(&p.encode()).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.layers.len(), 30, "every recurrent layer survives");
    }

    /// A truncated state blob is corrupt, not clamped — restoring a partial
    /// state would seed a conversation with a prompt memory that is real for
    /// some layers and zero for others, which is invisible from the outside.
    #[test]
    fn a_short_state_blob_is_refused() {
        let mut bad = real_checkpoint();
        bad.layers[0].state.pop();
        let err = BranchCheckpointPayload::decode(&bad.encode()).unwrap_err();
        assert!(err.to_string().contains("state blob"), "{err}");
    }

    /// The two payloads that share `RecordType::Snapshot` do **not** decode as
    /// each other.
    ///
    /// They are told apart by stream id, so nothing routes a conversation
    /// snapshot into a branch read today. This is the guard for the day
    /// something does — and it is not hypothetical tidiness: without the magic
    /// byte the two are wire-*compatible*. `(version, timeline, turn, schedule,
    /// n_layers)` and `(version, prefix_lo, prefix_hi, schedule, n_layers)` are
    /// the same widths in the same order, so a conversation snapshot decoded as
    /// a branch checkpoint cleanly, with every field landing somewhere
    /// plausible. The first version of this test asserted the refusal and
    /// failed, which is how the magic came to exist.
    #[test]
    fn the_two_snapshot_payloads_do_not_decode_as_each_other() {
        let conv = SnapshotPayload {
            timeline_id: 7,
            turn_index: 3,
            schedule_hash: 11,
            layers: vec![real_layer(0)],
        };
        let err = BranchCheckpointPayload::decode(&conv.encode()).unwrap_err();
        assert!(err.to_string().contains("not a branch checkpoint"), "{err}");
        assert!(SnapshotPayload::decode(&real_checkpoint().encode()).is_err());
    }
}

#[cfg(test)]
mod turn_coupling_tests {
    use super::*;

    /// Exact wire bytes — a coupling is durable state that must decode
    /// identically in every future build, so this asserts the encoding itself,
    /// not merely that it round-trips.
    #[test]
    fn payload_encodes_to_exact_bytes() {
        let p = TurnCouplingPayload {
            timeline_id: 42,
            from_turn: 7,
        };
        assert_eq!(p.encode(), br#"{"timeline_id":42,"from_turn":7}"#.to_vec());
    }

    #[test]
    fn payload_round_trips() {
        let p = TurnCouplingPayload {
            timeline_id: u64::MAX,
            from_turn: u32::MAX,
        };
        assert_eq!(TurnCouplingPayload::decode(&p.encode()).unwrap(), p);
    }

    /// The wire tag is durable: changing it silently reinterprets every existing
    /// coupling record as some other type.
    #[test]
    fn wire_tag_is_nineteen() {
        assert_eq!(RecordType::TurnCoupling.tag(), 19);
        assert_eq!(RecordType::from_tag(19), RecordType::TurnCoupling);
    }

    /// A log written before couplings existed has none of these records, and an
    /// unknown tag must stay skippable rather than abort the walk.
    #[test]
    fn corrupt_payload_is_an_error_not_a_panic() {
        assert!(TurnCouplingPayload::decode(b"not json").is_err());
        assert!(TurnCouplingPayload::decode(b"").is_err());
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

    // A generic record header for the codec / CRC tests. Uses `Tokens` (crc32
    // over the whole payload) — `Chunk` records store a Fletcher golden instead
    // and are verified separately, so they are not meaningful for these tests.
    fn sample_header(payload_len: u64, payload: &[u8]) -> RecordHeader {
        RecordHeader {
            record_type: RecordType::Tokens,
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
        assert_eq!(parsed.record_type, RecordType::Tokens);
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
        // A metadata (non-chunk) record: crc32 over the whole payload catches
        // any flipped byte.
        let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let header = RecordHeader {
            record_type: RecordType::Tokens,
            format: 0,
            payload_len: payload.len() as u64,
            crc: 0, // filled by encode_record
            stream_id: 1,
            chunk_index: 0,
            token_count: 0,
        };
        let mut bytes = encode_record(&header, &payload);
        // Flip a byte inside the payload — find it just after the header newline.
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
    fn recompute_chunk_golden_tracks_kv_not_prefix() {
        // The golden is Fletcher-32 over the KV bytes only.
        let cp = ChunkPayload {
            offset: 3,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
            k_pal: vec![0xAA; 3],
            v_pal: vec![0x55; 3],
            k_scale: vec![1.0, 2.0],
            v_scale: vec![3.0],
            kv_bytes: (0..96u32).map(|i| (i * 7 + 1) as u8).collect(),
        };
        let payload = cp.encode();
        let golden = candle::fletcher::fletcher32(&cp.kv_bytes);
        assert_eq!(recompute_chunk_golden(&payload).unwrap(), golden);

        // Flipping a KV byte changes the golden (a mismatch would be flagged).
        let (_, kv_range) = ChunkPayload::decode_with_kv_range(&payload).unwrap();
        let mut kv_flip = payload.clone();
        kv_flip[kv_range.start] ^= 0x01;
        assert_ne!(recompute_chunk_golden(&kv_flip).unwrap(), golden);

        // Flipping the host-authored prefix (the `offset` low byte) leaves the
        // golden unchanged — it covers only the arena KV bytes.
        let mut prefix_flip = payload.clone();
        prefix_flip[0] ^= 0x01;
        assert_eq!(recompute_chunk_golden(&prefix_flip).unwrap(), golden);
    }

    #[test]
    fn verify_record_crc_skips_chunk_records() {
        // Chunk integrity is the golden's job (checked at write/read), not this
        // metadata verifier — a chunk with a deliberately wrong crc still passes.
        let cp = ChunkPayload {
            offset: 0,
            k_formats: vec![4],
            v_formats: vec![5],
            k_pal: vec![],
            v_pal: vec![],
            k_scale: vec![],
            v_scale: vec![],
            kv_bytes: vec![1, 2, 3, 4],
        };
        let payload = cp.encode();
        let header = RecordHeader {
            record_type: RecordType::Chunk,
            format: 4,
            payload_len: payload.len() as u64,
            crc: 0xDEAD_BEEF,
            stream_id: 1,
            chunk_index: 0,
            token_count: 32,
        };
        assert!(verify_record_crc(&header, &payload).is_ok());
    }

    /// The digest wire tag is the enum discriminant — pinned by raw
    /// value so a reordering of the enum can't silently change the
    /// on-disk `HeaderIndex` format.
    #[test]
    fn record_type_tags_round_trip_with_pinned_values() {
        let pinned: [(RecordType, u8); 16] = [
            (RecordType::ModelSpec, 1),
            (RecordType::Template, 2),
            (RecordType::StreamDecl, 3),
            (RecordType::Chunk, 4),
            (RecordType::Tokens, 5),
            (RecordType::Commit, 7),
            (RecordType::Tokenizer, 9),
            (RecordType::Label, 10),
            (RecordType::ConvState, 11),
            (RecordType::TreeMetadata, 12),
            (RecordType::DebugId, 13),
            (RecordType::Tombstone, 14),
            (RecordType::ProjectionEvents, 15),
            (RecordType::Distilled, 16),
            (RecordType::WideQSig, 17),
            (RecordType::HeaderIndex, 18),
        ];
        for (rt, tag) in pinned {
            assert_eq!(rt.tag(), tag, "{rt:?} wire tag");
            assert_eq!(RecordType::from_tag(tag), rt);
        }
        // Retired tags 6 (the removed per-chunk Signatures — WideQSig
        // replaced it) and 8 (the removed checkpoint), plus future tags,
        // decode to Unknown.
        assert_eq!(RecordType::from_tag(6), RecordType::Unknown);
        assert_eq!(RecordType::from_tag(8), RecordType::Unknown);
        assert_eq!(RecordType::from_tag(200), RecordType::Unknown);
    }

    #[test]
    fn flipped_header_byte_caught_by_json_or_crc() {
        // Mutating a byte inside the header line either breaks JSON
        // parsing or — if the JSON still parses — leaves a stale
        // `crc` value in the header that no longer matches the
        // payload. The fast path catches the first case; verification
        // at the consumption point catches the second.
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

    #[test]
    fn distill_payload_roundtrips_both_modes() {
        for mode in [DistillMode::ProvenanceOnly, DistillMode::TextOnly] {
            let p = DistillPayload {
                timeline_id: 42,
                mode,
            };
            assert_eq!(DistillPayload::decode(&p.encode()).unwrap(), p);
        }
    }

    #[test]
    fn distill_payload_pre_mode_records_decode_provenance_only() {
        // Records written before `mode` existed carry only `timeline_id`; the
        // serde default keeps them decoding to the original behaviour so live
        // substrates (whose calibration corpus is already distilled) load.
        let legacy = br#"{"timeline_id":7}"#;
        let decoded = DistillPayload::decode(legacy).unwrap();
        assert_eq!(decoded.timeline_id, 7);
        assert_eq!(decoded.mode, DistillMode::ProvenanceOnly);
    }
}

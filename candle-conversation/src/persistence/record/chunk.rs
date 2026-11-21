//! The payload of a KV chunk record: quantization metadata, then the arena bytes.
//!
//! The metadata prefix is mandatory, since the quantized bytes cannot be decoded
//! without the palette maps and scales. A cold load can decode everything except
//! the KV bytes and leave those in the read buffer for a host-to-device copy. The
//! chunk's integrity is its GPU golden, a Fletcher-32 over the KV bytes, so
//! recomputing the golden checks the bytes and not the host-authored prefix.

use super::*;

/// Recompute a `Chunk` record's golden — Fletcher-32 over the KV-bytes slice of
/// its payload — to compare against the golden stored in `RecordHeader.crc`
/// (computed on the GPU at seal, before the DtoH copy). The metadata prefix
/// (`offset` / formats / palettes / scales) is host-authored and deliberately
/// not covered. Errors only if `payload` isn't a decodable `ChunkPayload`.
pub fn recompute_chunk_golden(payload: &[u8]) -> Result<u32> {
    let (_, kv_range) = ChunkPayload::decode_with_kv_range(payload)?;
    Ok(candle::fletcher::fletcher32(&payload[kv_range]))
}

// `ChunkPayload` — the one structured binary payload. Kept binary
// because chunks are the hot path; JSON-encoding KV bytes would
// inflate size and parse time for no benefit.

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

//! GPU-side slot-state cache for a single sequence.
//!
//! Holds a pinned host buffer of serialised `TokenSliceHost` bytes and a
//! matching device-side backing allocation.  Dirty chunk indices accumulate
//! on the [`GpuChunksGuard`]; on drop the guard coalesces adjacent indices
//! into contiguous byte ranges and issues one `stream.memcpy_htod` per run.

use std::sync::Arc;
use candle::cuda_backend::WrapErr;
use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use candle::quantized::pinned_staging::PinnedBuf;
use crate::kv_cache::arena_table::{ArenaFormatTag, ResolvedArenaInfo, N_PALETTE};
use super::types::ChunkWindow;

/// Cached pinned-host + device-side serialised slot-state for one sequence.
pub(crate) struct GpuChunks {
    /// Pinned write-combined host buffer containing serialised `TokenSlice` bytes.
    /// Starts empty (len = 0); grown on first `update` call.
    buf: PinnedBuf,
    /// Device-side backing buffer, always the same byte length as `buf`.
    /// `None` until the first non-empty update.
    gpu: Option<CudaSlice<u8>>,
    /// Stream used for all async H→D copies. `None` for CPU-backed tests even
    /// when the crate is compiled with the CUDA feature enabled.
    stream: Option<Arc<CudaStream>>,
    /// Byte size of one serialised chunk entry (e.g. one `TokenSliceHost`).
    /// Zero until the first call to [`GpuChunksGuard::update`].
    chunk_byte_size: usize,
}

impl std::fmt::Debug for GpuChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChunks")
            .field("bytes", &self.buf.len())
            .field("chunk_byte_size", &self.chunk_byte_size)
            .finish()
    }
}

impl GpuChunks {
    pub(crate) fn new(stream: Option<Arc<CudaStream>>) -> Self {
        Self {
            // alloc_owned(0) returns a zero-len Bump variant — no CUDA call.
            buf: PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail"),
            gpu: None,
            stream,
            chunk_byte_size: 0,
        }
    }

    pub(crate) fn as_mut(&mut self) -> GpuChunksGuard<'_> {
        GpuChunksGuard { inner: self, dirty_chunks: Vec::new() }
    }

    /// Returns the raw GPU device pointer for this sequence's slot-state buffer.
    /// Returns 0 if the buffer has not been allocated yet (no chunks).
    pub(crate) fn raw_device_ptr(&self) -> u64 {
        match (self.gpu.as_ref(), self.stream.as_ref()) {
            (Some(s), Some(stream)) => {
                let (ptr, _guard) = s.device_ptr(stream);
                ptr
            }
            _ => 0,
        }
    }

    /// Number of serialised chunk entries currently in the buffer.
    pub(crate) fn n_chunks(&self) -> usize {
        if self.chunk_byte_size > 0 {
            self.buf.len() / self.chunk_byte_size
        } else {
            0
        }
    }

}

/// Mutable accessor for [`GpuChunks`].
///
/// Chunk writes accumulate in `dirty_chunks`; on drop the guard coalesces
/// adjacent chunk indices into contiguous byte ranges and issues one
/// `stream.memcpy_htod` per contiguous run.
pub(crate) struct GpuChunksGuard<'a> {
    inner: &'a mut GpuChunks,
    /// Indices of chunk entries written since this guard was created.
    /// Maintained in ascending order so the drop coalescer can merge
    /// adjacent entries in a single pass.
    dirty_chunks: Vec<usize>,
}

impl GpuChunksGuard<'_> {
    /// Resize both the pinned host buffer and the GPU backing allocation to
    /// hold exactly `n_chunks` entries of `chunk_byte_size` bytes each.
    ///
    /// Existing content is preserved up to the smaller of the old and new
    /// sizes; any extension is zeroed.
    fn resize(&mut self, n_chunks: usize, chunk_byte_size: usize) -> candle::Result<()> {
        let byte_len = n_chunks
            .checked_mul(chunk_byte_size)
            .expect("overflow in resize");
        let mut new_buf = PinnedBuf::alloc_owned(byte_len)?;
        {
            let new_slice = new_buf.as_mut_slice();
            let copy_bytes = self.inner.buf.len().min(byte_len);
            new_slice[..copy_bytes].copy_from_slice(&self.inner.buf.as_slice()[..copy_bytes]);
            new_slice[copy_bytes..].fill(0);
        }
        // Sync before dropping the old GPU allocation: any in-flight
        // memcpy_htod issued by a prior guard drop may still be writing to it.
        if self.inner.gpu.is_some() {
            if let Some(stream) = self.inner.stream.as_ref() {
                stream.synchronize().w()?;
            }
        }
        // Dropping the old PinnedBuf calls cuMemFreeHost (no-op for len=0 Bump).
        self.inner.buf = new_buf;
        self.inner.chunk_byte_size = chunk_byte_size;

        if byte_len > 0 {
            if let Some(stream) = self.inner.stream.as_ref() {
                // SAFETY: alloc is untyped; caller fills before use.
                let gpu = unsafe { stream.alloc::<u8>(byte_len).w()? };
                self.inner.gpu = Some(gpu);
            } else {
                self.inner.gpu = None;
            }
        } else {
            self.inner.gpu = None;
        }

        Ok(())
    }

    /// Overwrite the serialised slot at `chunk_idx` in-place and mark it dirty.
    ///
    /// Returns an error if `chunk_idx` is out of range or the stored
    /// `chunk_byte_size` does not match the current model configuration.
    pub(crate) fn update_chunk(
        &mut self,
        chunk_idx: usize,
        chunk: &ChunkWindow,
        n_kv_head: usize,
        head_dim: usize,
        rope_base: u32,
        arena_info: &[ResolvedArenaInfo],
    ) -> candle::Result<()> {
        let chunk_byte_size = token_slice_serialized_size(n_kv_head, head_dim);
        let current_n = if self.inner.chunk_byte_size == chunk_byte_size && chunk_byte_size > 0 {
            self.inner.buf.len() / chunk_byte_size
        } else {
            0
        };
        if chunk_idx >= current_n {
            candle::bail!(
                "update_chunk: index {chunk_idx} out of range (buf holds {current_n} chunks)"
            );
        }
        let bs = chunk_idx * chunk_byte_size;
        let slot = &mut self.inner.buf.as_mut_slice()[bs..bs + chunk_byte_size];
        serialize_chunk_window(chunk, n_kv_head, head_dim, rope_base, arena_info, slot);
        self.dirty_chunks.push(chunk_idx);
        Ok(())
    }

    /// Clear the buffer and re-serialise all `chunks` from scratch, computing
    /// each chunk's `rope_base` from the cumulative usage of preceding chunks.
    ///
    /// `write_len` overrides the last chunk's serialised `len` field so callers
    /// can supply the true sequence-offset-derived length rather than the
    /// potentially-stale `chunk.usage` value.
    ///
    /// The actual H→D upload is deferred to guard drop as usual.
    pub(crate) fn rebuild_decode(
        &mut self,
        chunks: &[ChunkWindow],
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
        write_len: u16,
    ) -> candle::Result<()> {
        if chunks.is_empty() {
            self.clear();
            return Ok(());
        }
        self.dirty_chunks.clear();
        let chunk_byte_size = token_slice_serialized_size(n_kv_head, head_dim);
        self.resize(chunks.len(), chunk_byte_size)?;

        let mut rope_base = 0u32;
        for (i, chunk) in chunks.iter().enumerate() {
            let bs = i * chunk_byte_size;
            let slot = &mut self.inner.buf.as_mut_slice()[bs..bs + chunk_byte_size];
            let len = if i + 1 == chunks.len() { write_len } else { chunk.usage as u16 };
            serialize_chunk_window_with_len(chunk, n_kv_head, head_dim, rope_base, len, arena_info, slot);
            self.dirty_chunks.push(i);
            rope_base += chunk.usage;
        }
        Ok(())
    }

    /// Reset the host buffer to empty and cancel any pending dirty uploads.
    ///
    /// Drops the GPU-side allocation and zeroes the chunk count.  Called when
    /// a sequence's slot-state is fully invalidated (e.g. evicted or freed).
    pub(crate) fn clear(&mut self) {
        self.dirty_chunks.clear();
        // Sync before dropping the GPU allocation: in-flight memcpy_htod
        // operations from a prior guard drop may still be writing to it.
        if self.inner.gpu.is_some() {
            if let Some(stream) = self.inner.stream.as_ref() {
                if let Err(e) = stream.synchronize().w() {
                    log::warn!("GpuChunksGuard::clear: stream sync failed: {e:?}");
                }
            }
        }
        self.inner.buf =
            PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail");
        self.inner.gpu = None;
        self.inner.chunk_byte_size = 0;
    }
}

impl Drop for GpuChunksGuard<'_> {
    fn drop(&mut self) {
        if self.dirty_chunks.is_empty() {
            return;
        }

        // Normalise: sort ascending and remove duplicates so the coalescer's
        // single-pass adjacency check is correct even when append_chunks was
        // called multiple times with out-of-order or overlapping ranges.
        self.dirty_chunks.sort_unstable();
        self.dirty_chunks.dedup();

        // Split the borrow so we can hold &[u8] from buf alongside &mut gpu.
        let GpuChunks { buf, gpu, stream, chunk_byte_size } = &mut *self.inner;
        let chunk_byte_size = *chunk_byte_size;
        if chunk_byte_size == 0 {
            return;
        }
        let Some(gpu) = gpu.as_mut() else {
            return;
        };
        let Some(stream) = stream.as_ref() else {
            return;
        };
        let host: &[u8] = buf.as_slice();

        // Walk dirty_chunks (ascending) and coalesce adjacent indices into
        // contiguous byte ranges, issuing one memcpy_htod per run.
        let mut start = self.dirty_chunks[0];
        let mut end = start + 1;
        for &idx in &self.dirty_chunks[1..] {
            if idx == end {
                end += 1;
            } else {
                let (bs, be) = (start * chunk_byte_size, end * chunk_byte_size);
                if let Err(e) = stream
                    .memcpy_htod(&host[bs..be], &mut gpu.slice_mut(bs..be))
                    .w()
                {
                    log::warn!("GpuChunksGuard: memcpy_htod [{bs}..{be}] error: {e:?}");
                }
                start = idx;
                end = idx + 1;
            }
        }
        // Flush the final (or only) contiguous run.
        let (bs, be) = (start * chunk_byte_size, end * chunk_byte_size);
        if let Err(e) = stream
            .memcpy_htod(&host[bs..be], &mut gpu.slice_mut(bs..be))
            .w()
        {
            log::warn!("GpuChunksGuard: memcpy_htod [{bs}..{be}] error: {e:?}");
        }
    }
}

impl Drop for GpuChunks {
    fn drop(&mut self) {
        // Synchronise the stream before the pinned host buffer and GPU
        // allocation are freed.  Any in-flight memcpy_htod issued by a
        // prior GpuChunksGuard drop must complete before cuMemFreeHost
        // and cuMemFree are called on the backing buffers.
        if self.gpu.is_some() {
            if let Some(stream) = self.stream.as_ref() {
                if let Err(e) = stream.synchronize().w() {
                    log::warn!("GpuChunks::drop: stream sync failed: {e:?}");
                }
            }
        }
    }
}

impl Clone for GpuChunks {
    fn clone(&self) -> Self {
        // Pinned buffers and GPU allocations are not cloneable; forked
        // sequences start fresh on the same stream.
        Self {
            buf: PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail"),
            gpu: None,
            stream: self.stream.clone(),
            chunk_byte_size: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Returns the serialised byte-size of one `KvHead` entry.
///
/// Layout: `k_pal[head_dim/4] + v_pal[head_dim/4] + k_ptr[4]×8 + v_ptr[4]×8 + k_fmt[4] + v_fmt[4] + k_scale[4]×4 + v_scale[4]×4`
///
/// The pal_map packs 4 dims/byte (2 bits per dim), so each side is `head_dim/4`
/// bytes. The `/4` is the bit-packing density, *not* N_PALETTE — they
/// coincidentally both equal 4 today but are independent constants.
pub(crate) fn kv_head_serialized_size(head_dim: usize) -> usize {
    (head_dim / 4) * 2 // k_pal + v_pal, each head_dim/4 bytes
        + 32 // k_ptr: 4 × u64
        + 32 // v_ptr: 4 × u64
        + 4  // k_fmt: 4 × u8
        + 4  // v_fmt: 4 × u8
        + 16 // k_scale: 4 × f32 (outer scale per palette)
        + 16 // v_scale: 4 × f32 (outer scale per palette)
}

/// Returns the serialised byte-size of one `TokenSlice` entry.
pub(crate) fn token_slice_serialized_size(n_kv_head: usize, head_dim: usize) -> usize {
    8 + n_kv_head * kv_head_serialized_size(head_dim)
}

/// Write the identity 2-bit palette map for the given head dimension into `dst`.
///
/// Assigns dim `d` to palette `d / (head_dim / N_PALETTE)`, packed 4 dims
/// per byte in little-endian order: `(d3<<6)|(d2<<4)|(d1<<2)|d0`.
/// For N_PALETTE=4 this produces the (0,0,1,1,2,2,3,3,...) pattern.
/// `dst` must be exactly `(head_dim / 4).max(1)` bytes long.
#[cfg(test)]
pub(crate) fn write_identity_pal_map(head_dim: usize, dst: &mut [u8]) {
    let sub_hd = (head_dim / N_PALETTE).max(1);
    dst.fill(0);
    for d in 0..head_dim {
        let pal_idx = (d / sub_hd).min(N_PALETTE - 1) as u8;
        let byte_idx = d / 4;
        let bit_shift = (d % 4) * 2;
        dst[byte_idx] |= pal_idx << bit_shift;
    }
}

/// Serialise one `ChunkWindow` into `dst` with an explicit `len` override.
///
/// Identical to [`serialize_chunk_window`] except that `len` replaces
/// `chunk.usage as u16`.  Used by [`GpuChunksGuard::rebuild_decode`] to
/// supply the true sequence-offset-derived length for the write chunk.
pub(crate) fn serialize_chunk_window_with_len(
    chunk: &ChunkWindow,
    n_kv_head: usize,
    head_dim: usize,
    rope_base: u32,
    len: u16,
    arena_info: &[ResolvedArenaInfo],
    dst: &mut [u8],
) {
    debug_assert!(head_dim >= 4, "head_dim must be >= 4 for 2-bit pal_map packing");
    let pal_total = n_kv_head * (head_dim / 4);
    let scale_total = n_kv_head * N_PALETTE;
    debug_assert!(
        chunk.k_pal.is_empty() || chunk.k_pal.len() == pal_total,
        "k_pal length must be 0 or {pal_total}, got {}",
        chunk.k_pal.len()
    );
    debug_assert!(
        chunk.v_pal.is_empty() || chunk.v_pal.len() == pal_total,
        "v_pal length must be 0 or {pal_total}, got {}",
        chunk.v_pal.len()
    );
    debug_assert!(
        chunk.k_scale.is_empty() || chunk.k_scale.len() == scale_total,
        "k_scale length must be 0 or {scale_total}, got {}",
        chunk.k_scale.len()
    );
    debug_assert!(
        chunk.v_scale.is_empty() || chunk.v_scale.len() == scale_total,
        "v_scale length must be 0 or {scale_total}, got {}",
        chunk.v_scale.len()
    );
    let mut pos = 0;

    macro_rules! put {
        ($b:expr) => {{
            let b: &[u8] = $b;
            dst[pos..pos + b.len()].copy_from_slice(b);
            pos += b.len();
        }};
    }

    put!(&chunk.offset.to_le_bytes());
    put!(&len.to_le_bytes());
    put!(&rope_base.to_le_bytes());

    let pal_bytes = head_dim / 4;

    for h in 0..n_kv_head {
        dst[pos..pos + pal_bytes].copy_from_slice(&chunk.k_pal[h * pal_bytes..(h + 1) * pal_bytes]);
        pos += pal_bytes;
        dst[pos..pos + pal_bytes].copy_from_slice(&chunk.v_pal[h * pal_bytes..(h + 1) * pal_bytes]);
        pos += pal_bytes;

        let mut k_ptr = [0u64; N_PALETTE];
        let mut v_ptr = [0u64; N_PALETTE];
        let mut k_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];
        let mut v_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];

        for p in 0..N_PALETTE {
            let k_gid = chunk.gids.k_gid_pal(h, p);
            let v_gid = chunk.gids.v_gid_pal(h, p);
            if let Some(ai) = arena_info.get(k_gid.arena_idx()) {
                k_ptr[p] =
                    ai.base_ptr + k_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                k_fmt[p] = ai.k_format_tag.as_u8();
            }
            if let Some(ai) = arena_info.get(v_gid.arena_idx()) {
                v_ptr[p] =
                    ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                v_fmt[p] = ai.v_format_tag.as_u8();
            }
        }

        for &ptr in &k_ptr {
            put!(&ptr.to_le_bytes());
        }
        for &ptr in &v_ptr {
            put!(&ptr.to_le_bytes());
        }
        put!(&k_fmt);
        put!(&v_fmt);
        // k_scale[4] and v_scale[4]: f32 outer scale per palette (encoder *,
        // decoder /), default 1.0
        let scale_base = h * N_PALETTE;
        for p in 0..N_PALETTE {
            let s = chunk.k_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
        for p in 0..N_PALETTE {
            let s = chunk.v_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
    }
}

/// Serialise one `ChunkWindow` into `dst` in the `TokenSlice` layout the CUDA
/// kernel expects:
///
/// ```text
///   offset:  u16  (LE)
///   len:     u16  (LE)   ← chunk.usage truncated to u16
///   rope:    u32  (LE)
///   for each KV head:
///     k_pal[head_dim/4]        2-bit packed, identity routing
///     v_pal[head_dim/4]        same
///     k_ptr[N_PALETTE]  u64×4  resolved K device pointers
///     v_ptr[N_PALETTE]  u64×4  resolved V device pointers
///     k_fmt[N_PALETTE]  u8×4   K format tags
///     v_fmt[N_PALETTE]  u8×4   V format tags
/// ```
pub(crate) fn serialize_chunk_window(
    chunk: &ChunkWindow,
    n_kv_head: usize,
    head_dim: usize,
    rope_base: u32,
    arena_info: &[ResolvedArenaInfo],
    dst: &mut [u8],
) {
    debug_assert!(head_dim >= 4, "head_dim must be >= 4 for 2-bit pal_map packing");
    let pal_total = n_kv_head * (head_dim / 4);
    let scale_total = n_kv_head * N_PALETTE;
    debug_assert!(
        chunk.k_pal.is_empty() || chunk.k_pal.len() == pal_total,
        "k_pal length must be 0 or {pal_total}, got {}",
        chunk.k_pal.len()
    );
    debug_assert!(
        chunk.v_pal.is_empty() || chunk.v_pal.len() == pal_total,
        "v_pal length must be 0 or {pal_total}, got {}",
        chunk.v_pal.len()
    );
    debug_assert!(
        chunk.k_scale.is_empty() || chunk.k_scale.len() == scale_total,
        "k_scale length must be 0 or {scale_total}, got {}",
        chunk.k_scale.len()
    );
    debug_assert!(
        chunk.v_scale.is_empty() || chunk.v_scale.len() == scale_total,
        "v_scale length must be 0 or {scale_total}, got {}",
        chunk.v_scale.len()
    );
    let mut pos = 0;

    macro_rules! put {
        ($b:expr) => {{
            let b: &[u8] = $b;
            dst[pos..pos + b.len()].copy_from_slice(b);
            pos += b.len();
        }};
    }

    put!(&chunk.offset.to_le_bytes());
    put!(&(chunk.usage as u16).to_le_bytes());
    put!(&rope_base.to_le_bytes());

    let pal_bytes = head_dim / 4;

    for h in 0..n_kv_head {
        dst[pos..pos + pal_bytes].copy_from_slice(&chunk.k_pal[h * pal_bytes..(h + 1) * pal_bytes]);
        pos += pal_bytes;
        dst[pos..pos + pal_bytes].copy_from_slice(&chunk.v_pal[h * pal_bytes..(h + 1) * pal_bytes]);
        pos += pal_bytes;

        let mut k_ptr = [0u64; N_PALETTE];
        let mut v_ptr = [0u64; N_PALETTE];
        let mut k_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];
        let mut v_fmt = [ArenaFormatTag::BF16.as_u8(); N_PALETTE];

        for p in 0..N_PALETTE {
            let k_gid = chunk.gids.k_gid_pal(h, p);
            let v_gid = chunk.gids.v_gid_pal(h, p);
            if let Some(ai) = arena_info.get(k_gid.arena_idx()) {
                k_ptr[p] =
                    ai.base_ptr + k_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                k_fmt[p] = ai.k_format_tag.as_u8();
            }
            if let Some(ai) = arena_info.get(v_gid.arena_idx()) {
                v_ptr[p] =
                    ai.base_ptr + v_gid.chunk_idx() as u64 * ai.chunk_byte_stride as u64;
                v_fmt[p] = ai.v_format_tag.as_u8();
            }
        }

        for &ptr in &k_ptr {
            put!(&ptr.to_le_bytes());
        }
        for &ptr in &v_ptr {
            put!(&ptr.to_le_bytes());
        }
        put!(&k_fmt);
        put!(&v_fmt);
        // k_scale[4] and v_scale[4]: f32 outer scale per palette (encoder *,
        // decoder /), default 1.0
        let scale_base = h * N_PALETTE;
        for p in 0..N_PALETTE {
            let s = chunk.k_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
        for p in 0..N_PALETTE {
            let s = chunk.v_scale.get(scale_base + p).copied().unwrap_or(1.0);
            put!(&s.to_le_bytes());
        }
    }
}

// Unit tests live in tests/gpu_chunks_tests.rs (compiled under #[cfg(feature = "cuda")]).

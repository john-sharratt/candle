//! GPU-side slot-state cache for a single sequence.
//!
//! Holds a pinned host buffer and a matching device backing allocation, laid out
//! in two sections — `[ slice headers (16 B) | KvHead records ]` — so the slice
//! headers stay a contiguous 16-byte-stride array (what the kernel's `get_slice`
//! indexes) while each header's `kvheads_ptr` points into the records section.
//! Dirty chunk indices accumulate on the [`GpuChunksGuard`]; on drop the guard
//! coalesces adjacent indices into runs and, because of the two sections, issues
//! two `stream.memcpy_htod` per run (the headers range + the records range).

use super::slot_state_arena::{self, SlotStateSlot};
use super::types::ChunkWindow;
use crate::kv_cache::arena_table::ResolvedArenaInfo;
#[cfg(test)]
use crate::kv_cache::arena_table::N_PALETTE;
use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::cuda_backend::WrapErr;
use candle::quantized::pinned_staging::PinnedBuf;
use std::sync::Arc;

/// Cached pinned-host + device-side serialised slot-state for one sequence.
pub(crate) struct GpuChunks {
    /// Pinned write-combined host buffer containing serialised `TokenSlice` bytes.
    /// Starts empty (len = 0); grown on first `update` call.
    buf: PinnedBuf,
    /// Device-side backing: a slot from the region tier's doubling class
    /// family (`slot_state_arena`), **not** an allocation. `None` until the
    /// first non-empty update. Its capacity is a class width, so it is
    /// generally larger than `buf.len()`; the kernel's walk is bounded by
    /// `n_chunks`, never by the slot.
    slot: Option<SlotStateSlot>,
    /// Stream used for all async H→D copies. `None` for CPU-backed tests even
    /// when the crate is compiled with the CUDA feature enabled.
    stream: Option<Arc<CudaStream>>,
    /// Byte size of one serialised chunk entry (e.g. one `TokenSliceHost`).
    /// Zero until the first call to [`GpuChunksGuard::update`].
    chunk_byte_size: usize,
    /// Entries currently live.
    ///
    /// Kept explicitly because both buffers are now **capacities**: the pinned
    /// host buffer is grow-only and the device slot is a class width, so
    /// neither length divides down to the entry count any more.
    n_chunks: usize,
}

impl std::fmt::Debug for GpuChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChunks")
            .field("host_capacity", &self.buf.len())
            .field("chunk_byte_size", &self.chunk_byte_size)
            .field("n_chunks", &self.n_chunks)
            .finish()
    }
}

impl GpuChunks {
    pub(crate) fn new(stream: Option<Arc<CudaStream>>) -> Self {
        Self {
            // alloc_owned(0) returns a zero-len Bump variant — no CUDA call.
            buf: PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail"),
            slot: None,
            stream,
            chunk_byte_size: 0,
            n_chunks: 0,
        }
    }

    pub(crate) fn as_mut(&mut self) -> GpuChunksGuard<'_> {
        GpuChunksGuard {
            inner: self,
            dirty_chunks: Vec::new(),
        }
    }

    /// Returns the raw GPU device pointer for this sequence's slot-state buffer.
    /// Returns 0 if the buffer has not been allocated yet (no chunks).
    pub(crate) fn raw_device_ptr(&self) -> u64 {
        self.slot.as_ref().map_or(0, |s| s.ptr)
    }

    /// Release the device slot back to its class, if one is held.
    ///
    /// No fence. Every sequence's copies run on the device's *primary* stream,
    /// so a slot handed straight back out cannot be written before the copies
    /// that read it have run — they are enqueued ahead. This is what lets the
    /// full `stream.synchronize()` this path used to perform on every
    /// structural mutation disappear rather than merely move (audit A13).
    fn release_slot(&mut self) {
        if let (Some(slot), Some(stream)) = (self.slot.take(), self.stream.as_ref()) {
            slot_state_arena::release(stream, slot);
        }
    }

    /// Number of serialised chunk entries currently in the buffer.
    pub(crate) fn n_chunks(&self) -> usize {
        self.n_chunks
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
    /// Ensure the host buffer and the device slot can hold `n_chunks` entries
    /// of `chunk_byte_size` bytes.
    ///
    /// **Content is not preserved, and does not need to be.** The buffer has
    /// two sections — `[ slice headers | records ]` — so the records section
    /// moves whenever `n_chunks` changes, and the only caller
    /// ([`Self::rebuild_decode`]) rewrites every entry immediately after. The
    /// old code copied `min(old, new)` bytes forward and zeroed the rest,
    /// which was work spent producing bytes nobody read.
    ///
    /// Growth on the device is a **promotion**: claim a wider slot from the
    /// next class up and release the old one. No allocator call, no sync — and
    /// no copy, per the paragraph above. Growth on the host is grow-only: the
    /// pinned buffer is reused whenever it is already big enough, so a
    /// sequence crossing a 32-token boundary no longer pays a `cuMemFreeHost` +
    /// `cuMemHostAlloc` pair per layer.
    fn resize(&mut self, n_chunks: usize, chunk_byte_size: usize) -> candle::Result<()> {
        let byte_len = n_chunks
            .checked_mul(chunk_byte_size)
            .expect("overflow in resize");
        self.inner.chunk_byte_size = chunk_byte_size;
        self.inner.n_chunks = n_chunks;

        if byte_len == 0 {
            self.inner.release_slot();
            return Ok(());
        }

        // Host capacity follows the SAME doubling ladder as the device slot,
        // not `byte_len` exactly.
        //
        // Growing to the exact size would re-allocate on every `push_chunk` —
        // the very churn this change exists to remove — and, worse, would drop
        // the old pinned buffer while an async `memcpy_htod` may still be
        // reading it. Rounding to the ladder makes the reallocation
        // logarithmic in depth instead of linear, which is what makes the
        // `stream.synchronize()` below affordable: it now guards a rare event
        // rather than one per 32 tokens per layer.
        let want = slot_state_arena::class_bytes_for(byte_len)?;
        if self.inner.buf.len() < want {
            if let Some(stream) = self.inner.stream.as_ref() {
                // `cuMemFreeHost` unpins pages an in-flight H2D may still be
                // sourcing from, so the old buffer cannot be dropped until the
                // stream drains.
                stream.synchronize().w()?;
            }
            self.inner.buf = PinnedBuf::alloc_owned(want)?;
        }

        // Device: keep the slot if it still fits, else promote.
        let Some(stream) = self.inner.stream.as_ref().cloned() else {
            self.inner.slot = None;
            return Ok(());
        };
        let fits = self
            .inner
            .slot
            .as_ref()
            .is_some_and(|s| s.capacity() >= byte_len);
        if !fits {
            let next = slot_state_arena::claim(&stream, byte_len)?;
            self.inner.release_slot();
            self.inner.slot = Some(next);
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
        // The LIVE entry count, not one derived from the buffer length. Both
        // buffers are capacities now — the host side is grow-only and the
        // device slot is a class width — and `records_off` below is
        // `n * SLICE_HEADER_BYTES`, so a capacity-derived count would place
        // the records section past where `rebuild_decode` wrote it and point
        // every `kvheads_ptr` at unwritten bytes.
        let current_n = if self.inner.chunk_byte_size == chunk_byte_size && chunk_byte_size > 0 {
            self.inner.n_chunks
        } else {
            0
        };
        if chunk_idx >= current_n {
            candle::bail!(
                "update_chunk: index {chunk_idx} out of range (buf holds {current_n} chunks)"
            );
        }
        let rec_bytes = record_bytes(n_kv_head, head_dim);
        let records_off = current_n * SLICE_HEADER_BYTES;
        let base = self.inner.raw_device_ptr();
        let len = chunk.usage as u16;
        // Resident chunk: point at its meta-pool record; else inline (see
        // rebuild_decode for the rationale).
        let kvheads_ptr = match chunk.meta.as_ref().map(|m| m.device_addr()) {
            Some(addr) if addr != 0 => addr,
            _ => {
                let r0 = records_off + chunk_idx * rec_bytes;
                write_record_for_chunk(
                    &mut self.inner.buf.as_mut_slice()[r0..r0 + rec_bytes],
                    chunk,
                    n_kv_head,
                    head_dim,
                    arena_info,
                );
                base + (records_off + chunk_idx * rec_bytes) as u64
            }
        };
        let s0 = chunk_idx * SLICE_HEADER_BYTES;
        write_slice_header(
            &mut self.inner.buf.as_mut_slice()[s0..s0 + SLICE_HEADER_BYTES],
            chunk.offset,
            len,
            rope_base,
            kvheads_ptr,
        );
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
        write_idx: usize,
    ) -> candle::Result<()> {
        if chunks.is_empty() {
            self.clear();
            return Ok(());
        }
        self.dirty_chunks.clear();
        let n = chunks.len();
        let chunk_byte_size = token_slice_serialized_size(n_kv_head, head_dim);
        self.resize(n, chunk_byte_size)?;

        // Two sections: slice headers [0 .. n*16), then records. Resolve the GPU
        // base once (the allocation is fixed for the buffer's lifetime) so each
        // header's kvheads_ptr can point into the records section.
        let rec_bytes = record_bytes(n_kv_head, head_dim);
        let records_off = n * SLICE_HEADER_BYTES;
        let base = self.inner.raw_device_ptr();

        let mut rope_base = 0u32;
        for (i, chunk) in chunks.iter().enumerate() {
            // The writer chunk gets the seq_offset-derived `write_len`; every
            // other chunk (including trailing empties past the writer) keeps its
            // own stored usage.
            let len = if i == write_idx {
                write_len
            } else {
                chunk.usage as u16
            };
            // Resident chunk: point at its co-resident record in the meta-pool
            // slab and skip serializing a record here. Otherwise serialize an
            // inline record into this chunk's records-section slot and point at it.
            let kvheads_ptr = match chunk.meta.as_ref().map(|m| m.device_addr()) {
                Some(addr) if addr != 0 => addr,
                _ => {
                    let r0 = records_off + i * rec_bytes;
                    write_record_for_chunk(
                        &mut self.inner.buf.as_mut_slice()[r0..r0 + rec_bytes],
                        chunk,
                        n_kv_head,
                        head_dim,
                        arena_info,
                    );
                    base + (records_off + i * rec_bytes) as u64
                }
            };
            let s0 = i * SLICE_HEADER_BYTES;
            write_slice_header(
                &mut self.inner.buf.as_mut_slice()[s0..s0 + SLICE_HEADER_BYTES],
                chunk.offset,
                len,
                rope_base,
                kvheads_ptr,
            );
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
        // No sync and no free: the device side goes back to its class free
        // list (see `GpuChunks::release_slot`), and the pinned host buffer is
        // kept for the next fill rather than unpinned and re-pinned. `clear`
        // is called by *every* structural mutation — `push_chunk` alone fires
        // each time a sequence crosses a 32-token boundary — so what used to
        // happen here was the bulk of audit A13's ~3,000 alloc/free/sync
        // cycles per 32 decoded tokens at batch 64.
        self.inner.release_slot();
        self.inner.chunk_byte_size = 0;
        self.inner.n_chunks = 0;
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

        let GpuChunks {
            buf,
            slot,
            stream,
            chunk_byte_size,
            n_chunks,
        } = &mut *self.inner;
        let chunk_byte_size = *chunk_byte_size;
        let n_chunks = *n_chunks;
        if chunk_byte_size == 0 || n_chunks == 0 {
            return;
        }
        let Some(slot) = slot.as_ref() else {
            return;
        };
        let Some(stream) = stream.as_ref() else {
            return;
        };
        let host: &[u8] = buf.as_slice();

        // Two-section buffer: slice headers [0 .. n*16), then records. A
        // coalesced run of adjacent chunk indices is contiguous in *both*
        // sections, so each run uploads two ranges: the 16-byte headers and
        // the records.
        let rec_bytes = chunk_byte_size - SLICE_HEADER_BYTES;
        let records_off = n_chunks * SLICE_HEADER_BYTES;
        let upload = |range: std::ops::Range<usize>, what: &str| {
            // SAFETY: `range` lies inside the live entry count, which
            // `resize` sized the slot to hold; `host` is the pinned staging
            // buffer, alive for this call; and the copy is enqueued on the
            // same primary stream every reader of this slot uses.
            let res = unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                    slot.ptr + range.start as u64,
                    &host[range.clone()],
                    stream.cu_stream(),
                )
            };
            if let Err(e) = res {
                log::warn!("GpuChunksGuard: {what} memcpy_htod {range:?} error: {e:?}");
            }
        };
        let upload_run = |start: usize, end: usize| {
            upload(
                start * SLICE_HEADER_BYTES..end * SLICE_HEADER_BYTES,
                "header",
            );
            if rec_bytes > 0 {
                upload(
                    records_off + start * rec_bytes..records_off + end * rec_bytes,
                    "record",
                );
            }
        };

        let mut start = self.dirty_chunks[0];
        let mut end = start + 1;
        for &idx in &self.dirty_chunks[1..] {
            if idx == end {
                end += 1;
            } else {
                upload_run(start, end);
                start = idx;
                end = idx + 1;
            }
        }
        upload_run(start, end);
    }
}

impl Drop for GpuChunks {
    fn drop(&mut self) {
        // The device slot goes back to its class free list, which needs no
        // fence (see `release_slot`). The pinned host buffer still needs one:
        // `cuMemFreeHost` unpins pages an in-flight `memcpy_htod_async` may
        // still be reading, and unlike the device side there is no free list
        // holding it until the stream catches up. That half moves to the
        // pinned host reservation in step 7.
        self.release_slot();
        if !self.buf.is_empty() {
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
            slot: None,
            stream: self.stream.clone(),
            chunk_byte_size: 0,
            n_chunks: 0,
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

/// Fixed byte-size of one `TokenSlice` header: offset(2) + len(2) + rope(4) +
/// kvheads_ptr(8). The KvHead record lives out-of-line (see [`record_bytes`]).
pub(crate) const SLICE_HEADER_BYTES: usize = 16;

/// Byte-size of one chunk's out-of-line `KvHead[n_kv_head]` record.
pub(crate) fn record_bytes(n_kv_head: usize, head_dim: usize) -> usize {
    n_kv_head * kv_head_serialized_size(head_dim)
}

/// Per-chunk footprint in the `GpuChunks` buffer: the 16-byte slice header plus
/// its out-of-line record. The buffer is laid out in two sections —
/// `[ slice_header × n_chunks | record × n_chunks ]` — so slice headers stay a
/// contiguous 16-byte-stride array (what the kernel's `get_slice` indexes) while
/// each header's `kvheads_ptr` points into the records section.
pub(crate) fn token_slice_serialized_size(n_kv_head: usize, head_dim: usize) -> usize {
    SLICE_HEADER_BYTES + record_bytes(n_kv_head, head_dim)
}

/// Write a 16-byte slice header (`offset`, `len`, `rope`, `kvheads_ptr`).
pub(crate) fn write_slice_header(
    dst: &mut [u8],
    offset: u16,
    len: u16,
    rope: u32,
    kvheads_ptr: u64,
) {
    dst[0..2].copy_from_slice(&offset.to_le_bytes());
    dst[2..4].copy_from_slice(&len.to_le_bytes());
    dst[4..8].copy_from_slice(&rope.to_le_bytes());
    dst[8..16].copy_from_slice(&kvheads_ptr.to_le_bytes());
}

/// Serialize one chunk's `KvHead[n_kv_head]` record into `dst` (length
/// [`record_bytes`]). Delegates to the shared record serializer so the decode
/// buffer and the resident meta-pool produce byte-identical records.
fn write_record_for_chunk(
    dst: &mut [u8],
    chunk: &ChunkWindow,
    n_kv_head: usize,
    head_dim: usize,
    arena_info: &[ResolvedArenaInfo],
) {
    super::meta_pool::serialize_kv_heads(
        dst,
        &super::meta_pool::ChunkRecordSrc {
            gids: &chunk.gids,
            k_pal: chunk.k_pal.as_slice(),
            v_pal: chunk.v_pal.as_slice(),
            k_scale: chunk.k_scale.as_slice(),
            v_scale: chunk.v_scale.as_slice(),
            k_fmt: chunk.k_fmt.as_slice(),
            v_fmt: chunk.v_fmt.as_slice(),
        },
        n_kv_head,
        head_dim,
        arena_info,
    );
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

// Unit tests live in tests/gpu_chunks_tests.rs (compiled under #[cfg(feature = "cuda")]).

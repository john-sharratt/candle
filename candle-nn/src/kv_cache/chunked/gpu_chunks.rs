//! GPU-side slot-state cache for a single sequence.
//!
//! Holds a host copy of the serialised slot-state and a matching device slot,
//! laid out in two sections — `[ slice headers (16 B) × capacity | KvHead records ]`
//! — so the slice headers stay a contiguous 16-byte-stride array (what the
//! kernel's `get_slice` indexes) while each header's `kvheads_ptr` points into the
//! records section. The records begin past room for `capacity` headers, not past
//! the live ones, so a chunk appended within the capacity moves no record and the
//! entries before it stay valid as they stand: a sequence crossing a 32-token
//! boundary serialises its new tail ([`GpuChunksGuard::extend_decode`]), not its
//! whole history. Dirty chunk indices accumulate on the [`GpuChunksGuard`]; on drop
//! the guard coalesces adjacent indices into runs of two ranges each (the headers
//! range + the records range). Inside a slot sync, which has a
//! [`super::slot_upload_batch`] open, the ranges join the batch and every slot the
//! sync touched is copied by one scatter; anywhere else they are packed into a
//! pinned staging buffer and uploaded from there.
//!
//! **No copy ever reads the host copy.** An upload runs when the stream reaches
//! it, not when it is issued, so a copy sourced from the live bytes would carry
//! whatever they hold by then — a commit made while an upload is still queued
//! would hand the kernels ahead of it lengths from their future. Uploads read a
//! staging buffer instead, so the host copy is rewritten the moment host state
//! changes, and the host never waits for the GPU to reach an upload.

use super::head_gids::HeadGids;
use super::meta_pool::MetaGid;
use super::slot_state_arena::{self, SlotStateSlot};
use super::slot_upload_batch;
use super::types::ChunkWindow;
use crate::kv_cache::arena_table::ResolvedArenaInfo;
#[cfg(test)]
use crate::kv_cache::arena_table::N_PALETTE;
use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::cuda_backend::cudarc::driver::CudaEvent;
use candle::cuda_backend::WrapErr;
use candle::quantized::pinned_staging::{give_recycled_wc, take_recycled_wc, PinnedBuf};
use candle::CudaDevice;
use std::sync::Arc;

/// Everything one serialised chunk's headers dereference, held alive for the life of
/// any launch that reads them.
///
/// Both halves are addresses in a slice header: the record's own (`kvheads_ptr`) and,
/// through it, each band's. So both handles have to be held — see [`GpuChunks::pins`],
/// where holding only the first cost a night.
#[derive(Clone, Debug)]
pub struct ChunkPin {
    /// The chunk's bands.
    gids: HeadGids,
    /// The chunk's resident record, when it has one. `None` for a live writer window,
    /// whose heads are serialised inline and name no record slot.
    meta: Option<MetaGid>,
}

impl ChunkPin {
    fn of(chunk: &ChunkWindow) -> Self {
        Self {
            gids: chunk.gids.clone(),
            meta: chunk.meta.clone(),
        }
    }

    /// `HeadGids::alloc_id` of the bands this pin holds. Stable for as long as the pin
    /// lives, because the pin is what keeps the allocation alive.
    pub(crate) fn bands_alloc_id(&self) -> usize {
        self.gids.alloc_id()
    }

    /// Raw id of the record slot this pin holds, if the chunk has a record. Unique for
    /// as long as the pin lives, for the same reason.
    pub(crate) fn record_raw(&self) -> Option<i64> {
        self.meta.as_ref().map(MetaGid::raw)
    }

    /// Whether this pin already holds exactly what `chunk`'s header will name — its band
    /// allocation and its record slot — the cheap check `update_chunk` uses before
    /// replacing a pin.
    fn describes(&self, chunk: &ChunkWindow) -> bool {
        self.gids.is_same_alloc(&chunk.gids)
            && self.record_raw() == chunk.meta.as_ref().map(MetaGid::raw)
    }
}

/// Cached host + device-side serialised slot-state for one sequence.
pub(crate) struct GpuChunks {
    /// The serialised `TokenSlice` bytes — the authoritative copy every upload
    /// is taken from. Ordinary cacheable memory, because nothing reads it
    /// asynchronously: uploads go through [`Self::staging`]. Grow-only; its
    /// length is a capacity, and [`Self::n_chunks`] is the live entry count.
    host: Vec<u8>,
    /// Device-side backing: a slot from the region tier's doubling class
    /// family (`slot_state_arena`), **not** an allocation. `None` until the
    /// first non-empty update. Its capacity is a class width, so it is
    /// generally larger than the live entries; the kernel's walk is bounded by
    /// `n_chunks`, never by the slot.
    slot: Option<SlotStateSlot>,
    /// The device the slot lives on. `None` for CPU-backed tests even when the
    /// crate is compiled with the CUDA feature enabled.
    ///
    /// A device, not a stream, because this buffer is brought up to date from
    /// inside recorded waves (`docs/decode_graphs.md`): an upload is recorded
    /// into the wave's segment where it can be, and every other device call —
    /// a pinned-staging copy, a fence, a slab claim — runs behind the launches
    /// recorded so far, inside [`CudaDevice::pause_capture`], on the compute
    /// stream. A stream handle taken while recording would be the capture
    /// stream.
    device: Option<CudaDevice>,
    /// Byte size of one serialised chunk entry (e.g. one `TokenSliceHost`).
    /// Zero until the first call to [`GpuChunksGuard::update`].
    chunk_byte_size: usize,
    /// Cached copy of the records section in a stager generation. The records
    /// (per-chunk `KvHead` band pointers) are immutable once a chunk exists, so
    /// `snapshot_into_generation` copies them once per (epoch, chunk-count) and
    /// reuses the copy across the many per-token header snapshots that share the
    /// same chunks. Invalidated by epoch change (arena reset) or chunk append.
    gen_records: Option<GenRecordsCache>,
    /// Entries currently live.
    ///
    /// Kept explicitly because both buffers are **capacities**: the host copy
    /// is grow-only and the device slot is a class width, so neither length
    /// divides down to the entry count.
    n_chunks: usize,
    /// Entries the current layout has room for: the records section begins at
    /// `capacity × SLICE_HEADER_BYTES`. Fixed between full serialisations, so
    /// an entry appended below it is written where the layout already expects
    /// it. The slot class's width in entries, set by `resize`.
    capacity: usize,
    /// The writer entry of the most recent serialisation — the one entry whose
    /// length is the offset-derived write length rather than its chunk's
    /// stored usage, and whose device copy the decode kernel advances in
    /// place. An extend re-serialises from here at the latest.
    write_idx: usize,
    /// Chunks were appended to the host list since the last serialisation, so
    /// the `n_chunks` entries held are a strict prefix of it and the next sync
    /// extends rather than rebuilds. Set by [`Self::mark_appended`], cleared by
    /// any serialisation and by `clear`.
    appended: bool,
    /// The chunks this serialisation REFERENCES, held alive by their gids.
    ///
    /// A serialised slot-state is a page table: its records name arena base
    /// pointers, so the arenas behind them must outlive every launch that reads
    /// it. Nothing about the host chunk table guarantees that — the persistence
    /// thread's quantize-swap rewrites a slot's sealed chunks mid-wave, and once
    /// the last gid for a float source arena drops, the arena is released and
    /// re-tenanted under an in-flight kernel. So the serialisation carries its
    /// own pins, taken in the same pass that wrote the bytes.
    ///
    /// **The record handles are pinned for the same reason and it is not optional.**
    /// A header's `kvheads_ptr` is a *record's* address, and a record slot is an
    /// ordinary arena slot: a compaction mints a fresh record per relocated chunk —
    /// 800–1,400 a pass — which frees the old one and reissues its slot inside the
    /// same sweep. A launch holding only the band gids would keep the bands alive
    /// while the record its headers point at was handed to another chunk.
    ///
    /// Shared, so a launch holds the set it actually read with ONE refcount
    /// bump rather than a clone per chunk: `clear` installs a fresh vector and
    /// a launch still holding the old one keeps exactly the arenas its headers
    /// point at, for as long as it needs them.
    pins: Arc<Vec<ChunkPin>>,
    /// The pinned buffers an upload is copied through — two, so one can carry
    /// a copy still waiting in the stream while the next upload fills the
    /// other. See [`choose_staging`] for which one an upload takes.
    staging: [Staging; 2],
    /// The `staging` entry the most recent upload went through. Copies run in
    /// stream order, so its event retiring means every copy this object has
    /// issued has retired — the fence before the device slot changes hands.
    last_upload: Option<usize>,
}

/// One pinned staging buffer and the event recorded after the copies out of it.
struct Staging {
    /// Write-combined pinned memory — the CPU only writes it, the copy engine
    /// only reads it. Grow-only; replaced only once no copy reads it.
    buf: PinnedBuf,
    /// Recorded after the most recent copies out of [`Self::buf`]; `None` until
    /// the first. Created once and re-recorded: `cuEventRecord` overwrites the
    /// previous recording, and a create/destroy pair per upload would put two
    /// driver object lifecycles on every 32-token boundary of every sequence.
    done: Option<CudaEvent>,
}

impl Staging {
    fn empty() -> Self {
        Self {
            buf: Self::no_buf(),
            done: None,
        }
    }

    /// The zero-length placeholder a `Staging` holds before its first upload.
    /// `alloc_owned(0)` returns a zero-len `Bump` variant — no CUDA call.
    fn no_buf() -> PinnedBuf {
        PinnedBuf::alloc_owned(0).expect("zero-len PinnedBuf alloc cannot fail")
    }

    /// Take the pinned buffer out, leaving the placeholder.
    ///
    /// The caller must have fenced this staging's copies first: the recycler hands
    /// the same pages to the next taker, so a copy still reading them would read
    /// another owner's bytes.
    fn take_buf(&mut self) -> PinnedBuf {
        std::mem::replace(&mut self.buf, Self::no_buf())
    }

    /// Whether a copy out of this buffer may still be waiting in the stream.
    /// A non-blocking query — the host never waits here.
    fn busy(&self) -> bool {
        self.done.as_ref().is_some_and(|event| !event.is_complete())
    }
}

/// Which staging buffer the next upload goes through, and whether it must first
/// wait for that buffer's previous copy: `busy[i]` is whether a queued copy may
/// still read buffer `i`, and `last` is the buffer the previous upload used.
///
/// The previous upload's buffer when it is free — so an upload that never
/// collides with a queued copy (every decode-path upload) keeps reusing one
/// buffer, and the second is never grown. Otherwise the other one; if that is
/// busy too it is the older of the two, and the upload waits for it — the only
/// wait, and it takes two uploads queued behind unfinished work.
fn choose_staging(busy: [bool; 2], last: Option<usize>) -> (usize, bool) {
    match last {
        None => (0, false),
        Some(l) if !busy[l] => (l, false),
        Some(l) => {
            let other = 1 - l;
            (other, busy[other])
        }
    }
}

/// A records-section copy living in a stager generation's arena.
struct GenRecordsCache {
    /// Stager epoch the copy was made under; a mismatch means the arena reset.
    epoch: u64,
    /// Chunk count the copy covers; a mismatch means a chunk was appended.
    n_chunks: usize,
    /// Device pointer to the start of the copied records section.
    dev_ptr: u64,
    /// FNV-1a fingerprint of the records bytes at copy time. Checked (debug only)
    /// on every cache hit to catch a record mutating within one (epoch, n_chunks).
    records_checksum: u64,
}

impl std::fmt::Debug for GpuChunks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChunks")
            .field("host_capacity", &self.host.len())
            .field("chunk_byte_size", &self.chunk_byte_size)
            .field("n_chunks", &self.n_chunks)
            .finish()
    }
}

impl GpuChunks {
    pub(crate) fn new(device: Option<CudaDevice>) -> Self {
        Self {
            host: Vec::new(),
            slot: None,
            device,
            chunk_byte_size: 0,
            gen_records: None,
            n_chunks: 0,
            capacity: 0,
            write_idx: 0,
            appended: false,
            pins: Arc::new(Vec::new()),
            staging: [Staging::empty(), Staging::empty()],
            last_upload: None,
        }
    }

    /// Record that chunks were appended to the host list this buffer
    /// serialises. A buffer holding nothing has nothing to extend — its next
    /// sync rebuilds it — so it is left as it is.
    pub(crate) fn mark_appended(&mut self) {
        if self.n_chunks != 0 {
            self.appended = true;
        }
    }

    /// Whether chunks were appended since the last serialisation
    /// ([`Self::mark_appended`]).
    pub(crate) fn appended(&self) -> bool {
        self.appended
    }

    /// The writer entry of the most recent serialisation.
    pub(crate) fn write_idx(&self) -> usize {
        self.write_idx
    }

    /// The chunks this serialisation references, for a consumer to hold across
    /// its launch. One refcount bump — see [`Self::pins`].
    pub(crate) fn pins(&self) -> Arc<Vec<ChunkPin>> {
        Arc::clone(&self.pins)
    }

    /// Retire every copy this object has issued — call before the device slot
    /// changes hands or a staging buffer is freed.
    ///
    /// A copy still in flight toward a slot that has gone back to the
    /// `slot_state_arena` free list lands in whatever sequence claims it next:
    /// that sequence's attention kernel then dereferences the `kvheads_ptr`
    /// fields of this sequence's records — not its pointers, and not
    /// necessarily pointers at all — a device-side out-of-range access on every
    /// warp at once, poisoning the context at whatever call synchronises next.
    /// Stream ordering covers copies enqueued *before* the handover, not one
    /// already queued toward the slot; that one has to be retired.
    ///
    /// Waits on the most recent upload's event: copies run in stream order, so
    /// it retiring retires them all, and the kernels enqueued since keep
    /// running. Rewriting the host copy needs no fence at all — no copy reads
    /// it (see [`Self::staging`]).
    ///
    /// Only staged copies are tracked. A copy recorded into a wave's segment
    /// runs in issue order with everything after it on the compute stream, so
    /// the next tenant's own upload lands after it.
    fn fence_uploads(&mut self) {
        let Some(event) = self.last_upload.and_then(|i| self.staging[i].done.as_ref()) else {
            return;
        };
        // A query or a wait on an event is host protocol, refused on a thread
        // recording a wave — the query as much as the wait: it invalidates the
        // capture. Once fenced, `last_upload` clears, so a buffer pays this
        // pause once per staged upload, and one whose uploads are all recorded
        // never.
        let _paused = match self.device.as_ref().map(CudaDevice::pause_capture) {
            Some(Ok(paused)) => paused,
            Some(Err(e)) => {
                log::error!("GpuChunks: could not pause the wave capture to fence: {e:?}");
                None
            }
            None => None,
        };
        if event.is_complete() {
            self.last_upload = None;
            return;
        }
        if let Err(e) = event.synchronize().w() {
            log::warn!("GpuChunks: fencing a pending slot-state upload failed: {e:?}");
        }
        self.last_upload = None;
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
    /// **The caller must have fenced any pending upload first**
    /// ([`Self::fence_uploads`]). Stream ordering covers the copies that
    /// were *enqueued before* this slot changes hands, but it says nothing about
    /// a copy already in flight whose DESTINATION is this slot. Handing that
    /// slot to another sequence lets the pending transfer land in a buffer it
    /// does not own.
    ///
    /// Copies into it still pending in this thread's open upload batch are
    /// dropped: they were this sequence's bytes, and the slot's next tenant
    /// writes its own.
    fn release_slot(&mut self) {
        if let (Some(slot), Some(device)) = (self.slot.take(), self.device.as_ref()) {
            slot_upload_batch::forget(slot.ptr);
            slot_state_arena::release(device, slot);
        }
    }

    /// Number of serialised chunk entries currently in the buffer.
    pub(crate) fn n_chunks(&self) -> usize {
        self.n_chunks
    }

    /// Copy the current serialised slot-state into immutable buffers owned by
    /// the pinned-stager `generation`, returning the copy's device pointer (a
    /// contiguous `TokenSlice` header array the kernel's `get_slice` indexes).
    ///
    /// The live slot moves whenever a chunk is appended past its class
    /// (`rebuild_decode` → `resize` promotes to a wider `slot_state_arena` slot
    /// and returns the old one to its free list). A caller that captures
    /// `raw_device_ptr()` and defers its kernel launch — the wave prefill builds
    /// every per-token metadata snapshot up front, then runs the layer loop —
    /// would read a slot another sequence may since have claimed once a later
    /// snapshot crosses a chunk boundary. Copying into the generation (whose arena lives for the
    /// whole forward) makes the pointer stable and pins that token's exact slice
    /// content (per-token write-chunk length included).
    ///
    /// The two sections are handled differently to keep the cost O(prefill_len)
    /// rather than O(prefill_len²): the `KvHead` **records** are immutable once a
    /// chunk exists, so they are copied once per (epoch, chunk-count) and cached
    /// ([`GenRecordsCache`]) for reuse across the many per-token snapshots that
    /// share the same chunks; only the small 16-B **headers** — which carry the
    /// per-token write-chunk length — are copied every call. Float chunks inline
    /// their record, so each header's `kvheads_ptr` points into the source
    /// buffer and is rebased onto the cached records copy; resident (meta-pool)
    /// records point at the arena and are left untouched.
    ///
    /// `write_idx`/`write_len` override the write chunk's serialised length in
    /// the copied header. The live buffer only re-serialises the write length at
    /// a chunk boundary (the per-token advance normally rides the on-device
    /// `commit_decode_write_len_kernel`, which increments the *shared* buffer).
    /// Because each snapshot is a private copy, that on-device increment can no
    /// longer carry from one token to the next, so the caller supplies the
    /// sequence-offset-derived length for this snapshot's token directly.
    pub(crate) fn snapshot_into_generation(
        &mut self,
        generation: &candle::quantized::pinned_staging::Generation,
        write_idx: usize,
        write_len: u16,
    ) -> candle::Result<u64> {
        let n = self.n_chunks();
        if n == 0 {
            return Ok(0);
        }
        // Cacheable memory, read in place by every pass below (headers copy,
        // records copy, pointer rebase, debug checksum).
        let host: &[u8] = &self.host;
        // `write_idx` is `decode_write_chunk_idx()` (always `< host chunk count`);
        // after `sync_decode_gpu_chunks` the serialised buffer holds exactly that
        // many chunks, so `write_idx < n` is an invariant. Fail loudly rather
        // than silently skip the write-length patch (which would ship a stale
        // length and silently corrupt `q_pos`/the window walk).
        if write_idx >= n {
            candle::bail!(
                "snapshot_into_generation: write_idx {write_idx} >= n_chunks {n} (buffer/host chunk-count desync)"
            );
        }
        let headers_len = n * SLICE_HEADER_BYTES;
        // The live records: `n` of them, from where the layout places the
        // section — past room for `capacity` headers.
        let records_off = self.capacity * SLICE_HEADER_BYTES;
        let records_len = n * (self.chunk_byte_size - SLICE_HEADER_BYTES);
        let d_old = self.raw_device_ptr();
        let records_base_old = d_old + records_off as u64;
        let records_end_old = records_base_old + records_len as u64;

        // Records section: reuse the cached generation copy when it still matches
        // this generation's epoch and chunk count; otherwise copy it in. The
        // reuse is sound because a chunk's `KvHead` record (band pointers,
        // palette, outer scale) is fixed once the chunk exists and the host
        // buffer only re-serialises records at a chunk-boundary rebuild (which
        // changes `n_chunks` → cache miss). The debug checksum asserts that
        // invariant so a backing that mutates records within a chunk-count (e.g.
        // an adaptive-quant arena re-scaling the write chunk as it fills) trips
        // in tests instead of silently serving stale scales.
        let epoch = generation.epoch();
        let records_ptr = if records_len == 0 {
            0
        } else {
            let records_src = &host[records_off..records_off + records_len];
            let hit = self
                .gen_records
                .as_ref()
                .filter(|c| c.epoch == epoch && c.n_chunks == n);
            match hit {
                Some(c) => {
                    debug_assert_eq!(
                        records_checksum(records_src),
                        c.records_checksum,
                        "stale records cache reused: chunk records changed within one (epoch, n_chunks)"
                    );
                    c.dev_ptr
                }
                None => {
                    let ptr = copy_into_generation(generation, records_src)?;
                    self.gen_records = Some(GenRecordsCache {
                        epoch,
                        n_chunks: n,
                        dev_ptr: ptr,
                        // Only the debug staleness assert reads this; skip the
                        // byte-loop fingerprint in release.
                        records_checksum: if cfg!(debug_assertions) {
                            records_checksum(records_src)
                        } else {
                            0
                        },
                    });
                    ptr
                }
            }
        };

        // Header section: copied every call (carries the per-token write length),
        // with each inline `kvheads_ptr` rebased onto the cached records copy.
        let mut pinned = generation.alloc(headers_len)?;
        if !pinned.is_bump() {
            candle::bail!(
                "snapshot_into_generation: expected a bump-allocated staging buffer for {headers_len} bytes"
            );
        }
        let host_ptr = pinned.as_mut_slice().as_mut_ptr();
        pinned.as_mut_slice().copy_from_slice(&host[..headers_len]);
        let gpu = generation.submit(pinned)?;
        let headers_ptr = gpu.dev_ptr();

        // Rebase inline `kvheads_ptr` and patch the write-chunk length. The
        // original pointers are read from `host` (the source), NOT from `dst`:
        // `dst` is write-combined, and reading back bytes just stored to WC
        // memory can return stale data before the WC buffer drains.
        // SAFETY: `host_ptr` is the device-mapped bump slice we just filled; it
        // stays valid for the generation's lifetime and no kernel has read it
        // yet (build runs before the layer loop launches).
        unsafe {
            let dst = std::slice::from_raw_parts_mut(host_ptr, headers_len);
            rebase_and_patch_headers(
                dst,
                &host[..headers_len],
                n,
                records_base_old,
                records_end_old,
                records_ptr,
                write_idx,
                write_len,
            );
        }
        Ok(headers_ptr)
    }
}

/// Rebase inline `kvheads_ptr`s and patch the write-chunk length in a copied
/// `TokenSlice` header array. Pure byte arithmetic (no device, no unsafe) so it
/// is unit-testable against raw expected bytes.
///
/// `dst` is the freshly-copied header array to fix up; `src` is the authoritative
/// source header bytes to read the *original* pointers from (reading from `src`
/// rather than `dst` avoids a write-combined read-after-write hazard). For each
/// chunk, if its `kvheads_ptr` (u64 at header offset 8) falls in the source
/// records section `[records_base_old, records_end_old)` it is inline and is
/// rebased onto `records_ptr`; otherwise it is a resident/arena pointer and left
/// as copied. Finally the write chunk's `len` (u16 at header offset 2) is set to
/// `write_len`. Caller guarantees `write_idx < n` and `dst.len() == src.len() == n * 16`.
fn rebase_and_patch_headers(
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    records_base_old: u64,
    records_end_old: u64,
    records_ptr: u64,
    write_idx: usize,
    write_len: u16,
) {
    for i in 0..n {
        let off = i * SLICE_HEADER_BYTES + 8;
        let p = u64::from_le_bytes(src[off..off + 8].try_into().unwrap());
        if p >= records_base_old && p < records_end_old {
            let np = records_ptr + (p - records_base_old);
            dst[off..off + 8].copy_from_slice(&np.to_le_bytes());
        }
    }
    let loff = write_idx * SLICE_HEADER_BYTES + 2;
    dst[loff..loff + 2].copy_from_slice(&write_len.to_le_bytes());
}

/// FNV-1a checksum of a byte slice — a cheap fingerprint for the records-cache
/// staleness debug assertion (release builds never call it).
fn records_checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Copy `bytes` into a fresh device-mapped bump buffer in `generation` and
/// return its device pointer. The copied bytes are read verbatim by the GPU.
fn copy_into_generation(
    generation: &candle::quantized::pinned_staging::Generation,
    bytes: &[u8],
) -> candle::Result<u64> {
    let mut pinned = generation.alloc(bytes.len())?;
    if !pinned.is_bump() {
        candle::bail!(
            "copy_into_generation: expected a bump-allocated staging buffer for {} bytes",
            bytes.len()
        );
    }
    pinned.as_mut_slice().copy_from_slice(bytes);
    let gpu = generation.submit(pinned)?;
    Ok(gpu.dev_ptr())
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
    /// Ensure the host copy and the device slot can hold `n_chunks` entries of
    /// `chunk_byte_size` bytes, and lay the buffer out for the slot class's
    /// width in entries ([`GpuChunks::capacity`]).
    ///
    /// **Content is not preserved, and does not need to be.** The records
    /// section begins past room for `capacity` headers, so it moves whenever
    /// the class does, and the only caller ([`Self::rebuild_decode`]) rewrites
    /// every entry immediately after.
    ///
    /// Growth on the device is a **promotion**: claim a wider slot from the
    /// next class up and release the old one — no allocator call and no copy.
    /// The host copy grows along the same doubling ladder, so a sequence
    /// crossing a 32-token boundary does not reallocate it per layer; no copy
    /// reads it, so growing it needs no fence.
    fn resize(&mut self, n_chunks: usize, chunk_byte_size: usize) -> candle::Result<()> {
        let byte_len = n_chunks
            .checked_mul(chunk_byte_size)
            .expect("overflow in resize");
        self.inner.chunk_byte_size = chunk_byte_size;
        self.inner.n_chunks = n_chunks;

        if byte_len == 0 {
            self.inner.capacity = 0;
            // The slot changes hands: retire any copy still writing it.
            self.inner.fence_uploads();
            self.inner.release_slot();
            return Ok(());
        }

        let want = slot_state_arena::class_bytes_for(byte_len)?;
        // Every entry takes `chunk_byte_size` across the two sections, so the
        // class holds `want / chunk_byte_size` of them — at least `n_chunks`.
        self.inner.capacity = want / chunk_byte_size;
        if self.inner.host.len() < want {
            self.inner.host.resize(want, 0);
        }

        // Device: keep the slot if it still fits, else promote.
        let Some(device) = self.inner.device.clone() else {
            self.inner.slot = None;
            return Ok(());
        };
        let fits = self
            .inner
            .slot
            .as_ref()
            .is_some_and(|s| s.capacity() >= byte_len);
        if !fits {
            let next = slot_state_arena::claim(&device, byte_len)?;
            // The old slot changes hands: retire any copy still writing it.
            self.inner.fence_uploads();
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
        // Rewritten in place with no fence: no copy reads the host copy (see
        // `GpuChunks::staging`), so an upload still queued keeps the bytes it
        // was issued with however this changes them.
        let n_palette = chunk_n_palette(chunk, n_kv_head);
        let chunk_byte_size = token_slice_serialized_size(n_kv_head, head_dim, n_palette);
        // The LIVE entry count, not one derived from the buffer length: both
        // buffers are capacities — the host side is grow-only and the device
        // slot is a class width.
        let current_n = if self.inner.chunk_byte_size == chunk_byte_size && chunk_byte_size > 0 {
            self.inner.n_chunks
        } else {
            0
        };
        if chunk_idx >= current_n {
            // A chunk appended since the last serialisation: the pending extend
            // serialises it from the host state, this change included.
            if self.inner.appended && current_n > 0 {
                return Ok(());
            }
            candle::bail!(
                "update_chunk: index {chunk_idx} out of range (buf holds {current_n} chunks)"
            );
        }
        self.write_entry(
            chunk_idx,
            chunk,
            chunk.usage as u16,
            rope_base,
            n_kv_head,
            head_dim,
            n_palette,
            arena_info,
        );
        // Keep the pin describing what the bytes now reference. The gids are the
        // same object on every writer-length patch — the overwhelmingly common
        // call — so the compare short-circuits and the shared vector is left
        // alone; only a genuine re-gid pays `make_mut`'s copy, and only while a
        // launch is still holding the old set.
        //
        // The pins are written by the same pass that writes the bytes, so one
        // per live entry is an invariant, not an expectation. Say so here rather
        // than index into a short vector: an entry with no pin is a serialised
        // record whose arena nothing is holding, which surfaces as a freed arena
        // under a running kernel.
        if self.inner.pins.len() != current_n {
            candle::bail!(
                "update_chunk: {} pins for {current_n} serialised chunks — the \
                 slot-state buffer and its chunk references diverged",
                self.inner.pins.len()
            );
        }
        // Replaced whole when EITHER handle differs, not only the band allocation: the
        // record beside it can change while the gids do not — a compaction that copies a
        // record lower moves the chunk onto the copy and leaves its gids alone — and a pin
        // naming the old record then holds the wrong slot alive while the header just
        // written names the new one, which nothing in this buffer holds.
        if !self.inner.pins[chunk_idx].describes(chunk) {
            Arc::make_mut(&mut self.inner.pins)[chunk_idx] = ChunkPin::of(chunk);
        }
        self.dirty_chunks.push(chunk_idx);
        Ok(())
    }

    /// Clear the buffer and re-serialise all `chunks` from scratch, computing
    /// each chunk's `rope_base` as `base_pos` + the cumulative usage of
    /// preceding chunks (`base_pos` = tokens evicted off the front by the
    /// sliding-window ring; zero for non-windowed slots).
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
        base_pos: u32,
    ) -> candle::Result<()> {
        if chunks.is_empty() {
            self.clear();
            return Ok(());
        }
        self.dirty_chunks.clear();
        let n = chunks.len();
        // Pin what this pass is about to reference. A fresh vector, never a
        // mutation of the old one: a launch reading the previous serialisation
        // still holds that one and must keep ITS arenas, not these.
        self.inner.pins = Arc::new(chunks.iter().map(ChunkPin::of).collect());
        // All chunks in a backing share one band count; derive it from the first.
        let n_palette = chunk_n_palette(&chunks[0], n_kv_head);
        let chunk_byte_size = token_slice_serialized_size(n_kv_head, head_dim, n_palette);
        self.resize(n, chunk_byte_size)?;
        self.inner.appended = false;
        self.inner.write_idx = write_idx;

        // Seeded at `base_pos` (tokens evicted off the front by the sliding
        // window) so the serialised per-chunk `rope_base` stays ABSOLUTE after
        // the ring slides; zero for non-windowed slots → byte-identical.
        let mut rope_base = base_pos;
        for (i, chunk) in chunks.iter().enumerate() {
            let len = decode_entry_len(chunk, i, write_idx, write_len);
            self.write_entry(
                i, chunk, len, rope_base, n_kv_head, head_dim, n_palette, arena_info,
            );
            self.dirty_chunks.push(i);
            rope_base += chunk.usage;
        }
        Ok(())
    }

    /// Bring the buffer up to `chunks` after an append, serialising only the
    /// entries from `start` on, and return how many that was — or nothing at
    /// all, returning `None`, when the current layout cannot hold them, for the
    /// caller to [`Self::rebuild_decode`].
    ///
    /// The buffer's `n_chunks` entries are a prefix of `chunks`
    /// ([`GpuChunks::appended`]), and every entry below `start` is already what
    /// `rebuild_decode` would write now: an entry's bytes depend only on its own
    /// chunk, the usages before it and whether it is the writer, so the caller
    /// passes a `start` no later than the previous writer and the first entry a
    /// pending commit marked stale. The entries from `start` are written by the
    /// same rules as a rebuild, so the result describes the same K/V as the
    /// rebuild's, entry for entry. It is not always the same bytes: an entry
    /// below `start` serialised inline keeps its inline record after the chunk
    /// gains a resident one (a cold load's meta attach), where a rebuild would
    /// point at the resident record — two copies of one record.
    ///
    /// Bounded by the layout's [`GpuChunks::capacity`]: past it the records
    /// would have to move, and a rebuild promotes the slot to the next class —
    /// once per doubling, so a sequence growing to `n` chunks serialises O(n)
    /// entries in all rather than O(n²).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn extend_decode(
        &mut self,
        chunks: &[ChunkWindow],
        n_kv_head: usize,
        head_dim: usize,
        arena_info: &[ResolvedArenaInfo],
        write_len: u16,
        write_idx: usize,
        base_pos: u32,
        start: usize,
    ) -> candle::Result<Option<usize>> {
        let held = self.inner.n_chunks;
        let n = chunks.len();
        if !self.inner.appended || held == 0 || n < held || start > held || n > self.inner.capacity
        {
            return Ok(None);
        }
        let n_palette = chunk_n_palette(&chunks[0], n_kv_head);
        if token_slice_serialized_size(n_kv_head, head_dim, n_palette) != self.inner.chunk_byte_size
        {
            return Ok(None);
        }
        // The device slot is a class width at least `capacity` entries wide
        // (`resize`), so the new entries land inside it.
        self.inner.n_chunks = n;
        self.inner.appended = false;
        self.inner.write_idx = write_idx;
        // The pins follow the entries: unchanged below `start`, re-taken from
        // it. `make_mut` copies the vector when a launch still holds it, so that
        // launch keeps the set its headers were serialised against.
        {
            let pins = Arc::make_mut(&mut self.inner.pins);
            pins.truncate(start);
            pins.extend(chunks[start..].iter().map(ChunkPin::of));
        }
        let mut rope_base = base_pos;
        for c in &chunks[..start] {
            rope_base += c.usage;
        }
        for (i, chunk) in chunks.iter().enumerate().skip(start) {
            let len = decode_entry_len(chunk, i, write_idx, write_len);
            self.write_entry(
                i, chunk, len, rope_base, n_kv_head, head_dim, n_palette, arena_info,
            );
            self.dirty_chunks.push(i);
            rope_base += chunk.usage;
        }
        Ok(Some(n - start))
    }

    /// Write entry `i` — its slice header and, for a chunk with no resident
    /// record, its inline record — into the host copy at the layout
    /// [`GpuChunks::capacity`] fixes.
    #[allow(clippy::too_many_arguments)]
    fn write_entry(
        &mut self,
        i: usize,
        chunk: &ChunkWindow,
        len: u16,
        rope_base: u32,
        n_kv_head: usize,
        head_dim: usize,
        n_palette: usize,
        arena_info: &[ResolvedArenaInfo],
    ) {
        let rec_bytes = record_bytes(n_kv_head, head_dim, n_palette);
        let records_off = self.inner.capacity * SLICE_HEADER_BYTES;
        // Resident chunk: point at its co-resident record in the meta-pool
        // slab and serialise no record here. Otherwise serialise an inline
        // record into this entry's records-section slot and point at it.
        let kvheads_ptr = match chunk.meta.as_ref().map(|m| m.device_addr()) {
            Some(addr) if addr != 0 => addr,
            _ => {
                let r0 = records_off + i * rec_bytes;
                write_record_for_chunk(
                    &mut self.inner.host[r0..r0 + rec_bytes],
                    chunk,
                    n_kv_head,
                    head_dim,
                    n_palette,
                    arena_info,
                );
                self.inner.raw_device_ptr() + r0 as u64
            }
        };
        let s0 = i * SLICE_HEADER_BYTES;
        write_slice_header(
            &mut self.inner.host[s0..s0 + SLICE_HEADER_BYTES],
            chunk.offset,
            len,
            rope_base,
            kvheads_ptr,
        );
    }

    /// Reset the host buffer to empty and cancel any pending dirty uploads.
    ///
    /// Drops the GPU-side allocation and zeroes the chunk count.  Called when
    /// a sequence's slot-state is fully invalidated (e.g. evicted or freed).
    pub(crate) fn clear(&mut self) {
        // Dropping the dirty list cancels uploads not yet ISSUED; it does
        // nothing about one already enqueued, which is still writing this slot.
        // The slot becomes another owner's on the next line, so retire it first.
        self.dirty_chunks.clear();
        self.inner.fence_uploads();
        // No free: the device side goes back to its class free list, and the
        // host copy and staging buffers are kept for the next fill. `clear` is
        // called by every structural mutation but an append, and the fence
        // above fires only when a transfer is genuinely outstanding, which the
        // common `clear` is not.
        self.inner.release_slot();
        self.inner.capacity = 0;
        self.inner.write_idx = 0;
        self.inner.appended = false;
        self.inner.chunk_byte_size = 0;
        // The cached generation records described the old chunk set; drop it so
        // the next snapshot re-copies from the rebuilt buffer.
        self.inner.gen_records = None;
        self.inner.n_chunks = 0;
        // Release this serialisation's claim on its chunks. A launch still
        // reading the bytes holds its own reference to the same vector, so the
        // arenas it addresses stay alive until it drops.
        self.inner.pins = Arc::new(Vec::new());
    }
}

/// Serialised `len` of entry `i`: the offset-derived `write_len` for the writer,
/// the chunk's stored usage for every other entry (trailing empties past the
/// writer included).
fn decode_entry_len(chunk: &ChunkWindow, i: usize, write_idx: usize, write_len: u16) -> u16 {
    if i == write_idx {
        write_len
    } else {
        chunk.usage as u16
    }
}

/// The byte ranges of the slot buffer an upload must carry, for the dirty chunk
/// indices `dirty` (ascending, no duplicates) of a slot holding `n_chunks` live
/// entries in a layout of `capacity`, whose out-of-line records are `rec_bytes`
/// each.
///
/// The buffer has two sections — slice headers `[0 .. capacity*16)`, then
/// records — and a run of adjacent chunk indices is contiguous in *both*, so
/// each run is two ranges: its 16-byte headers and its records. A run covering
/// every live entry carries its headers on through the unused header room to
/// where the records begin, so the whole slot is one copy rather than two —
/// half a full rebuild's driver submissions, which is what a prefill pays for
/// every `(layer, slot)` it brings up to date, where each submission is a
/// separate trip into the driver and a queue already deep with kernels makes
/// each of them wait. The bytes past the live headers are never read: the
/// kernel's walk is bounded by the entry count. Ranges that touch are one
/// range.
fn upload_ranges(
    dirty: &[usize],
    n_chunks: usize,
    capacity: usize,
    rec_bytes: usize,
) -> Vec<std::ops::Range<usize>> {
    let Some(&first) = dirty.first() else {
        return Vec::new();
    };
    let records_off = capacity * SLICE_HEADER_BYTES;
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut push = |range: std::ops::Range<usize>| match ranges.last_mut() {
        Some(last) if last.end == range.start => last.end = range.end,
        _ => ranges.push(range),
    };
    let mut push_run = |start: usize, end: usize| {
        let headers_end = if start == 0 && end == n_chunks && rec_bytes > 0 {
            records_off
        } else {
            end * SLICE_HEADER_BYTES
        };
        push(start * SLICE_HEADER_BYTES..headers_end);
        if rec_bytes > 0 {
            push(records_off + start * rec_bytes..records_off + end * rec_bytes);
        }
    };
    let mut start = first;
    let mut end = start + 1;
    for &idx in &dirty[1..] {
        if idx == end {
            end += 1;
        } else {
            push_run(start, end);
            start = idx;
            end = idx + 1;
        }
    }
    push_run(start, end);
    ranges
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

        let chunk_byte_size = self.inner.chunk_byte_size;
        let n_chunks = self.inner.n_chunks;
        if chunk_byte_size == 0 || n_chunks == 0 {
            return;
        }
        let Some(slot_ptr) = self.inner.slot.as_ref().map(|s| s.ptr) else {
            return;
        };
        let Some(device) = self.inner.device.clone() else {
            return;
        };

        let rec_bytes = chunk_byte_size - SLICE_HEADER_BYTES;
        let all = upload_ranges(&self.dirty_chunks, n_chunks, self.inner.capacity, rec_bytes);

        // **Deferred into the open batch**, where a decode metadata build has
        // one: every slot the build brings up to date is copied by one launch
        // when it flushes, rather than by its own driver calls here.
        if slot_upload_batch::defer(&device, slot_ptr, &all, &self.inner.host) {
            return;
        }

        // **Recorded where the thread is recording a wave**: the bytes go into
        // the wave's ring and the copy into its segment, so bringing the slot up
        // to date does not end the segment — the per-layer header build of a
        // recorded verify and each step of a recorded draft walk run without a
        // cut. Whatever the ring will not take, or every range on a thread that
        // is not recording, goes through the pinned staging below.
        let staged_from = all
            .iter()
            .position(|r| {
                !device.record_upload(slot_ptr + r.start as u64, &self.inner.host[r.clone()])
            })
            .unwrap_or(all.len());
        let ranges = &all[staged_from..];
        if ranges.is_empty() {
            return;
        }
        // Behind every launch this thread has recorded, on the compute stream —
        // and the event waits below are refused on a recording thread.
        let _paused = match device.pause_capture() {
            Ok(paused) => paused,
            Err(e) => {
                log::error!("GpuChunksGuard: could not pause the wave capture to upload: {e:?}");
                None
            }
        };
        let stream = device.compute_stream();
        let total: usize = ranges.iter().map(|r| r.len()).sum();

        let inner = &mut *self.inner;
        let busy = [inner.staging[0].busy(), inner.staging[1].busy()];
        let (k, wait) = choose_staging(busy, inner.last_upload);
        let staging = &mut inner.staging[k];
        if wait {
            if let Some(event) = staging.done.as_ref() {
                if let Err(e) = event.synchronize().w() {
                    log::warn!("GpuChunks: waiting for a staging buffer failed: {e:?}");
                }
            }
        }
        if staging.buf.len() < total {
            // No copy reads this buffer any more — it was free, or waited on
            // above — so replacing it frees nothing a copy still needs.
            // **Through the recycler, because `cuMemHostAlloc` is ~1 ms.** Every
            // `(layer, slot)` owns two of these and a prefill rebuilds each about
            // once, so this branch was a first touch nearly every time it ran:
            // measured on the Flash-Next gate at 16 slots, 416 allocations and
            // 417 ms — 97.6% of the slot-state rebuild. Recycling takes it to 1.6 ms.
            // The sizes come from `class_bytes_for`, so the recycler's exact-length
            // keys are a short ladder that hits.
            match slot_state_arena::class_bytes_for(total).and_then(take_recycled_wc) {
                Ok(buf) => {
                    // The buffer it replaces is idle by the same argument as above,
                    // so it goes back for the next taker rather than to the driver.
                    give_recycled_wc(std::mem::replace(&mut staging.buf, buf));
                }
                Err(e) => {
                    // No pinned memory to stage through. Upload straight from
                    // the host copy and drain the stream before returning, so
                    // the copy has read those bytes before they can change: a
                    // stall, never a slot that is short.
                    log::warn!(
                        "GpuChunks: no {total}-byte staging buffer ({e:?}); uploading from \
                         the host copy and draining the stream"
                    );
                    for range in ranges {
                        // SAFETY: `range` lies inside the live entry count,
                        // which `resize` sized the slot to hold; the stream is
                        // drained below, before the host copy can change.
                        let res = unsafe {
                            memcpy_htod_async(
                                slot_ptr + range.start as u64,
                                &inner.host[range.clone()],
                                stream.cu_stream(),
                            )
                        };
                        if let Err(e) = res {
                            log::warn!("GpuChunksGuard: memcpy_htod {range:?} error: {e:?}");
                        }
                    }
                    if let Err(e) = stream.synchronize() {
                        log::warn!("GpuChunks: stream drain failed: {e:?}");
                    }
                    return;
                }
            }
        }

        // Pack the ranges and upload them out of the staging buffer. The bytes
        // a copy carries are the ones packed here, whatever the host copy
        // becomes before the stream reaches it.
        let mut cursor = 0usize;
        for range in ranges {
            let n = range.len();
            staging.buf.as_mut_slice()[cursor..cursor + n]
                .copy_from_slice(&inner.host[range.clone()]);
            // SAFETY: `range` lies inside the live entry count, which `resize`
            // sized the slot to hold; the source is this staging buffer, which
            // nothing rewrites or frees until the event recorded below has
            // retired; and the copy is ordered on the stream every reader of
            // this slot uses.
            let res = unsafe {
                memcpy_htod_async(
                    slot_ptr + range.start as u64,
                    &staging.buf.as_slice()[cursor..cursor + n],
                    stream.cu_stream(),
                )
            };
            if let Err(e) = res {
                log::warn!("GpuChunksGuard: memcpy_htod {range:?} error: {e:?}");
            }
            cursor += n;
        }

        // Record the point these copies finish at, so whoever next reuses the
        // staging buffer or hands the slot back retires exactly them rather
        // than draining the stream.
        if staging.done.is_none() {
            match stream.context().new_event(None) {
                Ok(event) => staging.done = Some(event),
                Err(e) => log::warn!("GpuChunks: could not create an upload event: {e:?}"),
            }
        }
        let recorded = staging
            .done
            .as_ref()
            .is_some_and(|event| event.record(&stream).is_ok());
        inner.last_upload = Some(k);
        if !recorded {
            // No event means no cheap fence. Fall back to the correct-but-blunt
            // ordering rather than leaving the transfer unguarded: a stall is
            // recoverable, a slot handed away mid-copy is not. The drain settles
            // the debt outright, so a stale recording left on the event cannot
            // make a later fence return early over a transfer that is still
            // live — there is none.
            log::warn!("GpuChunks: no slot-state upload event; draining the stream instead");
            if let Err(e) = stream.synchronize() {
                log::warn!("GpuChunks: fallback stream drain failed: {e:?}");
            }
        }
    }
}

impl Drop for GpuChunks {
    fn drop(&mut self) {
        // **Fence BEFORE releasing, not after.** A pending `memcpy_htod_async`
        // is still using storage that stops being ours right here: the staging
        // buffer it reads (`cuMemFreeHost` unpins its pages when the field
        // drops) and the slot it writes (it goes to a free list another sequence
        // claims from immediately). Stream ordering covers the copies enqueued
        // before the handover, not one already in flight toward the slot.
        self.fence_uploads();
        // The fence retired every copy out of these buffers, so their pages are
        // idle and the next `(layer, slot)` needing this size can have them
        // instead of paying `cuMemHostAlloc` again. Returned here and not in
        // `clear`, which keeps the buffers on purpose for the next fill.
        for s in self.staging.iter_mut() {
            give_recycled_wc(s.take_buf());
        }
        self.release_slot();
    }
}

impl Clone for GpuChunks {
    fn clone(&self) -> Self {
        // Pinned buffers and GPU allocations are not cloneable; forked
        // sequences start fresh on the same stream.
        Self {
            host: Vec::new(),
            slot: None,
            device: self.device.clone(),
            chunk_byte_size: 0,
            gen_records: None,
            n_chunks: 0,
            capacity: 0,
            write_idx: 0,
            appended: false,
            // Nothing is serialised here yet, so nothing is referenced.
            pins: Arc::new(Vec::new()),
            // Nothing was copied out of this one; it owns no slot and no copy.
            staging: [Staging::empty(), Staging::empty()],
            last_upload: None,
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
pub(crate) fn kv_head_serialized_size(head_dim: usize, n_palette: usize) -> usize {
    (head_dim / 4) * 2 // k_pal + v_pal, each head_dim/4 bytes (2-bit density)
        + n_palette * 26 // k_ptr+v_ptr (8 each) + k_fmt+v_fmt (1 each) + k_scale+v_scale (4 each)
}

/// Fixed byte-size of one `TokenSlice` header: offset(2) + len(2) + rope(4) +
/// kvheads_ptr(8). The KvHead record lives out-of-line (see [`record_bytes`]).
pub(crate) const SLICE_HEADER_BYTES: usize = 16;

/// Byte-size of one chunk's out-of-line `KvHead[n_kv_head]` record.
pub(crate) fn record_bytes(n_kv_head: usize, head_dim: usize, n_palette: usize) -> usize {
    n_kv_head * kv_head_serialized_size(head_dim, n_palette)
}

/// Per-chunk footprint in the `GpuChunks` buffer: the 16-byte slice header plus
/// its out-of-line record. The buffer is laid out in two sections —
/// `[ slice_header × capacity | record × capacity ]` — so slice headers stay a
/// contiguous 16-byte-stride array (what the kernel's `get_slice` indexes) while
/// each header's `kvheads_ptr` points into the records section.
pub(crate) fn token_slice_serialized_size(
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
) -> usize {
    SLICE_HEADER_BYTES + record_bytes(n_kv_head, head_dim, n_palette)
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
/// Per-head band count for this chunk's KvHead record, derived from its GID
/// count (`n_kv_head * n_palette * 2`): 8 for a single-latent chunk, 4 for GQA.
/// Falls back to [`N_PALETTE`] for an empty/placeholder chunk.
fn chunk_n_palette(chunk: &ChunkWindow, n_kv_head: usize) -> usize {
    let g = chunk.gids.len();
    if n_kv_head == 0 || g == 0 {
        crate::kv_cache::arena_table::N_PALETTE
    } else {
        (g / (n_kv_head * 2)).max(1)
    }
}

fn write_record_for_chunk(
    dst: &mut [u8],
    chunk: &ChunkWindow,
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
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
        n_palette,
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

// Integration-style unit tests live in tests/gpu_chunks_tests.rs. The pure
// header-rebase arithmetic is tested here (no device needed).
#[cfg(test)]
mod snapshot_tests {
    use super::{rebase_and_patch_headers, records_checksum, SLICE_HEADER_BYTES};

    /// Serialise one 16-byte `TokenSlice` header: offset u16 | len u16 | rope u32
    /// | kvheads_ptr u64.
    fn header(offset: u16, len: u16, rope: u32, kvheads_ptr: u64) -> [u8; 16] {
        let mut h = [0u8; 16];
        h[0..2].copy_from_slice(&offset.to_le_bytes());
        h[2..4].copy_from_slice(&len.to_le_bytes());
        h[4..8].copy_from_slice(&rope.to_le_bytes());
        h[8..16].copy_from_slice(&kvheads_ptr.to_le_bytes());
        h
    }

    fn read_ptr(bytes: &[u8], chunk: usize) -> u64 {
        let off = chunk * SLICE_HEADER_BYTES + 8;
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }
    fn read_len(bytes: &[u8], chunk: usize) -> u16 {
        let off = chunk * SLICE_HEADER_BYTES + 2;
        u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())
    }

    #[test]
    fn rebase_inline_leaves_resident_and_patches_write_len() {
        // 3 chunks, 8-byte records; source buffer based at 0x10000, headers
        // occupy [0, 48), records [48, 72). Chunk 0/1 inline (kvheads_ptr into
        // the records section), chunk 2 resident (arena pointer, out of range).
        const REC_BYTES: u64 = 8;
        let d_old: u64 = 0x1_0000;
        let n = 3usize;
        let headers_len = n * SLICE_HEADER_BYTES; // 48
        let records_base_old = d_old + headers_len as u64; // 0x10030
        let records_end_old = d_old + headers_len as u64 + n as u64 * REC_BYTES; // 0x10048
        let records_ptr: u64 = 0x2_0000; // relocated records copy base

        let mut src = Vec::new();
        src.extend_from_slice(&header(0, 32, 0, records_base_old)); // chunk 0 inline
        src.extend_from_slice(&header(0, 5, 32, records_base_old + REC_BYTES)); // chunk 1 inline
        src.extend_from_slice(&header(7, 0, 64, 0x9999_9999_9999)); // chunk 2 resident
        let mut dst = src.clone();

        rebase_and_patch_headers(
            &mut dst,
            &src,
            n,
            records_base_old,
            records_end_old,
            records_ptr,
            1,  // write chunk
            18, // new write length
        );

        // Inline pointers rebased onto the relocated records copy.
        assert_eq!(read_ptr(&dst, 0), records_ptr);
        assert_eq!(read_ptr(&dst, 1), records_ptr + REC_BYTES);
        // Resident pointer untouched.
        assert_eq!(read_ptr(&dst, 2), 0x9999_9999_9999);
        // Only the write chunk's length changed.
        assert_eq!(read_len(&dst, 0), 32);
        assert_eq!(read_len(&dst, 1), 18);
        assert_eq!(read_len(&dst, 2), 0);
        // Non-pointer, non-write-len bytes of the write header are preserved
        // (offset stays 0, rope stays 32).
        assert_eq!(&dst[16..18], &0u16.to_le_bytes()); // chunk 1 offset
        assert_eq!(&dst[20..24], &32u32.to_le_bytes()); // chunk 1 rope
    }

    #[test]
    fn edge_case_last_chunk_inline_ptr_is_in_range() {
        // The last inline record starts at records_end_old - REC_BYTES, which
        // must still satisfy the `< records_end_old` bound (no off-by-one).
        const REC_BYTES: u64 = 8;
        let d_old: u64 = 0;
        let n = 1usize;
        let records_base_old = d_old + SLICE_HEADER_BYTES as u64; // 16
        let records_end_old = records_base_old + REC_BYTES; // 24
        let records_ptr: u64 = 0x5000;
        let src = header(0, 1, 0, records_base_old).to_vec();
        let mut dst = src.clone();
        rebase_and_patch_headers(
            &mut dst,
            &src,
            n,
            records_base_old,
            records_end_old,
            records_ptr,
            0,
            1,
        );
        assert_eq!(read_ptr(&dst, 0), records_ptr);
    }

    #[test]
    fn checksum_detects_record_mutation() {
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut b = a;
        b[3] = 0xff;
        assert_eq!(records_checksum(&a), records_checksum(&a));
        assert_ne!(records_checksum(&a), records_checksum(&b));
    }
}

#[cfg(test)]
mod upload_range_tests {
    use super::upload_ranges;

    /// A slot of four chunks with 100-byte records in a layout of four: headers
    /// fill `[0, 64)`, records `[64, 464)`.
    const N: usize = 4;
    const REC: usize = 100;

    /// **A full rebuild is one copy.** Its headers end at 64 where its records
    /// begin, so the two halves are one range — the case a prefill hits for every
    /// `(layer, slot)` it serialises from nothing.
    #[test]
    fn a_whole_slot_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1, 2, 3], N, N, REC), vec![0..464]);
    }

    /// In a layout wider than the live entries the records begin at
    /// `capacity · 16`; a whole-slot run carries its headers on through the
    /// unused header room to them, and is still one copy. Three entries in a
    /// layout of eight: headers `[0, 48)`, room to 128, records `[128, 428)`.
    #[test]
    fn a_whole_slot_in_a_wider_layout_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1, 2], 3, 8, REC), vec![0..428]);
    }

    /// An extend's run — the previous writer and the chunk appended after it —
    /// is its headers and its records, at the layout's record offset. Entries
    /// 2 and 3 of four in a layout of eight: headers `[32, 64)`, records
    /// `[128 + 200, 128 + 400)`.
    #[test]
    fn an_extend_run_is_its_headers_and_its_records() {
        assert_eq!(upload_ranges(&[2, 3], 4, 8, REC), vec![32..64, 328..528]);
    }

    /// A run in the middle of the slot has headers and records far apart: two
    /// ranges, each exactly its run's bytes.
    #[test]
    fn an_interior_run_is_two_ranges() {
        assert_eq!(upload_ranges(&[1, 2], N, N, REC), vec![16..48, 164..364]);
    }

    /// Runs that do not touch stay separate, in run order: each run's headers,
    /// then its records.
    #[test]
    fn separated_runs_stay_separate() {
        assert_eq!(
            upload_ranges(&[0, 2], N, N, REC),
            vec![0..16, 64..164, 32..48, 264..364]
        );
    }

    /// A run that reaches the end of the slot but starts later does not touch
    /// its records: the headers end at 64, the records begin at 64 + 3·100.
    #[test]
    fn a_tail_run_is_two_ranges() {
        assert_eq!(upload_ranges(&[3], N, N, REC), vec![48..64, 364..464]);
    }

    /// A slot of two chunks, whole: headers `[0, 32)` run into records
    /// `[32, 232)`.
    #[test]
    fn a_two_chunk_slot_is_one_range() {
        assert_eq!(upload_ranges(&[0, 1], 2, 2, REC), vec![0..232]);
    }

    /// With no out-of-line record, only the headers travel — the live ones,
    /// never the unused room.
    #[test]
    fn a_slot_without_records_carries_headers_alone() {
        assert_eq!(upload_ranges(&[0, 1, 2, 3], N, N, 0), vec![0..64]);
        assert_eq!(upload_ranges(&[0, 1, 2], 3, 8, 0), vec![0..48]);
        assert_eq!(upload_ranges(&[1], N, N, 0), vec![16..32]);
    }

    /// Nothing dirty, nothing to upload.
    #[test]
    fn no_dirty_chunks_is_no_ranges() {
        assert!(upload_ranges(&[], N, N, REC).is_empty());
    }
}

#[cfg(test)]
mod staging_tests {
    use super::choose_staging;

    /// Every combination of which staging buffer may still be read by a queued
    /// copy and which one the previous upload used: the buffer the next upload
    /// takes, and whether it waits.
    #[test]
    fn an_upload_waits_only_when_both_staging_buffers_are_in_flight() {
        // No upload yet: the first buffer, no wait.
        assert_eq!(choose_staging([false, false], None), (0, false));
        // The previous upload's buffer is free: reuse it, so the second buffer
        // is never grown on a path that never collides.
        assert_eq!(choose_staging([false, false], Some(0)), (0, false));
        assert_eq!(choose_staging([false, true], Some(0)), (0, false));
        assert_eq!(choose_staging([false, false], Some(1)), (1, false));
        assert_eq!(choose_staging([true, false], Some(1)), (1, false));
        // The previous upload's buffer is still read by a queued copy: the
        // other one, which is free — no wait.
        assert_eq!(choose_staging([true, false], Some(0)), (1, false));
        assert_eq!(choose_staging([false, true], Some(1)), (0, false));
        // Both are read by queued copies: the other one is the older of the
        // two, and the upload waits for it.
        assert_eq!(choose_staging([true, true], Some(0)), (1, true));
        assert_eq!(choose_staging([true, true], Some(1)), (0, true));
    }
}

#[cfg(test)]
mod pin_tests {
    use super::*;
    use crate::kv_cache::chunked::gid_pool::ChunkGid;

    fn window(gids: HeadGids, meta: Option<MetaGid>) -> ChunkWindow {
        ChunkWindow {
            gids,
            usage: 32,
            offset: 0,
            k_pal: Arc::new(Vec::new()),
            v_pal: Arc::new(Vec::new()),
            k_scale: Arc::new(Vec::new()),
            v_scale: Arc::new(Vec::new()),
            k_fmt: Arc::new(Vec::new()),
            v_fmt: Arc::new(Vec::new()),
            meta,
        }
    }

    /// **A pin no longer describes a chunk whose record changed under unchanged gids.**
    ///
    /// That is what a compaction's record move produces: the chunk is moved onto a copy of
    /// its record and its bands stay put. A pin compared on bands alone would stay on the
    /// OLD record while `update_chunk` writes a header naming the new one, leaving the
    /// record the header dereferences held by nothing in the buffer.
    #[test]
    fn a_pin_is_stale_when_only_the_record_changed() {
        let bands = HeadGids::uniform(ChunkGid::detached(7), 1);
        let old = MetaGid::from_slot(ChunkGid::detached(100), bands.clone(), 0xA000);
        let new = MetaGid::from_slot(ChunkGid::detached(200), bands.clone(), 0xB000);

        let pin = ChunkPin::of(&window(bands.clone(), Some(old.clone())));
        assert!(pin.describes(&window(bands.clone(), Some(old))));
        assert!(
            !pin.describes(&window(bands.clone(), Some(new))),
            "same bands, different record: the pin must be replaced",
        );
        assert!(
            !pin.describes(&window(bands, None)),
            "a record dropped is a change too",
        );
        assert!(
            !pin.describes(&window(HeadGids::uniform(ChunkGid::detached(8), 1), None)),
            "and so is a band change",
        );
    }
}

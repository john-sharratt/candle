//! The per-sequence slot-state cache: a host copy and a device slot holding one
//! sequence's serialised chunks.
//!
//! The device slot comes from a class family, and it is retired before it
//! changes hands, because a copy still in flight toward a freed slot would land
//! in another sequence's kernel metadata. Uploads go through pinned staging
//! buffers, so the host copy can be rewritten the moment host state changes and
//! the host never waits for the GPU to reach an upload. A buffer held by a
//! recorded wave stays on the wave's stream, which is why it is tied to a device
//! and not to a stream.

use super::*;

/// Cached host + device-side serialised slot-state for one sequence.
pub(crate) struct GpuChunks {
    /// The serialised `TokenSlice` bytes — the authoritative copy every upload
    /// is taken from. Ordinary cacheable memory, because nothing reads it
    /// asynchronously: uploads go through [`Self::staging`]. Grow-only; its
    /// length is a capacity, and [`Self::n_chunks`] is the live entry count.
    pub(super) host: Vec<u8>,
    /// Device-side backing: a slot from the region tier's doubling class
    /// family (`slot_state_arena`), **not** an allocation. `None` until the
    /// first non-empty update. Its capacity is a class width, so it is
    /// generally larger than the live entries; the kernel's walk is bounded by
    /// `n_chunks`, never by the slot.
    pub(super) slot: Option<SlotStateSlot>,
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
    pub(super) device: Option<CudaDevice>,
    /// Byte size of one serialised chunk entry (e.g. one `TokenSliceHost`).
    /// Zero until the first call to [`GpuChunksGuard::update`].
    pub(super) chunk_byte_size: usize,
    /// Cached copy of the records section in a stager generation. The records
    /// (per-chunk `KvHead` band pointers) are immutable once a chunk exists, so
    /// `snapshot_into_generation` copies them once per (epoch, chunk-count) and
    /// reuses the copy across the many per-token header snapshots that share the
    /// same chunks. Invalidated by epoch change (arena reset) or chunk append.
    pub(super) gen_records: Option<GenRecordsCache>,
    /// Entries currently live.
    ///
    /// Kept explicitly because both buffers are **capacities**: the host copy
    /// is grow-only and the device slot is a class width, so neither length
    /// divides down to the entry count.
    pub(super) n_chunks: usize,
    /// Entries the current layout has room for: the records section begins at
    /// `capacity × SLICE_HEADER_BYTES`. Fixed between full serialisations, so
    /// an entry appended below it is written where the layout already expects
    /// it. The slot class's width in entries, set by `resize`.
    pub(super) capacity: usize,
    /// The writer entry of the most recent serialisation — the one entry whose
    /// length is the offset-derived write length rather than its chunk's
    /// stored usage, and whose device copy the decode kernel advances in
    /// place. An extend re-serialises from here at the latest.
    pub(super) write_idx: usize,
    /// Chunks were appended to the host list since the last serialisation, so
    /// the `n_chunks` entries held are a strict prefix of it and the next sync
    /// extends rather than rebuilds. Set by [`Self::mark_appended`], cleared by
    /// any serialisation and by `clear`.
    pub(super) appended: bool,
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
    pub(super) pins: Arc<Vec<ChunkPin>>,
    /// The pinned buffers an upload is copied through — two, so one can carry
    /// a copy still waiting in the stream while the next upload fills the
    /// other. See [`choose_staging`] for which one an upload takes.
    pub(super) staging: [Staging; 2],
    /// The `staging` entry the most recent upload went through. Copies run in
    /// stream order, so its event retiring means every copy this object has
    /// issued has retired — the fence before the device slot changes hands.
    pub(super) last_upload: Option<usize>,
    /// The vacate epoch at which [`Self::slot`] last read "stay"
    /// ([`Self::asked_to_move`]).
    pub(super) stays_at: u64,
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
            stays_at: vacate_epoch(),
        }
    }

    /// Whether the device slot stands on held ground being vacated, so the next
    /// sync should rebuild into a slot claimed afresh — which lands elsewhere —
    /// exactly as a promotion does. One atomic load while the vacate epoch
    /// stands still; asked again on every call until the slot has moved.
    pub(crate) fn asked_to_move(&mut self) -> bool {
        let epoch = vacate_epoch();
        if epoch == self.stays_at {
            return false;
        }
        let moving = match (&self.slot, &self.device) {
            (Some(slot), Some(device)) => slot_state_arena::vacating(device, slot),
            _ => false,
        };
        if !moving {
            self.stays_at = epoch;
        }
        moving
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
    pub(super) fn fence_uploads(&mut self) {
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
    pub(super) fn release_slot(&mut self) {
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
            stays_at: vacate_epoch(),
        }
    }
}

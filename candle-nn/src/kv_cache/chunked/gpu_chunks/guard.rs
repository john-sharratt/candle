//! Mutable access to a sequence's serialised slot state, which marks the chunks
//! it writes and keeps every pin in step with the bytes.
//!
//! Writes are recorded as dirty chunk indices and uploaded as copies of
//! contiguous byte ranges when the guard drops. Entries are rewritten in place
//! because no copy reads the host copy, and a pin is replaced whenever a
//! chunk's band allocation or record changes, so each live entry is held by
//! exactly one pin. Growth on the device is a promotion to a wider slot class,
//! not an allocation and a copy, so a sequence crossing a 32-token boundary
//! does not reallocate its buffers on every layer.

use super::*;

/// Mutable accessor for [`GpuChunks`].
///
/// Chunk writes accumulate in `dirty_chunks`; on drop the guard coalesces
/// adjacent chunk indices into contiguous byte ranges and issues one
/// `stream.memcpy_htod` per contiguous run.
pub(crate) struct GpuChunksGuard<'a> {
    pub(super) inner: &'a mut GpuChunks,
    /// Indices of chunk entries written since this guard was created.
    /// Maintained in ascending order so the drop coalescer can merge
    /// adjacent entries in a single pass.
    pub(super) dirty_chunks: Vec<usize>,
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
    pub(super) fn resize(&mut self, n_chunks: usize, chunk_byte_size: usize) -> candle::Result<()> {
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

        // Device: keep the slot if it still fits and its ground is not being
        // vacated, else claim afresh — a promotion, or a move.
        let Some(device) = self.inner.device.clone() else {
            self.inner.slot = None;
            return Ok(());
        };
        let fits = self
            .inner
            .slot
            .as_ref()
            .is_some_and(|s| s.capacity() >= byte_len)
            && !self.inner.asked_to_move();
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
    /// each chunk's `rope_base` as `base_pos` + the positions the preceding
    /// chunks take (their leads and tokens) + its own lead (`base_pos` = tokens
    /// evicted off the front by the sliding-window ring; zero for non-windowed
    /// slots).
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
            rope_base += chunk.lead;
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
            rope_base += c.span();
        }
        for (i, chunk) in chunks.iter().enumerate().skip(start) {
            let len = decode_entry_len(chunk, i, write_idx, write_len);
            rope_base += chunk.lead;
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
    pub(super) fn write_entry(
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

    /// [`Self::clear`], handing back this serialisation's pins instead of
    /// dropping them here. A freed view's pins are its parent's whole prefix —
    /// thousands of refcount decrements per layer — and the caller drops them
    /// beside the chunks, across threads, rather than one layer at a time under
    /// that layer's state lock.
    pub(crate) fn clear_detached(&mut self) -> Arc<Vec<ChunkPin>> {
        let pins = std::mem::replace(&mut self.inner.pins, Arc::new(Vec::new()));
        self.clear();
        pins
    }
}

/// Serialised `len` of entry `i`: the offset-derived `write_len` for the writer,
/// the chunk's stored usage for every other entry (trailing empties past the
/// writer included).
pub(super) fn decode_entry_len(
    chunk: &ChunkWindow,
    i: usize,
    write_idx: usize,
    write_len: u16,
) -> u16 {
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
pub(super) fn upload_ranges(
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
        let stream = device.cuda_stream();
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

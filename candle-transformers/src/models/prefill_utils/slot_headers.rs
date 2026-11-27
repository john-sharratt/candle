//! Uploads the per-slot headers and position map that a chunked attention launch
//! reads, and keeps those buffers alive until the forward ends.
//!
//! The headers point at the backing's persistent slot-state buffer. The position
//! map does not change from layer to layer, so the first layer builds it and
//! later layers reuse it unless the layout has moved. A windowed creep skews a
//! slot's layout by a trailing empty chunk, and that rebuilds the map rather than
//! failing the launch. A block table holding a stale or freed chunk is refused,
//! and naming the chunk. Any buffer a rebuild replaces is carried to the end of
//! the forward, because an earlier layer's kernel may still be reading it.

use super::*;

/// Uploaded per-slot `SlotHeader[b]` payloads for a chunked attention launch.
///
/// Holds the GPU-resident headers; the slice arrays they point at are the
/// backing's own persistent slot-state buffers, and the position_map upload is
/// layer-invariant and held in the per-forward [`SharedPm`] cache. Drop this
/// only after the launch.
#[cfg(feature = "cuda")]
pub(super) struct SlotHeaderUpload {
    /// Raw GPU address of `SlotHeader[b]`.
    pub(super) headers_ptr: u64,
    /// The headers' device bytes: a lease on the launch's wave arena, or a pool
    /// buffer when the launch has no wave. Held for the duration of the launch.
    pub(super) _headers: Tensor,
    /// Pins every chunk the uploaded slot headers address: an uploaded page
    /// table is a REFERENCE, so it must hold the referenced gids alive. While
    /// held, a concurrent quantize-swap (`quantize_sealed_in_place` on the
    /// persistence thread rewriting the slot's shared sealed-prefix chunks
    /// mid-wave) cannot drop the last gid, so the float source arena can never
    /// hit `live == 0` and be released/re-tenanted under the in-flight
    /// paged-prefill kernel. The multi-wave CFW creep opened a seconds-wide
    /// window for exactly that (pre-CFW prefills finished within one forward, so
    /// the window was ~ms). Dropped with this struct after the forward's logits
    /// readback — i.e. after the kernels have retired.
    ///
    /// The set comes from the serialisation itself, which takes its pins in the
    /// same pass that writes the bytes, so a launch claims exactly what its
    /// headers address with one refcount bump per slot rather than a clone per
    /// chunk — the difference between O(1) and O(depth) on every layer of every
    /// step.
    pub(super) _pinned_gids: Vec<Arc<Vec<ChunkPin>>>,
}

/// Per-forward cache of the layer-invariant uploaded `position_map`.
///
/// The position_map maps each token position to its `(slice_idx, in-block
/// offset)`, which depends only on the chunk token layout — **identical across
/// every layer of a forward** (a sequence's chunks seal at the same boundaries
/// in all layers; only the K/V values + arena pointers differ per layer). So the
/// first layer builds + uploads it and every later layer reuses the same device
/// buffer + per-slot byte offsets, eliminating the dominant per-layer host build
/// and PCIe upload. Built once per (prefill) forward and dropped with it.
///
/// The type is named in the CPU-fallback `paged_prefill_batched` signature too,
/// so it exists on both targets; only the GPU-buffer guard is CUDA-gated and the
/// cache is only ever populated on the CUDA path.
#[allow(dead_code)]
pub struct SharedPm {
    /// Keeps the uploaded position_map buffer alive for the whole forward.
    #[cfg(feature = "cuda")]
    pub(super) _gpu: GpuBuf,
    /// Device base address of the packed position_map buffer.
    pub(super) base_ptr: u64,
    /// Per-slot byte offset into the packed buffer, in slot order.
    pub(super) byte_offsets: Vec<usize>,
    /// Per-slot `(n_slices, covered_tokens)` at map build time, in slot order.
    /// The map encodes `(slice_idx, in_blk)` pairs that index THAT layout's
    /// slice array; every later layer re-derives its slice list from live
    /// state and must match this shape exactly, or the kernel would resolve
    /// positions through a stale map into a different slice list — a wild
    /// `slice_idx` walks off the slice array into unrelated stager memory and
    /// dereferences a garbage `kvheads_ptr` (CUDA_ERROR_ILLEGAL_ADDRESS with
    /// no attribution). Checked on every cache hit in `build_slot_headers`.
    pub(super) per_slot_shape: Vec<(u32, u32)>,
    /// Per-slot `write_slice` at map build time, in slot order. The map's write
    /// region names this slice, so a later layer that resolves a different
    /// writer would scatter its new tokens into a chunk the map does not
    /// describe.
    pub(super) per_slot_write: Vec<u32>,
    /// Position-map buffers a later layer's rebuild superseded, held alive until
    /// this whole cache is dropped at the end of the forward.
    ///
    /// A windowed-creep layer whose live layout diverges from the cached map
    /// rebuilds and republishes it (so the layers past the creep front reuse the
    /// new one), but the buffer it replaces cannot be freed here: earlier layers
    /// launched their attention kernels against that device address and may
    /// still be reading it. Freeing and reallocating mid-forward could hand the
    /// same pool address to this layer's upload while that read is in flight — a
    /// wild read or wrong attention with no attribution (AGENTS.md invariant 7).
    /// Carrying the superseded buffer here defers its free to forward end, which
    /// is after the logits readback — i.e. after every kernel has retired.
    #[cfg(feature = "cuda")]
    pub(super) superseded: Vec<GpuBuf>,
}

/// Classify how a later layer's live slot layout differs from the cached
/// position map: whether the map must be rebuilt (`stale`), and whether the
/// divergence is the benign windowed-creep skew or a real layout change worth a
/// loud, attributed line (`anomalous`).
///
/// `cached` is the per-slot `(n_slices, covered_tokens)` the map was built over
/// and `cached_write` the per-slot writer slice; `live` is this layer's
/// `(n_slices, covered_tokens, writer_slice)` derived from the slot-state buffer
/// and the caller's `offsets + q_lens`. A length mismatch means the wave's slot
/// membership moved between layers — never a creep artefact, so both flags. Per
/// slot: the creep skew pushes a trailing 0-token writer chunk ahead of the
/// lagging layers, so the slice count moves while the covered-token count does
/// not; a covered-token move (or the writer slipping into a chunk the map does
/// not describe) is a genuine layout divergence. A covered-token move is the one
/// that reads as anomalous — the rebuild keeps the forward correct either way,
/// but only creep is expected to move the slice count on its own.
///
/// Pure host logic, so these rules are unit-tested directly without a GPU (this
/// whole module is CUDA-only, but the function touches no device). See
/// [`SharedPm`] for why a rebuilt map must not index a stale layout.
pub(super) fn classify_pm_divergence(
    cached: &[(u32, u32)],
    cached_write: &[u32],
    live: &[(u32, u32, u32)],
) -> (bool, bool) {
    if cached.len() != live.len() {
        return (true, true);
    }
    let mut stale = false;
    let mut anomalous = false;
    for i in 0..live.len() {
        let (want_slices, want_covered) = cached[i];
        let (n_slices, this_covered, buf_write) = live[i];
        let coverage_moved = this_covered != want_covered;
        // The writer slice moved into a chunk the cached map does not describe.
        let writer_moved =
            buf_write != cached_write[i] && (cached_write[i] as usize) < n_slices as usize;
        if n_slices != want_slices || coverage_moved || writer_moved {
            stale = true;
        }
        if coverage_moved {
            anomalous = true;
        }
    }
    (stale, anomalous)
}

/// Build + upload the per-slot `SlotHeader` payloads (slices, position_map,
/// header records) for a chunked attention launch. Shared by paged prefill and
/// the paged-glue forward — both read a sealed prefix + a writer region the
/// same way; only the kernel they feed differs.
///
/// The slice array a header points at is the backing's own persistent
/// slot-state buffer — the same bytes the decode metadata builder hands the
/// decode kernel (`KvCache::sync_chunked_slot_states`). Serialising a second,
/// private copy of it per layer per launch is what made this the deepest
/// remaining cost in speculative decode: at 128K a slot is thousands of chunks
/// that are byte-identical from one verify step to the next, and re-deriving
/// them was 291 ms of an 1806 ms decode. What genuinely differs per launch is
/// the position map's write region and the headers, and those are all that is
/// built here. Each header carries its slot's RoPE rung, from the deepest
/// position the launch writes for it.
///
/// **Runs inside the recording wave capture, and in the common case does not
/// end its segment.** The headers are carved from the launch's `wave` arena and
/// their upload is recorded beside the kernel that reads them. The device work
/// that remains — re-serialising a slot whose chunk table changed, and the
/// first layer's position map — pauses the recording on its own. Wrapping the
/// whole build in an eager section instead cut every attention layer's segment
/// twice and left the build's host time on the GPU's critical path: on
/// Flash-Next's single-session verify, ~80 µs of idle per attention layer.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_slot_headers(
    caches: &[&mut KvCache],
    q_lens: &[usize],
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    offsets: &[usize],
    rope: &RopeRungs,
    device: &Device,
    wave: Option<&WaveGeneration>,
) -> Result<SlotHeaderUpload> {
    let t_build = profile_now();
    if caches.len() != offsets.len() || caches.len() != q_lens.len() {
        candle::bail!(
            "slot header build: {} caches against {} offsets and {} q_lens",
            caches.len(),
            offsets.len(),
            q_lens.len()
        );
    }
    let t_arena = profile_now();
    let arena_info = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        first
            .k_cache()
            .chunked_resolve_arena_info()
            .ok_or_else(|| candle::Error::Msg("expected chunked resolve_arena_info".into()))??
    };

    // Bring every slot's persistent slice array up to date. A slot whose chunk
    // table stood still since the last sync (the common case in decode: five
    // tokens a step against a 32-token chunk) hands back its live pointer with
    // no work at all; one whose table changed re-serialises there, once, for
    // every consumer of that buffer.
    pipeline_record("slot:arena", t_arena);
    let t_sync = profile_now();
    let (slot_states, pinned_gids, sync_stats) =
        KvCache::sync_chunked_slot_states(caches, offsets, &arena_info, generation)?;
    pipeline_record("slot:sync", t_sync);
    // What `slot:sync` spent, by how each slot was brought up to date.
    pipeline_record_duration(
        "slot:sync.rebuild",
        sync_stats.rebuild_time,
        sync_stats.rebuilds,
    );
    pipeline_record_duration(
        "slot:sync.resync",
        sync_stats.resync_time,
        sync_stats.resyncs,
    );
    pipeline_record_duration("slot:sync.reuse", sync_stats.reuse_time, sync_stats.reuses);
    pipeline_record_duration(
        "slot:sync.extend",
        sync_stats.extend_time,
        sync_stats.extends,
    );
    let table_changed = sync_stats.rebuilds > 0;

    // The position_map is layer-invariant (see [`SharedPm`]). The first layer of
    // a forward builds and uploads it from the slots' token layouts; every later
    // layer reuses the device buffer and needs no per-chunk work at all — its
    // whole contribution is checking that the layout the map was built over is
    // still the layout the kernel will walk, which the slot-state buffer reports
    // in `(n_slices, write_slice)` and the caller in `offsets + q_lens`.
    let pm_cached = shared_pm.borrow().is_some();

    // A live gid whose arena has no entry (or a zeroed hole entry) would silently
    // resolve to `base_ptr 0` in `from_gids` — an in-band value that is legal for
    // absent-palette sentinels (raw < 0) but, for a real gid, means the slot's
    // block table outlived its arena: the kernel would deref ~null
    // (CUDA_ERROR_ILLEGAL_ADDRESS) with zero attribution. Refuse to launch and
    // name the chunk instead.
    //
    // Scanned when the table was rebuilt, which is when a stale gid can enter
    // it: a slot that handed back its live pointer serialised the very same
    // gids on an earlier pass of this scan, and has held them pinned since; one
    // that was extended serialised only chunks appended since — allocated
    // fresh, so live — and the previous writer, which that earlier pass
    // covered. A whole-table walk per extend would put back, per layer, the
    // depth-proportional cost the extend exists to remove.
    if table_changed {
        let mut dangling: Option<(usize, i64, usize, u16, u16)> = None;
        for (slot_i, cache) in caches.iter().enumerate() {
            cache.k_cache().chunked_visit_live_chunks(|it| {
                for c in it {
                    if dangling.is_some() {
                        break;
                    }
                    for gid in c.gids.as_slice() {
                        let raw = gid.raw();
                        if raw < 0 {
                            continue;
                        }
                        let a = gid.arena_idx();
                        // Two failure classes, one check: a freed arena (zeroed
                        // hole entry), and a chunk_idx past the arena's real
                        // format-specific capacity (the raw-GID namespace is
                        // sized for the densest format, so an in-namespace idx
                        // can still address past this arena's end — a stale gid
                        // recycled across formats).
                        let live = arena_info.get(a).is_some_and(|ai| {
                            ai.chunk_byte_stride != 0
                                && (gid.chunk_idx() as u32) < ai.chunk_capacity
                        });
                        if !live {
                            dangling = Some((slot_i, raw, a, c.offset, c.token_count));
                            break;
                        }
                    }
                }
            });
        }
        if let Some((slot_i, raw, arena, offset, tokens)) = dangling {
            candle::bail!(
                "slot header build: batch slot {slot_i} holds a live chunk (gid {raw}, \
                 offset {offset}, tokens {tokens}) whose arena {arena} is freed — the \
                 block table lost its backing (freed with live KV)"
            );
        }
    }

    // Per-slot `write_slice` for the headers: read from the token layout on the
    // layer that builds the map, from the cache on every later layer.
    //
    // `pm_cached` starts from whether a map exists, but a divergence below
    // clears it and forces a rebuild — so it is `mut`.
    let mut pm_cached = pm_cached;
    let mut write_slices: Vec<u32> = Vec::with_capacity(caches.len());
    if pm_cached {
        // Position-map shape check: the cached map was built from an earlier
        // layer's slice layout; the kernel resolves every k_pos through it into
        // THIS layer's slice array. The map is layer-invariant only while the
        // slot's chunk layout is the same on every layer — and during a windowed
        // creep prefill it is not. The creep advances the layers incrementally
        // (layer 0 first), pushing an empty (0-token) writer chunk for the next
        // window ahead of the layers still pending resume, so between the layer
        // that built the map and a later layer a slot's slice count differs by
        // that one trailing empty chunk (same covered tokens). See the skew note
        // on `BatchedInferenceSession::sequence_block_count`. Reusing the stale
        // map would resolve k_pos through a slice index that layer does not have
        // — a garbage `kvheads_ptr` (CUDA_ERROR_ILLEGAL_ADDRESS, no attribution).
        //
        // So when the cached shape no longer matches this layer's live layout,
        // rebuild the map from the current layout and republish it for the layers
        // past the creep front, rather than refuse the launch. Each layer's map
        // then matches the slices its own kernel walks, which is all correctness
        // needs — attention is per-layer. Refusing instead stranded the forward,
        // and retried every wave it hung the engine with the GPU pegged and no
        // conversation ever opened. The rebuild reads the same live backing the
        // first layer's build already trusts, so it is no less safe than that
        // build is — and the buffer it replaces is not freed here but carried in
        // `SharedPm::superseded` until forward end (an earlier layer's kernel may
        // still be reading it).
        //
        // The benign creep skew adds a trailing 0-token chunk: the slice count
        // moves but the covered-token count does not. A covered-token move or a
        // change in the wave's slot count is a real layout divergence — not creep
        // — so it is worth a loud, attributed line even though the rebuild still
        // makes it safe.
        let (stale, anomalous) = {
            let cache = shared_pm.borrow();
            let s = cache
                .as_ref()
                .expect("pm_cached implies shared_pm is populated");
            // This layer's live `(n_slices, covered_tokens, writer_slice)` per
            // slot, against the `(n_slices, covered_tokens)` + writer the map was
            // built over. Classified by `classify_pm_divergence`, which is a pure
            // function so the stale/anomalous rules are unit-tested directly.
            let live: Vec<(u32, u32, u32)> = (0..caches.len())
                .map(|i| {
                    let (_, n_slices, buf_write) = slot_states[i];
                    (n_slices, (offsets[i] + q_lens[i]) as u32, buf_write)
                })
                .collect();
            classify_pm_divergence(&s.per_slot_shape, &s.per_slot_write, &live)
        };
        if stale {
            if anomalous {
                tracing::warn!(
                    "position map rebuilt mid-forward from a layout that is not the \
                     benign windowed-creep skew: the covered-token count or the wave's \
                     slot membership moved between layers. The rebuild keeps this \
                     forward correct, but suspect a concurrent chunk-table mutation if \
                     it recurs."
                );
            }
            // Do NOT free the cached map here — the build path takes it and carries
            // its buffer forward in `superseded`. This just routes both branches
            // below to the build path.
            pm_cached = false;
        } else {
            let cache = shared_pm.borrow();
            let s = cache
                .as_ref()
                .expect("pm_cached implies shared_pm is populated");
            write_slices.extend_from_slice(&s.per_slot_write[..caches.len()]);
        }
    }
    pipeline_record("slot:build", t_build);

    let t_pack = profile_now();
    // Position_map: layer-invariant, so build + upload it only on the first
    // layer of the forward and reuse the device buffer + per-slot byte offsets
    // for the rest (see [`SharedPm`]). Ragged: slot i writes q_lens[i] new
    // tokens, so a slot's map runs `offsets[i] + q_lens[i] == kv_lens[i]`
    // entries and the kernel resolves any k_pos in `[0, kv_lens[i])` with a
    // single lookup.
    let pm_byte_offsets: Vec<usize> = if pm_cached {
        let cache = shared_pm.borrow();
        let s = cache
            .as_ref()
            .expect("pm_cached implies shared_pm is populated");
        s.byte_offsets.clone()
    } else {
        let total: usize = offsets.iter().zip(q_lens.iter()).map(|(o, l)| o + l).sum();
        let mut pm_buf: Vec<u32> = Vec::with_capacity(total.max(1));
        let mut byte_offsets: Vec<usize> = Vec::with_capacity(caches.len());
        let mut per_slot_shape: Vec<(u32, u32)> = Vec::with_capacity(caches.len());
        for (slot_i, cache) in caches.iter().enumerate() {
            // The one per-chunk pass of the forward: the token layout the map
            // is built from. Nothing here resolves an arena pointer or
            // serialises a record — that is the slot-state buffer's job, done
            // once per table change.
            let writer_start_idx = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
            let mut chunks: Vec<(u16, u16)> = Vec::new();
            // Per chunk: whether the paged stores write into it, and palette 0's
            // K and V tags for the refusal below.
            let mut writable: Vec<(bool, u8, u8)> = Vec::new();
            let mut cum: u32 = 0;
            cache.k_cache().chunked_visit_live_chunks(|it| {
                for c in it {
                    cum = cum.saturating_add(c.token_count as u32);
                    chunks.push((c.offset, c.token_count));
                    let k_ok = c
                        .k_fmt
                        .iter()
                        .all(|&t| ArenaFormatTag::from_u8(t).takes_active_k_writes());
                    let v_ok = c
                        .v_fmt
                        .iter()
                        .all(|&t| ArenaFormatTag::from_u8(t).takes_active_v_writes());
                    writable.push((
                        k_ok && v_ok,
                        c.k_fmt.first().copied().unwrap_or(u8::MAX),
                        c.v_fmt.first().copied().unwrap_or(u8::MAX),
                    ));
                }
            });
            // Count invariant: the slices must cover AT LEAST the slot's recorded
            // sealed-KV offset. A shortfall (`cum < want`) means the block table
            // lost chunks (the host-side "computed write len N is invalid"
            // class); the kernel would seek a token past the covered range and
            // walk off the END of the slice array into adjacent stager memory —
            // garbage headers, garbage kvheads_ptr, CUDA_ERROR_ILLEGAL_ADDRESS
            // with no attribution. That is the only fatal case.
            //
            // An OVERAGE (`cum > want`) is the mid-creep skew and is safe,
            // REGARDLESS of `q_len`. A windowed advance moves some layers ahead of
            // the others, so `reconcile_entry_offsets` clamps the slot offset to
            // the MINIMUM coverage across layers (see `sequence_backing_tokens`) —
            // which leaves the ahead layers covering more than that clamped `want`.
            // The position map built below from THIS layer's chunks is then a
            // SUPERSET of `[0, want)`: every position the varlen metadata asks for
            // still resolves, and the extra covered tokens are simply unindexed.
            // New tokens are appended at the writer slice the layout resolves from
            // this layer's own chunks (`extend_for_write_region`), not blindly at
            // `want`, so a positive `q_len` lands correctly on top of the superset.
            // (This is the SAME event `classify_pm_divergence` above calls the
            // benign creep: there `cum` is measured against the cached map layer 0
            // published; here against the clamped `want`. Two baselines, one skew.)
            //
            // Bailing on the overage — in ANY form, including the earlier
            // `q_len > 0` variant of this guard — deadlocks the creep: the forward
            // fails, the lagging layers never catch up, and it fails again every
            // tick, wedging the slot for good. That regression showed up live as
            // "nothing came of it" across most characters. The overage genuinely
            // occurs during normal windowed decode (`q_len == 1`), so a shortfall
            // is the only fatal case; overage is tolerated.
            let want = offsets[slot_i];
            if (cum as usize) < want {
                candle::bail!(
                    "slot header build: batch slot {slot_i} slices cover {cum} tokens \
                     but the slot's recorded offset is {want} ({} slices) — block \
                     table lost chunks",
                    chunks.len()
                );
            }
            // The buffer's slice count is what the header promises the kernel;
            // this list is what the map indexes. `sync_chunked_slot_states`
            // refuses a divergence, and this is the same fact from the caller's
            // side — cheap, and it names the slot.
            let (_, n_slices, buf_write) = slot_states[slot_i];
            if n_slices as usize != chunks.len() {
                candle::bail!(
                    "slot header build: batch slot {slot_i} slot-state buffer holds \
                     {n_slices} slices but the host chunk list has {} — the buffer \
                     was not re-serialised after a chunk-table mutation",
                    chunks.len()
                );
            }
            let layout = SlotTokenLayout::new(chunks, writer_start_idx, CHUNK_SIZE);
            if layout.write_slice != buf_write
                && (layout.write_slice as usize) < layout.chunks.len()
            {
                candle::bail!(
                    "slot header build: batch slot {slot_i} writes slice {} by the \
                     token layout but slice {buf_write} by the slot-state buffer",
                    layout.write_slice
                );
            }
            // The chunk this slot's first new token lands in must be one the
            // paged stores write: `store_kv_chunk_arena` returns without a word
            // for a sealed format, so a quantized chunk with room left in it,
            // standing where the writer should be, drops every token of the
            // prefill — in every layer, with nothing faulting. Refuse and name
            // it instead. The writer is the layout's, the same rule the kernel's
            // `write_slice` follows; a layout with no writer is
            // `assert_write_region_capacity`'s to refuse.
            let w = layout.write_slice as usize;
            if q_lens[slot_i] > 0 {
                if let Some(&(false, k_tag, v_tag)) = writable.get(w) {
                    let tokens = layout.chunks[w].1;
                    candle::bail!(
                        "slot header build: batch slot {slot_i}'s write chunk {w} ({tokens} \
                         tokens) is in a format the paged stores cannot write (K tag {k_tag}, \
                         V tag {v_tag}) — every token of this prefill would be dropped without \
                         a fault. A sealed chunk with room left in it is standing in the \
                         writer region."
                    );
                }
            }
            byte_offsets.push(pm_buf.len() * 4);
            let start = pm_buf.len();
            layout.push_position_map(&mut pm_buf);
            layout.extend_for_write_region(&mut pm_buf, q_lens[slot_i], CHUNK_SIZE);
            per_slot_shape.push((n_slices, (pm_buf.len() - start) as u32));
            write_slices.push(layout.write_slice);
        }
        if pm_buf.is_empty() {
            pm_buf.push(0u32);
        }
        let pm_byte_len = pm_buf.len() * std::mem::size_of::<u32>();
        let mut pm_pinned = generation.alloc(pm_byte_len)?;
        // SAFETY: u32 has no padding and is trivially copyable; lengths match.
        let pm_bytes =
            unsafe { std::slice::from_raw_parts(pm_buf.as_ptr() as *const u8, pm_byte_len) };
        pm_pinned.copy_from_slice(pm_bytes);
        // The position map, the slice array and the slot headers are the chain
        // every block of the attention kernel walks before it can issue a K/V
        // load — header → slice → record, and per token the map — so they go to
        // device memory (`submit_resident`): a zero-copy mapping would make each
        // link a PCIe round trip on the kernel's critical path.
        let pm_gpu = generation.submit_resident(pm_pinned)?;
        let base_ptr = pm_gpu.dev_ptr();
        // Take the map this rebuild replaces (if any) and carry its buffer, and
        // any it had already carried, forward — freeing it now could pull the
        // device memory an earlier layer's in-flight kernel is still reading. It
        // is released with this whole cache at forward end, after the kernels
        // have retired. On the first (uncached) layer there is nothing to carry.
        let superseded = match shared_pm.borrow_mut().take() {
            Some(prev) => {
                let mut carried = prev.superseded;
                carried.push(prev._gpu);
                carried
            }
            None => Vec::new(),
        };
        *shared_pm.borrow_mut() = Some(SharedPm {
            _gpu: pm_gpu,
            base_ptr,
            byte_offsets: byte_offsets.clone(),
            per_slot_shape,
            per_slot_write: write_slices.clone(),
            superseded,
        });
        byte_offsets
    };
    let pm_base_ptr = shared_pm
        .borrow()
        .as_ref()
        .expect("position_map cache populated above")
        .base_ptr;

    let mut header_buf: Vec<u8> = Vec::with_capacity(caches.len() * SLOT_HEADER_BYTES);
    for (i, &write_slice) in write_slices.iter().enumerate() {
        let (slices_ptr, n_slices, _) = slot_states[i];
        SlotHeaderHost {
            n_slices,
            write_slice,
            slices_ptr,
            position_map_ptr: pm_base_ptr + pm_byte_offsets[i] as u64,
            // The deepest position this launch writes for the slot: its
            // tokens, the holes before them, and the new tokens.
            rope_rung: rope
                .rung_for(offsets[i] + caches[i].k_cache().chunked_lead_total() + q_lens[i])?,
        }
        .write(&mut header_buf);
    }

    // Device memory, as the map is — every block of the kernel walks the
    // headers before its first K/V load.
    let n_bytes = header_buf.len();
    let headers = wave_from_vec_ticketed(header_buf, n_bytes, device, wave.map(|w| w.ticket()))?;
    let headers_ptr = tensor_u8_device_ptr(&headers)?;
    pipeline_record("slot:pack", t_pack);

    Ok(SlotHeaderUpload {
        headers_ptr,
        _headers: headers,
        _pinned_gids: pinned_gids,
    })
}

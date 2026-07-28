#[cfg(feature = "cuda")]
use crate::models::profile::{pipeline_record, profile_now, profile_sync};
use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::GpuBuf;
use candle::*;
pub(crate) use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
pub(crate) use candle_nn::kv_cache::CHUNK_SIZE;

#[cfg(feature = "cuda")]
use {
    candle::backend::BackendStorage,
    candle::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr},
    candle_kernels::paged_glue::{run_paged_glue_bf16, run_paged_glue_fp16},
    candle_kernels::paged_prefill::*,
    candle_nn::kv_cache::ChunkedKvBacking,
    core::ffi::c_void,
    half::{bf16, f16},
};

#[cfg(feature = "cuda")]
use crate::models::prefill_capture::maybe_capture;
#[cfg(feature = "cuda")]
use crate::models::slot_state::{SlotStateHost, TokenSliceHost};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::HeadGids;

/// Uploaded per-slot `SlotHeader[b]` payloads for a chunked attention launch.
///
/// Holds the GPU-resident headers + the host `SlotStateHost` per slot (its
/// `position_map` drives glue write-target derivation). The three `GpuBuf`
/// guards keep the stager uploads alive for the duration of the kernel — drop
/// this only after the launch.
#[cfg(feature = "cuda")]
struct SlotHeaderUpload {
    /// Raw GPU address of `SlotHeader[b]` (24 bytes each).
    headers_ptr: u64,
    /// Keeps the stager uploads (headers, slices, records) alive for the
    /// duration of the kernel launch. The position_map upload is layer-invariant
    /// and held separately in the per-forward [`SharedPm`] cache.
    _guards: (GpuBuf, GpuBuf, GpuBuf),
    /// Pins every chunk the uploaded slot headers address: an uploaded page
    /// table is a REFERENCE, so it must hold the referenced gids alive. One
    /// `HeadGids` Arc clone per live chunk — while held, a concurrent
    /// quantize-swap (`quantize_sealed_in_place` on the persistence thread
    /// rewriting the slot's shared sealed-prefix chunks mid-wave) cannot drop
    /// the last gid, so the float source arena can never hit `live == 0` and
    /// be released/re-tenanted under the in-flight paged-prefill kernel. The
    /// multi-wave CFW creep opened a seconds-wide window for exactly that
    /// (pre-CFW prefills finished within one forward, so the window was ~ms).
    /// Dropped with this struct after the forward's logits readback — i.e.
    /// after the kernels have retired.
    _pinned_gids: Vec<HeadGids>,
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
    _gpu: GpuBuf,
    /// Device base address of the packed position_map buffer.
    base_ptr: u64,
    /// Per-slot byte offset into the packed buffer, in slot order.
    byte_offsets: Vec<usize>,
}

/// Build + upload the per-slot `SlotHeader` payloads (slices, position_map,
/// header records) for a chunked attention launch. Shared by paged prefill and
/// the paged-glue forward — both read a sealed prefix + a writer region the
/// same way; only the kernel they feed differs.
#[cfg(feature = "cuda")]
fn build_slot_headers(
    caches: &[&mut KvCache],
    q_lens: &[usize],
    n_kv_head: usize,
    head_dim: usize,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    expected_offsets: Option<&[usize]>,
) -> Result<SlotHeaderUpload> {
    let t_build = profile_now();
    let arena_info = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        first
            .k_cache()
            .chunked_resolve_arena_info()
            .ok_or_else(|| candle::Error::Msg("expected chunked resolve_arena_info".into()))??
    };

    // The position_map is layer-invariant (see [`SharedPm`]). The first layer of
    // a forward populates `shared_pm` and uploads it; later layers reuse it and
    // skip both the host build and the PCIe upload. When already cached we build
    // each slot's slices WITHOUT its position_map.
    let pm_cached = shared_pm.borrow().is_some();

    // Zero-clone slice build: visit each cache's live chunks by reference
    // (no SealedChunk materialization — its per-chunk clones and
    // arena_byte_size walks measured ~0.5 ms per layer-call at deep
    // prefixes, ~30x the slice build itself).
    let mut slots: Vec<SlotStateHost> = Vec::with_capacity(caches.len());
    // Reference pin: every chunk the headers will address (see
    // `SlotHeaderUpload::_pinned_gids`).
    let mut pinned_gids: Vec<HeadGids> = Vec::new();
    // A live gid whose arena has no entry (or a zeroed hole entry) would silently
    // resolve to `base_ptr 0` in `from_gids` — an in-band value that is legal for
    // absent-palette sentinels (raw < 0) but, for a real gid, means the slot's
    // block table outlived its arena: the kernel would deref ~null
    // (CUDA_ERROR_ILLEGAL_ADDRESS) with zero attribution. Refuse to launch and
    // name the chunk instead.
    let mut dangling: Option<(usize, i64, usize, u16, u16)> = None;
    for (slot_i, cache) in caches.iter().enumerate() {
        let writer_start_idx = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);
        let mut slices: Vec<TokenSliceHost> = Vec::new();
        let mut cum: u32 = 0;
        cache.k_cache().chunked_visit_live_chunks(|it| {
            for c in it {
                let rope_base = cum;
                cum = cum.saturating_add(c.token_count as u32);
                if dangling.is_none() {
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
                pinned_gids.push(c.gids.clone());
                slices.push(TokenSliceHost::from_live_chunk(
                    &c,
                    rope_base,
                    n_kv_head,
                    head_dim,
                    &arena_info,
                ));
            }
        });
        // Count invariant: the slices must cover EXACTLY the slot's recorded
        // sealed-KV offset. A shortfall means the block table lost chunks (the
        // host-side "computed write len N is invalid" class); the kernel would
        // seek a token past the covered range and walk off the END of the slice
        // array into adjacent stager memory — garbage headers, garbage
        // kvheads_ptr, CUDA_ERROR_ILLEGAL_ADDRESS with no attribution.
        if let Some(expected) = expected_offsets {
            let want = expected[slot_i];
            if (cum as usize) != want {
                candle::bail!(
                    "slot header build: batch slot {slot_i} slices cover {cum} tokens \
                     but the slot's recorded offset is {want} ({} slices) — block \
                     table lost chunks",
                    slices.len()
                );
            }
        }
        slots.push(SlotStateHost::from_slices(
            slices,
            writer_start_idx,
            !pm_cached,
        ));
    }
    if let Some((slot_i, raw, arena, offset, tokens)) = dangling {
        candle::bail!(
            "slot header build: batch slot {slot_i} holds a live chunk (gid {raw}, \
             offset {offset}, tokens {tokens}) whose arena {arena} is freed — the \
             block table lost its backing (freed with live KV)"
        );
    }

    // Extend each slot's position_map to cover the write region. Ragged: slot i
    // writes q_lens[i] new tokens, so after this `position_map.len() ==
    // offsets[i] + q_lens[i] == kv_lens[i]`, letting the kernel resolve any
    // k_pos in `[0, kv_lens[i])` via a single lookup. Skipped on a cache hit —
    // the cached upload already covers the (layer-invariant) write region.
    let chunk_size = CHUNK_SIZE;
    if !pm_cached {
        for (slot, &add) in slots.iter_mut().zip(q_lens.iter()) {
            slot.extend_for_write_region(add, chunk_size);
        }
    }
    pipeline_record("slot:build", t_build);

    let t_pack = profile_now();
    // Two-section upload. A records buffer (each *scratch* slice's out-of-line
    // KvHead[n_kv_head] record) is submitted FIRST so the slice headers can
    // embed each record's device address without self-referencing a single
    // buffer (the stager only yields a device pointer at submit). Resident
    // slices (`meta.is_some()`) skip the records buffer entirely and point their
    // `kvheads_ptr` at the device meta-pool slab — the residence win: no per-
    // forward head rebuild, no scratch upload for the sealed prefix.
    let rec_bytes = TokenSliceHost::record_size(n_kv_head, head_dim);
    let total_slices: usize = slots.iter().map(|s| s.slices.len()).sum();

    /// Where a slice's KvHead record lives: a resident device address, or a
    /// byte offset into the per-forward scratch records buffer.
    enum KvSrc {
        Resident(u64),
        Scratch(usize),
    }
    let mut records_buf: Vec<u8> = Vec::with_capacity(total_slices * rec_bytes);
    let mut srcs: Vec<KvSrc> = Vec::with_capacity(total_slices);
    for (slot, cache) in slots.iter().zip(caches.iter()) {
        for slice in &slot.slices {
            match &slice.meta {
                Some(meta) => {
                    let addr = cache.k_cache().chunked_meta_device_addr(meta);
                    // Invariant (enforced by `build_meta_records`, which returns
                    // None on a host-only pool): meta=Some ⇒ device_addr != 0.
                    // A resident slice has empty heads (no scratch fallback), so a
                    // 0 here would be a null `kvheads_ptr` — fail loudly in release
                    // rather than let the kernel deref null.
                    if addr == 0 {
                        candle::bail!(
                            "resident slice (meta=Some) resolved to device_addr 0 — \
                             record not device-resident"
                        );
                    }
                    srcs.push(KvSrc::Resident(addr));
                }
                None => {
                    let off = records_buf.len();
                    slice.serialize_record(&mut records_buf);
                    srcs.push(KvSrc::Scratch(off));
                }
            }
        }
    }
    if records_buf.is_empty() {
        records_buf.push(0u8);
    }
    let mut records_pinned = generation.alloc(records_buf.len())?;
    records_pinned.copy_from_slice(&records_buf);
    let records_gpu = generation.submit(records_pinned)?;
    let records_base = records_gpu.dev_ptr();

    // Slice headers (16 bytes each), in slot order, each pointing at its record
    // (resident address as-is, scratch offset rebased onto `records_base`).
    let mut slice_buf: Vec<u8> =
        Vec::with_capacity(total_slices * TokenSliceHost::SLICE_HEADER_SIZE);
    let mut slot_byte_offsets: Vec<usize> = Vec::with_capacity(slots.len());
    let mut k = 0usize;
    for slot in &slots {
        slot_byte_offsets.push(slice_buf.len());
        for slice in &slot.slices {
            let kvheads_ptr = match srcs[k] {
                KvSrc::Resident(addr) => addr,
                KvSrc::Scratch(off) => records_base + off as u64,
            };
            slice.serialize_slice_header(&mut slice_buf, kvheads_ptr);
            k += 1;
        }
    }
    if slice_buf.is_empty() {
        slice_buf.push(0u8);
    }
    let mut slices_pinned = generation.alloc(slice_buf.len())?;
    slices_pinned.copy_from_slice(&slice_buf);
    let slices_gpu = generation.submit(slices_pinned)?;
    let slices_base_ptr = slices_gpu.dev_ptr();

    // Position_map: layer-invariant, so build + upload it only on the first
    // layer of the forward and reuse the device buffer + per-slot byte offsets
    // for the rest (see [`SharedPm`]). On a cache hit the slots carry no
    // position_map (built with `build_position_map = false`), so the cached
    // offsets are authoritative.
    let pm_byte_offsets: Vec<usize> = if pm_cached {
        let cache = shared_pm.borrow();
        let s = cache
            .as_ref()
            .expect("pm_cached implies shared_pm is populated");
        s.byte_offsets.clone()
    } else {
        let total_pm_entries: usize = slots.iter().map(|s| s.position_map.len()).sum();
        let mut pm_buf: Vec<u32> = Vec::with_capacity(total_pm_entries.max(1));
        let mut byte_offsets: Vec<usize> = Vec::with_capacity(slots.len());
        for slot in &slots {
            byte_offsets.push(pm_buf.len() * 4);
            pm_buf.extend_from_slice(&slot.position_map);
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
        let pm_gpu = generation.submit(pm_pinned)?;
        let base_ptr = pm_gpu.dev_ptr();
        *shared_pm.borrow_mut() = Some(SharedPm {
            _gpu: pm_gpu,
            base_ptr,
            byte_offsets: byte_offsets.clone(),
        });
        byte_offsets
    };
    let pm_base_ptr = shared_pm
        .borrow()
        .as_ref()
        .expect("position_map cache populated above")
        .base_ptr;

    let mut header_buf: Vec<u8> = Vec::with_capacity(slots.len() * 24);
    for (i, slot) in slots.iter().enumerate() {
        let n_slices = slot.slices.len() as u32;
        let write_slice = slot.write_slice;
        let slices_ptr = slices_base_ptr + slot_byte_offsets[i] as u64;
        let position_map_ptr = pm_base_ptr + pm_byte_offsets[i] as u64;
        header_buf.extend_from_slice(&n_slices.to_le_bytes());
        header_buf.extend_from_slice(&write_slice.to_le_bytes());
        header_buf.extend_from_slice(&slices_ptr.to_le_bytes());
        header_buf.extend_from_slice(&position_map_ptr.to_le_bytes());
    }

    let mut pinned = generation.alloc(header_buf.len())?;
    pinned.copy_from_slice(&header_buf);
    let headers_gpu = generation.submit(pinned)?;
    let headers_ptr = headers_gpu.dev_ptr();
    pipeline_record("slot:pack", t_pack);

    Ok(SlotHeaderUpload {
        headers_ptr,
        _guards: (headers_gpu, slices_gpu, records_gpu),
        _pinned_gids: pinned_gids,
    })
}

/// Chunked KV + paged prefill attention over chunks.
///
/// This path avoids pooled KV materialization by:
/// 1) Ensuring (or creating) chunked KV backing
/// 2) Scattering this prefill segment into chunked arenas
/// 3) Running the paged prefill attention kernel that reads via `block_table`.
///
/// Returns `Some(outputs_per_sequence)` on success, `None` if inapplicable or if the attempt fails.
#[cfg(feature = "cuda")]
fn paged_prefill_batched_impl(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<Tensor> {
    // Ragged/varlen prefill. q/k/v arrive FLAT-packed:
    //   q: [total_q, n_head, head_dim], k/v: [total_q, n_kv_head, head_dim]
    // where total_q = Σ q_lens and the per-sequence token ranges are described
    // by `prefill_meta` (cu_seqlens_q / q_lens / kv_lens). `b_sz` is the number
    // of sequences (= caches.len() = q_lens.len()). Uniform prefill is the
    // special case where all q_lens are equal.
    let total_q: usize = q_lens.iter().sum();
    let max_add = q_lens.iter().copied().max().unwrap_or(0);
    if !(max_add >= 1
        && matches!(q.device(), Device::Cuda(_))
        && head_dim % 32 == 0
        && head_dim <= 256)
    {
        candle::bail!(
            "paged prefill batched attention not applicable: \
             total_q={}, device={:?}, head_dim={}",
            total_q,
            q.device(),
            head_dim
        );
    }

    // Reset caches where offset == 0.
    for (cache, &off) in caches.iter_mut().zip(offsets.iter()) {
        if off == 0 {
            cache.reset();
        }
    }

    let mut use_chunks = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();

    // If not already chunked, create chunked backing and (if needed) migrate the existing prefix.
    if !use_chunks {
        let max_len = offsets.iter().copied().max().unwrap_or(0);
        let max_after = offsets
            .iter()
            .zip(q_lens.iter())
            .map(|(&o, &l)| o.saturating_add(l))
            .max()
            .unwrap_or(max_add);

        // If we're creating chunked backing from scratch (no prefix to migrate), use the
        // incoming K/V dtype. Empty caches may otherwise report a default dtype (often F32)
        // even when the caller intends BF16/F16 execution.
        //
        // IMPORTANT: The paged prefill kernel only supports BF16/F16. If F32 is detected,
        // we use BF16 for the arenas and convert K/V on scatter. The Q/K/V inputs to the
        // attention kernel itself will be converted to BF16 by ensure_dtype anyway.
        let raw_dtype = if max_len > 0 {
            caches
                .first()
                .map(|c| c.dtype())
                .unwrap_or_else(|| k.dtype())
        } else {
            k.dtype()
        };
        let dtype = match raw_dtype {
            DType::F32 | DType::F64 => DType::BF16, // Paged prefill only supports BF16/F16
            other => other,
        };

        // Migration from contiguous to chunked KV is not supported.
        // Callers should ensure they start with chunked caches.
        if max_len > 0 {
            candle::bail!(
                "migration from contiguous KV to chunked backing is not supported; \
                    please ensure caches start with chunked backing for batched prefill"
            );
        }

        let backing =
            ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, q.device(), max_after)?;

        // Allocate chunks for exactly each sequence's own new-token count — never
        // over-allocate to max(q_lens), which would leave extra tail chunks that
        // desync the decode writer slice.
        backing.ensure_for_offsets(offsets, q_lens)?;

        // Switch caches to shared chunked backing.
        for (i, cache) in caches.iter_mut().enumerate() {
            cache.set_chunked_backing(&backing, i, None)?;
            cache.set_current_seq_len(offsets[i])?;
        }
        use_chunks = true;
    }

    if !use_chunks {
        candle::bail!("chunked backing unavailable");
    }

    // Ensure chunks for EXACTLY each sequence's own q_len — never over-allocate
    // to max(q_lens). Extra trailing chunks on a shorter sequence desync the
    // decode writer slice (the kernel's "one free slot per chunk" contract),
    // tripping the `ws_offset + ws_len < CHUNK_SIZE` assert on its next decode.
    // Per-cache calls use each cache's real batch_idx, so a single-element slice
    // targets the correct slot.
    // Entry drain: attributes GPU work still in flight when the prefill
    // call begins (enqueued by the caller) separately from this call's own
    // spans — without it the first sync'd span absorbs the caller's tail.
    let t_entry = profile_now();
    profile_sync(q.device());
    pipeline_record("prefill:entry", t_entry);
    let t_alloc = profile_now();
    for (i, &add) in q_lens.iter().enumerate() {
        KvCache::ensure_chunked_capacity_batch(&mut caches[i..i + 1], &offsets[i..i + 1], add)?;
    }
    profile_sync(q.device());
    pipeline_record("prefill:alloc", t_alloc);

    let t_meta = profile_now();
    let (compute_dtype, _chunk_size) = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;

        // Use the cache's dtype — for quantized backings this returns F16 (the dequant
        // output dtype), and for float backings it returns the arena's actual dtype.
        // The paged attention kernels run in F16/BF16 only, so collapse F32/F64 reference-mode
        // float arenas to BF16 — matching the decode path (decode_utils compute_dtype) and the
        // chunked-backing dtype selection above. Q/K/V are cast to this dtype before launch.
        let collapse_compute = |d: DType| match d {
            DType::F32 | DType::F64 => DType::BF16,
            other => other,
        };
        let k_compute = collapse_compute(first.k_cache().dtype());
        let v_compute = collapse_compute(first.v_cache().dtype());
        if k_compute != v_compute {
            candle::bail!(
                "K and V caches require different compute dtypes: K={:?} V={:?}",
                k_compute,
                v_compute
            );
        }
        let compute_dtype = k_compute;

        let chunk_size = first
            .k_cache()
            .chunked_chunk_size()
            .ok_or_else(|| candle::Error::Msg("expected chunked chunk_size".into()))?;
        (compute_dtype, chunk_size)
    };

    pipeline_record("prefill:metadata", t_meta);

    let t_pack = profile_now();
    // --- prefill:pack spans tensor packing, chunk_meta, head_gids construction ---
    // Some model paths provide non-contiguous K/V; ensure contiguity once.
    let k = if k.is_contiguous() {
        k.clone()
    } else {
        k.contiguous()?
    };
    let v = if v.is_contiguous() {
        v.clone()
    } else {
        v.contiguous()?
    };

    // Q/K/V already arrive FLAT-packed in cu_seqlens_q token order:
    //   q_packed: [total_q, n_head, head_dim], k/v: [total_q, n_kv_head, head_dim]
    // (k/v were made contiguous just above). No transpose/reshape needed.
    let q_packed = if q.is_contiguous() {
        q.clone()
    } else {
        q.contiguous()?
    };
    let k_packed = k;
    let v_packed = v;
    debug_assert_eq!(q_packed.dim(0)?, total_q);

    // Varlen metadata (device tensors). Renamed to `*_dev` so the host
    // `q_lens: &[usize]` param stays in scope for the per-seq seqlen update.
    let (cu_seqlens_q, q_lens_dev, kv_lens) = if let Some((cu, ql, kv)) = prefill_meta {
        (cu.clone(), ql.clone(), kv.clone())
    } else {
        // Fallback: rebuild the ragged metadata from the host q_lens.
        let mut cu = Vec::with_capacity(b_sz + 1);
        cu.push(0u32);
        let mut acc = 0u32;
        for &l in q_lens {
            acc += l as u32;
            cu.push(acc);
        }
        let cu_seqlens_q = Tensor::from_vec(cu, b_sz + 1, q.device())?;
        let q_lens_dev = Tensor::from_vec(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            b_sz,
            q.device(),
        )?;
        let kv_lens = Tensor::from_vec(
            offsets
                .iter()
                .zip(q_lens.iter())
                .map(|(&o, &l)| (o + l) as u32)
                .collect::<Vec<_>>(),
            b_sz,
            q.device(),
        )?;
        (cu_seqlens_q, q_lens_dev, kv_lens)
    };

    let softmax_scale = 1f32 / (head_dim as f32).sqrt();

    // Check storage policy to determine if reconcile is needed after prefill.
    // Validate that any quantized storage policy is kernel-native.
    // Reconcile and consolidation now happen once after all layers in batched_model.rs.
    let needs_reconcile = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        match first.k_cache().chunked_storage_policy() {
            Some(policy) => {
                let is_quant = policy.to_arena_key().is_quantized();
                if is_quant && !policy.is_kernel_native() {
                    candle::bail!(
                        "storage policy uses a quantized format that the kernel cannot read natively; \
                         all quantized formats must be kernel-native"
                    );
                }
                is_quant
            }
            None => false,
        }
    };

    let header_upload = build_slot_headers(
        caches,
        q_lens,
        n_kv_head,
        head_dim,
        generation,
        shared_pm,
        Some(offsets),
    )?;
    let headers_ptr = header_upload.headers_ptr;

    profile_sync(q.device());
    pipeline_record("prefill:pack", t_pack);

    // Optional kernel-replay capture: dumps this call's packed Q/K/V + cached KV
    // chunks + geometry to a fixture. No-op unless `ZEND_PREFILL_CAPTURE` is set;
    // fires once, on the first call past the kv_len threshold (one layer).
    maybe_capture(
        caches,
        offsets,
        &q_packed,
        &k_packed,
        &v_packed,
        q_lens,
        n_head,
        n_kv_head,
        head_dim,
        rope_offsets,
        rope_cs,
        rope_interleaved,
    );

    let t_kernel = profile_now();
    let out_packed = paged_prefill_attn_varlen_chunks(
        &q_packed,
        &cu_seqlens_q,
        &q_lens_dev,
        &kv_lens,
        &k_packed,
        &v_packed,
        headers_ptr,
        compute_dtype,
        n_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        rope_offsets,
        rope_cs,
        rope_interleaved,
        max_add,
    )?;
    profile_sync(q.device());
    pipeline_record("prefill:kernel", t_kernel);
    // Per-sequence written length (each sequence advanced by its own q_lens[i],
    // not the over-allocated max_add).
    for ((cache, &off), &add) in caches.iter_mut().zip(offsets.iter()).zip(q_lens.iter()) {
        cache.set_current_seq_len(off + add)?;
    }

    // After each prefill layer, eagerly quantize all fully-sealed chunks so that
    // float F16 arenas are freed as we go rather than accumulating to OOM.
    // The partial tail chunk (still being written to) is skipped automatically by
    // reconcile_multi, which only processes blocks where seq_len / CHUNK_SIZE > blk.
    //
    // After reconcile, float arenas hold only the sparse tail chunks. Consolidate
    // them into the minimum number of arenas and CUDA-free the rest. This is critical
    // for large batches (e.g. Q4_0Ã—460) where uncompacted tail arenas would exceed
    // the available VRAM budget (each F16 arena is 64 MiB regardless of occupancy).
    // Reconcile and consolidate now happen once after all layers complete,
    // in batched_model.rs. Per-layer reconcile was removed to avoid paying
    // the quantization cost on every layer during prefill.

    // Prefill and any post-prefill migrations are now complete. For pure-float
    // paths we can materialize the persistent decode slot buffers eagerly here.
    // Quantized paths are left to rebuild lazily on the next decode metadata
    // sync so they always see the final post-reconcile chunk routing.
    if !needs_reconcile {
        KvCache::prime_chunked_decode_slots_batch(caches)?;
    }

    // Return the attention output FLAT-packed in cu_seqlens_q token order:
    // [total_q, n_head, head_dim]. The caller reshapes to [total_q, n_head*head_dim]
    // and runs the output projection per-token (no per-sequence split needed).
    Ok(out_packed)
}

/// Public wrapper: full-context prefill through the INT8 prefix-attention
/// kernel (docs/archived/prefill_optimization.md) — GQA-packed M,
/// slice-aligned tiles, int8 m16n8k32 QK/PV computed directly over the
/// quantized arena.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<Tensor> {
    paged_prefill_batched_impl(
        caches,
        offsets,
        q,
        k,
        v,
        b_sz,
        q_lens,
        n_head,
        n_kv_head,
        head_dim,
        prefill_meta,
        rope_offsets,
        rope_cs,
        rope_interleaved,
        generation,
        shared_pm,
    )
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    _rope_offsets: &Tensor,
    _rope_cs: &Tensor,
    _rope_interleaved: bool,
    _generation: &Generation,
    _shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<Tensor> {
    // CPU fallback: per-sequence standard attention. The paged CUDA kernel is
    // the production path; this exists only for non-chunked CPU caches.
    // Q/K/V arrive FLAT-packed [total_q, n_*head, head_dim] in cu_seqlens order;
    // we slice each sequence's rows via the running prefix sum of q_lens.
    let first_cache = caches
        .first()
        .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
    if first_cache.k_cache().chunked_arena_chunks().is_some() {
        candle::bail!(
                "batched multi-token attention cannot use chunked caches without paged prefill support; \
                 paged prefill requires head_dim to be a multiple of 32 and <= 256; device must be CUDA; \
                 head_dim={}, has_chunked_caches=true, supports_paged_prefill=false",
                head_dim
            );
    }

    let mut all_outputs = Vec::with_capacity(b_sz);
    let mut cu = 0usize;
    for (batch_idx, cache) in caches.iter_mut().enumerate() {
        let seq_len = q_lens[batch_idx];
        // Slice this sequence's flat rows and restore [1, n_*head, seq_len, head_dim].
        let q_seq = q
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let k_seq = k
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        let v_seq = v
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        cu += seq_len;

        // Append new K/V to cache
        let (k_cached, v_cached) = cache.append(&k_seq, &v_seq)?;
        let cache_len = k_cached.dim(2)?;

        // Repeat K/V heads for MQA/GQA
        let k_rep = crate::utils::repeat_kv(k_cached, n_head / n_kv_head)?;
        let v_rep = crate::utils::repeat_kv(v_cached, n_head / n_kv_head)?;

        // Standard scaled dot-product attention with causal mask
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q_seq.matmul(&k_rep.t()?)? * scale)?;

        let offset = offsets[batch_idx];
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..cache_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, cache_len), q.device())?;
        let mask = mask.to_dtype(att.dtype())?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // [1, n_head, seq_len, head_dim] -> [seq_len, n_head, head_dim] (flat row block)
        let attn_out = att
            .matmul(&v_rep.contiguous()?)?
            .transpose(1, 2)?
            .reshape((seq_len, n_head, head_dim))?;
        all_outputs.push(attn_out);
    }

    // Concatenate per-sequence outputs back into the flat [total_q, n_head, head_dim].
    Tensor::cat(&all_outputs, 0)
}

/// CPU stub: the paged-glue forward is a CUDA-only kernel path.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn paged_glue_attn(
    _caches: &mut [&mut KvCache],
    _offsets: &[usize],
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _b_sz: usize,
    _q_lens: &[usize],
    _n_head: usize,
    _n_kv_head: usize,
    _head_dim: usize,
    _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    _col_actual_pos: &Tensor,
    _rope_cs: &Tensor,
    _rope_interleaved: bool,
    _fwd_window: usize,
    _generation: &Generation,
) -> Result<Tensor> {
    candle::bail!("paged-glue requires the cuda feature")
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct PagedPrefillInt8 {
    /// Longest per-sequence q_len in the batch — sizes the kernel's
    /// query-tile grid exactly.
    max_q_len: usize,
    softmax_scale: f32,
    cu_seqlens_q: Tensor,
    q_lens: Tensor,
    kv_lens: Tensor,
    k_packed: Tensor,
    v_packed: Tensor,
    /// Raw GPU virtual address of `SlotHeader[batch_size]`.
    headers_ptr: u64,
    batch_size: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Compute dtype for K/V/Q — F16 or BF16. Pre-resolved from K and V arena formats.
    compute_dtype: DType,
    /// RoPE position offsets per batch element, shape [batch_size], dtype U32.
    /// Zero values = natural positions (no extra shift). Non-zero = page-clone position delta.
    rope_offsets: Tensor,
    /// Precomputed cos/sin table on device, shape [max_pos, head_dim], dtype F32.
    /// Layout: rope_cs[pos * head_dim + d*2] = cos, [pos * head_dim + d*2+1] = sin.
    rope_cs: Tensor,
    /// RoPE pairing style: false=non-interleaved half-split (Qwen/GPT2), true=interleaved adjacent-pairs (Llama).
    rope_interleaved: bool,
}

#[cfg(feature = "cuda")]
impl PagedPrefillInt8 {
    fn cuda_fwd_t<
        Q: candle::cuda_backend::CudaDType + DeviceRepr, // Query type
        KV: candle::cuda_backend::CudaDType + DeviceRepr, // KV cache type
        O: candle::cuda_backend::CudaDType + DeviceRepr, // Output type
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let (total_q, n_head, head_dim) = q_l.shape().dims3()?;
        if n_head != self.n_head || head_dim != self.head_dim {
            candle::bail!(
                "paged-prefill-int8: q shape mismatch got {:?} expected (total_q, {}, {})",
                q_l.shape(),
                self.n_head,
                self.head_dim
            )
        }

        let dev = q.device();
        let stream = dev.cuda_stream();

        let (cu_seqlens_q_s, cu_seqlens_q_l) = self.cu_seqlens_q.storage_and_layout();
        let (q_lens_s, q_lens_l) = self.q_lens.storage_and_layout();
        let (kv_lens_s, kv_lens_l) = self.kv_lens.storage_and_layout();
        let (k_packed_s, k_packed_l) = self.k_packed.storage_and_layout();
        let (v_packed_s, v_packed_l) = self.v_packed.storage_and_layout();

        if cu_seqlens_q_l.shape().dims1()? != self.batch_size + 1 {
            candle::bail!(
                "paged-prefill-int8: cu_seqlens_q must have len batch+1 ({})",
                self.batch_size + 1
            )
        }
        if q_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-int8: q_lens must have len batch ({})",
                self.batch_size
            )
        }
        if kv_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-int8: kv_lens must have len batch ({})",
                self.batch_size
            )
        }
        if self.cu_seqlens_q.dtype() != DType::U32
            || self.q_lens.dtype() != DType::U32
            || self.kv_lens.dtype() != DType::U32
        {
            candle::bail!("paged-prefill-int8: cu_seqlens_q/q_lens/kv_lens must be U32")
        }

        let cu_seqlens_q = match &*cu_seqlens_q_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: cu_seqlens_q must be a cuda tensor"),
        }
        .slice(cu_seqlens_q_l.start_offset()..);

        let q_lens = match &*q_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: q_lens must be a cuda tensor"),
        }
        .slice(q_lens_l.start_offset()..);

        let kv_lens = match &*kv_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: kv_lens must be a cuda tensor"),
        }
        .slice(kv_lens_l.start_offset()..);

        let k_packed = match &*k_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-int8: k_packed must be a cuda tensor"),
        }
        .slice(k_packed_l.start_offset()..);

        let v_packed = match &*v_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-int8: v_packed must be a cuda tensor"),
        }
        .slice(v_packed_l.start_offset()..);

        let elem_count = q_l.shape().elem_count();
        let dst = unsafe { dev.alloc::<O>(elem_count)? };

        // Compute q_dtype code from Q's actual dtype
        // q_dtype codes: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
        let q_dtype_code: i32 = match q.dtype() {
            candle::DType::F16 => 1,
            candle::DType::BF16 => 2,
            dt => candle::bail!(
                "paged-prefill: unsupported Q dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        };

        match self.head_dim {
            64 | 96 | 128 | 256 => {}
            hd => candle::bail!(
                "paged-prefill: unsupported head_dim {hd} (must be 64, 96, 128, or 256)"
            ),
        }

        unsafe {
            let q = q.as_cuda_slice::<Q>()?;
            let q = q.slice(q_l.start_offset()..);
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr(&stream);
            let (k_ptr, _guard) = k_packed.device_ptr(&stream);
            let (v_ptr, _guard) = v_packed.device_ptr(&stream);
            let (cu_ptr, _guard) = cu_seqlens_q.device_ptr(&stream);
            let (q_lens_ptr, _guard) = q_lens.device_ptr(&stream);
            let (kv_lens_ptr, _guard) = kv_lens.device_ptr(&stream);
            // Extract rope_offsets pointer. The prefill CUDA kernel applies fused RoPE to Q
            // (in smem) and to new K tokens (k_pos >= prefix_len) before computing attention
            // scores and writing K/V to the arena.
            let rope_offsets_ptr = {
                let (ro_s, ro_l) = self.rope_offsets.storage_and_layout();
                let ro_slice = match &*ro_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("paged-prefill: rope_offsets must be a cuda tensor"),
                }
                .slice(ro_l.start_offset()..);
                let (ro_ptr, _ro_guard) = ro_slice.device_ptr(&stream);
                ro_ptr as *const u32
            };

            let rope_cs_ptr = {
                let (cs_s, cs_l) = self.rope_cs.storage_and_layout();
                let cs_slice = match &*cs_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("paged-prefill: rope_cs must be a cuda tensor"),
                }
                .slice(cs_l.start_offset()..);
                let (cs_ptr, _cs_guard) = cs_slice.device_ptr(&stream);
                cs_ptr as *const f32
            };

            let headers_ptr = self.headers_ptr as *const u8;

            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            candle::set_kernel_breadcrumb("run_paged_prefill_int8", file!(), line!());
            run_paged_prefill_int8(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                headers_ptr,
                cu_ptr as *const u32,
                q_lens_ptr as *const u32,
                kv_lens_ptr as *const u32,
                dst_ptr as *mut core::ffi::c_void,
                total_q as i32,
                self.batch_size as i32,
                self.n_head as i32,
                self.n_kv_head as i32,
                self.head_dim as i32,
                self.max_q_len as i32,
                self.softmax_scale,
                q_dtype_code,
                rope_offsets_ptr,
                rope_cs_ptr,
                self.rope_interleaved as i32,
                raw_stream,
            );
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, q_l.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
impl candle::CustomOp1 for PagedPrefillInt8 {
    fn name(&self) -> &'static str {
        "paged-prefill-int8"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-prefill-int8")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        if q.dtype() != self.compute_dtype {
            candle::bail!(
                "paged-prefill-int8: expected {:?} Q, got {:?}",
                self.compute_dtype,
                q.dtype()
            );
        }
        match self.compute_dtype {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16, f16>(q, q_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16, bf16, bf16>(q, q_l),
            dt => candle::bail!("paged-prefill-int8: unsupported compute dtype {:?}", dt),
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
/// Paged prefill attention over chunked KV arenas (no KV materialization),
/// through the INT8 prefix-attention kernel.
///
/// `q` has shape `(total_q, n_head, head_dim)`.
/// `headers_ptr` is the raw GPU address of `SlotHeader[batch_size]`, reusing the
/// same persistent slot-payload representation as decode.
/// `compute_dtype` is the pre-resolved F16 or BF16 dtype for Q/K/V (derived from arena formats).
pub(crate) fn paged_prefill_attn_varlen_chunks(
    q: &Tensor,
    cu_seqlens_q: &Tensor,
    q_lens: &Tensor,
    kv_lens: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    headers_ptr: u64,
    compute_dtype: DType,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    max_q_len: usize,
) -> Result<Tensor> {
    // The kernel's in-thread RoPE pairing needs head_dim % 64 == 0 with the
    // non-interleaved half-split pairing (Qwen/GPT2 style), and head_dim
    // 256's staging slabs exceed the 25.6 KB 4-blocks/SM union-arena budget.
    if head_dim != 64 && head_dim != 128 {
        candle::bail!("paged-prefill-int8 supports head_dim 64 or 128 (got {head_dim})")
    }
    if rope_interleaved {
        candle::bail!("paged-prefill-int8 does not support interleaved RoPE")
    }

    let q = q.to_dtype(compute_dtype)?;
    let k_packed = k_packed.to_dtype(compute_dtype)?;
    let v_packed = v_packed.to_dtype(compute_dtype)?;

    let (_total_q, q_n_head, q_head_dim) = q.dims3()?;
    if q_n_head != n_head || q_head_dim != head_dim {
        candle::bail!("paged-prefill-int8: q shape mismatch {:?}", q.dims())
    }

    let (kp_total_q, kp_n_kv, kp_hd) = k_packed.dims3()?;
    if kp_total_q != _total_q || kp_n_kv != n_kv_head || kp_hd != head_dim {
        candle::bail!(
            "paged-prefill-int8: k_packed shape mismatch {:?}",
            k_packed.dims()
        )
    }
    let (vp_total_q, vp_n_kv, vp_hd) = v_packed.dims3()?;
    if vp_total_q != _total_q || vp_n_kv != n_kv_head || vp_hd != head_dim {
        candle::bail!(
            "paged-prefill-int8: v_packed shape mismatch {:?}",
            v_packed.dims()
        )
    }

    let batch_size = cu_seqlens_q.dim(0)?.saturating_sub(1);

    let op = PagedPrefillInt8 {
        max_q_len,
        softmax_scale,
        cu_seqlens_q: cu_seqlens_q.clone(),
        q_lens: q_lens.clone(),
        kv_lens: kv_lens.clone(),
        k_packed: k_packed.clone(),
        v_packed: v_packed.clone(),
        headers_ptr,
        batch_size,
        n_head,
        n_kv_head,
        head_dim,
        compute_dtype,
        rope_offsets: rope_offsets.clone(),
        rope_cs: rope_cs.clone(),
        rope_interleaved,
    };
    q.apply_op1(op)
}

/// FFI signature shared by the FP16/BF16 glue-kernel entry points.
#[cfg(feature = "cuda")]
type GlueFfi = unsafe extern "C" fn(
    *const c_void, // q
    *const u8,     // headers
    *mut c_void,   // o
    i32,           // batch
    i32,           // max_glue
    i32,           // n_q_head
    i32,           // n_kv_head
    i32,           // head_dim
    f32,           // softmax_scale
    *const c_void, // k_new
    *const c_void, // v_new
    *const f32,    // rope_cs
    i32,           // rope_interleaved
    *const u32,    // cu_seqlens_q
    *const u32,    // q_lens
    *const u32,    // kv_lens
    *const u32,    // glue_write_slice
    *const u32,    // glue_write_in_blk
    *const u32,    // fwd_ahead
    *mut c_void,   // stream
);

/// Op for the paged-glue reprojection forward. Mirrors [`PagedPrefillInt8`]
/// but launches the decode-derivative glue kernel: each slot's `G` glue queries
/// attend its quantized sealed prefix (streamed-once dequant) plus earlier glue,
/// write their own K/V into the writer chunks, and mask by TRUE sequence
/// position via `col_actual_pos`. The output is `q`-shaped.
#[cfg(feature = "cuda")]
struct PagedGlueChunks {
    softmax_scale: f32,
    cu_seqlens_q: Tensor,
    q_lens: Tensor,
    kv_lens: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    /// Flat `[Σ q_lens]` U32 — gap chunk slice index per glue row (scatter target).
    glue_write_slice: Tensor,
    /// Flat `[Σ q_lens]` U32 — in-block offset per glue row.
    glue_write_in_blk: Tensor,
    /// Flat `[Σ q_lens]` U32 — forward bridge window per glue row (`0` == causal).
    fwd_ahead: Tensor,
    headers_ptr: u64,
    batch_size: usize,
    max_glue: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    compute_dtype: DType,
    rope_cs: Tensor,
    rope_interleaved: bool,
}

#[cfg(feature = "cuda")]
impl PagedGlueChunks {
    fn cuda_fwd_t<T: candle::cuda_backend::CudaDType + DeviceRepr>(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi: GlueFfi,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let (_total_q, n_head, head_dim) = q_l.shape().dims3()?;
        if n_head != self.n_head || head_dim != self.head_dim {
            candle::bail!(
                "paged-glue: q shape mismatch got {:?} expected (total_q, {}, {})",
                q_l.shape(),
                self.n_head,
                self.head_dim
            )
        }
        let dev = q.device();
        let stream = dev.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut c_void;

        // U32 metadata: extract the raw device address. Each source tensor is a
        // field of `self`, so its storage outlives this call; the device_ptr
        // guard is only needed for stream ordering, which the launch preserves.
        let u32_ptr = |t: &Tensor| -> Result<u64> {
            let (s, l) = t.storage_and_layout();
            let sl = match &*s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("paged-glue: expected cuda u32 tensor"),
            }
            .slice(l.start_offset()..);
            let (p, _g) = sl.device_ptr(&stream);
            Ok(p)
        };
        let cu_ptr = u32_ptr(&self.cu_seqlens_q)?;
        let ql_ptr = u32_ptr(&self.q_lens)?;
        let kv_ptr = u32_ptr(&self.kv_lens)?;
        let gws_ptr = u32_ptr(&self.glue_write_slice)?;
        let gwi_ptr = u32_ptr(&self.glue_write_in_blk)?;
        let fa_ptr = u32_ptr(&self.fwd_ahead)?;

        // Typed Q/K/V (compute dtype).
        let q_slice = q.as_cuda_slice::<T>()?.slice(q_l.start_offset()..);
        let (q_ptr, _qg) = q_slice.device_ptr(&stream);
        let (k_s, k_l) = self.k_new.storage_and_layout();
        let k_slice = match &*k_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("paged-glue: k_new must be a cuda tensor"),
        }
        .slice(k_l.start_offset()..);
        let (k_ptr, _kg) = k_slice.device_ptr(&stream);
        let (v_s, v_l) = self.v_new.storage_and_layout();
        let v_slice = match &*v_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("paged-glue: v_new must be a cuda tensor"),
        }
        .slice(v_l.start_offset()..);
        let (v_ptr, _vg) = v_slice.device_ptr(&stream);

        // RoPE cos/sin table (F32).
        let (cs_s, cs_l) = self.rope_cs.storage_and_layout();
        let cs_slice = match &*cs_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("paged-glue: rope_cs must be a cuda tensor"),
        }
        .slice(cs_l.start_offset()..);
        let (cs_ptr, _csg) = cs_slice.device_ptr(&stream);

        let elem_count = q_l.shape().elem_count();
        let dst = unsafe { dev.alloc::<T>(elem_count)? };

        unsafe {
            let (dst_ptr, _dg) = dst.device_ptr(&stream);
            candle::set_kernel_breadcrumb("run_paged_glue", file!(), line!());
            ffi(
                q_ptr as *const c_void,
                self.headers_ptr as *const u8,
                dst_ptr as *mut c_void,
                self.batch_size as i32,
                self.max_glue as i32,
                self.n_head as i32,
                self.n_kv_head as i32,
                self.head_dim as i32,
                self.softmax_scale,
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                cs_ptr as *const f32,
                self.rope_interleaved as i32,
                cu_ptr as *const u32,
                ql_ptr as *const u32,
                kv_ptr as *const u32,
                gws_ptr as *const u32,
                gwi_ptr as *const u32,
                fa_ptr as *const u32,
                raw_stream,
            );
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, q_l.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
impl candle::CustomOp1 for PagedGlueChunks {
    fn name(&self) -> &'static str {
        "paged-glue-chunks"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-glue-chunks")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        if q.dtype() != self.compute_dtype {
            candle::bail!(
                "paged-glue-chunks: expected {:?} Q, got {:?}",
                self.compute_dtype,
                q.dtype()
            );
        }
        match self.compute_dtype {
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, run_paged_glue_fp16),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, run_paged_glue_bf16),
            dt => candle::bail!("paged-glue-chunks: unsupported compute dtype {:?}", dt),
        }
    }
}

/// Reprojection glue forward over chunked KV. Each slot's `q_lens[i]` glue
/// queries are reserved IN PLACE as gap chunks at their logical positions; this
/// forward scatters each query's K/V into its gap (`glue_write_slice/in_blk`)
/// and computes its attention over the whole slot. Every column's sequence
/// position comes from its chunk `rope_base` (`slice_rope`) — the same
/// convention decode reads — so there is no `col_actual_pos`. Each glue token
/// attends backward over everything and forward up to `fwd_ahead[t]` tokens (its
/// bridge window). HD128 only — other head dims stay on the plain prefill path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_glue_attn(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    glue_write_slice: &Tensor,
    glue_write_in_blk: &Tensor,
    fwd_ahead: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    generation: &Generation,
) -> Result<Tensor> {
    if head_dim != 128 {
        candle::bail!("paged-glue requires head_dim==128 (got {head_dim})");
    }
    let total_q: usize = q_lens.iter().sum();
    let max_glue = q_lens.iter().copied().max().unwrap_or(0);
    let _ = offsets; // slot geometry is read from the cache, not the caller offset

    // Glue fires over an already-sealed prefix, so the caches must be chunked.
    let use_chunks = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();
    if !use_chunks {
        candle::bail!("paged-glue requires chunked caches");
    }

    // The gaps are already reserved chunks (the slot length counts them) — no
    // capacity allocation here. `kv_len` is the slot's actual length, read from
    // the cache so it is independent of how the caller accounts the offset.
    let kv_lens_host: Vec<usize> = caches.iter().map(|c| c.current_seq_len()).collect();

    let (compute_dtype, _max_blocks) = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        let k_compute = first.k_cache().dtype();
        let v_compute = first.v_cache().dtype();
        if k_compute != v_compute {
            candle::bail!(
                "paged-glue: K/V compute dtype mismatch: K={k_compute:?} V={v_compute:?}"
            );
        }
        (k_compute, first.k_cache().chunked_max_blocks())
    };

    let needs_reconcile = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        match first.k_cache().chunked_storage_policy() {
            Some(policy) => policy.to_arena_key().is_quantized(),
            None => false,
        }
    };

    let q_packed = if q.is_contiguous() {
        q.clone()
    } else {
        q.contiguous()?
    };
    let k_packed = if k.is_contiguous() {
        k.clone()
    } else {
        k.contiguous()?
    };
    let v_packed = if v.is_contiguous() {
        v.clone()
    } else {
        v.contiguous()?
    };

    let device = q.device();
    let _ = total_q;
    let (cu_seqlens_q, q_lens_dev) = if let Some((cu, ql, _kv)) = prefill_meta {
        (cu.clone(), ql.clone())
    } else {
        let mut cu = Vec::with_capacity(b_sz + 1);
        cu.push(0u32);
        let mut acc = 0u32;
        for &l in q_lens {
            acc += l as u32;
            cu.push(acc);
        }
        let cu_seqlens_q = Tensor::from_vec(cu, b_sz + 1, device)?;
        let q_lens_dev = Tensor::from_vec(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            b_sz,
            device,
        )?;
        (cu_seqlens_q, q_lens_dev)
    };
    // `kv_len` is the slot's actual length (the reserved gaps ARE the glue tokens,
    // already counted) — independent of how the caller accounts the offset.
    let kv_lens = Tensor::from_vec(
        kv_lens_host.iter().map(|&l| l as u32).collect::<Vec<_>>(),
        b_sz,
        device,
    )?;

    let t_hdr = profile_now();
    // The gaps are real chunks already (no trailing write region), so the slot
    // headers cover exactly `[0, kv_len)`; pass zero glue so build_slot_headers
    // does not extend a write region. Built fresh every call (always-miss).
    let zero_q = vec![0usize; b_sz];
    let glue_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
    let header_upload = build_slot_headers(
        caches,
        &zero_q,
        n_kv_head,
        head_dim,
        generation,
        &glue_pm,
        None,
    )?;
    profile_sync(device);
    pipeline_record("glue:hdr_meta", t_hdr);

    let t_kernel = profile_now();
    let softmax_scale = 1f32 / (head_dim as f32).sqrt();
    let op = PagedGlueChunks {
        softmax_scale,
        cu_seqlens_q,
        q_lens: q_lens_dev,
        kv_lens,
        k_new: k_packed.to_dtype(compute_dtype)?,
        v_new: v_packed.to_dtype(compute_dtype)?,
        // Per-token gap scatter target + forward bridge window, from the caller.
        glue_write_slice: glue_write_slice.clone(),
        glue_write_in_blk: glue_write_in_blk.clone(),
        fwd_ahead: fwd_ahead.clone(),
        headers_ptr: header_upload.headers_ptr,
        batch_size: b_sz,
        max_glue,
        n_head,
        n_kv_head,
        head_dim,
        compute_dtype,
        rope_cs: rope_cs.clone(),
        rope_interleaved,
    };
    let q_compute = q_packed.to_dtype(compute_dtype)?;
    let out = q_compute.apply_op1(op)?;
    profile_sync(device);
    pipeline_record("glue:kernel", t_kernel);

    // NO advance here: the glue tokens are reserved IN PLACE as gap chunks whose
    // `usage` is already counted in each slot's length (`reserve_glue_gap`). This
    // forward only SCATTERS their K/V into those gaps — it does not append a
    // trailing region. Advancing by `add` again (the old trailing-glue design)
    // would double-count the slot length and, because `writer_start` points past
    // the last chunk, `set_len`'s `writer_start.min(n-1)` fallback would inflate
    // the final gap chunk's usage — desyncing the slot and overflowing the next
    // prefill's write region. `header_upload` stays alive until return.
    if !needs_reconcile {
        KvCache::prime_chunked_decode_slots_batch(caches)?;
    }
    drop(header_upload);
    Ok(out)
}

/// Precompute cos/sin table for RoPE from inv_freq, computed with f64 precision.
///
/// Layout: `table[pos * head_dim + d * 2] = cos(pos * inv_freq[d])`,
///         `table[pos * head_dim + d * 2 + 1] = sin(pos * inv_freq[d])`.
///
/// Returns a tensor of shape `[max_blocks * 32, head_dim]`, dtype F32, on `device`.
pub fn compute_rope_cs(
    inv_freq: &Tensor,
    max_blocks: usize,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let max_pos = max_blocks * 32; // CHUNK_SIZE = 32
    let inv_freq_host = inv_freq.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let half_dim = inv_freq_host.len();
    let mut table = vec![0f32; max_pos * head_dim];
    for pos in 0..max_pos {
        let base = pos * head_dim;
        for d in 0..half_dim {
            let angle = pos as f64 * inv_freq_host[d] as f64;
            table[base + d * 2] = angle.cos() as f32;
            table[base + d * 2 + 1] = angle.sin() as f32;
        }
    }
    Tensor::from_vec(table, (max_pos, head_dim), device)
}

// ============================================================================
// Decode kernel — persistent slot buffer edition
// ============================================================================

/// Paged decode attention using persistent slot buffers.
///
/// Takes the slot pool `headers` tensor (16 bytes × n_active per slot) instead
/// of the old per-step chunk_meta / head_gids / kv_lens / per_head_table.
/// The kernel self-increments ws.len after scatter, so no write_offsets needed.
///
/// Runs the production INT8 split-KV / warp-stripe / batched-M decode kernel
/// (`run_paged_decode_*`) for head_dim 64/96/128/256.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn(
    q: &Tensor,
    headers_ptr: u64,
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) -> Result<Tensor> {
    let num_active_slots = q.dim(0)?;
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let op = PagedDecode {
        headers_ptr,
        arena_dtype,
        n_q_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        k_new,
        v_new,
        rope_cs: rope_cs.clone(),
        rope_interleaved,
        num_active_slots,
        emit_q8: false,
    };
    q.apply_op1(op)
}

/// B2 decode: like [`paged_decode_attn`] but the combine kernel emits the attention context
/// directly as q8a1024 blocks (head_dim 128 only). Returns a flat `[q8_bytes]` U8 tensor; the
/// caller wraps it as a `Q8a128Operand` (`rows = num_active_slots`, `cols = n_q_head·head_dim`)
/// and feeds `o_proj` via the int8 path — no FP store + standalone quantize.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn paged_decode_attn_q8(
    q: &Tensor,
    headers_ptr: u64,
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) -> Result<Tensor> {
    let num_active_slots = q.dim(0)?;
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let op = PagedDecode {
        headers_ptr,
        arena_dtype,
        n_q_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        k_new,
        v_new,
        rope_cs: rope_cs.clone(),
        rope_interleaved,
        num_active_slots,
        emit_q8: true,
    };
    q.apply_op1(op)
}

#[cfg(feature = "cuda")]
struct PagedDecode {
    headers_ptr: u64, // raw GPU virtual address of SlotHeader[num_active_slots]
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: Tensor,
    v_new: Tensor,
    rope_cs: Tensor,
    rope_interleaved: bool,
    num_active_slots: usize,
    /// B2: when true the combine kernel emits the attention context as q8a1024 blocks (head_dim
    /// 128 only) instead of an FP tensor, so `o_proj` consumes it via the int8 path with no
    /// standalone quantize. The op then returns a flat `[q8_bytes]` U8 tensor of the operand bytes.
    emit_q8: bool,
}

#[cfg(feature = "cuda")]
impl PagedDecode {
    /// q8a1024 byte size of the emitted context: `[num_active_slots × (n_q_head·head_dim)]`.
    fn q8_byte_size(&self) -> usize {
        let cols = self.n_q_head * self.head_dim;
        let total_tiles = self.num_active_slots * (cols / 128);
        total_tiles.div_ceil(8) * 1152
    }

    fn cuda_fwd_typed<
        Q: candle::cuda_backend::CudaDType + DeviceRepr + 'static,
        KV: candle::cuda_backend::CudaDType + DeviceRepr,
        O: candle::cuda_backend::CudaDType + DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi_fn: unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const u8,
            *mut core::ffi::c_void,
            i32,
            i32,
            i32,
            i32,
            f32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const f32,
            i32,
            *mut core::ffi::c_void,
        ),
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();

        let out_elem = self.num_active_slots * self.n_q_head * self.head_dim;
        let dst = unsafe { dev.alloc::<O>(out_elem)? };

        {
            let q_slice = q.as_cuda_slice::<Q>()?.slice(q_l.start_offset()..);
            let (q_ptr, _q_g) = q_slice.device_ptr(&stream);

            let headers_ptr = self.headers_ptr as *const u8;

            let (k_s, k_l) = self.k_new.storage_and_layout();
            let k_slice = match &*k_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: k_new must be CUDA"),
            }
            .slice(k_l.start_offset()..);
            let (k_ptr, _k_g) = k_slice.device_ptr(&stream);

            let (v_s, v_l) = self.v_new.storage_and_layout();
            let v_slice = match &*v_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: v_new must be CUDA"),
            }
            .slice(v_l.start_offset()..);
            let (v_ptr, _v_g) = v_slice.device_ptr(&stream);

            let (rcs_s, rcs_l) = self.rope_cs.storage_and_layout();
            let rcs_slice = match &*rcs_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("paged-decode-v2: rope_cs must be CUDA"),
            }
            .slice(rcs_l.start_offset()..);
            let (rcs_ptr, _rcs_g) = rcs_slice.device_ptr(&stream);

            let (dst_ptr, _dst_g) = dst.device_ptr(&stream);

            // Pass the device's dedicated stream so that both the decode
            // kernel and the follow-up commit_decode_write_len_kernel run on
            // the same stream as all GpuChunksGuard::memcpy_htod calls.
            // Passing null here would use the default CUDA stream, which
            // can race with the non-blocking dedicated stream.
            candle::set_kernel_breadcrumb("run_paged_decode", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            unsafe {
                ffi_fn(
                    q_ptr as *const core::ffi::c_void,
                    headers_ptr as *const u8,
                    dst_ptr as *mut core::ffi::c_void,
                    self.num_active_slots as i32,
                    self.n_q_head as i32,
                    self.n_kv_head as i32,
                    self.head_dim as i32,
                    self.softmax_scale,
                    k_ptr as *const core::ffi::c_void,
                    v_ptr as *const core::ffi::c_void,
                    rcs_ptr as *const f32,
                    self.rope_interleaved as i32,
                    raw_stream,
                );
            }
        } // all guards dropped here, dst no longer borrowed

        let dst_cs = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        let out_shape = Shape::from_dims(&[self.num_active_slots, self.n_q_head, self.head_dim]);
        Ok((dst_cs, out_shape))
    }

    /// B2 q8a1024-emitting variant: allocates the q8a1024 byte buffer and passes it as the
    /// kernel's `q8_out`, returning a flat `[q8_bytes]` U8 storage (no FP context store).
    fn cuda_fwd_typed_q8<
        Q: candle::cuda_backend::CudaDType + DeviceRepr + 'static,
        KV: candle::cuda_backend::CudaDType + DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi_fn: unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const u8,
            *mut core::ffi::c_void,
            i32,
            i32,
            i32,
            i32,
            f32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const f32,
            i32,
            *mut core::ffi::c_void,
        ),
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();

        let q8_bytes = self.q8_byte_size();
        let dst = unsafe { dev.alloc::<u8>(q8_bytes)? };

        {
            let q_slice = q.as_cuda_slice::<Q>()?.slice(q_l.start_offset()..);
            let (q_ptr, _q_g) = q_slice.device_ptr(&stream);

            let headers_ptr = self.headers_ptr as *const u8;

            let (k_s, k_l) = self.k_new.storage_and_layout();
            let k_slice = match &*k_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: k_new must be CUDA"),
            }
            .slice(k_l.start_offset()..);
            let (k_ptr, _k_g) = k_slice.device_ptr(&stream);

            let (v_s, v_l) = self.v_new.storage_and_layout();
            let v_slice = match &*v_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: v_new must be CUDA"),
            }
            .slice(v_l.start_offset()..);
            let (v_ptr, _v_g) = v_slice.device_ptr(&stream);

            let (rcs_s, rcs_l) = self.rope_cs.storage_and_layout();
            let rcs_slice = match &*rcs_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("paged-decode-v2: rope_cs must be CUDA"),
            }
            .slice(rcs_l.start_offset()..);
            let (rcs_ptr, _rcs_g) = rcs_slice.device_ptr(&stream);

            let (dst_ptr, _dst_g) = dst.device_ptr(&stream);

            candle::set_kernel_breadcrumb("run_paged_decode_q8", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            unsafe {
                ffi_fn(
                    q_ptr as *const core::ffi::c_void,
                    headers_ptr as *const u8,
                    dst_ptr as *mut core::ffi::c_void,
                    self.num_active_slots as i32,
                    self.n_q_head as i32,
                    self.n_kv_head as i32,
                    self.head_dim as i32,
                    self.softmax_scale,
                    k_ptr as *const core::ffi::c_void,
                    v_ptr as *const core::ffi::c_void,
                    rcs_ptr as *const f32,
                    self.rope_interleaved as i32,
                    raw_stream,
                );
            }
        } // all guards dropped here, dst no longer borrowed

        let dst_cs = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst_cs, Shape::from_dims(&[q8_bytes])))
    }
}

#[cfg(feature = "cuda")]
impl candle::CustomOp1 for PagedDecode {
    fn name(&self) -> &'static str {
        "paged-decode"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-decode")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match self.head_dim {
            64 | 96 | 128 | 256 => {}
            hd => candle::bail!(
                "paged-decode: unsupported head_dim {hd} (must be 64, 96, 128, or 256)"
            ),
        }

        // B2: emit the attention context as q8a1024 (head_dim 128 only) so o_proj runs int8.
        if self.emit_q8 {
            use candle_kernels::paged_decode::{
                run_paged_decode_bf16_q8, run_paged_decode_fp16_q8,
            };
            if self.head_dim != 128 {
                candle::bail!(
                    "paged-decode q8: q8a1024 emit requires head_dim 128, got {}",
                    self.head_dim
                );
            }
            return match self.arena_dtype {
                DType::F16 => self.cuda_fwd_typed_q8::<f16, f16>(q, q_l, run_paged_decode_fp16_q8),
                DType::BF16 | DType::F8E4M3 => {
                    self.cuda_fwd_typed_q8::<bf16, bf16>(q, q_l, run_paged_decode_bf16_q8)
                }
                dt => candle::bail!("paged-decode q8: unsupported arena dtype {:?}", dt),
            };
        }

        use candle_kernels::paged_decode::{run_paged_decode_bf16, run_paged_decode_fp16};
        match self.arena_dtype {
            DType::F16 => self.cuda_fwd_typed::<f16, f16, f16>(q, q_l, run_paged_decode_fp16),
            DType::BF16 | DType::F8E4M3 => {
                self.cuda_fwd_typed::<bf16, bf16, bf16>(q, q_l, run_paged_decode_bf16)
            }
            dt => candle::bail!(
                "paged-decode: unsupported arena dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::paged_prefill_batched as paged_prefill_flat;

    /// Serialize the GPU tests: the split-KV launcher's grow-on-demand
    /// partial pool is a function-local static sized for the production
    /// single-scheduler-thread model — concurrent test launches interleave
    /// its free/realloc. Same idiom as the prefill_ab harness's guard.
    fn gpu_serial() -> std::sync::MutexGuard<'static, ()> {
        static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
        M.lock().unwrap_or_else(|e| e.into_inner())
    }
    use candle::quantized::pinned_staging::{Generation, PinnedStager};
    use candle::{DType, Device, Result, Tensor};
    use candle_nn::kv_cache::KvCache;

    /// Test shim mapping the legacy uniform calling convention (q/k/v as
    /// `[b_sz, n_head, seq_len, head_dim]`, returning per-sequence
    /// `[1, n_head, seq_len, head_dim]`) onto the production flat/ragged
    /// [`paged_prefill_flat`]. Lets the kernel-correctness tests keep their
    /// original shapes/assertions while the production API is flat-packed.
    #[allow(clippy::too_many_arguments)]
    fn paged_prefill_uniform(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
        rope_offsets: &Tensor,
        rope_cs: &Tensor,
        rope_interleaved: bool,
        generation: &Generation,
    ) -> Result<Vec<Tensor>> {
        let total_q = b_sz * seq_len;
        let q_flat = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_head, head_dim))?;
        let k_flat = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let v_flat = v
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let q_lens = vec![seq_len; b_sz];
        let out = paged_prefill_flat(
            caches,
            offsets,
            &q_flat,
            &k_flat,
            &v_flat,
            b_sz,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope_offsets,
            rope_cs,
            rope_interleaved,
            generation,
            &std::cell::RefCell::new(None),
        )?;
        // Flat [total_q, n_head, head_dim] -> per-seq [1, n_head, seq_len, head_dim].
        let mut per_seq = Vec::with_capacity(b_sz);
        for i in 0..b_sz {
            per_seq.push(
                out.narrow(0, i * seq_len, seq_len)?
                    .reshape((1, seq_len, n_head, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        Ok(per_seq)
    }

    /// Helper: create a standard inv_freq tensor for tests (theta = 10000.0).
    /// Shape: (head_dim / 2,), dtype F32, on the given device.
    fn make_test_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        Tensor::from_vec(inv_freq, (half_dim,), device)
    }

    /// Zero inv_freq for correctness tests: RoPE rotation becomes identity so the
    /// paged-kernel output can be compared directly to a no-RoPE reference.
    fn make_zero_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        Tensor::zeros((head_dim / 2,), DType::F32, device)
    }

    /// Build a rope_cs table from inv_freq for decode tests.
    #[allow(dead_code)]
    fn make_test_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        let inv_freq = make_test_inv_freq(head_dim, device)?;
        super::compute_rope_cs(&inv_freq, max_blocks, head_dim, device)
    }

    /// Zero rope_cs for correctness tests (identity RoPE).
    #[allow(dead_code)]
    fn make_zero_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        let inv_freq = make_zero_inv_freq(head_dim, device)?;
        super::compute_rope_cs(&inv_freq, max_blocks, head_dim, device)
    }

    #[test]
    #[cfg(feature = "flash-attn")]
    fn paged_prefill_uses_flash_attn_kernel_smoke() -> Result<()> {
        // This forces a multi-token BF16 prefill so the flash-attn paged-prefill path is eligible.
        let device = Device::new_cuda(0)?;

        let b_sz = 1usize;
        let seq_len = 2usize;
        let n_head = 1usize;
        let n_kv_head = 1usize;
        let head_dim = 64usize;

        // Shapes: q (B, H, T, D), k/v (B, H_kv, T, D)
        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::BF16);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];

        // Enable the trace so the test output proves which kernel ran.
        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
        let generation = PinnedStager::new(device.as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_test_rope_cs(head_dim, 16, &device)?,
            false, // rope_interleaved
            &generation,
        )
        .expect("expected paged prefill to be applicable");

        assert_eq!(out.len(), 1);
        let y = &out[0];
        assert_eq!(y.dims(), &[1, n_head, seq_len, head_dim]);
        // Output dtype matches input dtype (BF16 in, BF16 out)
        assert_eq!(y.dtype(), DType::BF16);

        // Basic sanity: output should be finite.
        let max_abs = y
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite());

        Ok(())
    }

    // NOTE: there are intentionally no "fp8 paged prefill" tests. Prefill arenas
    // are always float (see chunked/io.rs: "Quantization only happens via
    // reconcile_sealed after chunks are complete"), and the prefill compute dtype
    // is the cache's reported dtype — F16/BF16 only (cuda_fwd rejects anything
    // else). FP8 is a post-seal KV *storage* format, exercised on read by the
    // fp8 paged-decode tests (decode_utils.rs: test_fp8_hd{64,128}_paged_decode),
    // not a prefill compute dtype. Forcing a cache to F8E4M3 before prefill is a
    // configuration production never uses.

    // ========================================================================
    // Correctness comparison tests: paged kernels vs reference matmul attention
    // ========================================================================

    /// Gold-standard causal attention via matmul.
    ///
    /// q: (1, n_head, seq_len, head_dim)
    /// k: (1, n_kv_head, kv_len, head_dim)  ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â full KV including prefix
    /// v: (1, n_kv_head, kv_len, head_dim)
    /// offset: number of prefix tokens (causal mask allows attending to prefix + up to current pos)
    ///
    /// Returns: (1, n_head, seq_len, head_dim) in F32
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let seq_len = q.dim(2)?;
        let kv_len = k.dim(2)?;

        // Convert everything to F32 for reference precision
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Repeat KV heads for GQA/MQA
        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // Causal mask: position i in the query (at absolute position offset+i)
        // can attend to KV positions 0..=offset+i
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), q.device())?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        Ok(out)
    }

    /// Reference decode attention: single query token against full KV cache.
    ///
    /// q: (batch_size, n_head, head_dim)
    /// k: (batch_size, n_kv_head, kv_len, head_dim)
    /// v: (batch_size, n_kv_head, kv_len, head_dim)
    ///
    /// Returns: (batch_size, n_head, head_dim) in F32
    #[allow(dead_code)]
    fn reference_decode_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let _batch_size = q.dim(0)?;
        let _kv_len = k.dim(2)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        // q: (batch, n_head, head_dim) -> (batch, n_head, 1, head_dim)
        let q = q.unsqueeze(2)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // No causal mask needed for decode ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all KV positions are valid
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // (batch, n_head, 1, head_dim)
        let out = out.squeeze(2)?; // (batch, n_head, head_dim)
        Ok(out)
    }

    /// Compute max absolute error between two tensors (both converted to F32).
    fn max_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.max(0)?.to_vec0::<f32>()
    }

    /// Compute mean absolute error between two tensors (both converted to F32).
    fn mean_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.mean_all()?.to_vec0::<f32>()
    }

    /// Helper: run paged prefill and return output in (1, n_head, seq_len, head_dim) shape.
    fn run_paged_prefill(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        let offsets = [offset];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, q.device())?;
        let generation = PinnedStager::new(q.device().as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_zero_rope_cs(head_dim, 16, q.device())?,
            false, // rope_interleaved
            &generation,
        )?;
        assert_eq!(out.len(), 1);
        Ok(out[0].clone())
    }

    /// Regression test for the int8 m16n8k32 MMA fragment loaders.
    ///
    /// Feeds known int8 `A` (16×32) and `B` (8×32) through the exact loaders the
    /// decode QK dot uses (`load_a_frag_m16k32` / `load_b_frag_n8k32` /
    /// `mma_int8_m16n8k32`) and asserts the int32 `C = A·Bᵀ` matches a trivial
    /// CPU reference — exactly, no tolerance. Guards against the
    /// `load_b_frag_n8k32` lane-decomposition bug that silently corrupted every
    /// int8 QK dot, and any future MMA-fragment regression. Fast (one warp, no
    /// model), so it runs on every `cargo test`.
    #[test]
    fn int8_mma_m16n8k32_fragment_layout() -> Result<()> {
        let _gpu = gpu_serial();
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();

        // Deterministic int8 inputs spanning a range of magnitudes and signs.
        let mut a = vec![0i8; 16 * 32];
        let mut b = vec![0i8; 8 * 32];
        for (i, x) in a.iter_mut().enumerate() {
            *x = ((i * 31 + 7) % 17) as i8 - 8;
        }
        for (i, x) in b.iter_mut().enumerate() {
            *x = ((i * 13 + 5) % 19) as i8 - 9;
        }
        // Reference: C[m][n] = Σ_k A[m][k]·B[n][k] (int32, exact).
        let mut c_ref = vec![0i32; 16 * 8];
        for m in 0..16 {
            for n in 0..8 {
                let mut s = 0i32;
                for k in 0..32 {
                    s += a[m * 32 + k] as i32 * b[n * 32 + k] as i32;
                }
                c_ref[m * 8 + n] = s;
            }
        }

        let a_slice = cuda.memcpy_stod(&a)?;
        let b_slice = cuda.memcpy_stod(&b)?;
        let c_slice = cuda.alloc_zeros::<i32>(16 * 8)?;
        unsafe {
            let (a_ptr, _ga) = a_slice.device_ptr(&stream);
            let (b_ptr, _gb) = b_slice.device_ptr(&stream);
            let (c_ptr, _gc) = c_slice.device_ptr(&stream);
            candle_kernels::paged_decode::mma_int8_m16n8k32_test(
                a_ptr as *const i8,
                b_ptr as *const i8,
                c_ptr as *mut i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            );
        }
        device.synchronize()?;
        let c_gpu = cuda.memcpy_dtov(&c_slice)?;
        assert_eq!(
            c_gpu, c_ref,
            "int8 m16n8k32 MMA fragment layout regressed (load_a/load_b/mma)"
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prefill correctness: no prefix (offset=0)
    // ------------------------------------------------------------------

    #[test]
    fn correctness_prefill_no_prefix_bf16() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (8, 8, 64, 128, "MHA hd64 long"),
            (8, 8, 128, 4, "MHA hd128 short"),
            (8, 8, 128, 64, "MHA hd128 long"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 64, 128, "GQA 32/8 hd64 long"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (32, 8, 128, 64, "GQA 32/8 hd128 long"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"), // num_groups=5, tricky for WARPS_TC
            (40, 8, 64, 32, "GQA 40/8 hd64 medium"),
            (40, 8, 128, 4, "GQA 40/8 hd128 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"), // Qwen2-style
            (14, 2, 64, 32, "GQA 14/2 hd64 medium"),
            (14, 2, 128, 4, "GQA 14/2 hd128 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
            (8, 1, 64, 32, "MQA hd64 medium"),
            (8, 1, 128, 4, "MQA hd128 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;

            // Reference: for no-prefix, full KV == the input K/V
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // BF16 tolerance: mean ~1e-2, max ~5e-2
            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    #[test]
    fn correctness_prefill_no_prefix_f16() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // F16 tolerance ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â slightly tighter since F16 has more mantissa bits than BF16
            assert!(
                mae < 0.05,
                "[{label}] F16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prefill correctness: with prefix (offset > 0, multi-turn)
    // ------------------------------------------------------------------

    #[test]
    fn correctness_prefill_with_prefix_bf16() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (8, 8, 64, 32, 8, "MHA hd64 prefix=32"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 64, 64, 16, "GQA 32/8 hd64 prefix=64"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
            (8, 1, 64, 10, 4, "MQA hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            // Create K/V for prefix (already in cache) and new segment
            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            // Set up chunked KV cache and write prefix data
            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &rope_zeros,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                false, // rope_interleaved
                &generation,
            )?;
            assert_eq!(out.len(), 1);
            let paged_out = &out[0];

            // Reference: full KV = prefix + new
            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Regression: prefix palette ROUTING across mixed per-slice maps.
    ///
    /// The prefix is a sealed partial chunk 0 (`p` tokens, identity palette
    /// map) followed by a full chunk 1 whose palette map is SHUFFLED
    /// (dim `d` → palette `d % 4`, rank `d / 4`) with its data permuted to
    /// match, so a correct reader reconstructs the same logical values.
    /// Each slice-aligned tile must be routed with ITS OWN slice's map:
    /// routing chunk 1's positions through chunk 0's map (a stale cached
    /// table) gathers a dim-permutation of the true values — order-1
    /// garbage on random data. Float chunks keep the check
    /// quantization-free, so the tolerance is tight fp16 noise and any
    /// routing regression fails hard. The partial chunk 0 also exercises
    /// the gap walk ahead of the shuffled map.
    #[test]
    fn correctness_prefill_straddle_shuffled_pal_map() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        const CHUNK: usize = 32;
        let (n_head, n_kv_head, head_dim) = (32usize, 8usize, 128usize);
        let p = 20usize; // partial chunk 0 → a gap before every later slice
        let new_len = 8usize;
        let prefix_len = p + CHUNK;
        let b_sz = 1;

        // Logical prefix K/V and the new segment.
        let k0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, p, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, p, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let k1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, CHUNK, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, CHUNK, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let new_q =
            Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?.to_dtype(dtype)?;
        let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Shuffled map: dim d → palette d % 4 (balanced: 32 dims per palette,
        // maximally different from identity's d / 32). The write path places
        // logical dim d' at (palette d'/32, rank d'%32) — identity placement —
        // so for logical dim d to land where the shuffled map expects it
        // (palette d%4, rank d/4), permute the head_dim axis by
        // d ↦ (d % 4) * 32 + d / 4 before writing.
        let mut inv = vec![0u32; head_dim];
        for d in 0..head_dim {
            inv[(d % 4) * 32 + d / 4] = d as u32;
        }
        let inv_t = Tensor::from_vec(inv, head_dim, &device)?;
        let k1p = k1.index_select(&inv_t, 3)?.contiguous()?;
        let v1p = v1.index_select(&inv_t, 3)?.contiguous()?;

        // Packed 2-bit shuffled map, repeated per head.
        let mut shuf = vec![0u8; head_dim / 4];
        for d in 0..head_dim {
            shuf[d / 4] |= ((d % 4) as u8) << ((d % 4) * 2);
        }
        let pal_all: Vec<u8> = (0..n_kv_head).flat_map(|_| shuf.iter().copied()).collect();

        // Backing: chunk 0 = sealed partial (identity map), chunk 1 = full
        // (shuffled map + permuted data), chunk 2 = empty writer.
        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, 3 * CHUNK)?;
        backing.ensure_for_offset(0, 0, 2 * CHUNK + new_len)?;
        backing.write_contiguous(0, 0, &k0, &v0)?;
        backing.write_contiguous(0, CHUNK, &k1p, &v1p)?;
        backing.set_block_window(0, 0, 0, p as u32)?;
        backing.set_block_window(0, 1, 0, CHUNK as u32)?;
        backing.test_set_block_palette(0, 1, pal_all.clone(), pal_all, Vec::new(), Vec::new())?;
        backing.test_set_writer_start(0, 2)?;

        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;
        cache0.set_current_seq_len(prefix_len)?;

        let offsets = [prefix_len];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
        let generation = backing.begin_stager_generation_required();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &new_q,
            &new_k,
            &new_v,
            b_sz,
            new_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_zero_rope_cs(head_dim, 16, &device)?,
            false, // rope_interleaved
            &generation,
        )?;
        assert_eq!(out.len(), 1);
        let paged_out = &out[0];

        // Reference attends the LOGICAL prefix (un-permuted) + new segment.
        let full_k = Tensor::cat(&[&k0, &k1, &new_k], 2)?;
        let full_v = Tensor::cat(&[&v0, &v1, &new_v], 2)?;
        let ref_out = reference_attention(
            &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
        )?;

        let paged_f32 = paged_out.to_dtype(DType::F32)?;
        let mae = mean_abs_error(&paged_f32, &ref_out)?;
        let max_err = max_abs_error(&paged_f32, &ref_out)?;
        assert!(
            mae < 0.05,
            "straddle shuffled-pal-map prefill mean error too large: {mae} — \
             the prefix tile loader routed a straddle slice through the wrong \
             palette table"
        );
        assert!(
            max_err < 0.2,
            "straddle shuffled-pal-map prefill max error too large: {max_err}"
        );
        println!("straddle shuffled-pal-map prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        Ok(())
    }

    #[test]
    fn correctness_prefill_with_prefix_f16() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &rope_zeros,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                false, // rope_interleaved
                &generation,
            )?;
            let paged_out = &out[0];

            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] F16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Decode correctness: prefill then single-token decode
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // GQA head mapping regression test (validates the kv_group_end fix)
    // ------------------------------------------------------------------

    /// This test specifically targets the GQA head mapping bug where WARPS_TC
    /// could overflow into the next KV group. It exercises configurations where
    /// num_groups is not a multiple of WARPS_TC (e.g., 40/8 = 5 groups).
    #[test]
    fn correctness_prefill_gqa_head_mapping_regression() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // These configurations are specifically chosen to trigger the GQA overflow bug:
        // - 40/8: num_groups=5, with WARPS_TC=2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ head_blocks_per_kv=3, last block has 1 warp active
        // - 28/4: num_groups=7, tricky for WARPS_TC alignment
        // - 48/8: num_groups=6, with WARPS_TC=2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ head_blocks_per_kv=3, all warps active
        // - 56/8: num_groups=7, same as 28/4 pattern
        // - 12/4: num_groups=3
        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (40, 8, 64, 4, "40/8 hd64 short"),
            (40, 8, 64, 32, "40/8 hd64 medium"),
            (40, 8, 64, 128, "40/8 hd64 long"),
            (40, 8, 128, 4, "40/8 hd128 short"),
            (40, 8, 128, 32, "40/8 hd128 medium"),
            (28, 4, 64, 4, "28/4 hd64 short"),
            (28, 4, 64, 32, "28/4 hd64 medium"),
            (48, 8, 64, 4, "48/8 hd64 short"),
            (48, 8, 64, 32, "48/8 hd64 medium"),
            (56, 8, 64, 4, "56/8 hd64 short"),
            (56, 8, 128, 4, "56/8 hd128 short"),
            (12, 4, 64, 4, "12/4 hd64 short"),
            (12, 4, 64, 32, "12/4 hd64 medium"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] GQA regression: mean error too large: {mae} (may indicate head mapping overflow)"
            );
            assert!(
                max_err < 0.2,
                "[{label}] GQA regression: max error too large: {max_err} (may indicate head mapping overflow)"
            );
            println!("[{label}] GQA regression OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Diagnostic test that prints per-head errors to help identify specific
    /// GQA group cross-contamination patterns.
    #[test]
    fn correctness_prefill_diagnostic_per_head() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // Use a config known to be tricky: 40 heads / 8 KV heads = 5 groups
        let n_head = 40;
        let n_kv_head = 8;
        let head_dim = 64;
        let seq_len = 8;
        let b_sz = 1;

        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let paged_out = run_paged_prefill(
            &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
        )?;
        let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

        let paged_f32 = paged_out.to_dtype(DType::F32)?;
        let num_groups = n_head / n_kv_head;

        println!("\n=== Per-head error for {n_head}/{n_kv_head} (groups={num_groups}) ===");
        for h in 0..n_head {
            let kv_group = h / num_groups;
            let paged_h = paged_f32.narrow(1, h, 1)?;
            let ref_h = ref_out.narrow(1, h, 1)?;
            let mae = mean_abs_error(&paged_h, &ref_h)?;
            let marker = if mae > 0.05 { " *** HIGH ***" } else { "" };
            println!("  head {h:2} (kv_group {kv_group}): mae={mae:.4e}{marker}");
        }

        let overall_mae = mean_abs_error(&paged_f32, &ref_out)?;
        assert!(
            overall_mae < 0.05,
            "Per-head diagnostic: overall mean error too large: {overall_mae}"
        );
        Ok(())
    }

    // ========================================================================
    // RoPE offset plumbing tests
    // ========================================================================
    //
    // These tests verify that the rope_offsets parameter is correctly plumbed
    // from the public API all the way to the CUDA kernel entry point without
    // causing panics or incorrect output.
    //
    // Since the kernels treat rope_offsets=nullptr as offset=0 (a no-op), and
    // rope_offsets=Some(zeros) also means offset=0, the two must produce
    // numerically identical results.

    /// Helper: run a single-batch prefill and return the output tensor.
    fn run_prefill_with_rope(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
        rope: &Tensor,
    ) -> candle::Result<Tensor> {
        use candle_nn::kv_cache::ChunkedKvBacking;
        let device = q.device();
        let b_sz = 1;

        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, device, seq_len)?;
        backing.ensure_for_offset(0, 0, seq_len)?;

        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;

        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let generation = backing.begin_stager_generation_required();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope,
            &make_test_rope_cs(head_dim, 16, device)?,
            false, // rope_interleaved
            &generation,
        )?;
        Ok(out.into_iter().next().unwrap())
    }

    #[test]
    fn rope_offset_prefill_none_succeeds() -> candle::Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (8, 8, 64, 16);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;
        assert_eq!(out.dims(), &[1, n_head, seq_len, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_abs.is_finite(),
            "rope=zeros output not finite: {max_abs}"
        );
        println!("rope_offset_prefill_none_succeeds OK: max_abs={max_abs:.4e}");
        Ok(())
    }

    #[test]
    fn rope_offset_prefill_zeros_differs_from_none() -> candle::Result<()> {
        let _gpu = gpu_serial();
        // Now that both paths always apply RoPE, this test confirms that different rope_offsets
        // values produce different outputs.
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (8, 8, 64, 16);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Run with rope=zeros (offset=0)
        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out_zeros = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;

        // Run with rope=offset16 â€” different base offset produces different rotation
        let rope_offset16 = Tensor::from_vec(vec![16u32], 1, &device)?;
        let out_offset16 = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_offset16,
        )?;

        let zeros_f32 = out_zeros.to_dtype(DType::F32)?;
        let offset16_f32 = out_offset16.to_dtype(DType::F32)?;

        // Both outputs must be finite
        let max_zeros = zeros_f32.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let max_offset16 = offset16_f32
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_zeros.is_finite(), "rope=zeros output not finite");
        assert!(max_offset16.is_finite(), "rope=offset16 output not finite");

        let mae = mean_abs_error(&zeros_f32, &offset16_f32)?;
        println!("rope_offset_prefill_zeros_differs_from_none: mae={mae:.4e}");
        // Different rope offsets must produce different outputs
        // RoPE preserves relative positions: shifting all token positions by the same constant
        // leaves the relative-position attention map unchanged. The two runs should be equivalent.
        assert!(
            mae < 1e-2,
            "rope=zeros and rope=offset16 should produce equivalent attention outputs \
             (same relative positions; mae={mae:.4e})"
        );
        Ok(())
    }

    // Reference RoPE rotation: apply to f32 slice of HEAD_DIM values.
    // Pairs: dim d with dim d+half_dim.
    fn apply_rope_to_vec(x: &[f32], head_dim: usize, pos: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = head_dim / 2;
        for d in 0..half {
            let theta = pos as f32 * 10000.0f32.powf(-2.0 * d as f32 / head_dim as f32);
            let (sin_v, cos_v) = theta.sin_cos();
            let x_lo = x[d];
            let x_hi = x[d + half];
            out[d] = x_lo * cos_v - x_hi * sin_v;
            out[d + half] = x_lo * sin_v + x_hi * cos_v;
        }
        out
    }

    #[test]
    fn rope_offset_prefill_functional() -> candle::Result<()> {
        let _gpu = gpu_serial();
        // Verifies: reference_attention(rotated_Q, rotated_K, V) â‰ˆ paged_kernel(unrotated_Q, unrotated_K, V, rope=zeros)
        // Tolerance < 0.05 for BF16.
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (4, 4, 64, 8);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Manually rotate [1, n_head, seq_len, head_dim]: token t -> pos t
        let apply_rope_4d = |t: &Tensor| -> candle::Result<Tensor> {
            let (b, h, s, d) = t.dims4()?;
            // Reshape to [b*h, s, d] and back so we can use to_vec3
            let flat = t.reshape((b * h, s, d))?;
            let data = flat.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            let mut out: Vec<f32> = Vec::with_capacity(b * h * s * d);
            for bh in 0..(b * h) {
                for si in 0..s {
                    out.extend_from_slice(&apply_rope_to_vec(&data[bh][si], d, si));
                }
            }
            Tensor::from_vec(out, (b * h, s, d), t.device())?
                .reshape((b, h, s, d))?
                .to_dtype(t.dtype())
        };

        // Pre-rotated versions
        let q_rotated = apply_rope_4d(&q)?.contiguous()?;
        let k_rotated = apply_rope_4d(&k)?.contiguous()?;

        // Branch A (reference): reference_attention with pre-rotated Q/K
        let ref_out =
            reference_attention(&q_rotated, &k_rotated, &v, n_head, n_kv_head, head_dim, 0)?;

        // Branch B: un-rotated Q, K  +  rope=zeros  (kernel rotates at tok_idx + 0)
        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out_kernel_rope = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;

        let mae = mean_abs_error(
            &ref_out.to_dtype(DType::F32)?,
            &out_kernel_rope.to_dtype(DType::F32)?,
        )?;
        println!(
            "rope_offset_prefill_functional: mae(reference_attention(rotated) vs kernel+rope0)={mae:.4e}"
        );
        // BF16 rounding: allow up to 0.05 (two roundings: manual rotation + kernel)
        assert!(
            mae < 0.05,
            "fused RoPE prefill functional mismatch: mae={mae}"
        );
        Ok(())
    }
}

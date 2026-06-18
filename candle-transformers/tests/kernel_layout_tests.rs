//! Kernel-isolation tests: do `paged-prefill` and `paged-decode` produce
//! the same output when the slot's K/V content sits at identical logical
//! positions but in different chunk shapes?
//!
//! Strategy
//! ────────
//! Each test case picks a **segment plan** — a list of segment sizes
//! summing to some `TOTAL`.  Two slots are then built:
//!
//! * **Control slot A** — one big prefill of all `TOTAL` tokens.
//!   Chunks land in the natural packed layout (every chunk full except
//!   possibly the last).
//! * **Test slot B** — same logical Q/K/V at the same logical
//!   positions, but built segment-by-segment: each segment is prefilled
//!   into a scratch slot, its sealed bytes are injected into slot B,
//!   and the next segment starts in a fresh chunk.  The resulting
//!   layout has whatever pattern of partial / full chunks the segment
//!   plan specifies (partial at start, partial in middle, partial at
//!   end, many tiny chunks, etc.).
//!
//! Then both slots are exercised with the same kernels:
//!
//! 1. **Decode**: same fresh Q feeds `paged_decode_attn` on each slot.
//!    Outputs are compared.  Any divergence isolates a bug to the
//!    decode kernel's read path under non-trivial chunk layouts.
//! 2. **Prefill (incremental)**: same fresh `EXTRA` Q/K/V is prefilled
//!    on each slot via `paged_prefill_batched`.  Outputs are compared.
//!    Any divergence isolates a bug to the prefill kernel's read path.
//!
//! All host-side metadata construction is shared — the same
//! `paged_prefill_batched` / `paged_decode_attn` wrappers and the same
//! `SlotStateHost::from_sealed_chunks` + position_map code that the
//! production scheduler uses.  Divergence therefore points at the
//! kernel, not the host.

#![cfg(feature = "cuda")]

use candle::quantized::pinned_staging::PinnedStager;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{ChunkedKvBacking, KvCache, CHUNK_SIZE};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_prefill_batched,
};

// ──────────────────────────────────────────────────────────────────────
// Test-config knobs
// ──────────────────────────────────────────────────────────────────────

const N_KV_HEAD: usize = 4;
const N_HEAD: usize = 4; // No GQA — keeps the test simple
const HEAD_DIM: usize = 64;
const MAX_BLOCKS: usize = 64; // Headroom for the larger layouts
const EXTRA_PREFILL_TOKENS: usize = 8;

/// fp16 attention has ~1e-3 numeric error per dot; the partial-chunk
/// layout shouldn't add anything visible on top.  Treat anything
/// >= 1e-2 as a real kernel divergence.
const DIFF_TOLERANCE: f32 = 1e-2;

// ──────────────────────────────────────────────────────────────────────
// Test cases — segment plans for slot B
// ──────────────────────────────────────────────────────────────────────

/// Each case is `(name, segments)`.  `segments` sums to the slot's
/// logical token count and dictates slot B's chunk layout.
/// Slot A always uses one big prefill for the same total.
const TEST_CASES: &[(&str, &[usize])] = &[
    // ── Tiny / minimal sanity ───────────────────────────────────────
    (
        "single_partial_only",
        // One short partial.  Both slots end up with [usage<32] at
        // chunk 0.  Tests that the trivial case still matches.
        &[7],
    ),
    (
        "single_full_only",
        // Exactly one full chunk.  Slot A and slot B should be
        // identical: [32].
        &[32],
    ),
    // ── Partial position scan ───────────────────────────────────────
    (
        "partial_at_start_then_full",
        // Slot B: [7, 32, 32, 32].  Control: [32, 32, 32, 7].
        &[7, 32, 32, 32],
    ),
    (
        "partial_at_end_only",
        // Slot B: [32, 32, 32, 25].  Same as the control's natural
        // packing — this is the "everything aligned" case, used as a
        // baseline that should pass even with no kernel fix.
        &[32, 32, 32, 25],
    ),
    (
        "partial_in_middle_only",
        // Slot B: [32, 7, 32, 32, 25].  Partial chunk surrounded by
        // full ones.  Tile boundaries cross the partial in both
        // directions (full→partial and partial→full).
        &[32, 7, 32, 32, 25],
    ),
    (
        "partial_at_start_and_end",
        // Slot B: [7, 32, 32, 32, 25].  The original failing case.
        &[7, 32, 32, 32, 25],
    ),
    (
        "multiple_partials_mixed",
        // Slot B: [5, 17, 32, 11, 32, 25, 6].  Several small partials
        // interleaved with full chunks.  Stresses palette/iter
        // re-init on every other tile.
        &[5, 17, 32, 11, 32, 25, 6],
    ),
    // ── Adjacent-partials stress ────────────────────────────────────
    (
        "two_adjacent_partials_then_full",
        // Slot B: [7, 5, 32, 32, 24].  Two partials side-by-side at
        // the start — a single tile may straddle three slices when
        // partials are smaller than the tile width.
        &[7, 5, 32, 32, 24],
    ),
    (
        "many_small_partials",
        // Slot B: [3, 3, 3, 3, 3, 3, 3, 3, 4].  Worst case for
        // tile-spans-multiple-slices: each tile of WARPS_PER_BLOCK
        // positions crosses many slice boundaries.
        &[3, 3, 3, 3, 3, 3, 3, 3, 4],
    ),
    // ── Larger slot ─────────────────────────────────────────────────
    (
        "larger_slot_with_partials",
        // 9 chunks instead of 5.  Same overall pattern as the
        // original failing case but on a longer slot.  Sum: 7 + 7*32
        // + 11 = 242.
        &[7, 32, 32, 32, 32, 32, 32, 32, 11],
    ),
    (
        "long_run_of_full_then_partial",
        // Sum: 14 × 32 + 9 = 457.  Stress the kernel's read scan
        // length and the very last partial.
        &[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 9],
    ),
];

// ──────────────────────────────────────────────────────────────────────
// Test entry points
// ──────────────────────────────────────────────────────────────────────

#[test]
fn kernel_layout_decode_matches_full_vs_partial() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, segments) in TEST_CASES {
        match run_decode_case(name, segments, &device, &stager) {
            Ok(diff) => eprintln!("decode {name:32}  diff = {diff:.6e}"),
            Err(e) => {
                eprintln!("decode {name:32}  FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "decode kernel divergence in {} layout(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn kernel_layout_prefill_matches_full_vs_partial() -> Result<()> {
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, segments) in TEST_CASES {
        match run_prefill_case(name, segments, &device, &stager) {
            Ok(diff) => eprintln!("prefill {name:32} diff = {diff:.6e}"),
            Err(e) => {
                eprintln!("prefill {name:32} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "prefill kernel divergence in {} layout(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

/// Byte-exact gate for the prefill position_map hoist.
///
/// The per-forward `SharedPm` cache reuses the first layer's uploaded
/// position_map for every later layer. That is sound only if two invariants of
/// `from_sealed_chunks` hold, which this test pins as raw-byte equalities:
///   1. The serialized slices (out-of-line KvHead records + 16-byte slice
///      headers) — the part re-uploaded per layer — are byte-identical whether
///      or not the position_map is built. So later layers can rebuild slices
///      while reusing the first layer's position_map without disagreement.
///   2. The position_map is a deterministic function of the chunk token layout
///      (which is identical across every layer of a forward), before and after
///      the write-region extension the prefill path applies.
#[test]
fn prefill_position_map_hoist_is_byte_exact() -> Result<()> {
    use candle_transformers::models::slot_state::{SlotStateHost, TokenSliceHost};
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);

    // Multi-chunk slot so slices + position_map are non-trivial.
    let segments: &[usize] = &[40, 40, 24];
    let total: usize = segments.iter().sum();
    let (q_master, k_master, v_master) = make_qkv(total, &device, 0xB17E_EAC7)?;
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, &device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, &device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, &device)?;
    let (backing, cache) = build_segmented_slot(
        segments,
        &q_master,
        &k_master,
        &v_master,
        &rope_cs,
        &rope_offsets_b1,
        &stager,
        &device,
    )?;

    let arena_info = backing.resolve_arena_info()?;
    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);

    // Serialize a slot's per-layer payload with slice headers rebased onto a
    // fixed origin, so the comparison is independent of allocation address.
    let serialize_slices = |slot: &SlotStateHost| -> Vec<u8> {
        let rec_bytes = TokenSliceHost::record_size(N_KV_HEAD, HEAD_DIM);
        let mut buf = Vec::new();
        for (i, s) in slot.slices.iter().enumerate() {
            s.serialize_record(&mut buf);
            s.serialize_slice_header(&mut buf, (i * rec_bytes) as u64);
        }
        buf
    };

    let with_pm = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    let no_pm = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        false,
    );

    // (1) The per-layer slices must not depend on whether the map was built.
    assert_eq!(
        serialize_slices(&with_pm),
        serialize_slices(&no_pm),
        "slice bytes must be independent of build_position_map",
    );
    assert!(
        !with_pm.position_map.is_empty(),
        "expected a non-empty position_map when requested",
    );
    assert!(
        no_pm.position_map.is_empty(),
        "expected an empty position_map when skipped",
    );

    // (2) Deterministic build — the layer-invariance the cache relies on.
    let mut a = with_pm;
    let mut b = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    assert_eq!(
        a.position_map, b.position_map,
        "position_map must be deterministic across builds",
    );
    a.extend_for_write_region(7, CHUNK_SIZE);
    b.extend_for_write_region(7, CHUNK_SIZE);
    assert_eq!(
        a.position_map, b.position_map,
        "write-region-extended position_map must be deterministic",
    );

    let _ = &backing;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Per-case runners
// ──────────────────────────────────────────────────────────────────────

fn run_decode_case(
    case_name: &str,
    segments: &[usize],
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let total: usize = segments.iter().sum();
    let (q_master, k_master, v_master) = make_qkv(total, device, hash_str(case_name))?;
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;

    let (backing_a, mut cache_a) = build_control_slot(
        total,
        &q_master,
        &k_master,
        &v_master,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    let (backing_b, cache_b) = build_segmented_slot(
        segments,
        &q_master,
        &k_master,
        &v_master,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;

    assert_slot_layouts(&cache_a, &cache_b, total, segments, case_name);

    // Single decode step with the same fresh Q on both slots.
    let (q_dec, k_new, v_new) = make_qkv(1, device, hash_str(case_name) ^ 0xD3C0DE)?;
    let q_dec_2d = q_dec.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let k_new_2d = k_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let v_new_2d = v_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;

    let out_a = decode_one_slot(
        &backing_a, &cache_a, &q_dec_2d, &k_new_2d, &v_new_2d, &rope_cs, stager, device,
    )?;
    let out_b = decode_one_slot(
        &backing_b, &cache_b, &q_dec_2d, &k_new_2d, &v_new_2d, &rope_cs, stager, device,
    )?;

    // Suppress unused warnings for cache_a — we keep it alive via the
    // backing reference.
    let _ = &mut cache_a;

    let diff = max_abs_diff_f32(&out_a.to_dtype(DType::F32)?, &out_b.to_dtype(DType::F32)?)?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "decode kernel diverged on layout {:?}: max abs diff = {:.6e} (expected < {:.0e})",
            segments,
            diff,
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

fn run_prefill_case(
    case_name: &str,
    segments: &[usize],
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let total: usize = segments.iter().sum();
    let (q_master, k_master, v_master) = make_qkv(total, device, hash_str(case_name))?;
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;

    let (backing_a, mut cache_a) = build_control_slot(
        total,
        &q_master,
        &k_master,
        &v_master,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    let (backing_b, mut cache_b) = build_segmented_slot(
        segments,
        &q_master,
        &k_master,
        &v_master,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    assert_slot_layouts(&cache_a, &cache_b, total, segments, case_name);

    // Run a fresh prefill of EXTRA tokens on both slots and compare
    // the resulting attention output.  The Q/K/V are the same; the K
    // scan covers everything written before + the new range.
    let (q_ext, k_ext, v_ext) =
        make_qkv(EXTRA_PREFILL_TOKENS, device, hash_str(case_name) ^ 0xEFEFEF)?;
    let out_a = run_prefill(
        &mut cache_a,
        &q_ext,
        &k_ext,
        &v_ext,
        EXTRA_PREFILL_TOKENS,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    let out_b = run_prefill(
        &mut cache_b,
        &q_ext,
        &k_ext,
        &v_ext,
        EXTRA_PREFILL_TOKENS,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    // Keep backings alive through the kernel; the cache borrows from them.
    let _ = &backing_a;
    let _ = &backing_b;

    let diff = max_abs_diff_f32(&out_a.to_dtype(DType::F32)?, &out_b.to_dtype(DType::F32)?)?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "prefill kernel diverged on layout {:?}: max abs diff = {:.6e} (expected < {:.0e})",
            segments,
            diff,
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

// ──────────────────────────────────────────────────────────────────────
// Slot builders
// ──────────────────────────────────────────────────────────────────────

/// Slot A: one big prefill of `total` tokens into a fresh backing.
fn build_control_slot(
    total: usize,
    q_master: &Tensor,
    k_master: &Tensor,
    v_master: &Tensor,
    rope_cs: &Tensor,
    rope_offsets_b1: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing = fresh_backing(device)?;
    let mut cache = bind_kv_cache(&backing, 0)?;
    let q = q_master.narrow(2, 0, total)?.contiguous()?;
    let k = k_master.narrow(2, 0, total)?.contiguous()?;
    let v = v_master.narrow(2, 0, total)?.contiguous()?;
    let _ = run_prefill(
        &mut cache,
        &q,
        &k,
        &v,
        total,
        rope_cs,
        rope_offsets_b1,
        stager,
        device,
    )?;
    Ok((backing, cache))
}

/// Slot B: build the segment-by-segment layout dictated by `segments`.
///
/// Each segment is prefilled into a scratch slot at state.offset=0,
/// its sealed chunks are captured and injected into slot B (preserving
/// the segment's partial-tail usage), and the next segment starts at
/// the next chunk boundary in slot B.
fn build_segmented_slot(
    segments: &[usize],
    q_master: &Tensor,
    k_master: &Tensor,
    v_master: &Tensor,
    rope_cs: &Tensor,
    rope_offsets_b1: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<(ChunkedKvBacking, KvCache)> {
    let backing = fresh_backing(device)?;
    let mut cache = bind_kv_cache(&backing, 0)?;
    let mut scratch = bind_kv_cache(&backing, 1)?;
    let mut start = 0usize;
    for &seg_len in segments {
        let end = start + seg_len;
        let q_seg = q_master.narrow(2, start, seg_len)?.contiguous()?;
        let k_seg = k_master.narrow(2, start, seg_len)?.contiguous()?;
        let v_seg = v_master.narrow(2, start, seg_len)?.contiguous()?;
        backing.truncate_sequence_to_blocks(1, 0)?;
        scratch.set_current_seq_len(0)?;
        let _ = run_prefill(
            &mut scratch,
            &q_seg,
            &k_seg,
            &v_seg,
            seg_len,
            rope_cs,
            rope_offsets_b1,
            stager,
            device,
        )?;
        // `prime_chunked_decode_slots_batch` ran at the end of the
        // segment's prefill and may have appended a trailing empty
        // chunk for a hypothetical follow-up decode.  Drop it
        // before sealing so it doesn't bleed into slot B's layout.
        let real_chunks = seg_len.div_ceil(CHUNK_SIZE);
        backing.truncate_sequence_to_blocks(1, real_chunks)?;
        let sealed = backing.record_turn(1)?;
        backing.inject_sealed_at_tail(0, &sealed)?;
        cache.set_current_seq_len(end)?;
        start = end;
    }
    let total: usize = segments.iter().sum();
    assert_eq!(cache.current_seq_len(), total);
    Ok((backing, cache))
}

/// Cross-check that slot A has all-full chunks (modulo a possible
/// trailing empty allocated by `prime_chunked_decode_slots_batch`) and
/// slot B's chunk usages exactly match `segments`.
fn assert_slot_layouts(
    cache_a: &KvCache,
    cache_b: &KvCache,
    total: usize,
    segments: &[usize],
    case_name: &str,
) {
    let chunks_a = cache_a.k_cache().chunked_live_chunks_as_sealed().unwrap();
    let chunks_b = cache_b.k_cache().chunked_live_chunks_as_sealed().unwrap();
    let usages_a: Vec<u16> = chunks_a.iter().map(|c| c.token_count).collect();
    let usages_b: Vec<u16> = chunks_b.iter().map(|c| c.token_count).collect();
    let sum_a: u32 = usages_a.iter().map(|&u| u as u32).sum();
    let sum_b: u32 = usages_b.iter().map(|&u| u as u32).sum();
    assert_eq!(
        sum_a, total as u32,
        "[{case_name}] slot A total token count mismatch: {usages_a:?}",
    );
    assert_eq!(
        sum_b, total as u32,
        "[{case_name}] slot B total token count mismatch: {usages_b:?}",
    );
    // All non-empty chunks of slot A should be full except possibly
    // the very last.
    let n_full_a = usages_a
        .iter()
        .filter(|&&u| u as usize == CHUNK_SIZE)
        .count();
    let n_partial_a = usages_a
        .iter()
        .filter(|&&u| u > 0 && (u as usize) < CHUNK_SIZE)
        .count();
    assert!(
        n_partial_a <= 1,
        "[{case_name}] slot A should have at most one partial chunk, got usages {usages_a:?}",
    );
    let _ = n_full_a;
    // Slot B should match `segments` exactly.
    let expected_b: Vec<u16> = segments.iter().map(|&n| n as u16).collect();
    assert_eq!(usages_b, expected_b, "[{case_name}] slot B layout mismatch",);
}

// ──────────────────────────────────────────────────────────────────────
// Low-level helpers — backing, cache, prefill, decode
// ──────────────────────────────────────────────────────────────────────

fn fresh_backing(device: &Device) -> Result<ChunkedKvBacking> {
    ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::F16, device, MAX_BLOCKS)
}

fn bind_kv_cache(backing: &ChunkedKvBacking, batch_idx: usize) -> Result<KvCache> {
    let mut cache = KvCache::new(2, 64);
    cache.force_dtype(DType::F16);
    cache.set_chunked_backing(backing, batch_idx, None)?;
    Ok(cache)
}

/// Flatten `make_qkv`'s `[1, n_head, n_tokens, head_dim]` into the FLAT-packed
/// `[total, n_head, head_dim]` the ragged prefill wants.
fn flatten_qkv(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    Ok((
        q.transpose(1, 2)?.squeeze(0)?.contiguous()?,
        k.transpose(1, 2)?.squeeze(0)?.contiguous()?,
        v.transpose(1, 2)?.squeeze(0)?.contiguous()?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn run_prefill(
    cache: &mut KvCache,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    rope_cs: &Tensor,
    rope_offsets: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    let offset = cache.current_seq_len();
    let (qf, kf, vf) = flatten_qkv(q, k, v)?;
    let generation = stager.begin_generation();
    let mut caches_arr: [&mut KvCache; 1] = [cache];
    let outs = paged_prefill_batched(
        &mut caches_arr[..],
        &[offset],
        &qf,
        &kf,
        &vf,
        1,
        &[seq_len],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        rope_offsets,
        rope_cs,
        false,
        0,
        &generation,
        &std::cell::RefCell::new(None),
    )?;
    caches_arr[0].set_current_seq_len(offset + seq_len)?;
    // `paged_prefill_batched` returns the flat attention output
    // [total_q, n_head, head_dim] (one sequence here).
    let _ = device;
    Ok(outs)
}

/// Reaches into a `KvCache` bound to slot 0 of `backing`, builds the
/// decode metadata (slot header with position_map + slice pointers),
/// and invokes `paged_decode_attn` for a single decode step.  Mirrors
/// `BatchedInferenceSession::build_decode_metadata` for one slot, so
/// the kernel exercise is identical to the production path.
#[allow(clippy::too_many_arguments)]
fn decode_one_slot(
    backing: &ChunkedKvBacking,
    cache: &KvCache,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    use candle::backend::BackendStorage;
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_transformers::models::slot_state::{
        tensor_u8_device_ptr, SlotStateHost, TokenSliceHost,
    };

    let seq_offset = cache.current_seq_len();
    let arena_info = backing.resolve_arena_info()?;
    backing.ensure_for_batch_entries(&[(0, seq_offset)], 1)?;

    let chunks = cache
        .k_cache()
        .chunked_live_chunks_as_sealed()
        .unwrap_or_default();
    let writer_start = cache.k_cache().chunked_writer_start_idx().unwrap_or(0);

    let mut slot = SlotStateHost::from_sealed_chunks(
        &chunks,
        N_KV_HEAD,
        HEAD_DIM,
        &arena_info,
        writer_start,
        true,
    );
    slot.extend_for_write_region(1, CHUNK_SIZE);

    // Two-section layout: out-of-line KvHead records first, then 16-byte slice
    // headers whose kvheads_ptr points into the records tensor.
    let rec_bytes = TokenSliceHost::record_size(N_KV_HEAD, HEAD_DIM);
    let mut records_buf = Vec::with_capacity(slot.slices.len() * rec_bytes);
    for s in &slot.slices {
        s.serialize_record(&mut records_buf);
    }
    let records_tensor = if records_buf.is_empty() {
        Tensor::zeros(1, DType::U8, device)?
    } else {
        Tensor::from_slice(&records_buf, records_buf.len(), device)?
    };
    let records_base = tensor_u8_device_ptr(&records_tensor)?;

    let mut slice_buf = Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (i, s) in slot.slices.iter().enumerate() {
        let kvheads_ptr = records_base + (i * rec_bytes) as u64;
        s.serialize_slice_header(&mut slice_buf, kvheads_ptr);
    }
    let slices_tensor = if slice_buf.is_empty() {
        Tensor::zeros(1, DType::U8, device)?
    } else {
        Tensor::from_slice(&slice_buf, slice_buf.len(), device)?
    };
    let slices_base_ptr = tensor_u8_device_ptr(&slices_tensor)?;

    let mut pm = slot.position_map.clone();
    if pm.is_empty() {
        pm.push(0);
    }
    let pm_tensor = Tensor::from_slice(&pm, pm.len(), device)?;
    let pm_base_ptr = {
        let (storage, layout) = pm_tensor.storage_and_layout();
        let cs = match &*storage {
            candle::Storage::Cuda(c) => c,
            _ => candle::bail!("expected CUDA storage"),
        };
        let stream = cs.device().cuda_stream();
        let s = cs.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
        let (p, _g) = s.device_ptr(&stream);
        p
    };

    let mut hdr = Vec::with_capacity(24);
    let n_slices = slot.slices.len() as u32;
    let write_slice = slot.write_slice;
    hdr.extend_from_slice(&n_slices.to_le_bytes());
    hdr.extend_from_slice(&write_slice.to_le_bytes());
    hdr.extend_from_slice(&slices_base_ptr.to_le_bytes());
    hdr.extend_from_slice(&pm_base_ptr.to_le_bytes());

    let generation = stager.begin_generation();
    let mut pinned = generation.alloc(hdr.len())?;
    pinned.copy_from_slice(&hdr);
    let headers_gpu = generation.submit(pinned)?;
    let headers_ptr = headers_gpu.dev_ptr();

    let softmax_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let out = paged_decode_attn(
        q,
        headers_ptr,
        DType::F16,
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        softmax_scale,
        k_new,
        v_new,
        rope_cs,
        false,
    )?;
    drop(headers_gpu);
    drop(slices_tensor);
    drop(pm_tensor);
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────
// Input tensors + numerical compare
// ──────────────────────────────────────────────────────────────────────

/// Build a deterministic synthetic Q/K/V triple for `n_tokens`
/// positions.  Hash-driven so two runs see identical inputs without
/// needing a seedable RNG.  Returned tensors are shape
/// `(1, n_head, n_tokens, head_dim)` with `n_head = N_HEAD` for Q and
/// `n_head = N_KV_HEAD` for K/V — the layout `paged_prefill_batched`
/// consumes after the transpose.
fn make_qkv(n_tokens: usize, device: &Device, seed: u64) -> Result<(Tensor, Tensor, Tensor)> {
    fn pseudo(i: usize, j: usize, k: usize, seed: u64) -> f32 {
        let mut x = (i as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F))
            .wrapping_add((k as u64).wrapping_mul(0x165667B19E3779F9))
            .wrapping_add(seed.wrapping_mul(0x94D049BB133111EB));
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        let v = (x as i64 as f32) / (i64::MAX as f32);
        v * 0.5
    }
    let mut q = Vec::with_capacity(n_tokens * N_HEAD * HEAD_DIM);
    let mut k = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    let mut v = Vec::with_capacity(n_tokens * N_KV_HEAD * HEAD_DIM);
    for t in 0..n_tokens {
        for h in 0..N_HEAD {
            for d in 0..HEAD_DIM {
                q.push(pseudo(t, h, d, seed ^ 0x111));
            }
        }
        for h in 0..N_KV_HEAD {
            for d in 0..HEAD_DIM {
                k.push(pseudo(t, h, d, seed ^ 0x222));
                v.push(pseudo(t, h, d, seed ^ 0x333));
            }
        }
    }
    let q = Tensor::from_vec(q, (1, n_tokens, N_HEAD, HEAD_DIM), device)?
        .transpose(1, 2)?
        .contiguous()?;
    let k = Tensor::from_vec(k, (1, n_tokens, N_KV_HEAD, HEAD_DIM), device)?
        .transpose(1, 2)?
        .contiguous()?;
    let v = Tensor::from_vec(v, (1, n_tokens, N_KV_HEAD, HEAD_DIM), device)?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((q, k, v))
}

fn hash_str(s: &str) -> u64 {
    // Tiny FNV-1a — deterministic, no dependency.
    let mut h: u64 = 0xCBF29CE484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

fn max_abs_diff_f32(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a - b)?.abs()?.flatten_all()?;
    let m = d.max(0)?;
    let m = m.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    Ok(m)
}

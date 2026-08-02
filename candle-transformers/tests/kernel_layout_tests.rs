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

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::{PinnedBuf, PinnedStager};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::{
    quantize_sealed_in_place, ChunkedKvBacking, CompressionPolicy, KvCache, KvFormat, SealedChunk,
    SealedSequence, CHUNK_SIZE,
};
use candle_transformers::models::prefill_utils::{
    compute_rope_cs, paged_decode_attn, paged_glue_attn, paged_prefill_batched,
};
use std::sync::{Arc, Mutex, MutexGuard};

// These tests share one GPU and the process-global quantized arena table, so
// they must not run concurrently. Each test acquires this guard first; poison
// from a panicking sibling is recovered (the next test rebuilds its own state).
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn gpu_serial() -> MutexGuard<'static, ()> {
    GPU_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

// ──────────────────────────────────────────────────────────────────────
// Test-config knobs
// ──────────────────────────────────────────────────────────────────────

const N_KV_HEAD: usize = 4;
const N_HEAD: usize = 4; // No GQA — keeps the test simple
const HEAD_DIM: usize = 128;
const MAX_BLOCKS: usize = 256; // Headroom for the larger layouts + quant candidate arenas
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
    let _serial = gpu_serial();
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
    let _serial = gpu_serial();
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
    let _serial = gpu_serial();
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
    // headers whose kvheads_ptr points at the slice's record. Float/transient
    // slices (meta.is_none()) serialize a scratch record here; quantized slices
    // (meta.is_some()) carry a device-resident meta-pool record and point
    // kvheads_ptr straight at its `device_addr()` — exactly as the production
    // `sync_decode_gpu_chunks` path does.
    let mut records_buf: Vec<u8> = Vec::new();
    let mut rec_offset: Vec<Option<usize>> = Vec::with_capacity(slot.slices.len());
    for s in &slot.slices {
        if s.meta.is_some() {
            rec_offset.push(None);
        } else {
            rec_offset.push(Some(records_buf.len()));
            s.serialize_record(&mut records_buf);
        }
    }
    let records_tensor = if records_buf.is_empty() {
        Tensor::zeros(1, DType::U8, device)?
    } else {
        Tensor::from_slice(&records_buf, records_buf.len(), device)?
    };
    let records_base = tensor_u8_device_ptr(&records_tensor)?;

    let mut slice_buf = Vec::with_capacity(slot.slices.len() * TokenSliceHost::SLICE_HEADER_SIZE);
    for (s, off) in slot.slices.iter().zip(&rec_offset) {
        let kvheads_ptr = match s.meta.as_ref() {
            Some(m) => m.device_addr(),
            None => records_base + off.expect("non-meta slice has a record offset") as u64,
        };
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

// ──────────────────────────────────────────────────────────────────────
// Offset>0 window coverage
//
// Every case in TEST_CASES builds its partial chunks by prefilling a
// segment fresh into a scratch slot at state.offset=0, so they only ever
// exercise offset==0 windows (pure end trims). The compression
// assistant-half is the only production path that injects an offset>0
// window: a chunk that physically holds tokens [a, a+N) but is read through
// the sub-window [a+off, a+N), sharing the physical chunk with the turn's
// user-half. These cases pin that read path — a suffix injected as an
// offset>0 window of a full prefill must produce the same attention as the
// identical suffix prefilled fresh at offset 0.
// ──────────────────────────────────────────────────────────────────────

/// `(name, total, window_start)`. `window_start % CHUNK_SIZE != 0` forces
/// the first windowed chunk to carry a non-zero `offset`.
const OFFSET_WINDOW_CASES: &[(&str, usize, usize)] = &[
    ("offset_window_single_chunk", 32, 10),
    ("offset_window_boundary_18", 64, 18),
    ("offset_window_multi_chunk", 70, 10),
    ("offset_window_deep", 200, 50),
];

#[test]
fn kernel_layout_offset_window_matches_fresh() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    // Each case twice: rope=false isolates chunk-read addressing; rope=true
    // additionally exercises the per-token RoPE position `slice_rope +
    // (within - off)` that the offset>0 read must compute.
    for &(name, total, window_start) in OFFSET_WINDOW_CASES {
        for with_rope in [false, true] {
            let label = if with_rope { "rope" } else { "flat" };
            match run_offset_window_case(name, total, window_start, with_rope, &device, &stager) {
                Ok(diff) => eprintln!("offset-window {name:28} [{label}] diff = {diff:.6e}"),
                Err(e) => {
                    eprintln!("offset-window {name:28} [{label}] FAILED: {e}");
                    failures.push(format!("{name} [{label}]: {e}"));
                }
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "offset>0 window divergence in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

/// Window `seq` to `[start_tok, token_count)`, sharing the physical chunks.
/// The chunk containing `start_tok` becomes an offset>0 partial; earlier
/// chunks drop out. Mirrors `conversation::window_sealed_tokens` for the
/// suffix case.
fn window_suffix(seq: &SealedSequence, start_tok: usize) -> SealedSequence {
    let win_start = start_tok.min(seq.token_count);
    let mut chunks: Vec<SealedChunk> = Vec::new();
    let mut acc = 0usize;
    for chunk in &seq.chunks {
        let c = chunk.token_count as usize;
        let chunk_start = acc;
        let chunk_end = acc + c;
        acc = chunk_end;
        if win_start >= chunk_end {
            continue;
        }
        let overlap_start = chunk_start.max(win_start);
        let overlap_len = (chunk_end - overlap_start) as u16;
        if overlap_start == chunk_start {
            chunks.push(chunk.clone());
        } else {
            let mut w = chunk.clone();
            w.offset = chunk.offset + (overlap_start - chunk_start) as u16;
            w.token_count = overlap_len;
            chunks.push(w);
        }
    }
    SealedSequence {
        chunks,
        token_count: seq.token_count - win_start,
        chunk_size: seq.chunk_size,
        location: seq.location,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_offset_window_case(
    case_name: &str,
    total: usize,
    window_start: usize,
    with_rope: bool,
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let (q_master, k_master, v_master) = make_qkv(total, device, hash_str(case_name))?;
    // rope=false (inv_freq=0) isolates chunk-read addressing; rope=true uses a
    // real geometric inv_freq so the per-token RoPE position the offset>0 read
    // computes (`slice_rope + (within - off)`) is exercised too.
    let inv_freq = if with_rope {
        let f: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / HEAD_DIM as f32))
            .collect();
        Tensor::from_vec(f, HEAD_DIM / 2, device)?
    } else {
        Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?
    };
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;
    let win_len = total - window_start;

    // Reference slot C: fresh offset-0 prefill of master[window_start..total].
    let q_suf = q_master.narrow(2, window_start, win_len)?.contiguous()?;
    let k_suf = k_master.narrow(2, window_start, win_len)?.contiguous()?;
    let v_suf = v_master.narrow(2, window_start, win_len)?.contiguous()?;
    let (backing_c, cache_c) = build_control_slot(
        win_len,
        &q_suf,
        &k_suf,
        &v_suf,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;

    // Test slot B: full prefill into scratch, seal, window [window_start,
    // total] (offset>0 first chunk), inject into slot 0.
    let backing_b = fresh_backing(device)?;
    let mut cache_b = bind_kv_cache(&backing_b, 0)?;
    let mut scratch = bind_kv_cache(&backing_b, 1)?;
    let q_all = q_master.narrow(2, 0, total)?.contiguous()?;
    let k_all = k_master.narrow(2, 0, total)?.contiguous()?;
    let v_all = v_master.narrow(2, 0, total)?.contiguous()?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    let real_chunks = total.div_ceil(CHUNK_SIZE);
    backing_b.truncate_sequence_to_blocks(1, real_chunks)?;
    let sealed = backing_b.record_turn(1)?;
    let windowed = window_suffix(&sealed, window_start);
    // The first windowed chunk must carry a non-zero offset, else the case
    // isn't testing what it claims.
    assert!(
        windowed.chunks.first().map(|c| c.offset).unwrap_or(0) > 0,
        "[{case_name}] window_start={window_start} produced an offset-0 first chunk",
    );
    backing_b.inject_sealed_at_tail(0, &windowed)?;
    cache_b.set_current_seq_len(win_len)?;

    // Same fresh decode Q on both slots.
    let (q_dec, k_new, v_new) = make_qkv(1, device, hash_str(case_name) ^ 0xD3C0DE)?;
    let q_dec_2d = q_dec.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let k_new_2d = k_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let v_new_2d = v_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;

    let out_c = decode_one_slot(
        &backing_c, &cache_c, &q_dec_2d, &k_new_2d, &v_new_2d, &rope_cs, stager, device,
    )?;
    let out_b = decode_one_slot(
        &backing_b, &cache_b, &q_dec_2d, &k_new_2d, &v_new_2d, &rope_cs, stager, device,
    )?;

    let diff = max_abs_diff_f32(&out_c.to_dtype(DType::F32)?, &out_b.to_dtype(DType::F32)?)?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "offset>0 window diverged (total={total}, window_start={window_start}): \
             max abs diff = {diff:.6e} (expected < {:.0e})",
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

// ──────────────────────────────────────────────────────────────────────
// Quantized offset>0 window coverage
//
// The fp16 cases above proved the decode kernel reads offset>0 windows at
// exact byte fidelity. The production compression path reads QUANTIZED
// sealed K/V, and quantization is per-32-token block. An offset>0 window
// reads slots [offset, 32) of a quantized block whose packing+scale cover
// the whole block — a path nothing else exercises. These cases compare a
// quantized offset>0 window against the fp16 window of the same tokens
// (rope and layout identical; only quantization differs), using the
// offset-0 full read as the quant-noise baseline. A correct dequant makes
// the offset>0 quant error track the offset-0 baseline; a broken one reads
// the wrong slots/scale and the error explodes.
// ──────────────────────────────────────────────────────────────────────

/// `(name, total, window_start, level)`. `total % CHUNK_SIZE == 0` keeps
/// the quantize on whole blocks; `window_start % CHUNK_SIZE != 0` forces an
/// offset>0 first window chunk. `level` is the CompressionPolicy level.
const OFFSET_WINDOW_QUANT_CASES: &[(&str, usize, usize, u8)] = &[
    ("q_c0_single_off10", 32, 10, 0),
    ("q_c3_single_off10", 32, 10, 3),
    ("q_c0_boundary_off18", 64, 18, 0),
    ("q_c3_boundary_off18", 64, 18, 3),
    ("q_c3_multi_off40", 96, 40, 3),
    ("q_c3_deep_off50", 128, 50, 3),
];

fn cuda_stream(device: &Device) -> Arc<CudaStream> {
    match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!("test gated on a CUDA device"),
    }
}

#[test]
fn kernel_layout_quantized_offset_window_matches_fp16() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, total, window_start, level) in OFFSET_WINDOW_QUANT_CASES {
        match run_offset_window_quant_case(name, total, window_start, level, &device, &stager) {
            Ok((base, win)) => {
                eprintln!(
                    "quant-offset {name:24} baseline(off0)={base:.4e}  window(off>0)={win:.4e}"
                )
            }
            Err(e) => {
                eprintln!("quant-offset {name:24} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "quantized offset>0 window divergence in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

/// Decode one sealed sequence on a fresh slot of `backing`: bind, inject,
/// set the logical length, single decode step. Returns the attention out.
#[allow(clippy::too_many_arguments)]
fn inject_and_decode(
    backing: &ChunkedKvBacking,
    slot: usize,
    sealed: &SealedSequence,
    q: &Tensor,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    let mut cache = bind_kv_cache(backing, slot)?;
    backing.inject_sealed_at_tail(slot, sealed)?;
    // Production primes a fresh writer chunk after injecting Arc-shared
    // sealed windows so the decode's write never lands in a shared chunk.
    backing.push_empty_writer_chunk(slot)?;
    cache.set_current_seq_len(sealed.token_count)?;
    let out = decode_one_slot(backing, &cache, q, k_new, v_new, rope_cs, stager, device)?;
    // Force any async kernel fault to surface here so the caller's stage
    // markers attribute the crash to the right decode.
    device.synchronize()?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_offset_window_quant_case(
    case_name: &str,
    total: usize,
    window_start: usize,
    level: u8,
    device: &Device,
    stager: &PinnedStager,
) -> Result<(f32, f32)> {
    let (q_master, k_master, v_master) = make_qkv(total, device, hash_str(case_name))?;
    // rope cancels in the quant-vs-fp16 comparison (both apply the same
    // position math to the same logical tokens); keep it off to isolate the
    // dequant of the windowed sub-block.
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;

    // Backing with the warm-protected adaptive candidate arenas for `level`,
    // mirroring the substrate engine's startup wiring.
    let policy = CompressionPolicy::new(level);
    let backing = ChunkedKvBacking::new_with_format_adaptive(
        4,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Float(DType::F16),
        KvFormat::Float(DType::F16),
        device,
        MAX_BLOCKS,
        Some(policy.clone()),
    )?;

    // Prefill the full master into a scratch slot and seal it (fp16).
    let mut scratch = bind_kv_cache(&backing, 0)?;
    let q_all = q_master.narrow(2, 0, total)?.contiguous()?;
    let k_all = k_master.narrow(2, 0, total)?.contiguous()?;
    let v_all = v_master.narrow(2, 0, total)?.contiguous()?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    let real_chunks = total.div_ceil(CHUNK_SIZE);
    backing.truncate_sequence_to_blocks(0, real_chunks)?;
    let src = backing.record_turn(0)?;

    // Quantize the sealed turn — the fp16 `src` survives alongside `warm`.
    let copy_stream = cuda_stream(device);
    let mut pinned: Option<PinnedBuf> = None;
    let warm = quantize_sealed_in_place(
        &backing,
        &[&src],
        &policy,
        device,
        &copy_stream,
        &mut pinned,
    )?;
    copy_stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("quant sync: {e}")))?;
    let warm = warm.into_iter().next().expect("one sealed in → one out");

    let src_win = window_suffix(&src, window_start);
    let warm_win = window_suffix(&warm, window_start);
    assert!(
        warm_win.chunks.first().map(|c| c.offset).unwrap_or(0) > 0,
        "[{case_name}] window produced offset-0 first chunk",
    );

    let (q_dec, k_new, v_new) = make_qkv(1, device, hash_str(case_name) ^ 0xD3C0DE)?;
    let q2 = q_dec.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let kn = k_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let vn = v_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;

    // Four decodes on fresh slots: fp16/quant × full/window.
    let out_fp16_full =
        inject_and_decode(&backing, 1, &src, &q2, &kn, &vn, &rope_cs, stager, device)?;
    let out_q_full =
        inject_and_decode(&backing, 2, &warm, &q2, &kn, &vn, &rope_cs, stager, device)?;
    let out_fp16_win = inject_and_decode(
        &backing, 3, &src_win, &q2, &kn, &vn, &rope_cs, stager, device,
    )?;
    let out_q_win = inject_and_decode(
        &backing, 4, &warm_win, &q2, &kn, &vn, &rope_cs, stager, device,
    )?;

    let baseline = max_abs_diff_f32(
        &out_q_full.to_dtype(DType::F32)?,
        &out_fp16_full.to_dtype(DType::F32)?,
    )?;
    let window = max_abs_diff_f32(
        &out_q_win.to_dtype(DType::F32)?,
        &out_fp16_win.to_dtype(DType::F32)?,
    )?;

    // A correct offset>0 dequant makes the window's quant error track the
    // offset-0 baseline. A broken one reads the wrong slots/scale and the
    // window error explodes far past it.
    let allow = baseline * 4.0 + 5e-3;
    if window > allow {
        candle::bail!(
            "quantized offset>0 window error {window:.4e} >> offset-0 baseline {baseline:.4e} \
             (allow {allow:.4e}); total={total} window_start={window_start} level={level}",
        );
    }
    Ok((baseline, window))
}

// Build a backing whose sealed turn exists in both fp16 (`src`) and quantized
// (`warm`) form, for tests that compare the two reads of the same window.
fn build_quant_pair(
    total: usize,
    level: u8,
    device: &Device,
    stager: &PinnedStager,
) -> Result<(ChunkedKvBacking, SealedSequence, SealedSequence, Tensor)> {
    let (q_master, k_master, v_master) = make_qkv(total, device, 0xA11CE)?;
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;
    let policy = CompressionPolicy::new(level);
    let backing = ChunkedKvBacking::new_with_format_adaptive(
        4,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Float(DType::F16),
        KvFormat::Float(DType::F16),
        device,
        MAX_BLOCKS,
        Some(policy.clone()),
    )?;
    let mut scratch = bind_kv_cache(&backing, 0)?;
    let q_all = q_master.narrow(2, 0, total)?.contiguous()?;
    let k_all = k_master.narrow(2, 0, total)?.contiguous()?;
    let v_all = v_master.narrow(2, 0, total)?.contiguous()?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    backing.truncate_sequence_to_blocks(0, total.div_ceil(CHUNK_SIZE))?;
    let src = backing.record_turn(0)?;
    let copy_stream = cuda_stream(device);
    let mut pinned: Option<PinnedBuf> = None;
    let warm = quantize_sealed_in_place(
        &backing,
        &[&src],
        &policy,
        device,
        &copy_stream,
        &mut pinned,
    )?;
    copy_stream
        .synchronize()
        .map_err(|e| candle::Error::Msg(format!("quant sync: {e}")))?;
    Ok((backing, src, warm.into_iter().next().unwrap(), rope_cs))
}

// ──────────────────────────────────────────────────────────────────────
// Glue (gap-fill) offset>0 window coverage
//
// Compression assembles its context via the paged-glue (gap-fill) kernel,
// which fills the glue tokens' K/V by attending to the sealed prefix —
// reading the injected offset>0 assistant-half window. The glue kernel uses
// a SEPARATE rope source (`col_actual_pos`) than decode, so its offset>0
// read is independently untested. These cases inject an offset>0 window as
// the prefix, run a glue forward over fresh glue tokens, and compare against
// the same glue over a fresh offset-0 prefill of the identical suffix. A
// correct glue read of the offset>0 prefix makes the two match exactly.
// ──────────────────────────────────────────────────────────────────────

const OFFSET_WINDOW_GLUE_CASES: &[(&str, usize, usize)] = &[
    ("glue_off10_single", 32, 10),
    ("glue_off18_boundary", 64, 18),
    ("glue_off40_multi", 96, 40),
    ("glue_off50_deep", 128, 50),
];

const GLUE_TOKENS: usize = 8;

/// Placeholder glue scatter descriptors for the current `paged_glue_attn`
/// signature. The two glue tests below were written against the old
/// `col_actual_pos` side-channel model; the kernel now reserves each glue
/// token as an in-place gap chunk and takes per-token scatter descriptors
/// (`glue_write_slice` / `glue_write_in_blk`) plus a per-token forward window
/// (`fwd_ahead`). Both tests are `#[ignore]`d pending a rewrite of their KV
/// setup to reserve gap chunks and derive real descriptors, and GPU
/// re-verification. This keeps the calls compiling in the meantime.
fn glue_descriptors(n_glue: usize, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
    let z = Tensor::zeros(n_glue, DType::U32, device)?;
    let fwd = Tensor::zeros(n_glue, DType::U32, device)?; // backward-only
    Ok((z.clone(), z, fwd))
}

#[test]
#[ignore = "needs port to the in-place gap-chunk glue model + GPU re-verification"]
fn kernel_layout_glue_offset_window_matches_fresh() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => {
            eprintln!("skipping: CUDA device required");
            return Ok(());
        }
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, total, window_start) in OFFSET_WINDOW_GLUE_CASES {
        match run_offset_window_glue_case(name, total, window_start, &device, &stager) {
            Ok(diff) => eprintln!("glue-offset {name:24} diff = {diff:.6e}"),
            Err(e) => {
                eprintln!("glue-offset {name:24} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "glue offset>0 window divergence in {} case(s):\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_offset_window_glue_case(
    case_name: &str,
    total: usize,
    window_start: usize,
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let (q_master, k_master, v_master) = make_qkv(total, device, hash_str(case_name))?;
    // Real geometric inv_freq so the glue's RoPE position — sourced from
    // `col_actual_pos`, the one path decode does not share — is exercised on
    // the offset>0 prefix read, not just the chunk addressing.
    let f: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(f, HEAD_DIM / 2, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;
    let win_len = total - window_start;

    // Reference slot C: fresh offset-0 prefill of master[window_start..total].
    let q_suf = q_master.narrow(2, window_start, win_len)?.contiguous()?;
    let k_suf = k_master.narrow(2, window_start, win_len)?.contiguous()?;
    let v_suf = v_master.narrow(2, window_start, win_len)?.contiguous()?;
    let (backing_c, mut cache_c) = build_control_slot(
        win_len,
        &q_suf,
        &k_suf,
        &v_suf,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;

    // Test slot B: full prefill into scratch, seal, window [window_start,
    // total] (offset>0 first chunk), inject, prime writer.
    let backing_b = fresh_backing(device)?;
    let mut cache_b = bind_kv_cache(&backing_b, 0)?;
    let mut scratch = bind_kv_cache(&backing_b, 1)?;
    let q_all = q_master.narrow(2, 0, total)?.contiguous()?;
    let k_all = k_master.narrow(2, 0, total)?.contiguous()?;
    let v_all = v_master.narrow(2, 0, total)?.contiguous()?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    backing_b.truncate_sequence_to_blocks(1, total.div_ceil(CHUNK_SIZE))?;
    let sealed = backing_b.record_turn(1)?;
    let windowed = window_suffix(&sealed, window_start);
    assert!(
        windowed.chunks.first().map(|c| c.offset).unwrap_or(0) > 0,
        "[{case_name}] window produced offset-0 first chunk",
    );
    backing_b.inject_sealed_at_tail(0, &windowed)?;
    backing_b.push_empty_writer_chunk(0)?;
    cache_b.set_current_seq_len(win_len)?;

    // Same glue tokens + col_actual_pos on both slots: prefix logical
    // positions [0, win_len) then glue [win_len, win_len+GLUE_TOKENS).
    let (qg, kg, vg) = make_qkv(GLUE_TOKENS, device, hash_str(case_name) ^ 0x6C0E_u64)?;
    let (qgf, kgf, vgf) = flatten_qkv(&qg, &kg, &vg)?;
    let (gw_slice, gw_in_blk, fwd_ahead) = glue_descriptors(GLUE_TOKENS, device)?;

    let gen_c = stager.begin_generation();
    let out_c = paged_glue_attn(
        &mut [&mut cache_c],
        &[win_len],
        &qgf,
        &kgf,
        &vgf,
        1,
        &[GLUE_TOKENS],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        &gw_slice,
        &gw_in_blk,
        &fwd_ahead,
        &rope_cs,
        false,
        &gen_c,
        &std::cell::RefCell::new(None),
    )?;
    let gen_b = stager.begin_generation();
    let out_b = paged_glue_attn(
        &mut [&mut cache_b],
        &[win_len],
        &qgf,
        &kgf,
        &vgf,
        1,
        &[GLUE_TOKENS],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        &gw_slice,
        &gw_in_blk,
        &fwd_ahead,
        &rope_cs,
        false,
        &gen_b,
        &std::cell::RefCell::new(None),
    )?;
    let _ = (&backing_b, &backing_c);

    let diff = max_abs_diff_f32(&out_b.to_dtype(DType::F32)?, &out_c.to_dtype(DType::F32)?)?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "glue offset>0 window diverged (total={total}, window_start={window_start}): \
             max abs diff = {diff:.6e} (expected < {:.0e})",
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

// ──────────────────────────────────────────────────────────────────────
// Quantized glue offset>0 window — the exact kernel combo compression uses:
// a gap-fill forward whose sealed prefix is a QUANTIZED offset>0 window,
// streamed dequant-once. Compares quant-glue vs fp16-glue for the same
// window, with the offset-0 full read as the quant-noise baseline.
// ──────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn glue_over(
    backing: &ChunkedKvBacking,
    slot: usize,
    sealed: &SealedSequence,
    qgf: &Tensor,
    kgf: &Tensor,
    vgf: &Tensor,
    rope_cs: &Tensor,
    stager: &PinnedStager,
    device: &Device,
) -> Result<Tensor> {
    let mut cache = bind_kv_cache(backing, slot)?;
    backing.inject_sealed_at_tail(slot, sealed)?;
    backing.push_empty_writer_chunk(slot)?;
    let win_len = sealed.token_count;
    cache.set_current_seq_len(win_len)?;
    let (gw_slice, gw_in_blk, fwd_ahead) = glue_descriptors(GLUE_TOKENS, device)?;
    let gen = stager.begin_generation();
    let out = paged_glue_attn(
        &mut [&mut cache],
        &[win_len],
        qgf,
        kgf,
        vgf,
        1,
        &[GLUE_TOKENS],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        &gw_slice,
        &gw_in_blk,
        &fwd_ahead,
        rope_cs,
        false,
        &gen,
        &std::cell::RefCell::new(None),
    )?;
    device.synchronize()?;
    Ok(out)
}

#[test]
#[ignore = "needs port to the in-place gap-chunk glue model + GPU re-verification"]
fn kernel_layout_quantized_glue_offset_window() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => return Ok(()),
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, total, ws) in &[
        ("qglue_off18", 64usize, 18usize),
        ("qglue_off40", 96, 40),
        ("qglue_off50", 128, 50),
    ] {
        let r = (|| -> Result<(f32, f32)> {
            let (backing, src, warm, rope_cs) = build_quant_pair(total, 3, &device, &stager)?;
            let (qg, kg, vg) = make_qkv(GLUE_TOKENS, &device, 0x6C0E)?;
            let (qgf, kgf, vgf) = flatten_qkv(&qg, &kg, &vg)?;
            let src_win = window_suffix(&src, ws);
            let warm_win = window_suffix(&warm, ws);
            let o_fp16_full = glue_over(
                &backing, 1, &src, &qgf, &kgf, &vgf, &rope_cs, &stager, &device,
            )?;
            let o_q_full = glue_over(
                &backing, 2, &warm, &qgf, &kgf, &vgf, &rope_cs, &stager, &device,
            )?;
            let o_fp16_win = glue_over(
                &backing, 3, &src_win, &qgf, &kgf, &vgf, &rope_cs, &stager, &device,
            )?;
            let o_q_win = glue_over(
                &backing, 4, &warm_win, &qgf, &kgf, &vgf, &rope_cs, &stager, &device,
            )?;
            let base = max_abs_diff_f32(
                &o_q_full.to_dtype(DType::F32)?,
                &o_fp16_full.to_dtype(DType::F32)?,
            )?;
            let win = max_abs_diff_f32(
                &o_q_win.to_dtype(DType::F32)?,
                &o_fp16_win.to_dtype(DType::F32)?,
            )?;
            Ok((base, win))
        })();
        match r {
            Ok((base, win)) => {
                eprintln!("qglue {name:16} baseline(off0)={base:.4e}  window(off>0)={win:.4e}");
                if win > base * 4.0 + 5e-3 {
                    failures.push(format!("{name}: window {win:.4e} >> baseline {base:.4e}"));
                }
            }
            Err(e) => {
                eprintln!("qglue {name:16} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "quantized glue offset>0 divergence:\n  - {}",
            failures.join("\n  - ")
        );
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Multi-segment composite: two windows that SHARE the boundary chunk
//
// This is the closest analog to the real compression assembly: a turn's
// user-half [0, split) and assistant-half [split, total) are derived as two
// windows of the same sealed sequence. They share the physical boundary
// chunk (the one straddling `split`) via the same ChunkGid — the first half
// reads its [0, split%32) slots (offset 0), the second its [split%32, 32)
// slots (offset>0). Injecting BOTH back-to-back into one slot and decoding
// must match a fresh prefill of the whole logical sequence. A host
// composition bug (rope_base accumulation across the shared chunk, or
// position_map ordering for the second window) shows up as divergence here
// where the single-window tests pass.
// ──────────────────────────────────────────────────────────────────────

/// `(name, total, split)` — `split % CHUNK_SIZE != 0` makes the boundary
/// chunk shared between the two windows with a non-zero second-window offset.
const MULTISEG_CASES: &[(&str, usize, usize)] = &[
    ("multiseg_split18", 64, 18),
    ("multiseg_split33", 70, 33),
    ("multiseg_split40", 96, 40),
    ("multiseg_split50", 128, 50),
];

/// Window `seq` to the token range `[start, end)`, sharing physical chunks.
fn window_range(seq: &SealedSequence, start_tok: usize, end_tok: usize) -> SealedSequence {
    let ws = start_tok.min(seq.token_count);
    let we = end_tok.min(seq.token_count).max(ws);
    let mut chunks: Vec<SealedChunk> = Vec::new();
    let mut acc = 0usize;
    for chunk in &seq.chunks {
        let c = chunk.token_count as usize;
        let (cs, ce) = (acc, acc + c);
        acc = ce;
        let os = cs.max(ws);
        let oe = ce.min(we);
        if os >= oe {
            continue;
        }
        let olen = (oe - os) as u16;
        if os == cs && olen as usize == c {
            chunks.push(chunk.clone());
        } else {
            let mut w = chunk.clone();
            w.offset = chunk.offset + (os - cs) as u16;
            w.token_count = olen;
            chunks.push(w);
        }
    }
    SealedSequence {
        chunks,
        token_count: we - ws,
        chunk_size: seq.chunk_size,
        location: seq.location,
    }
}

#[test]
fn kernel_layout_multiseg_shared_boundary_matches_fresh() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => return Ok(()),
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, total, split) in MULTISEG_CASES {
        match run_multiseg_case(name, total, split, &device, &stager) {
            Ok(diff) => eprintln!("multiseg {name:20} diff = {diff:.6e}"),
            Err(e) => {
                eprintln!("multiseg {name:20} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!("multiseg divergence:\n  - {}", failures.join("\n  - "));
    }
    Ok(())
}

fn run_multiseg_case(
    name: &str,
    total: usize,
    split: usize,
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let (qm, km, vm) = make_qkv(total, device, hash_str(name))?;
    let f: Vec<f32> = (0..HEAD_DIM / 2)
        .map(|i| 1.0f32 / 10000f32.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(f, HEAD_DIM / 2, device)?;
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;

    let q_all = qm.narrow(2, 0, total)?.contiguous()?;
    let k_all = km.narrow(2, 0, total)?.contiguous()?;
    let v_all = vm.narrow(2, 0, total)?.contiguous()?;

    // Reference slot: one fresh prefill of the whole logical sequence.
    let (backing_ref, cache_ref) = build_control_slot(
        total,
        &q_all,
        &k_all,
        &v_all,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;

    // Test slot: prefill+seal once, derive two windows sharing the boundary
    // chunk, inject both back-to-back.
    let backing = fresh_backing(device)?;
    let mut cache = bind_kv_cache(&backing, 0)?;
    let mut scratch = bind_kv_cache(&backing, 1)?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    backing.truncate_sequence_to_blocks(1, total.div_ceil(CHUNK_SIZE))?;
    let sealed = backing.record_turn(1)?;
    let seg_a = window_range(&sealed, 0, split);
    let seg_b = window_range(&sealed, split, total);
    assert!(
        seg_b.chunks.first().map(|c| c.offset).unwrap_or(0) > 0,
        "[{name}] second window has offset-0 first chunk (split aligned?)",
    );
    // The boundary chunk's gid must appear in BOTH windows (shared).
    let gid_id = |c: &SealedChunk| {
        let g = &c.gids.as_slice()[0];
        (g.arena_idx(), g.chunk_idx())
    };
    let a_last = seg_a.chunks.last().map(gid_id);
    let b_first = seg_b.chunks.first().map(gid_id);
    assert_eq!(
        a_last, b_first,
        "[{name}] boundary chunk not shared across windows"
    );
    backing.inject_sealed_at_tail(0, &seg_a)?;
    backing.inject_sealed_at_tail(0, &seg_b)?;
    backing.push_empty_writer_chunk(0)?;
    cache.set_current_seq_len(total)?;

    let (q_dec, k_new, v_new) = make_qkv(1, device, hash_str(name) ^ 0xD3C0DE)?;
    let q2 = q_dec.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let kn = k_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let vn = v_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;

    let out_ref = decode_one_slot(
        &backing_ref,
        &cache_ref,
        &q2,
        &kn,
        &vn,
        &rope_cs,
        stager,
        device,
    )?;
    let out_test = decode_one_slot(&backing, &cache, &q2, &kn, &vn, &rope_cs, stager, device)?;

    let diff = max_abs_diff_f32(
        &out_test.to_dtype(DType::F32)?,
        &out_ref.to_dtype(DType::F32)?,
    )?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "multiseg shared-boundary diverged (total={total}, split={split}): \
             max abs diff = {diff:.6e} (expected < {:.0e})",
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

// ──────────────────────────────────────────────────────────────────────
// Interspersed-glue composite: glue logically BEFORE an injected segment
// but physically appended after it (the gap-fill writes all glue at the
// tail). Mirrors compression's `[section | user_start-glue | half | …]`,
// where the glue's true position precedes the half it physically follows.
// The post-glue decode must position every token by its TRUE logical
// position; if it falls back to cumulative-physical rope, the half and the
// glue land at the wrong positions and this diverges from a fresh prefill.
// ──────────────────────────────────────────────────────────────────────

const GLUE_INTERSPERSED_CASES: &[(&str, usize, usize, usize)] = &[
    // (name, len_a, glue_len, len_b)
    ("inter_a16_g8_b24", 16, 8, 24),
    ("inter_a32_g8_b30", 32, 8, 30),
    ("inter_a40_g16_b40", 40, 16, 40),
];

#[test]
fn kernel_layout_glue_interspersed_matches_fresh() -> Result<()> {
    let _serial = gpu_serial();
    let device = match Device::cuda_if_available(0) {
        Ok(d) if d.is_cuda() => d,
        _ => return Ok(()),
    };
    let stager = PinnedStager::new_from_device(&device);
    let mut failures: Vec<String> = Vec::new();
    for &(name, la, g, lb) in GLUE_INTERSPERSED_CASES {
        match run_glue_interspersed_case(name, la, g, lb, &device, &stager) {
            Ok(diff) => eprintln!("inter-glue {name:20} diff = {diff:.6e}"),
            Err(e) => {
                eprintln!("inter-glue {name:20} FAILED: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        candle::bail!(
            "interspersed-glue divergence:\n  - {}",
            failures.join("\n  - ")
        );
    }
    Ok(())
}

fn run_glue_interspersed_case(
    name: &str,
    la: usize,
    g: usize,
    lb: usize,
    device: &Device,
    stager: &PinnedStager,
) -> Result<f32> {
    let total = la + g + lb;
    let (qm, km, vm) = make_qkv(total, device, hash_str(name))?;
    let inv_freq = Tensor::zeros(HEAD_DIM / 2, DType::F32, device)?; // isolate ordering, not rope
    let rope_cs = compute_rope_cs(&inv_freq, MAX_BLOCKS, HEAD_DIM, device)?;
    let rope_offsets_b1 = Tensor::zeros(1, DType::U32, device)?;
    let q_all = qm.narrow(2, 0, total)?.contiguous()?;
    let k_all = km.narrow(2, 0, total)?.contiguous()?;
    let v_all = vm.narrow(2, 0, total)?.contiguous()?;

    // Reference slot: fresh prefill of the whole logical sequence [A|glue|B],
    // then one decode step.
    let (backing_ref, cache_ref) = build_control_slot(
        total,
        &q_all,
        &k_all,
        &v_all,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;

    // Test slot: inject A=[0,la) and B=[la+g, total) (both sealed), then a glue
    // forward writes the glue [la, la+g) — logically between A and B, physically
    // after both. col_actual_pos carries each token's TRUE logical position.
    let backing = fresh_backing(device)?;
    let mut cache = bind_kv_cache(&backing, 0)?;
    let mut scratch = bind_kv_cache(&backing, 1)?;
    let _ = run_prefill(
        &mut scratch,
        &q_all,
        &k_all,
        &v_all,
        total,
        &rope_cs,
        &rope_offsets_b1,
        stager,
        device,
    )?;
    backing.truncate_sequence_to_blocks(1, total.div_ceil(CHUNK_SIZE))?;
    let sealed = backing.record_turn(1)?;
    let seg_a = window_range(&sealed, 0, la);
    let seg_b = window_range(&sealed, la + g, total);
    backing.inject_sealed_at_tail(0, &seg_a)?;
    backing.inject_sealed_at_tail(0, &seg_b)?;
    cache.set_current_seq_len(la + lb)?;

    // Glue tokens = master[la, la+g); col_actual_pos = A logical, B logical,
    // glue logical (prefix in inject/physical order, then glue).
    let qg = qm.narrow(2, la, g)?.contiguous()?;
    let kg = km.narrow(2, la, g)?.contiguous()?;
    let vg = vm.narrow(2, la, g)?.contiguous()?;
    let (qgf, kgf, vgf) = flatten_qkv(&qg, &kg, &vg)?;
    let (gw_slice, gw_in_blk, fwd_ahead) = glue_descriptors(g, device)?;
    let gen = stager.begin_generation();
    let _ = paged_glue_attn(
        &mut [&mut cache],
        &[la + lb],
        &qgf,
        &kgf,
        &vgf,
        1,
        &[g],
        N_HEAD,
        N_KV_HEAD,
        HEAD_DIM,
        None,
        &gw_slice,
        &gw_in_blk,
        &fwd_ahead,
        &rope_cs,
        false,
        &gen,
        &std::cell::RefCell::new(None),
    )?;
    device.synchronize()?;
    cache.set_current_seq_len(total)?;

    let (q_dec, k_new, v_new) = make_qkv(1, device, hash_str(name) ^ 0xD3C0DE)?;
    let q2 = q_dec.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let kn = k_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let vn = v_new.squeeze(2)?.to_dtype(DType::F16)?.contiguous()?;
    let out_ref = decode_one_slot(
        &backing_ref,
        &cache_ref,
        &q2,
        &kn,
        &vn,
        &rope_cs,
        stager,
        device,
    )?;
    let out_test = decode_one_slot(&backing, &cache, &q2, &kn, &vn, &rope_cs, stager, device)?;
    let diff = max_abs_diff_f32(
        &out_test.to_dtype(DType::F32)?,
        &out_ref.to_dtype(DType::F32)?,
    )?;
    if diff >= DIFF_TOLERANCE {
        candle::bail!(
            "interspersed-glue diverged (la={la}, g={g}, lb={lb}): \
             max abs diff = {diff:.6e} (expected < {:.0e})",
            DIFF_TOLERANCE,
        );
    }
    Ok(diff)
}

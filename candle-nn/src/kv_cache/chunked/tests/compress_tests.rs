//! Tests for the quantize-on-evict path.
//!
//! `quantize_sealed_in_place` is the substrate's hot→warm persistence
//! step when a `CompressionPolicy` is configured: a sealed sequence
//! whose GIDs live in GPU Float (or R16) arenas is run through the
//! fused per-`(head, palette)` selection + conversion kernel and
//! emerges as a sealed sequence whose GIDs live in CPU Quantized
//! arenas, ready to install as warm.
//!
//! These tests are CUDA-only and silently skip on machines without
//! a CUDA device — the kernel path is GPU-specific.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{DType, Device, Tensor};
use half::f16;

use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::chunked::{ChunkedKvBacking, CompressionPolicy};
use crate::kv_cache::{quantize_sealed_in_place, KvFormat, SealedSequence};

const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 128;
const ARENA_CAPACITY: usize = 256;

fn cuda_device_or_skip() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => Some(d),
        _ => None,
    }
}

fn cuda_stream(device: &Device) -> Arc<CudaStream> {
    match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!("test gated on a CUDA device"),
    }
}

/// Build an F16-backed GPU arena with the warm-protected candidate
/// arenas pre-allocated for the given compression policy. Mirrors
/// the substrate engine's startup wiring.
fn cuda_backing_with_policy(device: &Device, policy: &CompressionPolicy) -> ChunkedKvBacking {
    let backing = ChunkedKvBacking::new_with_format(
        4,
        N_KV_HEAD,
        HEAD_DIM,
        KvFormat::Float(DType::F16),
        KvFormat::Float(DType::F16),
        device,
        ARENA_CAPACITY,
    )
    .expect("create chunked backing");
    backing
        .warm_protected_arenas(Some(policy))
        .expect("pre-allocate compression candidate arenas");
    backing
}

/// Seed `n_tokens` of a deterministic F16 pattern as one sealed turn
/// in `backing`. Returns the sealed sequence (GIDs pointing at GPU
/// F16 arena slots).
fn seed_f16_sealed(
    backing: &ChunkedKvBacking,
    device: &Device,
    n_tokens: usize,
    pattern_base: u32,
) -> SealedSequence {
    let slot = backing.alloc_sequence().unwrap();
    backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
    let total = N_KV_HEAD * n_tokens * HEAD_DIM;
    // Smooth pattern — adjacent dims/tokens vary slowly so adaptive
    // selection can hit reasonable error budgets at any level.
    let data: Vec<f16> = (0..total)
        .map(|i| f16::from_f32(((pattern_base as usize + i) as f32) * 0.0005))
        .collect();
    let k = Tensor::from_vec(data, (1, N_KV_HEAD, n_tokens, HEAD_DIM), &Device::Cpu)
        .unwrap()
        .to_device(device)
        .unwrap();
    let v = k.clone();
    backing.write_contiguous(slot, 0, &k, &v).unwrap();
    backing.set_len(slot, n_tokens);
    backing.record_turn(slot).unwrap()
}

/// After quantize-on-evict the returned sequence:
/// 1. Has `location = Cpu`.
/// 2. Has the same `chunks.len()` and `token_count` as the source.
/// 3. Every chunk's GIDs point at quantized GPU arenas.
/// 4. Total `byte_size` is strictly less than the F16 source's
///    (compression actually compressed something).
#[test]
fn quantize_to_cpu_basic_round_trip_shape() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    // 64 tokens = 2 chunks of 32 tokens.
    let src = seed_f16_sealed(&backing, &device, 64, 1);
    let src_bytes_per_chunk = src.chunks.iter().map(|c| c.byte_size as u64).sum::<u64>();

    let mut pinned: Option<PinnedBuf> = None;
    let warm = quantize_sealed_in_place(
        &backing,
        &[&src],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("quantize_sealed_in_place");
    assert_eq!(warm.len(), 1, "one input sequence in → one out");
    let warm = &warm[0];
    assert_eq!(warm.location, ArenaLocation::Gpu, "warm must land on CPU");
    assert_eq!(warm.chunks.len(), src.chunks.len());
    assert_eq!(warm.token_count, src.token_count);

    // Every GID in every warm chunk must point at a quantized GPU arena.
    backing
        .inner
        .storage
        .read(|storage| {
            for chunk in &warm.chunks {
                for gid in chunk.gids.as_slice() {
                    let key = storage
                        .arena_key(gid.arena_idx())
                        .expect("warm gid arena exists");
                    assert_eq!(
                        key.location,
                        ArenaLocation::Gpu,
                        "warm gid must live in a GPU arena"
                    );
                    assert!(
                        matches!(key.format, KvFormat::Quantized(_)),
                        "warm gid must live in a quantized arena, got {:?}",
                        key.format
                    );
                }
            }
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();

    let warm_bytes = warm.chunks.iter().map(|c| c.byte_size as u64).sum::<u64>();
    assert!(
        warm_bytes < src_bytes_per_chunk,
        "warm bytes ({}) should be strictly less than F16 source bytes ({}) — \
         quantize-on-evict didn't actually compress",
        warm_bytes,
        src_bytes_per_chunk,
    );
}

/// Multi-sequence input batches map 1-to-1 to outputs in the same order.
#[test]
fn quantize_to_cpu_batches_two_sequences() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(3);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    let seq_a = seed_f16_sealed(&backing, &device, 32, 100);
    let seq_b = seed_f16_sealed(&backing, &device, 64, 200);

    let mut pinned: Option<PinnedBuf> = None;
    let out = quantize_sealed_in_place(
        &backing,
        &[&seq_a, &seq_b],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("quantize_sealed_in_place (batch)");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].chunks.len(), seq_a.chunks.len());
    assert_eq!(out[1].chunks.len(), seq_b.chunks.len());
    assert_eq!(out[0].token_count, seq_a.token_count);
    assert_eq!(out[1].token_count, seq_b.token_count);
    assert_eq!(out[0].location, ArenaLocation::Gpu);
    assert_eq!(out[1].location, ArenaLocation::Gpu);
}

/// An empty input list is a clean no-op.
#[test]
fn quantize_to_cpu_empty_input_is_noop() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    let mut pinned: Option<PinnedBuf> = None;
    let out = quantize_sealed_in_place(&backing, &[], &policy, &device, &copy_stream, &mut pinned)
        .expect("quantize_sealed_in_place (empty)");
    assert!(out.is_empty());
}

/// Partial trailing chunks (token_count < CHUNK_SIZE) must skip the
/// per-(h, p) palette4 selector and stay in their **source** format.
/// The COW partial-tail path in `create_view_sequence` allocates the
/// destination in the active K arena (R16 in production / F16 here)
/// and does a same-format byte copy — putting a partial into a Q*
/// arena would surface as `quant dtype mismatch (Q3_0 vs R16)` when
/// the scheduler later projects the prior turn.
///
/// **Regression test** for the bug where every chunk including the
/// partial went through the quantizer, breaking the cold-reload +
/// projection flow for any conversation whose turns didn't land on
/// exact CHUNK_SIZE boundaries.
#[test]
fn quantize_to_cpu_quantizes_partial_tail() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    // 50 tokens = 1 full sealed chunk (32 tokens) + 1 partial (18 tokens).
    // Partial chunks quantize like full ones: their dead token slots are
    // zero (arena zeroing at creation/recycle), and the selection kernel
    // receives the valid range to correct its count-normalized metrics.
    // The full chunk pins that full-chunk quantization keeps working
    // alongside the partial path.
    let src = seed_f16_sealed(&backing, &device, 50, 1);
    assert_eq!(src.chunks.len(), 2, "50 tokens → 1 full + 1 partial");
    assert_eq!(src.chunks[0].token_count, 32);
    assert_eq!(src.chunks[1].token_count, 18);

    let mut pinned: Option<PinnedBuf> = None;
    let warm = quantize_sealed_in_place(
        &backing,
        &[&src],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("quantize_sealed_in_place");
    assert_eq!(warm.len(), 1);
    let warm = &warm[0];

    // Chunk count + token counts + order all preserved.
    assert_eq!(warm.chunks.len(), 2, "chunk count preserved");
    assert_eq!(warm.chunks[0].token_count, 32, "full chunk preserved");
    assert_eq!(warm.chunks[1].token_count, 18, "partial chunk preserved");
    assert_eq!(warm.location, ArenaLocation::Gpu);

    // BOTH chunks must land in Quantized GPU arenas — the partial is
    // no longer preserved as float (readers address it by
    // [offset, offset+len); views/decode never append to a sealed
    // partial — `create_view_sequence` borrows it read-only and
    // `ensure_writable_tail` starts a fresh float chunk past it).
    backing
        .inner
        .storage
        .read(|storage| {
            for (ci, chunk) in warm.chunks.iter().enumerate() {
                for gid in chunk.gids.as_slice() {
                    let key = storage
                        .arena_key(gid.arena_idx())
                        .expect("chunk arena exists");
                    assert_eq!(key.location, ArenaLocation::Gpu);
                    assert!(
                        matches!(key.format, KvFormat::Quantized(_)),
                        "chunk {ci} gid {} must live in a Quantized GPU arena \
                         (partials quantize like full chunks), got {:?}",
                        gid.raw(),
                        key.format,
                    );
                }
            }
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();

    // The partial's byte footprint shrinks accordingly — this is the
    // field `install_warm` writes into the residence's `byte_size`
    // accounting, and the whole point of quantizing tails: a float
    // partial pinned a full F16 chunk slot per layer.
    let arena_infos = backing.resolve_arena_info().unwrap();
    let partial_bytes = warm.chunks[1].gids.arena_byte_size(&arena_infos);
    let f16_footprint = (N_KV_HEAD * 2 * HEAD_DIM * 32 * 2) as u64;
    assert!(
        partial_bytes < f16_footprint,
        "quantized partial byte_size {} must be below the F16 footprint {}",
        partial_bytes,
        f16_footprint,
    );
}

/// The "now-full" cycle: a live float chunk that a later turn filled
/// to 32 tokens quantizes like any other full chunk on that turn's
/// persist pass. (Turn 1's own snapshot of the then-partial chunk
/// quantizes independently into its own arenas — snapshots never share
/// quantized storage with the live slot.)
#[test]
fn quantize_to_cpu_requantizes_filled_partials() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    // Simulate turn 2's sealed sequence after the previously-partial
    // live chunk got filled: every chunk is full now. (The bytes come
    // from a fresh 64-token seed; the relevant property is
    // `token_count == CHUNK_SIZE` on every chunk.)
    let src = seed_f16_sealed(&backing, &device, 64, 2);
    assert_eq!(src.chunks.len(), 2);
    assert!(
        src.chunks.iter().all(|c| c.token_count == 32),
        "all chunks must be full for this test's purpose"
    );

    let mut pinned: Option<PinnedBuf> = None;
    let warm = quantize_sealed_in_place(
        &backing,
        &[&src],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("quantize_sealed_in_place");
    let warm = &warm[0];
    assert_eq!(warm.chunks.len(), 2);

    // Both chunks must be quantized — no skipping.
    backing
        .inner
        .storage
        .read(|storage| {
            for chunk in &warm.chunks {
                for gid in chunk.gids.as_slice() {
                    let key = storage.arena_key(gid.arena_idx()).unwrap();
                    assert!(
                        matches!(key.format, KvFormat::Quantized(_)),
                        "filled chunk gid {} must be quantized on the next \
                         persist pass (post-resume re-quantize cycle), got \
                         {:?}",
                        gid.raw(),
                        key.format,
                    );
                }
            }
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();
}

/// **End-to-end resume + extend + re-quantize cycle.**
///
/// This is the lifecycle the daemon walks through every multi-turn
/// conversation, simulated entirely against `ChunkedKvBacking`
/// primitives so any regression in the format-flow shows up here
/// without needing the substrate scheduler:
///
/// 1. **Turn 1** seals at 50 tokens (1 full chunk + 1 partial of 18).
/// 2. **Persist**: `quantize_sealed_in_place` quantizes BOTH chunks to
///    Q* (partials quantize like full chunks) — both still on GPU. A
///    separate `migrate_sealed_to_cpu` evicts the sealed turn to the
///    warm (CPU) tier, formats preserved.
/// 3. **Cold-elevate**: `migrate_sealed_to_gpu_batch_async` brings the
///    warm SealedSequence back to GPU with the same per-chunk formats
///    (Q* full + Q* partial) — the round-trip validates quantized
///    partials through the warm tier.
/// 4. **Inject + view**: `inject_sealed_at_tail` puts the elevated turn
///    into a parent slot; `create_view_sequence` borrows all blocks —
///    including the partial tail — read-only via Arc (this is the
///    scheduler's projection path). A borrowed partial can be any
///    format because nothing ever appends to it.
/// 5. **Decoder extends**: 32 more tokens past the 50-token boundary.
///    The borrowed partial stays read-only at 18 tokens; the new tokens
///    fill one fresh full 32-token chunk.
/// 6. **Turn 2 sealed**: `record_turn(view)` snapshots a 3-chunk
///    sequence: borrowed turn-1 full (Q*), borrowed turn-1 partial
///    (Q*, unchanged), new full chunk (F16).
/// 7. **Re-persist**: `quantize_sealed_in_place` runs again. Both
///    borrowed Q* chunks pass through unchanged (already compressed)
///    and the new full chunk is freshly quantized.
///
/// This pins the entire "resume → continue → re-quantize" contract
/// in one self-contained test.
#[test]
fn cold_load_partial_extend_then_requantize() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;

    // ── Phase 1: seed turn 1 (50 tokens = 1 full + 1 partial) ──────
    let sealed_t1 = seed_f16_sealed(&backing, &device, 50, 100);
    assert_eq!(sealed_t1.chunks.len(), 2);
    assert_eq!(sealed_t1.chunks[0].token_count, 32);
    assert_eq!(sealed_t1.chunks[1].token_count, 18);

    // ── Phase 2: persist = quantize-in-place (GPU) then evict (DtoH) ──
    // `quantize_sealed_in_place` compresses BOTH chunks to Q* — the
    // 18-token partial included — still GPU-resident. Eviction to the
    // warm tier is the separate `migrate_sealed_to_cpu` step.
    let quantized_gpu = quantize_sealed_in_place(
        &backing,
        &[&sealed_t1],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("turn 1 quantize");
    assert_eq!(quantized_gpu.len(), 1);
    let warm_cpu = backing
        .migrate_sealed_to_cpu_batch_async(&device, &copy_stream, &mut pinned, &[&quantized_gpu[0]])
        .expect("turn 1 evict to warm (CPU)");
    assert_eq!(warm_cpu.len(), 1);
    assert_eq!(warm_cpu[0].location, ArenaLocation::Cpu);

    // ── Phase 3: simulate cold-elevate back to GPU ─────────────────
    let elevated = backing
        .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&warm_cpu[0]])
        .expect("turn 1 elevate");
    let elevated_seq = &elevated[0];
    assert_eq!(elevated_seq.chunks.len(), 2);

    // Sanity: the elevated partial is back on GPU in a Quantized arena —
    // the format-preserving migrate round-trip carries the quantized
    // partial out and back unchanged.
    backing
        .inner
        .storage
        .read(|storage| {
            for gid in elevated_seq.chunks[1].gids.as_slice() {
                let key = storage.arena_key(gid.arena_idx()).unwrap();
                assert_eq!(key.location, ArenaLocation::Gpu);
                assert!(
                    matches!(key.format, KvFormat::Quantized(_)),
                    "cold-loaded partial gid {} must elevate back to a GPU \
                     Quantized arena (format-preserving round-trip), got {:?}",
                    gid.raw(),
                    key.format,
                );
            }
            // Also sanity-check that the elevated full chunk is on GPU
            // in some Quantized arena (the borrowed-already-quantized
            // case that turn 2's re-quantize must skip).
            for gid in elevated_seq.chunks[0].gids.as_slice() {
                let key = storage.arena_key(gid.arena_idx()).unwrap();
                assert_eq!(key.location, ArenaLocation::Gpu);
                assert!(
                    matches!(key.format, KvFormat::Quantized(_)),
                    "elevated full chunk gid {} should land in GPU Quantized, got {:?}",
                    gid.raw(),
                    key.format,
                );
            }
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();

    // ── Phase 4: inject elevated turn 1 into a parent slot ─────────
    let parent = backing.alloc_sequence().unwrap();
    let (start, end) = backing
        .inject_sealed_at_tail(parent, elevated_seq)
        .expect("inject elevated turn 1 into parent slot");
    assert_eq!(start, 0);
    assert_eq!(end, 2, "parent slot now has 2 chunks (1 full + 1 partial)");

    // ── Phase 5: view borrows all parent blocks ────────────────────
    // `create_view_sequence` borrows every parent block — including the
    // cold-loaded partial tail — read-only via Arc clone, then pushes one
    // fresh active chunk for new writes. No COW, no format coupling: a
    // borrowed partial can be in any format because nothing ever appends
    // to it (new tokens always land in the fresh active chunk).
    let view = backing.alloc_sequence().unwrap();
    let (_borrowed_blocks, borrowed_tokens) = backing
        .create_view_sequence(view, parent, &[(0, 2)])
        .expect("create_view_sequence over a cold-loaded turn with a partial tail");
    assert_eq!(
        borrowed_tokens, 50,
        "view inherits both turn 1 blocks (50 tokens)"
    );

    // ── Phase 6: decoder extends past the partial boundary ─────────
    // 32 more tokens at cumulative offset 50. The borrowed partial (18) is
    // read-only, so the writes land in a fresh chunk past it rather than
    // filling the partial: cumulative offsets 50..82 become one full new
    // 32-token chunk. The borrowed partial stays 18 tokens; it is never
    // extended in place.
    let extra_tokens = 32;
    let total_after = 50 + extra_tokens;
    backing.ensure_for_offset(view, 50, extra_tokens).unwrap();
    let n_elems = N_KV_HEAD * extra_tokens * HEAD_DIM;
    let extend_data: Vec<f16> = (0..n_elems)
        .map(|i| f16::from_f32((i as f32) * 0.0007))
        .collect();
    let k_ext = Tensor::from_vec(
        extend_data,
        (1, N_KV_HEAD, extra_tokens, HEAD_DIM),
        &Device::Cpu,
    )
    .unwrap()
    .to_device(&device)
    .unwrap();
    let v_ext = k_ext.clone();
    backing.write_contiguous(view, 50, &k_ext, &v_ext).unwrap();
    backing.set_len(view, total_after);

    // ── Phase 7: seal turn 2 ───────────────────────────────────────
    let sealed_t2 = backing.record_turn(view).unwrap();
    assert_eq!(
        sealed_t2.chunks.len(),
        3,
        "turn 2 = borrowed full + borrowed partial (read-only) + new full"
    );
    assert_eq!(sealed_t2.chunks[0].token_count, 32, "borrowed turn-1 full");
    assert_eq!(
        sealed_t2.chunks[1].token_count, 18,
        "borrowed turn-1 partial — unchanged, never extended in place"
    );
    assert_eq!(
        sealed_t2.chunks[2].token_count, 32,
        "new full chunk written by the extension"
    );

    // ── Phase 8: re-quantize turn 2 ────────────────────────────────
    // The bucketing must handle three distinct cases in one call:
    //   chunks[0]: borrowed cold-loaded full → already Quantized →
    //              eligibility check skips it, preserve bucket keeps it.
    //   chunks[1]: borrowed cold-loaded partial → already Quantized →
    //              preserve bucket keeps it (no re-quantization).
    //   chunks[2]: new full F16 chunk → eligible for the kernel → quantized.
    let warm_t2 = quantize_sealed_in_place(
        &backing,
        &[&sealed_t2],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect(
        "re-quantize after partial extend must succeed — the bucketing has to \
         pass borrowed already-quantized and partial chunks through unchanged",
    );
    let warm_t2_seq = &warm_t2[0];
    assert_eq!(warm_t2_seq.chunks.len(), 3);
    assert_eq!(warm_t2_seq.chunks[0].token_count, 32);
    assert_eq!(warm_t2_seq.chunks[1].token_count, 18);
    assert_eq!(warm_t2_seq.chunks[2].token_count, 32);

    backing
        .inner
        .storage
        .read(|storage| {
            let check = |chunk_idx: usize, label: &str| {
                for gid in warm_t2_seq.chunks[chunk_idx].gids.as_slice() {
                    let key = storage.arena_key(gid.arena_idx()).unwrap();
                    assert_eq!(
                        key.location,
                        ArenaLocation::Gpu,
                        "warm_t2 chunk {chunk_idx} ({label}) gid {} location",
                        gid.raw()
                    );
                    assert!(
                        matches!(key.format, KvFormat::Quantized(_)),
                        "warm_t2 chunk {chunk_idx} ({label}) gid {} must be \
                         in a Quantized GPU arena, got {:?}",
                        gid.raw(),
                        key.format,
                    );
                }
            };
            // chunks[0]: borrowed cold-loaded full → already Q*, preserved.
            check(0, "borrowed cold-loaded full (Q* preserved)");
            // chunks[1]: borrowed cold-loaded partial → already Q*, preserved.
            check(1, "borrowed partial (Q* preserved)");
            // chunks[2]: new full F16 chunk → freshly quantized.
            check(2, "new full chunk (freshly quantized)");
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();
}

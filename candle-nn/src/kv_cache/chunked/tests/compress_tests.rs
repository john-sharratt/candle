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
    backing.record_turn(slot, n_tokens).unwrap()
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
    let src_bytes_per_chunk = src
        .chunks
        .iter()
        .map(|c| c.byte_size as u64)
        .sum::<u64>();

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
    let out = quantize_sealed_in_place(
        &backing,
        &[],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
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
fn quantize_to_cpu_preserves_partial_chunk_format() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    // 50 tokens = 1 full sealed chunk (32 tokens) + 1 partial (18 tokens).
    // The bug only fires on partials, so we need at least one of each
    // — the full chunk also pins that full-chunk quantization keeps
    // working alongside the partial-skip branch.
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

    // Full chunk must land in a Quantized GPU arena; partial chunk
    // must stay in the source F16 Float GPU arena (the backing's K/V
    // active format).
    backing
        .inner
        .storage
        .read(|storage| {
            for gid in warm.chunks[0].gids.as_slice() {
                let key = storage
                    .arena_key(gid.arena_idx())
                    .expect("full chunk arena exists");
                assert_eq!(key.location, ArenaLocation::Gpu);
                assert!(
                    matches!(key.format, KvFormat::Quantized(_)),
                    "full chunk gid {} must live in a Quantized GPU arena, got {:?}",
                    gid.raw(),
                    key.format,
                );
            }
            for gid in warm.chunks[1].gids.as_slice() {
                let key = storage
                    .arena_key(gid.arena_idx())
                    .expect("partial chunk arena exists");
                assert_eq!(
                    key.location,
                    ArenaLocation::Gpu,
                    "partial chunk gid {} location",
                    gid.raw()
                );
                assert_eq!(
                    key.format,
                    KvFormat::Float(candle::DType::F16),
                    "partial chunk gid {} must stay in the source F16 Float \
                     arena — quantizing would break `create_view_sequence`'s \
                     COW partial-tail byte copy",
                    gid.raw(),
                );
            }
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();

    // Source-format preservation extends to byte_size: the partial's
    // arena_byte_size reflects the F16 footprint, not a (smaller)
    // quantized footprint. This is the field `install_warm` writes
    // into the residence's `byte_size` accounting.
    let arena_infos = backing.resolve_arena_info().unwrap();
    let partial_bytes = warm.chunks[1].gids.arena_byte_size(&arena_infos);
    let expected_min_bytes = (N_KV_HEAD * 2 * HEAD_DIM * 32 * 2) as u64;
    assert!(
        partial_bytes >= expected_min_bytes,
        "partial chunk byte_size {} should reflect an F16-sized \
         arena slot (≥ {} bytes), not a quantized footprint",
        partial_bytes,
        expected_min_bytes,
    );
}

/// The "now-full" cycle: a chunk that started life as a partial in
/// turn 1 becomes full after turn 2's decode fills the remaining
/// slots. On turn 2's persist pass, `quantize_sealed_in_place` must
/// treat the now-full chunk like any other full chunk and route it
/// through the palette4 selector. This pins the resume → continue →
/// re-quantize lifecycle: the second persist actually re-compresses
/// the previously-skipped block.
#[test]
fn quantize_to_cpu_requantizes_filled_partials() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let policy = CompressionPolicy::new(5);
    let backing = cuda_backing_with_policy(&device, &policy);
    let copy_stream = cuda_stream(&device);

    // Simulate turn 2's sealed sequence after the previously-partial
    // chunk got filled: pretend every chunk is full now. (The bytes
    // come from a fresh 64-token seed; the relevant property is
    // `token_count == CHUNK_SIZE` on every chunk, which is what the
    // quantize-vs-skip decision keys on.)
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
/// 2. **Persist**: `quantize_sealed_in_place` runs. Full chunk lands
///    in CPU Quantized; partial stays format-preserved (F16 here,
///    R16 in production).
/// 3. **Cold-elevate**: `migrate_sealed_to_gpu_batch_async` brings
///    the warm SealedSequence back to GPU with the same per-chunk
///    formats (Q* full + F16 partial).
/// 4. **Inject + view**: `inject_sealed_at_tail` puts the elevated
///    turn into a parent slot; `create_view_sequence` borrows all
///    blocks (this is exactly the scheduler's projection path).
///    The COW partial-tail step would fail with `quant dtype
///    mismatch` if the partial weren't in the active K format.
/// 5. **Decoder extends**: 30 more tokens go past the 50-token
///    boundary. First 14 fill the COW'd partial (18 → 32), then
///    16 land in a fresh new partial.
/// 6. **Turn 2 sealed**: `record_turn(view, 80)` snapshots a
///    3-chunk sequence: borrowed turn-1 full (Q*), formerly-partial
///    now-full (F16), new partial (F16).
/// 7. **Re-persist**: `quantize_sealed_in_place` runs again. The
///    borrowed Q* chunk must pass through to warm format-preserving
///    (already compressed). The COW'd-and-filled chunk must be
///    freshly quantized. The new partial must stay format-preserved.
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

    // ── Phase 2: simulate persistence Phase 1 (quantize on evict) ──
    let warm_cpu = quantize_sealed_in_place(
        &backing,
        &[&sealed_t1],
        &policy,
        &device,
        &copy_stream,
        &mut pinned,
    )
    .expect("turn 1 quantize");
    assert_eq!(warm_cpu.len(), 1);

    // ── Phase 3: simulate cold-elevate back to GPU ─────────────────
    let elevated = backing
        .migrate_sealed_to_gpu_batch_async(&device, &copy_stream, &mut pinned, &[&warm_cpu[0]])
        .expect("turn 1 elevate");
    let elevated_seq = &elevated[0];
    assert_eq!(elevated_seq.chunks.len(), 2);

    // Sanity: the elevated partial is back on GPU in F16 Float, so
    // the COW partial-tail byte copy in `create_view_sequence` will
    // succeed (same-format source ↔ destination).
    backing
        .inner
        .storage
        .read(|storage| {
            for gid in elevated_seq.chunks[1].gids.as_slice() {
                let key = storage.arena_key(gid.arena_idx()).unwrap();
                assert_eq!(key.location, ArenaLocation::Gpu);
                assert_eq!(
                    key.format,
                    KvFormat::Float(DType::F16),
                    "cold-loaded partial gid {} must elevate back to GPU F16 \
                     Float so the COW partial-tail path can byte-copy it",
                    gid.raw(),
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
    // The COW partial-tail path inside create_view_sequence is the
    // exact line that hit `quant dtype mismatch (Q3_0 vs R16)` in
    // production before the partial-skip fix.
    let view = backing.alloc_sequence().unwrap();
    let (_borrowed_blocks, borrowed_tokens) = backing
        .create_view_sequence(view, parent, &[(0, 2)])
        .expect(
            "create_view_sequence over a cold-loaded turn with a partial tail \
             must succeed — the COW byte copy needs same-format arenas",
        );
    assert_eq!(
        borrowed_tokens, 50,
        "view inherits both turn 1 blocks (50 tokens)"
    );

    // ── Phase 6: decoder extends past the partial boundary ─────────
    // 30 more tokens at offset 50:
    //   - 50..64 fills the view's COW'd-from-partial block (18 → 32).
    //   - 64..80 lands in a fresh block (16 tokens — new partial).
    let extra_tokens = 30;
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
    let sealed_t2 = backing.record_turn(view, total_after).unwrap();
    assert_eq!(
        sealed_t2.chunks.len(),
        3,
        "turn 2 = 1 borrowed full + 1 COW'd-filled + 1 new partial"
    );
    assert_eq!(sealed_t2.chunks[0].token_count, 32, "borrowed turn-1 full");
    assert_eq!(
        sealed_t2.chunks[1].token_count, 32,
        "COW'd-and-filled block (was the cold-loaded partial)"
    );
    assert_eq!(
        sealed_t2.chunks[2].token_count, 16,
        "new partial appended by the extension"
    );

    // ── Phase 8: re-quantize turn 2 ────────────────────────────────
    // The bucketing must handle three distinct cases in one call:
    //   chunks[0]: borrowed cold-loaded full → already Quantized →
    //              passes through to warm format-preserving (the
    //              eligibility check skips it, the preserve bucket
    //              picks it up).
    //   chunks[1]: COW'd-and-filled → was F16 (the COW dst format),
    //              now full → eligible for the kernel → re-quantized.
    //   chunks[2]: new partial → token_count < CHUNK_SIZE → preserve.
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
         pass borrowed already-quantized chunks through to warm format-preserving",
    );
    let warm_t2_seq = &warm_t2[0];
    assert_eq!(warm_t2_seq.chunks.len(), 3);
    assert_eq!(warm_t2_seq.chunks[0].token_count, 32);
    assert_eq!(warm_t2_seq.chunks[1].token_count, 32);
    assert_eq!(warm_t2_seq.chunks[2].token_count, 16);

    backing
        .inner
        .storage
        .read(|storage| {
            let check = |chunk_idx: usize, expect_quant: bool, label: &str| {
                for gid in warm_t2_seq.chunks[chunk_idx].gids.as_slice() {
                    let key = storage.arena_key(gid.arena_idx()).unwrap();
                    assert_eq!(
                        key.location,
                        ArenaLocation::Gpu,
                        "warm_t2 chunk {chunk_idx} ({label}) gid {} location",
                        gid.raw()
                    );
                    if expect_quant {
                        assert!(
                            matches!(key.format, KvFormat::Quantized(_)),
                            "warm_t2 chunk {chunk_idx} ({label}) gid {} must be \
                             in a Quantized GPU arena, got {:?}",
                            gid.raw(),
                            key.format,
                        );
                    } else {
                        assert_eq!(
                            key.format,
                            KvFormat::Float(DType::F16),
                            "warm_t2 chunk {chunk_idx} ({label}) gid {} must be \
                             F16 Float (preserved), got {:?}",
                            gid.raw(),
                            key.format,
                        );
                    }
                }
            };
            // chunks[0]: borrowed cold-loaded full → format-preserved
            //            from GPU Q* to CPU Q* (still Quantized, just
            //            different location).
            check(0, true, "borrowed cold-loaded full (Q* preserved)");
            // chunks[1]: COW'd-and-filled → freshly quantized.
            check(1, true, "COW'd-and-filled (freshly re-quantized)");
            // chunks[2]: new partial → format-preserved F16.
            check(2, false, "new partial (F16 preserved)");
            Ok::<(), candle::Error>(())
        })
        .unwrap()
        .unwrap();
}

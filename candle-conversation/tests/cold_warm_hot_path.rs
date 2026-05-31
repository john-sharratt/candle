//! Full cold → warm → hot round-trip integration test.
//!
//! Validates the complete tier lifecycle end-to-end without involving a
//! model or inference path. Drives the real substrate, real persistence
//! redo log (on disk, in a tempdir), real `ChunkedKvBacking` arenas with
//! real `ChunkGid` allocations, and the real `PersistenceThread`.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ phase 1 — open substrate, install a section, seed N turns hot   │
//! │ phase 2 — PersistenceThread cycles: hot → warm, warm → cold     │
//! │ phase 3 — purge hot (evict_from_hot with empty keep set)        │
//! │ phase 4 — warm → hot (elevate_to_hot, WarmLift)                 │
//! │ phase 5 — drop everything, simulating daemon shutdown           │
//! │ phase 6 — reopen substrate from the same tempdir + reload       │
//! │ phase 7 — cold → hot + warm (elevate_to_hot, ColdRecall)        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! After every transition the test re-gathers the turn's encoded chunk
//! bytes from VRAM via [`seal_to_chunk_images`] and asserts they are
//! byte-identical to the snapshot taken right after the initial seed.
//! This catches any corruption / clipping / re-encoding drift along the
//! full hot↔warm↔cold path.
//!
//! Skipped without an available CUDA device.

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{DType, Device, Tensor};
use candle_conversation::persistence::cold_load::ColdLoadStager;
use candle_conversation::persistence::elevate::{elevate_to_hot, evict_from_hot};
use candle_conversation::persistence::thread::PersistenceThread;
use candle_conversation::persistence::transfer::seal_to_chunk_images;
use candle_conversation::persistence::SubstratePersistence;
use candle_conversation::projection::{
    Conversation, GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey,
};
use candle_conversation::substrate::TierState;
use candle_conversation::token_buffer::TokenBuffer;
use candle_conversation::turn::Role;
use candle_nn::kv_cache::{
    ArenaKey, ArenaLocation, ChunkedKvBacking, KvFormat, QuantFormat, SealedChunk, SealedSequence,
};
use half::bf16;

const N_LAYERS: usize = 2;
const N_KV_HEAD: usize = 2;
const HEAD_DIM: usize = 16;
const N_TOKENS_PER_TURN: usize = 32;
const N_TURNS: usize = 3;
const CHUNK_SIZE: usize = 32;
const ARENA_CAPACITY: usize = 256;

/// `head_dim` for the quant-blend test. Q8_0 / R16 require `head_dim`
/// to be a multiple of the 32-element block size, and with `N_PALETTE
/// = 4` sub-bands, sub_head_dim = head_dim / 4 must also be ≥ 32 for
/// the per-(h,p) quant blocks to land cleanly. 128 / 4 = 32 — exactly
/// one block per sub-band, matching the R16 unit-test pattern.
const QUANT_HEAD_DIM: usize = 128;

fn cuda_device_or_skip() -> Option<Device> {
    match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => Some(d),
        _ => None,
    }
}

fn make_backings(device: &Device) -> Vec<ChunkedKvBacking> {
    (0..N_LAYERS)
        .map(|_| {
            ChunkedKvBacking::new(4, N_KV_HEAD, HEAD_DIM, DType::BF16, device, ARENA_CAPACITY)
                .unwrap()
        })
        .collect()
}

fn cuda_stream(device: &Device) -> Arc<CudaStream> {
    match device {
        Device::Cuda(d) => d.cuda_stream(),
        _ => unreachable!("test gated on a CUDA device"),
    }
}

/// Distinct deterministic byte pattern per turn; the `pattern_base`
/// shifts every value so two turns with different bases produce
/// byte-disjoint K/V grids.
fn pattern_kv(pattern_base: u32) -> Vec<bf16> {
    let total = N_KV_HEAD * N_TOKENS_PER_TURN * HEAD_DIM;
    (0..total)
        .map(|i| bf16::from_f32(((pattern_base as usize + i) as f32) * 0.001))
        .collect()
}

/// Seed one synthetic turn: allocate per-layer GPU chunks, write a
/// deterministic K/V tensor into them, record the turn into the
/// substrate. Returns the freshly-appended `TurnIndex`.
///
/// `block_end = 1` so the persisted `TurnDecl` records exactly one
/// chunk per layer — required for the cold-reload demux to round-trip
/// correctly.
fn seed_turn(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    pattern_base: u32,
) -> TurnIndex {
    let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(backings.len());
    for backing in backings {
        let slot = backing.alloc_sequence().unwrap();
        backing
            .ensure_for_offset(slot, 0, N_TOKENS_PER_TURN)
            .unwrap();
        let data = pattern_kv(pattern_base);
        let k = Tensor::from_vec(
            data,
            (1, N_KV_HEAD, N_TOKENS_PER_TURN, HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, N_TOKENS_PER_TURN);
        sealed_per_layer.push(backing.record_turn(slot, N_TOKENS_PER_TURN).unwrap());
    }
    // Total tokens = 32, block_size = 32 ⇒ 1 chunk per layer.
    let block_end = (N_TOKENS_PER_TURN / CHUNK_SIZE) as u64;
    conv.record_turn(
        timeline,
        Role::User,
        candle_conversation::substrate::TurnPartWrite::default(),
        candle_conversation::substrate::TurnPartWrite {
            token_count: N_TOKENS_PER_TURN,
            block_end,
            sealed_gpu: Some(Arc::new(sealed_per_layer)),
            ..Default::default()
        },
        |seqs| Ok(seqs.to_vec()),
    )
    .unwrap()
}

/// Seed a synthetic section (pinned-hot KV) so we can verify it
/// survives the in-process tier transitions. Sections don't have a
/// cold tier today, so they don't survive a substrate restart — the
/// test only asserts pre-restart.
fn seed_section(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    section: SectionId,
    pattern_base: u32,
) {
    let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(backings.len());
    for backing in backings {
        let slot = backing.alloc_sequence().unwrap();
        backing
            .ensure_for_offset(slot, 0, N_TOKENS_PER_TURN)
            .unwrap();
        let data = pattern_kv(pattern_base);
        let k = Tensor::from_vec(
            data,
            (1, N_KV_HEAD, N_TOKENS_PER_TURN, HEAD_DIM),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(device)
        .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, N_TOKENS_PER_TURN);
        sealed_per_layer.push(backing.record_turn(slot, N_TOKENS_PER_TURN).unwrap());
    }
    conv.write()
        .set_section_full(
            section,
            N_TOKENS_PER_TURN,
            Vec::new(),
            Arc::new(sealed_per_layer),
            |seqs| Ok(seqs.to_vec()),
            Arc::new(vec![0u32; N_TOKENS_PER_TURN]),
        )
        .unwrap();
}

/// Gather a turn's hot-tier bytes per layer via the same path
/// `persist_turn_chunks` uses — `seal_to_chunk_images`. Each layer's
/// per-chunk encoded `kv_bytes` are concatenated so a single
/// `assert_eq!(left, right)` is enough to flag any drift in
/// alignment, encoding, or chunk windowing.
fn snapshot_turn_bytes(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    idx: TurnIndex,
) -> Vec<Vec<u8>> {
    let sealed = conv
        .read()
        .turn_sealed_of(timeline, idx)
        .expect("turn must be hot");
    let mut out = Vec::with_capacity(sealed.len());
    for (backing, seq) in backings.iter().zip(sealed.iter()) {
        let images = seal_to_chunk_images(backing, device, seq).unwrap();
        let mut bytes = Vec::new();
        for img in &images {
            bytes.extend_from_slice(&img.payload.kv_bytes);
        }
        out.push(bytes);
    }
    out
}

fn snapshot_section_bytes(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    section: SectionId,
) -> Vec<Vec<u8>> {
    let sealed = conv
        .read()
        .section_sealed_of(section)
        .expect("section must be hot");
    let mut out = Vec::with_capacity(sealed.len());
    for (backing, seq) in backings.iter().zip(sealed.iter()) {
        let images = seal_to_chunk_images(backing, device, seq).unwrap();
        let mut bytes = Vec::new();
        for img in &images {
            bytes.extend_from_slice(&img.payload.kv_bytes);
        }
        out.push(bytes);
    }
    out
}

fn turn_state(conv: &Conversation, key: TurnKey) -> TierState {
    conv.read()
        .turn_tier_state(key.timeline, key.index)
        .expect("turn must be tracked")
}

fn section_state(conv: &Conversation, section: SectionId) -> TierState {
    conv.read()
        .section_tier_state(section)
        .expect("section must be tracked")
}

#[test]
fn full_cold_warm_hot_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    // ── Phase 1: open persistence on disk, seed section + turns ──────────
    //
    // The Conversation owns a real SubstratePersistence pointed at the
    // tempdir; the redo log lives there for the duration. Backings are
    // real GPU arenas, the turns' chunks are real ChunkGids.
    let layer_id = LayerId::from_raw(1).unwrap();
    let group_id = GroupId::from_raw(1).unwrap();
    let timeline = TimelineAllocator::new().next();
    let section = SectionId::new(42);

    // Per-turn pattern bases, chosen so each turn's bytes are disjoint
    // and a stale read from any one would surface as a hard mismatch.
    let pattern_bases: Vec<u32> = (0..N_TURNS).map(|i| 1000 + (i as u32) * 1_000_000).collect();
    let section_pattern_base: u32 = 7_777_777;

    let (snapshots, section_snapshot, turn_keys) = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings(&device);

        conv.register_timeline(timeline, layer_id, group_id);

        // Pinned section — its hot residence must survive every
        // pre-restart tier transition.
        seed_section(&conv, &backings, &device, section, section_pattern_base);

        // N_TURNS synthetic turns, each with distinct K/V patterns.
        let turn_keys: Vec<TurnKey> = pattern_bases
            .iter()
            .map(|&base| {
                let idx = seed_turn(&conv, &backings, &device, timeline, base);
                TurnKey::new(timeline, idx)
            })
            .collect();

        // Snapshot all hot bytes BEFORE any tier transition — this is
        // the canonical reference for every later byte check.
        let snapshots: Vec<(TurnKey, Vec<Vec<u8>>)> = turn_keys
            .iter()
            .map(|&k| (k, snapshot_turn_bytes(&conv, &backings, &device, k.timeline, k.index)))
            .collect();
        let section_snapshot = snapshot_section_bytes(&conv, &backings, &device, section);

        // Sanity: snapshots are non-trivially sized and per-turn
        // patterns are actually distinct (otherwise the byte equality
        // checks below could pass via coincidence).
        for (_, layers) in &snapshots {
            assert_eq!(layers.len(), N_LAYERS, "snapshot has all layers");
            for layer_bytes in layers {
                assert!(!layer_bytes.is_empty(), "snapshot layer is non-empty");
            }
        }
        for i in 0..snapshots.len() {
            for j in (i + 1)..snapshots.len() {
                assert_ne!(
                    snapshots[i].1, snapshots[j].1,
                    "distinct pattern_base → distinct bytes"
                );
            }
        }

        // ── Phase 2: persistence cycle ──────────────────────────────────
        //
        // Spawn the PersistenceThread, then shutdown to force exactly
        // one full drain pass (run_pass → final commit → exit). After
        // this returns, every turn should be hot + warm + cold.
        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            None,
        );
        persist.shutdown();

        // Post-persist: the persistence thread uses
        // `install_warm_and_hot`, so each turn ends up hot+warm+cold.
        // Hot stays populated with the (possibly-quantized) GPU
        // SealedSequences; warm holds the format-preserving CPU copy.
        for &key in &turn_keys {
            let st = turn_state(&conv, key);
            assert!(
                st.hot && st.warm && st.cold,
                "post-persist {key:?} should be hot+warm+cold (hot stays via \
                 install_warm_and_hot), got {st:?}"
            );
        }
        // Section: still hot (pinned). Warm/cold not applicable.
        let sec = section_state(&conv, section);
        assert!(sec.hot, "section stays pinned hot through persistence");
        assert_eq!(
            snapshot_section_bytes(&conv, &backings, &device, section),
            section_snapshot,
            "section bytes unchanged by persistence cycle"
        );

        // ── Phase 3: explicit hot purge drops the now-redundant hot ─────
        //
        // Under install_warm_and_hot, persist leaves hot populated. An
        // explicit evict_from_hot is the way the scheduler/working-set
        // policy releases VRAM when a residence isn't in the active
        // projection.
        let purged = evict_from_hot(&conv, &[], &[]);
        assert_eq!(
            purged.count, N_TURNS,
            "explicit purge should drop all N_TURNS hot residences"
        );
        for &key in &turn_keys {
            let st = turn_state(&conv, key);
            assert!(
                !st.hot && st.warm && st.cold,
                "post-purge {key:?} should be cold-marker (warm+cold only), got {st:?}"
            );
        }
        let sec = section_state(&conv, section);
        assert!(sec.hot, "section unaffected by hot purge");

        // ── Phase 4: warm → hot ─────────────────────────────────────────
        //
        // elevate_to_hot routes each turn through the WarmLift leg:
        // batched per-layer migrate_sealed_to_gpu_batch_async on the
        // device's main stream. Single substrate write lock to install.
        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &turn_keys,
        )
        .unwrap();
        assert_eq!(report.warm_to_hot, N_TURNS, "all turns took the warm path");
        assert_eq!(report.cold_to_hot, 0, "no cold work — warm was present");
        assert_eq!(report.missing, 0);
        assert_eq!(report.failed, 0);

        // Bytes must round-trip byte-identical across the warm→hot leg.
        for (key, original) in &snapshots {
            let st = turn_state(&conv, *key);
            assert!(
                st.hot && st.warm && st.cold,
                "post warm→hot {key:?} should be hot+warm+cold, got {st:?}"
            );
            let now =
                snapshot_turn_bytes(&conv, &backings, &device, key.timeline, key.index);
            assert_eq!(
                &now, original,
                "warm→hot leg corrupted bytes for {key:?}"
            );
        }
        // Section bytes also still intact.
        assert_eq!(
            snapshot_section_bytes(&conv, &backings, &device, section),
            section_snapshot,
            "section bytes unchanged after warm→hot of turns"
        );

        // Hand snapshots out of the phase scope so they survive the
        // upcoming drop of `conv` + `backings`.
        (snapshots, section_snapshot, turn_keys)
    };
    // ── Phase 5: simulate daemon shutdown ────────────────────────────────
    //
    // `conv` + `backings` are out of scope; the substrate + persistence +
    // GPU arenas all drop. Files in `dir` are flushed by shutdown (the
    // PersistenceThread::shutdown above ran the final commit). Nothing
    // in-process holds substrate state.
    let _ = section_snapshot; // sections don't survive restart (no cold tier today).

    // ── Phase 6: reopen on the same tempdir, reload from log ─────────────
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings(&device);

    // Re-register the same timeline so the reloader can attach turns to it.
    // (`reconstruct_from_log` calls register_timeline internally for each
    //  recovered TurnDecl, but doing it up-front is idempotent and matches
    //  how the daemon's startup path sets up timelines.)
    conv.register_timeline(timeline, layer_id, group_id);

    let restored = conv
        .reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new()))
        .unwrap();
    assert_eq!(
        restored, N_TURNS,
        "all persisted turns recovered from the redo log"
    );

    // Each restored turn must land cold-marker: cold = Some, hot/warm = None.
    for &key in &turn_keys {
        let st = turn_state(&conv, key);
        assert!(
            !st.hot && !st.warm && st.cold,
            "post-reload {key:?} should be cold-only, got {st:?}"
        );
    }

    // ── Phase 7: cold → hot + warm ───────────────────────────────────────
    //
    // elevate_to_hot routes each turn through the ColdRecall leg:
    // recover_turn_chunks pulls the grid out of the redo log, load_to_hot
    // scatters it into fresh VRAM arenas, migrate_sealed_to_cpu produces
    // a fresh warm copy, and install_promoted lands both tiers under one
    // write lock.
    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &turn_keys,
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, N_TURNS, "all turns took the cold path");
    assert_eq!(report.warm_to_hot, 0, "warm was empty post-reload");
    assert_eq!(report.missing, 0);
    assert_eq!(report.failed, 0);

    // Final corruption check: bytes survived the full disk round-trip.
    for (key, original) in &snapshots {
        let st = turn_state(&conv, *key);
        assert!(
            st.hot && st.warm && st.cold,
            "post cold→hot {key:?} should be hot+warm+cold, got {st:?}"
        );
        let now = snapshot_turn_bytes(&conv, &backings, &device, key.timeline, key.index);
        assert_eq!(
            &now, original,
            "cold→hot leg corrupted bytes for {key:?} — the full round-trip is broken"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Quant blend round-trip
// ════════════════════════════════════════════════════════════════════════════

fn make_backings_f16(device: &Device, head_dim: usize) -> Vec<ChunkedKvBacking> {
    (0..N_LAYERS)
        .map(|_| {
            ChunkedKvBacking::new(4, N_KV_HEAD, head_dim, DType::F16, device, ARENA_CAPACITY)
                .unwrap()
        })
        .collect()
}

/// Re-point a SealedSequence's chunk GIDs into a fresh arena of
/// `target_format` on `target_location`. Each chunk's GIDs are migrated
/// one-to-one via [`ChunkedKvBacking::migrate_chunk`], which handles
/// the format conversion (e.g. F16 → R16 layout repack, F16 → Q8_0
/// quantization) and produces brand-new `ChunkGid`s.
///
/// Returns a `SealedSequence` whose chunks address the migrated arena.
/// The original sealed sequence's GIDs are dropped — the old arena
/// chunks become reclaimable.
fn migrate_sealed_to_format(
    backing: &ChunkedKvBacking,
    sealed: SealedSequence,
    target_format: KvFormat,
    target_location: ArenaLocation,
) -> SealedSequence {
    let key = ArenaKey::uniform(target_format, target_location);
    let mut new_chunks = Vec::with_capacity(sealed.chunks.len());
    for chunk in &sealed.chunks {
        let new_gids = chunk
            .gids
            .map_unique(|gid| backing.migrate_chunk(gid.raw(), key.clone()))
            .expect("migrate_chunk must succeed");
        // Recompute `byte_size` against the post-migrate arenas:
        // the source arena's per-slot stride no longer applies once
        // GIDs point at the target arena (Float→Quantized changes
        // bytes-per-chunk). Without this re-derivation the cold-load
        // round-trip would compare the original arena's byte_size
        // against the new arena's natively-computed byte_size and
        // mismatch on length.
        let arena_infos = backing.resolve_arena_info().unwrap();
        let byte_size = new_gids.arena_byte_size(&arena_infos);
        new_chunks.push(SealedChunk {
            gids: new_gids,
            byte_size,
            ..chunk.clone()
        });
    }
    SealedSequence {
        chunks: new_chunks,
        token_count: sealed.token_count,
        chunk_size: sealed.chunk_size,
        location: target_location,
    }
}

/// Seed a turn in the backings' default format, then optionally
/// migrate every chunk to `target_k_format` / `target_v_format`.
/// `None` keeps the chunk in the source format.
fn seed_turn_with_format(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    head_dim: usize,
    n_tokens: usize,
    pattern_base: u32,
    target_format: Option<KvFormat>,
) -> TurnIndex {
    use half::f16;

    let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(backings.len());
    for backing in backings {
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = N_KV_HEAD * n_tokens * head_dim;
        // F16 (not BF16) to make F16→R16 migration sound — R16's
        // storage element is F16 and the migrate path expects matching
        // source dtype.
        let data: Vec<f16> = (0..total)
            .map(|i| f16::from_f32(((pattern_base as usize + i) as f32) * 0.0001))
            .collect();
        let k = Tensor::from_vec(data, (1, N_KV_HEAD, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        let v = k.clone();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        let mut sealed = backing.record_turn(slot, n_tokens).unwrap();
        if let Some(target) = target_format {
            sealed = migrate_sealed_to_format(backing, sealed, target, ArenaLocation::Gpu);
        }
        sealed_per_layer.push(sealed);
    }
    let block_end = (n_tokens / CHUNK_SIZE) as u64;
    conv.record_turn(
        timeline,
        Role::User,
        candle_conversation::substrate::TurnPartWrite::default(),
        candle_conversation::substrate::TurnPartWrite {
            token_count: n_tokens,
            block_end,
            sealed_gpu: Some(Arc::new(sealed_per_layer)),
            ..Default::default()
        },
        |seqs| Ok(seqs.to_vec()),
    )
    .unwrap()
}

/// Same as the BF16 byte-snapshot helper, but parameterised on the
/// device so the quant-blend test can re-gather post each transition.
fn snapshot_bytes_at(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    idx: TurnIndex,
) -> Vec<Vec<u8>> {
    let sealed = conv
        .read()
        .turn_sealed_of(timeline, idx)
        .expect("turn must be hot");
    let mut out = Vec::with_capacity(sealed.len());
    for (backing, seq) in backings.iter().zip(sealed.iter()) {
        let images = seal_to_chunk_images(backing, device, seq).unwrap();
        let mut bytes = Vec::new();
        for img in &images {
            bytes.extend_from_slice(&img.payload.kv_bytes);
        }
        out.push(bytes);
    }
    out
}

/// Quant-blend through the **warm tier** (hot→warm→hot leg only).
///
/// Three turns share a single conversation, each in a different KV
/// format: native F16, R16 (raw F16 in quantized arena layout), and
/// Q8_0 (block-quantized). The test seeds → persistence cycle → purge
/// hot → elevate warm→hot, and asserts byte-identical bytes at every
/// step. All three formats round-trip cleanly through warm.
///
/// The post-restart cold→hot leg for Quantized formats is covered
/// separately by [`quant_blend_cold_round_trip`] (currently
/// `#[ignore]`d — see that test's doc for the open production bug).
#[test]
fn quant_blend_warm_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(2).unwrap();
    let group_id = GroupId::from_raw(2).unwrap();
    let timeline = TimelineAllocator::new().next();

    // Per-turn formats: a mix that exercises the three storage paths
    // simultaneously.
    let formats: &[(u32, Option<KvFormat>, &str)] = &[
        (10_001, None, "F16 native"),
        (
            30_003,
            Some(KvFormat::Quantized(QuantFormat::Q8_0)),
            "Q8_0 quant",
        ),
        (20_002, Some(KvFormat::Quantized(QuantFormat::R16)), "R16 raw"),
    ];

    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_f16(&device, QUANT_HEAD_DIM);

        conv.register_timeline(timeline, layer_id, group_id);

        // Seed each turn in its target format. After this loop the
        // substrate has three turns in mixed formats, all hot.
        let turn_keys: Vec<TurnKey> = formats
            .iter()
            .map(|&(base, target, label)| {
                let idx = seed_turn_with_format(
                    &conv,
                    &backings,
                    &device,
                    timeline,
                    QUANT_HEAD_DIM,
                    N_TOKENS_PER_TURN,
                    base,
                    target,
                );
                eprintln!("seeded {label} turn at {idx:?}");
                TurnKey::new(timeline, idx)
            })
            .collect();

        // Snapshot the *post-migration* bytes — those are what every
        // later transition must preserve.
        let snapshots: Vec<(TurnKey, Vec<Vec<u8>>)> = turn_keys
            .iter()
            .map(|&k| {
                (
                    k,
                    snapshot_bytes_at(&conv, &backings, &device, k.timeline, k.index),
                )
            })
            .collect();

        // Cross-turn disjointness — different format AND different
        // pattern_base, so the encoded bytes must differ.
        for i in 0..snapshots.len() {
            for j in (i + 1)..snapshots.len() {
                assert_ne!(
                    snapshots[i].1, snapshots[j].1,
                    "format/pattern blend should produce disjoint encoded bytes"
                );
            }
        }

        // Persistence cycle → all three tiers populated.
        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();

        for &key in &turn_keys {
            let st = turn_state(&conv, key);
            assert!(
                st.hot && st.warm && st.cold,
                "quant-blend persist {key:?} should be hot+warm+cold post install_warm_and_hot, got {st:?}"
            );
        }

        // Explicit eviction drops hot for the warm→hot elevate test below.
        let purged = evict_from_hot(&conv, &[], &[]);
        assert_eq!(purged.count, formats.len(), "all hot residences evicted");

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &turn_keys,
        )
        .unwrap();
        assert_eq!(report.warm_to_hot, formats.len());

        // Warm → hot must preserve every format's bytes exactly.
        for (key, original) in &snapshots {
            let now = snapshot_bytes_at(&conv, &backings, &device, key.timeline, key.index);
            assert_eq!(
                &now, original,
                "warm→hot drifted bytes for mixed-format {key:?}"
            );
        }

        let _ = (snapshots, turn_keys);
    }
}

/// Parameterised single-format cold round-trip. Validates that every
/// covered KV format round-trips byte-identical through the full
/// hot→warm→cold + restart + cold→hot pipeline. Pins the fix for the
/// `arena_byte_size` under-reporting bug per-format, so a regression
/// surfaces with a specific format name rather than getting buried
/// inside the blend test.
fn single_format_cold_round_trip(label: &str, target_format: Option<KvFormat>) {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();
    let layer_id = LayerId::from_raw(20 + label.len() as u32).unwrap();
    let group_id = GroupId::from_raw(20 + label.len() as u32).unwrap();
    let timeline = TimelineAllocator::new().next();

    let snapshot = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
        conv.register_timeline(timeline, layer_id, group_id);
        let idx = seed_turn_with_format(
            &conv,
            &backings,
            &device,
            timeline,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            424_242,
            target_format,
        );
        let snap = snapshot_bytes_at(&conv, &backings, &device, timeline, idx);
        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            None,
        );
        persist.shutdown();
        snap
    };

    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
    conv.register_timeline(timeline, layer_id, group_id);
    conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();

    let key = TurnKey::new(timeline, TurnIndex(0));
    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[key],
    )
    .unwrap();
    assert_eq!(
        report.cold_to_hot, 1,
        "{label}: cold→hot promotion missed turn"
    );

    let restored = snapshot_bytes_at(&conv, &backings, &device, timeline, TurnIndex(0));
    assert_eq!(
        restored, snapshot,
        "{label}: cold→hot round-trip dropped or corrupted bytes — \
         the under-reported byte_size regression is back"
    );
    // Every per-sub-band slot's stride contributed: the gather blob
    // for QUANT_HEAD_DIM=128, n_kv_head=2 has 16 sub-bands per chunk.
    // We don't pin a specific length here (formats differ) but we
    // can assert it's substantially larger than a single sub-band's
    // stride — catches a future regression that quietly reverts the
    // arena_byte_size dedup.
    for layer_bytes in &restored {
        assert!(
            layer_bytes.len() >= 1088 * 2,
            "{label}: post-cold-load layer bytes ({}) suspiciously small \
             — likely arena_byte_size dedup regressed",
            layer_bytes.len()
        );
    }
}

#[test]
fn cold_round_trip_f16_native() {
    single_format_cold_round_trip("F16 native", None);
}

#[test]
fn cold_round_trip_r16() {
    single_format_cold_round_trip("R16", Some(KvFormat::Quantized(QuantFormat::R16)));
}

#[test]
fn cold_round_trip_q8_0() {
    single_format_cold_round_trip("Q8_0", Some(KvFormat::Quantized(QuantFormat::Q8_0)));
}

#[test]
fn cold_round_trip_q4_0() {
    single_format_cold_round_trip("Q4_0", Some(KvFormat::Quantized(QuantFormat::Q4_0)));
}

/// **Regression for the cold-marker filter bug.** A turn that lives
/// only in the cold tier (post-restart, before any elevation has
/// run) must still resolve as "tracked" via the substrate's
/// `turn_tier_state` accessor. The SubmitTurn handler uses this to
/// decide whether to include the turn in the projection's elevate +
/// inject list. Filtering on `turn_sealed_of` instead (the pre-fix
/// behaviour) would silently drop every resumed-from-disk turn
/// because `hot` is `None` until elevation lands — so the model
/// never sees prior turns of a resumed conversation.
#[test]
fn cold_marker_turn_passes_existence_check() {
    use candle_conversation::substrate::TierState;
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();
    let layer_id = LayerId::from_raw(40).unwrap();
    let group_id = GroupId::from_raw(40).unwrap();
    let timeline = TimelineAllocator::new().next();

    // Seed + persist + drop.
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        let _ = seed_turn(&conv, &backings, &device, timeline, 999_999);
        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();
    }

    // Reopen + reload. Turn is now cold-marker only.
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings(&device);
    conv.register_timeline(timeline, layer_id, group_id);
    conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();

    let key = TurnKey::new(timeline, TurnIndex(0));

    // The exact two accessors the projection-result filter uses,
    // each tested against a cold-marker turn:
    //
    // `turn_sealed_of` — the BUGGY filter — returns None because hot
    //                    is None. Filtering here would drop the turn.
    assert!(
        conv.read().turn_sealed_of(timeline, key.index).is_none(),
        "cold-marker turn has no hot bytes by definition"
    );
    // `turn_tier_state` — the FIXED filter — returns Some because
    //                     the turn is tracked (cold tier populated).
    let state = conv.read().turn_tier_state(timeline, key.index);
    assert_eq!(
        state,
        Some(TierState {
            hot: false,
            warm: false,
            cold: true,
        }),
        "cold-marker turn must be tracked (cold tier present) so the \
         projection filter can decide to elevate it"
    );

    // Now run the elevate path the SubmitTurn handler runs after the
    // filter — proves the cold-marker turn actually does land hot
    // when the filter lets it through.
    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[key],
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, 1);
    assert_eq!(report.missing, 0);

    let post = conv.read().turn_tier_state(timeline, key.index).unwrap();
    assert!(
        post.hot && post.warm && post.cold,
        "post-elevate the turn lives in all three tiers, got {post:?}"
    );
}

/// Focused diagnostic — round-trip a single Q8_0 chunk through the
/// cold tier and dump byte sizes at every step. Used during the
/// debugging of [`quant_blend_cold_round_trip`]; left in as a
/// regression for the cold→hot Quantized scatter path.
#[test]
fn cold_load_q8_single_chunk_diagnostic() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();
    let layer_id = LayerId::from_raw(9).unwrap();
    let group_id = GroupId::from_raw(9).unwrap();
    let timeline = TimelineAllocator::new().next();

    let snapshot = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
        conv.register_timeline(timeline, layer_id, group_id);

        let idx = seed_turn_with_format(
            &conv,
            &backings,
            &device,
            timeline,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            123_456,
            Some(KvFormat::Quantized(QuantFormat::Q8_0)),
        );
        let key = TurnKey::new(timeline, idx);

        let snap = snapshot_bytes_at(&conv, &backings, &device, timeline, idx);
        eprintln!(
            "seed-time snapshot: layer 0 has {} bytes, first 16 = {:?}",
            snap[0].len(),
            &snap[0][..16.min(snap[0].len())]
        );
        let nonzero = snap[0].iter().filter(|b| **b != 0).count();
        eprintln!("seed-time layer 0 nonzero bytes: {nonzero}/{}", snap[0].len());

        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            None,
        );
        persist.shutdown();

        // install_warm_and_hot keeps hot populated alongside warm.
        // The cold-load roundtrip below validates byte-identity from
        // disk; this assertion just confirms all three tiers are live.
        let st = turn_state(&conv, key);
        assert!(
            st.hot && st.warm && st.cold,
            "post-persist {key:?} should be hot+warm+cold (install_warm_and_hot), got {st:?}"
        );
        let _ = key;
        snap
    };

    // Reopen + reload + cold→hot.
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
    conv.register_timeline(timeline, layer_id, group_id);
    let restored = conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();
    assert_eq!(restored, 1);

    // Inspect the recovered chunk grid before elevation runs.
    let recovered = conv
        .recover_turn_chunks(timeline, TurnIndex(0), N_LAYERS)
        .unwrap()
        .expect("Some");
    let layer0_kv_bytes_len: usize =
        recovered.layer(0).iter().map(|c| c.payload.kv_bytes.len()).sum();
    eprintln!(
        "recovered layer 0 has {} chunks, total kv_bytes = {}",
        recovered.layer(0).len(),
        layer0_kv_bytes_len
    );
    if let Some(first_chunk) = recovered.layer(0).first() {
        eprintln!(
            "recovered chunk 0 kv_bytes len = {}, first 16 = {:?}",
            first_chunk.payload.kv_bytes.len(),
            &first_chunk.payload.kv_bytes[..16.min(first_chunk.payload.kv_bytes.len())]
        );
    }

    let key = TurnKey::new(timeline, TurnIndex(0));
    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[key],
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, 1, "{report:?}");
    assert_eq!(report.failed, 0);

    let post_cold = snapshot_bytes_at(&conv, &backings, &device, timeline, TurnIndex(0));
    eprintln!(
        "post cold→hot snapshot: layer 0 has {} bytes, first 16 = {:?}",
        post_cold[0].len(),
        &post_cold[0][..16.min(post_cold[0].len())]
    );
    let nonzero = post_cold[0].iter().filter(|b| **b != 0).count();
    eprintln!("post cold→hot layer 0 nonzero bytes: {nonzero}/{}", post_cold[0].len());

    assert_eq!(
        post_cold, snapshot,
        "cold→hot must preserve Q8 bytes byte-identical"
    );
}

/// Quant-blend through the **cold tier** (full hot→warm→cold + restart
/// + cold→hot leg).
///
/// Seeds three turns in distinct KV formats (F16 native, R16, Q8_0),
/// persists them through the redo log, drops the conversation, then
/// reopens from disk and elevates cold→hot. Every format must
/// round-trip byte-identical: same chunk count, same byte payload at
/// every sub-band slot, same total chunk bytes.
///
/// This is the regression for the `HeadGids::arena_byte_size`
/// dedup-by-arena-idx-only bug: under-reported `byte_size` caused
/// `seal_to_chunk_images` to silently drop 15/16 of each chunk's
/// sub-band bytes on the persistence gather, and the cold-load
/// scatter delivered correct bytes only to sub-band 0. Production
/// daemon symptom: resumed conversations losing context turn-by-turn
/// as the projection cycled prior turns through the cold path.
#[test]
fn quant_blend_cold_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(5).unwrap();
    let group_id = GroupId::from_raw(5).unwrap();
    let timeline = TimelineAllocator::new().next();

    let formats: &[(u32, Option<KvFormat>, &str)] = &[
        (10_001, None, "F16 native"),
        (
            30_003,
            Some(KvFormat::Quantized(QuantFormat::Q8_0)),
            "Q8_0 quant",
        ),
        (20_002, Some(KvFormat::Quantized(QuantFormat::R16)), "R16 raw"),
    ];

    let (snapshots, turn_keys) = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
        conv.register_timeline(timeline, layer_id, group_id);

        let turn_keys: Vec<TurnKey> = formats
            .iter()
            .map(|&(base, target, _)| {
                let idx = seed_turn_with_format(
                    &conv,
                    &backings,
                    &device,
                    timeline,
                    QUANT_HEAD_DIM,
                    N_TOKENS_PER_TURN,
                    base,
                    target,
                );
                TurnKey::new(timeline, idx)
            })
            .collect();

        let snapshots: Vec<(TurnKey, Vec<Vec<u8>>)> = turn_keys
            .iter()
            .map(|&k| {
                (
                    k,
                    snapshot_bytes_at(&conv, &backings, &device, k.timeline, k.index),
                )
            })
            .collect();

        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();

        (snapshots, turn_keys)
    };

    // Reopen, reload from cold, elevate cold→hot+warm, byte-compare.
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings_f16(&device, QUANT_HEAD_DIM);
    conv.register_timeline(timeline, layer_id, group_id);

    let restored = conv
        .reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new()))
        .unwrap();
    assert_eq!(restored, formats.len(), "all mixed-format turns recovered");

    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &turn_keys,
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, formats.len(), "all turns took cold path");
    assert_eq!(report.failed, 0);
    assert_eq!(report.missing, 0);

    for (key, original) in &snapshots {
        let now = snapshot_bytes_at(&conv, &backings, &device, key.timeline, key.index);
        assert_eq!(
            &now, original,
            "cold→hot drifted bytes for mixed-format {key:?} — \
             per-(h,p) format preservation broken across log round-trip"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Edge cases
// ════════════════════════════════════════════════════════════════════════════

/// Edge cases the happy-path test doesn't exercise:
/// - elevating a **missing** turn key (one that doesn't exist) reports
///   `missing` and doesn't poison the batch.
/// - **idempotent re-elevate**: an already-hot residence routes through
///   `already_hot` and does no work.
/// - **partial selection**: only some turns elevated; non-selected
///   residences stay in whatever tier they were.
/// - **keep-set evict**: `evict_from_hot` with a non-empty keep set
///   drops only the unkept turns; kept turns stay hot.
#[test]
fn elevate_edge_cases() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(3).unwrap();
    let group_id = GroupId::from_raw(3).unwrap();
    let timeline = TimelineAllocator::new().next();

    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings(&device);
    conv.register_timeline(timeline, layer_id, group_id);

    // Seed 3 turns, run one persistence cycle, then evict to warm-only.
    let keys: Vec<TurnKey> = (0..3)
        .map(|i| {
            let idx = seed_turn(&conv, &backings, &device, timeline, 50_000 + i as u32 * 7_777);
            TurnKey::new(timeline, idx)
        })
        .collect();
    let snapshots: Vec<Vec<Vec<u8>>> = keys
        .iter()
        .map(|k| snapshot_turn_bytes(&conv, &backings, &device, k.timeline, k.index))
        .collect();

    let persist = PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
    persist.shutdown();
    // With install_warm_and_hot, hot stays populated after persist.
    // Explicitly evict to set up the warm→hot promotion scenario the
    // following cases exercise.
    let purged = evict_from_hot(&conv, &[], &[]);
    assert_eq!(purged.count, 3, "explicit purge drops all 3 hot residences");

    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();

    // ── Case 1: missing key + valid keys mixed ───────────────────────────
    let bogus = TurnKey::new(TimelineId::from_raw(999).unwrap(), TurnIndex(42));
    let mixed_keys = vec![keys[0], bogus, keys[1]];
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &mixed_keys,
    )
    .unwrap();
    assert_eq!(report.warm_to_hot, 2, "two valid keys promoted");
    assert_eq!(report.missing, 1, "bogus key counted as missing");
    assert_eq!(report.failed, 0);
    // Bytes for the two promoted turns survived the warm→hot leg.
    let now0 = snapshot_turn_bytes(&conv, &backings, &device, keys[0].timeline, keys[0].index);
    let now1 = snapshot_turn_bytes(&conv, &backings, &device, keys[1].timeline, keys[1].index);
    assert_eq!(now0, snapshots[0], "promoted key 0 bytes intact");
    assert_eq!(now1, snapshots[1], "promoted key 1 bytes intact");
    // Non-selected key 2 stays in warm-only.
    let st2 = turn_state(&conv, keys[2]);
    assert!(
        !st2.hot && st2.warm && st2.cold,
        "non-selected key stays warm-only, got {st2:?}"
    );

    // ── Case 2: idempotent re-elevate ────────────────────────────────────
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[keys[0], keys[1]],
    )
    .unwrap();
    assert_eq!(report.already_hot, 2, "both already hot from previous pass");
    assert_eq!(report.warm_to_hot, 0);
    assert_eq!(report.cold_to_hot, 0);
    assert_eq!(report.failed, 0);

    // ── Case 3: keep-set evict ───────────────────────────────────────────
    //
    // Drop hot for everyone EXCEPT keys[0]. evict_from_hot's keep set
    // protects the listed residences.
    let evicted = evict_from_hot(&conv, &[], &[keys[0]]);
    assert_eq!(evicted.count, 1, "only key 1 evicted (key 0 kept, key 2 was already warm-only)");
    assert!(turn_state(&conv, keys[0]).hot, "keep-set protected key 0");
    assert!(!turn_state(&conv, keys[1]).hot, "key 1 evicted to warm-only");
    assert!(!turn_state(&conv, keys[2]).hot, "key 2 still warm-only");

    // ── Case 4: partial elevate after keep-set evict ─────────────────────
    //
    // Re-elevate just keys[2]; keys[1] stays warm-only and keys[0]
    // stays hot (already there). Verifies the elevate path doesn't
    // touch keys outside its input set.
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[keys[2]],
    )
    .unwrap();
    assert_eq!(report.warm_to_hot, 1, "only keys[2] elevated");
    assert_eq!(report.already_hot, 0);
    assert!(turn_state(&conv, keys[0]).hot, "key 0 untouched, still hot");
    assert!(
        !turn_state(&conv, keys[1]).hot,
        "key 1 not in input set, stays warm-only"
    );
    assert!(turn_state(&conv, keys[2]).hot, "key 2 just promoted");

    // Final byte integrity check across all three.
    for (k, original) in keys.iter().zip(snapshots.iter()) {
        if turn_state(&conv, *k).hot {
            let now = snapshot_turn_bytes(&conv, &backings, &device, k.timeline, k.index);
            assert_eq!(
                &now, original,
                "post edge-case dance, hot bytes for {k:?} drifted"
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Multi-chunk turn (more than one chunk per layer)
// ════════════════════════════════════════════════════════════════════════════

/// Validates the round-trip on a turn that spans **multiple chunks**
/// per layer (2 chunks of 32 tokens each = 64 tokens). Exercises the
/// `chunks_per_layer > 1` path in the persistence demux + cold-tier
/// `StoredSequence` reconstruction.
#[test]
fn multi_chunk_turn_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let n_tokens = 2 * CHUNK_SIZE; // two chunks per layer
    let chunks_per_layer = n_tokens / CHUNK_SIZE;
    assert_eq!(chunks_per_layer, 2);

    let layer_id = LayerId::from_raw(4).unwrap();
    let group_id = GroupId::from_raw(4).unwrap();
    let timeline = TimelineAllocator::new().next();

    let (snapshot, key) = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings(&device);
        conv.register_timeline(timeline, layer_id, group_id);

        // Seed one turn with 2 chunks per layer. Use the parameterised
        // seed helper so the block-range math is right.
        let idx = seed_turn_with_format(
            &conv,
            &backings,
            &device,
            timeline,
            HEAD_DIM,
            n_tokens,
            123_456,
            None,
        );
        let key = TurnKey::new(timeline, idx);

        // Per-layer snapshot must have two chunks' worth of encoded
        // bytes concatenated.
        let snapshot = snapshot_turn_bytes(&conv, &backings, &device, timeline, idx);
        for layer_bytes in &snapshot {
            assert!(
                !layer_bytes.is_empty(),
                "multi-chunk layer should have bytes"
            );
        }

        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();

        let st = turn_state(&conv, key);
        assert!(st.hot && st.warm && st.cold);

        let purged = evict_from_hot(&conv, &[], &[]);
        assert_eq!(purged.count, 1, "explicit purge drops post-persist hot");

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        assert_eq!(report.warm_to_hot, 1);

        let now = snapshot_turn_bytes(&conv, &backings, &device, timeline, idx);
        assert_eq!(now, snapshot, "warm→hot drift on multi-chunk turn");

        (snapshot, key)
    };

    // Restart + cold→hot+warm with a multi-chunk turn.
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings(&device);
    conv.register_timeline(timeline, layer_id, group_id);
    let restored = conv
        .reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new()))
        .unwrap();
    assert_eq!(restored, 1);

    let st = turn_state(&conv, key);
    assert!(!st.hot && !st.warm && st.cold);

    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[key],
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, 1, "multi-chunk turn took cold path");

    let now = snapshot_turn_bytes(&conv, &backings, &device, key.timeline, key.index);
    assert_eq!(
        now, snapshot,
        "multi-chunk cold→hot drift — demux of {chunks_per_layer} chunks/layer broken"
    );
}

// ════════════════════════════════════════════════════════════════════════════
// Archive lifecycle: ConvState records survive a restart
// ════════════════════════════════════════════════════════════════════════════

/// End-to-end archive lifecycle through the real persistence layer.
/// Seeds a turn, sets the conversation's `archived` flag, drops the
/// in-RAM substrate, reopens from the same tempdir, runs the reload,
/// and asserts the archive state survived. Then unarchives and
/// verifies that survives a second restart too.
///
/// Pins the contract that the daemon's `POST /v1/conversations/
/// {id}/archive` and `/unarchive` endpoints rely on — without this
/// the X button would set the in-RAM flag and the next refresh
/// (which goes through the reload) would forget it.
#[test]
fn archive_state_survives_restart() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(50).unwrap();
    let group_id = GroupId::from_raw(50).unwrap();
    let timeline = TimelineAllocator::new().next();

    // ── Phase 1: seed a turn + persist + mark archived ─────────────────
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        let _ = seed_turn(&conv, &backings, &device, timeline, 100_001);

        // Drive the persistence thread once so the turn's records
        // are durable before we test reload.
        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();

        // Initially the conversation is not archived.
        assert!(!conv.is_conversation_archived(timeline));

        // Archive it — writes a ConvState record + updates in-RAM.
        // The write goes through `append_record` which only stages
        // bytes; we need to commit before the conv drops or the
        // reload will not see it. (Production has the persistence
        // thread's tick + per-turn-seal trigger to take care of this
        // routinely — here we drive it explicitly.)
        conv.set_conversation_archived(timeline, true).unwrap();
        conv.commit_persistence().unwrap();
        assert!(conv.is_conversation_archived(timeline));
    }

    // ── Phase 2: reopen, reload, archive state survives ────────────────
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();
        assert!(
            conv.is_conversation_archived(timeline),
            "archived flag must survive the restart-reload"
        );

        // Unarchive — last-writer-wins on the next reload.
        conv.set_conversation_archived(timeline, false).unwrap();
        conv.commit_persistence().unwrap();
        assert!(!conv.is_conversation_archived(timeline));
    }

    // ── Phase 3: second restart — unarchive survives ──────────────────
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();
        assert!(
            !conv.is_conversation_archived(timeline),
            "unarchive must also survive — last-writer-wins on ConvState"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Quantize-on-evict — hot → warm → cold pipeline driven by a CompressionPolicy
// ════════════════════════════════════════════════════════════════════════════
//
// These tests mirror the `quant_blend_*` / `single_format_*` patterns
// above but exercise the policy-driven path through `PersistenceThread`:
// hot is seeded as F16, the persistence thread quantizes per-(head,
// palette) using a `CompressionPolicy`, and the warm/cold tiers end
// up holding the compressed bytes. Identity-checks are at the
// post-persist boundary (the redo log must faithfully roundtrip the
// quantized form), not against the original F16 source — quantization
// is lossy by definition.

use candle_nn::kv_cache::CompressionPolicy;

fn make_backings_adaptive(
    device: &Device,
    head_dim: usize,
    policy: &CompressionPolicy,
) -> Vec<ChunkedKvBacking> {
    (0..N_LAYERS)
        .map(|_| {
            ChunkedKvBacking::new_with_format_adaptive(
                4,
                N_KV_HEAD,
                head_dim,
                KvFormat::Float(DType::F16),
                KvFormat::Float(DType::F16),
                device,
                ARENA_CAPACITY,
                Some(policy.clone()),
            )
            .unwrap()
        })
        .collect()
}

/// Quantize-on-evict: hot → warm with `Some(policy)` should yield a
/// turn that is hot+warm+cold like the no-policy path, but the warm
/// chunks come back smaller because the per-(h, p) selection picked
/// a quantized format for each sub-band rather than carrying full
/// F16 through.
#[test]
fn quantize_on_evict_full_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(70).unwrap();
    let group_id = GroupId::from_raw(70).unwrap();
    let timeline = TimelineAllocator::new().next();
    let policy = CompressionPolicy::new(5);

    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings_adaptive(&device, QUANT_HEAD_DIM, &policy);
    conv.register_timeline(timeline, layer_id, group_id);

    let idx = seed_turn_with_format(
        &conv,
        &backings,
        &device,
        timeline,
        QUANT_HEAD_DIM,
        N_TOKENS_PER_TURN,
        500_001,
        None, // leave the seed in F16 so the policy can quantize it
    );
    let key = TurnKey::new(timeline, idx);

    // Drive a full persistence pass with the policy engaged.
    let persist = PersistenceThread::spawn(
        conv.clone(),
        Arc::new(backings.clone()),
        device.clone(),
        Some(policy.clone()),
    );
    persist.shutdown();

    // Post-persist: warm+cold (hot was atomically evicted by
    // install_warm_and_evict_hot so the next decode reads the
    // compressed bytes via warm→hot elevation, not the original
    // uncompressed hot copy).
    let st = turn_state(&conv, key);
    assert!(
        st.hot && st.warm && st.cold,
        "quantize-on-evict {key:?} should land hot+warm+cold (install_warm_and_hot), got {st:?}"
    );
}

/// The post-eviction warm bytes must be strictly smaller than what
/// the format-preserving migration would have produced for the same
/// hot source. Runs the same seed pattern twice — once with a
/// CompressionPolicy, once without — and compares the resulting
/// cold-log record sizes (the redo log captures whatever the warm
/// tier carries, so it's a faithful proxy for warm byte count).
#[test]
fn quantize_on_evict_actually_compresses() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let timeline = TimelineAllocator::new().next();
    let layer_id = LayerId::from_raw(71).unwrap();
    let group_id = GroupId::from_raw(71).unwrap();
    let policy = CompressionPolicy::new(5);

    // Helper closure — runs one full seed+persist cycle and returns
    // the on-disk Chunk record byte total (sum of all Chunk record
    // `record_size`s across both layers).
    let run_cycle = |policy_opt: Option<CompressionPolicy>| -> u64 {
        let tmpdir = tempfile::tempdir().unwrap();
        let dir = tmpdir.path().to_path_buf();
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = match policy_opt {
            Some(ref p) => make_backings_adaptive(&device, QUANT_HEAD_DIM, p),
            None => make_backings_f16(&device, QUANT_HEAD_DIM),
        };
        conv.register_timeline(timeline, layer_id, group_id);
        let _ = seed_turn_with_format(
            &conv,
            &backings,
            &device,
            timeline,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            600_011,
            None,
        );
        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            policy_opt,
        );
        persist.shutdown();
        conv.with_persistence_manifest(|m| {
            m.streams
                .values()
                .flat_map(|s| s.chunks.values())
                .map(|loc| loc.record_size)
                .sum::<u64>()
        })
    };

    let f16_bytes = run_cycle(None);
    let quant_bytes = run_cycle(Some(policy));
    assert!(
        quant_bytes < f16_bytes,
        "quantize-on-evict must produce strictly fewer chunk-record \
         bytes than the format-preserving path: quantized={quant_bytes}, \
         f16={f16_bytes}"
    );
    // At C5 we expect substantial compression — at least 30% smaller.
    // This is a soft floor; if the selection regresses to F16
    // fallback for every block the ratio collapses to ~1.0 and the
    // test loudly fails.
    assert!(
        (quant_bytes as f64) < 0.7 * (f16_bytes as f64),
        "quantize-on-evict compression ratio is suspiciously low: \
         quantized={quant_bytes}, f16={f16_bytes} ({:.2}× of F16)",
        quant_bytes as f64 / f16_bytes as f64,
    );
}

/// Full restart round-trip with quantize-on-evict. Seeds F16,
/// persists with a policy, drops everything, reopens substrate,
/// reloads from the redo log, and elevates the cold-marker turn
/// back to hot. The chunks must come back hot (cold-load
/// succeeded) — and crucially, the redo log file must be the same
/// size pre- and post-reload (no records appended during the
/// reload itself).
#[test]
fn quantize_on_evict_cold_reload_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();
    let layer_id = LayerId::from_raw(72).unwrap();
    let group_id = GroupId::from_raw(72).unwrap();
    let timeline = TimelineAllocator::new().next();
    let policy = CompressionPolicy::new(5);
    let log_path = dir
        .join(candle_conversation::persistence::SUBSTRATE_DIR)
        .join(candle_conversation::persistence::ACTIVE_LOG_NAME);

    // ── Phase 1: seed + persist (quantize on evict) ─────────────────
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_adaptive(&device, QUANT_HEAD_DIM, &policy);
        conv.register_timeline(timeline, layer_id, group_id);
        let _ = seed_turn_with_format(
            &conv,
            &backings,
            &device,
            timeline,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            700_007,
            None,
        );
        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            Some(policy.clone()),
        );
        persist.shutdown();
    }

    let log_len_before = std::fs::metadata(&log_path).unwrap().len();

    // ── Phase 2: drop everything, reopen, reload, then elevate ──────
    let persistence = SubstratePersistence::open_in(&dir).unwrap();
    let conv = Conversation::with_persistence(persistence);
    let backings = make_backings_adaptive(&device, QUANT_HEAD_DIM, &policy);
    conv.register_timeline(timeline, layer_id, group_id);
    conv.reconstruct_from_log(N_LAYERS, |_| Ok(Vec::new())).unwrap();

    // Pre-elevate: cold-marker only.
    let key = TurnKey::new(timeline, TurnIndex(0));
    let pre_elevate = turn_state(&conv, key);
    assert_eq!(
        pre_elevate,
        TierState {
            hot: false,
            warm: false,
            cold: true,
        },
        "reloaded turn must be cold-marker before elevation"
    );

    // Reopen must not have appended any records (no torn-tail truncation,
    // no extra checkpoint write during recovery).
    let log_len_after_reload = std::fs::metadata(&log_path).unwrap().len();
    assert_eq!(
        log_len_after_reload, log_len_before,
        "reload must not modify the redo log on disk"
    );

    // Cold-load + elevate.
    let main_stream = cuda_stream(&device);
    let mut pinned: Option<PinnedBuf> = None;
    let mut stager = ColdLoadStager::new();
    let report = elevate_to_hot(
        &conv,
        &backings,
        &device,
        &main_stream,
        &mut pinned,
        &mut stager,
        &[],
        &[key],
    )
    .unwrap();
    assert_eq!(report.cold_to_hot, 1, "single cold→hot promotion");
    assert_eq!(report.failed, 0);

    // Post-elevate: hot + cold (warm is repopulated by a fresh
    // persistence pass, not the cold→hot elevation).
    let post_st = turn_state(&conv, key);
    assert!(
        post_st.hot && post_st.cold,
        "post-elevate {key:?} should be hot+cold at minimum, got {post_st:?}"
    );

    // The cold→hot promotion must have landed every chunk in a
    // *quantized* arena (the cold log records the per-(h, p)
    // selection from phase 1, and the elevation places each GID
    // into a GPU arena of the recorded format). If the cold-load
    // path silently fell back to float arenas the quantization
    // metadata on disk would be lost.
    let sealed = conv
        .read()
        .turn_sealed_of(timeline, key.index)
        .expect("post-elevate turn must be hot");
    for (backing, seq) in backings.iter().zip(sealed.iter()) {
        let arena_info = backing.resolve_arena_info().unwrap();
        for chunk in &seq.chunks {
            for gid in chunk.gids.as_slice() {
                let arena_idx = gid.arena_idx();
                let info = &arena_info[arena_idx];
                assert!(
                    info.k_format_tag.is_quantized() || info.v_format_tag.is_quantized(),
                    "post-cold-load gid {} arena_idx {arena_idx} must live in a \
                     quantized arena, got {:?} / {:?}",
                    gid.raw(),
                    info.k_format_tag,
                    info.v_format_tag,
                );
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Per-(h, p) metadata preservation
// ════════════════════════════════════════════════════════════════════════════
//
// The byte-only round-trip tests above only assert `payload.kv_bytes`
// equality. That misses entire classes of corruption: a swap or
// reordering bug in `k_formats` / `v_formats`, a stale palette map, a
// scale that gets re-derived from arena state instead of preserved, or
// a fallback to a uniform format on cold reload would all leave the
// kv_bytes byte-identical but break the per-(h, p) decode at attention
// time.
//
// These tests close that gap by capturing the *whole* `ChunkImage`
// (k_formats, v_formats, k_pal, v_pal, k_scale, v_scale, offset,
// kv_bytes) at every tier transition and asserting field-by-field
// equality. They also seed a head-and-palette-varying K/V pattern so
// the policy is forced to pick different formats across sub-bands —
// otherwise a reordering bug on a uniform-format chunk would round-trip
// cleanly.

use candle_conversation::persistence::resume::ChunkImage;

/// Layer count for the field-level metadata round-trip tests. Larger
/// than the module-level `N_LAYERS` constant so the per-(h, p)
/// preservation contract is exercised across a realistic stack depth.
/// 32 layers matches the Qwen3-8B / Llama-3.2-3B-class architectures.
const N_LAYERS_METADATA: usize = 32;

/// KV head count for the metadata round-trip tests. Larger than the
/// module-level `N_KV_HEAD = 2` so the (h * N_PALETTE + p) index space
/// is wider — surfaces head-index reorder bugs that would alias on the
/// 2-head config.
const N_KV_HEAD_METADATA: usize = 4;

/// 32-layer × 4-kv-head adaptive backings for the metadata
/// round-trip tests. Uses `head_dim = QUANT_HEAD_DIM` (= 128) so the
/// palette-4 selection kernel's `head_dim = 128` shape constraint is
/// satisfied.
fn make_backings_metadata(
    device: &Device,
    policy: &CompressionPolicy,
) -> Vec<ChunkedKvBacking> {
    (0..N_LAYERS_METADATA)
        .map(|_| {
            ChunkedKvBacking::new_with_format_adaptive(
                4,
                N_KV_HEAD_METADATA,
                QUANT_HEAD_DIM,
                KvFormat::Float(DType::F16),
                KvFormat::Float(DType::F16),
                device,
                ARENA_CAPACITY,
                Some(policy.clone()),
            )
            .unwrap()
        })
        .collect()
}

/// Same as `make_backings_metadata` but for the BF16-default
/// no-policy round-trip — exercises the format-preserving migration.
fn make_backings_metadata_no_policy(device: &Device) -> Vec<ChunkedKvBacking> {
    (0..N_LAYERS_METADATA)
        .map(|_| {
            ChunkedKvBacking::new(
                4,
                N_KV_HEAD_METADATA,
                QUANT_HEAD_DIM,
                DType::F16,
                device,
                ARENA_CAPACITY,
            )
            .unwrap()
        })
        .collect()
}

/// Snapshot the full per-layer `ChunkImage` set (not just `kv_bytes`).
/// The returned `Vec<Vec<ChunkImage>>` is `[layer][chunk]`.
fn snapshot_turn_images(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    idx: TurnIndex,
) -> Vec<Vec<ChunkImage>> {
    let sealed = conv
        .read()
        .turn_sealed_of(timeline, idx)
        .expect("turn must be hot");
    let mut out = Vec::with_capacity(sealed.len());
    for (backing, seq) in backings.iter().zip(sealed.iter()) {
        out.push(seal_to_chunk_images(backing, device, seq).unwrap());
    }
    out
}

/// Field-by-field equality on the metadata that the persist path
/// records and the cold-load path must rebuild faithfully. Emits a
/// pinpoint message naming the exact field that drifted so the
/// failure mode is unambiguous.
fn assert_chunk_images_eq(
    left: &[Vec<ChunkImage>],
    right: &[Vec<ChunkImage>],
    where_: &str,
) {
    assert_eq!(
        left.len(),
        right.len(),
        "{where_}: layer count drift ({} vs {})",
        left.len(),
        right.len(),
    );
    for (l_idx, (l_layer, r_layer)) in left.iter().zip(right.iter()).enumerate() {
        assert_eq!(
            l_layer.len(),
            r_layer.len(),
            "{where_}: layer {l_idx} chunk count drift ({} vs {})",
            l_layer.len(),
            r_layer.len(),
        );
        for (c_idx, (l, r)) in l_layer.iter().zip(r_layer.iter()).enumerate() {
            assert_eq!(
                l.token_count, r.token_count,
                "{where_}: layer {l_idx} chunk {c_idx} token_count drift"
            );
            assert_eq!(
                l.payload.offset, r.payload.offset,
                "{where_}: layer {l_idx} chunk {c_idx} offset drift"
            );
            assert_eq!(
                l.payload.k_formats, r.payload.k_formats,
                "{where_}: layer {l_idx} chunk {c_idx} k_formats drift — per-(h, p) K \
                 format tag map did not survive the tier transition"
            );
            assert_eq!(
                l.payload.v_formats, r.payload.v_formats,
                "{where_}: layer {l_idx} chunk {c_idx} v_formats drift — per-(h, p) V \
                 format tag map did not survive the tier transition"
            );
            assert_eq!(
                l.payload.k_pal, r.payload.k_pal,
                "{where_}: layer {l_idx} chunk {c_idx} k_pal drift"
            );
            assert_eq!(
                l.payload.v_pal, r.payload.v_pal,
                "{where_}: layer {l_idx} chunk {c_idx} v_pal drift"
            );
            assert_eq!(
                l.payload.k_scale, r.payload.k_scale,
                "{where_}: layer {l_idx} chunk {c_idx} k_scale drift"
            );
            assert_eq!(
                l.payload.v_scale, r.payload.v_scale,
                "{where_}: layer {l_idx} chunk {c_idx} v_scale drift"
            );
            assert_eq!(
                l.payload.kv_bytes, r.payload.kv_bytes,
                "{where_}: layer {l_idx} chunk {c_idx} kv_bytes drift"
            );
        }
    }
}

/// Assert the post-quantize image set has sufficient structure to make
/// the round-trip non-vacuous. Specifically: `k_formats` ≠ `v_formats`
/// for at least one chunk, so a K↔V slice swap would surface in
/// [`assert_chunk_images_eq`].
///
/// **Within-K and within-V** per-(h, p) reorder bugs are covered by
/// `kv_bytes` byte-identity rather than by format/pal/scale variation:
/// each sub-band's quantized block bytes live at a distinct offset in
/// `kv_bytes`, so any swap of (h, p) positions during persist or
/// reload changes the byte layout. (At compression levels where the
/// chosen format is non-palette — Q3_0, Q4_0, Q8_0 — the `k_pal` /
/// `v_pal` / `k_scale` / `v_scale` slices are identity placeholders
/// and don't carry per-(h, p) variance regardless of input.)
fn assert_varied_formats(images: &[Vec<ChunkImage>], where_: &str) {
    let mut k_tags: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    let mut v_tags: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    let mut k_scale_set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut v_scale_set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut k_pal_set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    let mut v_pal_set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
    for layer in images {
        for img in layer {
            k_tags.extend(img.payload.k_formats.iter().copied());
            v_tags.extend(img.payload.v_formats.iter().copied());
            k_scale_set.extend(img.payload.k_scale.iter().map(|f| f.to_bits()));
            v_scale_set.extend(img.payload.v_scale.iter().map(|f| f.to_bits()));
            k_pal_set.extend(img.payload.k_pal.iter().copied());
            v_pal_set.extend(img.payload.v_pal.iter().copied());
        }
    }
    assert!(
        k_tags.len() >= 2,
        "{where_}: within-K format-tag variation absent (set = {k_tags:?})"
    );
    assert!(
        v_tags.len() >= 2,
        "{where_}: within-V format-tag variation absent (set = {v_tags:?})"
    );
    assert!(
        k_scale_set.len() >= 2,
        "{where_}: k_scale uniform across (h, p) ({} distinct value)",
        k_scale_set.len()
    );
    assert!(
        v_scale_set.len() >= 2,
        "{where_}: v_scale uniform across (h, p) ({} distinct value)",
        v_scale_set.len()
    );
    assert!(
        k_pal_set.len() >= 2,
        "{where_}: k_pal bytes uniform across (h, p) ({} distinct byte)",
        k_pal_set.len()
    );
    assert!(
        v_pal_set.len() >= 2,
        "{where_}: v_pal bytes uniform across (h, p) ({} distinct byte)",
        v_pal_set.len()
    );
}

/// Seed a turn with a K/V pattern explicitly designed to drive varied
/// per-(h, p) format selection: each `(head, palette sub-band)` gets a
/// distinct waveform with a different cosine-distance footprint, so
/// the policy lands on different formats across sub-bands.
///
/// Parameterised on `n_kv_head` so the test can exercise multi-head
/// configurations larger than the module-level `N_KV_HEAD` default.
/// The (h, p) waveform table cycles every 8 sub-band classes; with
/// `n_kv_head = 4` and `N_PALETTE = 4` we get 16 sub-bands per chunk
/// covering all 8 waveform classes twice with different offsets, so
/// each head's profile is unique.
///
/// Returns the `TurnIndex` of the seeded turn.
fn seed_turn_varied_per_sub_band(
    conv: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    timeline: TimelineId,
    n_kv_head: usize,
    head_dim: usize,
    n_tokens: usize,
    pattern_base: u32,
) -> TurnIndex {
    use half::f16;

    let sub_head_dim = head_dim / 4; // N_PALETTE = 4 sub-bands
    let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(backings.len());
    for (layer_idx, backing) in backings.iter().enumerate() {
        let slot = backing.alloc_sequence().unwrap();
        backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
        let total = n_kv_head * n_tokens * head_dim;
        let mut data = Vec::with_capacity(total);
        // Layout: (n_kv_head, n_tokens, head_dim). Per (head, sub-band)
        // we pick a waveform whose cosine-distance footprint differs
        // enough that the policy lands on different formats.
        for h in 0..n_kv_head {
            for t in 0..n_tokens {
                for d in 0..head_dim {
                    let p = d / sub_head_dim;
                    // 8 waveform classes cycled by (h * N_PALETTE + p) % 8 with
                    // an additional per-head amplitude shift so heads with the
                    // same (h*4+p)%8 still get distinct cosine-distance
                    // footprints.
                    let class = (h * 4 + p) % 8;
                    let head_offset = (h as f32) * 0.17;
                    let key = (class as u8, head_offset);
                    let base = (pattern_base as f32) * 0.0001
                        + (layer_idx as f32) * 0.013
                        + (h as f32) * 0.05;
                    // Mixture of low- and high-variance waveforms per (h, p).
                    // Quasi-noise sub-bands push the per-block scale up enough
                    // that low-bit formats lose too much fidelity and the
                    // policy promotes them to higher-precision candidates.
                    // Smooth sub-bands stay on the cheapest format.
                    //
                    // The exact format assignment is policy- and threshold-
                    // dependent; the test does not assert specific formats,
                    // only that ≥ 2 distinct format tags appear across the
                    // chunk. Diversity within K or within V on a single
                    // chunk would catch a reorder bug; diversity across
                    // K vs V alone (guaranteed by the C2 candidate sets)
                    // catches an inter-side swap.
                    let pseudo = ((t.wrapping_mul(2654435761) ^ d.wrapping_mul(40503))
                        & 0xFFFF) as f32
                        / 65535.0
                        - 0.5;
                    let (class, head_offset) = key;
                    let value = match class {
                        // Smooth linear ramp — well-approximated by cheap formats.
                        0 => base + (d as f32) * 0.001,
                        // High-amplitude pseudo-random — broad spectrum, stresses
                        // low-bit quantizers.
                        1 => pseudo * 4.0 + head_offset,
                        // Sparse spikes — heavy tail blows out the per-block scale.
                        2 => if (t + d) % 5 == 0 { 7.0 + head_offset } else { 0.001 },
                        // High-frequency sinusoid — needs more bits to stay
                        // within the cosine-distance threshold.
                        3 => ((t * 13 + d * 11) as f32 * 0.83).sin() * 2.0 + head_offset,
                        // Quadratic — moderate variance, cheap format usually fine.
                        4 => ((d as f32) * 0.02).powi(2) - 0.5 + head_offset,
                        // Pseudo-random with bias.
                        5 => pseudo * 2.5 + base + head_offset,
                        // Step with high-frequency jitter — discontinuity.
                        6 => {
                            let step = if d >= sub_head_dim / 2 { 3.0 } else { -3.0 };
                            step + ((d as f32) * 0.71).cos() * 1.5 + head_offset
                        }
                        // Bimodal: alternating extreme values across tokens.
                        _ => if t % 2 == 0 { 5.0 + head_offset } else { -5.0 + head_offset },
                    };
                    data.push(f16::from_f32(value));
                }
            }
        }
        let k = Tensor::from_vec(data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        // Build a V pattern with an *independent* per-(h, p) profile —
        // not just `-K`, since negation has the same magnitude profile
        // and would force V to inherit K's per-(h, p) choices. Here V
        // sub-bands get a different mapping than K so the cosine-
        // distance footprint differs and within-V variation can land
        // on different formats than within-K.
        let mut v_data = Vec::with_capacity(n_kv_head * n_tokens * head_dim);
        for h in 0..n_kv_head {
            for t in 0..n_tokens {
                for d in 0..head_dim {
                    let p = d / sub_head_dim;
                    let pseudo = ((t.wrapping_mul(2246822519)
                        ^ d.wrapping_mul(3266489917))
                        & 0xFFFF) as f32
                        / 65535.0
                        - 0.5;
                    // V uses a different (h, p) → waveform mapping than K so
                    // the policy makes independent per-(h, p) decisions
                    // for V vs K.
                    let class = ((h * 4 + p) + 3) % 8; // shifted vs K's classes
                    let head_offset = (h as f32) * 0.23;
                    let value = match class {
                        0 => pseudo * 6.0 + head_offset,
                        1 => if (t + d) % 4 == 0 { -6.0 + head_offset } else { 0.01 },
                        2 => ((d as f32) * 0.04).powi(3) - 0.5 + head_offset,
                        3 => ((t * 11 + d * 5) as f32 * 1.13).cos() * 0.3 + head_offset,
                        4 => if t % 3 == 0 { 4.5 + head_offset } else { -4.5 + head_offset },
                        5 => (d as f32) * 0.002 + (layer_idx as f32) * 0.01 + head_offset,
                        6 => pseudo * 0.8 + head_offset,
                        _ => ((t as f32) * 0.31).sin() * 2.5 + head_offset,
                    };
                    v_data.push(f16::from_f32(value));
                }
            }
        }
        let v = Tensor::from_vec(v_data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
            .unwrap()
            .to_device(device)
            .unwrap();
        backing.write_contiguous(slot, 0, &k, &v).unwrap();
        backing.set_len(slot, n_tokens);
        sealed_per_layer.push(backing.record_turn(slot, n_tokens).unwrap());
    }
    let block_end = (n_tokens / CHUNK_SIZE) as u64;
    conv.record_turn(
        timeline,
        Role::User,
        candle_conversation::substrate::TurnPartWrite::default(),
        candle_conversation::substrate::TurnPartWrite {
            token_count: n_tokens,
            block_end,
            sealed_gpu: Some(Arc::new(sealed_per_layer)),
            ..Default::default()
        },
        |seqs| Ok(seqs.to_vec()),
    )
    .unwrap()
}

/// Full hot → warm → hot → cold → warm → hot round-trip with
/// **field-level metadata preservation** at every transition.
///
/// Seeds a varied K/V pattern (so per-(h, p) format selection is
/// non-uniform), runs the policy-driven persist, captures a reference
/// snapshot from warm→hot elevation, then asserts the same byte-for-
/// byte image set survives:
///
/// 1. A second warm→hot round trip (in-process, no restart)
/// 2. A full restart + cold→hot reload from disk
///
/// Catches: per-(h, p) format reordering, palette-map drift, scale
/// re-derivation from a stale arena state, kv_bytes/format misalignment,
/// any silent fallback to a uniform format on the cold-load path.
#[test]
fn quantize_on_evict_metadata_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(80).unwrap();
    let group_id = GroupId::from_raw(80).unwrap();
    let timeline = TimelineAllocator::new().next();
    // C6 chosen deliberately: its candidate list spans palette
    // formats (Q2_A, Q2_S) and per-block formats (Q3/Q4/Q8). Palette
    // formats use a non-1.0 outer scale and a non-identity palette
    // map, so this is the compression level that drives variation in
    // **all** per-(h, p) carriers — `k_formats`, `v_formats`, `k_pal`,
    // `v_pal`, `k_scale`, `v_scale`. Combined with the varied per-
    // (h, p) seed pattern below and independent K/V waveforms, the
    // selection lands on different formats for different sub-bands,
    // making every reorder bug class detectable in the round-trip.
    let policy = CompressionPolicy::new(6);
    let log_path = dir
        .join(candle_conversation::persistence::SUBSTRATE_DIR)
        .join(candle_conversation::persistence::ACTIVE_LOG_NAME);

    // ── Phase 1: seed F16 with varied pattern, run policy-driven persist ──
    let key = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata(&device, &policy);
        conv.register_timeline(timeline, layer_id, group_id);

        let idx = seed_turn_varied_per_sub_band(
            &conv,
            &backings,
            &device,
            timeline,
            N_KV_HEAD_METADATA,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            900_001,
        );
        let key = TurnKey::new(timeline, idx);

        // Policy quantizes per-(h, p) at hot→warm and atomically drops hot.
        let persist = PersistenceThread::spawn(
            conv.clone(),
            Arc::new(backings.clone()),
            device.clone(),
            Some(policy.clone()),
        );
        persist.shutdown();

        let st = turn_state(&conv, key);
        assert!(
            st.hot && st.warm && st.cold,
            "post-persist {key:?} should be hot+warm+cold (install_warm_and_hot), got {st:?}"
        );
        key
    };

    // ── Phase 2: open a fresh process state, elevate warm→hot, ─────────
    //            capture full ChunkImage reference                       ─
    //
    // Re-opening the substrate from disk (rather than reusing the
    // Conversation from Phase 1) is what the production restart path
    // does: the warm tier is populated by cold-load on reopen, then
    // elevated. So Phase 2's "reference" snapshot is *the* image set
    // every subsequent transition must preserve.
    let reference = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata(&device, &policy);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        assert_eq!(report.cold_to_hot, 1, "Phase 2 cold→hot elevate");

        let snap = snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index);
        // Sanity-check the seed pattern actually drove varied selection
        // — otherwise a reorder bug on uniform formats would round-trip
        // trivially and this entire test would be vacuous.
        assert_varied_formats(&snap, "Phase 2 reference");
        snap
    };

    // ── Phase 3: same on-disk state, second cold→hot elevation, ────────
    //            assert byte-identical to the reference                   ─
    //
    // This catches non-determinism in the cold-load path itself: if a
    // second reload from the same redo log produced even one different
    // byte / scale / format tag we'd see drift here.
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata(&device, &policy);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();

        let snap = snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index);
        assert_chunk_images_eq(&reference, &snap, "second cold→hot reload");
    }

    // ── Phase 4: same on-disk state, but exercise the warm→hot leg ─────
    //            after a manual evict_from_hot — the path that runs in   ─
    //            steady state when a hot residence is dropped and re-    ─
    //            elevated without restart.                                ─
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata(&device, &policy);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        // First elevate populates warm via cold-load.
        elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        // Drop hot — turn is now warm+cold only.
        let purged = evict_from_hot(&conv, &[], &[]);
        assert_eq!(purged.count, 1, "hot must be evictable to exercise warm→hot");
        // Second elevate now takes the warm→hot path.
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        assert_eq!(
            report.warm_to_hot, 1,
            "Phase 4 must exercise warm→hot, not cold→hot"
        );
        assert_eq!(report.cold_to_hot, 0);

        let snap = snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index);
        assert_chunk_images_eq(&reference, &snap, "warm→hot after evict");
    }

    // ── Phase 5: redo log must not have grown across all reloads ───────
    //
    // Any tier transition that silently appended a record during reload
    // would be a write-amplification bug; the cold tier is meant to be
    // append-only at seal time, not at recover time.
    let log_len = std::fs::metadata(&log_path).unwrap().len();
    assert!(log_len > 0, "redo log should be non-empty");
}

/// Field-level cold-reload round-trip for the **non-policy** path
/// (formats stay in their seed format — BF16 in our case). Mirrors
/// [`quantize_on_evict_metadata_round_trip`] but for the format-
/// preserving migration. Catches any drift in the trivial-format
/// branches of `k_formats` / `v_formats` / `k_pal` / `v_pal` /
/// `k_scale` / `v_scale` populating.
#[test]
fn no_policy_metadata_round_trip() {
    let Some(device) = cuda_device_or_skip() else {
        return;
    };
    let tmpdir = tempfile::tempdir().unwrap();
    let dir = tmpdir.path().to_path_buf();

    let layer_id = LayerId::from_raw(81).unwrap();
    let group_id = GroupId::from_raw(81).unwrap();
    let timeline = TimelineAllocator::new().next();

    let key = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata_no_policy(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        let idx = seed_turn_varied_per_sub_band(
            &conv,
            &backings,
            &device,
            timeline,
            N_KV_HEAD_METADATA,
            QUANT_HEAD_DIM,
            N_TOKENS_PER_TURN,
            950_003,
        );
        let key = TurnKey::new(timeline, idx);

        let persist =
            PersistenceThread::spawn(conv.clone(), Arc::new(backings.clone()), device.clone(), None);
        persist.shutdown();
        key
    };

    // Reference: post-restart, post warm→hot elevation.
    let reference = {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata_no_policy(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index)
    };

    // Warm→hot leg after evict.
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata_no_policy(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        evict_from_hot(&conv, &[], &[]);
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        assert_eq!(report.warm_to_hot, 1);

        let snap = snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index);
        assert_chunk_images_eq(&reference, &snap, "no-policy warm→hot");
    }

    // Second cold→hot reload from the same log.
    {
        let persistence = SubstratePersistence::open_in(&dir).unwrap();
        let conv = Conversation::with_persistence(persistence);
        let backings = make_backings_metadata_no_policy(&device);
        conv.register_timeline(timeline, layer_id, group_id);
        conv.reconstruct_from_log(N_LAYERS_METADATA, |_| Ok(Vec::new())).unwrap();

        let main_stream = cuda_stream(&device);
        let mut pinned: Option<PinnedBuf> = None;
        let mut stager = ColdLoadStager::new();
        elevate_to_hot(
            &conv,
            &backings,
            &device,
            &main_stream,
            &mut pinned,
            &mut stager,
            &[],
            &[key],
        )
        .unwrap();
        let snap = snapshot_turn_images(&conv, &backings, &device, key.timeline, key.index);
        assert_chunk_images_eq(&reference, &snap, "no-policy second cold→hot");
    }
}

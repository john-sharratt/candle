//! Bulk hot-tier promotion — bring a batch of sections + turns into
//! VRAM in one orchestrated pass.
//!
//! Pairs with [`Substrate::evict_hot_except`] (working-set eviction):
//! together they form the workspace's tier-management hot path. The
//! scheduler calls `elevate_to_hot` ahead of a projection that needs
//! a working set of turns ready for decode; the persistence thread
//! independently caches hot→warm and persists warm→cold.
//!
//! ```text
//!   classify (substrate.read) ──┬─▶ already_hot   (skip)
//!                               │
//!                               ├─▶ warm_to_hot   ──▶ migrate_sealed_to_gpu_batch_async
//!                               │                      (one kv_migrate_on scatter per layer
//!                               │                       on the dedicated copy stream)
//!                               │
//!                               ├─▶ cold_to_hot   ──▶ recover_turn_chunks + load_to_hot
//!                               │                      (NVMe → pinned scratch → VRAM)
//!                               │
//!                               └─▶ missing       (warn, count)
//!
//!   install_promoted_hot (substrate.write) ──▶ residence.hot = Some(...)
//! ```

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{Device, Result};
use candle_nn::kv_cache::ChunkedKvBacking;

use super::cold_load::ColdLoadStager;
use crate::projection::{Conversation, SectionId, TurnKey};
use crate::substrate::{
    ColdRecall, EvictionReport, PromotionItemKind, PromotionPlan, PurgeReport, ResidenceIndex,
    WarmLift, WarmToHotEntry,
};
use candle_nn::kv_cache::SealedSequence;
use sysinfo::System;

/// Sum of `SealedChunk.byte_size` across every chunk of every layer
/// in a per-layer `Vec<SealedSequence>` — the per-tier memory cost
/// of holding this turn hot or warm.
fn sealed_total_bytes(seqs: &[SealedSequence]) -> u64 {
    seqs.iter()
        .flat_map(|s| s.chunks.iter())
        .map(|c| c.byte_size)
        .sum()
}

/// Per-pass summary of what `elevate_to_hot` did. Counts go alongside
/// the byte volumes actually moved between tiers, so callers can log
/// or assert on either dimension.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ElevationReport {
    /// Items that were already hot — no work was done for them.
    pub already_hot: usize,
    /// Items promoted warm → hot via the batched-async migrate path.
    pub warm_to_hot: usize,
    /// Items promoted cold → hot via the redo-log + cold-load path.
    pub cold_to_hot: usize,
    /// Items not present in the substrate. Logged and skipped.
    pub missing: usize,
    /// Items that hit an error mid-promotion. Best-effort means
    /// other items in the batch still succeed; the report tells the
    /// caller how many to retry.
    pub failed: usize,
    /// Total bytes moved warm → hot. Sum of each promoted item's
    /// `residence.byte_size` (the per-tier sealed payload size).
    pub bytes_warm_to_hot: u64,
    /// Total bytes moved cold → hot. Same accounting unit as
    /// `bytes_warm_to_hot`.
    pub bytes_cold_to_hot: u64,
    /// Warm-tier residences whose warm copy was dropped to make
    /// headroom for the upcoming cold→warm install (phase 2a purge).
    pub warm_purged: usize,
    /// RAM bytes freed by the phase 2a purge (sum of per-residence
    /// `byte_size`).
    pub bytes_warm_purged: u64,
}

impl ElevationReport {
    /// Total items the caller asked us to elevate (sum across buckets).
    pub fn total(&self) -> usize {
        self.already_hot + self.warm_to_hot + self.cold_to_hot + self.missing + self.failed
    }

    /// Total bytes actually moved into VRAM in this call.
    pub fn bytes_moved(&self) -> u64 {
        self.bytes_warm_to_hot + self.bytes_cold_to_hot
    }
}

/// Bring every named section + turn into VRAM, using whichever tier
/// they currently live in as the source.
///
/// **Best-effort.** Individual item failures are recorded in
/// `ElevationReport.failed` and logged via `tracing::warn`; the
/// remaining items still complete. The caller can re-issue
/// `elevate_to_hot` for retries.
///
/// Single substrate read at the start (classification) + at most one
/// substrate write at the end (bulk install). Between them, all GPU
/// work runs on `copy_stream` with the pinned host scratch caller
/// provided.
pub fn elevate_to_hot(
    conversation: &Conversation,
    backings: &[ChunkedKvBacking],
    device: &Device,
    copy_stream: &Arc<CudaStream>,
    pinned_scratch: &mut Option<PinnedBuf>,
    cold_stager: &mut ColdLoadStager,
    sections: &[SectionId],
    turns: &[TurnKey],
) -> Result<ElevationReport> {
    // ── Phase 1: classify ───────────────────────────────────────────────
    let plan: PromotionPlan = conversation
        .read()
        .snapshot_promotion_state(sections, turns);

    let mut report = ElevationReport {
        already_hot: plan.already_hot.len(),
        missing: plan.missing.len(),
        ..Default::default()
    };
    // `missing` is "substrate has no record of this id" — the
    // genuine error case.  `tier_less` (tracked in substrate but no
    // K/V to elevate, the design-intended state for ghost summary
    // turns) is silently skipped without alarming logs.
    for kind in &plan.missing {
        tracing::warn!("elevate_to_hot: item not found in substrate: {kind:?}");
    }

    let mut recalls: Vec<ColdRecall> = Vec::new();
    let mut lifts: Vec<WarmLift> = Vec::new();
    let n_layers = backings.len();

    // ── Phase 2a: warm purge (single batch, ahead of every cold→warm) ──
    //
    // Cold→hot pre-populates a fresh warm copy alongside the hot
    // install (see phase 2b), so each cold item adds RAM pressure
    // equal to its `byte_size`. Before doing any of that work, sum
    // the incoming RAM cost once and ask the substrate to drop LRU
    // warm residences until the OS would still have at least
    // `max(2 GiB, 5% × total_ram)` available after the upcoming
    // allocation lands. Section cold-load isn't wired up; only turns
    // contribute to the incoming budget.
    if !plan.cold_to_hot.is_empty() {
        let incoming_bytes: u64 = plan
            .cold_to_hot
            .iter()
            .filter(|c| matches!(c.kind, PromotionItemKind::Turn(_)))
            .flat_map(|c| c.cold.iter())
            .flat_map(|s| s.chunks.iter())
            .map(|c| c.record_len)
            .sum();
        if incoming_bytes > 0 {
            let mut sys = System::new();
            sys.refresh_memory();
            let total_ram = sys.total_memory();
            let available_ram = sys.available_memory();
            let purged: PurgeReport =
                conversation
                    .write()
                    .purge_warm_to_target(incoming_bytes, available_ram, total_ram);
            report.warm_purged = purged.count;
            report.bytes_warm_purged = purged.bytes;
        }
    }

    // ── Phase 2b.i: per-turn recover + load_to_hot (NVMe + GPU scatter)
    //
    // Walk every cold turn once: pull its chunk grid off disk via
    // `recover_turn_chunks`, then scatter into fresh GPU arena slots
    // via `load_to_hot`. Failures here are isolated to the failing
    // turn (it's accounted in the report and skipped from the rest
    // of the phase); the per-turn work can't be batched across turns
    // because each turn allocates its own scratch slot inside
    // load_stream.
    //
    // Section cold-load isn't wired up (see hot_section_or_skip in
    // scheduler/mod.rs); cold sections are warned and dropped.
    struct PendingRecall {
        kind: PromotionItemKind,
        residence: ResidenceIndex,
        hot_sealed: Vec<SealedSequence>,
        bytes_for_item: u64,
        timeline_raw: u64,
        turn_index: u32,
    }
    let mut pending: Vec<PendingRecall> = Vec::with_capacity(plan.cold_to_hot.len());
    for cold_entry in plan.cold_to_hot {
        match cold_entry.kind {
            PromotionItemKind::Turn(key) => {
                // Fused cold-load fast path: one stripe-coalesced disk
                // read straight into pinned host scratch, one HtoD, one
                // batched `kv_migrate` covering every layer's
                // destinations. Replaces the old
                // `recover_turn_chunks` + `load_to_hot` pair which paid
                // a per-chunk `Vec<u8>` allocation for `kv_bytes` and a
                // host-to-host memcpy from heap into the pinned scratch
                // before HtoD.
                let (hot_sealed, bytes_for_item) = match conversation.cold_load_turn_into_hot(
                    key.timeline,
                    key.index,
                    backings,
                    device,
                    cold_stager,
                ) {
                    Ok(Some(pair)) => pair,
                    Ok(None) => {
                        tracing::warn!(
                            "elevate_to_hot: cold-load found no chunks for turn {key:?}"
                        );
                        report.missing += 1;
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("elevate_to_hot: cold_load_turn_into_hot {key:?}: {e}");
                        report.failed += 1;
                        continue;
                    }
                };
                pending.push(PendingRecall {
                    kind: cold_entry.kind,
                    residence: cold_entry.residence,
                    hot_sealed,
                    bytes_for_item,
                    timeline_raw: key.timeline.raw(),
                    turn_index: key.index.0,
                });
            }
            PromotionItemKind::Section(sid) => {
                // Cold-load the section's chunks via the shared
                // `load_stream_into_hot` pipeline (same machinery
                // turns use).  Sections are read-mostly fixtures
                // installed at conversation construction; once a
                // reload installs cold-markers, the next projection
                // that depends on this section triggers the cold→hot
                // lift here.
                let chunks_per_layer = cold_entry
                    .cold
                    .first()
                    .map(|s| s.chunks.len())
                    .unwrap_or(0);
                if chunks_per_layer == 0 {
                    tracing::warn!(
                        "elevate_to_hot: section {sid:?} has empty cold refs — skipping"
                    );
                    report.failed += 1;
                    continue;
                }
                let backings_slice = backings;
                let hot_sealed = match conversation.cold_load_section_into_hot(
                    cold_entry.stream_id,
                    chunks_per_layer,
                    backings_slice,
                    device,
                    cold_stager,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::warn!(
                            "elevate_to_hot: cold-load section {sid:?}: {e}"
                        );
                        report.failed += 1;
                        continue;
                    }
                };
                let bytes_for_item: u64 = cold_entry
                    .cold
                    .iter()
                    .flat_map(|s| s.chunks.iter())
                    .map(|c| c.record_len)
                    .sum();
                pending.push(PendingRecall {
                    kind: cold_entry.kind,
                    residence: cold_entry.residence,
                    hot_sealed,
                    bytes_for_item,
                    // Sections don't have timeline/turn coordinates;
                    // the trace just sees `0` and the section id.
                    timeline_raw: 0,
                    turn_index: sid.raw(),
                });
            }
        }
    }

    // ── Phase 2b.ii: batched cold→warm migrate (per layer, all turns) ──
    //
    // Cold→warm materialises the CPU-arena copy alongside hot so the
    // next hot eviction is no-DMA (warm already there). We batch
    // ACROSS turns per layer — one `migrate_sealed_to_cpu_batch_async`
    // call per layer handles every pending turn's layer-`L` sequence
    // in a single gather kernel + DtoH. Without this batching, a
    // fresh-restart submit with N cold turns × L layers paid N×L
    // sync overheads instead of L. For 16 cold turns × 30 layers
    // that's ~480 sync points vs ~30 — same memory volume, far less
    // launch + sync overhead.
    //
    // Best-effort: a layer-batch failure drops warm for **all** turns
    // (graceful fallback to hot-only across the batch). The per-turn
    // robustness of the old loop is sacrificed for the batching win;
    // the failure mode is rare and the hot tier is still correct.
    let mut warm_per_turn: Vec<Vec<SealedSequence>> = (0..pending.len())
        .map(|_| Vec::with_capacity(n_layers))
        .collect();
    let mut batch_ok = !pending.is_empty();
    if batch_ok {
        for (layer, backing) in backings.iter().enumerate().take(n_layers) {
            let inputs: Vec<&SealedSequence> =
                pending.iter().map(|p| &p.hot_sealed[layer]).collect();
            match backing.migrate_sealed_to_cpu_batch_async(
                device,
                copy_stream,
                pinned_scratch,
                &inputs,
            ) {
                Ok(layer_warm) => {
                    if layer_warm.len() != pending.len() {
                        tracing::warn!(
                            "elevate_to_hot: cold→warm migrate layer {layer} returned \
                             {} sequences for {} inputs — landing hot-only for the \
                             whole cold batch",
                            layer_warm.len(),
                            pending.len()
                        );
                        batch_ok = false;
                        break;
                    }
                    for (i, seq) in layer_warm.into_iter().enumerate() {
                        warm_per_turn[i].push(seq);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "elevate_to_hot: cold→warm migrate layer {layer} batched failed: \
                         {e} — landing hot-only for the whole cold batch"
                    );
                    batch_ok = false;
                    break;
                }
            }
        }
    }

    // ── Phase 2b.iii: assemble ColdRecall installs ─────────────────────
    for (i, p) in pending.into_iter().enumerate() {
        let warm = if batch_ok {
            std::mem::take(&mut warm_per_turn[i])
        } else {
            Vec::new()
        };
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            timeline = p.timeline_raw,
            turn = p.turn_index,
            residence = p.residence.0,
            bytes = p.bytes_for_item,
            warm_landed = !warm.is_empty(),
            "cold recall (hot + warm) (turn)"
        );
        recalls.push(ColdRecall {
            kind: p.kind,
            residence: p.residence,
            hot: p.hot_sealed,
            warm,
        });
        report.cold_to_hot += 1;
        report.bytes_cold_to_hot = report.bytes_cold_to_hot.saturating_add(p.bytes_for_item);
    }

    // ── Phase 3: warm → hot (PCIe-bound, batched per layer) ────────────
    //
    // Build the per-layer input vectors once across all warm items,
    // then issue one `migrate_sealed_to_gpu_batch_async` per layer.
    // The async path internally batches the gather + HtoD + scatter
    // on the dedicated copy stream.
    let warm_items: Vec<WarmToHotEntry> = plan
        .warm_to_hot
        .into_iter()
        .filter(|w| {
            if w.warm.len() != n_layers {
                tracing::warn!(
                    "elevate_to_hot: warm layer-count mismatch for {kind:?} ({} vs {n_layers})",
                    w.warm.len(),
                    kind = w.kind
                );
                report.failed += 1;
                false
            } else {
                true
            }
        })
        .collect();

    if !warm_items.is_empty() {
        // Per-item collector of (layer-indexed) hot results.
        let mut hot_per_item: Vec<Vec<candle_nn::kv_cache::SealedSequence>> = (0..warm_items.len())
            .map(|_| Vec::with_capacity(n_layers))
            .collect();
        let mut layer_ok = true;
        for (layer, backing) in backings.iter().enumerate().take(n_layers) {
            let inputs: Vec<&candle_nn::kv_cache::SealedSequence> =
                warm_items.iter().map(|w| &w.warm[layer]).collect();
            match backing.migrate_sealed_to_gpu_batch_async(
                device,
                copy_stream,
                pinned_scratch,
                &inputs,
            ) {
                Ok(layer_hot) => {
                    for (i, h) in layer_hot.into_iter().enumerate() {
                        hot_per_item[i].push(h);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "elevate_to_hot: warm→hot layer {layer} batched migrate failed: {e}"
                    );
                    layer_ok = false;
                    break;
                }
            }
        }
        if layer_ok {
            for (i, w) in warm_items.into_iter().enumerate() {
                let hot = std::mem::take(&mut hot_per_item[i]);
                if hot.len() == n_layers {
                    let bytes_for_item = sealed_total_bytes(&hot);
                    match w.kind {
                        PromotionItemKind::Turn(key) => {
                            tracing::debug!(
                                target: "candle_conversation::persistence::tier",
                                timeline = key.timeline.raw(),
                                turn = key.index.0,
                                residence = w.residence.0,
                                bytes = bytes_for_item,
                                "warm lift (turn)"
                            );
                        }
                        PromotionItemKind::Section(sid) => {
                            tracing::debug!(
                                target: "candle_conversation::persistence::tier",
                                section = sid.raw(),
                                residence = w.residence.0,
                                bytes = bytes_for_item,
                                "warm lift (section)"
                            );
                        }
                    }
                    lifts.push(WarmLift {
                        kind: w.kind,
                        residence: w.residence,
                        hot,
                    });
                    report.warm_to_hot += 1;
                    report.bytes_warm_to_hot =
                        report.bytes_warm_to_hot.saturating_add(bytes_for_item);
                } else {
                    report.failed += 1;
                }
            }
        } else {
            // Whole warm batch failed — count each as a failure so
            // the caller knows nothing was promoted.
            report.failed += warm_items.len();
        }
    }

    // ── Phase 4: install (one substrate write) ─────────────────────────
    //
    // Both elevation legs land under a single write lock. `recalls`
    // installs warm + hot per item; `lifts` installs hot only (warm
    // was already present on the residence pre-promotion).
    if !recalls.is_empty() || !lifts.is_empty() {
        conversation.write().install_promoted(recalls, lifts);
    }

    // Aggregate summary — RUST_LOG=substrate::tier=info catches just
    // these lines without the per-item detail.
    if report.total() > 0 {
        tracing::info!(
            target: "candle_conversation::persistence::tier",
            already_hot = report.already_hot,
            warm_to_hot = report.warm_to_hot,
            cold_to_hot = report.cold_to_hot,
            missing = report.missing,
            failed = report.failed,
            bytes_warm_to_hot = report.bytes_warm_to_hot,
            bytes_cold_to_hot = report.bytes_cold_to_hot,
            "elevate_to_hot batch complete"
        );
    }

    Ok(report)
}

/// Working-set-aware bulk hot-tier eviction — the counterpart to
/// [`elevate_to_hot`].
///
/// Given the *same* `(sections, turns)` pair that's about to be
/// elevated, this drops hot bytes from every warm-backed hot residence
/// that is **not** in the incoming working set. Calling this
/// immediately before `elevate_to_hot` opens VRAM headroom for the
/// incoming scatter without churning bytes that will be re-promoted
/// in the very next call.
///
/// The actual filtered eviction happens inside
/// [`crate::substrate::Substrate::evict_hot_except`], which resolves
/// section / turn keys to residence indices and walks `hot_lru`
/// under a single write lock. This top-level function is a thin
/// orchestrator preserved for call-site symmetry with
/// `elevate_to_hot`.
pub fn evict_from_hot(
    conversation: &Conversation,
    sections: &[SectionId],
    turns: &[TurnKey],
) -> EvictionReport {
    conversation.write().evict_hot_except(sections, turns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId, TimelineAllocator, TimelineId, TurnIndex};
    use crate::substrate::TurnPartWrite;
    use crate::turn::Role;
    use candle::{DType, Tensor};
    use candle_nn::kv_cache::{ChunkedKvBacking, SealedSequence};
    use half::bf16;

    /// Skip-on-no-GPU helper. Mirrors the candle-nn batched-async tests.
    fn cuda_device_or_skip() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => Some(d),
            _ => None,
        }
    }

    /// Create a deterministic BF16 turn across `n_layers` backings and
    /// register it in `conv`. Returns the freshly-appended turn index.
    fn seed_turn(
        conv: &Conversation,
        backings: &[ChunkedKvBacking],
        device: &Device,
        timeline: TimelineId,
        n_kv_head: usize,
        head_dim: usize,
        n_tokens: usize,
        pattern_base: u32,
    ) -> TurnIndex {
        let mut sealed_per_layer: Vec<SealedSequence> = Vec::with_capacity(backings.len());
        for backing in backings {
            let slot = backing.alloc_sequence().unwrap();
            backing.ensure_for_offset(slot, 0, n_tokens).unwrap();
            let total = n_kv_head * n_tokens * head_dim;
            let data: Vec<bf16> = (0..total)
                .map(|i| bf16::from_f32(((pattern_base as usize + i) as f32) * 0.001))
                .collect();
            let k = Tensor::from_vec(data, (1, n_kv_head, n_tokens, head_dim), &Device::Cpu)
                .unwrap()
                .to_device(device)
                .unwrap();
            let v = k.clone();
            backing.write_contiguous(slot, 0, &k, &v).unwrap();
            backing.set_len(slot, n_tokens);
            sealed_per_layer.push(backing.record_turn(slot).unwrap());
        }
        conv.record_turn(
            timeline,
            Role::User,
            TurnPartWrite {
                token_count: n_tokens,
                sealed_gpu: Some(Arc::new(sealed_per_layer)),
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .unwrap()
    }

    /// For each layer, run a single-element batched async migrate to
    /// produce a CPU-backed `SealedSequence`, then collect across
    /// layers into the `Vec<SealedSequence>` shape `install_warm`
    /// expects. Lets a test install warm copies of an existing hot
    /// turn so subsequent eviction + elevation can be exercised.
    fn migrate_layers_to_cpu(
        backings: &[ChunkedKvBacking],
        device: &Device,
        copy_stream: &Arc<CudaStream>,
        pinned: &mut Option<PinnedBuf>,
        hot: &[SealedSequence],
    ) -> Vec<SealedSequence> {
        let mut out = Vec::with_capacity(hot.len());
        for (layer, seq) in hot.iter().enumerate() {
            let warm = backings[layer]
                .migrate_sealed_to_cpu_batch_async(device, copy_stream, pinned, &[seq])
                .unwrap();
            out.push(warm.into_iter().next().unwrap());
        }
        out
    }

    /// Standard test setup: ephemeral conversation, N CUDA-backed
    /// `ChunkedKvBacking`s (one per layer), a registered timeline.
    fn fresh_setup(
        device: &Device,
        n_layers: usize,
    ) -> (Conversation, Vec<ChunkedKvBacking>, TimelineId) {
        let conv = Conversation::ephemeral();
        let backings: Vec<ChunkedKvBacking> = (0..n_layers)
            .map(|_| ChunkedKvBacking::new(4, 2, 16, DType::BF16, device, 256).unwrap())
            .collect();
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let allocator = TimelineAllocator::new();
        let timeline = allocator.next();
        conv.register_timeline(timeline, layer, group);
        (conv, backings, timeline)
    }

    /// Cuda resources the elevate path needs.
    struct CudaResources {
        copy_stream: Arc<CudaStream>,
        pinned: Option<PinnedBuf>,
        stager: ColdLoadStager,
    }

    fn cuda_resources(device: &Device) -> CudaResources {
        let cuda_dev = match device {
            Device::Cuda(d) => d,
            _ => unreachable!(),
        };
        CudaResources {
            copy_stream: cuda_dev.cuda_context().new_stream().unwrap(),
            pinned: None,
            stager: ColdLoadStager::new(),
        }
    }

    /// `elevate_to_hot` on an already-hot turn is a pure-bookkeeping
    /// no-op: the report counts it under `already_hot` and no other
    /// bucket changes.
    #[test]
    fn already_hot_items_are_idempotent() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let idx = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 0);
        let mut r = cuda_resources(&device);

        let key = TurnKey::new(timeline, idx);
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &mut r.stager,
            &[],
            &[key],
        )
        .unwrap();

        assert_eq!(report.already_hot, 1);
        assert_eq!(report.warm_to_hot, 0);
        assert_eq!(report.cold_to_hot, 0);
        assert_eq!(report.missing, 0);
        assert_eq!(report.failed, 0);
        // Residence is still in hot_lru with a hot byte payload.
        assert!(conv.read().turn_sealed_of(timeline, idx).is_some());
    }

    /// Missing keys (timeline never registered, or turn index past
    /// the tail) get accounted under `missing` and otherwise leave
    /// the substrate alone.
    #[test]
    fn missing_items_are_reported() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, _timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let bogus = TurnKey::new(TimelineId::for_test(999), TurnIndex(42));
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &mut r.stager,
            &[],
            &[bogus],
        )
        .unwrap();

        assert_eq!(report.missing, 1);
        assert_eq!(report.already_hot, 0);
        assert_eq!(report.warm_to_hot, 0);
        assert_eq!(report.cold_to_hot, 0);
        assert_eq!(report.failed, 0);
    }

    /// Warm → hot promotion: seed a turn, install its warm copy, evict
    /// hot, run elevate. Final state: hot back, warm preserved, residence
    /// re-listed on the hot LRU.
    #[test]
    fn warm_to_hot_promotion_restores_vram_residence() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let idx = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 7);
        let mut r = cuda_resources(&device);

        // Build a CPU copy of the just-seeded hot bytes and install
        // it as warm.
        let residence = conv.read().turn_residence(timeline, idx).unwrap();
        let hot_arc = conv.read().turn_sealed_of(timeline, idx).unwrap();
        let warm =
            migrate_layers_to_cpu(&backings, &device, &r.copy_stream, &mut r.pinned, &hot_arc);
        // `hot_arc` is the only outstanding Arc apart from the substrate's
        // — drop it so install_warm + eviction don't see a phantom
        // borrower.
        drop(hot_arc);
        conv.write().install_warm(residence, warm);

        // Drop hot — turn is now warm-only.
        let evicted = conv.write().evict_hot_except(&[], &[]);
        assert_eq!(evicted.count, 1);
        assert!(evicted.bytes > 0, "evicted bytes must be non-zero");
        assert!(
            conv.read().turn_sealed_of(timeline, idx).is_none(),
            "post-eviction the turn must be cold-marker"
        );

        // Run elevate — it should promote warm → hot.
        let key = TurnKey::new(timeline, idx);
        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &mut r.stager,
            &[],
            &[key],
        )
        .unwrap();

        assert_eq!(report.warm_to_hot, 1);
        assert_eq!(report.already_hot, 0);
        assert_eq!(report.cold_to_hot, 0);
        assert_eq!(report.missing, 0);
        assert_eq!(report.failed, 0);

        // Hot bytes are back.
        let restored = conv.read().turn_sealed_of(timeline, idx);
        assert!(restored.is_some(), "post-elevate the turn must be hot");
        let restored = restored.unwrap();
        assert_eq!(restored.len(), 2, "n_layers per the test");
    }

    /// A batch mixing already-hot turns, a warm-only turn, and a
    /// missing key produces a report with each bucket populated.
    #[test]
    fn mixed_batch_reports_each_bucket() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        // Turn 0: stays hot.
        let hot_idx = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 0);

        // Turn 1: warm-only.
        let warm_idx = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 1000);
        let residence = conv.read().turn_residence(timeline, warm_idx).unwrap();
        let hot_arc = conv.read().turn_sealed_of(timeline, warm_idx).unwrap();
        let warm =
            migrate_layers_to_cpu(&backings, &device, &r.copy_stream, &mut r.pinned, &hot_arc);
        drop(hot_arc);
        conv.write().install_warm(residence, warm);

        // Bogus key — missing bucket.
        let missing = TurnKey::new(TimelineId::for_test(999), TurnIndex(42));

        // Evict hot just for turn 1 — install_warm only made it
        // dual-resident; eviction drops hot since warm is present.
        // `evict_hot_except(&[], &[])` is the unfiltered case: only
        // entries where BOTH hot is some AND warm is some are
        // eligible. Turn 0 has warm=None, so it stays hot.
        let evicted = conv.write().evict_hot_except(&[], &[]);
        assert_eq!(evicted.count, 1, "only warm-backed turn 1 should evict");

        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &mut r.stager,
            &[],
            &[
                TurnKey::new(timeline, hot_idx),
                TurnKey::new(timeline, warm_idx),
                missing,
            ],
        )
        .unwrap();

        assert_eq!(report.already_hot, 1, "turn 0 is still hot");
        assert_eq!(report.warm_to_hot, 1, "turn 1 got promoted");
        assert_eq!(report.missing, 1, "bogus key counted");
        assert_eq!(report.failed, 0);
        assert_eq!(report.total(), 3);
    }

    /// Empty input is a no-op — no read lock, no write lock, no GPU work.
    #[test]
    fn empty_input_is_noop() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, _timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let report = elevate_to_hot(
            &conv,
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &mut r.stager,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(report, ElevationReport::default());
    }

    /// Helper: seed a turn, build + install its warm copy. Returns the
    /// turn index. Leaves the residence dual-resident (hot + warm),
    /// which is the precondition `evict_*` looks for.
    fn seed_warm_backed_turn(
        conv: &Conversation,
        backings: &[ChunkedKvBacking],
        device: &Device,
        timeline: TimelineId,
        copy_stream: &Arc<CudaStream>,
        pinned: &mut Option<PinnedBuf>,
        pattern_base: u32,
    ) -> TurnIndex {
        let idx = seed_turn(conv, backings, device, timeline, 2, 16, 32, pattern_base);
        let residence = conv.read().turn_residence(timeline, idx).unwrap();
        let hot_arc = conv.read().turn_sealed_of(timeline, idx).unwrap();
        let warm = migrate_layers_to_cpu(backings, device, copy_stream, pinned, &hot_arc);
        drop(hot_arc);
        conv.write().install_warm(residence, warm);
        idx
    }

    /// All hot turns are also in the keep set → no eviction.
    #[test]
    fn keep_set_covers_all_hot_then_no_eviction() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let a = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            100,
        );
        let b = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            200,
        );

        let keep_turns = vec![TurnKey::new(timeline, a), TurnKey::new(timeline, b)];
        let report = evict_from_hot(&conv, &[], &keep_turns);

        assert_eq!(report.count, 0, "every hot residence is in keep set");
        assert_eq!(report.bytes, 0);
        // Both turns still resolve hot.
        assert!(conv.read().turn_sealed_of(timeline, a).is_some());
        assert!(conv.read().turn_sealed_of(timeline, b).is_some());
    }

    /// Empty keep set → unfiltered eviction: every warm-backed hot
    /// residence is dropped.
    #[test]
    fn empty_keep_set_evicts_every_warm_backed_residence() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let a = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            100,
        );
        let b = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            200,
        );

        let report = evict_from_hot(&conv, &[], &[]);

        assert_eq!(report.count, 2, "both warm-backed residences evicted");
        assert!(report.bytes > 0);
        // Both turns are now cold-marker (warm-only).
        assert!(conv.read().turn_sealed_of(timeline, a).is_none());
        assert!(conv.read().turn_sealed_of(timeline, b).is_none());
    }

    /// Mixed keep set: A stays, B evicts. Asserts each one ends in the
    /// expected tier and the report counts only B.
    #[test]
    fn mixed_keep_set_evicts_only_non_kept() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let a = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            100,
        );
        let b = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            200,
        );

        // Keep only A.
        let keep = vec![TurnKey::new(timeline, a)];
        let report = evict_from_hot(&conv, &[], &keep);

        assert_eq!(report.count, 1, "only B should be evicted");
        assert!(report.bytes > 0);
        // A is still hot, B is cold-marker (warm-only).
        assert!(conv.read().turn_sealed_of(timeline, a).is_some(), "A kept");
        assert!(
            conv.read().turn_sealed_of(timeline, b).is_none(),
            "B evicted"
        );
    }

    /// A turn with no warm backup is skipped even when it's not in the
    /// keep set — the loss-free invariant holds (dropping hot without
    /// warm would force a cold-load on re-read).
    #[test]
    fn hot_without_warm_is_skipped_even_when_not_kept() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        // Hot-only turn (no warm copy installed).
        let no_warm = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 0);
        // Warm-backed turn to give the report something to count.
        let warm_backed = seed_warm_backed_turn(
            &conv,
            &backings,
            &device,
            timeline,
            &r.copy_stream,
            &mut r.pinned,
            200,
        );

        // Empty keep set — would-evict-everything semantics, but the
        // hot-only turn must survive because its warm is None.
        let report = evict_from_hot(&conv, &[], &[]);

        assert_eq!(report.count, 1, "only the warm-backed turn is eligible");
        // Hot-only turn unaffected.
        assert!(
            conv.read().turn_sealed_of(timeline, no_warm).is_some(),
            "hot-only turn must remain hot — eviction without warm would lose data"
        );
        // Warm-backed turn evicted.
        assert!(conv.read().turn_sealed_of(timeline, warm_backed).is_none());
    }

    /// A `ColdRecall` install lands the residence as dual-tier (hot
    /// + warm). Subsequent hot eviction confirms warm was actually
    /// installed (otherwise the eviction couldn't preserve any
    /// backup).
    ///
    /// We exercise the substrate-side install directly rather than
    /// driving a real cold-load: `seed_turn` writes `block_end=0`
    /// into the persisted StreamDecl, so `recover_turn_chunks`
    /// returns None on the resulting log. The cold-load round-trip
    /// itself is covered by the candle-nn batched-async test suite;
    /// here we're verifying the install structure does what its
    /// type name says.
    #[test]
    fn cold_recall_install_lands_dual_tier_residence() {
        use crate::substrate::{ColdRecall, PromotionItemKind};
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        // Seed hot, capture hot, evict hot — residence ends with
        // hot = None, warm = None. (We don't install cold here; the
        // test only cares about the install-side behaviour.)
        let idx = seed_turn(&conv, &backings, &device, timeline, 2, 16, 32, 99);
        let residence = conv.read().turn_residence(timeline, idx).unwrap();
        let hot_arc = conv.read().turn_sealed_of(timeline, idx).unwrap();
        let hot_clone: Vec<SealedSequence> = hot_arc.iter().cloned().collect();
        let warm_clone =
            migrate_layers_to_cpu(&backings, &device, &r.copy_stream, &mut r.pinned, &hot_arc);
        drop(hot_arc);
        conv.write().clear_turn_sealed(timeline, idx);
        assert!(conv.read().turn_sealed_of(timeline, idx).is_none());

        // What phase 2b produces: one `ColdRecall` carrying both
        // hot and the fresh CPU-arena warm copy.
        let recall = ColdRecall {
            kind: PromotionItemKind::Turn(TurnKey::new(timeline, idx)),
            residence,
            hot: hot_clone,
            warm: warm_clone,
        };
        conv.write().install_promoted(vec![recall], Vec::new());

        // Both tiers populated. Confirm via the observable behaviour:
        // a subsequent unfiltered hot eviction drops hot because warm
        // exists, leaving the turn cold-marker.
        assert!(
            conv.read().turn_sealed_of(timeline, idx).is_some(),
            "hot installed"
        );
        let evicted = conv.write().evict_hot_except(&[], &[]);
        assert_eq!(
            evicted.count, 1,
            "warm backup present → eviction drops hot for one residence"
        );
        assert!(
            conv.read().turn_sealed_of(timeline, idx).is_none(),
            "post-eviction the turn is warm-only / cold-marker"
        );
    }
}

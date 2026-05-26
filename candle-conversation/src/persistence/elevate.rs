//! Bulk hot-tier promotion — bring a batch of sections + turns into
//! VRAM in one orchestrated pass.
//!
//! Pairs with [`Substrate::evict_hot_with_warm_backup`] (bulk eviction):
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
use super::transfer::load_to_hot;
use crate::projection::{Conversation, SectionId, TurnKey};
use crate::substrate::{
    EvictionReport, PromotionInstall, PromotionItemKind, PromotionPlan, WarmToHotEntry,
};
use candle_nn::kv_cache::SealedSequence;

/// Sum of `SealedChunk.byte_size` across every chunk of every layer
/// in a per-layer `Vec<SealedSequence>` — the per-tier memory cost
/// of holding this turn hot or warm.
fn sealed_total_bytes(seqs: &[SealedSequence]) -> u64 {
    seqs.iter()
        .flat_map(|s| s.chunks.iter())
        .map(|c| c.byte_size as u64)
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
    let plan: PromotionPlan = conversation.read().snapshot_promotion_state(sections, turns);

    let mut report = ElevationReport {
        already_hot: plan.already_hot.len(),
        missing: plan.missing.len(),
        ..Default::default()
    };
    for kind in &plan.missing {
        tracing::warn!("elevate_to_hot: item not found in substrate: {kind:?}");
    }

    let mut installs: Vec<PromotionInstall> = Vec::new();

    // ── Phase 2: cold → hot (NVMe-bound, per-item) ─────────────────────
    //
    // Each cold item does: recover_turn_chunks (or section equivalent)
    // → TurnChunkGrid → load_to_hot. Section cold-load isn't wired up
    // today (see ensure_section_hot in scheduler/mod.rs); cold sections
    // are warned and dropped on the floor.
    let n_layers = backings.len();
    for cold_entry in plan.cold_to_hot {
        match cold_entry.kind {
            PromotionItemKind::Turn(key) => {
                let grid = match conversation.recover_turn_chunks(
                    key.timeline,
                    key.index,
                    n_layers,
                ) {
                    Ok(Some(g)) => g,
                    Ok(None) => {
                        tracing::warn!(
                            "elevate_to_hot: cold-load found no chunks for turn {key:?}"
                        );
                        report.missing += 1;
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("elevate_to_hot: recover_turn_chunks {key:?}: {e}");
                        report.failed += 1;
                        continue;
                    }
                };
                let bytes_for_item: u64 = grid.bytes() as u64;
                match load_to_hot(backings, device, &grid, cold_stager) {
                    Ok(sealed) => {
                        tracing::debug!(
                            target: "candle_conversation::persistence::tier",
                            timeline = key.timeline.raw(),
                            turn = key.index.0,
                            residence = cold_entry.residence.0,
                            bytes = bytes_for_item,
                            "promoted cold → hot (turn)"
                        );
                        installs.push(PromotionInstall {
                            kind: cold_entry.kind,
                            residence: cold_entry.residence,
                            hot: sealed,
                        });
                        report.cold_to_hot += 1;
                        report.bytes_cold_to_hot =
                            report.bytes_cold_to_hot.saturating_add(bytes_for_item);
                    }
                    Err(e) => {
                        tracing::warn!("elevate_to_hot: load_to_hot {key:?}: {e}");
                        report.failed += 1;
                    }
                }
            }
            PromotionItemKind::Section(sid) => {
                tracing::warn!(
                    "elevate_to_hot: cold-load for section {sid:?} not supported; \
                     sections are expected to be pinned at conversation setup"
                );
                report.failed += 1;
            }
        }
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
        let mut hot_per_item: Vec<Vec<candle_nn::kv_cache::SealedSequence>> =
            (0..warm_items.len())
                .map(|_| Vec::with_capacity(n_layers))
                .collect();
        let mut layer_ok = true;
        for layer in 0..n_layers {
            let inputs: Vec<&candle_nn::kv_cache::SealedSequence> = warm_items
                .iter()
                .map(|w| &w.warm[layer])
                .collect();
            match backings[layer].migrate_sealed_to_gpu_batch_async(
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
                                "promoted warm → hot (turn)"
                            );
                        }
                        PromotionItemKind::Section(sid) => {
                            tracing::debug!(
                                target: "candle_conversation::persistence::tier",
                                section = sid.raw(),
                                residence = w.residence.0,
                                bytes = bytes_for_item,
                                "promoted warm → hot (section)"
                            );
                        }
                    }
                    installs.push(PromotionInstall {
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
    if !installs.is_empty() {
        conversation.write().install_promoted_hot(installs);
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
    use crate::token_buffer::TokenBuffer;
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
            let k = Tensor::from_vec(
                data,
                (1, n_kv_head, n_tokens, head_dim),
                &Device::Cpu,
            )
            .unwrap()
            .to_device(device)
            .unwrap();
            let v = k.clone();
            backing.write_contiguous(slot, 0, &k, &v).unwrap();
            backing.set_len(slot, n_tokens);
            sealed_per_layer.push(backing.record_turn(slot, n_tokens).unwrap());
        }
        conv.record_turn(
            timeline,
            Role::User,
            String::new(),
            TokenBuffer::default(),
            n_tokens,
            0,
            0,
            Arc::new(sealed_per_layer),
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
    fn fresh_setup(device: &Device, n_layers: usize) -> (Conversation, Vec<ChunkedKvBacking>, TimelineId) {
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
        let warm = migrate_layers_to_cpu(
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &hot_arc,
        );
        // `hot_arc` is the only outstanding Arc apart from the substrate's
        // — drop it so install_warm + eviction don't see a phantom
        // borrower.
        drop(hot_arc);
        conv.write().install_warm(residence, warm);

        // Drop hot — turn is now warm-only.
        let evicted = conv.write().evict_hot_with_warm_backup();
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
        let warm = migrate_layers_to_cpu(
            &backings,
            &device,
            &r.copy_stream,
            &mut r.pinned,
            &hot_arc,
        );
        drop(hot_arc);
        conv.write().install_warm(residence, warm);

        // Bogus key — missing bucket.
        let missing = TurnKey::new(TimelineId::for_test(999), TurnIndex(42));

        // Evict hot just for turn 1 — install_warm only made it
        // dual-resident; eviction drops hot since warm is present.
        // Note: this also evicts turn 0 because its warm is None, so
        // ... actually no, evict_hot_with_warm_backup only evicts
        // entries where BOTH hot is some AND warm is some. Turn 0
        // has warm=None, so it stays hot.
        let evicted = conv.write().evict_hot_with_warm_backup();
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

        let a = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 100);
        let b = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 200);

        let keep_turns = vec![TurnKey::new(timeline, a), TurnKey::new(timeline, b)];
        let report = evict_from_hot(&conv, &[], &keep_turns);

        assert_eq!(report.count, 0, "every hot residence is in keep set");
        assert_eq!(report.bytes, 0);
        // Both turns still resolve hot.
        assert!(conv.read().turn_sealed_of(timeline, a).is_some());
        assert!(conv.read().turn_sealed_of(timeline, b).is_some());
    }

    /// Empty keep set → behaves like `evict_hot_with_warm_backup`:
    /// every warm-backed hot residence is dropped.
    #[test]
    fn empty_keep_set_evicts_every_warm_backed_residence() {
        let Some(device) = cuda_device_or_skip() else {
            return;
        };
        let (conv, backings, timeline) = fresh_setup(&device, 2);
        let mut r = cuda_resources(&device);

        let a = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 100);
        let b = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 200);

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

        let a = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 100);
        let b = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 200);

        // Keep only A.
        let keep = vec![TurnKey::new(timeline, a)];
        let report = evict_from_hot(&conv, &[], &keep);

        assert_eq!(report.count, 1, "only B should be evicted");
        assert!(report.bytes > 0);
        // A is still hot, B is cold-marker (warm-only).
        assert!(conv.read().turn_sealed_of(timeline, a).is_some(), "A kept");
        assert!(conv.read().turn_sealed_of(timeline, b).is_none(), "B evicted");
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
        let warm_backed = seed_warm_backed_turn(&conv, &backings, &device, timeline,
            &r.copy_stream, &mut r.pinned, 200);

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
}

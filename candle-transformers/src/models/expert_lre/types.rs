//! Shared data types for the expert cache pipeline.
//!
//! These types are used across all submodules — cache bookkeeping, DMA
//! loading, pipeline dispatch, and the public API.

use super::compute::QMatMul;
use crate::models::profile::{ProfileMark, ProfileSnapshot};
use candle::quantized::GgmlDType;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

// ============================================================================
// Pipeline telemetry counters (always-on, minimal cost)
// ============================================================================

/// Lightweight telemetry counters for the expert pipeline.
///
/// Shared between the pipeline thread (writer) and the `ExpertCache` handle
/// (reader) via `Arc<Mutex<_>>`. The pipeline thread is the sole writer of the
/// tallies; a reader that wants them complete for a forward asks through
/// [`super::ExpertCache::expert_stats`], which settles the pipeline first.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Routed experts that were in VRAM when their layer's routing reached the
    /// pipeline thread.
    pub expert_hits: usize,
    /// Routed experts that were not: computed by the expert GEMMs' workers from
    /// pinned memory — `worker_pinned` + `worker_cold`.
    pub expert_misses: usize,
    /// Misses bucketize found in pinned memory (a warm or pad slot).
    pub worker_pinned: usize,
    /// Misses bucketize found cold — waited on until the stager published them.
    pub worker_cold: usize,
    /// Experts evicted from VRAM. A retarget, not a copy — every expert has a
    /// copy in the pack, and usually one in pinned memory.
    pub evictions: usize,
    /// Promotions into VRAM that landed by the copy engine, from a pinned slot
    /// (the speculative ones), and their bytes.
    pub promotions: usize,
    pub promotion_bytes: usize,
    /// Promotions the GEMM workers made: misses whose slices they also wrote
    /// into a VRAM slot from the promotion ring — no link traffic of their own.
    pub worker_promotions: usize,
    /// Cold experts the stager staged for a routed row, of them those copied
    /// from a pageable warm slot rather than read from the pack, the ones it
    /// staged ahead on the predictor's request, and the bytes it moved.
    pub staged_cold: usize,
    pub staged_paged: usize,
    pub staged_speculative: usize,
    pub staged_bytes: usize,
    /// Reader time summed over the stager's reads, ns — `staged_bytes` over
    /// this is the per-reader rate.
    pub stage_read_ns: u64,
    /// Pad slots the stager took back from a staged expert.
    pub pad_evictions: usize,
    /// **Gauge**, not a tally: experts the warm tier holds, of the model's
    /// total. Reported beside the miss counts because the two only make sense
    /// together — a cold count is a verdict on this number.
    pub warm_slots: usize,
    /// **Gauge**: of `warm_slots`, those in pageable memory beyond the
    /// page-lock ceiling — not device-readable, so staged like a pack record.
    pub warm_paged_slots: usize,
    /// **Gauge**: pad slots.
    pub pad_slots: usize,
    /// Experts in the model, so `warm_slots` reads as a fraction.
    pub total_experts: usize,
    /// **Gauge**: MoE layers in the model. Published beside `total_experts`
    /// because the rate model needs the two apart: a decode step costs
    /// `moe_layers` layers, and a layer's copy is capped at `total_experts /
    /// moe_layers` experts. Their product alone cannot say either.
    pub moe_layers: usize,
    /// Speculative promotions (predictor and wide-row lookahead) that landed.
    pub prefetch_promotions: usize,
    /// Speculatively promoted experts that the layer actually routed to.
    /// Numerator of prediction precision.
    pub predicted_hits: usize,
    /// Total speculatively promoted experts evaluated against actual routing.
    /// Denominator of prediction precision.
    pub predicted_total: usize,
    /// Speculative promotions still in flight when their target layer's
    /// routing arrived. The latency-bound signal for the dynamic load-ahead
    /// controller: late > 0 with bandwidth slack means the prefetcher should
    /// issue earlier (deepen N); late ≈ 0 with falling precision means it
    /// should shallow back.
    pub late_loads: usize,
    /// **Gauge**: the dynamic load-ahead depth `N` as of the last pass
    /// boundary — how many layers ahead the speculative prefetcher currently
    /// issues for.
    pub prefetch_depth: usize,
    /// Routed layers the pipeline thread has processed — one per MoE layer per
    /// forward.
    pub routed_messages: usize,
    /// Invocations the device had begun past the one being served, summed over
    /// routed layers: over `routed_messages`, how far the pipeline thread trails
    /// the GPU. Its ring stock reaches only the invocations that begin after it.
    pub pipeline_lag: u64,
    /// Ring slots bucketize gave remote experts, as collected from its log.
    pub ring_taken: usize,
    /// Of those, slots that still held a resident expert — a lazy victim the
    /// claim evicted on the device.
    pub victims_claimed: usize,
    /// Lazy victims bucketize passed over because their own launch routed them.
    pub victims_skipped: usize,
    /// Misses that took no ring slot and were not otherwise being promoted:
    /// computed from scratch, claimed by the next launch that routes them.
    pub ring_unslotted: usize,
    /// Of those, the ones a decode row routed — a decode row co-batched into a
    /// sweep defers its claims to the next narrow launch.
    pub decode_unslotted: usize,
    /// **Live** VRAM bytes held by resident expert slots — `occupied_slots ×
    /// slot_size`. Unlike the counters above (monotonic tallies), this is a
    /// gauge: it rises as experts load into VRAM and falls as they stream out
    /// to pinned RAM under pressure, so the whole-card VRAM decomposition can
    /// show the model's time-varying resident-expert footprint. Seeded at cache
    /// construction and refreshed by the pipeline thread each classify.
    pub resident_vram_bytes: usize,
    /// Gauge: span bytes the weight zone could concede to the KV side on demand
    /// — `(capacity − floor) × slot_bytes`. The elastic boundary already cedes
    /// this ground to stuck KV claims (`request_kv_ground`); publishing it lets
    /// the prefill width cap count it as admissible instead of pre-slicing the
    /// fleet at whatever happens to be standing free. Refreshed by the pipeline
    /// thread each classify, like `resident_vram_bytes`.
    pub zone_cedeable_bytes: usize,
    /// **Gauge**: the weight zone as it stands, and the range it may move in —
    /// `capacity`, `min_capacity` and `limit`, each in bytes.
    ///
    /// Published together because they are only meaningful together: the wave
    /// rate planner judges an admission on the residency it dislodges, which
    /// needs where the zone is *and* how far it can go. The KV side cannot
    /// derive them — `request_kv_ground` reports only what it was conceded after
    /// the fact, and the zone regrows, so a figure inferred from concessions
    /// reads the same whether or not the ground came back.
    pub zone_bytes: usize,
    pub zone_min_bytes: usize,
    pub zone_max_bytes: usize,
    /// **Gauge**: bytes one expert slot occupies — the unit every figure above
    /// is a multiple of, and the grain the rate model prices a routed expert in.
    pub expert_slot_bytes: usize,
}

impl PipelineStats {
    /// Create a new shared stats handle.
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }

    /// Snapshot the current counters (clone under lock).
    pub fn snapshot(shared: &Arc<Mutex<Self>>) -> Self {
        shared
            .lock()
            .map_or_else(|_| Self::default(), |s| s.clone())
    }

    /// Reset the per-interval tallies. The **gauges** survive it: they describe
    /// the cache's shape rather than what it did since the last reset.
    ///
    /// **The tallies are cleared by name rather than the gauges restored around
    /// a `default()`.** Both spellings zero the same fields today, but they fail
    /// in opposite directions when this struct grows: restoring meant every new
    /// gauge was silently zeroed on the next reset unless someone remembered to
    /// add it to the list, and a gauge that reads zero is indistinguishable from
    /// a cache that holds nothing. A new *tally* forgotten here merely
    /// accumulates across intervals, which shows up as a number that only ever
    /// rises — visible, rather than invisible.
    pub fn reset(shared: &Arc<Mutex<Self>>) {
        if let Ok(mut s) = shared.lock() {
            s.expert_hits = 0;
            s.expert_misses = 0;
            s.worker_pinned = 0;
            s.worker_cold = 0;
            s.evictions = 0;
            s.promotions = 0;
            s.promotion_bytes = 0;
            s.worker_promotions = 0;
            s.staged_cold = 0;
            s.staged_paged = 0;
            s.staged_speculative = 0;
            s.staged_bytes = 0;
            s.stage_read_ns = 0;
            s.pad_evictions = 0;
            s.prefetch_promotions = 0;
            s.predicted_hits = 0;
            s.predicted_total = 0;
            s.late_loads = 0;
            s.routed_messages = 0;
            s.pipeline_lag = 0;
            s.ring_taken = 0;
            s.victims_claimed = 0;
            s.victims_skipped = 0;
            s.ring_unslotted = 0;
            s.decode_unslotted = 0;
        }
    }

    /// Hit rate as a percentage (0.0–100.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.expert_hits + self.expert_misses;
        if total == 0 {
            100.0
        } else {
            (self.expert_hits as f64 / total as f64) * 100.0
        }
    }

    /// Prediction precision as a percentage (0.0–100.0): of all
    /// speculatively loaded experts, the fraction the layer actually routed
    /// to.  This isolates the transition-matrix predictor's quality from the
    /// overall cache hit rate.  Returns 0.0 when no speculative loads occurred.
    pub fn prediction_precision(&self) -> f64 {
        if self.predicted_total == 0 {
            0.0
        } else {
            (self.predicted_hits as f64 / self.predicted_total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod stats_tests {
    use super::PipelineStats;

    /// A reset clears the tallies and keeps the gauges — including the warm
    /// tier's pageable share, which a report reads after every config.
    #[test]
    fn a_reset_keeps_the_gauges() {
        let shared = PipelineStats::new_shared();
        {
            let mut s = shared.lock().unwrap();
            s.warm_slots = 13_508;
            s.warm_paged_slots = 2_138;
            s.pad_slots = 256;
            s.worker_cold = 11_225;
            s.promotion_bytes = 4096;
            s.stage_read_ns = 77;
        }
        PipelineStats::reset(&shared);
        let s = PipelineStats::snapshot(&shared);
        assert_eq!((s.warm_slots, s.warm_paged_slots, s.pad_slots), (13_508, 2_138, 256));
        assert_eq!((s.worker_cold, s.promotion_bytes, s.stage_read_ns), (0, 0, 0));
    }
}

// ============================================================================
// Mmap reference
// ============================================================================

/// Byte-range reference into the mmap for one expert's projection matrices.
///
/// Stores offsets, lengths, shapes, and dtypes for the three projections
/// (gate, up, down) so they can be loaded from the mmap on demand.
#[derive(Debug, Clone)]
pub struct MmapExpertRef {
    pub gate_offset: usize,
    pub gate_len: usize,
    pub up_offset: usize,
    pub up_len: usize,
    pub down_offset: usize,
    pub down_len: usize,
    pub gate_shape: Vec<usize>,
    pub up_shape: Vec<usize>,
    pub down_shape: Vec<usize>,
    pub gate_dtype: GgmlDType,
    pub up_dtype: GgmlDType,
    pub down_dtype: GgmlDType,
}

// ============================================================================
// Expert slot (VRAM resident)
// ============================================================================

/// A single VRAM slot holding one expert's three projection matrices.
///
/// The views over a weight-zone slot the pipeline thread keeps for its
/// bookkeeping. The expert GEMMs do not read them: they read the slot's
/// addresses out of bucketize's snapshot of the live table (`live_table`),
/// which the pipeline thread points at the slot once its bytes have landed.
///
/// **Sole ownership**: slots are owned directly by the pipeline thread.
/// No `Arc` wrapping — never cloned, never shared across threads.
pub struct ExpertSlot {
    pub gate_proj: QMatMul,
    pub up_proj: QMatMul,
    pub down_proj: QMatMul,
}

// ============================================================================
// Pipeline messages
// ============================================================================

/// One MoE layer's routing, handed to the pipeline thread.
///
/// Sent by the forward thread after it has enqueued the layer's bucketize. The
/// forward thread does not wait for an answer: nothing the GPU computes for the
/// layer depends on the pipeline thread, which keeps VRAM residency in step
/// with routing off the critical path.
#[cfg(feature = "cuda")]
pub struct RoutedLayer {
    /// The MoE row (layer index in the live table).
    pub row: usize,
    /// The forward thread's pass when this layer was routed — a pass is a run
    /// of invocations with strictly increasing row.
    pub pass: u64,
    /// The summary ring slot holding this layer's routing summary.
    pub slot: usize,
    /// The word bucketize stores last in the slot. Once the slot reads it,
    /// every kernel enqueued before this layer's bucketize — including every
    /// expert GEMM of an earlier layer — has completed, and the summary is
    /// readable.
    pub summary_word: u32,
    /// The invocation's ticket (`reclaim`): observed when the word is.
    pub ticket: u64,
    /// A prompt-prefill launch (`dispatch::PREFILL_LAUNCH_TOKENS`): its misses
    /// are pulled by enough workers to saturate the link, so a speculative copy
    /// for the next row competes with them.
    pub prefill_width: bool,
    /// Timestamp captured just before `send`, so the worker can measure the
    /// hand-off. Zero-sized off-`profile`.
    pub submitted_at: ProfileMark,
}

/// Message sent to the pipeline thread.
pub enum PipelineMessage {
    /// A layer's routing: score it, promote, prefetch.
    #[cfg(feature = "cuda")]
    Routed(RoutedLayer),
    /// Answer once every message sent before this one has been processed. What
    /// a reader of the counters, or of anything else the pipeline thread
    /// writes, sends first to read them complete.
    Settle {
        response_tx: mpsc::SyncSender<()>,
    },
    /// Snapshot and reset the pipeline thread’s profile accumulator.
    SnapshotProfile {
        /// Oneshot channel for returning the snapshot.
        response_tx: mpsc::SyncSender<ProfileSnapshot>,
    },
    /// Move the boundary: sell `regions` of weight-side ground to the KV side,
    /// or — with `regions` zero — take back whatever the KV side is holding
    /// spare.
    ///
    /// The eviction happens here, on the pipeline thread, where the cache
    /// state lives, while the *quantity* comes from the caller that knows it.
    /// Whether it may happen at all is decided at entry
    /// (`PipelineState::renegotiate_if_quiet`): never under a live wave, and
    /// never with an invocation the forward thread has begun still unserved.
    ///
    /// Answers with the bytes conceded — zero if the boundary could not or may
    /// not move.
    RenegotiateBoundary {
        /// Regions the KV side is asking for.
        regions: usize,
        /// Oneshot channel for the bytes handed to the KV side.
        response_tx: mpsc::SyncSender<u64>,
    },
}

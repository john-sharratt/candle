//! Shared data types for the expert cache pipeline.
//!
//! These types are used across all submodules — cache bookkeeping, the
//! pipeline thread's messages and telemetry, and the public API.

use super::compute::QMatMul;
use super::read_latency::ReadLatency;
use super::transition::HOPS as LOOK_AHEAD_HOPS;
use crate::models::profile::{ProfileMark, ProfileSnapshot};
use candle::quantized::GgmlDType;
use candle::LiveTensor;
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
    /// Promotions the GEMM workers made: misses whose slices they also wrote
    /// into a VRAM slot from the promotion ring — no link traffic of their own —
    /// and the read-ahead claims among the promotions they landed.
    pub worker_promotions: usize,
    /// Cold experts the stager staged for a routed row, of them those copied
    /// from a pageable warm slot rather than read from the pack, the ones it
    /// staged ahead on the predictor's request, and the bytes it moved.
    pub staged_cold: usize,
    pub staged_paged: usize,
    pub staged_speculative: usize,
    /// Cold experts the pipeline thread's predictions asked the stager to
    /// stage ahead. Over this, `staged_speculative` is how many reads the
    /// stager issued for them; the rest were readable by the time their turn
    /// came, or found no slot worth taking.
    pub stage_requests: usize,
    /// Of the experts staged ahead, those their row then routed — over
    /// `staged_speculative`, the stager's lookahead precision.
    pub staged_ahead_routed: usize,
    /// Of the experts staged ahead, those read-ahead listed from the pad as
    /// they landed — in time for a launch to read them ahead of their row.
    pub staged_listed: usize,
    pub staged_bytes: usize,
    /// Reader time summed over the stager's reads, ns — `staged_bytes` over
    /// this is the per-reader rate.
    pub stage_read_ns: u64,
    /// The stager's reads from the pack and from pageable warm slots, each
    /// with its slowest read and its count of slow ones — which of the two a
    /// stalled cold wait was waiting on.
    pub pack_reads: ReadLatency,
    pub paged_reads: ReadLatency,
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
    /// Read-ahead claims: ring slots bucketize gave a predicted warm expert of a
    /// later row, copied and published by its gate launch's workers — and their
    /// bytes, the link time read-ahead spent.
    pub ahead_claims: usize,
    pub ahead_bytes: usize,
    /// Of the read-ahead claims, those copied from the pad — staged ahead off
    /// the drive first, then read ahead into VRAM.
    pub ahead_pad_claims: usize,
    /// The router look-ahead's predictions by hop (`votes`): experts predicted
    /// for the row `h + 1` past the one that voted, judged against that row's
    /// routing, and of them the ones it routed.
    pub lookahead_judged: [usize; LOOK_AHEAD_HOPS],
    pub lookahead_hits: [usize; LOOK_AHEAD_HOPS],
    /// Experts read ahead that their row then routed. Numerator of prediction
    /// precision.
    pub predicted_hits: usize,
    /// Experts read ahead, judged against their row's routing. Denominator of
    /// prediction precision.
    pub predicted_total: usize,
    /// **Gauge**: how many rows ahead read-ahead looks — a launch at `r` reads
    /// for `r + 2 ..= r + prefetch_depth`.
    pub prefetch_depth: usize,
    /// **Gauge**: the read-ahead window as of the last pass boundary — slot
    /// images the link moves in one layer, at the read-ahead share.
    pub ahead_window: usize,
    /// **Gauge**: pad slots read-ahead's listings hold pinned — the pad-backed
    /// experts it may read ahead, and those whose readers have yet to finish.
    pub ahead_pinned: usize,
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
    /// gauge: it rises as experts are promoted into VRAM and falls as they are
    /// evicted or the zone concedes ground, so the whole-card VRAM decomposition
    /// can show the model's time-varying resident-expert footprint. Seeded at
    /// cache construction and refreshed by the pipeline thread each routed layer.
    pub resident_vram_bytes: usize,
    /// Gauge: span bytes the weight zone could concede to the KV side on demand
    /// — `(capacity − floor) × slot_bytes`. The elastic boundary already cedes
    /// this ground to stuck KV claims (`request_kv_ground`); publishing it lets
    /// the prefill width cap count it as admissible instead of pre-slicing the
    /// fleet at whatever happens to be standing free. Refreshed by the pipeline
    /// thread each routed layer, like `resident_vram_bytes`.
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
    /// The residency references: LRU's and Belady's hit rates, in percent,
    /// over the routing of the interval these tallies cover, at the zone's
    /// capacity (`ExpertCache::hit_references`). Not kept by the pipeline
    /// thread — a replay of the whole interval, too costly per layer — but
    /// filled by a reader that asks for them beside the tallies; 0 otherwise.
    pub hit_lru: f64,
    pub hit_ceiling: f64,
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
            s.worker_promotions = 0;
            s.staged_cold = 0;
            s.staged_paged = 0;
            s.staged_speculative = 0;
            s.stage_requests = 0;
            s.staged_ahead_routed = 0;
            s.staged_listed = 0;
            s.staged_bytes = 0;
            s.stage_read_ns = 0;
            s.pack_reads = ReadLatency::default();
            s.paged_reads = ReadLatency::default();
            s.pad_evictions = 0;
            s.ahead_claims = 0;
            s.ahead_bytes = 0;
            s.ahead_pad_claims = 0;
            s.predicted_hits = 0;
            s.predicted_total = 0;
            s.lookahead_judged = [0; LOOK_AHEAD_HOPS];
            s.lookahead_hits = [0; LOOK_AHEAD_HOPS];
            s.routed_messages = 0;
            s.pipeline_lag = 0;
            s.ring_taken = 0;
            s.victims_claimed = 0;
            s.victims_skipped = 0;
            s.ring_unslotted = 0;
            s.decode_unslotted = 0;
            s.hit_lru = 0.0;
            s.hit_ceiling = 0.0;
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

    /// The router look-ahead's precision at hop `h` (1-based), in percent; 0
    /// when nothing was judged there.
    pub fn lookahead_precision(&self, h: usize) -> f64 {
        let judged = self.lookahead_judged[h - 1];
        if judged == 0 {
            0.0
        } else {
            100.0 * self.lookahead_hits[h - 1] as f64 / judged as f64
        }
    }

    /// Prediction precision as a percentage (0.0–100.0): of the experts read
    /// ahead, the fraction their row actually routed to. This isolates the
    /// predictor's quality from the overall cache hit rate. Returns 0.0 when
    /// nothing was read ahead.
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
    use super::{PipelineStats, ReadLatency};

    /// A reset clears the tallies and keeps the gauges — including the warm
    /// tier's pageable share and the read-ahead window, which a report reads
    /// after every config.
    #[test]
    fn a_reset_keeps_the_gauges() {
        let shared = PipelineStats::new_shared();
        {
            let mut s = shared.lock().unwrap();
            s.warm_slots = 13_508;
            s.warm_paged_slots = 2_138;
            s.pad_slots = 256;
            s.ahead_window = 14;
            s.prefetch_depth = 4;
            s.worker_cold = 11_225;
            s.ahead_bytes = 4096;
            s.ahead_claims = 3;
            s.ahead_pad_claims = 2;
            s.staged_listed = 9;
            s.stage_requests = 4;
            s.lookahead_judged = [10, 8, 0, 0, 4];
            s.lookahead_hits = [7, 2, 0, 0, 1];
            assert_eq!(
                (1..=5)
                    .map(|h| s.lookahead_precision(h))
                    .collect::<Vec<_>>(),
                vec![70.0, 25.0, 0.0, 0.0, 25.0]
            );
            s.stage_read_ns = 77;
            s.pack_reads.record(250_000_000);
            s.paged_reads.record(1_000);
        }
        PipelineStats::reset(&shared);
        let s = PipelineStats::snapshot(&shared);
        assert_eq!(
            (s.warm_slots, s.warm_paged_slots, s.pad_slots),
            (13_508, 2_138, 256)
        );
        assert_eq!((s.ahead_window, s.prefetch_depth), (14, 4));
        assert_eq!(
            (
                s.worker_cold,
                s.ahead_bytes,
                s.ahead_claims,
                s.stage_read_ns
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            (s.ahead_pad_claims, s.staged_listed, s.stage_requests),
            (0, 0, 0)
        );
        assert_eq!((s.lookahead_judged, s.lookahead_hits), ([0; 5], [0; 5]));
        assert_eq!(
            (s.pack_reads, s.paged_reads),
            (ReadLatency::default(), ReadLatency::default())
        );
    }
}

// ============================================================================
// Router look-ahead
// ============================================================================

/// Some of the next rows' routers applied to this layer's input: `rows` is a
/// stacked projection, `[tokens, width]` f32, holding `hops` routers of
/// `n_experts` columns each from column `first_col`, for hops `first_hop ..
/// first_hop + hops` — hop `h`'s router is row `row + h`'s. The expert forward
/// turns them into votes for the experts those rows will route (`votes`).
#[derive(Clone, Copy)]
pub struct LookAhead<'a> {
    pub rows: &'a LiveTensor<'a>,
    pub first_col: usize,
    pub first_hop: usize,
    pub hops: usize,
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
    /// Hops of router look-ahead votes the invocation writes into its vote
    /// ring slot (`votes`); 0 when it writes none.
    pub voted_hops: usize,
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
    Settle { response_tx: mpsc::SyncSender<()> },
    /// A forward faulted (`fault`): drop the demand promotions still to land,
    /// whose slots its workers may have left part-written. Answers when done.
    DropDemandPromotions { response_tx: mpsc::SyncSender<()> },
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

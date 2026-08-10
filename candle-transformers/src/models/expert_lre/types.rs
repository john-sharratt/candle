//! Shared data types for the expert cache pipeline.
//!
//! These types are used across all submodules — cache bookkeeping, DMA
//! loading, pipeline dispatch, and the public API.

use super::compute::QMatMul;
use crate::models::profile::{ProfileMark, ProfileSnapshot};
use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::quantized::GgmlDType;
use candle::{DType, Device, Result, Tensor};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

// ============================================================================
// Pipeline telemetry counters (always-on, minimal cost)
// ============================================================================

/// Lightweight telemetry counters for the expert pipeline.
///
/// Shared between the pipeline thread (writer) and the `ExpertCache` handle
/// (reader) via `Arc<Mutex<_>>`.  The mutex is uncontended on the write
/// path (pipeline thread is sole writer); reads happen only between test
/// configs or at shutdown.
///
/// All fields are plain `usize` — atomic increments inside the already
/// exclusive pipeline thread (`&mut self`).  The `Mutex` only serialises
/// the cross-thread snapshot/reset from the handle.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Expert cache hits (already in VRAM).
    pub expert_hits: usize,
    /// Expert cache misses (loaded from pinned or mmap).
    pub expert_misses: usize,
    /// Experts evicted from VRAM (drip + end-of-pass). A drop, not a copy —
    /// the cold tier already holds every expert, so there is no matching
    /// device-to-host transfer to count.
    pub evictions: usize,
    /// H2D DMA transfers into VRAM, from either host tier.
    pub dma_loads: usize,
    /// Loads served by the warm tier (H2D from pinned host memory).
    pub warm_loads: usize,
    /// Loads that missed both resident tiers and read the pack file.
    pub cold_loads: usize,
    /// **Gauge**, not a tally: experts the warm tier holds, of the model's
    /// total. Reported beside the load counts because the two only make sense
    /// together — a cold-load count is a verdict on this number, and reading
    /// them in different places is how a warm tier sized at a third of the model
    /// went unnoticed while it sent two thirds of every miss to disk.
    pub warm_slots: usize,
    /// Experts in the model, so `warm_slots` reads as a fraction.
    pub total_experts: usize,
    /// Speculative prefetch loads that landed in VRAM.
    pub prefetch_loads: usize,
    /// Hint-driven speculative loads.
    pub hint_loads: usize,
    /// Speculatively loaded experts that the layer actually routed to.
    /// Numerator of prediction precision.
    pub predicted_hits: usize,
    /// Total speculatively loaded experts evaluated against actual routing.
    /// Denominator of prediction precision.
    pub predicted_total: usize,
    /// Number of times `fence_wait` blocked (non-zero wait).
    pub fence_stalls: usize,
    /// Total MoE work requests processed.
    pub work_requests: usize,
    /// **Live** VRAM bytes held by resident expert slots — `occupied_slots ×
    /// slot_size`. Unlike the counters above (monotonic tallies), this is a
    /// gauge: it rises as experts load into VRAM and falls as they stream out
    /// to pinned RAM under pressure, so the whole-card VRAM decomposition can
    /// show the model's time-varying resident-expert footprint. Seeded at cache
    /// construction and refreshed by the pipeline thread each classify.
    pub resident_vram_bytes: usize,
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

    /// Reset the per-interval tallies. The three **gauges** —
    /// `resident_vram_bytes`, `warm_slots`, `total_experts` — survive it: they
    /// describe the cache's shape rather than what it did since the last reset,
    /// and an inline-mode cache (which never re-seeds them via a classify) would
    /// otherwise read 0 forever.
    pub fn reset(shared: &Arc<Mutex<Self>>) {
        if let Ok(mut s) = shared.lock() {
            let gauges = (s.resident_vram_bytes, s.warm_slots, s.total_experts);
            *s = Self::default();
            (s.resident_vram_bytes, s.warm_slots, s.total_experts) = gauges;
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
/// Created on-demand with the correct dtype when an expert is loaded.
/// Stores pre-built `QMatMul` wrappers to avoid reconstruction per dispatch.
///
/// **Sole ownership**: slots are owned directly by the pipeline thread
/// (threaded mode) or the Mutex-protected inner (inline mode).
/// No `Arc` wrapping — never cloned, never shared across threads.
pub struct ExpertSlot {
    pub gate_proj: QMatMul,
    pub up_proj: QMatMul,
    pub down_proj: QMatMul,
}

// ============================================================================
// DMA fence
// ============================================================================

/// Opaque fence representing in-flight DMA work on the copy stream.
///
/// Returned by internal classify_and_load — waited on before computing
/// loaded experts.  On non-CUDA or when no copy stream exists this is a no-op.
pub struct CopyBatchFence {
    #[cfg(feature = "cuda")]
    pub(crate) event: Option<cudarc::driver::CudaEvent>,
}

impl CopyBatchFence {
    /// Create a no-op fence (nothing to wait on).
    pub fn noop() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            event: None,
        }
    }

    /// Wait for the fence on the given device's main stream.
    /// This is a GPU-side `cudaStreamWaitEvent` — the CPU does not block.
    pub(crate) fn wait(&self, _device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(ref event) = self.event {
            if let Device::Cuda(cuda_dev) = _device {
                cuda_dev
                    .cuda_stream()
                    .wait(event)
                    .map_err(candle::Error::wrap)?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Classification result
// ============================================================================

/// Result of expert classification: which are hits, which were loaded.
///
/// Used internally by both threaded and inline paths.
pub struct ClassifiedExperts {
    /// Experts that were already resident (cache hits).
    /// Each entry: (expert_idx, slot_idx).
    pub hits: Vec<(usize, usize)>,
    /// Experts that were loaded via DMA (cache misses).
    /// Each entry: (expert_idx, slot_idx).
    pub loaded: Vec<(usize, usize)>,
    /// DMA fence — wait on this before computing loaded experts.
    pub fence: CopyBatchFence,
}

// ============================================================================
// Work request / response for the pipeline thread
// ============================================================================

/// Work item submitted to the pipeline thread.
///
/// Contains everything the thread needs to perform the full MoE dispatch:
/// routing assignments, input tensor, and routing weights.  The thread
/// does classify → DMA → compute → return output.
/// The MoE input activation `[num_tokens, hidden_dim]`, threaded through the pipeline. `Q8` is the
/// B3 int8 path — the ln2-fused q8a1024 router input that the experts byte-gather directly (no
/// gather-then-quantize). It's cuda-only (the operand holds a `CudaSlice`); the non-CUDA pipeline
/// only ever sees `Float`. Defined here (not in candle-core) so the int8 arm can be cfg-gated
/// without dragging the cuda-only `Q8a128Operand` into the shared, non-cuda-gated pipeline structs.
pub enum MoeInput {
    Float(Tensor),
    #[cfg(feature = "cuda")]
    Q8(candle::quantized::cuda::Q8a128Operand<'static>),
}

impl MoeInput {
    /// `(num_tokens, hidden)` of the activation.
    pub fn shape(&self) -> Result<(usize, usize)> {
        match self {
            MoeInput::Float(t) => t.dims2(),
            #[cfg(feature = "cuda")]
            MoeInput::Q8(op) => Ok((op.rows, op.cols)),
        }
    }
}

pub struct MoeWorkRequest {
    /// Which MoE layer (0..num_moe_layers).
    pub moe_layer_idx: usize,
    /// Unique expert IDs selected by the router (sorted, deduplicated).
    pub expert_ids: Vec<usize>,
    /// Input hidden states `[num_tokens, hidden_dim]` — `Float` (Off) or the q8a1024 router
    /// input (int8). The experts gather from it per expert.
    pub input: MoeInput,
    /// Compute dtype of the expert output `ys` (a q8a1024 operand carries no dtype).
    pub out_dtype: DType,
    /// Flattened routing weights `[num_tokens * k]` (GPU tensor, F32).
    pub weights_flat: Tensor,
    /// Flat assignment array sorted by expert ID.
    /// Each entry: `(expert_id, token_idx, flat_weight_idx)`.
    pub assignments: Vec<(u32, u32, u32)>,
    /// The wave generation the submitting layer has open, if any.
    ///
    /// A [`WaveTicket`] is a `Copy` coordinate rather than a borrow, which is
    /// the whole reason it can be here: the expert chain runs on the pipeline
    /// thread, and no `&WaveGeneration` could cross this channel. The forward
    /// thread blocks on the response for the entire request (`submit_moe_work`
    /// sends and immediately `recv`s), so the generation is open throughout, and
    /// both threads issue on the same stream — so the arena's stream-ordered
    /// reclaim still holds. A ticket from a closed generation resolves to
    /// nothing, so the worst case is a pool allocation, never a stale range.
    pub wave: Option<WaveTicket>,
    /// Timestamp captured by the forward thread just before `send` — lets the worker measure the
    /// inbound handoff latency (`submit_inbound` = pickup − submit). Zero-sized off-`profile`.
    pub submitted_at: ProfileMark,
    /// Channel for the result + the worker's completion timestamp (`worker_done_at`), so the
    /// forward thread can measure the outbound handoff latency (`submit_outbound` = recv − done).
    pub response_tx: mpsc::SyncSender<(Result<Tensor>, ProfileMark)>,
}

// ============================================================================
// Pipeline message (Work or Hint)
// ============================================================================

/// Message sent to the pipeline thread: either a full work request or a
/// speculative prediction hint.
///
/// Hints are sent by the forward thread while the async routing DtoH is
/// in-flight, allowing the pipeline thread to start DMA for predicted
/// experts before the full work request arrives.
pub enum PipelineMessage {
    /// Full MoE dispatch: classify → DMA → compute → return.
    Work(MoeWorkRequest),
    /// Speculative prediction hint: start DMA for predicted experts.
    Hint {
        /// Which MoE layer the hint predicts for (typically current + 0,
        /// since the hint is sent before routing indices are available).
        layer_idx: usize,
        /// Expert IDs from the *previous* layer — used with the transition
        /// matrix to predict which experts this layer will need.
        prev_expert_ids: Vec<usize>,
    },
    /// Snapshot and reset the pipeline thread’s profile accumulator.
    SnapshotProfile {
        /// Oneshot channel for returning the snapshot.
        response_tx: mpsc::SyncSender<ProfileSnapshot>,
    },
}

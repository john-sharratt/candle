//! Shared data types for the expert cache pipeline.
//!
//! These types are used across all submodules — cache bookkeeping, DMA
//! loading, pipeline dispatch, and the public API.

use super::compute::QMatMul;
use crate::models::profile::ProfileSnapshot;
use candle::quantized::GgmlDType;
use candle::{Device, Result, Tensor};
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
    /// Experts evicted from VRAM (drip + end-of-pass).
    pub evictions: usize,
    /// H2D DMA transfers (pinned → VRAM).
    pub dma_loads: usize,
    /// D2H DMA transfers (VRAM → pinned).
    pub dma_evicts: usize,
    /// Speculative prefetch loads that landed in VRAM.
    pub prefetch_loads: usize,
    /// Hint-driven speculative loads.
    pub hint_loads: usize,
    /// Number of times `fence_wait` blocked (non-zero wait).
    pub fence_stalls: usize,
    /// Total MoE work requests processed.
    pub work_requests: usize,
}

impl PipelineStats {
    /// Create a new shared stats handle.
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }

    /// Snapshot the current counters (clone under lock).
    pub fn snapshot(shared: &Arc<Mutex<Self>>) -> Self {
        shared.lock().map_or_else(|_| Self::default(), |s| s.clone())
    }

    /// Reset all counters to zero.
    pub fn reset(shared: &Arc<Mutex<Self>>) {
        if let Ok(mut s) = shared.lock() {
            *s = Self::default();
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
pub struct MoeWorkRequest {
    /// Which MoE layer (0..num_moe_layers).
    pub moe_layer_idx: usize,
    /// Unique expert IDs selected by the router (sorted, deduplicated).
    pub expert_ids: Vec<usize>,
    /// Input hidden states `[num_tokens, hidden_dim]` (GPU tensor).
    pub xs: Tensor,
    /// Flattened routing weights `[num_tokens * k]` (GPU tensor, F32).
    pub weights_flat: Tensor,
    /// Flat assignment array sorted by expert ID.
    /// Each entry: `(expert_id, token_idx, flat_weight_idx)`.
    pub assignments: Vec<(u32, u32, u32)>,
    /// Channel to send the result back to the caller.
    pub response_tx: mpsc::SyncSender<Result<Tensor>>,
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

//! Expert cache with background pipeline thread and DMA / compute overlap.
//!
//! Provides the core infrastructure for MoE (Mixture-of-Experts) models that
//! offload expert weights to host memory (mmap) and load them on demand into
//! a fixed-size VRAM pool managed with score-based eviction.
//!
//! ## Architecture
//!
//! This implements the pipeline described in `docs/expert_pipeline_dataflow.md`
//!
//! - **Background pipeline thread**: a dedicated thread owns all mutable cache
//!   state (`ExpertCacheInner`) with `&mut self` — no Mutex on the hot path.
//!   Callers submit work via an MPSC channel and receive results via a oneshot.
//!   The thread runs the full classify → DMA → compute loop.
//!
//! - **Sole ownership**: expert slots are owned directly by the pipeline thread.
//!   No `Arc<ExpertSlot>`, no `RwLock`, no atomic reference counting.
//!
//! - **Score-based eviction**: each expert carries a lightly-decayed access
//!   frequency (hit: +1.0, prediction hit: +0.3, end-of-pass decay: ×0.85).
//!   Eviction selects the lowest-scored slot (frequency × position factor) —
//!   an O(n) scan over contiguous memory, roughly 1 μs.
//!
//! - **Two-phase dispatch**: the pipeline thread partitions routed experts
//!   into cache hits (WARM/READY) and misses (COLD→LOADING), runs hit compute
//!   concurrently with miss DMA, then runs miss compute after a single
//!   fence wait — all on the pipeline thread with `&mut self`.
//!
//! - **Flat sorted assignment array**: token→expert mappings are sorted by
//!   expert ID and sliced via binary search.  No per-expert HashMap or Vec.
//!
//! ## Eviction policy
//!
//! The cache uses a four-part eviction policy designed to eliminate eviction
//! cascades and maximise cache hit rates across forward passes:
//!
//! ### 1. End-of-pass batch eviction (proactive headroom)
//!
//! After the last MoE layer completes, the bottom 5% of occupied slots by
//! usage timestamp are evicted and returned to the free pool.  This creates
//! ~140 free slots (on a 2,805 slot budget) available for the next pass.
//! Early layers in the next pass find free slots instead of triggering
//! forced evictions, which eliminates the scan + evict overhead at the
//! layers where it matters most (layers with no compute overlap window).
//!
//! ### 2. Layer-aware forced eviction (for real cache misses)
//!
//! When a cache miss occurs and no free slots are available, eviction
//! follows a two-tier priority:
//!
//!   - **Behind-layer bias**: prefer evicting experts from layers that
//!     have already executed this pass (layer ≥ `PINNED_LAYERS`, layer <
//!     current layer).  Among those, pick the highest layer (furthest
//!     from reuse in the next pass), with lowest score as tie-breaker.
//!   - **Global score fallback**: if no behind-layer candidate exists,
//!     fall back to the globally lowest-scored slot, still
//!     respecting the pinning constraint.
//!
//! This ensures evicted experts are never needed later in the *same*
//! pass, preventing the eviction cascade problem where one eviction
//! triggers 3–5 downstream misses.
//!
//! ### 3. Early-layer pinning
//!
//! Experts in the first [`PINNED_LAYERS`] MoE layers (default: 3) are
//! never evicted.  These layers run first every pass with zero compute
//! to overlap with DMA — evicting them guarantees cold misses with
//! maximum stall.  Pinning costs ~24 slots (top-8 × 3 layers), <1% of
//! a typical slot budget.
//!
//! ### 4. Free-slot-only speculative prefetch
//!
//! Speculative prefetch is **structurally incapable** of causing eviction.
//! It only loads into free slots from the pool.  If no free slots exist,
//! prefetch is silently skipped.  Free headroom is created by the
//! end-of-pass batch eviction (point 1).  This guarantees that
//! mispredicted prefetches can never displace experts needed by real
//! cache misses.
//!
//! ### How the four pieces interact across a pass
//!
//! ```text
//! End of pass N:  batch-evict 5% → ~140 free slots created
//!                       ↓
//! Pass N+1, layers 0–2: pinned, always hit (zero stall)
//! Pass N+1, layers 3+:  misses consume free slots (no eviction scan)
//!                       prefetch fills remaining free slots
//!                       ↓
//! Mid-pass:             free slots depleted → prefetch auto-disables
//!                       forced evictions use behind-layer bias
//!                       (evict completed layers, never future ones)
//!                       ↓
//! End of pass N+1:      batch-evict 5% → cycle repeats
//! ```
//!
//! ## Transition matrix and speculative prefetch
//!
//! An online-learned transition matrix tracks expert→expert routing
//! patterns across adjacent MoE layers.  For each pair of consecutive
//! layers `(L, L+1)`, a `[E × E]` co-occurrence matrix records how often
//! an expert at layer L is followed by each expert at layer L+1.
//!
//! The matrix is built incrementally during inference — no calibration
//! pass required.  At each layer, the predictor ranks the top-`K` most likely
//! *non-cached* experts for the next layer; their DMA begins speculatively into
//! free slots while the current layer's compute runs.
//!
//! Correct predictions convert cold misses into free cache hits.
//! Incorrect predictions harmlessly occupy a free slot until normal
//! eviction reclaims it.
//!
//! ## Two operating modes
//!
//! | Mode | When | Thread? | Lock? |
//! |------|------|---------|-------|
//! | **Threaded** | mmap path (partial VRAM residency) | Yes — background thread owns all state | No — `&mut self` on thread |
//! | **Inline** | reader path (full VRAM residency) | No — all experts pre-loaded | Yes — `Mutex` (uncontended) |
//!
//! The inline mode is used when all experts fit in VRAM (the reader path).
//! No DMA ever occurs, so the background thread adds no value.  The Mutex
//! is uncontended because layers execute sequentially.
//!
//! ## Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`MmapExpertRef`] | Byte-range reference into the mmap for one expert's projections |
//! | [`ExpertSlot`]     | A single VRAM slot holding one expert's gate/up/down `QMatMul`s |
//! | [`CopyBatchFence`] | Opaque fence for in-flight DMA on the copy stream |
//! | [`ExpertCache`]    | Handle to the pipeline — threaded or inline |
//! | [`MoeWorkRequest`] | Work item sent to the pipeline thread |
//! | [`TransitionMatrix`] | Online-learned expert→expert routing predictor |
//! | [`PINNED_LAYERS`]  | Number of early layers exempt from eviction |
//!
//! ## Module structure
//!
//! | File | Contents |
//! |------|----------|
//! | [`types`]      | Shared data types (`MmapExpertRef`, `ExpertSlot`, etc.) |
//! | [`cache`]      | `ExpertCacheInner` — slot management and eviction policy |
//! | [`compute`]    | SwiGLU expert computation and `QMatMul` re-export |
//! | [`transition`] | `TransitionMatrix` — online-learned routing predictor |
//! | [`pinned`]     | `PinnedPool` — pinned host memory warm tier |
//! | [`pipeline`]   | `PipelineState`, background thread, DMA loading |
//! | [`handle`]     | `ExpertCache` public API and `PipelineMode` |

mod cache;
mod compute;
#[cfg(test)]
mod eval;
#[cfg(feature = "cuda")]
mod gpu_dispatch;
mod handle;
#[cfg(all(test, feature = "cuda"))]
mod matmul_baseline;
mod pinned;
mod pipeline;
mod transition;
mod types;

// Re-exports — the public API of this module.
pub use crate::models::profile::ProfileSnapshot;
#[cfg(feature = "cuda")]
pub use gpu_dispatch::GpuDispatchTables;
pub use handle::ExpertCache;
/// What the weight zone must be carved into to hold one expert.
///
/// The model loader needs this **before** the cache exists: the zone's slot size
/// decides its capacity, its capacity decides where the weight boundary sits,
/// and the boundary has to be placed before a single expert is uploaded into it.
#[cfg(feature = "cuda")]
pub(crate) use pinned::layer_geometries;
#[cfg(feature = "cuda")]
pub(crate) use pipeline::slot_bytes_for;
pub use types::{
    CopyBatchFence, ExpertSlot, MmapExpertRef, MoeInput, MoeWorkRequest, PipelineStats,
};

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
//!   The eviction key multiplies it by two more terms — what the reload would
//!   cost and how far the expert's layer is from being routed again — and the
//!   victims are chosen by an O(n) partial sort over contiguous memory.
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
//! Eviction is a **pure drop** — the cold pack is authoritative and the warm
//! tier is immutable, so releasing a slot moves no bytes and can fail in no
//! way.  That is what shapes the policy: there is no copy to hide, so nothing
//! is gained by freeing slots ahead of the demand for them, and eviction can
//! happen at the exact moment the demand is known.
//!
//! ### 1. Exact-demand batch eviction (the primary path)
//!
//! `classify_and_load` counts a layer's misses **before** issuing a single
//! load, and `ExpertCacheInner::demand_eviction` frees precisely
//! `misses − free` slots in one scan — scored at the wave's real layer, with
//! that layer's own hits protected.  One partial sort per layer, never a
//! per-miss scan.
//!
//! This replaces an end-of-pass batch eviction that freed a fixed fraction
//! proactively.  Two things were wrong with that: it over-evicted by its own
//! estimate error, and it scored mid-pass victims as if the pass were at layer
//! 0.  Knowing the exact deficit removes both, and the headroom it was
//! creating bought nothing once eviction stopped copying.
//!
//! ### 2. The eviction key: `score × reload_cost × position`
//!
//!   - **score** — the lightly-decayed access frequency above; the dominant
//!     term, so the cache behaves as LFU with a recency decay.
//!   - **`reload_cost`** — 1.0 when the warm tier holds a copy, else
//!     `COLD_RELOAD_PENALTY`.  Under the three-tier cache the two outcomes
//!     differ by an order of magnitude (a pinned-memory H2D against a
//!     page-cache-bypassing NVMe read), so a policy that ignores it is
//!     choosing blind.
//!   - **position** — a mild `[0.5, 1.0]` multiplier that FALLS with forward
//!     (wrapped) reuse distance.  Bélády's direction, and computable rather
//!     than predicted: the layer traversal is a cycle, so the distance to an
//!     expert's next use is a subtraction.  The layer about to be routed is
//!     most protected; the layer just executed is the preferred victim.
//!
//! ### 3. Layer-aware forced eviction (the backstop)
//!
//! The per-miss path in `ExpertCacheInner::allocate_slot`, reached only when
//! the batch scan came up short.  It prefers a low-scored expert from a layer
//! already executed this pass (behind the wave, so evicting it cannot cascade
//! into a downstream miss), and falls back to the globally lowest-scored
//! victim.  Both respect pinning and the in-flight protect set.
//!
//! ### 4. Early-layer pinning
//!
//! Experts in the first [`PINNED_LAYERS`] MoE layers are never evicted: they
//! run first every pass with no compute ahead of them to overlap a DMA
//! against, so evicting them guarantees a cold miss at maximum stall.
//!
//! The depth is what pinning is *worth*; what it may **cost** is derived from
//! capacity by `cache::affordable_pinned_layers`, because pinned experts are
//! capacity the cache can never reuse.  On a wide card the reservation is
//! noise.  It bites on many experts per layer against a small card — see that
//! function for the worked case.
//!
//! ### 5. Windowed prefetch eviction
//!
//! Speculative prefetch takes free slots first, and may make room only from
//! the **furthest-behind** layers (`cache::PREFETCH_EVICT_WINDOW`, wrapping
//! from the current layer).  Near-future layers are structurally out of its
//! reach, so a mispredicted prefetch cannot displace an expert this sweep is
//! about to need — while still letting prefetch run on a card with no standing
//! headroom, which a free-slot-only rule could not.
//!
//! ## Transition matrix and speculative prefetch
//!
//! An online-learned transition matrix tracks expert→expert routing
//! patterns across adjacent MoE layers.  For each pair of consecutive
//! layers `(L, L+1)`, a `[E × E]` co-occurrence matrix records how often
//! an expert at layer L is followed by each expert at layer L+1.
//!
//! The matrix is built incrementally during inference — no calibration pass
//! required, and no extra compute: it consumes routing IDs only, never live
//! activations, so it is free to evaluate and shared across every token in a
//! wave that routed to the same expert.  At each layer the predictor ranks the
//! likely *non-cached* experts for the next layer and their DMA begins while
//! the current layer computes.
//!
//! The fan-out is **not** a fixed top-`K`.  Each candidate must clear a
//! per-source relative confidence floor, ranked by pointwise mutual
//! information and capped at a fixed maximum, so depth tracks demand
//! *diversity* rather than demand *width* — see [`transition`] for why the cap
//! must not scale with the batch.
//!
//! Correct predictions convert cold misses into overlapped loads.  Incorrect
//! ones occupy a slot the windowed eviction above will reclaim, taken from
//! layers the wave has already left.
//!
//! Prediction is worth nothing at prefill width, where the next layer routes
//! to most of its experts and there is nothing to guess.  That regime is
//! served by `streamer` instead: bulk whole-layer streaming, off the
//! pipeline thread.
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
//! | [`pack`]       | `ExpertPack` — the authoritative cold tier on disk |
//! | [`pinned`]     | `WarmPool` — pinned host memory warm tier, and its draw |
//! | [`pipeline`]   | `PipelineState`, background thread, DMA loading |
//! | `streamer`     | Off-thread whole-layer streaming for prefill-width waves (cuda) |
//! | `gpu_dispatch` | Device-resident expert pointer tables for the grouped GEMM (cuda) |
//! | [`handle`]     | `ExpertCache` public API and `PipelineMode` |

mod cache;
pub(crate) mod compute;
#[cfg(test)]
mod eval;
#[cfg(feature = "cuda")]
mod gpu_dispatch;
mod handle;
#[cfg(all(test, feature = "cuda"))]
mod matmul_baseline;
mod pack;
mod pinned;
mod pipeline;
#[cfg(feature = "cuda")]
mod streamer;
mod transition;
mod types;

// Re-exports — the public API of this module.
pub use crate::models::profile::ProfileSnapshot;
#[cfg(feature = "cuda")]
pub use cache::minimum_resident_slots;
#[cfg(feature = "cuda")]
pub use gpu_dispatch::GpuDispatchTables;
pub use handle::ExpertCache;
#[cfg(feature = "cuda")]
pub use handle::ExpertCacheSetup;
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

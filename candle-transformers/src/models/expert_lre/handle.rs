//! Public API — the [`ExpertCache`] handle and its two operating modes.
//!
//! [`ExpertCache`] is the main entry point for the expert pipeline.  It
//! provides two modes:
//!
//! - **Threaded** (mmap path): a background thread owns all mutable state.
//!   The forward path submits work via a channel and blocks for the result.
//! - **Inline** (reader path): all experts are pre-loaded to VRAM.  No
//!   thread, no DMA.  A Mutex provides interior mutability (uncontended).

use super::cache::ExpertCacheInner;
#[cfg(not(feature = "cuda"))]
use super::compute::compute_expert_contribution_gpu_weights;
#[cfg(feature = "cuda")]
use super::compute::compute_experts_grouped;
#[cfg(feature = "cuda")]
use super::gpu_dispatch::GpuDispatchTables;
#[cfg(feature = "cuda")]
use super::pinned::{ExpertLocation, LayerGeometry, PinnedPool};
#[cfg(not(feature = "cuda"))]
use super::pipeline::prewarm_expert_cache;
#[cfg(feature = "cuda")]
use super::pipeline::startup_two_tier;
use super::pipeline::{spawn_pipeline_thread, PipelineState};
use super::transition::TransitionMatrix;
use super::types::{
    ClassifiedExperts, CopyBatchFence, ExpertSlot, MmapExpertRef, MoeInput, MoeWorkRequest,
    PipelineMessage, PipelineStats,
};
use crate::models::profile::{profile_now, ProfileAccumulator, ProfileMark, ProfileSnapshot};
use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::WeightZone;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

// ============================================================================
// Pinned pool sizing
// ============================================================================

/// Layers' worth of experts the pinned pool keeps free for the swap pipeline.
///
/// A swap is evict-then-load: moving an expert VRAM→pinned **consumes** a free
/// pinned slot, and the matching pinned→VRAM load gives one back. Net zero, but
/// the pool needs slack for the turnover — at zero free slots there is no way to
/// evict to make room to load, nor to load to make room to evict.
///
/// One layer is the batch bound: `evict_for_prefetch_batch` is asked for as many
/// experts as a layer is short, and a layer has `experts_per_layer` of them. Two
/// covers the prefetch running a layer ahead
/// ([`super::cache::PREFETCH_EVICT_WINDOW`] looks back that far for candidates).
///
/// This is reserved *from the boundary*, not shared with it — the two competing
/// for one pool is what made a retraction able to stall the pipeline.
pub(crate) const CHURN_RESERVE_LAYERS: usize = 2;

/// The smallest number of experts VRAM may hold, as a fraction of the model.
///
/// This is the **design input** the pinned pool is sized from, and therefore the
/// real answer to "how far can the boundary move". A quarter of the model leaves
/// the weight side able to give up three quarters of its residency to the KV
/// side for a wide prefill, and take it back for decode.
///
/// It is a choice, not a measurement, and it is bounded by host RAM: every slot
/// of shrink range costs one pinned slot, so lowering it trades host RAM for
/// VRAM roughly one for one. That is the trade the two-tier design exists to
/// make — host RAM is the plentiful side — but it is not free, and
/// `cuMemAllocHost` refusing at load is how a machine says the fraction is too
/// low for it.
fn min_vram_expert_slots(total_experts: usize) -> usize {
    total_experts / 4
}

// ============================================================================
// Pipeline mode
// ============================================================================

/// The two operating modes of the expert cache.
enum PipelineMode {
    /// Background thread owns all mutable state.  Used for the mmap path
    /// where DMA overlap is active and experts cycle through VRAM.
    Threaded {
        /// Channel to submit work/hints to the pipeline thread.
        tx: mpsc::SyncSender<PipelineMessage>,
    },
    /// All experts pre-loaded to VRAM.  No background thread.  Mutex
    /// protects the inner state (uncontended — layers execute sequentially).
    Inline {
        inner: Mutex<ExpertCacheInner>,
        device: Device,
    },
}

// ============================================================================
// ExpertCache — the public handle
// ============================================================================

/// Global expert cache / pipeline handle.
///
/// Shared across all `SparseMoeBlock`s via `Arc<ExpertCache>`.
///
/// ## Threaded mode (mmap path)
///
/// A background thread owns all mutable state (`ExpertCacheInner`, slots,
/// eviction scores, mmap, copy stream).  The forward path submits a `MoeWorkRequest`
/// via an MPSC channel and blocks on the oneshot response.  The thread
/// does the full classify → DMA → compute loop with `&mut self` — no locks.
///
/// ## Inline mode (reader path)
///
/// All experts are pre-loaded to VRAM.  No DMA, no background thread.
/// A `Mutex<ExpertCacheInner>` provides interior mutability (uncontended).
/// The forward path locks, computes all experts by slot index, releases.
pub struct ExpertCache {
    mode: PipelineMode,
    /// True when all experts fit in VRAM — hint sending is elided.
    all_resident: bool,
    /// Dedicated CUDA stream for async routing index DtoH.
    /// Lives on the forward thread — never crosses to the pipeline thread.
    #[cfg(feature = "cuda")]
    routing_stream: Option<Arc<CudaStream>>,
    /// Pinned host buffer for routing indices: `[max_tokens × k]` u32.
    /// Allocated via `cuMemAllocHost` — truly async DtoH destination.
    /// Reused every MoE layer (only one is active at a time).
    #[cfg(feature = "cuda")]
    routing_pinned: Option<PinnedRoutingBuffer>,
    /// Static GPU-native dispatch tables (all-resident cache only): per-expert
    /// weight pointers indexed on-device by `moe_bucketize`'s tile tables, so
    /// the expert forward needs no routing readback. `None` ⇒ host path.
    #[cfg(feature = "cuda")]
    gpu_dispatch: Option<GpuDispatchTables>,
    /// Set when the pipeline thread has exited (normally or by panic). Its
    /// `PipelineState` drop frees every expert slot, so the dispatch tables'
    /// captured weight pointers become dangling — the GPU-native gate checks
    /// this so a dead pipeline degrades to the host path's loud channel error
    /// instead of silently dereferencing freed VRAM. Never set in inline mode
    /// (no thread).
    pipeline_dead: Arc<AtomicBool>,
    /// Expert IDs from the most recently completed MoE layer.
    /// Used by the next layer's hint to predict via transition matrix.
    prev_layer_experts: Mutex<Vec<usize>>,
    /// Shared telemetry counters (always-on).
    stats: Arc<Mutex<PipelineStats>>,
    /// Timing accumulator for forward-thread spans (routing, submission).
    /// Only present when profiling is enabled.
    #[cfg(feature = "profile")]
    forward_profile: Mutex<ProfileAccumulator>,
}

/// Pinned host buffer for routing indices (async DtoH destination).
#[cfg(feature = "cuda")]
struct PinnedRoutingBuffer {
    /// Raw pointer from `cuMemAllocHost`.
    ptr: *mut u32,
    /// Capacity in u32 elements.
    capacity: usize,
}

#[cfg(feature = "cuda")]
impl PinnedRoutingBuffer {
    /// Allocate a pinned buffer for `capacity` u32 elements.
    fn new(capacity: usize) -> Result<Self> {
        let byte_size = capacity * std::mem::size_of::<u32>();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, byte_size) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle::bail!(
                "cuMemAllocHost for routing buffer failed: {:?} ({} bytes)",
                result,
                byte_size,
            );
        }
        Ok(Self {
            ptr: ptr as *mut u32,
            capacity,
        })
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedRoutingBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

// SAFETY: The pinned memory is GPU-accessible and not tied to a thread.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedRoutingBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedRoutingBuffer {}

impl ExpertCache {
    /// Create a new expert cache with a background pipeline thread.
    ///
    /// **CUDA path (two-tier):** GPU-repacks every expert from GGUF to K/128,
    /// fills VRAM slots first, overflows to a pinned host-memory pool.
    /// After startup the GGUF mmap is no longer needed.
    ///
    /// **Non-CUDA path:** fills VRAM from mmap (legacy GGML path).
    #[allow(clippy::too_many_arguments)]
    /// `zone` is the weight side of the device reservation, already sized: its
    /// capacity **is** the resident-expert count. There is no budget arithmetic
    /// left at this level — `VramGovernor::expert_budget` used to divide bytes by
    /// `max_expert_size` here, and the zone's capacity is that same quotient
    /// taken once, against a span whose extent is a fact rather than a forecast.
    pub fn new(
        mmap: Arc<memmap2::Mmap>,
        host_refs: Vec<Vec<MmapExpertRef>>,
        zone: WeightZone,
        device: &Device,
        experts_per_layer: usize,
        _gguf_path: Option<&std::path::Path>,
        progress: Option<&dyn Fn(usize, usize)>,
        int8mode: candle::quantized::Int8Mode,
    ) -> Result<Self> {
        let num_moe_layers = host_refs.len();
        let num_slots = zone.capacity();
        let mut inner = ExpertCacheInner::new(zone, num_moe_layers, experts_per_layer);

        // ── CUDA copy stream (pipeline thread) ──
        #[cfg(feature = "cuda")]
        let copy_stream: Option<Arc<CudaStream>> = if let Device::Cuda(cuda_dev) = device {
            match cuda_dev.cuda_context().new_stream() {
                Ok(stream) => {
                    tracing::info!("Expert cache: created copy stream for DMA overlap");
                    Some(stream)
                }
                Err(e) => {
                    tracing::warn!(
                        "Expert cache: failed to create copy stream, falling back to single-stream: {e}"
                    );
                    None
                }
            }
        } else {
            None
        };

        // ── CUDA routing stream + pinned buffer (forward thread) ──
        #[cfg(feature = "cuda")]
        let (routing_stream, routing_pinned): (
            Option<Arc<CudaStream>>,
            Option<PinnedRoutingBuffer>,
        ) = if let Device::Cuda(cuda_dev) = device {
            let stream = match cuda_dev.cuda_context().new_stream() {
                Ok(s) => {
                    tracing::info!("Expert cache: created routing stream for async DtoH");
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!("Expert cache: failed to create routing stream: {e}");
                    None
                }
            };
            // 1024 tokens × 8 experts = 8192 u32 elements = 32 KB
            let buf = match PinnedRoutingBuffer::new(1024 * experts_per_layer) {
                Ok(b) => {
                    tracing::info!(
                        "Expert cache: allocated {} KB pinned routing buffer",
                        (1024 * experts_per_layer * 4) / 1024,
                    );
                    Some(b)
                }
                Err(e) => {
                    tracing::warn!("Expert cache: failed to allocate pinned routing buffer: {e}");
                    None
                }
            };
            (stream, buf)
        } else {
            (None, None)
        };

        let transition_matrix = TransitionMatrix::new(num_moe_layers, experts_per_layer);
        if num_moe_layers > 1 {
            tracing::info!(
                "Expert cache: transition matrix enabled ({} layer pairs, {} experts/layer)",
                num_moe_layers - 1,
                experts_per_layer,
            );
        }

        // ── CUDA two-tier startup ──
        #[cfg(feature = "cuda")]
        let (pinned_pool, expert_locations, layer_geometries, all_resident) =
            if let Device::Cuda(cuda_dev) = device {
                let geoms = super::pinned::layer_geometries(&host_refs, int8mode)?;

                // **Size the pinned pool for the smallest VRAM residency the
                // boundary may ever reach, not for the one it happens to open
                // at.**
                //
                // An expert is in VRAM or pinned and nowhere else, so
                // `pinned_occupied = total_experts − vram_slots` at all times.
                // Shrink VRAM and pinned occupancy grows one for one — but the
                // pool is a single `cuMemAllocHost` that never grows, so
                // whatever slack it opens with is the *entire* budget for the
                // boundary to move.
                //
                // Sizing it from `num_slots` (the opening position) made that
                // budget accidental: it came out as 10% of wherever the boundary
                // started, and that 10% is also the swap pipeline's in-flight
                // depth — `evict_for_prefetch_batch` takes up to a layer's worth
                // at a time, with prefetch running a layer or two ahead. The two
                // uses shared one pool, so on a live rebuild the retraction asked
                // for thousands of regions, found the churn slack was all there
                // was, and delivered nothing.
                //
                // Sizing it from the *floor* instead makes the dependency run one
                // way: pick how small the weight side may get, and the pool is
                // whatever guarantees every expert has somewhere to be at that
                // extreme, plus churn depth on top. Above the floor the pool
                // simply carries empty slots, which is exactly what headroom
                // looks like.
                //
                // The allocation is the feasibility test. There is no need to ask
                // the OS what is available — `cuMemAllocHost` either succeeds or
                // it does not, and pinned pages are non-pageable, so a refusal is
                // a real answer and not a hint. Failure propagates and the
                // process stops at load with the numbers in the message.
                let total_experts = num_moe_layers * experts_per_layer;
                let all_resident = num_slots >= total_experts;
                let num_pinned = if all_resident {
                    0
                } else {
                    let floor = min_vram_expert_slots(total_experts);
                    let churn = CHURN_RESERVE_LAYERS * experts_per_layer;
                    total_experts.saturating_sub(floor) + churn
                };
                let slot_size = geoms
                    .iter()
                    .map(|g| g.total_repacked_size)
                    .max()
                    .unwrap_or(0);

                let mut pool = PinnedPool::new(num_pinned, slot_size)?;

                // Initialize location tracking. Every expert starts as `Pinned { slot_idx: 0 }`;
                // `startup_two_tier` below overwrites each entry with its real resident-VRAM or
                // pinned-slot location as it repacks the weights.
                let mut locations: Vec<Vec<ExpertLocation>> = Vec::with_capacity(num_moe_layers);
                for _ in 0..num_moe_layers {
                    let mut layer_locs = Vec::with_capacity(experts_per_layer);
                    for _ in 0..experts_per_layer {
                        layer_locs.push(ExpertLocation::Pinned { slot_idx: 0 });
                    }
                    locations.push(layer_locs);
                }

                // Run startup: GGUF → GPU repack → VRAM or pinned.
                startup_two_tier(
                    &mut inner,
                    &mut pool,
                    &mut locations,
                    &geoms,
                    &mmap,
                    &host_refs,
                    cuda_dev,
                    progress,
                );

                (pool, locations, geoms, all_resident)
            } else {
                // Non-CUDA device — empty pinned pool + empty locations.
                let pool = PinnedPool::empty();
                let locations: Vec<Vec<ExpertLocation>> = Vec::new();
                let geoms: Vec<LayerGeometry> = Vec::new();
                let all_resident = num_slots >= num_moe_layers * experts_per_layer;
                (pool, locations, geoms, all_resident)
            };

        // ── Non-CUDA prewarm path ──
        #[cfg(not(feature = "cuda"))]
        {
            let _ = progress; // not yet wired into the legacy prewarm path
            prewarm_expert_cache(&mut inner, &mmap, &host_refs, device, int8mode);
        }

        // Non-CUDA: all_resident is always false (no VRAM cache to fill).
        #[cfg(not(feature = "cuda"))]
        let all_resident = false;

        let stats = PipelineStats::new_shared();
        // Seed the resident-expert VRAM gauge with the startup footprint (occupied
        // slots × slot size) so it reads correctly before the first classify
        // refreshes it. `inner` + `layer_geometries` are still in scope here,
        // before they move into `PipelineState` below.
        #[cfg(feature = "cuda")]
        {
            let occupied = inner.num_slots() - inner.free_len();
            let slot_bytes = layer_geometries
                .iter()
                .map(|g| g.total_repacked_size)
                .max()
                .unwrap_or(0);
            let seeded = occupied * slot_bytes;
            tracing::info!(
                target: "candle_transformers::expert_lre",
                num_slots = inner.num_slots(),
                free_slots = inner.free_len(),
                occupied,
                slot_bytes,
                resident_gib = seeded as f64 / 1e9,
                "expert cache: seeded resident-VRAM gauge"
            );
            if let Ok(mut s) = stats.lock() {
                s.resident_vram_bytes = seeded;
            }
        }

        // With every expert staged into a permanent VRAM slot by the
        // (synchronous) startup above, the weight addresses are static:
        // capture them into device pointer tables while `inner` is still in
        // scope, so the expert forward dispatches entirely on-GPU (no per-layer
        // routing readback, no pipeline-thread handoff). Only meaningful when
        // all-resident — a paged cache's pointers move with eviction.
        #[cfg(feature = "cuda")]
        let gpu_dispatch = if all_resident {
            if let Device::Cuda(cuda_dev) = device {
                GpuDispatchTables::build(&inner, cuda_dev)
            } else {
                None
            }
        } else {
            None
        };

        let state = PipelineState {
            inner,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            copy_stream,
            #[cfg(feature = "cuda")]
            pinned_pool,
            #[cfg(feature = "cuda")]
            expert_locations,
            #[cfg(feature = "cuda")]
            layer_geometries,
            num_moe_layers,
            #[cfg(feature = "cuda")]
            all_resident,
            transition_matrix,
            last_moe_layer_idx: None,
            speculative_loads: HashSet::new(),
            pending_prefetch_fence: CopyBatchFence::noop(),
            hint_stats: (0, 0),
            profile: ProfileAccumulator::new(),
            stats: stats.clone(),
            pass_misses: 0,
            pass_drip_evicts: 0,
            eviction_rate: 0.07,
            drip_headroom: 0.02,
            #[cfg(not(feature = "cuda"))]
            mmap,
            #[cfg(not(feature = "cuda"))]
            host_refs,
            #[cfg(not(feature = "cuda"))]
            int8mode,
        };

        let pipeline_dead = Arc::new(AtomicBool::new(false));
        let tx = spawn_pipeline_thread(state, pipeline_dead.clone());

        Ok(Self {
            mode: PipelineMode::Threaded { tx },
            all_resident,
            #[cfg(feature = "cuda")]
            routing_stream,
            #[cfg(feature = "cuda")]
            routing_pinned,
            #[cfg(feature = "cuda")]
            gpu_dispatch,
            pipeline_dead,
            prev_layer_experts: Mutex::new(Vec::new()),
            stats,
            #[cfg(feature = "profile")]
            forward_profile: Mutex::new(ProfileAccumulator::new()),
        })
    }

    /// Construct an `ExpertCache` in inline mode with pre-populated slots.
    ///
    /// Used when all experts are already loaded to VRAM (e.g. the non-mmap
    /// reader path).  No background thread, no copy stream, no DMA.
    pub fn new_prepopulated(
        slots: Vec<Option<ExpertSlot>>,
        key_to_slot: HashMap<(usize, usize), usize>,
        last_used: Vec<u32>,
        generation: u32,
        slot_to_key: Vec<Option<(usize, usize)>>,
        device: &Device,
    ) -> Self {
        // Every slot is occupied and none ever moves, so the zone exists only to
        // answer "which index is where" — the addresses are the pre-built
        // storages' own, not the zone's, and nothing here allocates from it.
        let mut zone = WeightZone::new(0, 0, slots.len(), slots.len());
        for _ in 0..slots.len() {
            zone.alloc();
        }
        let inner = ExpertCacheInner {
            slots,
            zone,
            key_to_slot,
            last_used,
            generation,
            slot_to_key,
            expert_scores: vec![],
            num_moe_layers: 0,
            experts_per_layer: 0,
        };
        // All experts are VRAM-resident and never move, so their weight
        // addresses are static: capture them once into device pointer tables
        // and the expert forward dispatches entirely on-GPU (no per-layer
        // routing readback). Best-effort — `None` keeps the host path.
        #[cfg(feature = "cuda")]
        let gpu_dispatch = if let Device::Cuda(cuda_dev) = device {
            GpuDispatchTables::build(&inner, cuda_dev)
        } else {
            None
        };
        Self {
            mode: PipelineMode::Inline {
                inner: Mutex::new(inner),
                device: device.clone(),
            },
            all_resident: true, // prepopulated = all in VRAM
            #[cfg(feature = "cuda")]
            routing_stream: None,
            #[cfg(feature = "cuda")]
            routing_pinned: None,
            #[cfg(feature = "cuda")]
            gpu_dispatch,
            pipeline_dead: Arc::new(AtomicBool::new(false)),
            prev_layer_experts: Mutex::new(Vec::new()),
            stats: PipelineStats::new_shared(),
            #[cfg(feature = "profile")]
            forward_profile: Mutex::new(ProfileAccumulator::new()),
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────────────────────────

    /// Live VRAM bytes held by resident expert slots (`occupied_slots ×
    /// slot_size`) — the model's **time-varying** MoE weight footprint. Rises as
    /// experts load into VRAM and falls as they stream out to pinned RAM under
    /// pressure. Read lock-free from the shared stats gauge (seeded at
    /// construction, refreshed each classify). `0` on non-CUDA / no-expert models.
    pub fn resident_vram_bytes(&self) -> usize {
        PipelineStats::snapshot(&self.stats).resident_vram_bytes
    }

    /// Submit a full MoE dispatch to the pipeline.
    ///
    /// This is the primary entry point for `SparseMoeBlock::forward()`.
    /// In threaded mode, sends the work to the background thread and blocks
    /// until the result arrives.  In inline mode, executes synchronously.
    ///
    /// # Arguments
    ///
    /// * `moe_layer_idx` — which MoE layer (0..num_moe_layers)
    /// * `expert_ids` — unique expert IDs selected by the router (sorted)
    /// * `xs` — input hidden states `[num_tokens, hidden_dim]`
    /// * `weights_flat` — flattened routing weights `[num_tokens * k]`
    /// * `assignments` — flat sorted `(expert_id, token_idx, weight_idx)` array
    ///
    /// # Returns
    ///
    /// Output tensor `[num_tokens, hidden_dim]` — the weighted sum of expert outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_moe_work(
        &self,
        moe_layer_idx: usize,
        expert_ids: Vec<usize>,
        input: MoeInput,
        out_dtype: DType,
        weights_flat: &Tensor,
        assignments: Vec<(u32, u32, u32)>,
        wave: Option<WaveTicket>,
    ) -> Result<Tensor> {
        match &self.mode {
            PipelineMode::Threaded { tx } => {
                let t = profile_now();

                // Create a oneshot response channel.
                let (resp_tx, resp_rx) = mpsc::sync_channel(1);

                let request = MoeWorkRequest {
                    moe_layer_idx,
                    expert_ids,
                    input,
                    out_dtype,
                    weights_flat: weights_flat.clone(),
                    assignments,
                    // Captured right before `send` so the worker can split the inbound handoff
                    // (channel wakeup) out of the actual work.
                    wave,
                    submitted_at: profile_now(),
                    response_tx: resp_tx,
                };

                tx.send(PipelineMessage::Work(request)).map_err(|_| {
                    candle::Error::Msg("Expert pipeline thread died — channel closed".into())
                })?;

                // Block until the pipeline thread returns the result + its completion timestamp.
                let (result, worker_done_at) = resp_rx.recv().map_err(|_| {
                    candle::Error::Msg(
                        "Expert pipeline thread died — response channel closed".into(),
                    )
                })?;

                // submit_roundtrip = submit_inbound (worker) + pipe_worker_total (worker) +
                // submit_outbound (here). The two handoffs are the reclaimable threading tax;
                // pipe_worker_total is the irreducible classify + DMA + compute.
                self.record_profile("submit_outbound", worker_done_at);
                self.record_profile("submit_roundtrip", t);
                result
            }
            PipelineMode::Inline { inner, device } => {
                // Inline mode: no thread, just lock and compute.
                self.submit_inline(
                    inner,
                    device,
                    moe_layer_idx,
                    &expert_ids,
                    input,
                    out_dtype,
                    weights_flat,
                    &assignments,
                    wave,
                )
            }
        }
    }

    /// Inline execution path (reader/prepopulated mode).
    ///
    /// All experts are already resident — no DMA, no copy stream.
    /// Lock the Mutex (uncontended), compute all experts by slot index.
    #[allow(clippy::too_many_arguments)]
    fn submit_inline(
        &self,
        mutex: &Mutex<ExpertCacheInner>,
        device: &Device,
        moe_layer_idx: usize,
        expert_ids: &[usize],
        input: MoeInput,
        out_dtype: DType,
        weights_flat: &Tensor,
        assignments: &[(u32, u32, u32)],
        wave: Option<WaveTicket>,
    ) -> Result<Tensor> {
        let mut inner = mutex
            .lock()
            .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;

        // In inline mode, all experts are resident — classify as all hits.
        let mut hits: Vec<(usize, usize)> = Vec::with_capacity(expert_ids.len());
        for &expert_idx in expert_ids {
            if let Some(&slot_idx) = inner.key_to_slot.get(&(moe_layer_idx, expert_idx)) {
                inner.promote(slot_idx);
                hits.push((expert_idx, slot_idx));
            } else {
                return Err(candle::Error::Msg(format!(
                    "Inline mode: expert ({moe_layer_idx}, {expert_idx}) not found in cache"
                )));
            }
        }

        let expert_group = |eid: usize| -> (Vec<u32>, Vec<u32>) {
            let eid32 = eid as u32;
            let lo = assignments.partition_point(|a| a.0 < eid32);
            let hi = assignments.partition_point(|a| a.0 <= eid32);
            let toks: Vec<u32> = assignments[lo..hi].iter().map(|a| a.1).collect();
            let wids: Vec<u32> = assignments[lo..hi].iter().map(|a| a.2).collect();
            (toks, wids)
        };

        let (num_tokens, hidden) = input.shape()?;
        // The inline twin of the threaded pipeline's combine target. It runs on
        // the caller's thread and stream, so the layer's FFN generation *would*
        // bound it — but this mode exists only when there is no pipeline thread,
        // which production never configures, so it stays an ordinary allocation
        // rather than a second wave consumer for a path nothing takes.
        let mut ys = Tensor::zeros((num_tokens, hidden), out_dtype, device)?;
        let mut _inline_prof = ProfileAccumulator::new();
        #[cfg(feature = "cuda")]
        {
            let experts_data: Vec<(Vec<u32>, Vec<u32>)> =
                hits.iter().map(|&(eidx, _)| expert_group(eidx)).collect();
            let experts_vec: Vec<(&ExpertSlot, &[u32], &[u32])> = hits
                .iter()
                .zip(experts_data.iter())
                .filter_map(|(&(_, slot_idx), (toks, wids))| {
                    let slot = inner.slots[slot_idx].as_ref()?;
                    Some((slot, toks.as_slice(), wids.as_slice()))
                })
                .collect();
            if !experts_vec.is_empty() {
                compute_experts_grouped(
                    &input,
                    &mut ys,
                    &experts_vec,
                    weights_flat,
                    &mut _inline_prof,
                    wave,
                )?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Non-CUDA only ever sees `Float` (int8/q8a128 is cuda-only).
            let MoeInput::Float(xs) = &input;
            for &(eidx, slot_idx) in &hits {
                let slot = inner.slots[slot_idx].as_ref().ok_or_else(|| {
                    candle::Error::Msg(format!("inline slot {slot_idx} unexpectedly empty"))
                })?;
                let (toks, w_ids) = expert_group(eidx);
                compute_expert_contribution_gpu_weights(
                    xs,
                    &mut ys,
                    slot,
                    &toks,
                    weights_flat,
                    &w_ids,
                )?;
            }
        }

        Ok(ys)
    }

    /// Legacy API: classify and load experts, returning slot indices + fence.
    ///
    /// Only available in inline mode.  Panics in threaded mode (use
    /// `submit_moe_work` instead).
    pub fn classify_and_load(
        &self,
        moe_idx: usize,
        expert_ids: &[usize],
    ) -> Result<ClassifiedExperts> {
        match &self.mode {
            PipelineMode::Inline { inner, .. } => {
                let mut guard = inner
                    .lock()
                    .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;

                let mut hits: Vec<(usize, usize)> = Vec::new();
                for &expert_idx in expert_ids {
                    if let Some(&slot_idx) = guard.key_to_slot.get(&(moe_idx, expert_idx)) {
                        if guard.slots[slot_idx].is_some() {
                            guard.promote(slot_idx);
                            hits.push((expert_idx, slot_idx));
                        }
                    }
                }

                Ok(ClassifiedExperts {
                    hits,
                    loaded: vec![],
                    fence: CopyBatchFence::noop(),
                })
            }
            PipelineMode::Threaded { .. } => Err(candle::Error::Msg(
                "classify_and_load not available in threaded mode — use submit_moe_work".into(),
            )),
        }
    }

    /// Lock the inner cache state and run a closure with mutable access.
    ///
    /// Only available in inline mode.  Panics in threaded mode.
    pub fn with_inner<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut ExpertCacheInner) -> Result<R>,
    {
        match &self.mode {
            PipelineMode::Inline { inner, .. } => {
                let mut guard = inner
                    .lock()
                    .map_err(|_| candle::Error::Msg("ExpertCache Mutex poisoned".into()))?;
                f(&mut guard)
            }
            PipelineMode::Threaded { .. } => Err(candle::Error::Msg(
                "with_inner not available in threaded mode — use submit_moe_work".into(),
            )),
        }
    }

    /// Make the main (compute) stream wait for a batch fence.
    ///
    /// Only needed when using the legacy classify_and_load API.
    /// In threaded mode, fence waiting is handled by the pipeline thread.
    pub fn wait_for_copies(&self, fence: CopyBatchFence) -> Result<()> {
        let _fence = fence;
        #[cfg(feature = "cuda")]
        if let Some(ref event) = _fence.event {
            let device = match &self.mode {
                PipelineMode::Inline { device, .. } => device,
                PipelineMode::Threaded { .. } => {
                    return Err(candle::Error::Msg(
                        "wait_for_copies not available in threaded mode".into(),
                    ));
                }
            };
            if let Device::Cuda(cuda_dev) = device {
                cuda_dev
                    .cuda_stream()
                    .wait(event)
                    .map_err(candle::Error::wrap)?;
            }
        }
        Ok(())
    }

    /// Returns true if this cache uses the background pipeline thread.
    pub fn is_threaded(&self) -> bool {
        matches!(self.mode, PipelineMode::Threaded { .. })
    }

    /// Send a speculative prediction hint to the pipeline thread.
    ///
    /// The hint tells the pipeline thread which MoE layer is coming and
    /// which experts were used in the previous layer.  The pipeline thread
    /// uses the transition matrix to predict which experts will be needed
    /// and starts DMA before the full work request arrives.
    ///
    /// No-op in inline mode, when all experts are resident, or if the
    /// channel send fails.
    pub fn send_hint(&self, layer_idx: usize, prev_expert_ids: Vec<usize>) {
        if self.all_resident {
            return;
        }
        if let PipelineMode::Threaded { tx } = &self.mode {
            let _ = tx.send(PipelineMessage::Hint {
                layer_idx,
                prev_expert_ids,
            });
        }
    }

    /// Get the routing stream for async DtoH (if available).
    #[cfg(feature = "cuda")]
    pub fn routing_stream(&self) -> Option<&Arc<CudaStream>> {
        self.routing_stream.as_ref()
    }

    /// The static GPU-native dispatch tables, when this cache is all-resident
    /// and they were successfully built at construction. `Some` ⇒ the expert
    /// forward can run entirely on-device (no routing readback).
    #[cfg(feature = "cuda")]
    pub fn gpu_dispatch(&self) -> Option<&GpuDispatchTables> {
        self.gpu_dispatch.as_ref()
    }

    /// Whether the pipeline thread has exited — after which the dispatch
    /// tables' captured weight pointers are dangling and the GPU-native path
    /// must not run. Always `false` in inline mode.
    pub fn pipeline_dead(&self) -> bool {
        self.pipeline_dead.load(Ordering::Acquire)
    }

    /// The dispatch tables IF the GPU-native path is safe for `moe_layer_idx`
    /// with a router `n_experts` wide — the cache-owned safety conditions in
    /// one place so no caller can skip part of the chain:
    /// * tables built (all-resident, KO weights, null-stream compute),
    /// * the layer inside the covered row range,
    /// * router width == table row width (a mismatch would index other
    ///   layers' experts),
    /// * the pipeline thread alive (its death frees the slot weights the
    ///   tables point at — dangling-pointer dispatch).
    ///
    /// Callers still gate their own model-level conditions (top-k bound,
    /// routing capture, diagnostic env overrides).
    #[cfg(feature = "cuda")]
    pub fn live_gpu_dispatch(
        &self,
        moe_layer_idx: usize,
        n_experts: usize,
    ) -> Option<&GpuDispatchTables> {
        let gd = self.gpu_dispatch.as_ref()?;
        if gd.expert_base(moe_layer_idx).is_none()
            || gd.n_experts != n_experts
            || self.pipeline_dead()
        {
            return None;
        }
        Some(gd)
    }

    /// Base pointer of the pinned routing buffer, if it exists and holds `len`.
    ///
    /// A pointer rather than a `&mut [u32]`: the cache lives behind an `Arc`, so
    /// a `&mut` handed out from `&self` is one the borrow checker cannot police —
    /// nothing stops a second call from minting an overlapping one. The single
    /// writer here is the forward thread, and the ordering against the routing
    /// stream's DtoH is the caller's (it holds the events), so the caller is
    /// where the slice and its `unsafe` belong.
    #[cfg(feature = "cuda")]
    pub fn routing_pinned_ptr(&self, len: usize) -> Option<*mut u32> {
        let pinned = self.routing_pinned.as_ref()?;
        if len > pinned.capacity {
            return None;
        }
        Some(pinned.ptr)
    }

    /// Store the expert IDs from the most recently completed MoE layer.
    /// Called after routing indices are read and before submitting work.
    pub fn set_prev_layer_experts(&self, experts: Vec<usize>) {
        if let Ok(mut prev) = self.prev_layer_experts.lock() {
            *prev = experts;
        }
    }

    /// Get a clone of the previous layer's expert IDs for hint prediction.
    pub fn get_prev_layer_experts(&self) -> Vec<usize> {
        self.prev_layer_experts
            .lock()
            .map_or_else(|_| vec![], |v| v.clone())
    }

    /// Snapshot and reset all profile accumulators (forward + pipeline threads).
    ///
    /// Returns a merged [`ProfileSnapshot`] containing both forward-thread
    /// spans (prefixed `fwd_`) and pipeline-thread spans (`pipe_`, `cl_`,
    /// `gemm_`, `dma_`).
    pub fn snapshot_profiles(&self) -> ProfileSnapshot {
        // Forward thread profile (lives on this handle behind Mutex).
        let fwd = {
            #[cfg(feature = "profile")]
            {
                self.forward_profile.lock().map_or_else(
                    |_| ProfileSnapshot::default(),
                    |mut prof| {
                        let snap = prof.snapshot();
                        prof.reset();
                        snap
                    },
                )
            }
            #[cfg(not(feature = "profile"))]
            {
                ProfileSnapshot::default()
            }
        };

        // Pipeline thread profile (request via message).
        let pipe = match &self.mode {
            PipelineMode::Threaded { tx } => {
                let (resp_tx, resp_rx) = std::sync::mpsc::sync_channel(1);
                if tx
                    .send(PipelineMessage::SnapshotProfile {
                        response_tx: resp_tx,
                    })
                    .is_ok()
                {
                    resp_rx.recv().unwrap_or_default()
                } else {
                    ProfileSnapshot::default()
                }
            }
            PipelineMode::Inline { .. } => ProfileSnapshot::default(),
        };

        // Merge both into a single snapshot.
        let mut merged = fwd;
        merged.merge(&pipe);
        merged
    }

    /// Snapshot the current pipeline telemetry counters.
    pub fn expert_stats(&self) -> PipelineStats {
        PipelineStats::snapshot(&self.stats)
    }

    /// Reset all pipeline telemetry counters to zero.
    pub fn reset_expert_stats(&self) {
        PipelineStats::reset(&self.stats);
    }

    /// Record a profiling span from external callers (e.g. SparseMoeBlock).
    ///
    /// When profiling is disabled, this is an inline no-op.
    #[cfg(feature = "profile")]
    pub fn record_profile(&self, name: &'static str, start: ProfileMark) {
        if let Ok(mut prof) = self.forward_profile.lock() {
            prof.record(name, start);
        }
    }

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn record_profile(&self, _name: &'static str, _start: ProfileMark) {}
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // Profile report is now collected per-config via snapshot_profiles().
        // Only print at shutdown if un-collected data remains.
        #[cfg(feature = "profile")]
        {
            if let Ok(prof) = self.forward_profile.lock() {
                if prof.has_data() {
                    println!("{}", prof.report("Forward Thread Profile (uncollected)"));
                }
            }
        }
    }
}

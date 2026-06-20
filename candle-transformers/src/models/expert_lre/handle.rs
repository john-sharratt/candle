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
use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};

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
    pub fn new(
        mmap: Arc<memmap2::Mmap>,
        host_refs: Vec<Vec<MmapExpertRef>>,
        num_slots: usize,
        device: &Device,
        experts_per_layer: usize,
        _gguf_path: Option<&std::path::Path>,
        progress: Option<&dyn Fn(usize, usize)>,
        int8mode: candle::quantized::Int8Mode,
    ) -> Result<Self> {
        let num_moe_layers = host_refs.len();
        let mut inner = ExpertCacheInner::new(num_slots, num_moe_layers, experts_per_layer);

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
                // Build per-layer geometry. The pinned pool caches the *target* format: for Off
                // that is the gemx K/128 repack of the source dtype; for int8 it is the KO twin
                // (repacked once per expert, then DMA-reloaded on a miss — no per-miss re-quant).
                let mut geoms: Vec<LayerGeometry> = Vec::with_capacity(num_moe_layers);
                for moe_idx in 0..num_moe_layers {
                    let r = &host_refs[moe_idx][0];
                    let tko = |d: candle::quantized::GgmlDType| {
                        if int8mode.is_int8() {
                            d.to_ko(int8mode)
                        } else {
                            Ok(d)
                        }
                    };
                    let gate_dtype = tko(r.gate_dtype)?;
                    let up_dtype = tko(r.up_dtype)?;
                    let down_dtype = tko(r.down_dtype)?;
                    let gate_repacked_size = candle::quantized::repacked_size_bytes(
                        r.gate_shape[0],
                        r.gate_shape[1],
                        gate_dtype,
                    )?;
                    let up_repacked_size = candle::quantized::repacked_size_bytes(
                        r.up_shape[0],
                        r.up_shape[1],
                        up_dtype,
                    )?;
                    let down_repacked_size = candle::quantized::repacked_size_bytes(
                        r.down_shape[0],
                        r.down_shape[1],
                        down_dtype,
                    )?;
                    geoms.push(LayerGeometry {
                        gate_shape: r.gate_shape.clone(),
                        gate_dtype,
                        gate_repacked_size,
                        up_shape: r.up_shape.clone(),
                        up_dtype,
                        up_repacked_size,
                        down_shape: r.down_shape.clone(),
                        down_dtype,
                        down_repacked_size,
                        total_repacked_size: gate_repacked_size
                            + up_repacked_size
                            + down_repacked_size,
                    });
                }

                // Compute pinned pool size: total experts minus VRAM slots,
                // plus 10% headroom for runtime evictions (drip + end-of-pass
                // move VRAM experts to pinned — need free pinned slots for that).
                // When all experts fit in VRAM, skip the pinned pool entirely —
                // no eviction is needed and pinned RAM would be wasted.
                let total_experts = num_moe_layers * experts_per_layer;
                let all_resident = num_slots >= total_experts;
                let num_pinned = if all_resident {
                    0
                } else {
                    let eviction_headroom = num_slots / 10; // 10% of VRAM slots
                    total_experts.saturating_sub(num_slots) + eviction_headroom
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

        let tx = spawn_pipeline_thread(state);

        Ok(Self {
            mode: PipelineMode::Threaded { tx },
            all_resident,
            #[cfg(feature = "cuda")]
            routing_stream,
            #[cfg(feature = "cuda")]
            routing_pinned,
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
        Self {
            mode: PipelineMode::Inline {
                inner: Mutex::new(ExpertCacheInner {
                    slots,
                    free_slots: vec![],
                    key_to_slot,
                    last_used,
                    generation,
                    slot_to_key,
                    expert_scores: vec![],
                    num_moe_layers: 0,
                    experts_per_layer: 0,
                }),
                device: device.clone(),
            },
            all_resident: true, // prepopulated = all in VRAM
            #[cfg(feature = "cuda")]
            routing_stream: None,
            #[cfg(feature = "cuda")]
            routing_pinned: None,
            prev_layer_experts: Mutex::new(Vec::new()),
            stats: PipelineStats::new_shared(),
            #[cfg(feature = "profile")]
            forward_profile: Mutex::new(ProfileAccumulator::new()),
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public API
    // ────────────────────────────────────────────────────────────────────────

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

    /// Get mutable access to the pinned routing buffer.
    ///
    /// Returns `(buffer_ptr_as_mut_slice, capacity)` if available.
    /// The caller must ensure no concurrent DMA is in flight.
    #[cfg(feature = "cuda")]
    pub fn routing_pinned_mut(&self, len: usize) -> Option<&mut [u32]> {
        // SAFETY: We're the only thread accessing the routing buffer
        // (it lives on the forward thread), and the caller ensures
        // DMA has completed before reading.  We use interior mutability
        // via raw pointer since ExpertCache is behind Arc.
        let pinned = self.routing_pinned.as_ref()?;
        if len > pinned.capacity {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(pinned.ptr, len) })
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

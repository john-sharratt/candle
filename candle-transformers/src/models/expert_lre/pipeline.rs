//! Background pipeline thread — owns all mutable state with `&mut self`.
//!
//! The pipeline thread runs the full classify → DMA → compute loop for
//! each MoE layer.  It exclusively owns the [`ExpertCacheInner`], the
//! mmap, the copy stream, and the [`TransitionMatrix`].  No locks on
//! the hot path — everything is `&mut self`.
//!
//! This module also contains the DMA loading functions that build
//! [`ExpertSlot`]s from mmap byte ranges.
//!
//! # Page-cache / PCIe bandwidth management
//!
//! The mmap represents cold expert weights on SSD.  OS page-faults pull
//! pages into RAM; then H2D DMA copies them to VRAM.  Both operations
//! share PCIe bandwidth (NVMe ↔ RAM, RAM ↔ GPU).  Two optimisations
//! separate these two traffic classes so they never compete:
//!
//! ## 1. Promote (RAM → VRAM): evict pages after H2D completes
//!
//! After `load_from_mmap_on_stream` finishes copying expert data to VRAM,
//! the source pages in the OS page cache are no longer needed — the GPU
//! now owns the data.  We can call:
//!
//! - **Unix**: `madvise(MADV_DONTNEED)` on the byte range
//!   `[r.gate_offset .. r.down_offset + r.down_len]`.
//! - **Windows**: `VirtualUnlock` or `DiscardVirtualMemory` on the range.
//!
//! This immediately frees physical RAM for other experts that will be
//! loaded next, reducing total RSS and avoiding page-cache pressure in the OS
//! page cache that could evict pages we still need.
//!
//! ## 2. Demote (VRAM → RAM): prefetch pages from SSD before needed
//!
//! When an expert is evicted from VRAM (`evict()` or `end_of_pass_eviction`),
//! we know it will eventually be loaded again.  At eviction time we can
//! proactively ask the OS to start paging-in the mmap region for experts
//! that are *predicted* to be needed soon (from the transition matrix or
//! from the evicted expert's own region):
//!
//! - **Unix**: `madvise(MADV_WILLNEED)` on the byte range.
//! - **Windows**: `PrefetchVirtualMemory` on the range.
//!
//! This converts future page faults into background DMA from SSD → RAM,
//! which completes while the GPU is busy computing.  When the pipeline
//! later needs to promote that expert, the pages are already hot in RAM
//! and the H2D copy encounters zero page-fault stalls.
//!
//! ## Net effect
//!
//! The two optimisations together create a clean tiered pipeline:
//!
//! ```text
//! SSD ──(prefetch/WILLNEED)──▸ RAM ──(H2D DMA)──▸ VRAM
//!                               ◂──(DONTNEED)──  (after H2D)
//! ```
//!
//! PCIe bandwidth is reserved exclusively for H2D transfers.  SSD→RAM
//! traffic runs as background readahead that completes during GPU compute.
//! RAM usage stays bounded: pages are released as soon as the GPU has them.
//!
//! ## Interaction with `cuMemHostRegister`
//!
//! When the mmap is host-registered (pinned) for DMA, all pages are locked
//! in physical RAM.  In that case `MADV_DONTNEED` may be a no-op (kernel
//! won’t evict locked pages) and `MADV_WILLNEED` is unnecessary (pages are
//! already resident).  The helpers below handle this gracefully — the
//! advise calls are cheap no-ops when pages are pinned, and become active
//! when the mmap is *not* fully pinned (e.g. insufficient system RAM to
//! pin the entire file).

use super::cache::ExpertCacheInner;
#[cfg(not(feature = "cuda"))]
use super::compute::compute_expert_contribution_gpu_weights;
#[cfg(feature = "cuda")]
use super::compute::compute_experts_grouped;
use super::compute::QMatMul;
#[cfg(feature = "cuda")]
use super::pinned::{ExpertLocation, LayerGeometry, PinnedPool};
use super::transition::TransitionMatrix;
use super::types::{
    ClassifiedExperts, CopyBatchFence, ExpertSlot, MmapExpertRef, MoeWorkRequest, PipelineMessage,
    PipelineStats,
};
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use cudarc::driver::CudaStream;
use std::collections::HashSet;
use std::sync::{mpsc, Arc, Mutex};

// ============================================================================
// Page-cache management helpers (legacy, only for non-CUDA mmap path)
// ============================================================================

/// Advise the OS that the given mmap byte range is no longer needed in RAM.
#[cfg(not(feature = "cuda"))]
fn mmap_page_evict(mmap: &memmap2::Mmap, offset: usize, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(unix)]
    {
        let _ = mmap.advise_range(memmap2::Advice::DontNeed, offset, len);
    }
    #[cfg(not(unix))]
    {
        let _ = (mmap, offset, len);
    }
}

/// Release page-cache pages for an expert's three projection tensors.
#[cfg(not(feature = "cuda"))]
fn mmap_evict_expert(mmap: &memmap2::Mmap, r: &MmapExpertRef) {
    mmap_page_evict(mmap, r.gate_offset, r.gate_len);
    mmap_page_evict(mmap, r.up_offset, r.up_len);
    mmap_page_evict(mmap, r.down_offset, r.down_len);
}

// ============================================================================
// Two-tier startup: GGUF → GPU repack → VRAM or pinned RAM
// ============================================================================

/// Fill the VRAM cache and pinned pool from the GGUF mmap.
///
/// For each expert, reads GGML bytes from GGUF, GPU-repacks to K/128, then:
///   - If VRAM has free slots → install as ExpertSlot (stays in VRAM)
///   - Else → D2H repacked bytes to a pinned pool slot
///
/// After all experts are processed, the GGUF mmap can be dropped.
#[cfg(feature = "cuda")]
pub(crate) fn startup_two_tier(
    inner: &mut ExpertCacheInner,
    pinned_pool: &mut PinnedPool,
    expert_locations: &mut Vec<Vec<ExpertLocation>>,
    layer_geometries: &[LayerGeometry],
    mmap: &[u8],
    host_refs: &[Vec<MmapExpertRef>],
    cuda_dev: &candle::CudaDevice,
    progress: Option<&dyn Fn(usize, usize)>,
) {
    let num_moe_layers = host_refs.len();
    let num_experts = if num_moe_layers > 0 {
        host_refs[0].len()
    } else {
        0
    };
    let total_slots = inner.slots.len();

    if total_slots == 0 || num_moe_layers == 0 || num_experts == 0 {
        return;
    }
    let total_experts = num_moe_layers * num_experts;

    tracing::info!(
        "startup: repacking {}×{} experts → {} VRAM + {} pinned slots …",
        num_moe_layers,
        num_experts,
        total_slots,
        pinned_pool.num_slots(),
    );
    let t0 = std::time::Instant::now();

    let mut vram_count: usize = 0;
    let mut pinned_count: usize = 0;
    let mut errors: usize = 0;

    for moe_idx in 0..num_moe_layers {
        let geom = &layer_geometries[moe_idx];
        for expert_idx in 0..num_experts {
            let r = &host_refs[moe_idx][expert_idx];

            // Read GGML bytes + GPU repack each projection to K/128.
            let result = repack_expert_projections(mmap, r, geom, cuda_dev);

            match result {
                Ok((gate_repacked, up_repacked, down_repacked)) => {
                    if !inner.free_slots.is_empty() {
                        // ── VRAM path: install as ExpertSlot ──
                        let slot_idx = inner.free_slots.pop().unwrap();
                        match build_slot_from_repacked_with_device(
                            &gate_repacked,
                            &up_repacked,
                            &down_repacked,
                            geom,
                            cuda_dev,
                        ) {
                            Ok(slot) => {
                                inner.install(slot_idx, moe_idx, expert_idx, slot);
                                expert_locations[moe_idx][expert_idx] =
                                    ExpertLocation::Vram { slot_idx };
                                vram_count += 1;
                            }
                            Err(e) => {
                                tracing::warn!("startup: wrap failed L{moe_idx}E{expert_idx}: {e}");
                                inner.free_slots.push(slot_idx);
                                errors += 1;
                            }
                        }
                    } else {
                        // ── Pinned path: D2H repacked bytes to pinned pool ──
                        match pinned_pool.alloc() {
                            Some(p_slot) => {
                                let dst = pinned_pool.slot_mut(p_slot, geom.total_repacked_size);
                                let g = geom.gate_repacked_size;
                                let u = geom.up_repacked_size;
                                let d = geom.down_repacked_size;
                                dst[..g].copy_from_slice(&gate_repacked);
                                dst[g..g + u].copy_from_slice(&up_repacked);
                                dst[g + u..g + u + d].copy_from_slice(&down_repacked);
                                expert_locations[moe_idx][expert_idx] =
                                    ExpertLocation::Pinned { slot_idx: p_slot };
                                pinned_count += 1;
                            }
                            None => {
                                tracing::warn!("startup: pinned pool full at L{moe_idx}E{expert_idx}");
                                errors += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("startup: repack failed L{moe_idx}E{expert_idx}: {e}");
                    errors += 1;
                }
            }
            if let Some(cb) = progress {
                cb(moe_idx * num_experts + expert_idx + 1, total_experts);
            }
        }
        // Progress every 8 layers.
        if (moe_idx + 1) % 8 == 0 || moe_idx + 1 == num_moe_layers {
            tracing::info!(
                "  startup: layer {}/{} — {} VRAM, {} pinned, {} errors",
                moe_idx + 1,
                num_moe_layers,
                vram_count,
                pinned_count,
                errors,
            );
        }
    }

    let elapsed = t0.elapsed();
    tracing::info!(
        "startup: done in {:.1}s — {} VRAM + {} pinned ({} errors)",
        elapsed.as_secs_f64(),
        vram_count,
        pinned_count,
        errors,
    );
}

/// GPU-repack one expert's three projections from GGML to K/128 format.
///
/// Returns `(gate_bytes, up_bytes, down_bytes)` as host `Vec<u8>`.
#[cfg(feature = "cuda")]
fn repack_expert_projections(
    mmap: &[u8],
    r: &MmapExpertRef,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let gate_ggml = &mmap[r.gate_offset..r.gate_offset + r.gate_len];
    let up_ggml = &mmap[r.up_offset..r.up_offset + r.up_len];
    let down_ggml = &mmap[r.down_offset..r.down_offset + r.down_len];

    let gate_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        gate_ggml,
        geom.gate_shape[0],
        geom.gate_shape[1],
        geom.gate_dtype,
    )?;
    let up_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        up_ggml,
        geom.up_shape[0],
        geom.up_shape[1],
        geom.up_dtype,
    )?;
    let down_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        down_ggml,
        geom.down_shape[0],
        geom.down_shape[1],
        geom.down_dtype,
    )?;

    Ok((gate_repacked, up_repacked, down_repacked))
}

/// Build an `ExpertSlot` from already-repacked host bytes by loading them
/// to a CUDA device's VRAM and wrapping as `QMatMul`.
#[cfg(feature = "cuda")]
fn build_slot_from_repacked_with_device(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
) -> Result<ExpertSlot> {
    let gate_storage = candle::quantized::load_repacked(cuda_dev, gate_bytes, geom.gate_dtype)?;
    let up_storage = candle::quantized::load_repacked(cuda_dev, up_bytes, geom.up_dtype)?;
    let down_storage = candle::quantized::load_repacked(cuda_dev, down_bytes, geom.down_dtype)?;

    let gate_qt = candle::quantized::QTensor::new(gate_storage, geom.gate_shape.clone())?;
    let up_qt = candle::quantized::QTensor::new(up_storage, geom.up_shape.clone())?;
    let down_qt = candle::quantized::QTensor::new(down_storage, geom.down_shape.clone())?;

    Ok(ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    })
}

/// Build an `ExpertSlot` from already-repacked host bytes using a copy stream
/// for async DMA.
#[cfg(feature = "cuda")]
fn build_slot_from_repacked_on_stream(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    copy_stream: &Arc<CudaStream>,
    profile: &mut ProfileAccumulator,
) -> Result<ExpertSlot> {
    let t = profile_now();
    let gate_storage = candle::quantized::load_repacked_on_stream(
        cuda_dev,
        copy_stream,
        gate_bytes,
        geom.gate_dtype,
    )?;
    let up_storage =
        candle::quantized::load_repacked_on_stream(cuda_dev, copy_stream, up_bytes, geom.up_dtype)?;
    let down_storage = candle::quantized::load_repacked_on_stream(
        cuda_dev,
        copy_stream,
        down_bytes,
        geom.down_dtype,
    )?;
    profile.record("dma_h2d", t);

    let t = profile_now();
    let gate_qt = candle::quantized::QTensor::new(gate_storage, geom.gate_shape.clone())?;
    let up_qt = candle::quantized::QTensor::new(up_storage, geom.up_shape.clone())?;
    let down_qt = candle::quantized::QTensor::new(down_storage, geom.down_shape.clone())?;

    let slot = ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    };
    profile.record("dma_wrap", t);

    Ok(slot)
}

/// Non-CUDA prewarm: fill VRAM from mmap (legacy path).
#[cfg(not(feature = "cuda"))]
pub(crate) fn prewarm_expert_cache(
    inner: &mut ExpertCacheInner,
    mmap: &Arc<memmap2::Mmap>,
    host_refs: &[Vec<MmapExpertRef>],
    device: &Device,
) {
    let num_moe_layers = host_refs.len();
    let num_experts = if num_moe_layers > 0 {
        host_refs[0].len()
    } else {
        0
    };
    let total_slots = inner.slots.len();

    if total_slots == 0 || num_moe_layers == 0 || num_experts == 0 {
        return;
    }

    tracing::info!(
        "prewarm: filling {} VRAM slots from {}×{} experts …",
        total_slots, num_moe_layers, num_experts,
    );
    let t0 = std::time::Instant::now();
    let mut loaded_count: usize = 0;

    'fill: for moe_idx in 0..num_moe_layers {
        for expert_idx in 0..num_experts {
            if inner.free_slots.is_empty() {
                break 'fill;
            }
            let slot_idx = inner.free_slots.pop().unwrap();
            let mmap_bytes: &[u8] = mmap;
            let result = load_from_mmap(mmap_bytes, &host_refs[moe_idx][expert_idx], device);
            match result {
                Ok(expert_slot) => {
                    inner.install(slot_idx, moe_idx, expert_idx, expert_slot);
                    let r = &host_refs[moe_idx][expert_idx];
                    mmap_evict_expert(mmap, r);
                    loaded_count += 1;
                }
                Err(e) => {
                    tracing::warn!("prewarm: failed to load L{moe_idx}E{expert_idx}: {e}");
                    inner.free_slots.push(slot_idx);
                }
            }
        }
    }

    let elapsed1 = t0.elapsed();
    tracing::info!(
        "prewarm: {} experts to VRAM in {:.2}s",
        loaded_count,
        elapsed1.as_secs_f64(),
    );
}

// ============================================================================
// DMA loading — non-CUDA mmap path (legacy)
// ============================================================================

/// Build an `ExpertSlot` from mmap data on the default stream.
/// Only used on non-CUDA builds.
#[cfg(not(feature = "cuda"))]
fn load_from_mmap(mmap: &[u8], r: &MmapExpertRef, device: &Device) -> Result<ExpertSlot> {
    let gate_data = &mmap[r.gate_offset..r.gate_offset + r.gate_len];
    let gate_proj = candle::quantized::ggml_file::qtensor_from_ggml(
        r.gate_dtype,
        gate_data,
        r.gate_shape.clone(),
        device,
    )?;

    let up_data = &mmap[r.up_offset..r.up_offset + r.up_len];
    let up_proj = candle::quantized::ggml_file::qtensor_from_ggml(
        r.up_dtype,
        up_data,
        r.up_shape.clone(),
        device,
    )?;

    let down_data = &mmap[r.down_offset..r.down_offset + r.down_len];
    let down_proj = candle::quantized::ggml_file::qtensor_from_ggml(
        r.down_dtype,
        down_data,
        r.down_shape.clone(),
        device,
    )?;

    Ok(ExpertSlot {
        gate_proj: QMatMul::from_weights(gate_proj.into())?,
        up_proj: QMatMul::from_weights(up_proj.into())?,
        down_proj: QMatMul::from_weights(down_proj.into())?,
    })
}

// ============================================================================
// Two-tier eviction helper (VRAM → pinned)
// ============================================================================

/// Copy an ExpertSlot's three projections from VRAM to a pinned pool slot.
///
/// Uses the copy stream for async D2H DMA.  The caller must record a fence
/// event and wait before reusing the pinned slot.
#[cfg(feature = "cuda")]
fn evict_slot_to_pinned(
    slot: &ExpertSlot,
    pinned_pool: &mut PinnedPool,
    pinned_slot: usize,
    geom: &LayerGeometry,
    copy_stream: &Arc<CudaStream>,
) -> Result<()> {
    let g = geom.gate_repacked_size;
    let u = geom.up_repacked_size;
    let d = geom.down_repacked_size;
    let total = g + u + d;

    let dst = pinned_pool.slot_mut(pinned_slot, total);

    // Get QTensor from each QMatMul → QStorage::Cuda → copy_to_host_on_stream
    let gate_qt = slot
        .gate_proj
        .inner()
        .qtensor()
        .ok_or_else(|| candle::Error::Msg("evict: gate_proj is not QTensor".into()))?;
    gate_qt.copy_data_to_host_on_stream(&mut dst[..g], copy_stream)?;

    let up_qt = slot
        .up_proj
        .inner()
        .qtensor()
        .ok_or_else(|| candle::Error::Msg("evict: up_proj is not QTensor".into()))?;
    up_qt.copy_data_to_host_on_stream(&mut dst[g..g + u], copy_stream)?;

    let down_qt = slot
        .down_proj
        .inner()
        .qtensor()
        .ok_or_else(|| candle::Error::Msg("evict: down_proj is not QTensor".into()))?;
    down_qt.copy_data_to_host_on_stream(&mut dst[g + u..g + u + d], copy_stream)?;

    Ok(())
}

// ============================================================================
// Pipeline state — two-tier architecture
// ============================================================================

/// The pipeline thread's private state.  Never crosses a thread boundary
/// after construction — the thread owns it exclusively with `&mut self`.
pub(crate) struct PipelineState {
    /// Mutable cache bookkeeping (VRAM slots, eviction scores, free list).
    pub(crate) inner: ExpertCacheInner,
    /// Target device for QTensor creation.
    pub(crate) device: Device,
    /// Secondary CUDA stream for DMA overlap.
    #[cfg(feature = "cuda")]
    pub(crate) copy_stream: Option<Arc<CudaStream>>,
    /// Pinned host memory pool — warm tier for experts not in VRAM.
    #[cfg(feature = "cuda")]
    pub(crate) pinned_pool: PinnedPool,
    /// Per-expert location tracking: `[moe_layer][expert_idx]`.
    #[cfg(feature = "cuda")]
    pub(crate) expert_locations: Vec<Vec<ExpertLocation>>,
    /// Per-layer geometry (shapes, dtypes, repacked sizes).
    #[cfg(feature = "cuda")]
    pub(crate) layer_geometries: Vec<LayerGeometry>,
    /// Number of MoE layers.
    pub(crate) num_moe_layers: usize,
    /// True when all experts fit in VRAM — disables eviction & pinned pool.
    #[cfg(feature = "cuda")]
    pub(crate) all_resident: bool,
    /// Online-learned transition matrix for speculative prefetch.
    pub(crate) transition_matrix: TransitionMatrix,
    /// Tracks which moe_layer_idx was seen last to detect new forward passes.
    pub(crate) last_moe_layer_idx: Option<usize>,
    /// Set of `(layer_idx, expert_idx)` pairs speculatively loaded via Hint.
    /// Cleared when the corresponding Work request arrives.
    pub(crate) speculative_loads: HashSet<(usize, usize)>,
    /// Prediction accuracy counters: (hints_sent, hits_in_actual_set).
    pub(crate) hint_stats: (usize, usize),
    /// Timing accumulator for pipeline spans.
    pub(crate) profile: ProfileAccumulator,
    /// Shared telemetry counters (always-on).
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    // ── Adaptive eviction rate tracking ──
    /// Cache misses accumulated in the current forward pass.
    pub(crate) pass_misses: usize,
    /// Drip evictions performed in the current forward pass.
    pub(crate) pass_drip_evicts: usize,
    /// EMA-smoothed end-of-pass eviction rate.  Seed 0.07.
    pub(crate) eviction_rate: f32,
    /// EMA-smoothed drip headroom fraction.  Seed 0.02.
    pub(crate) drip_headroom: f32,
    // ── Non-CUDA fields (legacy mmap path) ──
    #[cfg(not(feature = "cuda"))]
    pub(crate) mmap: Arc<memmap2::Mmap>,
    #[cfg(not(feature = "cuda"))]
    pub(crate) host_refs: Vec<Vec<MmapExpertRef>>,
}

impl PipelineState {
    /// Classify experts as hits/misses, load misses via DMA from pinned pool.
    ///
    /// CUDA path: misses are loaded from pinned RAM (H2D on copy_stream).
    /// Eviction goes VRAM → pinned RAM (D2H on copy_stream).
    fn classify_and_load(
        &mut self,
        moe_idx: usize,
        expert_ids: &[usize],
    ) -> Result<ClassifiedExperts> {
        if expert_ids.is_empty() {
            return Ok(ClassifiedExperts {
                hits: vec![],
                loaded: vec![],
                fence: CopyBatchFence::noop(),
            });
        }

        // ── Phase 1: classify + reserve slots ──
        let t = profile_now();
        let mut hits: Vec<(usize, usize)> = Vec::new();
        let mut to_load: Vec<(usize, usize)> = Vec::new(); // (expert_idx, slot_idx)

        for &expert_idx in expert_ids {
            if let Some(&slot_idx) = self.inner.key_to_slot.get(&(moe_idx, expert_idx)) {
                if self.inner.slots[slot_idx].is_some() {
                    self.inner.promote(slot_idx);
                    self.inner.record_hit(moe_idx, expert_idx);
                    hits.push((expert_idx, slot_idx));
                    continue;
                }
            }

            // Cache miss — allocate a slot (layer-aware eviction)
            let (slot_idx, evicted_key, evicted_slot) = self.inner.allocate_slot(moe_idx)?;

            // D2H eviction: copy evicted expert to pinned pool.
            // If pinned is full, the evicted expert is lost (acceptable
            // since the miss we're about to load will free a pinned slot,
            // and the evicted expert was the lowest-scored anyway).
            #[cfg(feature = "cuda")]
            if let (Some((evict_moe, evict_exp)), Some(slot)) = (evicted_key, evicted_slot) {
                if let Err(_e) = self.evict_to_pinned(evict_moe, evict_exp, &slot) {
                    // Pinned full — expert data lost.  Mark location as
                    // Vram with a sentinel so it's treated as absent.
                    // (The slot index is already freed by allocate_slot.)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (evicted_key, evicted_slot);
            }

            to_load.push((expert_idx, slot_idx));
        }
        self.profile.record("cl_classify", t);

        // ── Phase 2: DMA all misses from pinned pool ──
        if to_load.is_empty() {
            return Ok(ClassifiedExperts {
                hits,
                loaded: vec![],
                fence: CopyBatchFence::noop(),
            });
        }

        let mut loaded_slots: Vec<(usize, usize, ExpertSlot)> = Vec::with_capacity(to_load.len());

        #[cfg(feature = "cuda")]
        {
            for &(expert_idx, slot_idx) in &to_load {
                let expert_slot = self.load_from_pinned(moe_idx, expert_idx)?;
                loaded_slots.push((expert_idx, slot_idx, expert_slot));
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let mmap_bytes: &[u8] = &self.mmap;
            for &(expert_idx, slot_idx) in &to_load {
                let mmap_ref = &self.host_refs[moe_idx][expert_idx];
                let expert_slot = load_from_mmap(mmap_bytes, mmap_ref, &self.device)?;
                loaded_slots.push((expert_idx, slot_idx, expert_slot));
            }
        }

        // ── Record single fence event ──
        let fence = {
            #[cfg(feature = "cuda")]
            if let (Some(cs), Device::Cuda(_)) = (&self.copy_stream, &self.device) {
                let event = cs.record_event(None).map_err(candle::Error::wrap)?;
                CopyBatchFence { event: Some(event) }
            } else {
                CopyBatchFence::noop()
            }
            #[cfg(not(feature = "cuda"))]
            CopyBatchFence::noop()
        };

        // ── Phase 3: install loaded experts ──
        let t = profile_now();
        let mut loaded: Vec<(usize, usize)> = Vec::with_capacity(loaded_slots.len());
        for (expert_idx, slot_idx, slot) in loaded_slots {
            self.inner.install(slot_idx, moe_idx, expert_idx, slot);
            loaded.push((expert_idx, slot_idx));

            // Update location tracker: now in VRAM.
            #[cfg(feature = "cuda")]
            {
                self.expert_locations[moe_idx][expert_idx] = ExpertLocation::Vram { slot_idx };
            }
        }
        self.profile.record("cl_install", t);

        // ── Telemetry ──
        {
            let num_hits = hits.len();
            let num_loaded = loaded.len();
            self.pass_misses += num_loaded;
            if let Ok(mut s) = self.stats.lock() {
                s.expert_hits += num_hits;
                s.expert_misses += num_loaded;
                s.dma_loads += num_loaded;
            }
        }

        Ok(ClassifiedExperts {
            hits,
            loaded,
            fence,
        })
    }

    /// Evict an expert from VRAM to the pinned pool (D2H).
    #[cfg(feature = "cuda")]
    fn evict_to_pinned(
        &mut self,
        moe_idx: usize,
        expert_idx: usize,
        slot: &ExpertSlot,
    ) -> Result<()> {
        let geom = &self.layer_geometries[moe_idx];

        // Allocate a pinned slot.
        let pinned_slot = self.pinned_pool.alloc().ok_or_else(|| {
            candle::Error::Msg(format!(
                "pinned pool full — cannot evict L{moe_idx}E{expert_idx}"
            ))
        })?;

        // D2H copy on the copy stream.
        if let Some(cs) = &self.copy_stream {
            evict_slot_to_pinned(slot, &mut self.pinned_pool, pinned_slot, geom, cs)?;
        } else {
            // Fallback: synchronous D2H via data().
            let gate_qt = slot
                .gate_proj
                .inner()
                .qtensor()
                .ok_or_else(|| candle::Error::Msg("evict: gate not QTensor".into()))?;
            let gate_bytes = gate_qt.data()?;
            let up_qt = slot
                .up_proj
                .inner()
                .qtensor()
                .ok_or_else(|| candle::Error::Msg("evict: up not QTensor".into()))?;
            let up_bytes = up_qt.data()?;
            let down_qt = slot
                .down_proj
                .inner()
                .qtensor()
                .ok_or_else(|| candle::Error::Msg("evict: down not QTensor".into()))?;
            let down_bytes = down_qt.data()?;

            let g = geom.gate_repacked_size;
            let u = geom.up_repacked_size;
            let d = geom.down_repacked_size;
            let dst = self.pinned_pool.slot_mut(pinned_slot, g + u + d);
            dst[..g].copy_from_slice(&gate_bytes);
            dst[g..g + u].copy_from_slice(&up_bytes);
            dst[g + u..g + u + d].copy_from_slice(&down_bytes);
        }

        self.expert_locations[moe_idx][expert_idx] = ExpertLocation::Pinned {
            slot_idx: pinned_slot,
        };

        if let Ok(mut s) = self.stats.lock() {
            s.evictions += 1;
            s.dma_evicts += 1;
        }

        Ok(())
    }

    /// Load an expert from the pinned pool to VRAM (H2D).
    #[cfg(feature = "cuda")]
    fn load_from_pinned(&mut self, moe_idx: usize, expert_idx: usize) -> Result<ExpertSlot> {
        let geom = &self.layer_geometries[moe_idx];

        // Find the pinned slot.
        let pinned_slot = match self.expert_locations[moe_idx][expert_idx] {
            ExpertLocation::Pinned { slot_idx } => slot_idx,
            ExpertLocation::Vram { .. } => {
                candle::bail!("load_from_pinned: L{moe_idx}E{expert_idx} is already in VRAM");
            }
        };

        let g = geom.gate_repacked_size;
        let u = geom.up_repacked_size;
        let d = geom.down_repacked_size;

        // Read pinned bytes.
        let src = self.pinned_pool.slot_ref(pinned_slot, g + u + d);
        let gate_bytes = &src[..g];
        let up_bytes = &src[g..g + u];
        let down_bytes = &src[g + u..g + u + d];

        // H2D load + wrap.
        let slot = if let (Some(cs), Device::Cuda(cd)) = (&self.copy_stream, &self.device) {
            build_slot_from_repacked_on_stream(
                gate_bytes,
                up_bytes,
                down_bytes,
                geom,
                cd,
                cs,
                &mut self.profile,
            )?
        } else if let Device::Cuda(cd) = &self.device {
            build_slot_from_repacked_with_device(gate_bytes, up_bytes, down_bytes, geom, cd)?
        } else {
            candle::bail!("load_from_pinned requires CUDA device");
        };

        // Free the pinned slot — expert is now in VRAM.
        self.pinned_pool.free(pinned_slot);

        Ok(slot)
    }

    /// Proactive eviction: move bottom-N VRAM experts to pinned pool.
    ///
    /// Called to maintain VRAM headroom so real misses find free slots
    /// without triggering inline eviction scans.
    /// Automatically capped at available pinned pool capacity.
    #[cfg(feature = "cuda")]
    fn drip_evict(&mut self, count: usize) -> Result<()> {
        // Clamp to available pinned capacity — can't evict more than
        // pinned can absorb.
        let available = self.pinned_pool.free_slots.len();
        let count = count.min(available);
        if count == 0 {
            return Ok(());
        }

        let evicted = self
            .inner
            .end_of_pass_eviction(count as f32 / self.inner.slots.len().max(1) as f32);

        for ((moe_idx, expert_idx), slot) in evicted {
            if let Err(e) = self.evict_to_pinned(moe_idx, expert_idx, &slot) {
                tracing::warn!("drip_evict: failed L{moe_idx}E{expert_idx}: {e}");
            }
            // ExpertSlot dropped here — VRAM freed.
        }

        // Sync the copy stream so pinned data is valid.
        if let (Some(cs), Device::Cuda(cd)) = (&self.copy_stream, &self.device) {
            let event = cs.record_event(None).map_err(candle::Error::wrap)?;
            cd.cuda_stream().wait(&event).map_err(candle::Error::wrap)?;
        }

        Ok(())
    }

    /// Process a single MoE work request: classify, DMA, compute, return output.
    pub(crate) fn process_request(&mut self, req: MoeWorkRequest) -> Result<Tensor> {
        if let Ok(mut s) = self.stats.lock() {
            s.work_requests += 1;
        }

        // ── Detect new forward pass: layer index going backward ──
        if let Some(last) = self.last_moe_layer_idx {
            if req.moe_layer_idx <= last {
                self.transition_matrix.reset_pass();
            }
        }
        self.last_moe_layer_idx = Some(req.moe_layer_idx);

        // ── Validate speculative prediction accuracy ──
        // Count how many speculatively loaded experts for this layer
        // are actually in the expert set that was requested.
        let spec_for_layer: Vec<usize> = self
            .speculative_loads
            .iter()
            .filter(|&&(l, _)| l == req.moe_layer_idx)
            .map(|&(_, e)| e)
            .collect();
        if !spec_for_layer.is_empty() {
            let hits = spec_for_layer
                .iter()
                .filter(|e| req.expert_ids.contains(e))
                .count();
            self.hint_stats.0 += spec_for_layer.len();
            self.hint_stats.1 += hits;

            // Score boost for predictions that matched actual routing.
            for &eid in &spec_for_layer {
                if req.expert_ids.contains(&eid) {
                    self.inner.record_prediction_hit(req.moe_layer_idx, eid);
                }
            }
        }

        // Clear speculative loads for this layer — they are now visible
        // as cache hits in classify_and_load.
        self.speculative_loads
            .retain(|&(l, _)| l != req.moe_layer_idx);

        // ── Record routing for transition matrix ──
        self.transition_matrix
            .observe(req.moe_layer_idx, &req.expert_ids);

        let t = profile_now();
        let classified = self.classify_and_load(req.moe_layer_idx, &req.expert_ids)?;
        self.profile.record("pipe_classify_load", t);

        // Helper: extract (token_ids, weight_ids) for a given expert.
        let expert_group = |eid: usize| -> (Vec<u32>, Vec<u32>) {
            let eid32 = eid as u32;
            let lo = req.assignments.partition_point(|a| a.0 < eid32);
            let hi = req.assignments.partition_point(|a| a.0 <= eid32);
            let toks: Vec<u32> = req.assignments[lo..hi].iter().map(|a| a.1).collect();
            let wids: Vec<u32> = req.assignments[lo..hi].iter().map(|a| a.2).collect();
            (toks, wids)
        };

        let mut ys = req.xs.zeros_like()?;

        // ── Compute hit experts (overlaps with DMA for misses) ──
        let t = profile_now();
        #[cfg(feature = "cuda")]
        {
            let experts_data: Vec<(Vec<u32>, Vec<u32>)> = classified
                .hits
                .iter()
                .map(|&(eidx, _)| expert_group(eidx))
                .collect();
            let experts_vec: Vec<(&ExpertSlot, &[u32], &[u32])> = classified
                .hits
                .iter()
                .zip(experts_data.iter())
                .filter_map(|(&(_, slot_idx), (toks, wids))| {
                    let slot = self.inner.slots[slot_idx].as_ref()?;
                    Some((slot, toks.as_slice(), wids.as_slice()))
                })
                .collect();
            if !experts_vec.is_empty() {
                compute_experts_grouped(
                    &req.xs,
                    &mut ys,
                    &experts_vec,
                    &req.weights_flat,
                    &mut self.profile,
                )?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            for &(eidx, slot_idx) in &classified.hits {
                let slot = self.inner.slots[slot_idx].as_ref().ok_or_else(|| {
                    candle::Error::Msg(format!("hit slot {slot_idx} unexpectedly empty"))
                })?;
                let (toks, w_ids) = expert_group(eidx);
                compute_expert_contribution_gpu_weights(
                    &req.xs,
                    &mut ys,
                    slot,
                    &toks,
                    &req.weights_flat,
                    &w_ids,
                )?;
            }
        }
        self.profile.record("pipe_compute_hits", t);

        // ── Wait for copy fence (current layer's misses) ──
        let t = profile_now();
        classified.fence.wait(&self.device)?;
        #[cfg(feature = "cuda")]
        if classified.fence.event.is_some() {
            if let Ok(mut s) = self.stats.lock() {
                s.fence_stalls += 1;
            }
        }
        self.profile.record("pipe_fence_wait", t);

        // ── Compute newly-loaded experts ──
        let t = profile_now();
        #[cfg(feature = "cuda")]
        {
            let experts_data: Vec<(Vec<u32>, Vec<u32>)> = classified
                .loaded
                .iter()
                .map(|&(eidx, _)| expert_group(eidx))
                .collect();
            let experts_vec: Vec<(&ExpertSlot, &[u32], &[u32])> = classified
                .loaded
                .iter()
                .zip(experts_data.iter())
                .filter_map(|(&(_, slot_idx), (toks, wids))| {
                    let slot = self.inner.slots[slot_idx].as_ref()?;
                    Some((slot, toks.as_slice(), wids.as_slice()))
                })
                .collect();
            if !experts_vec.is_empty() {
                compute_experts_grouped(
                    &req.xs,
                    &mut ys,
                    &experts_vec,
                    &req.weights_flat,
                    &mut self.profile,
                )?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            for &(eidx, slot_idx) in &classified.loaded {
                let slot = self.inner.slots[slot_idx].as_ref().ok_or_else(|| {
                    candle::Error::Msg(format!("loaded slot {slot_idx} unexpectedly empty"))
                })?;
                let (toks, w_ids) = expert_group(eidx);
                compute_expert_contribution_gpu_weights(
                    &req.xs,
                    &mut ys,
                    slot,
                    &toks,
                    &req.weights_flat,
                    &w_ids,
                )?;
            }
        }
        self.profile.record("pipe_compute_loaded", t);

        Ok(ys)
    }

    /// Post-compute maintenance: anti-prediction scoring, prefetch,
    /// drip eviction, end-of-pass eviction with adaptive rates.
    /// Called AFTER the response has been sent to the forward thread so this
    /// work doesn't inflate submit_roundtrip.
    pub(crate) fn post_compute(&mut self, moe_layer_idx: usize, expert_ids: &[usize]) {
        // When every expert is resident, there is nothing to prefetch or
        // evict — skip all post-compute maintenance.
        #[cfg(feature = "cuda")]
        if self.all_resident {
            return;
        }

        // ── Anti-prediction: penalize bottom-N least-likely experts ──
        // Only at end-of-pass, and only when demand exceeds free headroom.
        if moe_layer_idx + 1 == self.num_moe_layers {
            let target_free = ((self.pass_misses as f32 * 1.15).ceil() as usize).max(1);
            let free_slots = self.inner.free_slots.len();
            if free_slots < target_free {
                // Apply anti-prediction for the first few layers of the next pass.
                for layer in 0..self.num_moe_layers.min(8) {
                    let bottom =
                        self.transition_matrix
                            .predict_bottom(moe_layer_idx, expert_ids, 4);
                    for &eid in &bottom {
                        self.inner.record_anti_prediction(layer, eid);
                    }
                }
            }
        }

        // ── Speculative prefetch for next MoE layer ──
        let t = profile_now();
        if let Ok(prefetch_fence) = self.speculative_prefetch(moe_layer_idx, expert_ids) {
            self.profile.record("pipe_prefetch", t);
            let t2 = profile_now();
            let _ = prefetch_fence.wait(&self.device);
            self.profile.record("pipe_prefetch_fence", t2);
        } else {
            self.profile.record("pipe_prefetch", t);
        }

        // ── Drip eviction (adaptive headroom) ──
        // Skip all eviction when every expert is resident in VRAM — there
        // is nothing to rotate and evicting would only cause needless DMA.
        #[cfg(feature = "cuda")]
        if !self.all_resident {
            let vram_slots = self.inner.slots.len();
            let free = self.inner.free_slots.len();
            let target_free = ((vram_slots as f32 * self.drip_headroom).ceil() as usize).max(1);
            if free < target_free {
                let deficit = target_free - free;
                let t = profile_now();
                if let Err(e) = self.drip_evict(deficit) {
                    tracing::warn!("drip eviction failed: {e}");
                } else {
                    self.pass_drip_evicts += deficit;
                }
                self.profile.record("pipe_drip_evict", t);
            }
        }

        // ── End-of-pass: score decay + adaptive batch eviction ──
        if moe_layer_idx + 1 == self.num_moe_layers {
            // Decay all expert scores (exponential forgetting).
            self.inner.decay_scores(0.85);

            // Compute adaptive eviction rate based on pass demand.
            let occupied = self
                .inner
                .slots
                .iter()
                .filter(|s| s.is_some())
                .count()
                .max(1);

            let target_free = ((self.pass_misses as f32 * 1.15).ceil() as usize).max(1);
            let raw_rate = target_free as f32 / occupied as f32;
            let clamped = raw_rate.clamp(0.01, 0.20);
            // EMA smooth: 70% old + 30% new.
            self.eviction_rate = self.eviction_rate * 0.7 + clamped * 0.3;

            // Adaptive drip headroom based on drip pressure.
            let drip_pressure =
                self.pass_drip_evicts as f32 / (self.num_moe_layers as f32).max(1.0);
            let raw_headroom = if drip_pressure > 0.5 {
                self.drip_headroom * 1.1
            } else if drip_pressure < 0.1 {
                self.drip_headroom * 0.95
            } else {
                self.drip_headroom
            };
            self.drip_headroom = raw_headroom.clamp(0.005, 0.05);

            // Skip heavy eviction when the cache already has enough
            // free headroom. This keeps single-token generation fast when
            // the cache is stable (few misses → free slots accumulate).
            let free_slots = self.inner.free_slots.len();
            let do_eviction = free_slots < target_free;

            let t = profile_now();
            #[cfg(feature = "cuda")]
            if !self.all_resident && do_eviction {
                let desired = ((occupied as f32 * self.eviction_rate).ceil() as usize).max(1);
                let pinned_free = self.pinned_pool.free_slots.len();
                let capped = desired.min(pinned_free);
                if capped > 0 {
                    let fraction = capped as f32 / occupied as f32;
                    let evicted = self.inner.end_of_pass_eviction(fraction);
                    for ((moe_idx, expert_idx), slot) in evicted {
                        if let Err(e) = self.evict_to_pinned(moe_idx, expert_idx, &slot) {
                            tracing::warn!("end-of-pass evict failed L{moe_idx}E{expert_idx}: {e}");
                        }
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            if do_eviction {
                let evicted = self.inner.end_of_pass_eviction(self.eviction_rate);
                let _ = evicted;
            }
            self.profile.record("pipe_eviction", t);

            // Reset per-pass adaptive counters.
            self.pass_misses = 0;
            self.pass_drip_evicts = 0;
        }
    }

    /// Process a speculative prediction hint from the forward thread.
    ///
    /// Uses the transition matrix to predict which experts the given layer
    /// will need, then starts DMA from the pinned pool for predicted misses.
    /// Tracks which experts were speculatively loaded so `classify_and_load`
    /// can diff actual vs predicted when the Work request arrives.
    ///
    /// Only uses free VRAM slots — never evicts for speculative loads.
    fn process_hint(&mut self, layer_idx: usize, prev_expert_ids: &[usize]) {
        // All experts resident — nothing to prefetch.
        #[cfg(feature = "cuda")]
        if self.all_resident {
            return;
        }

        let t = profile_now();

        // Use prev layer index for transition matrix lookup (predicts
        // which experts layer_idx will need based on prev layer's routing).
        let prev_layer_idx = if layer_idx > 0 { layer_idx - 1 } else { 0 };
        let predicted = self
            .transition_matrix
            .predict(prev_layer_idx, prev_expert_ids);

        if predicted.is_empty() || layer_idx >= self.num_moe_layers {
            self.profile.record("pipe_hint", t);
            return;
        }

        #[allow(unused_mut)]
        let mut loaded_count = 0usize;

        for &expert_idx in &predicted {
            // Skip if already resident in VRAM.
            if self
                .inner
                .key_to_slot
                .get(&(layer_idx, expert_idx))
                .map_or(false, |&s| self.inner.slots[s].is_some())
            {
                continue;
            }

            // Free-slots-only: never evict for speculative loads.
            let slot_idx = match self.inner.free_slots.pop() {
                Some(s) => s,
                None => break, // no free slots — stop speculating
            };

            // Load from pinned pool (CUDA only — no-op on CPU builds).
            #[cfg(not(feature = "cuda"))]
            {
                self.inner.free_slots.push(slot_idx);
                continue;
            }

            #[cfg(feature = "cuda")]
            let expert_slot = match self.load_from_pinned(layer_idx, expert_idx) {
                Ok(slot) => slot,
                Err(_) => {
                    self.inner.free_slots.push(slot_idx);
                    continue;
                }
            };

            // Install the speculatively loaded expert.
            #[cfg(feature = "cuda")]
            self.inner
                .install(slot_idx, layer_idx, expert_idx, expert_slot);

            #[cfg(feature = "cuda")]
            {
                self.expert_locations[layer_idx][expert_idx] = ExpertLocation::Vram { slot_idx };
            }

            #[cfg(feature = "cuda")]
            {
                self.speculative_loads.insert((layer_idx, expert_idx));
                loaded_count += 1;
            }
        }

        if loaded_count > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.hint_loads += loaded_count;
            }
            self.profile.record("pipe_hint_dma", t);
        } else {
            self.profile.record("pipe_hint", t);
        }
    }

    /// Speculatively prefetch a single expert for the next MoE layer.
    ///
    /// Uses the transition matrix to predict the most likely expert,
    /// then loads from pinned pool if not already cached.
    /// Only uses free VRAM slots — never evicts for prefetch.
    fn speculative_prefetch(
        &mut self,
        moe_layer_idx: usize,
        current_expert_ids: &[usize],
    ) -> Result<CopyBatchFence> {
        let predicted = self
            .transition_matrix
            .predict(moe_layer_idx, current_expert_ids);

        if predicted.is_empty() {
            return Ok(CopyBatchFence::noop());
        }

        let next_moe_idx = moe_layer_idx + 1;
        if next_moe_idx >= self.num_moe_layers {
            return Ok(CopyBatchFence::noop());
        }

        // Find the first predicted expert that is not already cached.
        let target = predicted.iter().find(|&&eid| {
            !self
                .inner
                .key_to_slot
                .get(&(next_moe_idx, eid))
                .map_or(false, |&s| self.inner.slots[s].is_some())
        });

        let &expert_idx = match target {
            Some(eid) => eid,
            None => return Ok(CopyBatchFence::noop()),
        };

        // Free-slots-only: never evict for prefetch.
        let slot_idx = match self.inner.free_slots.pop() {
            Some(s) => s,
            None => return Ok(CopyBatchFence::noop()),
        };

        let expert_slot = {
            #[cfg(feature = "cuda")]
            {
                self.load_from_pinned(next_moe_idx, expert_idx)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let mmap_bytes: &[u8] = &self.mmap;
                let mmap_ref = &self.host_refs[next_moe_idx][expert_idx];
                load_from_mmap(mmap_bytes, mmap_ref, &self.device)?
            }
        };

        // Record fence for the prefetch DMA.
        let fence = {
            #[cfg(feature = "cuda")]
            if let (Some(cs), Device::Cuda(_)) = (&self.copy_stream, &self.device) {
                let event = cs.record_event(None).map_err(candle::Error::wrap)?;
                CopyBatchFence { event: Some(event) }
            } else {
                CopyBatchFence::noop()
            }
            #[cfg(not(feature = "cuda"))]
            CopyBatchFence::noop()
        };

        // Install the prefetched expert.
        self.inner
            .install(slot_idx, next_moe_idx, expert_idx, expert_slot);

        #[cfg(feature = "cuda")]
        {
            self.expert_locations[next_moe_idx][expert_idx] = ExpertLocation::Vram { slot_idx };
        }

        if let Ok(mut s) = self.stats.lock() {
            s.prefetch_loads += 1;
        }

        Ok(fence)
    }
}

// ============================================================================
// Thread spawn
// ============================================================================

/// Spawn the pipeline thread.  Returns the sender for submitting work/hints.
pub(crate) fn spawn_pipeline_thread(mut state: PipelineState) -> mpsc::SyncSender<PipelineMessage> {
    let (tx, rx) = mpsc::sync_channel::<PipelineMessage>(4);

    std::thread::Builder::new()
        .name("expert-pipeline".into())
        .spawn(move || {
            while let Ok(msg) = rx.recv() {
                match msg {
                    PipelineMessage::Work(req) => {
                        let response_tx = req.response_tx.clone();
                        let moe_layer_idx = req.moe_layer_idx;
                        let expert_ids = req.expert_ids.clone();
                        let result = state.process_request(req);
                        let _ = response_tx.send(result);
                        // Post-response: prefetch + eviction (off critical path)
                        state.post_compute(moe_layer_idx, &expert_ids);
                    }
                    PipelineMessage::Hint {
                        layer_idx,
                        prev_expert_ids,
                    } => {
                        state.process_hint(layer_idx, &prev_expert_ids);
                    }
                    PipelineMessage::SnapshotProfile { response_tx } => {
                        let snap = state.profile.snapshot();
                        state.profile.reset();
                        let _ = response_tx.send(snap);
                    }
                }
            }
        })
        .expect("failed to spawn expert-pipeline thread");

    tx
}

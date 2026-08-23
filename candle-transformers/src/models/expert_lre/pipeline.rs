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
//! When an expert is evicted from VRAM (`evict()` or `demand_eviction`),
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
use super::pack::{ExpertPack, PackRead, PackWriter, RecordLayout};
#[cfg(feature = "cuda")]
use super::pinned::{ExpertResidency, LayerGeometry, WarmPool};
#[cfg(feature = "cuda")]
use super::streamer::{StreamDone, StreamJob, StreamPlan, StreamerHandle};
use super::transition::TransitionMatrix;
#[cfg(not(feature = "cuda"))]
use super::types::MoeInput;
use super::types::{
    ClassifiedExperts, CopyBatchFence, ExpertSlot, MmapExpertRef, MoeWorkRequest, PipelineMessage,
    PipelineStats,
};
use crate::models::profile::{profile_now, ProfileAccumulator};
#[cfg(feature = "cuda")]
use crate::models::wave_buffers::wave_zeros_ticketed;
#[cfg(feature = "cuda")]
use candle::direct_io::AlignedScratch;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::PinnedBuf;
#[cfg(not(feature = "cuda"))]
use candle::quantized::Int8Mode;
use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{kv_spare_regions, set_weight_floor, weight_floor_after};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaEvent, CudaStream};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

// ============================================================================
// Dynamic load-ahead (prefetch depth N) controller constants
// ============================================================================

/// Upper clamp on the dynamic prefetch depth `N`. Adjacent-chained prediction
/// compounds precision per hop (~0.95×/hop from the 93–97% steady state), so
/// beyond a few hops the extra DMA is mostly waste even when latency-bound.
const PREFETCH_DEPTH_MAX: usize = 4;

/// Deepen only when a pass accumulated at least this many late-load experts.
/// A single straggler in one pass is noise — deepening the GLOBAL depth on it
/// puts extra hop batches on the pipe thread for every subsequent layer,
/// which costs wide waves more than the one late expert ever did.
const PREFETCH_LATE_FLOOR: usize = 2;

/// Chained hops beyond `L+1` are issued only for decode-narrow source sets
/// (at most this many routed experts). A prefill-width wave routes hundreds
/// of experts per layer; running multi-hop loads for it on the pipe thread
/// inside `post_compute` queues the next layer's work request behind the
/// batch — the measured full-next-layer dead end (cfg8 437→416) — while its
/// long compute window already hides single-hop latency (late ≈ 0 at width).
const PREFETCH_MULTI_HOP_MAX_SOURCES: usize = 32;

/// Upper bound on one whole-layer stream plan. Bounds the plan's bus time to
/// roughly one prefill layer's compute window (~64 slots × ~14 MB ≈ 0.9 GB ≈
/// tens of ms at effective PCIe rates) — a bigger plan cannot land before its
/// layer's request and only queues bytes in front of the demand path.
const STREAM_MAX_JOBS: usize = 64;

/// Deepen only when the pass's achieved copy bandwidth sits below this
/// fraction of the empirical ceiling. Late loads WITHOUT slack mean the link
/// is saturated — bandwidth-bound, not latency-bound — and issuing earlier
/// would only queue more bytes in front of the ones already late.
const PREFETCH_BW_SLACK: f32 = 0.8;

/// Shallow back when a pass finishes with zero late loads and its prediction
/// precision fell below this floor: the extra hops are landing in time but
/// loading the wrong experts, so the depth is buying waste, not coverage.
const PREFETCH_PRECISION_FLOOR: f32 = 0.8;

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
// Startup: build or reuse the pack, then fill the warm and hot tiers from it
// ============================================================================

/// Everything the startup fill writes into, gathered so the two entry points
/// below take one argument each instead of six.
#[cfg(feature = "cuda")]
pub(crate) struct StartupTargets<'a> {
    pub inner: &'a mut ExpertCacheInner,
    pub warm: &'a mut WarmPool,
    pub residency: &'a mut [Vec<ExpertResidency>],
    /// Warm slot `i` holds `membership[i]`, decided once by the stratified draw.
    pub membership: &'a [(usize, usize)],
    pub geoms: &'a [LayerGeometry],
    pub layouts: &'a [RecordLayout],
    pub stride: usize,
}

/// Repack every expert out of the GGUF, write the pack, and fill both resident
/// tiers from the bytes as they pass through.
///
/// This is the first-boot path and the only one that pays the ~42 s repack. It
/// writes every expert to the pack — including the ones that go straight to
/// VRAM — because the pack is authoritative: an expert that exists only in a
/// VRAM slot could not be evicted without losing it, which is the defect this
/// whole design removes.
#[cfg(feature = "cuda")]
pub(crate) fn startup_repack(
    t: StartupTargets<'_>,
    writer: &mut PackWriter,
    mmap: &[u8],
    host_refs: &[Vec<MmapExpertRef>],
    cuda_dev: &candle::CudaDevice,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    let num_moe_layers = host_refs.len();
    let num_experts = host_refs.first().map_or(0, |l| l.len());
    if num_moe_layers == 0 || num_experts == 0 {
        return Ok(());
    }
    let total_experts = num_moe_layers * num_experts;
    // Warm slot for each expert, so the fill can place bytes as they go by
    // rather than re-reading them afterwards.
    let mut warm_slot_of: Vec<Vec<Option<usize>>> = vec![vec![None; num_experts]; num_moe_layers];
    for (slot, &(layer, expert)) in t.membership.iter().enumerate() {
        warm_slot_of[layer][expert] = Some(slot);
    }

    tracing::info!(
        target: "candle_transformers::expert_lre",
        layers = num_moe_layers,
        experts = num_experts,
        vram_slots = t.inner.num_slots(),
        warm_slots = t.warm.num_slots(),
        "startup: repacking every expert into the pack, filling both tiers as it goes"
    );
    let t0 = std::time::Instant::now();
    let mut vram_count = 0usize;
    let mut warm_count = 0usize;

    for moe_idx in 0..num_moe_layers {
        let geom = &t.geoms[moe_idx];
        let layout = t.layouts[moe_idx];
        for expert_idx in 0..num_experts {
            let r = &host_refs[moe_idx][expert_idx];
            // A repack that fails leaves an expert with no valid record, and a
            // record of zeroes reads back as a plausible expert. There is no
            // partial answer here: the pack is authoritative or it is nothing.
            let (gate, up, down) = repack_expert_projections(mmap, r, geom, cuda_dev)?;
            writer.write_expert(moe_idx, expert_idx, &gate, &up, &down)?;

            let mut res = ExpertResidency::default();
            if let Some(warm_slot) = warm_slot_of[moe_idx][expert_idx] {
                let dst = t.warm.slot_mut(warm_slot, t.stride);
                write_record(dst, layout, &gate, &up, &down);
                res.ram = Some(warm_slot);
                warm_count += 1;
            }
            // The zone hands out the rightmost free slot, so the startup fill
            // packs experts against the span's right edge and the volatile left
            // margin is the last ground to be taken.
            if let Some(slot_idx) = t.inner.take_free() {
                let slot_base = t.inner.slot_base(slot_idx);
                // SAFETY: `slot_idx` was just handed out by the zone and is not
                // reclaimed until an eviction returns it.
                let slot = unsafe {
                    build_slot_from_repacked_with_device(
                        &gate, &up, &down, geom, cuda_dev, slot_base,
                    )?
                };
                t.inner.install(slot_idx, moe_idx, expert_idx, slot);
                res.vram = Some(slot_idx);
                vram_count += 1;
            }
            t.residency[moe_idx][expert_idx] = res;
            if let Some(cb) = progress {
                cb(moe_idx * num_experts + expert_idx + 1, total_experts);
            }
        }
        if (moe_idx + 1) % 8 == 0 || moe_idx + 1 == num_moe_layers {
            tracing::info!(
                target: "candle_transformers::expert_lre",
                layer = moe_idx + 1,
                of = num_moe_layers,
                vram_count,
                warm_count,
                "startup: repacking"
            );
        }
    }

    tracing::info!(
        target: "candle_transformers::expert_lre",
        secs = t0.elapsed().as_secs_f64(),
        vram_count,
        warm_count,
        warm_gib = t.warm.total_bytes() as f64 / 1e9,
        "startup: repack complete"
    );
    Ok(())
}

/// Fill both resident tiers from a pack that already exists.
///
/// The restart path, and the reason the pack is worth its 16.6 GiB: no repack,
/// no GGUF expert reads at all. Only the experts that land somewhere are read —
/// the rest stay on disk until something asks for them.
#[cfg(feature = "cuda")]
pub(crate) fn startup_from_pack(
    t: StartupTargets<'_>,
    pack: &ExpertPack,
    num_moe_layers: usize,
    num_experts: usize,
    cuda_dev: &candle::CudaDevice,
    progress: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    if num_moe_layers == 0 || num_experts == 0 {
        return Ok(());
    }
    let total_experts = num_moe_layers * num_experts;
    let t0 = std::time::Instant::now();

    // ── Warm tier: every membership record at once, at full queue depth ──
    //
    // The pool's slots are cut to the pack's stride, so each read lands in its
    // final home with nothing in between: no staging buffer, no host-to-host
    // copy, one NVMe DMA per expert.
    if !t.membership.is_empty() {
        let stride = t.stride;
        let mut rest = t.warm.span_mut(0, t.membership.len());
        let mut reads: Vec<PackRead<'_>> = Vec::with_capacity(t.membership.len());
        for &(layer, expert) in t.membership.iter() {
            let (head, tail) = rest.split_at_mut(stride);
            reads.push(PackRead {
                layer,
                expert,
                dest: head,
            });
            rest = tail;
        }
        pack.read_many(reads)?;
        for (slot, &(layer, expert)) in t.membership.iter().enumerate() {
            t.residency[layer][expert].ram = Some(slot);
        }
    }
    tracing::info!(
        target: "candle_transformers::expert_lre",
        warm_slots = t.membership.len(),
        secs = t0.elapsed().as_secs_f64(),
        "startup: warm tier filled from the pack"
    );

    // ── Hot tier: fill VRAM in layer order, from warm where possible ──
    let mut staging = ColdStaging::new(t.stride, COLD_STAGING_BUFFERS)?;
    let stream = cuda_dev.cuda_stream();
    let mut vram_count = 0usize;
    let mut cold_reads = 0usize;
    'fill: for moe_idx in 0..num_moe_layers {
        let geom = &t.geoms[moe_idx];
        let layout = t.layouts[moe_idx];
        for expert_idx in 0..num_experts {
            let Some(slot_idx) = t.inner.take_free() else {
                break 'fill;
            };
            let slot_base = t.inner.slot_base(slot_idx);
            // SAFETY (both arms): `slot_idx` was just handed out by the zone and
            // is not reclaimed while this runs.
            let slot = match t.residency[moe_idx][expert_idx].ram {
                // The warm pool is written once and never again, so it is a
                // source no later write can race — no event, no wait.
                Some(warm_slot) => unsafe {
                    build_slot_from_record_with_device(
                        t.warm.slot_ref(warm_slot, t.stride),
                        layout,
                        geom,
                        cuda_dev,
                        slot_base,
                    )?
                },
                None => {
                    let idx = staging.acquire()?;
                    pack.read_into(moe_idx, expert_idx, staging.buffer_mut(idx, t.stride))?;
                    let slot = unsafe {
                        build_slot_from_record_with_device(
                            staging.buffer_ref(idx, t.stride),
                            layout,
                            geom,
                            cuda_dev,
                            slot_base,
                        )?
                    };
                    staging.publish(idx, stream.record_event(None).map_err(candle::Error::wrap)?);
                    cold_reads += 1;
                    slot
                }
            };
            t.inner.install(slot_idx, moe_idx, expert_idx, slot);
            t.residency[moe_idx][expert_idx].vram = Some(slot_idx);
            vram_count += 1;
            if let Some(cb) = progress {
                cb(moe_idx * num_experts + expert_idx + 1, total_experts);
            }
        }
    }
    // The uploads above are asynchronous against pinned buffers this function
    // is about to drop.
    stream.synchronize().map_err(candle::Error::wrap)?;
    // The fill stops as soon as VRAM is full, so the remaining experts never
    // reach the progress callback. Land it on the total so a UI bar completes.
    if let Some(cb) = progress {
        cb(total_experts, total_experts);
    }

    tracing::info!(
        target: "candle_transformers::expert_lre",
        secs = t0.elapsed().as_secs_f64(),
        vram_count,
        cold_reads,
        warm_slots = t.membership.len(),
        warm_gib = t.warm.total_bytes() as f64 / 1e9,
        staging_mib = staging.total_bytes() as f64 / 1e6,
        pack = %pack.path().display(),
        "startup: filled from the pack — no repack this boot"
    );
    Ok(())
}

/// Pinned landing buffers the pipeline thread keeps for cold reads.
///
/// Deep enough that a buffer is never waited on in practice: it must exceed
/// ONE LAYER'S worst cold burst, not just its routed width. A wide prefill
/// wave misses tens of experts per layer past the warm tier, and a ring
/// shallower than that burst rewraps onto buffers whose uploads were
/// published microseconds earlier — whose events sit behind the copy stream's
/// ordered-after-compute waits, so each `acquire` becomes a host sync against
/// compute (measured: a 16-deep ring turned the batched cold read into a 5×
/// prefill collapse). At 64 records (~0.9 GB pinned on DeepSeek's 14.2 MB
/// stride) a burst never rewraps within its own layer, and by the next visit
/// every event has long retired — still a rounding error against the warm
/// tier, and the thing that keeps a cold miss from stalling the host.
#[cfg(feature = "cuda")]
pub(crate) const COLD_STAGING_BUFFERS: usize = 64;

/// Lay one expert's three projections into a record buffer at their spans.
///
/// The gaps between projections are alignment padding the kernels never read;
/// they are zeroed so a record is a deterministic function of its expert, which
/// is what lets the pack file be compared byte for byte between builds.
#[cfg(feature = "cuda")]
fn write_record(dst: &mut [u8], layout: RecordLayout, gate: &[u8], up: &[u8], down: &[u8]) {
    dst.fill(0);
    for (span, src) in [(layout.gate, gate), (layout.up, up), (layout.down, down)] {
        dst[span.offset..span.offset + src.len()].copy_from_slice(src);
    }
}

/// GPU-repack one expert's three projections from GGML to K/128 format.
///
/// Returns `(gate_bytes, up_bytes, down_bytes)` as host `Vec<u8>`.
#[cfg(feature = "cuda")]
pub(crate) fn repack_expert_projections(
    mmap: &[u8],
    r: &MmapExpertRef,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let gate_ggml = &mmap[r.gate_offset..r.gate_offset + r.gate_len];
    let up_ggml = &mmap[r.up_offset..r.up_offset + r.up_len];
    let down_ggml = &mmap[r.down_offset..r.down_offset + r.down_len];

    // `r.*_dtype` is the compact GGUF source; `geom.*_dtype` is the target the pinned pool caches
    // (== source for gemx; the KO twin for int8). For KO this repacks Q4_K→KO ONCE per expert.
    let gate_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        gate_ggml,
        geom.gate_shape[0],
        geom.gate_shape[1],
        r.gate_dtype,
        geom.gate_dtype,
    )?;
    let up_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        up_ggml,
        geom.up_shape[0],
        geom.up_shape[1],
        r.up_dtype,
        geom.up_dtype,
    )?;
    let down_repacked = candle::quantized::repack_to_host(
        cuda_dev,
        down_ggml,
        geom.down_shape[0],
        geom.down_shape[1],
        r.down_dtype,
        geom.down_dtype,
    )?;

    Ok((gate_repacked, up_repacked, down_repacked))
}

/// Alignment every projection's base within a slot is rounded up to — what the
/// tensor-core paths require of an operand.
#[cfg(feature = "cuda")]
const PROJECTION_ALIGN: usize = 256;

/// Byte offsets of gate / up / down within one expert slot, and the aligned
/// total.
///
/// A slot is one range of the weight zone holding all three projections, so the
/// zone can stay an array of equal-sized units — the property that makes "the
/// rightmost free spot" a single index and a relocation a memcpy. The three sit
/// at aligned offsets inside it.
///
/// The total is what [`slot_bytes_for`] maxes over layers, so a slot always
/// holds the widest layer's three projections and any layer's fit in any slot.
#[cfg(feature = "cuda")]
pub(crate) fn slot_offsets(geom: &LayerGeometry) -> (usize, usize, usize, usize) {
    let align = |n: usize| n.div_ceil(PROJECTION_ALIGN) * PROJECTION_ALIGN;
    let up = align(geom.gate_repacked_size);
    let down = up + align(geom.up_repacked_size);
    let total = down + align(geom.down_repacked_size);
    (0, up, down, total)
}

/// Bytes one zone slot must be to hold any layer's expert.
///
/// The max over layers of the aligned three-projection total. `max_expert_size`
/// used to be the max over the *unaligned* sum, which is the same number the
/// pinned pool uses for its own slots; this is that number plus at most two
/// alignments, and it is the one the zone is carved with.
#[cfg(feature = "cuda")]
pub(crate) fn slot_bytes_for(geoms: &[LayerGeometry]) -> usize {
    geoms.iter().map(|g| slot_offsets(g).3).max().unwrap_or(0)
}

/// Regions the weight side leaves above the KV side's recent high-water mark.
///
/// Headroom for the KV side to grow into without the weight side having to give
/// anything back — 32 regions is 512 MiB, which is several turns' worth at the
/// measured steady state. Taking right up to the mark would make every small
/// increase in KV demand a boundary move, and a boundary move costs an eviction.
///
/// Deriving this from the measured per-forward claim count was tried and is not
/// an improvement: on the quantized path that count is dominated by the
/// compressor creating size-class arenas as it works, so the "headroom" it
/// produces is the same order as the KV side itself and the weight side is
/// offered nothing at all.
#[cfg(feature = "cuda")]
const KV_REGION_SLACK: usize = 32;

/// Wrap an already-populated slot at `slot_base` as an `ExpertSlot`, without
/// moving any bytes.
///
/// For a relocation, where the copy has already put the bytes in place: the
/// three `QMatMul`s hold device pointers, so a slot that moves needs its
/// storages rebuilt over the new address even though its contents are identical.
///
/// # Safety
///
/// `slot_base` must name a slot the zone has handed out, already holding this
/// layer's three projections at [`slot_offsets`].
#[cfg(feature = "cuda")]
unsafe fn build_slot_view(
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    let (gate_off, up_off, down_off, _) = slot_offsets(geom);
    let view = |off: usize, bytes: usize, dtype, shape: &Vec<usize>| -> Result<_> {
        let storage =
            candle::quantized::view_repacked(cuda_dev, slot_base + off as u64, bytes, dtype)?;
        candle::quantized::QTensor::new(storage, shape.clone())
    };
    let gate_qt = view(
        gate_off,
        geom.gate_repacked_size,
        geom.gate_dtype,
        &geom.gate_shape,
    )?;
    let up_qt = view(up_off, geom.up_repacked_size, geom.up_dtype, &geom.up_shape)?;
    let down_qt = view(
        down_off,
        geom.down_repacked_size,
        geom.down_dtype,
        &geom.down_shape,
    )?;
    Ok(ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    })
}

/// Build an `ExpertSlot` from already-repacked host bytes, uploading them into
/// the weight-zone slot at `slot_base`.
///
/// # Safety
///
/// `slot_base` must name a slot the zone has handed out and not reclaimed, of at
/// least `slot_offsets(geom).3` bytes.
#[cfg(feature = "cuda")]
unsafe fn build_slot_from_repacked_with_device(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    build_slot_from_repacked_on_stream_inner(
        gate_bytes,
        up_bytes,
        down_bytes,
        geom,
        cuda_dev,
        &cuda_dev.cuda_stream(),
        slot_base,
        None,
    )
}

/// The one upload path, shared by the startup fill and the miss path.
///
/// Both used to allocate three buffers from the CUDA pool and hand ownership to
/// the storages. Now the three are views into a slot the zone owns: the upload
/// writes over whatever the previous tenant left, and dropping the storages
/// releases the views and not the memory
/// (`Backing::Lease(LeaseOrigin::Foreign)`). That is what makes an eviction a
/// bookkeeping change rather than a free, and a relocation a memcpy rather than
/// a reload.
///
/// # Safety
///
/// As [`build_slot_from_repacked_with_device`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn build_slot_from_repacked_on_stream_inner(
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    stream: &Arc<CudaStream>,
    slot_base: u64,
    profile: Option<&mut ProfileAccumulator>,
) -> Result<ExpertSlot> {
    let (gate_off, up_off, down_off, _) = slot_offsets(geom);
    let t = profile_now();
    let gate_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        stream,
        slot_base + gate_off as u64,
        gate_bytes,
        geom.gate_dtype,
    )?;
    let up_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        stream,
        slot_base + up_off as u64,
        up_bytes,
        geom.up_dtype,
    )?;
    let down_storage = candle::quantized::load_repacked_into(
        cuda_dev,
        stream,
        slot_base + down_off as u64,
        down_bytes,
        geom.down_dtype,
    )?;

    let gate_qt = candle::quantized::QTensor::new(gate_storage, geom.gate_shape.clone())?;
    let up_qt = candle::quantized::QTensor::new(up_storage, geom.up_shape.clone())?;
    let down_qt = candle::quantized::QTensor::new(down_storage, geom.down_shape.clone())?;

    let slot = ExpertSlot {
        gate_proj: QMatMul::from_qtensor_repacked(gate_qt)?,
        up_proj: QMatMul::from_qtensor_repacked(up_qt)?,
        down_proj: QMatMul::from_qtensor_repacked(down_qt)?,
    };
    if let Some(p) = profile {
        p.record("dma_h2d", t);
    }
    Ok(slot)
}

/// Build an `ExpertSlot` from one pack **record** — the form both resident
/// tiers hold — uploading it into the weight-zone slot at `slot_base`.
///
/// The record is the three projections at the offsets `layout` names, which is
/// the same arrangement a VRAM slot uses, so this is three subslices and the
/// ordinary upload. Everything downstream of the pack goes through here: warm
/// promotions, cold misses, and the startup fill.
///
/// # Safety
///
/// As [`build_slot_from_repacked_with_device`].
#[cfg(feature = "cuda")]
pub(crate) unsafe fn build_slot_from_record_on_stream(
    record: &[u8],
    layout: RecordLayout,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    stream: &Arc<CudaStream>,
    slot_base: u64,
    profile: Option<&mut ProfileAccumulator>,
) -> Result<ExpertSlot> {
    let at = |s: super::pack::RecordSpan| &record[s.offset..s.offset + s.bytes];
    build_slot_from_repacked_on_stream_inner(
        at(layout.gate),
        at(layout.up),
        at(layout.down),
        geom,
        cuda_dev,
        stream,
        slot_base,
        profile,
    )
}

/// [`build_slot_from_record_on_stream`] on the device's default stream.
///
/// # Safety
///
/// As [`build_slot_from_repacked_with_device`].
#[cfg(feature = "cuda")]
unsafe fn build_slot_from_record_with_device(
    record: &[u8],
    layout: RecordLayout,
    geom: &LayerGeometry,
    cuda_dev: &candle::CudaDevice,
    slot_base: u64,
) -> Result<ExpertSlot> {
    build_slot_from_record_on_stream(
        record,
        layout,
        geom,
        cuda_dev,
        &cuda_dev.cuda_stream(),
        slot_base,
        None,
    )
}

/// Pinned landing buffers for reads out of the cold tier.
///
/// A cold read has to land in pinned memory twice over: direct I/O needs a
/// sector-aligned destination, and the H2D that follows wants a source the DMA
/// engine can take without a bounce copy. `cuMemAllocHost` satisfies both.
///
/// The buffers rotate, and each carries the event for the upload it last fed.
/// **That event is why this is a ring and not one buffer**: the H2D is issued on
/// the copy stream and returns before the bytes have moved, so writing the next
/// record into the same buffer would race the copy reading it. Reusing a buffer
/// waits on its event first — a host-side wait, but only for a buffer that has
/// been round the ring since, which under any working warm tier is never.
/// One landing buffer: pinned if the machine will give it, sector-aligned host
/// memory otherwise.
///
/// **The fallback is not a nicety.** Both arms satisfy direct I/O's 4 KiB
/// alignment — `cuMemAllocHost` returns page-aligned memory and
/// [`AlignedScratch`] aligns explicitly — so the read works either way, and only
/// the H2D loses the no-bounce property. What it must never do is fall back to a
/// plain `Vec`, whose 8- or 16-byte alignment `ReadFile` with
/// `FILE_FLAG_NO_BUFFERING` rejects outright.
#[cfg(feature = "cuda")]
enum StagingBuf {
    Pinned(PinnedBuf),
    Aligned(AlignedScratch),
}

#[cfg(feature = "cuda")]
impl StagingBuf {
    fn alloc(len: usize) -> Result<Self> {
        match PinnedBuf::alloc_owned_default(len) {
            Ok(b) => Ok(Self::Pinned(b)),
            Err(_) => {
                let mut a = AlignedScratch::new();
                a.ensure(len).map_err(|e| {
                    candle::Error::Msg(format!("cold staging: aligned fallback of {len} B: {e}"))
                })?;
                Ok(Self::Aligned(a))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Pinned(b) => b.len(),
            Self::Aligned(a) => a.capacity(),
        }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        match self {
            Self::Pinned(b) => &mut b.as_mut_slice()[..len],
            Self::Aligned(a) => a.as_mut_slice(len),
        }
    }

    fn as_slice(&self, len: usize) -> &[u8] {
        match self {
            Self::Pinned(b) => &b.as_slice()[..len],
            Self::Aligned(a) => a.as_slice(len),
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct ColdStaging {
    bufs: Vec<StagingBuf>,
    /// The upload each buffer last fed, or `None` if it has not fed one since
    /// the last wait.
    events: Vec<Option<CudaEvent>>,
    next: usize,
}

#[cfg(feature = "cuda")]
impl ColdStaging {
    /// `count` buffers of `stride` bytes each.
    ///
    /// **Allocate this before the warm tier, not after.** It is small (46 MB)
    /// and mandatory; the warm tier is enormous and elastic, and halves itself
    /// when refused. Taking the elastic allocation first leaves this one to fail
    /// on a machine the warm tier has just filled — which is a model load that
    /// dies with `CUDA_ERROR_OUT_OF_MEMORY` where it should have been a slightly
    /// smaller warm tier. Mandatory-and-small first, elastic-and-large last.
    pub(crate) fn new(stride: usize, count: usize) -> Result<Self> {
        let mut bufs = Vec::with_capacity(count);
        for _ in 0..count {
            bufs.push(StagingBuf::alloc(stride)?);
        }
        Ok(Self {
            bufs,
            events: (0..count).map(|_| None).collect(),
            next: 0,
        })
    }

    /// Total bytes held, for the memory report.
    fn total_bytes(&self) -> usize {
        self.bufs.iter().map(|b| b.len()).sum()
    }

    /// The next buffer, once the upload it last fed has retired.
    fn acquire(&mut self) -> Result<usize> {
        let idx = self.next;
        self.next = (self.next + 1) % self.bufs.len();
        if let Some(event) = self.events[idx].take() {
            event.synchronize().map_err(candle::Error::wrap)?;
        }
        Ok(idx)
    }

    /// `n` distinct buffers (each waited-on if its last upload is still in
    /// flight), for one concurrent batch read. `n` must not exceed the ring.
    pub(crate) fn acquire_many(&mut self, n: usize) -> Result<Vec<usize>> {
        if n > self.bufs.len() {
            candle::bail!(
                "cold staging: {n} buffers requested, ring holds {}",
                self.bufs.len()
            );
        }
        (0..n).map(|_| self.acquire()).collect()
    }

    /// Mutable slices for a set of DISTINCT buffer indices, in the order given —
    /// the destinations of one `read_many` batch.
    pub(crate) fn buffers_mut_for(&mut self, idxs: &[usize], len: usize) -> Result<Vec<&mut [u8]>> {
        let mut taken: Vec<Option<&mut StagingBuf>> = self.bufs.iter_mut().map(Some).collect();
        idxs.iter()
            .map(|&i| {
                taken
                    .get_mut(i)
                    .and_then(|s| s.take())
                    .map(|b| b.as_mut_slice(len))
                    .ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "cold staging: buffer {i} requested twice or out of range"
                        ))
                    })
            })
            .collect()
    }

    /// Record that buffer `idx` is the source of an upload that has not landed.
    pub(crate) fn publish(&mut self, idx: usize, event: CudaEvent) {
        self.events[idx] = Some(event);
    }

    fn buffer_mut(&mut self, idx: usize, len: usize) -> &mut [u8] {
        self.bufs[idx].as_mut_slice(len)
    }

    pub(crate) fn buffer_ref(&self, idx: usize, len: usize) -> &[u8] {
        self.bufs[idx].as_slice(len)
    }
}

/// Device-direct prewarm: fill VRAM slots straight from the mmap. Non-CUDA-only — under CUDA the
/// experts are staged through the pack file, so this is unused there.
#[cfg(not(feature = "cuda"))]
pub(crate) fn prewarm_expert_cache(
    inner: &mut ExpertCacheInner,
    mmap: &Arc<memmap2::Mmap>,
    host_refs: &[Vec<MmapExpertRef>],
    device: &Device,
    mode: Int8Mode,
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
        total_slots,
        num_moe_layers,
        num_experts,
    );
    let t0 = std::time::Instant::now();
    let mut loaded_count: usize = 0;

    'fill: for moe_idx in 0..num_moe_layers {
        for expert_idx in 0..num_experts {
            let Some(slot_idx) = inner.take_free() else {
                break 'fill;
            };
            let mmap_bytes: &[u8] = mmap;
            let result = load_from_mmap(mmap_bytes, &host_refs[moe_idx][expert_idx], device, mode);
            match result {
                Ok(expert_slot) => {
                    inner.install(slot_idx, moe_idx, expert_idx, expert_slot);
                    let r = &host_refs[moe_idx][expert_idx];
                    mmap_evict_expert(mmap, r);
                    loaded_count += 1;
                }
                Err(e) => {
                    tracing::warn!("prewarm: failed to load L{moe_idx}E{expert_idx}: {e}");
                    inner.put_free(slot_idx);
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

/// Build an expert slot directly from the GGUF mmap. `mode` selects the weight form via
/// `QMatMul::from_weights_with_mode`: `Off` → FP GEMX (the dequant-weight grouped path); an int8
/// mode → the KO twin (device-direct `repack_ko`). Non-CUDA-only: under CUDA the experts (FP and
/// int8/KO alike) are staged through the pack file, so this is unused.
#[cfg(not(feature = "cuda"))]
fn load_from_mmap(
    mmap: &[u8],
    r: &MmapExpertRef,
    device: &Device,
    mode: Int8Mode,
) -> Result<ExpertSlot> {
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
        gate_proj: QMatMul::from_weights_with_mode(gate_proj.into(), mode)?,
        up_proj: QMatMul::from_weights_with_mode(up_proj.into(), mode)?,
        down_proj: QMatMul::from_weights_with_mode(down_proj.into(), mode)?,
    })
}

// ============================================================================
// Pipeline state — three tiers over an authoritative cold copy
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
    /// The cold tier: every expert, always, in kernel-ready form. `Arc`
    /// because the expert streamer reads it concurrently (positioned direct
    /// reads + interior-locked record cache — `&self` throughout).
    #[cfg(feature = "cuda")]
    pub(crate) pack: Arc<ExpertPack>,
    /// The warm tier: a stratified subset of the pack, pinned. Filled once,
    /// immutable after; `Arc`-shared read-only with the expert streamer.
    #[cfg(feature = "cuda")]
    pub(crate) warm: Arc<WarmPool>,
    /// Pinned landing buffers for reads that miss both resident tiers.
    #[cfg(feature = "cuda")]
    pub(crate) cold_staging: ColdStaging,
    /// Per-expert residency: `[moe_layer][expert_idx]`. Two independent facts
    /// over the pack, not a choice of one place.
    #[cfg(feature = "cuda")]
    pub(crate) residency: Vec<Vec<ExpertResidency>>,
    /// Per-layer geometry (shapes, dtypes, repacked sizes). `Arc`-shared with
    /// the expert streamer.
    #[cfg(feature = "cuda")]
    pub(crate) layer_geometries: Arc<Vec<LayerGeometry>>,
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
    /// Fences covering in-flight speculative DMA batches, keyed by TARGET
    /// layer — prefetch loads (recorded in `post_compute` for `L+1`) and hint
    /// loads (recorded in `process_hint` for the hinted layer). Recorded
    /// WITHOUT waiting — the DMA overlaps the forward thread's next-layer
    /// attention — and awaited when the covered layer's own work request
    /// arrives, BEFORE its hits are computed, by which point it has usually
    /// already signalled. Speculatively-loaded slots install (and so classify
    /// as hits) while their H2D is still in flight, so this wait is what keeps
    /// a hit from computing on half-copied expert bytes; waiting inline at
    /// load time instead would serialize the DMA on the pipe thread, stalling
    /// the next layer's work behind it.
    ///
    /// Per-layer keying is the load-ahead locking model: with prefetch depth
    /// N, a layer blocks only on the fence for ITS experts, never on DMAs
    /// issued for layers further ahead (a single latest-fence slot would
    /// serialize the main stream behind every outstanding target). All
    /// speculative copies share the one copy stream, so replacing a same-key
    /// fence with a later event is safe (a later event on the same stream
    /// implies completion of everything enqueued before it), and at most one
    /// entry per target layer ever exists.
    pub(crate) prefetch_fences: Vec<(usize, CopyBatchFence)>,
    /// Dynamic load-ahead depth `N`: how many layers ahead of the wave the
    /// speculative prefetcher issues for. Starts at 1; adapted once per pass
    /// by [`PipelineState::adapt_prefetch_depth`] on the late-load /
    /// bandwidth-slack signals, clamped to `1..=PREFETCH_DEPTH_MAX`.
    pub(crate) prefetch_depth: usize,
    /// Wall-clock start of the current forward pass (set at each pass
    /// boundary) — the denominator of the achieved-bandwidth measurement.
    pub(crate) pass_started: Option<std::time::Instant>,
    /// Copy-stream bytes enqueued this pass: demand misses + speculative
    /// prefetch + hint loads, priced at the zone's slot size.
    pub(crate) pass_dma_bytes: usize,
    /// In-flight prefetched experts that were late this pass (the per-pass
    /// slice of [`PipelineStats::late_loads`]).
    pub(crate) pass_late: usize,
    /// Per-pass prediction precision counters `(hits, total)` — the per-pass
    /// slice of `predicted_hits` / `predicted_total`.
    pub(crate) pass_pred: (usize, usize),
    /// Empirical achieved-bandwidth ceiling: running max of per-pass achieved
    /// GB/s. Self-calibrating and inclusive of every real-world overhead
    /// (WDDM, copy-stream contention), unlike the theoretical PCIe link rate,
    /// which is never reachable here and would make the controller
    /// over-deepen chasing it.
    pub(crate) bw_ceiling_gbps: f32,
    /// The off-thread expert streamer (whole-layer prefill streaming); `None`
    /// when all experts are resident or its staging ring could not be
    /// allocated. See `streamer.rs` for the protocol.
    #[cfg(feature = "cuda")]
    pub(crate) streamer: Option<StreamerHandle>,
    /// In-flight stream plans keyed by TARGET layer — the receiver half of
    /// each plan's one-shot outcome channel. Joined (blocking recv) when the
    /// target layer's work request arrives, drained at pass boundaries. At
    /// most one entry per layer (the issuer refuses a second plan while one
    /// is pending).
    #[cfg(feature = "cuda")]
    pub(crate) pending_streams: HashMap<usize, mpsc::Receiver<StreamDone>>,
    /// `(layer, expert)` pairs whose slots were installed at stream-issue
    /// time and whose bytes may still be in flight. Kept out of
    /// `speculative_loads` so whole-layer streaming does not poison the
    /// transition predictor's precision statistics; protected from demand
    /// eviction exactly like the speculative set, and cleared when the target
    /// layer's request arrives (successful entries become ordinary residents;
    /// failed ones were uninstalled at join).
    #[cfg(feature = "cuda")]
    pub(crate) stream_loads: HashSet<(usize, usize)>,
    /// Prediction accuracy counters: (hints_sent, hits_in_actual_set).
    pub(crate) hint_stats: (usize, usize),
    /// Timing accumulator for pipeline spans.
    pub(crate) profile: ProfileAccumulator,
    /// Shared telemetry counters (always-on).
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    // ── Non-CUDA legacy device-direct mmap path ──
    // Under CUDA every expert (FP and int8/KO) is staged through the pack file,
    // so these fields back only the non-CUDA `load_from_mmap` reload path.
    /// GGUF mmap, retained for non-CUDA device-direct slot builds.
    #[cfg(not(feature = "cuda"))]
    pub(crate) mmap: Arc<memmap2::Mmap>,
    /// Per-`[moe_layer][expert]` mmap offset/shape descriptors.
    #[cfg(not(feature = "cuda"))]
    pub(crate) host_refs: Vec<Vec<MmapExpertRef>>,
    /// Expert numeric mode (`Off` → FP, an int8 mode → KO twin) for `load_from_mmap`.
    #[cfg(not(feature = "cuda"))]
    pub(crate) int8mode: Int8Mode,
}

impl PipelineState {
    /// Classify experts as hits/misses and load the misses.
    ///
    /// A miss loads from the warm tier when the expert is there and from the
    /// pack when it is not. Displacing a resident expert to make room costs
    /// nothing but the bookkeeping: its bytes are already in the cold tier.
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

        // Defence in depth: every index into the per-layer expert tables below
        // assumes `expert_idx < n_experts`. A routing index at or past that
        // bound (e.g. a degenerate-token sentinel that slipped through) would
        // panic here and take down the whole expert-pipeline thread — after
        // which every forward fails forever. Drop such ids and log instead:
        // that token loses one expert (negligible) rather than bricking decode.
        #[cfg(feature = "cuda")]
        let n_experts = self.residency[moe_idx].len();
        #[cfg(not(feature = "cuda"))]
        let n_experts = self.host_refs[moe_idx].len();

        // Walk 1: split hits from misses WITHOUT allocating — the miss count is
        // what sizes the eviction, so it must be known before any slot moves.
        let mut miss_ids: Vec<usize> = Vec::new();
        for &expert_idx in expert_ids {
            if expert_idx >= n_experts {
                tracing::warn!(
                    moe_idx,
                    expert_idx,
                    n_experts,
                    "classify_and_load: routing index out of range — dropping (would otherwise panic the pipeline)"
                );
                continue;
            }
            if let Some(&slot_idx) = self.inner.key_to_slot.get(&(moe_idx, expert_idx)) {
                if self.inner.slots[slot_idx].is_some() {
                    self.inner.promote(slot_idx);
                    self.inner.record_hit(moe_idx, expert_idx);
                    hits.push((expert_idx, slot_idx));
                    continue;
                }
            }
            miss_ids.push(expert_idx);
        }

        // Exact-demand eviction: free EXACTLY what this layer's misses need
        // beyond the standing free list, in one batch scan preferring victims
        // from the window of layers directly behind the wave, with this layer's
        // hits AND the in-flight prefetch installs off-limits. Eviction is a
        // pure drop (cold pack authoritative, warm tier immutable), so nothing
        // is gained by doing it ahead of demand — and the headroom guessing
        // this replaces (per-layer drip + end-of-pass rate EMA) over-evicted by
        // its estimate error and scored mid-pass victims as if the pass were at
        // layer 0.
        // Protect set: this layer's hits (about to be computed with), plus
        // every in-flight speculative/streamed install whose target layer has
        // not consumed it yet (both sets clear when their layer's request
        // arrives). A just-installed prefetch has score ≈ 0 until its
        // prediction validates — unprotected, it would be the first victim
        // and its DMA a pure waste. For STREAMED installs protection is also
        // a matter of correctness: their bytes move on the streamer's stream,
        // so re-tenanting one from this thread's copy stream is an unordered
        // write race.
        let mut protect: std::collections::HashSet<usize> = hits.iter().map(|&(_, s)| s).collect();
        #[cfg(feature = "cuda")]
        let in_flight_iter = self
            .speculative_loads
            .iter()
            .chain(self.stream_loads.iter());
        #[cfg(not(feature = "cuda"))]
        let in_flight_iter = self.speculative_loads.iter();
        for &(layer, expert) in in_flight_iter {
            if let Some(&slot_idx) = self.inner.key_to_slot.get(&(layer, expert)) {
                if self.inner.slots[slot_idx].is_some() {
                    protect.insert(slot_idx);
                }
            }
        }

        let deficit = miss_ids.len().saturating_sub(self.inner.free_len());
        if deficit > 0 {
            let protect_list: Vec<usize> = protect.iter().copied().collect();
            for key in self.inner.demand_eviction(moe_idx, deficit, &protect_list) {
                #[cfg(feature = "cuda")]
                self.note_eviction(Some(key));
                #[cfg(not(feature = "cuda"))]
                let _ = key;
            }
        }

        // Walk 2: assign slots — the free list now covers the misses, so
        // `allocate_slot` takes its free path; its per-miss eviction scans
        // remain only as the backstop for a scan that came up short.
        for expert_idx in miss_ids {
            let (slot_idx, evicted_key) = self.inner.allocate_slot(moe_idx, &protect)?;
            #[cfg(feature = "cuda")]
            self.note_eviction(evicted_key);
            #[cfg(not(feature = "cuda"))]
            let _ = evicted_key;
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

        // **Every slot in `to_load` is allocated and keyless right now**, and it
        // stays keyless until the install below. So a failure anywhere in this
        // phase hands the batch back to the caller while the zone still counts
        // those slots as occupied — and a slot the zone holds with no expert in
        // it is one neither `alloc` nor the eviction scans can ever return. Ten
        // failed loads is ten slots gone for the life of the process.
        //
        // Hence the explicit unwinding rather than `?`: the error still
        // propagates unchanged, but the slots go back first.
        #[cfg(feature = "cuda")]
        let outcome = self.load_experts_batched(moe_idx, &to_load, &mut loaded_slots);

        #[cfg(not(feature = "cuda"))]
        let outcome = (|| -> Result<()> {
            let mmap_bytes: &[u8] = &self.mmap;
            for &(expert_idx, slot_idx) in &to_load {
                let mmap_ref = &self.host_refs[moe_idx][expert_idx];
                let expert_slot =
                    load_from_mmap(mmap_bytes, mmap_ref, &self.device, self.int8mode)?;
                loaded_slots.push((expert_idx, slot_idx, expert_slot));
            }
            Ok(())
        })();

        if let Err(e) = outcome {
            for &(_, slot_idx) in &to_load {
                if self.inner.slot_to_key[slot_idx].is_none() {
                    self.inner.zone.release(slot_idx);
                }
            }
            return Err(e);
        }
        self.pass_dma_bytes += loaded_slots.len() * self.inner.zone.slot_bytes();

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

            // A device copy now exists. Whether a host one also does is a
            // separate, immutable fact this does not touch.
            #[cfg(feature = "cuda")]
            {
                self.residency[moe_idx][expert_idx].vram = Some(slot_idx);
            }
        }
        self.profile.record("cl_install", t);

        // ── Telemetry ──
        {
            let num_hits = hits.len();
            let num_loaded = loaded.len();
            // Refresh the live resident-expert VRAM gauge from the current slot
            // occupancy (rises on install above, falls on evict elsewhere), so the
            // whole-card decomposition tracks experts paging VRAM↔pinned RAM.
            #[cfg(feature = "cuda")]
            let resident_vram = {
                let occupied = self.inner.num_slots() - self.inner.free_len();
                let slot_bytes = self
                    .layer_geometries
                    .iter()
                    .map(|g| g.total_repacked_size)
                    .max()
                    .unwrap_or(0);
                occupied * slot_bytes
            };
            // Ground the zone could concede to the KV side (capacity above its
            // floor) — the prefill width cap reads this through the stats
            // snapshot so it can admit waves the boundary would make room for.
            let cedeable = self
                .inner
                .zone
                .capacity()
                .saturating_sub(self.inner.zone.min_capacity())
                * self.inner.zone.slot_bytes();
            if let Ok(mut s) = self.stats.lock() {
                s.expert_hits += num_hits;
                s.expert_misses += num_loaded;
                s.dma_loads += num_loaded;
                s.zone_cedeable_bytes = cedeable;
                #[cfg(feature = "cuda")]
                {
                    s.resident_vram_bytes = resident_vram;
                }
            }
        }

        Ok(ClassifiedExperts {
            hits,
            loaded,
            fence,
        })
    }

    /// Record that an expert has left VRAM.
    ///
    /// **This is the whole of eviction.** One field assignment, no copy, no
    /// stream, no ordering, no destination slot to find, and no failure mode:
    /// the cold tier holds a valid copy of every expert at all times, and the
    /// warm tier — which is immutable, so it never reclaimed the slot — usually
    /// holds one too. Clearing `vram` re-exposes whichever copy is nearest.
    ///
    /// It used to be a 2.9 MB device-to-host copy into a pinned slot that had to
    /// be found first, and could fail with the expert's only copy in the slot
    /// being reused. Measured over one gate config that was 23,415 evictions —
    /// 68 GB of PCIe traffic, competing on the copy stream with an equal number
    /// of loads, all of it duplicating bytes the pack file already held.
    ///
    /// `None` means the allocator found a free slot and displaced nothing.
    #[cfg(feature = "cuda")]
    fn note_eviction(&mut self, evicted: Option<(usize, usize)>) {
        let Some((moe_idx, expert_idx)) = evicted else {
            return;
        };
        self.residency[moe_idx][expert_idx].vram = None;
        if let Ok(mut s) = self.stats.lock() {
            s.evictions += 1;
        }
    }

    /// Order the copy stream behind everything already issued on the compute
    /// stream, before a batch of H2D loads.
    ///
    /// **A slot is a fixed address now**, so an H2D that re-tenants one
    /// overwrites bytes a still-executing expert GEMM may be reading. The
    /// eviction policy deliberately prefers experts from layers already executed
    /// *this pass* — `allocate_slot`'s behind-layer scan takes
    /// `PINNED_LAYERS <= layer < current_layer` — and "already executed" means
    /// issued, not retired, so the previous layer's GEMM is exactly the kernel
    /// most likely to still be reading the slot being overwritten.
    ///
    /// The CUDA pool used to supply this ordering for free: `cuMemAllocAsync`
    /// returns memory whose `cuMemFreeAsync` has retired in stream order, so a
    /// reused buffer could not be written before its last reader finished. A
    /// fixed-address slot has no such guarantee, and the same gap in the KV side
    /// is what `region_pool::claim_region`'s quiesce exists to close.
    ///
    /// This is the cheap half of that: a GPU-side `cudaStreamWaitEvent`, so the
    /// **host does not block** — exactly as it did not under the pool. What is
    /// serialised is the copy stream against compute already issued, which is
    /// the same dependency the pool enforced internally.
    #[cfg(feature = "cuda")]
    fn order_copies_after_compute(&self) -> Result<()> {
        let (Some(cs), Device::Cuda(cd)) = (&self.copy_stream, &self.device) else {
            return Ok(());
        };
        let compute = cd.cuda_stream();
        let event = compute.record_event(None).map_err(candle::Error::wrap)?;
        cs.wait(&event).map_err(candle::Error::wrap)?;
        Ok(())
    }

    /// The other half: order the compute stream **behind** every copy issued so
    /// far, before a slot those copies wrote is computed on.
    ///
    /// [`Self::order_copies_after_compute`] stops a copy overwriting bytes a
    /// kernel is still reading. This stops a kernel reading bytes a copy has not
    /// finished writing, and the two are not interchangeable — a path that does
    /// only the first publishes a slot whose contents are still in flight.
    ///
    /// The load paths get this from [`CopyBatchFence`]: the batch records an
    /// event and whoever computes on the result waits for it
    /// (`prefetch_fence`, `classified.fence`, `prefetch_fences`). A
    /// relocation has no such consumer — it is not answering a work request, it
    /// runs at end of pass and hands its slots to whatever comes next — so it
    /// waits here instead, immediately, rather than threading a fence to a
    /// caller that does not exist.
    ///
    /// One call covers any number of copies: a later event on the same stream
    /// implies completion of everything enqueued before it. The wait is a
    /// GPU-side `cudaStreamWaitEvent`, so the host does not block.
    #[cfg(feature = "cuda")]
    fn order_compute_after_copies(&self) -> Result<()> {
        let (Some(cs), Device::Cuda(cd)) = (&self.copy_stream, &self.device) else {
            // Without a copy stream the copies were issued on the compute
            // stream itself, which already orders them.
            return Ok(());
        };
        let event = cs.record_event(None).map_err(candle::Error::wrap)?;
        cd.cuda_stream().wait(&event).map_err(candle::Error::wrap)?;
        Ok(())
    }

    /// Load an expert into the weight-zone slot at `slot_base`, from wherever
    /// its nearest copy is.
    ///
    /// **A total function with no error case for "where from".** The residency
    /// pair answers it directly:
    ///
    /// ```text
    /// ram = Some(s)  →  H2D from pinned slot s
    /// ram = None     →  read the pack, then H2D from the landing buffer
    /// ```
    ///
    /// There is no third case and no state this cannot serve. The warm copy is
    /// **not** surrendered on promotion: the warm tier is immutable, so the slot
    /// could not be reused by anything else, and keeping it is what makes the
    /// next eviction of this expert free. Without that the warm tier drains —
    /// every expert ever promoted would leave the warm set permanently, and
    /// since promotion is driven by demand, the ones lost first would be the
    /// most useful.
    ///
    /// The slot must already be held by the caller — every path reaches here
    /// after taking or forcing a slot, so the address is decided before the
    /// bytes move rather than by the allocator afterwards.
    #[cfg(feature = "cuda")]
    fn load_expert(
        &mut self,
        moe_idx: usize,
        expert_idx: usize,
        slot_base: u64,
    ) -> Result<ExpertSlot> {
        let Device::Cuda(cd) = &self.device else {
            candle::bail!("load_expert requires a CUDA device");
        };
        let geom = &self.layer_geometries[moe_idx];
        let layout = self.pack.layout(moe_idx);
        let stride = self.pack.stride();
        let stream = match &self.copy_stream {
            Some(cs) => cs.clone(),
            None => cd.cuda_stream(),
        };

        // SAFETY (both arms): `slot_base` names a slot the zone handed the
        // caller and has not reclaimed. Overwriting it is the point — a miss
        // replaces whatever the previous tenant left, in place.
        match self.residency[moe_idx][expert_idx].ram {
            Some(warm_slot) => {
                let src = self.warm.slot_ref(warm_slot, stride);
                if let Ok(mut s) = self.stats.lock() {
                    s.warm_loads += 1;
                }
                unsafe {
                    build_slot_from_record_on_stream(
                        src,
                        layout,
                        geom,
                        cd,
                        &stream,
                        slot_base,
                        Some(&mut self.profile),
                    )
                }
            }
            None => {
                let t = profile_now();
                let idx = self.cold_staging.acquire()?;
                self.pack.read_into(
                    moe_idx,
                    expert_idx,
                    self.cold_staging.buffer_mut(idx, stride),
                )?;
                self.profile.record("cold_read", t);
                let slot = unsafe {
                    build_slot_from_record_on_stream(
                        self.cold_staging.buffer_ref(idx, stride),
                        layout,
                        geom,
                        cd,
                        &stream,
                        slot_base,
                        Some(&mut self.profile),
                    )?
                };
                // The buffer cannot be written again until this upload lands.
                let event = stream.record_event(None).map_err(candle::Error::wrap)?;
                self.cold_staging.publish(idx, event);
                if let Ok(mut s) = self.stats.lock() {
                    s.cold_loads += 1;
                }
                Ok(slot)
            }
        }
    }

    /// Load a batch of misses for one layer into their reserved slots —
    /// the shared loader for the demand path (`classify_and_load`) and the
    /// layer-ahead prefetch.
    ///
    /// Cold misses take ONE concurrent striped NVMe read per staging-ring
    /// chunk (`read_many_unverified` — the same overlapped path the startup
    /// fill uses, minus its checksum); warm-backed misses follow as
    /// pinned→VRAM H2Ds pipelined by the copy stream. The former shape — a
    /// solo unbuffered read per expert, serial on this thread — made a layer
    /// with k cold misses pay k × (read latency) with the drive idle between
    /// them.
    #[cfg(feature = "cuda")]
    fn load_experts_batched(
        &mut self,
        moe_idx: usize,
        to_load: &[(usize, usize)],
        loaded_slots: &mut Vec<(usize, usize, ExpertSlot)>,
    ) -> Result<()> {
        // Before any byte moves: the slots below may still be under read by
        // the previous layer's GEMM.
        self.order_copies_after_compute()?;
        let cold: Vec<(usize, usize)> = to_load
            .iter()
            .copied()
            .filter(|&(e, _)| self.residency[moe_idx][e].ram.is_none())
            .collect();
        if !cold.is_empty() {
            let Device::Cuda(cd) = self.device.clone() else {
                candle::bail!("cold expert load requires a CUDA device");
            };
            let stream = match &self.copy_stream {
                Some(cs) => cs.clone(),
                None => cd.cuda_stream(),
            };
            let stride = self.pack.stride();
            let layout = self.pack.layout(moe_idx);
            for chunk in cold.chunks(COLD_STAGING_BUFFERS) {
                let t = profile_now();
                let idxs = self.cold_staging.acquire_many(chunk.len())?;
                self.profile.record("cold_acquire", t);
                let t = profile_now();
                {
                    let bufs = self.cold_staging.buffers_mut_for(&idxs, stride)?;
                    let reads: Vec<PackRead<'_>> = chunk
                        .iter()
                        .zip(bufs)
                        .map(|(&(expert_idx, _), dest)| PackRead {
                            layer: moe_idx,
                            expert: expert_idx,
                            dest,
                        })
                        .collect();
                    self.pack.read_many_unverified(reads)?;
                }
                self.profile.record("cold_read", t);
                for (&(expert_idx, slot_idx), &buf_idx) in chunk.iter().zip(&idxs) {
                    let slot_base = self.inner.slot_base(slot_idx);
                    let geom = &self.layer_geometries[moe_idx];
                    // SAFETY: `slot_base` names a slot the zone handed this
                    // batch and has not reclaimed; overwriting it is the point.
                    let slot = unsafe {
                        build_slot_from_record_on_stream(
                            self.cold_staging.buffer_ref(buf_idx, stride),
                            layout,
                            geom,
                            &cd,
                            &stream,
                            slot_base,
                            Some(&mut self.profile),
                        )?
                    };
                    // The buffer cannot be written again until this upload lands.
                    let event = stream.record_event(None).map_err(candle::Error::wrap)?;
                    self.cold_staging.publish(buf_idx, event);
                    if let Ok(mut s) = self.stats.lock() {
                        s.cold_loads += 1;
                    }
                    loaded_slots.push((expert_idx, slot_idx, slot));
                }
            }
        }
        // Warm-backed misses AFTER the cold reads: their pinned→VRAM H2Ds are
        // pipelined by the copy stream and need no host wait, while an
        // unbuffered NVMe read issued BEHIND a layer's whole warm H2D burst
        // contends with the in-flight DMA traffic.
        for &(expert_idx, slot_idx) in to_load {
            if self.residency[moe_idx][expert_idx].ram.is_none() {
                continue;
            }
            let slot_base = self.inner.slot_base(slot_idx);
            let expert_slot = self.load_expert(moe_idx, expert_idx, slot_base)?;
            loaded_slots.push((expert_idx, slot_idx, expert_slot));
        }
        Ok(())
    }

    /// Record `fence` as covering the in-flight speculative DMAs whose
    /// TARGET is `target_layer`. A no-op fence is dropped (nothing to
    /// cover). A same-key entry is replaced — all speculative copies share
    /// the one copy stream, so a later event implies completion of
    /// everything enqueued before it.
    fn record_prefetch_fence(&mut self, target_layer: usize, fence: CopyBatchFence) {
        if !fence.is_real() {
            return;
        }
        if let Some(entry) = self
            .prefetch_fences
            .iter_mut()
            .find(|(l, _)| *l == target_layer)
        {
            entry.1 = fence;
        } else {
            self.prefetch_fences.push((target_layer, fence));
        }
    }

    /// Wait (GPU-side) every prefetch fence whose target layer is at or
    /// behind `layer`, removing them from the ring; fences for layers still
    /// ahead stay pending so their DMAs keep overlapping compute. Returns
    /// true when `layer`'s OWN fence had not yet signalled at the moment of
    /// the wait — the prefetch was issued but did not land in time (a late
    /// load, the latency-bound signal for the load-ahead controller).
    fn wait_prefetch_fences_through(&mut self, layer: usize) -> Result<bool> {
        let mut late = false;
        let mut i = 0;
        while i < self.prefetch_fences.len() {
            if self.prefetch_fences[i].0 <= layer {
                let (l, fence) = self.prefetch_fences.swap_remove(i);
                if l == layer && !fence.is_complete() {
                    late = true;
                }
                fence.wait(&self.device)?;
            } else {
                i += 1;
            }
        }
        Ok(late)
    }

    /// Wait every outstanding prefetch fence, whatever its target. Runs at a
    /// pass boundary: the new pass reuses the same layer indices, so an
    /// unconsumed fence from the old pass would otherwise sit ahead of the
    /// wave and never be waited before its slots are computed as hits.
    fn drain_prefetch_fences(&mut self) -> Result<()> {
        for (_, fence) in std::mem::take(&mut self.prefetch_fences) {
            fence.wait(&self.device)?;
        }
        Ok(())
    }

    /// Issue a whole-layer stream plan for `issue_layer + 1` — the
    /// prefill-width analogue of the transition-matrix prefetch. Every
    /// non-resident expert of the target layer gets a slot (capacity from the
    /// behind-window batch eviction, then the free list; partial coverage is
    /// fine — the demand path loads the rest), the slot VIEW is installed
    /// immediately (so classify sees the experts as hits and never
    /// double-loads), and the byte-moving goes to the streamer thread with a
    /// compute-order event. The plan is joined when the target layer's
    /// request arrives ([`Self::join_stream_for`]).
    #[cfg(feature = "cuda")]
    fn stream_next_layer(&mut self, issue_layer: usize) {
        let target_layer = issue_layer + 1;
        if target_layer >= self.num_moe_layers
            || target_layer < self.inner.pinned_layers
            || self.streamer.is_none()
            || self.pending_streams.contains_key(&target_layer)
        {
            return;
        }
        let Device::Cuda(cd) = self.device.clone() else {
            return;
        };

        // Candidate misses: non-resident experts of the target layer that the
        // model has ROUTED TO BEFORE (score > 0), highest-scored first, capped
        // at what the compute window can carry. Streaming the literal whole
        // layer was measured here: hit rate rose (57→62% at config-8) but the
        // never-routed tail pushed total DMA volume up ~25% and the bus — not
        // the thread — is the constraint, so every config regressed and
        // virtually every stream fenced late. The score filter drops the
        // never-routed tail; the cap bounds a plan's bus time to roughly one
        // layer's compute window. Sources resolved NOW (the streamer carries
        // no residency state).
        let n_experts = self.residency[target_layer].len();
        let mut misses: Vec<(usize, f32, Option<usize>)> = (0..n_experts)
            .filter(|&e| {
                self.inner
                    .key_to_slot
                    .get(&(target_layer, e))
                    .is_none_or(|&s| self.inner.slots[s].is_none())
            })
            .filter_map(|e| {
                let score = self.inner.score(target_layer, e);
                (score > 0.0).then(|| (e, score, self.residency[target_layer][e].ram))
            })
            .collect();
        if misses.is_empty() {
            return;
        }
        misses.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        misses.truncate(STREAM_MAX_JOBS);
        let misses: Vec<(usize, Option<usize>)> =
            misses.into_iter().map(|(e, _, ram)| (e, ram)).collect();

        // Capacity: behind-window victims first (the double-buffer — the
        // layers the wave just left), then whatever the free list holds.
        // `take_free` stopping early bounds the plan to what the window could
        // supply.
        let need = misses.len().saturating_sub(self.inner.free_len());
        if need > 0 {
            for (slot_idx, evicted_key) in self.inner.evict_for_prefetch_batch(issue_layer, need) {
                self.note_eviction(evicted_key);
                self.inner.put_free(slot_idx);
            }
        }

        let mut jobs: Vec<StreamJob> = Vec::with_capacity(misses.len());
        for (expert_idx, warm_slot) in misses {
            let Some(slot_idx) = self.inner.take_free() else {
                break;
            };
            let slot_base = self.inner.slot_base(slot_idx);
            // SAFETY: `slot_base` names the slot just taken from the zone.
            // The view is pure geometry + address — the streamer moves the
            // bytes underneath it, fenced before this layer computes.
            let view = match unsafe {
                build_slot_view(&self.layer_geometries[target_layer], &cd, slot_base)
            } {
                Ok(v) => v,
                Err(_) => {
                    self.inner.put_free(slot_idx);
                    continue;
                }
            };
            self.inner.install(slot_idx, target_layer, expert_idx, view);
            self.residency[target_layer][expert_idx].vram = Some(slot_idx);
            self.stream_loads.insert((target_layer, expert_idx));
            jobs.push(StreamJob {
                expert_idx,
                slot_idx,
                slot_base,
                warm_slot,
            });
        }
        if jobs.is_empty() {
            return;
        }

        // Order the streamer's writes after the compute that may still be
        // reading the evicted tenants' bytes.
        let after = cd.cuda_stream().record_event(None).ok();
        let (done_tx, done_rx) = mpsc::channel::<StreamDone>();
        let n_jobs = jobs.len();
        let plan = StreamPlan {
            target_layer,
            jobs,
            after,
            done_tx,
        };
        let sent = self.streamer.as_ref().is_some_and(|s| s.send(plan));
        if sent {
            self.pending_streams.insert(target_layer, done_rx);
            tracing::trace!(
                target: "candle_transformers::expert_lre::streamer",
                target_layer,
                jobs = n_jobs,
                "whole-layer stream issued"
            );
        } else {
            // Streamer gone (thread died): take back every installed view so
            // the demand path reloads honestly.
            self.rollback_stream_installs(target_layer);
        }
    }

    /// Join the pending stream plan for `layer`, if any: blocks until the
    /// streamer has finished ENQUEUEING (usually already done — the reads
    /// overlapped the previous layer's compute), uninstalls any failed jobs,
    /// and records the plan's fence in the prefetch ring so the existing
    /// wait-at-need path covers the still-in-flight DMA. Returns the number
    /// of streamed installs for the layer (for late-load attribution).
    #[cfg(feature = "cuda")]
    fn join_stream_for(&mut self, layer: usize) -> usize {
        let Some(rx) = self.pending_streams.remove(&layer) else {
            return 0;
        };
        let t = profile_now();
        match rx.recv() {
            Ok(done) => {
                for (expert_idx, slot_idx) in done.failed {
                    if self.inner.slot_to_key[slot_idx] == Some((layer, expert_idx)) {
                        self.inner.evict(slot_idx);
                        self.inner.zone.release(slot_idx);
                        self.residency[layer][expert_idx].vram = None;
                    }
                    self.stream_loads.remove(&(layer, expert_idx));
                }
                self.pass_dma_bytes += done.bytes;
                self.record_prefetch_fence(layer, CopyBatchFence { event: done.fence });
            }
            Err(_) => {
                // Streamer died mid-plan: nothing enqueued reliably — take
                // every install back.
                self.rollback_stream_installs(layer);
            }
        }
        self.profile.record("stream_join", t);
        let streamed = self
            .stream_loads
            .iter()
            .filter(|&&(l, _)| l == layer)
            .count();
        self.stream_loads.retain(|&(l, _)| l != layer);
        streamed
    }

    /// Uninstall every still-tracked streamed slot for `layer` — the plan
    /// failed or the streamer died, so the views must not be computed with.
    #[cfg(feature = "cuda")]
    fn rollback_stream_installs(&mut self, layer: usize) {
        let entries: Vec<(usize, usize)> = self
            .stream_loads
            .iter()
            .copied()
            .filter(|&(l, _)| l == layer)
            .collect();
        for (l, expert_idx) in entries {
            if let Some(&slot_idx) = self.inner.key_to_slot.get(&(l, expert_idx)) {
                self.inner.evict(slot_idx);
                self.inner.zone.release(slot_idx);
                self.residency[l][expert_idx].vram = None;
            }
            self.stream_loads.remove(&(l, expert_idx));
        }
    }

    /// Join EVERY pending stream at a pass boundary — the new pass reuses the
    /// same layer indices, so an unjoined plan's fence would otherwise never
    /// be waited before its slots compute as hits.
    #[cfg(feature = "cuda")]
    fn drain_streams(&mut self) {
        let layers: Vec<usize> = self.pending_streams.keys().copied().collect();
        for layer in layers {
            self.join_stream_for(layer);
        }
    }

    /// Adapt the load-ahead depth `N` at a pass boundary, from the pass that
    /// just finished:
    ///
    /// * **Deepen** (`N+1`, ≤ [`PREFETCH_DEPTH_MAX`]) when the pass had late
    ///   loads AND bandwidth slack — prefetches were issued but did not land
    ///   in time while the link had headroom, so issuing a layer earlier
    ///   converts the latency into overlap. Late loads WITHOUT slack are
    ///   bandwidth-bound: deepening would only queue more bytes ahead of the
    ///   ones already late.
    /// * **Shallow** (`N−1`, ≥ 1) when the pass had zero late loads and its
    ///   prediction precision fell below [`PREFETCH_PRECISION_FLOOR`] — the
    ///   depth is landing in time but loading the wrong experts.
    ///
    /// The bandwidth reference is the EMPIRICAL running max of per-pass
    /// achieved GB/s ([`Self::bw_ceiling_gbps`]) rather than the theoretical
    /// PCIe rate: WDDM and copy-stream contention make the theoretical number
    /// unreachable, and a controller chasing it would deepen forever.
    ///
    /// Runs at every pass boundary; prefill passes self-select out of both
    /// arms in practice — their wide DMA saturates the link (no slack) and
    /// their compute window hides latency (no late loads) — so the depth is
    /// effectively steered by the decode-shaped passes, as intended.
    fn adapt_prefetch_depth(&mut self) {
        let elapsed = self.pass_started.take().map(|t| t.elapsed().as_secs_f32());
        let late = self.pass_late;
        let bytes = self.pass_dma_bytes;
        let (pred_hits, pred_total) = self.pass_pred;
        self.pass_late = 0;
        self.pass_dma_bytes = 0;
        self.pass_pred = (0, 0);

        let Some(dt) = elapsed else { return };
        if dt <= 0.0 {
            return;
        }
        let achieved_gbps = bytes as f32 / dt / 1e9;
        if achieved_gbps > self.bw_ceiling_gbps {
            self.bw_ceiling_gbps = achieved_gbps;
        }
        let slack = achieved_gbps < PREFETCH_BW_SLACK * self.bw_ceiling_gbps;

        if late >= PREFETCH_LATE_FLOOR && slack {
            self.prefetch_depth = (self.prefetch_depth + 1).min(PREFETCH_DEPTH_MAX);
        } else if late == 0
            && pred_total > 0
            && (pred_hits as f32) < PREFETCH_PRECISION_FLOOR * pred_total as f32
        {
            self.prefetch_depth = self.prefetch_depth.saturating_sub(1).max(1);
        }
        if let Ok(mut s) = self.stats.lock() {
            s.prefetch_depth = self.prefetch_depth;
        }
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
                // Streams first: joining records their fences into the ring
                // the next line drains.
                #[cfg(feature = "cuda")]
                self.drain_streams();
                self.drain_prefetch_fences()?;
                self.adapt_prefetch_depth();
            }
        }
        if self.pass_started.is_none() {
            self.pass_started = Some(std::time::Instant::now());
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
            self.pass_pred.0 += hits;
            self.pass_pred.1 += spec_for_layer.len();
            if let Ok(mut s) = self.stats.lock() {
                s.predicted_total += spec_for_layer.len();
                s.predicted_hits += hits;
            }

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

        // ── Join this layer's stream plan (if any) BEFORE classifying:
        // failed jobs must be uninstalled so they classify as honest misses,
        // and the plan's fence must be in the ring before the wait below. ──
        #[cfg(feature = "cuda")]
        let streamed = self.join_stream_for(req.moe_layer_idx);
        #[cfg(not(feature = "cuda"))]
        let streamed = 0usize;

        let t = profile_now();
        let classified = self.classify_and_load(req.moe_layer_idx, &req.expert_ids)?;
        self.profile.record("pipe_classify_load", t);

        // ── Whole-layer streaming for the NEXT layer, issued HERE — after
        // this layer's misses took their slots, before its compute — so the
        // streamer's reads overlap this layer's entire DMA + grouped-GEMM
        // window plus the forward thread's next attention pass. Issued from
        // post_compute (after the response) the only overlap was the
        // attention pass, and virtually every stream joined late. Only for
        // prefill-width waves: a wave routing at least half the layer means
        // the next layer will too — nothing to predict. Decode-width waves
        // take the transition-matrix prefetch in post_compute instead. ──
        #[cfg(feature = "cuda")]
        if self.streamer.is_some()
            && !self.all_resident
            && req.expert_ids.len() * 2 >= self.residency[req.moe_layer_idx].len().max(1)
        {
            let t = profile_now();
            self.stream_next_layer(req.moe_layer_idx);
            self.profile.record("stream_issue", t);
        }

        // ── Wait for this layer's deferred speculative-prefetch fence BEFORE
        // computing hits: a speculatively-loaded expert is installed (and
        // classifies as a hit) while its H2D may still be in flight. The DMA
        // overlapped the forward thread's attention window, so this is usually
        // already signalled; when it is NOT, the prefetch was issued too late
        // to hide the latency — counted as late loads (one per in-flight
        // install for the layer). Fences targeting layers still ahead are left
        // pending so their DMAs keep overlapping compute. ──
        let t = profile_now();
        let late = self.wait_prefetch_fences_through(req.moe_layer_idx)?;
        let in_flight = spec_for_layer.len() + streamed;
        if late && in_flight > 0 {
            self.pass_late += in_flight;
            if let Ok(mut s) = self.stats.lock() {
                s.late_loads += in_flight;
            }
        }
        self.profile.record("pipe_prefetch_fence", t);

        // Helper: extract (token_ids, weight_ids) for a given expert.
        let expert_group = |eid: usize| -> (Vec<u32>, Vec<u32>) {
            let eid32 = eid as u32;
            let lo = req.assignments.partition_point(|a| a.0 < eid32);
            let hi = req.assignments.partition_point(|a| a.0 <= eid32);
            let toks: Vec<u32> = req.assignments[lo..hi].iter().map(|a| a.1).collect();
            let wids: Vec<u32> = req.assignments[lo..hi].iter().map(|a| a.2).collect();
            (toks, wids)
        };

        let (num_tokens, hidden) = req.input.shape()?;
        // The combine target comes from the submitting layer's FFN span. This
        // thread cannot borrow that generation, but the request carries its
        // ticket, and the submitter blocks on the response for the whole
        // request — so the span is open from here to the hand-back, and the
        // returned tensor is a lease that frees nothing when the caller drops
        // it.
        // There is no wave domain without CUDA, so off-CUDA the target is an
        // ordinary owned allocation — the same split the `MoeInput` match below
        // makes, for the same reason.
        #[cfg(feature = "cuda")]
        let mut ys =
            wave_zeros_ticketed((num_tokens, hidden), req.out_dtype, &self.device, req.wave)?;
        #[cfg(not(feature = "cuda"))]
        let mut ys = Tensor::zeros((num_tokens, hidden), req.out_dtype, &self.device)?;
        // Non-CUDA only ever sees `Float` (int8/q8a128 is cuda-only).
        #[cfg(not(feature = "cuda"))]
        let MoeInput::Float(xs_float) = &req.input;

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

        // ── Compute EVERY routed expert, in one canonically-ordered pass ──
        //
        // Hits and newly-loaded experts are computed together and **ordered by
        // expert id**, because that order decides the arithmetic. Each call
        // reduces a token's contributions sequentially, and the scatter breaks
        // ties by position within the call, so splitting the experts into two
        // calls made a token's k terms sum in a grouping determined by which
        // experts happened to be resident. Residency varies run to run, float
        // addition is not associative, and the result was an engine that
        // returned different text for the same prompt
        // (`docs/deepseek_decode_reproducibility.md`). Expert id is a function
        // of routing alone, so this order is the same on every run.
        //
        // The cost is the overlap this used to buy: hit experts were computed
        // while the misses' DMA was still in flight. Correctness first — the
        // overlap-preserving shape is to keep both GEMM phases writing into
        // disjoint row ranges of one contribution buffer and scatter once.
        let t = profile_now();
        let all: Vec<(usize, usize)> = {
            let mut v: Vec<(usize, usize)> = classified
                .hits
                .iter()
                .chain(classified.loaded.iter())
                .copied()
                .collect();
            v.sort_by_key(|&(eidx, _)| eidx);
            v
        };
        #[cfg(feature = "cuda")]
        {
            let experts_data: Vec<(Vec<u32>, Vec<u32>)> =
                all.iter().map(|&(eidx, _)| expert_group(eidx)).collect();
            // **A missing slot is a dropped expert, not a skippable one.** The
            // old `filter_map(… as_ref()?)` silently omitted any expert whose
            // slot was empty at compute time, so its contribution vanished from
            // `ys` and the layer returned an answer computed from fewer than k
            // experts — indistinguishable, downstream, from a correct one.
            let mut experts_vec: Vec<(&ExpertSlot, &[u32], &[u32])> = Vec::with_capacity(all.len());
            for (&(eidx, slot_idx), (toks, wids)) in all.iter().zip(experts_data.iter()) {
                let Some(slot) = self.inner.slots[slot_idx].as_ref() else {
                    candle::bail!(
                        "expert {eidx} was classified resident in slot {slot_idx}, but that slot \
                         is empty at compute time: its contribution would be dropped from this \
                         layer's output"
                    );
                };
                experts_vec.push((slot, toks.as_slice(), wids.as_slice()));
            }
            if !experts_vec.is_empty() {
                compute_experts_grouped(
                    &req.input,
                    &mut ys,
                    &experts_vec,
                    &req.weights_flat,
                    &mut self.profile,
                    req.wave,
                )?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Same canonical order as the CUDA arm: this path accumulates into
            // `ys` one expert at a time, so it is subject to the identical
            // residency-dependent reassociation.
            for &(eidx, slot_idx) in &all {
                let slot = self.inner.slots[slot_idx].as_ref().ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "expert {eidx} was classified resident in slot {slot_idx}, but that \
                         slot is empty at compute time"
                    ))
                })?;
                let (toks, w_ids) = expert_group(eidx);
                compute_expert_contribution_gpu_weights(
                    xs_float,
                    &mut ys,
                    slot,
                    &toks,
                    &req.weights_flat,
                    &w_ids,
                )?;
            }
        }
        self.profile.record("pipe_compute_experts", t);

        Ok(ys)
    }

    /// Post-compute maintenance: speculative prefetch, drip eviction, and
    /// end-of-pass score decay + adaptive batch eviction.
    /// Called AFTER the response has been sent to the forward thread so this
    /// work doesn't inflate submit_roundtrip.
    pub(crate) fn post_compute(&mut self, moe_layer_idx: usize, expert_ids: &[usize]) {
        // When every expert is resident, there is nothing to prefetch or
        // evict — skip all post-compute maintenance.
        #[cfg(feature = "cuda")]
        if self.all_resident {
            return;
        }

        // ── Speculative prefetch for the next MoE layer (decode-width) ──
        // Prefill-width waves are handled by the whole-layer streamer, issued
        // BEFORE this layer's compute (see `process_request`) so its reads
        // get the full compute window; a stream already pending makes this a
        // no-op regardless (the issuer refuses duplicates). The fence is NOT
        // awaited here: the DMA runs on the copy stream while the forward
        // thread computes the next layer's attention, and the next work
        // request's compute phase waits on it (usually already signalled by
        // then). An inline wait would serialize the prefetch on the pipe
        // thread and stall the next layer's work behind it.
        let t = profile_now();
        #[cfg(feature = "cuda")]
        let stream_pending = self.pending_streams.contains_key(&(moe_layer_idx + 1));
        #[cfg(not(feature = "cuda"))]
        let stream_pending = false;
        if !stream_pending {
            // Advisory: a failed hop is dropped — the demand path loads
            // whatever the chain missed. Each hop records its own fence
            // internally.
            let _ = self.speculative_prefetch(moe_layer_idx, expert_ids);
        }
        self.profile.record("pipe_prefetch", t);

        // ── End-of-pass: score decay ──
        // Eviction is EXACT-DEMAND now: classify counts each layer's misses
        // before any load and frees precisely the deficit in one batch scan
        // ([`ExpertCacheInner::demand_eviction`]), scored at the wave's real
        // layer with the layer's hits protected. There is nothing left to do
        // between layers or between passes — eviction is a pure drop (cold
        // pack authoritative, warm tier immutable), so pre-freeing bought no
        // copy-hiding, and the headroom/rate estimators this replaces both
        // over-evicted by their estimate error and scored mid-pass victims as
        // if the pass were at layer 0.
        if moe_layer_idx + 1 == self.num_moe_layers {
            // Decay all expert scores (exponential forgetting).
            self.inner.decay_scores(0.85);

            // **The boundary does not move from here.**
            //
            // This was the growing direction's call site, and "end of pass" is
            // what made it look safe: the eviction above has produced whatever
            // free slots it was going to, and no GEMM for the pass is still being
            // *issued*. But end-of-pass on this thread is the middle of a forward
            // on the other one — `post_compute` runs the instant a MoE layer's
            // work is answered, while the forward thread is still inside
            // `ffn_forward` holding that layer's FFN wave guard. A wave arena is
            // live, and moving the boundary under one evicts and relocates slots
            // the wave may be reading.
            //
            // So the move lives where the wave genuinely is not: the wave loop's
            // inter-forward gap, beside the transient tier's hand-back
            // (`batched_model::forward_wave` phase 0 → `reclaim_spare_ground`).
            // The KV side's own direction is unchanged — it buys at the claim,
            // through `request_kv_ground`, and does not wait for a pass to end.
        }
    }

    /// Move the weight/KV boundary. `want` is regions the KV side is asking for;
    /// `None` asks the opposite question — how much it is holding free that the
    /// weight side could take back.
    ///
    /// **This is the elastic partition.** It runs on the pipeline thread because
    /// that is the only place the weight side is safe to change: this thread owns
    /// the cache. *When* it may run is a separate condition and a stricter one —
    /// no wave generation open on the span — which neither this function nor its
    /// thread can establish. Both callers reach it from outside a forward, and
    /// `set_weight_floor` checks rather than trusts that.
    ///
    /// # `want` is stated by the caller, never inferred here
    ///
    /// It used to come from `take_kv_demand`, which drained a running count of
    /// refused claims. That count was of *events*, and this function spent it as
    /// *regions*: one failed section-quantize drain walked the size-class ladder
    /// and left 4,436 behind it, against a KV side that was 28 regions short. The
    /// retraction that followed evicted 1,598 experts and put the zone below its
    /// own pinned working set, after which every forward failed `cannot evict
    /// (all pinned)` — 1,774 times, until the daemon was killed.
    ///
    /// Now the number arrives from whoever knows it: an arena claim that ran out
    /// asks for the regions it is claiming, and the scheduler's relief asks for
    /// its measured setpoint shortfall. Neither can accumulate, because neither
    /// survives the call that made it.
    ///
    /// Both directions are **non-destructive by preference**. Growing takes only
    /// free regions. Shrinking relocates the hottest doomed experts into free
    /// slots below the new frontier and drops the rest, exactly as an ordinary
    /// eviction does — so the worst case is a reload, never a loss.
    ///
    /// **The retraction stops at the zone's floor and nowhere else.** It was once
    /// capped at the expert cache's free pinned slots, because an evicted expert
    /// needed somewhere to be, and that cap was measured at 0–109 usable slots.
    /// With the cold tier authoritative there is no destination to find, so the
    /// range is the full `[floor, max_slots]` span the elastic partition was
    /// written to deliver — where `floor` is `minimum_resident_slots`, the fewest
    /// this cache can serve a token with. That floor is now the *only* thing
    /// bounding how much residency the KV side can buy, so it is measured from
    /// the pinning rule rather than guessed as a fraction (see `WeightZone::new`).
    ///
    /// **Ground changes hands only behind a device quiesce** — see
    /// [`Self::quiesce_before_handover`]. Removing the cap is what made that
    /// necessary: while a retraction moved nothing, no byte ever changed owner.
    ///
    /// Answers with the **bytes conceded to the KV side** — zero when the
    /// boundary held or moved the other way.
    #[cfg(feature = "cuda")]
    fn renegotiate_boundary(&mut self, want: Option<usize>) -> Result<u64> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(0);
        };
        let stream = cd.cuda_stream();
        // One conversion, one direction: regions the KV side is short (positive
        // ⇒ the floor moves right, the weight side shrinks) or holding spare
        // (negative ⇒ the floor moves left, the weight side grows).
        let delta = match want {
            Some(0) | None => {
                let spare = kv_spare_regions(&stream, KV_REGION_SLACK)?;
                if spare == 0 {
                    return Ok(0);
                }
                -(spare as isize)
            }
            Some(wanted) => wanted as isize,
        };
        let floor = weight_floor_after(&stream, delta)?;
        let before = self.inner.zone.capacity();
        let target = self.inner.zone.capacity_for_frontier(floor);
        if target == before {
            return Ok(0);
        }

        if target > self.inner.zone.capacity() {
            // Taking ground is a handover too: those regions were the KV side's
            // until this instant, and "free" there means no host-side gid names
            // them, not that no kernel is still reading them.
            self.quiesce_before_handover()?;
            // **The floor moves first, and the zone follows it.**
            //
            // `set_weight_floor` is refusable — it declines while a wave
            // generation is open on the span — and this ran the other way round:
            // `grow_zone` applied, then the publish, then `?` carried the refusal
            // out to a caller that logs a warning. What it left behind was a zone
            // one slot wider than the boundary the KV side had been told about,
            // so the next miss allocated the new top slot and handed the GEMM
            // `slot_base(capacity)` — an address exactly one `slot_bytes` *below*
            // `weight_floor`, which is KV ground.
            //
            // Measured: `gate weights for expert 43 are at 0xee10c0000, below the
            // weight floor 0xee1384000`, one slot down, from the first wave step
            // of a bulk ingest — and every later symptom (experts unable to
            // evict, KV unable to reclaim regions the failed steps never retired)
            // hung off that one refusal.
            //
            // Growing shrinks the KV side, so publishing first is also the safe
            // order in its own right: the KV side stops handing out the ground
            // before the weight side starts filling it. A refusal now lands with
            // nothing yet moved, and the next pass tries again.
            let grown_floor = self.inner.zone.frontier_after_growth(target);
            let gained = if grown_floor < self.inner.zone.frontier_for_capacity() {
                set_weight_floor(&stream, grown_floor)?;
                self.inner.grow_zone(target)
            } else {
                // `grow_to` clamps to the zone's limit, so a target past it moves
                // no boundary and must publish none.
                0
            };
            if gained > 0 {
                tracing::debug!(
                    target: "candle_transformers::expert_lre",
                    gained,
                    spare = -delta,
                    slots = self.inner.zone.capacity(),
                    "weight side took free KV regions"
                );
            }
            return Ok(0);
        }

        // **Refuse before touching anything, exactly as the growth path does.**
        //
        // This check used to sit at the end, just before `set_weight_floor` —
        // after `retract_zone` had already shrunk the zone, after the
        // relocations had moved experts, after the evictions, and after
        // `truncate_tables`. When it refused, `?` carried the error out through
        // a caller that logs it as a warning, so the whole retraction stayed
        // half-applied: the zone believed it was smaller while experts were
        // still live at indices past the new capacity, and the next pass handed
        // one of their addresses to a grouped GEMM.
        //
        // Measured: `boundary renegotiation failed: refusing to move the weight
        // boundary while a wave generation is open` at t=00:59:49.222, then at
        // t=00:59:49.844 a gate weight pointer exactly `slot_bytes` below
        // `weight_floor` — `slot_base(capacity)`, one slot past the last valid
        // index. That address is KV ground, and reading it is the
        // CUDA_ERROR_ILLEGAL_ADDRESS this hunt has been chasing.
        //
        // The growth branch above already had this right: quiesce, *then*
        // `grow_zone`. Retraction is the same handover and gets the same order.
        //
        // A zone already sitting on its floor concedes nothing, and `retract_to`
        // clamps to that floor rather than reporting it — so the answer is known
        // here, before a device-wide synchronize is spent on it. The relief
        // ladder asks on every rung of every failed wave; at the floor that was
        // 188 consecutive quiesces to arrive at zero each time.
        if target.max(self.inner.zone.min_capacity()) >= before {
            tracing::debug!(
                target: "candle_transformers::expert_lre",
                wanted = delta,
                slots = before,
                floor_slots = self.inner.zone.min_capacity(),
                "weight side is on its floor and can concede no further ground"
            );
            return Ok(0);
        }
        // In-flight stream plans hold slot addresses the retraction is about
        // to relocate or drop — and their byte-moves run on the STREAMER's
        // stream, which the compute-stream quiesce below does not cover until
        // the plan's copies are at least enqueued. Join them first: after the
        // join every streamed slot is an ordinary resident (or uninstalled),
        // and its fence sits in the ring where the context synchronize
        // completes it.
        #[cfg(feature = "cuda")]
        self.drain_streams();
        self.quiesce_before_handover()?;

        // The zone decides who moves and who goes; this performs it.
        let plan = self.inner.retract_zone(target);
        for &(from, to) in &plan.relocate {
            self.relocate_slot(from, to)?;
        }
        // **The relocations are published but not yet landed.** Each one
        // installed its destination into the tables the instant its copy was
        // *enqueued*, so as far as the next pass is concerned those experts are
        // resident — while the bytes are still moving on the copy stream. One
        // wait covers the whole plan and must happen before the boundary moves,
        // because moving it is what lets the next pass start issuing GEMMs
        // against the slots that just changed address.
        if !plan.relocate.is_empty() {
            self.order_compute_after_copies()?;
        }
        // The doomed remainder is dropped. This used to be the dangerous half of
        // a retraction — a D2H copy per expert, into a pool that could be full,
        // with the expert already out of the tables by the time the copy ran, so
        // a failure meant those bytes existed nowhere and every later forward
        // reported `Expert cache full, cannot evict (all pinned)` forever. It is
        // now a bookkeeping update that cannot fail.
        for &slot_idx in &plan.evict {
            let key = self.inner.evict(slot_idx);
            self.note_eviction(key);
        }
        // Only now: the relocations above read `slot_to_key` for the slots the
        // truncation removes.
        self.inner.truncate_tables();

        // **`residency` is the fifth table, and the truncation cannot reach it.**
        //
        // `truncate_tables` cleans `slots`, `last_used`, `slot_to_key` and
        // `key_to_slot` — every structure `inner` owns. `residency` lives on
        // *this* struct, so a retraction left it still claiming
        // `vram = Some(idx)` for slots the zone had just given up. Those entries
        // are what say "this expert is resident, in that slot", so the next pass
        // resolved one to an address at or below the frontier and handed it to a
        // grouped GEMM: `slot_base(idx)` for `idx >= capacity` is below
        // `weight_floor`, which is KV ground.
        //
        // Measured before this: gate weights for expert 49 exactly one
        // `slot_bytes` below the floor, and experts 27 and 28 sharing a single
        // address three slots below it — two keys resolving to one slot because
        // both had stale residency pointing into the conceded range.
        //
        // Cleared here rather than inside `truncate_tables` because that is a
        // method on `inner`, which has no access to `residency`; keeping the two
        // adjacent is what makes the pairing visible.
        let cap = self.inner.zone.capacity();
        for layer in self.residency.iter_mut() {
            for res in layer.iter_mut() {
                if res.vram.is_some_and(|s| s >= cap) {
                    res.vram = None;
                }
            }
        }
        // The conceded slots stop being ours the moment the floor moves.
        self.quiesce_before_handover()?;
        // **A refused publish here has to be undone, not carried out.**
        //
        // Retraction cannot publish first the way growth does — the floor moving
        // right is what hands the KV side ground the experts above it are only
        // now vacating. So the refusable step stays last, and the failure it can
        // still produce is repaired instead of propagated: the zone believes it
        // is smaller while `pool.weight_floor` says otherwise, and *both* sides
        // lose the ground — the weight side has evicted off it, the KV side was
        // never told it may use it. Worse, the next attempt computes its target
        // from the unmoved floor, finds it equal to the capacity already reached,
        // and returns zero: the relief loop then asks 188 times in a row and is
        // answered `conceded_mib=0` every time while nothing is actually pinned.
        //
        // Growing the zone back is a true rollback. The doomed slots were evicted
        // or relocated, so they are free; restoring the capacity restores the
        // agreement between the zone and the published floor, and the only cost
        // is reloading experts that were dropped for nothing.
        if let Err(e) = set_weight_floor(&stream, self.inner.zone.frontier_for_capacity()) {
            self.inner.grow_zone(before);
            return Err(e);
        }
        let conceded =
            (before - self.inner.zone.capacity()) as u64 * self.inner.zone.slot_bytes() as u64;
        tracing::debug!(
            target: "candle_transformers::expert_lre",
            wanted = delta,
            relocated = plan.relocate.len(),
            evicted = plan.evict.len(),
            slots = self.inner.zone.capacity(),
            floor_slots = self.inner.zone.min_capacity(),
            conceded_mib = conceded / (1 << 20),
            "weight side gave ground to KV"
        );
        Ok(conceded)
    }

    /// Retire every kernel in flight before a byte changes owner.
    ///
    /// **The boundary is the one place where memory changes side**, and neither
    /// side's own ordering reaches across it. The KV side quiesces before
    /// re-tenanting a *recycled* region (`region_pool::claim_region`), because a
    /// region on its free list may still be under read. But ground arriving from
    /// the weight side is not recycled — it is **fresh**, claimed by advancing
    /// `pool.next` past a ceiling that just moved, and that path has no wait at
    /// all. Nothing was wrong with that: until the concession cap was removed,
    /// the ceiling never moved over ground an expert had been sitting on, so a
    /// fresh region had never been anyone's.
    ///
    /// Now it has. `renegotiate_boundary` runs between forwards, and "between
    /// forwards" bounds only what is being *issued* — the last pass's expert
    /// GEMMs may still be executing. Publishing a lower floor lets the KV side
    /// memset and write bytes those GEMMs are still reading, which surfaces as
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` in whatever unrelated kernel is running when
    /// the fault lands. The same applies in reverse when the weight side takes
    /// regions back: an expert upload would overwrite bytes a KV kernel is
    /// reading.
    ///
    /// A GPU-side `cudaStreamWaitEvent` — what
    /// [`Self::order_compute_after_copies`] uses — cannot do this job. It orders
    /// two streams we know about; the readers here include the persistence
    /// thread's copy stream and the KV side's own work, and the *host* has to
    /// know the ground is quiet before it tells the other side it may have it.
    /// So this is a device-wide synchronize, exactly as `claim_region`'s quiesce
    /// is and for the same reason.
    ///
    /// It costs a full drain, and it is paid **only when the boundary actually
    /// moves** — a rare event at end of pass, against a retraction that already
    /// relocates or drops thousands of slots.
    #[cfg(feature = "cuda")]
    fn quiesce_before_handover(&self) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        let stream = cd.cuda_stream();
        let ctx = stream.context();
        ctx.bind_to_thread().map_err(candle::Error::wrap)?;
        ctx.synchronize().map_err(candle::Error::wrap)?;
        Ok(())
    }

    /// Move one expert's bytes from slot `from` to slot `to`, in place.
    ///
    /// A device-to-device copy of one slot — a few microseconds at card
    /// bandwidth — and a rewrite of the tables that name it. This is what keeps
    /// a *hot* expert alive when the boundary takes the ground it was sitting
    /// on, and it is only possible because every slot is the same size: the
    /// copy is a memcpy between two addresses of identical length, not a
    /// compaction.
    ///
    /// Issued on the copy stream, behind the same ordering as a load: the
    /// destination may have been read by the pass that just finished.
    ///
    /// **The tables are rewritten as soon as the copy is enqueued, so this
    /// returns with the expert published and its bytes still in flight.** The
    /// caller closes that with one [`Self::order_compute_after_copies`] for the
    /// whole plan — cheaper than a fence per slot, and correct for the same
    /// reason: a later event on the copy stream implies every copy before it.
    /// Calling this without that wait computes the next pass on a half-copied
    /// expert.
    #[cfg(feature = "cuda")]
    fn relocate_slot(&mut self, from: usize, to: usize) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        let Some(key) = self.inner.slot_to_key[from] else {
            // **Nothing to move, so give the destination back.**
            // `WeightZone::retract_to` marks `to` occupied when it builds the
            // plan — it deals in occupancy and cannot see keys — so returning
            // here without releasing leaves a slot the zone believes is taken
            // and the tables believe is empty. It is then invisible to `alloc`
            // (not on the free list) and to both eviction scans (no key), and
            // it never comes back.
            //
            // Self-amplifying, which is what made it fatal: one such slot in a
            // later retraction's doomed range produces another, and this run
            // retracted twenty-four times.
            self.inner.zone.release(to);
            return Ok(());
        };
        let geom = &self.layer_geometries[key.0];
        let bytes = slot_offsets(geom).3;
        let src = self.inner.slot_base(from);
        let dst = self.inner.slot_base(to);
        self.order_copies_after_compute()?;
        let stream = self.copy_stream.clone().unwrap_or_else(|| cd.cuda_stream());
        // SAFETY: both addresses name whole slots of the zone, of `slot_bytes`
        // each; `bytes` is this layer's aligned total and never exceeds it. The
        // source is live (the zone has not reclaimed it) and the destination is
        // free (the plan chose it from the free list), so the two cannot alias.
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream())
                .map_err(candle::Error::wrap)?;
        }
        // Rebuild the slot's storages over the new address, then move the
        // bookkeeping. The old `ExpertSlot`'s storages are leases, so dropping
        // them releases the views and touches no memory.
        // SAFETY: the copy above put this layer's three projections at `dst`,
        // at the offsets `slot_offsets` names, and the zone holds the slot.
        let moved = unsafe { build_slot_view(geom, cd, dst)? };
        self.inner.slots[from] = None;
        self.inner.slot_to_key[from] = None;
        self.inner.slots[to] = Some(moved);
        self.inner.slot_to_key[to] = Some(key);
        self.inner.key_to_slot.insert(key, to);
        self.inner.last_used[to] = self.inner.last_used[from];
        if let Some(res) = self.residency.get_mut(key.0).and_then(|l| l.get_mut(key.1)) {
            res.vram = Some(to);
        }
        Ok(())
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
            .predict_prefetch(prev_layer_idx, prev_expert_ids);

        if predicted.is_empty() || layer_idx >= self.num_moe_layers {
            self.profile.record("pipe_hint", t);
            return;
        }

        #[allow(unused_mut)]
        let mut loaded_count = 0usize;

        // A hint load takes only free slots, but "free" can mean "evicted this
        // pass and still under read" — see `order_copies_after_compute`.
        #[cfg(feature = "cuda")]
        if let Err(e) = self.order_copies_after_compute() {
            tracing::warn!("hint: could not order the copy stream ({e}); skipping");
            self.profile.record("pipe_hint", t);
            return;
        }

        for &expert_idx in &predicted {
            // Skip if already resident in VRAM.
            if self
                .inner
                .key_to_slot
                .get(&(layer_idx, expert_idx))
                .is_some_and(|&s| self.inner.slots[s].is_some())
            {
                continue;
            }

            // Free-slots-only: never evict for speculative loads.
            let slot_idx = match self.inner.take_free() {
                Some(s) => s,
                None => break, // no free slots — stop speculating
            };

            // Load from pinned pool (CUDA only — no-op on CPU builds).
            #[cfg(not(feature = "cuda"))]
            {
                self.inner.put_free(slot_idx);
                continue;
            }

            #[cfg(feature = "cuda")]
            let slot_base = self.inner.slot_base(slot_idx);
            #[cfg(feature = "cuda")]
            let expert_slot = match self.load_expert(layer_idx, expert_idx, slot_base) {
                Ok(slot) => slot,
                Err(_) => {
                    self.inner.put_free(slot_idx);
                    continue;
                }
            };

            // Install the speculatively loaded expert.
            #[cfg(feature = "cuda")]
            self.inner
                .install(slot_idx, layer_idx, expert_idx, expert_slot);

            #[cfg(feature = "cuda")]
            {
                self.residency[layer_idx][expert_idx].vram = Some(slot_idx);
                self.speculative_loads.insert((layer_idx, expert_idx));
                loaded_count += 1;
            }
        }

        if loaded_count > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.hint_loads += loaded_count;
            }
            self.pass_dma_bytes += loaded_count * self.inner.zone.slot_bytes();
            // Fence the batch: hint loads install their slots IMMEDIATELY (so
            // the next work request classifies them as hits), but the H2D
            // copies are still in flight on the copy stream. Without a fence a
            // hit whose bytes are half-copied computes a garbage expert
            // contribution — an intermittent, hard-to-attribute quality bug.
            // Recording into the fence ring under the hinted layer reuses the
            // deferred wait that runs before that layer's hit compute;
            // replacing an older same-key fence is safe because a later event
            // on the same copy stream implies completion of everything
            // enqueued before it.
            #[cfg(feature = "cuda")]
            if let (Some(cs), Device::Cuda(_)) = (&self.copy_stream, &self.device) {
                match cs.record_event(None) {
                    Ok(event) => {
                        self.record_prefetch_fence(
                            layer_idx,
                            CopyBatchFence { event: Some(event) },
                        );
                    }
                    // No event → no deferred cover for the in-flight copies.
                    // Fall back to draining the copy stream inline: rare and
                    // slow, but the alternative is the next work request
                    // computing a hint-installed expert on half-copied bytes.
                    Err(_) => {
                        let _ = cs.synchronize();
                    }
                }
            }
            self.profile.record("pipe_hint_dma", t);
        } else {
            self.profile.record("pipe_hint", t);
        }
    }

    /// Speculatively prefetch the experts the next `prefetch_depth` MoE
    /// layers will need, one confidence-gated hop at a time.
    ///
    /// Prediction stays ADJACENT — each hop asks the shared transition matrix
    /// to predict `target` from the previous hop's set (hop 1 from the layer's
    /// actual routing, hop 2 from hop 1's prediction, …), so precision
    /// compounds per hop and the chain self-terminates the moment a hop
    /// returns no confident prediction. Mispredictions cost slots and
    /// bandwidth, never correctness — the demand path loads whatever the
    /// chain missed. Each hop's DMA batch is fenced under ITS OWN target
    /// layer in the fence ring, so a layer's compute blocks only on the
    /// copies it will actually read, never on the deeper hops.
    ///
    /// The depth is the dynamic load-ahead `N` (see
    /// [`Self::adapt_prefetch_depth`]) — 1 in steady state, deepened only
    /// when late loads show the single-hop DMA is not landing in time.
    ///
    /// Two regimes, by the width of the wave that just computed:
    ///
    /// * A DECODE-width wave routes a handful of experts, so the next
    ///   layer's set is genuinely uncertain — the confidence-gated top-k.
    /// * A PREFILL-width wave routes most of the layer, and the next layer
    ///   will too. NOTE: a full-next-layer arm for prefill-width waves was
    ///   measured here and REGRESSED bulk throughput (cfg8 437→416): the
    ///   batch runs on this pipeline thread inside `post_compute`, so the
    ///   next layer's work request queues behind ~170 loads instead of
    ///   overlapping them. Streaming the next layer needs its reads OFF this
    ///   thread.
    fn speculative_prefetch(
        &mut self,
        moe_layer_idx: usize,
        current_expert_ids: &[usize],
    ) -> Result<()> {
        // Depth beyond the adjacent layer only for decode-narrow waves: wide
        // (prefill-shaped) sources hide single-hop latency under their compute
        // window, and their multi-hop batches would run on this pipe thread
        // ahead of the next work request (the measured dead end).
        let depth = if current_expert_ids.len() <= PREFETCH_MULTI_HOP_MAX_SOURCES {
            self.prefetch_depth
        } else {
            1
        };
        let mut source: Vec<usize> = current_expert_ids.to_vec();
        for hop in 1..=depth {
            let target_layer = moe_layer_idx + hop;
            if target_layer >= self.num_moe_layers {
                break;
            }
            let predicted = self
                .transition_matrix
                .predict_prefetch(target_layer - 1, &source);
            if predicted.is_empty() {
                // No confident basis — nothing to load here and nothing to
                // chain the next hop from.
                break;
            }
            // Don't LOAD into a pinned layer — its experts are always
            // resident (the pinned head layers run first every pass with no
            // compute to hide a reload, so they stay permanently pinned). The
            // prediction still chains through it. The count is the cache's own
            // (`affordable_pinned_layers` of the current capacity), not the
            // constant — on a cache too small to pin three layers it can be
            // fewer, or zero, and then every layer is prefetchable.
            if target_layer >= self.inner.pinned_layers {
                let fence = self.prefetch_into(moe_layer_idx, target_layer, &predicted)?;
                self.record_prefetch_fence(target_layer, fence);
            }
            source = predicted;
        }
        Ok(())
    }

    /// Load the non-resident members of `predicted` into `target_layer`'s
    /// slots from the warm/cold tiers, returning one fence covering the
    /// batch's copy-stream DMA. Misses are loaded into free slots; when the
    /// cache is full, ONE batched eviction (`evict_for_prefetch_batch`) frees
    /// the deficit from the safest furthest-window victims behind
    /// `issue_layer` — the wave's actual position — (the just-computed
    /// "behind" layers; the wave tail at the pinned boundary). Prefetched
    /// experts are tracked in `speculative_loads` so the target layer's work
    /// request can score prediction precision.
    fn prefetch_into(
        &mut self,
        issue_layer: usize,
        target_layer: usize,
        predicted: &[usize],
    ) -> Result<CopyBatchFence> {
        // Misses: predicted experts not already resident in VRAM.
        let misses: Vec<usize> = predicted
            .iter()
            .copied()
            .filter(|&e| {
                self.inner
                    .key_to_slot
                    .get(&(target_layer, e))
                    .is_none_or(|&s| self.inner.slots[s].is_none())
            })
            .collect();
        if misses.is_empty() {
            return Ok(CopyBatchFence::noop());
        }

        // Make room in ONE batched eviction: free the deficit between the misses
        // and the current free slots by evicting the safest furthest-window
        // victims — the just-computed "behind" layers (the double-buffer; the
        // wave tail at the pinned boundary). One scan, not one per victim.
        // Each eviction is a drop, so making room costs nothing but the scan.
        #[cfg(feature = "cuda")]
        {
            let need = misses.len().saturating_sub(self.inner.free_len());
            if need > 0 {
                for (slot_idx, evicted_key) in
                    self.inner.evict_for_prefetch_batch(issue_layer, need)
                {
                    self.note_eviction(evicted_key);
                    self.inner.put_free(slot_idx);
                }
            }
        }

        // Load the misses into (now-provisioned) free slots — stopping early if
        // the window couldn't supply enough room; the demand path loads the
        // rest. The slots just freed above were occupied by experts of layers
        // behind the wave, whose GEMMs may still be executing.
        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(misses.len());
        for &expert_idx in &misses {
            match self.inner.take_free() {
                Some(slot_idx) => pairs.push((expert_idx, slot_idx)),
                None => break,
            }
        }
        let mut loaded = 0usize;
        #[cfg(feature = "cuda")]
        {
            let mut loaded_slots: Vec<(usize, usize, ExpertSlot)> = Vec::with_capacity(pairs.len());
            let outcome = self.load_experts_batched(target_layer, &pairs, &mut loaded_slots);
            let filled: std::collections::HashSet<usize> =
                loaded_slots.iter().map(|&(_, s, _)| s).collect();
            for (expert_idx, slot_idx, slot) in loaded_slots {
                self.inner.install(slot_idx, target_layer, expert_idx, slot);
                self.residency[target_layer][expert_idx].vram = Some(slot_idx);
                // Track for prediction-precision measurement — validated when
                // the next layer's work request arrives.
                self.speculative_loads.insert((target_layer, expert_idx));
                loaded += 1;
            }
            // A failed batch hands back every slot it never filled — prefetch
            // is advisory, so the error itself is dropped (the demand path
            // will load whatever this round missed).
            if outcome.is_err() {
                for &(_, slot_idx) in &pairs {
                    if !filled.contains(&slot_idx) {
                        self.inner.put_free(slot_idx);
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        for &(expert_idx, slot_idx) in &pairs {
            let mmap_bytes: &[u8] = &self.mmap;
            let mmap_ref = &self.host_refs[target_layer][expert_idx];
            match load_from_mmap(mmap_bytes, mmap_ref, &self.device, self.int8mode) {
                Ok(s) => {
                    self.inner.install(slot_idx, target_layer, expert_idx, s);
                    self.speculative_loads.insert((target_layer, expert_idx));
                    loaded += 1;
                }
                Err(_) => {
                    self.inner.put_free(slot_idx);
                }
            }
        }

        if loaded == 0 {
            return Ok(CopyBatchFence::noop());
        }

        // Record one fence covering all of this round's prefetch DMA.
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

        if let Ok(mut s) = self.stats.lock() {
            s.prefetch_loads += loaded;
        }
        self.pass_dma_bytes += loaded * self.inner.zone.slot_bytes();
        // Prefetch funnel diagnostic: how demand narrows to actual loads —
        // transition predictions → non-resident misses → DMA'd. A wide wave
        // stalling on demand DMA with `loaded ≈ 0` here means the predictor
        // (not the load path) is the bottleneck. TRACE, not debug: it fires
        // once per MoE layer per forward (up to 48 lines per decoded token),
        // which floods a debug-level log.
        tracing::trace!(
            target: "candle_transformers::expert_lre::prefetch",
            layer = target_layer,
            issued_at = issue_layer,
            predicted = predicted.len(),
            misses = misses.len(),
            loaded,
            "speculative prefetch funnel",
        );

        Ok(fence)
    }
}

// ============================================================================
// Thread spawn
// ============================================================================

/// Sets its flag when dropped — including on unwind — so the thread's death is
/// observable without joining. A dead pipeline thread has dropped its
/// `PipelineState` (freeing every expert slot's VRAM), which invalidates any
/// captured expert weight pointers; `ExpertCache::pipeline_dead` gates the
/// GPU-native dispatch on this flag staying clear.
struct DeadFlagGuard(Arc<AtomicBool>);

impl Drop for DeadFlagGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// Spawn the pipeline thread.  Returns the sender for submitting work/hints.
pub(crate) fn spawn_pipeline_thread(
    mut state: PipelineState,
    dead_flag: Arc<AtomicBool>,
) -> mpsc::SyncSender<PipelineMessage> {
    let (tx, rx) = mpsc::sync_channel::<PipelineMessage>(4);

    std::thread::Builder::new()
        .name("expert-pipeline".into())
        .spawn(move || {
            let _dead_on_exit = DeadFlagGuard(dead_flag);
            while let Ok(msg) = rx.recv() {
                match msg {
                    PipelineMessage::Work(req) => {
                        let response_tx = req.response_tx.clone();
                        let moe_layer_idx = req.moe_layer_idx;
                        let expert_ids = req.expert_ids.clone();
                        // Inbound handoff latency: forward-thread `send` → worker pickup here.
                        state.profile.record("submit_inbound", req.submitted_at);
                        let wt = profile_now();
                        let result = state.process_request(req);
                        // Whole-request work (classify + DMA + compute), so the forward thread can
                        // subtract it from submit_roundtrip to isolate the channel handoff tax.
                        state.profile.record("pipe_worker_total", wt);
                        // Stamp completion before send so the forward thread can measure outbound.
                        let _ = response_tx.send((result, profile_now()));
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
                    // The give-back, reachable without a completed forward. The
                    // scheduler asks when a wave could not allocate; answering
                    // zero is a legitimate outcome (nothing spare, or a wave
                    // generation still open), and the caller retries or gives up
                    // on that basis rather than spinning.
                    PipelineMessage::RenegotiateBoundary {
                        regions,
                        response_tx,
                    } => {
                        #[cfg(feature = "cuda")]
                        let conceded = match state.renegotiate_boundary(Some(regions)) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                tracing::warn!("requested boundary move failed: {e}");
                                0
                            }
                        };
                        #[cfg(not(feature = "cuda"))]
                        let conceded = {
                            let _ = regions;
                            0u64
                        };
                        let _ = response_tx.send(conceded);
                    }
                }
            }
        })
        .expect("failed to spawn expert-pipeline thread");

    tx
}

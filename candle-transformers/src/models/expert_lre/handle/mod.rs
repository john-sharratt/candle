//! The expert cache handle: the entry point through which MoE layers route work
//! into the expert pipeline.
//!
//! `ExpertCache` owns the device-side expert forward and the channels to the two
//! host threads. The pipeline thread keeps VRAM residency in step with routing,
//! and the stager reads cold experts into the pad. The forward thread waits on
//! neither: it enqueues each MoE layer, tells both threads its routing, and moves
//! on, so the GPU waits only for a cold expert the stager is still reading. A
//! build without `cuda` cannot construct the cache. Where the CPU split is on, the
//! device gives the host pool its cheap pinned misses and collects the results.
//!
//! - `construction`: builds and fills the cache and checks its preconditions.
//! - `forward`: the routed forward and its fault handling.
//! - `ground`: moves the boundary between the weight zone and the KV side.
//! - `sizing`: sizes the warm tier from the machine's free memory.
//! - `telemetry`: counters, profiles and hit references.
//! - `tests_sizing`: tests for the warm tier's sizing.

#[cfg(feature = "cuda")]
use super::boundary::{grow_note, spare_for_growth, GrowOutcome};
use super::cache::{minimum_resident_slots, pinned_layer_count, ExpertCacheInner};
#[cfg(feature = "cuda")]
use super::cpu_pool::attach::start as start_cpu_service;
#[cfg(feature = "cuda")]
use super::cpu_pool::cold::PagedImages;
#[cfg(feature = "cuda")]
use super::cpu_pool::params::shared as cpu_params;
#[cfg(feature = "cuda")]
use super::dispatch::{CpuPath, Dispatch, OwnerTags};
#[cfg(feature = "cuda")]
use super::elastic::controller::{max_chunks, spawn_controller, ControllerCtx, Elastic};
#[cfg(feature = "cuda")]
use super::elastic::paged::chunk_slots;
#[cfg(feature = "cuda")]
use super::link_rate::measure_link_rate;
#[cfg(feature = "cuda")]
use super::live_table::LiveTable;
use super::pack::ExpertPack;
#[cfg(feature = "cuda")]
use super::pack::RecordLayout;
#[cfg(feature = "cuda")]
use super::pad::Pad;
#[cfg(feature = "cuda")]
use super::pinned::{can_go_cold, stratified_membership, ExpertResidency};
#[cfg(feature = "cuda")]
use super::pipeline::{spawn_pipeline_thread, PipelineState};
#[cfg(feature = "cuda")]
use super::promo::PromotionRing;
#[cfg(feature = "cuda")]
use super::residency::Residency;
#[cfg(feature = "cuda")]
use super::routing_trace::RoutingTrace;
#[cfg(feature = "cuda")]
use super::section::geometries_of;
#[cfg(feature = "cuda")]
use super::slot_image::{row_tile_bytes_for, slot_bytes_for, slot_offsets};
#[cfg(feature = "tensor-assert")]
use super::slot_owners::SlotOwners;
#[cfg(feature = "cuda")]
use super::stager::{spawn_stager, StagerCtx, StagerMsg};
#[cfg(feature = "cuda")]
use super::startup::{startup_from_pack, StartupRing, StartupTargets, PACK_TAIL_STEPS};
#[cfg(feature = "cuda")]
use super::types::LookAhead;
use super::types::{BoundaryAsk, PipelineMessage, PipelineStats};
#[cfg(feature = "cuda")]
use super::warm_tier::WarmTier;
use crate::models::load_phase::{LoadPhase, LoadProgress};
use crate::models::profile::ProfileSnapshot;
#[cfg(feature = "cuda")]
use candle::cuda_backend::CudaDevice;
#[cfg(feature = "cuda")]
use candle::quantized::cuda::Q8a128Operand;
#[cfg(feature = "cuda")]
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::Int8Mode;
#[cfg(feature = "cuda")]
use candle::{DType, LiveTensor};
use candle::{Device, Result};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;
use candle_nn::kv_cache::WeightZone;
use std::path::Path;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// Pad slots an all-resident cache takes: nothing is ever cold there, so the
/// pad serves only as the startup fill's landing ring.
#[cfg(feature = "cuda")]
const STARTUP_RING_SLOTS: usize = 64;

// Warm tier sizing

/// The seed for the warm tier's stratified draw.
///
/// Fixed rather than random so that two runs of the same build on the same model
/// warm the same experts: a change in expert-cache hit rate is then attributable
/// to the change under test, not to which experts got lucky at startup.
const WARM_DRAW_SEED: u64 = 0x5745_524D_5F53_4545;

use candle::vram::{gpu_addressable_room, host_ram_budget, total_physical_ram, PAGEABLE_RESERVE};

/// Layers of pad slots the stager owns: the layer a wave is computing, and the
/// [`STAGED_HOPS`] after it, which a prefill launch's router votes stage into
/// while the current one computes (`PipelineState::stage_voted`). With one
/// layer the next layer's reads could not start until every slot of the
/// current one was released — a prefill row routes about half its layer, and
/// Rule R′ protects all of it — so the drive idled through each layer's up and
/// down projections.
pub(crate) const PAD_LAYERS: usize = 1 + STAGED_HOPS;

/// Rows ahead a prefill launch votes on and stages: the next row's router,
/// applied to this layer's input. Two hops — and a third pad layer — measured
/// no faster on Flash-Next (16 GB laptop: C0×2 341.5 → 329.2, BF16×8 1108.8 →
/// 1113.3 t/s): the stagings issued barely moved (9,701 → 10,192 a ×2 config)
/// though twice as many were requested, so what holds them back is not how
/// early they are asked for.
pub(crate) const STAGED_HOPS: usize = 1;

// ExpertCache — the public handle

/// Consecutive forwards the CPU expert pool may fail before the split stops
/// using it: one is a pool preempted past the spin limit, three is a pool that
/// is not coming back.
#[cfg(feature = "cuda")]
const CPU_FAULT_LIMIT: u32 = 3;

/// Global expert cache / pipeline handle.
///
/// Shared across all MoE blocks via `Arc<ExpertCache>`. A background pipeline
/// thread owns all mutable cache state (`ExpertCacheInner`, slots, eviction
/// scores, the copy stream) with `&mut self`; the forward thread reaches it
/// only through the channel and the pass lock (`dispatch::PassState`).
pub struct ExpertCache {
    /// The elastic warm tier's controller, stopped first: it holds a sender
    /// to the stager. `None` for a cache with no tier to move.
    #[cfg(feature = "cuda")]
    _elastic: Option<Elastic>,
    /// Channel to the pipeline thread.
    #[cfg(feature = "cuda")]
    tx: mpsc::SyncSender<PipelineMessage>,
    /// Channel to the stager.
    #[cfg(feature = "cuda")]
    stager: mpsc::Sender<StagerMsg>,
    /// The device-side expert forward and everything it shares with the host
    /// threads: the live table, the summary ring, the abort word.
    #[cfg(feature = "cuda")]
    dispatch: Dispatch,
    /// Set when the pipeline thread or the stager has exited (normally or by
    /// panic). Their guards raise the abort word as they go, so every waiting
    /// worker gives its expert up and fails the forward instead of waiting on
    /// an expert that will never come.
    #[cfg(feature = "cuda")]
    pipeline_dead: Arc<AtomicBool>,
    #[cfg(feature = "cuda")]
    stager_dead: Arc<AtomicBool>,
    /// The pinned tiers the live table's entries point into — held here so
    /// they outlive every reader, whichever thread exits first.
    #[cfg(feature = "cuda")]
    _warm: Arc<WarmTier>,
    #[cfg(feature = "cuda")]
    _pad: Arc<Pad>,
    /// Whether a worker can ever wait on a cold expert or on the CPU pool — and
    /// so give one up (`fault`). Not when every expert is in VRAM or has a
    /// pinned warm copy and no CPU class runs: every evictable expert then
    /// holds a resident pad slot, so none is ever read from the pack, no churn
    /// copy arises for the stager to promote, and the resident set — which
    /// no slot outside it can join — never changes.
    #[cfg(feature = "cuda")]
    can_fault: bool,
    /// Consecutive forwards the CPU pool failed (`take_fault`): at
    /// [`CPU_FAULT_LIMIT`] the split stops using the pool until a restart.
    #[cfg(feature = "cuda")]
    cpu_faults: AtomicU32,
    /// The device the cache serves, for the region pool's growth question
    /// ([`Self::reclaim_spare_ground`]).
    #[cfg(feature = "cuda")]
    device: CudaDevice,
    /// Shared telemetry counters (always-on).
    stats: Arc<Mutex<PipelineStats>>,
    /// The routing the pipeline thread has served, for the residency
    /// ceiling ([`Self::hit_ceiling`]), and the slots the permanently resident
    /// layers hold, which the ceiling's replay does not get.
    #[cfg(feature = "cuda")]
    trace: Arc<Mutex<RoutingTrace>>,
    #[cfg(feature = "cuda")]
    pinned_slots: usize,
    /// The zone's serving floor in slots — `minimum_resident_slots` — which a
    /// guest's concession may go below and [`Self::restore_serving_floor`]
    /// brings it back to.
    #[cfg(feature = "cuda")]
    serving_floor_slots: usize,
}

/// Everything [`ExpertCache::new`] needs, gathered rather than passed as nine
/// positional arguments.
///
/// `zone` is the weight side of the device reservation, already sized: its
/// capacity **is** the resident-expert count. There is no budget arithmetic left
/// at this level — `VramGovernor::expert_budget` used to divide bytes by
/// `max_expert_size` here, and the zone's capacity is that same quotient taken
/// once, against a span whose extent is a fact rather than a forecast.
pub struct ExpertCacheSetup<'a> {
    /// The model pack's expert section, opened — every expert of every layer,
    /// and the geometry the cache sizes itself from (`section::geometries_of`).
    pub(crate) pack: ExpertPack,
    /// The weight side of the device reservation.
    pub zone: WeightZone,
    pub device: &'a Device,
    pub experts_per_layer: usize,
    /// Experts each token routes to (top-k) — sizes the bucketize workspace and
    /// is checked against the router's bound.
    pub experts_used: usize,
    /// Told each phase of the fill — pinning the host tier, reading it, filling
    /// VRAM — and how far through it is.
    pub progress: Option<LoadProgress<'a>>,
    pub int8mode: Int8Mode,
    /// Bytes the loader mapped of the model pack — its GGUF part, which holds
    /// every tensor but the experts.
    pub mapped_bytes: u64,
    /// Of `mapped_bytes`, those host RAM never serves after load — read through
    /// a bounded cache of the model's own (net of that cache's RAM), or uploaded
    /// to the device once and never read from the file again. Flash-Next's
    /// n-gram table is 54 GB of the mapping read through a 2 GiB row cache, and
    /// its dense stack is 5.2 GiB living on the card; left counted as live
    /// weight, the host budget reserves them as page cache out of the warm
    /// tier's RAM. `0` for a model whose whole mapping is paged — one that reads
    /// a weight from the mapping at run time.
    pub offloaded_bytes: u64,
}

mod construction;
mod forward;
mod ground;
mod sizing;
mod telemetry;
#[cfg(test)]
mod tests_sizing;

pub use sizing::*;

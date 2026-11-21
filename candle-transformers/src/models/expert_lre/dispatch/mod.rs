//! The expert forward on the device, and the hand-off of each layer's routing to
//! the host threads.
//!
//! Bucketize classifies each routed expert as resident in VRAM, pinned, or cold,
//! and writes a routing summary that the host threads read off the critical path.
//! The expert GEMMs copy remote experts from pinned memory themselves, and a cold
//! expert's workers wait only on a word the stager publishes. The forward thread
//! holds back only to keep the summary ring from being overwritten before both host
//! threads have read it, which is a check on two counters rather than on the driver
//! or the GPU.
//!
//! - `abort`: the fault word that ends waits on an unpublishable expert.
//! - `cpu_class`: the device side of the CPU expert class.
//! - `launch`: the three-part launch of one layer's experts.
//! - `read_ahead`: the read-ahead copy layout per row.
//! - `reservation`: the shared pass and reservation protocol.
//! - `ring`: the routing summary ring.
//!
//! - `tests_dispatch`: tests of the read-ahead layout and the summary waiter.

use super::cold_wait::ColdWait;
use super::cpu_pool::backoff::{Backoff, INTER_LAYER_SPINS, LOW_POWER_CYCLES};
use super::cpu_pool::cold::{ColdSource, PagedImages};
use super::cpu_pool::params::shared;
use super::cpu_pool::service::CpuService;
use super::fault::FaultWord;
use super::flush_schedule::{SegmentFill, MAX_UNFLUSHED_INVOCATIONS};
use super::live_table::{LiveTable, Proj};
use super::promo::PromotionRing;
use super::read_ahead_gate::{LinkTime, ReadAheadGate};
use super::reclaim::ReclaimClock;
#[cfg(feature = "tensor-assert")]
use super::slot_owners::SlotOwners;
use super::stager::StagerMsg;
use super::started::StartedRows;
use super::transition::HOPS;
use super::types::{LookAhead, PipelineMessage, RoutedLayer};
use super::votes::VoteRing;
use crate::models::batched_inference::MAX_PREFILL_TOKENS;
use crate::models::operand_guard::expect_dtype;
use crate::models::profile::{gpu_span, profile_now};
use crate::models::wave_buffers::{wave_empty, wave_root};
use candle::cuda_backend::CudaDevice;
#[cfg(feature = "profile")]
use candle::quantized::cuda::STALL_WORDS;
use candle::quantized::cuda::{
    fused_deterministic_scatter, fused_moe_gather_q8a128, grouped_int8_n_sub,
    grouped_qmatmul_dev_q8a128, moe_bucketize, moe_predict_votes, silu_mul_q8a128, BucketizeCpu,
    BucketizeLive, MoeBucketizeWorkspace, MoeLive, OwnerCheck, Q8a128Operand, ReadAhead,
};
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::SumScale;
use candle::{DType, Device, LiveTensor, Result};
use candle_kernels::simple::moe_bucketize::{
    CpuSplit, AHEAD_ITEM_WORDS, AHEAD_MAX, CPU_CARRY_WORDS, CPU_PROMOTE_NONE,
};
use candle_kernels::simple::moe_cpu_export::{run_moe_cpu_export, EXPORT_MAX_BLOCKS, EXPORT_WARPS};
use candle_nn::kv_cache::WaveGeneration;
use cudarc::driver::{sys, CudaSlice, DevicePtr};
use std::ffi::c_void;
use std::ptr::{null_mut, read_volatile, write_bytes, write_volatile};
use std::slice;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// Routing summaries in flight between the forward thread and the host threads.
/// Invocation `seq` writes slot `seq % SUMMARY_RING`; the forward thread does
/// not enqueue it until both readers have read the slot's previous tenant
/// (`Dispatch::hold_for_ring`). 64 is more than any forward's MoE layer count.
pub(crate) const SUMMARY_RING: usize = 64;

/// The pipeline channel's bound. The ring is protected by the readers' counters,
/// so this only bounds the queue.
pub(crate) const PIPELINE_CHANNEL_BOUND: usize = SUMMARY_RING;

/// The longest a single worker waits for a cold expert before it gives the
/// expert up and fails the forward (`fault`). A deadlock backstop, never meant
/// to fire: a wait this long means the stager is not delivering. Under WDDM it
/// also keeps the launch inside the display watchdog's 2 s (TDR); under TCC
/// there is no watchdog and it stays the backstop.
pub(crate) const SPIN_LIMIT_NS: u64 = 1_000_000_000;

/// Ranks past the router's own top `k` each token names in the look-ahead's
/// votes (`moe_predict_votes`).
///
/// The look-ahead applies a later layer's router to this layer's input, a layer
/// stale, so the experts that layer actually routes sit just below its top `k`
/// more often than the cut admits — at ×1 about four of the ten. The margin
/// predicts past the cut at the cost of copies that may go unused. It needs no
/// schedule by width: the margin picks rank below every routed pick
/// (`votes::ranked`), and the per-row caps, the read-ahead window and the
/// stager's in-flight limit all cut a list from the bottom — so at ×1 the margin
/// fills the room one token's ten leave, and at ×8 the routed picks fill it.
///
/// Measured on Qwen3.8-Flash-Next (RTX 4090 Laptop, two alternating pairs
/// against a margin of 0): hop-1 recall 51 → 65% at ×1 and 32 → 45% at ×8, ×8
/// decode +1.5–2.5% (94.9 → 96.3, 97.1 → 98.8 t/s; C5×8 99.3 → 100.7), ×8 cold
/// misses −5–6%, ×1 level. It pays only with room in the staging list: at a
/// `STAGE_CAP` of 32 the routed picks of a five-token verify wave already filled
/// it, and the margin moved recall one point.
pub(crate) const LOOK_AHEAD_MARGIN: usize = 6;

/// Worker blocks per expert launch, by its width.
///
/// A worker's copy of a slice is a few PCIe round trips, then its compute: a
/// worker is latency-bound, and the link fills only with many of them. Decode
/// launches take 32: on Flash-Next (RTX 4090 Laptop) 8 → 16 raised decode a
/// third (×8 68 → 89 t/s, ×1 22 → 29), 32 a few percent more (×8 93), and 64
/// measured level with 32 — the blocks past what the link needs launch only to
/// exit, on every launch. A prefill launch's
/// remote experts carry several token tiles each and the workers compute them
/// all: on the qwen36 gate's cold prefill (RTX 3090) 8 workers made it
/// compute-bound at 814 t/s, 32 ran at 1,083, 64 no faster. Where misses are
/// most of a launch they need more: Qwen3.8-Flash-Next's C10 ×8 prefill (a
/// working set 3× the zone) ran 969 t/s at 32, 1,031 at 64 and 1,044 at 128, and
/// 128 cost its ×16 decode ~4%. At the Q3_KO rung on the 16 GB laptop, whose
/// prefill is bound by the link (Gen4 ×8) once its staging keeps up, 128 ran
/// every prefill row level or ahead of 64 — C5×2 373.0 → 382.5, BF16×8 1131 →
/// 1141 t/s — with decode no lower. The workers take their own grid rows, so a
/// count above a projection's row tiles adds rows, never width.
pub(crate) const WORKERS_DECODE: usize = 32;

pub(crate) const WORKERS_PREFILL: usize = 128;

/// Launches over more tokens than this take [`WORKERS_PREFILL`].
pub(crate) const WORKERS_PREFILL_TOKENS: usize = 64;

/// Launches over more tokens than this are prompt prefill, whose misses saturate
/// the link (`RoutedLayer::prefill_width`). Well above [`WORKERS_PREFILL_TOKENS`]:
/// a many-sequence decode step with its drafts verified (×16 × 4 tokens, ×64 × 4)
/// lands between the two, and it still leaves the link room for speculation.
pub(crate) const PREFILL_LAUNCH_TOKENS: usize = 256;

/// The worker count for a launch over `num_tokens` tokens. Scratch is sized for
/// [`WORKERS_PREFILL`] slots.
pub(crate) fn workers_for(num_tokens: usize) -> usize {
    if num_tokens > WORKERS_PREFILL_TOKENS {
        WORKERS_PREFILL
    } else {
        WORKERS_DECODE
    }
}

/// Everything the device-side expert forward needs that the host threads do
/// not own.
pub(crate) struct Dispatch {
    pub(crate) table: Arc<LiveTable>,
    pub(crate) ring: Arc<SummaryRing>,
    /// The router look-ahead votes, slot for slot with `ring`.
    pub(crate) votes: Arc<VoteRing>,
    pub(crate) abort: Arc<AbortWord>,
    /// Where a worker that gave a cold expert up says so (`fault`); shared
    /// with the pipeline thread, which lands no demand promotion while it is
    /// set.
    pub(crate) fault: Arc<FaultWord>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) pass: Arc<Mutex<PassState>>,
    /// The ticket of the last routed layer the pipeline thread has served.
    pub(crate) served: Arc<AtomicU64>,
    /// The ticket of the last routing summary the stager has read.
    pub(crate) staged: Arc<AtomicU64>,
    pinned: PinnedRanges,
    workspace: Mutex<MoeBucketizeWorkspace>,
    forward: Mutex<ForwardSide>,
    /// `u64[3][n_experts]`: bucketize's snapshot of the routed entries — the
    /// three GEMMs' weight tables.
    snap: CudaSlice<u64>,
    /// `i32[n_experts][4]` remote list and `i32[3]` work counters.
    remote: CudaSlice<i32>,
    counters: CudaSlice<i32>,
    /// The promotion ring (none when every expert is in VRAM), and
    /// `u64[n_experts]`, the promotion slot bucketize gave each remote expert.
    promo: Option<Arc<PromotionRing>>,
    remote_dst: CudaSlice<u64>,
    /// `WORKERS_PREFILL` VRAM slots of `slot_bytes`, the workers' copies.
    scratch: CudaSlice<u8>,
    slot_bytes: usize,
    owner: OwnerTags,
    /// Read-ahead's device buffers, with the promotion ring: each row's slot
    /// image layout (`u64[rows][4]`: gate, up, down offset, image bytes), the
    /// items bucketize writes and the gate launch's workers copy, and their
    /// piece counters (`candle-kernels/src/moe_read_ahead.cuh`).
    ahead: Option<AheadBuffers>,
    /// Whether the invocation being recorded reads ahead — set by the pipeline
    /// thread from the cache's link time per decode invocation
    /// (`read_ahead_gate`). Off, bucketize is not given `ahead` and the caller
    /// passes no look-ahead routers.
    pub(crate) read_ahead: Arc<ReadAheadGate>,
    /// The CPU class, or `None` where it cannot run (no pool started).
    cpu: Option<CpuPath>,
    /// What the gate launches' workers waited for cold experts (`cold_wait`).
    pub(crate) cold_wait: ColdWait,
    /// Profile build only: per-row worker counters `[rows × STALL_WORDS]` (see
    /// `kernel.cuh`, "A live expert table").
    #[cfg(feature = "profile")]
    pub(crate) stall: CudaSlice<u64>,
    #[cfg(feature = "profile")]
    device: CudaDevice,
}

/// The expert GEMMs' worker counters, summed over rows since the last drain.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkerCounters {
    /// Workers' time waiting for a cold expert to be published, ns.
    pub(crate) cold_wait_ns: u64,
    /// Worker items — a remote expert's 32-row tile of one projection.
    pub(crate) items: u64,
    /// Bytes the workers copied into scratch.
    pub(crate) bytes: u64,
    /// Workers' time in the copy, ns.
    pub(crate) copy_ns: u64,
    /// Live launches.
    pub(crate) launches: u64,
    /// Read-ahead items the gate launches' workers copied and published, and
    /// their bytes.
    pub(crate) ahead_items: u64,
    pub(crate) ahead_bytes: u64,
}

mod abort;
mod cpu_class;
mod launch;
mod read_ahead;
mod reservation;
mod ring;
#[cfg(test)]
mod tests_dispatch;

pub(crate) use abort::*;
pub(crate) use cpu_class::*;
use read_ahead::*;
pub(crate) use reservation::*;
pub(crate) use ring::*;

//! The expert forward on the device, and its hand-off to the host threads.
//!
//! One MoE layer, as the forward thread enqueues it — none of it waits on the
//! host:
//!
//! ```text
//! bucketize   reads the layer's live-table entries for the routed experts,
//!             classifies each VRAM / pinned / cold, snapshots the entries into
//!             VRAM, orders the remote (pinned, cold) experts' tiles first and
//!             lists them, writes the routing summary into the mapped ring and
//!             its sequence word last
//! send        the row to the pipeline thread and the stager
//! gather → gate → up → SwiGLU → down → scatter
//! ```
//!
//! Each expert GEMM reads its weights from the snapshot. Its worker blocks copy
//! every remote expert's row tiles from pinned memory into VRAM scratch and
//! compute them; a cold expert's workers wait on its live gate entry, which the
//! stager publishes once the expert is in the pad. A remote expert bucketize
//! gave a slot from the promotion ring (`promo`) has its slices written there
//! instead, and computed from there, so the layer leaves it in VRAM. Whether a
//! launch claims slots at all is bucketize's to decide, per launch: a prompt
//! passing over the table (a sweep) claims none and stays in scratch. Nothing
//! the GPU waits on needs a driver call (`docs/moe_live_dispatch_design.md` §0).
//!
//! The forward thread holds back only to keep the summary ring from being
//! overwritten before both host threads have read it — a check against two
//! counters, never against the driver or the GPU.

use super::live_table::{LiveTable, Proj};
use super::promo::PromotionRing;
use super::reclaim::ReclaimClock;
#[cfg(feature = "tensor-assert")]
use super::slot_owners::SlotOwners;
use super::stager::StagerMsg;
use super::started::StartedRows;
use super::types::{PipelineMessage, RoutedLayer};
use crate::models::batched_inference::MAX_PREFILL_TOKENS;
use crate::models::profile::{gpu_span, profile_now};
use crate::models::wave_buffers::{wave_empty, wave_root};
use candle::cuda_backend::CudaDevice;
use candle::quantized::cuda::{
    fused_deterministic_scatter, fused_moe_gather_q8a128, grouped_int8_n_sub,
    grouped_qmatmul_dev_q8a128, moe_bucketize, silu_mul_q8a128, BucketizeLive,
    MoeBucketizeWorkspace, MoeLive, OwnerCheck, Q8a128Operand,
};
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::SumScale;
use candle::{DType, Device, LiveTensor, Result};
use candle_nn::kv_cache::WaveGeneration;
use cudarc::driver::{sys, CudaSlice, DevicePtr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// Routing summaries in flight between the forward thread and the host threads.
/// Invocation `seq` writes slot `seq % SUMMARY_RING`; the forward thread does
/// not enqueue it until both readers have read the slot's previous tenant
/// (`Dispatch::hold_for_ring`). 64 is more than any forward's MoE layer count.
pub(crate) const SUMMARY_RING: usize = 64;

/// The pipeline channel's bound. The ring is protected by the readers' counters,
/// so this only bounds the queue.
pub(crate) const PIPELINE_CHANNEL_BOUND: usize = SUMMARY_RING;

/// The longest a single worker may wait for a cold expert before it traps. A
/// backstop below the display watchdog's 2 s (TDR), never meant to fire: a wait
/// this long means the stager is not delivering.
pub(crate) const SPIN_LIMIT_NS: u64 = 1_500_000_000;

/// Worker blocks per expert launch, by its width.
///
/// 8 reach the link's rate (§0.10.1–2), and a decode launch's remote experts
/// carry a token tile each, so 8 is all decode needs; every block past what is
/// needed is a block launched to exit, on every launch. A prefill launch's
/// remote experts carry several token tiles each and the workers compute them
/// all: on the qwen36 gate's cold prefill (RTX 3090) 8 workers made it
/// compute-bound at 814 t/s, 32 ran at 1,083, 64 no faster. Where misses are
/// most of a launch they need more: Qwen3.8-Flash-Next's C10 ×8 prefill (a
/// working set 3× the zone) ran 969 t/s at 32, 1,031 at 64 and 1,044 at 128, and
/// 128 cost its ×16 decode ~4%. The workers take their own grid rows, so a count
/// above a projection's row tiles adds rows, never width.
pub(crate) const WORKERS_DECODE: usize = 8;
pub(crate) const WORKERS_PREFILL: usize = 64;

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

/// The routing summaries, in mapped pinned memory: bucketize writes a slot
/// through its device address and the host threads read it through its host
/// address once its sequence word is in.
pub(crate) struct SummaryRing {
    dev_base: u64,
    host: *mut u32,
    n_experts: usize,
}

// SAFETY: a slot is written by one bucketize and read by the host threads only
// after its sequence word; the forward thread's ring hold keeps it from being
// rewritten while it is read.
unsafe impl Send for SummaryRing {}
unsafe impl Sync for SummaryRing {}

impl SummaryRing {
    pub(crate) fn new(n_experts: usize) -> Result<Self> {
        let len = SUMMARY_RING * (n_experts + 1);
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, len * 4, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert cache: mapped routing-summary ring allocation failed: {r:?}");
        }
        let mut dev_base: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev_base, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert cache: routing-summary ring has no device address: {r:?}");
        }
        // SAFETY: `len` u32s just allocated. Zero is never a sequence word, so
        // no slot reads as written before its first bucketize.
        unsafe { std::ptr::write_bytes(raw as *mut u32, 0, len) };
        Ok(Self {
            dev_base,
            host: raw as *mut u32,
            n_experts,
        })
    }

    fn stride(&self) -> usize {
        self.n_experts + 1
    }

    fn dev_slot(&self, slot: usize) -> u64 {
        self.dev_base + (slot * self.stride() * 4) as u64
    }

    fn host_slot_ptr(&self, slot: usize) -> *mut u32 {
        assert!(slot < SUMMARY_RING, "summary ring slot {slot} of {SUMMARY_RING}");
        // SAFETY: in bounds by the assertion.
        unsafe { self.host.add(slot * self.stride()) }
    }

    /// Whether bucketize has written `slot` for sequence word `word`.
    ///
    /// The word is the last thing bucketize stores, behind a system fence, so
    /// once it reads `word` every count is visible — and bucketize running
    /// means every kernel enqueued before it on the compute stream has
    /// completed. That is the whole signal: no event and no copy.
    pub(crate) fn ready(&self, slot: usize, word: u32) -> bool {
        // SAFETY: the word is inside the mapped ring; the device writes it.
        let ready = unsafe { std::ptr::read_volatile(self.host_slot_ptr(slot).add(self.n_experts)) }
            == word;
        if ready {
            std::sync::atomic::fence(Ordering::Acquire);
        }
        ready
    }

    /// Wait for [`Self::ready`]. A word that never arrives means the device
    /// faulted or the forward thread's chain died; that is reported rather
    /// than waited out.
    pub(crate) fn wait(&self, slot: usize, word: u32) -> Result<()> {
        let start = std::time::Instant::now();
        let mut spins = 0u32;
        while !self.ready(slot, word) {
            spins += 1;
            if spins < 1024 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                if start.elapsed() > SUMMARY_DEADLINE {
                    candle::bail!(
                        "expert pipeline: routing summary for slot {slot} never arrived — the \
                         device stopped before this layer's bucketize"
                    );
                }
            }
        }
        Ok(())
    }

    /// The summary in `slot`: `count | pinned << 29 | cold << 30 | decode << 31`
    /// per expert.
    ///
    /// # Safety
    ///
    /// [`Self::ready`] must have returned true for this invocation, and the
    /// caller must not have released the slot to the forward thread yet.
    pub(crate) unsafe fn read(&self, slot: usize) -> &[u32] {
        std::slice::from_raw_parts(self.host_slot_ptr(slot), self.n_experts)
    }
}

/// How long the pipeline thread waits for a routing summary before it decides
/// the device is not going to produce one. Far above any layer's compute — a
/// worker traps after [`SPIN_LIMIT_NS`] — so it fires only on a dead device.
const SUMMARY_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

/// The sequence word bucketize stores for invocation `seq`: never zero, which
/// is what a fresh slot holds.
pub(crate) fn summary_word(seq: u64) -> u32 {
    (seq as u32).wrapping_add(1).max(1)
}

impl Drop for SummaryRing {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; the threads that read
        // it are joined before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// The word every waiting worker polls, and the means to raise it.
///
/// Raised when a cold expert can never be published — a failed pack read, a
/// stager or pipeline thread that died. Every waiting worker then traps, the
/// next synchronizing call reports the sticky error, and no token computed
/// from the layer is returned.
///
/// The word lives in mapped pinned memory and is raised with a plain host
/// store: raising it must not need the driver.
pub(crate) struct AbortWord {
    host: *mut u32,
    dev: u64,
}

// SAFETY: a mapped word written only by `raise` (a volatile store of 1) and
// read by the device and by `is_raised`.
unsafe impl Send for AbortWord {}
unsafe impl Sync for AbortWord {}

impl AbortWord {
    pub(crate) fn new() -> Result<Self> {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, 4, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert cache: mapped abort word allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert cache: abort word has no device address: {r:?}");
        }
        // SAFETY: four bytes just allocated.
        unsafe { std::ptr::write_volatile(raw as *mut u32, 0) };
        Ok(Self {
            host: raw as *mut u32,
            dev,
        })
    }

    /// The word's device address, for `MoeLive::abort`.
    pub(crate) fn ptr(&self) -> u64 {
        self.dev
    }

    /// End every wait, now and later. Idempotent.
    pub(crate) fn raise(&self) {
        // SAFETY: the mapped word this struct owns.
        unsafe { std::ptr::write_volatile(self.host, 1) };
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    pub(crate) fn is_raised(&self) -> bool {
        // SAFETY: the mapped word this struct owns.
        unsafe { std::ptr::read_volatile(self.host) != 0 }
    }
}

impl Drop for AbortWord {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; every reader is
        // joined or synchronized before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut std::ffi::c_void);
        }
    }
}

/// The forward thread's pass, shared with the pipeline thread under one lock.
///
/// A **pass** is a run of invocations with strictly increasing row: every trunk
/// forward, every sub-forward of a split prefill, every MTP draft step. Every
/// invocation begins by taking this lock and counting itself in `reserved`. A
/// boundary move (§9) runs only with every begun invocation served by the
/// pipeline thread, and holds this lock while it does, so no invocation can
/// begin under it.
pub(crate) struct PassState {
    pub(crate) pass: u64,
    pub(crate) reserved: u64,
}

/// The forward thread's own counters.
struct ForwardSide {
    seq: u64,
    last_row: Option<usize>,
}

/// The two pinned host ranges a remote entry lies in: the warm tier's pinned
/// part, and the pad.
pub(crate) type PinnedRanges = [(u64, u64); 2];

/// One invocation's host-side reservation, made by [`Dispatch::before`] and
/// consumed by [`Dispatch::record`]: its sequence number, the summary-ring slot
/// its bucketize writes, and its MoE row. The stager and the pipeline thread
/// have been told to expect it, so it must be recorded — dropping it leaves
/// both waiting on a summary word that never comes. Not `Copy`: `record` takes
/// it by value, so it is recorded once.
#[derive(Debug)]
#[must_use = "a reserved invocation must be recorded: the pipeline thread and the stager wait on its summary word"]
pub(crate) struct Reserved {
    seq: u64,
    slot: usize,
    row: usize,
}

/// Bucketize's owner check (`moe_bucketize.cu`, OWNER CHECK) and, with
/// `tensor-assert`, the slot tags it reads — held here, beside the launches
/// that read them, so they outlive every launch. Without the feature the check
/// is all zero and bucketize checks nothing.
pub(crate) struct OwnerTags {
    check: OwnerCheck,
    #[cfg(feature = "tensor-assert")]
    _owners: Arc<SlotOwners>,
}

impl OwnerTags {
    /// The check over `owners`, whose slot `s` ends `s · slot_bytes` below
    /// `zone_end`.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn new(owners: Arc<SlotOwners>, zone_end: u64, slot_bytes: usize) -> Self {
        Self {
            check: owners.check(zone_end, slot_bytes),
            _owners: owners,
        }
    }

    /// No tags: bucketize checks nothing.
    #[cfg(not(feature = "tensor-assert"))]
    pub(crate) fn none() -> Self {
        Self {
            check: OwnerCheck::default(),
        }
    }
}

/// Everything the device-side expert forward needs that the host threads do
/// not own.
pub(crate) struct Dispatch {
    pub(crate) table: Arc<LiveTable>,
    pub(crate) ring: Arc<SummaryRing>,
    pub(crate) abort: Arc<AbortWord>,
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
    /// Profile build only: per-row worker counters `[rows × 5]` (see
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
}

impl Dispatch {
    /// `slot_bytes` is the largest 32-row tile slice of any projection — one
    /// worker's scratch slot.
    pub(crate) fn new(
        device: &CudaDevice,
        table: Arc<LiveTable>,
        pinned: PinnedRanges,
        slot_bytes: usize,
        k: usize,
        promo: Option<Arc<PromotionRing>>,
        owner: OwnerTags,
    ) -> Result<Self> {
        let n_experts = table.n_experts();
        let ring = Arc::new(SummaryRing::new(n_experts)?);
        let abort = Arc::new(AbortWord::new()?);
        let clock = Arc::new(ReclaimClock::new(StartedRows::mapped(table.n_rows())?));
        // Sized here for the widest wave the engine composes — its prefill
        // ceiling, and as many decode and verify rows again — so the forward
        // never grows it. One workspace serves every layer: each layer's
        // tables are consumed by its own launches before the next layer's
        // bucketize writes them, in stream order.
        let workspace = MoeBucketizeWorkspace::new(device, 2 * MAX_PREFILL_TOKENS, k)?;
        // SAFETY (all three): written by bucketize before any reader.
        let snap = unsafe { device.alloc::<u64>(3 * n_experts)? };
        let remote = unsafe { device.alloc::<i32>(4 * n_experts)? };
        let counters = unsafe { device.alloc::<i32>(3)? };
        let remote_dst = unsafe { device.alloc::<u64>(n_experts)? };
        // SAFETY: each worker writes its slot before reading it.
        let scratch = unsafe { device.alloc::<u8>(WORKERS_PREFILL * slot_bytes)? };
        #[cfg(feature = "profile")]
        let stall = device.memcpy_stod(&vec![0u64; table.n_rows() * 5])?;
        Ok(Self {
            table,
            ring,
            abort,
            clock,
            pass: Arc::new(Mutex::new(PassState {
                pass: 0,
                reserved: 0,
            })),
            served: Arc::new(AtomicU64::new(0)),
            staged: Arc::new(AtomicU64::new(0)),
            pinned,
            workspace: Mutex::new(workspace),
            forward: Mutex::new(ForwardSide {
                seq: 0,
                last_row: None,
            }),
            snap,
            remote,
            counters,
            promo,
            remote_dst,
            scratch,
            slot_bytes,
            owner,
            #[cfg(feature = "profile")]
            stall,
            #[cfg(feature = "profile")]
            device: device.clone(),
        })
    }

    /// Read and zero the worker counters. Synchronizes the device — a profile
    /// snapshot, between phases, never inside a forward.
    #[cfg(feature = "profile")]
    pub(crate) fn drain_worker_counters(&self) -> Result<WorkerCounters> {
        let rows = self.device.memcpy_dtov(&self.stall)?;
        let stream = self.device.cuda_stream();
        let (ptr, _g) = self.stall.device_ptr(&stream);
        // SAFETY: the counter block this struct owns; the device is synchronized
        // (the read above), so no worker is adding to it.
        unsafe { cudarc::driver::result::memset_d8_sync(ptr, 0, rows.len() * 8) }
            .map_err(candle::Error::wrap)?;
        let mut c = WorkerCounters::default();
        for r in rows.chunks_exact(5) {
            c.cold_wait_ns += r[0];
            c.items += r[1];
            c.bytes += r[2];
            c.copy_ns += r[3];
            c.launches += r[4];
        }
        Ok(c)
    }

    /// Number this invocation, and start a new pass if its row is not past the
    /// previous one's. Returns `(seq, pass)`.
    fn begin_invocation(&self, row: usize) -> Result<(u64, u64)> {
        let (seq, new_pass) = {
            let mut f = self
                .forward
                .lock()
                .map_err(|_| candle::Error::Msg("expert dispatch: forward state poisoned".into()))?;
            let new_pass = f.last_row.is_none_or(|last| row <= last);
            f.last_row = Some(row);
            let seq = f.seq;
            f.seq += 1;
            (seq, new_pass)
        };
        let mut p = self
            .pass
            .lock()
            .map_err(|_| candle::Error::Msg("expert dispatch: pass state poisoned".into()))?;
        p.reserved += 1;
        if new_pass {
            p.pass += 1;
        }
        Ok((seq, p.pass))
    }

    /// Hold invocation `ticket` until both readers of the summary ring have
    /// read the slot's previous tenant, ticket `ticket - SUMMARY_RING`.
    fn hold_for_ring(&self, ticket: u64) -> Result<()> {
        let need = ticket.saturating_sub(SUMMARY_RING as u64);
        let mut spins = 0u32;
        while self.served.load(Ordering::Acquire) < need || self.staged.load(Ordering::Acquire) < need
        {
            if self.abort.is_raised() {
                candle::bail!("expert pipeline or stager died — the layer cannot be served");
            }
            spins += 1;
            if spins < 1024 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    /// The routed experts of one MoE layer, on the device.
    ///
    /// `acts` is the layer's q8a128 activation `[n_tokens, hidden]`; `weights`
    /// and `indices` the router's `[n_tokens, k]` output (f32, u32). Returns the
    /// routed sum `[n_tokens, hidden]` at `out_dtype`. The tokens in `decode`
    /// are decode rows, which the residency scoring weights differently.
    ///
    /// Three parts, in this order: [`Self::before`] (the host protocol — number
    /// the invocation, hold its summary-ring slot, enqueue its reclaim ticket,
    /// tell the stager and the pipeline thread), [`Self::record`] (the launches
    /// and nothing else) and [`Self::after`] (flush the submission). The
    /// messages go out before the launches because neither receiver assumes the
    /// summary word is published yet: the pipeline thread waits on the ring slot
    /// and the stager polls it. That split is what lets the launches be recorded
    /// into a graph and replayed between an unchanged `before` and `after`
    /// (`docs/decode_graphs.md` §4.5).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<'w>(
        &self,
        tx: &mpsc::SyncSender<PipelineMessage>,
        stager: &mpsc::Sender<StagerMsg>,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        row: usize,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let Device::Cuda(cuda_dev) = indices.device() else {
            candle::bail!("expert dispatch: expected a CUDA device")
        };
        let (num_tokens, _) = indices.dims2()?;
        let reserved = self.before(tx, stager, row, num_tokens)?;
        let ys = self.record(reserved, acts, weights, indices, decode, out_dtype, wave)?;
        self.after(cuda_dev)?;
        Ok(ys)
    }

    /// The host protocol ahead of one invocation's launches: its sequence
    /// number, its summary-ring slot held free of the previous tenant, and the
    /// `Routed` messages to the stager and the pipeline thread. The reclaim
    /// rule's key is not set here: bucketize stores the ticket into its row's
    /// started word when the device begins it (`started.rs`). What is set is
    /// the row's enqueued ticket, which marks it upcoming until the device
    /// begins it — a victim preference, not a key (`ReclaimClock::upcoming`).
    fn before(
        &self,
        tx: &mpsc::SyncSender<PipelineMessage>,
        stager: &mpsc::Sender<StagerMsg>,
        row: usize,
        num_tokens: usize,
    ) -> Result<Reserved> {
        let (seq, pass) = self.begin_invocation(row)?;
        let ticket = seq + 1;
        let slot = (seq % SUMMARY_RING as u64) as usize;
        self.hold_for_ring(ticket)?;
        self.clock.enqueued(row, ticket);

        let t = profile_now();
        let word = summary_word(seq);
        stager
            .send(StagerMsg::Routed {
                row,
                slot,
                summary_word: word,
                ticket,
            })
            .map_err(|_| candle::Error::Msg("expert stager died — channel closed".into()))?;
        tx.send(PipelineMessage::Routed(RoutedLayer {
            row,
            pass,
            slot,
            summary_word: word,
            ticket,
            prefill_width: num_tokens > PREFILL_LAUNCH_TOKENS,
            submitted_at: profile_now(),
        }))
        .map_err(|_| candle::Error::Msg("expert pipeline thread died — channel closed".into()))?;
        crate::models::profile::pipeline_record("moe:route_handoff", t);
        Ok(Reserved { seq, slot, row })
    }

    /// Submit what [`Self::record`] queued. Both host threads wait on the
    /// summary word bucketize writes, and nothing on the forward thread
    /// synchronizes, so on WDDM nothing else would flush it. Inside a wave
    /// capture this is where a segment ends: the recorded launches, bucketize
    /// among them, are launched as one graph before the query.
    fn after(&self, device: &CudaDevice) -> Result<()> {
        device.flush_launches()
    }

    /// The launches of one reserved invocation, and nothing else: bucketize,
    /// gather, the three grouped GEMMs, the SwiGLU and the scatter.
    #[allow(clippy::too_many_arguments)]
    fn record<'w>(
        &self,
        reserved: Reserved,
        acts: Q8a128Operand<'w>,
        weights: &LiveTensor<'_>,
        indices: &LiveTensor<'_>,
        decode: &DecodeRows,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let device = indices.device().clone();
        let Device::Cuda(cuda_dev) = &device else {
            candle::bail!("expert dispatch: expected a CUDA device")
        };
        let Reserved { seq, slot, row } = reserved;
        let (num_tokens, k) = indices.dims2()?;
        let hidden_dim = acts.cols;
        let table = &*self.table;
        let n_experts = table.n_experts();
        let (gate_dtype, down_dtype) = (table.gate_dtype(row), table.down_dtype(row));
        // **The token-tile width, chosen per launch from the expected rows per
        // expert** — the tile width is the GEMM's weight-reuse factor, so a prefill at
        // ~100–300 rows per expert run at the decode width re-streams and re-dequants
        // every expert 2–4× per projection. Rows per expert are not known without the
        // routing readback this path exists to avoid; `n_tokens·k / E` is what uniform
        // routing would give, a lower bound on the rows of an active expert and, at
        // prefill widths, near it — nearly every expert is active. One width for the
        // three projections, since they share one tile table, and wide only where both
        // dtypes have the wide kernels. Measured at the fixed decode width (RTX 3090):
        // Qwen3-30B-A3B prefill at ×10 ran 11% under wide tiles, Qwen3.8-Flash-Next
        // at ×8–×16 25–33%.
        let n_sub = grouped_int8_n_sub(
            (num_tokens * k) / n_experts.max(1),
            gate_dtype.is_ko() && down_dtype.is_ko(),
        );
        let tile_w = 16 * n_sub;
        let compute = cuda_dev.cuda_stream();
        let weights_flat = weights.flatten_all()?.contiguous()?;

        let mut ws = self
            .workspace
            .lock()
            .map_err(|_| candle::Error::Msg("moe bucketize workspace poisoned".into()))?;
        let snap = self.snap.device_ptr(&compute).0;
        let remote = self.remote.device_ptr(&compute).0;
        let counters = self.counters.device_ptr(&compute).0;
        let remote_dst = self.remote_dst.device_ptr(&compute).0;
        let g = gpu_span("moe:bucketize", &device);
        moe_bucketize(
            indices,
            n_experts,
            tile_w,
            &mut ws,
            Some(&BucketizeLive {
                gate_row: table.row_ptr(Proj::Gate, row),
                table_plane: table.plane(),
                snap,
                pinned: self.pinned,
                summary: self.ring.dev_slot(slot),
                summary_seq: summary_word(seq),
                remote,
                counters,
                row: row as i32,
                // Every launch is offered the ring; a prompt passing over the
                // table — more claiming experts than the ring's sweep word —
                // claims nothing and stays in scratch, decided by bucketize
                // for this launch alone, so a decode row co-batched with a
                // prompt costs at most a deferred claim on the next narrow
                // launch.
                promo: self.promo.as_ref().map(|p| p.ring()),
                remote_dst,
                started_rows: self.clock.started_ptr(),
                ticket: seq + 1,
                owner: self.owner.check,
            }),
            decode,
        )?;
        g.end();

        // ── The expert chain ──
        let a_ub = num_tokens * k;
        // Tight data-independent tile bound: full tiles ≤ ⌈a_ub/tile_w⌉ and each
        // expert adds at most one partial tile.
        let launch_tiles = a_ub.min(a_ub.div_ceil(tile_w) + n_experts);
        let hp = ws.header.device_ptr(&compute).0;
        let live = |proj: Proj| MoeLive {
            abort: self.abort.ptr(),
            live_row: table.row_ptr(proj, row),
            remote,
            remote_dst,
            header: hp,
            counter: counters + 4 * proj as u64,
            scratch: self.scratch.device_ptr(&compute).0,
            slot_bytes: self.slot_bytes as u64,
            dst_offset: table.offset(proj, row),
            #[cfg(feature = "profile")]
            stall: self.stall.device_ptr(&compute).0 + (row * 5 * 8) as u64,
            #[cfg(not(feature = "profile"))]
            stall: 0,
            spin_limit_ns: SPIN_LIMIT_NS,
            workers: workers_for(num_tokens) as i32,
        };

        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint_q8a128;
            let (r, c, n) = (acts.rows, acts.cols, acts.byte_len());
            let nm = candle::tensor_assert::site("moe.norm_out.L", row);
            acts.with_device_ptr(cuda_dev, |p| unsafe {
                checkpoint_q8a128(nm, p, r, c, n, cuda_dev)
            })?;
        }
        let g = gpu_span("moe:gather", &device);
        let stacked = fused_moe_gather_q8a128(&acts, &ws.tok_ids, a_ub, cuda_dev, wave_root(wave))?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint_q8a128;
            let (r, c, n) = (stacked.rows, stacked.cols, stacked.byte_len());
            let nm = candle::tensor_assert::site("moe.gathered.L", row);
            stacked.with_device_ptr(cuda_dev, |p| unsafe {
                checkpoint_q8a128(nm, p, r, c, n, cuda_dev)
            })?;
        }

        let g = gpu_span("moe:gate", &device);
        let gate_out = grouped_qmatmul_dev_q8a128(
            &stacked,
            &self.snap,
            0,
            n_experts,
            gate_dtype,
            table.gate_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Gate)),
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::{capture_gate_gemm, GemmCall};
            capture_gate_gemm(
                &GemmCall {
                    layer: row,
                    stacked: &stacked,
                    weight_ptrs: &self.snap,
                    expert_base: 0,
                    num_experts: n_experts,
                    weight_dtype: gate_dtype,
                    weight_nrows: table.gate_nrows(),
                    tile_expert: &ws.tile_expert,
                    tile_b_start: &ws.tile_b_start,
                    tile_b_cnt: &ws.tile_b_cnt,
                    launch_tiles,
                    out: &gate_out,
                },
                cuda_dev,
            )?;
        }
        let g = gpu_span("moe:up", &device);
        let up_out = grouped_qmatmul_dev_q8a128(
            &stacked,
            &self.snap,
            n_experts,
            n_experts,
            gate_dtype, // up shares gate's KO dtype
            table.gate_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Up)),
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        {
            use crate::models::nan_capture::checkpoint;
            use candle::tensor_assert::site;
            checkpoint(
                site("moe.up_out.L", row),
                &up_out,
                &[("gate_out", &gate_out)],
                cuda_dev,
            )?;
        }
        let g = gpu_span("moe:silu", &device);
        // Raw Σx — a language model's SwiGLU intermediate stays orders of
        // magnitude below f16's 65504; the down matmul reads this operand's own
        // `sum_scale`, so the two agree by construction.
        let inter_acts = silu_mul_q8a128(
            &gate_out,
            &up_out,
            cuda_dev,
            gate_out.cuda_backing(),
            SumScale::Raw,
        )?;
        g.end();
        let g = gpu_span("moe:down", &device);
        let down_out = grouped_qmatmul_dev_q8a128(
            &inter_acts,
            &self.snap,
            2 * n_experts,
            n_experts,
            down_dtype,
            table.down_nrows(),
            &ws.tile_expert,
            &ws.tile_b_start,
            &ws.tile_b_cnt,
            launch_tiles,
            n_sub,
            Some(&live(Proj::Down)),
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("moe.down_out.L", row),
            &down_out,
            &[],
            cuda_dev,
        )?;
        // The int8 matmul emits F32 and the scatter reads F32, narrowing once
        // at its store into `ys`'s dtype (hot-path invariant 1). The combine
        // target is defined in full by the scatter, so it is uninitialised
        // (invariant 6).
        let ys = wave_empty((num_tokens, hidden_dim), out_dtype, &device, wave)?;
        let g = gpu_span("moe:scatter", &device);
        fused_deterministic_scatter(
            &ys,
            &down_out,
            &ws.perm,
            &weights_flat,
            &ws.rw_ids,
            &ws.token_starts,
            num_tokens,
            cuda_dev,
        )?;
        g.end();
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("moe.routed_out.L", row),
            &ys,
            &[("down_out", &down_out)],
            cuda_dev,
        )?;
        Ok(ys)
    }
}

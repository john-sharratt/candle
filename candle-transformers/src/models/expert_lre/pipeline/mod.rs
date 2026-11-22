//! The expert pipeline thread: it decides which experts live in VRAM, off the
//! critical path of the GPU.
//!
//! The GPU never waits on this thread. The GPU promotes its misses into slots
//! this thread keeps stocked in the promotion ring, and this thread writes each
//! row's read-ahead prediction into the ring for launches that have yet to begin.
//! Eviction points an expert's entry back at its pinned copy under the reclaim
//! rule, and a slot is reused only once every invocation that could have
//! snapshotted its address is done. The thread alone owns the cache state and the
//! copy stream. The per-expert places it shares with the stager sit under the
//! residency lock.
//!
//! - `device`: lands promotions the GPU has filled and drops a faulted forward's demand claims.
//! - `listing`: the read-ahead listings and the experts staged for them.
//! - `prediction`: predictions of the rows ahead, kept until they are judged.
//! - `ring`: the promotion ring's stock and its lazy victims.
//! - `routing`: serves one routed layer.
//! - `spawn`: starts the thread and its dead-thread guard.
//! - `tests_ring`: tests for the ring's stock and fault handling.

use super::ahead_pins::{AheadPins, Candidate};
use super::blend::{cell_axes, gather, CellTable, MARKOV_MIN_CONF};
use super::cache::{ExpertCacheInner, DECODE_RECENCY_DECAY};
use super::cpu_pool::params::shared as cpu_params;
use super::dispatch::{AbortWord, PassState, SummaryRing};
use super::fault::FaultWord;
use super::layer_trace::{publish as publish_layer_trace, RowRouting, TraceBuilder};
use super::pinned::LayerGeometry;
use super::promo::{ticket_from_word, PromotionRing, Victim, AHEAD_CAP};
use super::read_ahead::{listing_readable, median, read_ahead_window, READ_AHEAD_DEPTH};
use super::read_ahead_gate::{link_us, LinkTime, ReadAheadGate};
use super::reclaim::ReclaimClock;
use super::regret::{eviction_costs, just_ahead, prediction_band, RegretLedger, VictimTag, TAGS};
use super::residency::{Fallback, Residency};
use super::routing_trace::RoutingTrace;
use super::slot_image::{build_slot_view, slot_offsets};
use super::stack_distance::{marginal_window, MarginalSlots};
use super::stager::StagerMsg;
use super::transition::{TransitionMatrix, HOPS};
use super::types::{PipelineMessage, PipelineStats, RoutedLayer};
use super::votes::{ranked, voted_set, VoteRing, Votes};
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::{Device, Result};
use candle_kernels::simple::moe_bucketize::{AHEAD_MAX, SUMMARY_COUNT_MASK, SUMMARY_CPU_BIT};
use cudarc::driver::CudaStream;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::Instant;

/// The fewest free slots the promotion ring is kept stocked with.
///
/// Above it the stock is predicted: the next few rows' non-resident experts at
/// the share of experts the current row routed (`process_routed`), and stock
/// past that is taken back when the ring is quiet (`trim_ring`). Every stocked
/// slot is an expert evicted ahead of need, so a standing deep stock is wrong —
/// on the qwen36 gate (RTX 3090) 256 cost C10×16 1,692 misses where 32 cost 278
/// — and so is one sized from the row just served: the startup fill leaves the
/// early layers resident and the late ones cold, so a pass reaching its first
/// cold layer found 32 slots for ~256 misses, and the rest crossed the link a
/// second time by the copy engine (756 of a cold prefill's, 1,169 t/s against
/// 1,382 with the prediction).
const RING_TARGET: usize = 32;

/// How many rows past the one just served the promotion ring is stocked for.
/// The stock has to be in the ring when the GPU's bucketize of those rows reads
/// it, and this thread stocks only after reading the served row's summary — a
/// refill per routed layer, made with no driver call, so it trails the GPU by
/// about a layer.
const RING_HORIZON: usize = 2;

/// Prompt-only experts are promoted only when the expert zone can hold at
/// least this share of the model's experts (as `numerator / denominator`) —
/// at its limit, a property of the model and the card, not at the size the KV
/// side leaves it at the moment: the zone starts small and shrinks under a
/// wide wave, and keying on that switched promotion off exactly when a cold or
/// wide prompt needed it.
///
/// A prompt sweeps the expert table about once, so what promoting its misses
/// buys depends on room. With most of the model resident there is little to
/// evict and a promoted prompt expert is often read again (Qwen3.6-35B-A3B on
/// an RTX 3090, ~95% at the limit: its prefill falls up to 20% unpromoted).
/// With the working set several times the zone, a prompt cycles the whole zone
/// through its own one-shot experts (Qwen3.8-Flash-Next on the same card, ~48%:
/// promoted, a two-sequence prompt took ~12,600 promotions into a ~12,000-slot
/// zone), and its prefill rows run up to 5% faster unpromoted — the workers'
/// copies of a miss are all a prompt needs. The two models sit far apart;
/// three quarters divides them.
const PROMPT_ROOM: (usize, usize) = (3, 4);

/// The summary bits the pipeline reads. The count is the field below the CPU
/// class's bit (`moe_bucketize_cpu.cuh`), which marks an expert the host pool
/// computed: still a miss for scoring, but no copy over the link.
const SUMMARY_COUNT: u32 = SUMMARY_COUNT_MASK;

const SUMMARY_CPU: u32 = 1 << SUMMARY_CPU_BIT;

const SUMMARY_PINNED: u32 = 1 << 29;

const SUMMARY_COLD: u32 = 1 << 30;

const SUMMARY_DECODE: u32 = 1 << 31;

/// The pipeline thread's private state. Never crosses a thread boundary after
/// construction — the thread owns it exclusively with `&mut self`.
pub(crate) struct PipelineState {
    /// VRAM slots, eviction scores, the zone and its free list.
    pub(crate) inner: ExpertCacheInner,
    pub(crate) device: Device,
    /// The boundary moves' relocation copies.
    pub(crate) copy_stream: Arc<CudaStream>,
    pub(crate) residency: Arc<Mutex<Residency>>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) ring: Arc<SummaryRing>,
    pub(crate) abort: Arc<AbortWord>,
    /// The live launches' fault word: while set, no demand promotion lands.
    fault: Arc<FaultWord>,
    /// The forward thread's pass, shared for the boundary rule.
    pub(crate) pass_state: Arc<Mutex<PassState>>,
    /// The pass of the routed layer being served.
    pub(crate) pass: Option<u64>,
    /// The ticket of the last routed layer served — the forward thread's ring
    /// hold and the boundary rule's `served`. Written by this thread only.
    pub(crate) routed_served: Arc<AtomicU64>,
    /// Speculative staging requests for cold predicted experts.
    pub(crate) stager: mpsc::Sender<StagerMsg>,
    /// Per-layer geometry.
    pub(crate) layer_geometries: Arc<Vec<LayerGeometry>>,
    pub(crate) num_moe_layers: usize,
    /// True when every expert fits in VRAM — nothing to promote or prefetch.
    pub(crate) all_resident: bool,
    /// Online-learned transition matrix for speculative prefetch.
    pub(crate) transition_matrix: TransitionMatrix,
    /// `(layer, expert)` pairs read ahead and not yet judged: their layer's
    /// routing scores the prediction.
    pub(crate) speculative_loads: HashSet<(usize, usize)>,
    /// The experts the GPU is promoting — a miss's claim or a read-ahead.
    promoting: HashSet<(usize, usize)>,
    /// The promotion ring (none when every expert is in VRAM); the offer
    /// behind each ring index; the slots standing in it, which nothing else
    /// may evict, relocate or offer again until a claim or a withdrawal returns
    /// them; how far this thread has collected the device's takes; the slots
    /// the GPU is filling; and how many offers the ring is kept stocked with.
    promo_ring: Option<Arc<PromotionRing>>,
    ring_slots: Vec<Option<Offer>>,
    offered: HashSet<usize>,
    /// Standing offers whose victim's pad slot is pinned for them.
    pad_pinned_offers: usize,
    ring_taken: u32,
    device_promotions: Vec<DevicePromotion>,
    ring_target: usize,
    /// Read-ahead's window inputs: the link's measured rate (bytes/s), the
    /// largest slot image, the intervals between consecutive rows' summaries
    /// this pass, the last summary's arrival and row, and the window the ring
    /// carries — set once per pass from the previous pass's median interval.
    link_rate: f64,
    image_bytes: usize,
    layer_intervals: Vec<f64>,
    last_summary: Option<(Instant, usize)>,
    window: u32,
    /// The pass being walked, for the telemetry page's layer view.
    layer_trace: TraceBuilder,
    /// The pad pins read-ahead's listings hold (`ahead_pins`).
    ahead_pins: AheadPins,
    /// Per row, its latest prediction, best first — what the row's listing is
    /// vetted from: at the prediction, and again as the stager lands what it
    /// staged ahead for the row.
    predicted: Vec<Vec<usize>>,
    /// Speculative reads the stager has landed, `(row, expert)`: each a staged
    /// expert its row may now list from the pad.
    landed_ahead: mpsc::Receiver<(usize, usize)>,
    /// Every routed invocation served, for the residency references
    /// (`ExpertCache::hit_references`).
    trace: Arc<Mutex<RoutingTrace>>,
    /// The misses a zone a window larger would have served
    /// (`stack_distance`), over the evictable rows' experts.
    marginal: MarginalSlots,
    /// The router look-ahead's votes (`votes`), and per row and hop the latest
    /// prediction made for it, which the row's routing judges.
    votes: Arc<VoteRing>,
    pending: Vec<[Option<Prediction>; HOPS]>,
    /// The link time per decode invocation, and the forward's read of the
    /// decision it drives (`read_ahead_gate`); and the read-ahead claims
    /// collected since the last decode invocation was counted.
    link_time: LinkTime,
    read_ahead: Arc<ReadAheadGate>,
    ahead_uncounted: usize,
    /// What each combination of evidence has been worth, per hop (`blend`).
    cells: CellTable,
    /// Claimed victims awaiting their row's next routing (`regret`), and the
    /// latest measured mean pack read — what a pack-only miss waits on.
    regret: RegretLedger,
    pack_read_secs: Option<f64>,
    /// Timing accumulator for pipeline spans.
    pub(crate) profile: ProfileAccumulator,
    /// Shared telemetry counters (always-on).
    pub(crate) stats: Arc<Mutex<PipelineStats>>,
    /// The pinned experts' bytes while a guest stands in their ground
    /// ([`BoundaryAsk::StashPinned`]), with the frontier they occupied.
    pub(crate) pinned_stash: Option<(u64, Vec<u8>)>,
}

impl PipelineState {
    /// The fields with no startup state of their own.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        inner: ExpertCacheInner,
        device: Device,
        copy_stream: Arc<CudaStream>,
        residency: Arc<Mutex<Residency>>,
        clock: Arc<ReclaimClock>,
        ring: Arc<SummaryRing>,
        abort: Arc<AbortWord>,
        fault: Arc<FaultWord>,
        pass_state: Arc<Mutex<PassState>>,
        routed_served: Arc<AtomicU64>,
        stager: mpsc::Sender<StagerMsg>,
        landed_ahead: mpsc::Receiver<(usize, usize)>,
        layer_geometries: Arc<Vec<LayerGeometry>>,
        all_resident: bool,
        promo_ring: Option<Arc<PromotionRing>>,
        link_rate: f64,
        stats: Arc<Mutex<PipelineStats>>,
        trace: Arc<Mutex<RoutingTrace>>,
        votes: Arc<VoteRing>,
        read_ahead: Arc<ReadAheadGate>,
    ) -> Result<Self> {
        let num_moe_layers = layer_geometries.len();
        let experts = inner.experts_per_layer;
        let ring_cap = promo_ring.as_ref().map_or(0, |r| r.cap());
        let image_bytes = layer_geometries
            .iter()
            .map(|g| slot_offsets(g).3)
            .max()
            .unwrap_or(0);
        if let Some(ring) = &promo_ring {
            ring.set_depth(READ_AHEAD_DEPTH);
        }
        if let Ok(mut s) = stats.lock() {
            s.prefetch_depth = READ_AHEAD_DEPTH as usize;
        }
        Ok(Self {
            inner,
            device,
            copy_stream,
            residency,
            clock,
            ring,
            abort,
            fault,
            pass_state,
            pass: None,
            routed_served,
            stager,
            layer_geometries,
            num_moe_layers,
            all_resident,
            transition_matrix: TransitionMatrix::new(num_moe_layers, experts),
            speculative_loads: HashSet::new(),
            promoting: HashSet::new(),
            promo_ring,
            ring_slots: vec![None; ring_cap],
            offered: HashSet::new(),
            pad_pinned_offers: 0,
            ring_taken: 0,
            device_promotions: Vec::new(),
            ring_target: RING_TARGET.min(ring_cap / 2),
            link_rate,
            image_bytes,
            layer_intervals: Vec::new(),
            last_summary: None,
            window: 0,
            layer_trace: TraceBuilder::default(),
            ahead_pins: AheadPins::new(num_moe_layers),
            predicted: vec![Vec::new(); num_moe_layers],
            landed_ahead,
            trace,
            marginal: MarginalSlots::new(num_moe_layers * experts),
            votes,
            pending: vec![Default::default(); num_moe_layers],
            link_time: LinkTime::new(all_resident),
            read_ahead,
            ahead_uncounted: 0,
            cells: CellTable::default(),
            regret: RegretLedger::new(num_moe_layers),
            pack_read_secs: None,
            profile: ProfileAccumulator::new(),
            stats,
            pinned_stash: None,
        })
    }

    pub(crate) fn residency(&self) -> Result<MutexGuard<'_, Residency>> {
        self.residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))
    }
}

mod device;
mod listing;
mod prediction;
mod ring;
mod routing;
mod spawn;
#[cfg(test)]
mod tests_ring;

use device::*;
use prediction::*;
use ring::*;
pub(crate) use spawn::*;

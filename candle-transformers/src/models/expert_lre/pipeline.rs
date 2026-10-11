//! The pipeline thread — VRAM residency policy, off the critical path.
//!
//! The GPU never waits on this thread. A routed expert that is not in VRAM is
//! computed by its layer's GEMM workers from pinned memory (a warm or pad
//! slot), or from the pad once the stager has staged it; this thread decides
//! which experts *live* in VRAM, and moves bytes only with the copy engine:
//!
//! - **Promotion of misses, by the GPU.** This thread keeps the promotion ring
//!   (`promo`) stocked with free VRAM slots (free, or victims the reclaim rule
//!   allows). Bucketize hands them to the layer's remote experts, the GEMM
//!   workers write each slice they copy into the slot as well as their scratch,
//!   and once the invocation has completed this thread points the expert's
//!   entry at the slot: a miss is promoted with no second crossing of the link.
//! - **Read-ahead, by the GPU too.** The Markov transition matrix predicts the
//!   next layers' experts (a row routing most of its experts predicts the next
//!   rows' scored ones instead), and this thread writes each row's prediction
//!   into the promotion ring (`PromotionRing::predict`) while a launch that
//!   reads it has yet to begin. A later launch claims offers for the predicted
//!   experts whose pinned copy this thread vetted (a warm slot, or a pad slot
//!   it pinned — `ahead_pins`) with the link time its own misses leave
//!   (`read_ahead`), and its gate launch's workers copy them into the slots and
//!   publish their entries — no copy issued here and no landing waited for. A
//!   predicted cold expert is handed to the stager to stage into the pad, and
//!   listed from there as the stager reports it landed.
//! - **Eviction** points an entry back at the expert's pinned copy (or 0) and
//!   frees the slot under the reclaim rule (`reclaim`): an entry may go to 0
//!   only for a quiet row, and a slot is reused only once every invocation that
//!   could have snapshotted its address is done.
//!
//! It exclusively owns the [`ExpertCacheInner`], the copy stream and the
//! [`TransitionMatrix`]. The per-expert places it shares with the stager live in
//! the [`Residency`] lock, which also writes the live table.

use super::ahead_pins::{AheadPins, Candidate};
use super::blend::{cell_axes, gather, CellTable, MARKOV_MIN_CONF};
use super::cache::{ExpertCacheInner, DECODE_RECENCY_DECAY};
use super::dispatch::{AbortWord, PassState, SummaryRing};
use super::fault::FaultWord;
use super::pinned::LayerGeometry;
use super::promo::{ticket_from_word, PromotionRing, Victim, AHEAD_CAP};
use super::read_ahead::{listing_readable, median, read_ahead_window, READ_AHEAD_DEPTH};
use super::read_ahead_gate::{link_us, LinkTime, ReadAheadGate};
use super::reclaim::ReclaimClock;
use super::regret::{eviction_costs, just_ahead, prediction_band, RegretLedger, VictimTag, TAGS};
use super::residency::{Fallback, Residency};
use super::routing_trace::RoutingTrace;
use super::slot_image::{build_slot_view, slot_offsets};
use super::stager::StagerMsg;
use super::transition::{TransitionMatrix, HOPS};
use super::types::{PipelineMessage, PipelineStats, RoutedLayer};
use super::votes::{ranked, VoteRing, Votes};
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::{Device, Result};
use candle_kernels::simple::moe_bucketize::AHEAD_MAX;
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

/// What the promotion ring is kept at: how many offers it holds, the reserve
/// below which a prompt-only expert takes no slot, and the sweep word — the
/// most claiming experts a launch may have and still claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RingStock {
    target: usize,
    reserve: u32,
    sweep: u32,
}

/// The promotion ring's stock, from the stock the next rows' decode misses
/// want (`decode_ahead`), the stock all their misses want (`all_ahead`), and
/// the zone's **empty** slots — the free list plus the empty offers already
/// standing in the ring.
///
/// **Every empty slot is stocked.** A ring slot is where a worker's copy of a
/// missed expert lands and is computed from (`moe_live_worker`), so a miss that
/// finds one is promoted for the copy the layer made anyway. An empty slot
/// evicts nothing to stand there, so the ring holds all of them (to half its
/// capacity) whatever the prediction says; the prediction sizes only the stock
/// of lazy victims, which a claim evicts. Stocked to the prediction alone, a
/// 16 GB card's zone stayed ~1,400 slots short of full for a whole
/// Qwen3.6-35B-A3B run while its misses crossed the link and were thrown away.
///
/// With room (`PROMPT_ROOM`) nothing is reserved, so a prompt-only miss takes
/// the stock like a decode one — in a launch that is not a sweep. Without room
/// decode's own target is the reserve, so the stock a prompt-only expert may
/// take is only what stands past it: empty slots, never a victim decode would
/// not have made.
///
/// **The sweep word is the prediction, never the empties.** A launch whose
/// claiming experts outnumber what the next rows were predicted to want is a
/// prompt passing over the table, and claims nothing: its misses would evict
/// one-shot experts into the zone decode reuses. Counting empty slots into it
/// would let a cold prompt claim victims as soon as the zone had a hole. That
/// holds with room too: a prefill launch's misses run to thousands against a
/// prediction capped at half the ring, so even a model whose zone holds most of
/// its experts promotes a prompt's misses only from launches narrow enough to
/// fit the prediction — stencils, short prompts, the rows a decode step
/// verifies.
///
/// **Read-ahead stocks on top.** A launch reads ahead with up to `window` offers
/// past its own misses, so the stock grows by the window — in the target only:
/// the reserve and the sweep word decide what misses may claim, and read-ahead
/// spends only the stock above the reserve (`moe_bucketize.cu`). When decode's
/// own target reaches half the ring, the target is clamped there too and equals
/// the reserve: read-ahead then gets only the zone's empty slots, since the
/// ring has no room left to stock a window past what decode's misses want.
fn ring_stock(
    decode_ahead: usize,
    all_ahead: usize,
    prompt_room: bool,
    empty_slots: usize,
    window: usize,
    cap: usize,
) -> RingStock {
    let target = |ahead: usize| (ahead + ahead / 4).clamp(RING_TARGET, cap / 2);
    let empties = empty_slots.min(cap / 2);
    let stocked = |t: usize| (t + window).min(cap / 2).max(empties);
    if prompt_room {
        let all = target(all_ahead);
        return RingStock {
            target: stocked(all),
            reserve: 0,
            sweep: all as u32,
        };
    }
    let decode = target(decode_ahead);
    RingStock {
        target: stocked(decode),
        reserve: decode as u32,
        sweep: decode as u32,
    }
}

/// The summary bits the pipeline reads.
const SUMMARY_COUNT: u32 = 0x1fff_ffff;
const SUMMARY_PINNED: u32 = 1 << 29;
const SUMMARY_COLD: u32 = 1 << 30;
const SUMMARY_DECODE: u32 = 1 << 31;

/// Whether a device promotion lands now: its invocation is `complete`, and
/// either it is a read-ahead claim (which never waits on a cold expert) or no
/// forward has `faulted` — a demand claim's item may have been given up, its
/// slot never written, and it waits for the fault to be taken, which drops it
/// (`PipelineState::drop_demand_promotions`).
///
/// `faulted` is read only after `complete` was: a worker claims the fault word
/// and fences before any later launch begins, and a later launch's start is
/// what makes a claim complete — so a claim of the faulted launch judged
/// complete always reads the word set.
fn lands(complete: bool, ahead: bool, faulted: impl FnOnce() -> bool) -> bool {
    complete && (ahead || !faulted())
}

/// Pending promotions split into the read-ahead claims, which a fault leaves to
/// land, and the demand claims it drops — each in its pending order.
fn split_ahead(pending: Vec<DevicePromotion>) -> (Vec<DevicePromotion>, Vec<DevicePromotion>) {
    pending.into_iter().partition(|p| p.ahead)
}

/// One prediction for a row ahead, kept until that row's routing judges it: the
/// look-ahead's own list (for its precision and recall), every candidate any
/// predictor proposed that a copy could serve, with its cell (`blend`), the
/// list used, which of the row's experts were held — resident or being
/// promoted — when it was made, so the list is also judged on the experts a
/// copy could have served, and the held experts a predictor named, with their
/// cell: what an eviction of one of them knew it was throwing away (`regret`).
#[derive(Clone, Debug, Default)]
struct Prediction {
    votes: Option<Vec<usize>>,
    candidates: Vec<(usize, usize)>,
    selected: Vec<usize>,
    held: Vec<bool>,
    held_named: Vec<(usize, usize)>,
}

/// Markov candidates proposed per row ahead — past any cap, so the cells see
/// Markov's weaker evidence too.
const MARKOV_CANDIDATES: usize = 64;

/// Recently routed candidates proposed per row ahead, for the same reason.
const RECENT_CANDIDATES: usize = 64;

/// A slot the GPU is filling with a missed expert: whole once invocation
/// `ticket` has completed.
struct DevicePromotion {
    row: usize,
    expert: usize,
    slot: usize,
    ticket: u64,
    /// A read-ahead claim: its gate launch's workers copy it whole and publish
    /// it, whatever else the launch gives up (`fault`).
    ahead: bool,
}

/// What stands at one ring index: a zone slot, and — when it was offered as a
/// lazy victim — the resident expert it still holds and the fallback held for
/// it as long as the offer stands (a pad slot pinned, a cold one marked).
#[derive(Clone, Copy, Debug)]
struct Offer {
    slot: usize,
    victim: Option<((usize, usize), Fallback)>,
}

/// A lazy victim chosen for an offer: its slot, its expert, what the device is
/// told, and the fallback held for it.
struct Chosen {
    slot: usize,
    key: (usize, usize),
    victim: Victim,
    fallback: Fallback,
}

/// The share of the pad lazy victims may hold pinned (`1 / PAD_PIN_SHARE`). A
/// pinned pad slot cannot be evicted, and the stager stages every cold miss
/// into the pad: offers holding most of it would leave a cold wait nowhere to
/// land.
const PAD_PIN_SHARE: usize = 4;

/// The share of the pad read-ahead's listings may hold pinned
/// (`1 / AHEAD_PIN_SHARE`), beside the lazy victims' share: together they leave
/// the stager most of the pad for cold demand rows.
const AHEAD_PIN_SHARE: usize = 8;

/// Experts predicted for a row within two of the served one: too near for a
/// launch to read ahead (`read_ahead::listing_readable`), so the prediction
/// only stages its cold experts into the pad for the row's own workers, and
/// the pad — one layer of slots, shared with every demand row — is what
/// bounds it. A row past that is listed for read-ahead as well and takes the
/// ring's list (`AHEAD_CAP`).
///
/// 48 so the look-ahead's margin picks (`LOOK_AHEAD_MARGIN`) reach the stager:
/// at 32 a verify wave's routed picks alone filled the list. See the margin for
/// what the pair measured.
const STAGE_CAP: usize = 48;

/// The score a non-resident expert needs to be predicted for its row from
/// decode's own history (`scored_absent`): a decode routing credits 1.0 and a
/// pass decays it by `DECODE_RECENCY_DECAY` (0.85), so this admits an expert
/// decode routed there within the last four steps.
const RECENT_MIN_SCORE: f32 = 0.5;

/// How many experts a row is predicted at `hop` rows ahead.
fn prediction_cap(hop: usize) -> usize {
    if hop >= 3 {
        AHEAD_CAP
    } else {
        STAGE_CAP
    }
}

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
            ahead_pins: AheadPins::new(num_moe_layers),
            predicted: vec![Vec::new(); num_moe_layers],
            landed_ahead,
            trace,
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
        })
    }

    pub(crate) fn residency(&self) -> Result<MutexGuard<'_, Residency>> {
        self.residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))
    }

    /// Serve one routed layer: read its summary, score it, promote its misses,
    /// and prefetch ahead. Computes nothing and gates nothing — the layer's
    /// expert chain was enqueued without it.
    pub(crate) fn process_routed(&mut self, msg: RoutedLayer) -> Result<()> {
        let row = msg.row;

        // ── A new pass: the forward thread started its rows over ──
        if self.pass != Some(msg.pass) {
            if self.pass.is_some() {
                self.transition_matrix.reset_pass();
                self.set_read_ahead_window();
                self.set_eviction_order();
            }
            self.pass = Some(msg.pass);
            self.last_summary = None;
        }

        // ── The routing summary ──
        let t = profile_now();
        // While it waits, land what the GPU has finished promoting, so the
        // residency this thread ranks victims and predictions on is current,
        // and list what the stager has staged ahead while a launch can still
        // read the listing.
        self.list_landed()?;
        let ring = self.ring.clone();
        ring.wait(msg.slot, msg.summary_word, || {
            self.land_device_promotions(false)?;
            self.list_landed()
        })?;
        let arrived = Instant::now();
        if let Some((at, prev)) = self.last_summary {
            if row == prev + 1 {
                self.layer_intervals
                    .push(arrived.duration_since(at).as_secs_f64());
            }
        }
        self.last_summary = Some((arrived, row));
        self.clock.observe(msg.ticket);
        let lag = self.clock.latest_started().saturating_sub(msg.ticket);
        self.profile.record("pipe_routed_wait", t);
        // SAFETY: `wait` returned for this invocation, and the forward thread
        // does not rewrite the slot — or its vote slot — until `routed_served`
        // passes this ticket.
        let summary: Vec<u32> = unsafe { self.ring.read(msg.slot) }.to_vec();
        let votes =
            (msg.voted_hops > 0).then(|| unsafe { self.votes.read(msg.slot, msg.voted_hops) });
        self.routed_served.store(msg.ticket, Ordering::Release);

        let t = profile_now();
        self.collect_device_promotions(msg.ticket)?;
        self.land_device_promotions(false)?;
        self.release_ahead_pins(false)?;

        let mut expert_ids: Vec<usize> = Vec::new();
        let mut decode: HashSet<usize> = HashSet::new();
        let (mut pinned, mut cold) = (0usize, 0usize);
        for (e, &w) in summary.iter().enumerate() {
            if w & SUMMARY_COUNT == 0 {
                continue;
            }
            expert_ids.push(e);
            if w & SUMMARY_DECODE != 0 {
                decode.insert(e);
            }
            pinned += usize::from(w & SUMMARY_PINNED != 0);
            cold += usize::from(w & SUMMARY_COLD != 0);
        }
        if let Ok(mut t) = self.trace.lock() {
            let routed: Vec<(usize, u32, bool)> = expert_ids
                .iter()
                .map(|&e| (e, summary[e] & SUMMARY_COUNT, decode.contains(&e)))
                .collect();
            t.push(row, &routed, row < self.inner.pinned_layers);
        }
        // Claims are judged, and the row's visit recorded, by decode routing
        // alone: a prompt launch routes most of the row, so it would count
        // nearly every victim routed again and mark nearly every resident just
        // hit — the regret rates that order eviction (`set_eviction_order`)
        // and the recency passes would learn the prompt, not decode.
        if !msg.prefill_width {
            let v = self.regret.judge(row, msg.ticket, &expert_ids);
            if let Ok(mut s) = self.stats.lock() {
                for k in 0..2 {
                    s.claim_evicted[k] += v.evicted[k];
                    s.claim_regretted[k] += v.regretted[k];
                    s.claim_promoted[k] += v.promoted[k];
                    s.claim_paid[k] += v.paid[k];
                }
                for t in 0..TAGS {
                    s.victim_tag_evicted[t] += v.tag_evicted[t];
                    s.victim_tag_regretted[t] += v.tag_regretted[t];
                }
            }
            self.inner.mark_visit(row, &expert_ids);
        }

        // ── Read-ahead precision: of the experts read ahead for this row, the
        // ones it routed ──
        let spec_for_layer: Vec<usize> = self
            .speculative_loads
            .iter()
            .filter(|&&(l, _)| l == row)
            .map(|&(_, e)| e)
            .collect();
        if !spec_for_layer.is_empty() {
            let hits = spec_for_layer
                .iter()
                .filter(|e| expert_ids.binary_search(e).is_ok())
                .count();
            if let Ok(mut s) = self.stats.lock() {
                s.predicted_total += spec_for_layer.len();
                s.predicted_hits += hits;
            }
            for &eid in &spec_for_layer {
                if expert_ids.binary_search(&eid).is_ok() {
                    self.inner.record_prediction_hit(row, eid);
                }
            }
        }
        self.speculative_loads.retain(|&(l, _)| l != row);

        // ── The latest predictions for this row, judged — by decode routing
        // only, for the cells learn what decode's next rows route; a prompt
        // reaching the row first discards them ──
        for hop in 1..=HOPS {
            let Some(p) = self.pending[row][hop - 1].take() else {
                continue;
            };
            if msg.prefill_width {
                continue;
            }
            let routed = |e: &usize| expert_ids.binary_search(e).is_ok();
            let (mut union_hits, mut markov, mut recent) = (0, (0, 0), (0, 0));
            for &(e, cell) in &p.candidates {
                let hit = routed(&e);
                self.cells.record(hop, cell, hit);
                union_hits += usize::from(hit);
                let (_, m, r) = cell_axes(cell);
                if m > 0 {
                    markov.0 += 1;
                    markov.1 += usize::from(hit);
                }
                if r {
                    recent.0 += 1;
                    recent.1 += usize::from(hit);
                }
            }
            let selected_hits = p.selected.iter().filter(|e| routed(e)).count();
            let copy_routed = expert_ids
                .iter()
                .filter(|&&e| !p.held.get(e).copied().unwrap_or(false))
                .count();
            if let Ok(mut s) = self.stats.lock() {
                let h = hop - 1;
                if let Some(votes) = &p.votes {
                    s.lookahead_judged[h] += votes.len();
                    s.lookahead_hits[h] += votes.iter().filter(|e| routed(e)).count();
                    s.lookahead_routed[h] += expert_ids.len();
                }
                s.copy_routed[h] += copy_routed;
                s.union_hits[h] += union_hits;
                s.markov_judged[h] += markov.0;
                s.markov_hits[h] += markov.1;
                s.recent_judged[h] += recent.0;
                s.recent_hits[h] += recent.1;
                s.selected_judged[h] += p.selected.len();
                s.selected_hits[h] += selected_hits;
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.cells = self.cells;
        }

        // The predictor learns from decode routing alone: a prompt launch
        // routes most of the row, so its transitions are near-uniform and
        // would wash out the sharper statistics decode's steps carry. A cache
        // holding every expert never predicts (`predict_rows` below), so it
        // keeps no tables — at a wide decode step their update is hundreds of
        // thousands of scattered writes an invocation, on the thread every
        // host meeting point waits for.
        if !msg.prefill_width && !self.all_resident {
            let mut decode_ids: Vec<usize> = decode.iter().copied().collect();
            decode_ids.sort_unstable();
            self.transition_matrix.observe(row, &decode_ids);
        }

        // ── Hits and misses, by this thread's own bookkeeping ──
        let mut hits = 0usize;
        let mut misses = 0usize;
        // Misses that took no ring slot and are not otherwise being promoted:
        // their workers computed them from scratch, and the next launch that
        // routes them claims one. Those a decode row routed are counted apart —
        // a decode row co-batched into a sweep defers its claims this way.
        let (mut unslotted, mut decode_unslotted) = (0usize, 0usize);
        // Whether prompt-only experts are promoted (`PROMPT_ROOM`).
        let prompt_room =
            self.inner.zone.limit() * PROMPT_ROOM.1 >= self.inner.total_experts() * PROMPT_ROOM.0;
        for &e in &expert_ids {
            match self.inner.key_to_slot.get(&(row, e)) {
                Some(&slot) if self.inner.slots[slot].is_some() => {
                    self.inner.promote(slot);
                    // A decode row's reuse of this expert is near-certain from
                    // one step to the next; a prefill row's is close to zero, so
                    // the two must not bid for residency on equal footing.
                    if decode.contains(&e) {
                        self.inner.record_hit(row, e);
                    } else {
                        self.inner.record_prefill_hit(row, e);
                    }
                    hits += 1;
                }
                _ => {
                    misses += 1;
                    // The GPU promotes it if bucketize had a ring slot for it.
                    // A decode miss holds its slot to the next step, which
                    // routes to it again; a prefill-only miss earns no benefit
                    // of the doubt, so its score biases it toward the next
                    // eviction.
                    let routed_by_decode = decode.contains(&e);
                    if routed_by_decode {
                        self.inner.record_decode_miss(row, e);
                    } else {
                        self.inner.record_prefill_elevate(row, e);
                    }
                    if !self.promoting.contains(&(row, e)) {
                        unslotted += 1;
                        decode_unslotted += usize::from(routed_by_decode);
                    }
                }
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.routed_messages += 1;
            s.pipeline_lag += lag;
            s.ring_unslotted += unslotted;
            s.decode_unslotted += decode_unslotted;
            s.expert_hits += hits;
            s.expert_misses += misses;
            s.worker_pinned += pinned;
            s.worker_cold += cold;
        }
        // ── Whether reading ahead pays (`read_ahead_gate`), judged on decode
        // invocations — the width the look-ahead runs at. The misses bucketize
        // found cold wait on a pack read; the rest, and the read-ahead claims,
        // are pinned copies over the link ──
        if !msg.prefill_width {
            let copies = misses.saturating_sub(cold) + std::mem::take(&mut self.ahead_uncounted);
            let cold_secs = self
                .pack_read_secs
                .unwrap_or(self.image_bytes as f64 / self.link_rate.max(1.0));
            let us = link_us(copies, cold, self.image_bytes, self.link_rate, cold_secs);
            let was = self.link_time.on();
            let on = self.link_time.observe(us);
            self.read_ahead.set(on);
            if on != was {
                tracing::info!(
                    link_us_per_invocation = self.link_time.us(),
                    "expert cache: read-ahead {}",
                    if on { "on" } else { "off" }
                );
            }
            if let Ok(mut s) = self.stats.lock() {
                s.ahead_link_us = self.link_time.us();
            }
        }
        self.profile.record("pipe_classify", t);

        if !self.all_resident {
            // Stock every empty slot, and past them what the next
            // `RING_HORIZON` rows will miss — their non-resident experts, at the
            // share of experts this row routed that take ring slots — since this
            // thread's refill trails the GPU's bucketize; never below the floor.
            // Without room, prompt traffic gets only stock past decode's
            // reserve, and a launch claiming more than the predicted demand is a
            // sweep (`ring_stock`). Stock the demand no longer wants is taken
            // back (`trim_ring`).
            if let Some(ring) = &self.promo_ring {
                let empty_slots = self.inner.free_len() + self.empty_offers(ring);
                let width = self.inner.experts_per_layer.max(1);
                let absent: usize = (row + 1..(row + 1 + RING_HORIZON).min(self.num_moe_layers))
                    .map(|r| {
                        (0..width)
                            .filter(|&e| !self.inner.key_to_slot.contains_key(&(r, e)))
                            .count()
                    })
                    .sum();
                let ahead = |promoted: usize| (absent * promoted).div_ceil(width);
                let stock = ring_stock(
                    ahead(decode.len()),
                    ahead(expert_ids.len()),
                    prompt_room,
                    empty_slots,
                    self.window as usize,
                    ring.cap(),
                );
                self.ring_target = stock.target;
                ring.set_reserve(stock.reserve);
                ring.set_sweep(stock.sweep);
            }

            // ── Stock the ring for the layers to come ──
            let t = profile_now();
            self.trim_ring()?;
            self.refill_ring(row)?;
            self.profile.record("pipe_ring", t);

            // ── Predictions for read-ahead, only where the link has room ──
            // A prefill-width launch's misses are pulled by enough workers to
            // saturate the link, so reading ahead for it takes bandwidth from
            // them and saves nothing — its launches' own misses use up their
            // window anyway (`read_ahead`). Measured with copy-engine prefetch
            // on Qwen3.8-Flash-Next ×16 prefill (RTX 3090, a working set 3× the
            // zone): 40,823 promotions, 52.6 GiB, nearly every one late —
            // prefill 845 t/s with them, 1,075 without. A decode-width launch
            // leaves the link mostly idle, and there reading ahead pays: ×16
            // decode 237 t/s with it, 220 without.
            //
            // A narrower launch that routes prompt-only experts — any expert
            // no decode-scored row routed — predicts only without room
            // (`PROMPT_ROOM`); one whose experts are all decode's predicts as a
            // decode launch does. With room the ring already promotes the
            // prompt's misses and reading ahead only competes with them:
            // Qwen3.6-35B-A3B's one-sequence prompts ran 4–9% slower with it.
            // Without room, it is the one way a prompt's experts reach VRAM
            // ahead of the workers: Qwen3.8-Flash-Next's warm one-sequence
            // prompt ran 4% faster with it.
            let t = profile_now();
            if !msg.prefill_width && (decode.len() == expert_ids.len() || !prompt_room) {
                let wide = expert_ids.len() * 2 >= self.inner.experts_per_layer.max(1);
                // Invocations the device has begun past this one, as of now.
                let begun_past = self.clock.latest_started().saturating_sub(msg.ticket);
                self.predict_rows(row, &expert_ids, votes.as_ref(), wide, begun_past)?;
            }
            self.profile.record("pipe_prefetch", t);
        }
        // Scores age once per pass that routed a decode-scored row, at its last
        // MoE layer. That is nearly every pass — a prompt's last row is scored
        // as decode (`residency_rows`) — so the recency term counts passes; a
        // launch with no decode-scored row at all leaves the scores as they are.
        if row + 1 == self.num_moe_layers && !decode.is_empty() {
            self.inner.decay_scores(DECODE_RECENCY_DECAY);
        }
        self.publish_gauges();
        Ok(())
    }

    /// Collect what the GPU has taken from the promotion ring, in log order.
    /// A skipped victim's offer comes back with its expert still resident. A
    /// claimed victim was evicted by the device — its entries retargeted before
    /// its slot was written — so its bookkeeping follows at once, ahead of any
    /// later claim that could promote the same expert again. Each claimed slot
    /// becomes a device promotion, whole once its invocation is done; a
    /// read-ahead claim's expert is already published by the device, and its
    /// row's routing will judge the prediction. `near` is the ticket being
    /// served — the logged words are resolved against it.
    fn collect_device_promotions(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let head = ring.head();
        let (mut taken, mut claimed, mut skipped) = (0usize, 0usize, 0usize);
        let (mut ahead, mut ahead_bytes, mut ahead_pad) = (0usize, 0usize, 0usize);
        while self.ring_taken != head {
            let i = self.ring_taken as usize % ring.cap();
            let logged = ring.log(self.ring_taken);
            let offer = self.ring_slots[i]
                .take()
                .expect("a taken ring index holds the offer made there");
            self.ring_taken = self.ring_taken.wrapping_add(1);
            self.offered.remove(&offer.slot);
            if logged.skipped() {
                self.release_offer(&offer)?;
                skipped += 1;
                continue;
            }
            if let Some(((vr, ve), _)) = offer.victim {
                let evicted = self.inner.evict(offer.slot);
                if evicted != Some((vr, ve)) {
                    candle::bail!(
                        "expert pipeline: ring slot {} was offered holding ({vr}, {ve}) but held \
                         {evicted:?} when the device claimed it",
                        offer.slot
                    );
                }
                // The entry now names the fallback the device wrote (the
                // stager may have booked a cold one first); only then may the
                // stager evict a pad copy it names.
                self.residency()?.device_evicted(vr, ve);
                self.release_offer(&offer)?;
                let tag = self.victim_tag(vr, ve);
                self.regret.evicted(
                    vr,
                    ve,
                    logged.ahead,
                    ticket_from_word(logged.word, near),
                    tag,
                );
                claimed += 1;
            }
            taken += 1;
            if logged.ahead {
                ahead += 1;
                ahead_bytes += slot_offsets(&self.layer_geometries[logged.row]).3;
                // Listed from the pad: its pin holds until this lands.
                ahead_pad += usize::from(self.ahead_pins.lists(logged.row, logged.expert));
                self.speculative_loads.insert((logged.row, logged.expert));
            }
            self.promoting.insert((logged.row, logged.expert));
            self.regret.promoted(
                logged.row,
                logged.expert,
                logged.ahead,
                ticket_from_word(logged.word, near),
            );
            self.device_promotions.push(DevicePromotion {
                row: logged.row,
                expert: logged.expert,
                slot: offer.slot,
                ticket: ticket_from_word(logged.word, near),
                ahead: logged.ahead,
            });
        }
        self.ahead_uncounted += ahead;
        if let Ok(mut s) = self.stats.lock() {
            s.ring_taken += taken;
            s.victims_claimed += claimed;
            s.victims_skipped += skipped;
            s.evictions += claimed;
            s.ahead_claims += ahead;
            s.ahead_bytes += ahead_bytes;
            s.ahead_pad_claims += ahead_pad;
        }
        Ok(())
    }

    /// Empty offers standing in the ring, untaken.
    fn empty_offers(&self, ring: &PromotionRing) -> usize {
        let (head, tail) = (ring.head(), ring.tail());
        let mut i = head;
        let mut n = 0usize;
        while i != tail {
            if self.ring_slots[i as usize % ring.cap()].is_some_and(|o| o.victim.is_none()) {
                n += 1;
            }
            i = i.wrapping_add(1);
        }
        n
    }

    /// Up to `n` lazy victims, ranked against `row`: resident experts whose
    /// slot a miss may claim, each with the entries the device retargets it
    /// to (`Residency::displaced_entries`), never a slot already offered or
    /// one being promoted into. The fallback is held here for the life of the
    /// offer: a pad slot pinned, while fewer than a `PAD_PIN_SHARE`th of the
    /// pad is pinned so (past that, a pad-backed expert is passed over); a cold
    /// one marked, so the stager reads a cold summary of it as the device's
    /// eviction. Rows with an invocation enqueued are offered last: the wave is
    /// about to read their experts.
    ///
    /// The ranking runs without the residency lock — it is a scan of the whole
    /// zone, and the stager publishes pad copies under that lock while cold
    /// workers wait on them — so it asks for twice what it needs, and the lock
    /// is held only to confirm and hold the ranked slots' fallbacks, in rank
    /// order, keeping the first `n`.
    fn lazy_victims(&mut self, row: usize, n: usize) -> Result<Vec<Chosen>> {
        let upcoming: Vec<bool> = (0..self.num_moe_layers)
            .map(|r| self.clock.upcoming(r))
            .collect();
        let keys = &self.inner.slot_to_key;
        let offered = &self.offered;
        let promoting = &self.promoting;
        let width = self.inner.experts_per_layer;
        let candidate = |slot: usize| {
            !offered.contains(&slot)
                && keys[slot].is_some_and(|(r, e)| !promoting.contains(&(r, e)))
        };
        let ask = 2 * n;
        let mut slots = self
            .inner
            .rank_victims(row, ask, |slot, layer| !upcoming[layer] && candidate(slot));
        if slots.len() < ask {
            let more = self
                .inner
                .rank_victims(row, ask - slots.len(), |slot, layer| {
                    upcoming[layer] && candidate(slot)
                });
            slots.extend(more);
        }
        let pad_cap = self.stats.lock().map_or(0, |s| s.pad_slots / PAD_PIN_SHARE);
        let mut pinned = self.pad_pinned_offers;
        let mut chosen = Vec::with_capacity(n);
        let mut places = self
            .residency
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))?;
        for slot in slots {
            if chosen.len() == n {
                break;
            }
            let Some((r, e)) = keys[slot] else { continue };
            let Some((retarget, fallback)) = places.displaced_entries(r, e) else {
                continue;
            };
            match fallback {
                Fallback::Pad if pinned >= pad_cap => continue,
                Fallback::Pad => {
                    places.pin_pad(r, e, 1);
                    pinned += 1;
                }
                Fallback::Cold => places.set_offered_cold(r, e, true),
                Fallback::Warm => {}
            }
            chosen.push(Chosen {
                slot,
                key: (r, e),
                victim: Victim {
                    index: (r * width + e) as u64,
                    retarget,
                },
                fallback,
            });
        }
        drop(places);
        self.pad_pinned_offers = pinned;
        Ok(chosen)
    }

    /// An offer is over — claimed, skipped or taken back: release the fallback
    /// it held.
    fn release_offer(&mut self, offer: &Offer) -> Result<()> {
        match offer.victim {
            Some(((r, e), Fallback::Pad)) => {
                self.residency()?.pin_pad(r, e, -1);
                self.pad_pinned_offers -= 1;
            }
            Some(((r, e), Fallback::Cold)) => self.residency()?.set_offered_cold(r, e, false),
            Some((_, Fallback::Warm)) | None => {}
        }
        Ok(())
    }

    /// Point every device-promoted expert whose invocation has completed — all
    /// of them when `idle` (the device has been synchronized) — at its slot.
    fn land_device_promotions(&mut self, idle: bool) -> Result<()> {
        let Device::Cuda(cd) = self.device.clone() else {
            return Ok(());
        };
        let mut i = 0;
        let mut landed = 0usize;
        while i < self.device_promotions.len() {
            let p = &self.device_promotions[i];
            let complete = idle || self.clock.reclaimable(p.ticket);
            if !lands(complete, p.ahead, || self.fault.is_set()) {
                i += 1;
                continue;
            }
            let p = self.device_promotions.swap_remove(i);
            self.promoting.remove(&(p.row, p.expert));
            if let Some(ring) = &self.promo_ring {
                ring.clear_mark(p.row, p.expert);
            }
            // Promoted twice — by the GPU on two invocations: nothing ever
            // named this slot, so it is free.
            if self
                .inner
                .key_to_slot
                .get(&(p.row, p.expert))
                .is_some_and(|&s| self.inner.slots[s].is_some())
            {
                self.inner.put_free(p.slot);
                continue;
            }
            let base = self.inner.slot_base(p.slot);
            // SAFETY: the invocation whose workers filled the slot has completed.
            let view = unsafe { build_slot_view(&self.layer_geometries[p.row], &cd, base)? };
            self.inner.install(p.slot, p.row, p.expert, view);
            self.residency()?
                .set_vram(p.row, p.expert, Some((p.slot, base)));
            landed += 1;
        }
        if landed > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.worker_promotions += landed;
            }
        }
        Ok(())
    }

    /// Keep `ring_target` offers standing in the promotion ring: every empty
    /// slot first, then lazy victims, which stay resident until a miss claims
    /// them.
    fn refill_ring(&mut self, row: usize) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let tail = ring.tail();
        let stocked = tail.wrapping_sub(ring.head()) as usize;
        let room = ring.cap() - tail.wrapping_sub(self.ring_taken) as usize;
        let want = self.ring_target.saturating_sub(stocked).min(room);
        if want == 0 {
            return Ok(());
        }
        let mut offers: Vec<(Offer, Option<Victim>)> = Vec::with_capacity(want);
        while offers.len() < want {
            match self.inner.take_free() {
                Some(slot) => offers.push((Offer { slot, victim: None }, None)),
                None => break,
            }
        }
        if offers.len() < want {
            for c in self.lazy_victims(row, want - offers.len())? {
                offers.push((
                    Offer {
                        slot: c.slot,
                        victim: Some((c.key, c.fallback)),
                    },
                    Some(c.victim),
                ));
            }
        }
        for (offer, victim) in offers {
            let at = ring.tail() as usize % ring.cap();
            self.ring_slots[at] = Some(offer);
            self.offered.insert(offer.slot);
            ring.offer(self.inner.slot_base(offer.slot), victim);
        }
        Ok(())
    }

    /// Return an offer the device never took: an empty slot to the zone's free
    /// list, a lazy victim to plain residency — its expert never left — with
    /// its fallback released.
    fn take_back(&mut self, offer: Offer) -> Result<()> {
        self.offered.remove(&offer.slot);
        self.release_offer(&offer)?;
        if offer.victim.is_none() {
            self.inner.put_free(offer.slot);
        }
        Ok(())
    }

    /// Take back the offers past `ring_target` — stock the demand no longer
    /// wants.
    ///
    /// Only when no bucketize can be reading the ring: under the pass lock (no
    /// invocation can begin) with every invocation begun observed (its
    /// bucketize has run). Between a decode step and the next that holds; deep
    /// in a run-ahead it does not, and the stock waits.
    fn trim_ring(&mut self) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let pass_state = self.pass_state.clone();
        let p = pass_state
            .lock()
            .map_err(|_| candle::Error::Msg("expert pipeline: pass state poisoned".into()))?;
        if p.reserved != self.clock.observed() {
            return Ok(());
        }
        let (head, tail) = (ring.head(), ring.tail());
        let stocked = tail.wrapping_sub(head) as usize;
        if stocked <= self.ring_target {
            return Ok(());
        }
        let keep = head.wrapping_add(self.ring_target as u32);
        ring.withdraw_to(keep);
        drop(p);
        let mut i = keep;
        while i != tail {
            if let Some(offer) = self.ring_slots[i as usize % ring.cap()].take() {
                self.take_back(offer)?;
            }
            i = i.wrapping_add(1);
        }
        Ok(())
    }

    /// A forward faulted (`fault`) and the device has been synchronized: its
    /// workers gave demand items up, and a given-up item's promotion slot was
    /// never written. Collect what the device took, and drop every demand
    /// promotion still to land — its slot back to the zone, its mark cleared —
    /// instead of installing bytes that were never copied. Every demand claim
    /// made since the fault is still pending (`land_device_promotions` holds
    /// them while the word is set). Read-ahead claims stay: they never wait on
    /// a cold expert, and the device has published them. `near` is the ticket
    /// last served.
    pub(crate) fn drop_demand_promotions(&mut self, near: u64) -> Result<()> {
        self.collect_device_promotions(near)?;
        self.drop_pending_demand();
        Ok(())
    }

    /// Drop every demand promotion still to land — its slot back to the zone,
    /// its mark cleared — keeping the read-ahead claims.
    fn drop_pending_demand(&mut self) {
        let (ahead, demand) = split_ahead(std::mem::take(&mut self.device_promotions));
        self.device_promotions = ahead;
        for p in demand {
            self.promoting.remove(&(p.row, p.expert));
            self.regret.withdraw_promotion(p.row, p.expert);
            if let Some(ring) = &self.promo_ring {
                ring.clear_mark(p.row, p.expert);
            }
            self.inner.put_free(p.slot);
        }
    }

    /// Take every offer out of the promotion ring — before a boundary move,
    /// with the device synchronized: land what the GPU filled, take back what
    /// it never took, and release the pad pins read-ahead's old listings still
    /// held, whose readers are all done. With a fault not yet taken, its demand
    /// promotions are dropped here rather than left pending: the move may
    /// retract the zone below their slots.
    pub(crate) fn drain_ring(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        self.collect_device_promotions(near)?;
        if self.fault.is_set() {
            self.drop_pending_demand();
        }
        self.land_device_promotions(true)?;
        ring.withdraw();
        let untaken: Vec<Offer> = self
            .ring_slots
            .iter_mut()
            .filter_map(Option::take)
            .collect();
        for offer in untaken {
            self.take_back(offer)?;
        }
        self.ring_taken = ring.head();
        self.release_ahead_pins(true)
    }

    /// `experts` predicted for row `row`, best first — the row's latest
    /// prediction, which its listing is vetted from now (`list_row`) and again
    /// as the stager lands what it stages ahead for it (`list_landed`).
    fn predict_ahead(&mut self, row: usize, experts: &[usize]) -> Result<()> {
        if row < self.inner.pinned_layers {
            return Ok(());
        }
        self.predicted[row] = experts.to_vec();
        self.list_row(row, true)?;
        Ok(())
    }

    /// Whether a launch that reads `row`'s listing has yet to begin
    /// (`read_ahead::listing_readable`): the served row is the last whose
    /// summary this thread read, and the served ticket is where the forward
    /// thread's ring hold stands — the previous pass's last between passes.
    fn listing_readable(&self, row: usize) -> bool {
        let served_row = self.last_summary.map(|(_, r)| r);
        let served_ticket = self.routed_served.load(Ordering::Acquire);
        listing_readable(row, served_row, served_ticket, |r| self.clock.begun(r))
    }

    /// Write `row`'s read-ahead listing from its latest prediction: the
    /// predicted experts not in VRAM and not being promoted that have a pinned
    /// copy, each with its slot image as the vetted source, replacing the last
    /// listing — only while a launch that reads it has yet to begin
    /// (`listing_readable`), since a launch at `r` reads the lists of `r + 2`
    /// on. A warm copy is listed as it is; a pad copy only with its slot
    /// pinned for the listing (`ahead_pins`), since the stager may otherwise
    /// reuse it under another row's launch. With `stage`, the cold ones (no
    /// device-readable copy) go to the stager to stage into the pad, whether
    /// or not the row is listed now — staged in time for its own workers at the
    /// least, and listed from the pad as each lands (`list_landed`). Returns
    /// the experts listed.
    fn list_row(&mut self, row: usize, stage: bool) -> Result<Vec<usize>> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(Vec::new());
        };
        let readable = self.listing_readable(row);
        if !readable && !stage {
            return Ok(Vec::new());
        }
        let cap = self
            .stats
            .lock()
            .map_or(0, |s| s.pad_slots / AHEAD_PIN_SHARE);
        let mut to_stage = Vec::new();
        let mut planned = None;
        {
            let mut places = self
                .residency
                .lock()
                .map_err(|_| candle::Error::Msg("expert pipeline: residency poisoned".into()))?;
            let predicted = &self.predicted[row];
            let mut candidates = Vec::with_capacity(predicted.len().min(AHEAD_CAP));
            for &e in predicted {
                if self.is_held(row, e) {
                    continue;
                }
                let p = places.place(row, e);
                if p.entry() == 0 {
                    if stage {
                        to_stage.push(e);
                    }
                } else if p.vram.is_none() && candidates.len() < AHEAD_CAP {
                    candidates.push(Candidate {
                        expert: e,
                        image: p.entry(),
                        pad: p.pad.is_some(),
                    });
                }
            }
            if readable {
                // Pinned under the residency lock the candidates were read
                // under, so each pad slot listed is the one its entry names.
                let plan = self.ahead_pins.plan(row, &candidates, cap);
                for &e in &plan.pin {
                    places.pin_pad(row, e, 1);
                }
                planned = Some(plan.listing);
            }
        }
        let mut listed = Vec::new();
        if let Some(listing) = planned {
            ring.predict(row, &listing);
            self.ahead_pins.commit(self.clock.readers_key());
            if let Ok(mut s) = self.stats.lock() {
                s.ahead_pinned = self.ahead_pins.held();
            }
            listed = listing.into_iter().map(|(e, _)| e).collect();
        }
        if !to_stage.is_empty() {
            if let Ok(mut s) = self.stats.lock() {
                s.stage_requests += to_stage.len();
            }
            // A closed channel means the stager is gone; its guard has raised
            // the abort word, which the next routed layer reports.
            let _ = self.stager.send(StagerMsg::Stage {
                row,
                experts: to_stage,
            });
        }
        Ok(listed)
    }

    /// The experts the stager has staged ahead since this was last called,
    /// each now in the pad: every row among them whose latest prediction still
    /// names the expert, and that a launch has yet to read, is listed again —
    /// from that prediction, so now with the staged expert from the pad.
    fn list_landed(&mut self) -> Result<()> {
        let mut landed: Vec<(usize, usize)> = Vec::new();
        while let Ok(l) = self.landed_ahead.try_recv() {
            landed.push(l);
        }
        if landed.is_empty() {
            return Ok(());
        }
        let mut rows: Vec<usize> = landed.iter().map(|&(r, _)| r).collect();
        rows.sort_unstable();
        rows.dedup();
        let mut listed = 0usize;
        for row in rows {
            let staged: Vec<usize> = landed
                .iter()
                .filter(|&&(r, e)| r == row && self.predicted[row].contains(&e))
                .map(|&(_, e)| e)
                .collect();
            if staged.is_empty() {
                continue;
            }
            let now = self.list_row(row, false)?;
            listed += staged.iter().filter(|e| now.contains(e)).count();
        }
        if listed > 0 {
            if let Ok(mut s) = self.stats.lock() {
                s.staged_listed += listed;
            }
        }
        Ok(())
    }

    /// Release the pad pins read-ahead's listings dropped once every
    /// invocation that could have read them has finished — all of them when
    /// `idle` (the device has been synchronized).
    fn release_ahead_pins(&mut self, idle: bool) -> Result<()> {
        let clock = &self.clock;
        let due = self.ahead_pins.due(|key| idle || clock.reclaimable(key));
        if !due.is_empty() {
            let mut places = self.residency()?;
            for (row, e) in due {
                places.pin_pad(row, e, -1);
            }
            drop(places);
            if let Ok(mut s) = self.stats.lock() {
                s.ahead_pinned = self.ahead_pins.held();
            }
        }
        Ok(())
    }

    /// What was known of `row`'s expert `e` as a claim evicts it: the best
    /// measured probability a live prediction for the row gave it (bands below
    /// and from one half), whether the row routed it on its last visit,
    /// whether the row lies just ahead of the one last served, and whether a
    /// warm copy backs it. "Last visit" and "last served" are as this thread
    /// has processed them: while it trails the GPU, a claim from a later
    /// invocation is tagged against the visit before.
    fn victim_tag(&self, row: usize, e: usize) -> VictimTag {
        let best = self.pending[row]
            .iter()
            .enumerate()
            .filter_map(|(h, p)| Some((h + 1, p.as_ref()?)))
            .flat_map(|(hop, p)| {
                p.held_named
                    .iter()
                    .filter(move |&&(x, _)| x == e)
                    .map(move |&(_, cell)| self.cells.probability(hop, cell))
            })
            .fold(None, |m: Option<f64>, p| Some(m.map_or(p, |m| m.max(p))));
        let n = self.num_moe_layers;
        let served = self.last_summary.map_or(row, |(_, r)| r);
        VictimTag {
            predicted: prediction_band(best),
            hit_last: self.inner.hit_last_visit(row, e),
            ahead: just_ahead(row, served, n),
            warm: self.inner.is_warm_backed(row, e),
        }
    }

    /// Whether `row`'s expert `e` is resident or being promoted: listing it
    /// would buy no copy.
    fn is_held(&self, row: usize, e: usize) -> bool {
        self.inner
            .key_to_slot
            .get(&(row, e))
            .is_some_and(|&s| self.inner.slots[s].is_some())
            || self.promoting.contains(&(row, e))
    }

    /// `row`'s experts not in VRAM that decode routed recently — a score of at
    /// least `min` (`RECENT_MIN_SCORE`: within the last few steps) — highest
    /// first, at most `cap`: what the zone let go of and the next step is
    /// likely to want back.
    fn scored_absent(&self, row: usize, min: f32, cap: usize) -> Vec<usize> {
        let mut cands: Vec<(usize, f32)> = (0..self.inner.experts_per_layer)
            .filter(|&e| !self.inner.key_to_slot.contains_key(&(row, e)))
            .filter_map(|e| {
                let s = self.inner.score(row, e);
                (s >= min).then_some((e, s))
            })
            .collect();
        cands.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(CmpOrdering::Equal)
                .then(a.0.cmp(&b.0))
        });
        cands.truncate(cap);
        cands.into_iter().map(|(e, _)| e).collect()
    }

    /// Predict the experts the next rows will need, out to the last row a
    /// launch reads ahead for. Each hop's candidates come from three
    /// predictors:
    ///
    /// 1. the router look-ahead's votes (`votes`) — that row's own router
    ///    applied to this layer's input, its margin picks included;
    /// 2. for a narrow launch, the Markov table from this row's routing to the
    ///    hop's arrivals (`transition`), with its confidence — a launch routing
    ///    most of the row (`wide`) leaves it nothing specific to imply;
    /// 3. the target row's experts the zone let go of that decode routed
    ///    recently (`scored_absent`) — any scored one for a wide launch, which
    ///    will route most of that row too.
    ///
    /// The list is their union ranked by what each combination of evidence has
    /// been worth (`blend`), cut at the hop's cap — not the predictors in turn.
    /// In turn, the votes filled every cap and the Markov table never reached a
    /// list, though a strong Markov prediction alone is routed more often than a
    /// one-token vote from hop 3 on, and the two agreeing more often than
    /// either: on Qwen3.8-Flash-Next 92% at hop 1, 86% at hop 5. Ranked by the
    /// cells, the same candidates gave hops 3–5 recall 26/24/22 → 35/34/34% and
    /// precision 55–63 → 84–85% at ×8, and 38/35/31 → 43/41/38% recall at ×1;
    /// ×1 decode rose 28.5–31.6 → 35.3 t/s (warm), ×8 to 96.0 and 106.1.
    ///
    /// A wrong prediction costs a pad slot or a ring offer and the link time
    /// its window allowed, never correctness. Every candidate is kept to be
    /// judged by the row's routing (`Prediction`), so the cells learn from the
    /// whole set, not just the list.
    ///
    /// **Only rows the device has not begun.** The device has begun
    /// `begun_past` invocations past this one, so the rows that many hops on
    /// are already running: a read-ahead or a staging for them cannot land in
    /// time, and predicting them only deepens this thread's lag. Those hops are
    /// skipped. On a card whose GPU outruns this thread the skip is what keeps
    /// it from holding the forward at the summary ring — Qwen3.8-Flash-Next's
    /// gate (RTX PRO 5000) at one to four sequences ran the ring's full 64
    /// invocations behind, and with the skip at most 36: BF16 ×4 decode 677 →
    /// 797 t/s, ×1 warm 200 → 283, the ×2 rows 337–368 → 423–446. On a card
    /// that streams its experts the lag stays short and every hop runs.
    fn predict_rows(
        &mut self,
        row: usize,
        current: &[usize],
        votes: Option<&Votes>,
        wide: bool,
        begun_past: u64,
    ) -> Result<()> {
        let width = self.inner.experts_per_layer;
        let voted_hops = votes.map_or(0, |v| v.hops(width));
        for hop in 1..=HOPS {
            let target = row + hop;
            if target >= self.num_moe_layers {
                break;
            }
            if hop as u64 <= begun_past {
                continue;
            }
            let cap = prediction_cap(hop);
            let hop_votes = votes
                .filter(|_| hop <= voted_hops)
                .map(|v| v.hop(hop, width));
            let by_votes = votes
                .filter(|_| hop <= voted_hops)
                .map(|v| ranked(&v.words, width, hop, cap));
            let markov = if wide {
                Vec::new()
            } else {
                self.transition_matrix.candidates(
                    row,
                    hop,
                    current,
                    MARKOV_MIN_CONF,
                    MARKOV_CANDIDATES,
                )
            };
            let min = if wide {
                f32::MIN_POSITIVE
            } else {
                RECENT_MIN_SCORE
            };
            let recent = self.scored_absent(target, min, RECENT_CANDIDATES);
            let held: Vec<bool> = (0..width).map(|e| self.is_held(target, e)).collect();
            // Only an expert a copy could serve is listed: a held one would
            // spend a slot of the cap and buy nothing.
            let (named_held, candidates): (Vec<_>, Vec<_>) = gather(hop_votes, &markov, &recent)
                .into_iter()
                .partition(|c| held[c.expert]);
            let selected = self.cells.select(hop, &candidates, cap);
            self.pending[target][hop - 1] = Some(Prediction {
                votes: by_votes,
                candidates: candidates.iter().map(|c| (c.expert, c.cell)).collect(),
                selected: selected.clone(),
                held,
                held_named: named_held.iter().map(|c| (c.expert, c.cell)).collect(),
            });
            if selected.is_empty() {
                continue;
            }
            // A pinned layer's experts are all resident; `predict_ahead`
            // passes it over.
            self.predict_ahead(target, &selected)?;
        }
        Ok(())
    }

    /// Size read-ahead's window from the pass just finished: the median
    /// interval between its consecutive rows' summaries, at the link's measured
    /// rate (`read_ahead`). A pass with no such interval leaves the window as
    /// it was.
    fn set_read_ahead_window(&mut self) {
        let Some(layer_secs) = median(&mut self.layer_intervals) else {
            return;
        };
        self.layer_intervals.clear();
        self.window = read_ahead_window(self.link_rate, layer_secs, self.image_bytes, AHEAD_MAX);
        if let Some(ring) = &self.promo_ring {
            ring.set_window(self.window);
        }
        if let Ok(mut s) = self.stats.lock() {
            s.ahead_window = self.window as usize;
        }
    }

    /// Order the eviction passes by each class's expected miss cost — its
    /// running regret rate (`regret`) times what restoring one costs: the copy
    /// over the link for a warm-backed expert (one image at the measured link
    /// rate), and the pack read the layer then waits on (the stager's mean,
    /// latest measured) plus that copy for a pack-only one. Before the first
    /// pack read is measured the default order stands.
    fn set_eviction_order(&mut self) {
        if let Ok(s) = self.stats.lock() {
            if let Some(read) = s.pack_reads.mean_secs() {
                self.pack_read_secs = Some(read);
            }
        }
        let Some(read) = self.pack_read_secs else {
            return;
        };
        if self.link_rate <= 0.0 {
            return;
        }
        let copy = self.image_bytes as f64 / self.link_rate;
        let cost = eviction_costs(self.regret.class_rates(), copy, read);
        self.inner.set_eviction_costs(cost);
        if let Ok(mut s) = self.stats.lock() {
            s.eviction_cost_us = cost.map(|c| c * 1e6);
        }
    }

    /// The zone's gauges, refreshed each routed layer and after every boundary
    /// move.
    pub(super) fn publish_gauges(&self) {
        let slot_bytes = self.inner.zone.slot_bytes();
        let occupied = self.inner.num_slots() - self.inner.free_len();
        if let Ok(mut s) = self.stats.lock() {
            s.resident_vram_bytes = occupied * slot_bytes;
            s.zone_cedeable_bytes = self
                .inner
                .zone
                .capacity()
                .saturating_sub(self.inner.zone.min_capacity())
                * slot_bytes;
            s.zone_bytes = self.inner.zone.capacity() * slot_bytes;
            s.zone_min_bytes = self.inner.zone.min_capacity() * slot_bytes;
            s.zone_max_bytes = self.inner.zone.limit() * slot_bytes;
            s.expert_slot_bytes = slot_bytes;
        }
    }

    /// Make the device's CUDA context current on the calling thread — the
    /// pipeline thread, at start.
    fn bind_device_to_thread(&self) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(());
        };
        cd.cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(candle::Error::wrap)
    }
}

// ============================================================================
// Thread spawn
// ============================================================================

/// Marks the thread dead and raises the abort word when dropped — including on
/// unwind. Nothing the GPU waits on comes from this thread, but a dead pipeline
/// means the cache's state can no longer be trusted, so every waiting worker
/// gives its expert up and the forward fails (`fault`).
struct DeadFlagGuard {
    dead: Arc<AtomicBool>,
    abort: Arc<AbortWord>,
}

impl Drop for DeadFlagGuard {
    fn drop(&mut self) {
        self.abort.raise();
        self.dead.store(true, Ordering::Release);
    }
}

/// Spawn the pipeline thread. Returns the sender for routed layers and the
/// rest of the message set.
pub(crate) fn spawn_pipeline_thread(
    mut state: PipelineState,
    dead_flag: Arc<AtomicBool>,
) -> mpsc::SyncSender<PipelineMessage> {
    let (tx, rx) = mpsc::sync_channel::<PipelineMessage>(super::dispatch::PIPELINE_CHANNEL_BOUND);

    std::thread::Builder::new()
        .name("expert-pipeline".into())
        .spawn(move || {
            let _dead_on_exit = DeadFlagGuard {
                dead: dead_flag,
                abort: state.abort.clone(),
            };
            // This thread builds the views of landed slots and issues the
            // boundary moves' relocations, which need the device's context
            // current on it.
            if let Err(e) = state.bind_device_to_thread() {
                tracing::error!(
                    target: "candle_transformers::expert_lre",
                    "expert pipeline: could not bind the CUDA context to its thread: {e}"
                );
                return;
            }
            while let Ok(msg) = rx.recv() {
                match msg {
                    PipelineMessage::Routed(routed) => {
                        // Hand-off latency: forward-thread `send` → pickup here.
                        state.profile.record("pipe_inbound", routed.submitted_at);
                        let wt = profile_now();
                        let row = routed.row;
                        if let Err(e) = state.process_routed(routed) {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                row,
                                "expert pipeline: routed layer could not be served, aborting: {e}"
                            );
                            return;
                        }
                        state.profile.record("pipe_worker_total", wt);
                    }
                    PipelineMessage::Settle { response_tx } => {
                        // Land whatever has completed, so a reader of the
                        // counters sees it.
                        let near = state.routed_served.load(Ordering::Acquire);
                        if let Err(e) = state
                            .collect_device_promotions(near)
                            .and_then(|()| state.land_device_promotions(false))
                        {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                "expert pipeline: a promotion could not land, aborting: {e}"
                            );
                            return;
                        }
                        let _ = response_tx.send(());
                    }
                    PipelineMessage::DropDemandPromotions { response_tx } => {
                        let near = state.routed_served.load(Ordering::Acquire);
                        if let Err(e) = state.drop_demand_promotions(near) {
                            tracing::error!(
                                target: "candle_transformers::expert_lre",
                                "expert pipeline: a faulted forward's promotions could not be \
                                 dropped, aborting: {e}"
                            );
                            return;
                        }
                        let _ = response_tx.send(());
                    }
                    PipelineMessage::SnapshotProfile { response_tx } => {
                        let snap = state.profile.snapshot();
                        state.profile.reset();
                        let _ = response_tx.send(snap);
                    }
                    // The give-back, reachable without a completed forward.
                    // Answering zero is a legitimate outcome (nothing spare, a
                    // wave still live, an invocation still unserved).
                    PipelineMessage::RenegotiateBoundary { ask, response_tx } => {
                        let conceded = match state.renegotiate_if_quiet(ask) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                tracing::warn!("requested boundary move failed: {e}");
                                0
                            }
                        };
                        let _ = response_tx.send(conceded);
                    }
                }
            }
        })
        .expect("failed to spawn expert-pipeline thread");

    tx
}

#[cfg(test)]
mod tests {
    use super::{lands, ring_stock, split_ahead, DevicePromotion, RingStock};
    use std::cell::Cell;

    fn promo(row: usize, expert: usize, slot: usize, ahead: bool) -> DevicePromotion {
        DevicePromotion {
            row,
            expert,
            slot,
            ticket: 40 + row as u64,
            ahead,
        }
    }

    /// A fault drops the demand claims and keeps the read-ahead ones, each side
    /// in pending order.
    #[test]
    fn a_fault_keeps_read_ahead_claims_and_drops_demand_ones() {
        let (ahead, demand) = split_ahead(vec![
            promo(3, 7, 100, false),
            promo(5, 2, 101, true),
            promo(4, 9, 102, false),
            promo(6, 1, 103, true),
        ]);
        let key = |v: &[DevicePromotion]| {
            v.iter()
                .map(|p| (p.row, p.expert, p.slot))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&ahead), vec![(5, 2, 101), (6, 1, 103)]);
        assert_eq!(key(&demand), vec![(3, 7, 100), (4, 9, 102)]);
    }

    /// A complete claim lands unless it is a demand claim under a fault; an
    /// incomplete one never lands, and the fault word is read only for a
    /// complete demand claim — after the completion it is ordered behind.
    #[test]
    fn a_demand_claim_waits_out_a_fault_and_a_read_ahead_claim_does_not() {
        let reads = Cell::new(0);
        let word = |set: bool| {
            let reads = &reads;
            move || {
                reads.set(reads.get() + 1);
                set
            }
        };
        assert!(lands(true, false, word(false)));
        assert!(!lands(true, false, word(true)));
        assert_eq!(reads.get(), 2);
        assert!(lands(true, true, word(true)));
        assert!(!lands(false, false, word(false)));
        assert!(!lands(false, true, word(false)));
        assert_eq!(reads.get(), 2, "read only for a complete demand claim");
    }

    fn stock(target: usize, reserve: u32, sweep: u32) -> RingStock {
        RingStock {
            target,
            reserve,
            sweep,
        }
    }

    /// With room the ring stocks for every miss, ×1.25, reserves nothing, and
    /// lets a launch claim up to that prediction.
    #[test]
    fn with_room_the_ring_stocks_for_all_misses_and_reserves_nothing() {
        assert_eq!(ring_stock(8, 100, true, 0, 0, 1024), stock(125, 0, 125));
        // The floor and the half-capacity ceiling.
        assert_eq!(ring_stock(0, 4, true, 0, 0, 1024), stock(32, 0, 32));
        assert_eq!(ring_stock(0, 1_000, true, 0, 0, 1024), stock(512, 0, 512));
        // More empty slots than the prediction: every one is stocked, and the
        // sweep word stays the prediction.
        assert_eq!(ring_stock(8, 100, true, 300, 0, 1024), stock(300, 0, 125));
    }

    /// Without room decode's target is the reserve and the sweep word, and the
    /// stock past it is empty slots only — a full zone stocks exactly decode's
    /// target.
    #[test]
    fn without_room_the_stock_past_decodes_reserve_is_empty_slots() {
        // Full zone: stock = reserve = sweep = decode's target.
        assert_eq!(ring_stock(40, 100, false, 0, 0, 1024), stock(50, 50, 50));
        // Fewer empties than decode's target: the target, made of lazy victims.
        assert_eq!(ring_stock(40, 100, false, 25, 0, 1024), stock(50, 50, 50));
        // Plenty empty: every one stocked, decode's target still reserved, and
        // a prompt claiming past it still a sweep.
        assert_eq!(ring_stock(40, 100, false, 300, 0, 1024), stock(300, 50, 50));
        // Never past half the ring.
        assert_eq!(
            ring_stock(40, 1_000, false, 1_000, 0, 256),
            stock(128, 50, 50)
        );
        // A prompt-only launch: decode's floor is still reserved.
        assert_eq!(ring_stock(0, 100, false, 10, 0, 1024), stock(32, 32, 32));
    }

    /// Read-ahead's window is stocked past the misses' target, never past half
    /// the ring, and moves neither the reserve nor the sweep word.
    #[test]
    fn the_read_ahead_window_is_stocked_on_top() {
        assert_eq!(ring_stock(40, 100, false, 0, 14, 1024), stock(64, 50, 50));
        assert_eq!(ring_stock(8, 100, true, 0, 14, 1024), stock(139, 0, 125));
        assert_eq!(ring_stock(40, 100, false, 0, 100, 256), stock(128, 50, 50));
        // Empty slots past target + window are still all stocked.
        assert_eq!(
            ring_stock(40, 100, false, 300, 14, 1024),
            stock(300, 50, 50)
        );
    }
}

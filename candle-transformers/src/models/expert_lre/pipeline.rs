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
//! - **Prefetch, by the copy engine.** The Markov transition matrix predicts the
//!   next layers' experts; a predicted expert with a pinned copy is copied into
//!   a slot on the copy stream and its entry pointed there once the copy is
//!   **reported** complete. The copies are issued by the copier thread
//!   (`copier`), never by this one: issuing a copy can stall in the driver
//!   for as long as the forward the GPU is running, and this thread's ring
//!   must not stall with it. A predicted cold one is handed to the stager to
//!   stage into the pad. A prefill-width row promotes the next row's scored
//!   experts the same way.
//! - **Eviction** points an entry back at the expert's pinned copy (or 0) and
//!   frees the slot under the reclaim rule (`reclaim`): an entry may go to 0
//!   only for a quiet row, and a slot is reused only once every invocation that
//!   could have snapshotted its address is done.
//!
//! It exclusively owns the [`ExpertCacheInner`], the copy stream and the
//! [`TransitionMatrix`]. The per-expert places it shares with the stager live in
//! the [`Residency`] lock, which also writes the live table.

use super::cache::{ExpertCacheInner, DECODE_RECENCY_DECAY};
use super::copier::{Copier, CopyJob};
use super::dispatch::{AbortWord, PassState, SummaryRing};
use super::pinned::LayerGeometry;
use super::promo::{ticket_from_word, PromotionRing, Victim};
use super::reclaim::{ReclaimClock, RetireList};
use super::residency::{Fallback, Residency};
use super::slot_image::{build_slot_view, slot_offsets};
use super::stager::StagerMsg;
use super::transition::TransitionMatrix;
use super::types::{PipelineMessage, PipelineStats, RoutedLayer};
use crate::models::profile::{profile_now, ProfileAccumulator};
use candle::{Device, Result};
use cudarc::driver::CudaStream;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};

// ============================================================================
// Dynamic load-ahead (prefetch depth N) controller constants
// ============================================================================

/// Upper clamp on the dynamic prefetch depth `N`. Adjacent-chained prediction
/// compounds precision per hop (~0.95×/hop from the 93–97% steady state), so
/// beyond a few hops the extra DMA is mostly waste even when latency-bound.
const PREFETCH_DEPTH_MAX: usize = 4;

/// Deepen only when a pass accumulated at least this many late promotions.
/// A single straggler in one pass is noise.
const PREFETCH_LATE_FLOOR: usize = 2;

/// Chained hops beyond `L+1` are issued only for decode-narrow source sets
/// (at most this many routed experts). A prefill-width row's compute window
/// hides single-hop latency, and multi-hop batches for it would only queue
/// bytes ahead of the next row's.
const PREFETCH_MULTI_HOP_MAX_SOURCES: usize = 32;

/// Upper bound on one lookahead: the next row's scored experts, highest first.
/// Bounds its bus time to roughly one layer's compute window; a bigger batch
/// cannot land before its row and only queues bytes.
const LOOKAHEAD_MAX: usize = 64;

/// Deepen only when the pass's achieved copy bandwidth sits below this
/// fraction of the empirical ceiling. Late promotions WITHOUT slack mean the
/// link is saturated, and issuing earlier would only queue more bytes.
const PREFETCH_BW_SLACK: f32 = 0.8;

/// Shallow back when a pass finishes with zero late promotions and its
/// prediction precision fell below this floor: the extra hops are landing in
/// time but bringing in the wrong experts.
const PREFETCH_PRECISION_FLOOR: f32 = 0.8;

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
/// refill per routed layer, made with no driver call (`copier`), so it trails
/// the GPU by about a layer.
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
fn ring_stock(
    decode_ahead: usize,
    all_ahead: usize,
    prompt_room: bool,
    empty_slots: usize,
    cap: usize,
) -> RingStock {
    let target = |ahead: usize| (ahead + ahead / 4).clamp(RING_TARGET, cap / 2);
    let empties = empty_slots.min(cap / 2);
    if prompt_room {
        let all = target(all_ahead);
        return RingStock {
            target: all.max(empties),
            reserve: 0,
            sweep: all as u32,
        };
    }
    let decode = target(decode_ahead);
    RingStock {
        target: decode.max(empties),
        reserve: decode as u32,
        sweep: decode as u32,
    }
}

/// The summary bits the pipeline reads.
const SUMMARY_COUNT: u32 = 0x1fff_ffff;
const SUMMARY_PINNED: u32 = 1 << 29;
const SUMMARY_COLD: u32 = 1 << 30;
const SUMMARY_DECODE: u32 = 1 << 31;

/// A slot the GPU is filling with a missed expert: whole once invocation
/// `ticket` has completed.
struct DevicePromotion {
    row: usize,
    expert: usize,
    slot: usize,
    ticket: u64,
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

/// One speculative promotion copy in flight on the copier, named by its job id.
struct Promotion {
    id: u64,
    row: usize,
    expert: usize,
    slot: usize,
    bytes: usize,
    /// The copy reads the expert's pad slot, pinned against pad eviction until
    /// the copy is done.
    from_pad: bool,
}

/// The pipeline thread's private state. Never crosses a thread boundary after
/// construction — the thread owns it exclusively with `&mut self`.
pub(crate) struct PipelineState {
    /// VRAM slots, eviction scores, the zone and its free list.
    pub(crate) inner: ExpertCacheInner,
    pub(crate) device: Device,
    /// Every promotion and relocation copy, in order: promotions issued by the
    /// copier, relocations by this thread with the copier flushed.
    pub(crate) copy_stream: Arc<CudaStream>,
    copier: Copier,
    /// The next promotion's job id.
    next_copy: u64,
    pub(crate) residency: Arc<Mutex<Residency>>,
    pub(crate) clock: Arc<ReclaimClock>,
    pub(crate) ring: Arc<SummaryRing>,
    pub(crate) abort: Arc<AbortWord>,
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
    /// `(layer, expert)` pairs promoted speculatively and not yet judged: their
    /// layer's routing scores the prediction.
    pub(crate) speculative_loads: HashSet<(usize, usize)>,
    /// Promotion copies in flight, and the experts they — or the GPU — are
    /// promoting.
    promotions: Vec<Promotion>,
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
    /// Evicted slots waiting for their old readers (`reclaim`).
    retired: RetireList<usize>,
    /// Dynamic load-ahead depth `N`, adapted once per pass by
    /// [`PipelineState::adapt_prefetch_depth`].
    pub(crate) prefetch_depth: usize,
    /// Wall-clock start of the current pass — the denominator of the achieved
    /// promotion bandwidth.
    pub(crate) pass_started: Option<std::time::Instant>,
    /// Promotion bytes enqueued this pass.
    pub(crate) pass_dma_bytes: usize,
    /// Speculative promotions late this pass.
    pub(crate) pass_late: usize,
    /// Per-pass prediction precision counters `(hits, total)`.
    pub(crate) pass_pred: (usize, usize),
    /// Running max of per-pass achieved GB/s — the controller's ceiling.
    pub(crate) bw_ceiling_gbps: f32,
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
        pass_state: Arc<Mutex<PassState>>,
        routed_served: Arc<AtomicU64>,
        stager: mpsc::Sender<StagerMsg>,
        layer_geometries: Arc<Vec<LayerGeometry>>,
        all_resident: bool,
        promo_ring: Option<Arc<PromotionRing>>,
        stats: Arc<Mutex<PipelineStats>>,
    ) -> Result<Self> {
        let num_moe_layers = layer_geometries.len();
        let experts = inner.experts_per_layer;
        let ring_cap = promo_ring.as_ref().map_or(0, |r| r.cap());
        let copier = Copier::spawn(copy_stream.clone())?;
        Ok(Self {
            inner,
            device,
            copy_stream,
            copier,
            next_copy: 0,
            residency,
            clock,
            ring,
            abort,
            pass_state,
            pass: None,
            routed_served,
            stager,
            layer_geometries,
            num_moe_layers,
            all_resident,
            transition_matrix: TransitionMatrix::new(num_moe_layers, experts),
            speculative_loads: HashSet::new(),
            promotions: Vec::new(),
            promoting: HashSet::new(),
            promo_ring,
            ring_slots: vec![None; ring_cap],
            offered: HashSet::new(),
            pad_pinned_offers: 0,
            ring_taken: 0,
            device_promotions: Vec::new(),
            ring_target: RING_TARGET.min(ring_cap / 2),
            retired: RetireList::new(),
            prefetch_depth: 1,
            pass_started: None,
            pass_dma_bytes: 0,
            pass_late: 0,
            pass_pred: (0, 0),
            bw_ceiling_gbps: 0.0,
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
                self.adapt_prefetch_depth();
            }
            self.pass = Some(msg.pass);
        }
        if self.pass_started.is_none() {
            self.pass_started = Some(std::time::Instant::now());
        }

        // ── The routing summary ──
        let t = profile_now();
        self.ring.wait(msg.slot, msg.summary_word)?;
        self.clock.observe(msg.ticket);
        let lag = self.clock.latest_started().saturating_sub(msg.ticket);
        self.profile.record("pipe_routed_wait", t);
        // SAFETY: `wait` returned for this invocation, and the forward thread
        // does not rewrite the slot until `routed_served` passes this ticket.
        let summary: Vec<u32> = unsafe { self.ring.read(msg.slot) }.to_vec();
        self.routed_served.store(msg.ticket, Ordering::Release);

        let t = profile_now();
        self.collect_device_promotions(msg.ticket)?;
        for slot in self.retired.drain(&self.clock) {
            self.inner.put_free(slot);
        }
        self.poll_promotions()?;
        self.land_device_promotions(false)?;

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

        // ── Prediction precision, and late speculative promotions ──
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
            self.pass_pred.0 += hits;
            self.pass_pred.1 += spec_for_layer.len();
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
        let late = self.promotions.iter().filter(|p| p.row == row).count();
        if late > 0 {
            self.pass_late += late;
            if let Ok(mut s) = self.stats.lock() {
                s.late_loads += late;
            }
        }

        self.transition_matrix.observe(row, &expert_ids);

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

            // ── Speculation, only where the link has room for it ──
            // A prefill-width launch's misses are pulled by enough workers to
            // saturate the link, so a speculative copy for the next row takes
            // bandwidth from them and saves nothing: either it lands late and the
            // workers have pulled the same bytes, or it lands in time and they
            // would have pulled them at the same cost. Measured on
            // Qwen3.8-Flash-Next ×16 prefill (RTX 3090, a working set 3× the
            // zone): 40,823 copy-engine promotions, 52.6 GiB, nearly every one
            // late — prefill 845 t/s with them, 1,075 without. A decode-width
            // launch leaves the link mostly idle, and there the look-ahead pays:
            // ×16 decode 237 t/s with it, 220 without.
            //
            // A narrower launch that routes prompt-only experts — any expert
            // no decode-scored row routed — speculates only without room
            // (`PROMPT_ROOM`); one whose experts are all decode's speculates as
            // a decode launch does. With room the ring already promotes the
            // prompt's misses and a speculative copy only competes with them:
            // Qwen3.6-35B-A3B's one-sequence prompts ran 4–9% slower with it.
            // Without room, it is the one way a prompt's experts reach VRAM
            // ahead of the workers: Qwen3.8-Flash-Next's warm one-sequence
            // prompt ran 4% faster with it.
            let t = profile_now();
            if !msg.prefill_width && (decode.len() == expert_ids.len() || !prompt_room) {
                if expert_ids.len() * 2 >= self.inner.experts_per_layer.max(1) {
                    self.lookahead(row)?;
                } else {
                    self.speculative_prefetch(row, &expert_ids)?;
                }
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

    /// Land every promotion whose copy has completed: build the slot's views,
    /// install it, and point the expert's entry at VRAM.
    /// The copier reports completions; this thread makes no driver call for it.
    pub(crate) fn poll_promotions(&mut self) -> Result<()> {
        for id in self.copier.completed()? {
            let Some(i) = self.promotions.iter().position(|p| p.id == id) else {
                candle::bail!("expert copier completed copy {id}, which was never submitted");
            };
            let p = self.promotions.swap_remove(i);
            self.land(p)?;
        }
        Ok(())
    }

    fn land(&mut self, p: Promotion) -> Result<()> {
        let Device::Cuda(cd) = &self.device else {
            candle::bail!("expert promotion requires a CUDA device");
        };
        if p.from_pad {
            self.residency()?.pin_pad(p.row, p.expert, -1);
        }
        self.promoting.remove(&(p.row, p.expert));
        // The GPU promoted it first: nothing ever named this slot.
        if self
            .inner
            .key_to_slot
            .get(&(p.row, p.expert))
            .is_some_and(|&s| self.inner.slots[s].is_some())
        {
            self.inner.put_free(p.slot);
            return Ok(());
        }
        let base = self.inner.slot_base(p.slot);
        // SAFETY: the copy that filled the slot has completed.
        let view = unsafe { build_slot_view(&self.layer_geometries[p.row], cd, base)? };
        self.inner.install(p.slot, p.row, p.expert, view);
        self.residency()?
            .set_vram(p.row, p.expert, Some((p.slot, base)));
        self.speculative_loads.insert((p.row, p.expert));
        if let Ok(mut s) = self.stats.lock() {
            s.promotions += 1;
            s.promotion_bytes += p.bytes;
            s.prefetch_promotions += 1;
        }
        Ok(())
    }

    /// Up to `n` VRAM slots no kernel can be reading, for a speculative
    /// promotion: free ones first, then the best victims (ranked against
    /// `row`, the row the wave is at) from quiet rows in the window behind the
    /// wave. A row with no invocation in flight may have its entry go to 0
    /// (`reclaim`); a victim whose old readers may still run goes on the retire
    /// list and is reused once they are done. A slot standing in the promotion
    /// ring is the ring's, and never taken here.
    fn take_slots(&mut self, row: usize, n: usize) -> Result<Vec<usize>> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            match self.inner.take_free() {
                Some(s) => out.push(s),
                None => break,
            }
        }
        if out.len() == n {
            return Ok(out);
        }
        let quiet: Vec<bool> = (0..self.num_moe_layers)
            .map(|r| self.clock.quiet(r))
            .collect();
        // A row with an invocation enqueued is held back: the wave is about to
        // read its experts, so evicting one buys a miss within the pass. The
        // started-word key leaves every row ahead of the GPU quiet, so with the
        // forward recorded ahead of it nothing else keeps them out: on
        // Qwen3.6-35B-A3B (RTX 3090) a prompt's misses doubled and prefill fell
        // up to 24%.
        let upcoming: Vec<bool> = (0..self.num_moe_layers)
            .map(|r| self.clock.upcoming(r))
            .collect();
        let offered = &self.offered;
        let victims = self
            .inner
            .rank_victims(row, n - out.len(), true, |slot, layer| {
                quiet[layer] && !upcoming[layer] && !offered.contains(&slot)
            });
        let mut evicted = 0usize;
        for victim in victims {
            let Some((vr, ve)) = self.inner.evict(victim) else {
                out.push(victim);
                continue;
            };
            self.residency()?.set_vram(vr, ve, None);
            evicted += 1;
            let key = self.clock.retire_key(vr);
            if self.clock.reclaimable(key) {
                out.push(victim);
            } else {
                self.retired.push(key, victim);
            }
        }
        if let Ok(mut s) = self.stats.lock() {
            s.evictions += evicted;
        }
        Ok(out)
    }

    /// Collect what the GPU has taken from the promotion ring, in log order.
    /// A skipped victim's offer comes back with its expert still resident. A
    /// claimed victim was evicted by the device — its entries retargeted before
    /// its slot was written — so its bookkeeping follows at once, ahead of any
    /// later claim that could promote the same expert again. Each claimed slot
    /// becomes a device promotion, whole once its invocation is done. `near`
    /// is the ticket being served — the logged words are resolved against it.
    fn collect_device_promotions(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        let head = ring.head();
        let (mut taken, mut claimed, mut skipped) = (0usize, 0usize, 0usize);
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
                claimed += 1;
            }
            taken += 1;
            self.promoting.insert((logged.row, logged.expert));
            self.device_promotions.push(DevicePromotion {
                row: logged.row,
                expert: logged.expert,
                slot: offer.slot,
                ticket: ticket_from_word(logged.word, near),
            });
        }
        if let Ok(mut s) = self.stats.lock() {
            s.ring_taken += taken;
            s.victims_claimed += claimed;
            s.victims_skipped += skipped;
            s.evictions += claimed;
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
        let mut slots = self.inner.rank_victims(row, ask, false, |slot, layer| {
            !upcoming[layer] && candidate(slot)
        });
        if slots.len() < ask {
            let more = self
                .inner
                .rank_victims(row, ask - slots.len(), false, |slot, layer| {
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
            let ticket = self.device_promotions[i].ticket;
            if !(idle || self.clock.reclaimable(ticket)) {
                i += 1;
                continue;
            }
            let p = self.device_promotions.swap_remove(i);
            self.promoting.remove(&(p.row, p.expert));
            if let Some(ring) = &self.promo_ring {
                ring.clear_mark(p.row, p.expert);
            }
            // Promoted twice — by the GPU on two invocations, or by the copy
            // engine as well: nothing ever named this slot, so it is free.
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

    /// Take every offer out of the promotion ring — before a boundary move,
    /// with the device synchronized: land what the GPU filled, take back what
    /// it never took.
    pub(crate) fn drain_ring(&mut self, near: u64) -> Result<()> {
        let Some(ring) = self.promo_ring.clone() else {
            return Ok(());
        };
        self.collect_device_promotions(near)?;
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
        Ok(())
    }

    /// Prefetch `experts` of `row`, predicted ahead of it, into VRAM by the copy
    /// engine — those with a pinned copy; the cold ones go to the stager to
    /// stage ahead. Victims only from behind the wave (`take_slots`).
    /// `issue_row` is the row the wave is at — what victims are ranked against.
    fn promote(&mut self, issue_row: usize, row: usize, experts: &[usize]) -> Result<usize> {
        let Device::Cuda(_) = &self.device else {
            return Ok(0);
        };
        if row < self.inner.pinned_layers {
            return Ok(0);
        }
        let bytes = slot_offsets(&self.layer_geometries[row]).3;
        let mut to_stage: Vec<usize> = Vec::new();
        let mut sources: Vec<(usize, u64, bool)> = Vec::new();
        for &e in experts {
            if self.promoting.contains(&(row, e))
                || self
                    .inner
                    .key_to_slot
                    .get(&(row, e))
                    .is_some_and(|&s| self.inner.slots[s].is_some())
            {
                continue;
            }
            match self.residency()?.place(row, e).pinned_source() {
                Some((src, from_pad)) => sources.push((e, src, from_pad)),
                None => to_stage.push(e),
            }
        }
        let slots = self.take_slots(issue_row, sources.len())?;
        let mut issued = 0usize;
        for ((e, src, from_pad), slot) in sources.into_iter().zip(slots) {
            let dst = self.inner.slot_base(slot);
            if from_pad {
                self.residency()?.pin_pad(row, e, 1);
            }
            // `src` is a pinned slot image of at least `bytes`, kept unwritten
            // until the copy lands (a warm slot is immutable; a pad slot is
            // pinned against eviction above); `dst` is a slot the zone handed
            // out with no reader left (`take_slots`), named by no entry until
            // the copy is reported complete.
            let id = self.next_copy;
            self.next_copy += 1;
            if let Err(err) = self.copier.submit(CopyJob {
                id,
                dst,
                src,
                bytes,
            }) {
                if from_pad {
                    self.residency()?.pin_pad(row, e, -1);
                }
                self.inner.put_free(slot);
                return Err(err);
            }
            self.promotions.push(Promotion {
                id,
                row,
                expert: e,
                slot,
                bytes,
                from_pad,
            });
            self.promoting.insert((row, e));
            self.pass_dma_bytes += bytes;
            issued += 1;
        }
        if !to_stage.is_empty() {
            // A closed channel means the stager is gone; its guard has raised
            // the abort word, which the next routed layer reports.
            let _ = self.stager.send(StagerMsg::Stage {
                row,
                experts: to_stage,
            });
        }
        Ok(issued)
    }

    /// A decode-width launch that routed most of the row's experts (many
    /// sequences decoding together): the next row will too, so promote its scored
    /// experts, highest first — nothing to predict.
    fn lookahead(&mut self, row: usize) -> Result<()> {
        let target = row + 1;
        if target >= self.num_moe_layers || target < self.inner.pinned_layers {
            return Ok(());
        }
        let mut cands: Vec<(usize, f32)> = (0..self.inner.experts_per_layer)
            .filter(|&e| !self.inner.key_to_slot.contains_key(&(target, e)))
            .filter_map(|e| {
                let s = self.inner.score(target, e);
                (s > 0.0).then_some((e, s))
            })
            .collect();
        cands.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(LOOKAHEAD_MAX);
        let experts: Vec<usize> = cands.into_iter().map(|(e, _)| e).collect();
        self.promote(row, target, &experts)?;
        Ok(())
    }

    /// Speculatively promote the experts the next `prefetch_depth` MoE layers
    /// will need, one confidence-gated hop at a time.
    ///
    /// Prediction stays ADJACENT — each hop asks the shared transition matrix
    /// to predict `target` from the previous hop's set — so precision compounds
    /// per hop and the chain stops at the first hop with no confident
    /// prediction. Mispredictions cost slots and bandwidth, never correctness.
    fn speculative_prefetch(&mut self, row: usize, current: &[usize]) -> Result<()> {
        let depth = if current.len() <= PREFETCH_MULTI_HOP_MAX_SOURCES {
            self.prefetch_depth
        } else {
            1
        };
        let mut source: Vec<usize> = current.to_vec();
        for hop in 1..=depth {
            let target = row + hop;
            if target >= self.num_moe_layers {
                break;
            }
            let predicted = self.transition_matrix.predict_prefetch(target - 1, &source);
            if predicted.is_empty() {
                break;
            }
            // A pinned layer's experts are all resident; the prediction still
            // chains through it.
            if target >= self.inner.pinned_layers {
                self.promote(row, target, &predicted)?;
            }
            source = predicted;
        }
        Ok(())
    }

    /// Adapt the load-ahead depth `N` at a pass boundary, from the pass that
    /// just finished: deepen when promotions were late and the link had slack,
    /// shallow when nothing was late and precision fell below the floor.
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

    /// The zone's gauges, refreshed each routed layer.
    fn publish_gauges(&self) {
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

    /// Return every retired slot to the zone — before a boundary move, with the
    /// device synchronized. A retired slot waits on a ticket the host has not
    /// yet observed; once the device is idle its readers are done whatever was
    /// observed, and a slot left on the list past a retraction names ground the
    /// per-slot tables no longer cover.
    pub(crate) fn release_retired(&mut self) {
        for slot in self.retired.drain_all() {
            self.inner.put_free(slot);
        }
    }

    /// Wait out every promotion copy and land it — before a boundary move,
    /// which relocates and drops slots, and issues its relocations on the
    /// copy stream after the copier has nothing left to issue.
    pub(crate) fn finish_promotions(&mut self) -> Result<()> {
        self.copier.flush()?;
        self.poll_promotions()?;
        debug_assert!(self.promotions.is_empty());
        Ok(())
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
/// traps and the next synchronizing call reports it.
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
            // This thread builds the views of landed slots, which needs the
            // device's context current on it. Promotion copies and their
            // events are the copier thread's.
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
                            .and_then(|()| state.poll_promotions())
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
                    PipelineMessage::SnapshotProfile { response_tx } => {
                        let snap = state.profile.snapshot();
                        state.profile.reset();
                        let _ = response_tx.send(snap);
                    }
                    // The give-back, reachable without a completed forward.
                    // Answering zero is a legitimate outcome (nothing spare, a
                    // wave still live, an invocation still unserved).
                    PipelineMessage::RenegotiateBoundary {
                        regions,
                        response_tx,
                    } => {
                        let conceded = match state.renegotiate_if_quiet(Some(regions)) {
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
    use super::{ring_stock, RingStock};

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
        assert_eq!(ring_stock(8, 100, true, 0, 1024), stock(125, 0, 125));
        // The floor and the half-capacity ceiling.
        assert_eq!(ring_stock(0, 4, true, 0, 1024), stock(32, 0, 32));
        assert_eq!(ring_stock(0, 1_000, true, 0, 1024), stock(512, 0, 512));
        // More empty slots than the prediction: every one is stocked, and the
        // sweep word stays the prediction.
        assert_eq!(ring_stock(8, 100, true, 300, 1024), stock(300, 0, 125));
    }

    /// Without room decode's target is the reserve and the sweep word, and the
    /// stock past it is empty slots only — a full zone stocks exactly decode's
    /// target.
    #[test]
    fn without_room_the_stock_past_decodes_reserve_is_empty_slots() {
        // Full zone: stock = reserve = sweep = decode's target.
        assert_eq!(ring_stock(40, 100, false, 0, 1024), stock(50, 50, 50));
        // Fewer empties than decode's target: the target, made of lazy victims.
        assert_eq!(ring_stock(40, 100, false, 25, 1024), stock(50, 50, 50));
        // Plenty empty: every one stocked, decode's target still reserved, and
        // a prompt claiming past it still a sweep.
        assert_eq!(ring_stock(40, 100, false, 300, 1024), stock(300, 50, 50));
        // Never past half the ring.
        assert_eq!(ring_stock(40, 1_000, false, 1_000, 256), stock(128, 50, 50));
        // A prompt-only launch: decode's floor is still reserved.
        assert_eq!(ring_stock(0, 100, false, 10, 1024), stock(32, 32, 32));
    }
}

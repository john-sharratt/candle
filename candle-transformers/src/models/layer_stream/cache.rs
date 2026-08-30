//! The layer cache: residency, the tiers under it, and the transfers between.
//!
//! Ties [`residency`](super::residency) (which layer goes where) to the tiers
//! that hold the bytes — the weight zone's slots, a pinned warm run, and the
//! cold pack — and issues the copies on a dedicated stream so they overlap the
//! compute of the layers already resident.
//!
//! # The protocol
//!
//! Per forward, per layer:
//!
//! ```text
//! ensure(L)           the wave moves to L; join L's fence if one is in flight;
//!                     hand back its view
//! prefetch()          plan from L, issue what the plan asks for
//! ```
//!
//! `ensure` is the only place the forward can stall, and it stalls on exactly
//! one event: the fence recorded after `L`'s last copy. A layer that was already
//! resident joins nothing.
//!
//! **The wave moves inside `ensure`, not on a call of its own.** Asking for a
//! layer *is* the statement that the forward has arrived at it, and there is no
//! caller that would want one without the other: an `ensure` not preceded by its
//! `advance` measures every distance from the wrong position, which silently
//! picks the wrong victim rather than failing. Two calls that must always be
//! made together, in one order, are one call.
//!
//! # Why a fence per layer and not one per plan
//!
//! A plan issues several layers' copies back to back, and the wave reaches them
//! one at a time. A single fence at the end of the plan would make the wave wait
//! for `L+3`'s bytes before reading `L+1`. One event per layer, recorded after
//! that layer's copies, lets each be joined at the moment it is actually needed
//! — the same wait-at-need shape the expert streamer uses.

use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaStream};
use candle::quantized::Int8Mode;
use candle::{CudaDevice, Result};
use std::sync::Arc;

use super::descriptor::LayerImage;
use super::pack::{LayerPack, PackRead};
use super::residency::{LayerResidency, LoadOp, PlanScratch, Residence};
use super::view::{build_layer_view, StreamedLayer};
use super::warm::{warm_membership, warm_slots_for};
use crate::models::expert_lre::pinned::WarmPool;
use crate::models::layer_stream::zone::{LayerPlacement, ZonePlan};
use candle::vram::PinnedUse;

pub use super::residency::COMMITTED_DEPTH;

/// A failed [`LayerCache::issue`], and whether the slot survived it.
///
/// The distinction is the whole reason this type exists: the rollback may put
/// the evicted tenant back only when nothing was enqueued, and must clear the
/// slot's view when something was.
enum Failed {
    /// The H2D was never enqueued — the slot still holds its previous tenant,
    /// bytes and view both.
    Untouched(candle::Error),
    /// The copy is in flight over the slot. The previous tenant is gone.
    Clobbered(candle::Error),
}

/// Buffers in the cold-read staging ring.
///
/// Each is one record wide, so the ring costs `STAGING_SLOTS × stride` of pinned
/// host memory — ~1.1 GiB at the 27B's ~290 MB record. That is real, and it is
/// why this is four rather than the plan's full width: the ring only has to
/// cover the reads *in flight at once*, and a read is finished with its buffer
/// the moment the H2D behind it lands. Four gives the copy stream three landed
/// buffers of slack before a fill has to wait on one.
pub const STAGING_SLOTS: usize = 4;

/// Counters for the gate's report.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LayerCacheStats {
    /// The layer was already resident when the wave reached it — no fence, no
    /// bytes moved.
    pub hits: usize,
    /// Promoted from the pinned host tier.
    pub warm_hits: usize,
    /// Read from the pack.
    pub cold_reads: usize,
    /// The wave joined a layer's fence before reading it.
    ///
    /// **Not a measure of time lost.** A join whose copy has already landed
    /// costs nothing, and on a ring smaller than the model *every* streamed
    /// layer is joined by construction — the count is bounded below by the
    /// number of layers that had to stream, which §9.2 says is
    /// `total − resident` and no policy can reduce. What it does distinguish is
    /// the two regimes: a model that fits reports joins near zero, and a model
    /// that does not reports one per streamed layer.
    ///
    /// Measuring the *wait* would need the fence's elapsed time, which is a
    /// device round trip on the hot path and is deliberately not taken here.
    pub fence_joins: usize,
    pub evictions: usize,
    /// Opportunistic loads dropped because their transfer failed.
    pub abandoned: usize,
    /// Layers with a permanent address **right now**, not at the carve.
    ///
    /// State rather than a counter, and filled by [`LayerCache::stats`] at read
    /// time for that reason. It is here because the carve is the only residency
    /// anything printed, and the carve is not the number that decides
    /// throughput: the boundary grows into spare KV ground over the first
    /// forwards, so a zone that opens at 47 of 64 may be running at 64. Reading
    /// that off the transfer counters is an inference; this is the observation.
    pub homed: usize,
    /// Layers that still cross PCIe every forward. `0` means the zone holds the
    /// trunk and the streaming machinery is inert.
    pub streaming: usize,
}

/// Turns a slot's views into what the forward reads.
///
/// A trait rather than a bare `Fn` bound because the model's assembler is a
/// named type — it carries the per-layer residues and images — and a type that
/// has to be spelled in a struct field cannot be a closure. The blanket impl
/// keeps closures working, which is what lets the cache's own tests exercise it
/// with no model at all.
pub trait SlotAssembler<T> {
    fn assemble(&self, view: StreamedLayer, layer: usize) -> Result<T>;
}

impl<T, F: Fn(StreamedLayer, usize) -> Result<T>> SlotAssembler<T> for F {
    fn assemble(&self, view: StreamedLayer, layer: usize) -> Result<T> {
        self(view, layer)
    }
}

/// Where a layer's bytes come from when it is not resident.
enum Source {
    /// A pinned host slot — one H2D of contiguous bytes.
    Warm(usize),
    /// The pack, staged through a pinned buffer.
    Cold,
}

/// Residency plus the tiers and the stream that serve it.
///
/// Generic over what a slot's bytes are *presented* as. The cache moves bytes
/// and tracks residency; turning a slot into something a forward can read is
/// model knowledge, so it is supplied as `assemble` rather than known here.
/// The qwen35 lineage passes a closure that builds a whole `QuantLayer` (see
/// [`super::assemble`]), which is what lets the forward keep taking
/// `&QuantLayer` and never learn a slot exists. A test can pass the identity
/// and get the raw views back.
pub struct LayerCache<T, A>
where
    A: SlotAssembler<T>,
{
    device: CudaDevice,
    /// The copy stream. Separate from the compute stream so a transfer overlaps
    /// the layers already resident rather than serialising behind them.
    copy: Arc<CudaStream>,
    images: Vec<LayerImage>,
    /// The numeric mode the model was loaded for, so a rebuilt view reports the
    /// same path the loader chose. A slot holds a KO twin or a source quant, and
    /// which one it is decides whether the weight is fed q8a128 activations.
    mode: Int8Mode,
    residency: LayerResidency,
    pack: LayerPack,
    warm: WarmPool,
    /// `layer → warm slot`, for the static membership.
    warm_slot: Vec<Option<usize>>,
    /// Staging for a cold read whose layer has no warm slot.
    ///
    /// A **ring**, and it has to be. The read fills a buffer on the host and the
    /// upload out of it is asynchronous on the copy stream, so the buffer is
    /// still being read by the DMA after `issue` returns. One buffer would be
    /// overwritten by the next cold read in the same plan while the previous
    /// transfer was mid-flight, and the layer that transfer was carrying would
    /// land holding a different layer's bytes.
    ///
    /// That is not hypothetical: it is what a single buffer did. The bug hid for
    /// as long as the warm tier happened to cover every streamed layer (`cold 0`,
    /// so the path never ran twice in a plan); sizing the warm tier honestly
    /// against host RAM introduced 132 cold reads per config on the 27B and the
    /// model started producing garbage logits within three configs.
    staging: WarmPool,
    /// `staging slot → the event marking its last upload complete`. A slot is
    /// reused only after this is joined, which is what makes the ring a ring
    /// rather than a rotation.
    staging_done: Vec<Option<CudaEvent>>,
    /// Next staging slot to hand out.
    staging_next: usize,
    /// Where each zone slot is and how big it is.
    ///
    /// **Not one size.** A homed slot is exactly its layer's image; only the
    /// floating cell is sized for the largest streamable layer, because only it
    /// has to hold more than one thing. See [`super::zone`].
    slots: Vec<LayerPlacement>,
    /// `slot → view`, rebuilt whenever a slot's tenant changes.
    ///
    /// Behind an `Arc` so [`Self::ensure`] can hand the caller a handle rather
    /// than a borrow. A borrow would be tied to the `&mut self` that `ensure`
    /// takes, which makes the documented per-layer protocol — `ensure` then
    /// `prefetch` — fail to compile: the forward would be holding a reference
    /// into the very structure the next call mutates. The `Arc` also states the
    /// lifetime honestly, because `prefetch` may evict the slot the handle came
    /// from, and the handle must keep the assembled matmuls alive until the
    /// layer's compute has been issued.
    views: Vec<Option<Arc<T>>>,
    /// Turns a slot's views into what the forward reads. Called once per
    /// tenancy — when a layer lands in a slot, not when it is read.
    assemble: A,
    /// `layer → fence`, recorded after that layer's copies.
    fences: Vec<Option<CudaEvent>>,
    /// The planner's buffers, held so a per-layer plan allocates nothing.
    plan_scratch: PlanScratch,
    plan_ops: Vec<LoadOp>,
    stats: LayerCacheStats,
}

impl<T, A: SlotAssembler<T>> std::fmt::Debug for LayerCache<T, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerCache")
            .field("layers", &self.images.len())
            .field("slots", &self.slots.len())
            .field("warm", &self.warm.num_slots())
            .field("stats", &self.stats)
            .finish()
    }
}

impl<T, A: SlotAssembler<T>> LayerCache<T, A> {
    /// Build a cache over the zone `plan` describes, filling the warm tier from
    /// the pack.
    ///
    /// The plan's addresses descend from the zone's top in protection order, so
    /// slot 0 is the rightmost ground — where the pinned head lands, which
    /// retraction can never reach — and the frontier holds the floating cell.
    ///
    /// `pub(crate)` because it takes the cold tier by value: a caller outside
    /// this crate has no way to have opened one, and the entry point that
    /// builds both together is the model loader's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &CudaDevice,
        images: Vec<LayerImage>,
        mode: Int8Mode,
        pack: LayerPack,
        plan: &ZonePlan,
        warm_budget_slots: usize,
        assemble: A,
    ) -> Result<Self>
    where
        A: SlotAssembler<T>,
    {
        let num_layers = images.len();
        let pinned = pack.pinned_layers();
        let stride = pack.stride();

        // **The staging ring is claimed first, and that ordering is the point.**
        //
        // It is small and *structurally required* — without a buffer a cold read
        // cannot happen at all — while the warm tier is large and elastic, and
        // exists only to make cold reads rarer. Sizing the elastic thing first
        // lets it spend the whole host budget and leave the mandatory one to fail,
        // which is precisely backwards: the tier would be at its maximum and
        // every cold read it could not cover would hard-fail for want of a
        // buffer.
        //
        // The ring cannot be *measured* by the host probe that sets
        // `warm_budget_slots`, because that probe necessarily runs before this
        // constructor is entered. So the ring is charged against the budget
        // rather than observed in it: the caller deducts `STAGING_SLOTS` from
        // what the probe allowed, both tiers being denominated in the same
        // record stride. Claiming the ring first then makes the deduction true
        // — the mandatory buffers exist before the elastic tier can spend
        // anything.
        //
        // Each buffer is one record wide, so it carries the same sector-aligned
        // stride the pack reads want and a cold read lands in one directly.
        let staging = WarmPool::new(STAGING_SLOTS, stride, PinnedUse::Staging);
        let staging_ring = staging.num_slots();

        // The warm tier serves the layers that stream, and it is drawn from the
        // **top** of the model so that it stays correct as the held prefix
        // grows — see `warm`'s header. It is filled once and never redrawn,
        // which is only sound because the top run is streamed under every prefix
        // the tier can reach.
        let want = warm_slots_for(num_layers, pinned, warm_budget_slots);
        let warm = WarmPool::new(want, stride, PinnedUse::WeightWarmTier);
        let members = warm_membership(num_layers, pinned, warm.num_slots());
        let slots = slot_table(plan);
        let mut warm_slot = vec![None; num_layers];
        for (slot, &layer) in members.iter().enumerate() {
            warm_slot[layer] = Some(slot);
        }

        let mut cache = Self {
            device: device.clone(),
            copy: device
                .cuda_context()
                .new_stream()
                .map_err(candle::Error::wrap)?,
            residency: LayerResidency::new(plan, pinned),
            images,
            mode,
            pack,
            warm,
            warm_slot,
            staging,
            // Sized from the pool's answer, not the request: it steps down until
            // the host accepts, so these must agree or the ring indexes past its
            // events.
            staging_done: (0..staging_ring).map(|_| None).collect(),
            staging_next: 0,
            slots,
            views: Vec::new(),
            assemble,
            fences: Vec::new(),
            plan_scratch: PlanScratch::default(),
            plan_ops: Vec::new(),
            stats: LayerCacheStats::default(),
        };
        cache.views = (0..cache.slots.len()).map(|_| None).collect();
        cache.fences = (0..num_layers).map(|_| None).collect();
        cache.fill_warm(&members)?;
        cache.build_pinned_views()?;
        Ok(cache)
    }

    /// Build the views over the pinned head's slots.
    ///
    /// The pinned layers are the one set that never passes through
    /// [`Self::issue`] — they have no pack record and are never loaded — so
    /// nothing else would ever build their views, and the first `ensure(0)`
    /// would find a resident layer whose slot has no matmuls over it. That is
    /// what the residency placing them in [`LayerResidency::new`] leaves for
    /// this to finish.
    ///
    /// A view is geometry over an address, so it is built here even though the
    /// bytes are not in the slot yet: **the caller must upload the pinned
    /// layers' images into `slot_base[0..pinned]` before the first `ensure`**.
    /// That is the loader's job, because the pinned head is the part of the
    /// model that comes straight from the checkpoint and never round-trips
    /// through a tier.
    fn build_pinned_views(&mut self) -> Result<()> {
        for layer in 0..self.residency.pinned() {
            let Some(slot) = self.residency.residence(layer).slot() else {
                continue;
            };
            // SAFETY: `slot_base[slot]` names a slot the zone handed out, and
            // the caller's upload puts this layer's image at these offsets
            // before any kernel reads it.
            let view = unsafe {
                build_layer_view(
                    &self.images[layer],
                    &self.device,
                    self.slots[slot].base,
                    self.mode,
                )
            }?;
            self.views[slot] = Some(Arc::new(self.assemble.assemble(view, layer)?));
        }
        Ok(())
    }

    /// Read the warm tier's members out of the pack, once.
    ///
    /// Verified, unlike the runtime miss path: this reads every member exactly
    /// once with idle cores, where a per-read checksum inside a forward would
    /// multiply its latency.
    fn fill_warm(&mut self, members: &[usize]) -> Result<()> {
        if members.is_empty() {
            return Ok(());
        }
        let stride = self.pack.stride();
        // The pool is one contiguous allocation and the members are a
        // contiguous ascending run, so the whole tier is one span of bytes.
        let span = self.warm.span_mut(0, members.len());
        let mut targets = Vec::with_capacity(members.len());
        for (chunk, &layer) in span.chunks_exact_mut(stride).zip(members) {
            targets.push(PackRead { layer, dest: chunk });
        }
        self.pack.read_many(targets)
    }

    pub fn stats(&self) -> LayerCacheStats {
        LayerCacheStats {
            homed: self.residency.homed(),
            streaming: self.residency.streaming(),
            ..self.stats
        }
    }

    pub fn residency(&self) -> &LayerResidency {
        &self.residency
    }

    /// Bytes the zone currently holds, from the frontier to its top.
    ///
    /// What the boundary trades against KV. Slots are not one size, so this is a
    /// sum rather than a product — and the difference is the point: the same
    /// ground holds more layers when each is charged its own size.
    pub fn zone_bytes(&self) -> usize {
        self.slots.iter().map(|s| s.bytes).sum()
    }

    /// The lowest address the zone occupies.
    pub fn floor(&self) -> u64 {
        self.slots.last().map(|s| s.base).unwrap_or(0)
    }

    /// The widest layer image, in bytes — this consumer's allocation unit.
    ///
    /// The boundary's growth direction spends ground in whole layers, so a grant
    /// smaller than this buys nothing at all. The pool has to be told that, or it
    /// sizes grants for a consumer whose unit is an expert slot and this one
    /// discards every one of them.
    ///
    /// The **widest** rather than the mean, because the planner packs from the
    /// top of the protection order and the next layer it would take back is not
    /// known here — sizing on the mean would leave grants that buy a layer only
    /// when the next one happens to be small.
    pub fn widest_layer_bytes(&self) -> usize {
        self.images.iter().map(|i| i.total).max().unwrap_or(0)
    }

    /// The layout `budget` bytes ending at `end` would buy.
    ///
    /// The boundary's question in both directions: it proposes a budget and
    /// reads back how many layers that holds, without anything having moved.
    /// A budget under the model's floor — the pinned head plus one streaming
    /// cell — is an error, which is how the retraction direction learns it can
    /// concede nothing rather than producing a zone that cannot run.
    pub fn plan_for(
        &self,
        end: u64,
        budget: usize,
    ) -> Result<crate::models::layer_stream::zone::ZonePlan> {
        crate::models::layer_stream::zone::plan_zone(
            &self.images,
            self.residency.pinned(),
            end,
            budget,
        )
        .map_err(|e| candle::Error::Msg(e.to_string()))
    }

    /// The device the slots live on, for the boundary move's stream.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// A layer's image — the offsets its projections sit at inside a slot.
    pub fn image(&self, layer: usize) -> Result<&LayerImage> {
        self.images
            .get(layer)
            .ok_or_else(|| candle::Error::Msg(format!("layer cache: no image for L{layer}")))
    }

    /// The device address of the slot `layer` currently occupies.
    ///
    /// For the loader's pinned-head upload, which is the one write to a slot
    /// that does not come through [`Self::issue`]: the pinned layers have no
    /// record in any tier, so their bytes come straight from the checkpoint.
    pub fn slot_base_of(&self, layer: usize) -> Result<u64> {
        let slot =
            self.residency.residence(layer).slot().ok_or_else(|| {
                candle::Error::Msg(format!("layer cache: L{layer} holds no slot"))
            })?;
        self.slots
            .get(slot)
            .map(|s| s.base)
            .ok_or_else(|| candle::Error::Msg(format!("layer cache: slot {slot} is past the zone")))
    }

    /// Move the wave to `layer` and hand back its matmuls, joining its transfer
    /// first if one is in flight.
    ///
    /// The only place a forward can block on this subsystem, and the only place
    /// the wave moves — see the module header for why those are one call.
    pub fn ensure(&mut self, layer: usize) -> Result<Arc<T>> {
        self.residency.set_wave(layer);
        match self.residency.residence(layer) {
            Residence::Resident(_) => self.stats.hits += 1,
            Residence::Loading(_) => {
                self.stats.fence_joins += 1;
                self.join(layer)?;
                self.residency.finish_load(layer);
            }
            Residence::Absent => {
                // The wave reached a layer no plan had placed — capacity is
                // tight enough that the committed prefix did not cover it. Load
                // it now and pay the full latency, which is the honest cost
                // rather than a wrong answer.
                let mut ops = std::mem::take(&mut self.plan_ops);
                self.residency
                    .plan_into(COMMITTED_DEPTH, &mut self.plan_scratch, &mut ops);
                let found = ops.iter().copied().find(|o| o.layer == layer);
                self.plan_ops = ops;
                let op = found.ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "layer cache: L{layer} is absent and no slot can hold it — the \
                             zone is smaller than the committed prefix needs"
                    ))
                })?;
                // Through `issue_all` rather than `issue` so a failure rolls the
                // `Loading` mark back — the layer must not be left looking
                // in-flight over a slot holding its predecessor.
                self.issue_all(&[op])?;
                self.stats.fence_joins += 1;
                self.join(layer)?;
                self.residency.finish_load(layer);
            }
        }
        let slot = self
            .residency
            .residence(layer)
            .slot()
            .ok_or_else(|| candle::Error::Msg(format!("layer cache: L{layer} has no slot")))?;
        self.views[slot]
            .as_ref()
            .map(Arc::clone)
            .ok_or_else(|| candle::Error::Msg(format!("layer cache: slot {slot} has no view")))
    }

    /// Fill every slot before the first forward runs.
    ///
    /// The zone comes up holding only the pinned head, so without this the first
    /// forward pays for the whole residency — every slot's first tenant — inside
    /// a wave, while the scheduler is timing it. Measured on the 9B: the gate's
    /// first config ran at **18.3 t/s against a resident model's 80.8**, and
    /// every config after it was at parity, because by then the slots were full.
    ///
    /// The bytes have to move either way; this puts them at load, where the cost
    /// belongs and where nothing is waiting on them. A daemon pays it once at
    /// startup instead of on its first request.
    pub fn warm_start(&mut self) -> Result<()> {
        // Plan from the front, since that is where the first forward begins, and
        // repeat until a pass places nothing — the plan fills free slots before
        // it evicts, so this terminates when the zone is full.
        //
        // **Termination is on `live()`; honesty is on `abandoned`.** `live()` is
        // invariant under a load that fails and rolls back (`begin_load` adds
        // one, the rollback removes it), so a pack that is briefly unreadable —
        // an NVMe hiccup, EBUSY — makes every op of the first pass fail, leaves
        // `live()` unchanged, and breaks this loop after one iteration. It must
        // still break, or a persistently failing read spins here forever; what it
        // must not do is then report success. The first forward would pay a
        // synchronous fetch per layer: the exact 18.3 t/s regression this
        // function exists to prevent, logged as "filled".
        let mut abandoned_during_fill = 0usize;
        loop {
            let before = self.residency.live();
            let failures = self.stats.abandoned;
            self.prefetch()?;
            abandoned_during_fill += self.stats.abandoned - failures;
            if self.residency.live() == before {
                break;
            }
        }
        // One join for all of them rather than a fence apiece: nothing is
        // overlapping this, so there is no reason to keep the events around for
        // the first forward to trip over.
        self.copy.synchronize().map_err(candle::Error::wrap)?;
        for layer in 0..self.images.len() {
            self.residency.finish_load(layer);
            self.fences[layer] = None;
        }
        // A fill that ended because its reads were failing is a different event
        // from one that ended because the zone was full, and they must not share
        // a message: the first means the next forward pays synchronously for
        // every layer it touches, which is a startup fault worth seeing.
        if abandoned_during_fill > 0 && self.residency.live() < self.residency.homed() {
            tracing::warn!(
                target: "candle_transformers::layer_stream",
                resident = self.residency.live(),
                of = self.images.len(),
                homed = self.residency.homed(),
                abandoned = abandoned_during_fill,
                "layer zone fill stopped early — {abandoned_during_fill} reads failed, so the \
                 first forward will fetch its layers synchronously"
            );
        } else {
            tracing::info!(
                target: "candle_transformers::layer_stream",
                resident = self.residency.live(),
                of = self.images.len(),
                homed = self.residency.homed(),
                streaming = self.residency.streaming(),
                "layer zone filled before the first forward"
            );
        }
        Ok(())
    }

    /// Plan from the wave's position and issue what it asks for.
    ///
    /// Called after the layer's compute is issued, so the copies overlap it.
    pub fn prefetch(&mut self) -> Result<()> {
        // Taken and put back so the planner's output buffer is reused across
        // every layer of every forward — `issue` needs `&mut self`, which a
        // borrow of the field would forbid, and the swap costs three words.
        let mut ops = std::mem::take(&mut self.plan_ops);
        self.residency
            .plan_into(COMMITTED_DEPTH, &mut self.plan_scratch, &mut ops);
        let r = self.issue_all(&ops);
        self.plan_ops = ops;
        r
    }

    fn issue_all(&mut self, ops: &[LoadOp]) -> Result<()> {
        for &op in ops {
            if let Err(failed) = self.issue(op) {
                // **Roll back before deciding what to do about it.** `issue`
                // marks the layer `Loading` in its first statement, so a failure
                // anywhere after that leaves it in flight with no fence behind
                // it — and `ensure` reads `Loading` as "join and promote", joins
                // nothing, and hands back a view over a slot still holding the
                // tenant this op evicted. Wrong weights, no error, and the
                // committed path was the one that skipped the rollback.
                let (e, intact) = match failed {
                    Failed::Untouched(e) => (e, true),
                    Failed::Clobbered(e) => {
                        // The copy is in flight over the slot, so the view built
                        // for its previous tenant now describes another layer's
                        // bytes. Dropping it is what stops `reshape` — or any
                        // reader going through `slot_layer` — handing a forward
                        // the wrong weights.
                        self.views[op.slot] = None;
                        (e, false)
                    }
                };
                self.residency.abandon_load(op, intact);
                self.fences[op.layer] = None;
                if op.committed {
                    return Err(e);
                }
                // An opportunistic load is allowed to fail: its slot is back and
                // the next plan tries again. Only the committed prefix is
                // load-bearing.
                tracing::debug!(
                    target: "candle_transformers::layer_stream",
                    layer = op.layer,
                    "opportunistic layer load failed ({e}); abandoning"
                );
                self.stats.abandoned += 1;
            }
        }
        Ok(())
    }

    /// Move one layer's bytes into its slot and record its fence.
    ///
    /// Failures are reported either side of **the point of no return**, which is
    /// the H2D: before it nothing has moved and the slot still holds its
    /// previous tenant; after it those bytes are being overwritten whether the
    /// rest of this function succeeds or not. `issue_all` needs the difference to
    /// know whether it may put the evicted tenant back — see
    /// [`LayerResidency::abandon_load`].
    fn issue(&mut self, op: LoadOp) -> std::result::Result<(), Failed> {
        // Set the instant the first H2D is enqueued, and never unset: from that
        // point the slot's previous contents are gone regardless of what fails
        // afterwards.
        let mut clobbered = false;
        match self.issue_inner(op, &mut clobbered) {
            Ok(()) => Ok(()),
            Err(e) if clobbered => Err(Failed::Clobbered(e)),
            Err(e) => Err(Failed::Untouched(e)),
        }
    }

    fn issue_inner(&mut self, op: LoadOp, clobbered: &mut bool) -> Result<()> {
        self.residency.begin_load(op);
        if op.evicted.is_some() {
            self.stats.evictions += 1;
        }
        // The slot's previous tenant may still be under read by a GEMM the
        // compute stream has issued. Ordering the copy stream after the compute
        // stream's current point is what keeps the eviction from overwriting
        // bytes that are still being read.
        let ready = self
            .device
            .cuda_stream()
            .record_event(None)
            .map_err(candle::Error::wrap)?;
        self.copy.wait(&ready).map_err(candle::Error::wrap)?;

        let base = self.slots[op.slot].base;
        // **The layer's own bytes, not the slot's.** A homed slot is exactly
        // this size anyway; the floating cell is sized for the largest
        // streamable layer, and copying its full width would move bytes that are
        // not part of this layer's image across the bus on every fetch.
        let bytes = self.images[op.layer].total;
        let source = match self.warm_slot[op.layer] {
            Some(w) => Source::Warm(w),
            None => Source::Cold,
        };
        match source {
            Source::Warm(w) => {
                self.stats.warm_hits += 1;
                let src = self.warm.slot_ref(w, bytes);
                *clobbered = true;
                self.upload(src, base)?;
            }
            Source::Cold => {
                self.stats.cold_reads += 1;
                let stride = self.pack.stride();
                // **Modulo what the pool actually gave, not what was asked
                // for.** `WarmPool::new` steps its request down until
                // `cuMemAllocHost` accepts, so a host under pressure hands back
                // two buffers or one — and indexing past that is an assertion
                // inside `slot_mut`, i.e. a panic on the forward thread.
                let ring = self.staging.num_slots();
                if ring == 0 {
                    candle::bail!(
                        "layer cache: L{} needs a cold read and the staging ring is empty — \
                         the host could not pin a single {}-byte buffer",
                        op.layer,
                        stride
                    );
                }
                let slot = self.staging_next % ring;
                self.staging_next = (slot + 1) % ring;
                // **Wait for this buffer's last upload before refilling it.**
                // The fill below is a host write; the upload out of it is a DMA
                // the copy stream is still running. Without this join the two
                // overlap and the earlier layer lands holding this one's bytes.
                if let Some(done) = self.staging_done[slot].take() {
                    done.synchronize().map_err(candle::Error::wrap)?;
                }
                let dest = self.staging.slot_mut(slot, stride);
                self.pack.read_into(op.layer, dest)?;
                let src = self.staging.slot_ref(slot, bytes);
                *clobbered = true;
                self.upload(src, base)?;
                self.staging_done[slot] =
                    Some(self.copy.record_event(None).map_err(candle::Error::wrap)?);
            }
        }
        let fence = self.copy.record_event(None).map_err(candle::Error::wrap)?;
        self.fences[op.layer] = Some(fence);

        // The view is geometry over an address, so it is built once per tenancy
        // rather than per read. It is safe to build before the copy lands — no
        // kernel reads it until `ensure` has joined the fence.
        //
        // SAFETY: `base` names a slot the zone handed out and has not
        // reclaimed, and the copy above wrote this layer's image into it.
        let view =
            unsafe { build_layer_view(&self.images[op.layer], &self.device, base, self.mode) }?;
        self.views[op.slot] = Some(Arc::new(self.assemble.assemble(view, op.layer)?));
        Ok(())
    }

    /// One H2D of a whole layer image on the copy stream.
    fn upload(&self, src: &[u8], dst: u64) -> Result<()> {
        use candle::cuda_backend::cudarc::driver::sys;
        let n = src.len();
        // SAFETY: `src` is pinned host memory of at least `n` bytes and `dst`
        // names `n` bytes of a slot the zone owns. Async on the copy stream —
        // the fence recorded after it is what orders any read.
        let res = unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src.as_ptr() as *const _, n, self.copy.cu_stream())
        };
        if res != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("layer cache: H2D of {n} B failed: {res:?}");
        }
        Ok(())
    }

    /// Make the compute stream wait for `layer`'s transfer.
    ///
    /// A GPU-side wait, not a host synchronize: the host has nothing to do with
    /// the ordering, and draining here would serialise the very overlap this
    /// subsystem exists to create.
    fn join(&mut self, layer: usize) -> Result<()> {
        if let Some(fence) = self.fences[layer].take() {
            self.device
                .cuda_stream()
                .wait(&fence)
                .map_err(candle::Error::wrap)?;
        }
        Ok(())
    }

    /// Adopt a new zone layout — the boundary move, in either direction.
    ///
    /// **Between forwards only** (`docs/qwen38_layer_streaming.md` §6): the
    /// caller has already established that no wave is open.
    ///
    /// One entry point rather than a retract and a grow, because with a nested
    /// protection order they are the same operation. A layer keeping its home
    /// keeps its **address**, so nothing is relocated, no surviving view is
    /// rebuilt, and no layer is re-fetched merely for having moved — moving a
    /// tenant between slot indices without copying its weights would point a
    /// live view at another layer's bytes, which is why the order is nested in
    /// the first place.
    ///
    /// What does change: layers that lost their home drop their views, and
    /// layers that gained one gain an empty one for the next plan to fill.
    pub fn reshape(&mut self, plan: &ZonePlan) -> Result<Vec<usize>> {
        let dropped = self.residency.reshape(plan);
        for layer in &dropped {
            self.fences[*layer] = None;
        }
        let next = slot_table(plan);
        // A view is geometry over an address. Keep the ones whose slot still
        // names the same address and still holds the same layer; drop the rest.
        let mut views: Vec<Option<Arc<T>>> = (0..next.len()).map(|_| None).collect();
        for (slot, place) in next.iter().enumerate() {
            let same_address = self.slots.get(slot).is_some_and(|s| s.base == place.base);
            if same_address && self.residency.slot_layer(slot).is_some() {
                views[slot] = self.views.get(slot).and_then(|v| v.clone());
            }
        }
        self.views = views;
        self.slots = next;
        Ok(dropped)
    }
}

/// The plan's placements in slot order: homed layers by protection rank, then
/// the floating cell.
///
/// The one transcription of that numbering — [`LayerResidency::new`] derives the
/// same indices from the same plan, and a disagreement would have every view
/// pointing one layer off.
fn slot_table(plan: &ZonePlan) -> Vec<LayerPlacement> {
    let mut homed: Vec<LayerPlacement> = plan.homes.iter().flatten().copied().collect();
    homed.sort_by_key(|p| std::cmp::Reverse(p.base));
    homed.extend(plan.floating);
    homed
}

/// The pack path for a checkpoint, beside it rather than in the workspace.
///
/// Same placement rule as the expert pack: several workspaces on one machine
/// share a checkpoint, and rebuilding a 15 GiB pack per workspace is the cost
/// that placement avoids.
///
/// **`narrow` is in the name because it is in the bytes.** The streaming
/// narrowing schedule is chosen from the card's total VRAM, and it changes the
/// twin dtype — and so the record length — of `ffn_down`, `attn_q` and
/// `attn_qkv`. Two cards of different size sharing one checkpoint directory is
/// the case the placement rule above exists for, so without this the 16 GB box
/// and the 72 GB box each find the other's pack, `check_geometry` rejects it on
/// the per-projection dtype, and both rebuild a multi-GiB pack on every start,
/// forever. Named apart, the two packs simply coexist.
pub fn pack_path_for(
    gguf: &std::path::Path,
    int8mode: Int8Mode,
    narrow: Option<usize>,
) -> std::path::PathBuf {
    match narrow {
        Some(n) => gguf.with_extension(format!("layers.{}.n{n}.pack", int8mode as u32)),
        None => gguf.with_extension(format!("layers.{}.pack", int8mode as u32)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pack_sits_beside_the_checkpoint() {
        let p = pack_path_for(
            std::path::Path::new("/models/Qwen3.8-27B-Q4_K_M.gguf"),
            Int8Mode::Performance,
            None,
        );
        assert_eq!(p.parent(), Some(std::path::Path::new("/models")));
        assert!(p.to_string_lossy().ends_with(".pack"));
    }

    #[test]
    fn the_numeric_mode_is_part_of_the_pack_name() {
        // Two modes target different KO twins, so one checkpoint has two packs
        // and neither may be mistaken for the other.
        let gguf = std::path::Path::new("/models/m.gguf");
        assert_ne!(
            pack_path_for(gguf, Int8Mode::Performance, None),
            pack_path_for(gguf, Int8Mode::Precision, None)
        );
    }

    #[test]
    fn the_narrowing_schedule_is_part_of_the_pack_name() {
        // A narrowed build writes different record lengths, so a card that
        // narrows and a card that does not must not contend for one file —
        // otherwise each rejects the other's geometry and rebuilds forever.
        let gguf = std::path::Path::new("/models/m.gguf");
        let wide = pack_path_for(gguf, Int8Mode::Performance, None);
        let narrow = pack_path_for(gguf, Int8Mode::Performance, Some(64));
        assert_ne!(wide, narrow);
        // And two different schedules are two different packs.
        assert_ne!(narrow, pack_path_for(gguf, Int8Mode::Performance, Some(48)));
        assert!(narrow.to_string_lossy().ends_with(".pack"));
    }

    // ── The stack, end to end, on the real device ────────────────────────
    //
    // Synthetic layers at a few hundred KB each rather than a checkpoint's
    // 240 MB, so the whole 64-layer sweep costs megabytes and seconds. What it
    // exercises is everything except the model loader: images, the pack's
    // write/read/checksum path, the warm tier's fill and promotion, slot
    // uploads on the copy stream, the per-layer fences, and the residency
    // policy driving all of it.
    //
    // The assertion is byte identity: after the wave reaches each layer, the
    // slot holds *that* layer's payload and not its predecessor's. A fence
    // that is joined too late, a slot handed to two layers, or an eviction of
    // something still in flight all show up here as the wrong bytes.

    use super::super::pack::{header_for, LayerPack, PackIdentity, PackWriter};
    use crate::models::layer_stream::descriptor::{
        layer_image, FfnForm, LayerTensor, MixKind, Projection,
    };
    use candle::cuda_backend::cudarc::driver::CudaSlice;
    use candle::quantized::GgmlDType;

    const TEST_LAYERS: usize = 64;
    const TEST_PINNED: usize = 2;
    /// Small enough to be quick, large enough to cross a sector.
    const PROJ_BYTES: usize = 8192;

    fn test_image(kind: MixKind) -> LayerImage {
        let p = |role, rows, cols| Projection {
            role,
            shape: [rows, cols],
            // The KO twin, not the source dtype — a slot holds repacked bytes.
            dtype: GgmlDType::Q4_KO,
            payload: PROJ_BYTES,
            extent: PROJ_BYTES,
        };
        let mut projs: Vec<Projection> = match kind {
            MixKind::DeltaNet => vec![
                p(LayerTensor::Wqkv, 10240, 5120),
                p(LayerTensor::Wz, 6144, 5120),
                p(LayerTensor::WOut, 5120, 6144),
            ],
            MixKind::Attention => vec![
                p(LayerTensor::Wq, 12288, 5120),
                p(LayerTensor::Wk, 1024, 5120),
                p(LayerTensor::Wv, 1024, 5120),
                p(LayerTensor::Wo, 5120, 6144),
            ],
        };
        projs.extend([
            p(LayerTensor::FfnGateUp, 34816, 5120),
            p(LayerTensor::FfnDown, 5120, 17408),
        ]);
        layer_image(kind, FfnForm::Fused, &projs).unwrap()
    }

    /// The lineage's 3:1 interleave: three DeltaNet layers then an attention
    /// one, so both kinds and both image widths are in the same run.
    fn test_images() -> Vec<LayerImage> {
        (0..TEST_LAYERS)
            .map(|i| {
                test_image(if i % 4 == 3 {
                    MixKind::Attention
                } else {
                    MixKind::DeltaNet
                })
            })
            .collect()
    }

    /// A payload unique to `(layer, projection)`, so a slot holding the wrong
    /// layer is visible in its first byte.
    ///
    /// Never zero: a freshly allocated slot reads as zeros, so a zero-valued
    /// tag would make "this layer's bytes arrived" and "nothing was ever
    /// written here" indistinguishable — which is exactly the failure this test
    /// exists to catch.
    fn payload(layer: usize, idx: usize, len: usize) -> Vec<u8> {
        let tag = 1 + ((layer * 7 + idx) % 254) as u8;
        vec![tag; len]
    }

    #[test]
    #[ignore = "needs a CUDA device; allocates a few hundred MB of VRAM and writes a \
                temporary pack. Run with: cargo test -p candle-transformers \
                --features cuda --lib layer_stream::cache::tests::a_wave_streams_every_layer \
                -- --ignored --nocapture"]
    fn a_wave_streams_every_layer_through_a_small_ring() {
        // Both regimes §7 says must be one code path: a zone far smaller than
        // the model, which streams most of it, and a zone that holds it, where
        // "it fits" is the degenerate case and nothing ever moves after the
        // first pass.
        let images = test_images();
        let whole: usize = images.iter().map(|i| i.total).sum();
        let tight = run_sweep(crate::models::layer_stream::slot_bytes_for_layers(&images) * 10);
        let roomy = run_sweep(whole);
        let (Some(tight), Some(roomy)) = (tight, roomy) else {
            return; // no device; run_sweep already said so
        };

        assert_eq!(tight.abandoned, 0, "an opportunistic load failed");
        assert!(
            tight.evictions > 0,
            "a ring of 10 over {TEST_LAYERS} layers must evict"
        );
        assert!(
            tight.warm_hits > 0,
            "the warm tier was never promoted from — its membership is unused"
        );

        // The model fits: after the first pass every layer is resident, so the
        // second pass is pure hits and moves no bytes at all.
        assert_eq!(
            roomy.evictions, 0,
            "a ring holding the whole model must never evict"
        );
        assert_eq!(
            roomy.warm_hits + roomy.cold_reads,
            TEST_LAYERS - TEST_PINNED,
            "a layer was loaded more than once despite fitting"
        );
        assert!(
            roomy.hits > tight.hits,
            "residency bought nothing: roomy {} vs tight {}",
            roomy.hits,
            tight.hits
        );
    }

    /// **Several cold reads in one plan must not overwrite each other's staging.**
    ///
    /// A cold read fills a pinned host buffer and then uploads out of it
    /// *asynchronously*, so the buffer is still under DMA when `issue` returns.
    /// With a single staging buffer the next cold read in the same plan refills
    /// it mid-flight and the earlier layer lands holding this one's bytes.
    ///
    /// The sweep above cannot see that: it synchronizes and verifies after every
    /// layer, which drains the copy stream between reads and serialises exactly
    /// the overlap the bug needs. So this one warms **nothing** — every read is
    /// cold — runs a whole pass with no intermediate synchronize, and checks the
    /// slots once at the end.
    ///
    /// It is a real bug this reproduces, not a hypothetical: the 27B ran clean
    /// for as long as its warm tier happened to cover every streamed layer, and
    /// began emitting garbage logits within three configs of the tier being
    /// sized honestly against host RAM.
    #[test]
    #[ignore = "needs a CUDA device; allocates a few hundred MB of VRAM and writes a \
                temporary pack. Run with: cargo test -p candle-transformers \
                --features cuda --lib \
                layer_stream::cache::tests::concurrent_cold_reads_keep_their_own_bytes \
                -- --ignored --nocapture"]
    fn concurrent_cold_reads_keep_their_own_bytes() {
        let Some(bad) = run_cold_only() else {
            return; // no device
        };
        assert_eq!(
            bad, 0,
            "{bad} slots held another layer's bytes after a pass of cold reads — \
             the staging ring is being reused before its upload has landed"
        );
    }

    /// One pass with an empty warm tier and no per-layer readback; answers with
    /// the number of resident slots holding the wrong layer's bytes.
    fn run_cold_only() -> Option<usize> {
        use candle::Device;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return None;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda yields a cuda device")
        };

        let images = test_images();
        let slot_bytes = crate::models::layer_stream::slot_bytes_for_layers(&images);
        let identity = PackIdentity {
            source_len: 4242,
            source_sum: 0x1234_5678,
            int8_mode: Int8Mode::Performance as u32,
            repack_fp: 0xFEED_FACE_CAFE_BEEF,
        };

        let dir = std::env::temp_dir().join(format!("candle_layer_cold_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.layers.pack");
        let header = header_for(&images, identity, TEST_PINNED, slot_bytes);
        let mut w = PackWriter::create(&path, header).unwrap();
        for (li, img) in images.iter().enumerate().skip(TEST_PINNED) {
            let bufs: Vec<Vec<u8>> = img
                .placements
                .iter()
                .enumerate()
                .map(|(i, p)| payload(li, i, p.bytes))
                .collect();
            let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
            w.write_layer(li, &refs).unwrap();
        }
        let path = w.finish().unwrap();
        let pack = LayerPack::open(&path, identity, &images, TEST_PINNED).unwrap();

        // One contiguous arena, as the span is: the plan lays layers down inside
        // it at their own sizes, so the addresses under test are the ones
        // production computes rather than a hand-built table.
        let (arena, base, plan) = arena_for(cuda, &images, slot_bytes * 12);

        // **Warm budget zero**, so every promotion is a pack read and the
        // staging ring is the only path bytes take.
        let mut cache = LayerCache::new(
            cuda,
            images.clone(),
            Int8Mode::Performance,
            pack,
            &plan,
            0,
            |view, _layer| Ok(view),
        )
        .unwrap();

        // One pass, no synchronize inside it: the copy stream stays deep, which
        // is the condition the bug needs.
        for li in 0..TEST_LAYERS {
            cache.ensure(li).unwrap();
            cache.prefetch().unwrap();
        }
        cuda.cuda_stream().synchronize().unwrap();

        let s = cache.stats();
        eprintln!(
            "[layer-stream] cold-only pass: warm {} cold {} evictions {}",
            s.warm_hits, s.cold_reads, s.evictions
        );
        assert!(
            s.cold_reads > STAGING_SLOTS,
            "the pass must issue more cold reads than the ring has buffers, or \
             a stale buffer can never be reused"
        );

        // Every slot still holding a layer must hold *that* layer's bytes.
        //
        // The pinned head is excluded because this fixture never places it: it
        // has no pack record by construction, so the loader uploads it by hand
        // and there is nothing here for a cold read to have got wrong.
        let mut wrong = 0usize;
        for layer in TEST_PINNED..TEST_LAYERS {
            if !cache.residency().residence(layer).is_readable() {
                continue;
            }
            let got = read_layer(cuda, &arena, base, &cache, &images, layer);
            for (i, p) in images[layer].placements.iter().enumerate() {
                let want = payload(layer, i, p.bytes);
                if got[p.offset..p.offset + p.bytes] != want[..] {
                    eprintln!("  L{layer} projection {i} holds foreign bytes");
                    wrong += 1;
                    break;
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
        Some(wrong)
    }

    /// A contiguous device arena and the [`ZonePlan`] laid out inside it.
    ///
    /// The plan's addresses are absolute, so the arena's own base is where `end`
    /// is taken from — exactly as the span's right edge serves in production.
    fn arena_for(
        cuda: &CudaDevice,
        images: &[LayerImage],
        budget: usize,
    ) -> (
        CudaSlice<u8>,
        u64,
        crate::models::layer_stream::zone::ZonePlan,
    ) {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        let whole: usize = images.iter().map(|i| i.total).sum();
        let budget = budget.min(whole);
        let arena = unsafe { cuda.alloc::<u8>(budget).unwrap() };
        let stream = cuda.cuda_stream();
        // The guard is dropped before the arena moves out; the address stays
        // valid because the caller holds the allocation for the whole test.
        let base = {
            let (p, _g) = arena.device_ptr(&stream);
            p
        };
        let plan = crate::models::layer_stream::zone::plan_zone(
            images,
            TEST_PINNED,
            base + budget as u64,
            budget,
        )
        .expect("the fixture budget must clear the model's floor");
        (arena, base, plan)
    }

    /// A layer's bytes, read back from wherever the cache put it.
    fn read_layer<T, A: SlotAssembler<T>>(
        cuda: &CudaDevice,
        arena: &CudaSlice<u8>,
        base: u64,
        cache: &LayerCache<T, A>,
        images: &[LayerImage],
        layer: usize,
    ) -> Vec<u8> {
        let at = cache.slot_base_of(layer).unwrap();
        let off = (at - base) as usize;
        let n = images[layer].total;
        cuda.cuda_stream()
            .memcpy_dtov(&arena.slice(off..off + n))
            .unwrap()
    }

    /// Two forwards over a zone of `budget` bytes, checking byte identity at
    /// every layer. `None` when there is no device.
    fn run_sweep(budget: usize) -> Option<LayerCacheStats> {
        use candle::Device;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return None;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda yields a cuda device")
        };

        let images = test_images();
        let slot_bytes = crate::models::layer_stream::slot_bytes_for_layers(&images);
        let identity = PackIdentity {
            source_len: 4242,
            source_sum: 0x1234_5678,
            int8_mode: Int8Mode::Performance as u32,
            repack_fp: 0xFEED_FACE_CAFE_BEEF,
        };

        // ── the cold tier ──
        let dir = std::env::temp_dir().join(format!(
            "candle_layer_cache_{}_{budget}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synthetic.layers.pack");
        let header = header_for(&images, identity, TEST_PINNED, slot_bytes);
        let mut w = PackWriter::create(&path, header).unwrap();
        for (li, img) in images.iter().enumerate().skip(TEST_PINNED) {
            let bufs: Vec<Vec<u8>> = img
                .placements
                .iter()
                .enumerate()
                .map(|(i, p)| payload(li, i, p.bytes))
                .collect();
            let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
            w.write_layer(li, &refs).unwrap();
        }
        let path = w.finish().unwrap();
        let pack = LayerPack::open(&path, identity, &images, TEST_PINNED).unwrap();

        // ── the hot tier ──
        // Held by value and written through `&mut`: `CudaSlice::clone` is a
        // deep copy in cudarc, so cloning to get a writable handle would upload
        // into a temporary and leave the real ground untouched.
        let (mut arena, base, plan) = arena_for(cuda, &images, budget);

        // The identity assembler: this test is about bytes and residency, not
        // about any model's layer type, so a slot is presented as its own
        // views. That the cache can be exercised with no model at all is the
        // point of it being generic over the payload.
        let mut cache = LayerCache::new(
            cuda,
            images.clone(),
            // The fixture's images are KO twins, so the mode must be an int8 one
            // for `from_qtensor_view` to accept them.
            Int8Mode::Performance,
            pack,
            &plan,
            20,
            |view, _layer| Ok(view),
        )
        .unwrap();
        // The pinned head must be placed by hand: the pack holds no record for
        // it, which is the point of pinning.
        for (li, img) in images.iter().enumerate().take(TEST_PINNED) {
            let mut image = vec![0u8; img.total];
            for (i, p) in img.placements.iter().enumerate() {
                image[p.offset..p.offset + p.bytes].copy_from_slice(&payload(li, i, p.bytes));
            }
            let off = (cache.slot_base_of(li).unwrap() - base) as usize;
            cuda.cuda_stream()
                .memcpy_htod(&image, &mut arena.slice_mut(off..off + image.len()))
                .unwrap();
        }

        // ── two full forwards, so the wrap is exercised as well as the sweep ──
        let mut checked = 0usize;
        for pass in 0..2 {
            for (li, img) in images.iter().enumerate().take(TEST_LAYERS) {
                cache.ensure(li).unwrap();
                cache.prefetch().unwrap();

                // Read the ground back and check it is this layer's bytes.
                cuda.cuda_stream().synchronize().unwrap();
                let got = read_layer(cuda, &arena, base, &cache, &images, li);
                for (i, p) in img.placements.iter().enumerate() {
                    let want = payload(li, i, p.bytes);
                    assert_eq!(
                        &got[p.offset..p.offset + p.bytes],
                        want.as_slice(),
                        "pass {pass} L{li} projection {i} ({:?}) holds the wrong layer's bytes",
                        p.role
                    );
                }
                checked += 1;
            }
        }

        let s = cache.stats();
        eprintln!(
            "[layer-stream] {} of {TEST_LAYERS} homed: {checked} layer reads over 2 forwards",
            cache.residency().homed()
        );
        eprintln!(
            "[layer-stream]   hits {} warm {} cold {} joins {} evictions {} abandoned {}",
            s.hits, s.warm_hits, s.cold_reads, s.fence_joins, s.evictions, s.abandoned
        );
        assert_eq!(
            checked,
            TEST_LAYERS * 2,
            "a layer was skipped rather than checked"
        );

        std::fs::remove_dir_all(&dir).ok();
        Some(s)
    }
}

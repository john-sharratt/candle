//! The layer cache's operations: filling its tiers, moving the wave to a layer,
//! and issuing the transfers between them.
//!
//! Each layer's bytes move from the pack or the pinned warm tier into a zone slot
//! on a copy stream, and the wave joins a layer's fence only when it reaches that
//! layer, so the copies overlap the resident compute. A failed issue rolls the
//! layer's loading mark back before anything else, because a layer left looking
//! in flight would hand a forward the evicted tenant's weights. Warm start fills
//! every slot at load, so the first forward does not pay for the whole residency
//! inside a wave.

use super::*;

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
        pinned: usize,
        warm_budget_slots: usize,
        assemble: A,
    ) -> Result<Self>
    where
        A: SlotAssembler<T>,
    {
        let num_layers = images.len();
        let pinned = pinned.min(num_layers);
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
        // Bounded by what the device can still address as well as by the host
        // budget: under WDDM every pinned byte is charged against the non-local
        // budget (`warm_tier::ADDRESSABLE_MARGIN`); under TCC and Linux there is
        // none, and the driver's refusal bounds it. Read after the staging ring.
        let want = addressable_pinned_slots(
            gpu_addressable_room(device)?,
            stride,
            warm_slots_for(num_layers, pinned, warm_budget_slots),
        );
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
        cache.fill_pinned()?;
        Ok(cache)
    }

    /// Upload the pinned head into its slots and build the views over them.
    ///
    /// The pinned layers are the one set that never passes through
    /// [`Self::issue`] — they are resident for the life of the cache and never
    /// loaded on demand — so nothing else would ever fill their slots or build
    /// their views, and the first `ensure(0)` would find a resident layer whose
    /// slot has no matmuls over it. [`LayerResidency::new`] places them; this
    /// finishes the job, from the pack like every other layer, one record at a
    /// time through the staging ring and synchronously, at load.
    ///
    /// **The whole image is copied, zeros included.** The pack writer zeroes
    /// each record before placing its payloads, so the gaps the GGML kernels
    /// read past each row (`MATRIX_ROW_PADDING`) arrive as defined zeros rather
    /// than whatever the zone last held there.
    pub(super) fn fill_pinned(&mut self) -> Result<()> {
        let stride = self.pack.stride();
        for layer in 0..self.residency.pinned() {
            let Some(slot) = self.residency.residence(layer).slot() else {
                continue;
            };
            if self.staging.num_slots() == 0 {
                candle::bail!(
                    "layer cache: the staging ring is empty, so the pinned head has nowhere to \
                     land — the host could not pin a single {stride}-byte buffer"
                );
            }
            let base = self.slots[slot].base;
            let bytes = self.images[layer].total;
            let dest = self.staging.slot_mut(0, stride);
            self.pack.read_into(layer, dest)?;
            let src = self.staging.slot_ref(0, bytes);
            self.upload(src, base)?;
            // The ring slot is refilled by the next layer's read, which is a
            // host write: the upload out of it has to have landed first.
            self.copy.synchronize().map_err(candle::Error::wrap)?;
            // SAFETY: `base` names a slot the zone handed out, and the upload
            // above put this layer's image at these offsets.
            let view =
                unsafe { build_layer_view(&self.images[layer], &self.device, base, self.mode) }?;
            self.views[slot] = Some(Arc::new(self.assemble.assemble(view, layer)?));
        }
        Ok(())
    }

    /// Read the warm tier's members out of the pack, once.
    ///
    /// Verified, unlike the runtime miss path: this reads every member exactly
    /// once with idle cores, where a per-read checksum inside a forward would
    /// multiply its latency.
    pub(super) fn fill_warm(&mut self, members: &[usize]) -> Result<()> {
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

    pub(super) fn issue_all(&mut self, ops: &[LoadOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        // A wave recording on this thread launches its open segment first, so
        // the compute-stream event each copy waits on lands behind every GEMM
        // recorded so far — including the ones still reading an evicted slot.
        let device = self.device.clone();
        let _eager = device.pause_capture()?;
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
    pub(super) fn issue(&mut self, op: LoadOp) -> std::result::Result<(), Failed> {
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

    pub(super) fn issue_inner(&mut self, op: LoadOp, clobbered: &mut bool) -> Result<()> {
        self.residency.begin_load(op);
        if op.evicted.is_some() {
            self.stats.evictions += 1;
        }
        // The slot's previous tenant may still be under read by a GEMM the
        // device's stream has issued. Ordering the copy stream after that
        // stream's current point is what keeps the eviction from overwriting
        // bytes that are still being read. Recorded with no capture open on
        // this thread: an event recorded into a segment is a graph node, and
        // the copy stream waiting on it would be taken into the capture.
        let ready = {
            let _eager = self.device.pause_capture()?;
            self.device
                .cuda_stream()
                .record_event(None)
                .map_err(candle::Error::wrap)?
        };
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
    pub(super) fn upload(&self, src: &[u8], dst: u64) -> Result<()> {
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
    pub(super) fn join(&mut self, layer: usize) -> Result<()> {
        if let Some(fence) = self.fences[layer].take() {
            // Paused, the wait goes on the device's stream every recorded
            // segment launches onto, so the layer's GEMMs recorded after it
            // run behind the copy. A resident layer has no fence and never
            // ends a segment here.
            let _eager = self.device.pause_capture()?;
            self.device
                .cuda_stream()
                .wait(&fence)
                .map_err(candle::Error::wrap)?;
        }
        Ok(())
    }

    /// Adopt a new zone layout — the boundary move, in either direction.
    ///
    /// **Between forwards only**: the
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
pub(super) fn slot_table(plan: &ZonePlan) -> Vec<LayerPlacement> {
    let mut homed: Vec<LayerPlacement> = plan.homes.iter().flatten().copied().collect();
    homed.sort_by_key(|p| std::cmp::Reverse(p.base));
    homed.extend(plan.floating);
    homed
}

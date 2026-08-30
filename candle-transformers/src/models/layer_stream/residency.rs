//! Which layer is where, and what to load next.
//!
//! Pure bookkeeping over indices — no device, no bytes — so every rule here is
//! exercised without a GPU.
//!
//! # There is no replacement policy, because there is nothing to replace
//!
//! This module used to implement Bélády's rule: a forward walks `0 … N-1` and
//! then walks it again, so `distance(layer) = (layer − wave) mod N` is the exact
//! time to a layer's next use, and evicting the argmax is optimal replacement
//! rather than an approximation of it.
//!
//! It was optimal and it was the wrong question. On a cyclic reference string
//! the furthest-future resident is always the layer just executed, so each step
//! evicts the layer behind and loads the layer ahead: residency becomes a window
//! sliding one step per layer, and a window that slides `N` steps per forward has
//! thrown out everything it held by the time the forward ends. Measured on the
//! 27B at capacity 21 of 64: **22 hits over 11 forwards**, exactly the two pinned
//! layers each time, and raising capacity did not change it.
//!
//! The fix is not a better replacement rule. It is to stop treating a layer's
//! address as something to bid for:
//!
//! * a layer that fits **has a home**, keeps it for the life of the model, and is
//!   never a candidate for anything;
//! * a layer that does not fit passes through **one floating cell**, and which
//!   layers those are is fixed by [`super::order`] rather than chosen at runtime.
//!
//! What is left is not a cache. `plan` answers "is the next missing layer in
//! flight yet", and the only decisions in this file are made by the zone at carve
//! time.
//!
//! # The wave still matters, for scheduling rather than for eviction
//!
//! [`Self::distance`] survives because the *order* of fetches is still the sweep
//! order: the next layer to bring into the cell is the next missing one ahead of
//! the wave, and the run of resident layers before it is the time available to
//! hide the transfer. That is the quantity [`super::order`] maximises.
//!
//! # Filling homes is a second, unrelated kind of load
//!
//! When the boundary hands ground back, layers gain homes that hold nothing yet.
//! Those loads do not contend for the cell — each goes to its own address — so
//! any number may be in flight at once, and they are issued nearest-to-the-wave
//! first so the forward in progress benefits soonest. A home, once filled, is
//! never emptied except by a retraction.

use super::zone::ZonePlan;

/// How many upcoming layers must be in flight regardless of pressure.
///
/// One, and it cannot usefully be more: there is a single floating cell, and the
/// layer occupying it is the layer the wave is standing on. A deeper commitment
/// would need somewhere to put it.
///
/// It survives as a named constant because the cache still distinguishes a load
/// the forward will stall on from one it may abandon, and because the pack build
/// and the boundary both price "the working window" and must agree on its size.
pub const COMMITTED_DEPTH: usize = 1;

/// Where a layer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residence {
    /// In a slot and readable now.
    Resident(usize),
    /// A slot is assigned and a transfer is in flight; the fence must be joined
    /// before the layer is read.
    Loading(usize),
    /// Nowhere on the device.
    Absent,
}

impl Residence {
    /// The slot backing this layer, resident or in flight.
    pub fn slot(self) -> Option<usize> {
        match self {
            Self::Resident(s) | Self::Loading(s) => Some(s),
            Self::Absent => None,
        }
    }

    /// Whether a read may proceed without joining a fence.
    pub fn is_readable(self) -> bool {
        matches!(self, Self::Resident(_))
    }
}

/// One transfer the caller should issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOp {
    /// The layer to bring in.
    pub layer: usize,
    /// The slot to bring it into.
    pub slot: usize,
    /// The layer displaced, if the slot was not free. Its bytes are simply
    /// dropped — the cold tier is authoritative, so an eviction moves nothing.
    pub evicted: Option<usize>,
    /// Whether the forward will stall on this transfer if it does not complete.
    ///
    /// True only for the layer the wave is standing on, which must land or the
    /// model cannot proceed. Everything else — the cell's lookahead, a home being
    /// backfilled after growth — is free to be abandoned and retried, because the
    /// wave will ask again and the blocking path in `ensure` will cover it.
    pub committed: bool,
}

/// The planner's working buffer, owned by whoever plans repeatedly.
#[derive(Debug, Default)]
pub struct PlanScratch {
    order: Vec<(usize, usize)>,
}

/// Slot assignment and the load plan for a dense layer stack.
#[derive(Debug, Clone)]
pub struct LayerResidency {
    /// Trunk depth.
    num_layers: usize,
    /// Leading layers with no record in the cold tier. They cannot stream, so
    /// they always have a home and are never given up.
    pinned: usize,
    /// `layer → its permanent slot`, or `None` when the layer streams.
    home: Vec<Option<usize>>,
    /// The slot streamed layers pass through, absent when none do.
    cell: Option<usize>,
    /// `layer → residence`.
    where_is: Vec<Residence>,
    /// Streaming layers in eviction order — the order they were given up in and
    /// the order a growth takes them back.
    missing: Vec<usize>,
    /// The layer the wave is executing, or about to.
    wave: usize,
}

impl LayerResidency {
    /// The residency a [`ZonePlan`] describes, holding nothing yet.
    ///
    /// Slots are numbered by the plan's protection order — slot 0 is the
    /// rightmost ground, beside the dense block — with the floating cell last.
    /// Nothing is marked resident: even a homed layer has to be read in, which
    /// is what `warm_start` does before the first forward.
    pub fn new(plan: &ZonePlan, pinned: usize) -> Self {
        let num_layers = plan.homes.len();
        let mut home = vec![None; num_layers];
        let mut slot = 0usize;
        // Slot indices follow the protection order, which is the order the plan
        // laid the addresses down in — so slot index and address rank agree and
        // the boundary can talk about either.
        let mut homed: Vec<(usize, u64)> = plan
            .homes
            .iter()
            .enumerate()
            .filter_map(|(l, h)| h.map(|h| (l, h.base)))
            .collect();
        homed.sort_by_key(|&(_, base)| std::cmp::Reverse(base));
        for (l, _) in homed {
            home[l] = Some(slot);
            slot += 1;
        }
        let cell = plan.floating.map(|_| slot);
        let pinned = pinned.min(num_layers);
        let mut where_is = vec![Residence::Absent; num_layers];
        // **The pinned head starts resident.** It has no record in any tier, so
        // it can never be loaded through `issue` and a plan that named it would
        // have nowhere to read it from. Its bytes come straight from the
        // checkpoint, written by the loader into the addresses this plan just
        // fixed — so by the time anything reads this, they are there.
        for (layer, slot) in home.iter().enumerate().take(pinned) {
            if let Some(s) = slot {
                where_is[layer] = Residence::Resident(*s);
            }
        }
        Self {
            num_layers,
            pinned,
            home,
            cell,
            where_is,
            missing: plan.missing.clone(),
            wave: 0,
        }
    }

    /// The permanent slot `layer` was given, if it has one.
    pub fn home_of(&self, layer: usize) -> Option<usize> {
        self.home.get(layer).copied().flatten()
    }

    /// Slots the zone hands out: one per homed layer, plus the cell.
    pub fn capacity(&self) -> usize {
        self.home.iter().filter(|h| h.is_some()).count() + usize::from(self.cell.is_some())
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn pinned(&self) -> usize {
        self.pinned
    }

    /// Layers with a permanent address — the residency the zone bought.
    pub fn homed(&self) -> usize {
        self.home.iter().filter(|h| h.is_some()).count()
    }

    /// Layers that cross PCIe on every forward.
    pub fn streaming(&self) -> usize {
        self.missing.len()
    }

    /// Streaming layers, in the order they were given up.
    pub fn missing(&self) -> &[usize] {
        &self.missing
    }

    /// Whether the zone holds the trunk, so nothing moves after load.
    pub fn is_whole(&self) -> bool {
        self.cell.is_none()
    }

    pub fn wave(&self) -> usize {
        self.wave
    }

    pub fn residence(&self, layer: usize) -> Residence {
        self.where_is
            .get(layer)
            .copied()
            .unwrap_or(Residence::Absent)
    }

    /// The layer a slot holds, resident or in flight.
    pub fn slot_layer(&self, slot: usize) -> Option<usize> {
        self.where_is.iter().position(|r| r.slot() == Some(slot))
    }

    /// Layers resident or in flight.
    pub fn live(&self) -> usize {
        self.where_is.iter().filter(|r| r.slot().is_some()).count()
    }

    /// Move the wave to `layer`. Every distance is measured from here.
    pub fn set_wave(&mut self, layer: usize) {
        self.wave = if self.num_layers == 0 {
            0
        } else {
            layer % self.num_layers
        };
    }

    /// Steps until `layer` is next read, from the wave's position.
    ///
    /// The layer under the wave is `0`; the layer just executed is `N-1`.
    pub fn distance(&self, layer: usize) -> usize {
        if self.num_layers == 0 {
            return 0;
        }
        (layer + self.num_layers - self.wave) % self.num_layers
    }

    /// Mark an in-flight load complete.
    pub fn finish_load(&mut self, layer: usize) {
        if let Some(Residence::Loading(slot)) = self.where_is.get(layer).copied() {
            self.where_is[layer] = Residence::Resident(slot);
        }
    }

    /// Abandon an in-flight load, releasing its slot.
    /// Undo a [`Self::begin_load`] whose transfer failed.
    ///
    /// `slot_intact` says whether the H2D was ever enqueued. It decides the fate
    /// of the **evicted tenant**, which `begin_load` marked `Absent` on the
    /// assumption the transfer would happen:
    ///
    /// - `true` — nothing moved, so the slot still holds the victim's image and
    ///   its assembled view is still correct. Forgetting it costs a full
    ///   re-stream (~200 MiB, synchronous inside the forward) of a layer that
    ///   never left the card, on every transient failure.
    /// - `false` — the copy is in flight over those bytes. The victim is
    ///   genuinely gone and must stay `Absent`, or a later read is handed one
    ///   layer's view over another's bytes.
    pub fn abandon_load(&mut self, op: LoadOp, slot_intact: bool) {
        if let Some(Residence::Loading(_)) = self.where_is.get(op.layer).copied() {
            self.where_is[op.layer] = Residence::Absent;
        }
        if slot_intact {
            if let Some(victim) = op.evicted {
                self.where_is[victim] = Residence::Resident(op.slot);
            }
        }
    }

    /// Apply a planned op: the slot is reserved and the transfer is in flight.
    pub fn begin_load(&mut self, op: LoadOp) {
        if let Some(victim) = op.evicted {
            self.where_is[victim] = Residence::Absent;
        }
        self.where_is[op.layer] = Residence::Loading(op.slot);
    }

    /// What to load next.
    pub fn plan(&self, committed: usize) -> Vec<LoadOp> {
        let mut scratch = PlanScratch::default();
        let mut ops = Vec::new();
        self.plan_into(committed, &mut scratch, &mut ops);
        ops
    }

    /// [`Self::plan`] over a caller-owned buffer.
    ///
    /// Two independent kinds of work, in this order:
    ///
    /// 1. **Empty homes**, nearest the wave first. These appear only after the
    ///    boundary hands ground back; each has its own address, so they do not
    ///    contend and all of them may be in flight together.
    /// 2. **The cell's next tenant** — the next missing layer strictly ahead of
    ///    the wave — but only when the cell is free. It is not free while it
    ///    holds the layer the wave is standing on, which is the whole reason the
    ///    gap between missing layers is the time available to hide a transfer.
    ///
    /// `committed` is accepted for symmetry with what the cache passes and is
    /// bounded by the one slot that can stall a forward; see [`COMMITTED_DEPTH`].
    pub fn plan_into(&self, committed: usize, scratch: &mut PlanScratch, ops: &mut Vec<LoadOp>) {
        ops.clear();
        if self.num_layers == 0 {
            return;
        }

        // ── 1. Homes that hold nothing ──
        let order = &mut scratch.order;
        order.clear();
        for (layer, h) in self.home.iter().enumerate() {
            let Some(slot) = *h else { continue };
            if self.where_is[layer].slot().is_none() {
                order.push((self.distance(layer), slot));
            }
        }
        order.sort_unstable();
        for &(dist, slot) in order.iter() {
            let layer = self.wave_plus(dist);
            ops.push(LoadOp {
                layer,
                slot,
                evicted: None,
                committed: dist == 0 && committed > 0,
            });
        }

        // ── 2. The cell ──
        let Some(cell) = self.cell else { return };
        let tenant = self.slot_layer(cell);
        // Busy while the wave is standing on its tenant: the compute reading
        // those bytes has been issued but the view must stay valid until the
        // wave moves on.
        if tenant.is_some_and(|t| self.distance(t) == 0) {
            return;
        }
        let Some(next) = self.next_missing() else {
            return;
        };
        if self.where_is[next].slot().is_some() {
            return;
        }
        ops.push(LoadOp {
            layer: next,
            slot: cell,
            evicted: tenant,
            committed: self.distance(next) == 0 && committed > 0,
        });
    }

    /// The missing layer the wave reaches next, wrapping past the end of the
    /// sweep.
    ///
    /// The wrap is not an edge case to tidy away — it is where the lead time for
    /// the *first* missing layer of the next forward comes from. Without it the
    /// sweep would end with the cell idle and begin by stalling.
    fn next_missing(&self) -> Option<usize> {
        (0..self.num_layers)
            .map(|d| self.wave_plus(d))
            .find(|&l| self.home[l].is_none())
    }

    fn wave_plus(&self, d: usize) -> usize {
        (self.wave + d) % self.num_layers
    }

    /// Adopt a new zone layout, returning the layers that lost their home.
    ///
    /// The boundary move. Because the resident set is a prefix of the protection
    /// order at every budget, a retraction only ever drops from the end of that
    /// prefix and a growth only ever appends to it — so a layer that keeps its
    /// home keeps its **address**, and nothing is copied, no view is rebuilt, and
    /// no layer is re-fetched for having moved.
    ///
    /// Layers that gain a home gain an empty one; the next plan fills them.
    pub fn reshape(&mut self, plan: &ZonePlan) -> Vec<usize> {
        let next = Self::new(plan, self.pinned);
        let mut dropped = Vec::new();
        for layer in 0..self.num_layers {
            let was = self.home[layer];
            let now = next.home[layer];
            match (was, now) {
                // Kept its home, and the same one: residence is untouched.
                (Some(a), Some(b)) if a == b => {}
                // Lost its home, or was handed a different one. Either way what
                // it held is no longer where it is, so it starts again.
                //
                // **This arm is also what evicts the cell's tenant.** A layer in
                // the cell is homeless before and after — `(None, None)` — which
                // lands here, and it must: the cell sits at the zone's frontier,
                // so its address is precisely what a retraction moves. Narrowing
                // this arm to spare layers whose home is unchanged would have to
                // keep excluding `(None, None)`, or a streaming layer would keep
                // a residence pointing at ground the KV side now owns.
                _ => {
                    if self.where_is[layer].slot().is_some() {
                        dropped.push(layer);
                    }
                    self.where_is[layer] = Residence::Absent;
                }
            }
        }
        self.home = next.home;
        self.cell = next.cell;
        self.missing = next.missing;
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::layer_stream::descriptor::{
        layer_image, FfnForm, LayerImage, LayerTensor, MixKind, Projection, PROJECTION_ALIGN,
    };
    use crate::models::layer_stream::order::{eviction_order, protection_order};
    use crate::models::layer_stream::zone::plan_zone;
    use candle::quantized::GgmlDType;

    const N: usize = 16;
    const PINNED: usize = 2;
    const END: u64 = 1 << 40;

    fn image(units: usize) -> LayerImage {
        let roles = [
            LayerTensor::Wqkv,
            LayerTensor::Wz,
            LayerTensor::WOut,
            LayerTensor::FfnGateUp,
            LayerTensor::FfnDown,
        ];
        let rest = units - (roles.len() - 1);
        let p: Vec<Projection> = roles
            .iter()
            .enumerate()
            .map(|(i, &role)| Projection {
                role,
                shape: [32, 128],
                dtype: GgmlDType::Q4_KO,
                payload: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
                extent: PROJECTION_ALIGN * if i == 0 { rest } else { 1 },
            })
            .collect();
        layer_image(MixKind::DeltaNet, FfnForm::Fused, &p).unwrap()
    }

    fn model() -> Vec<LayerImage> {
        (0..N).map(|_| image(10)).collect()
    }

    /// A residency holding `resident` layers, warmed so every home is filled.
    fn res(resident: usize) -> (Vec<LayerImage>, LayerResidency) {
        let m = model();
        let per = m[0].total;
        let budget = if resident >= N {
            per * N
        } else {
            per * (resident + 1)
        };
        let plan = plan_zone(&m, PINNED, END, budget).unwrap();
        let mut r = LayerResidency::new(&plan, PINNED);
        warm(&mut r);
        (m, r)
    }

    /// Run plans until nothing more is placed, as `warm_start` does.
    fn warm(r: &mut LayerResidency) {
        loop {
            let ops = r.plan(COMMITTED_DEPTH);
            if ops.is_empty() {
                break;
            }
            for op in ops {
                r.begin_load(op);
                r.finish_load(op.layer);
            }
        }
    }

    /// Sweep one forward, counting the layers that had to be fetched and the
    /// ones that were not ready when the wave arrived.
    fn sweep(r: &mut LayerResidency) -> (usize, usize) {
        let (mut fetched, mut stalled) = (0, 0);
        for l in 0..N {
            r.set_wave(l);
            if !r.residence(l).is_readable() {
                stalled += 1;
                // The blocking path: load it now.
                let op = r
                    .plan(COMMITTED_DEPTH)
                    .into_iter()
                    .find(|o| o.layer == l)
                    .expect("the wave's layer must be plannable");
                r.begin_load(op);
                fetched += 1;
            }
            r.finish_load(l);
            // Compute is issued; prefetch overlaps it.
            for op in r.plan(COMMITTED_DEPTH) {
                r.begin_load(op);
                r.finish_load(op.layer);
                fetched += 1;
            }
        }
        (fetched, stalled)
    }

    /// **A zone that holds the trunk moves nothing, ever.** No cell is
    /// allocated, no plan is produced, and a sweep issues not one transfer —
    /// which is the property the whole two-tier split exists to make reachable.
    #[test]
    fn a_whole_zone_streams_nothing() {
        let (_, mut r) = res(N);
        assert!(r.is_whole());
        assert_eq!(r.homed(), N);
        assert_eq!(r.streaming(), 0);
        assert_eq!(r.capacity(), N, "no cell is allocated when none is needed");
        let (fetched, stalled) = sweep(&mut r);
        assert_eq!((fetched, stalled), (0, 0));
    }

    /// Every layer is readable when the wave reaches it, at every capacity the
    /// zone can take — the correctness floor, independent of how fast it is.
    #[test]
    fn every_layer_is_readable_when_the_wave_arrives() {
        for resident in PINNED..=N {
            let (_, mut r) = res(resident);
            for _ in 0..3 {
                for l in 0..N {
                    r.set_wave(l);
                    if !r.residence(l).is_readable() {
                        let op = r
                            .plan(COMMITTED_DEPTH)
                            .into_iter()
                            .find(|o| o.layer == l)
                            .unwrap_or_else(|| panic!("resident={resident}: L{l} unplannable"));
                        assert!(op.committed, "the wave's own layer must be committed");
                        r.begin_load(op);
                    }
                    r.finish_load(l);
                    assert!(r.residence(l).is_readable(), "resident={resident} L{l}");
                    for op in r.plan(COMMITTED_DEPTH) {
                        r.begin_load(op);
                        r.finish_load(op.layer);
                    }
                }
            }
        }
    }

    /// Transfers per forward are `N − homed`, and not one byte more.
    ///
    /// This is the number the old sliding window could not hold: it re-fetched
    /// everything outside the pinned head on every forward regardless of
    /// capacity. Here a homed layer is fetched once, at load, and never again.
    ///
    /// Measured over two sweeps rather than one, because the lookahead crosses
    /// the sweep boundary in both directions: the warm-up leaves one missing
    /// layer already in the cell, and a sweep that ends on a missing layer
    /// carries none into the next. Either sweep alone can be one off; the pair
    /// cannot, and the pair is what a rate is computed from.
    #[test]
    fn a_forward_moves_exactly_the_layers_with_no_home() {
        for resident in [PINNED, PINNED + 3, N - 1] {
            let (_, mut r) = res(resident);
            let homed = r.homed();
            let (first, _) = sweep(&mut r);
            let (steady, _) = sweep(&mut r);
            assert_eq!(steady, N - homed, "homed={homed}");
            // The first sweep is one cheaper: the warm-up left a missing layer
            // already in the cell. No sweep may ever exceed the count, which is
            // the statement that nothing is churned in and out.
            assert!(
                first <= N - homed,
                "homed={homed}: first sweep moved {first}"
            );
        }
    }

    /// The pinned head always has a home and never streams, at any capacity.
    #[test]
    fn the_pinned_head_never_streams() {
        for resident in PINNED..=N {
            let (_, r) = res(resident);
            for l in 0..PINNED {
                assert!(r.residence(l).is_readable(), "resident={resident} L{l}");
                assert!(!r.missing().contains(&l));
            }
        }
    }

    /// The cell is loaded **ahead** of the wave, not on arrival — the lookahead
    /// that turns the gap between missing layers into hidden latency.
    #[test]
    fn the_cell_runs_ahead_of_the_wave() {
        let (_, mut r) = res(N - 4);
        warm(&mut r);
        // Stand on a layer, issue its compute, and prefetch.
        let missing: Vec<usize> = r.missing().to_vec();
        assert!(!missing.is_empty());
        for &m in &missing {
            // Park the wave a couple of layers before a missing one.
            let before = (m + N - 2) % N;
            r.set_wave(before);
            for op in r.plan(COMMITTED_DEPTH) {
                r.begin_load(op);
                r.finish_load(op.layer);
            }
            r.set_wave(before);
            let ops = r.plan(COMMITTED_DEPTH);
            assert!(
                ops.is_empty() || ops.iter().all(|o| o.layer != before),
                "the wave's own layer should already be in hand"
            );
        }
    }

    /// The cell is not overwritten while the wave is standing on its tenant.
    #[test]
    fn the_cell_is_not_reused_under_the_wave() {
        let (_, mut r) = res(N - 4);
        let m = r.missing()[0];
        r.set_wave(m);
        if !r.residence(m).is_readable() {
            for op in r.plan(COMMITTED_DEPTH) {
                r.begin_load(op);
                r.finish_load(op.layer);
            }
        }
        r.set_wave(m);
        assert!(r.residence(m).is_readable());
        let ops = r.plan(COMMITTED_DEPTH);
        assert!(
            ops.iter().all(|o| o.evicted != Some(m)),
            "planned over the layer under the wave: {ops:?}"
        );
    }

    /// **Growth takes back the last layers given up, and nothing that stayed
    /// moves.** A layer keeping its home keeps its address, so it is not
    /// re-fetched — the property that makes a boundary move cheap.
    #[test]
    fn growth_keeps_addresses_and_backfills_the_new_homes() {
        let m = model();
        let per = m[0].total;
        let small = plan_zone(&m, PINNED, END, per * 6).unwrap();
        let big = plan_zone(&m, PINNED, END, per * 10).unwrap();
        let mut r = LayerResidency::new(&small, PINNED);
        warm(&mut r);
        let before: Vec<Residence> = (0..N).map(|l| r.residence(l)).collect();
        let was_missing = r.missing().to_vec();

        let dropped = r.reshape(&big);
        // Nothing that had a home in the small plan lost it.
        for (l, was) in before.iter().enumerate() {
            if small.homes[l].is_some() {
                assert!(big.homes[l].is_some(), "L{l} was displaced by growth");
                assert_eq!(*was, r.residence(l), "L{l} moved");
            }
        }
        // Only the cell's tenant is dropped, because only the cell's address
        // moved.
        assert!(dropped.len() <= 1, "growth dropped {dropped:?}");
        // The layers that gained homes are the tail of what was missing — the
        // ones given up most recently.
        let gained: Vec<usize> = was_missing
            .iter()
            .copied()
            .filter(|&l| big.homes[l].is_some())
            .collect();
        let tail: Vec<usize> = was_missing[was_missing.len() - gained.len()..].to_vec();
        assert_eq!(gained, tail, "growth took layers back out of order");
        // And they are planned for immediately, so the gain is realised rather
        // than waiting for a fault.
        let ops = r.plan(COMMITTED_DEPTH);
        for g in &gained {
            assert!(ops.iter().any(|o| o.layer == *g), "L{g} was not backfilled");
        }
    }

    /// Retraction is the same statement backwards: it drops from the end of the
    /// protection order, and every survivor keeps its address.
    #[test]
    fn retraction_drops_the_least_protected_and_moves_no_survivor() {
        let m = model();
        let per = m[0].total;
        let big = plan_zone(&m, PINNED, END, per * 12).unwrap();
        let small = plan_zone(&m, PINNED, END, per * 6).unwrap();
        let mut r = LayerResidency::new(&big, PINNED);
        warm(&mut r);
        let before: Vec<Residence> = (0..N).map(|l| r.residence(l)).collect();

        r.reshape(&small);
        let evict = eviction_order(N, PINNED);
        let lost: Vec<usize> = (0..N)
            .filter(|&l| big.homes[l].is_some() && small.homes[l].is_none())
            .collect();
        // What was given up is a prefix of the eviction order beyond what was
        // already missing.
        for l in &lost {
            let rank = evict.iter().position(|x| x == l).unwrap();
            assert!(rank < r.streaming(), "L{l} given up out of order");
        }
        for (l, was) in before.iter().enumerate() {
            if small.homes[l].is_some() {
                assert_eq!(*was, r.residence(l), "survivor L{l} moved");
            }
        }
    }

    /// Slot indices follow the protection order, so slot 0 is the rightmost
    /// ground and the cell is last. The boundary relies on this to talk about a
    /// slot and an address interchangeably.
    #[test]
    fn slot_indices_follow_the_protection_order() {
        let m = model();
        let per = m[0].total;
        let plan = plan_zone(&m, PINNED, END, per * 8).unwrap();
        let r = LayerResidency::new(&plan, PINNED);
        let prot = protection_order(N, PINNED);
        let mut want = 0usize;
        for &l in &prot {
            if let Some(h) = plan.homes[l] {
                assert_eq!(r.home[l], Some(want), "L{l} at {:?}", h.base);
                want += 1;
            }
        }
        assert_eq!(r.cell, Some(want));
    }

    /// The lookahead wraps past the end of the sweep, so the first missing layer
    /// of the next forward is already in flight when this one ends.
    #[test]
    fn the_lookahead_wraps_around_the_end_of_the_sweep() {
        let (_, mut r) = res(N - 3);
        let missing = r.missing().to_vec();
        let last = *missing.iter().max().unwrap();
        let first = *missing.iter().min().unwrap();
        assert_ne!(last, first, "the fixture needs two missing layers");
        // Stand past the last missing layer of the sweep.
        r.set_wave(last);
        for op in r.plan(COMMITTED_DEPTH) {
            r.begin_load(op);
            r.finish_load(op.layer);
        }
        r.set_wave((last + 1) % N);
        let ops = r.plan(COMMITTED_DEPTH);
        assert!(
            ops.iter().any(|o| o.layer == first),
            "past the last miss, the next forward's first miss should load: {ops:?}"
        );
    }

    /// An abandoned load frees the cell rather than leaving it reserved for a
    /// transfer that will never land.
    #[test]
    fn an_abandoned_load_releases_the_cell() {
        let (_, mut r) = res(N - 3);
        let cell = r.cell.expect("this fixture streams");
        // The warm-up already put the next miss in the cell, so step past it to
        // reach the point where the *following* one is planned.
        let held = r.slot_layer(cell).expect("warm-up fills the cell");
        r.set_wave((held + 1) % N);
        let op = r
            .plan(COMMITTED_DEPTH)
            .into_iter()
            .find(|o| o.slot == cell)
            .expect("a missing layer should be in flight for the cell");
        let m = op.layer;
        r.begin_load(op);
        assert!(matches!(r.residence(m), Residence::Loading(_)));
        r.abandon_load(op, false);
        assert_eq!(r.residence(m), Residence::Absent);
        // And it is planned again rather than skipped.
        assert!(r.plan(COMMITTED_DEPTH).iter().any(|o| o.layer == m));
    }

    /// **A load abandoned before its transfer started gives the cell's tenant
    /// back.**
    ///
    /// `begin_load` marks the victim `Absent` on the assumption the copy will
    /// happen. When it never does — a failed event record, a pack read that
    /// errored — the slot still holds the victim's image and its view is still
    /// correct, so forgetting it costs a full re-stream (~200 MiB, synchronous
    /// inside the forward) of a layer that never left the card.
    #[test]
    fn an_abandoned_load_that_never_copied_restores_the_evicted_tenant() {
        let (_, mut r) = res(N - 3);
        let cell = r.cell.expect("this fixture streams");
        let held = r.slot_layer(cell).expect("warm-up fills the cell");
        r.set_wave((held + 1) % N);
        let op = r
            .plan(COMMITTED_DEPTH)
            .into_iter()
            .find(|o| o.slot == cell && o.evicted.is_some())
            .expect("the cell's next op evicts its tenant");
        let victim = op.evicted.expect("filtered on it");
        r.begin_load(op);
        assert_eq!(
            r.residence(victim),
            Residence::Absent,
            "begin_load clears the victim, as it must while the copy is planned"
        );

        r.abandon_load(op, true);
        assert_eq!(
            r.residence(victim),
            Residence::Resident(op.slot),
            "nothing was copied, so the victim's bytes are still in the slot"
        );
        assert_eq!(r.residence(op.layer), Residence::Absent);

        // And with the copy in flight it must stay gone — those bytes are being
        // overwritten whatever else failed.
        r.begin_load(op);
        r.abandon_load(op, false);
        assert_eq!(r.residence(victim), Residence::Absent);
    }
}

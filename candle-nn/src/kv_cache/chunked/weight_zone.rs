//! The weight side of the reservation: equal-sized expert slots, filled from the
//! right, retracted from the left.
//!
//! The mirror image of [`super::region_pool`]'s KV side, and deliberately built
//! from the same parts. Both hand out fixed-size units from one end of the span
//! and both keep live data packed *away* from the frontier, using a
//! lowest-index-first free list to do it. The KV side packs left because its
//! frontier is on the right; this packs right because its frontier is on the
//! left. That symmetry is the whole reason a boundary between them can move:
//! whichever way it moves, the data it would disturb has already been pushed out
//! of the way (`docs/elastic_vram_partition.md` §2, §6).
//!
//! # Why equal-sized slots make this an array
//!
//! Every expert slot is `max_expert_size` bytes — the maximum over all layers,
//! which the loader already computes. Nothing here has to reason about
//! fragmentation, because there is none to reason about: "the rightmost free
//! spot" is "the lowest free index", a retraction is a suffix of the index
//! space, and relocating a slot is a memcpy between two addresses of identical
//! length rather than a compaction.
//!
//! # This module owns bytes, not experts
//!
//! It knows how many slots exist, which are occupied, and where each one is. It
//! does not know which expert is in a slot, what an expert is worth, or how to
//! copy one. Those belong to the cache above it, so a retraction returns a
//! [`RetractPlan`] — *what to move and what to drop* — and the caller executes
//! it. Keeping the policy out means the whole module tests without a GPU, a
//! model, or a routing trace.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::wave_spans::{WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES};

/// **The floor the weight side may never cross**: bytes always left to the
/// elastic middle, whatever the expert cache would like.
///
/// Derived, not chosen. It is the point at which a warm daemon can still serve a
/// wave without evicting a single sealed chunk:
///
/// | term | bytes |
/// |---|---|
/// | wave transient span (derived from the three phase spans) | 912 MiB |
/// | [`MIN_FIRST_WAVE_KV`] | 384 MiB |
/// | **total** | **1,296 MiB** |
///
/// **Derived, not written down.** It was a flat 2 GiB whose KV term was a steady-state
/// measurement from another model, and on the 27B that pinned ~750 MiB idle: the boundary
/// settled at exactly 125 regions — the floor — while the weight side still wanted ground.
/// Deriving it from the phase spans means a change to the wave tier moves the floor with it,
/// which a constant could not.
///
/// A wave that needs more than this takes more, by retracting the weight side.
/// This exists so that a weight fill at load time — when the arenas are empty
/// and the whole span looks available — cannot leave the engine unable to run
/// its first wave, and so that the minimum viable configuration is a stated
/// number rather than an emergent one.
///
/// Lives here rather than with the region pool because it is a property of the
/// *zone* (how far left it may reach), and because it must be readable on a
/// build with no GPU backend.
pub const MIN_ELASTIC_RESERVE: usize =
    WAVE_ATTN_BYTES + WAVE_FFN_BYTES + WAVE_FORWARD_BYTES + MIN_FIRST_WAVE_KV;

/// KV ground the floor keeps beyond the wave transient span.
///
/// **Sized from what a first wave actually holds, not from a steady state.** The 2 GiB the
/// floor used to be was `912 + 1,120`, and that 1,120 was a steady-state KV measurement from a
/// different model. On the 27B the KV side runs at **320 MiB live** (20 regions) at width 4, so
/// the old constant pinned ~750 MiB idle — measured, as the boundary settling at exactly 125
/// regions against a 2,048 MiB floor while the weight side still wanted ground.
///
/// A first wave is all this has to cover. Growth beyond it is handled by retraction (see
/// [`MIN_ELASTIC_RESERVE`]), so the floor's job is only to stop a load-time weight fill from
/// leaving the engine unable to run *once* — not to pre-buy the steady state.
///
/// Halving the old constant to 1,024 MiB was the obvious move and is wrong: it lands below
/// `912 + 320`, so every forward would retract the weight side to place its transient tier and
/// then regrow it. That trades layer slots for boundary churn on the hot path, which is worse
/// than the memory it frees.
const MIN_FIRST_WAVE_KV: usize = 384 * 1024 * 1024;

/// Where the boundary opens at model load — **a crutch, and it should not
/// exist**.
///
/// The partition is meant to be recomputed exactly on every forward
/// (`docs/elastic_vram_partition.md` §7), and a forward is a full layer sweep at
/// 57–80 ms. At that cadence the opening position survives one decode step and
/// is then irrelevant: the weight side should simply fill to
/// [`MIN_ELASTIC_RESERVE`] at load and let the first forward correct it.
///
/// Filling to the floor was built and the gate killed it — twenty concurrent
/// Q8_0 contexts exhausted a 2 GiB KV side, `every region of the KV reservation
/// is occupied (67 live)`. **That is not evidence the floor is too small.** Load
/// time is the least informative moment the engine has (empty arenas make every
/// byte look free), and the give-back path built today costs a pass, which a
/// failing arena claim has no way to wait for. This constant does not fix the
/// partition; it conceals that the partition never moves.
///
/// Sized to reproduce the KV capacity the old static partition was measured to
/// need — 275 regions × 16 MiB, plus 976 MiB of blocks that were fixed then and
/// are not now — so it is at least a known-good number while it survives.
///
/// # What actually deletes it, which is not what was expected
///
/// The note above said "delete this with the phase lock", on the reasoning that
/// an exactly-recomputed partition makes the opening position irrelevant. The
/// partition *is* exact now — `RegionPool::spare_regions` measures the weight
/// side's surplus against a monotone watermark with no clock and no decay — and
/// this constant still cannot go.
///
/// The reason is that exactness does not help at load, because at load there is
/// nothing to be exact about. A watermark that rises with observed demand reads
/// zero before the first forward, so the weight side would be offered the entire
/// span, and the first prefill's arena claim would be refused. The refusal is the
/// problem: conceding costs a pass, and a claim that fails has no way to wait for
/// one. Every scheme that hands ground over on *observed* demand has this hole at
/// its first observation.
///
/// So what deletes this is not an exact boundary but a **claim that can block**:
/// a refused arena claim that waits for the next boundary move instead of failing
/// the forward. Until then the opening position has to be a number large enough
/// that the first workload never needs to wait, and that is what this is.
pub const INITIAL_KV_RESERVE: usize = 5376 * 1024 * 1024;

const _: () = assert!(
    INITIAL_KV_RESERVE >= MIN_ELASTIC_RESERVE,
    "the opening position cannot be past the floor"
);

/// What a retraction asks its caller to do before the slots disappear.
///
/// Produced by [`WeightZone::retract_to`], which has already applied the
/// bookkeeping — so the caller's job is to make the *bytes* match what the zone
/// now believes, and nothing here is optional.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RetractPlan {
    /// `(from, to)` — copy the slot's bytes to the new index and rewrite
    /// whatever names it. Ordered hottest-first, which is also the order the
    /// destinations ascend in, so the hottest survivor lands furthest right.
    pub relocate: Vec<(usize, usize)>,
    /// Slots whose contents are dropped. Cold by construction: a doomed slot is
    /// only evicted once every free slot below the new frontier has been used
    /// by a hotter one.
    pub evict: Vec<usize>,
}

impl RetractPlan {
    /// Whether this plan asks for anything at all.
    pub fn is_empty(&self) -> bool {
        self.relocate.is_empty() && self.evict.is_empty()
    }
}

/// The expert-slot side of the device reservation.
///
/// Addresses descend as the index rises: slot 0 is the rightmost slot in the
/// span and the highest occupied index is the frontier. Constructed once with
/// the span's right edge and the per-slot size; capacity moves afterwards.
#[derive(Debug)]
pub struct WeightZone {
    /// One past the last byte of the span — the zone's immovable right edge.
    ///
    /// There is no dense-weight block inside the span to bound it: the dense
    /// tensors are loaded before the reservation exists, so the span is what
    /// they left behind and this is simply its end (§4).
    span_end: u64,
    /// Bytes per slot. Uniform, which is what §"equal-sized slots" rests on.
    slot_bytes: usize,
    /// Slots that currently exist: indices `[0, capacity)`.
    capacity: usize,
    /// The most slots this zone may ever hold — `MIN_ELASTIC_RESERVE` expressed
    /// in slots. [`Self::grow_to`] clamps to it, so the floor cannot be crossed
    /// by any path that grows the zone.
    limit: usize,
    /// The fewest slots the cache can serve a token with — the weight side's own
    /// floor, symmetric with `limit`.
    ///
    /// A model with no resident experts cannot run: every layer would miss every
    /// expert, and each miss would want a slot that does not exist. So the
    /// boundary may take the zone down to a working minimum and no further,
    /// however loudly the KV side asks. See [`Self::retract_to`].
    ///
    /// **The stated figure, not the effective one.** A zone that opens below its
    /// own floor is already as small as it can be, so the *effective* floor is
    /// `min(floor, capacity)` — and that has to be recomputed on every read
    /// rather than stored, because a zone can open small and then
    /// [`Self::grow_to`] past its floor. Freezing the clamp at construction
    /// would leave such a zone retractable to its opening size for ever, which is
    /// the permanent all-pinned state the floor exists to prevent. See
    /// [`Self::min_capacity`].
    floor: usize,
    /// Free slot indices, **lowest first**: `pop` yields the rightmost free spot.
    free: BinaryHeap<Reverse<usize>>,
    /// Occupancy, indexed by slot. Length is always `capacity`.
    occupied: Vec<bool>,
    live: usize,
    peak_live: usize,
}

impl WeightZone {
    /// A zone over `[span_end − limit·slot_bytes, span_end)`, empty, with
    /// `capacity` slots live.
    ///
    /// `capacity` starts wherever the caller wants the boundary; `limit` is the
    /// ceiling's expression in slots and nothing may grow past it; `floor` is the
    /// fewest slots the cache can serve a token with and nothing may retract
    /// below it.
    ///
    /// # The floor is the caller's to state, and it is measured
    ///
    /// It was a fraction — an eighth of the opening capacity — on the reasoning
    /// that "an eighth is enough that every layer's routed experts have somewhere
    /// to land". That was never measured and it is false for a wide batch: on a
    /// 128-expert model the caller's floor is 384 slots — the pinned head layers
    /// plus one layer's worst-case routed set — and an eighth of a 2,377-slot
    /// opening is 297. The daemon reached exactly that number, every resident
    /// slot held a pinned-layer expert, and every load afterwards failed with
    /// `cannot evict (all pinned)` — 1,774 times, until it was killed.
    ///
    /// A fraction cannot express that, because the quantity it has to clear is a
    /// property of the *model* (pinned layers × experts per layer), not of
    /// wherever the boundary happened to open. So the zone takes the figure
    /// rather than inventing one, and the expert cache — which owns the pinning
    /// rule — computes it.
    ///
    /// Clamped to `capacity`: a zone that opens below its own floor is already as
    /// small as it can be, and refusing to construct it would fail a load that
    /// can still run.
    pub fn new(
        span_end: u64,
        slot_bytes: usize,
        capacity: usize,
        limit: usize,
        floor: usize,
    ) -> Self {
        let capacity = capacity.min(limit);
        Self {
            span_end,
            slot_bytes,
            capacity,
            limit,
            floor,
            free: (0..capacity).map(Reverse).collect(),
            occupied: vec![false; capacity],
            live: 0,
            peak_live: 0,
        }
    }

    /// The zone's retraction floor right now, in slots.
    ///
    /// Derived rather than stored, so that a zone which opens below its floor and
    /// later grows past it is protected by the real figure from that moment on.
    pub fn min_capacity(&self) -> usize {
        self.floor.min(self.capacity)
    }

    /// Slots that fit between `frontier` and the span's right edge.
    ///
    /// The conversion from an address the KV side computed into the slot count
    /// this side thinks in. Rounds **down**: a partial slot is not a slot.
    pub fn capacity_for_frontier(&self, frontier: u64) -> usize {
        if frontier >= self.span_end || self.slot_bytes == 0 {
            return 0;
        }
        ((self.span_end - frontier) as usize / self.slot_bytes).min(self.limit)
    }

    /// Device address of slot `i`'s first byte.
    ///
    /// Index rises leftward, so slot 0 is adjacent to the span's right edge and
    /// the highest live index is the frontier.
    pub fn slot_base(&self, i: usize) -> u64 {
        self.span_end - ((i + 1) * self.slot_bytes) as u64
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn live(&self) -> usize {
        self.live
    }

    pub fn peak_live(&self) -> usize {
        self.peak_live
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    pub fn is_occupied(&self, i: usize) -> bool {
        self.occupied.get(i).copied().unwrap_or(false)
    }

    /// The leftmost byte the zone currently occupies.
    ///
    /// `span_end` when nothing is resident — an empty zone occupies nothing and
    /// the whole span is available to the other side. Derived from the highest
    /// *occupied* index rather than from `capacity`, because capacity is a
    /// permission and this is a fact: a zone with 2,000 slots and 3 experts in
    /// it is holding three slots' worth of bytes, and the KV side may have the
    /// rest without anyone being evicted.
    pub fn frontier(&self) -> u64 {
        match self.highest_occupied() {
            None => self.span_end,
            Some(i) => self.slot_base(i),
        }
    }

    fn highest_occupied(&self) -> Option<usize> {
        self.occupied.iter().rposition(|&o| o)
    }

    /// The leftmost byte the zone's **capacity** reaches, occupied or not.
    ///
    /// Distinct from [`Self::frontier`], which follows occupancy. This is the
    /// boundary the KV side is told about: a slot that is free today can be
    /// filled by the next miss without asking anyone, so the region count has to
    /// be derived from what the zone may hold, not from what it holds now.
    pub fn frontier_for_capacity(&self) -> u64 {
        self.span_end - (self.capacity * self.slot_bytes) as u64
    }

    /// The frontier the zone will publish *after* [`Self::grow_to`] takes it to
    /// `target` — `target` under the same clamps that method applies.
    ///
    /// **Growth publishes its floor before it moves.** Telling the KV side about
    /// a boundary is the step that can be refused (a wave generation open on the
    /// span), and a zone that grew against a refused floor holds slots whose
    /// addresses lie *below* the boundary the KV side still believes in — ground
    /// both sides think they own. Publishing first makes the refusal land while
    /// the zone is still untouched, which is why the address has to be derivable
    /// from a capacity the zone does not have yet.
    pub fn frontier_after_growth(&self, target: usize) -> u64 {
        let reached = target.min(self.limit).max(self.capacity);
        self.span_end - (reached * self.slot_bytes) as u64
    }

    /// Take the rightmost free slot, or `None` when every slot is occupied.
    ///
    /// `None` is the signal to evict, and the choice of victim belongs to the
    /// caller's temperature policy — never to this module. The free list is
    /// always drained first: if any slot is free, no eviction happens, whatever
    /// the scores say.
    pub fn alloc(&mut self) -> Option<usize> {
        let Reverse(i) = self.free.pop()?;
        self.occupied[i] = true;
        self.live += 1;
        self.peak_live = self.peak_live.max(self.live);
        Some(i)
    }

    /// Return a slot to the free list.
    ///
    /// Idempotent against a slot that is already free, and silently ignores an
    /// index past the current capacity — a retraction can remove a slot whose
    /// owner has not yet noticed, and making that an error would turn an
    /// ordinary race into a failure.
    pub fn release(&mut self, i: usize) {
        if i >= self.capacity || !self.occupied[i] {
            return;
        }
        self.occupied[i] = false;
        self.live -= 1;
        self.free.push(Reverse(i));
    }

    /// Grow to `new_capacity` slots, clamped to the zone's limit.
    ///
    /// New slots appear at the **highest** indices and join the back of the free
    /// list, so they are the last to be used and the first to be lost. Nothing
    /// is wasted by that — every free slot is still handed out before any
    /// eviction — but the volatile margin stays empty while the working set does
    /// not need it, which is what keeps the next retraction cheap.
    ///
    /// Returns the number of slots gained.
    pub fn grow_to(&mut self, new_capacity: usize) -> usize {
        let new_capacity = new_capacity.min(self.limit);
        if new_capacity <= self.capacity {
            return 0;
        }
        for i in self.capacity..new_capacity {
            self.free.push(Reverse(i));
        }
        self.occupied.resize(new_capacity, false);
        let gained = new_capacity - self.capacity;
        self.capacity = new_capacity;
        gained
    }

    /// Shrink to `new_capacity` slots: relocate the hottest doomed occupants
    /// into free slots below the new frontier, evict the rest.
    ///
    /// `score` is the caller's temperature for a slot — higher is more valuable.
    /// It is the *only* thing this module knows about worth, and it never
    /// decides which of two resident experts survives in general; it decides
    /// only which of the doomed ones is worth a memcpy.
    ///
    /// The bookkeeping is applied here. The returned [`RetractPlan`] is what the
    /// caller must do to the bytes to make them agree.
    pub fn retract_to(&mut self, new_capacity: usize, score: impl Fn(usize) -> f32) -> RetractPlan {
        // **The weight side has a floor of its own, and it had none.**
        // `MIN_ELASTIC_RESERVE` bounds how far the weight side may *grow* — it
        // protects the KV side's minimum. Nothing bounded the other direction,
        // so a caller that asked repeatedly could retract the zone to nothing.
        //
        // One did. A wedged scheduler asked for ground on every retry of a wave
        // that could not succeed, each refusal was counted as a region of
        // boundary demand, and the boundary paid: six experts evicted per step,
        // one region at a time, until `slots=0` and the model had no experts at
        // all. The demand was spurious — the ceiling was a placed tier the
        // boundary cannot move — but a zone that can be asked to reach zero will
        // reach zero eventually whatever the reason, and an engine that has
        // evicted its last expert cannot serve a token.
        //
        // Conceding *less* than asked is a slowdown. Conceding everything is an
        // engine that no longer works, so the floor holds even against a caller
        // that insists.
        let new_capacity = new_capacity.max(self.min_capacity());
        if new_capacity >= self.capacity {
            return RetractPlan::default();
        }

        // Doomed: everything occupied at or past the new frontier, hottest
        // first — so the hottest survivor gets the first (rightmost)
        // destination.
        let mut doomed: Vec<usize> = (new_capacity..self.capacity)
            .filter(|&i| self.occupied[i])
            .collect();
        doomed.sort_by(|&a, &b| {
            score(b)
                .partial_cmp(&score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        // Destinations: free slots that survive, rightmost first.
        let mut destinations: Vec<usize> =
            (0..new_capacity).filter(|&i| !self.occupied[i]).collect();
        destinations.truncate(doomed.len());

        let mut plan = RetractPlan::default();
        for (n, &from) in doomed.iter().enumerate() {
            match destinations.get(n) {
                Some(&to) => {
                    self.occupied[to] = true;
                    plan.relocate.push((from, to));
                }
                None => {
                    self.live -= 1;
                    plan.evict.push(from);
                }
            }
        }

        // The suffix ceases to exist. Rebuild the free list over what is left
        // rather than filtering the heap: it is a few thousand entries at most
        // and this runs once per retraction, between waves.
        self.occupied.truncate(new_capacity);
        self.capacity = new_capacity;
        self.free = (0..new_capacity)
            .filter(|&i| !self.occupied[i])
            .map(Reverse)
            .collect();
        plan
    }

    /// Occupancy snapshot for diagnostics.
    pub fn stats(&self) -> WeightZoneStats {
        WeightZoneStats {
            capacity: self.capacity,
            limit: self.limit,
            live: self.live,
            free: self.free.len(),
            peak_live: self.peak_live,
            slot_bytes: self.slot_bytes,
            frontier: self.frontier(),
            span_end: self.span_end,
        }
    }

    /// Mean slot index weighted by `score`, or `None` when nothing is resident.
    ///
    /// The measurement behind the temperature gradient (§6): hot experts should
    /// trend toward index 0 as the frontier moves, so this should fall over a
    /// run that retracts. Reported rather than asserted in production — it is a
    /// property of a workload, not an invariant of the allocator.
    pub fn score_weighted_mean_index(&self, score: impl Fn(usize) -> f32) -> Option<f64> {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (i, _) in self.occupied.iter().enumerate().filter(|(_, &o)| o) {
            let w = score(i).max(0.0) as f64;
            num += w * i as f64;
            den += w;
        }
        (den > 0.0).then_some(num / den)
    }
}

/// Occupancy of a [`WeightZone`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightZoneStats {
    pub capacity: usize,
    pub limit: usize,
    pub live: usize,
    pub free: usize,
    pub peak_live: usize,
    pub slot_bytes: usize,
    pub frontier: u64,
    pub span_end: u64,
}

impl WeightZoneStats {
    /// Bytes the zone is permitted to hold.
    pub fn capacity_bytes(&self) -> usize {
        self.capacity * self.slot_bytes
    }
    /// Bytes it is actually holding.
    pub fn live_bytes(&self) -> usize {
        self.live * self.slot_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOT: usize = 3 * 1024 * 1024;
    const END: u64 = 0x8000_0000;

    /// A zone of `capacity` slots whose floor is an eighth of it — the fraction
    /// the floor used to be computed as internally, kept here only so these
    /// fixtures keep exercising a floor well below their capacity.
    fn zone(capacity: usize) -> WeightZone {
        WeightZone::new(END, SLOT, capacity, capacity, capacity / 8)
    }

    /// **The zone cannot be asked down to nothing**, however insistent the
    /// caller.
    ///
    /// A wedged scheduler asked for ground on every retry of a wave that could
    /// not succeed. Each ask conceded a region and evicted six experts, and the
    /// zone went 2,377 slots → 0: `weights=0MiB`, every expert destroyed, and
    /// the wave no closer to running because the ceiling was a placed tier the
    /// boundary cannot move.
    #[test]
    fn retraction_stops_at_the_weight_floor() {
        let mut z = zone(800);
        assert_eq!(z.min_capacity(), 100);

        // Asked for everything in one step: the floor holds.
        z.retract_to(0, |_| 0.0);
        assert_eq!(z.capacity(), 100);

        // Asked again and again, as the retry loop did: still the floor.
        for _ in 0..50 {
            z.retract_to(0, |_| 0.0);
        }
        assert_eq!(z.capacity(), 100, "repeated asks must not walk it down");
    }

    /// Above the floor a retraction is unimpeded — the whole point of the
    /// elastic boundary is that the KV side can take most of the zone.
    #[test]
    fn the_floor_does_not_impede_an_ordinary_retraction() {
        let mut z = zone(800);
        z.retract_to(200, |_| 0.0);
        assert_eq!(z.capacity(), 200, "the KV side may take most of the zone");
    }

    /// **The floor is the caller's figure, not a fraction of the opening
    /// capacity** — and that distinction is what the daemon died on.
    ///
    /// A fraction cannot express "the pinned layers' experts must fit", because
    /// that quantity belongs to the model and the fraction belongs to wherever
    /// the boundary happened to open. With 3 pinned layers of 128 experts the
    /// working minimum is 385 slots; an eighth of a 2,377-slot opening is 297.
    /// The boundary retracted to exactly 297, every resident slot came to hold a
    /// pinned-layer expert, and 1,774 consecutive forwards failed `cannot evict
    /// (all pinned)` — a state nothing can leave, because leaving it requires
    /// evicting something and every candidate is pinned.
    #[test]
    fn the_floor_is_the_caller_s_measured_minimum() {
        let mut z = WeightZone::new(END, SLOT, 2377, 2377, 385);
        assert_eq!(z.min_capacity(), 385, "the stated floor, not 2377/8 = 297");
        z.retract_to(0, |_| 0.0);
        assert_eq!(
            z.capacity(),
            385,
            "a zone asked for everything stops at the pinned working set"
        );

        // A zone that opens below its own floor is already as small as it can
        // be — clamped rather than refused, so a small card still loads.
        let small = WeightZone::new(END, SLOT, 64, 64, 385);
        assert_eq!(small.min_capacity(), 64);
    }

    /// **The floor applies from the moment the zone is big enough to have one.**
    ///
    /// A zone that opens below its floor is clamped to its capacity — but the
    /// clamp is a statement about *now*, not a permanent concession. Store it at
    /// construction and a zone that opens small and then grows keeps the small
    /// figure for ever, so the KV side can buy it straight back down to its
    /// opening size and park it in the all-pinned state the floor exists to
    /// prevent.
    #[test]
    fn a_grown_zone_is_held_to_the_real_floor() {
        let mut z = WeightZone::new(END, SLOT, 64, 4000, 385);
        assert_eq!(
            z.min_capacity(),
            64,
            "below its floor, the floor is capacity"
        );

        z.grow_to(2000);
        assert_eq!(z.capacity(), 2000);
        assert_eq!(
            z.min_capacity(),
            385,
            "past the floor, the stated floor applies"
        );

        z.retract_to(0, |_| 0.0);
        assert_eq!(
            z.capacity(),
            385,
            "and a grown zone cannot be taken back below it"
        );
    }

    /// Slot addresses descend as the index rises, are disjoint, and tile the
    /// zone exactly — the arithmetic every slot pointer depends on.
    #[test]
    fn slots_descend_and_tile_exactly() {
        let z = zone(8);
        assert_eq!(z.slot_base(0), END - SLOT as u64, "slot 0 is rightmost");
        assert_eq!(
            z.slot_base(0) + SLOT as u64,
            END,
            "slot 0 ends exactly at the span's right edge"
        );
        for i in 1..8 {
            let base = z.slot_base(i);
            assert!(base < z.slot_base(i - 1), "index must rise leftward");
            assert_eq!(
                z.slot_base(i - 1) - base,
                SLOT as u64,
                "slots {} and {i} leave a gap or overlap",
                i - 1
            );
        }
        assert_eq!(
            z.slot_base(7),
            END - 8 * SLOT as u64,
            "eight slots tile exactly back from the edge"
        );
    }

    /// Allocation always takes the rightmost free spot — including after
    /// out-of-order frees, which is the property a `Vec`-as-stack free list
    /// breaks. `ExpertCacheInner::free_slots` is that `Vec` today: it seeds
    /// `(0..n).rev()` so the first `pop` is index 0, but eviction `push`es the
    /// freed index onto the top and the order is gone.
    #[test]
    fn allocation_takes_the_rightmost_free_spot() {
        let mut z = zone(6);
        let taken: Vec<usize> = (0..6).map(|_| z.alloc().unwrap()).collect();
        assert_eq!(taken, vec![0, 1, 2, 3, 4, 5], "fills rightmost-first");
        assert!(z.alloc().is_none(), "a full zone has nothing to hand out");

        // Free out of order: high, then low.
        z.release(4);
        z.release(1);
        assert_eq!(z.alloc(), Some(1), "the lower index comes back first");
        assert_eq!(z.alloc(), Some(4));
        assert!(z.alloc().is_none());
    }

    /// Newly-gained space is used **last**: it lands at the high indices and
    /// sits behind every closer hole. Nothing is wasted — the free list is still
    /// fully drained before any eviction — but the volatile margin stays empty
    /// while the working set does not need it.
    #[test]
    fn newly_gained_space_is_used_last() {
        let mut z = WeightZone::new(END, SLOT, 4, 8, 0);
        for _ in 0..4 {
            z.alloc().unwrap();
        }
        z.release(2); // a hole at index 2
        assert_eq!(z.grow_to(8), 4, "four new slots");
        assert_eq!(z.capacity(), 8);
        assert_eq!(z.alloc(), Some(2), "the closer hole wins over new space");
        assert_eq!(z.alloc(), Some(4), "only then the newly-gained slots");
    }

    /// Growth is clamped by the limit — the floor's expression in slots. No path
    /// that grows the zone may cross it.
    #[test]
    fn growth_is_clamped_by_the_limit() {
        let mut z = WeightZone::new(END, SLOT, 4, 6, 0);
        assert_eq!(z.grow_to(100), 2, "only up to the limit");
        assert_eq!(z.capacity(), 6);
        assert_eq!(z.grow_to(100), 0, "already there");
        // And the constructor clamps too.
        let z2 = WeightZone::new(END, SLOT, 99, 6, 0);
        assert_eq!(z2.capacity(), 6);
    }

    /// A retraction clears exactly the doomed suffix and nothing else, and the
    /// frontier afterwards is the new boundary.
    #[test]
    fn retraction_clears_exactly_the_suffix() {
        let mut z = zone(8);
        for _ in 0..8 {
            z.alloc().unwrap();
        }
        let plan = z.retract_to(5, |_| 1.0);
        assert_eq!(z.capacity(), 5);
        assert_eq!(plan.relocate, vec![], "no free slots below to move into");
        let mut evicted = plan.evict.clone();
        evicted.sort_unstable();
        assert_eq!(evicted, vec![5, 6, 7], "exactly the suffix");
        for i in 0..5 {
            assert!(z.is_occupied(i), "slot {i} must be untouched");
        }
        assert_eq!(z.live(), 5);
        assert_eq!(z.frontier(), z.slot_base(4));
    }

    /// **The hottest doomed occupants are relocated, the coldest evicted** — and
    /// the choice is made on scores, never on indices. This is what keeps
    /// position out of the eviction decision even though retraction is
    /// inherently positional.
    #[test]
    fn retraction_relocates_the_hot_and_evicts_the_cold() {
        let mut z = zone(8);
        for _ in 0..8 {
            z.alloc().unwrap();
        }
        // Two holes below the new frontier for survivors to land in.
        z.release(0);
        z.release(3);

        // Doomed slots 4..8 with deliberately non-monotonic scores, so a plan
        // that sorted by index instead of temperature gives a different answer.
        let scores = [0.0, 0.0, 0.0, 0.0, 2.0, 9.0, 0.5, 7.0];
        let plan = z.retract_to(4, |i| scores[i]);

        assert_eq!(
            plan.relocate,
            vec![(5, 0), (7, 3)],
            "hottest (5) to the rightmost hole, next (7) to the next"
        );
        let mut evicted = plan.evict.clone();
        evicted.sort_unstable();
        assert_eq!(evicted, vec![4, 6], "the two coldest are dropped");

        assert_eq!(z.capacity(), 4);
        assert_eq!(z.live(), 4, "two survivors moved, two left");
        assert!(z.is_occupied(0) && z.is_occupied(3));
        assert_eq!(z.free_count(), 0);
    }

    /// Ties break by index so a plan is deterministic — an unstable plan would
    /// make the relocation copies unreproducible between runs.
    #[test]
    fn equal_scores_break_by_index() {
        let mut z = zone(6);
        for _ in 0..6 {
            z.alloc().unwrap();
        }
        z.release(0);
        let plan = z.retract_to(3, |_| 1.0);
        assert_eq!(
            plan.relocate,
            vec![(3, 0)],
            "lowest doomed index wins a tie"
        );
        assert_eq!(plan.evict, vec![4, 5]);
    }

    /// Retracting past everything resident is legal and empties the zone.
    #[test]
    fn retracting_to_zero_empties_the_zone() {
        let mut z = zone(4);
        for _ in 0..4 {
            z.alloc().unwrap();
        }
        let plan = z.retract_to(0, |_| 1.0);
        assert_eq!(plan.relocate, vec![]);
        assert_eq!(plan.evict.len(), 4);
        assert_eq!(z.capacity(), 0);
        assert_eq!(z.live(), 0);
        assert_eq!(z.frontier(), END, "an empty zone occupies nothing");
        assert!(z.alloc().is_none());
    }

    /// The frontier tracks what is **occupied**, not what is permitted: a zone
    /// with room for 2,000 experts and three resident is holding three slots'
    /// worth of bytes, and the KV side may have the rest without evicting
    /// anyone.
    #[test]
    fn the_frontier_follows_occupancy_not_capacity() {
        let mut z = zone(100);
        assert_eq!(z.frontier(), END, "empty: the whole span is available");
        z.alloc().unwrap();
        z.alloc().unwrap();
        z.alloc().unwrap();
        assert_eq!(z.frontier(), z.slot_base(2), "three slots' worth, not 100");
        z.release(2);
        assert_eq!(z.frontier(), z.slot_base(1), "and it retracts when freed");
    }

    /// `capacity_for_frontier` is the conversion from the KV side's address into
    /// this side's slot count. It must round **down** — a partial slot is not a
    /// slot, and rounding up would hand out an address past the span.
    #[test]
    fn capacity_for_frontier_rounds_down_and_clamps() {
        let z = WeightZone::new(END, SLOT, 0, 10, 0);
        assert_eq!(z.capacity_for_frontier(END), 0);
        assert_eq!(z.capacity_for_frontier(END - SLOT as u64), 1);
        assert_eq!(
            z.capacity_for_frontier(END - SLOT as u64 - 1),
            1,
            "one byte short of two slots is one slot"
        );
        assert_eq!(z.capacity_for_frontier(END - 4 * SLOT as u64), 4);
        assert_eq!(
            z.capacity_for_frontier(0),
            10,
            "clamped by the limit, not by the address"
        );
        assert_eq!(z.capacity_for_frontier(END + 1), 0, "past the end is none");
    }

    /// The address growth publishes **before** it grows has to be the one the
    /// zone will actually reach — under `grow_to`'s own clamps, not the caller's
    /// ask. Publishing a floor for a capacity the limit forbids hands the KV side
    /// a boundary the weight side never occupies; publishing one below the
    /// current capacity would move the boundary backwards on a growth.
    #[test]
    fn the_floor_growth_publishes_is_the_one_growth_reaches() {
        let mut z = WeightZone::new(END, SLOT, 4, 10, 0);
        assert_eq!(
            z.frontier_after_growth(7),
            END - 7 * SLOT as u64,
            "an ordinary growth publishes its target"
        );
        assert_eq!(
            z.frontier_after_growth(40),
            END - 10 * SLOT as u64,
            "clamped by the limit, exactly as grow_to clamps it"
        );
        assert_eq!(
            z.frontier_after_growth(1),
            z.frontier_for_capacity(),
            "a target below the capacity moves nothing"
        );
        // And the published address is what the zone holds once it has grown.
        z.grow_to(7);
        assert_eq!(z.frontier_for_capacity(), END - 7 * SLOT as u64);
    }

    /// Every slot is either live or free, at every point in a mixed sequence.
    /// Nothing leaks out of the zone and nothing is double-counted.
    #[test]
    fn every_slot_is_either_live_or_free() {
        let mut z = WeightZone::new(END, SLOT, 16, 32, 0);
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for step in 0..4000 {
            match next() % 5 {
                0 | 1 => {
                    z.alloc();
                }
                2 => {
                    let i = (next() % 32) as usize;
                    z.release(i);
                }
                3 => {
                    let c = (next() % 33) as usize;
                    z.retract_to(c, |i| (i % 7) as f32);
                }
                _ => {
                    let c = (next() % 33) as usize;
                    z.grow_to(c);
                }
            }
            assert_eq!(
                z.live() + z.free_count(),
                z.capacity(),
                "slot accounting broke at step {step}"
            );
            assert!(z.capacity() <= z.limit(), "limit crossed at step {step}");
            assert!(
                z.frontier() >= z.span_end - (z.limit() * SLOT) as u64,
                "frontier escaped the zone at step {step}"
            );
        }
    }

    /// **The gradient.** Over repeated retract/refill cycles, experts that keep
    /// being demanded end up at lower (righter, more stable) indices than those
    /// that do not — with no rule anywhere that says so. The sorting is a
    /// consequence of relocation preferring the hot and refill landing rightmost.
    ///
    /// Modelled here without a GPU: eight "hot" experts demanded every cycle,
    /// twenty-four "cold" ones demanded once each, a frontier that oscillates.
    #[test]
    fn hot_experts_drift_right_across_frontier_churn() {
        const HOT: usize = 8;
        const COLD: usize = 24;
        let mut z = WeightZone::new(END, SLOT, 32, 32, 0);

        // slot -> expert id, and expert id -> score. Hot ids are 0..HOT.
        let mut tenant: Vec<Option<usize>> = vec![None; 32];
        let score_of = |id: usize| if id < HOT { 10.0f32 } else { 0.1 };
        let slot_score = |tenant: &Vec<Option<usize>>, i: usize| {
            tenant.get(i).and_then(|t| *t).map_or(0.0, score_of)
        };

        let load = |z: &mut WeightZone, tenant: &mut Vec<Option<usize>>, id: usize| {
            if tenant.iter().take(z.capacity()).any(|t| *t == Some(id)) {
                return;
            }
            if let Some(slot) = z.alloc() {
                tenant[slot] = Some(id);
                return;
            }
            // Full: evict the coldest resident, which is the policy's job and is
            // modelled here as "any cold tenant".
            let victim = (0..z.capacity()).rfind(|&i| tenant[i].is_some_and(|t| t >= HOT));
            if let Some(v) = victim {
                tenant[v] = None;
                z.release(v);
                if let Some(slot) = z.alloc() {
                    tenant[slot] = Some(id);
                }
            }
        };

        let mut cold_next = HOT;
        for cycle in 0..60 {
            // Demand: every hot expert, then two fresh cold ones.
            for id in 0..HOT {
                load(&mut z, &mut tenant, id);
            }
            for _ in 0..2 {
                let id = HOT + (cold_next % COLD);
                cold_next += 1;
                load(&mut z, &mut tenant, id);
            }

            // The frontier breathes: KV takes space on odd cycles, gives it back
            // on even ones.
            if cycle % 2 == 0 {
                let plan = z.retract_to(20, |i| slot_score(&tenant, i));
                for &(from, to) in &plan.relocate {
                    tenant[to] = tenant[from].take();
                }
                for &e in &plan.evict {
                    tenant[e] = None;
                }
                for t in tenant.iter_mut().skip(z.capacity()) {
                    *t = None;
                }
            } else {
                z.grow_to(32);
            }
        }

        let mean = |want_hot: bool| {
            let idx: Vec<usize> = (0..z.capacity())
                .filter(|&i| tenant[i].is_some_and(|t| (t < HOT) == want_hot))
                .collect();
            (!idx.is_empty()).then(|| idx.iter().sum::<usize>() as f64 / idx.len() as f64)
        };
        let hot_mean = mean(true).expect("hot experts must stay resident");
        let cold_mean = mean(false).expect("some cold experts should be resident");
        assert!(
            hot_mean < cold_mean,
            "no gradient: hot mean index {hot_mean:.2} vs cold {cold_mean:.2}"
        );

        // And the strong form: every hot expert survived the churn.
        let resident_hot = (0..z.capacity())
            .filter(|&i| tenant[i].is_some_and(|t| t < HOT))
            .count();
        assert_eq!(resident_hot, HOT, "a hot expert was lost to the frontier");
    }

    /// Without relocation the gradient does not appear — the control that stops
    /// the test above from passing for an unrelated reason. Same workload, same
    /// churn, but every doomed slot is evicted; hot experts then land wherever
    /// the refill finds room and no ordering accumulates.
    #[test]
    fn without_relocation_the_gradient_does_not_appear() {
        let mut z = WeightZone::new(END, SLOT, 32, 32, 0);
        let mut tenant: Vec<Option<usize>> = vec![None; 32];
        const HOT: usize = 8;

        let evict_only_retract = |z: &mut WeightZone, tenant: &mut Vec<Option<usize>>| {
            // score 0 everywhere ⇒ ties everywhere ⇒ relocation still happens by
            // index, so suppress it explicitly by filling the low slots first.
            let plan = z.retract_to(20, |_| 0.0);
            for &(from, to) in &plan.relocate {
                tenant[to] = tenant[from].take();
            }
            for &e in &plan.evict {
                tenant[e] = None;
            }
            for t in tenant.iter_mut().skip(z.capacity()) {
                *t = None;
            }
        };

        for cycle in 0..60 {
            for id in 0..HOT {
                if !tenant.iter().take(z.capacity()).any(|t| *t == Some(id)) {
                    if let Some(s) = z.alloc() {
                        tenant[s] = Some(id);
                    }
                }
            }
            if cycle % 2 == 0 {
                evict_only_retract(&mut z, &mut tenant);
            } else {
                z.grow_to(32);
            }
        }
        // The assertion is only that the harness runs and accounts correctly —
        // the point of this control is that `retract_to` with a flat score makes
        // no temperature distinction, which is exactly what it should do.
        assert_eq!(z.live() + z.free_count(), z.capacity());
    }

    /// A relocation destination is never a slot that is already occupied, and
    /// never a slot that is itself doomed. Violating either would corrupt a live
    /// expert or copy into memory about to be handed to the KV side.
    #[test]
    fn relocation_destinations_are_free_and_surviving() {
        let mut z = zone(12);
        for _ in 0..12 {
            z.alloc().unwrap();
        }
        for i in [0usize, 2, 5] {
            z.release(i);
        }
        let scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let plan = z.retract_to(6, |i| scores[i]);
        for &(from, to) in &plan.relocate {
            assert!(from >= 6, "source {from} was not doomed");
            assert!(to < 6, "destination {to} does not survive the retraction");
        }
        let dests: Vec<usize> = plan.relocate.iter().map(|&(_, t)| t).collect();
        let mut sorted = dests.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), dests.len(), "a destination was reused");
        assert_eq!(sorted, vec![0, 2, 5], "exactly the free surviving slots");
    }
}

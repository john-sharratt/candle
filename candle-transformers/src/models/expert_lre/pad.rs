//! The pad — a mutable pinned tier the stager stages cold experts into.
//!
//! Slot images at the pack's stride, in one `cuMemAllocHost` block allocated at
//! startup before the warm tier (it is mandatory, the warm tier elastic). The
//! device reads a pad slot at its host address, exactly as it reads a pinned
//! warm slot, so a staged expert is computed by the expert GEMMs' workers
//! straight from here; the pipeline thread may later promote it into VRAM.
//!
//! **At least one layer of slots** (`n_experts`): a layer's whole cold set is
//! then always stageable without reusing a slot the same layer routes, so the
//! demand path never needs a "consumed" signal from the GPU. Every slot beyond
//! that is cache.
//!
//! **One layer, deliberately.** A replay of Flash-Next's decode routing (RTX 4090
//! Laptop) says an LRU pad of 4096 slots holding only demand misses would serve
//! 40–65% of the cold misses again, where one layer serves almost none. Built as
//! eight layers it did not: the speculative staging (a few in ten routed) filled
//! the pad and evicted the reusable copies, the victim scan sorts every slot per
//! read, and the warm tier it is cut from halved — decode fell a third, prefill
//! more. A larger pad wants staging admitted by its precision and an ordered
//! victim index first.
//!
//! Eviction is LRE over the stager's own score table: it reads every row's
//! routing summary anyway, so it credits each staged expert a routed row sends
//! to (decode +1.0, prefill +0.1, as the VRAM scores) and decays the table at
//! each pass boundary. A slot whose expert is also in VRAM is the cheapest
//! victim — its entry already names VRAM, so evicting it changes nothing the
//! device reads.
//!
//! **A slot staged ahead is fresh until its row is reached.** A speculative read
//! lands an expert nothing has routed yet, so its score is 0 — the lowest there
//! is — and without protection it would be the next read's first victim, the
//! predictions evicting each other before the wave arrives. A fresh slot is never
//! a speculative read's victim, and a demand read's only after every other
//! candidate. The stager settles a row's fresh slots when it begins that row
//! ([`PadBook::settle_row`]): a routed one is credited like any other, a
//! mispredicted one simply loses its protection.

use super::cache::PREFILL_HIT_SCORE;
use super::pinned::WarmPool;
use candle::Result;
use std::cmp::Ordering;

/// One pad slot's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PadSlot {
    Free,
    /// A read is landing in it, for `(row, expert)`.
    Loading(usize, usize),
    /// It holds `(row, expert)`'s record, published.
    Held(usize, usize),
}

/// The pad's slot states, free list and scores — the stager's private policy.
pub(crate) struct PadBook {
    slots: Vec<PadSlot>,
    /// Per slot: staged ahead, and its row not yet begun.
    fresh: Vec<bool>,
    free: Vec<usize>,
    scores: Vec<f32>,
    n_experts: usize,
}

/// Which held slots a victim search may return.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fresh {
    /// Never a fresh slot — a speculative read must not evict another.
    Spare,
    /// A fresh slot only after every other candidate — a demand read must
    /// make progress.
    Last,
}

impl PadBook {
    pub(crate) fn new(n_slots: usize, rows: usize, n_experts: usize) -> Self {
        Self {
            slots: vec![PadSlot::Free; n_slots],
            fresh: vec![false; n_slots],
            // Popped from the back: slot 0 first.
            free: (0..n_slots).rev().collect(),
            scores: vec![0.0; rows * n_experts],
            n_experts,
        }
    }

    pub(crate) fn state(&self, slot: usize) -> PadSlot {
        self.slots[slot]
    }

    pub(crate) fn take_free(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// `slot` (free, or a victim just taken) starts receiving `(row, expert)`,
    /// `ahead` of its row when the read is speculative.
    pub(crate) fn start_load(&mut self, slot: usize, row: usize, expert: usize, ahead: bool) {
        self.slots[slot] = PadSlot::Loading(row, expert);
        self.fresh[slot] = ahead;
    }

    #[cfg(test)]
    pub(crate) fn is_fresh(&self, slot: usize) -> bool {
        self.fresh[slot]
    }

    /// The stager has begun `row`: its slots staged ahead are fresh no more.
    /// Returns the experts they hold, so the caller can count the routed ones.
    pub(crate) fn settle_row(&mut self, row: usize) -> Vec<usize> {
        let mut settled = Vec::new();
        for (slot, s) in self.slots.iter().enumerate() {
            if let PadSlot::Loading(r, e) | PadSlot::Held(r, e) = *s {
                if r == row && self.fresh[slot] {
                    self.fresh[slot] = false;
                    settled.push(e);
                }
            }
        }
        settled
    }

    /// `row`'s experts scoring at least `min`, highest first, at most `k` —
    /// what the recent passes routed there, and so what the next pass is
    /// likely to.
    pub(crate) fn top_scored(&self, row: usize, min: f32, k: usize) -> Vec<(usize, f32)> {
        let base = row * self.n_experts;
        let mut top: Vec<(usize, f32)> = self.scores[base..base + self.n_experts]
            .iter()
            .enumerate()
            .filter(|&(_, &s)| s >= min)
            .map(|(e, &s)| (e, s))
            .collect();
        top.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        top.truncate(k);
        top
    }

    /// `slot`'s read has landed; it now holds its expert. Returns the expert.
    pub(crate) fn landed(&mut self, slot: usize) -> (usize, usize) {
        let PadSlot::Loading(row, expert) = self.slots[slot] else {
            panic!("pad: slot {slot} landed while {:?}", self.slots[slot]);
        };
        self.slots[slot] = PadSlot::Held(row, expert);
        (row, expert)
    }

    /// A routed row sent tokens to `(row, expert)`.
    pub(crate) fn credit(&mut self, row: usize, expert: usize, decode: bool) {
        self.scores[row * self.n_experts + expert] += if decode { 1.0 } else { PREFILL_HIT_SCORE };
    }

    pub(crate) fn decay(&mut self, factor: f32) {
        self.scores.iter_mut().for_each(|s| *s *= factor);
    }

    pub(crate) fn score(&self, row: usize, expert: usize) -> f32 {
        self.scores[row * self.n_experts + expert]
    }

    /// Up to `count` held slots to evict, best first. `admit(row, expert)`
    /// returns `None` for a slot that may not go, else whether its expert is
    /// also in VRAM — those go first, then the slots that are not fresh, then
    /// the lowest score. `fresh` says whether a fresh slot may go at all.
    pub(crate) fn victims(
        &self,
        count: usize,
        fresh: Fresh,
        admit: impl Fn(usize, usize) -> Option<bool>,
    ) -> Vec<usize> {
        let mut cands: Vec<(usize, bool, bool, f32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match *s {
                PadSlot::Held(row, expert) => {
                    let is_fresh = self.fresh[slot];
                    if is_fresh && fresh == Fresh::Spare {
                        return None;
                    }
                    admit(row, expert).map(|vram| (slot, vram, is_fresh, self.score(row, expert)))
                }
                _ => None,
            })
            .collect();
        cands.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.2.cmp(&b.2))
                .then(a.3.partial_cmp(&b.3).unwrap_or(Ordering::Equal))
                .then(a.0.cmp(&b.0))
        });
        cands.into_iter().take(count).map(|c| c.0).collect()
    }
}

/// The pad's memory.
pub(crate) struct Pad {
    pool: WarmPool,
    stride: usize,
}

impl Pad {
    /// `n_slots` slots of `stride` bytes, all or nothing: the pad is what makes
    /// a cold expert computable, so a machine that cannot pin it cannot run the
    /// model's expert path.
    pub(crate) fn new(n_slots: usize, stride: usize) -> Result<Self> {
        let pool = WarmPool::new(n_slots, stride, candle::vram::PinnedUse::Staging);
        if pool.num_slots() != n_slots {
            candle::bail!(
                "expert pad: {n_slots} pinned slots of {stride} B were refused (got {}) — one \
                 layer of experts must be pinnable for cold experts to be staged",
                pool.num_slots()
            );
        }
        Ok(Self { pool, stride })
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.pool.num_slots()
    }

    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Slot `i`'s address, host and device alike.
    pub(crate) fn slot_addr(&self, i: usize) -> u64 {
        self.pool.slot_addr(i)
    }

    pub(crate) fn range(&self) -> (u64, u64) {
        self.pool.range()
    }

    /// Slot `i`'s bytes, for a writer that owns the slot. The stager hands one
    /// slot to one reader at a time and publishes it only after the write.
    ///
    /// # Safety
    ///
    /// No other reference to slot `i`'s bytes may be live, on the host or in a
    /// copy, while the returned slice is.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn slot_mut(&self, i: usize) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.slot_addr(i) as *mut u8, self.stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Slots come off the free list in order and move Loading → Held; a held
    /// slot taken again as a victim reloads in place.
    #[test]
    fn a_slot_cycles_through_its_states() {
        let mut b = PadBook::new(3, 2, 4);
        assert_eq!(b.take_free(), Some(0));
        b.start_load(0, 1, 3, false);
        assert_eq!(b.state(0), PadSlot::Loading(1, 3));
        assert_eq!(b.landed(0), (1, 3));
        assert_eq!(b.state(0), PadSlot::Held(1, 3));
        b.start_load(0, 0, 2, false);
        assert_eq!(b.landed(0), (0, 2), "a victim reloaded");
        assert_eq!(b.take_free(), Some(1));
        assert_eq!(b.take_free(), Some(2));
        assert_eq!(b.take_free(), None);
    }

    /// Credit, decay and victim order: a VRAM-backed expert first whatever its
    /// score, then the lowest score; loading and refused slots never.
    #[test]
    fn victims_prefer_vram_backed_then_the_coldest() {
        let mut b = PadBook::new(5, 2, 4);
        for (slot, (row, e)) in [(1, 0), (1, 1), (1, 2), (0, 3), (0, 0)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(b.take_free(), Some(slot));
            b.start_load(slot, row, e, false);
            if slot != 4 {
                b.landed(slot);
            }
        }
        b.credit(1, 0, true); // 1.0
        b.credit(1, 1, false); // 0.1
        b.credit(1, 2, true);
        b.credit(1, 2, true); // 2.0
        b.decay(0.5);
        assert_eq!(b.score(1, 0), 0.5);
        assert_eq!(b.score(1, 2), 1.0);
        // (0, 3) is refused; (1, 2) is VRAM-backed; slot 4 is still loading.
        let v = b.victims(10, Fresh::Spare, |row, e| match (row, e) {
            (0, 3) => None,
            (1, 2) => Some(true),
            _ => Some(false),
        });
        assert_eq!(v, vec![2, 1, 0]);
        assert_eq!(
            b.victims(1, Fresh::Spare, |_, _| Some(false)),
            vec![3],
            "the unscored slot first"
        );
    }

    /// A slot staged ahead is never a speculative read's victim and a demand
    /// read's only last, until its row is begun; a demand load into the slot
    /// clears it too.
    #[test]
    fn a_fresh_slot_is_spared_until_its_row_is_settled() {
        let mut b = PadBook::new(4, 3, 4);
        // Slots 0 and 1 staged ahead for row 2, slot 2 an old demand copy of
        // row 0 with a score, slot 3 an unscored demand copy of row 1.
        for (slot, (row, e, ahead)) in [(2, 0, true), (2, 1, true), (0, 1, false), (1, 3, false)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(b.take_free(), Some(slot));
            b.start_load(slot, row, e, ahead);
            b.landed(slot);
        }
        b.credit(0, 1, true);
        assert!(b.is_fresh(0) && b.is_fresh(1) && !b.is_fresh(2));
        assert_eq!(b.victims(4, Fresh::Spare, |_, _| Some(false)), vec![3, 2]);
        assert_eq!(
            b.victims(4, Fresh::Last, |_, _| Some(false)),
            vec![3, 2, 0, 1],
            "fresh slots after every other candidate"
        );
        assert_eq!(
            b.settle_row(1),
            Vec::<usize>::new(),
            "row 1 staged nothing ahead"
        );
        assert!(b.is_fresh(0), "settling another row changes nothing");
        assert_eq!(b.settle_row(2), vec![0, 1]);
        assert!(!b.is_fresh(0) && !b.is_fresh(1));
        assert_eq!(b.settle_row(2), Vec::<usize>::new(), "settled once");
        assert_eq!(
            b.victims(4, Fresh::Spare, |_, _| Some(false)),
            vec![0, 1, 3, 2]
        );
        b.start_load(3, 2, 2, true);
        assert!(b.is_fresh(3));
        b.start_load(3, 1, 0, false);
        assert!(!b.is_fresh(3), "a demand load is not fresh");
    }

    /// A row's scored experts, highest first, ties by expert, floored and capped.
    #[test]
    fn top_scored_ranks_a_rows_experts_above_the_floor() {
        let mut b = PadBook::new(1, 2, 5);
        b.credit(1, 3, true);
        b.credit(1, 3, true); // 2.0
        b.credit(1, 0, true); // 1.0
        b.credit(1, 4, true); // 1.0
        b.credit(1, 2, false); // 0.1
        b.credit(0, 1, true); // another row
        assert_eq!(b.top_scored(1, 0.5, 8), vec![(3, 2.0), (0, 1.0), (4, 1.0)]);
        assert_eq!(b.top_scored(1, 0.5, 2), vec![(3, 2.0), (0, 1.0)]);
        assert_eq!(
            b.top_scored(1, 0.0, 8).len(),
            5,
            "a zero floor admits every expert"
        );
    }
}

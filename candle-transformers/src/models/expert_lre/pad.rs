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
//! Eviction is LRE over the stager's own score table: it reads every row's
//! routing summary anyway, so it credits each staged expert a routed row sends
//! to (decode +1.0, prefill +0.1, as the VRAM scores) and decays the table at
//! each pass boundary. A slot whose expert is also in VRAM is the cheapest
//! victim — its entry already names VRAM, so evicting it changes nothing the
//! device reads.

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
    free: Vec<usize>,
    scores: Vec<f32>,
    n_experts: usize,
}

impl PadBook {
    pub(crate) fn new(n_slots: usize, rows: usize, n_experts: usize) -> Self {
        Self {
            slots: vec![PadSlot::Free; n_slots],
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

    /// `slot` (free, or a victim just taken) starts receiving `(row, expert)`.
    pub(crate) fn start_load(&mut self, slot: usize, row: usize, expert: usize) {
        self.slots[slot] = PadSlot::Loading(row, expert);
    }

    /// `slot`'s read has landed; it now holds its expert. Returns the expert.
    pub(crate) fn landed(&mut self, slot: usize) -> (usize, usize) {
        let PadSlot::Loading(row, expert) = self.slots[slot] else {
            panic!("pad: slot {slot} landed while {:?}", self.slots[slot]);
        };
        self.slots[slot] = PadSlot::Held(row, expert);
        (row, expert)
    }

    /// `slot` holds nothing any more; its bytes may be overwritten.
    pub(crate) fn release(&mut self, slot: usize) {
        self.slots[slot] = PadSlot::Free;
        self.free.push(slot);
    }

    /// `slot` holds nothing, but its bytes may still be read — it is on a
    /// retire list and comes back through [`Self::release`].
    pub(crate) fn vacate(&mut self, slot: usize) {
        self.slots[slot] = PadSlot::Free;
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
    /// also in VRAM — those go first, then the lowest score.
    pub(crate) fn victims(
        &self,
        count: usize,
        admit: impl Fn(usize, usize) -> Option<bool>,
    ) -> Vec<usize> {
        let mut cands: Vec<(usize, bool, f32)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match *s {
                PadSlot::Held(row, expert) => {
                    admit(row, expert).map(|vram| (slot, vram, self.score(row, expert)))
                }
                _ => None,
            })
            .collect();
        cands.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal))
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

    /// Slots come off the free list in order, move Loading → Held, and go back
    /// when released.
    #[test]
    fn a_slot_cycles_through_its_states() {
        let mut b = PadBook::new(3, 2, 4);
        assert_eq!(b.take_free(), Some(0));
        b.start_load(0, 1, 3);
        assert_eq!(b.state(0), PadSlot::Loading(1, 3));
        assert_eq!(b.landed(0), (1, 3));
        assert_eq!(b.state(0), PadSlot::Held(1, 3));
        assert_eq!(b.take_free(), Some(1));
        b.release(0);
        assert_eq!(b.state(0), PadSlot::Free);
        assert_eq!(b.take_free(), Some(0));
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
            b.start_load(slot, row, e);
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
        let v = b.victims(10, |row, e| match (row, e) {
            (0, 3) => None,
            (1, 2) => Some(true),
            _ => Some(false),
        });
        assert_eq!(v, vec![2, 1, 0]);
        assert_eq!(
            b.victims(1, |_, _| Some(false)),
            vec![3],
            "the unscored slot first"
        );
    }
}

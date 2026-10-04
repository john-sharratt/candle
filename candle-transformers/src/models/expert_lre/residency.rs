//! Where every expert's copies are, and the live-table entry that follows.
//!
//! Two threads change where an expert is readable from: the pipeline thread
//! (VRAM promotion and eviction) and the stager (pad staging and pad eviction).
//! An entry's value depends on facts both own — an expert evicted from VRAM
//! falls back to its pad copy, if the stager has one — so every change goes
//! through one [`Residency`] behind one `Mutex`, which also writes the entry.
//! The lock is never held across a CUDA call or a pack read.
//!
//! **The entry is a pure function of the places**: the VRAM slot if there is
//! one, else the pad slot, else a pinned warm slot, else 0. A pageable warm slot
//! is not device-readable, so an expert held only there is cold (the stager
//! stages it into the pad with a `memcpy`).
//!
//! Whether a change is *allowed* now — an entry going to 0 needs a quiet row, a
//! slot that lost its tenant waits on a retire key — is the caller's to decide
//! before it asks (`reclaim`); this type answers what the entry would become.

use super::live_table::LiveTable;
use std::sync::Arc;

/// One expert's copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Place {
    /// Weight-zone slot and its device address.
    pub(crate) vram: Option<(usize, u64)>,
    /// Pad slot and its host address (device-readable at the same address).
    pub(crate) pad: Option<(usize, u64)>,
    /// Warm-tier slot, with its address when it is pinned.
    pub(crate) warm: Option<(usize, Option<u64>)>,
    /// Promotion copies reading the pad slot right now. A pinned pad slot is
    /// never evicted.
    pub(crate) pins: u32,
}

impl Place {
    /// The live-table entry these places make: a slot image's first byte, or 0.
    pub(crate) fn entry(&self) -> u64 {
        if let Some((_, b)) = self.vram {
            b
        } else if let Some((_, b)) = self.pad {
            b
        } else if let Some((_, Some(b))) = self.warm {
            b
        } else {
            0
        }
    }

    /// A pinned host copy the copy engine may promote from, and whether it is
    /// the pad's.
    pub(crate) fn pinned_source(&self) -> Option<(u64, bool)> {
        if let Some((_, b)) = self.pad {
            Some((b, true))
        } else if let Some((_, Some(b))) = self.warm {
            Some((b, false))
        } else {
            None
        }
    }
}

/// Every expert's places, and the table they are published into.
pub(crate) struct Residency {
    places: Vec<Place>,
    n_experts: usize,
    table: Arc<LiveTable>,
}

impl Residency {
    /// Every expert with no copy anywhere and every entry 0. Fill with the
    /// setters, then [`Self::publish_all`].
    pub(crate) fn new(table: Arc<LiveTable>) -> Self {
        let n_experts = table.n_experts();
        Self {
            places: vec![Place::default(); table.n_rows() * n_experts],
            n_experts,
            table,
        }
    }

    fn at(&self, row: usize, expert: usize) -> usize {
        row * self.n_experts + expert
    }

    pub(crate) fn place(&self, row: usize, expert: usize) -> Place {
        self.places[self.at(row, expert)]
    }

    /// Change one place with `f`, and write the entry if its value moved.
    /// Returns `(before, after)`.
    fn change(&mut self, row: usize, expert: usize, f: impl FnOnce(&mut Place)) -> (u64, u64) {
        let i = self.at(row, expert);
        let before = self.places[i].entry();
        f(&mut self.places[i]);
        let after = self.places[i].entry();
        if after != before {
            if after == 0 {
                self.table.clear(row, expert);
            } else {
                self.table.publish(row, expert, after);
            }
        }
        (before, after)
    }

    /// Record the expert in VRAM slot `slot` at `base`, or out of VRAM.
    pub(crate) fn set_vram(
        &mut self,
        row: usize,
        expert: usize,
        vram: Option<(usize, u64)>,
    ) -> (u64, u64) {
        self.change(row, expert, |p| p.vram = vram)
    }

    /// Record the expert in pad slot `slot` at `base`, or out of the pad.
    pub(crate) fn set_pad(
        &mut self,
        row: usize,
        expert: usize,
        pad: Option<(usize, u64)>,
    ) -> (u64, u64) {
        self.change(row, expert, |p| p.pad = pad)
    }

    /// Record the expert's warm slot — at startup, once; the warm tier is
    /// immutable.
    pub(crate) fn set_warm(&mut self, row: usize, expert: usize, warm: Option<(usize, Option<u64>)>) {
        let i = self.at(row, expert);
        self.places[i].warm = warm;
    }

    /// Whether the expert keeps a device-readable host copy (a pad slot or a
    /// pinned warm slot) — so leaving VRAM retargets its entry to an address,
    /// never to 0.
    pub(crate) fn has_pinned_copy(&self, row: usize, expert: usize) -> bool {
        self.place(row, expert).pinned_source().is_some()
    }

    /// A promotion copy starts (`+1`) or ends (`-1`) reading the pad slot.
    pub(crate) fn pin_pad(&mut self, row: usize, expert: usize, delta: i32) {
        let i = self.at(row, expert);
        let pins = self.places[i].pins as i64 + delta as i64;
        assert!(pins >= 0, "residency: pad pin count of ({row}, {expert}) below zero");
        self.places[i].pins = pins as u32;
    }

    /// Write every entry from the places — at startup, before the first
    /// forward.
    pub(crate) fn publish_all(&self) {
        for row in 0..self.table.n_rows() {
            for e in 0..self.n_experts {
                let v = self.place(row, e).entry();
                if v == 0 {
                    self.table.clear(row, e);
                } else {
                    self.table.publish(row, e, v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::live_table::Proj;
    use super::*;

    fn residency() -> (Residency, Arc<LiveTable>) {
        let t = Arc::new(LiveTable::host_only(2, 4, [0, 0x100, 0x300]));
        (Residency::new(t.clone()), t)
    }

    /// Every row of the design's transition table (§0.4), with the entry each
    /// leaves behind: VRAM over pad over pinned warm over 0.
    #[test]
    fn the_entry_follows_the_nearest_copy() {
        let (mut r, t) = residency();
        let pinned_warm = 0x7100_0000u64;
        r.set_warm(1, 2, Some((9, Some(pinned_warm))));
        r.publish_all();
        assert_eq!(t.entry(Proj::Gate, 1, 2), pinned_warm, "warm only");
        assert_eq!(t.entry(Proj::Down, 1, 2), pinned_warm + 0x300);

        // Promotion lands: VRAM.
        assert_eq!(r.set_vram(1, 2, Some((40, 0xa000_0000))), (pinned_warm, 0xa000_0000));
        assert_eq!(t.entry(Proj::Up, 1, 2), 0xa000_0100);
        // The stager publishes a pad copy: the entry stays VRAM.
        assert_eq!(r.set_pad(1, 2, Some((3, 0x9000_0000))), (0xa000_0000, 0xa000_0000));
        assert_eq!(t.entry(Proj::Gate, 1, 2), 0xa000_0000);
        // VRAM eviction falls back to the pad.
        assert_eq!(r.set_vram(1, 2, None), (0xa000_0000, 0x9000_0000));
        // Pad eviction falls back to the pinned warm slot.
        assert_eq!(r.set_pad(1, 2, None), (0x9000_0000, pinned_warm));
        assert_eq!(t.entry(Proj::Gate, 1, 2), pinned_warm);
    }

    /// A pageable warm slot is not device-readable: an expert held only there
    /// is cold, and a pad eviction that leaves only it writes 0 everywhere.
    #[test]
    fn a_pageable_warm_copy_is_cold() {
        let (mut r, t) = residency();
        r.set_warm(0, 1, Some((70, None)));
        r.publish_all();
        assert_eq!(t.entry(Proj::Gate, 0, 1), 0);
        r.set_pad(0, 1, Some((0, 0x9000_0000)));
        assert_eq!(t.entry(Proj::Gate, 0, 1), 0x9000_0000);
        assert_eq!(r.set_pad(0, 1, None), (0x9000_0000, 0));
        for p in [Proj::Gate, Proj::Up, Proj::Down] {
            assert_eq!(t.entry(p, 0, 1), 0);
        }
    }

    /// A promotion's pin is counted, and the copy source prefers the pad.
    #[test]
    fn pins_count_and_the_pad_is_the_preferred_source() {
        let (mut r, _) = residency();
        r.set_warm(1, 0, Some((5, Some(0x7100_0000))));
        assert_eq!(r.place(1, 0).pinned_source(), Some((0x7100_0000, false)));
        r.set_pad(1, 0, Some((2, 0x9000_0000)));
        assert_eq!(r.place(1, 0).pinned_source(), Some((0x9000_0000, true)));
        r.pin_pad(1, 0, 1);
        r.pin_pad(1, 0, 1);
        assert_eq!(r.place(1, 0).pins, 2);
        r.pin_pad(1, 0, -2);
        assert_eq!(r.place(1, 0).pins, 0);
        assert_eq!(r.place(1, 1).pinned_source(), None);
    }
}

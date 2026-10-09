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

use super::live_table::{LiveTable, Proj};
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
    /// Standing in the promotion ring as a lazy victim whose fallback is cold:
    /// a claim zeroes its entries on the device, so a routing summary that
    /// finds it cold is the eviction, and the stager books it
    /// ([`Residency::device_evicted`]) before staging it.
    pub(crate) offered_cold: bool,
}

/// Where a lazy victim's entries go when a miss claims its slot — what an
/// eviction here would publish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fallback {
    /// Its pinned warm slot.
    Warm,
    /// Its pad slot, pinned against the stager for the life of the offer.
    Pad,
    /// Nowhere device-readable: entries 0, and the next launch that routes it
    /// waits on the stager.
    Cold,
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

    /// The pinned host copy the device reads the expert from once its VRAM
    /// slot is claimed, and whether it is the pad's.
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
    pub(crate) fn set_warm(
        &mut self,
        row: usize,
        expert: usize,
        warm: Option<(usize, Option<u64>)>,
    ) {
        let i = self.at(row, expert);
        self.places[i].warm = warm;
    }

    /// The entries (gate, up, down) a VRAM-resident expert falls back to when
    /// the device evicts it — exactly what an eviction here would publish: its
    /// pad slot's, else its pinned warm slot's, else 0 — and which that is, or
    /// `None` when it is not in VRAM.
    ///
    /// The caller holds the fallback for as long as the offer stands: a pad
    /// slot pinned (`pin_pad`), since the stager owns pad eviction and must not
    /// reuse the slot the device may retarget to; a cold one marked
    /// (`set_offered_cold`), so the stager reads a cold summary of it as the
    /// device's eviction.
    pub(crate) fn displaced_entries(
        &self,
        row: usize,
        expert: usize,
    ) -> Option<([u64; 3], Fallback)> {
        let p = self.place(row, expert);
        p.vram?;
        Some(match p.pinned_source() {
            Some((base, from_pad)) => (
                [
                    base + self.table.offset(Proj::Gate, row),
                    base + self.table.offset(Proj::Up, row),
                    base + self.table.offset(Proj::Down, row),
                ],
                if from_pad {
                    Fallback::Pad
                } else {
                    Fallback::Warm
                },
            ),
            None => ([0; 3], Fallback::Cold),
        })
    }

    /// Mark (or clear) the expert as a lazy victim whose fallback is cold.
    pub(crate) fn set_offered_cold(&mut self, row: usize, expert: usize, offered: bool) {
        let i = self.at(row, expert);
        self.places[i].offered_cold = offered;
    }

    /// The device evicted the expert from VRAM by claiming its slot: its
    /// entries already name its fallback. Book it here, publishing the same
    /// value. Idempotent — the stager calls it on the first cold summary, the
    /// pipeline thread when it collects the claim, whichever comes first.
    pub(crate) fn device_evicted(&mut self, row: usize, expert: usize) -> (u64, u64) {
        self.change(row, expert, |p| {
            p.vram = None;
            p.offered_cold = false;
        })
    }

    /// Something the device may read the pad slot through starts (`+1`) or
    /// stops (`-1`) standing: a read-ahead listing (`ahead_pins`), or a lazy
    /// victim's offer whose fallback is the slot. The stager evicts no pinned
    /// slot.
    pub(crate) fn pin_pad(&mut self, row: usize, expert: usize, delta: i32) {
        let i = self.at(row, expert);
        let pins = self.places[i].pins as i64 + delta as i64;
        assert!(
            pins >= 0,
            "residency: pad pin count of ({row}, {expert}) below zero"
        );
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
    use super::*;

    /// A lazy victim's fallback is exactly the entry eviction would publish —
    /// its pad slot over its pinned warm slot, plus the row's projection
    /// offsets, else 0 — and only an expert in VRAM is one.
    #[test]
    fn a_lazy_victim_falls_back_to_what_eviction_publishes() {
        let (mut r, t) = residency();
        let (warm, pad) = (0x7100_0000u64, 0x9000_0000u64);
        r.set_warm(1, 2, Some((9, Some(warm))));
        assert_eq!(r.displaced_entries(1, 2), None, "not in VRAM");
        r.set_vram(1, 2, Some((40, 0xa000_0000)));
        assert_eq!(
            r.displaced_entries(1, 2),
            Some(([warm, warm + 0x100, warm + 0x300], Fallback::Warm))
        );
        r.set_pad(1, 2, Some((3, pad)));
        assert_eq!(
            r.displaced_entries(1, 2),
            Some(([pad, pad + 0x100, pad + 0x300], Fallback::Pad)),
            "the pad copy is nearer"
        );
        r.set_vram(1, 2, None);
        assert_eq!(
            t.entry(Proj::Gate, 1, 2),
            pad,
            "what the device writes, eviction publishes"
        );

        r.set_warm(0, 1, Some((4, None)));
        r.set_vram(0, 1, Some((41, 0xa100_0000)));
        assert_eq!(
            r.displaced_entries(0, 1),
            Some(([0; 3], Fallback::Cold)),
            "a pageable warm copy is not device-readable"
        );
    }

    /// A device eviction books the fallback the device wrote, clears the
    /// offered-cold mark, and is idempotent.
    #[test]
    fn a_device_eviction_is_booked_once_whoever_sees_it_first() {
        let (mut r, t) = residency();
        r.set_vram(0, 1, Some((41, 0xa100_0000)));
        r.set_offered_cold(0, 1, true);
        assert!(r.place(0, 1).offered_cold);
        assert_eq!(r.device_evicted(0, 1), (0xa100_0000, 0));
        assert_eq!(t.entry(Proj::Gate, 0, 1), 0);
        assert!(!r.place(0, 1).offered_cold);
        assert_eq!(
            r.device_evicted(0, 1),
            (0, 0),
            "a second booking changes nothing"
        );
    }

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
        assert_eq!(
            r.set_vram(1, 2, Some((40, 0xa000_0000))),
            (pinned_warm, 0xa000_0000)
        );
        assert_eq!(t.entry(Proj::Up, 1, 2), 0xa000_0100);
        // The stager publishes a pad copy: the entry stays VRAM.
        assert_eq!(
            r.set_pad(1, 2, Some((3, 0x9000_0000))),
            (0xa000_0000, 0xa000_0000)
        );
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

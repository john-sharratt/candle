//! Reports how much of the span the gallery holds, and how much of that a turn's
//! pages can give back.
//!
//! Reserved bytes are whole regions, which the memory report charges to the
//! gallery, while page bytes are what eviction can actually free. The ceiling is
//! measured in page bytes, because a region is freed only once every page in its
//! arena has gone, so a ceiling on regions could stay breached however many turns
//! were evicted. The per-turn run list is what breaks the gallery down by whose
//! turns it holds, and what the sweep of turns no scan can read is made from.

use super::*;

impl GalleryArena {
    /// Reservation bytes the gallery's arenas hold — whole regions, whatever is in
    /// them. What the gallery denies the rest of the span, and so what the memory
    /// report accounts to it.
    pub fn resident_bytes(&self) -> u64 {
        (arena_regions(&self.device, SlotTenant::Gallery) * SpanRegion::bytes()) as u64
    }

    /// Bytes of the pages resident turns hold — what eviction can give back. Less
    /// than [`Self::resident_bytes`] by every arena's free slots and unused tail.
    ///
    /// **The ceiling is measured in this, not in regions.** Evicting a turn frees
    /// its pages, but a region goes back only when every page in its arena has
    /// gone, so a ceiling on regions can stay breached however many turns are
    /// evicted — and would then evict the whole gallery, working set and all, to
    /// satisfy a figure eviction does not directly move.
    pub fn page_bytes(&self) -> u64 {
        arena_held_bytes(&self.device, SlotTenant::Gallery) as u64
    }

    /// Number of turns currently resident.
    pub fn resident_turns(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Every resident turn: its stream, the bytes of the pages it holds, how
    /// long since a scan last used it, and who it is held for — what a
    /// breakdown of the gallery by whose turns it holds, and the scheduler's
    /// sweep of the turns no scan can read, are made from.
    pub fn resident_runs(&self) -> Vec<ResidentRun> {
        let now = Instant::now();
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(sid, rt)| ResidentRun {
                sid: *sid,
                bytes: self.run_bytes(rt),
                idle: now.saturating_duration_since(rt.used),
                holder: rt.holder,
            })
            .collect()
    }

    /// Pages this arena's resident turns hold — its own share, where
    /// [`Self::page_bytes`] counts every gallery page on the device.
    #[cfg(test)]
    pub(super) fn live_pages(&self) -> usize {
        self.residency
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|rt| rt.run.slots.len())
            .sum()
    }

    /// The residency mutation count — churn, reported per scan (see
    /// [`GalleryArena::residency_gen`]'s field).
    #[inline]
    pub(in super::super) fn residency_gen(&self) -> u64 {
        self.residency_gen.load(Ordering::Relaxed)
    }

    /// Resident turns, and of them how many a scan holds pinned now — what the
    /// keep trade may not take whatever their age.
    pub fn residency_counts(&self) -> (usize, usize) {
        let res = self.residency.lock().unwrap_or_else(|e| e.into_inner());
        (res.len(), res.values().filter(|rt| rt.pinned != 0).count())
    }
}

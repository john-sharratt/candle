//! The weight side's growth ledger over one window of a run.
//!
//! `expert_lre::grow_tally` counts every growth answer since boot: `(asked, no_spare,
//! spare_regions_offered, target_unchanged, target_backwards, floor_refused, at_limit,
//! slots_gained)`. The uptake gate is judged on the drain alone, so its verdict has to
//! come from what the ledger recorded during the drain — the difference of two
//! snapshots — and not from the totals, which also hold the load-time fill and every
//! regrowth after a concession in the phases before it.

/// The ledger's movement between two snapshots of `grow_tally`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GrowWindow([u64; 8]);

impl GrowWindow {
    pub(crate) fn between(start: [u64; 8], end: [u64; 8]) -> Self {
        Self(std::array::from_fn(|i| end[i].saturating_sub(start[i])))
    }

    /// The zone reached every expert slot the model has, and nothing short of that
    /// limit stopped it: in the window it answered `at_limit`, and never with a refusal
    /// that had room to grow (`target_unchanged`), a target below the zone
    /// (`target_backwards`) or a floor that would not move (`floor_refused`) — the
    /// defects the uptake gate exists to catch. Slots gained on the way there are the
    /// uptake; the released ground past the limit has no expert to hold.
    ///
    /// Measured on Qwen3-30B-A3B: a drain that released 3,120 MiB grew the zone by the
    /// 686 MiB it was short of the whole model and then answered `at_limit` 191 times —
    /// 21% by the ratio, and every byte it had a slot for.
    pub(crate) fn at_limit(&self) -> bool {
        let g = &self.0;
        g[6] > 0 && g[3] == 0 && g[4] == 0 && g[5] == 0
    }

    pub(crate) fn counts(&self) -> [u64; 8] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::GrowWindow;

    /// The Qwen3-30B-A3B Q4 probe on the RTX 3090 (2026-10-04): 3,086 slots gained
    /// before the drain — the load-time fill and phase A's regrowth — and the drain
    /// itself answered `at_limit` with the 16,992 MiB zone holding the whole model.
    #[test]
    fn a_drain_answered_only_at_limit_is_at_limit_whatever_came_before() {
        let start = [683, 330, 15_500, 0, 0, 0, 2, 3_086];
        let end = [1_334, 654, 31_015, 0, 0, 0, 651, 3_086];
        let w = GrowWindow::between(start, end);
        assert_eq!(w.counts(), [651, 324, 15_515, 0, 0, 0, 649, 0]);
        assert!(w.at_limit());
    }

    /// The same probe's next run: the drain began 686 MiB short of the whole model,
    /// gained 248 slots, and then answered `at_limit` — it took all it had room for.
    #[test]
    fn a_drain_that_grew_to_its_limit_is_at_limit() {
        let start = [1_102, 914, 7_003, 0, 0, 0, 141, 4_133];
        let end = [1_307, 926, 20_917, 0, 0, 0, 332, 4_381];
        let w = GrowWindow::between(start, end);
        assert_eq!(w.counts(), [205, 12, 13_914, 0, 0, 0, 191, 248]);
        assert!(w.at_limit());
    }

    /// Growth that never met the limit is scored on its uptake.
    #[test]
    fn a_drain_that_gained_without_reaching_its_limit_is_scored() {
        let w = GrowWindow::between([10, 0, 0, 0, 0, 0, 0, 100], [20, 0, 40, 0, 0, 0, 0, 140]);
        assert!(!w.at_limit());
    }

    #[test]
    fn a_target_below_the_zone_in_the_drain_is_never_excused() {
        let w = GrowWindow::between([0; 8], [9, 0, 30, 0, 1, 0, 5, 0]);
        assert!(!w.at_limit());
    }

    #[test]
    fn a_floor_that_refused_in_the_drain_is_never_excused() {
        let w = GrowWindow::between([0; 8], [9, 0, 30, 0, 0, 2, 5, 0]);
        assert!(!w.at_limit());
    }

    #[test]
    fn a_zone_with_room_that_did_not_move_is_never_excused() {
        let w = GrowWindow::between([0; 8], [9, 0, 30, 4, 0, 0, 5, 0]);
        assert!(!w.at_limit());
    }

    #[test]
    fn a_drain_with_no_growth_answer_is_not_at_limit() {
        let w = GrowWindow::between([5, 1, 2, 0, 0, 0, 7, 9], [5, 1, 2, 0, 0, 0, 7, 9]);
        assert!(!w.at_limit());
    }
}

//! Holding back a KV compaction pass that has nothing left to do.
//!
//! The per-wave gate opens on holes and sparse arenas, and a pool can carry both
//! while no pass is able to lower its frontier — measured live on Flash-Next, a
//! pinned arena at the top held `frontier_before=703 frontier_after=703`. The gate
//! stayed open, so the same pass ran after every decode wave: ~8,300 moves, nine
//! fresh arenas claimed and nine released, nothing reclaimed, ~145 ms of the loop
//! each time — a decode stall every ~2 s that bought nothing.
//!
//! A pass that ends where it began records the pools' occupancy here, and the next
//! one waits until that occupancy changes. Any arrival, eviction or seal that moves
//! a region re-opens the gate, so compaction still runs the moment there is
//! something new to pack.

/// The occupancy a pass is judged against: regions held, the frontier, and the
/// sparse arenas beneath it — the three figures the gate opens on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoolShape {
    pub live: usize,
    pub frontier: usize,
    pub sparse: usize,
}

/// The shape a pass last failed to improve, if the most recent pass did.
#[derive(Debug, Default)]
pub(super) struct CompactionStall {
    stalled_at: Option<PoolShape>,
}

impl CompactionStall {
    /// Whether a pass over `now` would repeat one that already made no progress
    /// on exactly this shape.
    pub fn holds(&self, now: PoolShape) -> bool {
        self.stalled_at == Some(now)
    }

    /// Record how a pass that started from `before` ended. A pass that reclaimed
    /// no region and left its frontier where it found it stalls the gate on
    /// `before`; any progress clears it. A pass its time budget clipped is
    /// progress whatever its frontier: it packed a prefix, and the next resumes
    /// from there.
    pub fn record(&mut self, before: PoolShape, outcome: PassOutcome) {
        let progressed = outcome.clipped
            || outcome.reclaimed_regions > 0
            || outcome.frontier_after < outcome.frontier_before;
        self.stalled_at = (!progressed).then_some(before);
    }
}

/// What a pass reports about its own progress.
#[derive(Debug, Clone, Copy)]
pub(super) struct PassOutcome {
    pub reclaimed_regions: usize,
    pub frontier_before: usize,
    pub frontier_after: usize,
    pub clipped: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: PoolShape = PoolShape {
        live: 690,
        frontier: 703,
        sparse: 9,
    };

    fn outcome(reclaimed_regions: usize, frontier_after: usize) -> PassOutcome {
        PassOutcome {
            reclaimed_regions,
            frontier_before: 703,
            frontier_after,
            clipped: false,
        }
    }

    #[test]
    fn a_clipped_pass_never_stalls_the_gate() {
        let mut stall = CompactionStall::default();
        stall.record(
            SHAPE,
            PassOutcome {
                clipped: true,
                ..outcome(0, 703)
            },
        );
        assert!(
            !stall.holds(SHAPE),
            "the next pass resumes where it stopped"
        );
    }

    #[test]
    fn a_pass_that_moved_nothing_holds_the_gate_on_that_shape() {
        let mut stall = CompactionStall::default();
        assert!(!stall.holds(SHAPE), "nothing has stalled yet");
        stall.record(SHAPE, outcome(0, 703));
        assert!(stall.holds(SHAPE));
    }

    #[test]
    fn any_change_in_occupancy_reopens_the_gate() {
        let mut stall = CompactionStall::default();
        stall.record(SHAPE, outcome(0, 703));
        for moved in [
            PoolShape { live: 691, ..SHAPE },
            PoolShape {
                frontier: 704,
                ..SHAPE
            },
            PoolShape { sparse: 8, ..SHAPE },
        ] {
            assert!(!stall.holds(moved), "{moved:?} is a new shape");
        }
    }

    #[test]
    fn a_pass_that_reclaimed_or_lowered_the_frontier_clears_the_stall() {
        let mut stall = CompactionStall::default();
        stall.record(SHAPE, outcome(0, 703));
        stall.record(SHAPE, outcome(2, 703));
        assert!(!stall.holds(SHAPE), "it reclaimed two regions");
        stall.record(SHAPE, outcome(0, 703));
        stall.record(SHAPE, outcome(0, 699));
        assert!(!stall.holds(SHAPE), "it lowered the frontier");
    }
}

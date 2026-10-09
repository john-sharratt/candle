//! Whether a compaction pass could lower the frontier, and holding back one that
//! already showed it could not.
//!
//! **A pass is worth its quiesce only if it can lower the frontier.** The frontier —
//! one past the highest live region — is what the weight side loses: the wave tier
//! stands above it and `weight_floor` is measured from there. Packing the KV pools
//! beneath a region another tenant holds moves holes around and lowers nothing.
//! Measured on Flash-Next's 128K prefill: with the QSA index and recurrent state at
//! the top, every KV pass moved the same 4,560 chunks, reclaimed nothing, left the
//! frontier where it was, and drained the device for 0.26–1.8 s before each — the
//! prefill's forwards serialised behind passes that bought nothing. So the gate asks
//! first *who stands at the frontier* ([`PoolShape::frontier_pass`]): a KV arena
//! there is a KV pass's to move, a span tenant's region is the span tenants' pass.
//!
//! **And not again over ground the last pass could not improve.** A pass that ends
//! where it began records the shape it started from and which pass it was, and the
//! gate holds *that* pass until something new is there to pack — a hole opened below
//! the frontier, or an arena gone sparse. It used to reopen on *any* change of
//! occupancy, and a prefill changes occupancy every forward: each one claims chunks
//! and moves the frontier up, so the guard never held while the turn grew, and the
//! same futile pass ran after every forward. Growth alone opens nothing to pack.
//!
//! **A stall belongs to the pass that stalled.** The other pass is a different
//! question about different ground: a span tenant that could not move says nothing
//! about whether the KV arena a later claim put on top of it can. One stall gating
//! both kept a KV pass that could lower the frontier shut until unrelated churn freed
//! a region.

/// Regions a pass must be able to recover before it is worth its copies: holes
/// below the frontier plus, for the KV pools, arenas a pack would empty.
///
/// **The same bar for both passes.** A span-tenant pass is a batch of whole-block
/// device copies — a recurrent state is ~3 MiB — and offered on a single hole it
/// lowered the frontier by one region at a time, counted that as progress, never
/// stalled, and reran at the gate's pace: the copy-heavy cadence that took the
/// probe's efficiency gate from 98–99% to 76–87%.
pub(super) const MIN_FREEABLE_ARENAS: usize = 8;

/// The occupancy a pass is judged against: regions held, the frontier, and the
/// sparse KV arenas beneath it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PoolShape {
    pub live: usize,
    pub frontier: usize,
    pub sparse: usize,
}

/// The pass that could lower the frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FrontierPass {
    /// A KV arena (band or record) stands at the frontier, and the pools hold
    /// enough air — holes below it, or arenas a pack would empty — to move it down.
    Kv,
    /// A span tenant's region stands at the frontier — recurrent state, the QSA
    /// index, the provenance gallery — over free ground it could move into.
    SpanTenants,
}

impl PoolShape {
    /// Free regions stranded below the frontier.
    pub fn holes(&self) -> usize {
        self.frontier.saturating_sub(self.live)
    }

    /// The pass that could lower this frontier, given `kv_top`, the highest region
    /// any KV arena holds. `None` when no pass could: nothing is live, or the KV
    /// side holds the top with too little air to move it, or a span tenant holds it
    /// over too few holes to be worth moving.
    pub fn frontier_pass(&self, kv_top: Option<usize>) -> Option<FrontierPass> {
        let top = self.frontier.checked_sub(1)?;
        if kv_top == Some(top) {
            (self.holes() + self.sparse >= MIN_FREEABLE_ARENAS).then_some(FrontierPass::Kv)
        } else {
            (self.holes() >= MIN_FREEABLE_ARENAS).then_some(FrontierPass::SpanTenants)
        }
    }
}

/// The pass that last failed to improve its ground, and the shape it started
/// from, if the most recent pass did.
#[derive(Debug, Default)]
pub(super) struct CompactionStall {
    stalled_at: Option<(FrontierPass, PoolShape)>,
}

impl CompactionStall {
    /// Whether `pass` over `now` would repeat one that already made no progress:
    /// the same pass stalled last, and nothing has been freed since — no new hole
    /// below the frontier and no arena gone sparser — whatever the frontier and the
    /// live count did in between.
    pub fn holds(&self, pass: FrontierPass, now: PoolShape) -> bool {
        self.stalled_at.is_some_and(|(stalled, s)| {
            stalled == pass && now.holes() <= s.holes() && now.sparse <= s.sparse
        })
    }

    /// Record how `pass`, started from `before`, ended. A pass that reclaimed no
    /// region and left its frontier where it found it stalls the gate on `before`;
    /// any progress clears it. A pass its time budget clipped is progress whatever
    /// its frontier: it packed a prefix, and the next resumes from there.
    pub fn record(&mut self, pass: FrontierPass, before: PoolShape, outcome: PassOutcome) {
        let progressed = outcome.clipped
            || outcome.reclaimed_regions > 0
            || outcome.frontier_after < outcome.frontier_before;
        self.stalled_at = (!progressed).then_some((pass, before));
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

    const KV: FrontierPass = FrontierPass::Kv;

    fn outcome(reclaimed_regions: usize, frontier_after: usize) -> PassOutcome {
        PassOutcome {
            reclaimed_regions,
            frontier_before: 703,
            frontier_after,
            clipped: false,
        }
    }

    /// A KV arena at the frontier with enough air is the KV pass's; one with too
    /// little is nobody's.
    #[test]
    fn a_kv_arena_at_the_frontier_is_the_kv_passes() {
        assert_eq!(SHAPE.frontier_pass(Some(702)), Some(FrontierPass::Kv));
        let tight = PoolShape {
            live: 700,
            frontier: 703,
            sparse: 4,
        };
        // 3 holes + 4 sparse arenas = 7, one short of a pass.
        assert_eq!(tight.frontier_pass(Some(702)), None);
    }

    /// The case that ran a futile pass after every 128K prefill forward: the KV
    /// pools sparse beneath a frontier a span tenant holds. That is the span
    /// tenants' pass to make, never the KV pools'.
    #[test]
    fn a_span_tenant_at_the_frontier_is_the_span_tenants_pass() {
        let index_on_top = PoolShape {
            live: 155,
            frontier: 167,
            sparse: 40,
        };
        assert_eq!(
            index_on_top.frontier_pass(Some(162)),
            Some(FrontierPass::SpanTenants)
        );
        // With no hole below it, there is nowhere for that tenant to go.
        let packed = PoolShape {
            live: 167,
            frontier: 167,
            sparse: 40,
        };
        assert_eq!(packed.frontier_pass(Some(162)), None);
        // And with no KV arena at all, the top is a span tenant's.
        assert_eq!(
            index_on_top.frontier_pass(None),
            Some(FrontierPass::SpanTenants)
        );
    }

    /// A span tenant over a single hole is not worth its copies: moving it would
    /// lower the frontier by one region, at the cost of whole-block copies, and
    /// the pass would never stall. It takes as many holes as a KV pass takes air.
    #[test]
    fn a_span_tenant_over_a_few_holes_has_no_pass() {
        let one_hole = PoolShape {
            live: 166,
            frontier: 167,
            sparse: 40,
        };
        assert_eq!(one_hole.frontier_pass(Some(162)), None);
        let seven = PoolShape {
            live: 160,
            ..one_hole
        };
        assert_eq!(seven.frontier_pass(Some(162)), None);
        let eight = PoolShape {
            live: 159,
            ..one_hole
        };
        assert_eq!(
            eight.frontier_pass(Some(162)),
            Some(FrontierPass::SpanTenants)
        );
    }

    /// **A stall belongs to the pass that stalled.** A span tenant that could not
    /// move holds the span-tenant pass; a KV arena a later claim put on top is the
    /// KV pass's, which nothing has tried.
    #[test]
    fn a_stall_holds_only_the_pass_that_stalled() {
        let mut stall = CompactionStall::default();
        stall.record(FrontierPass::SpanTenants, SHAPE, outcome(0, 703));
        assert!(stall.holds(FrontierPass::SpanTenants, SHAPE));
        assert!(!stall.holds(FrontierPass::Kv, SHAPE));
    }

    /// Nothing live, nothing to lower.
    #[test]
    fn an_empty_span_has_no_pass() {
        let empty = PoolShape {
            live: 0,
            frontier: 0,
            sparse: 0,
        };
        assert_eq!(empty.frontier_pass(None), None);
    }

    #[test]
    fn a_clipped_pass_never_stalls_the_gate() {
        let mut stall = CompactionStall::default();
        stall.record(
            KV,
            SHAPE,
            PassOutcome {
                clipped: true,
                ..outcome(0, 703)
            },
        );
        assert!(
            !stall.holds(KV, SHAPE),
            "the next pass resumes where it stopped"
        );
    }

    #[test]
    fn a_pass_that_moved_nothing_holds_the_gate_on_that_shape() {
        let mut stall = CompactionStall::default();
        assert!(!stall.holds(KV, SHAPE), "nothing has stalled yet");
        stall.record(KV, SHAPE, outcome(0, 703));
        assert!(stall.holds(KV, SHAPE));
    }

    /// The prefill that grows: every forward claims chunks and lifts the frontier,
    /// and none of that is new ground to pack. The stall holds through it.
    #[test]
    fn growth_alone_does_not_reopen_the_gate() {
        let mut stall = CompactionStall::default();
        stall.record(KV, SHAPE, outcome(0, 703));
        for grown in [
            PoolShape {
                live: 700,
                frontier: 713,
                sparse: 9,
            },
            PoolShape {
                live: 760,
                frontier: 765,
                sparse: 8,
            },
        ] {
            assert!(stall.holds(KV, grown), "{grown:?} freed nothing");
        }
    }

    /// Ground freed below the frontier is new work: a hole opened, or an arena
    /// went sparse.
    #[test]
    fn freed_ground_reopens_the_gate() {
        let mut stall = CompactionStall::default();
        stall.record(KV, SHAPE, outcome(0, 703));
        for freed in [
            PoolShape { live: 689, ..SHAPE },
            PoolShape {
                sparse: 10,
                ..SHAPE
            },
        ] {
            assert!(
                !stall.holds(KV, freed),
                "{freed:?} has something new to pack"
            );
        }
    }

    #[test]
    fn a_pass_that_reclaimed_or_lowered_the_frontier_clears_the_stall() {
        let mut stall = CompactionStall::default();
        stall.record(KV, SHAPE, outcome(0, 703));
        stall.record(KV, SHAPE, outcome(2, 703));
        assert!(!stall.holds(KV, SHAPE), "it reclaimed two regions");
        stall.record(KV, SHAPE, outcome(0, 703));
        stall.record(KV, SHAPE, outcome(0, 699));
        assert!(!stall.holds(KV, SHAPE), "it lowered the frontier");
    }
}

//! Which arenas a relocation pass should empty, which arenas may receive their
//! chunks, and whether the pass is worth running at all.
//!
//! # Two jobs, and this is only one of them
//!
//! Returning ground to the weight side takes two things: arenas have to become
//! **empty**, and the arenas that remain have to sit **low** in the span, because
//! what the weight zone and the wave tier grow into is
//! `weight_floor − live_end()` and `live_end` is the highest live region.
//!
//! Those are separate jobs and the engine already has a safe answer to the
//! second. `compact_arenas_down` relocates whole slabs toward the low end, and it
//! changes no chunk's identity at all — nothing has to be found, nothing has to
//! be repointed, and it has never corrupted anything. So the frontier is
//! compaction's business.
//!
//! What compaction cannot do is make an arena empty, because it moves slabs
//! rather than their contents. That is this pass's whole job, and framing it that
//! way makes the policy obvious: **drain the arenas that are cheapest to drain**.
//! An arena holding 8 live slots costs 8 copies and returns a whole arena; one
//! holding 2,048 costs 2,048 and returns exactly the same arena.
//!
//! # What the first version got wrong
//!
//! It chose the keep set by walking up from the lowest rank accumulating capacity
//! until it covered the class's live total, and evacuated everything above. That
//! ranks by *benefit* and ignores *cost* — and since the sparse arenas are
//! frequently the low ones, it asked full arenas high in the span to empty into
//! sparse arenas low in it.
//!
//! Measured on run 58, class 8192: six arenas at 100% held 12,288 of 19,360 live
//! slots while twenty-five arenas sat at 0.4–13% holding ~7,072 between them. The
//! pass moved 387,759 bands — about 48,470 chunks — against a sparse population of
//! roughly 884 chunks. Fifty-five times more copying than the cheap work
//! contained, and arena headroom went from 14 to 97 while it ran.
//!
//! # The policy
//!
//! Donors are arenas below [`SPARSE_NUMERATOR`]/[`SPARSE_DENOMINATOR`] occupancy,
//! taken **cheapest first**. Recipients are every other arena with room, taken
//! **lowest rank first**, so chunks still land low without that being the
//! selection criterion. A donor is never a recipient, which is what stops the
//! ping-pong the previous arena-order pass suffered — draining an arena is
//! precisely what makes it the one with the most free slots.

use super::gid_pool::ArenaOccupancy;

/// Arenas a pass must expect to empty before it is worth its copies.
///
/// The unit is arenas because an arena is what `release_empty_arenas` can hand
/// back and what pins a region. A pass that frees none has bought nothing.
pub const MIN_RELOCATION_GAIN: usize = 4;

/// Occupancy at or below which an arena is worth draining, as a fraction.
///
/// A quarter: past that the copies stop being cheap relative to the one arena
/// they return, and the ground is better left for compaction to slide downward
/// intact. Run 58's class 8192 had twenty-five arenas under 13%, so the cheap
/// population is real and this bar is not the binding constraint.
pub const SPARSE_NUMERATOR: usize = 1;
/// Denominator of the sparsity bar — see [`SPARSE_NUMERATOR`].
pub const SPARSE_DENOMINATOR: usize = 4;

/// The arenas one size class should drain, and where their chunks may go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelocationPlan {
    /// Arena indices to drain, cheapest (fewest live slots) first.
    pub evacuate: Vec<usize>,
    /// `(arena, free slots)` a chunk may move into, lowest rank first. Never
    /// contains a donor.
    pub recipients: Vec<(usize, usize)>,
    /// Slot stride of the class this plan covers.
    pub slot_bytes: usize,
}

impl RelocationPlan {
    /// Whether `arena_idx` is one this pass wants drained.
    pub fn wants(&self, arena_idx: usize) -> bool {
        self.evacuate.contains(&arena_idx)
    }

    /// Claim a destination slot, answering the arena to allocate it from.
    ///
    /// Returns `None` when every recipient is full, which is the caller's signal
    /// to leave the band where it is rather than let the allocator create a
    /// fresh arena — a pass that ends by adding an arena has moved chunks to no
    /// purpose.
    ///
    /// The room is spent optimistically, so a caller whose allocation then fails
    /// must hand it back with [`Self::release`] — otherwise the budget drifts
    /// below what the arenas actually hold and donors the planner proved
    /// affordable end up half-drained, which frees nothing.
    pub fn claim(&mut self) -> Option<usize> {
        let (idx, free) = self.recipients.first_mut()?;
        let target = *idx;
        *free -= 1;
        if *free == 0 {
            self.recipients.remove(0);
        }
        Some(target)
    }

    /// Return a slot claimed from `arena_idx` that the caller could not use.
    ///
    /// Restores the arena to the front of the queue when the failed claim was
    /// its last: it is still the lowest-ranked recipient, and dropping it for
    /// good would push the next chunk higher up the span for no reason.
    pub fn release(&mut self, arena_idx: usize) {
        if let Some((_, free)) = self.recipients.iter_mut().find(|(i, _)| *i == arena_idx) {
            *free += 1;
            return;
        }
        self.recipients.insert(0, (arena_idx, 1));
    }
}

/// Plan one size class from its arena census, or `None` when it is not worth a
/// pass.
///
/// `rows` may hold every class; only those matching `slot_bytes` are read.
pub fn plan_class(rows: &[ArenaOccupancy], slot_bytes: usize) -> Option<RelocationPlan> {
    let mine: Vec<&ArenaOccupancy> = rows.iter().filter(|a| a.slot_bytes == slot_bytes).collect();
    if mine.len() < 2 {
        // One arena cannot be packed into anything.
        return None;
    }

    // **Cheapest first.** Every donor returns one arena, so the only thing that
    // separates them is what draining costs. An already-empty arena is skipped:
    // `release_empty_arenas` takes it without a single copy, and counting it
    // would let a class needing no work look like one that does.
    let mut donors: Vec<&ArenaOccupancy> = mine
        .iter()
        .copied()
        .filter(|a| a.live > 0 && a.live * SPARSE_DENOMINATOR <= a.capacity * SPARSE_NUMERATOR)
        .collect();
    donors.sort_unstable_by_key(|a| a.live);
    if donors.len() < MIN_RELOCATION_GAIN {
        return None;
    }

    // Recipients: anything that is not a donor and has room, lowest in the span
    // first. Rank orders them because a chunk may as well land low, but it is
    // not what selects them — excluding donors is, and that is what stops a
    // drained arena from becoming the obvious place to put the next chunk.
    let donor_ids: std::collections::HashSet<usize> = donors.iter().map(|a| a.arena_idx).collect();
    let mut recipients: Vec<(usize, usize, usize)> = mine
        .iter()
        .filter(|a| !donor_ids.contains(&a.arena_idx))
        .map(|a| (a.rank, a.arena_idx, a.capacity.saturating_sub(a.live)))
        .filter(|(_, _, free)| *free > 0)
        .collect();
    recipients.sort_unstable_by_key(|(rank, _, _)| *rank);

    // **Trim the donors to what the recipients can actually take.** A donor only
    // pays when it empties *completely*, so a partial drain is pure cost. Taking
    // them cheapest-first and stopping at the first that does not fit keeps every
    // copy this pass makes attached to an arena it will finish.
    let mut room: usize = recipients.iter().map(|(_, _, free)| *free).sum();
    let mut affordable: Vec<usize> = Vec::new();
    for d in &donors {
        if d.live > room {
            break;
        }
        room -= d.live;
        affordable.push(d.arena_idx);
    }
    if affordable.len() < MIN_RELOCATION_GAIN {
        return None;
    }

    Some(RelocationPlan {
        evacuate: affordable,
        recipients: recipients
            .into_iter()
            .map(|(_, idx, free)| (idx, free))
            .collect(),
        slot_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(arena_idx: usize, rank: usize, capacity: usize, live: usize) -> ArenaOccupancy {
        ArenaOccupancy {
            arena_idx,
            slot_bytes: 4096,
            capacity,
            live,
            rank,
        }
    }

    /// Run 58's class 8192, in miniature: a few full arenas holding most of the
    /// data, many nearly-empty ones holding little. The cheap ones drain; the
    /// full ones are left for compaction to slide down intact.
    #[test]
    fn the_sparse_arenas_drain_and_the_full_ones_are_left_alone() {
        let rows = vec![
            arena(281, 20, 2048, 2048),
            arena(303, 21, 2048, 2048),
            // The mid-occupancy arenas are where the drained chunks go: too full
            // to be worth draining, not full enough to be useless as a
            // destination. Run 58's class 8192 had four of these (68%, 79%, 42%,
            // 36%) against twenty-five sparse ones.
            arena(30, 5, 2048, 1408),
            arena(251, 6, 2048, 872),
            arena(1, 0, 2048, 264),
            arena(5, 1, 2048, 176),
            arena(20, 2, 2048, 88),
            arena(240, 3, 2048, 8),
        ];
        let plan = plan_class(&rows, 4096).expect("four sparse arenas is worth a pass");
        assert_eq!(
            plan.evacuate,
            vec![240, 20, 5, 1],
            "cheapest first, and no full or mid-occupancy arena among them",
        );
        assert!(
            !plan.wants(281) && !plan.wants(303),
            "a full arena costs 2,048 copies to return the same one arena",
        );
        assert!(
            !plan.wants(30) && !plan.wants(251),
            "a mid-occupancy arena is a destination, not a donor",
        );
    }

    /// A donor must never be a destination, or draining it makes it the most
    /// attractive place to put the next chunk and the pass chases its own tail.
    #[test]
    fn a_donor_is_never_a_recipient() {
        let rows = vec![
            arena(10, 0, 100, 50),
            arena(11, 1, 100, 1),
            arena(12, 2, 100, 2),
            arena(13, 3, 100, 3),
            arena(14, 4, 100, 4),
        ];
        let plan = plan_class(&rows, 4096).expect("four sparse arenas");
        for (idx, _) in &plan.recipients {
            assert!(!plan.wants(*idx), "arena {idx} is both donor and recipient");
        }
        assert_eq!(
            plan.recipients,
            vec![(10, 50)],
            "only the non-donor has room"
        );
    }

    /// Destinations are handed out lowest-rank first, so chunks land low even
    /// though rank is not what picks the donors.
    #[test]
    fn destinations_are_handed_out_lowest_in_the_span_first() {
        let rows = vec![
            arena(90, 7, 100, 90),
            arena(80, 2, 100, 95),
            arena(11, 10, 100, 1),
            arena(12, 11, 100, 1),
            arena(13, 12, 100, 1),
            arena(14, 13, 100, 1),
        ];
        let mut plan = plan_class(&rows, 4096).expect("four sparse arenas");
        assert_eq!(plan.claim(), Some(80), "rank 2 before rank 7");
        // Arena 80 has 5 free; take the rest, then it must fall out.
        for _ in 0..4 {
            assert_eq!(plan.claim(), Some(80));
        }
        assert_eq!(plan.claim(), Some(90), "exhausted recipients are dropped");
    }

    /// Room bounds the pass: donors that will not fit are dropped rather than
    /// half-drained, because a half-drained arena frees nothing.
    #[test]
    fn donors_that_do_not_fit_are_dropped_not_half_drained() {
        // 6 free slots in the only recipient; donors cost 1,2,3,4.
        let rows = vec![
            arena(10, 0, 100, 94),
            arena(11, 1, 100, 1),
            arena(12, 2, 100, 2),
            arena(13, 3, 100, 3),
            arena(14, 4, 100, 4),
        ];
        assert_eq!(
            plan_class(&rows, 4096),
            None,
            "1+2+3 fits but 4 does not, leaving three arenas of gain — under the floor",
        );
    }

    /// Arenas already empty need no copies and must not inflate the gain.
    #[test]
    fn empty_arenas_are_not_donors() {
        let rows = vec![
            arena(10, 0, 100, 50),
            arena(11, 1, 100, 0),
            arena(12, 2, 100, 0),
            arena(13, 3, 100, 0),
            arena(14, 4, 100, 0),
            arena(15, 5, 100, 0),
        ];
        assert_eq!(
            plan_class(&rows, 4096),
            None,
            "five reclaimable arenas, not one chunk to move",
        );
    }

    /// A well-filled class has no cheap arenas and is refused outright.
    #[test]
    fn a_dense_class_is_refused() {
        let rows: Vec<ArenaOccupancy> = (0..8).map(|i| arena(i, i, 100, 80)).collect();
        assert_eq!(
            plan_class(&rows, 4096),
            None,
            "nothing under a quarter full"
        );
    }

    /// A plan is per class: a band only fits a slot of its own stride.
    #[test]
    fn other_classes_are_not_planned_into_this_one() {
        let mut rows: Vec<ArenaOccupancy> = (0..6).map(|i| arena(i, i, 100, 2)).collect();
        rows.push(arena(99, 9, 100, 90));
        for a in rows.iter_mut().take(2) {
            a.slot_bytes = 8192;
            a.arena_idx += 100;
        }
        let plan = plan_class(&rows, 4096).expect("four 4096 donors remain");
        assert!(
            plan.evacuate.iter().all(|i| *i < 100),
            "no 8192 arena planned for a 4096 pass: {:?}",
            plan.evacuate,
        );
    }

    /// One arena is the degenerate case and never reads as a pass.
    #[test]
    fn a_single_arena_class_is_never_worth_a_pass() {
        assert_eq!(plan_class(&[arena(10, 0, 100, 5)], 4096), None);
    }
}

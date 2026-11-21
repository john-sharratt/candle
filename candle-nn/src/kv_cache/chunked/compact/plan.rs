//! Sets the budget for a pass and plans the moves that empty the frontier.
//!
//! Pools are planned highest arena first, because the topmost arena is the one
//! that costs the weight side ground. Each pool's plan is capped, so a pass can
//! empty a whole arena, and planning takes a quarter of the budget while claiming
//! takes the rest. A clipped pass resumes where it stopped. A capped plan is
//! reported apart from a clipped one, because it counts as progress only when the
//! pass also reclaimed something.

use super::*;

/// Claims per batch before the clock is consulted.
///
/// A time budget, not a move cap, because chunks are not equal: the ladder runs
/// from 320 B to 16 KiB, so the same "one move" is fifty times the bandwidth at
/// one rung than another and a count cannot bound a duration. Batching keeps the
/// clock check off the per-claim path while staying fine-grained enough that a
/// budget of a few milliseconds is respected.
pub(super) const MOVES_PER_BATCH: usize = 512;

/// Moves one pass plans per pool — no more than the pass can claim.
///
/// **Planning what the budget cannot claim is what stopped compaction.** The walk is
/// ~0.12 µs a move and the claim ~0.55 µs, so a pass spends its budget on whichever
/// it is handed more of. Planned without a cap, a fragmented pool at a high frontier
/// produced 570,000–770,000 moves: the walk alone took 75–100 ms of the 80 ms budget,
/// the pass claimed only its one guaranteed batch of 512, and a top arena holding
/// 768–1,280 live chunks was never emptied — the frontier held at 376–517 regions
/// pass after pass, and Qwen3-30B-A3B's engine probe sat at 36% efficiency on the
/// RTX 4090 Laptop. The passes that did lower it had planned 75,000–145,000 and
/// claimed 60,000–115,000.
///
/// **Capping a pool keeps exactly the moves that matter.** The two-cursor walk takes
/// its sources off the right cursor, highest address first, so a capped plan is the
/// pool's top arenas — the ones holding the frontier — and what it drops is the low
/// tail the budget would never have reached. A capped plan marks the pass clipped, so
/// the next resumes from the shorter distance. At this cap even four active pools
/// plan in ~16 ms, leaving the rest of the budget for ~100,000 claims.
///
/// A capped plan is reported as `plan_capped`, not as `clipped`: it is only progress
/// when the pass also reclaimed something.
///
/// Sized for a pass of [`CAP_BUDGET`]; a pass given a longer budget plans in
/// proportion, so it claims what that many passes would have.
pub(super) const PLANNED_MOVES_PER_POOL: usize = 32_768;

/// The pass budget [`PLANNED_MOVES_PER_POOL`] was measured against.
pub(super) const CAP_BUDGET: Duration = Duration::from_millis(80);

/// One pool's plan cap for a pass of `budget`: [`PLANNED_MOVES_PER_POOL`] scaled by
/// `budget / CAP_BUDGET`, or one whole arena of the pool when that is more.
///
/// **Scaled with the budget, so one long pass does the work of several short
/// ones.** What a pass's holders cost to sweep does not depend on what it moved —
/// the substrate walks every residence whatever the map holds — so a caller that
/// would otherwise run passes back to back buys the same moves for one sweep by
/// running one pass of their combined budget.
///
/// **A pass must be able to empty the arena holding the frontier, not just drain
/// it.** Allocation is leftmost by address, so once a pass has packed every lower
/// arena of a pool full, the only free slots in the pool are the ones it just
/// vacated at the top — and the next allocations land exactly there. An arena left
/// part-drained is refilled before the next pass reaches it. Capped below an
/// arena's capacity, the smallest rung (52,428 slots an arena) could never be
/// emptied in one pass: on Qwen3-30B-A3B the class-0 arena at the frontier was
/// planned first on pass after pass, and its live count went 2,499 → 7,266 →
/// 4,957 → 2,750 → 5,228 with the frontier pinned at 364 regions.
pub(super) fn planned_moves_cap(key: ArenaKey, budget: Duration) -> usize {
    let scaled =
        (PLANNED_MOVES_PER_POOL as u128 * budget.as_micros() / CAP_BUDGET.as_micros()) as usize;
    scaled.max(key.chunks())
}

/// The copy plan, as the migration kernel wants it: one entry per relocated slot,
/// in three parallel arrays.
///
/// Built entirely on the host by [`ChunkedKvBacking::claim_moves`] and issued in a
/// single launch, so the whole pass costs two kernel calls — this copy and the
/// band-pointer patch — however many chunks it relocates.
#[derive(Default)]
pub(super) struct CopyRecords {
    pub(super) srcs: Vec<i64>,
    pub(super) dsts: Vec<i64>,
    pub(super) lens: Vec<i64>,
    /// Moves dropped because the slot claimed for them was one this pass had already
    /// planned to read — see the note in [`ChunkedKvBacking::claim_moves`].
    ///
    /// Carried here rather than counted only in the global tally so a single pass's
    /// line says whether the collision it declined was real. Without that the
    /// disappearance of a fault is unattributable: the guard that would have caught
    /// the collision downstream finds nothing precisely *because* this declined it,
    /// so silence proves both "it fired" and "there was nothing to fire at".
    pub(super) source_collisions: usize,
}

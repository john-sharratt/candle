//! How much of a character's conversation stays verbatim in the redo log.
//!
//! # The bound nothing else was providing
//!
//! Three bounds sit on a character's conversation, and none of them was on the
//! substrate:
//!
//! | Bound | What it limits |
//! |---|---|
//! | [`crate::engine::mind::CONTEXT_WINDOW_TURNS`] | what is prefilled onto the GPU per turn |
//! | [`crate::engine::window::DEFAULT_TURNS`] | what perception carries verbatim |
//! | **this** | what stays live in the redo log |
//!
//! An NPC never stops living, so its timeline never stops growing — and an
//! untombstoned turn is a turn compaction can *never* reclaim. The compactor
//! only relocates a sealed segment once enough of it is dead
//! (`SEGMENT_COMPACT_MIN_DEAD`, 10%), and a log where every turn stays live
//! never reaches that threshold on any segment. It runs, finds nothing, and the
//! log grows at whatever rate the cast talks.
//!
//! Measured, not theorised: a cast of sixteen mostly-idle Makers wrote **~370 GB
//! in under two hours** and stopped only when the disk filled — 39
//! four-gigabyte segments, none of which could be reclaimed because not one of
//! them held a dead record.
//!
//! # A hole, not a deletion
//!
//! Retiring a turn writes a **turn-scoped tombstone**, which is a different
//! thing from retiring a conversation. The timeline stays live and the turn
//! stays in the index; what goes is the turn's bulk content. Compaction sheds
//! the records, keeps the `StreamDecl`, and a reload restores the turn as an
//! empty hole — so turn numbering is stable across the drop and nothing
//! downstream renumbers or dangles.
//!
//! # The sweep works oldest-first, from a watermark
//!
//! Each insert asks *what below the horizon is not yet tombstoned*, rather than
//! assuming exactly one turn fell out since the last insert. The two agree while
//! nothing goes wrong, and differ in every case where something does: a write
//! that failed, a daemon restarted mid-conversation, a turn count that resumed
//! from a reload. Counting forward would skip those turns permanently — they sit
//! below the horizon forever after, so nothing would look at them again.
//!
//! **Oldest-first, and that direction is load-bearing.** Sweeping *down* from
//! the horizon and stopping at the first turn already tombstoned reads as the
//! cheaper walk, and it is — until a gap is longer than one insert's budget.
//! Then it tombstones a contiguous block in the *middle*, the next insert stops
//! dead at the top of that block, and everything below it stays live for good.
//! Going up from the watermark keeps the tombstoned run contiguous from turn
//! zero, which is the invariant that makes stopping early safe at all.
//!
//! The watermark is per conversation. A conversation this process opened starts
//! at zero; one it **rejoined** starts at the first turn still live, found by
//! [`resume_watermark`] — the sweep would otherwise spend the first few hundred
//! turns of every restart re-walking ground it retired before the process
//! started, which a conversation that never ends only makes worse.
//!
//! # Opt-in, because most conversations are finite
//!
//! An assistant conversation ends, and its transcript is the product — retiring
//! its tail would delete the thing the user came for. A mind that runs forever
//! is the unusual case, so it asks for this rather than being given it.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

/// How many turns of a character's conversation stay verbatim on disk.
///
/// **Chosen to sit above every other bound with room to spare.** The perception
/// window carries 64 turns and the GPU tail 32 exchanges — the same 64 turns
/// counted the other way — so at 160 a turn is retired only once it is more
/// than twice as old as anything that would still read it. The margin is the
/// point, and it is what moves when either window does: retention is the one
/// bound whose mistake is unrecoverable, since a dropped turn does not come
/// back.
pub const KEEP_TURNS: u64 = 160;

/// Retention must stay clear of both live windows, or a turn is retired out
/// from under something still reading it. Held at compile time for the same
/// reason `mind.rs` holds its own pair that way: a change to any of the three
/// has to reckon with the others even in a build nobody runs the tests for.
const _: () = assert!(KEEP_TURNS > crate::engine::window::DEFAULT_TURNS as u64);
const _: () = assert!(KEEP_TURNS > (crate::engine::mind::CONTEXT_WINDOW_TURNS as u64) * 2);

/// How many turns below the horizon one insert will look back over.
///
/// A bound on the work, not on the correctness: a gap longer than this is closed
/// over the next few inserts rather than all at once, because a single insert
/// stalling on ten thousand tombstone writes would be a worse failure than the
/// slow catch-up it was trying to avoid.
pub const LOOK_BACK: u32 = 256;

/// The newest turn that has fallen out of retention, if any.
///
/// Turn indices are zero-based, so a conversation of `n` turns holds `0..n`, the
/// oldest kept is `n - keep`, and the newest expired is one below that.
pub fn horizon(turn_count: u64, keep: u64) -> Option<u32> {
    turn_count
        .checked_sub(keep)
        .and_then(|dropped| dropped.checked_sub(1))
        .map(|i| i as u32)
}

/// Where the sweep should start on a conversation this process did not open.
///
/// Zero is always *correct* — an already-tombstoned turn costs a lookup and no
/// write, so a sweep from the bottom converges on the truth. What it is not is
/// affordable forever. [`retire_expired`] looks back at most [`LOOK_BACK`] turns
/// per insert, so a resumed conversation a million turns deep would spend four
/// thousand of its turns walking ground that was already retired before it could
/// reach the ground that was not — and a mind that never stops living is exactly
/// the conversation that gets that deep. Resume is what makes this reachable at
/// all: before it, every restart began at turn zero of a brand-new timeline.
///
/// A binary search is available because [`retire_expired`] sweeps **oldest-first
/// and stops on failure**, which makes the tombstoned turns a contiguous run
/// from turn zero — the same invariant that lets the sweep stop early. So the
/// watermark is the partition point, and finding it costs `log2(turns)` lookups
/// instead of one per turn.
///
/// If the invariant were ever broken — a gap left by something other than this
/// module — the search lands inside the run rather than at its top, and the
/// ordinary oldest-first sweep closes the rest over the next few inserts. The
/// bad case is slow, not wrong.
pub fn resume_watermark(engine: &ConversationEngine, timeline: TimelineId, turn_count: u64) -> u32 {
    let at = first_live_turn(turn_count, |turn| engine.is_turn_tombstoned(timeline, turn));
    if at > 0 {
        tracing::info!(
            watermark = at,
            turn_count,
            "resumed conversation is already retired up to here; the sweep starts above it"
        );
    }
    at
}

/// The search itself, over the answer rather than over an engine.
///
/// Separated so the walk is testable without a model on a GPU behind it — the
/// property being checked is arithmetic, and arranging a real conversation of a
/// million turns to check it is not.
fn first_live_turn(turn_count: u64, tombstoned: impl Fn(u32) -> bool) -> u32 {
    // Turn indices are `u32`; a conversation deeper than that is not a case this
    // saturates into wrongly — it clamps to the top of the index space, which is
    // past every turn, and the sweep then has nothing to do.
    let mut lo = 0u32;
    let mut hi = u32::try_from(turn_count).unwrap_or(u32::MAX);
    while lo < hi {
        // Halfway *between*, not `(lo + hi) / 2`: the sum of two indices near
        // the top of the space overflows, and a conversation that deep is the
        // one this exists for.
        let mid = lo + (hi - lo) / 2;
        if tombstoned(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Tombstone everything from `watermark` up to the horizon, oldest first.
///
/// `watermark` is the first turn this conversation has *not* yet retired; it is
/// advanced past everything this sweep dealt with and handed back, so the next
/// insert resumes where this one stopped. Starting it at zero is always safe —
/// an already-tombstoned turn costs a lookup and no write.
///
/// Returns the new watermark and how many turns were actually written. Failure
/// is logged rather than propagated, on the same reasoning as
/// [`crate::engine::mind`]'s retirement path: a tombstone that did not land
/// leaves one turn's bulk on disk and the next insert tries again, whereas
/// refusing to let a character think because its log could not be trimmed would
/// be far worse. The watermark does **not** advance past a failure, so the retry
/// is real rather than nominal.
pub fn retire_expired(
    engine: &ConversationEngine,
    timeline: TimelineId,
    turn_count: u64,
    keep: u64,
    watermark: u32,
) -> (u32, u32) {
    // **Say so when there is nothing to do.**
    //
    // Retention's failure mode is silence: it only spoke when it retired
    // something, so "the horizon has never opened" and "working perfectly" were
    // the same empty log. A live daemon ran for days that way — 37,705 turn
    // streams on disk, 70 tombstones between them, and not one line saying the
    // sweep had considered anything. The counts are cheap and they are the
    // whole diagnosis.
    let Some(horizon) = horizon(turn_count, keep) else {
        tracing::debug!(
            turn_count,
            keep,
            "nothing below the retention horizon yet — the conversation is still \
             shorter than what it keeps"
        );
        return (watermark, 0);
    };
    tracing::debug!(
        turn_count,
        keep,
        horizon,
        watermark,
        "retiring the tail below the horizon"
    );
    let mut at = watermark;
    let mut retired = 0;
    let mut budget = LOOK_BACK;
    while at <= horizon && budget > 0 {
        budget -= 1;
        if engine.is_turn_tombstoned(timeline, at) {
            at += 1;
            continue;
        }
        match engine.tombstone_turn(timeline, at) {
            Ok(()) => {
                retired += 1;
                at += 1;
            }
            Err(e) => {
                // The watermark stays put, so this turn is the first one the
                // next sweep tries. Logged once — a broken log fails every
                // write, and one line beats two hundred and fifty-six.
                tracing::warn!(
                    turn = at,
                    "turn could not be retired: {e:?} — its bulk stays live in the log, and \
                     the next insert will try again"
                );
                break;
            }
        }
    }
    if retired > 0 {
        tracing::debug!(
            retired,
            through = at.saturating_sub(1),
            kept = keep,
            "turns retired from the log; the conversation keeps its shape"
        );
    }
    (at, retired)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_has_expired_until_the_conversation_is_deeper_than_the_horizon() {
        for n in 0..=KEEP_TURNS {
            assert_eq!(horizon(n, KEEP_TURNS), None, "at {n} turns");
        }
    }

    #[test]
    fn the_first_turn_past_the_horizon_is_the_very_first_turn() {
        // One turn more than it keeps: turn 0 is the one that fell out.
        assert_eq!(horizon(KEEP_TURNS + 1, KEEP_TURNS), Some(0));
    }

    #[test]
    fn the_horizon_advances_one_turn_at_a_time() {
        let mut seen = Vec::new();
        for n in KEEP_TURNS + 1..KEEP_TURNS + 20 {
            seen.push(horizon(n, KEEP_TURNS).expect("past the horizon"));
        }
        let expected: Vec<u32> = (0..19).collect();
        assert_eq!(seen, expected);
    }

    /// **The horizon must name a turn that exists.**
    ///
    /// It is an index into turns numbered `0..turn_count`, so it can never
    /// reach the newest turn and can never run past the end. That held only
    /// while the count it is given advances at the same rate as the indices,
    /// and for a long time it did not: the caller passed
    /// `Sequence::turn_count`, which counts a user and an assistant message
    /// separately, so the horizon moved two turns for every one that existed.
    ///
    /// Measured live: `turn_count=65 → horizon=0`, and by 65 real exchanges the
    /// horizon had passed the last turn — every turn retired, `keep_turns: 64`
    /// keeping nothing. The count now comes from the substrate
    /// (`ConversationEngine::timeline_turn_count`), which counts the way the
    /// indices do.
    #[test]
    fn the_horizon_never_names_a_turn_the_conversation_does_not_have() {
        for turns in KEEP_TURNS + 1..KEEP_TURNS + 500 {
            let h = horizon(turns, KEEP_TURNS).expect("past the horizon") as u64;
            assert!(
                h < turns,
                "horizon {h} is not a turn of a {turns}-turn conversation"
            );
            // And it always leaves exactly `keep` turns standing.
            assert_eq!(turns - (h + 1), KEEP_TURNS, "at {turns} turns");
        }
    }

    #[test]
    fn what_stays_is_always_exactly_the_horizon_however_long_the_conversation_runs() {
        // The property that makes this "bounded" rather than "slower-growing".
        for n in [KEEP_TURNS + 1, KEEP_TURNS + 500, 100_000] {
            let newest_retired = horizon(n, KEEP_TURNS).unwrap() as u64;
            let oldest_kept = newest_retired + 1;
            let newest_held = n - 1;
            assert_eq!(newest_held - oldest_kept + 1, KEEP_TURNS, "at {n} turns");
        }
    }

    #[test]
    fn a_horizon_of_zero_never_underflows() {
        // Not a configuration anybody should choose — the schema loader refuses
        // it — but it must not panic if one arrives another way.
        assert_eq!(horizon(0, 0), None, "an empty conversation expired a turn");
        assert_eq!(horizon(1, 0), Some(0));
        assert_eq!(horizon(9, 0), Some(8));
    }

    #[test]
    fn retention_stays_clear_of_every_live_window() {
        // The compile-time assertions above cover this; stating it as a test is
        // what makes the *reason* visible when one of them fires.
        assert!(
            KEEP_TURNS > crate::engine::window::DEFAULT_TURNS as u64,
            "a turn would be retired while perception still carries it"
        );
        assert!(
            KEEP_TURNS > crate::engine::mind::CONTEXT_WINDOW_TURNS as u64 * 2,
            "a turn would be retired while the GPU tail still prefills it"
        );
    }

    /// A resumed conversation starts its sweep above what it already retired.
    ///
    /// The whole point: without it, [`retire_expired`]'s [`LOOK_BACK`] budget
    /// means a conversation a million turns deep spends ~4,000 inserts walking
    /// ground that was retired before the process started.
    #[test]
    fn a_resumed_sweep_starts_at_the_first_turn_still_live() {
        for retired in [0u32, 1, 63, 64, 1_000, 999_999] {
            let at = first_live_turn(1_000_000, |turn| turn < retired);
            assert_eq!(at, retired, "with {retired} turns already retired");
        }
    }

    /// A conversation whose every turn is retired has nothing above the
    /// watermark, and must say so rather than pointing at a turn that is not
    /// there — the sweep would then re-tombstone from the top forever.
    #[test]
    fn a_wholly_retired_conversation_puts_the_watermark_past_its_last_turn() {
        assert_eq!(first_live_turn(500, |_| true), 500);
    }

    /// Nothing retired — the ordinary first run — starts at zero, which is what
    /// every conversation opened by this process gets.
    #[test]
    fn a_conversation_with_nothing_retired_starts_at_zero() {
        assert_eq!(first_live_turn(500, |_| false), 0);
        assert_eq!(first_live_turn(0, |_| false), 0);
    }

    /// **The search must not need the invariant to be sound.**
    ///
    /// It is only a binary search because [`retire_expired`] sweeps oldest-first
    /// and stops on failure, which keeps the tombstoned turns contiguous from
    /// zero. If something ever broke that, landing inside the run is acceptable
    /// — the oldest-first sweep closes the rest over the next few inserts — but
    /// landing *past* a live turn is not, because everything below the horizon
    /// is then stranded for good. So: whatever it returns, no live turn sits
    /// below it.
    #[test]
    fn a_gap_in_the_retired_run_never_strands_a_live_turn_below_the_watermark() {
        // Retired 0..40 and 60..100, live in between — the shape a failed write
        // followed by a later catch-up would leave.
        let holey = |turn: u32| turn < 40 || (60..100).contains(&turn);
        let at = first_live_turn(100, holey);
        assert!(
            !holey(at) || at == 100,
            "the watermark names a turn that is already retired"
        );
        assert!(at <= 40, "live turns 40..60 were skipped over");
    }

    /// The midpoint is taken *between* the bounds, not as their sum. A
    /// conversation near the top of the index space is exactly the one a mind
    /// that never stops living produces, and `(lo + hi) / 2` overflows there.
    #[test]
    fn a_conversation_at_the_top_of_the_index_space_does_not_overflow() {
        let count = u32::MAX as u64;
        assert_eq!(
            first_live_turn(count, |turn| turn < u32::MAX - 1),
            u32::MAX - 1
        );
        // And a count past what an index can name clamps rather than wrapping.
        assert_eq!(first_live_turn(u64::MAX, |_| false), 0);
    }

    /// The sweep's own logic, over a fake in place of an engine.
    ///
    /// Worth having separately from the integration test: what this checks is
    /// that a *gap* closes, and manufacturing a gap through a real engine means
    /// making a write fail, which is far harder to arrange than to reason about.
    #[derive(Default)]
    struct Fake {
        tombstoned: std::collections::HashSet<u32>,
        at: u32,
    }

    impl Fake {
        /// The same walk as [`retire_expired`], against the fake.
        fn sweep(&mut self, turn_count: u64, keep: u64) -> u32 {
            let Some(horizon) = horizon(turn_count, keep) else {
                return 0;
            };
            let mut retired = 0;
            let mut budget = LOOK_BACK;
            while self.at <= horizon && budget > 0 {
                budget -= 1;
                if self.tombstoned.insert(self.at) {
                    retired += 1;
                }
                self.at += 1;
            }
            retired
        }

        /// Whether every turn retired so far forms one run from zero.
        ///
        /// The invariant the whole design rests on: a hole below the watermark
        /// is a turn nothing will ever look at again.
        fn contiguous_from_zero(&self) -> bool {
            (0..self.tombstoned.len() as u32).all(|i| self.tombstoned.contains(&i))
        }
    }

    #[test]
    fn a_steady_conversation_retires_exactly_one_turn_per_insert() {
        let mut f = Fake::default();
        assert_eq!(f.sweep(KEEP_TURNS + 1, KEEP_TURNS), 1);
        for n in KEEP_TURNS + 2..KEEP_TURNS + 40 {
            assert_eq!(f.sweep(n, KEEP_TURNS), 1, "at {n} turns");
        }
        assert_eq!(f.tombstoned.len(), 39);
        assert!(f.contiguous_from_zero());
    }

    #[test]
    fn a_gap_is_closed_rather_than_skipped_forever() {
        // **The reason this is a sweep and not a counter.** A daemon restarted
        // mid-conversation resumes at a depth far past the horizon; counting
        // forward would retire one turn and leave everything between the two
        // depths live for good, since nothing would look below the horizon
        // again.
        let mut f = Fake::default();
        assert_eq!(f.sweep(KEEP_TURNS + 100, KEEP_TURNS), 100, "gap not closed");
        assert!(f.contiguous_from_zero());
    }

    /// **The bug the down-walk had, as a test.**
    ///
    /// Sweeping down from the horizon and stopping at the first tombstoned turn
    /// tombstones a block in the middle when the gap is longer than one budget.
    /// The next insert then stops at the top of that block and never reaches
    /// what is under it — a permanent hole, of exactly the turns retention
    /// exists to reclaim.
    #[test]
    fn a_gap_longer_than_the_look_back_closes_over_several_inserts() {
        let mut f = Fake::default();
        let depth = KEEP_TURNS + LOOK_BACK as u64 + 50;
        assert_eq!(
            f.sweep(depth, KEEP_TURNS),
            LOOK_BACK,
            "one insert did more than its budget"
        );
        assert!(f.contiguous_from_zero(), "the first sweep left a hole");

        // The next insert resumes at the watermark rather than at the horizon.
        assert_eq!(f.sweep(depth + 1, KEEP_TURNS), 51);
        assert!(f.contiguous_from_zero(), "the catch-up left a hole");
        assert_eq!(
            f.sweep(depth + 2, KEEP_TURNS),
            1,
            "steady state, one a turn"
        );
    }

    #[test]
    fn sweeping_twice_at_the_same_depth_writes_nothing_the_second_time() {
        let mut f = Fake::default();
        assert_eq!(f.sweep(KEEP_TURNS + 5, KEEP_TURNS), 5);
        assert_eq!(
            f.sweep(KEEP_TURNS + 5, KEEP_TURNS),
            0,
            "a re-run rewrote tombstones that already stood"
        );
    }

    #[test]
    fn a_watermark_starting_at_zero_costs_lookups_and_not_writes() {
        // What a restarted daemon does: its watermark is zero against a
        // conversation whose turns are already retired, and it must not rewrite
        // any of them.
        let mut f = Fake::default();
        f.sweep(KEEP_TURNS + 30, KEEP_TURNS);
        let mut restarted = Fake {
            tombstoned: f.tombstoned.clone(),
            at: 0,
        };
        assert_eq!(
            restarted.sweep(KEEP_TURNS + 30, KEEP_TURNS),
            0,
            "a restart rewrote tombstones that already stood"
        );
        assert_eq!(restarted.at, 30, "the watermark did not catch up");
    }
}

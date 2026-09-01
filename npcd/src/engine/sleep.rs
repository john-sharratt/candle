//! The day boundary: a character sleeps, its conversation is retired, and it
//! wakes tomorrow on a fresh one.
//!
//! # Why a day at all
//!
//! A character that never rolls over accumulates one conversation for the
//! lifetime of the world. Nothing breaks immediately — the substrate is
//! append-only and the gather does not care how long a conversation is — but two
//! things degrade. The verbatim window stops meaning anything (it is a fixed
//! tail off an unbounded transcript), and consolidation never has a boundary to
//! run at. The sleep fold is the mind design's *hard forget*, and a hard forget
//! needs an edge.
//!
//! # What sleeping does, in order
//!
//! 1. The character perceives that the day is ending — it is an event like any
//!    other, so anything it wants to do about it, it can.
//! 2. The day's conversation is **tombstoned**: retired from selection, not
//!    deleted. The turns stay in the redo log. This is the crucial distinction —
//!    a tombstone stops a conversation being gathered by default, and leaves
//!    every one of its turns reachable when something makes them relevant again.
//! 3. Consolidation folds the day into the memory layer. What survives is what
//!    the fold keeps; the rest fades.
//! 4. A new conversation opens for the next day, and the window rolls over.
//!
//! # The clock is narrative, not wall
//!
//! Days here are the world's days. A world running at 60× has a day every
//! twenty-four minutes, and characters sleep on that schedule — which is the
//! point, since the world's inhabitants live in the world's time. The day number
//! comes from [`crate::clock`], never from the host's calendar.

use serde::Serialize;

/// World-clock milliseconds in one narrative day.
///
/// Twenty-four hours of world time. A world that wants shorter days changes its
/// pace rather than this constant — the day is a day, and how fast it passes is
/// the clock's business.
pub const DAY_MS: u64 = 24 * 60 * 60 * 1000;

/// Which day a world-clock instant falls in. Day 0 is the world's first.
pub fn day_of(world_ms: u64) -> u64 {
    world_ms / DAY_MS
}

/// How far into its day an instant is: 0.0 at dawn, approaching but never
/// reaching 1.0.
///
/// `f64`, and not for pedantry. A day is 86 400 000 ms and `f32` carries about
/// seven significant digits, so the last several seconds before midnight all
/// round to exactly `1.0` — a phase that says "the next day has started" during
/// a period when [`day_of`] still says it has not. The two would disagree for
/// the last moments of every day, which is precisely when a roll-over is being
/// decided.
pub fn phase_of(world_ms: u64) -> f64 {
    (world_ms % DAY_MS) as f64 / DAY_MS as f64
}

/// What a character does about the day, right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DayAction {
    /// Nothing to do; the day the character is on is the day it is.
    Continue,
    /// The world has moved into a new day. Roll over.
    RollOver { from: u64, to: u64 },
}

/// Tracks which day each character is living in.
///
/// Deliberately not a timer. A timer would fire on wall-clock elapsed time and
/// be wrong the moment the narrative clock is paused, jumped, or re-paced —
/// which the console can do at any moment. Asking "what day is it, and what day
/// does this character think it is" is correct under every one of those.
#[derive(Debug, Default)]
pub struct DayTracker {
    /// The day this character last woke into. `None` before it has ever woken.
    current: Option<u64>,
}

impl DayTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// The day this character believes it is in.
    pub fn current(&self) -> Option<u64> {
        self.current
    }

    /// Compare the world's day against the character's and say what to do.
    ///
    /// Does not mutate — a caller that decides not to roll over (because the
    /// character is mid-interaction, say) must be able to ask again later and get
    /// the same answer. [`Self::rolled_over_to`] is how the decision is recorded.
    pub fn evaluate(&self, world_ms: u64) -> DayAction {
        let today = day_of(world_ms);
        match self.current {
            None => DayAction::Continue,
            Some(c) if c == today => DayAction::Continue,
            // A clock jumped backwards — an operator rewinding the world — is a
            // roll-over like any other. The character does not get to be in a
            // day the world has left, whichever direction it left in.
            Some(c) => DayAction::RollOver { from: c, to: today },
        }
    }

    /// Record that the character is now living in `day`.
    pub fn rolled_over_to(&mut self, day: u64) {
        self.current = Some(day);
    }

    /// First wake: adopt the world's day without treating it as a roll-over.
    pub fn start_at(&mut self, world_ms: u64) {
        self.current = Some(day_of(world_ms));
    }
}

/// The conversation id a character uses for a given day.
///
/// Derived rather than allocated, so a restart mid-day rejoins the conversation
/// it was already on instead of opening a second one for the same day. That bug
/// is invisible until somebody asks why a character has two of everything.
pub fn conversation_id(npc_id: u64, day: u64) -> String {
    format!("npc-{npc_id}-day-{day}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: u64 = 60 * 60 * 1000;

    #[test]
    fn days_are_counted_from_world_zero() {
        assert_eq!(day_of(0), 0);
        assert_eq!(day_of(DAY_MS - 1), 0);
        assert_eq!(day_of(DAY_MS), 1);
        assert_eq!(day_of(DAY_MS * 9 + 3 * H), 9);
    }

    #[test]
    fn phase_runs_from_dawn_to_dawn() {
        assert_eq!(phase_of(0), 0.0);
        assert!((phase_of(DAY_MS / 2) - 0.5).abs() < 1e-9);
        assert_eq!(phase_of(DAY_MS), 0.0, "a new day starts at dawn again");
    }

    /// **The phase and the day must never disagree.** In `f32` the last several
    /// seconds of every day round to a phase of exactly 1.0 while [`day_of`]
    /// still reports the day that is ending — a contradiction arriving precisely
    /// when a roll-over is being decided. This is why the phase is `f64`.
    #[test]
    fn the_last_millisecond_of_a_day_is_still_that_day() {
        let end = DAY_MS - 1;
        assert_eq!(day_of(end), 0);
        assert!(
            phase_of(end) < 1.0,
            "phase reached 1.0 while day_of still says day 0 — they disagree at midnight"
        );
        // And the same at a later day, where the absolute magnitude is bigger
        // and a narrower type would round harder.
        let late = DAY_MS * 900 - 1;
        assert_eq!(day_of(late), 899);
        assert!(phase_of(late) < 1.0);
    }

    /// A character that has never woken has no day to be wrong about.
    #[test]
    fn a_character_that_has_never_woken_does_not_roll_over() {
        let t = DayTracker::new();
        assert_eq!(t.evaluate(DAY_MS * 5), DayAction::Continue);
        assert_eq!(t.current(), None);
    }

    #[test]
    fn staying_within_a_day_is_not_a_roll_over() {
        let mut t = DayTracker::new();
        t.start_at(3 * H);
        assert_eq!(t.evaluate(4 * H), DayAction::Continue);
        assert_eq!(t.evaluate(23 * H), DayAction::Continue);
    }

    #[test]
    fn crossing_midnight_rolls_over() {
        let mut t = DayTracker::new();
        t.start_at(23 * H);
        assert_eq!(
            t.evaluate(DAY_MS + H),
            DayAction::RollOver { from: 0, to: 1 }
        );
    }

    /// The console can jump the clock. A character must not be left living in a
    /// day the world has left, whichever direction it left in.
    #[test]
    fn a_backwards_clock_jump_is_still_a_roll_over() {
        let mut t = DayTracker::new();
        t.start_at(DAY_MS * 5);
        assert_eq!(
            t.evaluate(DAY_MS * 2),
            DayAction::RollOver { from: 5, to: 2 }
        );
    }

    /// A multi-day jump is one roll-over, not one per day skipped — the
    /// character was not awake for the days in between and has nothing to fold
    /// from them.
    #[test]
    fn a_long_jump_is_a_single_roll_over() {
        let mut t = DayTracker::new();
        t.start_at(0);
        assert_eq!(
            t.evaluate(DAY_MS * 40),
            DayAction::RollOver { from: 0, to: 40 }
        );
    }

    /// Evaluating must not mutate, or a caller that defers the roll-over (a
    /// character mid-interaction) silently loses it.
    #[test]
    fn evaluating_twice_gives_the_same_answer() {
        let mut t = DayTracker::new();
        t.start_at(0);
        let first = t.evaluate(DAY_MS * 2);
        assert_eq!(t.evaluate(DAY_MS * 2), first);
        assert_eq!(t.current(), Some(0), "evaluate mutated the tracker");
        t.rolled_over_to(2);
        assert_eq!(t.evaluate(DAY_MS * 2), DayAction::Continue);
    }

    /// A restart mid-day must rejoin the day's conversation rather than open a
    /// second one for the same day.
    #[test]
    fn a_days_conversation_id_is_derived_not_allocated() {
        assert_eq!(conversation_id(7, 3), "npc-7-day-3");
        assert_eq!(conversation_id(7, 3), conversation_id(7, 3));
        assert_ne!(conversation_id(7, 3), conversation_id(7, 4));
        assert_ne!(conversation_id(8, 3), conversation_id(7, 3));
    }
}

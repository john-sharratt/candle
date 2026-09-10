//! The air handling, which the whole building breathes through.
//!
//! # Why this one matters most
//!
//! It is the fixture with the widest reach: it publishes [`Cond::Gusting`] and
//! [`Cond::Cold`], and those are what the rat startles at and what the growth
//! lives or dies by. A purge here is the first domino in most of the building's
//! interesting incidents, and it is the reason the air runs on a cycle rather
//! than at random — a cycle means the rest of the building can be *waiting* for
//! something, which is the difference between a machine and a mood.
//!
//! # The cycle
//!
//! Low → nominal → high → purge → low. It climbs on load from the compute floor
//! and falls back when that quietens, so the building's noise floor tracks what
//! it is doing. A purge is the loud end and only happens off the top of the
//! cycle, which is what stops the startle being cheap.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Low,
    Nominal,
    High,
    Purging,
}

pub struct AirHandling {
    level: Level,
    due: Due,
    rng: Rng,
    /// Set when the compute floor says it is working, cleared when it stops.
    /// The air does not know what compute *is* — only that something published
    /// [`Cond::Working`], which is the whole of the coupling.
    under_load: bool,
    /// A filter change is due after this many cycles, and is the one thing here
    /// that arrives on a count rather than a clock.
    to_filter: u32,
}

impl AirHandling {
    pub fn new(seed: u64) -> AirHandling {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(20), Duration::from_secs(90));
        let to_filter = 6 + rng.below(8) as u32;
        AirHandling {
            level: Level::Nominal,
            due: Due::at(first),
            rng,
            under_load: false,
            to_filter,
        }
    }
}

impl Fixture for AirHandling {
    fn id(&self) -> &'static str {
        "air"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        match self.level {
            Level::Purging => {
                out.push(Cond::Gusting);
                out.push(Cond::Loud);
            }
            Level::High => out.push(Cond::Loud),
            // The quiet end is a real condition, not the absence of one: it is
            // what lets a small sound carry, and the rat is bolder in it.
            Level::Low => {
                out.push(Cond::Quiet);
                out.push(Cond::Cold);
            }
            Level::Nominal => {}
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }

        // A purge is loud, so it waits for the top of the cycle rather than
        // firing whenever the dice say. Being *earned* is what makes it read as
        // the building doing something rather than as noise.
        if self.level == Level::High && self.rng.one_in(3) {
            self.level = Level::Purging;
            self.due.again(w, Duration::from_secs(8));
            self.to_filter = self.to_filter.saturating_sub(1);
            return Some(
                Stirring::new(
                    "air",
                    "The air handling purges, and every loose thing in the room lifts and settles.",
                    Salience::NORMAL,
                )
                .tagged(&[Cond::Gusting, Cond::Loud]),
            );
        }

        if self.level == Level::Purging {
            self.level = Level::Low;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(90), Duration::from_secs(400)),
            );
            return Some(Stirring::new(
                "air",
                "The air handling drops away to its minimum, and the quiet it leaves is louder than the noise was.",
                Salience::IDLE,
            ).tagged(&[Cond::Quiet]));
        }

        if self.to_filter == 0 {
            self.to_filter = 8 + self.rng.below(10) as u32;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(120), Duration::from_secs(360)),
            );
            return Some(Stirring::new(
                "air",
                "The air turns colder and cleaner as the handling swaps to a fresh filter bank.",
                Salience::IDLE,
            ).tagged(&[Cond::Cold]));
        }

        // Otherwise it tracks load: up while the compute floor is working, down
        // when it is not.
        //
        // The two arms that hold their level are the ones the room hears most,
        // because a building sits at a level far longer than it changes one — so
        // those get several lines each and the transitions, which are rare, get
        // one apiece. Sizing the variety by how often an arm is *reached* is
        // what stops the commonest state being the most repetitive one.
        let (next, lines): (Level, &[&str]) = match (self.level, self.under_load) {
            (Level::Low, true) => (
                Level::Nominal,
                &["The air handling picks up off its minimum and finds a working note."],
            ),
            (Level::Nominal, true) => (
                Level::High,
                &[
                    "The air handling steps up a notch and holds there, pulling heat off the \
                   compute floor.",
                ],
            ),
            (Level::High, true) => (
                Level::High,
                &[
                    "The air handling is running hard enough that the room has a draught along \
                     the floor.",
                    "A vent grille rattles in its frame under the weight of air going through it.",
                    "The note the air handling is holding wanders up a tone and comes back.",
                    "Paper lifts at the corner on every flat surface in the room and settles \
                     again.",
                    "There is a whistle somewhere in the ducting that was not there at the lower \
                     speed.",
                ],
            ),
            (Level::High, false) => (
                Level::Nominal,
                &["The air handling eases off its high note as the load comes off."],
            ),
            (Level::Nominal, false) => (
                Level::Low,
                &[
                    "The air handling settles to its low cycle, and the room goes several degrees \
                   cooler.",
                ],
            ),
            (Level::Low, false) => (
                Level::Low,
                &[
                    "The air moving through the vents reverses direction along the floor.",
                    "A vent damper somewhere overhead closes by half a step and stays there.",
                    "The air in the room has gone still enough that the dust in it is visible.",
                    "A slow draught comes off the floor grille and dies away again.",
                    "The air handling ticks over at its minimum, barely moving anything.",
                    "Cold air spills out of a duct at the far end of the room and stops.",
                ],
            ),
            (Level::Purging, _) => unreachable!("handled above"),
        };
        let line = self.rng.pick(lines).copied().unwrap_or(lines[0]);

        self.level = next;
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(45), Duration::from_secs(240)),
        );
        let mut s = Stirring::new("air", line, Salience::IDLE);
        if next == Level::Low {
            s = s.tagged(&[Cond::Cold, Cond::Quiet]);
        }
        Some(s)
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // The compute floor's load is what the air is for. It reads the flag
        // rather than the fixture, so anything else that runs hot couples in
        // for nothing.
        if what.tags.contains(&Cond::Working) {
            self.under_load = true;
        }
        if what.from == "compute" && !what.tags.contains(&Cond::Working) {
            self.under_load = false;
        }

        // A seal shutting or a supply transferring changes what the air is
        // working against, and it responds sooner than its own cycle would.
        if what.tags.contains(&Cond::Unstable) && self.rng.one_in(2) {
            self.due.hold(w, Duration::from_secs(15));
        }
    }
}

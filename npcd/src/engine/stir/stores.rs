//! The stores, which is where the rat's work gets found.
//!
//! # The other direction of the coupling
//!
//! Most of the wiring in this module is a fixture reacting to a condition *as
//! it happens* — the gust startles the rat in the same moment. This one is the
//! opposite and the more satisfying: the rat goes into the stores, gnaws at
//! something, and this fixture quietly counts it. Nobody is told. Then, ten or
//! twenty minutes later, somebody opens a crate and finds the packaging gone
//! through.
//!
//! **Consequence arriving after the fact is what makes a world feel like it
//! kept running while you were not looking**, and it costs one counter.
//!
//! # It is mostly a quiet room
//!
//! Its ordinary output is a room settling, which is deliberate: the vault needs
//! somewhere for the noise floor to be low, or the rat's scratching has nothing
//! to be heard against.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

/// What gets found gone through, once somebody looks. Each is the back half of
/// a sentence the fixture builds, so each has to read as one.
const SPOILED: &[&str] = &[
    "a case of ration packs opened at the corner and half emptied",
    "the insulation stripped off a spare cable run and carried off somewhere",
    "a sack of dry stores split along the bottom seam",
    "the seal chewed off a crate of filter elements",
    "a box of paper records shredded into bedding",
];

pub struct Stores {
    due: Due,
    rng: Rng,
    /// Damage done and not yet found. The rat puts it up; opening a crate takes
    /// it down.
    spoiled: u32,
    /// Total found over the run. Once it is more than one or two the room stops
    /// treating each as a surprise, which is how a place gets a reputation.
    found: u32,
}

impl Stores {
    pub fn new(seed: u64) -> Stores {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(120), Duration::from_secs(600));
        Stores {
            due: Due::at(first),
            rng,
            spoiled: 0,
            found: 0,
        }
    }

    /// Damage done and not yet found. Nothing else needs this, but a room that
    /// cannot be asked what it knows is a room that cannot be inspected.
    pub fn spoiled(&self) -> u32 {
        self.spoiled
    }
}

impl Fixture for Stores {
    fn id(&self) -> &'static str {
        "stores"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        // Once enough has been found, the room itself is the evidence — and the
        // address system will start reading the containment log out at people.
        if self.found >= 2 {
            out.push(Cond::Vermin);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(150), Duration::from_secs(700)),
        );

        // **The delayed consequence.** Something the rat did a while ago,
        // arriving now, in a room the rat has long since left.
        if self.spoiled > 0 && !self.rng.one_in(3) {
            self.spoiled -= 1;
            self.found += 1;
            let what = self.rng.pick(SPOILED).copied().unwrap_or(SPOILED[0]);
            let line = match self.found {
                1 => format!("A crate in the stores turns out to have {what}."),
                _ => format!(
                    "Something else in the stores has been got at — {what}, the same as the last \
                     one.",
                ),
            };
            return Some(Stirring::new("stores", line, Salience::NORMAL).tagged(&[Cond::Vermin]));
        }

        let damp = w.is(Cond::Damp);
        let cold = w.is(Cond::Cold);
        let line: &str = match (damp, cold, self.rng.below(9)) {
            (true, _, 0..=1) => {
                "The cardboard on the lower shelves in the stores has gone soft \
                                 and is sagging under its own weight."
            }
            (_, true, 2) => {
                "The metal shelving in the stores ticks as it contracts, one upright \
                             at a time."
            }
            (_, _, 3) => "A stack of crates in the stores settles half an inch and stops.",
            (_, _, 4) => {
                "The stores door swings a few degrees on its own and comes to rest \
                          against the stop."
            }
            (_, _, 5) => {
                "Something rolls off a shelf in the stores and comes to a halt on the \
                          floor."
            }
            (_, _, 6) => "A label peels away from a crate in the stores and drops.",
            (_, _, 7) => {
                "The inventory terminal in the stores wakes up, finds nothing to do, and \
                          dims again."
            }
            _ => "A drum in the stores shifts against its neighbour with a low hollow note.",
        };
        Some(Stirring::new("stores", line, Salience::IDLE))
    }

    fn notice(&mut self, what: &Stirring, _w: &Watch) {
        // The rat only reports where it is and what it is doing. The stores read
        // that the way a person would — something was in here, so something in
        // here has been got at — without either fixture knowing the other.
        if what.from == "rat" && what.text.contains("stores") && self.spoiled < 4 {
            self.spoiled += 1;
        }
    }
}

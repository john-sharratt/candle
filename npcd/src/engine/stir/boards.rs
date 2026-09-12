//! The status boards — the building's written account of itself.
//!
//! # It reports what actually happened
//!
//! The address system reads the building's *state*; this reads its *history*.
//! [`Fixture::notice`] keeps a short log of what the other fixtures have
//! actually done, and the board then puts a fault entry up naming that. So
//! *"A fault entry has come up on the status board against the coolant
//! gallery."* appears **because the coolant loop went wrong a minute ago**, and
//! there is no path by which it can name a system that has not.
//!
//! That is the most useful thing in the vault for a character with any sense: a
//! board is where you go to find out what you missed, and here it genuinely is.
//!
//! # Why it lags
//!
//! Entries queue and clear slowly. A board that showed the truth instantly
//! would be a second copy of the event feed; one that is a minute or two behind
//! is a thing a character can be *wrong* about, which is far more interesting.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Due, Fixture, Rng, Stirring, Watch};

/// How many entries it will hold before the oldest falls off the bottom.
const ENTRIES: usize = 6;

/// What a fixture is called on a fault board. The board does not know what any
/// of these things *are* — it knows what they are filed under.
fn gallery(from: &str) -> Option<&'static str> {
    match from {
        "air" => Some("the air handling plant"),
        "lights" => Some("the lighting ring on this level"),
        "power" => Some("the main supply bus"),
        "coolant" => Some("the coolant gallery"),
        "compute" => Some("the compute floor"),
        "structure" => Some("the outer seal line"),
        "growth" => Some("environmental control"),
        "stores" => Some("the stores inventory"),
        "rat" => Some("biological containment"),
        _ => None,
    }
}

pub struct Boards {
    due: Due,
    rng: Rng,
    /// Faults waiting to be shown, oldest first. Bounded, because a board is a
    /// screen and not an archive.
    pending: Vec<&'static str>,
    /// Faults currently up. Clears slowly, and clearing is the good news.
    showing: usize,
}

impl Boards {
    pub fn new(seed: u64) -> Boards {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(45), Duration::from_secs(300));
        Boards {
            due: Due::at(first),
            rng,
            pending: Vec::new(),
            showing: 0,
        }
    }

    /// How many faults are up. The stores fixture is not the only thing that
    /// might want to ask.
    pub fn showing(&self) -> usize {
        self.showing
    }
}

impl Fixture for Boards {
    fn id(&self) -> &'static str {
        "boards"
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(70), Duration::from_secs(360)),
        );

        // A queued fault is the interesting case, and it names a real one.
        if !self.pending.is_empty() {
            let what = self.pending.remove(0);
            self.showing += 1;
            return Some(Stirring::new(
                "boards",
                format!("A fault entry has come up on the status board against {what}."),
                Salience::NORMAL,
            ));
        }

        // Faults come off the board when somebody elsewhere has dealt with
        // them, which is the building's only evidence of other people in it.
        if self.showing > 0 && self.rng.one_in(3) {
            self.showing -= 1;
            let line = match self.showing {
                0 => {
                    "The last fault entry clears off the status board, and it is showing all \
                      green for the first time in a while."
                }
                _ => {
                    "A fault entry clears off the status board, acknowledged by somebody on \
                      another level."
                }
            };
            return Some(Stirring::new("boards", line, Salience::IDLE));
        }

        let busy = self.showing > 0;
        let line = match busy {
            true => *self
                .rng
                .pick(&[
                    "The status board cycles through its open faults and starts the list again.",
                    "A line on the status board has begun to flash amber and nobody has \
                     acknowledged it.",
                    "The status board's summary count goes up by one without any entry appearing \
                     to explain it.",
                ])
                .unwrap_or(
                    &"The status board cycles through its open faults and starts the list again.",
                ),
            false => *self
                .rng
                .pick(&[
                    "A terminal in the corner wakes, shows a login prompt, and goes back to sleep.",
                    "The status board redraws itself from the top for no reason anybody asked for.",
                    "A trend graph on the wall panel steps along one division.",
                    "The status board's clock and the wall clock disagree by about a minute.",
                    "A console fan spins up under the bench and settles back down.",
                    "The wall panel dims itself, having decided the room is empty.",
                ])
                .unwrap_or(&"The status board redraws itself from the top."),
        };
        Some(Stirring::new("boards", line, Salience::IDLE))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        if what.from == "boards" {
            return;
        }
        // Only real faults get filed — a vent changing note is not a fault, and
        // a board that logged everything would be as useless here as it is in a
        // real building.
        let worth_filing = what.salience >= Salience::NORMAL && !what.tags.is_empty();
        if !worth_filing {
            return;
        }
        if let Some(name) = gallery(what.from) {
            if self.pending.len() < ENTRIES && !self.pending.contains(&name) {
                self.pending.push(name);
            }
            // Something that stopped a person in the room is on the board
            // sooner than the board's own leisurely cycle.
            if what.salience.preempts() {
                self.due.hold(
                    w,
                    self.rng
                        .between(Duration::from_secs(10), Duration::from_secs(45)),
                );
            }
        }
    }
}

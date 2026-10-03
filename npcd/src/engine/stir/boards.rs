//! The status boards — the building's written account of itself.
//!
//! # It reports what actually stands
//!
//! The world tells the board which faults stand on real objects ([`Report`]),
//! in this room and in every other occupied one. The board files an entry for
//! each and writes it on itself, where a character can `read` it: which system,
//! which object, which room, what is wrong. When the fault is gone — somebody
//! reset the object — the entry comes down, and that is written too.
//!
//! So *"A fault entry has come up on the status board against the main supply
//! bus."* appears **because a breaker panel somewhere is tripped**, and the board
//! will say which. There is no path by which it can name a system that has not
//! failed, and no entry it cannot be walked to and checked.
//!
//! # Why it lags
//!
//! It looks every minute or two. A board that showed the truth instantly would
//! be a second copy of the event feed; one that is a little behind is a thing a
//! character can be *wrong* about, which is far more interesting.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Due, Fixture, Report, Rng, Stirring, Watch};

/// The part a board's entries are written on.
const BOARD: &str = "status-board";

pub struct Boards {
    due: Due,
    rng: Rng,
    /// The faults currently up, in the order they were filed.
    up: Vec<Report>,
}

impl Boards {
    pub fn new(seed: u64) -> Boards {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(45), Duration::from_secs(300));
        Boards {
            due: Due::at(first),
            rng,
            up: Vec::new(),
        }
    }

    /// How many faults are up.
    pub fn showing(&self) -> usize {
        self.up.len()
    }
}

impl Fixture for Boards {
    fn id(&self) -> &'static str {
        "boards"
    }

    fn needs(&self) -> &'static [&'static str] {
        &[BOARD]
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

        // A fault that stands and is not up yet goes up.
        if let Some(r) = w.standing.iter().find(|r| !self.up.contains(r)) {
            self.up.push(r.clone());
            return Some(
                Stirring::new(
                    "boards",
                    format!(
                        "A fault entry has come up on the status board against {}.",
                        r.system
                    ),
                    Salience::NORMAL,
                )
                .posted_on(
                    BOARD,
                    format!(
                        "Fault against {}: {}, at {} in {}.",
                        r.system, r.trouble, r.object, r.room
                    ),
                ),
            );
        }

        // A fault that is up and no longer stands comes down, which is the
        // building's only evidence that somebody has been and dealt with it.
        if let Some(at) = self.up.iter().position(|r| !w.standing.contains(r)) {
            let r = self.up.remove(at);
            return Some(
                Stirring::new(
                    "boards",
                    format!(
                        "The fault entry against {} clears off the status board.",
                        r.system
                    ),
                    Salience::IDLE,
                )
                .posted_on(
                    BOARD,
                    format!("Cleared: {}, at {} in {}.", r.system, r.object, r.room),
                ),
            );
        }

        let line = match self.up.is_empty() {
            false => *self
                .rng
                .pick(&[
                    "The status board cycles through its open faults and starts the list again.",
                    "A line on the status board has begun to flash amber and nobody has \
                     acknowledged it.",
                ])
                .unwrap_or(
                    &"The status board cycles through its open faults and starts the list again.",
                ),
            true => *self
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
}

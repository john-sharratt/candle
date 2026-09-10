//! Something living in a sealed research base, which is a containment failure
//! rather than a bit of atmosphere.
//!
//! # Why this fixture is the interesting one
//!
//! It is the only part of the building with somewhere to *be*. Everything else
//! is a condition — the air is loud or it is not — and a condition cannot go
//! anywhere. A rat has a place, a nerve, and an appetite, so two sightings are
//! a story instead of two draws from a list: the thing you heard in the ceiling
//! is the thing now in the stores.
//!
//! # The state machine
//!
//! ```text
//!   Hidden ──heard──> InTheWalls ──quiet & bold──> InTheOpen
//!      ^                   |                            |
//!      |                   |                       gust / noise
//!      |                   v                            v
//!      └───────────── Bolting <──────────────────────────┘
//!                         |
//!                    reaches the stores
//!                         v
//!                      AtTheStores ── gnaws, and the stores find out later
//! ```
//!
//! # What couples it
//!
//! * **Startle.** Anything tagged [`Cond::Gusting`] or [`Cond::Alarmed`] sends
//!   it bolting from wherever it is. This is the vent-scares-the-rat incident,
//!   and neither fixture knows the other exists — the air publishes a flag and
//!   the rat reads it.
//! * **Nerve.** It is bolder in the [`Cond::Dark`] and in the
//!   [`Cond::Quiet`], and it will not cross a room that is [`Cond::Loud`]. So
//!   the air handling's cycle decides how much of it anybody ever sees.
//! * **Knocking something over.** A bolt across an occupied room is the one
//!   thing here that reaches the preempt bar, because a person cannot not
//!   notice it.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Where {
    /// Nowhere anybody can hear. The resting state, and where it returns to.
    Hidden,
    /// Behind the panelling or above the ceiling — audible, not visible.
    InTheWalls,
    /// On the floor of a room, which is when a person sees it.
    InTheOpen,
    /// Going, fast, in a straight line, over whatever is in the way.
    Bolting,
    /// At the one place in the vault worth the trip.
    AtTheStores,
}

/// The things a bolting rat can put over. Each is a whole sentence because the
/// rat's line and the object's line run together in one room.
const KNOCKS: &[&str] = &[
    "and puts a rack of plates over on its way",
    "and takes a stack of filed pages off a table with it",
    "and sends something metal spinning off a bench",
    "and knocks a stool across the floor",
    "and drags a length of cabling out of its clip",
];

pub struct Rat {
    at: Where,
    due: Due,
    rng: Rng,
    /// Rises in the dark and the quiet, falls when it is startled. Decides
    /// whether it will come out at all.
    nerve: i32,
    /// How much it has eaten through at the stores. The stores fixture asks
    /// about this rather than being told, which is the other direction of the
    /// coupling.
    gnawed: u32,
    /// Set the moment something frightens it, cleared when it acts on it.
    startled: bool,
}

impl Rat {
    pub fn new(seed: u64) -> Rat {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(60), Duration::from_secs(600));
        Rat {
            at: Where::Hidden,
            due: Due::at(first),
            rng,
            nerve: 0,
            gnawed: 0,
            startled: false,
        }
    }

    /// How much damage it has done. Read by the stores.
    pub fn gnawed(&self) -> u32 {
        self.gnawed
    }

    fn settle(&mut self, w: &Watch) {
        let gap = match self.at {
            // It does not go straight back out after a fright.
            Where::Hidden => self
                .rng
                .between(Duration::from_secs(240), Duration::from_secs(900)),
            Where::AtTheStores => self
                .rng
                .between(Duration::from_secs(60), Duration::from_secs(180)),
            _ => self
                .rng
                .between(Duration::from_secs(40), Duration::from_secs(220)),
        };
        self.due.again(w, gap);
    }
}

impl Fixture for Rat {
    fn id(&self) -> &'static str {
        "rat"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        // Only once it is somewhere it could be found. A rat nobody has any
        // evidence of is not a containment failure yet.
        if !matches!(self.at, Where::Hidden) {
            out.push(Cond::Vermin);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        // A fright jumps the queue: being startled is not something that waits
        // for a timer.
        if self.startled {
            self.startled = false;
            let was = self.at;
            self.at = Where::Bolting;
            self.nerve = -2;
            self.settle(w);

            let line = match was {
                Where::InTheOpen => {
                    let knock = self.rng.pick(KNOCKS).copied().unwrap_or_default();
                    // Loud, sudden, and in the room with you.
                    return Some(
                        Stirring::new(
                            "rat",
                            format!("Something bolts across the floor {knock}."),
                            Salience::URGENT,
                        )
                        .tagged(&[Cond::Vermin, Cond::Loud]),
                    );
                }
                Where::AtTheStores => {
                    "Something goes out of the stores at speed, knocking a shelf as it leaves."
                }
                _ => "Something bolts along the inside of the wall and is gone.",
            };
            return Some(Stirring::new("rat", line, Salience::NORMAL).tagged(&[Cond::Vermin]));
        }

        if !self.due.ready(w) {
            return None;
        }

        // Nerve is the whole of its judgement: dark and quiet embolden it, and
        // a loud room keeps it in the walls however long it has waited.
        if w.is(Cond::Dark) {
            self.nerve += 2;
        }
        if w.is(Cond::Quiet) {
            self.nerve += 1;
        }
        if w.is(Cond::Loud) {
            self.nerve -= 2;
        }
        self.nerve = self.nerve.clamp(-3, 5);

        let next = match (self.at, self.nerve) {
            (Where::Bolting, _) => Where::Hidden,
            (Where::Hidden, n) if n >= 1 => Where::InTheWalls,
            (Where::Hidden, _) => Where::Hidden,
            (Where::InTheWalls, n) if n >= 3 => Where::InTheOpen,
            (Where::InTheWalls, n) if n <= -1 => Where::Hidden,
            (Where::InTheWalls, _) => Where::InTheWalls,
            (Where::InTheOpen, n) if n >= 4 => Where::AtTheStores,
            (Where::InTheOpen, n) if n <= 0 => Where::InTheWalls,
            (Where::InTheOpen, _) => Where::InTheOpen,
            (Where::AtTheStores, n) if n <= 0 => Where::InTheWalls,
            (Where::AtTheStores, _) => Where::AtTheStores,
        };

        let was = self.at;
        self.at = next;
        self.settle(w);

        // Standing still is usually not worth saying. It says something the
        // first few times and then shuts up, which is what stops a rat that is
        // going nowhere from being the loudest thing in the vault.
        if was == next && !self.rng.one_in(3) {
            return None;
        }

        let line: &str = match (was, next) {
            (Where::Hidden, Where::InTheWalls) => {
                "Something moves in the duct overhead, unhurried, and stops."
            }
            (Where::InTheWalls, Where::InTheWalls) => self
                .rng
                .pick(&[
                    "Something is working at the insulation behind a wall panel.",
                    "A scratching starts up inside the wall and keeps going.",
                    "Something small crosses the ceiling void, from one side of the room to the other.",
                ])
                .unwrap_or(&"A scratching starts up inside the wall and keeps going."),
            (Where::InTheWalls, Where::InTheOpen) => {
                "A rat comes out along the base of the wall and stops in the open."
            }
            (Where::InTheOpen, Where::InTheOpen) => self
                .rng
                .pick(&[
                    "The rat crosses the floor at the wall, in no particular hurry.",
                    "The rat sits up on its back legs and works at something it is holding.",
                    "The rat has got up onto a bench and is going along the back of it.",
                ])
                .unwrap_or(&"The rat crosses the floor at the wall, in no particular hurry."),
            (Where::InTheOpen, Where::AtTheStores) => {
                "The rat goes into the stores through the gap under the door."
            }
            (Where::AtTheStores, Where::AtTheStores) => {
                self.gnawed += 1;
                "Something is at work in the stores, on the far side of the shelving."
            }
            (Where::AtTheStores, _) | (Where::InTheOpen, _) => {
                "The rat goes back into the wall and the room is its own again."
            }
            // Anywhere else out of the walls is back into cover: the ladder
            // only goes one rung at a time, so there is nowhere else to reach.
            (Where::InTheWalls, _) => {
                "The scratching inside the wall stops, and does not start again."
            }
            (Where::Bolting, _) => "The building is quiet where something was moving a moment ago.",
            (Where::Hidden, _) => return None,
        };

        let salience = match next {
            // A rat in the open, in front of you, is worth a turn. It is not
            // worth stopping mid-sentence for — that is reserved for the bolt.
            Where::InTheOpen | Where::AtTheStores => Salience::NORMAL,
            _ => Salience::IDLE,
        };
        Some(Stirring::new("rat", line, salience).tagged(&[Cond::Vermin]))
    }

    fn notice(&mut self, what: &Stirring, _w: &Watch) {
        if what.from == "rat" {
            return;
        }
        // **The incident.** A purge, an alarm, or anything else sudden and loud
        // sends it. The rat has never heard of the air handling; it knows what
        // a gust is.
        let frightening = what.tags.contains(&Cond::Gusting)
            || what.tags.contains(&Cond::Alarmed)
            || (what.tags.contains(&Cond::Loud) && what.salience.preempts());

        if frightening && !matches!(self.at, Where::Hidden) {
            self.startled = true;
        }

        // The dark is the other direction: it does not startle, it emboldens.
        if what.tags.contains(&Cond::Dark) {
            self.nerve = (self.nerve + 2).clamp(-3, 5);
        }
    }
}

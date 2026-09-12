//! The lighting, which is what the building's other troubles show up in first.
//!
//! # Why it mostly reacts
//!
//! It has a timer of its own, but the interesting half is [`Fixture::notice`]:
//! a light bank is the cheapest instrument in the vault for reading the supply,
//! so an unstable bus shows here before it shows anywhere else. A character
//! that has learnt that is a character reading its building.
//!
//! Publishing [`Cond::Dark`] is what makes this matter to anything else — the
//! rat is bolder in it, and the growth does better without light.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Full,
    /// One tube gone, or a bank on standby. Working gloom.
    Down,
    /// Out. Rare, short, and the thing the rat waits for.
    Out,
}

pub struct Lighting {
    state: State,
    due: Due,
    rng: Rng,
    /// Tubes that have failed and not been replaced. A building that is
    /// slowly going dark reads differently from one that flickers.
    failed: u32,
}

impl Lighting {
    pub fn new(seed: u64) -> Lighting {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(120), Duration::from_secs(600));
        Lighting {
            state: State::Full,
            due: Due::at(first),
            rng,
            failed: 0,
        }
    }
}

impl Fixture for Lighting {
    fn id(&self) -> &'static str {
        "lights"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        match self.state {
            State::Out => out.push(Cond::Dark),
            // Enough gone to matter is dark enough for what lives in the walls.
            State::Down if self.failed >= 3 => out.push(Cond::Dark),
            _ => {}
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }

        // Being out is a moment, never a state it settles into.
        if self.state == State::Out {
            self.state = State::Down;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(90), Duration::from_secs(400)),
            );
            return Some(Stirring::new(
                "lights",
                "The lights come back, and come back brighter than they were before.",
                Salience::NORMAL,
            ));
        }

        let unstable = w.is(Cond::Unstable);
        let roll = self.rng.below(if unstable { 5 } else { 12 });
        let (line, salience, tags): (&str, Salience, &[Cond]) = match roll {
            0 if unstable => {
                self.state = State::Out;
                self.due.again(w, Duration::from_secs(6));
                (
                    "The lights go out entirely, and the room is black enough to feel.",
                    Salience::URGENT,
                    &[Cond::Dark],
                )
            }
            1 => {
                self.failed += 1;
                self.state = State::Down;
                (
                    "One tube in the ceiling fails, and the room is subtly the wrong colour.",
                    Salience::NORMAL,
                    &[],
                )
            }
            2 => (
                "The lights in the ceiling flicker and steady.",
                Salience::IDLE,
                &[],
            ),
            3 => (
                "The lighting shifts colour temperature as it changes supply.",
                Salience::IDLE,
                &[],
            ),
            4 if self.failed > 0 => {
                self.failed = self.failed.saturating_sub(1);
                self.state = match self.failed {
                    0 => State::Full,
                    _ => State::Down,
                };
                (
                    "A failed tube in the ceiling has been changed, and that end of the room is honest again.",
                    Salience::IDLE,
                    &[],
                )
            }
            5 => {
                self.state = State::Down;
                (
                    "A bank of lights drops to standby, leaving half the room in working gloom.",
                    Salience::NORMAL,
                    &[],
                )
            }
            6 => (
                "A ceiling tube buzzes at the end of its life and settles back down.",
                Salience::IDLE,
                &[],
            ),
            7 => (
                "The ceiling lights dim for a moment and come back to full.",
                Salience::IDLE,
                &[],
            ),
            8 => (
                "An emergency light over the door tests itself, green, and goes out again.",
                Salience::IDLE,
                &[],
            ),
            9 => (
                "The lighting in the far half of the room comes up as something moves under it.",
                Salience::IDLE,
                &[],
            ),
            10 => (
                "A ceiling fitting has started to hum at a pitch that is hard to ignore.",
                Salience::IDLE,
                &[],
            ),
            _ => (
                "The light in the room shifts as a fitting overhead cycles itself.",
                Salience::IDLE,
                &[],
            ),
        };

        if self.state != State::Out {
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(100), Duration::from_secs(500)),
            );
        }
        Some(Stirring::new("lights", line, salience).tagged(tags))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        if what.from == "lights" {
            return;
        }
        // **The supply shows here first.** A light bank is the cheapest
        // instrument in the vault, and this is the coupling that makes the
        // cascade read as one incident rather than three coincidences.
        if what.tags.contains(&Cond::Unstable) {
            self.due.hold(
                w,
                self.rng
                    .between(Duration::from_secs(2), Duration::from_secs(12)),
            );
        }
    }
}

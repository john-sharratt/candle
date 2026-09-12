//! The supply, and the first domino in the building's other big incident.
//!
//! # The cascade
//!
//! A dip here publishes [`Cond::Unstable`], which the lights read (they
//! flicker, and may fail, which publishes [`Cond::Dark`], which the rat reads)
//! and which the compute floor reads (a node drops, load falls, the air eases
//! off). One event, four fixtures, and no fixture naming another.
//!
//! # Why it has a health rather than a die roll
//!
//! A supply that dips at random is weather. A supply that has been dipping and
//! is now worse is a *situation* — a character that saw the first two has
//! reason to expect the third, and the breaker going is then something it could
//! have seen coming. `strain` is that memory, and it decays, so a bad patch
//! passes rather than damning the building for ever.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

pub struct PowerBus {
    due: Due,
    rng: Rng,
    /// How unhappy the supply is. Climbs on every dip and on heavy load,
    /// decays on quiet passes. At the top a breaker goes.
    strain: i32,
    /// True between a dip and its recovery, which is the window in which the
    /// lights and the compute floor read it.
    wobbling: bool,
    /// Which supply it is on. Changing it is the loudest ordinary thing here.
    supply: u8,
}

impl PowerBus {
    pub fn new(seed: u64) -> PowerBus {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(90), Duration::from_secs(500));
        PowerBus {
            due: Due::at(first),
            rng,
            strain: 0,
            wobbling: false,
            supply: 1,
        }
    }
}

impl Fixture for PowerBus {
    fn id(&self) -> &'static str {
        "power"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        if self.wobbling || self.strain >= 4 {
            out.push(Cond::Unstable);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }

        // A supply that has been holding for a while settles down again.
        if self.wobbling {
            self.wobbling = false;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(120), Duration::from_secs(600)),
            );
            let line = *self
                .rng
                .pick(&[
                    "The supply steadies, and the machines that were complaining stop.",
                    "The supply comes back onto its proper frequency, and the room's hum settles \
                     with it.",
                    "Whatever the supply was doing, it has stopped doing it, and the panel is \
                     green again.",
                ])
                .unwrap_or(&"The supply steadies, and the machines that were complaining stop.");
            return Some(Stirring::new("power", line, Salience::IDLE));
        }

        // The top of the strain curve. Loud, consequential, and earned by
        // everything that came before it.
        if self.strain >= 6 {
            self.strain = 0;
            self.wobbling = true;
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(30), Duration::from_secs(90)),
            );
            return Some(
                Stirring::new(
                    "power",
                    "A breaker goes somewhere else in the vault, and you hear what it cost — \
                     half the machines in earshot stopping at once.",
                    Salience::URGENT,
                )
                .tagged(&[Cond::Unstable, Cond::Loud]),
            );
        }

        // Load makes it worse; a quiet building lets it recover.
        match w.is(Cond::Working) {
            true => self.strain += 1,
            false => self.strain = (self.strain - 1).max(0),
        }

        let roll = self.rng.below(10);
        let (line, salience, tags): (&str, Salience, &[Cond]) = match roll {
            0..=2 if self.strain >= 3 => {
                self.wobbling = true;
                self.strain += 1;
                (
                    "The supply dips, and every machine in earshot complains at once.",
                    Salience::NORMAL,
                    &[Cond::Unstable],
                )
            }
            3..=4 => {
                self.supply = 3 - self.supply.min(2);
                self.wobbling = true;
                (
                    "The power bus transfers to its other supply, and everything in the room blinks.",
                    Salience::NORMAL,
                    &[Cond::Unstable],
                )
            }
            5 => (
                "A backup supply clicks in, and clicks out again a second later.",
                Salience::IDLE,
                &[],
            ),
            6 => (
                "The load-shed warning comes up on the supply panel and clears itself.",
                Salience::IDLE,
                &[],
            ),
            7 if self.strain >= 2 => (
                "The lighting and the vents dim together for a moment, which they should not do.",
                Salience::NORMAL,
                &[Cond::Unstable],
            ),
            8 => (
                "A cabinet fan on the supply panel starts up and runs for a few seconds.",
                Salience::IDLE,
                &[],
            ),
            9 => (
                "The supply panel logs something to itself and shows no sign of what.",
                Salience::IDLE,
                &[],
            ),
            _ => (
                "The supply hums a fraction higher than it was, and holds there.",
                Salience::IDLE,
                &[],
            ),
        };

        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(60), Duration::from_secs(420)),
        );
        Some(Stirring::new("power", line, salience).tagged(tags))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // A hard-working building is a strain on the supply. Read off the flag,
        // so anything that publishes `Working` counts.
        if what.tags.contains(&Cond::Working) {
            self.strain += 1;
        }
        // And a supply already wobbling reacts sooner than its own timer.
        if self.wobbling && what.tags.contains(&Cond::Unstable) && what.from != "power" {
            self.due.hold(w, Duration::from_secs(20));
        }
    }
}

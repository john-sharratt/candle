//! The compute floor, which is what a research base is for.
//!
//! # Why it drives the others
//!
//! It is the only fixture that publishes [`Cond::Working`], and three separate
//! things read it: the air steps up to pull the heat away, the supply takes
//! strain, and the coolant's next problem arrives sooner. So the building's
//! whole noise floor tracks whether anybody is getting anything done, which is
//! the cheapest way to make it feel like a place with a purpose.
//!
//! Its jobs also *finish*, which is the one thing here that is unambiguously
//! good news — a building where every event is a small disaster reads as a
//! building falling down.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

pub struct ComputeFloor {
    due: Due,
    rng: Rng,
    /// Jobs in flight. Above zero is [`Cond::Working`].
    running: u32,
    /// Nodes that have dropped and not come back.
    down: u32,
}

impl ComputeFloor {
    pub fn new(seed: u64) -> ComputeFloor {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(30), Duration::from_secs(200));
        ComputeFloor {
            due: Due::at(first),
            rng,
            running: 0,
            down: 0,
        }
    }
}

impl Fixture for ComputeFloor {
    fn id(&self) -> &'static str {
        "compute"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        if self.running > 0 {
            out.push(Cond::Working);
            out.push(Cond::Loud);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(60), Duration::from_secs(300)),
        );

        // An unstable supply takes nodes down; that is the cascade arriving.
        if w.is(Cond::Unstable) && self.rng.one_in(3) {
            self.down += 1;
            self.running = self.running.saturating_sub(1);
            return Some(Stirring::new(
                "compute",
                "A node drops out of the array, and the rest of it audibly takes up the slack.",
                Salience::NORMAL,
            ));
        }

        match self.rng.below(8) {
            0..=2 if self.running == 0 => {
                self.running += 1;
                Some(
                    Stirring::new(
                        "compute",
                        "The compute stacks spool up under load, and the floor picks up their hum.",
                        Salience::IDLE,
                    )
                    .tagged(&[Cond::Working, Cond::Loud]),
                )
            }
            3 if self.running > 0 => {
                self.running -= 1;
                // Deliberately untagged: the absence of `Working` is what the
                // air reads to ease off.
                Some(Stirring::new(
                    "compute",
                    "The compute stacks fall quiet as something long-running finishes.",
                    Salience::IDLE,
                ))
            }
            4 if self.down > 0 => {
                self.down -= 1;
                Some(Stirring::new(
                    "compute",
                    "A node that had dropped comes back into the array and starts taking work again.",
                    Salience::IDLE,
                ))
            }
            5 if self.down > 0 => Some(Stirring::new(
                "compute",
                "A fault light is up on the compute stacks, with no alarm behind it.",
                Salience::NORMAL,
            )),
            6 if self.running > 0 => Some(
                Stirring::new(
                    "compute",
                    "A coil somewhere in the compute stacks changes pitch and settles at the new one.",
                    Salience::IDLE,
                )
                .tagged(&[Cond::Working]),
            ),
            // The catch-all, and therefore the line the room hears most often —
            // so it is the one that needs the spread.
            _ => {
                let line = *self
                    .rng
                    .pick(&[
                        "The compute stacks tick over at idle, doing nothing anybody asked for.",
                        "A cooling fan on the compute stacks cycles up and back down on its own.",
                        "An indicator runs left to right along the front of the compute stacks \
                         and starts again.",
                        "Somewhere in the compute stacks a disk seeks, several times, and stops.",
                        "The compute stacks drop half a tone as something in them finishes.",
                        "A relay clicks over inside the compute stacks and nothing follows it.",
                    ])
                    .unwrap_or(&"The compute stacks tick over at idle.");
                Some(Stirring::new("compute", line, Salience::IDLE))
            }
        }
    }
}

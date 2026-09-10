//! What grows in a sealed building once something has been leaking for long
//! enough.
//!
//! # The payoff at the end of the slow burn
//!
//! This is the fixture that makes the coolant loop worth having. The coolant
//! goes Sound → Low → Hammering → Weeping → Pooling over the better part of an
//! hour, and if nobody sees to it there is standing water; this fixture eats
//! that water. So a character who ignored *"the coolant pressure drops off its
//! mark"* meets, an hour later, *"there is a bloom of something pale in the
//! corner where the wall meets the floor."* Nothing authored the link — the
//! coolant publishes [`Cond::Damp`] and this reads it.
//!
//! # It has to be slow to mean anything
//!
//! `fed` only rises while the building is damp, and only falls slowly, so a
//! bloom is genuinely evidence of a long-standing fault rather than a die roll.
//! [`Cond::Dark`] helps it and [`Cond::Cold`] holds it back, which is why the
//! lighting and the shell matter to it too.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Extent {
    /// Nothing to see. Where a well-kept building stays.
    None,
    /// A smell before anything is visible, which is the honest order.
    Smell,
    /// Visible, small, in the corner nobody looks at.
    Patch,
    /// Across a seam, into a duct, and now a problem.
    Spreading,
}

pub struct Growth {
    extent: Extent,
    due: Due,
    rng: Rng,
    /// How much damp it has had. The whole clock of the thing.
    fed: i32,
}

impl Growth {
    pub fn new(seed: u64) -> Growth {
        let mut rng = Rng::new(seed);
        // It starts late. Nothing grows in the first minute of anything.
        let first = rng.between(Duration::from_secs(400), Duration::from_secs(1200));
        Growth {
            extent: Extent::None,
            due: Due::at(first),
            rng,
            fed: 0,
        }
    }
}

impl Fixture for Growth {
    fn id(&self) -> &'static str {
        "growth"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        // Once it is into the seams it is holding water of its own.
        if self.extent >= Extent::Spreading {
            out.push(Cond::Damp);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(300), Duration::from_secs(1500)),
        );

        // Damp feeds it, dark helps, cold holds it back, and a dry building
        // slowly takes it away again.
        match w.is(Cond::Damp) {
            true => {
                self.fed += 2;
                if w.is(Cond::Dark) {
                    self.fed += 1;
                }
                if w.is(Cond::Cold) {
                    self.fed -= 1;
                }
            }
            false => self.fed -= 1,
        }
        self.fed = self.fed.clamp(0, 12);

        let want = match self.fed {
            0..=2 => Extent::None,
            3..=5 => Extent::Smell,
            6..=9 => Extent::Patch,
            _ => Extent::Spreading,
        };

        // Only speak when it has actually moved. A bloom that reports itself
        // every ten minutes is wallpaper.
        if want == self.extent {
            return match (self.extent, self.rng.one_in(4)) {
                (Extent::None, _) | (_, false) => None,
                (Extent::Smell, _) => Some(Stirring::new(
                    "growth",
                    "The air in this corner of the room has a wet, cellar-like smell to it.",
                    Salience::IDLE,
                )),
                (Extent::Patch, _) => Some(Stirring::new(
                    "growth",
                    "The pale patch on the wall by the floor is no smaller than it was.",
                    Salience::IDLE,
                )),
                (Extent::Spreading, _) => Some(
                    Stirring::new(
                        "growth",
                        "A run of dark growth has worked its way along the wall seam and into the \
                         cable tray.",
                        Salience::NORMAL,
                    )
                    .tagged(&[Cond::Damp]),
                ),
            };
        }

        let advancing = want > self.extent;
        self.extent = want;

        let (line, salience): (&str, Salience) = match (self.extent, advancing) {
            (Extent::Smell, true) => (
                "There is a faint sourness in the air that was not in it earlier.",
                Salience::IDLE,
            ),
            (Extent::Patch, true) => (
                "A bloom of something pale has come up where the wall meets the floor.",
                Salience::NORMAL,
            ),
            (Extent::Spreading, true) => (
                "The growth on the wall has reached the duct grille and is going in behind it.",
                Salience::NORMAL,
            ),
            (Extent::Patch, false) => (
                "The growth on the wall has dried back to a chalky outline.",
                Salience::IDLE,
            ),
            (Extent::Smell, false) => (
                "The wet smell in this corner has thinned out to almost nothing.",
                Salience::IDLE,
            ),
            (Extent::None, _) => (
                "The wall by the floor has dried out, and there is nothing left on it but a stain.",
                Salience::IDLE,
            ),
            (Extent::Spreading, false) => (
                "The growth in the cable tray has stopped where it is.",
                Salience::IDLE,
            ),
        };

        Some(Stirring::new("growth", line, salience))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // A fresh leak is a meal, and it does not wait for its own slow clock to
        // come round before it counts.
        if what.tags.contains(&Cond::Leaking) {
            self.fed = (self.fed + 1).min(12);
            if self.extent == Extent::None {
                self.due.hold(
                    w,
                    self.rng
                        .between(Duration::from_secs(200), Duration::from_secs(800)),
                );
            }
        }
    }
}

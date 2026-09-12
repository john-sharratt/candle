//! The coolant loop, and the building's one slow-burning failure.
//!
//! # A progression, not an event
//!
//! Everything else here recovers. This one does not, unless somebody sees to
//! it: pressure falls, a valve starts hammering, a seal weeps, a puddle forms,
//! and the puddle is what the growth feeds on hours later. A character that
//! ignored the first line meets the last one, which is the difference between
//! entropy that decorates and entropy that means something.
//!
//! It publishes [`Cond::Damp`] once it is losing fluid, and [`Cond::Leaking`]
//! for anything that wants to know why.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    /// Nominal. Ticks, hums, changes note.
    Sound,
    /// Off its mark. Recoverable, and says so.
    Low,
    /// Hammering. The point at which a person would go and look.
    Hammering,
    /// Weeping at a joint. Now there is water.
    Weeping,
    /// A puddle, and the level below it eventually knows.
    Pooling,
}

pub struct CoolantLoop {
    stage: Stage,
    due: Due,
    rng: Rng,
    /// How long it has been left. Only rises; the fixture cannot mend itself,
    /// which is the point of it.
    neglected: u32,
}

impl CoolantLoop {
    pub fn new(seed: u64) -> CoolantLoop {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(150), Duration::from_secs(700));
        CoolantLoop {
            stage: Stage::Sound,
            due: Due::at(first),
            rng,
            neglected: 0,
        }
    }

    /// How bad it has got. The growth asks.
    pub fn stage_depth(&self) -> u8 {
        self.stage as u8
    }
}

impl Fixture for CoolantLoop {
    fn id(&self) -> &'static str {
        "coolant"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        if self.stage >= Stage::Weeping {
            out.push(Cond::Damp);
            out.push(Cond::Leaking);
        }
        if self.stage >= Stage::Hammering {
            out.push(Cond::Loud);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }

        // A sound loop mostly just makes noise about itself.
        if self.stage == Stage::Sound {
            self.due.again(
                w,
                self.rng
                    .between(Duration::from_secs(120), Duration::from_secs(600)),
            );
            // It only goes wrong under strain, so a quiet building stays dry.
            if w.is(Cond::Unstable) && self.rng.one_in(3) {
                self.stage = Stage::Low;
                return Some(Stirring::new(
                    "coolant",
                    "The coolant pressure drops off its mark and does not come straight back.",
                    Salience::NORMAL,
                ));
            }
            let line = *self
                .rng
                .pick(&[
                    "The coolant pumps change note.",
                    "A coolant line ticks somewhere behind a panel as it cools.",
                    "Condensation runs down a coolant line and drips onto the floor plate.",
                    "The coolant loop settles into a lower, steadier note.",
                ])
                .unwrap_or(&"The coolant pumps change note.");
            return Some(Stirring::new("coolant", line, Salience::IDLE));
        }

        // Once it is off its mark it only goes one way.
        self.neglected += 1;
        let advance = self.neglected >= 2 && self.rng.one_in(2);
        if advance {
            self.stage = match self.stage {
                Stage::Low => Stage::Hammering,
                Stage::Hammering => Stage::Weeping,
                Stage::Weeping | Stage::Pooling => Stage::Pooling,
                Stage::Sound => Stage::Low,
            };
            self.neglected = 0;
        }

        // A stage is held for several passes before it advances, so each needs
        // more than one way of saying itself — otherwise the slow burn, which is
        // the most consequential thing the building does, is also the most
        // repetitive.
        let (lines, tags): (&[&str], &[Cond]) = match self.stage {
            Stage::Low => (
                &[
                    "The coolant pressure is still under its mark, and has been for a while now.",
                    "The coolant pumps are working harder than they should have to for the flow \
                     they are getting.",
                    "A gauge on the coolant gallery is sitting in the amber and not moving.",
                ],
                &[],
            ),
            Stage::Hammering => (
                &[
                    "A coolant valve hammers once, hard enough to be felt through the floor.",
                    "The coolant line knocks three times in quick succession and goes quiet.",
                    "There is air in the coolant loop, and it makes itself heard every time the \
                     pump comes round.",
                ],
                &[Cond::Loud],
            ),
            Stage::Weeping => (
                &[
                    "A coolant joint has begun to weep, and there is a wet line down the wall \
                     under it.",
                    "A drip has started somewhere behind the coolant gallery panelling, slow and \
                     regular.",
                    "The lagging on a coolant line has gone dark along its underside.",
                ],
                &[Cond::Damp, Cond::Leaking],
            ),
            Stage::Pooling => (
                &[
                    "Coolant has pooled on the floor under the loop, and is finding its way to \
                     the drain by the long route.",
                    "There is standing coolant across the floor plates now, wide enough to have \
                     to step over.",
                    "The puddle under the coolant gallery has reached the cable tray stanchion \
                     and is going round it.",
                ],
                &[Cond::Damp, Cond::Leaking],
            ),
            Stage::Sound => unreachable!("handled above"),
        };
        let line = self.rng.pick(lines).copied().unwrap_or(lines[0]);
        let salience = Salience::NORMAL;

        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(90), Duration::from_secs(400)),
        );
        Some(Stirring::new("coolant", line, salience).tagged(tags))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // Heat is what it is fighting. A hard-working compute floor brings the
        // next problem forward.
        if what.tags.contains(&Cond::Working) && self.stage > Stage::Sound {
            self.due.hold(
                w,
                self.rng
                    .between(Duration::from_secs(20), Duration::from_secs(90)),
            );
        }
    }
}

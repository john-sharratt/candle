//! The fabric — the hull, the seals, the doors, and what the cold does to all
//! three.
//!
//! # Why it is on the wall clock and nothing else is
//!
//! Everything else here runs on `since_start`, because a daemon that came up at
//! four in the morning should not behave differently from one that came up at
//! noon. The fabric is the exception: metal contracts as the outside cools, so
//! **the small hours are when a building talks to itself**. That is a real
//! property of a real base and it costs one line — [`Watch::hour`] — to have.
//!
//! # What it gives the rest
//!
//! * [`Cond::Cold`], which the growth reads (cold slows it) and which makes the
//!   creaking more likely in turn.
//! * [`Cond::Gusting`] when a seal lets go, which is the *second* way the rat
//!   gets frightened — the air handling is not the only thing in the vault that
//!   can move a volume of air suddenly.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

pub struct Structure {
    due: Due,
    rng: Rng,
    /// How far the fabric has cooled. Rises through the small hours, falls
    /// through the day, and decides how much the building creaks.
    chill: i32,
    /// A seal that has been complaining and has not been seen to.
    weak_seal: bool,
}

impl Structure {
    pub fn new(seed: u64) -> Structure {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(60), Duration::from_secs(400));
        Structure {
            due: Due::at(first),
            rng,
            chill: 0,
            weak_seal: false,
        }
    }

    /// Whether the hour is one of the cold ones. Between one and six the
    /// outside has had all night to take the heat out of the shell.
    fn small_hours(w: &Watch) -> bool {
        matches!(w.hour(), 1..=6)
    }
}

impl Fixture for Structure {
    fn id(&self) -> &'static str {
        "structure"
    }

    fn signals(&self, out: &mut Vec<Cond>) {
        if self.chill >= 3 {
            out.push(Cond::Cold);
        }
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(80), Duration::from_secs(420)),
        );

        // The shell tracks the hour, slowly.
        match Structure::small_hours(w) {
            true => self.chill = (self.chill + 1).min(6),
            false => self.chill = (self.chill - 1).max(0),
        }

        // A seal that has been complaining eventually lets go, and that is a
        // volume of air arriving somewhere it was not.
        if self.weak_seal && self.rng.one_in(3) {
            self.weak_seal = false;
            return Some(
                Stirring::new(
                    "structure",
                    "A door seal lets go all at once, with a bang and a shove of cold air across \
                     the floor.",
                    Salience::URGENT,
                )
                .tagged(&[Cond::Gusting, Cond::Loud, Cond::Cold]),
            );
        }

        // Cold fabric is a noisy fabric, so the odds change with the hour.
        let cold = self.chill >= 3;
        let roll = self.rng.below(if cold { 7 } else { 11 });
        let (line, salience, tags): (&str, Salience, &[Cond]) = match roll {
            0 => (
                "Something in the structure of the building settles with a single deep crack.",
                Salience::NORMAL,
                &[],
            ),
            1 if cold => (
                "The wall panels tick as the shell contracts against the cold outside.",
                Salience::IDLE,
                &[Cond::Cold],
            ),
            2 => {
                self.weak_seal = true;
                (
                    "A door seal hisses where it is not sitting properly, and keeps hissing.",
                    Salience::IDLE,
                    &[],
                )
            }
            3 => (
                "A pressure door somewhere further in cycles, closes, and locks.",
                Salience::IDLE,
                &[],
            ),
            4 => (
                "The floor plates shift underfoot as somebody's weight moves across them elsewhere.",
                Salience::IDLE,
                &[],
            ),
            5 if cold => (
                "Frost has come up on the inside of the outer window, in fern shapes.",
                Salience::IDLE,
                &[Cond::Cold],
            ),
            6 => (
                "A gantry overhead takes up its load with a long metallic groan.",
                Salience::IDLE,
                &[],
            ),
            7 => (
                "A hatch cover rattles once in its frame and is still.",
                Salience::IDLE,
                &[],
            ),
            8 => (
                "The building takes a gust from outside, and the whole shell leans into it.",
                Salience::NORMAL,
                &[],
            ),
            9 => (
                "Grit blows against the outer skin of the building in a long hiss.",
                Salience::IDLE,
                &[],
            ),
            _ => (
                "The joints in the walkway creak as the building takes up its own weight.",
                Salience::IDLE,
                &[],
            ),
        };

        Some(Stirring::new("structure", line, salience).tagged(tags))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // A sudden pressure change is what finishes a marginal seal, so the air
        // handling's purge shortens the fuse on a fault that was already there.
        if self.weak_seal && what.tags.contains(&Cond::Gusting) && what.from != "structure" {
            self.due.hold(
                w,
                self.rng
                    .between(Duration::from_secs(5), Duration::from_secs(30)),
            );
        }
    }
}

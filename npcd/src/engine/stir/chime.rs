//! The clock, which is the only thing in the vault that tells anybody what time
//! it is.
//!
//! # Why a clock earns its place
//!
//! Every other fixture is an event without a date — a vent purges, a light
//! fails, and none of it tells a character whether it has been here ten minutes
//! or six hours. A chime does. It is the cheapest possible sense of elapsed
//! time, it costs one comparison against [`Watch::hour`], and a character that
//! has heard three of them knows something it could not otherwise know.
//!
//! # It only strikes once
//!
//! `struck` is the hour it last sounded, so however often
//! [`crate::engine::stir::Building::next_event`] is called, the hour is rung
//! once. This is the one fixture where a repeat would be a bug rather than
//! merely dull.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Cond, Due, Fixture, Rng, Stirring, Watch};

pub struct Chime {
    rng: Rng,
    /// The hour it last struck. `None` before the first one, so a run that
    /// starts at twenty past does not immediately ring for an hour it missed.
    struck: Option<u64>,
    /// The half hour it last marked, on the same principle.
    halved: Option<u64>,
    /// A quiet fault of its own: a clock that is slow is a good detail, and a
    /// character noticing it is a character paying attention.
    slow: bool,
    due: Due,
}

impl Chime {
    pub fn new(seed: u64) -> Chime {
        let mut rng = Rng::new(seed);
        let slow = rng.one_in(4);
        Chime {
            rng,
            struck: None,
            halved: None,
            slow,
            due: Due::at(Duration::ZERO),
        }
    }

    /// How to say a number of strokes without a number, because the room reads
    /// prose and "The clock strikes 3." is not prose.
    fn strokes(h: u64) -> &'static str {
        const NAMES: [&str; 12] = [
            "twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven",
        ];
        NAMES[(h % 12) as usize]
    }
}

impl Fixture for Chime {
    fn id(&self) -> &'static str {
        "chime"
    }

    fn signals(&self, _out: &mut Vec<Cond>) {}

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        let secs = w.secs_today();
        let hour = secs / 3600;
        let into = secs % 3600;

        // The first look only records where the clock is. Ringing for an hour
        // that had already gone by before the daemon came up would be a lie.
        if self.struck.is_none() {
            self.struck = Some(hour);
            // And the same for the half, if this hour's has already gone.
            self.halved = (into >= 1800).then_some(hour);
            return None;
        }

        // A slow clock strikes late, which is the whole of the joke.
        let lag = match self.slow {
            true => 90,
            false => 0,
        };

        if self.struck != Some(hour) && into >= lag && into < lag + 300 {
            self.struck = Some(hour);
            let word = Chime::strokes(hour);
            let line = match self.slow {
                true => format!(
                    "The clock on the wall strikes {word}, a good while after it should have.",
                ),
                false => format!("The clock on the wall strikes {word}, unhurried, and stops."),
            };
            // Loud enough to be worth a turn, never loud enough to interrupt
            // one — an hour striking is not an emergency.
            return Some(Stirring::new("chime", line, Salience::NORMAL));
        }

        if self.halved != Some(hour) && (1800..2100).contains(&into) {
            self.halved = Some(hour);
            return Some(Stirring::new(
                "chime",
                "The clock on the wall gives a single stroke for the half hour.",
                Salience::IDLE,
            ));
        }

        // Between the hours it is only a mechanism, and only occasionally.
        if !self.due.ready(w) {
            return None;
        }
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(400), Duration::from_secs(1600)),
        );

        let line = *self
            .rng
            .pick(&[
                "The clock on the wall ticks over onto the next minute with an audible step.",
                "The second hand on the wall clock catches, and then goes on.",
                "The clock's escapement changes note for a few beats and settles again.",
            ])
            .unwrap_or(
                &"The clock on the wall ticks over onto the next minute with an audible step.",
            );
        Some(Stirring::new("chime", line, Salience::IDLE))
    }
}

//! The standing recordings — what the Creator left on the address system.
//!
//! # It holds no words of its own
//!
//! Every other fixture in this module carries its own prose, because a vent
//! sounds like a vent wherever it is fitted. This one carries none: the
//! recordings are *the world's*, authored on the building in
//! `<mind>/map/…`, handed in at construction, and quoted verbatim. A vault run
//! by different people says different things over the tannoy, and none of that
//! belongs in the engine.
//!
//! Given no recordings it is silent, which is the correct behaviour for a
//! building nobody left a message in — not a fallback, and not a default set
//! of words that would then be the engine's opinion about somebody else's
//! world.
//!
//! # Why it deals from a bag
//!
//! The one thing a recording must not do is come round twice while the first
//! is still in mind. Drawing at random gets you a repeat every dozen plays
//! however long the list is; dealing from a shuffled bag and only reshuffling
//! when it is empty means **every recording plays once before any plays
//! twice**. With a couple of dozen loaded and one every half hour, that is a
//! day between repeats.

use std::time::Duration;

use crate::engine::event::Salience;
use crate::engine::stir::{Due, Fixture, Rng, Stirring, Watch};

/// How a recording announces itself. The framing is the engine's — a tannoy is
/// a tannoy — and the words after the colon are the world's.
const FRAMINGS: &[&str] = &[
    "The address system plays one of the standing recordings",
    "A recorded voice comes up on the address system",
    "The address system works down to the bottom of its notice list",
    "One of the old recordings plays itself out over the address system",
    "The address system clears its throat and plays a standing notice",
];

pub struct Broadcast {
    /// The world's recordings, in the order it gave them.
    said: Vec<String>,
    /// Indices not yet played this time round, in the order they will play.
    bag: Vec<usize>,
    due: Due,
    rng: Rng,
}

impl Broadcast {
    /// A tannoy loaded with whatever the world left on it.
    pub fn new(seed: u64, said: Vec<String>) -> Broadcast {
        let mut rng = Rng::new(seed);
        let first = rng.between(Duration::from_secs(300), Duration::from_secs(1500));
        Broadcast {
            said,
            bag: Vec::new(),
            due: Due::at(first),
            rng,
        }
    }

    /// How many recordings are loaded. A vault with two of them will sound like
    /// a vault with two of them, and that is worth being able to check.
    pub fn loaded(&self) -> usize {
        self.said.len()
    }

    /// The next one, dealt from the bag. Refills and shuffles when empty.
    fn draw(&mut self) -> Option<&str> {
        if self.bag.is_empty() {
            self.bag = (0..self.said.len()).collect();
            // Fisher-Yates, so the refill is a genuine reshuffle rather than
            // the same order with a different starting point.
            for i in (1..self.bag.len()).rev() {
                self.bag.swap(i, self.rng.below(i + 1));
            }
        }
        let at = self.bag.pop()?;
        self.said.get(at).map(String::as_str)
    }
}

impl Fixture for Broadcast {
    fn id(&self) -> &'static str {
        "broadcast"
    }

    fn consider(&mut self, w: &Watch) -> Option<Stirring> {
        if self.said.is_empty() || !self.due.ready(w) {
            return None;
        }
        // Half an hour or so between them. Often enough to be part of the
        // building, rare enough that a character still looks up.
        self.due.again(
            w,
            self.rng
                .between(Duration::from_secs(900), Duration::from_secs(2700)),
        );

        let framing = self.rng.pick(FRAMINGS).copied().unwrap_or(FRAMINGS[0]);
        let words = self.draw()?;
        // Worth a turn, never an interruption. A recording that stopped a
        // character mid-sentence would be the building shouting at people, and
        // it is a building talking to itself.
        Some(Stirring::new(
            "broadcast",
            format!("{framing}: \"{words}\""),
            Salience::NORMAL,
        ))
    }

    fn notice(&mut self, what: &Stirring, w: &Watch) {
        // It defers to a live alarm rather than talking over one. A standing
        // notice about tidying your bench, played under an evacuation tone, is
        // funny exactly once.
        if what.salience.preempts() && what.from != "broadcast" {
            // `defer`, not `hold`: this is "wait at least this long", and `hold`
            // would happily pull a distant recording *forward* to meet it.
            //
            // And it is the length of an alarm, not the length of a gap between
            // recordings. Something loud happens often enough that deferring by
            // a full gap each time starved the tannoy down to four plays in a
            // day — every deferral pushing past the next one, for ever. All this
            // has to do is let the noise finish.
            self.due.defer(
                w,
                self.rng
                    .between(Duration::from_secs(240), Duration::from_secs(600)),
            );
        }
    }
}

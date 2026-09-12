//! The clock that lets the building speak, one room at a time.
//!
//! [`crate::engine::stir`] builds a vault that has things happen in it.
//! Something has to ask it, and this is that something.
//!
//! # A building per room, and only for rooms somebody is in
//!
//! The obvious design is one [`Building`] for the whole world, and it is wrong.
//! A rat is not in the whole world, it is in a room — so a single building
//! would have the same rat bolt across the floor of two levels at once, and a
//! character walking next door to look would find the one place it certainly is
//! not. Each room gets its own, and they are unrelated: the coolant gallery on
//! the casting floor can be pooling while the one on the chronicle level is
//! sound, because they are different pipes.
//!
//! Rooms are fitted lazily, the first time anybody stands in one, and are never
//! asked while empty. That is not only an economy — **a room nobody is in has
//! no history, and nobody can contradict it.** Advancing an unwatched room
//! would burn its whole slow burn against an empty floor, so a Maker walking
//! into the coolant gallery for the first time would find it already pooling
//! for reasons nobody witnessed.
//!
//! # The timer is per room, and jittered
//!
//! A fixed interval synchronises the building: every room speaking on the same
//! beat is one loud world rather than several quiet ones. Each room draws its
//! own gap in [`WAIT`], so two Makers on two levels are never being interrupted
//! together.
//!
//! The gap is also the **idle thinking rate**. Nothing polls any more — a mind
//! thinks when something reaches it and not otherwise — so in a room where
//! nobody is talking, this is the only thing that will ever wake anybody. Too
//! short and every character is thinking about a fan every minute; too long and
//! a Maker left alone is inert. A few minutes is a person noticing their
//! surroundings about as often as a person does.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::{Duration, Instant};

use npc_map::salience::Weight;
use npc_map::world::{Where, World};

use crate::engine::event::Salience;
use crate::engine::stir::{Building, Stirring};

/// How long a room waits between looks. Drawn afresh after every one.
///
/// **This is the idle thinking rate**, not merely a rate of scenery: with idle
/// ticks gone, a character with nobody to talk to thinks when its room does
/// something and at no other time. Three rooms on a two-to-seven minute timer
/// gave a cast that took five turns in six minutes, which reads as a dead world
/// however healthy it is — so the dial is here and it is short.
///
/// Still a range, and still jittered, for the reason it always was: rooms on a
/// fixed interval speak on the same beat, and a building that does everything at
/// once is one loud world rather than several quiet ones.
pub const WAIT: Range<Duration> = Duration::from_secs(10)..Duration::from_secs(30);

/// A room that has somebody in it, and what its building is doing.
struct Room {
    building: Building,
    /// When it is next worth asking, measured from [`Rooms::started`].
    due: Duration,
}

/// Every room in one world that anybody has stood in.
pub struct Rooms {
    /// When this world's fixtures started. All of their clocks run from here,
    /// so a room fitted an hour in does not begin its coolant loop as though
    /// the daemon had only just come up.
    started: Instant,
    fitted: BTreeMap<Where, Room>,
    /// Seeds the per-room buildings, so two rooms fitted in the same second do
    /// not run identically.
    next_seed: u64,
}

impl Default for Rooms {
    fn default() -> Rooms {
        Rooms::new()
    }
}

impl Rooms {
    pub fn new() -> Rooms {
        Rooms {
            started: Instant::now(),
            fitted: BTreeMap::new(),
            next_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// How many rooms are running. What a health check and a test read.
    pub fn len(&self) -> usize {
        self.fitted.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fitted.is_empty()
    }

    /// Ask every occupied room whether anything happened, and write what did
    /// into the world.
    ///
    /// Returns how many rooms stirred. Called on the world's own clock, under
    /// the world's own lock, **before perception is taken** — a stirring that
    /// landed after the sweep would be read a moment late by everybody, and the
    /// character that walked in during that moment would never read it at all.
    pub fn stir(&mut self, world: &mut World) -> usize {
        self.stir_at(world, self.started.elapsed())
    }

    /// The same, at a stated point in the run rather than at the real one.
    ///
    /// The split is the same one [`Building::next_event`] makes and is there for
    /// the same reason: the fixtures are stateful and time-aware, so the only
    /// way to see a coolant loop reach its fourth stage is to be able to say
    /// when it is. A test that had to wait out `WAIT` in real seconds would not
    /// be written.
    pub fn stir_at(&mut self, world: &mut World, since: Duration) -> usize {
        let mut occupied: Vec<Where> = world.actors().map(|a| a.at.clone()).collect();
        occupied.sort();
        occupied.dedup();

        let mut spoke = 0;
        for at in occupied {
            let room = self.fit(world, &at, since);
            if since < room.due {
                continue;
            }
            // Drawn before the event rather than after, so a room whose
            // fixtures were all quiet still waits its turn — otherwise an
            // unlucky room is asked every 500ms until something answers.
            room.due = since + room.building.jitter(WAIT.start, WAIT.end);
            let Some(s) = room.building.next_event(since) else {
                continue;
            };
            let rung = rung(&s);
            // A place the map does not hold is the one way this can fail, and it
            // is worth saying so: the room was fitted from an actor standing
            // there, so a refusal means a body is somewhere the map does not
            // have, which is a fault well upstream of here.
            match world.stir(&at, &s.text, rung) {
                Ok(()) => {
                    // **Debug, because Pulse already shows it.** A stirring that
                    // reaches anybody appears in that character's tick as what
                    // it perceived, beside what it did about it, which is both
                    // more use than a log line and where somebody would look.
                    // This says the same thing one step earlier, which is only
                    // worth reading when the two disagree — a room that spoke
                    // and nobody heard.
                    tracing::debug!(
                        room = %format!("{}/{}", at.area, at.node),
                        from = s.from,
                        weight = ?rung,
                        "{}",
                        s.text
                    );
                    spoke += 1;
                }
                Err(e) => tracing::warn!(
                    room = %format!("{}/{}", at.area, at.node),
                    "the building could not speak: {e:?}"
                ),
            }
        }
        spoke
    }

    /// The room's building, fitting one if this is the first time anybody has
    /// stood here.
    fn fit(&mut self, world: &World, at: &Where, since: Duration) -> &mut Room {
        if !self.fitted.contains_key(at) {
            // The world's own recordings, off the map. The engine holds none —
            // see `stir::broadcast`.
            let said = world.map().announcements_for(&at.area).to_vec();
            let recordings = said.len();
            let mut building = Building::new(self.next_seed, said);
            // Advanced whether or not this room ever speaks, so no two rooms
            // fitted in one pass run identically.
            self.next_seed = self
                .next_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // The first look is one ordinary wait away, not immediate: a Maker
            // walking into a room should meet the room, not a fan changing note
            // in the same instant it arrives.
            let due = since + building.jitter(WAIT.start, WAIT.end);
            tracing::debug!(
                room = %format!("{}/{}", at.area, at.node),
                fixtures = building.len(),
                recordings,
                first_look_s = (due - since).as_secs(),
                "a building is now running in a room somebody is standing in"
            );
            self.fitted.insert(at.clone(), Room { building, due });
        }
        self.fitted.get_mut(at).expect("fitted above")
    }
}

/// A stirring's salience as the ladder the map speaks in.
///
/// Two scales meet here and neither is going away. `stir` reasons in floats
/// because a fixture wants to say *slightly more than nothing*; the map reasons
/// in named rungs because [`Weight`] is deliberately not tunable. The bands are
/// the map's own bars, so the mapping is a lookup rather than a judgement:
/// anything that would preempt does, anything that would end a standing wait
/// wakes, and the rest is a note or ambient.
fn rung(s: &Stirring) -> Weight {
    match s.salience.get() {
        v if v >= Salience::PREEMPT_AT => Weight::Preempt,
        v if v >= Salience::ROUSES_AT => Weight::Wake,
        v if v >= Weight::Note.as_f32() => Weight::Note,
        _ => Weight::Ambient,
    }
}

#[cfg(test)]
mod tests;

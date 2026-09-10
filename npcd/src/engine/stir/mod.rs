//! What a building does when nobody is doing anything.
//!
//! # Why this exists
//!
//! A character thinks when something reaches it and not otherwise — see
//! [`crate::engine::tick`]. That rule is right, and it has one consequence: a
//! room where nothing happens produces characters who never think. Two Makers
//! standing in a quiet vault are not idle, they are inert, and the world has no
//! way to start.
//!
//! The wrong fix is to give characters their own clock back, which is what had
//! one of them gesturing into an empty room every four seconds until its window
//! held nothing but its own last sentence. The right one is that **a research
//! base is not quiet.** Air moves, seals cycle, compute runs, something gets
//! into the ducts. Those are facts about the building; they happen whether or
//! not anybody is thinking; and a character perceiving one has something real
//! to react to.
//!
//! # Why the fixtures watch each other
//!
//! A list of sentences, however long, is memoryless — two draws from it are
//! unrelated, so a room is never *going* anywhere. What makes a building feel
//! alive is that its parts are coupled: the air handling purges, the gust
//! startles whatever is living behind the panelling, it bolts across the floor
//! and puts something over. Three fixtures, one incident, and none of it
//! authored as a sentence.
//!
//! So a fixture is not a list. It is a small stateful thing that
//!
//!   * keeps its own state and its own clock,
//!   * **publishes** what is true of it right now, as [`Cond`] flags,
//!   * **reads** every sibling's flags before deciding whether to speak, and
//!   * **is told** what any sibling just did, and may change its mind about
//!     itself as a result.
//!
//! [`Building::next_event`] is the whole loop: gather the flags, ask every
//! fixture in turn whether it wants to speak, take one, and tell everybody what
//! happened. Coupling is therefore a property of the *fixtures*, not of a table
//! somewhere — a new one wires itself in by reading the flags it cares about.
//!
//! # A stirring names its own subject
//!
//! Every line is a whole sentence saying what it is about: *"The lights in the
//! ceiling flicker and steady."* Never *"flickers and steadies."*
//!
//! Several of these run together in a room's prose, and a line opening with a
//! bare pronoun attaches itself to whichever subject came last — so a character
//! reasons about the wrong thing having happened, which is worse than being
//! told nothing. `npc_map::part::Part::short` carries the same rule for the same
//! reason, and `every_line_names_its_own_subject` holds it here.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::engine::event::Salience;

pub mod air;
pub mod ambient;
pub mod announce;
pub mod boards;
pub mod broadcast;
pub mod chime;
pub mod compute;
pub mod coolant;
pub mod growth;
pub mod lights;
pub mod power;
pub mod rat;
pub mod stores;
pub mod structure;

/// Something that is true of the building right now, that another fixture may
/// care about.
///
/// **The whole coupling surface.** A fixture publishes these about itself and
/// reads them about everything else, so no fixture needs to know another's type
/// — the rat does not know what a vent is, it knows what [`Cond::Gusting`]
/// means. A new fixture couples itself in by publishing and reading flags, and
/// nothing central has to be edited to allow it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cond {
    /// Air is moving hard — a purge, a gust, a pressure release.
    Gusting,
    /// The lights are out or badly down.
    Dark,
    /// Cold enough to notice.
    Cold,
    /// Wet enough for things to grow.
    Damp,
    /// The supply is not steady.
    Unstable,
    /// Loud enough to cover a small sound.
    Loud,
    /// Quiet enough that a small sound carries.
    Quiet,
    /// Something is loose in the building that should not be.
    Vermin,
    /// An alarm is up.
    Alarmed,
    /// Something is losing fluid.
    Leaking,
    /// The compute floor is working hard.
    Working,
}

/// One thing the building did.
#[derive(Clone, Debug)]
pub struct Stirring {
    /// Which fixture did it.
    pub from: &'static str,
    /// What happened, as the room reads it. A whole sentence naming its own
    /// subject — see the module note.
    pub text: String,
    /// How much it demands. **Most of these must sit below the preempt bar**
    /// (`Salience::PREEMPT_AT`): a building that stops everybody every time a
    /// fan changes note is the four-second treadmill in better prose.
    pub salience: Salience,
    /// What this event asserts about the building, for siblings to react to.
    ///
    /// Carried on the event rather than only on the fixture's own flags because
    /// a *moment* is not a state: a gust is over by the time anybody polls, and
    /// the rat has to hear it happen rather than find it still happening.
    pub tags: Vec<Cond>,
}

impl Stirring {
    pub fn new(from: &'static str, text: impl Into<String>, salience: Salience) -> Stirring {
        Stirring {
            from,
            text: text.into(),
            salience,
            tags: Vec::new(),
        }
    }

    pub fn tagged(mut self, tags: &[Cond]) -> Stirring {
        self.tags.extend_from_slice(tags);
        self
    }
}

/// The building as it stands, for a fixture deciding what to do.
pub struct Watch {
    /// How long the daemon has been up. Fixtures schedule against this rather
    /// than against a wall clock, so a world that starts at an odd hour behaves
    /// the same as one that starts on the hour.
    pub since_start: Duration,
    /// Real time, for the things that genuinely care what hour it is.
    pub wall: SystemTime,
    /// Everything every fixture is currently publishing about itself.
    pub conds: Vec<Cond>,
}

impl Watch {
    pub fn is(&self, c: Cond) -> bool {
        self.conds.contains(&c)
    }

    /// Hour of the day, 0–23, in whatever the host's clock says.
    pub fn hour(&self) -> u64 {
        self.secs_today() / 3600
    }

    /// Seconds since local midnight, near enough — the chime wants to land on
    /// the hour and does not care which side of a leap second it is.
    pub fn secs_today(&self) -> u64 {
        self.wall
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() % 86_400)
            .unwrap_or(0)
    }
}

/// One part of the building that can act on its own.
pub trait Fixture: Send {
    /// What it is, for the feed and for a sibling naming a source.
    fn id(&self) -> &'static str;

    /// What is true of it right now. Read by every sibling before it decides.
    fn signals(&self, _out: &mut Vec<Cond>) {}

    /// Whether it wants to speak, and what it says.
    ///
    /// Called with the building's state around it. Returning `None` is the
    /// ordinary answer — most fixtures are quiet most of the time, and a
    /// fixture that always has something to say is a fixture nobody can hear.
    fn consider(&mut self, w: &Watch) -> Option<Stirring>;

    /// What a sibling just did. Free to change this fixture's own state.
    ///
    /// **This is where coupling lives.** The rat learns it has been startled
    /// here, the growth learns it has been given damp, the insects learn the
    /// lights have gone.
    fn notice(&mut self, _what: &Stirring, _w: &Watch) {}
}

/// A small deterministic generator.
///
/// Its own rather than a dependency: the whole need is a few bits of jitter per
/// fixture, and seeding it per fixture keeps one busy part from shifting the
/// phase of every other — which is what makes a shared generator produce
/// buildings that lurch in unison.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // Stir the seed before using it. Xorshift takes a long time to
        // decorrelate two nearby starting states, and a bare `seed | 1` is
        // worse than that — it maps 2 and 3 to the same generator, so two
        // buildings a seed apart ran identically. One splitmix64 round
        // separates them, and its output is only zero for one input in 2^64.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        // Never zero: xorshift is stuck there for ever.
        Rng((z ^ (z >> 31)) | 1)
    }

    /// Not `next`: a `next(&mut self) -> u64` on a plain struct reads as an
    /// iterator that has forgotten to be one, and clippy says so.
    pub fn roll(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// `0..n`, for small `n`.
    pub fn below(&mut self, n: usize) -> usize {
        match n {
            0 => 0,
            n => (self.roll() % n as u64) as usize,
        }
    }

    /// True with probability `1/n`.
    pub fn one_in(&mut self, n: usize) -> bool {
        n > 0 && self.below(n) == 0
    }

    /// One of these, or `None` if empty.
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> Option<&'a T> {
        match xs.is_empty() {
            true => None,
            false => xs.get(self.below(xs.len())),
        }
    }

    /// A duration between `lo` and `hi`.
    pub fn between(&mut self, lo: Duration, hi: Duration) -> Duration {
        let (lo, hi) = (lo.as_millis() as u64, hi.as_millis() as u64);
        match hi > lo {
            true => Duration::from_millis(lo + self.roll() % (hi - lo)),
            false => Duration::from_millis(lo),
        }
    }
}

/// A due time that a fixture keeps for itself.
///
/// Every fixture has at least one of these. Kept as a *deadline* rather than a
/// countdown so nothing has to be told how much time passed, which means a
/// fixture is correct however irregularly [`Building::next_event`] is called.
#[derive(Clone, Debug)]
pub struct Due(Duration);

impl Due {
    pub fn at(when: Duration) -> Due {
        Due(when)
    }

    pub fn ready(&self, w: &Watch) -> bool {
        w.since_start >= self.0
    }

    pub fn again(&mut self, w: &Watch, gap: Duration) {
        self.0 = w.since_start + gap;
    }

    /// Bring it forward — for a fixture that has just heard something it wants
    /// to react to sooner than its own clock would have let it.
    pub fn hold(&mut self, w: &Watch, gap: Duration) {
        self.again(w, gap);
    }

    /// Push it back, never forward.
    ///
    /// The distinction from [`Due::hold`] is not cosmetic. `hold` *sets* the
    /// deadline, so a fixture using it to say "not for a while yet" will
    /// happily pull a distant deadline nearer and get the opposite of what it
    /// asked for. This takes the later of the two, which is what "wait at least
    /// this long" actually means.
    pub fn defer(&mut self, w: &Watch, gap: Duration) {
        self.0 = self.0.max(w.since_start + gap);
    }
}

/// How often the building's answer is simply that nothing much happened.
///
/// **This is the plausibility dial, and it belongs to the parent.** Each
/// mechanism fixture is individually reasonable — a supply dips a few times a
/// shift, a light fails now and then, a loop goes off its mark once in a while
/// — but a dozen individually reasonable mechanisms all reporting into one room
/// is a base where something goes wrong every twenty seconds, which is a base
/// about to be evacuated rather than one being worked in.
///
/// No single fixture can fix that, because none of them can see the others'
/// rate. The mix between *the building at rest* and *the building doing
/// something* is a property of the whole building, so the whole building is
/// where it is set: this many times in [`AT_REST`], [`ambient::Ambient`] is
/// asked first and, being nearly always ready, answers.
const AT_REST: usize = 4;

/// Everything in the building that can act on its own.
pub struct Building {
    fixtures: Vec<Box<dyn Fixture>>,
    started: SystemTime,
    /// Where [`ambient::Ambient`] sits, found by id rather than assumed, so
    /// reordering the list above cannot silently turn the dial off.
    at_rest: usize,
    rng: Rng,
}

impl Building {
    /// The building, with everything in it running.
    ///
    /// `said` is the world's standing recordings — whatever the people who
    /// built this place left on the address system, in their own words. Every
    /// other fixture here is a fact about buildings in general and carries its
    /// own prose; this is the one thing that is a fact about *this* building,
    /// so it comes in from outside and the engine never authors it. An empty
    /// list is a building nobody left a message in, and it says nothing.
    pub fn new(seed: u64, said: Vec<String>) -> Building {
        let mut rng = Rng::new(seed);
        let mut seed_for = || rng.roll();
        let fixtures: Vec<Box<dyn Fixture>> = vec![
            Box::new(ambient::Ambient::new(seed_for())),
            Box::new(broadcast::Broadcast::new(seed_for(), said)),
            Box::new(air::AirHandling::new(seed_for())),
            Box::new(lights::Lighting::new(seed_for())),
            Box::new(power::PowerBus::new(seed_for())),
            Box::new(coolant::CoolantLoop::new(seed_for())),
            Box::new(compute::ComputeFloor::new(seed_for())),
            Box::new(structure::Structure::new(seed_for())),
            Box::new(rat::Rat::new(seed_for())),
            Box::new(growth::Growth::new(seed_for())),
            Box::new(announce::Announcements::new(seed_for())),
            Box::new(chime::Chime::new(seed_for())),
            Box::new(boards::Boards::new(seed_for())),
            Box::new(stores::Stores::new(seed_for())),
        ];
        let at_rest = fixtures
            .iter()
            .position(|f| f.id() == "ambient")
            .expect("the building is always fitted with its quiet");
        Building {
            fixtures,
            started: SystemTime::now(),
            at_rest,
            rng: Rng::new(seed ^ 0x5eed),
        }
    }

    /// How many fixtures are running.
    pub fn len(&self) -> usize {
        self.fixtures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fixtures.is_empty()
    }

    /// The building as it stands, at `since_start` into the run.
    fn watch(&self, since_start: Duration) -> Watch {
        let mut conds = Vec::new();
        for f in &self.fixtures {
            f.signals(&mut conds);
        }
        conds.sort();
        conds.dedup();
        Watch {
            since_start,
            wall: self.started + since_start,
            conds,
        }
    }

    /// Ask the building whether anything just happened.
    ///
    /// # The loop
    ///
    /// 1. Gather what every fixture is publishing, so each of them decides
    ///    against the same picture rather than against a building that changes
    ///    underneath them mid-pass.
    /// 2. Ask each in turn, from a **random** start, until one speaks. At most
    ///    one thing happens per call: a building that emitted from every fixture
    ///    that happened to be ready would arrive in bursts and be silent
    ///    between them.
    ///
    ///    The start has to be random rather than a rotating cursor. The room
    ///    asks every few minutes and a dozen fixtures run on timers of one to
    ///    ten, so on most calls several are ready at once and whoever is asked
    ///    first wins. A `+1` cursor under those conditions is not a tie-break,
    ///    it is a **rota** — air, lights, power, coolant, compute, structure,
    ///    round and round — and a building that takes its turns in order reads
    ///    as a list being walked, which is the thing this module exists to stop
    ///    being.
    /// 3. Tell **every** fixture what happened, including the one that did it.
    ///    This is the half that makes an incident rather than an event — the
    ///    gust is what the rat hears, and the rat bolting is what knocks
    ///    something over.
    ///
    /// `None` means the building was quiet, which is the ordinary answer.
    pub fn next_event(&mut self, since_start: Duration) -> Option<Stirring> {
        let w = self.watch(since_start);

        let n = self.fixtures.len();
        // Most of the time the building is asked whether it is quiet, and it
        // is. The rest of the time anybody may answer. See [`AT_REST`].
        let start = match self.rng.below(AT_REST) {
            0 => self.rng.below(n),
            _ => self.at_rest,
        };

        let mut spoke = None;
        for i in 0..n {
            let at = (start + i) % n;
            if let Some(s) = self.fixtures[at].consider(&w) {
                spoke = Some(s);
                break;
            }
        }

        let s = spoke?;
        // Everybody hears it, including whoever did it — a fixture is allowed to
        // react to its own noise, and several do.
        for f in &mut self.fixtures {
            f.notice(&s, &w);
        }
        Some(s)
    }

    /// Run the building forward and collect what it said.
    ///
    /// For a caller that wants a stretch of building rather than a moment —
    /// the room timer asks for one at a time; this is what the tests and the
    /// demonstration use.
    pub fn run(
        &mut self,
        from: Duration,
        step: Duration,
        steps: usize,
    ) -> Vec<(Duration, Stirring)> {
        let mut out = Vec::new();
        let mut t = from;
        for _ in 0..steps {
            if let Some(s) = self.next_event(t) {
                out.push((t, s));
            }
            t += step;
        }
        out
    }

    /// A little jitter, for a caller scheduling the next look.
    pub fn jitter(&mut self, lo: Duration, hi: Duration) -> Duration {
        self.rng.between(lo, hi)
    }
}

#[cfg(test)]
mod tests;

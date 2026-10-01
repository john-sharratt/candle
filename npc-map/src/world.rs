//! Who is where, and what they are holding.
//!
//! The map says what the vault *is*; this says what is true of it right now.
//! One [`World`] is shared by every NPC in it — sixteen Makers reading and
//! changing the same state — so everything here is written for that case
//! rather than adapted to it later.
//!
//! # Shared means three things, not one
//!
//! **A claim is global.** Holding a character on the casting level makes that
//! character unholdable at an easel two floors up, because the reason to claim
//! one is coherence and coherence does not partition by floor. [`World::take`]
//! checks every hold in the building, not the ones in the room.
//!
//! **Every action is one step.** Checking a station is free and then sitting
//! at it is two operations and a race; [`World::take`] does both or neither.
//! That is the whole reason state changes live here rather than in the caller.
//!
//! **What one does, the next sees.** A percept is computed from this state at
//! the moment it is asked for, so a station taken by one Maker is a station
//! the next Maker sees taken. Nothing is cached per NPC except how far it has
//! read the log.
//!
//! The synchronisation story is therefore one lock around one `World`. Every
//! mutation is a handful of map operations, so a shared mutex costs nothing
//! worth measuring and buys an invariant that is otherwise impossible to hold:
//! there is exactly one answer to *who has that character*.
//!
//! # What this is not
//!
//! It records events; it does not interpret them. Who could make out what is
//! [`crate::witness`]'s question, and what is true of a place at an instant is
//! [`crate::perceive`]'s. This module's whole job is that both of them are
//! reading the same state and neither of them can change it behind the other's
//! back.
//!
//! # Leaving is releasing
//!
//! A station holds its subject *until you leave*, which is what the level
//! descriptions say and therefore what this does: [`World::set_off`] releases
//! whatever the mover was holding. The alternative — refusing to let a holder
//! walk away — makes a forgotten claim permanent, and sixteen robots that
//! never sleep would fill the building with them.
//!
//! # Moving takes time; the world reports how it went
//!
//! [`World::set_off`] starts a journey and returns its length. The body then
//! covers one place per [`World::step`], and learns it arrived — or that it
//! never will — from an event, because by the time there is an answer the
//! question is several ticks old.
//!
//! This is the one place where an actor is told about its own doings. Every
//! other event about you is something you already know; a journey's outcome is
//! not, and [`Happening::is_outcome`] is what marks the difference.
//!
//! The exception, [`World::teleport`], goes to exactly one place in the
//! building and goes there instantly. Everything else is walked.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use crate::lift::{Lift, Moment};
use crate::load::MapSet;
use crate::mutate::MapEdit;
use crate::part::PartKind;
use crate::route;
use crate::salience::Weight;
use crate::schema::{AreaKind, Node, NodeKind};
use crate::witness::{Reach, Scope};

pub use crate::schema::Where;

/// A step of world time. Advances once per change, so ordering is total and
/// "since you last looked" is a comparison rather than a diff.
///
/// It is not a clock. One tick is one thing happening, and everything that
/// happens in the same [`World::step`] shares a tick, because a step is one
/// moment however many bodies move in it.
pub type Tick = u64;

/// A station taken, and what it claims.
///
/// `subject` is `None` for a station that binds nothing — the watch desk,
/// which is worked at and holds nobody.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hold {
    pub subject: Option<String>,
    pub part: String,
    pub since: Tick,
}

/// A journey in progress: where it is going, and what is left of the way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Walk {
    pub toward: Where,
    /// The places still to be reached, in order. One per [`World::step`].
    pub ahead: VecDeque<Where>,
    pub set_out: Tick,
}

impl Walk {
    /// How many doorways are left. What the *map* costs; not what the journey
    /// costs, which is [`Walk::to_go`].
    pub fn moves_left(&self) -> usize {
        self.ahead.len()
    }

    /// Where the next leg ends — as far as a body gets in one tick.
    ///
    /// A leg is a maximal run of the route that stays on one side of an area
    /// boundary: everything up to the lift, then the ride, then everything
    /// from the lift to the door. Those are the places somebody would actually
    /// stop, so they are the places a journey pauses and a mind gets a turn.
    pub fn leg_end(&self, from: &Where) -> Option<Where> {
        let first = self.ahead.front()?;
        let crossing = first.area != from.area;
        let mut prev = first;
        let mut end = first;
        for next in self.ahead.iter().skip(1) {
            if (next.area != prev.area) != crossing {
                break;
            }
            prev = next;
            end = next;
        }
        Some(end.clone())
    }

    /// How many stops are left — how many ticks, counting the one that
    /// arrives. Never zero: a journey with nothing left has been taken off.
    pub fn to_go(&self, from: &Where) -> usize {
        let mut stops = 0;
        let mut at = from.clone();
        let mut rest = self.clone();
        while let Some(end) = rest.leg_end(&at) {
            let covered = rest
                .ahead
                .iter()
                .position(|w| w == &end)
                .expect("a leg ends on the route it was cut from");
            rest.ahead.drain(..=covered);
            at = end;
            stops += 1;
        }
        stops
    }
}

/// One body in the world.
#[derive(Clone, Debug)]
pub struct Actor {
    pub id: String,
    pub name: String,
    pub at: Where,
    pub hold: Option<Hold>,
    /// The journey under way, if any. A walking body is always standing
    /// somewhere real — there is no space between rooms — so `at` is the last
    /// place reached and this is the rest of the way.
    pub walk: Option<Walk>,
    /// How far this actor has read the log. Everything after it is new.
    pub looked: Tick,
}

/// Why a journey ended before it arrived.
///
/// Every one of these is something the walker itself did. Nothing in the vault
/// stops a body getting where it is going — no locked doors, no crowding in a
/// corridor — so a walk that fails failed because its owner changed its mind,
/// and the reason names which way it changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lost {
    /// Set off somewhere else instead.
    Diverted,
    /// Teleported away mid-route.
    Teleported,
    /// Stopped at a station along the way.
    SatDown,
}

impl fmt::Display for Lost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Phrased to follow a clause about the journey that failed: "never got
        // to the command room, having teleported."
        match self {
            Lost::Diverted => write!(f, "having set off somewhere else"),
            Lost::Teleported => write!(f, "having teleported"),
            Lost::SatDown => write!(f, "having stopped to work"),
        }
    }
}

/// Something that happened somewhere.
///
/// Three of these — [`SetOut`](Happening::SetOut), [`GotThere`](Happening::GotThere)
/// and [`LostTheWay`](Happening::LostTheWay) — happen *to* one body and are
/// seen by nobody else. Intent is not visible and neither is arriving; what a
/// bystander sees is a person leaving a room and a person coming into one,
/// which is [`Left`](Happening::Left) and [`Arrived`](Happening::Arrived).
/// [`crate::witness`] enforces that; the record keeps all of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Happening {
    Arrived,
    Left,
    /// Began a journey. Private to the walker.
    SetOut {
        toward: Where,
    },
    /// The journey finished. Private to the walker, and the answer to a
    /// question it asked several ticks ago.
    GotThere {
        toward: Where,
    },
    /// The journey ended without arriving. Private to the walker.
    LostTheWay {
        toward: Where,
        why: Lost,
    },
    /// Two bodies travelling crossed paths and stopped — a collision at a shared
    /// doorway or lift that ends both journeys where they met. `into` is the id
    /// of the other body. Unlike a journey's other outcomes it is **not**
    /// private: the one bumped into is standing right there and knows it
    /// happened, and a bystander sees two people run into each other. It is an
    /// outcome all the same — the walker is told, because a collision it did not
    /// plan is news — which is why it rides the `is_outcome` gate without the
    /// `is_private` one.
    Bumped {
        into: String,
    },
    /// A station lit up. The subject is carried but is only *shown* to
    /// somebody in the same room.
    TookStation {
        subject: Option<String>,
    },
    LeftStation {
        subject: Option<String>,
    },
    /// Somebody spoke. `to` is who they aimed it at, and is a fact about the
    /// utterance rather than about who received it — at an ordinary pitch
    /// everybody in the room hears it either way. `None` is spoken to the room
    /// at large. `voice` is how it was pitched, which is what decides who else
    /// makes it out — see [`Voice`].
    Said {
        to: Option<String>,
        words: String,
        voice: Voice,
    },
    /// Somebody did something without speaking — a look held, a hand raised, a
    /// chair pushed back.
    ///
    /// **The room's non-verbal channel, and it is not decoration.** Without it
    /// every act a character can take that anybody else perceives is speech, so
    /// a character with nothing to say has nothing to *do* that registers, and
    /// the acts it reaches for instead — turning to face a thing, looking at a
    /// wall — change nothing and are perceived by nobody. Two characters then
    /// share a room while being, to each other, motionless.
    ///
    /// `to` aims it the same way [`Happening::Said`] does: at somebody, in
    /// front of everybody, who all see who it was meant for.
    Did {
        to: Option<String>,
        what: String,
    },
    /// The building did something. Nobody did it.
    ///
    /// **The one happening with no actor**, and that is the whole of why it
    /// exists. Everything else here is somebody's doing, so a room where nobody
    /// is doing anything is a room where nothing is true — which is exactly the
    /// state a character standing alone in a corridor is in, and it is why one
    /// of them stood there gesturing at nothing every four seconds. A vent
    /// cycling, a light failing, a rat crossing the floor: those happen whether
    /// or not anybody is thinking, and a character perceiving one has something
    /// real to react to.
    ///
    /// `text` is a whole sentence naming its own subject — *"The lights in the
    /// ceiling flicker and steady."* — because there is no actor to put in
    /// front of it. `npcd`'s `engine::stir` produces them and holds that rule;
    /// `crate::witness::narrate` relies on it, standing the sentence on its own
    /// rather than trying to attribute it.
    ///
    /// `weight` rides on the event rather than being derived from it. Every
    /// other happening's weight follows from what *kind* of thing it is, but a
    /// fan changing note and a breaker going are the same kind of thing and are
    /// worth entirely different amounts, so the thing that knows says so.
    Stirred {
        text: String,
        weight: Weight,
    },
}

impl Happening {
    /// Whether this is the world answering something the actor set in motion,
    /// rather than something the actor did.
    ///
    /// The distinction only exists because movement takes time. A body is not
    /// told what it just did — it knows — but it must be told how a journey it
    /// began four ticks ago turned out, because by then the answer is news.
    pub fn is_outcome(&self) -> bool {
        matches!(
            self,
            Happening::GotThere { .. } | Happening::LostTheWay { .. } | Happening::Bumped { .. }
        )
    }

    /// Whether this happened inside one body and is visible to nobody else.
    ///
    /// The journey internals are — where a body meant to go, and its own sense of
    /// having got there or given up. A [`Happening::Bumped`] is an outcome but
    /// **not** private: it is a collision two bodies share, so the private set is
    /// listed rather than derived from [`Self::is_outcome`].
    pub fn is_private(&self) -> bool {
        matches!(
            self,
            Happening::SetOut { .. } | Happening::GotThere { .. } | Happening::LostTheWay { .. }
        )
    }
}

/// How an utterance was pitched — the one thing about it that decides who else
/// can make it out.
///
/// Carried on the event rather than inferred from who was listening, because
/// the world records what happened in full and [`crate::witness`] narrows it on
/// the way out: a whisper is a thing everybody in the room saw happen and one
/// person heard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Voice {
    /// An ordinary voice. Everybody in the room hears it; nobody outside does.
    Said,
    /// Raised. The room hears it, and so does every room that can see into it
    /// — the one kind of speech that carries through a doorway.
    Shouted,
    /// Lowered. Only the person it is aimed at makes out the words; everybody
    /// else in the room sees it happen and hears nothing of it.
    Whispered,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub at: Tick,
    pub actor: String,
    pub place: Where,
    pub what: Happening,
}

/// Why an action did not happen.
///
/// Typed rather than a message, because every one of these is an ordinary
/// outcome an NPC has to be able to act on — a full room is not an error, it
/// is a fact about the room — and because a test asserting on prose is a test
/// that breaks when the prose improves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    NoSuchActor(String),
    NoSuchPlace(Where),
    /// There is no route from where the actor is to where it asked to go.
    ///
    /// A refusal rather than a journey that fails later, because a body that
    /// knows a building knows before it stands up. Walking to a door to find
    /// out it leads nowhere is what a body without a memory does.
    NoWay {
        from: Where,
        to: Where,
    },
    /// This world names nowhere to teleport to.
    NoTeleport,
    /// Somebody was addressed who is not in the room. Talking *about* an
    /// absent person is undirected speech with their name in it.
    NotHere {
        who: String,
    },
    /// A body addressed itself. Muttering is [`World::say`].
    SpeakingToYourself,
    /// Nothing here is worked at.
    NothingToWorkAt,
    /// Every station in the room is taken.
    EveryStationTaken {
        of: u32,
    },
    /// Somebody else has it, anywhere in the world.
    AlreadyHeld {
        subject: String,
        by: String,
    },
    /// This actor is already at a station.
    AlreadyAtAStation,
    /// A station that claims something was taken without saying what.
    SubjectNeeded {
        binds: String,
    },
    /// A station that claims nothing was taken with a subject anyway.
    SubjectRefused,
    NotAtAStation,
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::NoSuchActor(id) => write!(f, "there is no actor `{id}`"),
            Refused::NoSuchPlace(w) => write!(f, "there is no place `{w}`"),
            Refused::NoWay { from, to } => write!(f, "no way from `{from}` to `{to}`"),
            Refused::NoTeleport => write!(f, "there is nowhere to teleport to"),
            Refused::NotHere { who } => write!(f, "`{who}` is not in this room"),
            Refused::SpeakingToYourself => write!(f, "cannot address yourself"),
            Refused::NothingToWorkAt => write!(f, "nothing here is worked at"),
            Refused::EveryStationTaken { of } => {
                write!(f, "all {of} stations here are taken")
            }
            Refused::AlreadyHeld { subject, by } => {
                write!(f, "`{subject}` is already held by {by}")
            }
            Refused::AlreadyAtAStation => write!(f, "already at a station"),
            Refused::SubjectNeeded { binds } => write!(f, "a station here takes {binds}"),
            Refused::SubjectRefused => write!(f, "a station here takes hold of nothing"),
            Refused::NotAtAStation => write!(f, "not at a station"),
        }
    }
}

impl std::error::Error for Refused {}

type Done<T = ()> = Result<T, Refused>;

/// How many events a world keeps.
///
/// # Why there is a bound at all
///
/// A world never stops happening. Every arrival, departure, utterance, gesture
/// and station change appends an event, and nothing was taking any of them away
/// — so the log grew for as long as the daemon ran, and [`crate::witness::since`]
/// walks it **from the beginning on every read**, once per body per moment. An
/// unbounded log is therefore not only unbounded memory, it is a per-tick cost
/// that climbs for the life of the process. A cast standing still still talks.
///
/// # Why this many, and why losing the rest is safe
///
/// The log exists to be *drained*, not stored. Every body's cursor advances on
/// every moment of the perception sweep — including bodies with no mind bound,
/// which are swept precisely so their cursor cannot fall behind — and a moment
/// is 500 ms. So what a reader actually needs is the handful of events since it
/// last looked, and this window is four thousand.
///
/// The cost of trimming is only paid by a reader that has not looked in
/// thousands of events, which the sweep makes impossible for anything with a
/// body in the world. What is lost is history nobody was going to be handed:
/// the record a character *keeps* is its substrate, written when it perceived
/// the event, and that is not this.
///
/// [`crate::witness::since`] already tolerates the loss — it recovers a
/// reader's position by walking arrivals and falls back to where the body is
/// standing when its own arrival has been trimmed out from under it.
pub const KEEP_EVENTS: usize = 4096;

/// How far over [`KEEP_EVENTS`] the log runs before it is cut back.
///
/// Purely an amortisation: see [`World::forget_old_events`].
const EVENT_SLACK: usize = 512;

/// The state of one world, shared by everything in it.
#[derive(Debug, Clone)]
pub struct World {
    map: MapSet,
    actors: BTreeMap<String, Actor>,
    /// What every body that has ever been here is called.
    ///
    /// **A name has to outlive the body.** The log keeps an event after the
    /// actor it names has gone, and a reader is shown that event later — so a
    /// name looked up only in `actors` falls back to the id the moment somebody
    /// leaves, and the room is told `visitor:u_154797f3 left` instead of who
    /// went. Keyed by body rather than by event, so it grows with the number of
    /// distinct people this world has ever held — a cast and its visitors —
    /// rather than with everything that has happened.
    names: BTreeMap<String, String>,
    log: Vec<Event>,
    now: Tick,
    /// The building's one lift, and the landings it serves. `shaft[i]` is the
    /// core of floor `i`, in level order (bottom to top by `ordinal`); the lift
    /// is `None` in a world with fewer than two floors to join. See
    /// [`crate::lift`].
    lift: Option<Lift>,
    shaft: Vec<Where>,
    /// Who is riding, and the floor each is bound for. They come off at their
    /// floor when the car opens there.
    riders: BTreeMap<String, usize>,
}

impl World {
    pub fn new(map: MapSet) -> World {
        let shaft = Self::build_shaft(&map);
        // The car starts at the floor the building says a body arrives on, so
        // the first Maker down finds it waiting rather than having to call it up
        // from somewhere. Falls back to the bottom of the shaft.
        let start = map
            .arrival()
            .and_then(|at| shaft.iter().position(|c| c.area == at.area))
            .unwrap_or(0);
        let lift = (shaft.len() >= 2).then(|| Lift::new(shaft.len(), start));
        World {
            map,
            actors: BTreeMap::new(),
            names: BTreeMap::new(),
            log: Vec::new(),
            now: 0,
            lift,
            shaft,
            riders: BTreeMap::new(),
        }
    }

    /// The cores of the building's levels, bottom to top by `ordinal` — the
    /// landings the lift stops at, one per floor. Everything the lift needs to
    /// know about the map is captured here once, so the car itself stays a plain
    /// state machine that knows only floor indices.
    fn build_shaft(map: &MapSet) -> Vec<Where> {
        let mut levels: Vec<&crate::schema::Area> =
            map.areas().filter(|a| a.kind == AreaKind::Level).collect();
        levels.sort_by_key(|a| a.ordinal.unwrap_or(u32::MAX));
        levels
            .into_iter()
            .filter_map(|a| {
                a.nodes
                    .iter()
                    .find(|n| n.kind == NodeKind::Core)
                    .map(|n| Where::new(a.id.clone(), n.id.clone()))
            })
            .collect()
    }

    /// The building's lift, if it has one.
    pub fn lift(&self) -> Option<&Lift> {
        self.lift.as_ref()
    }

    /// The landings the lift serves, floor by floor (each level's core).
    pub fn shaft(&self) -> &[Where] {
        &self.shaft
    }

    /// Which floor a place is on — the shaft index of its level — if the lift
    /// serves that level.
    pub fn floor_of(&self, place: &Where) -> Option<usize> {
        self.shaft.iter().position(|c| c.area == place.area)
    }

    /// The floor whose landing a body is standing on, if it is on one — where it
    /// can call or board the lift. `None` when the body is not at a lift core.
    pub fn at_landing(&self, id: &str) -> Option<usize> {
        let at = &self.actor(id)?.at;
        self.shaft.iter().position(|c| c == at)
    }

    /// The floor a body is riding to, if it has boarded and not yet been set down.
    /// A rider waits on its origin landing until the car opens at this floor (see
    /// [`World::ride_lift`]); until then it has already chosen, so it is neither
    /// offered the lift again nor told the car has left without it.
    pub fn riding(&self, id: &str) -> Option<usize> {
        self.riders.get(id).copied()
    }

    /// The lift facts a tool grammar is gated on, for the body `id`, in one pass:
    /// whether it stands on a landing (`lift_call`/`lift_use` are possible at
    /// all), whether the car is open there (ready to board), and the other levels
    /// it could ride to. A body already aboard is offered nothing — it has chosen
    /// and is waiting — so all three come back empty for it.
    ///
    /// The single source for this, so the situation the character is shown and the
    /// grammar it is masked to cannot compute it two different ways.
    pub fn lift_within(&self, id: &str) -> (bool, bool, Vec<String>) {
        if self.riding(id).is_some() {
            return (false, false, Vec::new());
        }
        let Some(here) = self.at_landing(id) else {
            return (false, false, Vec::new());
        };
        let lift_here = self.lift.as_ref().is_some_and(|l| l.boardable_at(here));
        let floors = self
            .floor_names()
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != here)
            .map(|(_, n)| n)
            .collect();
        (true, lift_here, floors)
    }

    /// The level id of floor `i`, for naming a floor in prose.
    pub fn floor_area(&self, floor: usize) -> Option<&str> {
        self.shaft.get(floor).map(|c| c.area.as_str())
    }

    /// The name of floor `i`'s level — "the chronicle level" — as a rider would
    /// say it.
    pub fn floor_name(&self, floor: usize) -> Option<&str> {
        let area = &self.shaft.get(floor)?.area;
        self.map.get(area).map(|a| a.name.as_str())
    }

    /// The floor a level name picks out, matched case-insensitively — the mirror
    /// of [`World::floor_name`], for turning a rider's chosen level back into a
    /// shaft index.
    pub fn floor_named(&self, name: &str) -> Option<usize> {
        let want = name.trim();
        (0..self.shaft.len()).find(|&i| {
            self.floor_name(i)
                .is_some_and(|n| n.eq_ignore_ascii_case(want))
        })
    }

    /// Every floor's level name, in shaft order — the set `lift_use` binds its
    /// `floor` to. The caller drops the one the rider is on.
    pub fn floor_names(&self) -> Vec<String> {
        (0..self.shaft.len())
            .filter_map(|i| self.floor_name(i).map(str::to_string))
            .collect()
    }

    /// Call the car to `floor`. A no-op in a world with no lift, or for a floor
    /// off the end of the shaft.
    pub fn call_lift(&mut self, floor: usize) {
        if let Some(lift) = self.lift.as_mut() {
            lift.call(floor);
        }
    }

    /// Board `rider` into the car and send it to `dest`; the rider comes off
    /// there when the doors open. `false` — refused — when the rider is not on a
    /// landing with the car open at it, there is no lift, or `dest` is off the
    /// shaft. The rider stays on its landing until the car reaches `dest`, so a
    /// body waiting with it is company until then.
    pub fn ride_lift(&mut self, rider: &str, dest: usize) -> bool {
        let Some(from) = self.at_landing(rider) else {
            return false;
        };
        if dest >= self.shaft.len() {
            return false;
        }
        if !self.lift.as_ref().is_some_and(|l| l.boardable_at(from)) {
            return false;
        }
        self.riders.insert(rider.to_string(), dest);
        self.lift.as_mut().expect("checked above").call(dest);
        true
    }

    /// Advance the car one moment, on its own clock — it moves whether or not
    /// anybody is walking. Each moment is voiced on the floor it happened at, so
    /// a body on a landing hears the car close, pass, or arrive; and a rider
    /// bound for a floor the car has just opened at is set down there.
    fn step_lift(&mut self) {
        let moments = match self.lift.as_mut() {
            Some(lift) => lift.step(),
            None => return,
        };
        if moments.is_empty() {
            return;
        }
        self.now += 1;
        let at = self.now;
        for m in &moments {
            let (floor, text) = match m {
                Moment::Closed(f) => (*f, "The lift doors close and it sets off."),
                Moment::Passed(f) => (*f, "The lift passes the landing without stopping."),
                Moment::Arrived(f) => (*f, "The lift arrives, and its doors open."),
            };
            if let Some(core) = self.shaft.get(floor).cloned() {
                self.log.push(Event {
                    at,
                    actor: String::new(),
                    place: core,
                    what: Happening::Stirred {
                        text: text.into(),
                        weight: Weight::Note,
                    },
                });
            }
        }
        // Set down every rider bound for a floor the car has just opened at.
        let opened: Vec<usize> = moments
            .iter()
            .filter_map(|m| match m {
                Moment::Arrived(f) => Some(*f),
                _ => None,
            })
            .collect();
        for floor in opened {
            let Some(core) = self.shaft.get(floor).cloned() else {
                continue;
            };
            let arrivals: Vec<String> = self
                .riders
                .iter()
                .filter(|(_, f)| **f == floor)
                .map(|(r, _)| r.clone())
                .collect();
            for r in arrivals {
                self.riders.remove(&r);
                self.land(&r, &core, at);
            }
        }
    }

    /// What a body is called, whether or not it is still here.
    ///
    /// The one lookup that answers for somebody who has left, which is what a
    /// reader being told about a departure needs.
    pub fn name_of(&self, id: &str) -> Option<&str> {
        self.actors
            .get(id)
            .map(|a| a.name.as_str())
            .or_else(|| self.names.get(id).map(String::as_str))
    }

    pub fn map(&self) -> &MapSet {
        &self.map
    }

    pub fn now(&self) -> Tick {
        self.now
    }

    pub fn actor(&self, id: &str) -> Option<&Actor> {
        self.actors.get(id)
    }

    pub fn actors(&self) -> impl Iterator<Item = &Actor> {
        self.actors.values()
    }

    /// Everybody standing in one place, in a stable order.
    pub fn actors_at(&self, place: &Where) -> Vec<&Actor> {
        self.actors.values().filter(|a| &a.at == place).collect()
    }

    /// Somebody `speaker` could call after: named `name`, no longer in the room
    /// but one doorway away, in a room the speaker's voice still reaches
    /// ([`Reach::InSight`] of the *walker's* scope, since it is the walker who
    /// must make out the voice).
    ///
    /// **It does not re-check that they were just in the speaker's room**, and it
    /// does not need to: that guarantee is the caller's. The grammar only ever
    /// offers an addressee drawn from the room's own company (`tell`/`ask`/
    /// `whisper` bind `to` to `Choices::Company`), so a named addressee who is no
    /// longer present is, by construction, somebody who was standing here a
    /// moment ago and has stepped out. The world's part is only to find where
    /// they went and confirm the line can still carry. Speech does not otherwise
    /// cross a room — this is the one aimed exception, and it reaches only the
    /// one it is aimed at.
    pub fn within_earshot(&self, speaker: &str, name: &str) -> Option<String> {
        let want = name.trim();
        let place = self.actor(speaker)?.at.clone();
        self.actors
            .values()
            .filter(|a| a.id != speaker && a.at != place)
            .filter(|a| a.name.eq_ignore_ascii_case(want))
            .find(|a| Scope::at(self, &a.at).reach(&place) == Reach::InSight)
            .map(|a| a.id.clone())
    }

    /// Who holds a subject, anywhere in the world.
    ///
    /// The search is over every actor rather than every actor on a level,
    /// which is what makes a claim mean the same thing on the casting floor
    /// and at an easel two levels up.
    pub fn holder_of(&self, subject: &str) -> Option<&Actor> {
        self.actors.values().find(|a| {
            a.hold
                .as_ref()
                .and_then(|h| h.subject.as_deref())
                .is_some_and(|s| s == subject)
        })
    }

    /// How many stations in a place are being worked at.
    pub fn stations_taken(&self, place: &Where) -> u32 {
        self.actors_at(place)
            .iter()
            .filter(|a| a.hold.is_some())
            .count() as u32
    }

    /// How many stations a place has, from the map.
    pub fn stations_here(&self, place: &Where) -> u32 {
        self.node(place)
            .map(|n| self.map.stations_at(n))
            .unwrap_or(0)
    }

    pub fn node(&self, place: &Where) -> Option<&Node> {
        self.map.get(&place.area)?.node(&place.node)
    }

    /// Put a body into the world.
    pub fn enter(&mut self, id: impl Into<String>, name: impl Into<String>, place: Where) -> Done {
        let id = id.into();
        if self.node(&place).is_none() {
            return Err(Refused::NoSuchPlace(place));
        }
        let name = name.into();
        self.now += 1;
        self.names.insert(id.clone(), name.clone());
        self.actors.insert(
            id.clone(),
            Actor {
                id: id.clone(),
                name,
                at: place.clone(),
                hold: None,
                walk: None,
                looked: self.now,
            },
        );
        self.log.push(Event {
            at: self.now,
            actor: id,
            place,
            what: Happening::Arrived,
        });
        Ok(())
    }

    /// Take a body out of the world.
    ///
    /// **The other half of [`World::enter`], and it was missing.** Every actor
    /// this world had ever seen stayed in it for the life of the daemon, which
    /// was true enough while the only bodies were characters that never left —
    /// and stopped being true the moment a person could walk into a room to
    /// talk to one. Somebody who closes the conversation has gone, and a world
    /// that still lists them is a world where a character is told it has
    /// company that is not there.
    ///
    /// **The hold goes first.** A body leaving while it holds a station would
    /// leave that station claimed by nobody, and nothing else releases it: the
    /// claim is keyed on an actor that no longer exists, so no act could ever
    /// give it up. That is a chronicle terminal nobody can ever sit at again.
    ///
    /// `Err(NoSuchActor)` for a body that was not here, so leaving twice is
    /// reported rather than silently fine — a caller that thinks it removed
    /// somebody twice has lost track of which session it is ending.
    pub fn leave(&mut self, id: &str) -> Done {
        self.now += 1;
        let at = self.now;
        let Some(actor) = self.actors.get(id) else {
            return Err(Refused::NoSuchActor(id.into()));
        };
        let place = actor.at.clone();
        if actor.hold.is_some() {
            self.release_at(id, at)?;
        }
        self.actors.remove(id);
        // Logged as a departure, so everybody standing there perceives it the
        // way they perceive an arrival. Somebody vanishing from a room without
        // the room being told is the same defect as somebody appearing in it
        // unannounced.
        self.log.push(Event {
            at,
            actor: id.into(),
            place,
            what: Happening::Left,
        });
        Ok(())
    }

    /// Reshape the world's map while it runs — add or drown a room, open a gate
    /// that was not there (effector design Appendix F).
    ///
    /// **Apply-to-a-copy, validate, then swap.** The edit is handed to
    /// [`MapSet::apply`], which rebuilds and re-validates the whole set; a
    /// mutation that would break the map returns `Err` here and **nothing
    /// changes** — the running world keeps the map it had. Only a set that
    /// passes the same gate a fresh load does is swapped in.
    ///
    /// **No body is left nowhere.** A body standing where a node used to be is
    /// relocated to the area's way in ([`MapSet::arrival_in`], then the world's
    /// arrival), releasing whatever station it held first — the hold's node is
    /// gone, so nothing else ever could. The relocation is logged as a departure
    /// from the vanished place and an arrival at the new one, so both rooms
    /// perceive it the ordinary way. If a stranded body has nowhere at all to go
    /// — an edit that leaves the world with no arrival — the reshape is refused
    /// rather than stranding it.
    ///
    /// Returns the ids relocated, in a stable order (empty when the reshape
    /// displaced nobody). `Err(reason)` carries the world's own words on why the
    /// edit could not be made — this is an authoring act reached from the map
    /// stations, not a body act, so it answers with a reason string rather than a
    /// [`Refused`].
    pub fn reshape(&mut self, edit: &MapEdit) -> Result<Vec<String>, String> {
        // The candidate set. A mutation that will not validate refuses here,
        // before anything in the live world is touched.
        let new_map = self.map.apply(edit).map_err(|e| e.to_string())?;

        // Bodies standing where a node no longer exists, and where each will go.
        // Resolved against the *new* map, and refused outright if any has nowhere
        // to land, so the swap below cannot strand anyone.
        let mut moves: Vec<(String, Where, Where)> = Vec::new();
        for actor in self.actors.values() {
            if new_map.node_at(&actor.at).is_some() {
                continue;
            }
            let to = new_map
                .arrival_in(&actor.at.area)
                .or_else(|| new_map.arrival())
                .ok_or_else(|| {
                    format!(
                        "that would leave {} standing nowhere, and there is no way into the world \
                         to move them to",
                        self.name_of(&actor.id).unwrap_or(&actor.id)
                    )
                })?;
            moves.push((actor.id.clone(), actor.at.clone(), to));
        }

        // Commit: the map is swapped, then the lift shaft re-derived from it.
        self.map = new_map;
        self.reshaft();

        // Relocate the stranded, releasing a now-impossible hold first, and log
        // the move both ways so the vanished room and the new one each see it.
        let mut relocated = Vec::new();
        for (id, from, to) in moves {
            self.now += 1;
            let at = self.now;
            if self.actors.get(&id).and_then(|a| a.hold.as_ref()).is_some() {
                self.release_at(&id, at).map_err(|e| e.to_string())?;
            }
            let Some(actor) = self.actors.get_mut(&id) else {
                continue;
            };
            actor.at = to.clone();
            actor.walk = None;
            self.riders.remove(&id);
            self.log.push(Event {
                at,
                actor: id.clone(),
                place: from,
                what: Happening::Left,
            });
            self.log.push(Event {
                at,
                actor: id.clone(),
                place: to,
                what: Happening::Arrived,
            });
            relocated.push(id);
        }
        Ok(relocated)
    }

    /// Re-derive the lift shaft after the map has changed.
    ///
    /// The shaft is a pure function of the map's levels and their cores
    /// ([`World::build_shaft`]), so it is simply rebuilt. The car is only
    /// disturbed when the shaft actually changed — a mutation that adds a room to
    /// a level leaves the floors exactly as they were, and the lift with them.
    /// When the floors do change, the car is set back to the building's arrival
    /// floor (as a fresh world starts it) and anyone mid-ride is set down, since
    /// a floor index no longer names the same landing.
    fn reshaft(&mut self) {
        let shaft = Self::build_shaft(&self.map);
        if shaft == self.shaft {
            return;
        }
        let start = self
            .map
            .arrival()
            .and_then(|at| shaft.iter().position(|c| c.area == at.area))
            .unwrap_or(0);
        self.lift = (shaft.len() >= 2).then(|| Lift::new(shaft.len(), start));
        self.shaft = shaft;
        self.riders.clear();
    }

    /// Set off for somewhere, and find out later whether you got there.
    ///
    /// Returns how many **stops** the journey takes — how many world ticks it
    /// will cost, not how many doorways it passes. Nothing else about it is
    /// known yet: the body stands up, leaves whatever it was holding, and
    /// covers one leg per [`World::tick`].
    ///
    /// **Why this is not one call that ends where it started going.** A vault
    /// where crossing the building is free is a vault where the building does
    /// not exist: nobody is *away* when somebody looks for them, nobody is
    /// caught at the lift, and the green room being near the casting bands
    /// means nothing. Distance only matters if it is paid, and paying it is
    /// what makes the journey's outcome something the world has to report
    /// rather than something the caller already knows.
    pub fn set_off(&mut self, id: &str, to: Where) -> Done<usize> {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        let from = actor.at.clone();
        if self.node(&to).is_none() {
            return Err(Refused::NoSuchPlace(to));
        }
        let Some(mut path) = route::route(&self.map, &from, &to) else {
            return Err(Refused::NoWay { from, to });
        };
        // The route includes where you are standing, which is not a move.
        path.remove(0);
        if path.is_empty() {
            return Ok(0);
        }

        self.end_walk(id, &from, Lost::Diverted);
        // Walking away cancels a lift you were waiting on: you are no longer on
        // the landing to be carried off, so the car must not set you down later.
        self.riders.remove(id);
        if self.actors[id].hold.is_some() {
            self.release(id)?;
        }

        self.now += 1;
        let walk = Walk {
            toward: to.clone(),
            ahead: path.into(),
            set_out: self.now,
        };
        let stops = walk.to_go(&from);
        self.actors.get_mut(id).expect("checked above").walk = Some(walk);
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place: from,
            what: Happening::SetOut { toward: to },
        });
        Ok(stops)
    }

    /// Advance the world one moment, moving everyone who is on their way.
    ///
    /// Returns how many bodies moved. Everything in one tick shares a [`Tick`],
    /// because a tick *is* one moment: two Makers who set off together stay
    /// level with each other, and a watcher sees them arrive at the same
    /// instant rather than in whatever order the actor map happens to be in.
    ///
    /// **One tick is one leg, not one doorway.** There is no physics here and
    /// no metres — a body covers as much ground as it can without changing
    /// level, so a room on your own floor is one tick away and the far corner
    /// of the building is three: out to the lift, up, and along. Those are the
    /// points where somebody would actually stop and reconsider, which is why
    /// they are the points a mind gets a turn at.
    pub fn tick(&mut self) -> usize {
        // **Before the early return below, not after it.** Most moments move
        // nobody, so a trim placed after the `legs.is_empty()` exit would run
        // only while somebody happened to be walking — which is exactly not the
        // condition that grows the log.
        self.forget_old_events();
        // The lift runs on its own clock, whether or not anybody is walking.
        self.step_lift();
        let legs: Vec<(String, Where)> = self
            .actors
            .values()
            .filter_map(|a| Some((a.id.clone(), a.walk.as_ref()?.leg_end(&a.at)?)))
            .collect();
        if legs.is_empty() {
            return 0;
        }
        self.now += 1;
        let at = self.now;

        // Where each mover starts and where its leg ends, captured (owned, so
        // the land calls below may borrow the world mutably) before anyone moves
        // — so a *swap*, two bodies exchanging rooms across the same edge, can be
        // told apart from an ordinary arrival.
        let from: BTreeMap<String, Where> = legs
            .iter()
            .map(|(id, _)| (id.clone(), self.actors[id].at.clone()))
            .collect();
        let to: BTreeMap<String, Where> =
            legs.iter().map(|(id, d)| (id.clone(), d.clone())).collect();

        // **Swaps first.** Two travellers crossing the same lift or stair in
        // opposite directions meet on it: the one heading into the other's room
        // stops there, and the other never leaves — they end up together in the
        // doorway. Resolved before the ordinary land so no departure is logged
        // that then has to be taken back. This is the crossing a chase between
        // two levels is made of, where an arrival never catches anybody because
        // the one it is chasing has left by the time it lands.
        let ids: Vec<String> = legs.iter().map(|(id, _)| id.clone()).collect();
        let mut caught: BTreeSet<String> = BTreeSet::new();
        let mut bumps: Vec<(String, String, Where)> = Vec::new();
        for a in &ids {
            if caught.contains(a) {
                continue;
            }
            let (af, at2) = (from[a].clone(), to[a].clone());
            let partner = ids.iter().find(|b| {
                b.as_str() != a
                    && !caught.contains(*b)
                    && from.get(*b) == Some(&at2)
                    && to.get(*b) == Some(&af)
            });
            if let Some(b) = partner.cloned() {
                // `a` walks into `b`'s room (`at2` is where `b` still stands);
                // `b` holds there. Both stop, together, where they met.
                self.land(a, &at2, at);
                caught.insert(a.clone());
                caught.insert(b.clone());
                bumps.push((a.clone(), b, at2));
            }
        }

        // Everyone not caught in a swap takes their leg.
        for (id, dest) in &legs {
            if !caught.contains(id) {
                self.land(id, dest, at);
            }
        }

        // **Co-locations.** A traveller still on its way that has landed where
        // somebody already is stops there too — walked into them. One that has
        // *arrived* (its walk cleared on landing) went there on purpose and is
        // joining them, not colliding.
        for (id, _) in &legs {
            if caught.contains(id) {
                continue;
            }
            let Some(a) = self.actors.get(id) else {
                continue;
            };
            if a.walk.is_none() {
                continue;
            }
            let place = a.at.clone();
            if let Some(other) = self.actors.values().find(|o| &o.id != id && o.at == place) {
                let oid = other.id.clone();
                caught.insert(id.clone());
                if other.walk.is_some() {
                    caught.insert(oid.clone());
                }
                bumps.push((id.clone(), oid, place));
            }
        }

        // A collision ends the journey where it happened — the walk is cleared
        // directly rather than through `end_walk`, because the `Bumped` event is
        // the news the walker reads, not a `LostTheWay` on top of it. And each
        // pair is recorded once: it is perceived from both sides already (the
        // actor reads "you bump into …", the other "… bumps into you").
        for id in &caught {
            if let Some(a) = self.actors.get_mut(id) {
                a.walk = None;
            }
        }
        let mut seen: BTreeSet<(String, String)> = BTreeSet::new();
        for (actor, into, place) in bumps {
            let key = if actor < into {
                (actor.clone(), into.clone())
            } else {
                (into.clone(), actor.clone())
            };
            if seen.insert(key) {
                self.log.push(Event {
                    at,
                    actor,
                    place,
                    what: Happening::Bumped { into },
                });
            }
        }
        legs.len()
    }

    /// Tick until nobody is on their way, and say how many it took.
    ///
    /// For a caller that wants a journey settled — a test, a scene being set
    /// up, a crew dispatched before the day starts. An NPC does not call this;
    /// it lives a tick at a time and reads its outcomes.
    pub fn settle(&mut self) -> usize {
        let mut ticks = 0;
        while self.tick() > 0 {
            ticks += 1;
        }
        ticks
    }

    /// Put a body somewhere, now — **the one way a body ever moves.**
    ///
    /// [`World::tick`] is a caller of this and so is a game that owns its own
    /// movement: where this world advances a leg per tick, Battle Cities walks
    /// a tile grid at its own pace and says where the body ended up. Both
    /// produce the same events, because both come through here. Nothing about
    /// how long a journey takes lives in this crate.
    ///
    /// A body landing on its own route continues the journey; landing on its
    /// destination finishes it; landing anywhere else abandons it — and says so.
    pub fn place(&mut self, id: &str, to: Where) -> Done {
        if !self.actors.contains_key(id) {
            return Err(Refused::NoSuchActor(id.into()));
        }
        if self.node(&to).is_none() {
            return Err(Refused::NoSuchPlace(to));
        }
        if self.actors[id].at == to {
            return Ok(());
        }
        self.now += 1;
        let at = self.now;
        self.land(id, &to, at);
        Ok(())
    }

    /// A body arriving somewhere, at a moment the caller has already opened.
    ///
    /// Split out so that everyone moving in one tick shares its tick — a leg
    /// that bumped the clock per body would let a watcher see four Makers cross
    /// a lift lobby in single file when they crossed it together.
    fn land(&mut self, id: &str, to: &Where, at: Tick) {
        let Some(actor) = self.actors.get(id) else {
            return;
        };
        let from = actor.at.clone();
        // Any relocation clears a pending ride: the lift's own set-down has
        // already taken the rider off the roster before landing it here, so this
        // only catches a body moved by some other means (placed, teleported)
        // while it was waiting, which must not then be carried off a second time.
        self.riders.remove(id);
        // Leaving is releasing, wherever the move came from.
        if actor.hold.is_some() {
            let _ = self.release_at(id, at);
        }

        self.actors.get_mut(id).expect("checked above").at = to.clone();
        self.log.push(Event {
            at,
            actor: id.into(),
            place: from,
            what: Happening::Left,
        });
        self.log.push(Event {
            at,
            actor: id.into(),
            place: to.clone(),
            what: Happening::Arrived,
        });

        // A bystander sees a body leave and a body come in, and that is all.
        // Where it meant to go, and whether it has got there, are the walker's
        // own — see `Happening::is_private`.
        let Some(walk) = self.actors[id].walk.as_ref() else {
            return;
        };
        if &walk.toward == to {
            let toward = walk.toward.clone();
            self.actors.get_mut(id).expect("checked above").walk = None;
            self.log.push(Event {
                at,
                actor: id.into(),
                place: to.clone(),
                what: Happening::GotThere { toward },
            });
        } else if let Some(covered) = walk.ahead.iter().position(|w| w == to) {
            self.actors
                .get_mut(id)
                .expect("checked above")
                .walk
                .as_mut()
                .expect("checked above")
                .ahead
                .drain(..=covered);
        } else {
            // Carried somewhere off the route by something outside its own
            // intent. The journey is over, and the body has to be told.
            self.end_walk_at(id, to, Lost::Diverted, at);
        }
    }

    /// Teleport to the one place this world says a body may jump to.
    ///
    /// The only journey in the vault that costs nothing, and it is *one*
    /// journey: [`Area::teleport_to`](crate::schema::Area::teleport_to) names a
    /// single place, so every trip out is still walked and the building keeps
    /// its distances. A Maker upstairs with something to report does not spend
    /// three ticks reaching the room that wants to hear it, which is the whole
    /// of what this buys.
    pub fn teleport(&mut self, id: &str) -> Done {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        let from = actor.at.clone();
        let to = self
            .map
            .teleport_to(&from.area)
            .ok_or(Refused::NoTeleport)?;
        if from == to {
            return Ok(());
        }
        self.now += 1;
        let at = self.now;
        // Ends the journey before the move, so a body teleporting to where it
        // was already walking is answered as having arrived rather than as
        // having been carried off its route.
        self.end_walk_at(id, &to, Lost::Teleported, at);
        self.land(id, &to, at);
        Ok(())
    }

    /// End whatever journey a body is on, and answer for it.
    ///
    /// `ending_at` is where the body is left standing. If that is where it was
    /// going, the journey *succeeded* however it got there — a Maker recalled
    /// to the room it was walking to arrived, and telling it that it never got
    /// there while it stands in the room would be a plain falsehood.
    ///
    /// Nothing at all when the body was not going anywhere. Everything that
    /// stops a walk comes through here, so there is one place that knows a
    /// journey has to be answered rather than merely dropped.
    fn end_walk(&mut self, id: &str, ending_at: &Where, why: Lost) {
        if self.actors.get(id).is_some_and(|a| a.walk.is_some()) {
            self.now += 1;
            let at = self.now;
            self.end_walk_at(id, ending_at, why, at);
        }
    }

    /// [`World::end_walk`] at a moment the caller has already opened.
    fn end_walk_at(&mut self, id: &str, ending_at: &Where, why: Lost, at: Tick) {
        let Some(actor) = self.actors.get_mut(id) else {
            return;
        };
        let Some(walk) = actor.walk.take() else {
            return;
        };
        let place = actor.at.clone();
        let arrived = &walk.toward == ending_at;
        self.log.push(Event {
            at,
            actor: id.into(),
            place,
            what: if arrived {
                Happening::GotThere {
                    toward: walk.toward,
                }
            } else {
                Happening::LostTheWay {
                    toward: walk.toward,
                    why,
                }
            },
        });
    }

    /// The shortest way from where a body stands to somewhere else, ends
    /// included. What it would walk if it set off now.
    pub fn route(&self, id: &str, to: &Where) -> Option<Vec<Where>> {
        route::route(&self.map, &self.actor(id)?.at, to)
    }

    /// Sit at a station here, claiming `subject` if the station takes one.
    ///
    /// One step: the room having a free station, the subject being unheld
    /// anywhere in the world, and the actor sitting down all happen together
    /// or none of them do.
    pub fn take(&mut self, id: &str, subject: Option<&str>) -> Done {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        if actor.hold.is_some() {
            return Err(Refused::AlreadyAtAStation);
        }
        let place = actor.at.clone();
        let node = self
            .node(&place)
            .ok_or(Refused::NoSuchPlace(place.clone()))?;

        let Some((part, _)) = self.map.parts_of(node, PartKind::Station).next() else {
            return Err(Refused::NothingToWorkAt);
        };
        let part_id = part.id.clone();
        let binds = part.binds.clone();

        match (&binds, subject) {
            (Some(binds), None) => {
                return Err(Refused::SubjectNeeded {
                    binds: binds.clone(),
                })
            }
            (None, Some(_)) => return Err(Refused::SubjectRefused),
            _ => {}
        }

        // Who has the thing you came for, before whether there is anywhere to
        // sit. In a room of one station both are true at once, and the more
        // useful answer is the one that names somebody to go and ask — and it
        // stays the more useful answer as rooms get bigger, where a full room
        // and a taken subject stop coinciding.
        if let Some(subject) = subject {
            if let Some(other) = self.holder_of(subject) {
                return Err(Refused::AlreadyHeld {
                    subject: subject.into(),
                    by: other.name.clone(),
                });
            }
        }

        let capacity = self.map.stations_at(node);
        if self.stations_taken(&place) >= capacity {
            return Err(Refused::EveryStationTaken { of: capacity });
        }

        // A body that stops at a station on its way somewhere has stopped, and
        // the walk it will now never finish has to be answered rather than
        // dropped.
        self.end_walk(id, &place, Lost::SatDown);

        self.now += 1;
        let hold = Hold {
            subject: subject.map(String::from),
            part: part_id,
            since: self.now,
        };
        self.actors.get_mut(id).expect("checked above").hold = Some(hold.clone());
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::TookStation {
                subject: hold.subject,
            },
        });
        Ok(())
    }

    /// Get up, and let go of whatever was claimed.
    pub fn release(&mut self, id: &str) -> Done {
        self.now += 1;
        let at = self.now;
        self.release_at(id, at)
    }

    /// [`World::release`] at a moment the caller has already opened.
    fn release_at(&mut self, id: &str, at: Tick) -> Done {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        let Some(hold) = actor.hold.clone() else {
            return Err(Refused::NotAtAStation);
        };
        let place = actor.at.clone();
        self.actors.get_mut(id).expect("checked above").hold = None;
        self.log.push(Event {
            at,
            actor: id.into(),
            place,
            what: Happening::LeftStation {
                subject: hold.subject,
            },
        });
        Ok(())
    }

    /// Say something to the room. Nobody outside it hears.
    pub fn say(&mut self, id: &str, words: impl Into<String>) -> Done {
        self.utter(id, None, words, Voice::Said)
    }

    /// Call out, to anybody within earshot: the room, and every room that can
    /// see into it. Aimed at nobody — a shout is for whoever hears it.
    pub fn shout(&mut self, id: &str, words: impl Into<String>) -> Done {
        self.utter(id, None, words, Voice::Shouted)
    }

    /// Say something to one person too quietly for anybody else to make out.
    ///
    /// The rest of the room still sees it happen — two heads together is a
    /// thing people notice — and hears none of it. The listener must be here.
    pub fn whisper(&mut self, id: &str, to: &str, words: impl Into<String>) -> Done {
        self.utter(id, Some(to), words, Voice::Whispered)
    }

    /// Say something to somebody in particular.
    ///
    /// It still lands in the room. **Direction is part of the utterance, not a
    /// filter on who receives it** — you aim your voice at a person and
    /// everybody standing there hears you do it, and knows it was not for them.
    /// That asymmetry is the whole social value of a shared room: overhearing
    /// *the creator told Maker-01 to get out* is a different fact from being
    /// told it, and both are true of the same event.
    ///
    /// The listener must be in the room. Talking *about* somebody who is not
    /// there is [`World::say`] with their name in the words.
    pub fn tell(&mut self, id: &str, to: &str, words: impl Into<String>) -> Done {
        self.utter(id, Some(to), words, Voice::Said)
    }

    /// Do something in the room without speaking. Everybody standing there sees
    /// it; nobody outside does — the same reach as [`World::say`], because it is
    /// the same room.
    ///
    /// # `what` is a predicate, verb and all
    ///
    /// It is rendered as `"{who} {what}"` — `witness::verb_phrase` supplies a
    /// verb for every *other* happening (`said …`, `came in`, `took …`) and
    /// takes this one as given, because only the caller knows whether the deed
    /// was a gesture, a hand laid on somebody, or a body going still.
    ///
    /// So `"gestures towards the table"`, not `"towards the table"`. A fragment
    /// reaches the other characters as a sentence with the verb missing —
    /// measured live, nine gestures arrived as *"Yaelis Vayne towards the table,
    /// indicating the standing orders"*.
    ///
    /// Do **not** name the target in `what`: pass it as `to` and the aiming is
    /// added once, as `", at X"`. Both is how one act came to read
    /// `"to Wren: steadying her, at Wren"`.
    pub fn show(&mut self, id: &str, to: Option<&str>, what: impl Into<String>) -> Done {
        let (place, to) = self.aim(id, to)?;
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Did {
                to,
                what: what.into(),
            },
        });
        Ok(())
    }

    /// Raise your voice after somebody who has just left the room.
    ///
    /// The one aimed line that carries through a doorway. [`World::tell`] and its
    /// kin refuse an addressee who is not here, because two people talking need
    /// to be in the same room — but somebody mid-sentence when the other steps
    /// out has one parting line to call after them, and losing it turned a moving
    /// cast's exchanges into questions nobody ever answered.
    ///
    /// Delivered under the same rule the world enforces itself rather than trusts
    /// from the caller: `to` must be one doorway away ([`Reach::InSight`]),
    /// exactly what [`World::within_earshot`] resolves. It logs at the speaker's
    /// place, so [`crate::witness`] carries it to the one it is aimed at and to
    /// nobody else.
    pub fn call_after(
        &mut self,
        id: &str,
        to: &str,
        words: impl Into<String>,
        voice: Voice,
    ) -> Done {
        let place = self
            .actor(id)
            .ok_or_else(|| Refused::NoSuchActor(id.into()))?
            .at
            .clone();
        if id == to {
            return Err(Refused::SpeakingToYourself);
        }
        let in_sight = self.actor(to).is_some_and(|a| {
            a.at != place && Scope::at(self, &a.at).reach(&place) == Reach::InSight
        });
        if !in_sight {
            return Err(Refused::NotHere { who: to.into() });
        }
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Said {
                to: Some(to.into()),
                words: words.into(),
                voice,
            },
        });
        Ok(())
    }

    /// The building does something in a room. Nobody did it.
    ///
    /// Takes a place rather than an actor, which is the only entry point here
    /// that does — see [`Happening::Stirred`]. A room that is not in the map is
    /// refused rather than logged, because an event nobody can ever be standing
    /// in is one that will sit in the window until it is trimmed.
    pub fn stir(&mut self, place: &Where, text: impl Into<String>, weight: Weight) -> Done {
        if self.node(place).is_none() {
            return Err(Refused::NoSuchPlace(place.clone()));
        }
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            // No actor. Every reader compares this against its own id to decide
            // whether an event is its own doing, and no body is called this, so
            // a stirring is nobody's doing to everybody.
            actor: String::new(),
            place: place.clone(),
            what: Happening::Stirred {
                text: text.into(),
                weight,
            },
        });
        Ok(())
    }

    /// Resolve who an utterance or gesture is aimed at, and where it happens.
    ///
    /// Shared by speech and gesture because the rule is about the *room*, not
    /// about the channel: you can only aim at somebody standing with you, and
    /// aiming at yourself is not a thing you can do.
    fn aim(&self, id: &str, to: Option<&str>) -> Result<(Where, Option<String>), Refused> {
        let actor = self.actors.get(id).ok_or(Refused::NoSuchActor(id.into()))?;
        let place = actor.at.clone();
        if let Some(to) = to {
            if to == id {
                return Err(Refused::SpeakingToYourself);
            }
            if !self.actors.get(to).is_some_and(|a| a.at == place) {
                return Err(Refused::NotHere { who: to.into() });
            }
        }
        Ok((place, to.map(String::from)))
    }

    fn utter(
        &mut self,
        id: &str,
        to: Option<&str>,
        words: impl Into<String>,
        voice: Voice,
    ) -> Done {
        let (place, to) = self.aim(id, to)?;
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Said {
                to,
                words: words.into(),
                voice,
            },
        });
        Ok(())
    }

    /// Mark everything up to now as read by this body.
    ///
    /// The cursor is state and so it lives here, but *reading* the stream is
    /// [`crate::witness`]'s: what a body could make out of an event is a
    /// perception rule, not a fact about the event, and the world records
    /// events in full whoever ends up seeing them.
    pub fn mark_seen(&mut self, id: &str) {
        let now = self.now;
        if let Some(actor) = self.actors.get_mut(id) {
            actor.looked = now;
        }
    }

    /// The whole log, in order. Read by [`crate::witness`] on behalf of one
    /// body, and by anything else that wants the record — a dispatch board, a
    /// replay, an audit.
    ///
    /// **A window, not an archive** — see [`KEEP_EVENTS`].
    pub fn log(&self) -> &[Event] {
        &self.log
    }

    /// Drop the oldest events past [`KEEP_EVENTS`].
    ///
    /// Trimmed in batches rather than one per push: draining a single event
    /// from the front of a full log is a memmove of the whole window, and the
    /// log takes an event every time anybody does anything. Letting it run
    /// [`EVENT_SLACK`] over and then cutting back to the bound makes that one
    /// memmove per `EVENT_SLACK` events instead of one per event.
    fn forget_old_events(&mut self) {
        if self.log.len() <= KEEP_EVENTS + EVENT_SLACK {
            return;
        }
        let drop = self.log.len() - KEEP_EVENTS;
        self.log.drain(..drop);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    // ── the event window ────────────────────────────────────────────────────

    /// Fill the log by talking. Speech is the cheapest event to make a lot of
    /// and the one a busy room actually produces most of.
    fn chatter(w: &mut World, n: usize) {
        for i in 0..n {
            w.say("m1", format!("line {i}")).unwrap();
        }
    }

    fn one_speaker() -> World {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        w
    }

    /// **The log is a window, not an archive.** Nothing was taking events away,
    /// so it grew for the life of the process — and `witness::since` walks it
    /// from the beginning once per body per moment, so it was a per-tick cost
    /// that climbed for ever as well as unbounded memory.
    #[test]
    fn the_event_log_stops_growing() {
        let mut w = one_speaker();
        for _ in 0..12 {
            chatter(&mut w, KEEP_EVENTS);
            w.tick();
            assert!(
                w.log().len() <= KEEP_EVENTS + EVENT_SLACK,
                "log reached {} against a bound of {}",
                w.log().len(),
                KEEP_EVENTS
            );
        }
    }

    /// And it is the **oldest** that go. A window that dropped the newest would
    /// be worse than no window: the whole point is what just happened.
    #[test]
    fn the_window_keeps_the_newest_events() {
        let mut w = one_speaker();
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 200);
        w.tick();

        let said: Vec<&str> = w
            .log()
            .iter()
            .filter_map(|e| match &e.what {
                Happening::Said { words, .. } => Some(words.as_str()),
                _ => None,
            })
            .collect();
        let last = format!("line {}", KEEP_EVENTS + EVENT_SLACK + 199);
        assert_eq!(said.last().copied(), Some(last.as_str()));
        assert!(!said.contains(&"line 0"), "the oldest line survived a trim");
    }

    /// **Trimming runs on a still world.** Most moments move nobody, so `tick`
    /// returns early — and a trim placed after that exit would only run while
    /// somebody happened to be walking, which is not the condition that fills
    /// the log.
    #[test]
    fn a_world_where_nobody_is_walking_still_trims() {
        let mut w = one_speaker();
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 50);
        assert_eq!(w.tick(), 0, "somebody was walking");
        assert_eq!(w.log().len(), KEEP_EVENTS);
    }

    /// Cut back to the bound rather than to the moment it crossed it, so the
    /// memmove is paid once per `EVENT_SLACK` events instead of once per event.
    #[test]
    fn trimming_is_amortised_over_the_slack() {
        let mut w = one_speaker();
        // Filled by measurement rather than by arithmetic: entering the world
        // is itself an event, so counting only the lines said is off by one and
        // the test would be asserting against the wrong side of the threshold.
        while w.log().len() < KEEP_EVENTS + EVENT_SLACK {
            w.say("m1", "filling").unwrap();
        }
        w.tick();
        assert_eq!(
            w.log().len(),
            KEEP_EVENTS + EVENT_SLACK,
            "cut at the threshold instead of past it — that is a memmove per event"
        );

        w.say("m1", "the one over").unwrap();
        w.tick();
        assert_eq!(w.log().len(), KEEP_EVENTS, "did not cut back to the bound");
    }

    /// **A reader still gets its news across a trim.** The cursor is a tick
    /// rather than an index into the log, so dropping the front cannot shift
    /// what a body has already read out from under it — the failure an
    /// index-based cursor would have had, silently, as a reader served somebody
    /// else's events.
    #[test]
    fn a_reader_is_not_disturbed_by_a_trim() {
        let mut w = one_speaker();
        w.enter("m2", "Maker-02", at("green-room")).unwrap();
        w.mark_seen("m2");
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 10);
        w.tick();

        w.say("m1", "the one that matters").unwrap();
        let heard = crate::witness::since(&w, "m2");
        assert!(
            heard.iter().any(|x| matches!(&x.what,
                Happening::Said { words, .. } if words == "the one that matters")),
            "a trim lost the reader its news"
        );
    }

    #[test]
    fn a_body_enters_where_it_is_told_to() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        assert_eq!(w.actor("m1").unwrap().at, at("core"));
    }

    // ── leaving ─────────────────────────────────────────────────────────────

    #[test]
    fn a_body_that_leaves_is_gone_from_the_room_and_the_world() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.enter("op", "Johnathan", at("core")).unwrap();
        assert_eq!(w.actors_at(&at("core")).len(), 2);

        w.leave("op").unwrap();
        assert!(w.actor("op").is_none());
        assert_eq!(w.actors_at(&at("core")).len(), 1);
    }

    /// **The hold goes with them.** A body leaving while it holds a station
    /// would leave that station claimed by an actor that no longer exists — and
    /// nothing else can release it, because every act that would is keyed on
    /// the actor. That is a terminal nobody can ever sit at again.
    #[test]
    fn leaving_gives_up_whatever_was_being_held() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.take("m1", Some("cindy")).unwrap();
        assert!(w.holder_of("cindy").is_some());

        w.leave("m1").unwrap();
        assert!(
            w.holder_of("cindy").is_none(),
            "a station stayed claimed by somebody who had gone"
        );
    }

    /// Leaving twice is reported rather than silently fine: a caller that
    /// thinks it removed somebody twice has lost track of which session it is
    /// ending.
    #[test]
    fn leaving_a_body_that_is_not_here_is_refused() {
        let mut w = vault();
        assert!(matches!(w.leave("nobody"), Err(Refused::NoSuchActor(_))));
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.leave("m1").unwrap();
        assert!(matches!(w.leave("m1"), Err(Refused::NoSuchActor(_))));
    }

    /// The room is told, the way it is told about an arrival — somebody
    /// vanishing unannounced is the same defect as somebody appearing that way.
    #[test]
    fn the_room_perceives_a_departure() {
        let mut w = vault();
        w.enter("watcher", "Maker-01", at("core")).unwrap();
        w.enter("goer", "Johnathan", at("core")).unwrap();
        // Spend what is already waiting, so what is left is only the leaving.
        let mut eyes = crate::delta::Attention::new();
        eyes.take(&mut w, "watcher");
        w.leave("goer").unwrap();

        let seen = eyes.take(&mut w, "watcher");
        let left = seen
            .events
            .iter()
            .find(|e| e.what == Happening::Left)
            .unwrap_or_else(|| panic!("the room was not told: {seen:?}"));
        // **Named, not identified.** The body is gone by the time anybody reads
        // this, so a name looked up in the actor list falls back to the id and
        // the room is told `visitor:u_8812 left` instead of who went.
        assert_eq!(left.name, "Johnathan");
    }

    #[test]
    fn entering_nowhere_is_refused() {
        let mut w = vault();
        let err = w.enter("m1", "Maker-01", at("the-moon")).unwrap_err();
        assert!(matches!(err, Refused::NoSuchPlace(_)));
    }

    #[test]
    fn walking_finds_its_own_way_round_the_ring() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.set_off("m1", at("green-room")).unwrap();
        w.settle();
        assert_eq!(w.actor("m1").unwrap().at, at("green-room"));
    }

    #[test]
    fn a_route_exists_between_any_two_rooms_on_a_level() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        let route = w
            .route("m1", &at("relations"))
            .expect("the ring joins everything");
        assert_eq!(route.first().unwrap(), &at("band-one"));
        assert_eq!(route.last().unwrap(), &at("relations"));
    }

    /// A plain room node for a reshape test — somewhere to stand that opens off
    /// one existing room.
    fn room(id: &str, off: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind: NodeKind::Social,
            name: id.into(),
            plural: false,
            stand: None,
            off: off.iter().map(|s| s.to_string()).collect(),
            character: None,
            parts: vec![],
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }
    }

    /// **A reshape adds a room, and it is walkable at once.** The new node opens
    /// off the green room; after the reshape the green room opens back onto it —
    /// the door woven both ways, exactly as a loaded map's is — and a body can
    /// route to it.
    #[test]
    fn a_reshape_adds_a_walkable_room() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        let moved = w
            .reshape(&MapEdit::AddNode {
                area: "vault-casting".into(),
                node: Box::new(room("annex", &["green-room"])),
            })
            .expect("a room off the green room is a valid reshape");
        assert!(moved.is_empty(), "adding a room displaces nobody");
        let green = w
            .map()
            .get("vault-casting")
            .unwrap()
            .node("green-room")
            .unwrap();
        assert!(
            green.exits.contains(&"annex".to_string()),
            "the door was not woven back: {:?}",
            green.exits
        );
        // And it is genuinely reachable from where the body stands.
        let route = w.route("m1", &at("annex")).expect("the annex is walkable");
        assert_eq!(route.last().unwrap(), &at("annex"));
    }

    /// **A body standing where a room is drowned is relocated, not stranded.**
    /// It is moved to the area's way in, and the move is logged both ways so the
    /// vanished room and the new one each perceive it.
    #[test]
    fn a_reshape_relocates_a_body_off_a_drowned_room() {
        let mut w = vault();
        w.reshape(&MapEdit::AddNode {
            area: "vault-casting".into(),
            node: Box::new(room("annex", &["green-room"])),
        })
        .unwrap();
        w.enter("m1", "Maker-01", at("annex")).unwrap();
        assert_eq!(w.actor("m1").unwrap().at, at("annex"));

        let moved = w
            .reshape(&MapEdit::RemoveNode(at("annex")))
            .expect("drowning an empty leaf room is valid");
        assert_eq!(moved, vec!["m1".to_string()], "the body was not relocated");
        // vault-casting's way in is its core, where the stranded body lands.
        assert_eq!(w.actor("m1").unwrap().at, at("core"));
        assert!(w
            .map()
            .get("vault-casting")
            .unwrap()
            .node("annex")
            .is_none());
    }

    /// **An impossible reshape refuses and changes nothing.** A room opening off
    /// a door to nowhere cannot validate, so the running world keeps the map it
    /// had — the annex never appears.
    #[test]
    fn an_impossible_reshape_leaves_the_world_untouched() {
        let mut w = vault();
        let before = w.map().get("vault-casting").unwrap().nodes.len();
        let err = w
            .reshape(&MapEdit::AddNode {
                area: "vault-casting".into(),
                node: Box::new(room("annex", &["nowhere"])),
            })
            .expect_err("a door to nowhere cannot be reshaped in");
        assert!(err.contains("not a node here"), "{err}");
        assert_eq!(
            w.map().get("vault-casting").unwrap().nodes.len(),
            before,
            "a refused reshape must not change the map"
        );
    }

    /// **Adding a room to a level leaves the lift exactly as it was.** The shaft
    /// is a function of the levels and their cores, and neither changed — so the
    /// car is not disturbed and nobody mid-ride is set down.
    #[test]
    fn a_reshape_within_a_level_does_not_touch_the_lift() {
        let mut w = vault();
        let shaft_before = w.shaft().to_vec();
        w.reshape(&MapEdit::AddNode {
            area: "vault-casting".into(),
            node: Box::new(room("annex", &["green-room"])),
        })
        .unwrap();
        assert_eq!(
            w.shaft(),
            shaft_before.as_slice(),
            "the shaft must be unchanged"
        );
        assert!(w.lift().is_some(), "the lift is still there");
    }

    /// **The vault's lift serves every level, in order.** The shaft is built
    /// from the map — one landing per level, bottom to top — so the car knows
    /// only floor indices and the world knows which core each is.
    #[test]
    fn the_lift_serves_a_landing_on_every_level() {
        let w = vault();
        let shaft = w.shaft();
        assert!(shaft.len() >= 2, "the vault has more than one level");
        // Every landing is a real node, and each is on a distinct level.
        let areas: BTreeSet<&str> = shaft.iter().map(|c| c.area.as_str()).collect();
        assert_eq!(areas.len(), shaft.len(), "two floors shared a level");
        assert!(w.lift().is_some(), "a multi-floor building has a lift");
    }

    /// **The lift carries a rider to the floor they chose.** Call it, board it,
    /// ride it: the rider is set down on the landing of the level they picked,
    /// and the trip takes real time (the car moves a floor at a moment).
    #[test]
    fn the_lift_carries_a_rider_to_the_floor_they_chose() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        let (from, dest) = (0usize, shaft.len() - 1);
        w.enter("m1", "Maker-01", shaft[from].clone()).unwrap();
        w.mark_seen("m1");

        // Bring the car to our landing and wait for its doors.
        w.call_lift(from);
        for _ in 0..50 {
            if w.lift().unwrap().boardable_at(from) {
                break;
            }
            w.tick();
        }
        assert!(
            w.lift().unwrap().boardable_at(from),
            "the car never came to be boarded"
        );

        // Board for the top floor; it should not arrive the same moment.
        assert!(w.ride_lift("m1", dest), "boarding was refused");
        assert_eq!(
            w.actor("m1").unwrap().at,
            shaft[from],
            "carried off instantly"
        );

        // Ride it out. The rider is set down on the chosen landing.
        for _ in 0..50 {
            if w.actor("m1").unwrap().at == shaft[dest] {
                break;
            }
            w.tick();
        }
        assert_eq!(
            w.actor("m1").unwrap().at,
            shaft[dest],
            "the rider was not set down at the floor they chose"
        );
        assert!(w.at_landing("m1") == Some(dest));
    }

    /// **Two riders bound for different floors are both set down.** The car is
    /// shared, and the far rider presses last — which a single-target car would
    /// obey by carrying the near rider straight past their floor. Each must be
    /// let off where they chose.
    #[test]
    fn two_riders_bound_for_different_floors_are_both_set_down() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        assert!(shaft.len() >= 3, "need three floors for a near and a far");
        let (from, near, far) = (0usize, 1usize, shaft.len() - 1);
        w.enter("m1", "Maker-01", shaft[from].clone()).unwrap();
        w.enter("m2", "Maker-02", shaft[from].clone()).unwrap();
        w.mark_seen("m1");
        w.mark_seen("m2");

        w.call_lift(from);
        for _ in 0..50 {
            if w.lift().unwrap().boardable_at(from) {
                break;
            }
            w.tick();
        }
        assert!(w.lift().unwrap().boardable_at(from), "the car never came");

        // Both board while the doors are open — the far one last.
        assert!(w.ride_lift("m1", near), "m1 boarding refused");
        assert!(w.ride_lift("m2", far), "m2 boarding refused");

        for _ in 0..50 {
            let done =
                w.actor("m1").unwrap().at == shaft[near] && w.actor("m2").unwrap().at == shaft[far];
            if done {
                break;
            }
            w.tick();
        }
        assert_eq!(
            w.actor("m1").unwrap().at,
            shaft[near],
            "the near rider was carried past their floor"
        );
        assert_eq!(
            w.actor("m2").unwrap().at,
            shaft[far],
            "the far rider never arrived"
        );
    }

    /// **Walking off the landing cancels the ride.** A rider waits on its origin
    /// landing until the car opens at its floor; if it changes its mind and walks
    /// away, it must not be teleported to the old destination when the car later
    /// gets there.
    #[test]
    fn walking_off_the_landing_cancels_the_ride() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        let (from, dest) = (0usize, shaft.len() - 1);
        w.enter("m1", "Maker-01", shaft[from].clone()).unwrap();
        w.mark_seen("m1");

        w.call_lift(from);
        for _ in 0..50 {
            if w.lift().unwrap().boardable_at(from) {
                break;
            }
            w.tick();
        }
        assert!(w.ride_lift("m1", dest), "boarding refused");
        assert_eq!(w.riding("m1"), Some(dest));

        // Change your mind: walk to a room on this level instead.
        let room = Where::new(shaft[from].area.clone(), "command-room");
        w.set_off("m1", room).unwrap();
        assert_eq!(w.riding("m1"), None, "walking away left the ride pending");

        // Ride the car all the way out; the walker is never carried off to dest.
        for _ in 0..50 {
            w.tick();
        }
        assert_ne!(
            w.actor("m1").unwrap().at,
            shaft[dest],
            "a cancelled rider was still teleported to the old destination"
        );
    }

    /// A body on a landing hears the car work — its doors, its arrival — so the
    /// lift is a thing that happens in the world, not a silent teleport.
    #[test]
    fn a_body_on_a_landing_hears_the_lift_arrive() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        let floor = shaft.len() - 1; // somewhere the car is not already parked
        w.enter("m1", "Maker-01", shaft[floor].clone()).unwrap();
        w.mark_seen("m1");
        w.call_lift(floor);
        for _ in 0..50 {
            if w.lift().unwrap().boardable_at(floor) {
                break;
            }
            w.tick();
        }
        let heard: Vec<_> = crate::witness::since(&w, "m1")
            .into_iter()
            .filter(|s| matches!(&s.what, Happening::Stirred { .. }))
            .collect();
        assert!(
            heard.iter().any(|s| matches!(&s.what,
                Happening::Stirred { text, .. } if text.contains("lift arrives"))),
            "the body on the landing did not hear the lift arrive: {heard:?}"
        );
    }

    #[test]
    fn walking_out_of_a_room_lets_go_of_what_was_held() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.take("m1", Some("a-character")).unwrap();
        w.set_off("m1", at("green-room")).unwrap();
        // Let go on standing up, not on arriving. A claim held for the length
        // of a walk across the building is a claim nobody else can have while
        // its holder is in a corridor.
        assert!(w.actor("m1").unwrap().hold.is_none());
        assert!(w.holder_of("a-character").is_none());
    }

    /// **Two bodies crossing at the lift run into each other and both stop.**
    /// The whole point of the collision: a pair chasing each other between
    /// levels, forever swapping, finally end up in one place. Both leave the
    /// casting level for elsewhere, so their first leg ends at the lift (`core`),
    /// where they meet — still on their way — and stop there.
    #[test]
    fn two_travellers_crossing_at_the_lift_bump_and_both_stop() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.enter("m2", "Maker-02", at("green-room")).unwrap();
        let elsewhere = Where::new("vault-chronicle", "core");
        w.set_off("m1", elsewhere.clone()).unwrap();
        w.set_off("m2", elsewhere).unwrap();

        // One leg carries both to the casting lift, still bound onward.
        w.tick();
        assert_eq!(w.actor("m1").unwrap().at, at("core"), "m1 not at the lift");
        assert_eq!(w.actor("m2").unwrap().at, at("core"), "m2 not at the lift");

        // The collision stops both — neither carries on to the other level.
        assert!(
            w.actor("m1").unwrap().walk.is_none(),
            "m1 kept walking through the collision"
        );
        assert!(w.actor("m2").unwrap().walk.is_none(), "m2 kept walking");

        // And it is recorded, once, as a collision between the two of them.
        let bumps: Vec<_> = w
            .log()
            .iter()
            .filter(|e| matches!(&e.what, Happening::Bumped { .. }))
            .collect();
        assert_eq!(bumps.len(), 1, "one collision per pair: {bumps:?}");
        assert!(
            matches!(&bumps[0].what, Happening::Bumped { into } if into == "m2")
                && bumps[0].actor == "m1",
            "{:?}",
            bumps[0]
        );
    }

    /// **Two bodies swapping levels cross on the lift and end up together.**
    /// This is the exact shape of the chase the collision exists to break: each
    /// heads for the level the other is on, over the one lift between them, and
    /// without this they would swap past each other for ever. Instead they meet
    /// on it and both stop, in the same room.
    #[test]
    fn two_travellers_swapping_levels_meet_and_stop_together() {
        let mut w = vault();
        let command = Where::new("vault-command", "core");
        let chronicle = Where::new("vault-chronicle", "core");
        w.enter("m1", "Maker-01", command.clone()).unwrap();
        w.enter("m2", "Maker-02", chronicle.clone()).unwrap();
        w.set_off("m1", chronicle.clone()).unwrap();
        w.set_off("m2", command).unwrap();

        w.tick();
        let p1 = w.actor("m1").unwrap().at.clone();
        let p2 = w.actor("m2").unwrap().at.clone();
        assert_eq!(p1, p2, "the swap left them apart: {p1:?} vs {p2:?}");
        assert!(w.actor("m1").unwrap().walk.is_none(), "m1 kept going");
        assert!(w.actor("m2").unwrap().walk.is_none(), "m2 kept going");

        let bumps: Vec<_> = w
            .log()
            .iter()
            .filter(|e| matches!(&e.what, Happening::Bumped { .. }))
            .collect();
        assert_eq!(
            bumps.len(),
            1,
            "one collision for the crossing pair: {bumps:?}"
        );
    }

    /// Arriving where somebody is standing is joining them, not a collision — you
    /// walked there on purpose. Only a body still on its way bumps.
    #[test]
    fn arriving_where_somebody_stands_is_not_a_bump() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        // m2 walks to green-room, where m1 is — a same-level journey it finishes
        // in one leg, so it arrives rather than crossing anybody mid-way.
        w.set_off("m2", at("green-room")).unwrap();
        w.settle();
        assert_eq!(w.actor("m2").unwrap().at, at("green-room"));
        assert!(
            !w.log()
                .iter()
                .any(|e| matches!(&e.what, Happening::Bumped { .. })),
            "arriving at a room where somebody stands read as a collision"
        );
    }
}

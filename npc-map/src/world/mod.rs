//! The shared state of a vault: who is where, what each body holds, and the
//! journeys bodies are on.
//!
//! One world is shared by every character in it, so each change is one step
//! under one lock, and a claim is checked across the whole building rather than
//! the room alone. The map says what the vault is; this module says what is true
//! of it now. A journey costs time and is answered by an event when it ends, and
//! leaving a station releases it. The world records what happened without
//! interpreting it; who perceives what is decided elsewhere.
//!
//! - `bodies`: entering, leaving and reshaping.
//! - `events`: the happenings the log keeps, and who may perceive them.
//! - `speech`: speaking, gesturing and decisions.
//! - `stations`: holds on stations and subjects.
//! - `movement`: journeys, one leg per tick.
//! - `lifts`: the building's lift.
//! - `world_state`: the state shared by every copy, and the event window.
//! - Tests: `tests_bodies`, `tests_lift`, `tests_log`, `tests_map`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::lift::{Lift, Moment};
use crate::load::MapSet;
use crate::mutate::MapEdit;
use crate::part::PartKind;
use crate::route;
use crate::salience::Weight;
use crate::schema::{AreaKind, Node, NodeKind};
use crate::state::{BodyState, LiftState, State, WalkState};

pub use crate::schema::Where;

/// A step of world time. Advances once per change, so ordering is total and
/// "since you last looked" is a comparison rather than a diff.
///
/// It is not a clock. One tick is one thing happening, and everything that
/// happens in the same [`World::step`] shares a tick, because a step is one
/// moment however many bodies move in it.
pub type Tick = u64;

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
    /// Who is waiting on a landing for the car to come, and the floor each will
    /// ride to. They board when the car opens at their landing — see
    /// [`World::ride_lift`].
    waiting: BTreeMap<String, usize>,
    /// The company each body was in when it began deciding, for the bodies that
    /// are deciding now. See [`World::begin_decision`].
    deciding: BTreeMap<String, Deliberation>,
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
            waiting: BTreeMap::new(),
            deciding: BTreeMap::new(),
        }
    }
}

mod bodies;
mod events;
mod lifts;
mod movement;
mod speech;
mod stations;
#[cfg(test)]
mod tests_bodies;
#[cfg(test)]
mod tests_lift;
#[cfg(test)]
mod tests_log;
#[cfg(test)]
mod tests_map;
mod world_state;

pub use events::*;
pub use movement::*;
use speech::*;
pub use stations::*;

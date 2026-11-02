//! What two copies of one world must agree on.
//!
//! A world's characters can run on a member that follows the world rather
//! than running it, and that member builds each character's turn from its own
//! copy: what it perceives, where it may go, what the lift offers it. So
//! everything a turn is built from that the world's authority changes has to
//! reach every copy, and [`State`] is that set — read off a world by
//! [`World::state`](crate::world::World::state) and put into one by
//! [`World::restore_body`](crate::world::World::restore_body) and
//! [`World::restore_lift`](crate::world::World::restore_lift).
//!
//! `World::state` names every field of the world, its bodies, their holds and
//! their walks without a `..`, so a field added to any of them does not
//! compile until it is said to be either here or one copy's own.

use std::collections::BTreeMap;

use crate::lift::Doors;
use crate::schema::Where;

/// Everything about a world that its copies share.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Each body, by id.
    pub bodies: BTreeMap<String, BodyState>,
    /// The building's lift, when it has one.
    pub lift: Option<LiftState>,
}

/// One body, as every copy of its world has it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyState {
    /// Where it stands — the last place reached, when it is walking.
    pub at: Where,
    /// The station it holds: the part, and the subject it claims.
    pub hold: Option<(String, Option<String>)>,
    /// The journey under way: where to, and the places still to be reached.
    pub walk: Option<WalkState>,
}

/// A journey under way, as every copy of its world has it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkState {
    pub toward: Where,
    pub ahead: Vec<Where>,
}

/// The lift, as every copy of its world has it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftState {
    pub floor: usize,
    pub doors: Doors,
    pub stops: Vec<usize>,
    /// Who is aboard, and the floor each is bound for.
    pub riders: BTreeMap<String, usize>,
    /// Who is waiting on a landing for the car, and the floor each will ride to.
    pub waiting: BTreeMap<String, usize>,
}

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

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use crate::load::MapSet;
use crate::part::PartKind;
use crate::route;
use crate::schema::Node;

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
    /// A station lit up. The subject is carried but is only *shown* to
    /// somebody in the same room.
    TookStation {
        subject: Option<String>,
    },
    LeftStation {
        subject: Option<String>,
    },
    /// Somebody spoke. `to` is who they aimed it at, and is a fact about the
    /// utterance rather than about who received it — everybody in the room
    /// hears it either way. `None` is spoken to the room at large.
    Said {
        to: Option<String>,
        words: String,
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
            Happening::GotThere { .. } | Happening::LostTheWay { .. }
        )
    }

    /// Whether this happened inside one body and is visible to nobody else.
    pub fn is_private(&self) -> bool {
        self.is_outcome() || matches!(self, Happening::SetOut { .. })
    }
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

/// The state of one world, shared by everything in it.
#[derive(Debug, Clone)]
pub struct World {
    map: MapSet,
    actors: BTreeMap<String, Actor>,
    log: Vec<Event>,
    now: Tick,
}

impl World {
    pub fn new(map: MapSet) -> World {
        World {
            map,
            actors: BTreeMap::new(),
            log: Vec::new(),
            now: 0,
        }
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
        self.now += 1;
        self.actors.insert(
            id.clone(),
            Actor {
                id: id.clone(),
                name: name.into(),
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
        for (id, to) in &legs {
            self.land(id, to, at);
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
        self.utter(id, None, words)
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
        self.utter(id, Some(to), words)
    }

    /// Do something in the room without speaking. Everybody standing there sees
    /// it; nobody outside does — the same reach as [`World::say`], because it is
    /// the same room.
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

    fn utter(&mut self, id: &str, to: Option<&str>, words: impl Into<String>) -> Done {
        let (place, to) = self.aim(id, to)?;
        self.now += 1;
        self.log.push(Event {
            at: self.now,
            actor: id.into(),
            place,
            what: Happening::Said {
                to,
                words: words.into(),
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
    pub fn log(&self) -> &[Event] {
        &self.log
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

    #[test]
    fn a_body_enters_where_it_is_told_to() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        assert_eq!(w.actor("m1").unwrap().at, at("core"));
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
}

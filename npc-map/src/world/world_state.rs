//! Holds the state of one world that every body in it shares, and the cursor each
//! body has read up to.
//!
//! Every field is named, so a copy built from another member's state cannot
//! silently lack one. Shared state is laid over the map, and each body's
//! position, holds and walks travel in full. What happened is recorded in order
//! on the world's own clock, and a body's cursor is how far it has read. Restoring
//! a body or the lift from another copy settles them silently, because nobody in
//! the building did anything.

use super::*;

impl World {
    /// Take in what happened in this world elsewhere — on the member that is
    /// its authority — so the bodies here perceive it as they would anything
    /// done in front of them. Each event is stamped with a tick of this
    /// world's own clock, so readers' cursors here order it among the rest.
    /// Returns the tick the last one took.
    pub fn witness(&mut self, events: impl IntoIterator<Item = Event>) -> Tick {
        for mut event in events {
            self.now += 1;
            event.at = self.now;
            self.log.push(event);
        }
        self.forget_old_events();
        self.now
    }

    /// Everything about this world its copies share ([`crate::state`]).
    ///
    /// **Every field is named, none with `..`.** A world's characters can run
    /// on a member that follows it, built each turn from that member's copy, so
    /// a field one copy keeps and the others never hear of is a turn built from
    /// a world that is not there. Riders left out of the lift's record were
    /// exactly that: a drone its authority had aboard was offered the rooms of
    /// its landing as if it had never asked, walked off, and gave up the ride,
    /// and four drones went back and forth between the lift and the nest hall.
    /// A field added to the world, a body, a hold or a walk does not compile
    /// here until it is said to be shared or one copy's own.
    pub fn state(&self) -> State {
        let World {
            // The map is its files and what the world's builders made, laid
            // over it from the world's records — shared, but not from here.
            map: _,
            actors,
            // Names come with each body as it is embodied, in every copy.
            names: _,
            // What happened reaches the others as it happens (`witness`), each
            // copy stamping it on its own clock.
            log: _,
            now: _,
            lift,
            // Derived from the map.
            shaft: _,
            riders,
            waiting,
            // A decision is made where the mind making it runs, and travels
            // with the act it chose (`deliberation`, `restore_decision`).
            deciding: _,
        } = self;
        let bodies = actors
            .iter()
            .map(|(id, actor)| {
                let Actor {
                    id: _,
                    name: _,
                    at,
                    hold,
                    walk,
                    // How far this copy's reader of the body has read its log.
                    looked: _,
                } = actor;
                let hold = hold.as_ref().map(|h| {
                    let Hold {
                        subject,
                        part,
                        // On this copy's clock.
                        since: _,
                    } = h;
                    (part.clone(), subject.clone())
                });
                let walk = walk.as_ref().map(|w| {
                    let Walk {
                        toward,
                        ahead,
                        // On this copy's clock.
                        set_out: _,
                    } = w;
                    WalkState {
                        toward: toward.clone(),
                        ahead: ahead.iter().cloned().collect(),
                    }
                });
                let body = BodyState {
                    at: at.clone(),
                    hold,
                    walk,
                };
                (id.clone(), body)
            })
            .collect();
        let lift = lift.as_ref().map(|car| LiftState {
            floor: car.floor(),
            doors: car.doors(),
            stops: car.stops(),
            riders: riders.clone(),
            waiting: waiting.clone(),
        });
        State { bodies, lift }
    }

    /// Set `id` as another copy of the world has it: where it stands, what it
    /// holds, and the journey it is on. Nothing is logged: the moves a body
    /// makes are told as they happen ([`Self::witness`]), and this only
    /// settles where they got it. A walk through a place the map does not
    /// have is dropped; a body moved is off any landing it was riding from,
    /// until the lift's own state says otherwise ([`Self::restore_lift`]).
    pub fn restore_body(&mut self, id: &str, body: &BodyState) -> Done {
        if self.node(&body.at).is_none() {
            return Err(Refused::NoSuchPlace(body.at.clone()));
        }
        let walk = body.walk.as_ref().filter(|w| {
            self.node(&w.toward).is_some() && w.ahead.iter().all(|p| self.node(p).is_some())
        });
        let now = self.now;
        let actor = self
            .actors
            .get_mut(id)
            .ok_or(Refused::NoSuchActor(id.into()))?;
        let moved = actor.at != body.at;
        actor.at = body.at.clone();
        let same_walk = match (&actor.walk, walk) {
            (Some(had), Some(w)) => had.toward == w.toward && had.ahead.iter().eq(w.ahead.iter()),
            (None, None) => true,
            _ => false,
        };
        if !same_walk {
            actor.walk = walk.map(|w| Walk {
                toward: w.toward.clone(),
                ahead: w.ahead.iter().cloned().collect(),
                set_out: now,
            });
        }
        let held = actor.hold.as_ref().map(|h| (&h.part, &h.subject));
        if held != body.hold.as_ref().map(|(p, s)| (p, s)) {
            actor.hold = body.hold.clone().map(|(part, subject)| Hold {
                subject,
                part,
                since: now,
            });
        }
        if moved {
            self.riders.remove(id);
            self.waiting.remove(id);
        }
        Ok(())
    }

    /// Set the lift as another copy of the world has it ([`Lift::restore`]),
    /// and who is aboard it and waiting for it. Nothing is logged: nobody in
    /// the building did anything. A rider bound for a floor off the shaft is
    /// dropped, as the car's own stops are. Whether the world has a lift.
    pub fn restore_lift(&mut self, car: &LiftState) -> bool {
        let Some(lift) = self.lift.as_mut() else {
            return false;
        };
        lift.restore(car.floor, car.doors, &car.stops);
        let floors = self.shaft.len();
        let on_shaft = |m: &BTreeMap<String, usize>| -> BTreeMap<String, usize> {
            m.iter()
                .filter(|(_, f)| **f < floors)
                .map(|(b, f)| (b.clone(), *f))
                .collect()
        };
        self.riders = on_shaft(&car.riders);
        self.waiting = on_shaft(&car.waiting);
        true
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
    pub(super) fn forget_old_events(&mut self) {
        if self.log.len() <= KEEP_EVENTS + EVENT_SLACK {
            return;
        }
        let drop = self.log.len() - KEEP_EVENTS;
        self.log.drain(..drop);
    }
}

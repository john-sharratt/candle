//! Moves bodies along their journeys, one leg per tick, and answers each journey
//! with how it ended.
//!
//! Journeys are walked rather than teleported, so distance is paid and an
//! outcome has to be reported rather than known in advance. A body set off
//! elsewhere, teleported, placed or stopped has its journey answered in one
//! place, so it is never told it failed to reach the room it is standing in.
//! Within a tick, swaps are resolved before ordinary arrivals, so no departure
//! is logged that would have to be taken back. A traveller that lands where
//! someone already stands bumps into them only while it is still on its way.

use super::*;

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

/// Why a journey ended before it arrived.
///
/// Every one of these is something the walker itself did. Nothing in the vault
/// stops a body getting where it is going — no locked doors, no crowding in a
/// corridor — so a walk that fails failed because its owner changed its mind,
/// and the reason names which way it changed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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

impl World {
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
        // **Setting off again for where you are already going is carrying on.**
        // Ending the walk and starting it over told the body it "never got to
        // the lift, having set off somewhere else" in the same breath as
        // "got to the lift" — a contradiction it then reasoned about as a trap.
        if let Some(walk) = self.actors[id].walk.as_ref().filter(|w| w.toward == to) {
            return Ok(walk.to_go(&from));
        }

        self.end_walk(id, &from, Lost::Diverted);
        // Walking away cancels a lift you were waiting on: you are no longer on
        // the landing to be carried off, so the car must not set you down later.
        self.riders.remove(id);
        self.waiting.remove(id);
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
        self.board_for_the_way();
        // A body riding — waiting on its landing for the car, or aboard — goes
        // nowhere on foot: the car sets it down, and its walk carries on from
        // the landing it is set down on.
        let legs: Vec<(String, Where)> = self
            .actors
            .values()
            .filter(|a| self.riding(&a.id).is_none())
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
        //
        // **Never on a landing.** A landing is where bodies wait for the car
        // together, and a journey to another floor passes through one: ending
        // it there for meeting the people already waiting stopped every ride a
        // queue had formed for.
        for (id, _) in &legs {
            if caught.contains(id) {
                continue;
            }
            let Some(a) = self.actors.get(id) else {
                continue;
            };
            if a.walk.is_none() || self.shaft.contains(&a.at) {
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
        // A moment that moves nobody is not the end while somebody rides: the
        // car is still on its way to them, or with them aboard.
        while self.tick() > 0 || !self.riders.is_empty() || !self.waiting.is_empty() {
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
    pub(super) fn land(&mut self, id: &str, to: &Where, at: Tick) {
        let Some(actor) = self.actors.get(id) else {
            return;
        };
        let from = actor.at.clone();
        // Any relocation clears a pending ride: the lift's own set-down has
        // already taken the rider off the roster before landing it here, so this
        // only catches a body moved by some other means (placed, teleported)
        // while it was waiting, which must not then be carried off a second time.
        self.riders.remove(id);
        self.waiting.remove(id);
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
    pub(super) fn end_walk(&mut self, id: &str, ending_at: &Where, why: Lost) {
        if self.actors.get(id).is_some_and(|a| a.walk.is_some()) {
            self.now += 1;
            let at = self.now;
            self.end_walk_at(id, ending_at, why, at);
        }
    }

    /// [`World::end_walk`] at a moment the caller has already opened.
    pub(super) fn end_walk_at(&mut self, id: &str, ending_at: &Where, why: Lost, at: Tick) {
        let Some(actor) = self.actors.get_mut(id) else {
            return;
        };
        let Some(walk) = actor.walk.take() else {
            return;
        };
        let place = actor.at.clone();
        // A ride taken for the journey ends with it: the car must not carry
        // off a body that has given up the way that went through it.
        self.riders.remove(id);
        self.waiting.remove(id);
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
}

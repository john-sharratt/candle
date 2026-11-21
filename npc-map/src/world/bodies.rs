//! Puts bodies into and takes them out of a world, and reshapes its map while
//! it runs.
//!
//! A body that leaves gives up what it holds, since a station claimed by an
//! actor nothing can release would never be usable again. Leaving twice is
//! reported, so a caller that has lost track of its session is told. Arrivals
//! and departures are announced to the room, because a body vanishing
//! unannounced is as wrong as one appearing that way. A reshape is applied to a
//! copy and validated before anything changes, and a body left standing where a
//! room was drowned is moved to the area's way in and logged both ways.

use super::*;

impl World {
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
        self.deciding.remove(id);
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
            self.waiting.remove(&id);
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
}

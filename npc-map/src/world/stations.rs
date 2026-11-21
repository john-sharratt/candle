//! Holds station claims, so each station is taken once and each subject is held
//! by one body anywhere in the world.
//!
//! A claim is global, because the reason to hold one is coherence and coherence
//! does not partition by floor. Taking a station is a single step: a free
//! station, an unheld subject and the sitting down all happen together or not at
//! all. A body that stops at a station on its way has its walk answered rather
//! than dropped, and getting up releases what was claimed.

use super::*;

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

impl World {
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
    pub(super) fn release_at(&mut self, id: &str, at: Tick) -> Done {
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
}

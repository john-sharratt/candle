//! Runs the building's lift: its shaft, the calls and rides made on it, and the
//! car's movement between floors.
//!
//! The shaft is derived from the map's levels and their landings, and the car is
//! a state machine that knows only floor indices. A rider is waiting for the car
//! from the moment it asks, and stays on its landing until the car opens there,
//! so it is neither offered other walks nor carried off without it. The car
//! moves on its own clock whether or not anybody walks, and each moment is
//! voiced on the floor where it happens. A reshape that adds a room within a
//! level leaves the lift alone; a change to the floors returns the car to the
//! arrival floor.

use super::*;

impl World {
    /// The cores of the building's levels, bottom to top by `ordinal` — the
    /// landings the lift stops at, one per floor. Everything the lift needs to
    /// know about the map is captured here once, so the car itself stays a plain
    /// state machine that knows only floor indices.
    pub(super) fn build_shaft(map: &MapSet) -> Vec<Where> {
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

    /// The floor a body is riding to, if it has asked to ride and not yet been
    /// set down — waiting for the car on its landing, or aboard. A rider stays on
    /// its origin landing until the car opens at its floor (see
    /// [`World::ride_lift`]); until then it has already chosen, so it is neither
    /// offered the lift again nor told the car has left without it.
    pub fn riding(&self, id: &str) -> Option<usize> {
        self.riders
            .get(id)
            .or_else(|| self.waiting.get(id))
            .copied()
    }

    /// Whether a body has asked to ride and is waiting on its landing for the
    /// car to come.
    pub fn waiting_for_lift(&self, id: &str) -> bool {
        self.waiting.contains_key(id)
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

    /// Ride to `dest`: board now when the car is open at the rider's landing,
    /// otherwise call it there and wait, boarding when it opens. Either way the
    /// rider comes off at `dest` when the doors open there, and stays on its
    /// landing until then, so a body waiting with it is company. `false` —
    /// refused — when the rider is not on a landing, there is no lift, `dest` is
    /// off the shaft, or `dest` is the floor it is on.
    ///
    /// **Asking to ride is waiting for the car.** Calling it and boarding were
    /// two acts with a wait between, and a body given a turn in the wait spent it
    /// walking away: measured, Makers called the car and left the landing before
    /// it came, five and six times running, until the cast agreed among
    /// themselves that the lift was broken.
    pub fn ride_lift(&mut self, rider: &str, dest: usize) -> bool {
        let Some(from) = self.at_landing(rider) else {
            return false;
        };
        if dest >= self.shaft.len() || dest == from {
            return false;
        }
        let Some(lift) = self.lift.as_mut() else {
            return false;
        };
        if lift.boardable_at(from) {
            self.riders.insert(rider.to_string(), dest);
            lift.call(dest);
        } else {
            self.waiting.insert(rider.to_string(), dest);
            lift.call(from);
        }
        true
    }

    /// Advance the car one moment, on its own clock — it moves whether or not
    /// anybody is walking. Each moment is voiced on the floor it happened at, so
    /// a body on a landing hears the car close, pass, or arrive; and a rider
    /// bound for a floor the car has just opened at is set down there.
    pub(super) fn step_lift(&mut self) {
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
            // Board whoever was waiting on this landing, and send the car on to
            // where each is going.
            let boarding: Vec<(String, usize)> = self
                .waiting
                .iter()
                .filter(|(r, _)| self.at_landing(r) == Some(floor))
                .map(|(r, d)| (r.clone(), *d))
                .collect();
            for (r, dest) in boarding {
                self.waiting.remove(&r);
                self.riders.insert(r, dest);
                if let Some(lift) = self.lift.as_mut() {
                    lift.call(dest);
                }
            }
        }
    }

    /// Ride the lift for every walk whose next leg changes floor.
    ///
    /// **A journey to another floor is one journey.** The route runs through
    /// the landings — out to this floor's, between landings, on from the far
    /// one — and the leg between landings is the car's, not the walker's: a
    /// body on its landing whose way goes on from another landing asks to ride
    /// there ([`World::ride_lift`]), waits for the car with whoever else is
    /// waiting, and is set down on the landing its walk goes on from, which
    /// [`World::land`] reads as the walk's next stop. Asked to make the ride
    /// itself — walk to the lift, call it, name the level, walk on — a cast
    /// stood on the landing and never once called the car.
    ///
    /// A world with no lift, or a leg that crosses a door rather than the
    /// shaft, is walked as any other leg.
    pub(super) fn board_for_the_way(&mut self) {
        let rides: Vec<(String, usize)> = self
            .actors
            .values()
            .filter(|a| self.riding(&a.id).is_none())
            .filter_map(|a| {
                let walk = a.walk.as_ref()?;
                let next = walk.ahead.front()?;
                self.at_landing(&a.id)?;
                if next.area == a.at.area || !self.shaft.contains(next) {
                    return None;
                }
                let end = walk.leg_end(&a.at)?;
                let floor = self.shaft.iter().position(|c| c == &end)?;
                Some((a.id.clone(), floor))
            })
            .collect();
        for (rider, floor) in rides {
            self.ride_lift(&rider, floor);
        }
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
    pub(super) fn reshaft(&mut self) {
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
        self.waiting.clear();
    }
}

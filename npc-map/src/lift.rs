//! The lift — a car that shuttles between the floors of a building.
//!
//! # Why a state machine, and not a portal
//!
//! A portal joins two rooms and stepping through it is instant, which is right
//! for a doorway and wrong for a lift. A lift is a *shared, slow* thing: there
//! is one car, it is on one floor at a time, and getting from one level to
//! another means calling it, waiting for it, getting in, and riding. That
//! waiting and riding is not friction for its own sake — it is the one place two
//! bodies crossing a building on their own errands are made to be in the same
//! room at the same time, which is what lets them run into each other and talk.
//! So the lift is modelled as what it is: a machine with a position, doors, and
//! a direction, stepped one moment at a time.
//!
//! # What it is not
//!
//! It is not the map and it does not know about actors. A floor is an index into
//! an ordered shaft (`0` at one end), nothing more; who is standing on a landing
//! or riding in the car is the world's business, layered on top. Keeping the car
//! pure is what makes its behaviour — the sequence of doors and floors a call
//! produces — testable on its own, without a building around it.
//!
//! # One car, many buttons
//!
//! The car serves a *set* of stops, not one at a time: a call adds the floor to
//! that set, and the car works through them, stopping at each in turn until none
//! is left. This is the whole reason it is a lift and not two portals — a second
//! Maker pressing a button while the first is aboard must not carry the first
//! past their floor. It heads for the nearest pending stop, so a floor on the way
//! is reached before a farther one; each stop reached is served (doors open) and
//! dropped, and the car goes idle only once the set is empty.

use std::collections::BTreeSet;

/// The doors, at whatever floor the car is level with. While the car is between
/// floors the doors are [`Doors::Closed`]; there is nowhere to open onto.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Doors {
    Open,
    Closed,
}

/// One thing the car did in a single [`Lift::step`], in the order it happened.
/// The world turns each into something the people on the floor it names would
/// hear — doors closing, a car passing, a car arriving.
///
/// The floor a moment names is the one it *happened at*, so the world knows
/// which landing to voice it on: [`Moment::Closed`] and the first move name the
/// floor left behind, [`Moment::Passed`] a floor gone by, [`Moment::Arrived`]
/// the floor reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moment {
    /// The doors closed on this floor; the car is about to move.
    Closed(usize),
    /// The car passed this floor without stopping.
    Passed(usize),
    /// The car reached this floor and its doors opened.
    Arrived(usize),
}

/// A lift serving an ordered run of floors. A floor is an index into that run,
/// `0` at one end of the shaft; the car moves one floor per [`Lift::step`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lift {
    /// How many floors the shaft serves. Floor indices are `0..floors`.
    floors: usize,
    /// The floor the car is level with — the last one it reached. While moving,
    /// this advances one floor at a time; it is never "between" two.
    at: usize,
    doors: Doors,
    /// The floors the car still has to reach, from calls and ridden destinations.
    /// Empty is idle: parked, doors open. The car serves the nearest first and
    /// drops each as it opens there.
    stops: BTreeSet<usize>,
}

impl Lift {
    /// A lift of `floors` floors, parked at `start` with its doors open. An idle
    /// lift keeps its doors open, so somebody arriving at its floor can step
    /// straight in.
    ///
    /// Panics if `floors` is zero or `start` is not a floor — a lift with no
    /// shaft, or parked off the end of one, is a bug in the building, not a
    /// state to represent.
    pub fn new(floors: usize, start: usize) -> Lift {
        assert!(floors > 0, "a lift must serve at least one floor");
        assert!(
            start < floors,
            "the car cannot start off the end of the shaft"
        );
        Lift {
            floors,
            at: start,
            doors: Doors::Open,
            stops: BTreeSet::new(),
        }
    }

    /// How many floors the shaft serves.
    pub fn floors(&self) -> usize {
        self.floors
    }

    /// The floor the car is level with.
    pub fn floor(&self) -> usize {
        self.at
    }

    pub fn doors(&self) -> Doors {
        self.doors
    }

    /// Whether the car is parked — no stops left to serve.
    pub fn is_idle(&self) -> bool {
        self.stops.is_empty()
    }

    /// Whether the car still has a stop to reach (doors shut and moving, or open
    /// at one stop with another still ahead).
    pub fn is_moving(&self) -> bool {
        !self.stops.is_empty()
    }

    /// The pending stop nearest the car's current floor — where [`Lift::step`]
    /// heads next. Ties go to the lower floor, so the sequence is deterministic.
    fn nearest_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .copied()
            .min_by_key(|&f| (f.abs_diff(self.at), f))
    }

    /// The floor you can step in or out at right now — the car is level with it
    /// and its doors are open. `None` while it is moving.
    pub fn open_floor(&self) -> Option<usize> {
        (self.doors == Doors::Open).then_some(self.at)
    }

    /// Whether somebody standing on `floor` can get in or out this moment.
    pub fn boardable_at(&self, floor: usize) -> bool {
        self.open_floor() == Some(floor)
    }

    /// Add `floor` to the car's stops — a call from a landing or a choice from
    /// inside; the car does not know the difference and does not need to.
    /// Off-the-shaft floors are ignored, and a call to the floor the car is
    /// already open at does nothing, so pressing the button you are standing at
    /// is harmless.
    ///
    /// The car does not move here; the stop is recorded, and [`Lift::step`]
    /// carries it one floor at a time. A call while the car is already on its way
    /// adds a stop rather than replacing the current one — the car queues, and
    /// nobody aboard is carried past their floor.
    pub fn call(&mut self, floor: usize) {
        if floor >= self.floors {
            return;
        }
        if self.at == floor && self.doors == Doors::Open {
            return;
        }
        self.stops.insert(floor);
    }

    /// Advance the car one moment, returning what it did. Idle returns nothing.
    ///
    /// The sequence a call produces is: the doors close on the floor left behind,
    /// then one [`Moment::Passed`] per floor gone by, then [`Moment::Arrived`] on
    /// the floor reached, where the doors open again and that stop is dropped.
    /// With more than one stop pending it heads for the nearest, serves it, and
    /// comes round for the rest — so a call to the current floor while the doors
    /// are already open produces nothing, but a stop still ahead keeps it moving.
    pub fn step(&mut self) -> Vec<Moment> {
        let Some(target) = self.nearest_stop() else {
            return Vec::new();
        };
        if target == self.at {
            // The nearest stop is the floor the car sits at. Serve it: open up if
            // the doors are shut, and drop it either way.
            self.stops.remove(&self.at);
            return match self.doors {
                Doors::Closed => {
                    self.doors = Doors::Open;
                    vec![Moment::Arrived(self.at)]
                }
                Doors::Open => Vec::new(),
            };
        }
        // A move is coming. The doors close first, on their own moment, so a body
        // on the landing has the beat between the car being open and it leaving.
        if self.doors == Doors::Open {
            self.doors = Doors::Closed;
            return vec![Moment::Closed(self.at)];
        }
        // Doors shut, between floors: cover one, and serve it if it is a stop.
        self.at = if target > self.at {
            self.at + 1
        } else {
            self.at - 1
        };
        if self.stops.remove(&self.at) {
            self.doors = Doors::Open;
            vec![Moment::Arrived(self.at)]
        } else {
            vec![Moment::Passed(self.at)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_lift_is_parked_open_at_its_start_floor() {
        let lift = Lift::new(6, 5);
        assert_eq!(lift.floor(), 5);
        assert_eq!(lift.doors(), Doors::Open);
        assert!(lift.is_idle());
        assert_eq!(lift.open_floor(), Some(5));
        assert!(lift.boardable_at(5));
        assert!(!lift.boardable_at(4));
    }

    #[test]
    #[should_panic(expected = "at least one floor")]
    fn a_lift_with_no_shaft_is_a_bug() {
        Lift::new(0, 0);
    }

    #[test]
    #[should_panic(expected = "off the end")]
    fn a_car_parked_off_the_shaft_is_a_bug() {
        Lift::new(3, 3);
    }

    #[test]
    fn an_idle_step_does_nothing() {
        let mut lift = Lift::new(4, 1);
        assert_eq!(lift.step(), Vec::new());
        assert_eq!(lift.floor(), 1);
        assert!(lift.boardable_at(1));
    }

    #[test]
    fn calling_the_floor_it_is_open_at_is_a_no_op() {
        let mut lift = Lift::new(4, 2);
        lift.call(2);
        assert!(
            lift.is_idle(),
            "a call to the current open floor set it going"
        );
        assert_eq!(lift.step(), Vec::new());
        assert!(lift.boardable_at(2));
    }

    #[test]
    fn a_call_off_the_shaft_is_ignored() {
        let mut lift = Lift::new(4, 0);
        lift.call(9);
        assert!(lift.is_idle());
    }

    /// **The whole shape of a call**, floor by floor: the doors close on the
    /// floor left, the car passes each floor between, and it arrives — doors
    /// open — on the one called. A call up from 1 to 4 is Closed(1), Passed(2),
    /// Passed(3), Arrived(4): four moments, four ticks.
    #[test]
    fn a_call_closes_passes_and_arrives_floor_by_floor() {
        let mut lift = Lift::new(6, 1);
        lift.call(4);
        assert!(lift.is_moving());
        assert_eq!(lift.step(), vec![Moment::Closed(1)]);
        assert_eq!(lift.doors(), Doors::Closed);
        assert_eq!(lift.open_floor(), None, "no floor is boardable mid-shaft");
        assert_eq!(lift.step(), vec![Moment::Passed(2)]);
        assert_eq!(lift.floor(), 2);
        assert_eq!(lift.step(), vec![Moment::Passed(3)]);
        assert_eq!(lift.step(), vec![Moment::Arrived(4)]);
        assert_eq!(lift.floor(), 4);
        assert_eq!(lift.doors(), Doors::Open);
        assert!(lift.is_idle());
        assert!(lift.boardable_at(4));
        assert!(!lift.boardable_at(1));
    }

    /// The same downward: a call from 5 to 2 descends past 4 and 3.
    #[test]
    fn a_call_down_descends_past_the_floors_between() {
        let mut lift = Lift::new(6, 5);
        lift.call(2);
        let mut moments = Vec::new();
        for _ in 0..4 {
            moments.extend(lift.step());
        }
        assert_eq!(
            moments,
            vec![
                Moment::Closed(5),
                Moment::Passed(4),
                Moment::Passed(3),
                Moment::Arrived(2),
            ]
        );
        assert!(lift.boardable_at(2));
    }

    /// A call to the very next floor is Closed then Arrived — no floor passed.
    #[test]
    fn a_call_to_the_next_floor_is_close_then_arrive() {
        let mut lift = Lift::new(4, 1);
        lift.call(2);
        assert_eq!(lift.step(), vec![Moment::Closed(1)]);
        assert_eq!(lift.step(), vec![Moment::Arrived(2)]);
        assert!(lift.is_idle());
    }

    /// **Boardable only when level and open.** The car is boardable at its floor
    /// while idle, not while its doors are shut, and never at a floor it is not
    /// level with.
    #[test]
    fn boardable_tracks_the_doors_and_the_floor() {
        let mut lift = Lift::new(5, 0);
        assert!(lift.boardable_at(0));
        lift.call(3);
        // Doors close: no longer boardable, even before it has moved.
        lift.step();
        assert!(!lift.boardable_at(0));
        // Mid-shaft: boardable nowhere.
        lift.step();
        assert_eq!(lift.open_floor(), None);
        // Finish the trip.
        while lift.is_moving() {
            lift.step();
        }
        assert!(lift.boardable_at(3));
        assert!(!lift.boardable_at(0));
    }

    /// **A second button queues; it does not replace.** Heading to 4, then called
    /// back to 0 one floor along, the car serves both — nearest first — rather
    /// than abandoning either. From floor 1 the nearest stop is 0, so it turns
    /// down to 0, opens, then comes back up to 4. Nobody bound for 4 is dropped.
    #[test]
    fn a_second_call_queues_and_both_are_served() {
        let mut lift = Lift::new(6, 0);
        lift.call(4);
        lift.step(); // Closed(0)
        lift.step(); // Passed(1)
        assert_eq!(lift.floor(), 1);
        lift.call(0); // a second caller, below us
        let mut arrivals = Vec::new();
        while lift.is_moving() {
            for m in lift.step() {
                if let Moment::Arrived(f) = m {
                    arrivals.push(f);
                }
            }
        }
        assert_eq!(arrivals, vec![0, 4], "both stops served, nearest first");
        assert!(lift.is_idle());
        assert!(lift.boardable_at(4));
    }

    /// **The stranding case, head on.** A rider aboard for floor 3 must reach 3
    /// even when a second caller, on floor 5, presses the button while the car is
    /// on its way up. The car opens at 3 (setting the first rider down) and only
    /// then goes on to 5 — it never passes 3 without stopping.
    #[test]
    fn a_stop_on_the_way_is_reached_before_a_farther_call() {
        let mut lift = Lift::new(6, 0);
        lift.call(3); // rider aboard, bound for 3
        lift.step(); // Closed(0)
        lift.step(); // Passed(1)
        lift.call(5); // a caller up on 5, mid-journey
        let mut arrivals = Vec::new();
        while lift.is_moving() {
            for m in lift.step() {
                if let Moment::Arrived(f) = m {
                    arrivals.push(f);
                }
            }
        }
        assert_eq!(
            arrivals,
            vec![3, 5],
            "the car opened at 3 before going on to 5"
        );
    }

    /// Re-aimed to the floor it is passing through, the car stops there — a
    /// change of mind that happens to name the floor under its feet arrives at
    /// once rather than overshooting.
    #[test]
    fn re_aiming_to_the_current_floor_arrives_there() {
        let mut lift = Lift::new(6, 0);
        lift.call(5);
        lift.step(); // Closed(0)
        lift.step(); // Passed(1)
        assert_eq!(lift.floor(), 1);
        lift.call(1); // to where it now is
        assert_eq!(lift.step(), vec![Moment::Arrived(1)]);
        assert!(lift.boardable_at(1));
    }

    /// A whole journey settles in a bounded number of steps and leaves the car
    /// idle and open at the destination — the property the world relies on to
    /// know a ride is over.
    #[test]
    fn a_journey_settles_open_at_the_destination() {
        for (from, to) in [(0usize, 5usize), (5, 0), (2, 3), (3, 2)] {
            let mut lift = Lift::new(6, from);
            lift.call(to);
            let mut steps = 0;
            while lift.is_moving() {
                lift.step();
                steps += 1;
                assert!(steps < 100, "a {from}->{to} ride never settled");
            }
            assert_eq!(lift.floor(), to);
            assert_eq!(lift.doors(), Doors::Open);
            assert!(lift.boardable_at(to));
        }
    }
}

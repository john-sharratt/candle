//! Tests that the lift serves every level and carries each rider to the floor
//! it chose.
//!
//! A rider is set down on its own floor even when a rider for a farther floor
//! presses last. Asking to ride while the car is elsewhere calls it and waits,
//! and walking off the landing cancels the ride. A body on a landing hears the
//! car arrive.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_bodies::tests::vault;

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

    /// **Asking to ride with the car elsewhere calls it and waits.** The rider
    /// is riding from the moment it asks — offered nothing else to do with the
    /// lift — boards when the car opens on its landing, and is set down where it
    /// chose.
    #[test]
    fn asking_to_ride_with_the_car_away_calls_it_and_boards_when_it_comes() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        let (from, dest) = (0usize, shaft.len() - 1);
        // Send the car away from the rider's floor first.
        w.call_lift(dest);
        for _ in 0..50 {
            if w.lift().unwrap().boardable_at(dest) {
                break;
            }
            w.tick();
        }
        assert!(!w.lift().unwrap().boardable_at(from));
        w.enter("m1", "Maker-01", shaft[from].clone()).unwrap();
        w.mark_seen("m1");

        assert!(w.ride_lift("m1", dest), "asking to ride was refused");
        assert!(w.waiting_for_lift("m1"));
        assert_eq!(w.riding("m1"), Some(dest));
        assert_eq!(w.lift_within("m1"), (false, false, Vec::new()));
        assert!(!w.ride_lift("m1", from), "riding to the floor it is on");

        for _ in 0..120 {
            if w.actor("m1").unwrap().at == shaft[dest] {
                break;
            }
            w.tick();
        }
        assert_eq!(w.actor("m1").unwrap().at, shaft[dest], "never carried");
        assert_eq!(w.riding("m1"), None);
        assert!(!w.waiting_for_lift("m1"));
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
}

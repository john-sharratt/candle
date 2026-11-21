//! Tests the entry and exit of bodies, the holds they carry, and the walks and
//! rides that follow from them.
//!
//! These pin what each act must leave behind: a hold is released when its body
//! leaves, the room perceives the departure, and a walk to another floor rides
//! the lift and arrives. Two travellers meeting at a landing ride on together,
//! and arriving where someone already stands is joining them, not a collision.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
                .expect("the vault must load"),
        )
    }

    pub(in super::super) fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    #[test]
    fn a_body_enters_where_it_is_told_to() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        assert_eq!(w.actor("m1").unwrap().at, at("core"));
    }

    // ── leaving ─────────────────────────────────────────────────────────────

    #[test]
    fn a_body_that_leaves_is_gone_from_the_room_and_the_world() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.enter("op", "Johnathan", at("core")).unwrap();
        assert_eq!(w.actors_at(&at("core")).len(), 2);

        w.leave("op").unwrap();
        assert!(w.actor("op").is_none());
        assert_eq!(w.actors_at(&at("core")).len(), 1);
    }

    /// **The hold goes with them.** A body leaving while it holds a station
    /// would leave that station claimed by an actor that no longer exists — and
    /// nothing else can release it, because every act that would is keyed on
    /// the actor. That is a terminal nobody can ever sit at again.
    #[test]
    fn leaving_gives_up_whatever_was_being_held() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.take("m1", Some("cindy")).unwrap();
        assert!(w.holder_of("cindy").is_some());

        w.leave("m1").unwrap();
        assert!(
            w.holder_of("cindy").is_none(),
            "a station stayed claimed by somebody who had gone"
        );
    }

    /// Leaving twice is reported rather than silently fine: a caller that
    /// thinks it removed somebody twice has lost track of which session it is
    /// ending.
    #[test]
    fn leaving_a_body_that_is_not_here_is_refused() {
        let mut w = vault();
        assert!(matches!(w.leave("nobody"), Err(Refused::NoSuchActor(_))));
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.leave("m1").unwrap();
        assert!(matches!(w.leave("m1"), Err(Refused::NoSuchActor(_))));
    }

    /// The room is told, the way it is told about an arrival — somebody
    /// vanishing unannounced is the same defect as somebody appearing that way.
    #[test]
    fn the_room_perceives_a_departure() {
        let mut w = vault();
        w.enter("watcher", "Maker-01", at("core")).unwrap();
        w.enter("goer", "Johnathan", at("core")).unwrap();
        // Spend what is already waiting, so what is left is only the leaving.
        let mut eyes = crate::delta::Attention::new();
        eyes.take(&mut w, "watcher");
        w.leave("goer").unwrap();

        let seen = eyes.take(&mut w, "watcher");
        let left = seen
            .events
            .iter()
            .find(|e| e.what == Happening::Left)
            .unwrap_or_else(|| panic!("the room was not told: {seen:?}"));
        // **Named, not identified.** The body is gone by the time anybody reads
        // this, so a name looked up in the actor list falls back to the id and
        // the room is told `visitor:u_8812 left` instead of who went.
        assert_eq!(left.name, "Johnathan");
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

    /// **A walk to another floor rides the lift and walks on.** One `set_off`:
    /// out to this floor's landing, waiting there for the car, carried, set down
    /// on the far landing, and on into the room — arriving, not lost on the way.
    #[test]
    fn a_walk_to_another_floor_rides_the_lift_and_walks_on() {
        let mut w = vault();
        let room = Where::new("vault-command", "command-room");
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.set_off("m1", room.clone()).unwrap();

        // The first leg ends at this floor's landing, and the next asks to ride.
        w.tick();
        assert_eq!(w.actor("m1").unwrap().at, at("core"), "not at the lift");
        w.tick();
        let command = w.floor_of(&room).expect("the command level is served");
        assert_eq!(w.riding("m1"), Some(command), "never asked to ride");

        // The body never crosses the shaft on foot: it is on this landing until
        // the car sets it down on the far one.
        let mut rode = 0;
        while w.riding("m1").is_some() && rode < 200 {
            assert_eq!(w.actor("m1").unwrap().at, at("core"), "left before the car");
            w.tick();
            rode += 1;
        }
        assert!(rode > 0, "the ride took no time");
        w.settle();
        assert_eq!(w.actor("m1").unwrap().at, room, "never got to the room");
        assert!(w.actor("m1").unwrap().walk.is_none());
        assert!(
            w.log().iter().any(|e| e.actor == "m1"
                && matches!(&e.what, Happening::GotThere { toward } if *toward == room)),
            "the journey was not answered as arrived"
        );
        assert!(
            !w.log()
                .iter()
                .any(|e| e.actor == "m1" && matches!(&e.what, Happening::LostTheWay { .. })),
            "the ride lost the way"
        );
    }

    /// **Two bodies meeting at the lift ride on together.** A landing is where a
    /// queue for the car forms, and a walk that passes through one is not ended
    /// by the people already standing there.
    #[test]
    fn two_travellers_meeting_at_the_lift_ride_on_together() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("band-one")).unwrap();
        w.enter("m2", "Maker-02", at("green-room")).unwrap();
        let elsewhere = Where::new("vault-chronicle", "core");
        w.set_off("m1", elsewhere.clone()).unwrap();
        w.set_off("m2", elsewhere.clone()).unwrap();

        w.tick();
        assert_eq!(w.actor("m1").unwrap().at, at("core"), "m1 not at the lift");
        assert_eq!(w.actor("m2").unwrap().at, at("core"), "m2 not at the lift");
        assert!(
            w.actor("m1").unwrap().walk.is_some(),
            "m1 stopped at the lift"
        );
        assert!(
            w.actor("m2").unwrap().walk.is_some(),
            "m2 stopped at the lift"
        );

        w.settle();
        assert_eq!(w.actor("m1").unwrap().at, elsewhere);
        assert_eq!(w.actor("m2").unwrap().at, elsewhere);
        assert!(
            !w.log()
                .iter()
                .any(|e| matches!(&e.what, Happening::Bumped { .. })),
            "a queue at the lift was taken for a collision"
        );
    }

    /// **Two bodies swapping levels each reach the other's.** The shaft is the
    /// car's, not a corridor they can meet in, so neither is stopped by the
    /// other on the way.
    #[test]
    fn two_travellers_swapping_levels_each_reach_the_other_s() {
        let mut w = vault();
        let command = Where::new("vault-command", "core");
        let chronicle = Where::new("vault-chronicle", "core");
        w.enter("m1", "Maker-01", command.clone()).unwrap();
        w.enter("m2", "Maker-02", chronicle.clone()).unwrap();
        w.set_off("m1", chronicle.clone()).unwrap();
        w.set_off("m2", command.clone()).unwrap();

        w.settle();
        assert_eq!(w.actor("m1").unwrap().at, chronicle);
        assert_eq!(w.actor("m2").unwrap().at, command);
    }

    /// **Giving up the way gives up its ride.** A body waiting at the landing
    /// for a journey's car that sets off somewhere on its own floor instead is
    /// not carried off when the car comes.
    #[test]
    fn setting_off_elsewhere_from_the_landing_gives_up_the_journey_s_ride() {
        let mut w = vault();
        let shaft: Vec<Where> = w.shaft().to_vec();
        let casting = w.floor_of(&at("core")).expect("casting is served");
        let far = (0..shaft.len())
            .find(|f| *f != casting)
            .expect("two floors");
        // The car away, so the rider must wait for it.
        w.call_lift(far);
        for _ in 0..100 {
            if w.lift().unwrap().boardable_at(far) {
                break;
            }
            w.tick();
        }
        w.enter("m1", "Maker-01", at("core")).unwrap();
        w.set_off("m1", shaft[far].clone()).unwrap();
        w.tick();
        assert_eq!(w.riding("m1"), Some(far), "never asked to ride");

        w.set_off("m1", at("green-room")).unwrap();
        assert_eq!(w.riding("m1"), None, "the ride outlived the journey");
        w.settle();
        assert_eq!(w.actor("m1").unwrap().at, at("green-room"));
    }

    /// Arriving where somebody is standing is joining them, not a collision — you
    /// walked there on purpose. Only a body still on its way bumps.
    #[test]
    fn arriving_where_somebody_stands_is_not_a_bump() {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        // m2 walks to green-room, where m1 is — a same-level journey it finishes
        // in one leg, so it arrives rather than crossing anybody mid-way.
        w.set_off("m2", at("green-room")).unwrap();
        w.settle();
        assert_eq!(w.actor("m2").unwrap().at, at("green-room"));
        assert!(
            !w.log()
                .iter()
                .any(|e| matches!(&e.what, Happening::Bumped { .. })),
            "arriving at a room where somebody stands read as a collision"
        );
    }
}

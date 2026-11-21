//! Tests of the scan act: who it names, where it resolves a place, and what an empty room or a survey says.
//!
//! A scan names who else is on the floor and where. It resolves a place the grammar offered even after the
//! body has walked away from it. It points an empty floor at the handset only for a character that carries
//! one. A survey of a room lists the stations with the tower's state, the character's own conversations, and
//! the machines that have no address, which are worked with `operate`.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use npc_map::world::Where;

    use serde_json::json;

    use super::super::tests_acts::tests::{act, waste};

    /// The tower's one floor, with each `(id, name, node)` standing on it.
    pub(in super::super) fn tower_with(bodies: &[(&str, &str, &str)]) -> Hosted {
        let h = Hosted::load(
            "battle-cities",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped maps must load");
        h.with(|w| {
            for (id, name, node) in bodies {
                w.enter(*id, *name, Where::new("tower-redoubt", *node))
                    .unwrap();
            }
        });
        h
    }

    /// **A scan names who else is on the floor, and where.** The room looked at
    /// was empty; the person the character wanted was two doors down.
    #[test]
    fn a_scan_names_who_else_is_on_the_floor() {
        let h = tower_with(&[
            ("c1", "Wren Weaver", "muster-hall"),
            ("c2", "Yaelis Vayne", "barracks"),
        ]);
        let out = perform(
            &h,
            "c1",
            &act("scan", json!({"at": "tower-redoubt/gatehouse"})),
        );
        let line = out.line().unwrap();
        assert!(line.contains("Nobody there"), "{line}");
        assert!(
            line.contains("Elsewhere on this floor: Yaelis Vayne is in the barracks."),
            "{line}"
        );
        assert!(
            !line.contains("message them"),
            "somebody is on the floor: {line}"
        );
    }

    /// **A place the grammar offered is a place the scan resolves, however far
    /// the body has walked during the decode.** The character read the rooms
    /// off the muster hall, chose to look at the rampart, and was a leg onto it
    /// by the time the act landed — where "the rampart" is the room it stands
    /// in and so is no longer in the list of places to look at. It was refused
    /// "You cannot see the rampart from here" for naming exactly what it was
    /// offered.
    #[test]
    fn a_scan_resolves_a_place_offered_before_the_body_walked_there() {
        let h = tower_with(&[("c1", "Wren Weaver", "muster-hall")]);
        h.with(|w| w.begin_decision("c1"));
        let offered = crate::engine::body::reachable(&h, "c1");
        assert!(
            offered.iter().any(|n| n == "the rampart"),
            "the rampart was not offered: {offered:?}"
        );
        h.with(|w| {
            w.set_off("c1", Where::new("tower-redoubt", "rampart"))
                .unwrap();
            w.settle();
        });
        assert_eq!(
            crate::engine::body::reachable(&h, "c1"),
            offered,
            "the places on offer moved with the body during the decode"
        );

        let out = perform(&h, "c1", &act("scan", json!({"at": "the rampart"})));
        assert!(
            out.line().unwrap().starts_with("You look at the rampart."),
            "{out:?}"
        );

        h.with(|w| w.end_decision("c1"));
        let after = crate::engine::body::reachable(&h, "c1");
        assert!(
            !after.iter().any(|n| n == "the rampart"),
            "once decided, the body's own room is not a place to look at: {after:?}"
        );
        assert!(
            after.iter().any(|n| n == "the muster hall"),
            "once decided, the room it left is: {after:?}"
        );
    }

    /// With nobody else on the floor, the handset is where to look.
    #[test]
    fn a_scan_of_an_empty_floor_points_at_the_handset() {
        let h = tower_with(&[("c1", "Wren Weaver", "muster-hall")]);
        h.with_sim(|s| crate::sim::seed::issue_handset(s, "c1", "Wren Weaver"));
        let out = perform(
            &h,
            "c1",
            &act("scan", json!({"at": "tower-redoubt/gatehouse"})),
        );
        assert!(
            out.line()
                .unwrap()
                .contains("If you are looking for somebody, message them"),
            "{out:?}"
        );
    }

    /// Without a handset it does not point at something the character cannot
    /// use — the same gate that keeps the messaging acts out of its grammar.
    #[test]
    fn without_a_handset_an_empty_floor_says_nothing_about_messaging() {
        let h = tower_with(&[("c1", "Wren Weaver", "muster-hall")]);
        let out = perform(
            &h,
            "c1",
            &act("scan", json!({"at": "tower-redoubt/gatehouse"})),
        );
        assert!(!out.line().unwrap().contains("message them"), "{out:?}");
    }

    #[test]
    fn a_coordinate_off_the_edge_of_the_world_is_refused_and_says_where_the_edge_is() {
        let h = waste();
        let out = perform(&h, "c1", &act("scan", json!({"x":"999999","y":"0"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("off the map"), "{out:?}");
    }

    /// **A scan that names nothing surveys the room**: one line per station, with
    /// the exact address to `invoke` and the tower's own state in a few words.
    #[test]
    fn a_scan_naming_nothing_surveys_the_stations_in_the_room() {
        let h = tower_with(&[("c1", "Wren Weaver", "bridge")]);
        let out = perform(&h, "c1", &act("scan", json!({})));
        assert!(out.happened(), "{out:?}");
        let line = out.line().unwrap();
        assert!(line.starts_with("Here you can work:\n- "), "{line}");
        assert!(line.contains("Bridge console:"), "{line}");
        assert!(line.contains("shields down"), "{line}");
        assert!(
            line.contains("/command_tower` (needs action)"),
            "the survey did not name the invoke address: {line}"
        );
        assert!(line.contains("`invoke` its address"), "{line}");
    }

    /// **The survey carries what the character holds as well as what the room
    /// offers**: its conversations, whatever the room.
    #[test]
    fn a_survey_lists_the_characters_conversations() {
        let h = waste();
        h.with_sim(|s| {
            crate::sim::seed::issue_handset(s, "c1", "Wren");
            let g = s.threads.open_group(
                "the crew",
                &[Who::new("c1", "Wren"), Who::new("c2", "Soren")],
            );
            s.threads
                .send_on(&g, "c2", Said::words("how much ore is left?"))
                .unwrap();
        });
        let out = perform(&h, "c1", &act("scan", json!({})));
        let line = out.line().unwrap();
        assert!(line.contains("On your phone:\n"), "{line}");
        assert!(
            line.contains("- the crew (a group of 2): 1 unread.\n"),
            "{line}"
        );
        assert!(
            !line.contains("ore is left"),
            "the content stays in `read`: {line}"
        );
    }

    /// **A character with no handset is not told about one.**
    #[test]
    fn a_survey_leaves_out_a_phone_the_character_does_not_carry() {
        let h = tower_with(&[("c1", "Wren Weaver", "rampart")]);
        h.with_sim(|s| {
            s.threads.open_group(
                "the crew",
                &[Who::new("c1", "Wren Weaver"), Who::new("c2", "Soren")],
            );
        });
        let out = perform(&h, "c1", &act("scan", json!({})));
        assert!(!out.line().unwrap().contains("On your phone"), "{out:?}");
    }

    /// **A room with nothing to work says so rather than listing nothing.**
    #[test]
    fn a_survey_of_an_empty_room_points_elsewhere() {
        let h = tower_with(&[("c1", "Wren Weaver", "barracks")]);
        let out = perform(&h, "c1", &act("scan", json!({})));
        assert_eq!(
            out.line().unwrap(),
            "Nothing in this room answers to you. To see somewhere else, name a place."
        );
    }

    /// **A machine with no address is still read by a scan.** The rampart's
    /// turrets afford no `invoke`; they are worked with `operate`, and a scan
    /// shows each in its state with the states it can be put to — the reading a
    /// mission to "read the turret" asks for.
    #[test]
    fn a_survey_reads_the_machines_that_have_no_address() {
        let h = tower_with(&[("c1", "Wren Weaver", "rampart")]);
        let out = perform(&h, "c1", &act("scan", json!({})));
        let line = out.line().unwrap();
        assert!(
            line.starts_with("Here you can work:\n- Wall turret 1: "),
            "{line}"
        );
        assert_eq!(line.matches("with `operate`.").count(), 4, "{line}");
    }
}

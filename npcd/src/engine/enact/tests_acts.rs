//! Tests that each act in the catalogue reaches the world it names, and that giving, gathering and
//! using carry the state the world records.
//!
//! The catalogue test checks that every act this module names is dispatched by something in the engine.
//! The acts performed on other paths are named explicitly, so a new one has to be argued for. The other
//! tests check the effect of each act on the world: a giving moves the thing and leaves the giver short,
//! a stimpak is spent while a scanner is not, a stance stands until it is broken off, and a verdict
//! attaches to the thing judged.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::engine::act::Act;

    use crate::engine::talk::hand_off::HAND_OFF_ACTS;

    use crate::engine::{beyond, zen_task};

    use npc_map::world::Where;

    use serde_json::json;

    pub(in super::super) fn act(tool: &'static str, args: Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    /// A world with two bodies in the ruins, kitted out.
    pub(in super::super) fn waste() -> Hosted {
        let h = Hosted::load(
            "battle-cities",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped maps must load");
        let at = Where::new("the-waste", "ruins");
        h.with(|w| {
            w.enter("c1", "Wren", at.clone()).unwrap();
            w.enter("c2", "Soren", at).unwrap();
        });
        h.with_sim(|s| {
            crate::sim::seed::outfit(s, "c1");
            crate::sim::seed::outfit(s, "c2");
        });
        h
    }

    #[test]
    fn every_act_in_the_catalog_is_dispatched_by_somebody() {
        // The other half of `every_act_offered_is_one_that_does_something`: that
        // test asks whether an offered act reaches a world, this one asks
        // whether anything in this module names an act that does not exist.
        for name in crate::engine::tools::CATALOG.iter().map(|t| t.name) {
            // `send_image` is the one act performed above this layer: it
            // renders through the image guest and lands in an interaction, so
            // no body reaches a world for it. Named here rather than filtered
            // by category, so a second such act has to be argued for.
            if name == "send_image" {
                continue;
            }
            // The effector device's two verbs reach the world through the
            // effector router on the async fast path (`Runtime::enact_device`),
            // not through any synchronous body/enact dispatch — so this layer
            // deliberately does not perform them.
            if crate::engine::body::is_device(name) {
                continue;
            }
            // The house's acts likewise run on the async path
            // (`Runtime::bridge_act`, `Runtime::build_act`).
            if crate::engine::body::is_of_the_house(name) {
                continue;
            }
            // A mission's task acts call zend, also on the async path
            // (`Runtime::zen_task_act`).
            if zen_task::is_mine(name) {
                continue;
            }
            // What a personality reaches beyond the world runs on the async
            // path too (`Runtime::beyond_act`).
            if beyond::is_mine(name) {
                continue;
            }
            // A conversation's hand-off is performed as the world's `message`
            // (`talk::hand_off::as_performed`), which is dispatched here.
            if HAND_OFF_ACTS.iter().any(|t| t.name == name) {
                continue;
            }
            assert!(
                crate::engine::body::is_of_the_body(name) || is_mine(name),
                "`{name}` is in the catalog and nothing performs it"
            );
        }
    }

    #[test]
    fn giving_moves_the_thing_and_leaves_the_giver_short() {
        let h = waste();
        let before = h.sim(|s| s.pack("c1").get("bolt").unwrap().count);
        let out = perform(
            &h,
            "c1",
            &act(
                "give",
                json!({"what":"bolt rounds","to":"Soren","count":"20"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            h.sim(|s| s.pack("c1").get("bolt").unwrap().count),
            before - 20
        );
        assert_eq!(h.sim(|s| s.pack("c2").get("bolt").unwrap().count), 80);
    }

    #[test]
    fn giving_more_than_is_carried_moves_nothing() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act(
                "give",
                json!({"what":"bolt rounds","to":"Soren","count":"999"}),
            ),
        );
        assert!(!out.happened());
        assert_eq!(h.sim(|s| s.pack("c1").get("bolt").unwrap().count), 60);
    }

    #[test]
    fn a_stimpak_used_on_somebody_is_spent_and_a_scanner_is_not() {
        let h = waste();
        assert!(perform(
            &h,
            "c1",
            &act("use", json!({"what":"stimpak","on":"Soren"}))
        )
        .happened());
        assert_eq!(h.sim(|s| s.pack("c1").get("stimpak").unwrap().count), 2);

        assert!(perform(&h, "c1", &act("use", json!({"what":"advanced scanner"}))).happened());
        assert_eq!(
            h.sim(|s| s.pack("c1").get("scanner").unwrap().count),
            1,
            "gear was consumed"
        );
    }

    #[test]
    fn gathering_takes_from_the_ground_and_puts_it_in_the_pack() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act("gather", json!({"what":"the burnt-out carrier"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            h.sim(|s| s.pack("c1").get("metal").map(|i| i.count)),
            Some(10)
        );
        assert_eq!(
            h.sim(|s| s.field.deposit("burnt_carrier").unwrap().remaining),
            20
        );
    }

    #[test]
    fn gathering_something_that_is_not_here_takes_nothing() {
        let h = waste();
        assert!(!perform(&h, "c1", &act("gather", json!({"what":"the ore seam"}))).happened());
        assert!(h.sim(|s| s.pack("c1").get("ore").is_none()));
    }

    #[test]
    fn a_stance_stands_until_it_is_broken_off() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act(
                "engage",
                json!({"posture":"press","target":"a mech","priority":"nearest"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            h.sim(|s| s.field.stance("c1").unwrap().posture.clone()),
            "press"
        );

        assert!(perform(&h, "c1", &act("engage", json!({"posture":"break off"}))).happened());
        assert!(
            h.sim(|s| s.field.stance("c1").is_none()),
            "breaking off left a stance"
        );
    }

    #[test]
    fn engaging_something_that_is_not_here_is_refused() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act("engage", json!({"posture":"press","target":"a stinger"})),
        );
        assert!(!out.happened());
        assert!(h.sim(|s| s.field.stance("c1").is_none()));
    }

    #[test]
    fn a_door_takes_its_own_states_and_not_a_turrets() {
        let h = waste();
        h.with(|w| {
            w.place("c1", Where::new("tower-redoubt", "gatehouse"))
                .unwrap()
        });
        let out = perform(
            &h,
            "c1",
            &act("operate", json!({"what":"the blast door","mode":"locked"})),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(
            h.sim(|s| s
                .devices
                .by_name_at("tower-redoubt/gatehouse", "the blast door")
                .unwrap()
                .mode
                .clone()),
            "locked"
        );

        let bad = perform(
            &h,
            "c1",
            &act(
                "operate",
                json!({"what":"the blast door","mode":"air only"}),
            ),
        );
        assert!(!bad.happened());
        assert!(
            bad.line().unwrap().contains("open"),
            "the refusal should name what it does take: {bad:?}"
        );
    }

    #[test]
    fn sleeping_is_recorded_so_something_can_wake_you() {
        let h = waste();
        assert!(perform(&h, "c1", &act("sleep", json!({"until":"dawn"}))).happened());
        assert_eq!(
            h.sim(|s| s.ledger.asleep("c1").map(str::to_string)),
            Some("dawn".into())
        );
    }

    #[test]
    fn a_verdict_attaches_to_the_thing_judged() {
        let h = waste();
        let out = perform(
            &h,
            "c1",
            &act(
                "record_verdict",
                json!({"on":"the ridge survey","judgement":"it will not do","what_would_change_it":"the count checked"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert_eq!(h.sim(|s| s.ledger.verdicts_on("the ridge survey").len()), 1);
    }
}

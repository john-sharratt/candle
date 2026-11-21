//! Tests of the tower's commands: its costs, its decisions and its states under contact.
//!
//! The tower refuses what it cannot afford and says what it can. A drill is paid for or it does not turn.
//! Only the holder of a decision may command the tower while it stands, and an unheld decision does not stop
//! the tower being commanded. Folding the tower away leaves a contact and its decision behind, and a tower run
//! down to its reserve can dig in and recover.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::sim::upkeep::{CONTACT_EVERY, RESERVE};

    use serde_json::json;

    use std::time::Duration;

    use super::super::tests_acts::tests::{act, waste};

    #[test]
    fn the_tower_refuses_what_it_cannot_afford_and_names_what_it_can() {
        let h = waste();
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            t.draw(Resource::Energy, t.stock_of(Resource::Energy));
        });
        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"relocate","x":10,"y":10})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("cannot relocate"), "{out:?}");
    }

    #[test]
    fn folding_the_tower_moves_it_and_spends_the_energy() {
        let h = waste();
        let before = h.sim(|s| s.tower.as_ref().unwrap().stock_of(Resource::Energy));
        let out = perform(
            &h,
            "c1",
            &act(
                "command_tower",
                json!({"action":"relocate","x":-300,"y":180}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let t = s.tower.as_ref().unwrap();
            assert_eq!((t.at.x, t.at.y), (-300, 180));
            assert!(t.stock_of(Resource::Energy) < before);
        });
    }

    /// The tower with a contact sighted and its decision on the board.
    pub(in super::super) fn under_contact() -> Hosted {
        let h = waste();
        h.with_sim(|s| {
            s.tower.as_mut().unwrap().advance(CONTACT_EVERY);
            s.watch_tower();
        });
        h
    }

    pub(in super::super) fn the_decision(h: &Hosted) -> String {
        h.sim(|s| s.tower_decisions().remove(0).what)
    }

    /// **Somebody holding the decision is the tower's one voice**: a bystander
    /// is told who, by name, and the tower is left as it was.
    #[test]
    fn the_tower_is_not_commanded_over_whoever_is_deciding_it() {
        let h = under_contact();
        let what = the_decision(&h);
        h.with_sim(|s| s.ledger.take_order(&what, "c1").unwrap());

        let out = perform(
            &h,
            "c2",
            &act("command_tower", json!({"action":"raise shields"})),
        );

        assert!(!out.happened(), "{out:?}");
        let line = out.line().unwrap();
        assert!(line.contains("Wren"), "the holder was not named: {line}");
        assert!(
            line.contains("contact 1"),
            "the decision was not named: {line}"
        );
        assert!(!h.sim(|s| s.tower.as_ref().unwrap().shields));
    }

    /// **Nobody is in the way until somebody has taken it.**
    #[test]
    fn an_unheld_decision_does_not_stop_the_tower_being_commanded() {
        let h = under_contact();
        let out = perform(
            &h,
            "c2",
            &act("command_tower", json!({"action":"raise shields"})),
        );
        assert!(out.happened(), "{out:?}");
        assert!(h.sim(|s| s.tower.as_ref().unwrap().shields));
    }

    /// **The holder commands the tower, and that settles the decision**, so it
    /// is not put to the crew again while the contact stands.
    #[test]
    fn the_holder_commands_the_tower_and_settles_the_decision() {
        let h = under_contact();
        let what = the_decision(&h);
        h.with_sim(|s| s.ledger.take_order(&what, "c1").unwrap());

        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"raise shields"})),
        );

        assert!(out.happened(), "{out:?}");
        h.with_sim(|s| {
            assert!(s.tower_decisions().is_empty(), "the decision stayed open");
            assert!(
                s.watch_tower().is_empty(),
                "a settled decision was raised again"
            );
        });
        let free = perform(
            &h,
            "c2",
            &act("command_tower", json!({"action":"drop shields"})),
        );
        assert!(
            free.happened(),
            "the tower stayed locked after it was decided: {free:?}"
        );
    }

    /// **Folding away is the answer to a contact**: what was closing on the
    /// tower is left behind, and so is the decision about it.
    #[test]
    fn folding_the_tower_leaves_the_contact_behind() {
        let h = under_contact();
        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"relocate","x":10,"y":10})),
        );
        assert!(out.happened(), "{out:?}");
        h.with_sim(|s| {
            assert!(s.tower.as_ref().unwrap().contact.is_none());
            s.watch_tower();
            assert!(s.tower_decisions().is_empty());
        });
    }

    /// **Shields are refused on a stockpile that cannot hold them**, not raised
    /// to fail on the next beat.
    #[test]
    fn shields_are_refused_when_there_is_no_energy_to_hold_them() {
        let h = waste();
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            t.draw(Resource::Energy, t.stock_of(Resource::Energy));
        });
        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"raise shields"})),
        );
        assert!(!out.happened(), "{out:?}");
        assert!(!h.sim(|s| s.tower.as_ref().unwrap().shields));
    }

    /// **A drill is paid for or it does not turn**: a tower that cannot afford the
    /// depth is told how deep it can go, and is not dug in for nothing.
    #[test]
    fn a_drill_the_stockpile_cannot_pay_for_is_refused_and_says_how_deep_it_can_go() {
        let h = waste();
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            t.draw(Resource::Energy, t.stock_of(Resource::Energy) - 50);
        });
        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"drill down","depth":100})),
        );
        assert!(!out.happened(), "{out:?}");
        assert!(out.line().unwrap().contains("25 metres"), "{out:?}");
        h.sim(|s| {
            let t = s.tower.as_ref().unwrap();
            assert_eq!(t.depth, 0);
            assert_eq!(t.stock_of(Resource::Energy), 50);
        });
    }

    /// **A tower run down to its reserve can dig in, and the ground brings it
    /// back.**
    #[test]
    fn a_run_down_tower_digs_in_and_recovers() {
        let h = waste();
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            t.draw(Resource::Energy, t.stock_of(Resource::Energy) - RESERVE);
        });
        let out = perform(
            &h,
            "c1",
            &act("command_tower", json!({"action":"drill down","depth":50})),
        );
        assert!(out.happened(), "{out:?}");
        h.with_sim(|s| {
            let t = s.tower.as_mut().unwrap();
            let low = t.stock_of(Resource::Energy);
            t.advance(Duration::from_secs(60 * 30));
            assert!(
                t.stock_of(Resource::Energy) > low,
                "the ground gave nothing"
            );
        });
    }

    #[test]
    fn a_batch_the_stockpile_cannot_cover_is_refused_and_says_what_is_short() {
        let h = waste();
        let out = perform(&h, "c1", &act("produce", json!({"what":"a companion"})));
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("nanobots"), "{out:?}");
    }
}

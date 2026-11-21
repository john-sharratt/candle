//! Tests for the table's chapters, collection, draws and reports.
//!
//! They check that each mission closed turns a chapter and a routine turns none,
//! that a body made unordered is stood down and told, and that lodged missions are
//! drawn oldest first. They also check that clearing the table keeps what an
//! operator lodged, that starting over forgets what was finished, and that a read
//! which is gone is struck from every place a mission waits.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::Missions;
    use crate::engine::mission::bank::Facts;
    use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome, Todo};
    use crate::engine::mission_gen::world::{Draw, Routine};

    fn a_mission(prompt: &str) -> Mission {
        Mission::new(
            prompt,
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        )
    }

    /// **Each mission closed — reported or called off — turns a chapter**, and
    /// the mission closed last is kept to close it with.
    #[test]
    fn each_mission_closed_turns_a_chapter() {
        let mut m = Missions::default();
        assert_eq!(m.chapter("bram"), 0);
        m.assign("bram", a_mission("first"));
        m.report("bram", Outcome::Pass, "done", None);
        assert_eq!(m.chapter("bram"), 1);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("first")
        );
        m.assign("bram", a_mission("second"));
        assert!(m.cancel("bram"));
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("second"),
            "a mission called off closes a chapter too"
        );
        assert!(!m.cancel("bram"), "nothing open, nothing closed");
        assert_eq!(m.chapter("bram"), 2);
        assert_eq!(m.chapter("cindy"), 0);
    }

    /// **A world with routines of its own hands them out in place of the
    /// bank's**, closed by the report at its table; with none, the bank again.
    #[test]
    fn a_world_s_own_routines_replace_the_bank() {
        let routines = || {
            vec![Routine {
                id: "ward-shift".to_string(),
                brief: "Take a shift on the ward.".to_string(),
                steps: vec!["go to the ward room".into()],
                ..Default::default()
            }]
        };
        let facts = Facts {
            table: Some("the watch room"),
            ..Facts::default()
        };
        let mut m = Missions::default();
        m.set_routines(routines(), Draw::Routines);
        let got = m.collect("kettle", &facts).unwrap().clone();
        assert_eq!(got.prompt, "Take a shift on the ward.");
        assert_eq!(
            got.origin,
            Origin::Random {
                routine: "ward-shift".to_string()
            }
        );
        let steps: Vec<&str> = got.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the ward room",
                "go back to the table in the watch room and report it"
            ]
        );

        let mut adopted = Missions::default();
        adopted.adopt_workflows(&mut m);
        let again = adopted.collect("kettle", &facts).unwrap().clone();
        assert_eq!(again.prompt, got.prompt, "adopted with the workflows");

        adopted.set_routines(Vec::new(), Draw::Routines);
        let bank = adopted.collect("biscuit", &facts).unwrap().clone();
        assert!(
            matches!(&bank.origin, Origin::Random { routine } if routine != "ward-shift"),
            "{:?}",
            bank.origin
        );
    }

    /// **A routine handed in turns no chapter**, and a chapter closes with the
    /// work that turned it, not with a routine handed in after it.
    #[test]
    fn a_routine_closed_turns_no_chapter() {
        let routine = || {
            Mission::new(
                "read the breaker panel",
                vec![Todo::new("go to the plant room")],
                Origin::Random {
                    routine: "read-a-station".to_string(),
                },
            )
        };
        let mut m = Missions::default();
        m.assign("bram", routine());
        m.report("bram", Outcome::Pass, "it reads tight", None);
        assert_eq!(m.chapter("bram"), 0);
        assert!(m.last_closed("bram").is_none());

        m.assign("bram", a_mission("the work"));
        m.report("bram", Outcome::Pass, "done", None);
        m.assign("bram", routine());
        m.report("bram", Outcome::Pass, "it reads tight", None);
        assert_eq!(m.chapter("bram"), 1);
        assert_eq!(
            m.last_closed("bram").map(|x| x.prompt.as_str()),
            Some("the work")
        );
    }

    /// **A character made unordered is stood down from what it carries, and
    /// told**; what was lodged for it goes, and marking it again changes
    /// nothing.
    #[test]
    fn an_unordered_character_carries_nothing_and_is_told_so() {
        let mut m = Missions::default();
        m.assign("zen", a_mission("visit the hatchery"));
        m.lodge("zen", a_mission("then the ward"));
        m.assign("wick", a_mission("read a cradle"));
        m.set_unordered("zen", true);
        assert!(m.is_unordered("zen"));
        assert!(!m.is_on_mission("zen"));
        assert!(!m.has_lodged("zen"));
        assert_eq!(m.take_stood_down(), ["zen"]);
        assert!(!m.is_unordered("wick"));
        assert!(m.is_on_mission("wick"), "nobody else is touched");

        m.set_unordered("zen", true);
        assert!(m.take_stood_down().is_empty(), "nothing left to stand down");
        m.set_unordered("zen", false);
        assert!(!m.is_unordered("zen"));
    }

    #[test]
    fn a_lodged_mission_is_drawn_oldest_first_and_then_the_queue_empties() {
        let mut m = Missions::default();
        assert!(!m.has_lodged("bram"));
        m.lodge("bram", a_mission("first"));
        m.lodge("bram", a_mission("second"));
        assert!(m.has_lodged("bram"));

        assert_eq!(m.take_lodged("bram").unwrap().prompt, "first");
        assert_eq!(m.take_lodged("bram").unwrap().prompt, "second");
        assert!(m.take_lodged("bram").is_none(), "queue empties");
        assert!(!m.has_lodged("bram"));
    }

    /// **Clearing the table clears what was posted by name too**, and leaves
    /// what an operator lodged.
    #[test]
    fn discarding_the_table_takes_posts_set_by_name_and_keeps_the_operators() {
        let mut m = Missions::default();
        let posted = |brief: &str| {
            Mission::new(
                brief,
                vec![Todo::new("go to the ward room")],
                Origin::Ordered {
                    by: "Zen".into(),
                    poster: "zen".into(),
                    job: None,
                },
            )
        };
        m.lodge("jitter", posted("Watch the ward."));
        m.lodge("jitter", posted("Check the heat."));
        m.lodge("jitter", a_mission("the operator's"));
        m.lodge("wick", posted("Read the board."));
        assert_eq!(m.discard_pool(), 3);
        assert!(!m.has_lodged("wick"), "an emptied queue goes");
        assert_eq!(m.take_lodged("jitter").unwrap().prompt, "the operator's");
        assert!(m.take_lodged("jitter").is_none());
    }

    #[test]
    fn assigning_a_mission_makes_it_the_active_one() {
        let mut m = Missions::default();
        assert!(!m.is_on_mission("wren"));
        m.assign("wren", a_mission("survey the ridge"));
        assert!(m.is_on_mission("wren"));
        assert_eq!(m.active("wren").unwrap().prompt, "survey the ridge");
        // A second assignment replaces the first — a body carries one at a time.
        m.assign("wren", a_mission("read the canon"));
        assert_eq!(m.active("wren").unwrap().prompt, "read the canon");
    }

    #[test]
    fn steps_are_checked_off_and_added_only_on_an_open_mission() {
        let mut m = Missions::default();
        // No mission: nothing to act on.
        assert!(!m.check_off("pax", "step one", StepOutcome::Achieved));
        assert!(!m.add_todo("pax", "a new step"));

        m.assign("pax", a_mission("do the thing"));
        assert!(
            m.check_off("pax", "STEP one", StepOutcome::Achieved),
            "trim + case-insensitive"
        );
        assert!(
            !m.check_off("pax", "step one", StepOutcome::Thwarted),
            "already done"
        );
        assert!(m.add_todo("pax", "a discovered step"));
        assert!(
            !m.add_todo("pax", "  a discovered step  "),
            "open duplicate"
        );
        assert_eq!(m.active("pax").unwrap().todo.len(), 3);
    }

    #[test]
    fn reporting_closes_the_open_mission_and_keeps_it_for_the_operator() {
        let mut m = Missions::default();
        assert!(
            m.report("soren", Outcome::Pass, "n/a", None).is_none(),
            "nothing to report without an open mission"
        );

        m.assign("soren", a_mission("check the record"));
        let closed = m
            .report(
                "soren",
                Outcome::Pass,
                "it holds against the storyline",
                Some("the eastern date is wrong".to_string()),
            )
            .expect("an open mission is reported");
        assert!(!closed.is_open());
        assert_eq!(closed.report.as_ref().unwrap().outcome, Outcome::Pass);

        // The body is now free of its mission, but the operator can still read it.
        assert!(!m.is_on_mission("soren"));
        assert!(m.active("soren").is_none());
        let done = m.done("soren").expect("kept for the operator");
        assert_eq!(done.answer.as_deref(), Some("the eastern date is wrong"));
        assert_eq!(m.latest("soren").unwrap().prompt, "check the record");
    }

    #[test]
    fn collecting_takes_a_lodged_mission_first_then_draws_from_the_bank() {
        let mut m = Missions::default();
        let makers = vec!["Wren".to_string(), "Pax".to_string()];
        let records = vec!["the-charge".to_string()];
        let facts = Facts {
            makers: &makers,
            records: &records,
            ..Facts::default()
        };

        // A lodged mission is taken first, verbatim, and becomes active.
        m.lodge("bram", a_mission("the lodged one"));
        assert_eq!(m.collect("bram", &facts).unwrap().prompt, "the lodged one");
        assert!(m.is_on_mission("bram"));
        assert!(!m.has_lodged("bram"), "the lodged queue is drawn down");

        // With nothing lodged, a table that draws nothing hands out nothing —
        // the default — and one that draws routines gives one from the bank: a
        // valid, open, non-empty, non-destructive mission it can act on.
        m.cancel("bram");
        assert!(m.collect("bram", &facts).is_none(), "a routine was drawn");
        m.set_routines(Vec::new(), Draw::Routines);
        let drawn = m.collect("bram", &facts).unwrap().clone();
        assert!(drawn.is_open());
        assert!(!drawn.todo.is_empty(), "a routine gives steps to act on");
        assert!(matches!(drawn.origin, Origin::Random { .. }));
    }

    /// **One who gives orders collects nothing from the table** — not what is
    /// lodged for it, not what waits for whoever takes it, not a routine.
    #[test]
    fn one_who_gives_orders_collects_nothing_from_the_table() {
        let mut m = Missions::default();
        m.set_routines(Vec::new(), Draw::Routines);
        m.set_unordered("zen", true);
        m.lodge("zen", a_mission("lodged for zen"));
        m.post(a_mission("for whoever takes it"), None);
        assert!(m.collect("zen", &Facts::default()).is_none());
        assert!(!m.is_on_mission("zen"));
        assert!(
            m.collect("wick", &Facts::default()).is_some(),
            "one who takes orders still collects"
        );
    }

    /// **Starting over forgets what was finished**: nothing is carried, and
    /// nothing is kept as done or as the last mission closed.
    #[test]
    fn starting_over_forgets_what_was_finished() {
        let mut m = Missions::default();
        m.assign("zen", a_mission("read the reports"));
        m.report("zen", Outcome::Pass, "read", None).unwrap();
        m.assign("zen", a_mission("watch earth"));
        assert!(m.done("zen").is_some());
        m.start_over("zen");
        assert!(!m.is_on_mission("zen"));
        assert!(m.done("zen").is_none(), "a finished mission was kept");
        assert!(m.latest("zen").is_none(), "something was kept as latest");
    }

    /// A read struck wherever a mission stands — carried, lodged or pooled —
    /// and each place it was struck named.
    #[test]
    fn a_gone_read_is_struck_from_every_mission_wherever_it_waits() {
        let gone = "layers/stories/gone.md";
        let with_read = |p: &str| {
            Mission::new(
                p,
                vec![Todo::new(format!("read {gone}")), Todo::report("report")],
                Origin::Random {
                    routine: "r".into(),
                },
            )
        };
        let mut m = Missions::default();
        m.assign("vespera", with_read("carried"));
        m.lodge("sila", with_read("lodged"));
        m.pool.push(with_read("pooled"));
        m.assign("paxon", a_mission("reads nothing"));
        let struck = m.strike_gone_reads(&|_, p| p != gone);
        assert_eq!(
            struck,
            [
                (Some("vespera".to_string()), vec![gone.to_string()]),
                (Some("sila".to_string()), vec![gone.to_string()]),
                (None, vec![gone.to_string()]),
            ]
        );
        for mission in [&m.active["vespera"], &m.lodged["sila"][0], &m.pool[0]] {
            assert_eq!(mission.todo.len(), 1, "{}", mission.prompt);
            assert!(mission.todo[0].reports);
        }
    }

    #[test]
    fn cancelling_removes_the_open_mission_without_recording_it() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        assert!(m.cancel("bram"), "an open mission is cancelled");
        assert!(!m.is_on_mission("bram"));
        assert!(
            m.done("bram").is_none(),
            "a cancelled mission is not a finished one"
        );
        assert!(!m.cancel("bram"), "nothing to cancel twice");
    }

    #[test]
    fn missions_survive_a_round_trip_through_json() {
        let mut m = Missions::default();
        m.assign("bram", a_mission("do the thing"));
        m.add_todo("bram", "look twice");
        m.lodge("bram", a_mission("then this"));
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", Some("it holds".to_string()));
        let blob = serde_json::to_value(&m).unwrap();
        let back: Missions = serde_json::from_value(blob).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            back.active("bram").unwrap().todo.last().unwrap().text,
            "look twice"
        );
        assert_eq!(
            back.done("yen").unwrap().answer.as_deref(),
            Some("it holds")
        );
    }

    #[test]
    fn latest_is_the_open_mission_over_a_finished_one() {
        let mut m = Missions::default();
        m.assign("yen", a_mission("first task"));
        m.report("yen", Outcome::Pass, "done", None);
        m.assign("yen", a_mission("second task"));
        // With both a finished and an open mission, the open one is what an
        // operator's question is about.
        assert_eq!(m.latest("yen").unwrap().prompt, "second task");
    }
}

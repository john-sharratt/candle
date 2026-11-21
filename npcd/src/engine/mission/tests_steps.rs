//! Tests for how each kind of step is signed off, and what a report is held to.
//!
//! A journey is signed off by arriving in the named room on the named level, and a
//! reading by a scan that shows the machine. Being with somebody, or speaking to
//! them, signs off the steps about them by whole name or first name. A stuck report
//! counts only once something was tried, and a document's faults count again only
//! after a rewrite. A refused document's writing is reopened, and a read of a
//! document no longer on the record is struck from the steps.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::{Mission, Origin, Outcome, StepOutcome, Todo, Work, UNTRIED_TURN_AWAYS};

    /// **A filed report signs off the report step**, achieved whatever the
    /// verdict, and leaves the other steps as they were.
    #[test]
    fn filing_the_report_ticks_the_report_step() {
        for outcome in [Outcome::Pass, Outcome::Fail] {
            let mut m = Mission::new(
                "Read the board.",
                vec![
                    Todo::new("scan the status board"),
                    Todo::report("go back to the table and report it"),
                ],
                Origin::Lodged {
                    by: "u_op".to_string(),
                },
            );
            m.complete(outcome, "it is raised", None);
            assert!(m.todo[1].done, "the report step was left open");
            assert_eq!(m.todo[1].outcome, Some(StepOutcome::Achieved));
            assert!(!m.todo[0].done, "a step never taken was signed off");
        }
    }

    pub(in super::super) fn a_mission() -> Mission {
        Mission::new(
            "Do the thing.",
            vec![Todo::new("step one"), Todo::new("step two")],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        )
    }

    pub(in super::super) fn read_the_valve() -> Mission {
        Mission::new(
            "Go to the plant room and read the coolant valve.",
            vec![
                Todo::new("go to the plant room"),
                Todo::new("scan the coolant valve and read what state it is in"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        )
    }

    /// **A stuck report counts only once something was tried.** Over a
    /// document's faults, a turn-away counts once per rewrite: the same draft
    /// reported stuck again is not a try. On the way, one counts only after the
    /// world refused an act.
    #[test]
    fn only_a_real_turn_away_counts_toward_believing_stuck() {
        let mut m = a_mission();
        assert_eq!(
            m.refuse_stuck_over_faults(7),
            1,
            "the first draft was a try"
        );
        assert_eq!(
            m.refuse_stuck_over_faults(7),
            1,
            "the same draft again is not"
        );
        assert_eq!(m.refuse_stuck_over_faults(8), 2, "a rewrite is");

        let mut m = a_mission();
        assert_eq!(m.refuse_stuck_on_the_way(), 0, "nothing tried");
        m.refused_an_act();
        assert_eq!(m.refuse_stuck_on_the_way(), 1);
        assert_eq!(m.refuse_stuck_on_the_way(), 1, "one refusal counts once");

        // A step that connects to a terminal is done when that terminal is
        // worked, by its name with or without its article, and by no other.
        let mut m = Mission::new(
            "keep watch",
            vec![
                Todo::new("connect to the cluster and read its status"),
                Todo::new("connect to WhiteWorld and run its health check"),
            ],
            Origin::Own,
        );
        assert!(super::engine_sees(
            "connect to the cluster and read its status"
        ));
        assert!(!m.worked_terminal("the lights"));
        assert!(m.worked_terminal("the cluster"));
        assert!(!m.worked_terminal("the cluster"), "signed off already");
        assert!(m.worked_terminal("whiteworld"));
        assert!(m.all_todos_done());

        // Never trying counts in the end: one for every run of turn-aways.
        let mut m = a_mission();
        let counted: Vec<u32> = (0..UNTRIED_TURN_AWAYS * 2)
            .map(|_| m.refuse_stuck_on_the_way())
            .collect();
        assert_eq!(counted, [0, 0, 1, 1, 1, 2]);
    }

    /// **Arriving signs the journey off; arriving somewhere else does not.** The
    /// room is compared by name, with or without its article.
    #[test]
    fn arriving_in_the_named_room_signs_off_the_journey_to_it() {
        let mut m = read_the_valve();
        assert!(!m.arrived_in("the command room", "the command level"));
        assert!(!m.todo[0].done);
        assert!(m.arrived_in("The plant room", "the command level"));
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        assert!(
            !m.arrived_in("the plant room", "the command level"),
            "already signed off"
        );
        assert_eq!(
            m.next_step().map(|t| t.text.as_str()),
            Some("scan the coolant valve and read what state it is in")
        );
    }

    pub(in super::super) fn write_a_year() -> Mission {
        Mission::new(
            "Write the year the sky went out.",
            vec![
                Todo::new("go to band one on the casting level"),
                Todo::new("read layers/eras/the-awakening.md"),
                Todo::new("read layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"),
                Todo::new("write layers/life/keeper/2488 The Year After.md and commit it"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:keeper".into(),
                operation: 1,
                step: "write".into(),
            },
        )
        .with_work(Work {
            writes: "layers/life/keeper/2488 The Year After.md".into(),
            reads: vec!["layers/eras/the-awakening.md".into()],
            min_words: 250,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        })
    }

    /// **A room of the same name on another level is not the one asked for**;
    /// the level decides.
    #[test]
    fn a_journey_that_names_its_level_is_signed_off_only_there() {
        let mut m = write_a_year();
        assert!(!m.arrived_in("band one", "the chronicle"));
        assert!(!m.todo[0].done);
        assert!(m.arrived_in("band one", "the casting level"));
        assert!(m.todo[0].done);
    }

    /// **Reading a named document and committing the written one sign their
    /// steps off**, whatever case or slashes the path came back in.
    #[test]
    fn reading_and_committing_the_named_documents_sign_their_steps_off() {
        let mut m = write_a_year();
        assert!(!m.read_doc("layers/eras/the-salvation.md"));
        assert!(m.read_doc("Layers/eras/the-awakening.md"));
        assert!(m.todo[1].done);
        assert!(m.read_doc("/layers/life/keeper/2487-03-08 The Second the Sky Went Out.md"));
        assert!(!m.committed(&["layers/life/keeper/2489 Another.md".to_string()]));
        assert!(!m.todo[3].done);
        assert!(m.committed(&[
            "layers/eras/the-awakening.md".to_string(),
            "layers/life/keeper/2488 The Year After.md".to_string()
        ]));
        assert!(m.todo[3].done);
        assert!(m.written_up());
        assert!(m.arrived_in("band one", "the casting level"));
        assert!(m.next_step().unwrap().reports);
    }

    /// **Written up means committed while carried** — a write step struck on
    /// somebody's word is not the document on the record.
    #[test]
    fn only_a_commit_writes_a_mission_up() {
        let mut m = write_a_year();
        assert!(!m.written_up());
        m.check_off(
            "write layers/life/keeper/2488 The Year After.md and commit it",
            StepOutcome::Thwarted,
        );
        assert!(!m.written_up(), "thwarted is not written");
        assert!(read_the_valve().written_up(), "nothing to write");
    }

    /// **Only the engine signs off what the engine sees** — the journey, the
    /// reads and the write — and anything else may be signed off on the
    /// character's word.
    #[test]
    fn the_steps_the_engine_sees_are_named() {
        for step in &write_a_year().todo[..4] {
            assert!(super::engine_sees(&step.text), "{}", step.text);
        }
        assert!(super::engine_sees(
            "scan the coolant valve and read what state it is in"
        ));
        assert!(!super::engine_sees("form your own view of it"));
    }

    /// **A step settled in the body's own mind is told so**; a journey, a
    /// reading or a word to somebody is not one.
    /// **A judgement keeps its place, and does not hide what follows it**: the
    /// next step is still the authored one, and the step to do is the journey
    /// after it — until only the report is left.
    #[test]
    fn a_judgement_step_does_not_hide_the_journey_after_it() {
        use super::{judgement_then, Mission, Origin, Todo};
        let mut m = Mission::new(
            "Look in on them.",
            vec![
                Todo::new("decide what they might need"),
                Todo::new("go to the store room"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "x".into(),
            },
        );
        let text = |t: Option<&Todo>| t.map(|t| t.text.clone());
        assert_eq!(
            text(m.next_step()).as_deref(),
            Some("decide what they might need")
        );
        assert_eq!(
            text(m.step_to_do()).as_deref(),
            Some("go to the store room")
        );
        m.todo[1].done = true;
        assert!(m.step_to_do().is_none(), "only the report is left");
        assert_eq!(
            judgement_then("go to the store room", Some("the store room: `move_to` it")),
            "Nothing in the world marks that done: settle it from what you have seen, and say it \
             in your report. Then go on to what is still to do: go to the store room. It is in \
             the store room: `move_to` it."
        );
    }

    #[test]
    fn a_judgement_step_is_known_by_its_verb() {
        use super::is_judgement;
        for step in [
            "settle how the Sharratt family is today",
            "Decide what they might need before they ask for it",
            "note whether the wifi is up, how fast it is, and what is using most of it",
            "work out where the two meet",
        ] {
            assert!(is_judgement(step), "{step}");
        }
        for step in [
            "go to the Sharratt room",
            "scan the sensor panel and read what state it is in",
            "ask Wick what they found",
            "notes from the ward",
        ] {
            assert!(!is_judgement(step), "{step}");
        }
    }

    /// **A scan that shows the machine signs off the reading of it**, and a
    /// mission with a reading still to make holds `report_done` up — until the
    /// reading is made, when only the report is left.
    #[test]
    fn a_scan_that_shows_the_machine_signs_off_the_reading() {
        let mut m = read_the_valve();
        let machines = ["the coolant valve".to_string()];
        assert_eq!(
            m.unread_step(&machines),
            Some("scan the coolant valve and read what state it is in")
        );
        assert_eq!(
            m.unread_step(&["the breaker panel".to_string()]),
            None,
            "a reading of nothing the world holds is not the engine's to hold up"
        );
        assert!(!m.read_off(&["the plant panel".to_string()]));
        assert!(m.read_off(&[
            "the plant panel".to_string(),
            "the coolant valve".to_string()
        ]));
        assert_eq!(m.unread_step(&machines), None);
        assert_eq!(
            m.next_step().map(|t| t.text.as_str()),
            Some("go to the plant room"),
            "the journey was never signed off, and reading does not sign it"
        );
        assert!(m.arrived_in("the plant room", "the command level"));
        assert!(m.next_step().unwrap().reports);
        assert!(!m.todo[2].done, "nothing but the report closes the report");
    }

    /// **Being with somebody signs off finding them; speaking to them signs off
    /// asking them too.** By whole name or first name; nobody else's name does.
    #[test]
    fn meeting_and_speaking_to_somebody_sign_off_the_steps_about_them() {
        let mut m = Mission::new(
            "Find out what Paxon Vael is working on.",
            vec![
                Todo::new("find Paxon Vael"),
                Todo::new("ask Paxon what he is working on"),
                Todo::new("tell Ione Valtiere what you learnt"),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        );
        assert!(!m.met("Vespera Kaine"));
        assert!(m.met("Paxon Vael"));
        assert!(m.todo[0].done);
        assert!(!m.todo[1].done, "being in the room is not asking");
        assert!(m.spoke_with("Paxon Vael"));
        assert!(m.todo[1].done, "asked by first name");
        assert!(!m.todo[2].done);
        assert!(!m.spoke_with("Pax"), "a fragment of a name is nobody");
        assert!(
            !m.spoke_with("the channel"),
            "an article is not a first name"
        );
        assert!(m.spoke_with("Ione Valtiere"));
        assert!(m.next_step().unwrap().reports);
    }

    /// **An order-giver's steps are signed off by its acts landing**: a post
    /// signs off the step that posts work, a reading of the reports the step
    /// that reads what came back, and a read of the phone the step that reads
    /// the messages — each only its own.
    #[test]
    fn an_order_givers_steps_are_signed_off_by_its_acts() {
        let mut m = Mission::new(
            "Keep the table.",
            vec![
                Todo::new("read what came back at the order table"),
                Todo::new("Post work on the order table"),
                Todo::new("read your messages"),
            ],
            Origin::Own,
        );
        assert!(m.posted_work());
        assert!(!m.posted_work(), "signed off already");
        let open: Vec<&str> = m
            .todo
            .iter()
            .filter(|t| !t.done)
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(
            open,
            [
                "read what came back at the order table",
                "read your messages"
            ]
        );
        assert!(m.read_results());
        assert!(m.read_messages());
        assert!(m.all_todos_done());
    }

    /// **A document that does not stand is to be written again**: a done
    /// writing step is reopened, and a mission with none gains one before its
    /// report.
    #[test]
    fn a_refused_document_puts_its_writing_back_in_front() {
        use super::{Mission, Origin, Todo};
        let doc = "layers/life/keeper/2950 The Archive.md";
        let mut drafted = Mission::new(
            "write it",
            vec![
                Todo::new(format!("write {doc} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        );
        drafted.committed(&[doc.to_string()]);
        assert!(drafted.next_step().is_some_and(|t| t.reports));
        drafted.reopen_write(doc);
        assert_eq!(
            drafted.next_step().map(|t| t.text.as_str()),
            Some(format!("write {doc} and commit it").as_str())
        );

        let mut checked = Mission::new(
            "check it",
            vec![
                Todo::new(format!("read {doc}")),
                Todo::report("go back to the table and report your verdict"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        );
        checked.read_doc(doc);
        checked.reopen_write(doc);
        let steps: Vec<&str> = checked.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                format!("read {doc}").as_str(),
                format!("change {doc} and commit it").as_str(),
                "go back to the table and report your verdict"
            ]
        );
    }

    /// **A read of a document the record no longer holds is struck**, step and
    /// listed read both; one already done, and every other step, stays.
    #[test]
    fn a_read_of_a_document_gone_from_the_record_is_struck() {
        use super::{Mission, Origin, Todo, Work};
        let gone = "layers/stories/the-anchor-s-last-breath.md";
        let kept = "layers/eras/the-portal-retreat.md";
        let writes = "layers/stories/the-first-breach.md";
        let mut m = Mission::new(
            "write it",
            vec![
                Todo::new("go to the first writing room on the story level"),
                Todo::new(format!("read {kept}")),
                Todo::new(format!("read {gone}")),
                Todo::new(format!("write {writes} and commit it")),
                Todo::report("go back to the table and report it"),
            ],
            Origin::Random {
                routine: "r".into(),
            },
        )
        .with_work(Work {
            writes: writes.into(),
            reads: vec![
                kept.into(),
                "Layers/Stories/The-Anchor-S-Last-Breath.md".into(),
            ],
            min_words: 0,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        });
        let holds = |op: Option<u64>, p: &str| {
            assert_eq!(op, None, "not an operation's step");
            p != gone
        };
        assert_eq!(m.strike_gone_reads(&holds), [gone]);
        let steps: Vec<&str> = m.todo.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(
            steps,
            [
                "go to the first writing room on the story level",
                "read layers/eras/the-portal-retreat.md",
                "write layers/stories/the-first-breach.md and commit it",
                "go back to the table and report it"
            ]
        );
        assert_eq!(m.work.as_ref().unwrap().reads, [kept]);
        assert!(
            m.strike_gone_reads(&holds).is_empty(),
            "nothing left to strike"
        );

        // A read already made stays on the list as done: it happened.
        let mut read = Mission::new(
            "check it",
            vec![Todo::new(format!("read {gone}")), Todo::report("report")],
            Origin::Random {
                routine: "r".into(),
            },
        );
        read.read_doc(gone);
        assert!(read.strike_gone_reads(&holds).is_empty());
        assert_eq!(read.todo.len(), 2);
    }

    #[test]
    fn a_step_names_its_aim() {
        use super::Aim;
        assert_eq!(
            Aim::of("go to the plant room"),
            Some(Aim::Room("plant room".into()))
        );
        let read = Aim::of("scan the coolant valve and read what state it is in").unwrap();
        assert!(read.is_machine("the coolant valve"));
        assert!(!read.is_machine("the breaker panel"));
        let ask = Aim::of("ask Paxon Vael what he has been working on").unwrap();
        assert!(ask.is_person("Paxon Vael"));
        assert!(!ask.is_room("the plant room", "the command level"));
        // A destination that names its level is that room on that level only.
        let band = Aim::of("go to band one on the casting level").unwrap();
        assert!(band.is_room("band one", "the casting level"));
        assert!(!band.is_room("band one", "the chronicle"));
        assert!(Aim::of("go to the plant room")
            .unwrap()
            .is_room("the plant room", "anywhere"));
        let go_to_him = Aim::of("go to Paxon Vael").unwrap();
        assert!(go_to_him.is_person("Paxon Vael"));
        assert_eq!(Aim::of("form your own view of it"), None);
    }

    /// A step the engine cannot observe never holds a report up.
    #[test]
    fn a_step_the_engine_cannot_see_does_not_hold_the_report_up() {
        let m = Mission::new(
            "Hear Pax out.",
            vec![Todo::new("find Pax"), Todo::new("form your own view of it")],
            Origin::Lodged {
                by: "u_abc".to_string(),
            },
        );
        assert_eq!(m.unread_step(&["Pax".to_string()]), None);
    }

    #[test]
    fn a_new_mission_is_open_with_no_report_and_no_answer() {
        let m = a_mission();
        assert!(m.is_open());
        assert!(m.report.is_none());
        assert!(m.answer.is_none());
        assert!(!m.all_todos_done());
    }

    #[test]
    fn check_off_ticks_the_named_step_case_insensitively_and_reports_the_hit() {
        let mut m = a_mission();
        assert!(
            m.check_off("  STEP one ", StepOutcome::Achieved),
            "trim + case-insensitive match"
        );
        assert!(m.todo[0].done);
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        assert!(!m.todo[1].done);
        assert_eq!(m.todo[1].outcome, None);
        // A second tick of the same step finds nothing left to tick.
        assert!(!m.check_off("step one", StepOutcome::Thwarted));
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Achieved));
        // A step that is not on the list ticks nothing.
        assert!(!m.check_off("step three", StepOutcome::Achieved));
    }

    #[test]
    fn a_thwarted_step_is_done_and_reads_to_the_character_as_one_it_could_not_do() {
        let mut m = a_mission();
        assert!(m.check_off("step one", StepOutcome::Thwarted));
        assert!(m.todo[0].done);
        assert_eq!(m.todo[0].outcome, Some(StepOutcome::Thwarted));
        assert_eq!(StepOutcome::Achieved.as_str(), "achieved");
        assert_eq!(StepOutcome::Thwarted.as_str(), "thwarted");
        assert_eq!(
            m.task_text().as_deref(),
            Some("[could not be done] step one\n[ ] step two")
        );
        let mut achieved = a_mission();
        achieved.check_off("step one", StepOutcome::Achieved);
        assert_ne!(m.content_fingerprint(), achieved.content_fingerprint());
    }

    #[test]
    fn a_step_signed_off_before_outcomes_were_recorded_loads_with_none() {
        let t: Todo = serde_json::from_str(r#"{"text":"old","done":true}"#).unwrap();
        assert!(t.done);
        assert_eq!(t.outcome, None);
        assert_eq!(
            serde_json::to_string(&Todo::new("x")).unwrap(),
            r#"{"text":"x","done":false,"outcome":null,"reports":false,"critical":false}"#
        );
    }

    #[test]
    fn all_todos_done_only_once_every_step_is_ticked() {
        let mut m = a_mission();
        assert!(!m.all_todos_done());
        assert!(m.check_off("step one", StepOutcome::Achieved));
        assert!(!m.all_todos_done());
        assert!(m.check_off("step two", StepOutcome::Thwarted));
        assert!(m.all_todos_done());
    }

    #[test]
    fn add_todo_appends_but_refuses_blank_and_open_duplicates() {
        let mut m = a_mission();
        assert!(m.add_todo("step three"));
        assert_eq!(m.todo.len(), 3);
        assert!(!m.add_todo("   "), "blank refused");
        assert!(!m.add_todo(" STEP three "), "open duplicate refused");
        assert_eq!(m.todo.len(), 3);
        // Once a step is done, the same text may be added again as new work.
        assert!(m.check_off("step three", StepOutcome::Achieved));
        assert!(m.add_todo("step three"));
        assert_eq!(m.todo.len(), 4);
    }

    #[test]
    fn complete_files_the_report_and_closes_the_mission() {
        let mut m = a_mission();
        m.complete(Outcome::Pass, "went fine", Some("the answer".to_string()));
        assert!(!m.is_open());
        assert_eq!(m.report.as_ref().unwrap().outcome, Outcome::Pass);
        assert_eq!(m.report.as_ref().unwrap().notes, "went fine");
        assert_eq!(m.answer.as_deref(), Some("the answer"));
        assert_eq!(Outcome::Pass.as_str(), "pass");
        assert_eq!(Outcome::Fail.as_str(), "fail");
    }
}

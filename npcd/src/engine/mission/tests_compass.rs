//! Tests for the compass lines: the act at each desk, the way to it, and the
//! refusal carried in front of the Maker.
//!
//! A document whose text the brief gives is written with its write act at a desk,
//! and by way away from one. A refused document keeps its refusal in front of the
//! Maker, bounded in length, with both ways out left open rather than a bare
//! instruction not to write again. A publish not yet landed points at the merge
//! table. The fingerprint changes only when what the character reads changes, so
//! filing a report does not force a re-seal.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::bank::{random, Duty, Facts, ROUTINES};

    use super::{Mission, Origin, Outcome, StepOutcome, Todo};

    use super::super::tests_steps::tests::{a_mission, read_the_valve};

    pub(in super::super) fn makers() -> Vec<String> {
        vec!["Wren".to_string(), "Pax".to_string(), "Soren".to_string()]
    }

    pub(in super::super) fn records() -> Vec<String> {
        vec!["the-charge".to_string(), "the-awakening".to_string()]
    }

    /// **Work is posted with the act at its address**, a standing job by name
    /// when there are any, and a notice is said not to be work — in every case.
    #[test]
    fn the_line_for_posting_work_names_the_act_and_the_jobs() {
        use super::post_work_line;
        use crate::engine::mission_gen::world::Routine;
        let jobs = [
            Routine {
                id: "ward-patrol".into(),
                ..Default::default()
            },
            Routine {
                id: "morning-briefing".into(),
                ..Default::default()
            },
        ];
        let at = "http://local/command/order-table~0/order_post";
        assert_eq!(
            post_work_line(&jobs, Some(at)),
            "Work is posted with `invoke` `http://local/command/order-table~0/order_post` — a \
             standing `job` by name when one fits what the families need: `ward-patrol`, \
             `morning-briefing`; or a `brief` and its `steps` for what none of them covers. A \
             notice on a board is not work: nobody takes it up. The table holds as many missions \
             nobody has taken as there are hands to take them; past that, a new one waits for a \
             report."
        );
        assert_eq!(
            post_work_line(&jobs, None),
            "Work is posted with `order_post` at the order table — a standing `job` by name when \
             one fits what the families need: `ward-patrol`, `morning-briefing`; or a `brief` and \
             its `steps` for what none of them covers. A notice on a board is not work: nobody \
             takes it up. The table holds as many missions nobody has taken as there are hands \
             to take them; past that, a new one waits for a report."
        );
        assert_eq!(
            post_work_line(&[], Some(at)),
            "Work is posted with `invoke` `http://local/command/order-table~0/order_post`, with a \
             `brief` and its `steps`. A notice on a board is not work: nobody takes it up. The \
             table holds as many missions nobody has taken as there are hands to take them; past \
             that, a new one waits for a report."
        );
        assert_eq!(
            post_work_line(&[], None),
            "Work is posted with `order_post` at the order table, with a `brief` and its `steps`. \
             A notice on a board is not work: nobody takes it up. The table holds as many \
             missions nobody has taken as there are hands to take them; past that, a new one \
             waits for a report."
        );
    }

    /// **A document whose text the brief gives is written as given**: at a desk
    /// the compass names `file_write` and the commit at their addresses, and
    /// away from one, the acts and the way.
    #[test]
    fn the_compass_for_a_given_text_names_file_write() {
        use super::given_compass;
        let p = "missions/moon/ops/check-the-wifi.yaml";
        assert_eq!(
            given_compass(
                p,
                Some("http://local/archive/map-table~0/file_write"),
                Some("http://local/archive/map-table~0/bench_commit"),
                None,
                None
            ),
            "Your mission, next: write missions/moon/ops/check-the-wifi.yaml exactly as your \
             brief gives it, whole: `invoke` http://local/archive/map-table~0/file_write with \
             `path` \"missions/moon/ops/check-the-wifi.yaml\" and that text; then `invoke` \
             http://local/archive/map-table~0/bench_commit with `why`, one line saying what it \
             is."
        );
        assert_eq!(
            given_compass(p, None, None, Some("the watch room: `move_to` it"), None),
            "Your mission, next: write missions/moon/ops/check-the-wifi.yaml exactly as your \
             brief gives it, whole, with `file_write` at a desk, then commit it with \
             `bench_commit`. The nearest desk is in the watch room: `move_to` it."
        );
        // Refused at its last report, it is told why first.
        assert_eq!(
            given_compass(
                p,
                None,
                None,
                None,
                Some("It is not a mission template: `steps` is missing.")
            ),
            "When you last reported missions/moon/ops/check-the-wifi.yaml, the table would not \
             take it: It is not a mission template: `steps` is missing.\nPut that right in the \
             text and write it again. Your mission, next: write \
             missions/moon/ops/check-the-wifi.yaml exactly as your brief gives it, whole, with \
             `file_write` at a desk, then commit it with `bench_commit`."
        );
    }

    /// **The compass for the year says the act at the machine, or the way to it.**
    #[test]
    fn the_compass_for_the_year_says_the_act_or_the_way() {
        use super::time_compass;
        assert_eq!(
            time_compass(2837, true, None),
            "Your mission, next: set your time to 2837. A time machine is here: `time_travel` \
             naming the year 2837."
        );
        assert_eq!(
            time_compass(
                2837,
                false,
                Some("the first time room, on the time level: `move_to` the first time room")
            ),
            "Your mission, next: set your time to 2837. The time machines are in the first time \
             room, on the time level: `move_to` the first time room; there, `time_travel` naming \
             the year 2837."
        );
        assert!(time_compass(2837, false, None).contains("the lift takes you there"));
    }

    /// **At a desk, the compass says the exact act; away from one, the way.**
    #[test]
    fn the_compass_for_a_document_says_the_act_that_does_it() {
        use super::{doc_compass, DeskVerbs, DocStep, COMPASS_REFUSAL_CHARS};
        let desk = DeskVerbs {
            read: Some("http://local/desk/1/file_read".into()),
            compose: Some("http://local/desk/1/compose".into()),
            edit: None,
            commit: Some("http://local/desk/1/bench_commit".into()),
        };
        assert_eq!(
            doc_compass(&DocStep::Read("layers/eras/x.md".into()), &desk, None, None),
            "Your mission, next: read layers/eras/x.md. `invoke` http://local/desk/1/file_read \
             with `path` \"layers/eras/x.md\" — it comes back a page at a time."
        );
        let written = doc_compass(
            &DocStep::Write("layers/stories/y.md".into()),
            &desk,
            None,
            Some(537),
        );
        assert_eq!(
            written,
            "Your mission, next: commit layers/stories/y.md. It is written — 537 words in your \
             working copy. Read it back with `invoke` http://local/desk/1/file_read with `path` \
             \"layers/stories/y.md\"; if it stands, `invoke` http://local/desk/1/bench_commit with \
             `why`, one line saying what it is. Put one passage right first if one is wrong, but \
             do not write it again."
        );
        let write = doc_compass(
            &DocStep::Write("layers/stories/y.md".into()),
            &desk,
            None,
            None,
        );
        assert!(
            write.contains(
                "Sit down and write it whole: `invoke` http://local/desk/1/compose — you write it \
                 through"
            ) && write.contains("Then `invoke` http://local/desk/1/bench_commit with `why`"),
            "{write}"
        );
        assert_eq!(
            doc_compass(
                &DocStep::Read("layers/eras/x.md".into()),
                &DeskVerbs::default(),
                Some("band one, on the story level: `move_to` band one"),
                None
            ),
            "Your mission, next: read layers/eras/x.md. Documents are read at a desk: the nearest \
             is in band one, on the story level: `move_to` band one."
        );
        // A refused document: the refusal stands in front of the Maker, and
        // both ways out of it are open — never "do not write it again".
        let refused = super::refused_compass(
            "layers/stories/y.md",
            "it copies the brief's words: 'Out of scope.'",
            &desk,
            None,
        );
        assert!(refused.contains("it copies the brief's words"), "{refused}");
        assert!(refused.contains("`file_edit`"), "{refused}");
        assert!(refused.contains("http://local/desk/1/compose"), "{refused}");
        assert!(!refused.contains("do not write it again"), "{refused}");
        // A refusal past the bound is carried by its opening: one kept from
        // before the gate named five of a kind ran to 412,000 characters.
        let huge = "x".repeat(COMPASS_REFUSAL_CHARS + 50);
        let carried = super::refused_compass("layers/stories/y.md", &huge, &desk, None);
        assert!(carried.contains(&format!(
            "{} … (the table said more; report it again to hear all of it)\n",
            "x".repeat(COMPASS_REFUSAL_CHARS)
        )));
        assert!(!carried.contains(&"x".repeat(COMPASS_REFUSAL_CHARS + 1)));
        // A terminal that reads but does not compose is not where a piece is
        // written: the way goes to a writing desk.
        let reads_only = DeskVerbs {
            read: Some("http://local/chronicle/t/file_read".into()),
            commit: Some("http://local/chronicle/t/bench_commit".into()),
            ..DeskVerbs::default()
        };
        assert_eq!(
            doc_compass(
                &DocStep::Write("layers/stories/y.md".into()),
                &reads_only,
                Some("band one, on the story level: `move_to` band one"),
                None
            ),
            "Your mission, next: write layers/stories/y.md. A piece is written at a writing \
             desk: the nearest is in band one, on the story level: `move_to` band one."
        );
    }

    /// The paths a step names, by the step's own verb.
    #[test]
    fn a_step_names_the_document_it_reads_or_writes() {
        assert_eq!(
            super::reading_doc("read layers/eras/the-awakening.md closely"),
            Some("layers/eras/the-awakening.md".into())
        );
        assert_eq!(
            super::writing_doc("write layers/life/keeper/2488 The Year After.md and commit it"),
            Some("layers/life/keeper/2488 the year after.md".into())
        );
        assert_eq!(
            super::writing_doc("change worlds/battle-cities.yaml so both agree"),
            Some("worlds/battle-cities.yaml".into())
        );
        assert_eq!(super::reading_doc("read the record 'the-charge'"), None);
        assert_eq!(super::writing_doc("go to band one"), None);
        // Told to the character as the step spells it.
        assert_eq!(
            super::DocStep::of("write layers/life/keeper/2488 The Year After.md and commit it"),
            Some(super::DocStep::Write(
                "layers/life/keeper/2488 The Year After.md".into()
            ))
        );
        assert_eq!(
            super::DocStep::of("read layers/eras/the-awakening.md"),
            Some(super::DocStep::Read("layers/eras/the-awakening.md".into()))
        );
        assert_eq!(super::DocStep::of("go to band one"), None);
        // A title whose lowercase is longer than itself does not split a
        // character.
        assert_eq!(
            super::DocStep::of("write layers/stories/İstanbul Gate.MD and commit it"),
            Some(super::DocStep::Write(
                "layers/stories/İstanbul Gate.MD".into()
            ))
        );
    }

    /// **The compass names the next step and where the person it names is.**
    #[test]
    fn the_compass_says_where_the_person_the_step_names_is() {
        let ask = Todo::new("ask Paxon Vael what he has been working on");
        let elsewhere = [
            (
                "Paxon Vael".to_string(),
                Some("the plant room: `move_to` it".to_string()),
            ),
            ("Vespera Kaine".to_string(), None),
        ];
        assert_eq!(
            super::compass_line(&ask, &elsewhere, None),
            "Your mission, next: ask Paxon Vael what he has been working on. Paxon Vael is in \
             the plant room: `move_to` it — or `message` them, which reaches them wherever they \
             are."
        );
        let here = [("Paxon Vael".to_string(), None)];
        assert_eq!(
            super::compass_line(&ask, &here, None),
            "Your mission, next: ask Paxon Vael what he has been working on. Paxon Vael is here \
             with you: say it to them now — `message` them."
        );
        assert_eq!(
            super::compass_line(
                &Todo::new("go to band three"),
                &here,
                Some("Band three is on the record level: go to the lift.")
            ),
            "Your mission, next: go to band three. Band three is on the record level: go to the \
             lift."
        );
        assert!(super::compass_report_line(
            Some("the command room: `move_to` it"),
            false,
            &[],
            false
        )
        .contains("Go back to the table, in the command room: `move_to` it, and report"));
        let review = super::compass_report_line(None, true, &[], true);
        assert!(
            review.contains("That verdict is the whole report."),
            "{review}"
        );
        assert!(review.contains("`report_rejected`"), "{review}");
        let seen = ["in the plant room, the coolant valve: tight".to_string()];
        let now = super::compass_report_line(None, true, &seen, false);
        assert!(now.contains("report it now"), "{now}");
        assert!(
            now.ends_with(" What you saw on it: in the plant room, the coolant valve: tight."),
            "{now}"
        );
    }

    /// **A publish not yet landed points at the merge table, by its own
    /// addresses when it is here** — never "report it now", which the order
    /// table refuses until the work has landed.
    #[test]
    fn a_publish_not_landed_points_at_the_merge_table() {
        let doc = "layers/stories/the-opening.md";
        let here = super::publish_compass(
            doc,
            Some("http://local/command/merge-table~1/merge_status"),
            Some("http://local/command/merge-table~1/merge_publish"),
            None,
        );
        assert_eq!(
            here,
            "Your mission, next: land layers/stories/the-opening.md in the record. The merge \
             table is here: `invoke` http://local/command/merge-table~1/merge_status to see what \
             it changes and whether it merges cleanly, then `invoke` \
             http://local/command/merge-table~1/merge_publish to land it. Report at the table \
             once it has landed."
        );
        let away = super::publish_compass(doc, None, None, Some("the command room: `move_to` it"));
        assert_eq!(
            away,
            "Your mission, next: land layers/stories/the-opening.md in the record, at the merge \
             table in the command room: `move_to` it. There, `merge_status` shows whether it \
             merges cleanly and `merge_publish` lands it. Report at the table once it has landed."
        );
        let lost = super::publish_compass(doc, None, None, None);
        assert!(!lost.contains("report it now"), "{lost}");
        assert!(lost.contains("`merge_publish` lands it"), "{lost}");
    }

    #[test]
    fn the_progress_line_names_the_next_step_or_the_report() {
        let m = read_the_valve();
        assert_eq!(
            super::progress_line("You are in the plant room", m.todo.get(1), &[]),
            "You are in the plant room: that step of your mission is done. Next: scan the \
             coolant valve and read what state it is in."
        );
        assert_eq!(
            super::progress_line(
                "You are in the plant room",
                m.todo.get(1),
                &[
                    "the breaker panel".to_string(),
                    "the coolant valve".to_string()
                ]
            ),
            "You are in the plant room: that step of your mission is done. Next: scan the \
             coolant valve and read what state it is in. It is here, in this room: `scan` \
             naming no place, and it lists it with the state it is in."
        );
        let last = super::progress_line("You have read it", m.todo.get(2), &[]);
        assert!(
            last.starts_with("You have read it: that was the last step of your mission."),
            "{last}"
        );
        assert!(last.contains("`report_done`"), "{last}");
        assert_eq!(super::progress_line("You have read it", None, &[]), last);
    }

    #[test]
    fn mission_and_task_text_render_the_ask_and_the_ticked_steps() {
        let mut m = a_mission();
        m.check_off("step one", StepOutcome::Achieved);
        assert_eq!(m.mission_text(), "Do the thing.");
        assert_eq!(
            m.task_text().as_deref(),
            Some("[done] step one\n[ ] step two")
        );
        // A mission with no steps renders no task section (task_intro stays hidden).
        let bare = Mission::new(
            "Just be.",
            vec![],
            Origin::Random {
                routine: "x".into(),
            },
        );
        assert!(bare.task_text().is_none());
    }

    #[test]
    fn standing_text_names_the_ask_the_steps_and_where_it_ends() {
        let m = a_mission();
        let text = m.standing_text();
        assert!(text.starts_with("What has been asked of you: Do the thing."));
        assert!(text.contains("The steps that see it through:"));
        assert!(text.contains("step one") && text.contains("step two"));
        // It ends by pointing back to the table to report and take the next — the
        // loop closes there, not out in the world — by scanning it and invoking
        // its report verbs.
        assert!(
            text.contains("go back to the table where work is handed out, `scan` it")
                && text.contains("`report_done`")
                && text.contains("`report_stuck`")
        );
        // A mission carried on its ask alone still ends at the table.
        let bare = Mission::new(
            "Just be.",
            vec![],
            Origin::Random {
                routine: "x".into(),
            },
        );
        assert!(bare
            .standing_text()
            .contains("go back to the table where work is handed out"));
    }

    #[test]
    fn the_prompt_is_the_standing_text_under_its_own_heading_and_carries_the_fingerprint() {
        let m = a_mission();
        let prompt = m.prompt();
        assert!(!prompt.text.starts_with("What has been asked of you"));
        assert_eq!(
            m.standing_text(),
            format!("What has been asked of you: {}", prompt.text)
        );
        assert_eq!(prompt.fingerprint, m.content_fingerprint());
    }

    #[test]
    fn content_fingerprint_tracks_the_rendered_prompt_only() {
        let m1 = a_mission();
        let mut m2 = a_mission();
        // Filing a report or an answer does not change what the character reads,
        // so the fingerprint is unchanged and no re-seal is forced.
        m2.answer = Some("built up".to_string());
        m2.report = Some(super::Report {
            outcome: Outcome::Pass,
            notes: "done".to_string(),
        });
        assert_eq!(m1.content_fingerprint(), m2.content_fingerprint());
        // Ticking a step does change the rendered task text, so it re-fingerprints.
        let mut m3 = a_mission();
        m3.check_off("step one", StepOutcome::Achieved);
        assert_ne!(m1.content_fingerprint(), m3.content_fingerprint());
    }

    #[test]
    fn a_random_mission_is_open_non_empty_and_tagged_with_a_known_routine() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        for seed in 0..50u64 {
            let m = random(&facts, seed);
            assert!(m.is_open());
            assert!(!m.prompt.trim().is_empty());
            assert!(!m.todo.is_empty(), "a routine gives steps to act on");
            match &m.origin {
                Origin::Random { routine } => {
                    assert!(
                        ROUTINES.contains(&routine.as_str()),
                        "known routine: {routine}"
                    );
                }
                other => panic!("bank produces Random origins, not {other:?}"),
            }
        }
    }

    #[test]
    fn random_missions_vary_with_the_seed() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        let prompts: std::collections::HashSet<String> =
            (0..30u64).map(|s| random(&facts, s).prompt).collect();
        assert!(
            prompts.len() > 3,
            "the bank is diverse, not one repeated mission"
        );
    }

    #[test]
    fn a_lone_character_with_no_records_still_gets_an_actionable_mission() {
        // No other makers, no archive: routines needing them are skipped and the
        // always-available "walk the halls" fallback fires.
        let facts = Facts::default();
        // The only routines needing no other maker and no record.
        let no_material = ["take-stock", "walk-the-halls"];
        for seed in 0..20u64 {
            let m = random(&facts, seed);
            assert!(!m.todo.is_empty());
            match &m.origin {
                Origin::Random { routine } => assert!(
                    no_material.contains(&routine.as_str()),
                    "a lone character gets a routine needing no material, got {routine}"
                ),
                other => panic!("bank produces Random origins, not {other:?}"),
            }
        }
    }

    #[test]
    fn a_world_with_machines_gets_a_mission_to_go_and_read_one() {
        let mk = makers();
        let duties = vec![
            Duty {
                room: "the foundry".to_string(),
                device: "the coolant valve".to_string(),
            },
            Duty {
                room: "the armoury".to_string(),
                device: "the breaker panel".to_string(),
            },
        ];
        let facts = Facts {
            makers: &mk,
            duties: &duties,
            table: Some("the muster hall"),
            ..Facts::default()
        };
        for seed in 0..20u64 {
            let m = random(&facts, seed);
            assert_eq!(
                m.origin,
                Origin::Random {
                    routine: "read-a-station".to_string()
                }
            );
            let duty = duties
                .iter()
                .find(|d| m.prompt.contains(&d.device))
                .expect("the mission names one of the machines");
            assert!(m.prompt.contains(&duty.room), "{}", m.prompt);
            assert_eq!(m.todo[0].text, format!("go to {}", duty.room));
            assert_eq!(
                m.todo.last().unwrap().text,
                "go back to the table in the muster hall and report it",
                "it ends at the table, by the room's name"
            );
        }
    }

    #[test]
    fn every_routine_ends_with_a_step_that_reports_at_the_table() {
        let mk = makers();
        let rc = records();
        let facts = Facts {
            makers: &mk,
            records: &rc,
            ..Facts::default()
        };
        for seed in 0..30u64 {
            let m = random(&facts, seed);
            assert_eq!(
                m.todo.last().unwrap().text,
                "go back to the table and report it"
            );
            assert!(m.todo.last().unwrap().reports);
            assert_eq!(m.todo.iter().filter(|t| t.reports).count(), 1);
        }
    }
}

//! Tests the station acts: the subject rule, taking a station in one step, the
//! collision that names the other party, and the findings that craft acts leave.
//!
//! Each act that reports work leaves a trace of it, and a machine name is listed
//! once. The preposition names the thing written into, and a second body cannot
//! write into work another body holds. Filing and letting go have their own rules,
//! settling needs the other party named, and blame says where the chain goes quiet.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use serde_json::json;

    use super::super::tests_mission::tests::vault;

    use super::super::tests_documents::tests::act;

    /// **A machine name is listed once**, though seven plant rooms each hold a
    /// breaker panel.
    #[test]
    fn a_machine_not_found_lists_each_name_once() {
        let h = vault();
        let why = h.sim(|s| machine_named(s, "the air handler")).unwrap_err();
        assert_eq!(why.matches("the breaker panel").count(), 1, "{why}");
        assert!(
            h.sim(|s| s
                .devices
                .iter()
                .filter(|d| d.name == "the breaker panel")
                .count())
                > 1,
            "the vault has a panel in more than one room"
        );
    }

    /// **An act that reports work leaves a trace of it.**
    ///
    /// These four returned prose and changed nothing: the character did the
    /// reading, said so, and the next character to pick the piece up found no
    /// sign of it — so the work was done again, and again, with nothing
    /// accumulating. They read as working from every angle except the only one
    /// that counts.
    #[test]
    fn the_craft_readings_leave_a_finding_behind_them() {
        let h = vault();
        let before = h.sim(|s| s.ledger.verdicts_on("the third era").len());

        assert!(perform(
            &h,
            "m1",
            &act("structure_lay_out_scenes", json!({"what":"the third era"}))
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act(
                "structure_test_the_want",
                json!({"what":"the third era","scene":"the gate"})
            ),
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act("structure_find_the_slack", json!({"what":"the third era"}))
        )
        .happened());

        let after: Vec<String> = h.sim(|s| {
            s.ledger
                .verdicts_on("the third era")
                .iter()
                .map(|v| v.judgement.clone())
                .collect()
        });
        assert_eq!(
            after.len(),
            before + 3,
            "a reading was reported and left nothing: {after:?}"
        );
        assert!(after.iter().any(|j| j.contains("scenes")), "{after:?}");
        assert!(after.iter().any(|j| j.contains("want")), "{after:?}");
        assert!(after.iter().any(|j| j.contains("slack")), "{after:?}");
    }

    /// Tidying an index leaves the index changed, rather than saying it did.
    #[test]
    fn tidying_an_index_writes_the_way_in() {
        let h = vault();
        h.sim(|s| {
            assert!(s
                .record
                .by_name("the third era")
                .unwrap()
                .description
                .is_none())
        });
        let out = perform(
            &h,
            "m1",
            &act("record_tidy_index", json!({"what":"the third era"})),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let entry = s
                .record
                .by_name("the third era")
                .unwrap()
                .description
                .clone()
                .expect("the index entry was not written");
            assert!(entry.contains("named span"), "{entry}");
        });
    }

    /// A reading of something the world does not have is refused, rather than
    /// leaving a finding about nothing.
    #[test]
    fn a_reading_of_nothing_is_refused() {
        let h = vault();
        for tool in [
            "structure_lay_out_scenes",
            "structure_find_the_slack",
            "record_tidy_index",
        ] {
            let out = perform(&h, "m1", &act(tool, json!({"what":"a thing nobody has"})));
            assert!(!out.happened(), "{tool} reported work on nothing");
        }
        let out = perform(
            &h,
            "m1",
            &act(
                "structure_test_the_want",
                json!({"what":"a thing nobody has","scene":"x"}),
            ),
        );
        assert!(!out.happened(), "{out:?}");
    }

    /// **A fault is raised in a machine the vault has**, by its own name, and
    /// nothing else reaches the board.
    #[test]
    fn a_fault_names_a_machine_the_vault_has() {
        let h = vault();
        let name = h
            .sim(|s| s.devices.iter().next().map(|d| d.name.clone()))
            .expect("the vault has machines");
        let orders = |h: &Hosted| h.sim(|s| s.ledger.clone());
        let before = orders(&h);
        for (tool, second) in [("plant_raise_fault", "why"), ("plant_note_drift", "drift")] {
            let out = perform(
                &h,
                "m1",
                &act(
                    tool,
                    json!({"what":"The breaker panel is closed", second:"x"}),
                ),
            );
            assert!(!out.happened(), "{tool}: {out:?}");
            assert!(format!("{out:?}").contains("`what` is the machine, by its name: "));
        }
        assert_eq!(orders(&h), before, "nothing reached the board");
        let out = perform(
            &h,
            "m1",
            &act(
                "plant_raise_fault",
                json!({"what": format!("the {}", name.trim_start_matches("the ")), "why":"it hums"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        assert!(format!("{out:?}").contains(&format!(
            "Raised, and you may be wrong in public: {name} — it hums"
        )));
    }

    #[test]
    fn every_station_and_bench_act_is_dispatched() {
        for t in crate::engine::station::STATION_ACTS
            .iter()
            .chain(crate::engine::bench::BENCH_ACTS)
            .chain(crate::engine::mission_acts::MISSION_ACTS)
        {
            assert!(
                is_mine(t.name),
                "`{}` is declared and not dispatched",
                t.name
            );
        }
    }

    #[test]
    fn writing_takes_the_thing_and_appends_to_it() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"}),
            ),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let i = s.record.by_name("the third silence").expect("a gap");
            assert!(i.body.contains("night at the gate"), "{}", i.body);
            assert_eq!(i.holder.as_deref(), Some("m1"));
        });
    }

    /// The subject is the preposition, never `what`.
    ///
    /// `story_draft` carries the gap in `for` and the prose in `what`. A
    /// generic "first argument that looks like a subject" read `what` first and
    /// wrote the draft into a thing named after the draft's own text — then
    /// reported success, which is the failure that does not announce itself.
    #[test]
    fn the_thing_written_into_is_the_one_the_preposition_names() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"}),
            ),
        );
        h.sim(|s| {
            assert!(
                s.record.by_name("a night at the gate").is_none(),
                "wrote into the prose"
            );
            assert!(!s
                .record
                .by_name("the third silence")
                .unwrap()
                .body
                .is_empty());
        });
    }

    #[test]
    fn a_second_maker_cannot_write_into_work_somebody_is_holding() {
        let h = vault();
        perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"mine"}),
            ),
        );
        let out = perform(
            &h,
            "m2",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"mine too"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("m1"), "{out:?}");
    }

    /// A filed thing is changed by making a change and warning whoever it
    /// breaks, not by writing over it where it stands.
    #[test]
    fn a_filed_thing_is_not_written_over() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act(
                "chronicle_add_entry",
                json!({"to":"the third era","what":"…"}),
            ),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("filed"), "{out:?}");
    }

    #[test]
    fn a_commit_that_collides_names_the_other_party_rather_than_failing_blankly() {
        let h = vault();
        // m1 opens it; m2 tries to take it and is told who has it.
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third era"}))
        )
        .happened());
        let out = perform(
            &h,
            "m2",
            &act("bench_branch", json!({"what":"the third era"})),
        );
        assert!(!out.happened());
        assert!(
            out.line().unwrap().contains("m1"),
            "the holder was not named: {out:?}"
        );
    }

    #[test]
    fn the_working_loop_runs_branch_to_commit() {
        let h = vault();
        assert!(perform(
            &h,
            "m1",
            &act("bench_branch", json!({"what":"the third silence"}))
        )
        .happened());
        assert!(perform(
            &h,
            "m1",
            &act(
                "story_draft",
                json!({"for":"the third silence","what":"a night at the gate"})
            )
        )
        .happened());
        assert!(perform(&h, "m1", &act("bench_stage", json!({}))).happened());
        let out = perform(
            &h,
            "m1",
            &act("bench_commit", json!({"why":"filled the longest silence"})),
        );
        assert!(out.happened(), "{out:?}");
        h.sim(|s| {
            let i = s.record.by_name("the third silence").unwrap();
            assert_eq!(i.state, State::Filed);
            assert!(i.holder.is_none(), "filing did not release it");
        });
    }

    #[test]
    fn letting_something_go_without_a_reason_is_refused() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act("record_let_go", json!({"what":"the third era"})),
        );
        assert!(!out.happened());
        assert!(out.line().unwrap().contains("reason"), "{out:?}");
    }

    #[test]
    fn settling_needs_the_other_side_named() {
        let h = vault();
        let alone = perform(
            &h,
            "m1",
            &act(
                "chronicle_settle_boundary",
                json!({"between":"the third era"}),
            ),
        );
        assert!(!alone.happened());
        assert!(alone.line().unwrap().contains("two"), "{alone:?}");

        let both = perform(
            &h,
            "m1",
            &act(
                "chronicle_settle_boundary",
                json!({"between":"the third era","and":"the fourth era"}),
            ),
        );
        assert!(both.happened(), "{both:?}");
    }

    #[test]
    fn blame_says_where_the_chain_goes_quiet() {
        let h = vault();
        let out = perform(
            &h,
            "m1",
            &act("bench_blame", json!({"what":"the third era"})),
        );
        assert!(out.line().unwrap().contains("quiet"), "{out:?}");

        perform(
            &h,
            "m1",
            &act(
                "record_write_provenance",
                json!({"of":"the third era","from":"the western intake"}),
            ),
        );
        let then = perform(
            &h,
            "m1",
            &act("bench_blame", json!({"what":"the third era"})),
        );
        assert!(then.line().unwrap().contains("western intake"), "{then:?}");
    }
}

//! Tests the mission lifecycle: giving, gating and reviewing operations, and when
//! a report of done or stuck is taken.
//!
//! A mission is given once to a free body, and a finished one reported stuck is
//! filed done with what was found on it. A mission that writes a record is done
//! only when its reporter has committed the document, and an unsigned critical step
//! blocks the report. A draft rejected in review goes back to its writer, and a
//! rejection must quote the draft's own words.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::engine::mission::{Origin, Todo};

    use crate::engine::mission_gen::world::Draw;

    use crate::engine::survey;

    use npc_map::world::Where;

    use serde_json::json;

    use super::super::tests_documents::tests::{act, vault_with_documents};

    /// **A piece under its floor is told how far short it is**, and to add to
    /// what is there — not that it is a sketch, which Makers read as a verdict
    /// on the writing and answered by resubmitting it, or cutting it shorter.
    #[test]
    fn a_short_piece_is_told_the_gap_and_to_add_to_it() {
        let why = short_by("layers/life/zen/2491 The Silence.md", 218, 250, None);
        assert!(
            why.starts_with(
                "layers/life/zen/2491 The Silence.md is 218 of the 250 words asked for — about \
                 32 more."
            ),
            "{why}"
        );
        assert!(why.contains("keep what is there and add to it"), "{why}");
        assert!(!why.contains("sketch"), "{why}");
    }

    /// **And it is told what the piece is to hold.** A Maker short of its
    /// length rewrote the same words, sure there was nothing more to say; the
    /// brief's own account of what happens is the scene to add from.
    #[test]
    fn a_short_piece_is_told_what_happens_in_it() {
        let brief = "Write Zen's life where the record has nothing: the year 2491.\n\n\
                     What happens: Zen walks the archive at the end of a maintenance cycle and \
                     finds one log entry nobody wrote.\n\nIt must agree with: The Awakening.";
        let happens = what_happens(brief);
        assert_eq!(
            happens.as_deref(),
            Some(
                "Zen walks the archive at the end of a maintenance cycle and finds one log \
                 entry nobody wrote."
            )
        );
        let why = short_by("layers/life/zen/2491 The Silence.md", 127, 250, happens);
        assert!(
            why.ends_with(
                "What happens in it, as your brief puts it: Zen walks the archive at the end of \
                 a maintenance cycle and finds one log entry nobody wrote. Write that, moment by \
                 moment."
            ),
            "{why}"
        );
        assert_eq!(what_happens("A brief with no such paragraph."), None);
    }

    /// The vault with two Makers on the chronicle, its table handing out the
    /// bank's routines — so `take_up` gives a free Maker a reading to work.
    pub(in super::super) fn vault() -> Hosted {
        let h = Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the vault must load");
        h.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::Routines));
        h.with(|w| {
            w.enter(
                "m1",
                "Perrin Vastwood",
                Where::new("vault-chronicle", "early-range"),
            )
            .unwrap();
            w.enter(
                "m2",
                "Orion Vance",
                Where::new("vault-chronicle", "early-range"),
            )
            .unwrap();
        });
        h
    }

    /// **Work with no operation behind it, every step done, is filed done when
    /// it is reported stuck** — with what was seen on it as its answer —
    /// rather than refused with "report it done", which three characters
    /// went between for hours.
    #[test]
    fn a_finished_routine_reported_stuck_is_filed_done() {
        let h = vault();
        let mut m = Mission::new(
            "Go to the plant room and read the breaker panel.",
            vec![Todo::report("go back to the table and report it")],
            Origin::Random {
                routine: "read-a-station".into(),
            },
        );
        m.observed
            .push("in the plant room, the breaker panel: closed".into());
        h.with_sim(|s| s.missions.assign("m1", m));
        match perform(
            &h,
            "m1",
            &act(
                "report_stuck",
                json!({"why": "I reported it already and the table still shows it"}),
            ),
        ) {
            Outcome::Did(line) => assert!(line.starts_with("Reported done"), "{line}"),
            other => panic!("a finished routine was not filed: {other:?}"),
        }
        assert!(!h.sim(|s| s.missions.is_on_mission("m1")));
        // Its words first, what the engine saw beside them as evidence.
        assert_eq!(
            h.sim(|s| s.missions.done("m1").and_then(|m| m.answer.clone()))
                .as_deref(),
            Some(
                "I reported it already and the table still shows it (seen: in the plant room, \
                 the breaker panel: closed)"
            )
        );
    }

    /// The operation [`on_operation`] opens: the first, so always 1.
    pub(in super::super) const OP: u64 = 1;

    /// Both Makers on a step of one operation, [`OP`], whose store takes their
    /// commits (`docs/replicated_substrate.md` §10.3) — held in memory, as a
    /// disconnected world's is, so no substrate is needed.
    pub(in super::super) fn on_operation(h: &Hosted, root: &std::path::Path) {
        use std::collections::BTreeMap;

        use crate::engine::mission::{Mission, Origin, Todo};
        use crate::opstore::MindStores;
        use crate::sim::operations::tests::workflows;
        let stores = MindStores::new(Some(root));
        stores.set_quiet(Box::new(|_| true));
        h.set_op_stores(stores);
        h.with_sim(|s| {
            s.missions.set_workflows(workflows());
            let id = s
                .missions
                .launch(
                    "story",
                    "untold",
                    "era:layers/eras/third.md",
                    1,
                    "the bench's work",
                    "layers/stories/the-bench.md",
                    "What happens: work at the bench.",
                    BTreeMap::new(),
                    Vec::new(),
                )
                .unwrap();
            assert_eq!(id, OP);
            for body in ["m1", "m2"] {
                s.missions.assign(
                    body,
                    Mission::new(
                        "Work at the bench.",
                        vec![Todo::new("work at the bench")],
                        Origin::Generated {
                            generator: "untold".into(),
                            target: "era:layers/eras/third.md".into(),
                            operation: id,
                            step: "write".into(),
                        },
                    ),
                );
            }
        });
    }

    /// What `key` holds in operation [`OP`]'s store.
    pub(in super::super) fn stored(h: &Hosted, key: &str) -> Option<String> {
        h.op_shelf(OP)?.read(key)
    }

    /// **A mission's commit carries its own document and nothing else**: an
    /// edit left in the working set from before stays open, uncommitted, and
    /// the commit says so.
    #[test]
    fn a_missions_commit_carries_only_its_document() {
        use crate::engine::mission::{Mission, Origin, Todo, Work};
        let (h, _root) = vault_with_documents("commit-only");
        let path = "layers/stories/the-water-schedule.md";
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Tell the story of the water schedule.",
                    vec![
                        Todo::new(format!("write {path} and commit it")),
                        Todo::report("go back to the table and report it"),
                    ],
                    Origin::Generated {
                        generator: "untold".into(),
                        target: "era:layers/eras/third.md".into(),
                        operation: OP,
                        step: "write".into(),
                    },
                )
                .with_work(Work {
                    writes: path.into(),
                    reads: vec![],
                    min_words: 0,
                    edit_optional: false,
                    anew: false,
                    checks: Vec::new(),
                    tools: Vec::new(),
                }),
            )
        });
        // Left in the working set from earlier work.
        let shelf = h.shelf_of("m1");
        h.with_sim(|s| {
            s.bench
                .write("m1", "x", "layers/eras/third.md", "stray\n", &*shelf)
                .unwrap()
        });
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": path, "content": "# The Water Schedule\n\nThe clerks finished it.\n"})
            ),
        )
        .happened());
        let committed = perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why": "the story of the water schedule"}),
            ),
        );
        let line = committed.line().unwrap().to_string();
        assert!(
            line.contains(
                "Still open, and not part of this commit: layers/eras/third.md (changed)."
            ),
            "{line}"
        );
        assert!(stored(&h, path).is_some());
        assert_ne!(
            stored(&h, "layers/eras/third.md").as_deref(),
            Some("stray\n")
        );
        assert_eq!(
            h.sim(|s| s.bench.diff("m1")),
            ["layers/eras/third.md (changed)"]
        );
    }

    /// **A mission's own piece is never taken out by the one carrying it**: it
    /// is mended in place, and the refusal says so.
    #[test]
    fn a_missions_own_piece_cannot_be_deleted() {
        use crate::engine::mission::{Mission, Origin, Todo, Work};
        let (h, _root) = vault_with_documents("keep-piece");
        let path = "layers/stories/the-water-schedule.md";
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Mend the story of the water schedule.",
                    vec![Todo::new(format!("change {path} and commit it"))],
                    Origin::Generated {
                        generator: "untold".into(),
                        target: "era:layers/eras/third.md".into(),
                        operation: OP,
                        step: "review".into(),
                    },
                )
                .with_work(Work {
                    writes: path.into(),
                    reads: vec![],
                    min_words: 0,
                    edit_optional: false,
                    anew: false,
                    checks: Vec::new(),
                    tools: Vec::new(),
                }),
            )
        });
        match perform(&h, "m1", &act("file_delete", json!({ "path": path }))) {
            Outcome::Refused(why) => {
                assert!(why.contains("is the piece your mission is for"), "{why}")
            }
            other => panic!("the mission's own piece was taken out: {other:?}"),
        }
    }

    /// **A mission that writes the record is reported done only once its
    /// document is committed by the one reporting** — whatever its steps say.
    /// A story was reported done with its write step struck "thwarted" on the
    /// character's word and no document anywhere. Committing it also signs the
    /// write step off and says what is next.
    #[test]
    fn a_mission_that_writes_the_record_is_done_when_its_document_is_committed() {
        use crate::engine::mission::{Mission, Origin, StepOutcome, Todo, Work};
        let (h, root) = vault_with_documents("gate");
        let path = "layers/stories/the-water-schedule.md";
        let mission = || {
            Mission::new(
                "Tell the story of the water schedule.",
                vec![
                    Todo::new(format!("write {path} and commit it")),
                    Todo::report("go back to the table and report it"),
                ],
                Origin::Generated {
                    generator: "untold".into(),
                    target: "era:layers/eras/third.md".into(),
                    operation: OP,
                    step: "write".into(),
                },
            )
            .with_work(Work {
                writes: path.into(),
                reads: vec![],
                min_words: 6,
                edit_optional: false,
                anew: false,
                checks: Vec::new(),
                tools: Vec::new(),
            })
        };
        h.with_sim(|s| s.missions.assign("m1", mission()));
        let report = || {
            perform(
                &h,
                "m1",
                &act(
                    "report_done",
                    json!({"found": "I wrote the story of the water schedule."}),
                ),
            )
        };

        // Struck on its word, with nothing on the record: still refused.
        h.with_sim(|s| {
            s.missions.check_off(
                "m1",
                &format!("write {path} and commit it"),
                StepOutcome::Thwarted,
            )
        });
        match report() {
            Outcome::Refused(why) => assert!(why.contains("is not committed"), "{why}"),
            other => panic!("reported done with nothing written: {other:?}"),
        }

        // Another document is not this mission's to write.
        match perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": "layers/eras/third.md", "content": "new\n"}),
            ),
        ) {
            Outcome::Refused(why) => assert!(
                why.starts_with(&format!("Your mission writes {path} and nothing else.")),
                "{why}"
            ),
            other => panic!("a mission wrote outside its document: {other:?}"),
        }
        // Its own document under other capitals is another document.
        let shouted = path.replace(".md", ".Md");
        assert!(matches!(
            perform(
                &h,
                "m1",
                &act("file_write", json!({"path": shouted, "content": "x\n"})),
            ),
            Outcome::Refused(_)
        ));

        // Committed short of the floor: refused while it can still be
        // finished.
        h.with_sim(|s| s.missions.assign("m1", mission()));
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": path, "content": "# The Water Schedule\n"})
            ),
        )
        .happened());
        match perform(&h, "m1", &act("bench_commit", json!({"why": "a start"}))) {
            Outcome::Refused(why) => assert!(why.contains("is 4 of the 6 words"), "{why}"),
            other => panic!("a sketch was committed: {other:?}"),
        }
        assert!(!root.join(path).is_file(), "nothing reached the disk");

        // Written whole and committed: the commit says the work is done, and
        // the report is taken.
        assert!(perform(
            &h,
            "m1",
            &act(
                "file_write",
                json!({"path": path, "content": "# The Water Schedule\n\nThe clerks finished it.\n"})
            ),
        )
        .happened());
        let committed = perform(
            &h,
            "m1",
            &act(
                "bench_commit",
                json!({"why": "the story of the water schedule"}),
            ),
        );
        assert!(
            committed
                .line()
                .unwrap()
                .contains("last step of your mission"),
            "{committed:?}"
        );
        assert!(stored(&h, path).is_some());
        assert!(
            !root.join(path).is_file(),
            "only a publish reaches the record"
        );
        // **Finished work cannot be reported stuck.** It used to be taken, and
        // a stuck report failed the operation and threw the draft away.
        match perform(
            &h,
            "m1",
            &act(
                "report_stuck",
                json!({"why": "the mission is done, but I cannot report it"}),
            ),
        ) {
            Outcome::Refused(why) => {
                assert!(why.starts_with("Nothing here is stuck"), "{why}");
                assert!(
                    why.contains(&format!("{path} is written and committed")),
                    "{why}"
                );
                assert!(why.contains("`report_done`"), "{why}");
            }
            other => panic!("finished work was reported stuck: {other:?}"),
        }
        let stuck = || {
            perform(
                &h,
                "m1",
                &act(
                    "report_stuck",
                    json!({"why": "the story stands, but I cannot file it"}),
                ),
            )
        };
        assert!(
            matches!(stuck(), Outcome::Refused(ref why) if why.starts_with("Nothing here is stuck")),
            "the second is turned away too"
        );
        // **Told twice, the third is the done report it matches.**
        match stuck() {
            Outcome::Did(line) => assert!(line.starts_with("Reported done"), "{line}"),
            other => panic!("a third stuck report on finished work was not filed done: {other:?}"),
        }
        let answer = h
            .sim(|s| s.missions.done("m1").and_then(|m| m.answer.clone()))
            .expect("the mission is filed done");
        assert!(
            answer.starts_with("the story stands, but I cannot file it"),
            "filed in the carrier's words: {answer}"
        );
    }

    /// **An operation runs its workflow from `missions.yaml` at the bench: its
    /// draft is held to the step's checks at its report; the table's reading
    /// sends it to a review by somebody else, who may reject it only with the
    /// draft's own words, and a rejected draft goes to be fixed.** Every
    /// mission here is built from the workflow's own steps.
    #[test]
    fn an_operation_is_gated_reviewed_by_another_and_rejected_back_to_a_fix() {
        use std::collections::BTreeMap;

        use crate::engine::mission::bank::Facts;
        use crate::engine::mission_gen::corpus::Corpus;
        use crate::engine::mission_gen::step::step_mission;
        use crate::sim::operations::tests::workflows;
        let (h, root) = vault_with_documents("operation");
        h.with_sim(|s| {
            s.missions.cancel("m1");
            s.missions.cancel("m2");
        });
        let path = "layers/stories/the-water-schedule.md";
        let fields: BTreeMap<String, String> = [
            ("title", "The Water Schedule"),
            ("when", "the first dry season"),
            ("where", "the cistern hall"),
            ("who", "the clerks"),
            ("happens", "The clerks count the cisterns before dawn."),
            ("turns", "A clerk strikes a cistern from the count."),
            ("quote", "the third era"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let id = h.with_sim(|s| {
            s.missions.set_workflows(workflows());
            s.missions
                .launch(
                    "story",
                    "untold",
                    "era:layers/eras/third.md",
                    1,
                    "the water schedule",
                    path,
                    "What happens: The clerks count the cisterns before dawn.",
                    fields,
                    Vec::new(),
                )
                .unwrap()
        });
        // Each Maker step set on the table as the engine's loop sets it.
        let put_up = || {
            let corpus = Corpus::read(
                &std::sync::Arc::new(crate::mind::files::MindFiles::folder(&root)),
                "creators-vault",
            );
            h.with_sim(|s| {
                let ops = s.missions.operations();
                let offer = ops.offer(id).unwrap();
                let m = step_mission(ops.get(id).unwrap(), &offer, &corpus, None).unwrap();
                s.missions.offer_step(id, m);
            });
        };
        put_up();
        h.with_sim(|s| {
            let drew = s.missions.collect("m1", &Facts::default()).unwrap();
            assert_eq!(drew.operation(), Some((id, "write")));
            let w = drew.work.as_ref().unwrap();
            assert!(w.anew && !w.edit_optional && w.tools.is_empty());
        });
        // Four hundred words, no phrase said twice, and no heading.
        let prose: String = (0..50)
            .map(|i| format!("Clerk{i} tallied cistern{i} beside wall{i} before dawn{i} broke."))
            .collect::<Vec<_>>()
            .chunks(5)
            .map(|c| c.join(" "))
            .collect::<Vec<_>>()
            .join("\n\n");
        let write = |text: &str| {
            assert!(perform(
                &h,
                "m1",
                &act("file_write", json!({"path": path, "content": text}))
            )
            .happened());
            assert!(
                perform(&h, "m1", &act("bench_commit", json!({"why": "the story"}))).happened()
            );
        };
        let report = |who: &str| {
            perform(
                &h,
                who,
                &act(
                    "report_done",
                    json!({"found": "I wrote the story of the water schedule."}),
                ),
            )
        };
        // A sentence said twice is the writer's to mend.
        write(&format!(
            "{prose}\n\nThe clerks went home before the count was done. The clerks went home \
             before the count was done."
        ));
        match report("m1") {
            Outcome::Refused(why) => {
                assert!(why.contains("is not ready to stand in the record"), "{why}");
                assert!(why.contains("is said twice"), "{why}");
                assert!(
                    !why.contains("heading"),
                    "the missing heading is the engine's to add: {why}"
                );
            }
            other => panic!("an ungated draft was reported: {other:?}"),
        }
        // Refused, the draft is work to do again: the mission's next step is
        // its writing, not the report, so nothing tells the writer it is done.
        let next = h.sim(|s| {
            s.missions
                .active("m1")
                .and_then(|m| m.next_step())
                .map(|t| t.text.clone())
        });
        assert!(
            next.as_deref().is_some_and(|t| t.starts_with("write ")),
            "{next:?}"
        );
        // Reported again unchanged, it is told the fault again — what to
        // change — not only that its writing is still to do.
        for tool in ["report_done", "report_stuck"] {
            let args = match tool {
                "report_done" => json!({"found": "I wrote the story of the water schedule."}),
                _ => json!({"why": "I have written it and cannot report it."}),
            };
            match perform(&h, "m1", &act(tool, args)) {
                Outcome::Refused(why) => {
                    assert!(why.contains("is said twice"), "{tool}: {why}");
                    assert!(!why.contains("You have not done this yet"), "{tool}: {why}");
                }
                other => panic!("{tool} was taken over a fault: {other:?}"),
            }
        }
        // Mended, and still without a heading: the engine gives it its title.
        write(&prose);
        assert!(report("m1").happened(), "the mended draft passes the gate");
        let draft = || h.op_shelf(id).unwrap().read(path);
        let stood = draft().unwrap();
        assert!(stood.starts_with("# The Water Schedule\n\n"), "{stood}");
        assert!(
            !root.join(path).exists(),
            "only a publish reaches the record"
        );

        assert_eq!(
            h.sim(|s| s.missions.operations().awaiting_table()),
            vec![(id, "reading".to_string())],
            "the draft goes to the table's reading"
        );

        // The table found it sound; the review is set, and the writer cannot
        // draw it — the other Maker does.
        h.with_sim(|s| {
            s.missions
                .table_took(id, "sound", "The table's verdict: sound.")
                .unwrap();
        });
        put_up();
        h.with_sim(|s| {
            let writer_draws = s
                .missions
                .collect("m1", &Facts::default())
                .is_some_and(|m| m.operation().is_some());
            assert!(!writer_draws, "not its own draft");
            s.missions.cancel("m1");
            let reviewer = s.missions.collect("m2", &Facts::default()).unwrap();
            assert_eq!(reviewer.operation(), Some((id, "review")));
            assert!(reviewer.may_reject());
            assert!(reviewer.prompt.contains("The table found nothing to mend."));
        });
        // It reads what the step lists.
        let reads: Vec<String> = h.sim(|s| {
            s.missions
                .active("m2")
                .unwrap()
                .todo
                .iter()
                .filter_map(|t| t.text.strip_prefix("read ").map(str::to_string))
                .collect()
        });
        assert!(reads.iter().any(|r| r == path), "{reads:?}");
        for r in &reads {
            assert!(perform(&h, "m2", &act("file_read", json!({"path": r}))).happened());
        }
        assert!(matches!(
            perform(&h, "m2", &act("report_rejected", json!({"why": "bad"}))),
            Outcome::Refused(_)
        ));
        // A reason that points at nothing in the draft is refused.
        match perform(
            &h,
            "m2",
            &act(
                "report_rejected",
                json!({"why": "It counts the cisterns fifty times and nothing happens in it."}),
            ),
        ) {
            Outcome::Refused(why) => assert!(why.contains("does not quote"), "{why}"),
            other => panic!("an unquoted rejection was taken: {other:?}"),
        }
        let rejected = perform(
            &h,
            "m2",
            &act(
                "report_rejected",
                json!({"why": "\"Clerk7 tallied cistern7 beside wall7\" — and so on fifty times; \
                                nothing happens in it."}),
            ),
        );
        assert!(rejected.happened());
        assert!(
            format!("{rejected:?}").contains("it goes back to be fixed, with your reasons"),
            "{rejected:?}"
        );
        // **A rejected draft stays in the operation's store, to be fixed.**
        assert!(draft().is_some(), "the draft is what the fix mends");
        let step = h.sim(|s| {
            s.missions
                .operations()
                .get(id)
                .unwrap()
                .step()
                .map(str::to_string)
        });
        assert_eq!(step.as_deref(), Some("fix"));
    }

    /// **A scan that signs a reading off keeps what it read**, as an arrival
    /// does. Signed off with nothing kept, the report prompt had nothing to
    /// quote, and a Maker reported the previous mission's reading for the
    /// machine this one sent it to.
    #[test]
    fn a_scan_keeps_the_reading_it_signs_off() {
        let h = vault();
        assert!(take_up(&h, "m1").is_some());
        let (device, mode, at) = h.sim(|s| {
            let m = s.missions.active("m1").unwrap();
            let d = s
                .devices
                .iter()
                .find(|d| m.todo.iter().any(|t| t.text.contains(&d.name)))
                .expect("the first routine reads a machine");
            (d.name.clone(), d.mode.clone(), d.at.clone())
        });
        let (area, node) = at.split_once('/').unwrap();
        h.with(|w| w.place("m1", Where::new(area, node))).unwrap();
        assert!(h.sim(|s| s.missions.active("m1").unwrap().observed.is_empty()));

        assert!(survey::here(&h, "m1").happened());

        let observed = h.sim(|s| s.missions.active("m1").unwrap().observed.clone());
        assert!(
            observed
                .iter()
                .any(|o| o.ends_with(&format!(", {device}: {mode}"))),
            "{observed:?}"
        );
    }

    /// **At a desk already, a stuck report is told the act there, not a walk to
    /// it.**
    #[test]
    fn a_stuck_report_at_the_desk_is_told_the_act_there() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let h = vault();
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Write the story.",
                    vec![Todo::new("write layers/stories/y.md")],
                    Origin::Lodged { by: "op".into() },
                ),
            )
        });
        assert_eq!(
            perform(
                &h,
                "m1",
                &act(
                    "report_stuck",
                    json!({"why": "there is no desk to write it at"})
                )
            ),
            Outcome::Refused(
                "It is not stuck yet — You are at a desk, in the early range: write it with \
                 `compose`, then `bench_commit` it. Do that, then come back to the table and \
                 `invoke` its `report_done` with what you found."
                    .into()
            )
        );
    }

    /// **A mission whose text its brief gives is told `file_write` at the desk,
    /// not `compose`.**
    #[test]
    fn a_stuck_report_at_the_desk_with_given_text_is_told_file_write() {
        use crate::engine::mission::{Mission, Origin, Todo, Work};
        let h = vault();
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Write the template.",
                    vec![Todo::new("write layers/stories/y.md")],
                    Origin::Lodged { by: "op".into() },
                )
                .with_work(Work {
                    writes: "layers/stories/y.md".into(),
                    reads: vec![],
                    min_words: 0,
                    edit_optional: false,
                    anew: false,
                    checks: Vec::new(),
                    tools: vec!["file_write".into()],
                }),
            )
        });
        assert_eq!(
            perform(
                &h,
                "m1",
                &act(
                    "report_stuck",
                    json!({"why": "there is no desk to write it at"})
                )
            ),
            Outcome::Refused(
                "It is not stuck yet — You are at a desk, in the early range: `file_write` it \
                 exactly as your brief gives it, then `bench_commit` it. Do that, then come back \
                 to the table and `invoke` its `report_done` with what you found."
                    .into()
            )
        );
    }

    /// **A judgement does not let a report past the journey after it.**
    #[test]
    fn a_report_is_held_by_a_journey_behind_a_judgement() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let h = vault();
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Look in on the stacks.",
                    vec![
                        Todo::new("decide what the stacks might need"),
                        Todo::new("go to the stacks"),
                        Todo::report("go back to the table and report it"),
                    ],
                    Origin::Random {
                        routine: "x".into(),
                    },
                ),
            )
        });
        match perform(
            &h,
            "m1",
            &act(
                "report_done",
                json!({"found": "the stacks need nothing at all today"}),
            ),
        ) {
            Outcome::Refused(why) => {
                assert!(
                    why.starts_with("You have not done this yet: go to the stacks."),
                    "{why}"
                );
                assert!(why.contains("`move_to`"), "{why}");
            }
            other => panic!("a report past an unmade journey was taken: {other:?}"),
        }
    }

    /// **A mission with an unsigned critical step is not reported done**: the
    /// step is an authorizer's to sign (§15 of `docs/npc_conversations.md`).
    #[test]
    fn a_mission_waiting_on_a_signature_is_not_reported_done() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let h = vault();
        let mut lease = Todo::new("sign the lease");
        lease.critical = true;
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Rent the flat.",
                    vec![lease],
                    Origin::Lodged { by: "u_op".into() },
                ),
            )
        });
        match perform(
            &h,
            "m1",
            &act(
                "report_done",
                json!({"found": "the flat is ours and the keys are here"}),
            ),
        ) {
            Outcome::Refused(why) => assert_eq!(
                why,
                "This mission waits on an authorizer's yes to \"sign the lease\": it is not done \
                 until they sign it. Message them and ask."
            ),
            other => panic!("reported done past an unsigned step: {other:?}"),
        }
    }

    /// **An order is not handed to a body the world gives no orders.**
    #[test]
    fn an_order_is_not_handed_to_an_unordered_body() {
        let h = vault();
        h.with_sim(|s| {
            s.ledger.set_order("mind the drill", "m1", None);
            s.set_unordered("m2", true);
        });
        assert_eq!(
            perform(
                &h,
                "m1",
                &act(
                    "orders_hand_to",
                    json!({"what": "mind the drill", "to": "Orion Vance"})
                )
            ),
            Outcome::Refused(
                "Orion Vance takes no orders: work for Orion Vance is not handed over, it is \
                 asked for."
                    .into()
            )
        );
        assert!(h.sim(|s| s.unheld_orders().contains(&"mind the drill".to_string())));
        assert!(h.sim(|s| s.orders_held_by("m2")).is_empty());
    }

    /// **An order handed over by name is held by the body of that name** — the
    /// id every reader of the board matches — and refused for a name nobody
    /// answers to.
    #[test]
    fn an_order_handed_by_name_is_held_by_that_body() {
        let h = vault();
        h.with_sim(|s| {
            s.ledger.set_order("mind the drill", "m1", None);
        });
        let hand = |to: &str| {
            perform(
                &h,
                "m1",
                &act(
                    "orders_hand_to",
                    json!({"what": "mind the drill", "to": to}),
                ),
            )
        };
        assert_eq!(
            hand("Nobody Atall"),
            Outcome::Refused("There is nobody here called Nobody Atall.".into())
        );
        assert_eq!(
            hand("orion vance"),
            Outcome::Did("Orion Vance has it: mind the drill".into())
        );
        assert_eq!(h.sim(|s| s.orders_held_by("m2")), ["mind the drill"]);
        // Held, it is refused to anybody else, and the holder is named.
        assert_eq!(
            hand("Perrin Vastwood"),
            Outcome::Refused("Orion Vance is holding that one.".into())
        );
    }

    /// A mission given by the table, worked with its progress recorded, and
    /// reported — the whole loop through `perform`. `perform` does not gate on
    /// availability (the grammar does), so this drives the acts directly.
    #[test]
    fn a_mission_is_given_worked_and_reported() {
        let h = vault();
        // Nothing to report before one is given.
        assert!(matches!(
            perform(&h, "m1", &act("report_done", json!({"found":"nothing"}))),
            Outcome::Refused(_)
        ));

        // The table gives a mission (a bank routine here — the vault has no
        // records indexed) and it is the body's open one.
        assert!(take_up(&h, "m1").is_some());
        assert!(h.sim(|s| s.missions.is_on_mission("m1")));

        // One at a time: a body carrying a mission is given nothing more, so
        // one in progress is never discarded by drawing another.
        let brief = h.sim(|s| s.missions.active("m1").unwrap().mission_text());
        assert_eq!(take_up(&h, "m1"), None);
        assert_eq!(
            h.sim(|s| s.missions.active("m1").unwrap().mission_text()),
            brief
        );

        // The bank's first routine goes to a room and reads a machine there, and
        // a journey not made cannot be reported done: the desk says what is left
        // and the way to it, and that `report_stuck` is the honest close if it
        // cannot be made.
        match perform(
            &h,
            "m1",
            &act("report_done", json!({"found":"it all looks fine to me"})),
        ) {
            Outcome::Refused(why) => {
                assert!(
                    why.starts_with("You have not done this yet: go to "),
                    "{why}"
                );
                assert!(why.contains("`move_to`"), "{why}");
                assert!(why.contains("report_stuck"), "{why}");
            }
            other => panic!("a journey not made cannot be reported done, got {other:?}"),
        }
        // Going there and reading it signs both steps off.
        let room = h.sim(|s| {
            let step = &s.missions.active("m1").unwrap().todo[0].text;
            step.trim_start_matches("go to ").to_string()
        });
        let machines: Vec<String> = h.sim(|s| s.devices.iter().map(|d| d.name.clone()).collect());
        assert!(h.with_sim(|s| s.missions.arrived_in("m1", &room, "the command level")));
        assert!(h.with_sim(|s| s.missions.read_off("m1", &machines)));
        h.with_sim(|s| {
            s.missions
                .observe("m1", "in the plant room, the coolant valve: open")
        });

        // A name is not an account: turned back with what was seen.
        match perform(&h, "m1", &act("report_done", json!({"found":"Paxon Vael"}))) {
            Outcome::Refused(why) => {
                assert!(why.contains("is not an account of what you found"), "{why}");
                assert!(why.contains("the coolant valve: open"), "{why}");
            }
            other => panic!("a name is not an account, got {other:?}"),
        }

        // Reporting done closes it, frees the character, and files the answer so
        // an operator can still read it.
        assert!(perform(
            &h,
            "m1",
            &act(
                "report_done",
                json!({"found":"the ledger is two years out"})
            )
        )
        .happened());
        assert!(
            !h.sim(|s| s.missions.is_on_mission("m1")),
            "reporting frees the character for the next mission"
        );
        assert_eq!(
            h.sim(|s| s.missions.done("m1").unwrap().answer.clone()),
            Some("the ledger is two years out".to_string())
        );
    }

    /// **A writing step is not stuck while a desk can be reached**: the report,
    /// made away from one, is turned away with the way to one.
    #[test]
    fn a_writing_step_is_not_stuck_while_a_desk_can_be_reached() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let (h, _root) = vault_with_documents("stuck-at-desk");
        h.with(|w| w.place("m1", Where::new("vault-chronicle", "stacks")))
            .unwrap();
        h.with_sim(|s| {
            s.missions.assign(
                "m1",
                Mission::new(
                    "Tell the story of the water schedule.",
                    vec![
                        Todo::new("write layers/stories/the-water-schedule.md and commit it"),
                        Todo::report("go back to the table and report it"),
                    ],
                    Origin::Random {
                        routine: "x".into(),
                    },
                ),
            )
        });
        match perform(
            &h,
            "m1",
            &act(
                "report_stuck",
                json!({"why":"I keep arriving at the lift and cannot reach the desk"}),
            ),
        ) {
            Outcome::Refused(why) => {
                assert!(why.starts_with("It is not stuck yet — "), "{why}");
                assert!(why.contains("desk"), "{why}");
                assert!(why.contains("`move_to`"), "{why}");
            }
            other => panic!("a reachable desk is not stuck, got {other:?}"),
        }
    }

    /// **Stuck is turned away while the next step is within reach**, with where
    /// and how — and a character that never tries the way is never believed,
    /// however often it reports.
    #[test]
    fn stuck_is_refused_while_the_next_step_is_within_reach() {
        let h = vault();
        assert!(take_up(&h, "m1").is_some());
        for _ in 0..4 {
            match crate::engine::body::perform(
                &h,
                "m1",
                &act("report_stuck", json!({"why":"it will not answer"})),
            ) {
                Outcome::Refused(why) => {
                    assert!(why.starts_with("It is not stuck yet — "), "{why}");
                    assert!(why.contains("`move_to`"), "{why}");
                }
                other => panic!("a step never tried is not stuck, got {other:?}"),
            }
        }
        assert!(h.sim(|s| s.missions.is_on_mission("m1")));
    }

    /// **A turn-away counts once the world has refused what the character
    /// tried.** Twice told the way and twice refused on it, the third report is
    /// taken: that is a real block.
    #[test]
    fn stuck_is_believed_after_the_way_was_tried_and_refused_twice() {
        let h = vault();
        assert!(take_up(&h, "m1").is_some());
        let body = crate::engine::body::perform;
        let tried = || {
            let out = body(
                &h,
                "m1",
                &act("move_to", json!({"destination":"the far side of the moon"})),
            );
            assert!(
                matches!(out, Outcome::Refused(_)),
                "a place off the map is refused"
            );
        };
        let stuck = || {
            body(
                &h,
                "m1",
                &act("report_stuck", json!({"why":"it will not answer"})),
            )
        };
        for _ in 0..2 {
            tried();
            assert!(matches!(stuck(), Outcome::Refused(_)));
        }
        tried();
        assert!(
            stuck().happened(),
            "the third time, after trying, the report is taken"
        );
        assert!(!h.sim(|s| s.missions.is_on_mission("m1")));
    }
}

//! Tests for the command table: work given, reports taken, and drones at rest.
//!
//! Missions are checked against what the room offers rather than against a
//! coordinate: the table is recognised by its report act, and a character is only
//! given work when there is a way to a table to report at. Work posted for a free
//! character goes to the drone resting longest, and a suspended one waits. Missions
//! and working copies are checked to survive a restart.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::sim::shelf::Published;

    use super::super::tests_support::tests::{
        embody, embody_at, rt, vaulted, waiting, ROOMS, WORLD,
    };

    /// **A fresh world opens the command table by default, and it gives out
    /// what is asked for.** An untouched world comes up open, so a free
    /// character has no standing task; with nothing posted for it, it is at
    /// rest rather than handed a routine, and the moment work is posted the
    /// table gives it that. (Shutting it is durable and an operator's to do —
    /// see [`Self::host`].)
    #[tokio::test]
    async fn a_fresh_world_opens_the_command_table_by_default() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        assert_eq!(rt.nudge_for(1), None);
        let hosted = rt.hosted.get(WORLD).unwrap();
        rt.assign_work(&hosted);
        assert!(
            !hosted.sim(|s| s.missions.is_on_mission("m1")),
            "a routine was drawn"
        );
        assert!(rt.at_rest(&hosted, "m1"), "free, with nothing asked of it");
        hosted.with_sim(|s| {
            s.missions.post(
                Mission::new(
                    "Read the board.",
                    vec![Todo::report("report it")],
                    Origin::Lodged { by: "op".into() },
                ),
                None,
            )
        });
        rt.assign_work(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission("m1")));
    }

    /// **A Maker gone quiet on its mission is brought back by the world, not by
    /// its own turn.** The standing task used to be asked at the head of the
    /// character's turn — which only comes when something wakes it — and a
    /// character with a mission had none at all. A Maker at a quiet desk with
    /// nothing waking it stayed there. Now the moment asks, and a character
    /// quiet past the gate is told its mission's next step.
    #[tokio::test]
    async fn a_maker_gone_quiet_on_its_mission_is_told_its_next_step() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        rt.set_table_open(false);
        let mission = Mission::new(
            "Read the record 'the-charge' and check it against the storyline.",
            vec![Todo::new("go to the archives")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.assign("m1", mission));
        // Everything embodying queued is read, and the character goes quiet.
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert!(!rt.scheduler.due_now(1).contains(&1), "not quiet");

        // Not yet: a moment inside the quiet gate brings nothing.
        rt.set_clock(Arc::new(|_| IDLE_AFTER_MS / 2));
        rt.moment(&hosted);
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert!(!rt.scheduler.due_now(1).contains(&1), "told too soon");

        // Past it, the moment tells it what to do next, and that wakes it.
        rt.set_clock(Arc::new(|_| IDLE_AFTER_MS * 2));
        rt.moment(&hosted);
        let told = rt
            .scheduler
            .tick(1, 0, IDLE_AFTER_MS * 2, |_, _| Vec::new())
            .expect("it was woken");
        let read = told.perceived.join("\n");
        assert!(read.contains("archives"), "{read}");
    }

    /// **A step that sets the year is pointed at the machine that sets it.**
    /// Told only the step, Makers wandered between levels for an hour; the
    /// compass names the act, and the way to a time machine from where they
    /// stand.
    #[tokio::test]
    async fn the_compass_for_the_year_names_the_machine_and_the_way() {
        use crate::engine::mission::{time_step_text, Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let mission = Mission::new(
            "Write the year.",
            vec![Todo::new(time_step_text(2837))],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.assign("m1", mission));
        let told = rt.mission_compass(1).expect("a compass");
        assert!(
            told.starts_with("Your mission, next: set your time to 2837."),
            "{told}"
        );
        assert!(told.contains("The time machines are in"), "{told}");
        assert!(
            told.contains("`time_travel` naming the year 2837"),
            "{told}"
        );
    }

    /// **A body that moves by attention is pointed at `focus`, never the lift.**
    /// Zen has no `move_to` and no lift; told to ride the lift to the room its
    /// mission named, it decided the lift was locked and asked the Drones all
    /// night to override it.
    #[tokio::test]
    async fn the_compass_points_a_body_that_attends_at_focus() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).unwrap();
        let mission = Mission::new(
            "Read the board.",
            vec![Todo::new("go to the command room")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        hosted.with_sim(|s| s.missions.assign("m1", mission));
        let walks = rt.mission_compass(1).expect("a compass");
        assert_eq!(
            walks,
            "Your mission, next: go to the command room. It is in the command room, on the \
             command level: `move_to` the command room."
        );

        hosted.with_sim(|s| s.set_attends("m1", true));
        let attends = rt.mission_compass(1).expect("a compass");
        assert_eq!(
            attends,
            "Your mission, next: go to the command room. It is in the command room: `focus` on it."
        );
    }

    /// **A judgement about somebody is not a walk to them.** A step that only
    /// names a person to judge them is told it is settled in the mind, and not
    /// where they are; a step that goes to them still is.
    #[tokio::test]
    async fn a_judgement_about_somebody_does_not_send_the_body_after_them() {
        use crate::engine::mission::{Mission, Origin, Todo, JUDGEMENT_LINE};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "band-one");
        let hosted = rt.hosted.get(WORLD).unwrap();
        let assign = |step: &str| {
            let m = Mission::new(
                "Look in on Maker-02.",
                vec![
                    Todo::new(step),
                    Todo::report("go back to the table and report it"),
                ],
                Origin::Lodged {
                    by: "u_op".to_string(),
                },
            );
            hosted.with_sim(|s| s.missions.assign("m1", m));
        };
        assign("note what Maker-02 needs");
        assert_eq!(
            rt.mission_compass(1).expect("a compass"),
            format!("Your mission, next: note what Maker-02 needs. {JUDGEMENT_LINE}")
        );
        hosted.with_sim(|s| s.missions.cancel("m1"));
        assign("ask Maker-02 what they need");
        let asks = rt.mission_compass(1).expect("a compass");
        assert!(asks.contains("Maker-02 is in"), "{asks}");
    }

    /// **A step that posts work names the jobs it may post.** Told only "post
    /// work on the order table", Zen never saw the catalogue and invented a
    /// crisis a turn instead.
    #[tokio::test]
    async fn the_compass_to_post_work_names_the_standing_jobs() {
        use crate::engine::mission::{Mission, Origin, Todo};
        use crate::engine::mission_gen::world::{Draw, Routine};
        use npc_map::world::Where;
        let rt = vaulted();
        let hosted = rt.hosted.get(WORLD).unwrap();
        let table = Where::new("vault-command", "command-room");
        hosted.with(|w| w.enter("m1", "Zen".to_string(), table).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.embody(1, WORLD, "m1", 0).expect("bound");
        let job = |id: &str| Routine {
            id: id.into(),
            brief: "b".into(),
            steps: vec!["go to the ward room".into()],
            ..Default::default()
        };
        hosted.with_sim(|s| {
            s.missions.set_unordered("m1", true);
            s.missions.set_routines(
                vec![job("ward-patrol"), job("morning-briefing")],
                Draw::None,
            );
            s.missions.assign(
                "m1",
                Mission::new(
                    "Set the work.",
                    vec![Todo::new("post work on the order table")],
                    Origin::Lodged {
                        by: "u_op".to_string(),
                    },
                ),
            );
        });
        let told = rt.mission_compass(1).expect("a compass");
        assert!(
            told.starts_with(
                "Your mission, next: post work on the order table. Work is posted with `invoke` \
                 `http://local/"
            ),
            "{told}"
        );
        assert!(
            told.ends_with(
                "/order_post` — a standing `job` by name when one fits what the families need: \
                 `ward-patrol`, `morning-briefing`; or a `brief` and its `steps` for what none of \
                 them covers. A notice on a board is not work: nobody takes it up. The table holds \
                 as many missions nobody has taken as there are hands to take them; past that, a \
                 new one waits for a report."
            ),
            "{told}"
        );
    }

    /// **A carried mission is in the system prompt, not restated.** Given a
    /// mission, the persona a turn reads carries its ask, its steps and where it
    /// ends, and the quiet-turn nudge says nothing — neither the "nothing has
    /// been asked of you" default nor the mission's wording again.
    #[tokio::test]
    async fn a_character_on_a_mission_carries_it_in_its_prompt_and_is_not_nudged() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        // Shut the (now default-open) table so "no mission" reads as the
        // NO_MISSION default rather than a call to the table.
        rt.set_table_open(false);
        // No mission yet: the default stands.
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));

        // Give it a mission, and the standing task becomes the mission.
        let mission = Mission::new(
            "Read the record 'the-charge' and check it against the storyline.",
            vec![
                Todo::new("go to the archives"),
                Todo::new("read 'the-charge'"),
            ],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with_sim(|s| s.missions.assign("m1", mission));

        assert_eq!(rt.nudge_for(1), None, "the mission is not restated");
        let carried = rt
            .place_persona(1, OwnedPersona::default())
            .mission
            .expect("the persona carries the mission");
        let text = carried.text;
        assert!(text.starts_with("Read the record 'the-charge'"), "{text}");
        assert!(
            text.contains("[ ] go to the archives"),
            "the steps are shown: {text}"
        );
        assert!(
            text.contains("go back to the table where work is handed out, `scan` it")
                && text.contains("`report_done`"),
            "the mission points back to the table's device: {text}"
        );
        assert!(
            !text.contains("What has been asked of you"),
            "the prompt's own heading supplies the lead-in: {text}"
        );
    }

    /// **Calling a mission off tells the character.** The notice is queued only
    /// when there was a mission to withdraw, and the mission is gone either way.
    #[tokio::test]
    async fn cancelling_a_mission_tells_the_character_it_is_withdrawn() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let before = rt.scheduler.inbox_depth(1).unwrap();
        assert!(
            !rt.cancel_mission(1, 0),
            "nothing to cancel on a character with no mission"
        );
        assert_eq!(rt.scheduler.inbox_depth(1).unwrap(), before, "no notice");

        let mission = Mission::new(
            "Read the record.",
            vec![Todo::new("go to the archives")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with_sim(|s| s.missions.assign("m1", mission));
        assert!(rt.cancel_mission(1, 0));
        assert_eq!(
            rt.scheduler.inbox_depth(1).unwrap(),
            before + 1,
            "the withdrawal is queued"
        );
        assert!(!rt
            .hosted
            .get(WORLD)
            .unwrap()
            .sim(|s| s.missions.is_on_mission("m1")));
        assert!(
            !MISSION_WITHDRAWN.is_empty() && !MISSION_WITHDRAWN.contains("ledger"),
            "the notice names no subject"
        );
    }

    /// **Uncommitted work outlives a restart.** A body's working copy, written
    /// with the world's state on its beat, is open again in the world a fresh
    /// runtime hosts over the same substrate.
    #[tokio::test]
    async fn working_copies_are_written_and_restored_on_the_next_boot() {
        use crate::npcs::Npcs;

        let dir = std::env::temp_dir().join(format!(
            "npcd-benches-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mind = dir.join("mind");
        std::fs::create_dir_all(mind.join("layers/stories")).unwrap();
        let npcs = Npcs::load(&dir).unwrap();
        const DOC: &str = "layers/stories/the-count.md";

        let first = rt();
        first.set_substrate(npcs.substrate());
        let hosted = first
            .host(WORLD, Path::new(ROOMS))
            .expect("the vault loads");
        let published = Published::folder(mind.clone());
        hosted.with_sim(|s| {
            s.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
                &mind,
            ))));
            s.bench
                .write(
                    "m1",
                    "a story",
                    DOC,
                    "# The Count\n\nYou count the fleet.\n",
                    &published,
                )
                .unwrap();
        });
        first.beat(&hosted);

        let second = rt();
        second.set_substrate(npcs.substrate());
        let revived = second
            .host(WORLD, Path::new(ROOMS))
            .expect("the vault loads");
        let back = revived.with_sim(|s| {
            s.set_bench_root(crate::mind::Mind::new(Some(std::path::PathBuf::from(
                &mind,
            ))));
            s.bench.read("m1", DOC, &published)
        });
        assert_eq!(back.as_deref(), Ok("# The Count\n\nYou count the fleet.\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **A mission outlives a restart.** Written with the world's state on its
    /// beat, a world hosted by a fresh runtime over the same substrate comes up
    /// carrying the same open mission, its steps and the queue behind it.
    #[tokio::test]
    async fn missions_are_written_and_restored_on_the_next_boot() {
        use crate::engine::mission::bank::Facts;
        use crate::engine::mission::{Mission, Origin, Outcome, Todo};
        use crate::npcs::Npcs;

        let dir = std::env::temp_dir().join(format!(
            "npcd-missions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let npcs = Npcs::load(&dir).unwrap();
        let ask = |text: &str| {
            Mission::new(
                text,
                vec![Todo::new("go to the archives")],
                Origin::Random {
                    routine: "x".into(),
                },
            )
        };

        let first = rt();
        first.set_substrate(npcs.substrate());
        let hosted = first
            .host(WORLD, Path::new(ROOMS))
            .expect("the vault loads");
        hosted.with_sim(|s| {
            s.missions.assign("m1", ask("Read the record."));
            s.missions.lodge("m1", ask("Then the ledger."));
        });
        first.beat(&hosted);

        let second = rt();
        second.set_substrate(npcs.substrate());
        let revived = second
            .host(WORLD, Path::new(ROOMS))
            .expect("the vault loads");
        let open = revived.with_sim(|s| s.missions.active("m1").map(|m| m.prompt_text()));
        assert!(
            open.as_deref()
                .unwrap_or_default()
                .starts_with("Read the record."),
            "{open:?}"
        );
        let facts = Facts::default();
        let queued = revived.with_sim(|s| {
            s.missions.report("m1", Outcome::Pass, "ok", None);
            s.missions.collect("m1", &facts).unwrap().mission_text()
        });
        assert_eq!(
            queued, "Then the ledger.",
            "the queue behind it came back too"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── the command table: work given, and drones at rest ───────────────────

    /// **Standing at the command table is recognised by what the room offers,
    /// not by a coordinate.** `report_done` is stationed at the order table,
    /// so "am I at the table" is exactly "is that act among the ones this room
    /// hands me" — which is true in the command room and nowhere else.
    #[tokio::test]
    async fn a_character_is_at_the_table_only_where_a_mission_can_be_reported() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        // Arrives at the command room, where the order table stands.
        rt.embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap();
        assert!(
            rt.at_command_table(1),
            "the room the order table is stationed in is the table"
        );

        // A second character out in the casting wing is not at the table.
        embody(&rt, 2, "m2", "green-room");
        assert!(
            !rt.at_command_table(2),
            "a room with no order table is not the table"
        );
    }

    /// **An open table gives a free character its work where it stands, and a
    /// shut one gives none.** In a world that draws routines, a character away
    /// from the table with no mission is handed one — told with its brief,
    /// never called to fetch it — and one already carrying is handed nothing
    /// more.
    #[tokio::test]
    async fn the_open_table_gives_a_free_character_its_work_where_it_stands() {
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        // Out in the wing, away from the table.
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::Routines));
        let carrying = || hosted.sim(|s| s.missions.active("m1").map(|m| m.standing_text()));

        // Shut table: nothing is given. (The default is open, so this shuts it
        // first to exercise the closed→open transition.)
        rt.set_table_open(false);
        waiting(&rt, 1);
        rt.assign_work(&hosted);
        assert_eq!(carrying(), None, "the table is not open");
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));

        // Opened: it is given a routine, and told so with the brief.
        rt.set_table_open(true);
        assert_eq!(rt.nudge_for(1), None, "the table gives it its work");
        rt.assign_work(&hosted);
        let brief = carrying().expect("given a mission");
        assert_eq!(waiting(&rt, 1), [format!("{MISSION_GIVEN}\n{brief}")]);
        assert!(
            !rt.scheduler.is_resting(1),
            "a world that draws rests nobody"
        );

        // Carrying one, it is given nothing more.
        rt.assign_work(&hosted);
        assert_eq!(carrying(), Some(brief));
        assert!(waiting(&rt, 1).is_empty(), "told once");
    }

    /// **A drone with no mission is at rest, and only a mission wakes it**
    /// (`docs/zen_everywhere.md` §3). In a world that draws nothing, a drone
    /// with nothing posted for it is held: no standing task, no turn, and what
    /// is said to it waits. Work posted for nobody by name goes to the drone
    /// longest at rest, which wakes carrying it and reads its brief first;
    /// reported, with nothing more waiting, it comes to rest again.
    #[tokio::test]
    async fn a_drone_with_no_mission_rests_until_the_table_gives_it_one() {
        use crate::engine::event::{EventKind, Salience};
        use crate::engine::mission::{Mission, Origin, Outcome as Verdict, Todo};
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::None));
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        waiting(&rt, 1);
        waiting(&rt, 2);

        // The first comes to rest a moment before the second.
        rt.follow_rest(&hosted);
        assert!(rt.scheduler.is_resting(1) && rt.scheduler.is_resting(2));
        rt.scheduler.rest(2, false);
        std::thread::sleep(Duration::from_millis(5));
        rt.follow_rest(&hosted);
        assert!(rt.scheduler.resting_since(1) < rt.scheduler.resting_since(2));

        // At rest it has no standing task and takes no turn, whatever arrives.
        assert_eq!(rt.nudge_for(1), None);
        rt.standing_tasks(&hosted);
        rt.scheduler.deliver(
            1,
            0,
            Salience::URGENT,
            EventKind::Description {
                text: "A door slams.".into(),
            },
        );
        assert!(
            rt.scheduler.begin_tick(1).is_none(),
            "a drone at rest thought"
        );
        assert!(rt.scheduler.due_now(u64::MAX - 1).is_empty());

        // Work for whoever is free goes to the one longest at rest.
        let board = || {
            Mission::new(
                "Read the board.",
                vec![Todo::new("go to the dispatch room")],
                Origin::Lodged { by: "op".into() },
            )
        };
        hosted.with_sim(|s| s.missions.post(board(), None));
        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission("m1")));
        assert!(!hosted.sim(|s| s.missions.is_on_mission("m2")));
        assert!(!rt.scheduler.is_resting(1), "the mission woke it");
        assert!(rt.scheduler.is_resting(2), "nothing for the other");
        let read = waiting(&rt, 1);
        assert_eq!(read[0], "A door slams.", "what waited is read");
        assert!(
            read[1].starts_with(MISSION_GIVEN) && read[1].contains("Read the board."),
            "{read:?}"
        );

        // Reported with nothing more waiting, it rests again.
        hosted.with_sim(|s| {
            s.missions
                .report("m1", Verdict::Pass, "two jobs stalled", None)
        });
        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(rt.scheduler.is_resting(1));

        // Work set for one by name is that one's, however long another rested.
        hosted.with_sim(|s| s.missions.post(board(), Some("m2")));
        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission("m2")));
        assert!(!rt.scheduler.is_resting(2) && rt.scheduler.is_resting(1));

        // A shut table rests nobody: with no table to be given work by, a
        // character with no mission has the ordinary standing task.
        rt.set_table_open(false);
        rt.follow_rest(&hosted);
        assert!(!rt.scheduler.is_resting(1));
        assert!(rt.nudge_for(1).is_some());
    }

    /// **Work for a suspended drone waits for it.** It is no hand: a mission
    /// for whoever is free passes it by, and one set for it by name stays on
    /// the table until it is resumed.
    #[tokio::test]
    async fn work_is_not_given_to_a_suspended_drone() {
        use crate::engine::mission::{Mission, Origin, Todo};
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::None));
        embody(&rt, 1, "m1", "green-room");
        rt.scheduler.suspend(1, true);
        let watch = Mission::new(
            "Watch Earth.",
            vec![Todo::new("go to the observatory")],
            Origin::Lodged { by: "op".into() },
        );
        hosted.with_sim(|s| s.missions.post(watch, Some("m1")));
        rt.assign_work(&hosted);
        assert!(!hosted.sim(|s| s.missions.is_on_mission("m1")));
        assert!(hosted.sim(|s| !s.missions.is_hand("m1")));

        rt.scheduler.suspend(1, false);
        rt.assign_work(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission("m1")));
    }

    /// **A character carrying a mission can give it up from anywhere**: away
    /// from the table, its `report_stuck` is in reach and nothing else of the
    /// table's; carrying nothing, not even that.
    #[tokio::test]
    async fn a_mission_can_be_given_up_away_from_the_table() {
        use crate::engine::mission::{Mission, Origin, Todo};
        let rt = vaulted();
        embody(&rt, 2, "m2", "green-room");
        assert!(!rt.at_command_table(2), "it stands away from the table");
        let (hosted, body) = rt.body_of(2).unwrap();
        let acts = |rt: &Runtime| -> Vec<String> {
            rt.invokable_urls(&hosted, &body)
                .into_iter()
                .map(|i| i.act.to_string())
                .collect()
        };
        assert!(
            !acts(&rt)
                .iter()
                .any(|a| a == mission_acts::REPORT_STUCK.name),
            "carrying nothing, there is nothing to give up"
        );
        hosted.with_sim(|s| {
            s.missions.assign(
                &body,
                Mission::new(
                    "Land the template.",
                    vec![Todo::new("go to the watch room")],
                    Origin::Lodged {
                        by: "u_op".to_string(),
                    },
                ),
            )
        });
        let offered = acts(&rt);
        assert!(
            offered.iter().any(|a| a == mission_acts::REPORT_STUCK.name),
            "{offered:?}"
        );
        assert!(
            !offered.iter().any(|a| a == mission_acts::REPORT_DONE.name),
            "done is reported at the table: {offered:?}"
        );
    }

    /// **A post is handed to its carrier, and a report tells its poster**: one
    /// for whoever is free goes to one free hand and nobody else is told; one
    /// by name goes to its own; the poster hears of neither until the report.
    #[tokio::test]
    async fn a_post_is_handed_to_its_carrier_and_a_report_tells_its_poster() {
        use crate::engine::mission_gen::world::Draw;
        use crate::engine::order_table;
        let rt = vaulted();
        for id in [1, 3] {
            rt.scheduler.wake(id, 0, 0);
        }
        rt.embody_in_world(1, WORLD, Kind::default(), "Wick", None, 0)
            .unwrap();
        embody(&rt, 2, "m2", "green-room");
        let zen = Kind {
            unordered: true,
            ..Kind::default()
        };
        rt.embody_in_world(3, WORLD, zen, "Zen", None, 0).unwrap();
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::None));
        let (_, wick) = rt.body_of(1).unwrap();
        let (_, poster) = rt.body_of(3).unwrap();
        let depth = |id| rt.scheduler.inbox_depth(id).unwrap();
        let notify = || {
            rt.assign_work(&hosted);
            for n in hosted.with_sim(|s| s.missions.take_notices()) {
                rt.give_notice(&hosted, n);
            }
        };
        let before = (depth(1), depth(2), depth(3));
        let args = |v: serde_json::Value| v.as_object().unwrap().clone();
        assert!(order_table::post(
            &hosted,
            &poster,
            &args(
                serde_json::json!({"brief": "Read the board.", "steps": "go to the dispatch room"})
            )
        )
        .happened());
        notify();
        assert_eq!(
            (depth(1), depth(2), depth(3)),
            (before.0 + 1, before.1, before.2 + 1),
            "handed to the first hand, the mind that is the world told it was taken up, \
             and nobody else told"
        );
        assert!(hosted.sim(|s| s.missions.is_on_mission(&wick)));

        assert!(order_table::post(
            &hosted,
            &poster,
            &args(serde_json::json!({
                "brief": "Watch Earth.",
                "steps": "go to the observatory",
                "for": "Maker-02"
            })),
        )
        .happened());
        notify();
        assert_eq!(
            (depth(1), depth(2), depth(3)),
            (before.0 + 1, before.1 + 1, before.2 + 2)
        );

        hosted.with_sim(|s| {
            s.missions.report(
                &wick,
                Verdict::Pass,
                "two jobs have stalled on the board",
                None,
            );
        });
        notify();
        assert_eq!(depth(3), before.2 + 3, "the poster is told");
    }

    /// **A body that moves by attention `focus`es on any room at once**, and
    /// neither walks nor rides; one that walks has no rooms to focus on.
    #[tokio::test]
    async fn a_body_that_moves_by_attention_focuses_rather_than_walks() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let attends = Kind {
            attends: true,
            ..Kind::default()
        };
        rt.embody_in_world(1, WORLD, attends, "Warden", None, 0)
            .unwrap();
        embody(&rt, 2, "m2", "green-room");
        let within = rt.within(1);
        assert!(within.places.is_empty(), "{:?}", within.places);
        assert!(within.floors.is_empty() && !within.at_lift);
        assert!(
            within.rooms.iter().any(|r| r == "the green room"),
            "{:?}",
            within.rooms
        );
        assert!(
            !within.rooms.iter().any(|r| r == "the command room"),
            "not where it is"
        );
        assert!(rt.within(2).rooms.is_empty(), "a walker has none");

        let hosted = rt.hosted.get(WORLD).unwrap();
        let (_, body) = rt.body_of(1).unwrap();
        let act = |tool: &'static str, v: serde_json::Value| Act {
            tool,
            args: v.as_object().unwrap().clone(),
        };
        assert_eq!(
            body::perform(
                &hosted,
                &body,
                &act("focus", serde_json::json!({"on": "the green room"}))
            ),
            Outcome::Did("Your attention is on the green room.".into())
        );
        assert_eq!(hosted.place_of(&body), "vault-casting/green-room");
        assert!(!body::perform(
            &hosted,
            &body,
            &act(
                "move_to",
                serde_json::json!({"destination": "the command room"})
            )
        )
        .happened());
        assert!(
            !body::perform(&hosted, &body, &act("lift_call", serde_json::json!({}))).happened()
        );
        assert!(
            !within.rooms.iter().any(|r| r.starts_with("the lift")),
            "a landing is not a room to attend to: {:?}",
            within.rooms
        );
        let (_, walker) = rt.body_of(2).unwrap();
        assert!(!body::perform(
            &hosted,
            &walker,
            &act("focus", serde_json::json!({"on": "the command room"}))
        )
        .happened());
    }

    /// **A free Maker is handed the operation waiting on the open table** — and
    /// never the review of a draft it wrote: with only that waiting, in a world
    /// that draws nothing, it is at rest.
    #[tokio::test]
    async fn the_open_table_hands_a_free_maker_the_operation_waiting() {
        use crate::engine::mission::{Origin, Outcome as Verdict, Todo, Work};
        use crate::engine::mission_gen::world::Draw;
        use crate::sim::operations::tests::workflows;
        use std::collections::BTreeMap;
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        rt.embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap();
        rt.set_table_open(true);
        let hosted = rt.hosted.get(WORLD).unwrap();
        let (_, body) = rt.body_of(1).unwrap();
        // A step of the operation, as the engine's loop sets it on the table.
        let step = |prompt: &str, tools: Vec<String>, report: &str| {
            Mission::new(
                prompt,
                vec![Todo::new("write it"), Todo::report(report)],
                Origin::Generated {
                    generator: "life-event".into(),
                    target: "life:keeper".into(),
                    operation: 0,
                    step: String::new(),
                },
            )
            .with_work(Work {
                writes: "layers/life/keeper/2488 X.md".into(),
                reads: vec![],
                min_words: 0,
                edit_optional: false,
                anew: false,
                checks: Vec::new(),
                tools,
            })
        };
        let id = hosted.with_sim(|s| {
            s.missions.set_routines(Vec::new(), Draw::None);
            s.missions.set_workflows(workflows());
            let id = s
                .missions
                .launch(
                    "life-event",
                    "life-event",
                    "life:keeper",
                    1,
                    "Keeper's next year",
                    "layers/life/keeper/2488 X.md",
                    "What happens: Keeper waits.",
                    BTreeMap::new(),
                    Vec::new(),
                )
                .unwrap();
            s.missions.offer_step(
                id,
                step(
                    "Write the next year of Keeper's life.",
                    Vec::new(),
                    "report it",
                ),
            );
            id
        });
        waiting(&rt, 1);
        rt.assign_work(&hosted);
        let handed = waiting(&rt, 1);
        assert!(
            handed.len() == 1 && handed[0].starts_with(MISSION_GIVEN),
            "{handed:?}"
        );
        assert_eq!(
            hosted.sim(|s| s
                .missions
                .active(&body)
                .and_then(|m| m.operation().map(|(i, st)| (i, st.to_string())))),
            Some((id, "write".to_string()))
        );

        // Its own draft's review is not handed to it.
        hosted.with_sim(|s| {
            s.missions.report(&body, Verdict::Pass, "written", None);
            s.missions
                .table_took(id, "sound", "nothing to mend")
                .unwrap();
            let review = step("Review it.", vec!["report_rejected".into()], "report it");
            s.missions.offer_step(id, review);
        });
        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(!hosted.sim(|s| s.missions.is_on_mission(&body)));
        assert!(rt.scheduler.is_resting(1), "nothing it may take");
        assert_eq!(rt.at_table_report(1), None);

        // A reviewer whose reading is done is asked for its verdict, not for
        // what it found.
        let review = Mission::new(
            "Review it.",
            vec![Todo::report("report your verdict")],
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:keeper".into(),
                operation: id,
                step: "review".into(),
            },
        )
        .with_work(Work {
            writes: "layers/life/keeper/2488 X.md".into(),
            reads: vec![],
            min_words: 0,
            edit_optional: true,
            anew: false,
            checks: Vec::new(),
            tools: vec!["report_rejected".into()],
        });
        hosted.with_sim(|s| s.missions.assign(&body, review));
        assert_eq!(rt.at_table_report(1).as_deref(), Some(VERDICT_AT_THE_TABLE));
    }

    /// **A character back at the table with its mission's work done is told to
    /// report it**, every tick it stands there — and not before the work is
    /// done, not away from the table, and not once it has reported. Arriving in
    /// the room the mission sent it to signs that journey off and says what is
    /// next.
    #[tokio::test]
    async fn the_table_calls_for_the_report_once_the_work_is_done() {
        use crate::engine::mission::{Mission, Origin, Outcome as Verdict, Todo};
        let rt = vaulted();
        rt.set_table_open(true);
        let read_the_valve = || {
            Mission::new(
                "Read the coolant valve.",
                vec![
                    Todo::new("go to the plant room"),
                    Todo::new("scan the coolant valve and read what state it is in"),
                    Todo::report("go back to the table and report it"),
                ],
                Origin::Lodged {
                    by: "u_op".to_string(),
                },
            )
        };
        let hosted = rt.hosted.get(WORLD).unwrap();

        // In the plant room: the journey and the reading of the valve standing
        // there are signed off, once, with the reading named, and it is told the
        // report is what is left. Not at the table, so not called to report.
        embody_at(&rt, 1, "m1", Where::new("vault-command", "plant"));
        hosted.with_sim(|s| s.missions.assign("m1", read_the_valve()));
        let arrived = rt.mark_arrival(1).expect("the journey ends here");
        assert!(
            arrived.starts_with("You are in the plant room, and the coolant valve here reads "),
            "{arrived}"
        );
        assert!(arrived.contains(": that was the last step"), "{arrived}");
        assert!(hosted.sim(|s| s.missions.active("m1").unwrap().todo[1].done));
        assert_eq!(rt.mark_arrival(1), None, "signed off once");
        assert_eq!(rt.at_table_report(1), None, "not at the table");

        // At the table with the journey still to make: not called to report.
        embody_at(&rt, 2, "m2", Where::new("vault-command", "command-room"));
        hosted.with_sim(|s| s.missions.assign("m2", read_the_valve()));
        assert_eq!(rt.at_table_report(2), None, "the work is not done");

        // At the table with the work done: report it now.
        hosted.with_sim(|s| {
            s.missions
                .arrived_in("m2", "the plant room", "the command level");
            s.missions
                .read_off("m2", &["the coolant valve".to_string()]);
        });
        assert_eq!(rt.at_table_report(2).as_deref(), Some(REPORT_AT_THE_TABLE));

        // Reported, there is nothing to report, and nothing said at the table.
        hosted.with_sim(|s| s.missions.report("m2", Verdict::Pass, "ok", None));
        assert_eq!(rt.at_table_report(2), None);
    }

    /// **A Maker standing at a landing is offered the ride, and shown it.** Sent
    /// to the time level, Makers stood at the story level's lift and told each
    /// other there was no lift act there; this holds the grammar to offering
    /// `lift_use`, bound to the time level among the floors, with the act
    /// described whatever else ranks.
    #[tokio::test]
    async fn a_maker_at_a_landing_is_offered_the_ride_to_the_time_level() {
        use crate::engine::acts::LIFT_USE;
        use crate::engine::tools::{shown_wherever_offered, specs_within, Mode};
        let rt = vaulted();
        embody_at(&rt, 1, "m1", Where::new("vault-story", "core"));
        let within = rt.within(1);
        assert!(within.at_lift, "not on the landing");
        assert!(
            within.floors.iter().any(|f| f == "the time level"),
            "{:?}",
            within.floors
        );
        let specs = specs_within(Mode::Physical, &within);
        assert!(
            specs.iter().any(|s| s.name == "lift_use"),
            "{:?}",
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
        );
        assert!(shown_wherever_offered(&LIFT_USE));
    }

    /// **A room on another floor is one `move_to` away, and the grammar offers
    /// it.** Makers sent to the time level stood on its landing all evening
    /// without calling the car; the time rooms are now names `move_to` takes
    /// from anywhere in the vault, and the world rides the lift for them.
    #[tokio::test]
    async fn a_maker_is_offered_the_rooms_of_another_floor_to_move_to() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let within = rt.within(1);
        let hosted = rt.hosted.get(WORLD).unwrap();
        let time_rooms: Vec<String> = hosted.read(|w| {
            let here = w.actor("m1").unwrap().at.clone();
            body::destinations_from(w, &here)
                .into_iter()
                .filter(|(_, p)| {
                    w.map()
                        .get(&p.area)
                        .is_some_and(|a| a.name == "the time level")
                })
                .map(|(name, _)| name)
                .collect()
        });
        assert!(
            !time_rooms.is_empty(),
            "no room on the time level is offered"
        );
        for name in &time_rooms {
            assert!(
                within.places.contains(name),
                "`{name}` is not in the grammar: {:?}",
                within.places
            );
        }
    }

    /// **A character is only given work it has a table to report at.** The
    /// waste is open ground with no door, stair or lift to any building, so a
    /// character out there has no way to a command table: a world that draws
    /// gives it no routine, and it keeps the ordinary standing task. One in
    /// the vault, however far from the table, can ride to it and is given work
    /// — as is one in the Redoubt, which has a table of its own in the muster
    /// hall.
    #[tokio::test]
    async fn no_work_is_given_to_a_character_with_no_way_to_a_table() {
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        embody_at(&rt, 1, "m1", Where::new("the-waste", "ruins"));
        embody(&rt, 2, "m2", "green-room");
        embody_at(&rt, 3, "m3", Where::new("tower-redoubt", "gatehouse"));
        rt.set_table_open(true);
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::Routines));

        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(!hosted.sim(|s| s.missions.is_on_mission("m1")));
        assert!(!rt.scheduler.is_resting(1), "not the table's to rest");
        assert_eq!(rt.at_table_report(1), None, "not standing at a table");
        assert_eq!(
            rt.nudge_for(1).as_deref(),
            Some(NO_MISSION),
            "it keeps the ordinary standing task"
        );
        assert!(hosted.sim(|s| s.missions.is_on_mission("m2")));
        assert!(hosted.sim(|s| s.missions.is_on_mission("m3")));
    }
}

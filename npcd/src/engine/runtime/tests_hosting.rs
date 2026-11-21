//! Tests for hosting worlds, embodying characters and supervising their tasks.
//!
//! These cover a world being hosted, held and unhosted, a character embodied twice
//! being refused, a deleted character's loop actually stopping, and a body id that
//! a restart derives the same way. A map that does not load leaves the daemon
//! running, since an authoring mistake should not take the console down with it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        at, character, embody, rt, vaulted, wait_for, ROOMS, WORLD,
    };

    /// **A character suspended in its record takes no turn, whatever the
    /// scheduler's copy says.** The copy is brought to the records at install;
    /// left behind them — as a member that missed a sync leaves it — the turn's
    /// own look at the record holds the character, and the copy is put right.
    /// A running character is not held.
    #[test]
    fn a_character_suspended_in_its_record_is_held_at_its_turn() {
        let dir = std::env::temp_dir().join(format!("npcd-held-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let npcs = Npcs::load(&dir).unwrap();
        npcs.import(vec![character(1, "suspended"), character(2, "idle")])
            .unwrap();
        let rt = rt();
        rt.set_npcs(Arc::new(npcs));
        rt.scheduler.wake(1, 0, 0);
        rt.scheduler.wake(2, 0, 0);
        assert!(rt.scheduler.is_suspended(1), "installed with the cast");

        rt.scheduler.follow_suspended(HashSet::new());
        assert!(!rt.scheduler.is_suspended(1), "the copy left behind");
        assert!(rt.held_by_record(1));
        assert!(rt.scheduler.is_suspended(1), "the copy put right");
        assert!(!rt.held_by_record(2));
        assert!(!rt.scheduler.is_suspended(2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── the cast, supervised ────────────────────────────────────────────────

    /// The restart pacing: doubling to a ceiling, and a healthy stretch
    /// starting it over — the backoff observed to grow and then reset.
    #[test]
    fn a_restart_backoff_doubles_to_its_ceiling_and_a_healthy_run_resets_it() {
        let mut backoff = Backoff::new();
        let crashed_fast = Duration::from_secs(1);
        assert_eq!(backoff.after(crashed_fast), Duration::from_secs(1));
        assert_eq!(backoff.after(crashed_fast), Duration::from_secs(2));
        assert_eq!(backoff.after(crashed_fast), Duration::from_secs(4));
        for _ in 0..10 {
            backoff.after(crashed_fast);
        }
        assert_eq!(
            backoff.after(crashed_fast),
            Duration::from_secs(60),
            "the ceiling did not hold"
        );
        assert_eq!(
            backoff.after(Backoff::HEALTHY_AFTER),
            Duration::from_secs(1),
            "a healthy stretch did not start the backoff over"
        );
    }

    /// **Deleting a character cancels its loop, not just its supervisor.**
    ///
    /// The supervisor runs each attempt as its own spawned task, and aborting
    /// only the wrapper DETACHES that attempt: the parked loop lives on —
    /// holding its runtime and, mid-decode, its scheduler slot — while the
    /// deletion looks done, and the character's re-creation then runs beside
    /// the zombie. Both futures hold an `Arc<Runtime>` clone, so the count
    /// returning to its baseline is the proof both actually died. A
    /// multi-thread runtime, because the parked task has to genuinely run to
    /// be genuinely cancelled.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deleting_a_character_cancels_its_parked_loop() {
        let rt = rt();
        // An inbox with nothing due parks the loop in `wait_due` indefinitely
        // — the exact await the cancellation has to land on.
        rt.scheduler.wake(1, 0, 0);
        let baseline = Arc::strong_count(&rt);
        rt.spawn_character(1);
        wait_for(
            || Arc::strong_count(&rt) >= baseline + 2,
            "the supervisor and its attempt to hold the runtime",
        )
        .await;
        rt.stop_character(1);
        wait_for(
            || Arc::strong_count(&rt) == baseline,
            "the supervisor AND its attempt to be dropped",
        )
        .await;
        assert!(
            rt.tasks.lock().unwrap().is_empty(),
            "the roster kept a stopped character"
        );
    }

    /// **Everything this daemon writes goes under `--data`.**
    ///
    /// The conversation layer's `workspace_path` falls back to the *process
    /// working directory*, so a runtime that does not carry `--data` cannot pass
    /// it on and the redo log lands wherever the daemon was launched from. That
    /// happened: the character store sat correctly in `npcd/substrate` at 65 MB
    /// while the real log grew to **190 GB** in the repo root, invisible to the
    /// storage view, until it filled the disk and surfaced as a linker error.
    #[test]
    fn the_runtime_knows_where_the_daemon_writes() {
        let data = std::env::temp_dir().join("npcd-data-probe");
        let rt = Runtime::new(Mind::new(None), &data).unwrap();
        assert_eq!(
            rt.data, data,
            "the runtime cannot tell the engine where to write"
        );
        // And it is not the working directory, which is the fallback that caused
        // the whole problem.
        assert_ne!(
            rt.data,
            std::env::current_dir().unwrap_or_default(),
            "the data directory resolved to the process working directory"
        );
    }

    #[test]
    fn a_fresh_runtime_is_not_ready_and_has_no_minds() {
        let rt = rt();
        assert!(!rt.is_ready());
        assert!(rt.minds.read().unwrap().is_none());
        assert_eq!(rt.scheduler.population(), 0);
    }

    /// Stopping must be observable before the thread notices it, or a shutdown
    /// races the driver's next pass.
    #[test]
    fn stopping_is_visible_immediately() {
        let rt = rt();
        assert!(!rt.stopping());
        rt.stop();
        assert!(rt.stopping());
    }

    /// The clock is asked per character. Two characters in differently-paced
    /// worlds must not be handed the same instant — and a resolver called once
    /// per pass instead of once per character is exactly how that happens.
    #[test]
    fn the_world_clock_is_resolved_per_character() {
        let rt = Runtime::new(Mind::new(None), &std::env::temp_dir()).unwrap();
        // Before a clock is installed there is no time to report, and zero is
        // the only honest answer. The driver does not run in this window.
        assert_eq!(rt.world_ms(3), 0);

        rt.set_clock(Arc::new(|npc_id| npc_id * 1_000));
        assert_eq!(rt.world_ms(3), 3_000);
        assert_eq!(rt.world_ms(7), 7_000);
    }

    #[tokio::test]
    async fn a_hosted_world_is_reachable_and_moving() {
        let rt = rt();
        assert!(rt.hosted.get(WORLD).is_none(), "hosted before it was asked");

        rt.host(WORLD, Path::new(ROOMS)).expect("the vault loads");
        assert!(rt.hosted.get(WORLD).is_some());
        let running = rt.moments();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].0, WORLD);
        assert!(!running[0].2, "hosted paused");
    }

    #[test]
    fn a_map_that_does_not_load_leaves_the_daemon_running() {
        // An authoring mistake must not take the console and the whole
        // authored corpus down with it.
        let rt = rt();
        assert!(rt
            .host(WORLD, Path::new(env!("CARGO_MANIFEST_DIR")))
            .is_err());
        assert!(rt.hosted.get(WORLD).is_none());
        assert!(
            rt.moments().is_empty(),
            "a metronome outlived a failed load"
        );
    }

    #[tokio::test]
    async fn a_world_can_be_held_still_and_let_go() {
        let rt = vaulted();
        assert!(rt.moments()[0].2, "not paused");
        assert!(rt.hold_world(WORLD, false));
        assert!(!rt.moments()[0].2, "not resumed");
        assert!(
            !rt.hold_world("nowhere", true),
            "held a world it has not got"
        );
    }

    #[tokio::test]
    async fn embodying_the_same_character_twice_elsewhere_is_refused() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with(|w| w.enter("m2", "Maker-02", at("band-one")).unwrap());
        assert!(rt.embody(1, WORLD, "m2", 0).is_err());
        assert!(rt.embody(2, WORLD, "m1", 0).is_err(), "two minds, one body");
    }

    #[tokio::test]
    async fn embodying_into_a_world_or_a_body_that_is_not_there_is_refused() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let no_world = rt.embody(1, "elsewhere", "m1", 0).unwrap_err().to_string();
        assert!(no_world.contains("elsewhere"), "{no_world}");

        let no_body = rt.embody(1, WORLD, "nobody", 0).unwrap_err().to_string();
        assert!(no_body.contains("nobody"), "{no_body}");
        assert!(!rt.bodies.is_bound(1));
    }

    #[tokio::test]
    async fn disembodying_lets_a_character_settle_back_to_reacting() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        assert!(rt.disembody(1, 0));
        assert_eq!(rt.scheduler.pace_of(1), Some(Pace::AMBIENT));
        assert!(rt.body_of(1).is_none());
        assert!(!rt.disembody(1, 0), "disembodied twice");
    }

    /// **A deleted character leaves the world.** Its mind stopped and its
    /// conversation retired, but its body stayed in the room — on the pulse
    /// page, and in the company every neighbour was told about.
    #[tokio::test]
    async fn a_forgotten_character_leaves_the_world_and_its_body() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "band-one");
        rt.forget_character(1);
        let hosted = rt.hosted.get(WORLD).unwrap();
        assert!(
            hosted.read(|w| w.actor("m1").is_none()),
            "the body is still in the world"
        );
        assert!(!rt.bodies.is_bound(1), "the binding outlived the character");
        assert_eq!(rt.scheduler.pace_of(1), None, "its mind is still scheduled");
        assert!(
            !rt.within(2).company.iter().any(|n| n == "Maker-01"),
            "a neighbour is still told it is here"
        );
        // Forgetting twice is not an error: the second has nothing to take.
        rt.forget_character(1);
    }

    #[tokio::test]
    async fn unhosting_a_world_stops_it_and_frees_the_characters_in_it() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "band-one");

        assert!(rt.unhost(WORLD));
        assert!(rt.moments().is_empty(), "the metronome kept beating");
        assert!(rt.hosted.get(WORLD).is_none());
        assert!(
            !rt.bodies.is_bound(1),
            "a character kept a body that is gone"
        );
        assert!(!rt.bodies.is_bound(2));
        assert!(!rt.unhost(WORLD), "unhosted twice");
    }

    #[test]
    fn a_body_id_is_derived_so_a_restart_finds_the_same_one() {
        // Derived rather than stored, the same reason a timeline id is: a
        // restart reconstructs the correspondence without a table that could
        // be lost or disagree.
        assert_eq!(Runtime::body_id(7), "npc-7");
        assert_ne!(Runtime::body_id(7), Runtime::body_id(8));
    }

    // ── one daemon, several worlds ──────────────────────────────────────────

    /// **Why a hosted world has an id at all.** Every character is created with
    /// a `world_id` naming the world document it belongs to, so the places it
    /// can stand in must be findable by that same name. Anything else needs a
    /// second table to reconcile the two, and a second table is a thing that can
    /// disagree.
    #[tokio::test]
    async fn a_characters_world_id_finds_the_places_it_can_stand_in() {
        let rt = vaulted();
        // What a character carries is a string from its own document.
        let world_id: String = WORLD.to_string();
        let places = rt.hosted.get(&world_id).expect("hosted under its own id");
        assert_eq!(places.id(), world_id);

        places.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.embody(1, &world_id, "m1", 0).expect("bound by world id");
        assert_eq!(
            rt.body_of(1).map(|(w, b)| (w.id().to_string(), b)),
            Some((world_id, "m1".into()))
        );
    }

    #[tokio::test]
    async fn two_worlds_are_two_places_and_a_body_in_one_is_not_in_the_other() {
        let rt = rt();
        rt.host("creators-vault", Path::new(ROOMS)).unwrap();
        rt.host("second-world", Path::new(ROOMS)).unwrap();
        rt.hold_world("creators-vault", true);
        rt.hold_world("second-world", true);
        assert_eq!(rt.moments().len(), 2);

        let first = rt.hosted.get("creators-vault").unwrap();
        let second = rt.hosted.get("second-world").unwrap();
        first.with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        assert!(
            second.read(|w| w.actor("m1").is_none()),
            "one world, not two"
        );

        // The same body id in two worlds is two bodies, and two characters may
        // have them.
        second.with(|w| w.enter("m1", "Someone Else", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.scheduler.wake(2, 0, 0);
        rt.embody(1, "creators-vault", "m1", 0).unwrap();
        rt.embody(2, "second-world", "m1", 0)
            .expect("a different world");
    }

    #[tokio::test]
    async fn unhosting_one_world_leaves_the_others_running() {
        let rt = rt();
        rt.host("creators-vault", Path::new(ROOMS)).unwrap();
        rt.host("second-world", Path::new(ROOMS)).unwrap();
        rt.hold_world("creators-vault", true);
        rt.hold_world("second-world", true);

        rt.hosted
            .get("creators-vault")
            .unwrap()
            .with(|w| w.enter("m1", "Maker-01", at("band-one")).unwrap());
        rt.hosted
            .get("second-world")
            .unwrap()
            .with(|w| w.enter("m9", "Maker-09", at("band-one")).unwrap());
        rt.scheduler.wake(1, 0, 0);
        rt.scheduler.wake(9, 0, 0);
        rt.embody(1, "creators-vault", "m1", 0).unwrap();
        rt.embody(9, "second-world", "m9", 0).unwrap();

        assert!(rt.unhost("creators-vault"));
        assert!(rt.body_of(1).is_none(), "its character kept a body");
        assert!(
            rt.body_of(9).is_some(),
            "another world's character lost one"
        );
        assert_eq!(rt.moments().len(), 1);
    }

    #[tokio::test]
    async fn only_authored_worlds_with_rooms_are_hosted() {
        // Driven by the registry rather than by what is on disk: a map
        // directory nothing authored must not be hosted under an id no
        // character can name, and an authored world with no map is a world
        // whose characters have lore and no bodies — the common case.
        let dir = std::env::temp_dir().join("npcd-host-authored");
        let _ = std::fs::remove_dir_all(&dir);
        let maps = dir.join(MAPS);
        std::fs::create_dir_all(maps.join("has-rooms")).unwrap();
        std::fs::create_dir_all(maps.join("never-authored")).unwrap();
        for f in std::fs::read_dir(ROOMS).unwrap().flatten() {
            if f.path().is_file() {
                std::fs::copy(f.path(), maps.join("has-rooms").join(f.file_name())).unwrap();
            }
        }
        std::fs::create_dir_all(maps.join("has-rooms").join("parts")).unwrap();
        for f in std::fs::read_dir(Path::new(ROOMS).join("parts"))
            .unwrap()
            .flatten()
        {
            std::fs::copy(
                f.path(),
                maps.join("has-rooms").join("parts").join(f.file_name()),
            )
            .unwrap();
        }

        let rt = rt();
        // Only what the registry knows is offered. `never-authored` has a
        // directory on disk and no document, so it is not among these.
        let done = rt.host_authored(&dir, ["has-rooms", "no-rooms"]);
        // `no-rooms` is authored with nowhere to stand — lore and no bodies,
        // which is the common case and not an error.
        let ids: Vec<&str> = done.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["has-rooms"]);
        assert!(done[0].1.is_ok());
        assert!(rt.hosted.get("has-rooms").is_some());
        assert!(rt.hosted.get("never-authored").is_none());

        rt.unhost("has-rooms");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

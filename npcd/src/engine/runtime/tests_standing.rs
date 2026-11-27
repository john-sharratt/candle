//! Tests for the standing instruction a character is given when nothing asks of it.
//!
//! The instruction is checked for what it leaves out: no place, no object and no
//! verb of motion. Company changes it to one that names the people present, and
//! somebody in the next room is not company. Each test checks which instruction a
//! character is given at a given point, including for a mind with no body and for
//! a character that takes no orders.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        at, embody, run, vaulted, waiting, whole_of, window, ROOMS, WORLD,
    };

    // ── the standing instruction ────────────────────────────────────────────

    #[tokio::test]
    async fn a_character_with_nothing_asked_of_it_is_pointed_at_the_work() {
        // Having nothing to do is itself a standing instruction. An NPC that
        // stands still because nothing was assigned reads as scenery.
        //
        // **At the work, not at the map.** This used to assert "not been in" —
        // the instruction was to go and see an unvisited room, and characters
        // did precisely that: a seventy-eight room tour, hours long, that
        // produced nothing to talk about, because a room is not a subject.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        rt.embody_in_world(1, WORLD, Kind::default(), "Maker-01", None, 0)
            .unwrap();
        // The command table opens by default now; shut it to read the
        // no-mission-no-table standing task this test is about.
        rt.set_table_open(false);

        let nudge = rt.nudge_for(1).expect("something to be getting on with");
        assert_eq!(nudge, NO_MISSION);
        assert!(nudge.contains("the work is in front of you"), "{nudge}");
        // **No verb of motion.** The standing task is the most recent thing in
        // the window, so whatever it tells a character to *do* is what the
        // character does — and every version of this that ended with somewhere
        // to go produced a cast that only ever went there.
        for motion in ["go ", "walk", "find somewhere", "where people are"] {
            assert!(
                !nudge.to_lowercase().contains(motion),
                "the standing task tells it to move (`{motion}`): {nudge}"
            );
        }
    }

    /// **Company changes the standing task.** Told to go somewhere new every
    /// quiet turn, two characters explored a seventy-eight room building and
    /// never held a conversation: each moved every four seconds, so sharing a
    /// room lasted one tick and neither had a reason to stay for the second.
    #[tokio::test]
    async fn a_character_that_is_not_alone_is_told_to_stay_and_talk() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        // Shut the (now default-open) table so company, not the table, is what
        // changes the standing task here.
        rt.set_table_open(false);
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));

        // Somebody walks in, and what there is to do changes with them.
        embody(&rt, 2, "m2", "green-room");
        let together = rt.nudge_for(1).expect("bound");
        // Speaking, and about something in particular — "say what you have
        // been looking at" got three characters agreeing about silence for a
        // hundred turns, because none of them had to name a thing.
        // **Naming the act is what makes an instruction land.** "Talk to them"
        // is a wish; "`message`" is the branch the grammar has an arm for when
        // the company is characters, and the difference showed up as a cast
        // that only ever walked.
        assert!(
            together.contains("`message`"),
            "the instruction names no act to carry it out: {together}"
        );
        assert!(
            together.contains("particular thing"),
            "it permits a vague answer: {together}"
        );
        // **And it names them.**
        //
        // This is the whole difference between an instruction a character can
        // act on and one it cannot. `tell` and `ask` take a name and refuse one
        // they cannot find, and the only other place a name appears — the
        // situation percept — is suppressed while nothing moves. Told to talk
        // to an unnamed somebody, a live character addressed `you`, was
        // refused, and did it again.
        assert!(
            together.contains("Maker-02"),
            "the standing task names no addressee: {together}"
        );
        assert!(
            !together.contains("{who}"),
            "the placeholder was never filled: {together}"
        );
        // Each is told about the other, never about itself.
        let other = rt.nudge_for(2).expect("bound");
        assert!(other.contains("Maker-01"), "{other}");
        assert!(
            !other.contains("Maker-02"),
            "it was told about itself: {other}"
        );

        // And when they part, it changes back.
        rt.hosted
            .get(WORLD)
            .unwrap()
            .with(|w| w.set_off("m2", at("band-one")).unwrap());
        rt.hosted.get(WORLD).unwrap().tick();
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));
    }

    /// **A mind with no body sees the whole world, once, and again only when
    /// it changes**: everybody inside it, by room, with what each is doing —
    /// and never its own seat. A body that takes orders is carried no such
    /// view.
    #[tokio::test]
    async fn a_mind_with_no_body_sees_everything_inside_it() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "band-one");
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with_sim(|s| s.missions.set_unordered("m1", true));

        environment::advance(&hosted, &rt.bodies, &rt.scheduler, &|_| 0);
        run(&rt, 1);
        let seen = window(&rt, 1).join("\n");
        assert!(
            seen.contains("Everything inside you, as it stands now:"),
            "{seen}"
        );
        assert!(seen.contains("Maker-02, at rest"), "{seen}");
        assert!(
            !seen.contains("Maker-01,"),
            "its own seat is not in the view: {seen}"
        );
        assert!(
            !window(&rt, 2).join("\n").contains("Everything inside you"),
            "a body sees its room"
        );

        // Unchanged, it is not carried again; changed, it is.
        assert!(!rt.scheduler.carry_whole(1, 0, whole_of(&hosted, "m1")));
        embody(&rt, 3, "m3", "green-room");
        assert!(rt.scheduler.carry_whole(1, 0, whole_of(&hosted, "m1")));
    }

    /// **One who gives orders is given no work when it has none**: its
    /// missions come from the people who message it, so with an empty agenda
    /// it is told only that its time is its own, and nothing is put on it.
    #[tokio::test]
    async fn one_who_gives_orders_is_given_nothing_with_an_empty_agenda() {
        use npc_map::world::Where;
        let rt = vaulted();
        let hosted = rt.hosted.get(WORLD).unwrap();
        hosted.with(|w| {
            w.enter(
                "m1",
                "Zen".to_string(),
                Where::new("vault-casting", "green-room"),
            )
            .unwrap()
        });
        rt.scheduler.wake(1, 0, 0);
        rt.embody(1, WORLD, "m1", 0).expect("bound");
        hosted.with_sim(|s| s.missions.set_unordered("m1", true));
        assert_eq!(rt.nudge_for(1).as_deref(), Some(YOUR_OWN_TIME));
        assert!(
            !hosted.sim(|s| s.missions.is_on_mission("m1")),
            "work was put on its agenda"
        );
    }

    /// **The world gives an unordered character no orders.** At the open table
    /// it is given no work and is never at rest; idle, it is woken only with
    /// [`YOUR_OWN_TIME`]; and the table offers it no report, while a character
    /// beside it is given work as ever. Embodied unordered while it carries a
    /// mission, it is stood down from it.
    #[tokio::test]
    async fn an_unordered_character_is_given_no_orders() {
        use crate::engine::mission::{Mission, Origin, Todo};
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let unordered = Kind {
            unordered: true,
            ..Kind::default()
        };
        rt.embody_in_world(1, WORLD, unordered, "Zen", None, 0)
            .unwrap();
        rt.set_table_open(true);
        assert!(rt.at_command_table(1), "it is standing at the table");
        assert_eq!(rt.at_table_report(1), None);
        let (hosted, body) = rt.body_of(1).unwrap();
        // A world that hands out routines, so the one that takes orders is given
        // one and the one that gives them is not.
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::Routines));
        rt.assign_work(&hosted);
        rt.follow_rest(&hosted);
        assert!(!hosted.sim(|s| s.missions.is_on_mission(&body)));
        assert!(!rt.scheduler.is_resting(1));
        assert_eq!(
            rt.nudge_for(1).as_deref(),
            Some(YOUR_OWN_TIME),
            "woken, and set nothing"
        );
        assert!(!rt.takes_orders(1));
        let acts: Vec<String> = rt
            .invokable_urls(&hosted, &body)
            .into_iter()
            .map(|i| i.act.to_string())
            .collect();
        assert!(
            !acts.iter().any(|a| a == mission_acts::REPORT_DONE.name),
            "{acts:?}"
        );

        embody(&rt, 2, "m2", "green-room");
        rt.assign_work(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission("m2")));
        assert!(rt.takes_orders(2));

        hosted.with_sim(|s| {
            s.missions.set_unordered(&body, false);
            s.missions.assign(
                &body,
                Mission::new(
                    "Visit the hatchery.",
                    vec![Todo::new("go to the hatchery")],
                    Origin::Lodged {
                        by: "u_op".to_string(),
                    },
                ),
            );
        });
        rt.embody_in_world(1, WORLD, unordered, "Zen", None, 0)
            .unwrap();
        assert!(!hosted.sim(|s| s.missions.is_on_mission(&body)));
        assert_eq!(
            hosted.with_sim(|s| s.missions.take_stood_down()),
            [body],
            "it is told its mission was withdrawn"
        );
    }

    /// **A mind that takes no orders has no body** (`docs/zen_everywhere.md`
    /// §4). Nothing of the room its seat is in is carried to it — no room's
    /// situation, nothing that happens there; it is carried the whole world
    /// instead, once — and it is in nobody's company, with nowhere to walk,
    /// ride or turn its attention and no station to work but the order table.
    /// It is still told what comes by other roads: the outcome of a mission
    /// it posted. A body beside it perceives as ever.
    #[tokio::test]
    async fn a_mind_that_takes_no_orders_has_no_body() {
        use crate::engine::enact::NO_BODY;
        use crate::engine::mission_gen::world::Draw;
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let zen = Kind {
            unordered: true,
            attends: true,
            ..Kind::default()
        };
        rt.embody_in_world(1, WORLD, zen, "Zen", None, 0).unwrap();
        // A Drone in the same room, at the table.
        rt.scheduler.wake(2, 0, 0);
        rt.embody_in_world(2, WORLD, Kind::default(), "Wick", None, 0)
            .unwrap();
        rt.set_table_open(false);
        let hosted = rt.hosted.get(WORLD).unwrap();
        let (_, seat) = rt.body_of(1).unwrap();
        let (_, wick) = rt.body_of(2).unwrap();

        // Nothing of the room reached it as it was seated. A moment carries it
        // the whole world, once, and never the room: Wick is in the view as
        // somebody inside it, and nothing says where the seat is.
        assert_eq!(rt.scheduler.inbox_depth(1), Some(0), "told where it is");
        rt.moment(&hosted);
        let seen = waiting(&rt, 1).join("\n");
        assert!(seen.contains("Everything inside you"), "{seen}");
        assert!(seen.contains("Wick, at rest"), "{seen}");
        assert!(!seen.contains("You are in"), "told where it is: {seen}");
        let carried = rt.scheduler.inbox_depth(1);
        rt.moment(&hosted);
        assert_eq!(
            rt.scheduler.inbox_depth(1),
            carried,
            "carried again unchanged"
        );
        assert!(
            waiting(&rt, 2)
                .iter()
                .any(|l| l.contains("the command room")),
            "a body beside it is told where it stands"
        );

        // In nobody's company, and nowhere to go.
        let within = rt.within(1);
        assert!(within.company.is_empty() && within.people.is_empty());
        assert!(within.places.is_empty() && within.rooms.is_empty());
        assert!(within.floors.is_empty() && !within.at_lift && !within.away_from_home);
        assert!(within.parts.is_empty(), "{:?}", within.parts);
        // The order table is still its to post on, and nothing else is.
        let acts: Vec<&str> = within.invokable.iter().map(|i| i.act).collect();
        assert!(acts.contains(&"order_post"), "{acts:?}");
        assert!(
            acts.iter().all(|a| mission_acts::is_mine(a)),
            "only the table's acts: {acts:?}"
        );

        let act = |tool: &'static str, v: serde_json::Value| Act {
            tool,
            args: v.as_object().unwrap().clone(),
        };
        for (tool, args) in [
            ("focus", serde_json::json!({"on": "the green room"})),
            (
                "move_to",
                serde_json::json!({"destination": "the green room"}),
            ),
            ("recall", serde_json::json!({})),
            ("lift_call", serde_json::json!({})),
        ] {
            assert_eq!(
                body::perform(&hosted, &seat, &act(tool, args)),
                Outcome::Refused(NO_BODY.into()),
                "{tool}"
            );
        }
        assert_eq!(hosted.place_of(&seat), "vault-command/command-room");

        // What comes by another road still reaches it: the outcome of a
        // mission it posted.
        hosted.with_sim(|s| s.missions.set_routines(Vec::new(), Draw::None));
        rt.set_table_open(true);
        let posted = body::perform(
            &hosted,
            &seat,
            &act(
                "order_post",
                serde_json::json!({"brief": "Read the board.", "steps": "go to the dispatch room"}),
            ),
        );
        assert!(posted.happened(), "{posted:?}");
        rt.moment(&hosted);
        assert!(hosted.sim(|s| s.missions.is_on_mission(&wick)));
        // It is the world, so it is told what happens inside it: the work
        // taken up, and — at the next moment's sweep — what it now sees, Wick
        // carrying it.
        rt.moment(&hosted);
        let told = waiting(&rt, 1);
        assert!(
            told.iter()
                .any(|l| l == "Wick has taken up work: Read the board."),
            "{told:?}"
        );
        assert!(
            told.iter()
                .any(|l| l.contains("Wick, carrying \"Read the board.\"")),
            "{told:?}"
        );
        hosted.with_sim(|s| {
            s.missions
                .report(&wick, Verdict::Pass, "two jobs stalled", None)
        });
        rt.moment(&hosted);
        let told = waiting(&rt, 1);
        assert!(
            told.iter().any(|l| l
                .starts_with("Wick has reported it done: Read the board. What Wick reported: \"two jobs stalled\"")),
            "{told:?}"
        );
    }

    /// **A mind with no body reaches real things through one terminal at a
    /// time** (`docs/zen_everywhere.md` §5). It lists what there is, connects,
    /// and has that terminal's verbs alone in its grammar; a verb of another
    /// terminal is refused, a body is refused all of it, and what a terminal
    /// raises is told to the mind and to nobody with a body.
    #[tokio::test]
    async fn a_mind_with_no_body_works_through_one_terminal_at_a_time() {
        use crate::bridge::finding::{Finding, DEVICE, SONOS, WIZ};
        use crate::engine::terminal::{ALL_LIGHTS, LIGHTS, NETWORK};
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let zen = Kind {
            unordered: true,
            attends: true,
            ..Kind::default()
        };
        rt.embody_in_world(1, WORLD, zen, "Zen", None, 0).unwrap();
        embody(&rt, 2, "m2", "green-room");
        rt.set_table_open(false);
        let hosted = rt.hosted.get(WORLD).unwrap();
        let found = |hardware: &str, name: &str, profile: &str| Finding {
            address: "192.168.1.20".into(),
            hardware: hardware.into(),
            names: vec![name.into()],
            profile: profile.into(),
            ..Finding::default()
        };
        hosted.with_sim(|s| {
            s.terminals.surveyed(vec![
                found("a8:bb:50:00:00:01", "wiz_d8a011", WIZ),
                found("a8:bb:50:00:00:02", "hall light", WIZ),
                found("f0:f6:c1:a0:79:b4", "SonosZP", SONOS),
                found("3c:22:fb:00:00:09", "Pixel-8", DEVICE),
            ])
        });
        let act = |tool: &'static str, v: serde_json::Value| Act {
            tool,
            args: v.as_object().unwrap().clone(),
        };
        let none = serde_json::json!({});

        // Disconnected: the terminal acts, and no terminal's verbs.
        let free = rt.within(1);
        assert_eq!(free.terminals, [LIGHTS, NETWORK, "SonosZP"]);
        assert!(free.grants.iter().any(|g| g == "terminal_connect"));
        assert!(free.lights.is_empty() && free.sonos.is_empty());
        let listed = rt
            .terminal_act(1, &act("terminal_list", none.clone()))
            .await;
        assert!(listed.landed, "{}", listed.answer);
        assert!(
            listed
                .answer
                .contains("- the lights: every light in the house, 2 of them, 2 answering"),
            "{}",
            listed.answer
        );
        let early = rt
            .terminal_act(
                1,
                &act(
                    "device_rename",
                    serde_json::json!({"device": "x", "called": "y"}),
                ),
            )
            .await;
        assert!(!early.landed && early.answer.starts_with("You are connected to nothing"));

        // Connected to the lights: the light acts, on one bulb or all.
        let lit = rt
            .terminal_act(
                1,
                &act("terminal_connect", serde_json::json!({"terminal": LIGHTS})),
            )
            .await;
        assert!(
            lit.landed && lit.answer.starts_with("You are connected to the lights."),
            "{}",
            lit.answer
        );
        let within = rt.within(1);
        assert_eq!(within.lights, [ALL_LIGHTS, "hall light", "wiz_d8a011"]);
        assert!(within.sonos.is_empty(), "another terminal's device");
        assert!(!within.terminals.iter().any(|t| t == LIGHTS));
        // Another terminal's verb is not this one's.
        let stray = rt
            .terminal_act(
                1,
                &act("sonos_play", serde_json::json!({"device": "SonosZP"})),
            )
            .await;
        assert!(!stray.landed && stray.answer.contains("has no `sonos_play`"));
        // With no agent in reach a light is refused, never silently skipped.
        let dark = rt
            .terminal_act(
                1,
                &act("wiz_off", serde_json::json!({"device": ALL_LIGHTS})),
            )
            .await;
        assert!(!dark.landed && dark.answer.starts_with("0 of 2 lights took it."));

        // Connecting elsewhere lets the last go: the network names devices.
        let net = rt
            .terminal_act(
                1,
                &act("terminal_connect", serde_json::json!({"terminal": NETWORK})),
            )
            .await;
        assert!(net.landed, "{}", net.answer);
        assert!(rt.within(1).lights.is_empty());
        let named = rt
            .terminal_act(
                1,
                &act(
                    "device_rename",
                    serde_json::json!({"device": "wiz_d8a011", "called": "kitchen light"}),
                ),
            )
            .await;
        assert_eq!(
            named.answer,
            "wiz_d8a011 is called kitchen light from now on."
        );
        let watched = rt
            .terminal_act(
                1,
                &act(
                    "presence_watch",
                    serde_json::json!({"device": "Pixel-8", "who": "Kate"}),
                ),
            )
            .await;
        assert!(watched.landed, "{}", watched.answer);
        assert!(hosted.sim(|s| s.terminals.watches()));

        // What a terminal raises reaches the mind, and nobody with a body.
        waiting(&rt, 1);
        waiting(&rt, 2);
        rt.raise_terminal(&hosted, NETWORK, "Kate is home.", true);
        assert_eq!(
            waiting(&rt, 1),
            ["Raised by the house network, a terminal of yours: Kate is home."]
        );
        assert!(waiting(&rt, 2).is_empty());

        // Let go, it is connected to nothing; and a body has no terminals.
        let gone = rt
            .terminal_act(1, &act("terminal_disconnect", none.clone()))
            .await;
        assert_eq!(gone.answer, "You let go of the house network.");
        assert!(
            !rt.terminal_act(1, &act("terminal_disconnect", none.clone()))
                .await
                .landed
        );
        assert!(rt.within(2).terminals.is_empty());
        assert!(!rt.within(2).grants.iter().any(|g| g == "terminal_list"));
        let body = rt.terminal_act(2, &act("terminal_list", none)).await;
        assert!(!body.landed && body.answer.starts_with("Terminals are not yours"));
    }

    /// **A personality's `idle:` is its idle wake**, for one that takes no
    /// orders and for one with no table to be called to.
    #[tokio::test]
    async fn a_personality_s_idle_thought_is_its_idle_wake() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        let zen = Kind {
            unordered: true,
            idle: Some("You look in on the groups, through their terminals."),
            ..Kind::default()
        };
        rt.embody_in_world(1, WORLD, zen, "Zen", None, 0).unwrap();
        assert_eq!(
            rt.nudge_for(1).as_deref(),
            Some("You look in on the groups, through their terminals.")
        );
        embody(&rt, 2, "m2", "green-room");
        rt.set_table_open(false);
        rt.set_idle(2, Some("You tend the hatchery."));
        assert_eq!(rt.nudge_for(2).as_deref(), Some("You tend the hatchery."));
        rt.set_idle(2, None);
        assert_eq!(rt.nudge_for(2).as_deref(), Some(NO_MISSION));
    }

    /// **An edited personality reaches characters already standing in their
    /// world**: read again, `takes_orders` marks them, and unmarks them, with no
    /// embodiment in between; one whose persona cannot be read keeps its mark.
    #[tokio::test]
    async fn an_edited_personality_marks_its_characters_where_they_stand() {
        use std::sync::atomic::{AtomicU8, Ordering};
        let rt = vaulted();
        rt.embody_in_world(1, WORLD, Kind::default(), "Zen", None, 0)
            .unwrap();
        assert!(rt.takes_orders(1));
        // 0: no persona to read, 1: takes no orders, 2: takes them.
        let says = Arc::new(AtomicU8::new(1));
        let reads = Arc::clone(&says);
        rt.set_persona_source(Arc::new(move |_| match reads.load(Ordering::SeqCst) {
            0 => None,
            n => Some(OwnedPersona {
                unordered: n == 1,
                ..OwnedPersona::default()
            }),
        }));
        rt.mark_orders().await;
        assert!(!rt.takes_orders(1), "marked where it stands");
        says.store(0, Ordering::SeqCst);
        rt.mark_orders().await;
        assert!(!rt.takes_orders(1), "an unread persona keeps the mark");
        says.store(2, Ordering::SeqCst);
        rt.mark_orders().await;
        assert!(rt.takes_orders(1), "and unmarked");
    }

    /// **A cast embodied while this member followed its world is marked once the
    /// member becomes the world's authority**: a character whose personality
    /// takes no orders is stood down from the mission it carried, rather than
    /// working it on until the personalities are next read.
    #[tokio::test]
    async fn taking_a_world_marks_the_cast_embodied_as_its_replica() {
        use crate::engine::mission::{Mission, Origin, Todo};

        let rt = vaulted();
        rt.standing.set_role(WORLD, Role::Replica, 0);
        rt.embody_in_world(1, WORLD, Kind::default(), "Zen", None, 0)
            .unwrap();
        let (hosted, body) = rt.body_of(1).unwrap();
        hosted.with_sim(|s| {
            s.missions.assign(
                &body,
                Mission::new(
                    "Visit the hatchery.",
                    vec![Todo::new("go to the hatchery")],
                    Origin::Random {
                        routine: "visit-the-hatchery".into(),
                    },
                ),
            )
        });
        rt.set_persona_source(Arc::new(|_| {
            Some(OwnedPersona {
                unordered: true,
                ..OwnedPersona::default()
            })
        }));
        rt.mark_orders_in(WORLD).await;
        assert!(rt.takes_orders(1), "a replica marks nothing");
        assert!(hosted.sim(|s| s.missions.active(&body).is_some()));

        rt.standing.set_role(WORLD, Role::Authority { epoch: 1 }, 0);
        rt.mark_orders_in(WORLD).await;
        assert!(!rt.takes_orders(1), "the authority marks it");
        assert!(
            hosted.sim(|s| s.missions.active(&body).is_none()),
            "and stands it down from what it carried"
        );
    }

    /// **One who gives orders is told what the table holds of theirs, every
    /// turn a request stands** — nothing, or each brief and who has it — so a
    /// remembered "I posted it" cannot stand in for the post.
    #[test]
    fn a_standing_request_reads_the_order_table_as_it_is() {
        let post = "http://local/command/order-table~0/order_post";
        let none = orders_line(&[], Some(post));
        assert!(
            none.starts_with("Nothing you have posted is on the order table"),
            "{none}"
        );
        assert!(
            none.contains("post it with `invoke` http://local/command/order-table~0/order_post"),
            "{none}"
        );
        assert_eq!(
            orders_line(
                &[
                    ("Read the sensor panel.".into(), Some("Jitter".into())),
                    ("Watch Earth".into(), None),
                ],
                Some(post)
            ),
            "On the order table, of yours: \"Read the sensor panel\" — Jitter has it; \"Watch \
             Earth\" — waiting to be taken."
        );
        let told = still_asked("John", "Have a Drone read the panel.", Some(&none));
        assert!(
            told.contains("Have a Drone read the panel.\nNothing you have posted"),
            "{told}"
        );
        assert!(
            told.ends_with(
                "until it is posted, nobody is doing it.\nDo it now: post it, if it is a Drone's \
                 to carry, or else take the act that answers it yourself. Then tell John what \
                 came of it — what you read, or that you could not. Telling them you are about \
                 to is not an answer."
            ),
            "{told}"
        );
        assert_eq!(
            still_asked("John", "Read the panel.", None),
            "Still yours to do, and nothing comes before it: what John asked of you on the \
             phone.\nRead the panel.\nDo it yourself, now, with the act that answers it, then \
             tell John what came of it — what you read, or that you could not. Telling them you \
             are about to is not an answer."
        );
    }

    /// **The standing task is for the quiet turns, and the situation is not
    /// news.**
    ///
    /// This gate has been wrong in both directions. Gated on inbox *depth* it
    /// never fired, because a character with a body is handed the room it is
    /// standing in every moment, so the depth is never zero — and the task
    /// written for the case of having company was the one that never arrived.
    /// Ungated it fired every turn, restating "nothing has been asked of you"
    /// more recently than anything a companion had actually said.
    ///
    /// So the question is whether anything *happened*, which is what
    /// [`crate::engine::tick::Inbox::has_news`] answers.
    #[tokio::test]
    async fn the_standing_task_waits_for_a_turn_with_nothing_in_it() {
        use crate::engine::event::{EventKind, Salience};

        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        // Where a character is standing is a fact about the room, not something
        // that happened — and this is the case the old depth gate mistook for a
        // busy character, because a body is handed its situation whenever the
        // world moves under it.
        rt.scheduler.deliver(
            1,
            0,
            Salience::IDLE,
            EventKind::Situation {
                text: "You are in the green room.".into(),
            },
        );
        assert!(
            rt.scheduler.inbox_depth(1).unwrap() > 0,
            "the situation should be queued"
        );
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(10_000),
            "standing in a room is not something that happened"
        );

        // Nor is the standing task itself, or it would keep its own gate open.
        rt.scheduler.deliver(
            1,
            0,
            Salience::IDLE,
            EventKind::Nudge {
                text: NO_MISSION.into(),
            },
        );
        assert_eq!(rt.scheduler.quiet_for(1, 10_000), Some(10_000));

        // Somebody speaks, and now there is something to answer.
        rt.scheduler.deliver(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::You,
            },
        );
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(0),
            "being spoken to is news"
        );

        // **And the quiet does not restart the moment it is read.** Draining the
        // speech leaves the character having been spoken to at t=0, not having
        // been alone forever — which is the whole point of the clock: the gap
        // between two turns of a conversation is not idleness.
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert_eq!(
            rt.scheduler.quiet_for(1, 10_000),
            Some(10_000),
            "ten seconds after the last thing said to it"
        );
        assert!(
            rt.scheduler.quiet_for(1, 10_000).unwrap() < IDLE_AFTER_MS,
            "ten seconds is a pause in a conversation, not an idle character"
        );
    }

    #[tokio::test]
    async fn somebody_in_the_next_room_is_not_company() {
        // Company is who is *here*. Being able to see somebody through a
        // doorway is a reason to go to them, not a conversation.
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "ring-north");
        // Shut the (now default-open) table so the standing task reflects
        // company (there is none here), not a call to the table.
        rt.set_table_open(false);
        assert_eq!(rt.nudge_for(1).as_deref(), Some(NO_MISSION));
    }

    #[tokio::test]
    async fn a_character_with_no_body_is_not_told_to_explore_anything() {
        // It has nowhere to go and nobody to talk to; instructing it otherwise
        // is instructing it to do something it cannot.
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert_eq!(rt.nudge_for(1), None);
    }

    /// The standing instruction is a **disposition**, not a script. A named
    /// destination would be an order every character in the world followed
    /// identically — which looks like emergence and is the opposite of it.
    #[test]
    fn the_standing_instruction_names_no_particular_place_or_person() {
        let vault = npc_map::MapSet::load_dir(ROOMS).unwrap();
        let words = NO_MISSION.to_lowercase();
        for area in vault.areas() {
            for node in &area.nodes {
                assert!(
                    !words.contains(&node.name.to_lowercase()),
                    "it names {}",
                    node.name
                );
            }
        }
        assert!(!words.contains("maker-"), "it names somebody");
    }

    /// A standing instruction read by every world's cast may not hand them a
    /// thing to have found: the example a character is given is the thing it
    /// then reports reading.
    #[test]
    fn the_standing_instructions_offer_no_example_object() {
        for text in [NO_MISSION, IN_COMPANY] {
            let words = text.to_lowercase();
            for thing in ["ledger", "story", "wall", "date", "page", "entry"] {
                assert!(!words.contains(thing), "`{thing}` in: {text}");
            }
        }
    }

    #[test]
    fn a_standing_instruction_replaces_the_one_before_it() {
        use crate::engine::event::EventKind;
        let a = EventKind::Nudge { text: "one".into() };
        let b = EventKind::Nudge { text: "two".into() };
        assert_eq!(a.replaces().as_deref(), Some("nudge"));
        assert_eq!(a.replaces(), b.replaces(), "two tasks at once");
        // Its own band, beside the situation — they are different things and
        // neither may retire the other.
        assert_ne!(
            a.replaces(),
            EventKind::Situation { text: "x".into() }.replaces()
        );
    }

    #[test]
    fn a_thought_it_cannot_shake_and_the_standing_task_share_a_band() {
        use crate::engine::event::EventKind;
        let task = EventKind::Nudge {
            text: "new errand".into(),
        };
        let thought = EventKind::MindControl {
            text: "old errand".into(),
        };
        assert_eq!(task.replaces(), thought.replaces(), "two tasks at once");
    }

    #[test]
    fn a_standing_instruction_never_interrupts_and_reads_as_written() {
        use crate::engine::event::{Event, EventKind, Salience};
        let e = Event::new(
            1,
            0,
            Salience::IDLE,
            EventKind::Nudge {
                text: format!("  {NO_MISSION}\n"),
            },
        );
        assert!(!e.preempts(), "a standing task interrupted a character");
        assert_eq!(e.prose(), NO_MISSION);
    }
}

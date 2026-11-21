//! Tests for how acts reach the world, the character and the feed.
//!
//! Each test follows a path a person or a character actually takes: a message
//! reaching a character's conversation on the next moment, a pause ended by being
//! spoken to, a refused act read as refused. The feed and the character are checked
//! separately, since each is given its own sentence for the same act. Several
//! assert that nothing reaches a character outside a world moment.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use serde_json::json;

    use super::super::tests_support::tests::{act, at, embody, rt, run, vaulted, window, WORLD};

    /// **A message sent to a character has to reach its conversation.**
    ///
    /// The whole path, end to end, because every piece of it was individually
    /// correct while the thing a person actually does — message a character and
    /// wait for an answer — did nothing at all. Sending writes the pair's
    /// thread; the world's next moment is what carries the line to the
    /// character's conversation with them; the mind's inbox is never handed it
    /// (`docs/npc_conversations.md` §9).
    #[tokio::test]
    async fn a_message_reaches_the_characters_conversation_on_the_next_moment() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).expect("hosted");

        let sent = rt
            .message_npc(
                1,
                &Who::new("u_john", "Johnathan Sharratt"),
                "are you there",
                None,
            )
            .expect("the character could be messaged");
        assert_eq!(sent.with, "Maker-01");
        assert_eq!(sent.waiting_for_them, 1, "it was not written to the thread");

        // Drain what embodying already queued, so what is asserted below can
        // only have come from the message.
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());
        assert_eq!(rt.scheduler.inbox_depth(1), Some(0), "not drained");

        let moment = environment::advance(&hosted, &rt.bodies, &rt.scheduler, &|_| 0);
        assert_eq!(moment.messaged, 1, "the moment carried nothing to anybody");

        assert_eq!(
            rt.talks.waiting(1),
            1,
            "the line never reached the conversation"
        );
        assert_eq!(
            rt.scheduler.inbox_depth(1),
            Some(0),
            "the mind was handed it"
        );
    }

    /// And it is handed over **once**. The sweep runs twice a second; a cursor
    /// that did not move would read the same message to the character for ever,
    /// which is the failure the phone's own cursor exists to prevent.
    #[tokio::test]
    async fn a_message_is_handed_to_the_mind_exactly_once() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let hosted = rt.hosted.get(WORLD).expect("hosted");
        rt.message_npc(
            1,
            &Who::new("u_john", "Johnathan Sharratt"),
            "are you there",
            None,
        );
        rt.scheduler.tick(1, 0, 0, |_, _| Vec::new());

        assert_eq!(
            environment::advance(&hosted, &rt.bodies, &rt.scheduler, &|_| 0).messaged,
            1
        );
        assert_eq!(
            environment::advance(&hosted, &rt.bodies, &rt.scheduler, &|_| 0).messaged,
            0,
            "the same message was handed over twice"
        );
    }

    #[tokio::test]
    async fn an_act_that_happens_inside_a_head_never_reaches_the_world() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        assert_eq!(
            rt.act_on_world(1, &act("note_concern", json!({}))),
            Outcome::NotOfTheBody
        );
    }

    #[tokio::test]
    async fn a_character_with_no_body_cannot_act_on_a_world() {
        let rt = vaulted();
        rt.scheduler.wake(1, 0, 0);
        assert_eq!(
            rt.act_on_world(1, &act("speak", json!({"intent": "anything"}))),
            Outcome::NotOfTheBody
        );
    }

    #[tokio::test]
    async fn speaking_reaches_the_other_character_in_the_room() {
        // The whole chain, in one daemon: an act from one mind lands in the
        // world, is perceived by the body beside it, and is read by that
        // body's mind.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let said = rt
            .act_on_world(
                1,
                &act(
                    "speak",
                    json!({"intent": "the redoubt burned twice", "to": "Maker-02"}),
                ),
            )
            .line()
            .expect("a body act")
            .to_string();
        assert!(said.contains("You tell Maker-02"), "{said}");

        // Not yet: perception happens on the world's clock, not the speaker's.
        run(&rt, 2);
        assert!(!window(&rt, 2).join("\n").contains("redoubt"));

        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        run(&rt, 3);

        let heard = window(&rt, 2).join("\n");
        assert!(heard.contains("says to you"), "{heard}");
        assert!(heard.contains("redoubt"), "{heard}");
    }

    /// **Being spoken to ends a pause.**
    ///
    /// The deadlock the typed wait needed three mechanisms to prevent cannot
    /// form here: Wyneth stops, and Perrin speaking brings her straight back
    /// through the ordinary sweep. No condition to satisfy, no patience, no
    /// rule excluding Perrin from waiting back — and nothing announcing to the
    /// room that she stopped, which it no longer is.
    #[tokio::test]
    async fn a_pause_is_ended_by_the_room() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let w = act("reflect", json!({}));
        assert!(
            rt.record_act(1, &w).feed.starts_with("reflect"),
            "the act stood"
        );
        rt.arm_pause(1, &w, 0);
        assert!(
            !rt.scheduler.due_now(1).contains(&1),
            "she did not actually stop"
        );

        // Perrin is told nothing about it.
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        run(&rt, 2);
        let seen = window(&rt, 2).join("\n");
        assert!(!seen.contains("lets the moment pass"), "{seen}");

        // And Perrin speaking brings her back at once, through nothing but the
        // ordinary delivery path.
        assert!(rt
            .act_on_world(2, &act("shout", json!({"intent": "that I am here"})))
            .happened());
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        assert!(rt.scheduler.due_now(1).contains(&1), "she was not woken");
    }

    /// A stall nobody interrupts ends by itself.
    ///
    /// The failure this replaces: a wait nothing could answer left the
    /// character asleep forever while every view of it read the same word a
    /// merely quiet character reads, so the two were indistinguishable from
    /// outside. A stall cannot do that — its deadline is the whole of it.
    ///
    /// Driven by `move_to`, which is one of the two acts that occupy a body.
    /// It was `reflect`, from when reflection was the only thing that stalled;
    /// a thought does not take time, so it no longer does.
    #[tokio::test]
    async fn a_pause_nobody_interrupts_ends_on_its_own() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        run(&rt, 1);

        rt.arm_pause(1, &act("move_to", json!({"destination": "band one"})), 0);
        assert!(
            !rt.scheduler.due_now(1_000).contains(&1),
            "it came back a second later"
        );
        // Nobody says anything, ever, and it comes back anyway.
        assert!(
            rt.scheduler.due_now(10_000_000).contains(&1),
            "it stopped for good"
        );
    }

    #[tokio::test]
    async fn an_act_that_landed_is_recorded_as_an_act_whatever_the_world_said_back() {
        // Speech is the case that made this visible. The world answers a
        // successful `shout` in narration — "You shout, for anyone within
        // earshot." — and
        // recording *that* put one prose sentence in a column of single-word
        // acts, both in the feed and in the character's own window.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let said = rt.record_act(
            1,
            &act("shout", json!({"intent": "the redoubt burned twice"})),
        );
        // The feed carries both halves: what was asked, then what came of it.
        assert_eq!(
            said.feed,
            "shout — the redoubt burned twice → You shout, for anyone within earshot."
        );
        // The character, meanwhile, is told only what the world did with it —
        // it does not need to be told the name of the act it just chose.
        assert_eq!(said.answer, "You shout, for anyone within earshot.");

        // A body act and a head act now read the same way as each other, which
        // is the point — one shape, whether or not a world was involved.
        let noted = rt.record_act(1, &act("note_concern", json!({"about": "the ledger"})));
        assert!(noted.feed.starts_with("note_concern"), "{}", noted.feed);

        // And the world did take the speech: the shape of the record is not
        // the act being quietly dropped.
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        run(&rt, 10_000);
        let heard = window(&rt, 2).join("\n");
        assert!(heard.contains("redoubt"), "{heard}");
    }

    /// A refusal keeps the world's words **and** names the act it refused.
    ///
    /// The prose is what tells the character where it could actually have gone,
    /// so losing it to a uniform shape would be the reverse of the bug above.
    /// But the words alone read as a sentence dropped into a column of acts —
    /// "Wailen Wylde is already on it." beside "invite — …", with nothing
    /// saying they were the same kind of event or that one had failed.
    #[tokio::test]
    async fn a_refusal_is_recorded_in_the_world_s_words_because_the_prose_is_the_point() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        let r = rt.record_act(
            1,
            &act("move_to", json!({"destination": "the observatory"})),
        );
        let line = r.feed;
        assert!(
            line.contains("nowhere called \"the observatory\""),
            "{line}"
        );
        assert!(line.contains("band one"), "it named nowhere real: {line}");
        // Scannable like every other line: the act's name first.
        assert!(line.starts_with("move_to"), "{line}");
        // And unmistakably not a completed one.
        assert!(
            line.contains('✗'),
            "a refusal read as a completed act: {line}"
        );

        // The character is told the world's words and nothing else — the mark
        // and the act's name are for somebody watching the feed, and a
        // character does not need to be told the name of the act it just chose.
        assert!(!r.answer.contains('✗'), "{}", r.answer);
        assert!(r.answer.contains("nowhere called"), "{}", r.answer);
    }

    /// **The character is told what happened, not what it asked for.**
    ///
    /// The feed renders an act as an act — `gesture — …` — which is right for
    /// a column somebody scans. Sent to the model as a `<tool_response>` it is
    /// the character's own arguments handed back as though they were an
    /// outcome, which is what it read live: "gesture — for Wailen to see that
    /// the room is ours for now", every turn, never once told whether anybody
    /// saw it.
    #[tokio::test]
    async fn the_answer_is_the_worlds_verdict_and_the_feed_line_is_the_act() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let r = rt.record_act(1, &act("gesture", json!({"intent": "at the door"})));
        // The feed names the act, what it asked for, and what came of it.
        assert!(r.feed.starts_with("gesture — at the door → "), "{}", r.feed);
        assert!(r.feed.contains("You show"), "{}", r.feed);
        // The character gets the verdict alone — not its own arguments back.
        assert!(
            r.answer.starts_with("You show"),
            "the character was handed its own arguments: {}",
            r.answer
        );
        assert!(!r.answer.contains("gesture —"), "{}", r.answer);
    }

    /// **A tell survives its addressee walking out during the decode.** The
    /// company is fixed before the grammar is built; the addressee then leaves
    /// the speaker's sight while the decode runs; the act that follows lands and
    /// the addressee reads it, instead of the character being told "is not here"
    /// about somebody its own grammar offered. Once the decision ends, the same
    /// line is refused again.
    #[tokio::test]
    async fn a_tell_to_somebody_who_walked_out_during_the_decode_lands() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        rt.begin_decision(1);
        let world = rt.hosted.get(WORLD).unwrap();
        world.with(|w| {
            w.set_off("m2", Where::new("vault-chronicle", "core"))
                .unwrap();
            w.settle();
        });
        let tell = act(
            "tell",
            json!({"to": "Maker-02", "intent": "the redoubt held"}),
        );
        let landed = rt.record_act(1, &tell);
        assert!(landed.landed, "{}", landed.answer);
        assert!(
            landed.answer.contains("as they walk away"),
            "{}",
            landed.answer
        );

        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        run(&rt, 10_000);
        let heard = window(&rt, 2).join("\n");
        assert!(heard.contains("redoubt"), "{heard}");

        rt.end_decision(1);
        let refused = rt.record_act(1, &tell);
        assert!(!refused.landed, "{}", refused.answer);
        assert!(refused.answer.contains("is not here"), "{}", refused.answer);
    }

    /// **What a body holds is read from both places it can be held.** A claimed
    /// station lives in the world and an order in the sim; either puts `release`
    /// in the grammar, and giving it back takes it out again.
    #[tokio::test]
    async fn release_is_offered_exactly_while_something_is_held() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "band-one");
        embody(&rt, 2, "m2", "band-one");
        let offered = |npc: u64| {
            tools::specs_within(tools::Mode::Physical, &rt.within(npc))
                .iter()
                .any(|t| t.name == "release")
        };
        assert!(!offered(1), "offered to a body holding nothing");

        let world = rt.hosted.get(WORLD).unwrap();
        world.with(|w| w.take("m1", Some("cindy")).unwrap());
        assert!(offered(1), "a claimed station was not read as held");
        assert!(
            !offered(2),
            "somebody else's claim was offered to a bystander"
        );
        assert!(rt.record_act(1, &act("release", json!({}))).landed);
        assert!(
            !offered(1),
            "still offered after the station was given back"
        );

        let order = "close the longest silence in the record";
        assert!(
            rt.record_act(2, &act("claim", json!({"what": order})))
                .landed
        );
        assert!(offered(2), "a taken order was not read as held");
        assert!(rt.record_act(2, &act("release", json!({}))).landed);
        assert!(!offered(2), "still offered after the order was put back");
    }

    /// **A reflect's row is its thought, then what came back.** Its situation
    /// and feeling are the reflection's input and are not listed; and what a
    /// pause says back is only that it happened — never the character's own
    /// words. When a reflection runs, its answer takes the arrow's side instead.
    #[tokio::test]
    async fn a_reflect_is_recorded_as_its_thought_and_what_came_of_it() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let r = rt.record_act(
            1,
            &act(
                "reflect",
                json!({
                    "situation": "alone in the green room",
                    "inner_thoughts": "the box has given up a fold",
                    "feeling": "weary",
                }),
            ),
        );
        assert_eq!(
            r.feed,
            format!(
                "reflect — the box has given up a fold → {}",
                body::NO_REFLECTION
            )
        );
        assert_eq!(r.answer, body::NO_REFLECTION);
    }

    /// **One dream at a time, per character, and the slot always comes
    /// back.** What a reflect finds taken is the dream, never the reflection:
    /// it still answers, and only skips the dream.
    #[test]
    fn a_character_writes_one_dream_at_a_time() {
        let rt = rt();
        let first = rt.claim_dream(1).expect("a free slot");
        assert!(rt.claim_dream(1).is_none(), "two dreams at once");
        assert!(rt.claim_dream(2).is_some(), "one slot per character");
        drop(first);
        assert!(rt.claim_dream(1).is_some(), "the slot never came back");
    }

    #[tokio::test]
    async fn an_act_the_world_refuses_reads_as_refused_rather_than_as_done() {
        // The distinction the whole path exists to preserve: a character that
        // cannot tell a refused act from a successful one spends the rest of
        // the day reasoning from a move it never made.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");

        let out = rt.act_on_world(
            1,
            &act("move_to", json!({"destination": "the observatory"})),
        );
        let Outcome::Refused(out) = out else {
            panic!("a walk to nowhere is a refusal, not {out:?}");
        };
        assert!(out.contains("nowhere called \"the observatory\""), "{out}");
        assert!(out.contains("band one"), "it named nowhere real: {out}");
        assert!(rt
            .hosted
            .get(WORLD)
            .unwrap()
            .read(|w| w.actor("m1").unwrap().walk.is_none()));
    }

    #[tokio::test]
    async fn moving_is_a_journey_the_world_advances_rather_than_the_act() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        let out = rt
            .act_on_world(1, &act("move_to", json!({"destination": "band one"})))
            .line()
            .expect("a body act")
            .to_string();
        assert!(out.contains("one stop"), "{out}");

        let world = rt.hosted.get(WORLD).unwrap();
        // Still where it was: setting off is not arriving.
        assert_eq!(
            world.read(|w| w.actor("m1").unwrap().at.clone()),
            at("green-room")
        );
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        assert_eq!(
            world.read(|w| w.actor("m1").unwrap().at.clone()),
            at("band-one")
        );

        run(&rt, 10_000);
        let read = window(&rt, 1).join("\n");
        assert!(read.contains("You got to band one"), "{read}");
    }

    /// **Arriving on a floor tells you who else is on it, and where** — the
    /// person left behind in the room it walked out of, here.
    #[tokio::test]
    async fn arriving_names_who_else_is_on_the_floor() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        let _ = rt.act_on_world(1, &act("move_to", json!({"destination": "band one"})));
        let world = rt.hosted.get(WORLD).unwrap();
        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);

        run(&rt, 10_000);
        let read = window(&rt, 1).join("\n");
        assert!(read.contains("Elsewhere on this floor:"), "{read}");
        assert!(read.contains("in the green room"), "{read}");
    }

    #[tokio::test]
    async fn what_a_character_did_comes_back_in_its_own_turn() {
        // The actor is not left waiting for the world to tell it what it just
        // did — the act goes into its own window in the same turn. What it does
        // wait for is *everyone else* perceiving it.
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let spoke = act("shout", json!({"intent": "first"}));
        assert!(
            rt.act_on_world(2, &spoke).happened(),
            "the world took the speech"
        );
        // And what gets recorded for it is the act, in the shape every act has.
        assert_eq!(spoke.summary(), "shout — first");

        run(&rt, 2);
        assert!(
            !window(&rt, 1).join("\n").contains("first"),
            "it arrived outside a world moment"
        );
    }

    /// Every path into a character's window goes through a world moment. An
    /// act that pushed perception itself would be a second one, unbatched and
    /// off the sweep — and it would be invisible, because it would work.
    #[tokio::test]
    async fn nothing_perceives_anything_outside_a_world_moment() {
        let rt = vaulted();
        embody(&rt, 1, "m1", "green-room");
        embody(&rt, 2, "m2", "green-room");
        run(&rt, 1);

        let world = rt.hosted.get(WORLD).unwrap();
        for intent in ["one", "two", "three"] {
            rt.act_on_world(1, &act("speak", json!({ "intent": intent })));
        }
        run(&rt, 2);
        let before = window(&rt, 2).len();

        environment::advance(&world, &rt.bodies, &rt.scheduler, &|_| 0);
        // Past its heartbeat: speech to the room is worth a turn but does not
        // interrupt one, so it waits rather than preempting.
        run(&rt, 10_000);
        let after = window(&rt, 2).join("\n");
        assert!(
            window(&rt, 2).len() > before,
            "the moment delivered nothing"
        );
        for intent in ["one", "two", "three"] {
            assert!(after.contains(intent), "{intent} was lost: {after}");
        }
    }
}

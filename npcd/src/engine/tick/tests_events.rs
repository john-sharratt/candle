//! Tests for what reaches a character and when it is read: news, wakes, and the
//! turn an act's answer is owed.
//!
//! A character busy elsewhere keeps what arrives on its stored wake. An act that
//! answered brings the character back to a turn that can use the answer, while
//! an ordinary act leaves it unscheduled, which stops a character being asked
//! again into its own silence. Each test holds one rule of the standing task or
//! of that follow-up turn.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_scheduling::tests::{say, sched};

    use super::super::tests_suspension::tests::pause_ms;

    /// **A character thinking somewhere else loses nothing that arrives
    /// meanwhile.** Its task is awaiting a reflection's answer rather than
    /// sitting in [`Scheduler::wait_due`], so nothing drains its inbox — and
    /// every arrival stores a wake on its waker, so the moment the task
    /// returns to its wait it thinks at once, on all of it.
    #[tokio::test]
    async fn what_arrives_while_a_character_thinks_elsewhere_is_read_on_return() {
        let s = sched();
        s.wake(1, 0, 0);

        // The character's task is elsewhere: nothing calls wait_due or tick.
        s.deliver(1, 0, Salience::NORMAL, say("somebody comes in"));
        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        assert_eq!(s.inbox_depth(1), Some(2), "something was dropped");

        // Back from the other conversation: the stored wake completes at once…
        let due = tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the stored wake was lost");
        assert!(due);
        // …and the one turn reads everything that queued up meanwhile.
        let now = s.now_ms();
        let rec = s.tick(1, now, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 2, "on everything it missed");
    }

    /// **An act recorded before its result is completed with it.** A reflect
    /// goes into the window as the act alone and is answered seconds later by
    /// its own conversation; the row has to end up reading what came back, not
    /// stay a call with nothing after it.
    #[test]
    fn an_act_recorded_before_its_result_is_completed_with_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a box sags"));
        assert!(s.is_due(1, 1_000));
        s.tick(1, 1_000, 0, |_, _| vec!["reflect — it gave up".into()]);

        let done = "reflect — it gave up → I keep counting what gives up".to_string();
        assert!(s.amend_act(1, "reflect — it gave up", done.clone()));
        let window: Vec<String> = s
            .window_of(1, |w| w.turns().map(|t| t.text.clone()).collect())
            .unwrap();
        assert!(window.contains(&done), "{window:?}");
        assert!(
            !window.iter().any(|t| t == "reflect — it gave up"),
            "{window:?}"
        );

        assert!(
            !s.amend_act(1, "nothing like it", "x".into()),
            "nothing to amend"
        );
    }

    /// **The gap between two turns of a conversation is not idleness.**
    ///
    /// The bug this is here for: asked as "is there news queued right now", the
    /// answer is *no* in every gap between two things happening — and a
    /// conversation is mostly gaps. Two characters alternated for a hundred
    /// turns, each hearing the other speak and then, in the pause before the
    /// reply, being told that nothing had been asked of it.
    /// **A world announcement is news.** It is something that happened, put to
    /// everyone at once — so it resets the quiet clock and is not talked over by
    /// the standing task on the very next tick. Left off the news list, an
    /// announcement is read once and then buried under "nothing has been asked
    /// of you" in the most recent window slot.
    #[test]
    fn a_world_announcement_counts_as_news() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(
            1,
            0,
            Salience::URGENT,
            EventKind::Announcement {
                text: "the gate is sealed".into(),
            },
        );
        assert_eq!(s.quiet_for(1, 0), Some(0), "an unread announcement is news");
    }

    #[test]
    fn a_pause_in_a_conversation_does_not_read_as_a_quiet_character() {
        let s = sched();
        s.wake(1, 0, 0);

        // Somebody speaks at t=0 and the character reads it.
        s.deliver(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        assert_eq!(s.quiet_for(1, 0), Some(0), "unread news is news");
        s.tick(1, 0, 0, |_, _| Vec::new());

        // Its companion is now thinking. Several heartbeats pass with nothing
        // arriving — which is what waiting for a reply looks like from here.
        for gap in [4_000, 8_000, 20_000, 45_000] {
            assert_eq!(
                s.quiet_for(1, gap),
                Some(gap),
                "the clock runs from the last thing said, not from the last drain"
            );
        }

        // The situation and the standing task keep arriving throughout, and
        // neither is something that happened — so neither may reset the clock,
        // or the task would hold its own gate open forever.
        s.deliver(
            1,
            46_000,
            Salience::IDLE,
            EventKind::Situation {
                text: "You are in the green room.".into(),
            },
        );
        s.deliver(
            1,
            46_000,
            Salience::IDLE,
            EventKind::Nudge {
                text: "go and look around".into(),
            },
        );
        assert_eq!(s.quiet_for(1, 50_000), Some(50_000));

        // And a genuinely abandoned character does eventually come back round.
        assert!(s.quiet_for(1, 600_000).unwrap() > 45_000);
    }

    /// **The standing task is a cadence, not a latch.**
    ///
    /// It is not news, so it cannot move the news clock — which left the gate,
    /// once open, open forever. A character nobody was talking to got the task
    /// again on every tick, each copy superseding the last into the most recent
    /// position in the window. Identical instruction, identical context,
    /// identical decode: the character repeated one act until something outside
    /// it changed.
    #[test]
    fn the_standing_task_is_restated_on_a_cadence_rather_than_every_tick() {
        const AFTER: u64 = 90_000;
        // The world clock is an absolute time, not a counter that starts at
        // zero with the character — so a fresh one is already long past any
        // threshold, which is what makes it due immediately.
        const T0: u64 = 1_000_000;
        let s = sched();
        s.wake(1, 0, T0);

        // Never told anything: due at once, which is right — a character with
        // nothing ever asked of it is the one the task exists for.
        assert!(s.nudge_due(1, T0, AFTER));
        s.deliver(
            1,
            T0,
            Salience::IDLE,
            EventKind::Nudge {
                text: "go and look around".into(),
            },
        );
        s.tick(1, 0, T0, |_, _| Vec::new());

        // Still nothing happening — and it is NOT due again on the next tick,
        // nor the one after, nor a minute later.
        for gap in [4_000, 8_000, 45_000, 89_000] {
            assert!(
                !s.nudge_due(1, T0 + gap, AFTER),
                "restated {gap}ms after the last one"
            );
        }
        // A full cadence later, it is.
        assert!(s.nudge_due(1, T0 + AFTER, AFTER));

        // And news resets the other clock, so a character that gets spoken to
        // is not handed the task on the very next quiet tick.
        s.deliver(
            1,
            T0 + AFTER,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        s.tick(1, 0, T0 + AFTER, |_, _| Vec::new());
        assert!(
            !s.nudge_due(1, T0 + AFTER + 10_000, AFTER),
            "ten seconds after being spoken to"
        );
    }

    /// **A character doing what it was asked on the phone is not told that
    /// nothing calls for it.** What the conversation settled is news, and each
    /// act it takes afterwards keeps it out of the quiet the standing task
    /// waits for; once it has stopped for a full cadence, the task is due.
    #[test]
    fn a_settled_conversation_and_the_acts_after_it_hold_the_standing_task_off() {
        const AFTER: u64 = 90_000;
        const T0: u64 = 1_000_000;
        let s = sched();
        s.wake(1, 0, T0);
        s.deliver(
            1,
            T0,
            Salience::IDLE,
            EventKind::Nudge {
                text: "nothing calls for you".into(),
            },
        );
        s.tick(1, 0, T0, |_, _| Vec::new());

        // Long quiet, then the phone: asked to survey the house network.
        let asked = T0 + 10 * AFTER;
        assert!(s.nudge_due(1, asked, AFTER));
        s.deliver(
            1,
            asked,
            Salience::NORMAL,
            EventKind::Concluded {
                party: "John".into(),
                applied: "Meaning to: survey the house network and report.".into(),
            },
        );
        assert!(!s.nudge_due(1, asked, AFTER), "unread, it is news");
        s.tick(1, 0, asked, |_, _| vec!["terminal_connect".into()]);
        assert!(!s.nudge_due(1, asked + 60_000, AFTER));

        // A slow turn later it acts again, past a cadence from the request.
        let surveyed = asked + 80_000;
        s.think_again(1);
        s.tick(1, 0, surveyed, |_, _| vec!["survey_network".into()]);
        assert!(
            !s.nudge_due(1, asked + AFTER + 5_000, AFTER),
            "mid-task, a cadence after the request"
        );

        // Done and gone quiet: the standing task comes round again.
        assert!(s.nudge_due(1, surveyed + AFTER, AFTER));
    }

    /// **A mind may rest once it has turned to what last reached it.** With
    /// nothing arrived it has; a request arriving makes it not so until a turn
    /// of its own has acted; and news landing after that act is unanswered
    /// again.
    #[test]
    fn a_character_has_acted_since_news_only_after_a_turn_that_acted() {
        const T0: u64 = 1_000_000;
        let s = sched();
        assert!(s.acted_since_news(9), "nothing has reached a stranger");
        s.wake(1, 0, T0);
        assert!(s.acted_since_news(1), "nothing has reached it yet");

        s.deliver(
            1,
            T0 + 1_000,
            Salience::NORMAL,
            EventKind::Concluded {
                party: "John".into(),
                applied: "Asked how the cluster is.".into(),
            },
        );
        assert!(!s.acted_since_news(1), "asked, and nothing done about it");
        s.tick(1, 0, T0 + 2_000, |_, _| Vec::new());
        assert!(
            !s.acted_since_news(1),
            "a turn that did nothing answers nothing"
        );
        s.think_again(1);
        s.tick(1, 0, T0 + 3_000, |_, _| vec!["cluster_status".into()]);
        assert!(s.acted_since_news(1));

        s.deliver(1, T0 + 4_000, Salience::NORMAL, say("is it up?"));
        assert!(!s.acted_since_news(1), "asked again");
    }

    /// **Being told a hand went idle asks nothing.** A notice — a Drone come to
    /// rest — schedules no turn, and a mind that has acted on what last reached
    /// it may still rest on it: it is not news to answer.
    #[test]
    fn a_notice_neither_schedules_a_turn_nor_stands_unanswered() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("is the cluster up?"));
        s.tick(1, 0, 0, |_, _| vec!["cluster_status".to_string()])
            .expect("it ticked");
        assert!(s.acted_since_news(1));
        assert!(!s.is_due(1, 1), "a quiet character was due");

        let notice = EventKind::Notice {
            text: "Wick has come to rest, with nothing to carry.".into(),
        };
        s.deliver(1, 1, Salience::NORMAL, notice);
        assert!(!s.is_due(1, 2), "a notice scheduled a turn");
        assert!(s.acted_since_news(1), "a notice stood as news to answer");
    }

    /// **What a person asked stands until the character has worked at it and
    /// told them.** Telling them before any act came back leaves it standing;
    /// telling somebody else leaves it standing; and one nothing came of is
    /// dropped after [`ASKED_FOR_MS`].
    #[test]
    fn a_request_stands_until_it_is_worked_at_and_told() {
        const T0: u64 = 1_000_000;
        let s = sched();
        s.wake(1, 0, T0);
        assert_eq!(s.asked(1, T0), None);
        s.ask(1, "John", " Asked whether the television is on. ", T0);
        let standing = Some((
            "John".to_string(),
            "Asked whether the television is on.".to_string(),
        ));
        assert_eq!(s.asked(1, T0 + 1_000), standing);

        s.told(1, "John");
        assert_eq!(s.asked(1, T0 + 2_000), standing, "a promise, not an answer");
        s.worked(1);
        s.told(1, "Jitter");
        assert_eq!(s.asked(1, T0 + 3_000), standing, "told to somebody else");
        s.told(1, "John");
        assert_eq!(s.asked(1, T0 + 4_000), None);

        s.ask(1, "John", "Asked for the lights off.", T0);
        assert!(s.asked(1, T0 + ASKED_FOR_MS - 1).is_some());
        assert_eq!(s.asked(1, T0 + ASKED_FOR_MS), None, "nothing came of it");
        assert_eq!(s.asked(1, T0 + 1), None, "and it does not come back");
    }

    /// **The standing task never reaches a character deep in conversation**:
    /// one talking to a companion is not quiet, so [`Inbox::nudge_due`]'s
    /// quiet-since-news half stays shut.
    #[test]
    fn the_standing_task_waits_out_a_conversation() {
        const AFTER: u64 = 30_000;
        const T0: u64 = 1_000_000;
        let s = sched();
        s.wake(1, 0, T0);

        // A character being spoken to is NOT quiet, so the standing task's full
        // gate stays shut…
        s.deliver(
            1,
            T0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        s.tick(1, 0, T0, |_, _| Vec::new());
        assert!(
            !s.nudge_due(1, T0 + AFTER, 90_000),
            "the standing task must not reach a character mid-conversation"
        );
    }

    pub(in super::super) fn overheard() -> EventKind {
        EventKind::Speech {
            speaker: "Orion Vance".into(),
            text: "to somebody else".into(),
            to: crate::engine::event::Addressed::Other {
                who: "Maker-03".into(),
            },
        }
    }

    /// **Being told something is not being asked something.** Speech aimed past
    /// a character is information: it is queued for whenever the character next
    /// thinks, but it does not schedule a turn, and does not cut a pause short.
    /// A room of three that each answer whatever they overhear sustains a
    /// conversation nobody was asked to have.
    #[test]
    fn overheard_speech_does_not_schedule_a_turn() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("the room is quiet"));
        s.tick(1, 0, 0, |_, _| vec!["reflect — nothing here".to_string()])
            .expect("it ticked");
        assert!(!s.is_due(1, 1), "a quiet character was due");

        s.deliver(1, 1, Salience::from(npc_map::Weight::Note), overheard());
        assert!(!s.is_due(1, 2), "overhearing scheduled a turn");

        s.pause_for(1, 2, pause_ms());
        s.deliver(1, 3, Salience::from(npc_map::Weight::Note), overheard());
        assert!(!s.is_due(1, 4), "overhearing cut a pause short");
        assert!(s.is_due(1, 2 + pause_ms() + 1), "the pause never ended");
    }

    /// **What was overheard is read with whatever next wakes the character.**
    /// Not dropped — the mind design is explicit that evidence never is.
    #[test]
    fn overheard_speech_is_read_when_something_else_wakes_the_character() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("the room is quiet"));
        s.tick(1, 0, 0, |_, _| Vec::new()).expect("it ticked");
        s.deliver(1, 1, Salience::from(npc_map::Weight::Note), overheard());

        s.deliver(1, 2, Salience::NORMAL, say("the lights flicker"));
        assert!(s.is_due(1, 2), "a description did not wake it");
        let record = s.tick(1, 2, 2, |_, _| Vec::new()).expect("it ticked");
        let read = record.perceived.join(" ");
        assert!(read.contains("to somebody else"), "{read}");
        assert!(read.contains("lights flicker"), "{read}");
    }

    /// **Overhearing during a decode does not buy another turn.** The tick's
    /// end puts the character back on the schedule only for what wakes it.
    #[test]
    fn overhearing_while_thinking_does_not_schedule_the_next_turn() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something to answer"));
        s.tick(1, 0, 0, |_, _| {
            s.deliver(1, 0, Salience::from(npc_map::Weight::Note), overheard());
            vec!["speak — yes".to_string()]
        })
        .expect("it ticked");
        assert!(!s.is_due(1, 1), "overheard speech re-woke the character");

        s.deliver(1, 1, Salience::NORMAL, say("something else"));
        s.tick(1, 1, 1, |_, _| {
            s.deliver(1, 1, Salience::NORMAL, say("a thing that happened"));
            vec!["speak — yes".to_string()]
        })
        .expect("it ticked");
        assert!(s.is_due(1, 1), "an arrival during a decode lost its wake");
    }

    /// **An act that answered brings the character straight back.**
    ///
    /// What a board says exists in that act's outcome and nowhere else —
    /// nothing in the world perceives a document being read — so a character
    /// left unscheduled after asking is one holding the answer with no turn in
    /// which to use it. It read the board, went to sleep, and read the board
    /// again the next time anything woke it.
    #[test]
    fn an_act_that_answered_gets_a_turn_to_use_the_answer() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("go and read the board"));
        s.tick(1, 0, 0, |_, _| {
            s.think_again(1);
            vec!["read — the muster board".to_string()]
        })
        .expect("it ticked");

        assert!(
            s.is_due(1, 0),
            "it was told something and given nowhere to put it"
        );
        // **Due is not enough: the turn has to run.** The answer rides on the
        // conversation, not in the queue, so this pass drains nothing — and it
        // used to end there, with the character due, unticked, and the answer
        // unread until something else happened to wake it.
        let next = s
            .tick(1, 0, 0, |_, _| vec!["tell — the three names".to_string()])
            .expect("the turn it was owed ran");
        assert!(next.perceived.is_empty());
        // And having had it, it is quiet again until the world says something.
        assert!(!s.is_due(1, 1_000), "a turn owed once ran twice");
    }

    /// **A turn the decode never took is given back.** The queue was drained
    /// before the decode, so a decode that failed used to leave what the
    /// character had been told consumed and nothing scheduled again. Handed
    /// back, the same events are read by the turn that runs — after the wait,
    /// not before — and are not written into the window a second time.
    #[test]
    fn a_failed_turn_is_tried_again_on_the_same_events() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("the board says three names"));
        let start = s.begin_tick(1).expect("something to think about");
        let events = start.events.clone();
        let in_window = s.window_of(1, |w| w.len()).unwrap();
        s.end_tick(1, 0, 0, start, Vec::new());

        assert!(s.retry(1, 1_000, 5_000, events));
        assert!(!s.is_due(1, 2_000), "tried again before the wait was out");
        assert!(s.is_due(1, 6_000), "never tried again");
        let again = s.begin_tick(1).expect("the turn it was given back ran");
        assert_eq!(again.events.len(), 1, "the events were not handed back");
        assert_eq!(
            s.window_of(1, |w| w.len()).unwrap(),
            in_window,
            "written into the window twice"
        );
    }

    /// And every *other* act leaves it unscheduled, which is the half that
    /// stops the treadmill: a character that speaks and is immediately asked
    /// again speaks again, into the same silence, until its own window holds
    /// nothing but its own voice.
    #[test]
    fn an_ordinary_act_does_not_buy_another_turn() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("anything"));
        s.tick(1, 0, 0, |_, _| vec!["tell — something back".to_string()])
            .expect("it ticked");

        assert!(
            !s.is_due(1, u64::MAX - 1),
            "speaking scheduled its own next turn"
        );
    }
}

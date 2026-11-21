//! Tests for the scheduler's core behaviour: due-ness, preemption, batching and
//! ages.
//!
//! The scheduling is tested without a model, since the test supplies each
//! tick's acts. The central case is a character that runs out of things to
//! think about and must stay reachable. A quiet character is never due, a
//! high-salience arrival preempts, and a burst of arrivals is one wake.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    pub(in super::super) fn sched() -> Scheduler {
        Scheduler::new(8)
    }

    pub(in super::super) fn say(text: &str) -> EventKind {
        EventKind::Description { text: text.into() }
    }

    /// **A character that runs out of things to think about must stay
    /// reachable.**
    ///
    /// The wedge this closes, in the order it happens live:
    ///
    /// 1. An act that answered calls `think_again` — `due_at = 0`, `scheduled`
    ///    set — and nothing new arrives, so the tick ends with an empty queue
    ///    and `due_at` still `0`.
    /// 2. The next pass pops the character's heap entry and ticks it. The drain
    ///    yields nothing, so the tick returns early.
    /// 3. That early return used to leave `due_at` at `0` with the entry now
    ///    consumed — so `deliver`'s `already_queued` check (`due_at == 0`, read
    ///    as "an entry is standing in the heap") suppressed every future push.
    ///
    /// From there the character is unschedulable for the life of the daemon:
    /// events accumulate, `readiness` reports `Interrupted`, and nothing runs it.
    /// Measured on the live daemon — three characters wedged inside a minute,
    /// inboxes past sixty, and an URGENT broadcast could not move them.
    #[test]
    fn a_tick_that_finds_nothing_leaves_the_character_reachable() {
        let s = sched();
        s.wake(1, 0, 0);

        // An act that answered: due now, and no event behind it.
        s.deliver(1, 0, Salience::NORMAL, say("the board says three names"));
        assert!(s.is_due(1, 1_000));
        s.tick(1, 1_000, 0, |_, _| vec!["read the board".into()]);
        s.think_again(1);

        // The turn it is owed, on what the read came back with: nothing is
        // queued, and it runs anyway.
        assert!(s.is_due(1, 2_000), "it was not even due");
        let owed = s
            .tick(
                1,
                2_000,
                0,
                |_, _| vec!["write down the three names".into()],
            )
            .expect("the turn it was owed ran");
        assert!(owed.perceived.is_empty(), "{:?}", owed.perceived);

        // The pass that finds an empty queue, and nothing owed.
        if s.is_due(1, 2_500) {
            assert!(
                s.tick(1, 2_500, 0, |_, _| vec![]).is_none(),
                "there was nothing to drain and nothing owed, so there is no record"
            );
        }

        // The character must now be schedulable again by an ordinary arrival.
        s.deliver(1, 0, Salience::NORMAL, say("somebody comes in"));
        assert!(
            s.is_due(1, 3_000),
            "the character was left claiming to be queued while nothing could \
             ever run it — every later arrival is silently dropped on the floor"
        );
        let rec = s.tick(1, 3_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 1);
    }

    #[test]
    fn a_woken_character_joins_the_population() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert_eq!(s.population(), 2);
        s.retire(1);
        assert_eq!(s.population(), 1);
    }

    /// Waking twice must not reset an existing character's inbox — a mind-file
    /// reload calls this for the whole cast.
    #[test]
    fn waking_an_existing_character_is_idempotent() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.wake(1, 0, 0);
        assert_eq!(s.population(), 1);
        assert_eq!(s.census()[0].inbox_depth, 1, "the reload emptied the inbox");
    }

    #[test]
    fn an_event_for_an_unknown_character_is_refused() {
        let s = sched();
        assert!(!s.deliver(99, 0, Salience::NORMAL, say("x")));
    }

    /// An empty inbox is **quiet**, not blocked — the state that costs nothing
    /// and the one a character is in nearly all the time.
    #[test]
    fn an_empty_inbox_reads_quiet() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(s.census()[0].readiness, Readiness::Quiet);
        s.deliver(1, 0, Salience::NORMAL, say("a noise"));
        assert_eq!(s.census()[0].readiness, Readiness::Waiting);
    }

    /// **The tick must record why it ran, not what it looks like afterwards.**
    /// `drain` clears the interrupt flag, so reading readiness after it reports
    /// `Quiet` for every tick — and the Pulse feed's cause column, the one
    /// thing that says why a character woke, silently becomes a constant.
    #[test]
    fn a_ticks_recorded_cause_is_its_state_before_the_drain() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        let rec = s.tick(1, 0, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.cause, Readiness::Interrupted);

        s.deliver(1, 0, Salience::NORMAL, say("footsteps"));
        let rec = s.tick(1, 0, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.cause, Readiness::Waiting);
    }

    /// A high-salience arrival forces a tick now rather than waiting for the
    /// heartbeat.
    #[test]
    fn a_high_salience_event_preempts_and_becomes_due_immediately() {
        let s = sched();
        s.wake(1, 10_000, 0);
        assert!(!s.is_due(1, 10_000), "not due yet");
        s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        assert_eq!(s.census()[0].readiness, Readiness::Interrupted);
        assert!(s.is_due(1, 10_000));
    }

    /// **A burst of urgent arrivals is one wake, not one wake each.** The
    /// waker coalesces — twenty deliveries store one permit — and the single
    /// tick that permit buys drains the whole burst.
    #[tokio::test]
    async fn a_burst_of_preempting_arrivals_is_one_wake_that_drains_it_all() {
        let s = sched();
        s.wake(1, 10_000, 0);
        for _ in 0..20 {
            s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        }

        assert!(tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the burst stored no wake"));
        let rec = s.tick(1, 10_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 20);

        // An arrival during the decode is a different case and must still wake
        // the character: its permit was stored while the task was ticking, so
        // the next wait completes at once.
        s.deliver(1, 0, Salience::URGENT, say("and again"));
        assert!(s.is_due(1, 10_000));
        assert!(tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the mid-decode arrival's wake was lost"));
    }

    /// Ordinary traffic waits for the character's own heartbeat. It is not
    /// dropped — nothing ever is — it simply does not jump the queue.
    #[test]
    fn ordinary_traffic_waits_without_being_dropped() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a rumour"));
        assert_eq!(s.census()[0].inbox_depth, 1);
        assert_eq!(s.census()[0].readiness, Readiness::Waiting);
    }

    /// A busy character drains everything at once — one better-informed step,
    /// not several thrashing ones.
    #[test]
    fn a_tick_drains_the_whole_batch() {
        let s = sched();
        s.wake(1, 0, 0);
        for i in 0..5 {
            s.deliver(1, 0, Salience::NORMAL, say(&format!("event {i}")));
        }
        let rec = s
            .tick(1, 0, 0, |events, _| {
                assert_eq!(events.len(), 5, "the tick did not drain the batch");
                vec!["waits".into()]
            })
            .expect("ticked");
        assert_eq!(rec.perceived.len(), 5);
        assert_eq!(rec.inbox_after, 0);
        assert_eq!(s.census()[0].readiness, Readiness::Quiet);
    }

    /// **A character with nothing to react to does not think.**
    ///
    /// It used to: an empty inbox synthesised a heartbeat event and the
    /// character took a turn on it, so that "nothing arrived" could not mean
    /// "dead forever". The cost was not the decode, it was what the decode
    /// wrote — an act, appended to the character's own window, every beat. An
    /// embodied character is floored at four seconds, so a quiet room filled a
    /// twenty-four-turn window with copies of one gesture inside two minutes,
    /// and after that the next token was certain and the character could not
    /// say anything else.
    #[test]
    fn a_quiet_character_does_not_think_at_all() {
        let s = sched();
        s.wake(1, 0, 0);
        assert!(
            s.tick(1, 0, 0, |_, _| panic!(
                "thought with nothing to think about"
            ))
            .is_none(),
            "an empty inbox produced a turn"
        );
    }

    /// And it is not *scheduled*, either — nothing to serve, so nothing is due
    /// however long you wait. Being quiet costs no decodes and no heap.
    #[test]
    fn a_quiet_character_is_never_due() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        assert!(s.is_due(1, 0));
        s.tick(1, 0, 0, |_, _| vec!["speak — yes".into()])
            .expect("ticked");

        for at in [0, 10_000, 600_000, u64::MAX - 1] {
            assert!(
                !s.is_due(1, at),
                "a character with an empty inbox came due at {at}"
            );
        }
    }

    /// Alertness with no combat branch: something happens, the metabolism
    /// tightens, and it decays back rather than snapping.
    #[test]
    fn alertness_tightens_on_a_preempt() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            IDLE_HEARTBEAT.as_millis() as u64
        );

        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            ALERT_HEARTBEAT.as_millis() as u64
        );
        // It does not decay from here. Alertness used to relax over the ticks a
        // quiet character took, and a quiet character no longer takes any — so
        // what this number records is that something recently happened, which
        // is what the Pulse feed shows. Nothing schedules from it.
    }

    /// **A pace no longer schedules anything**, and this is what that means.
    ///
    /// `Pace::WORKING` floored an embodied character's heartbeat at four
    /// seconds and never let it relax, so *being in a world* meant thinking
    /// every four seconds whether or not anything had happened. That floor was
    /// the engine of the repetition: not the sampler, not the grammar — a
    /// character scheduled to speak fifteen times a minute into a room where
    /// nothing was going on.
    ///
    /// Neither pace makes a character due now. Whether it thinks is decided by
    /// its inbox, and both of these are empty.
    #[test]
    fn no_pace_makes_a_quiet_character_think() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert!(s.set_pace(2, Pace::WORKING, 0));

        for at in [0, 4_000, 120_000, 600_000] {
            assert!(!s.is_due(1, at), "the ambient one was due at {at}");
            assert!(
                !s.is_due(2, at),
                "a working pace still put a quiet character on the schedule at {at}"
            );
        }
    }

    #[test]
    fn a_character_is_ambient_until_it_is_told_otherwise() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(s.pace_of(1), Some(Pace::AMBIENT));
        assert_eq!(s.pace_of(2), None, "an absent character has a pace");
        assert!(
            !s.set_pace(2, Pace::WORKING, 0),
            "paced an absent character"
        );
    }

    /// **Changing a pace must not conjure a thought.**
    ///
    /// It used to pull a character forward to the new floor, which is how
    /// `embody` — which sets `WORKING` — put every character in a world onto a
    /// four-second treadmill. Setting a pace now says how alert the character
    /// reads as and nothing else.
    #[test]
    fn changing_a_pace_does_not_schedule_a_thought() {
        let s = sched();
        s.wake(1, 0, 0);
        // A freshly-woken character starts unscheduled; a tick on something
        // real parks it there again, and neither pace change below moves it.
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.tick(1, 0, 0, |_, _| vec![]);

        s.set_pace(1, Pace::WORKING, 0);
        assert_eq!(s.pace_of(1), Some(Pace::WORKING));
        for at in [0, 4_000, 120_000] {
            assert!(
                !s.is_due(1, at),
                "quickening scheduled a character with an empty inbox at {at}"
            );
        }

        s.set_pace(1, Pace::AMBIENT, 0);
        assert_eq!(s.pace_of(1), Some(Pace::AMBIENT));
        assert!(!s.is_due(1, u64::MAX - 1));
    }

    /// A pace cannot be set faster than the world moves — below that a
    /// character spends decodes discovering nothing has changed.
    #[test]
    fn a_pace_cannot_be_set_faster_than_the_world_moves() {
        assert_eq!(Pace::at_most(Duration::from_millis(1)), Pace::WORKING);
        assert_eq!(Pace::at_most(ALERT_HEARTBEAT), Pace::WORKING);
        assert_eq!(Pace::at_most(IDLE_HEARTBEAT), Pace::AMBIENT);
        assert_eq!(
            Pace::at_most(Duration::from_secs(30)).slowest(),
            Duration::from_secs(30)
        );
    }

    /// A tick with real events must not relax — the character heard something.
    #[test]
    fn a_tick_with_real_events_does_not_relax() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        let alert = s.census()[0].heartbeat_ms;
        s.deliver(1, 0, Salience::NORMAL, say("footsteps"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            alert,
            "relaxed despite an event"
        );
    }

    /// Perception lands in the window before the decode reads it, and acts land
    /// after — the transcript has to be in the order it happened.
    #[test]
    fn perception_precedes_the_decode_and_acts_follow_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("the gate groans open"));
        s.tick(1, 0, 100, |_, window| {
            assert_eq!(
                window.len(),
                1,
                "the decode could not see what it perceived"
            );
            vec!["steps back".into()]
        });
        let c = &s.census()[0];
        assert_eq!(c.window_turns, 2, "the act did not land in the window");
    }

    #[test]
    fn a_broadcast_reaches_everyone() {
        let s = sched();
        for id in 1..=4 {
            s.wake(id, 0, 0);
        }
        // The clock is per character: thunder over two worlds happens at two
        // times, and each character's event must carry its own world's instant.
        assert_eq!(
            s.broadcast(|id| id * 100, Salience::NORMAL, say("thunder")),
            4
        );
        for c in s.census() {
            assert_eq!(c.inbox_depth, 1);
        }
    }

    /// **How long since it last acted**, which is the figure the cast strip
    /// leads with — and the one that says whether a world is alive at all.
    ///
    /// An age rather than a timestamp, measured against the scheduler's own
    /// clock. Neither of the other two would do: the driver's starts minutes
    /// after `Scheduler::new`, so a duration written by one and read against the
    /// other is out by however long loading took, and the wall clock belongs to
    /// this machine while the reader is a browser on somebody else's.
    #[test]
    fn a_character_reports_how_long_since_it_last_acted() {
        let s = sched();
        s.wake(1, 0, 0);
        // Never, not "a very long time" — a character that has not acted is a
        // different fact from one that has gone quiet, and the strip says so.
        assert_eq!(s.census()[0].acted_ms_ago, None);

        // Thinking is not acting. A tick that decided on nothing leaves the
        // question where it was, or it answers "how long since it last ran".
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.tick(1, 0, 0, |_, _| Vec::new()).expect("it ticked");
        assert_eq!(
            s.census()[0].acted_ms_ago,
            None,
            "a turn that produced nothing counted as an act"
        );

        s.deliver(1, 0, Salience::NORMAL, say("something else"));
        s.tick(1, 0, 0, |_, _| vec!["tell — anything".to_string()])
            .expect("it ticked");
        let age = s.census()[0]
            .acted_ms_ago
            .expect("it acted and reported nothing");
        assert!(
            age < 5_000,
            "the age is measured against the wrong clock: {age}"
        );
    }

    /// A retired character is gone: not due, not tickable — and its own task,
    /// wherever it is waiting, is woken to find that out and exit.
    #[tokio::test]
    async fn a_retired_character_is_not_scheduled_and_its_wait_ends() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("x"));
        s.retire(1);
        assert!(!s.is_due(1, u64::MAX));
        assert!(s.tick(1, 0, 0, |_, _| vec![]).is_none());
        assert!(
            !tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
                .await
                .expect("the retired character's wait never ended"),
            "a wait on a retired character claimed it was due"
        );
    }
}

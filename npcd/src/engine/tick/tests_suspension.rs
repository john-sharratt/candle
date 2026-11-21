//! Tests for the scheduler's stall mechanism: suspension, rest and pauses.
//!
//! A suspended or resting character leaves the schedule and comes back when it
//! is let go, woken early by anything that reaches it. The tests check the
//! mechanism; which acts stall and for how long belongs to the cost table. A
//! pause is only a due time, so one cut short or taken mid-tick is neither spun
//! nor overwritten.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use super::super::tests_scheduling::tests::{say, sched};

    /// A stall long enough to be visible against the heartbeat.
    ///
    /// **A number of this module's own, on purpose.** What is under test here
    /// is the scheduler's stall *mechanism* — that a stalled character leaves
    /// the schedule, comes back on time, and is woken early by anything that
    /// arrives. Which acts stall, and for how long, is a question for the cost
    /// table and is answered by its own tests. Reading the table from here
    /// coupled the two: when `reflect` stopped stalling, seven scheduler tests
    /// failed for a reason that had nothing to do with scheduling.
    pub(in super::super) fn pause_ms() -> u64 {
        30_000
    }

    /// **A suspended character is never due, and comes back once resumed with
    /// everything that reached it meanwhile.** It is left out of `due_now`
    /// however urgent its inbox, refused a turn, and nothing it was told is
    /// lost: the one turn after the resume reads all of it.
    #[test]
    fn a_suspended_character_is_never_due_and_returns_once_resumed() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert!(s.suspend(1, true));
        assert!(!s.suspend(1, true), "suspending it twice changes nothing");
        assert!(s.is_suspended(1));
        assert!(!s.is_suspended(2));

        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        s.deliver(1, 0, Salience::NORMAL, say("somebody comes in"));
        s.deliver(2, 0, Salience::NORMAL, say("somebody comes in"));
        assert_eq!(s.due_now(1_000), vec![2]);
        assert!(!s.is_due(1, u64::MAX - 1));
        assert!(
            s.begin_tick(1).is_none(),
            "a suspended character took a turn"
        );
        assert_eq!(s.inbox_depth(1), Some(2), "the refused turn drained it");

        assert!(s.suspend(1, false));
        assert!(!s.is_suspended(1));
        assert_eq!(s.due_now(1_000), vec![1, 2]);
        let rec = s.tick(1, 1_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(
            rec.perceived,
            vec!["an alarm".to_string(), "somebody comes in".to_string()]
        );
    }

    /// **A character at rest is held as a suspended one is**, and is let go
    /// only when neither holds it: woken from rest while suspended it stays
    /// held, and the turn after both are lifted reads what reached it.
    #[test]
    fn a_character_at_rest_is_held_until_it_is_woken() {
        let s = sched();
        s.wake(1, 0, 0);
        assert!(s.rest(1, true));
        assert!(!s.rest(1, true), "put at rest twice changes nothing");
        assert!(s.is_resting(1) && s.is_held(1));
        assert!(s.resting_since(1).is_some());

        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        assert!(s.due_now(1_000).is_empty());
        assert!(s.begin_tick(1).is_none(), "a character at rest took a turn");
        assert_eq!(s.inbox_depth(1), Some(1), "the refused turn drained it");

        // Suspended meanwhile, waking from rest does not let it go.
        s.suspend(1, true);
        assert!(s.rest(1, false));
        assert!(!s.is_resting(1) && s.is_held(1));
        assert!(s.due_now(1_000).is_empty());
        assert_eq!(s.resting_since(1), None);

        s.suspend(1, false);
        assert_eq!(s.due_now(1_000), vec![1]);
        let rec = s.tick(1, 1_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived, vec!["an alarm".to_string()]);
    }

    /// **The turn after a long suspension reads the newest of each source and
    /// a count of the rest**, never the whole backlog: 25 things happen around
    /// a suspended character, and it resumes to the newest 20, behind one line
    /// standing for the 5 before them.
    #[test]
    fn a_resumed_turn_reads_its_backlog_bounded_per_source() {
        let s = sched();
        s.wake(1, 0, 0);
        s.suspend(1, true);
        for i in 0..25 {
            s.deliver(1, 0, Salience::NORMAL, say(&format!("line {i}")));
        }
        assert_eq!(s.inbox_depth(1), Some(25), "nothing is dropped while held");

        s.suspend(1, false);
        let rec = s.tick(1, 1_000, 0, |_, _| vec![]).expect("ticked");
        let mut want =
            vec!["While you were away, 5 more things that happened around you.".to_string()];
        want.extend((5..25).map(|i| format!("line {i}")));
        assert_eq!(rec.perceived, want);
    }

    /// **The scheduler holds the cast's suspended set as it is handed, and
    /// says what moved.** A character the set no longer names is resumed, one
    /// it names newly is suspended, and one in neither is untouched — including
    /// one with no inbox here, which is suspended already on the day it is
    /// woken.
    #[test]
    fn following_the_suspended_set_reports_each_change_once() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        s.suspend(1, true);

        let moved = s.follow_suspended(HashSet::from([2, 9]));
        assert_eq!(moved, vec![(1, false), (2, true), (9, true)]);
        assert!(s.follow_suspended(HashSet::from([2, 9])).is_empty());
        assert_eq!(s.suspended(), HashSet::from([2, 9]));

        s.wake(9, 0, 0);
        s.deliver(9, 0, Salience::URGENT, say("an alarm"));
        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        assert_eq!(
            s.due_now(1_000),
            vec![1],
            "a woken suspended character is due"
        );
    }

    /// **A suspended task parks, and the resume is what moves it.** An arrival
    /// rings the waker and the task goes straight back to waiting; resuming
    /// rings it and the task finds itself due.
    #[tokio::test]
    async fn a_suspended_task_waits_until_it_is_resumed() {
        let s = Arc::new(sched());
        s.wake(1, 0, 0);
        s.suspend(1, true);
        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        assert!(
            tokio::time::timeout(Duration::from_millis(200), s.wait_due(1))
                .await
                .is_err(),
            "a suspended character's task came due"
        );

        let waiting = tokio::spawn({
            let s = Arc::clone(&s);
            async move { s.wait_due(1).await }
        });
        tokio::task::yield_now().await;
        s.suspend(1, false);
        let due = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("the resume did not wake the task")
            .unwrap();
        assert!(due);
    }

    /// The neighbouring path, which does **not** wedge — and is here to keep it
    /// that way.
    ///
    /// A pause leaves `due_at` at a real deadline rather than at `0`, so
    /// `deliver`'s `already_queued` check reads false and an arrival still
    /// queues it. That is the whole reason `think_again` was the one that broke:
    /// the two differ only in the value they park `due_at` at, and one of those
    /// values happens to be the sentinel the queue check keys on. Anything that
    /// makes a pause park at zero brings the wedge back here.
    #[test]
    fn a_character_that_wakes_from_a_pause_to_an_empty_queue_stays_reachable() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a light goes out"));
        s.tick(1, 1_000, 0, |_, _| vec!["reflect".into()]);
        s.pause_for(1, 1_000, pause_ms());

        // Its deadline comes round with nothing waiting.
        assert!(s.is_due(1, 1_000 + pause_ms() + 1));
        assert!(s
            .tick(1, 1_000 + pause_ms() + 1, 0, |_, _| vec![])
            .is_none());

        s.deliver(1, 0, Salience::URGENT, say("a door slams"));
        assert!(
            s.is_due(1, 1_000 + pause_ms() + 2_000),
            "an urgent arrival could not wake it"
        );
    }

    /// **A paused character stops thinking**, which is the whole of what a
    /// pause is for.
    ///
    /// The free-text `wait` it began as left the heartbeat alone, so the
    /// character woke four seconds later and decided to wait again, and again —
    /// a busy-loop wearing the word. Here the deadline *is* the schedule.
    #[test]
    fn a_pause_takes_the_character_out_of_the_schedule() {
        let s = sched();
        s.wake(1, 0, 0);
        // Due because something reached it, not because it woke.
        s.deliver(1, 0, Salience::NORMAL, say("anything"));
        assert!(s.is_due(1, 10_000), "an arrival did not schedule it");
        s.tick(1, 0, 0, |_, _| vec![]);

        assert!(
            s.pause_for(1, 0, pause_ms()),
            "there was a character to stop"
        );
        assert!(
            !s.is_due(1, pause_ms() - 1),
            "it came back before the pause was up"
        );
        assert!(
            s.is_due(1, pause_ms() + 1),
            "the pause never ended, so it will never think again"
        );
    }

    #[test]
    fn pausing_a_character_that_is_not_there_says_so() {
        let s = sched();
        assert!(!s.pause_for(99, 0, pause_ms()));
    }

    /// **Anything that reaches a paused character brings it straight back**,
    /// and nothing had to be written to make it so.
    ///
    /// This is the whole benefit over the typed wait it replaced. That was a
    /// subscription to one named thing, so everything *else* the world did —
    /// being messaged, being spoken to by the wrong person, an operator saying
    /// something in the room — landed in the inbox and sat there until the
    /// patience ran out, two minutes after whoever did it had gone. It needed a
    /// rousing bar, tuned against a ladder, to get half of that back.
    ///
    /// A pause is only a due time, and `deliver` moves every character's due
    /// time to now, so all of it comes back at once — including the quiet
    /// traffic, which under the old rule was deliberately left out and is the
    /// one case here that changed.
    #[test]
    fn anything_arriving_at_all_cuts_a_pause_short() {
        for (what, salience, kind) in [
            (
                "being spoken to",
                Salience::URGENT,
                EventKind::Speech {
                    speaker: "Orion Vance".into(),
                    text: "look at me".into(),
                    to: crate::engine::event::Addressed::You,
                },
            ),
            (
                "a message",
                Salience::URGENT,
                EventKind::Message {
                    thread: "Johnathan Sharratt".into(),
                    from: "Johnathan Sharratt".into(),
                    unread: 1,
                },
            ),
            (
                "the room changing under you",
                Salience::IDLE,
                EventKind::Situation {
                    text: "You are in the green room.".into(),
                },
            ),
        ] {
            let s = sched();
            s.wake(1, 0, 0);
            s.pause_for(1, 0, pause_ms());
            assert!(!s.is_due(1, 1), "it did not stop");

            s.deliver(1, 0, salience, kind);
            assert!(s.is_due(1, 1), "{what} did not bring it back");
        }
    }

    /// **A pause cut short goes back to the ordinary schedule.**
    ///
    /// The path that used to spin: `deliver` sets `due_at = 0` for an arrival,
    /// and the end of `tick` puts the deadline back. A pause that survived its
    /// own interruption left the character due at zero for ever, ticking flat
    /// out on an empty queue, and every view of it read "healthy".
    #[test]
    fn a_pause_cut_short_does_not_leave_the_character_spinning() {
        let s = sched();
        s.wake(1, 0, 0);
        s.pause_for(1, 0, pause_ms());
        s.deliver(
            1,
            0,
            Salience::URGENT,
            EventKind::Speech {
                speaker: "Orion Vance".into(),
                text: "look at me".into(),
                to: crate::engine::event::Addressed::You,
            },
        );
        assert!(s.is_due(1, 0), "it did not wake");
        s.tick(1, 0, 0, |_, _| vec!["speak — yes".to_string()])
            .expect("it ticked");
        assert!(
            !s.is_due(1, 0),
            "due again at the same instant — the busy-loop is back"
        );
    }

    /// **A pause taken during a tick survives the end of that tick.**
    ///
    /// The act loop arms the pause from inside the closure, and the line at the
    /// end of `tick` runs afterwards. Without the flag it overwrites the
    /// deadline with `NEVER` — on the empty queue a paused character has by
    /// definition — which is not "wait two minutes" but "never think again",
    /// and reads from every angle as a healthy quiet character.
    #[test]
    fn a_pause_taken_mid_tick_is_not_overwritten_by_the_end_of_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something to answer"));
        s.tick(1, 0, 0, |_, _| {
            s.pause_for(1, 0, pause_ms());
            vec!["reflect — that nothing here needs me".to_string()]
        })
        .expect("it ticked");

        assert!(!s.is_due(1, pause_ms() - 1), "the pause was cut short");
        assert!(
            s.is_due(1, pause_ms() + 1),
            "the pause became a character that never thinks again"
        );
    }

    /// A character parked at `NEVER` sleeps on its waker alone: the wait is
    /// still pending after real time has passed, and one delivery ends it.
    #[tokio::test]
    async fn an_unscheduled_character_sleeps_until_something_arrives() {
        let s = Arc::new(sched());
        s.wake(1, 0, 0);

        let waiting = tokio::spawn({
            let s = Arc::clone(&s);
            async move { s.wait_due(1).await }
        });
        // Genuinely parked: nothing has arrived, so the wait must not end.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished(), "a quiet character's wait returned");

        s.deliver(1, 0, Salience::NORMAL, say("something"));
        let due = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("the delivery did not wake the wait")
            .expect("the waiting task panicked");
        assert!(due);
    }
}

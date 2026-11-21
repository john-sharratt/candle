//! Tests for the journal cadence, the day roll-over and the recent-tick feed.
//!
//! A stretch of a character's life is waiting only once the cadence is reached
//! and no draft is running. A day roll-over keeps the turns it clears for the
//! journal. The recent feed stays bounded, and a character id past 2^53 must
//! reach the wire exactly.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use journal_core::entry::{Entry, Item};

    use journal_core::period::Period;

    use super::*;

    use crate::engine::journal::state::EVERY_TURNS;

    use super::super::tests_scheduling::tests::{say, sched};

    #[test]
    fn the_day_rolls_over_once_and_clears_the_window() {
        use crate::clock::DAY_MS;
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("yesterday"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert!(s.census()[0].window_turns > 0);

        let rolled = s.roll_day(1, DAY_MS + 5).unwrap();
        assert_eq!((rolled.from, rolled.to), (0, 1));
        assert_eq!(s.census()[0].window_turns, 0, "the tail survived the day");
        assert_eq!(s.census()[0].day, Some(1));
        assert!(s.roll_day(1, DAY_MS + 6).is_none(), "rolled over twice");
    }

    /// Deliver `n` descriptions and let the character think over them, so its
    /// window holds `n` turns.
    pub(in super::super) fn lived(s: &Scheduler, id: u64, n: usize) {
        for i in 0..n {
            s.deliver(id, 0, Salience::NORMAL, say(&format!("event {i}")));
        }
        s.tick(id, 0, 0, |_, _| vec![]);
    }

    #[test]
    fn a_stretch_is_waiting_once_the_cadence_is_reached_and_not_while_a_draft_runs() {
        let s = sched();
        s.wake(0, 0, 0);
        lived(&s, 0, EVERY_TURNS as usize);
        let waiting = s.journal_waiting(0).expect("a stretch is waiting");
        assert_eq!(waiting.turns, EVERY_TURNS as usize);
        let due = s.journal_begin(0).unwrap();
        assert_eq!(due.span.turns.len(), EVERY_TURNS as usize);
        assert!(s.journal_waiting(0).is_none(), "one draft at a time");
        due.state.lock().unwrap().journal_mut().abandon();
        assert!(
            s.journal_waiting(0).is_none(),
            "an abandoned draft is not retried until new turns land"
        );
        lived(&s, 0, EVERY_TURNS as usize);
        let waiting = s.journal_waiting(0).expect("a new stretch is waiting");
        assert_eq!(
            waiting.turns,
            2 * EVERY_TURNS as usize,
            "the abandoned turns are covered again"
        );
    }

    #[test]
    fn a_draft_can_be_forced_before_the_cadence_but_only_over_new_turns() {
        let s = sched();
        s.wake(0, 0, 0);
        assert!(s.journal_begin(0).is_none(), "nothing has happened yet");
        lived(&s, 0, 3);
        let due = s.journal_begin(0).expect("three turns are enough to force");
        assert_eq!(due.span.turns.len(), 3);
        assert!(s.journal_begin(0).is_none(), "one draft at a time");
        assert_eq!(s.journal_of(0, |j| j.journal().pending()), Some(true));
        assert!(s.journal_begin(99).is_none());
    }

    #[test]
    fn a_journal_is_read_in_place_and_a_stranger_has_none() {
        let s = sched();
        s.wake(0, 0, 0);
        assert_eq!(s.journal_of(0, |j| j.journal().written()), Some(0));
        assert_eq!(s.journal_of(99, |j| j.journal().written()), None);
    }

    #[test]
    fn nothing_is_waiting_before_the_cadence_or_for_a_stranger() {
        let s = sched();
        s.wake(0, 0, 0);
        lived(&s, 0, 3);
        assert!(s.journal_waiting(0).is_none());
        assert!(s.journal_waiting(99).is_none());
    }

    #[test]
    fn the_census_reports_the_journal() {
        let s = sched();
        s.wake(0, 0, 0);
        let row = &s.census()[0];
        assert_eq!(
            (row.journal_written, row.journal_open, row.journal_drafting),
            (0, 0, false)
        );

        lived(&s, 0, EVERY_TURNS as usize);
        let due = s.journal_begin(0).expect("a draft is due");
        assert!(s.census()[0].journal_drafting, "a draft in flight shows");

        let mut writing = Entry {
            id: 0,
            from_turn: due.span.from_turn,
            to_turn: due.span.to_turn,
            from_ms: 0,
            to_ms: 1,
            summary: "Asked about the shield.".into(),
            claims: vec![],
            intend: vec![],
            opened: vec![Item {
                id: 0,
                text: "who holds the shield?".into(),
            }],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
        };
        writing = due.state.lock().unwrap().journal().stage(writing);
        due.state.lock().unwrap().journal_mut().kept(writing);
        let row = &s.census()[0];
        assert_eq!(
            (row.journal_written, row.journal_open, row.journal_drafting),
            (1, 1, false)
        );
    }

    #[test]
    fn a_restored_journal_is_the_one_the_character_drafts_against() {
        let s = sched();
        s.wake(0, 0, 0);
        let mut restored = JournalState::new(0);
        restored.journal_mut().kept(Entry {
            id: 0,
            from_turn: 1,
            to_turn: 2,
            from_ms: 0,
            to_ms: 1,
            summary: "Woke.".into(),
            claims: vec![],
            intend: vec![],
            opened: vec![],
            resolved: vec![],
            period: Period::Step,
            digest: String::new(),
        });
        s.set_journal(0, restored);
        lived(&s, 0, EVERY_TURNS as usize + 2);
        let due = s.journal_begin(0).expect("a draft is due");
        assert_eq!(
            due.span.from_turn, 3,
            "the span starts after what is already written"
        );
        assert_eq!(due.state.lock().unwrap().journal().entries().count(), 1);
    }

    #[test]
    fn the_turns_a_day_is_about_to_clear_are_still_waiting_afterwards() {
        use crate::clock::DAY_MS;
        let s = sched();
        s.wake(1, 0, 0);
        lived(&s, 1, 5);
        s.roll_day(1, DAY_MS + 5).unwrap();
        assert_eq!(s.census()[0].window_turns, 0);
        let waiting = s
            .journal_waiting(1)
            .expect("the unwritten turns are asked about before they are lost");
        assert_eq!(waiting.turns, 5);
        let due = s.journal_begin(1).unwrap();
        assert_eq!(due.span.turns.len(), 5);
    }

    #[test]
    fn a_day_with_nothing_unwritten_leaves_nothing_waiting() {
        use crate::clock::DAY_MS;
        let s = sched();
        s.wake(1, 0, 0);
        s.roll_day(1, DAY_MS + 5).unwrap();
        assert!(s.journal_waiting(1).is_none());
    }

    /// The feed is an instrument, not a log. It must stay bounded under load.
    #[test]
    fn the_recent_feed_is_bounded() {
        let s = Scheduler::new(4);
        s.wake(1, 0, 0);
        for i in 0..20 {
            // Something to think about each time: a character with an empty
            // inbox does not tick, so a bare `tick` produces no record at all.
            s.deliver(1, i, Salience::NORMAL, say("something"));
            s.tick(1, 0, 0, |_, _| vec![]);
        }
        let r = s.recent_where(100, |_| true);
        assert_eq!(r.len(), 4);
        // Newest last, and contiguous.
        assert!(r.windows(2).all(|w| w[1].tick == w[0].tick + 1));
    }

    /// **A feed for one character is filtered within the same window**, and
    /// its newest tick is found without copying any: six ticks alternating
    /// between two characters, ticks 0–5.
    #[test]
    fn the_feed_filters_within_its_window_and_finds_a_characters_newest() {
        let s = Scheduler::new(8);
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        for i in 0..6 {
            let npc = if i % 2 == 0 { 2 } else { 1 };
            s.deliver(npc, i, Salience::NORMAL, say("something"));
            s.tick(npc, 0, 0, |_, _| vec![]).expect("ticked");
        }
        let ones: Vec<u64> = s
            .recent_where(4, |t| t.npc_id == 1)
            .iter()
            .map(|t| t.tick)
            .collect();
        assert_eq!(ones, [3, 5]);
        assert_eq!(s.latest_tick_of(1), Some(5));
        assert_eq!(s.latest_tick_of(9), None);
    }

    /// **An id past 2^53 must survive the wire.**
    ///
    /// Character ids use the whole `u64` range. As a JSON number a large one
    /// arrives at the console rounded, so the same character would carry
    /// different ids in the feed and the census — the cast strip colouring one
    /// row and the feed another, and a click filtering to a character that does
    /// not exist. Invisible until an id happens to be large, which is why it is
    /// asserted against a real one.
    #[test]
    fn a_large_character_id_reaches_the_wire_exactly() {
        let big = 6_817_662_845_163_923_144u64;
        assert!(big > (1u64 << 53), "the test's id is not past the danger");

        let s = sched();
        s.wake(big, 0, 0);
        s.deliver(big, 0, Salience::NORMAL, say("something to answer"));
        s.tick(big, 0, 0, |_, _| vec![]).expect("ticked");

        let wire_id = npcs::npc_id_wire(big);
        let json = serde_json::to_string(&s.recent_where(1, |_| true)[0]).unwrap();
        assert!(
            json.contains(&format!("\"npc_id\":\"{wire_id}\"")),
            "a tick's id is not a string: {json}"
        );
        let census = serde_json::to_string(&s.census()[0]).unwrap();
        assert!(
            census.contains(&format!("\"npc_id\":\"{wire_id}\"")),
            "a census row's id is not a string: {census}"
        );

        // And it round-trips: the exact value comes back, not a rounded one.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            npcs::npc_id_of_wire(v["npc_id"].as_str().unwrap()).unwrap(),
            big
        );
    }
}

//! Tests the world's event window, and the reading of events that another copy
//! of the world produced.
//!
//! The log is a window: its oldest events are dropped in batches, a reader's
//! position survives a trim, and trimming runs on a still world as well. A
//! world that follows another copy takes in witnessed events as if they were
//! spoken locally, and state copied from one world into another reads back the
//! same.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_bodies::tests::{at, vault};

    // ── following a world kept elsewhere ────────────────────────────────────

    /// **What the authority's world did reaches the bodies here.** An event
    /// sent as JSON and witnessed is heard by a body in the room as if spoken
    /// here, stamped on this world's clock.
    #[test]
    fn a_witnessed_event_is_heard_like_a_local_one() {
        let mut authority = one_speaker();
        authority.say("m1", "from the other machine").unwrap();
        let sent = serde_json::to_string(authority.log().last().unwrap()).unwrap();

        let mut here = one_speaker();
        here.enter("m2", "Maker-02", at("green-room")).unwrap();
        here.mark_seen("m2");
        let before = here.now;
        let event: Event = serde_json::from_str(&sent).unwrap();
        let stamped = here.witness([event]);
        assert!(stamped > before);
        let heard = crate::witness::since(&here, "m2");
        assert!(heard.iter().any(|x| matches!(&x.what,
            Happening::Said { words, .. } if words == "from the other machine")));
    }

    /// A body restored from another copy settles silently, its walk ended
    /// when that copy has it standing.
    #[test]
    fn a_restored_body_moves_nobody_audibly() {
        let mut w = one_speaker();
        w.enter("m2", "Maker-02", at("band-one")).unwrap();
        w.set_off("m2", at("green-room")).unwrap();
        let logged = w.log().len();
        let standing = |place: &str| BodyState {
            at: at(place),
            hold: None,
            walk: None,
        };
        w.restore_body("m2", &standing("green-room")).unwrap();
        assert_eq!(w.log().len(), logged, "nothing logged");
        assert_eq!(w.actor("m2").unwrap().at, at("green-room"));
        assert!(w.actor("m2").unwrap().walk.is_none());
        assert!(w.restore_body("m2", &standing("no-such-room")).is_err());
    }

    /// **A copy given another's state is that state.** Everything a copy of a
    /// world shares — a body standing, one holding a station, one partway
    /// along a walk, the lift on its way with one body aboard and another
    /// waiting for it — read off one world and put into another built from
    /// the same map, reads back the same.
    #[test]
    fn a_world_s_state_put_into_another_copy_reads_back_the_same() {
        let mut source = vault();
        let shaft: Vec<Where> = source.shaft().to_vec();
        let top = shaft.len() - 1;
        for (id, place) in [
            ("stands", at("green-room")),
            ("holds", at("band-one")),
            ("walks", at("green-room")),
            ("aboard", shaft[0].clone()),
            ("waits", shaft[top].clone()),
        ] {
            source.enter(id, id, place).unwrap();
        }
        source.take("holds", Some("cindy")).unwrap();
        source.call_lift(0);
        while !source.lift().unwrap().boardable_at(0) {
            source.tick();
        }
        assert!(source.ride_lift("aboard", top));
        assert!(source.ride_lift("waits", 0));
        source.set_off("walks", shaft[top].clone()).unwrap();
        let state = source.state();
        assert!(state.bodies["walks"].walk.is_some(), "on its way");
        let car = state.lift.as_ref().unwrap();
        assert_eq!(car.riders.get("aboard"), Some(&top));
        assert_eq!(car.waiting.get("waits"), Some(&0));
        assert_eq!(car.stops, vec![top]);
        assert_eq!(
            state.bodies["holds"].hold.as_ref().map(|h| h.1.as_deref()),
            Some(Some("cindy"))
        );

        let mut copy = vault();
        for id in state.bodies.keys() {
            copy.enter(id, id, at("green-room")).unwrap();
        }
        for (id, body) in &state.bodies {
            copy.restore_body(id, body).unwrap();
        }
        assert!(copy.restore_lift(car));
        assert_eq!(copy.state(), state);
    }

    /// A decision begun in one copy of a world, carried to another, lets a
    /// line be called after somebody who left the room there.
    #[test]
    fn a_decision_carried_over_calls_after_the_same_company() {
        let mut replica = one_speaker();
        replica.enter("m2", "Maker-02", at("green-room")).unwrap();
        replica.begin_decision("m1");
        let (from, company) = replica.deliberation("m1").unwrap();
        assert_eq!(
            (from.clone(), company.clone()),
            (at("green-room"), vec!["m2".to_string()])
        );

        let mut authority = one_speaker();
        authority.enter("m2", "Maker-02", at("band-one")).unwrap();
        authority.restore_decision("m1", from, company);
        assert!(authority
            .call_after("m1", "m2", "wait for me", Voice::Said)
            .is_ok());
        authority.end_decision("m1");
        assert!(authority.deliberation("m1").is_none());
        authority.restore_decision("nobody", at("green-room"), vec![]);
        assert!(authority.deliberation("nobody").is_none(), "no such body");
    }

    // ── the event window ────────────────────────────────────────────────────

    /// Fill the log by talking. Speech is the cheapest event to make a lot of
    /// and the one a busy room actually produces most of.
    pub(in super::super) fn chatter(w: &mut World, n: usize) {
        for i in 0..n {
            w.say("m1", format!("line {i}")).unwrap();
        }
    }

    pub(in super::super) fn one_speaker() -> World {
        let mut w = vault();
        w.enter("m1", "Maker-01", at("green-room")).unwrap();
        w
    }

    /// **The log is a window, not an archive.** Nothing was taking events away,
    /// so it grew for the life of the process — and `witness::since` walks it
    /// from the beginning once per body per moment, so it was a per-tick cost
    /// that climbed for ever as well as unbounded memory.
    #[test]
    fn the_event_log_stops_growing() {
        let mut w = one_speaker();
        for _ in 0..12 {
            chatter(&mut w, KEEP_EVENTS);
            w.tick();
            assert!(
                w.log().len() <= KEEP_EVENTS + EVENT_SLACK,
                "log reached {} against a bound of {}",
                w.log().len(),
                KEEP_EVENTS
            );
        }
    }

    /// And it is the **oldest** that go. A window that dropped the newest would
    /// be worse than no window: the whole point is what just happened.
    #[test]
    fn the_window_keeps_the_newest_events() {
        let mut w = one_speaker();
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 200);
        w.tick();

        let said: Vec<&str> = w
            .log()
            .iter()
            .filter_map(|e| match &e.what {
                Happening::Said { words, .. } => Some(words.as_str()),
                _ => None,
            })
            .collect();
        let last = format!("line {}", KEEP_EVENTS + EVENT_SLACK + 199);
        assert_eq!(said.last().copied(), Some(last.as_str()));
        assert!(!said.contains(&"line 0"), "the oldest line survived a trim");
    }

    /// **Trimming runs on a still world.** Most moments move nobody, so `tick`
    /// returns early — and a trim placed after that exit would only run while
    /// somebody happened to be walking, which is not the condition that fills
    /// the log.
    #[test]
    fn a_world_where_nobody_is_walking_still_trims() {
        let mut w = one_speaker();
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 50);
        assert_eq!(w.tick(), 0, "somebody was walking");
        assert_eq!(w.log().len(), KEEP_EVENTS);
    }

    /// Cut back to the bound rather than to the moment it crossed it, so the
    /// memmove is paid once per `EVENT_SLACK` events instead of once per event.
    #[test]
    fn trimming_is_amortised_over_the_slack() {
        let mut w = one_speaker();
        // Filled by measurement rather than by arithmetic: entering the world
        // is itself an event, so counting only the lines said is off by one and
        // the test would be asserting against the wrong side of the threshold.
        while w.log().len() < KEEP_EVENTS + EVENT_SLACK {
            w.say("m1", "filling").unwrap();
        }
        w.tick();
        assert_eq!(
            w.log().len(),
            KEEP_EVENTS + EVENT_SLACK,
            "cut at the threshold instead of past it — that is a memmove per event"
        );

        w.say("m1", "the one over").unwrap();
        w.tick();
        assert_eq!(w.log().len(), KEEP_EVENTS, "did not cut back to the bound");
    }

    /// **A reader still gets its news across a trim.** The cursor is a tick
    /// rather than an index into the log, so dropping the front cannot shift
    /// what a body has already read out from under it — the failure an
    /// index-based cursor would have had, silently, as a reader served somebody
    /// else's events.
    #[test]
    fn a_reader_is_not_disturbed_by_a_trim() {
        let mut w = one_speaker();
        w.enter("m2", "Maker-02", at("green-room")).unwrap();
        w.mark_seen("m2");
        chatter(&mut w, KEEP_EVENTS + EVENT_SLACK + 10);
        w.tick();

        w.say("m1", "the one that matters").unwrap();
        let heard = crate::witness::since(&w, "m2");
        assert!(
            heard.iter().any(|x| matches!(&x.what,
                Happening::Said { words, .. } if words == "the one that matters")),
            "a trim lost the reader its news"
        );
    }
}

//! Where each of a member's own events stands (§6.3): **unsent** (written to
//! the log, on no socket yet), **in flight** on a socket (sent, no `RECEIPT`
//! covers it), or **received** (a `RECEIPT` covered it). Only `oseq`s are kept:
//! the log is the source of truth, and an event is read back from it when it
//! is sent.
//!
//! Received events are forgotten, so the outbox holds only what is still owed.
//!
//! **A socket is one session.** Frame numbers are a session's own, counted
//! from its start, so a socket id names one session and is never reused: a
//! cluster that reconnects opens a new id. A write that completes after its
//! session ended — the writer's future resolving once the socket has dropped
//! and its events were made owed again — names a socket that is no longer
//! open, and its event stays owed. Recorded in flight under the dead session's
//! frame number, no later receipt on the cluster's new session would cover it:
//! on BlackWorld (2026-11-18) 964 events stood in flight that way for five
//! hours, the oldest never moving.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One of the member's sessions, by the caller's own numbering: a new id for
/// every session, never reused.
pub type SocketId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    Unsent,
    /// Handed to `socket`'s writer, its frame number not known yet.
    Queued {
        socket: SocketId,
    },
    /// Sent on `socket`; its last fragment was that session's frame `frame`.
    InFlight {
        socket: SocketId,
        frame: u64,
    },
    Received,
}

#[derive(Debug, Default)]
pub struct Outbox {
    unsent: BTreeSet<u64>,
    queued: BTreeMap<u64, SocketId>,
    in_flight: BTreeMap<u64, (SocketId, u64)>,
    /// Per socket, its in-flight events by the frame that completed them.
    by_frame: HashMap<SocketId, BTreeMap<u64, Vec<u64>>>,
    /// The sockets open now: an event is in flight only on one of them.
    open: BTreeSet<SocketId>,
    /// The highest `oseq` the outbox has been told of.
    written: u64,
}

impl Outbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// The member wrote its event `oseq`: it is owed.
    pub fn written(&mut self, oseq: u64) {
        self.written = self.written.max(oseq);
        if !self.in_flight.contains_key(&oseq) && !self.queued.contains_key(&oseq) {
            self.unsent.insert(oseq);
        }
    }

    /// After a process start (§7.5): every own event through `written` — all
    /// the previous process wrote — is owed to no socket. The resync's `HAVE`
    /// offers them, and each member lacking any pulls exactly those by
    /// catch-up; pushing them again would send a member alone in its group its
    /// whole history, ahead of every request on the ordered lane. Events
    /// written since the start, above `written`, stay as they are.
    pub fn restart(&mut self, written: u64) {
        self.unsent.retain(|o| *o > written);
        self.queued.retain(|o, _| *o > written);
        self.in_flight.retain(|o, _| *o > written);
        for frames in self.by_frame.values_mut() {
            for oseqs in frames.values_mut() {
                oseqs.retain(|o| *o > written);
            }
            frames.retain(|_, oseqs| !oseqs.is_empty());
        }
        self.written = self.written.max(written);
    }

    /// The unsent events, in `oseq` order: the order they are sent again in.
    pub fn unsent(&self) -> impl Iterator<Item = u64> + '_ {
        self.unsent.iter().copied()
    }

    /// Session `socket` began: events may be queued and in flight on it.
    pub fn opened(&mut self, socket: SocketId) {
        self.open.insert(socket);
    }

    /// `oseq` was handed to `socket`'s writer.
    pub fn queue(&mut self, oseq: u64, socket: SocketId) {
        if self.unsent.remove(&oseq) {
            self.queued.insert(oseq, socket);
        }
    }

    /// `oseq`, queued on `socket`, never went out: the session ended first.
    /// One queued since on another socket is that socket's, and stays.
    pub fn unqueue(&mut self, oseq: u64, socket: SocketId) {
        if self.queued.get(&oseq) == Some(&socket) {
            self.queued.remove(&oseq);
            self.unsent.insert(oseq);
        }
    }

    /// `oseq` is no longer held: maintenance dropped it as superseded, and
    /// what superseded it is owed instead.
    pub fn forget(&mut self, oseq: u64) {
        self.unsent.remove(&oseq);
        self.queued.remove(&oseq);
    }

    /// `oseq` went out on `socket`, its last fragment as frame `frame`. A
    /// socket no longer open is a session that ended: no receipt will come for
    /// its frame, so the event stays owed (module docs).
    pub fn sent(&mut self, oseq: u64, socket: SocketId, frame: u64) {
        if !self.open.contains(&socket) {
            self.unqueue(oseq, socket);
            return;
        }
        self.unsent.remove(&oseq);
        self.queued.remove(&oseq);
        if let Some((s, f)) = self.in_flight.insert(oseq, (socket, frame)) {
            self.forget_frame(s, f, oseq);
        }
        self.by_frame
            .entry(socket)
            .or_default()
            .entry(frame)
            .or_default()
            .push(oseq);
    }

    /// `socket`'s gateway took in every frame through `through`.
    pub fn receipt(&mut self, socket: SocketId, through: u64) {
        let Some(frames) = self.by_frame.get_mut(&socket) else {
            return;
        };
        let rest = frames.split_off(&(through + 1));
        for (_, oseqs) in std::mem::replace(frames, rest) {
            for oseq in oseqs {
                self.in_flight.remove(&oseq);
            }
        }
    }

    /// `socket` dropped: everything queued or in flight on it is owed again,
    /// and a write it completes later leaves its event owed.
    pub fn dropped(&mut self, socket: SocketId) {
        self.open.remove(&socket);
        for (_, oseqs) in self.by_frame.remove(&socket).unwrap_or_default() {
            for oseq in oseqs {
                self.in_flight.remove(&oseq);
                self.unsent.insert(oseq);
            }
        }
        let queued: Vec<u64> = self
            .queued
            .iter()
            .filter(|(_, s)| **s == socket)
            .map(|(o, _)| *o)
            .collect();
        for oseq in queued {
            self.queued.remove(&oseq);
            self.unsent.insert(oseq);
        }
    }

    /// Whether any event at or below `through` is still owed anywhere but on
    /// `socket`: unsent, or queued or in flight on another socket.
    pub fn owed_elsewhere(&self, through: u64, socket: SocketId) -> bool {
        self.unsent.range(..=through).next().is_some()
            || self.queued.range(..=through).any(|(_, s)| *s != socket)
            || self
                .in_flight
                .range(..=through)
                .any(|(_, (s, _))| *s != socket)
    }

    pub fn state(&self, oseq: u64) -> EventState {
        if self.unsent.contains(&oseq) {
            EventState::Unsent
        } else if let Some(socket) = self.queued.get(&oseq) {
            EventState::Queued { socket: *socket }
        } else if let Some((socket, frame)) = self.in_flight.get(&oseq) {
            EventState::InFlight {
                socket: *socket,
                frame: *frame,
            }
        } else if oseq <= self.written {
            EventState::Received
        } else {
            EventState::Unsent
        }
    }

    /// `(unsent, in flight)`, for the status page; a queued event is in
    /// flight.
    pub fn counts(&self) -> (usize, usize) {
        (self.unsent.len(), self.queued.len() + self.in_flight.len())
    }

    /// The oldest event queued or in flight, with the socket it is on: what a
    /// stall of the receipts names.
    pub fn oldest_in_flight(&self) -> Option<(u64, SocketId)> {
        let in_flight = self.in_flight.iter().next().map(|(o, (s, _))| (*o, *s));
        let queued = self.queued.iter().next().map(|(o, s)| (*o, *s));
        match (in_flight, queued) {
            (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    fn forget_frame(&mut self, socket: SocketId, frame: u64, oseq: u64) {
        if let Some(frames) = self.by_frame.get_mut(&socket) {
            if let Some(list) = frames.get_mut(&frame) {
                list.retain(|o| *o != oseq);
                if list.is_empty() {
                    frames.remove(&frame);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the events past a socket's last `RECEIPT` go back to unsent when it
    /// drops, and they leave again in `oseq` order; nothing a receipt covered
    /// is owed again.
    #[test]
    fn a_dropped_socket_returns_only_its_unreceipted_events() {
        let mut o = Outbox::new();
        o.opened(0);
        o.opened(1);
        for oseq in 1..=6 {
            o.written(oseq);
        }
        o.sent(1, 0, 10);
        o.sent(2, 0, 12);
        o.sent(3, 1, 4);
        o.sent(4, 0, 15);
        o.receipt(0, 12);
        assert_eq!(o.state(1), EventState::Received);
        assert_eq!(o.state(2), EventState::Received);
        assert_eq!(
            o.state(4),
            EventState::InFlight {
                socket: 0,
                frame: 15
            }
        );
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![5, 6]);
        o.dropped(0);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![4, 5, 6]);
        assert_eq!(
            o.state(3),
            EventState::InFlight {
                socket: 1,
                frame: 4
            }
        );
        assert_eq!(o.counts(), (3, 1));
    }

    /// With every socket down, events stay unsent — nothing is held but the
    /// numbers.
    #[test]
    fn events_with_no_socket_stay_unsent() {
        let mut o = Outbox::new();
        o.written(1);
        o.written(2);
        assert_eq!(o.state(2), EventState::Unsent);
        assert_eq!(o.counts(), (2, 0));
    }

    /// **A restart pushes none of the previous process's events.** The
    /// resync's `HAVE` offers them and a member lacking any pulls it; owing
    /// them here re-sent a lone member's whole history — 2.5M records on `.5` —
    /// ahead of every request.
    #[test]
    fn a_restart_owes_nothing_the_last_process_wrote() {
        let mut o = Outbox::new();
        o.restart(10);
        assert_eq!(o.unsent().count(), 0);
        assert_eq!(o.state(1), EventState::Received);
        assert_eq!(o.state(10), EventState::Received);
        assert_eq!(o.state(11), EventState::Unsent, "not written yet");
    }

    /// A queued event is no longer unsent, goes back if its socket drops or its
    /// session ends before it is written, and counts as owed elsewhere only
    /// for another socket.
    #[test]
    fn a_queued_event_is_neither_unsent_nor_lost() {
        let mut o = Outbox::new();
        o.opened(0);
        o.opened(1);
        for oseq in 1..=3 {
            o.written(oseq);
        }
        o.queue(1, 0);
        o.queue(2, 1);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![3]);
        assert_eq!(o.state(1), EventState::Queued { socket: 0 });
        o.queue(1, 1);
        assert_eq!(o.state(1), EventState::Queued { socket: 0 }, "queued once");
        o.sent(3, 0, 9);
        assert!(o.owed_elsewhere(3, 0), "2 is queued on socket 1");
        assert!(!o.owed_elsewhere(1, 0));
        o.dropped(1);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![2]);
        o.unqueue(1, 1);
        assert_eq!(o.state(1), EventState::Queued { socket: 0 }, "not 1's");
        o.unqueue(1, 0);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(o.counts(), (2, 1));
    }

    /// A restart leaves alone what was written since the start, and forgets
    /// anything at or below `written` it was holding.
    #[test]
    fn a_restart_leaves_events_written_since_alone() {
        let mut o = Outbox::new();
        o.opened(0);
        o.written(9);
        o.written(12);
        o.sent(12, 0, 1);
        o.restart(10);
        assert_eq!(o.unsent().count(), 0, "9 belongs to the last process");
        assert_eq!(
            o.state(12),
            EventState::InFlight {
                socket: 0,
                frame: 1
            }
        );
    }

    /// The oldest event owed to a socket is named whether it is queued or in
    /// flight; nothing is named once every one is received.
    #[test]
    fn the_oldest_event_in_flight_is_named() {
        let mut o = Outbox::new();
        o.opened(0);
        o.opened(1);
        assert_eq!(o.oldest_in_flight(), None);
        for oseq in 1..=4 {
            o.written(oseq);
        }
        o.sent(3, 1, 7);
        o.queue(2, 0);
        assert_eq!(o.oldest_in_flight(), Some((2, 0)));
        o.sent(2, 0, 8);
        o.receipt(0, 8);
        assert_eq!(o.oldest_in_flight(), Some((3, 1)));
        o.receipt(1, 7);
        assert_eq!(
            o.oldest_in_flight(),
            None,
            "1 and 4 are unsent, not in flight"
        );
    }

    /// An event sent again on another socket is tracked there alone.
    #[test]
    fn a_resend_moves_the_event_to_its_new_socket() {
        let mut o = Outbox::new();
        o.opened(0);
        o.opened(1);
        o.written(1);
        o.sent(1, 0, 3);
        o.dropped(0);
        o.queue(1, 1);
        o.sent(1, 1, 2);
        o.receipt(0, 100);
        assert_eq!(
            o.state(1),
            EventState::InFlight {
                socket: 1,
                frame: 2
            }
        );
        o.receipt(1, 2);
        assert_eq!(o.state(1), EventState::Received);
    }

    /// **A write that completes after its session dropped leaves its event
    /// owed**, not in flight under the dead session's frame number. The
    /// cluster's new session counts its frames from the start, so its receipts
    /// never reach that number: 964 events stood in flight for five hours on
    /// BlackWorld (2026-11-18). The event goes out again on the new session and
    /// is received there.
    #[test]
    fn a_write_completing_on_a_dropped_session_stays_owed() {
        let mut o = Outbox::new();
        o.opened(0);
        o.written(1);
        o.written(2);
        o.queue(1, 0);
        o.queue(2, 0);
        o.dropped(0);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![1, 2]);
        // The cluster reconnects as a new session.
        o.opened(1);
        o.queue(2, 1);
        // Session 0's writer reports both written, late, at a frame far past
        // anything session 1 has sent.
        o.sent(1, 0, 5_000_000);
        o.sent(2, 0, 5_000_001);
        assert_eq!(o.state(1), EventState::Unsent, "owed again, not in flight");
        assert_eq!(
            o.state(2),
            EventState::Queued { socket: 1 },
            "1's, untouched"
        );
        assert_eq!(o.counts(), (1, 1));
        o.queue(1, 1);
        o.sent(1, 1, 3);
        o.sent(2, 1, 4);
        o.receipt(1, 4);
        assert_eq!(o.state(1), EventState::Received);
        assert_eq!(o.state(2), EventState::Received);
        assert_eq!(o.counts(), (0, 0));
        assert_eq!(o.oldest_in_flight(), None);
    }
}

//! Where each of a member's own events stands (§6.3): **unsent** (written to
//! the log, on no socket yet), **in flight** on a socket (sent, no `RECEIPT`
//! covers it), or **received** (a `RECEIPT` covered it). Only `oseq`s are kept:
//! the log is the source of truth, and an event is read back from it when it
//! is sent.
//!
//! Received events are forgotten, so the outbox holds only what is still owed.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One of the member's sockets, by the caller's own numbering.
pub type SocketId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventState {
    Unsent,
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
    in_flight: BTreeMap<u64, (SocketId, u64)>,
    /// Per socket, its in-flight events by the frame that completed them.
    by_frame: HashMap<SocketId, BTreeMap<u64, Vec<u64>>>,
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
        if !self.in_flight.contains_key(&oseq) {
            self.unsent.insert(oseq);
        }
    }

    /// After a process start (§7.5): every own event at or below `received` is
    /// held by some member, and every one above it through `written` is owed.
    pub fn restart(&mut self, received: u64, written: u64) {
        self.unsent.clear();
        self.in_flight.clear();
        self.by_frame.clear();
        self.written = written;
        self.unsent.extend(received + 1..=written);
    }

    /// The unsent events, in `oseq` order: the order they are sent again in.
    pub fn unsent(&self) -> impl Iterator<Item = u64> + '_ {
        self.unsent.iter().copied()
    }

    /// `oseq` went out on `socket`, its last fragment as frame `frame`.
    pub fn sent(&mut self, oseq: u64, socket: SocketId, frame: u64) {
        self.unsent.remove(&oseq);
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

    /// `socket` dropped: everything in flight on it is owed again.
    pub fn dropped(&mut self, socket: SocketId) {
        for (_, oseqs) in self.by_frame.remove(&socket).unwrap_or_default() {
            for oseq in oseqs {
                self.in_flight.remove(&oseq);
                self.unsent.insert(oseq);
            }
        }
    }

    pub fn state(&self, oseq: u64) -> EventState {
        if self.unsent.contains(&oseq) {
            EventState::Unsent
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

    /// `(unsent, in flight)`, for the status page.
    pub fn counts(&self) -> (usize, usize) {
        (self.unsent.len(), self.in_flight.len())
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

    /// A restart owes exactly the events above the highest any member
    /// acknowledged.
    #[test]
    fn a_restart_owes_what_no_member_acknowledged() {
        let mut o = Outbox::new();
        o.restart(7, 10);
        assert_eq!(o.unsent().collect::<Vec<_>>(), vec![8, 9, 10]);
        assert_eq!(o.state(7), EventState::Received);
        assert_eq!(o.state(9), EventState::Unsent);
    }

    /// An event sent again on another socket is tracked there alone.
    #[test]
    fn a_resend_moves_the_event_to_its_new_socket() {
        let mut o = Outbox::new();
        o.written(1);
        o.sent(1, 0, 3);
        o.dropped(0);
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
}

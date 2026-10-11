//! The writer's three lanes (§4.2), as a pure scheduler.
//!
//! - **control** goes first, always;
//! - **ordered** is one FIFO of stream frames and live events, in the order
//!   produced;
//! - **catch-up** goes only when the ordered lane is empty and the socket's
//!   unsent bytes are under the low-water mark.
//!
//! Each lane keeps the order things were queued in: nothing overtakes a frame
//! or an event queued before it. Events are sent one at a time, so a member's
//! events leave in the order it produced them.
//!
//! On the ordered lane a large event yields **one fragment per turn**: after
//! each fragment its turn goes back behind the frames queued since, but ahead
//! of any later event. Those frames go out before its next fragment, so no
//! frame waits behind more than one 1 MiB fragment of it. The catch-up lane
//! sends an event's fragments back to back: a `DONE` queued after a fetched
//! record never reaches the requester before the record's last fragment.

use std::collections::VecDeque;

use tokio::sync::oneshot;

use super::gauge::Lane;
use crate::frame::{Event, Frame};

/// The socket's unsent bytes under which catch-up frames may go (§7.4). A
/// live event queued behind catch-up waits for at most this much to drain —
/// with small records, tens of thousands of frames — so it is kept to what
/// keeps a link busy, not more.
pub const CATCHUP_LOW_WATER: usize = 1 << 20;

/// An event waiting to go, as its fragments, and who to tell the session
/// frame number its last fragment went out as.
pub struct PendingEvent {
    pub fragments: VecDeque<Event>,
    pub catchup: bool,
    pub done: Option<oneshot::Sender<u64>>,
}

impl PendingEvent {
    /// A fetched record — flagged catch-up or addressed to a requester — goes
    /// on the catch-up lane; a live event on the ordered lane.
    pub fn lane(&self) -> Lane {
        if self.catchup || self.fragments.front().is_some_and(|f| f.reply_to.is_some()) {
            Lane::CatchUp
        } else {
            Lane::Live
        }
    }
}

enum Slot {
    Frame(Frame),
    /// The turn of the event at the head of the lane's event queue: event
    /// slots are in the same order as the events.
    Event,
}

struct Fifo {
    slots: VecDeque<Slot>,
    events: VecDeque<PendingEvent>,
    /// Whether a large event yields its turn after each fragment.
    interleave: bool,
}

impl Fifo {
    fn new(interleave: bool) -> Self {
        Self {
            slots: VecDeque::new(),
            events: VecDeque::new(),
            interleave,
        }
    }

    fn push_frame(&mut self, frame: Frame) {
        self.slots.push_back(Slot::Frame(frame));
    }

    fn push_event(&mut self, event: PendingEvent) {
        self.slots.push_back(Slot::Event);
        self.events.push_back(event);
    }

    /// The next frame, gauged as `frames` when it is a frame and as `events`
    /// when it is an event's fragment.
    fn pop(&mut self, frames: Lane, events: Lane) -> Option<Next> {
        match self.slots.pop_front()? {
            Slot::Frame(frame) => Some(Next {
                frame,
                done: None,
                lane: frames,
            }),
            Slot::Event => {
                let head = self.events.front_mut().expect("an event slot has an event");
                let fragment = head.fragments.pop_front().expect("events are never empty");
                let catchup = head.catchup;
                let done = if head.fragments.is_empty() {
                    self.events.pop_front().expect("head").done
                } else {
                    // Its next turn: behind the frames queued since, ahead of
                    // the next event — or at once, on a lane that does not
                    // interleave.
                    let at = if self.interleave {
                        self.slots
                            .iter()
                            .position(|s| matches!(s, Slot::Event))
                            .unwrap_or(self.slots.len())
                    } else {
                        0
                    };
                    self.slots.insert(at, Slot::Event);
                    None
                };
                Some(Next {
                    frame: Frame::Event {
                        catchup,
                        event: fragment,
                    },
                    done,
                    lane: events,
                })
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// The next frame to write, the completion to fire once it is written, and
/// the lane whose gauge it comes off.
pub struct Next {
    pub frame: Frame,
    pub done: Option<oneshot::Sender<u64>>,
    pub lane: Lane,
}

pub struct Lanes {
    control: VecDeque<Frame>,
    ordered: Fifo,
    catchup: Fifo,
}

impl Default for Lanes {
    fn default() -> Self {
        Self::new()
    }
}

impl Lanes {
    pub fn new() -> Self {
        Self {
            control: VecDeque::new(),
            ordered: Fifo::new(true),
            catchup: Fifo::new(false),
        }
    }

    pub fn control(&mut self, frame: Frame) {
        self.control.push_back(frame);
    }

    pub fn ordered(&mut self, frame: Frame) {
        self.ordered.push_frame(frame);
    }

    pub fn catchup(&mut self, frame: Frame) {
        self.catchup.push_frame(frame);
    }

    pub fn event(&mut self, event: PendingEvent) {
        if event.lane() == Lane::CatchUp {
            self.catchup.push_event(event);
        } else {
            self.ordered.push_event(event);
        }
    }

    /// The next frame to write, given the socket's unsent bytes.
    pub fn next(&mut self, unsent_bytes: usize) -> Option<Next> {
        if let Some(frame) = self.control.pop_front() {
            return Some(Next {
                frame,
                done: None,
                lane: Lane::Control,
            });
        }
        if let Some(next) = self.ordered.pop(Lane::Stream, Lane::Live) {
            return Some(next);
        }
        if unsent_bytes < CATCHUP_LOW_WATER {
            return self.catchup.pop(Lane::CatchUp, Lane::CatchUp);
        }
        None
    }

    /// Catch-up frames wait only on the socket draining.
    pub fn catchup_waiting(&self) -> bool {
        self.control.is_empty() && self.ordered.is_empty() && !self.catchup.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.control.is_empty() && self.ordered.is_empty() && self.catchup.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::split;
    use crate::frame::{name, Data, Done, Kind, Receipt};

    fn data(stream: u32) -> Frame {
        Frame::Data(Data {
            stream,
            bytes: vec![0; 10],
        })
    }

    fn event(oseq: u64, len: usize, catchup: bool) -> (PendingEvent, oneshot::Receiver<u64>) {
        let (tx, rx) = oneshot::channel();
        (
            PendingEvent {
                fragments: split(1, oseq, &vec![1; len], None).into(),
                catchup,
                done: Some(tx),
            },
            rx,
        )
    }

    fn drain(l: &mut Lanes, unsent: usize) -> Vec<Frame> {
        let mut out = Vec::new();
        while let Some(n) = l.next(unsent) {
            if let Some(done) = n.done {
                // A test that does not watch an event's completion drops its
                // receiver.
                let _ = done.send(out.len() as u64 + 1);
            }
            out.push(n.frame);
        }
        out
    }

    fn label(f: &Frame) -> String {
        match f {
            Frame::Data(d) => format!("D{}", d.stream),
            Frame::Event { event, .. } => format!("E{}.{}", event.oseq, event.offset >> 20),
            other => name(other.kind()).to_string(),
        }
    }

    /// Control first; then the ordered FIFO with a large event taking one turn
    /// at a time; catch-up last.
    #[test]
    fn control_first_and_a_large_event_takes_one_turn_at_a_time() {
        let mut l = Lanes::new();
        let (big, mut big_done) = event(1, 3 << 20, false);
        l.ordered(data(1));
        l.event(big);
        l.ordered(data(2));
        l.ordered(data(3));
        l.control(Frame::Receipt(Receipt { through: 1 }));
        l.catchup(Frame::Done(Done {
            request: 1,
            reply_to: 1,
        }));
        let order: Vec<String> = drain(&mut l, 0).iter().map(label).collect();
        assert_eq!(
            order,
            ["RECEIPT", "D1", "E1.0", "D2", "D3", "E1.1", "E1.2", "DONE"]
        );
        assert_eq!(
            big_done.try_recv().unwrap(),
            7,
            "completes on its last fragment"
        );
    }

    /// Each frame names the gauge it comes off: control and stream frames
    /// none, a live event's fragments the live gauge, everything on the
    /// catch-up lane the catch-up gauge.
    #[test]
    fn each_frame_names_its_lane() {
        let mut l = Lanes::new();
        let (live, _) = event(1, 10, false);
        let (fetched, _) = event(2, 10, true);
        l.control(Frame::Receipt(Receipt { through: 1 }));
        l.ordered(data(1));
        l.event(live);
        l.event(fetched);
        l.catchup(Frame::Done(Done {
            request: 1,
            reply_to: 1,
        }));
        let mut lanes = Vec::new();
        while let Some(n) = l.next(0) {
            lanes.push(n.lane);
        }
        assert_eq!(
            lanes,
            [
                Lane::Control,
                Lane::Stream,
                Lane::Live,
                Lane::CatchUp,
                Lane::CatchUp
            ]
        );
    }

    /// A member's events leave in the order produced, never interleaved with
    /// each other; a frame queued between two events goes between a large
    /// one's fragments, and one queued after the second waits for it.
    #[test]
    fn events_go_one_at_a_time_in_order() {
        let mut l = Lanes::new();
        let (a, _) = event(1, 2 << 20, false);
        let (b, _) = event(2, 10, false);
        l.event(a);
        l.ordered(data(3));
        l.event(b);
        l.ordered(data(4));
        let order: Vec<String> = drain(&mut l, 0).iter().map(label).collect();
        assert_eq!(order, ["E1.0", "D3", "E1.1", "E2.0", "D4"]);
    }

    /// On the catch-up lane a fetched record's fragments go back to back, and
    /// the `DONE` queued after the records goes after the last of them.
    #[test]
    fn a_done_never_overtakes_the_records_before_it() {
        let mut l = Lanes::new();
        let (a, _) = event(1, 2 << 20, true);
        let (b, _) = event(2, 10, true);
        l.event(a);
        l.event(b);
        l.catchup(Frame::Done(Done {
            request: 1,
            reply_to: 1,
        }));
        let order: Vec<String> = drain(&mut l, 0).iter().map(label).collect();
        assert_eq!(order, ["E1.0", "E1.1", "E2.0", "DONE"]);
    }

    /// Catch-up waits while the socket holds 1 MiB unsent, and while the
    /// ordered lane has anything.
    #[test]
    fn catchup_waits_for_an_empty_ordered_lane_and_a_drained_socket() {
        let mut l = Lanes::new();
        let (fetched, _) = event(9, 10, true);
        l.event(fetched);
        assert!(l.next(CATCHUP_LOW_WATER).is_none());
        assert!(l.catchup_waiting());
        l.ordered(data(1));
        assert_eq!(label(&l.next(0).unwrap().frame), "D1");
        let f = l.next(CATCHUP_LOW_WATER - 1).unwrap().frame;
        assert!(matches!(f, Frame::Event { catchup: true, .. }));
        assert_eq!(f.kind(), Kind::Event);
        assert!(l.is_empty());
    }
}

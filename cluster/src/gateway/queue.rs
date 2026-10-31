//! Rule 3's queue (`docs/cooperative_admission.md` §4.3): a unit that fits on
//! no eligible card waits here, most urgent first and then oldest first, until
//! a card has room — released whenever a card's count falls or a `CAPACITY`
//! arrives — or until [`HOLD`] has passed.

use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use super::load::Picked;
use super::registry::LATENCY_APP;
use crate::dispatch::Dispatch;

/// How long a unit waits for room before it is answered `503`: the catch-up
/// hold of `docs/replicated_substrate.md` §5.4.
pub const HOLD: Duration = Duration::from_secs(30);

/// How urgent a waiting unit is, most urgent first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Urgency {
    /// A person is waiting on it as it runs: every zend request — a live chat,
    /// a passthrough or MCP conversation — and a member's ask for a peer (a
    /// draw's content check).
    Latency,
    /// Every other request.
    Deadline,
}

impl Urgency {
    pub fn of(app: &str, sender: Option<u64>) -> Urgency {
        if app == LATENCY_APP || sender.is_some() {
            Urgency::Latency
        } else {
            Urgency::Deadline
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Urgency::Latency => "latency",
            Urgency::Deadline => "deadline",
        }
    }
}

/// What a unit asks to be placed by: the same as [`super::Hub::pick_from`]'s
/// arguments, kept so the unit can be placed again as it waits.
#[derive(Debug, Clone)]
pub struct Placing {
    /// The gateway's id for the request, which every placement line names
    /// (`docs/cooperative_admission.md` §4.6).
    pub req: u64,
    pub app: String,
    pub dispatch: Dispatch,
    pub draw: Option<u64>,
    pub key: Option<String>,
    pub pinned: Option<u64>,
    /// The member that sent it up asking for a peer.
    pub sender: Option<u64>,
    /// The route it matched, which an `Any` request goes only to a member
    /// declaring ([`super::registry::Registry::declaring`]).
    pub route: Option<RouteAt>,
}

/// A request's method and path, as a member's table is matched with them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteAt {
    pub method: String,
    pub path: String,
}

/// One waiting unit.
pub(super) struct Waiter {
    pub seq: u64,
    pub urgency: Urgency,
    pub since: Instant,
    pub ask: Placing,
    pub tx: oneshot::Sender<Picked>,
}

/// The units waiting at this gateway.
#[derive(Default)]
pub(super) struct Queue {
    waiters: Vec<Waiter>,
    next: u64,
}

impl Queue {
    /// Queue `ask`; its sequence number names it to [`Self::forget`].
    pub fn push(&mut self, ask: Placing, now: Instant, tx: oneshot::Sender<Picked>) -> u64 {
        self.next += 1;
        let seq = self.next;
        self.waiters.push(Waiter {
            seq,
            urgency: Urgency::of(&ask.app, ask.sender),
            since: now,
            ask,
            tx,
        });
        seq
    }

    /// Drop unit `seq`; whether it was still waiting.
    pub fn forget(&mut self, seq: u64) -> bool {
        let before = self.waiters.len();
        self.waiters.retain(|w| w.seq != seq);
        self.waiters.len() != before
    }

    /// Every waiting unit whose client still waits, most urgent first and then
    /// oldest first; those whose client went away are dropped. The caller puts
    /// back, with [`Self::keep`], each one it could not place.
    pub fn take_in_order(&mut self) -> Vec<Waiter> {
        let mut waiting: Vec<Waiter> = std::mem::take(&mut self.waiters)
            .into_iter()
            .filter(|w| !w.tx.is_closed())
            .collect();
        waiting.sort_by_key(|w| (w.urgency, w.seq));
        waiting
    }

    /// Put `waiter` back, still waiting.
    pub fn keep(&mut self, waiter: Waiter) {
        self.waiters.push(waiter);
    }

    pub fn len(&self) -> usize {
        self.waiters.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(app: &str, sender: Option<u64>) -> Placing {
        Placing {
            req: 0,
            app: app.into(),
            dispatch: Dispatch::Best,
            draw: None,
            key: None,
            pinned: None,
            sender,
            route: None,
        }
    }

    /// **Latency first, then arrival**: a zend request and a member's ask for a
    /// peer go ahead of an npcd request queued before them; within a class,
    /// the older first. A client that went away is dropped, and a unit
    /// forgotten is gone.
    #[test]
    fn latency_first_then_oldest_first() {
        let mut q = Queue::default();
        let now = Instant::now();
        let (tx1, _rx1) = oneshot::channel();
        let (tx2, _rx2) = oneshot::channel();
        let (tx3, _rx3) = oneshot::channel();
        let (tx4, rx4) = oneshot::channel();
        let (tx5, _rx5) = oneshot::channel();
        let npcd = q.push(ask("npcd", None), now, tx1);
        let chat = q.push(ask("zend", None), now, tx2);
        let check = q.push(ask("npcd", Some(9)), now, tx3);
        let gone = q.push(ask("zend", None), now, tx4);
        let late = q.push(ask("npcd", None), now, tx5);
        drop(rx4);
        assert!(q.forget(late));
        assert!(!q.forget(late), "already gone");
        let order: Vec<(u64, Urgency)> = q
            .take_in_order()
            .into_iter()
            .map(|w| (w.seq, w.urgency))
            .collect();
        assert_eq!(
            order,
            vec![
                (chat, Urgency::Latency),
                (check, Urgency::Latency),
                (npcd, Urgency::Deadline),
            ]
        );
        assert!(
            !order.iter().any(|(s, _)| *s == gone),
            "its client went away"
        );
        assert_eq!(q.len(), 0, "taken");
    }
}

//! What a gateway relays of the substrate (§4.3, §4.4, §7.2, §7.5, §9.1): live
//! events to the app's other members, catch-up traffic to the one member it
//! answers, the `RECEIPT`s that tell a sender what has been relayed, the
//! lagging state of a member that cannot keep up, and world pulses. The
//! gateway never decodes a record; it moves bytes between sockets.

use std::collections::BTreeMap;
use std::time::Instant;

use super::hub::{hex, Hub, CATCHUP_CAP, LAG_CAP};
use super::registry::Registry;
use crate::fragment::Whole;
use crate::frame::{Done, Frame, MemberState, Receipt};
use crate::session::SessionHandle;

impl Hub {
    /// A record from `from`: relayed live to the app's other members, or to
    /// the requester a fetched one is addressed to; then `from` is receipted
    /// when its pacing says so. The recipients are chosen under the hub's
    /// lock and the record queued for them outside it — a large record is
    /// copied once per recipient — and the receipt counts it only once every
    /// recipient's queue holds it (§4.3).
    pub(super) fn relay_event(&self, app: &str, from: u64, whole: Whole, frame: u64) {
        let bytes = whole.bytes.len();
        let mut lagged = Vec::new();
        let targets: Vec<SessionHandle> = {
            let mut guard = self.state();
            let s = &mut *guard;
            match whole.reply_to {
                Some(to) => match s.sockets.get_mut(&(app.to_string(), to)) {
                    Some(x) if over_cap(bytes, x.handle.gauges().catchup()) => {
                        x.dropped += 1;
                        Vec::new()
                    }
                    Some(x) => vec![x.handle.clone()],
                    None => Vec::new(),
                },
                None => {
                    let mut targets = Vec::new();
                    for ((a, n), x) in s.sockets.iter_mut() {
                        if a != app || *n == from {
                            continue;
                        }
                        let Some(m) = s.registry.member(app, *n) else {
                            continue;
                        };
                        if matches!(m.state(), MemberState::Upgrading | MemberState::Fault) {
                            continue;
                        }
                        if m.lagging {
                            x.skipped += 1;
                            continue;
                        }
                        if x.handle.gauges().live() + bytes > LAG_CAP {
                            x.skipped += 1;
                            lagged.push(*n);
                            continue;
                        }
                        targets.push(x.handle.clone());
                    }
                    for n in &lagged {
                        s.registry.set_lagging(app, *n, true);
                    }
                    targets
                }
            }
        };
        for handle in targets {
            let _ = handle.event(
                whole.origin,
                whole.oseq,
                &whole.bytes,
                whole.reply_to,
                whole.reply_to.is_some(),
            );
        }
        let receipt = self
            .state()
            .sockets
            .get_mut(&(app.to_string(), from))
            .and_then(|x| {
                x.pacer
                    .relayed(frame, bytes, Instant::now())
                    .map(|through| (x.handle.clone(), through))
            });
        if let Some((handle, through)) = receipt {
            let _ = handle.control(Frame::Receipt(Receipt { through }));
        }
        for n in lagged {
            self.note(
                app,
                hex(n),
                format!("lagging: over {} MiB of live events unsent", LAG_CAP >> 20),
            );
        }
    }

    /// A `PULSE` from `from` (§9.1): to every other member of the app on the
    /// ordered lane, except one whose lane holds more than half its lag cap —
    /// a transient is never queued behind pressure, it is dropped and
    /// counted, since the next state record settles the world anyway.
    pub(super) fn relay_pulse(&self, app: &str, from: u64, frame: Frame) {
        let targets: Vec<SessionHandle> = {
            let mut guard = self.state();
            let s = &mut *guard;
            let mut targets = Vec::new();
            for ((a, n), x) in s.sockets.iter_mut() {
                if a != app || *n == from {
                    continue;
                }
                if x.handle.gauges().live() > LAG_CAP / 2 {
                    x.pulses_dropped += 1;
                    continue;
                }
                targets.push(x.handle.clone());
            }
            targets
        };
        for handle in targets {
            let _ = handle.ordered(frame.clone());
        }
    }

    /// `node`'s socket's timer: a `RECEIPT` owed for 100 ms, and the end of
    /// its lagging once its lane has drained and it has caught up.
    pub(super) fn receipt_tick(&self, app: &str, node: u64) {
        let (receipt, recovered, synced) = {
            let mut guard = self.state();
            let s = &mut *guard;
            let Some(x) = s.sockets.get_mut(&(app.to_string(), node)) else {
                return;
            };
            let receipt = x
                .pacer
                .tick(Instant::now())
                .map(|through| (x.handle.clone(), through));
            let drained = x.handle.gauges().live() == 0;
            let acks = x.acks.clone();
            let lagging = s.registry.member(app, node).is_some_and(|m| m.lagging);
            let caught_up = s
                .registry
                .members(app)
                .iter()
                .filter(|m| m.node != node && m.high_water > 0)
                .all(|m| {
                    acks.iter()
                        .any(|(origin, through)| *origin == m.node && *through >= m.high_water)
                });
            let recovered = lagging && drained && caught_up;
            if recovered {
                s.registry.set_lagging(app, node, false);
            }
            // Synced: its own `ACK`s cover every other origin this gateway has
            // heard, members away included — judged here, not by the member,
            // whose catch-up can end early (a `DONE` with nobody to answer).
            let syncing = s.registry.member(app, node).is_some_and(|m| m.syncing);
            let synced = syncing && {
                // The most any member serving now has applied of each origin.
                let mut serving: BTreeMap<u64, u64> = BTreeMap::new();
                for ((a, n), x) in &s.sockets {
                    let routable = s.registry.member(a, *n).is_some_and(|m| m.routable());
                    if a != app || *n == node || !routable {
                        continue;
                    }
                    for (origin, through) in &x.acks {
                        let top = serving.entry(*origin).or_default();
                        *top = (*top).max(*through);
                    }
                }
                Registry::in_sync(node, &acks, &serving)
            };
            if synced {
                s.registry.set_syncing(app, node, false);
            }
            (receipt, recovered, synced)
        };
        if let Some((handle, through)) = receipt {
            let _ = handle.control(Frame::Receipt(Receipt { through }));
        }
        if recovered {
            self.note(app, hex(node), "caught up: no longer lagging".into());
        }
        if synced {
            self.note(app, hex(node), "in sync with the group: serving requests".into());
        }
    }

    /// `ACK`, `CATCHUP`, `HEADERS`, `FETCH`, `DONE` and `HAVE` from `from`.
    pub(super) fn relay_substrate(&self, app: &str, from: u64, frame: Frame) {
        match frame {
            Frame::Ack(ack) => {
                let mut s = self.state();
                if let Some(own) = ack.applied.iter().find(|(o, _)| *o == from) {
                    s.registry.high_water(app, from, own.1);
                }
                if let Some(x) = s.sockets.get_mut(&(app.to_string(), from)) {
                    x.acks = ack.applied;
                }
            }
            Frame::Catchup(c) => {
                let requester = c.reply_to;
                let answerer = match c.to {
                    Some(n) => self
                        .state()
                        .sockets
                        .contains_key(&(app.to_string(), n))
                        .then_some(n),
                    None => self.state().registry.answerer(app, from),
                };
                match answerer {
                    Some(n) => {
                        self.note(
                            app,
                            hex(from),
                            format!("catch-up {} asked of {}", c.request, hex(n)),
                        );
                        self.to_member(app, n, Frame::Catchup(c));
                    }
                    // No member ready to answer, and writes seen here that the
                    // requester lacks: their member is away — restarting, or
                    // still loading. A `DONE` now would tell a newcomer it had
                    // caught up on nothing, and it joined empty for good (an
                    // admission catch-up answered in 15 ms while .5 restarted).
                    // Left unanswered, the requester asks again once it goes
                    // quiet, by when a member may answer.
                    None if self.state().registry.owed(app, requester, &c.since) => {
                        self.note(
                            app,
                            hex(from),
                            format!(
                                "catch-up {} left open: no member can answer what it lacks yet",
                                c.request
                            ),
                        );
                    }
                    // No other member here and nothing seen it lacks: nothing
                    // is owed through this gateway.
                    None => self.to_catchup(
                        app,
                        requester,
                        Frame::Done(Done {
                            request: c.request,
                            reply_to: requester,
                        }),
                    ),
                }
            }
            Frame::Headers(h) => {
                let to = h.reply_to;
                self.to_catchup(app, to, Frame::Headers(h));
            }
            Frame::Done(d) => {
                let to = d.reply_to;
                self.to_catchup(app, to, Frame::Done(d));
            }
            Frame::Fetch(f) => {
                let to = f.from_node;
                self.to_member(app, to, Frame::Fetch(f));
            }
            Frame::Have(h) => {
                self.note(app, hex(from), "offered what it holds: a resync".into());
                let s = self.state();
                for ((a, n), x) in &s.sockets {
                    if a == app && *n != from {
                        let _ = x.handle.catchup(Frame::Have(h.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    /// Queue a catch-up frame for `node`, unless [`dropped_at_cap`].
    fn to_catchup(&self, app: &str, node: u64, frame: Frame) {
        let mut s = self.state();
        let Some(x) = s.sockets.get_mut(&(app.to_string(), node)) else {
            return;
        };
        if dropped_at_cap(&frame, x.handle.gauges().catchup()) {
            x.dropped += 1;
            return;
        }
        let _ = x.handle.catchup(frame);
    }
}

/// Whether a catch-up frame is dropped, with `queued` bytes already on the
/// member's catch-up lane: when it would pass [`CATCHUP_CAP`].
///
/// **A `DONE` is never dropped.** A dropped `HEADERS` or fetched record is
/// repaired by the requester — it asks again from where it stands, and a fetch
/// ending short of a record fetches it again — but a dropped `DONE` leaves its
/// catch-up waiting until it goes quiet, 30 s on. It is a few bytes.
fn dropped_at_cap(frame: &Frame, queued: usize) -> bool {
    !matches!(frame, Frame::Done(_)) && over_cap(frame.encoded_len(), queued)
}

/// Whether `bytes` more would take a catch-up lane holding `queued` past
/// [`CATCHUP_CAP`].
///
/// **A record larger than the cap counts as the cap**: it goes onto an empty
/// lane, and waits for one. The cap bounds a lane's backlog, not the size of
/// what crosses it. Measured against its own length, a record over 16 MiB
/// was dropped however empty the lane — every try, so it was left after its
/// third and the next catch-up dropped it again: npcd on `.5`, 2026-10-17,
/// caught up 6 of 192 records (24 of 21,290 MiB) in ten minutes, every
/// record it landed a small one.
fn over_cap(bytes: usize, queued: usize) -> bool {
    queued + bytes.min(CATCHUP_CAP) > CATCHUP_CAP
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Headers;

    fn headers() -> Frame {
        Frame::Headers(Headers {
            request: 1,
            reply_to: 7,
            answerer: 3,
            origin: 9,
            from: 1,
            to: 1,
            gone: vec![],
            records: vec![],
        })
    }

    /// Past the cap a `HEADERS` is dropped and a `DONE` still goes.
    #[test]
    fn a_done_passes_the_cap_and_headers_do_not() {
        let done = Frame::Done(Done {
            request: 1,
            reply_to: 7,
        });
        assert!(!dropped_at_cap(&headers(), 0));
        assert!(dropped_at_cap(&headers(), CATCHUP_CAP));
        assert!(!dropped_at_cap(&done, CATCHUP_CAP));
        assert!(!dropped_at_cap(&done, usize::MAX / 2));
    }

    /// **A record over the cap crosses an empty lane and waits for a busy
    /// one**; one under it fills the lane up to the cap, as before.
    #[test]
    fn a_record_larger_than_the_cap_needs_only_an_empty_lane() {
        let big = 110 << 20;
        assert!(!over_cap(big, 0));
        assert!(over_cap(big, 1));
        assert!(!over_cap(CATCHUP_CAP, 0));
        assert!(!over_cap(CATCHUP_CAP - 10, 10));
        assert!(over_cap(CATCHUP_CAP - 10, 11));
        assert!(!over_cap(1 << 20, CATCHUP_CAP - (1 << 20)));
    }
}

//! What a gateway relays of the substrate (§4.3, §4.4, §7.2, §7.5, §9.1): live
//! events to the app's other members, catch-up traffic to the one member it
//! answers, the `RECEIPT`s that tell a sender what has been relayed, the
//! lagging state of a member that cannot keep up, and world pulses. The
//! gateway never decodes a record; it moves bytes between sockets.

use std::time::Instant;

use super::hub::{hex, Hub, CATCHUP_CAP, LAG_CAP};
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
                    Some(x) if x.handle.gauges().catchup() + bytes > CATCHUP_CAP => {
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
        let (receipt, recovered) = {
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
            (receipt, recovered)
        };
        if let Some((handle, through)) = receipt {
            let _ = handle.control(Frame::Receipt(Receipt { through }));
        }
        if recovered {
            self.note(app, hex(node), "caught up: no longer lagging".into());
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
                    // No other member here: nothing is owed through this
                    // gateway.
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

    /// Queue a catch-up frame for `node`, unless it would pass the cap.
    fn to_catchup(&self, app: &str, node: u64, frame: Frame) {
        let mut s = self.state();
        let Some(x) = s.sockets.get_mut(&(app.to_string(), node)) else {
            return;
        };
        if x.handle.gauges().catchup() + frame.encoded_len() > CATCHUP_CAP {
            x.dropped += 1;
            return;
        }
        let _ = x.handle.catchup(frame);
    }
}

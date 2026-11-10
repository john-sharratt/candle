//! What a gateway relays of the substrate (§4.3, §4.4, §7.2, §7.5, §9.1): live
//! events to the app's other members, catch-up traffic to the one member it
//! answers, the `RECEIPT`s that tell a sender what has been relayed, the
//! lagging state of a member that cannot keep up, and world pulses. The
//! gateway never decodes a record; it moves bytes between sockets.

use std::collections::BTreeMap;
use std::time::Instant;

use super::hub::{hex, Hub, CATCHUP_CAP, LAG_CAP};
use super::registry::{Held, Registry};
use crate::coord::CoordState;
use crate::fragment::Whole;
use crate::frame::{Done, Frame, Lost, MemberState, Receipt};
use crate::session::SessionHandle;

/// What a member's `COORD` says it holds of each origin: the highest `oseq` and
/// how many below it are missing. Nothing for a member that has sent none.
fn held_of(coord: Option<&CoordState>) -> BTreeMap<u64, Held> {
    coord
        .into_iter()
        .flat_map(|c| c.applied.iter())
        .filter_map(|line| {
            let origin = u64::from_str_radix(&line.origin, 16).ok()?;
            Some((
                origin,
                Held {
                    top: line.top,
                    missing: line.missing,
                },
            ))
        })
        .collect()
}

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
                        log_drop(app, to, x.dropped, bytes, x.handle.gauges().catchup());
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
                        // A member in fault mode still hears live events: it
                        // applies the tombstones among them, which is how a
                        // full disk frees itself, and the upgrade decisions,
                        // by which it installs a certified build, and drops
                        // the rest at once, to be fetched by its catch-up on
                        // leaving (§7.7).
                        if m.state() == MemberState::Upgrading {
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
    /// its lagging once its lane has drained.
    ///
    /// **A drained lane ends the lagging; being in sync ends the syncing.** A
    /// lagging member is sent no live event, so the writes it lacks reach it
    /// only by catch-up — which trails any origin still writing. Waiting for its
    /// `ACK`s to reach that origin's `high_water` before sending it live events
    /// again waited for ever: WhiteWorld's zend, lagging behind `.5`'s ingest,
    /// was routed nothing for as long as `.5` ingested (2026-10-18). Drained, it
    /// takes live events again and is syncing: routed no request until its
    /// `ACK`s are in sync with the members serving, judged with
    /// [`super::registry::SYNC_SLACK`] like any member joining beside them.
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
            let lagging = s.registry.member(app, node).is_some_and(|m| m.lagging);
            let recovered = lagging && drained;
            if recovered {
                s.registry.set_lagging(app, node, false);
                s.registry.set_syncing(app, node, true);
            }
            // Synced: its own `ACK`s cover every other origin this gateway has
            // heard, members away included — judged here, not by the member,
            // whose catch-up can end early (a `DONE` with nobody to answer).
            let syncing = s.registry.member(app, node).is_some_and(|m| m.syncing);
            let synced = syncing && {
                // The most any member serving now has applied of each origin.
                let mut serving: BTreeMap<u64, u64> = BTreeMap::new();
                for m in s.registry.members(app) {
                    if m.node == node || !m.routable() {
                        continue;
                    }
                    for (origin, through) in &m.applied {
                        let top = serving.entry(*origin).or_default();
                        *top = (*top).max(*through);
                    }
                }
                let acks = s
                    .registry
                    .member(app, node)
                    .map(|m| m.applied.clone())
                    .unwrap_or_default();
                // What it holds of each origin past a hole, and what each
                // serving origin misses of itself: both from `COORD`.
                let held = held_of(
                    s.sockets
                        .get(&(app.to_string(), node))
                        .and_then(|x| x.coord.as_ref()),
                );
                let mut origin_missing: BTreeMap<u64, u64> = BTreeMap::new();
                for m in s.registry.members(app) {
                    if m.node == node || !m.routable() {
                        continue;
                    }
                    let coord = s
                        .sockets
                        .get(&(app.to_string(), m.node))
                        .and_then(|x| x.coord.as_ref());
                    if let Some(own) = held_of(coord).get(&m.node) {
                        origin_missing.insert(m.node, own.missing);
                    }
                }
                Registry::in_sync(node, &acks, &serving, &held, &origin_missing)
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
            self.note(
                app,
                hex(node),
                "lane drained: no longer lagging, syncing until in sync".into(),
            );
        }
        if synced {
            self.note(
                app,
                hex(node),
                "in sync with the group: serving requests".into(),
            );
        }
    }

    /// `CATCHUP`, `HEADERS`, `FETCH`, `DONE` and `HAVE` from `from`. Its `ACK`
    /// is a state report (`reports::apply_report`).
    pub(super) fn relay_substrate(&self, app: &str, from: u64, frame: Frame) {
        match frame {
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
                        self.state()
                            .open
                            .asked_catchup(app, n, requester, c.request);
                        self.to_member(app, n, Frame::Catchup(c));
                    }
                    // No member ready to answer, and writes seen here that the
                    // requester lacks: their member is away — restarting, or
                    // still loading. A `DONE` now would tell a newcomer it had
                    // caught up on nothing, and it joined empty for good (an
                    // admission catch-up answered in 15 ms while .5 restarted).
                    // `LOST` instead: the requester asks again later, by when a
                    // member may answer — told now, not left to a silence that
                    // a reply still arriving over a slow link also gives.
                    None if self.state().registry.owed(app, requester, &c.since) => {
                        self.note(
                            app,
                            hex(from),
                            format!(
                                "catch-up {} left open: no member can answer what it lacks yet",
                                c.request
                            ),
                        );
                        self.to_member(
                            app,
                            requester,
                            Frame::Lost(Lost {
                                request: c.request,
                                reply_to: requester,
                            }),
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
                self.state().open.done(app, to, d.request);
                self.to_catchup(app, to, Frame::Done(d));
            }
            Frame::Fetch(f) => {
                let to = f.from_node;
                let answers = {
                    let mut s = self.state();
                    let here = s.sockets.contains_key(&(app.to_string(), to));
                    if here {
                        s.open.asked_fetch(app, to, f.reply_to, f.request);
                    }
                    here
                };
                if answers {
                    self.to_member(app, to, Frame::Fetch(f));
                } else {
                    // Its answerer is gone: nothing will come.
                    self.to_member(
                        app,
                        f.reply_to,
                        Frame::Lost(Lost {
                            request: f.request,
                            reply_to: f.reply_to,
                        }),
                    );
                }
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
            log_drop(
                app,
                node,
                x.dropped,
                frame.encoded_len(),
                x.handle.gauges().catchup(),
            );
            return;
        }
        let _ = x.handle.catchup(frame);
    }
}

/// Whether the `dropped`th catch-up frame dropped for one socket is logged:
/// the first and then each power of two, so a lane that stays full is seen
/// without a line for every frame it refuses.
fn logs_drop(dropped: u64) -> bool {
    dropped.is_power_of_two()
}

/// The log line for a catch-up frame of `bytes` dropped for `node` with
/// `queued` bytes already on its lane, when [`logs_drop`] says so. A dropped
/// record is fetched again by its requester; one that keeps being dropped is a
/// catch-up that never finishes.
fn log_drop(app: &str, node: u64, dropped: u64, bytes: usize, queued: usize) {
    if logs_drop(dropped) {
        tracing::warn!(
            target: "cluster::gateway::relay",
            app,
            node = %hex(node),
            dropped,
            bytes,
            queued,
            cap = CATCHUP_CAP,
            "a catch-up frame was dropped at the member's lane cap"
        );
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
        let big = CATCHUP_CAP * 2;
        assert!(!over_cap(big, 0));
        assert!(over_cap(big, 1));
        assert!(!over_cap(CATCHUP_CAP, 0));
        assert!(!over_cap(CATCHUP_CAP - 10, 10));
        assert!(over_cap(CATCHUP_CAP - 10, 11));
        assert!(!over_cap(1 << 20, CATCHUP_CAP - (1 << 20)));
    }

    /// **A lane that keeps refusing is logged at doubling counts**: the first
    /// drop and each power of two, not every frame.
    #[test]
    fn drops_are_logged_at_doubling_counts() {
        let logged: Vec<u64> = (1..=40).filter(|d| logs_drop(*d)).collect();
        assert_eq!(logged, vec![1, 2, 4, 8, 16, 32]);
        assert!(!logs_drop(0));
    }
}

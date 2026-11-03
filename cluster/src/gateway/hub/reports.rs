//! A node's state reports (`docs/replicated_substrate.md` §5.4): what a
//! session writes into its node's own entries — its claims, capacity, routes,
//! completeness, coordination state, what it has applied, the builds it holds.
//! Each is applied under the state's guard ([`apply_report`]); the hub then
//! does what follows from it, the relays and notes, outside the guard
//! ([`Applied`]). While a replacement of the session is pending they are held
//! back ([`Held`]), each kind kept to what it last said, and applied under the
//! guard that undoes the replacement when its handshake fails.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use super::{hex, State};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::coord::CoordState;
use crate::dispatch::RouteTable;
use crate::frame::{Claim, Complete, Frame, Unclaim};
use crate::gateway::coord_log::changes;
use crate::gateway::upgrades::Action;

/// What a session reports of its node's state: dropped once the session is
/// replaced, for the node's newer session says it of the process that is
/// there now. A request routed on, or a one-time reply or offer — a vote, a
/// catch-up answer, a build chunk, a resync's `HAVE` — still passes: a
/// process still running sends it once, and a peer awaits it.
pub(in crate::gateway) fn is_state_report(frame: &Frame) -> bool {
    matches!(
        frame,
        Frame::Claim(_)
            | Frame::Unclaim(_)
            | Frame::Capacity(_)
            | Frame::Routes(_)
            | Frame::Complete(_)
            | Frame::Coord(_)
            | Frame::Ack(_)
            | Frame::BuildOffer(_)
            | Frame::BuildHave(_)
    )
}

/// What follows from a report applied: done by the hub once the guard it was
/// applied under is released.
#[derive(Default)]
pub(in crate::gateway) struct Applied {
    /// Coordination-log lines, by subject.
    pub notes: Vec<(String, String)>,
    /// Relayed to the app's other members.
    pub to_others: Vec<Frame>,
    /// Relayed to every other member of every app.
    pub to_every_other: Vec<Frame>,
    /// The app's membership listing changed.
    pub members_changed: bool,
    /// A `CAPACITY` arrived: the units waiting for room are placed again.
    pub capacity: bool,
    pub actions: Vec<Action>,
}

/// Apply `frame`, a state report from `node` of `app`, under the state's
/// guard: `Err` for one that breaks the protocol.
pub(in crate::gateway) fn apply_report(
    s: &mut State,
    app: &str,
    node: u64,
    frame: Frame,
) -> Result<Applied, Close> {
    let mut done = Applied::default();
    match frame {
        Frame::Capacity(c) => {
            let state = c.state;
            if let Some(x) = s.sockets.get_mut(&(app.to_string(), node)) {
                x.capacity_at = Instant::now();
            }
            if s.registry.set_silent(app, node, false) {
                done.notes
                    .push((hex(node), "reporting again: routed as ever".into()));
            }
            done.members_changed = s.registry.capacity(app, node, c);
            done.actions = s.upgrades.capacity(app, node, state, now_ms());
            done.capacity = true;
        }
        Frame::Routes(r) => {
            let table = RouteTable::decode(&r.table)
                .map_err(|e| Close::new(CloseCode::Malformed, format!("ROUTES: {e}")))?;
            s.registry.routes(app, node, table);
        }
        Frame::Claim(c) => {
            if s.registry.claims(app).claim(&c.key, c.epoch) {
                done.notes.push((
                    String::from_utf8_lossy(&c.key).into_owned(),
                    format!("claimed by {} at {}", hex(node), c.epoch.ms),
                ));
            }
            done.to_others.push(Frame::Claim(c));
        }
        Frame::Unclaim(u) => {
            // Relayed only when it let a claim go: a release that matched
            // nothing — a duplicate, or one superseded — would fire every
            // peer's release hook for a key it holds no word of.
            if s.registry.claims(app).unclaim(&u.key, u.epoch) {
                s.registry.released(app, &u.key, node, u.through);
                done.notes.push((
                    String::from_utf8_lossy(&u.key).into_owned(),
                    format!("released by {} through oseq {}", hex(node), u.through),
                ));
                done.to_others.push(Frame::Unclaim(u));
            }
        }
        // The member holds these keys complete: each one's next turn may go
        // to it (`docs/cooperative_admission.md` §2.2). The gateway's own — no
        // other member reads it.
        Frame::Complete(c) => s.registry.completed(app, node, &c.keys),
        Frame::Coord(c) => {
            let state = CoordState::decode(&c.state).ok_or_else(|| {
                Close::new(
                    CloseCode::Malformed,
                    "a COORD that is not a coordination state",
                )
            })?;
            let key = (app.to_string(), node);
            let Some(socket) = s.sockets.get_mut(&key) else {
                return Ok(done);
            };
            socket.coord = Some(state.clone());
            let subject = hex(node);
            for line in changes(s.reported.get(&key), &state) {
                done.notes.push((subject.clone(), line));
            }
            if let Some(view) = &state.upgrade {
                let floor = s.admission.floor(app).cloned();
                done.actions = s.upgrades.coord(app, view, floor.as_ref(), now_ms());
            }
            s.reported.insert(key, state);
        }
        Frame::Ack(ack) => s.registry.applied(app, node, ack.applied),
        Frame::BuildOffer(b) => {
            s.held.announce(node, &b, true);
            done.to_every_other.push(Frame::BuildOffer(b));
        }
        Frame::BuildHave(b) => {
            s.held.announce(node, &b, false);
            done.to_every_other.push(Frame::BuildHave(b));
        }
        _ => {}
    }
    Ok(done)
}

/// A key's claim reports held back: the release of a claim the gateway
/// holds, and the latest claim.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct KeyHeld {
    unclaim: Option<Unclaim>,
    claim: Option<Claim>,
}

/// The state reports a session sent while a replacement of it was pending,
/// each kind kept to what it last said — so however long the handshake, they
/// stay as few as the kinds and keys they name, and none of what they ended
/// at is lost: the latest `CAPACITY`, `ROUTES`, `COORD` and `ACK`, each
/// build's latest offer and holding, every key reported complete, and each
/// key's net claim — a claim and the release of that same claim, both held,
/// come to nothing.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(in crate::gateway) struct Held {
    ack: Option<Frame>,
    capacity: Option<Frame>,
    routes: Option<Frame>,
    coord: Option<Frame>,
    builds: BTreeMap<(bool, u32, String, String), Frame>,
    complete: BTreeSet<Vec<u8>>,
    keys: BTreeMap<Vec<u8>, KeyHeld>,
}

impl Held {
    /// Hold `frame`, a state report, back.
    pub fn keep(&mut self, frame: Frame) {
        match frame {
            Frame::Ack(_) => self.ack = Some(frame),
            Frame::Capacity(_) => self.capacity = Some(frame),
            Frame::Routes(_) => self.routes = Some(frame),
            Frame::Coord(_) => self.coord = Some(frame),
            Frame::BuildOffer(ref b) | Frame::BuildHave(ref b) => {
                let offer = matches!(frame, Frame::BuildOffer(_));
                let id = (
                    offer,
                    b.build.count,
                    b.build.hash.clone(),
                    b.build.local.clone(),
                );
                self.builds.insert(id, frame);
            }
            Frame::Complete(c) => self.complete.extend(c.keys),
            Frame::Claim(c) => {
                let key = c.key.clone();
                self.keys.entry(key).or_default().claim = Some(c);
            }
            Frame::Unclaim(u) => {
                let held = self.keys.entry(u.key.clone()).or_default();
                match held.claim.as_ref().is_some_and(|c| c.epoch == u.epoch) {
                    // The release of a claim itself held back: neither
                    // reached the gateway, and neither is to.
                    true => held.claim = None,
                    false => held.unclaim = Some(u),
                }
            }
            _ => {}
        }
    }

    /// What is held, in the order to apply it: each key's release before its
    /// claim.
    pub fn into_frames(self) -> Vec<Frame> {
        let mut frames: Vec<Frame> = [self.ack, self.capacity, self.routes, self.coord]
            .into_iter()
            .flatten()
            .collect();
        frames.extend(self.builds.into_values());
        if !self.complete.is_empty() {
            frames.push(Frame::Complete(Complete {
                keys: self.complete.into_iter().collect(),
            }));
        }
        for held in self.keys.into_values() {
            frames.extend(held.unclaim.map(Frame::Unclaim));
            frames.extend(held.claim.map(Frame::Claim));
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Capacity, Epoch, MemberState};

    fn capacity(state: MemberState) -> Frame {
        Frame::Capacity(Capacity {
            state,
            ..Capacity::default()
        })
    }

    fn claim(key: &[u8], ms: u64) -> Claim {
        Claim {
            key: key.to_vec(),
            epoch: Epoch { ms, node: 1 },
        }
    }

    fn unclaim(key: &[u8], ms: u64) -> Unclaim {
        Unclaim {
            key: key.to_vec(),
            epoch: Epoch { ms, node: 1 },
            through: 9,
        }
    }

    /// **Held reports keep what each kind ended at, however many arrive**: the
    /// latest `CAPACITY`, every key reported complete, and each key's net
    /// claim — a claim and its own release both held come to nothing, a
    /// release of a claim the gateway holds stays, and a later claim follows
    /// it.
    #[test]
    fn held_reports_keep_what_each_kind_ended_at() {
        let mut held = Held::default();
        for _ in 0..10_000 {
            held.keep(capacity(MemberState::Loading));
        }
        held.keep(capacity(MemberState::Ready));
        held.keep(Frame::Complete(Complete {
            keys: vec![b"a".to_vec()],
        }));
        held.keep(Frame::Complete(Complete {
            keys: vec![b"b".to_vec(), b"a".to_vec()],
        }));
        held.keep(Frame::Claim(claim(b"gone", 5)));
        held.keep(Frame::Unclaim(unclaim(b"gone", 5)));
        held.keep(Frame::Unclaim(unclaim(b"moved", 3)));
        held.keep(Frame::Claim(claim(b"moved", 7)));
        assert_eq!(
            held.into_frames(),
            vec![
                capacity(MemberState::Ready),
                Frame::Complete(Complete {
                    keys: vec![b"a".to_vec(), b"b".to_vec()],
                }),
                Frame::Unclaim(unclaim(b"moved", 3)),
                Frame::Claim(claim(b"moved", 7)),
            ]
        );
    }

    /// A resync's `HAVE`, a catch-up and a vote are not state reports; a
    /// node's capacity, claims and acknowledgements are.
    #[test]
    fn state_reports_are_what_a_node_writes_of_itself() {
        assert!(is_state_report(&capacity(MemberState::Ready)));
        assert!(is_state_report(&Frame::Claim(claim(b"k", 1))));
        assert!(!is_state_report(&Frame::Ping([0; 8])));
    }
}

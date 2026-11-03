//! A node's state reports (`docs/replicated_substrate.md` §5.4): what a
//! session writes into its node's own entries — its claims, capacity, routes,
//! completeness, coordination state, what it has applied, the builds it holds.
//! Each is checked and decoded as it arrives ([`check_report`]), and applied
//! under the state's guard ([`apply_report`]); the hub then does what follows
//! from it, the relays and notes, outside the guard ([`Applied`]). While a
//! replacement of the session is pending they are held back ([`Held`]), each
//! kind kept to what it last said, and applied under the guard that undoes the
//! replacement when its handshake fails.

use std::collections::BTreeMap;
#[cfg(test)]
use std::fmt;
use std::time::Instant;

use super::{hex, State};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::coord::CoordState;
use crate::dispatch::RouteTable;
use crate::frame::{Ack, BuildArtifacts, Capacity, Claim, Complete, Frame, Unclaim};
use crate::gateway::coord_log::changes;
use crate::gateway::upgrades::Action;

/// A state report, checked and decoded ([`check_report`]): what is applied or
/// held back, so nothing held can fail to apply. What a session reports of
/// its node's state is dropped once the session is replaced, for the node's
/// newer session says it of the process that is there now.
pub(in crate::gateway) enum Report {
    Claim(Claim),
    Unclaim(Unclaim),
    Capacity(Capacity),
    Routes(RouteTable),
    Complete(Complete),
    /// Boxed: a coordination state is many times the size of any other report.
    Coord(Box<CoordState>),
    Ack(Ack),
    BuildOffer(BuildArtifacts),
    BuildHave(BuildArtifacts),
}

#[cfg(test)]
impl PartialEq for Report {
    fn eq(&self, other: &Report) -> bool {
        match (self, other) {
            (Report::Claim(a), Report::Claim(b)) => a == b,
            (Report::Unclaim(a), Report::Unclaim(b)) => a == b,
            (Report::Capacity(a), Report::Capacity(b)) => a == b,
            (Report::Routes(a), Report::Routes(b)) => a.encode() == b.encode(),
            (Report::Complete(a), Report::Complete(b)) => a == b,
            (Report::Coord(a), Report::Coord(b)) => a == b,
            (Report::Ack(a), Report::Ack(b)) => a == b,
            (Report::BuildOffer(a), Report::BuildOffer(b)) => a == b,
            (Report::BuildHave(a), Report::BuildHave(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(test)]
impl fmt::Debug for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Report::Claim(c) => write!(f, "Claim({c:?})"),
            Report::Unclaim(u) => write!(f, "Unclaim({u:?})"),
            Report::Capacity(c) => write!(f, "Capacity({:?})", c.state),
            Report::Routes(_) => write!(f, "Routes(..)"),
            Report::Complete(c) => write!(f, "Complete({c:?})"),
            Report::Coord(c) => write!(f, "Coord({c:?})"),
            Report::Ack(a) => write!(f, "Ack({a:?})"),
            Report::BuildOffer(b) => write!(f, "BuildOffer({b:?})"),
            Report::BuildHave(b) => write!(f, "BuildHave({b:?})"),
        }
    }
}

/// A frame from a member, as [`check_report`] sorts it.
pub(in crate::gateway) enum Arrived {
    /// A state report, checked and decoded.
    Report(Report),
    /// Anything else — a request routed on, or a one-time reply or offer: a
    /// vote, a catch-up answer, a build chunk, a resync's `HAVE`. It passes
    /// whatever session it came on: a process still running sends it once,
    /// and a peer awaits it.
    Other(Frame),
}

/// Sort `frame`, from `node`, as it arrives: a state report is checked — a
/// claim or release of its own node's, a `ROUTES` table and a `COORD` that
/// decode — before it is applied or held back, and decoded once for both.
/// `Err` closes the socket `4400`.
pub(in crate::gateway) fn check_report(frame: Frame, node: u64) -> Result<Arrived, Close> {
    let names = |kind: &str, other: u64| -> Result<(), Close> {
        match other == node {
            true => Ok(()),
            false => Err(Close::new(
                CloseCode::Malformed,
                format!("a {kind} from {} names node {}", hex(node), hex(other)),
            )),
        }
    };
    Ok(Arrived::Report(match frame {
        Frame::Claim(c) => {
            names("CLAIM", c.epoch.node)?;
            Report::Claim(c)
        }
        Frame::Unclaim(u) => {
            names("UNCLAIM", u.epoch.node)?;
            Report::Unclaim(u)
        }
        Frame::Routes(r) => Report::Routes(
            RouteTable::decode(&r.table)
                .map_err(|e| Close::new(CloseCode::Malformed, format!("ROUTES: {e}")))?,
        ),
        Frame::Coord(c) => {
            Report::Coord(Box::new(CoordState::decode(&c.state).ok_or_else(|| {
                Close::new(
                    CloseCode::Malformed,
                    "a COORD that is not a coordination state",
                )
            })?))
        }
        Frame::Capacity(c) => Report::Capacity(c),
        Frame::Complete(c) => Report::Complete(c),
        Frame::Ack(a) => Report::Ack(a),
        Frame::BuildOffer(b) => Report::BuildOffer(b),
        Frame::BuildHave(b) => Report::BuildHave(b),
        other => return Ok(Arrived::Other(other)),
    }))
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

/// Apply `report`, from `node` of `app`, under the state's guard.
pub(in crate::gateway) fn apply_report(
    s: &mut State,
    app: &str,
    node: u64,
    report: Report,
) -> Applied {
    let mut done = Applied::default();
    match report {
        Report::Capacity(c) => {
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
        Report::Routes(table) => s.registry.routes(app, node, table),
        Report::Claim(c) => {
            if s.registry.claims(app).claim(&c.key, c.epoch) {
                done.notes.push((
                    String::from_utf8_lossy(&c.key).into_owned(),
                    format!("claimed by {} at {}", hex(node), c.epoch.ms),
                ));
            }
            done.to_others.push(Frame::Claim(c));
        }
        Report::Unclaim(u) => {
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
        Report::Complete(c) => s.registry.completed(app, node, &c.keys),
        Report::Coord(state) => {
            let state = *state;
            let key = (app.to_string(), node);
            let Some(socket) = s.sockets.get_mut(&key) else {
                return done;
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
        Report::Ack(ack) => s.registry.applied(app, node, ack.applied),
        Report::BuildOffer(b) => {
            s.held.announce(node, &b, true);
            done.to_every_other.push(Frame::BuildOffer(b));
        }
        Report::BuildHave(b) => {
            s.held.announce(node, &b, false);
            done.to_every_other.push(Frame::BuildHave(b));
        }
    }
    done
}

/// A key's claim reports held back, each with when it arrived: the release
/// of a claim the gateway holds, and the latest claim.
#[derive(Default)]
struct KeyHeld {
    unclaim: Option<(u64, Unclaim)>,
    claim: Option<(u64, Claim)>,
}

/// A build's offer or holding, each with when its last report arrived: the
/// artifacts of every report for it, merged, as `HeldBuilds::announce` adds
/// them report by report.
type BuildHeld = (u64, BuildArtifacts);

/// The state reports a session sent while a replacement of it was pending,
/// each kind kept to what it last said — so however long the handshake, they
/// stay as few as the kinds and keys they name, and none of what they ended
/// at is lost: the latest `CAPACITY`, `ROUTES`, `COORD` and `ACK`, each
/// build's offer and holding with every artifact named, every key reported
/// complete, and each key's net claim — a release ends the claims held back
/// before it, and is kept itself. Each is applied in the order it
/// arrived — the latest of its kind, but a release repeated keeps its first
/// place — so what depends on an earlier report, a key's `COMPLETE` on its
/// `UNCLAIM`, an `ACK`'s placeability on a `CAPACITY`, sees it.
#[derive(Default)]
pub(in crate::gateway) struct Held {
    next: u64,
    ack: Option<(u64, Report)>,
    capacity: Option<(u64, Report)>,
    routes: Option<(u64, Report)>,
    coord: Option<(u64, Report)>,
    offers: BTreeMap<(u32, String, String), BuildHeld>,
    haves: BTreeMap<(u32, String, String), BuildHeld>,
    complete: BTreeMap<Vec<u8>, u64>,
    keys: BTreeMap<Vec<u8>, KeyHeld>,
}

/// `held`'s artifacts and those of `b`, the newer, merged; arrived at `seq`.
fn merge(held: Option<BuildHeld>, b: BuildArtifacts, seq: u64) -> BuildHeld {
    let Some((_, mut into)) = held else {
        return (seq, b);
    };
    for a in b.artifacts {
        if !into.artifacts.contains(&a) {
            into.artifacts.push(a);
        }
    }
    (seq, into)
}

impl Held {
    /// Hold `report` back.
    pub fn keep(&mut self, report: Report) {
        let seq = self.next;
        self.next += 1;
        match report {
            Report::Ack(_) => self.ack = Some((seq, report)),
            Report::Capacity(_) => self.capacity = Some((seq, report)),
            Report::Routes(_) => self.routes = Some((seq, report)),
            Report::Coord(_) => self.coord = Some((seq, report)),
            Report::BuildOffer(b) => {
                let id = (b.build.count, b.build.hash.clone(), b.build.local.clone());
                let held = self.offers.remove(&id);
                self.offers.insert(id, merge(held, b, seq));
            }
            Report::BuildHave(b) => {
                let id = (b.build.count, b.build.hash.clone(), b.build.local.clone());
                let held = self.haves.remove(&id);
                self.haves.insert(id, merge(held, b, seq));
            }
            Report::Complete(c) => {
                for key in c.keys {
                    self.complete.insert(key, seq);
                }
            }
            Report::Claim(c) => {
                let key = c.key.clone();
                self.keys.entry(key).or_default().claim = Some((seq, c));
            }
            Report::Unclaim(u) => {
                // A release ends the node's claim on the key whatever epoch
                // either names (`Claims::unclaim`): a claim held back before
                // it comes to nothing, and the release itself is kept, for
                // the gateway may hold an earlier claim of the node's it ends.
                let held = self.keys.entry(u.key.clone()).or_default();
                held.claim = None;
                // The same release again keeps its first place, so what
                // followed the first — the key's `COMPLETE` — still follows.
                let at = match &held.unclaim {
                    Some((first, before)) if before.epoch == u.epoch => *first,
                    _ => seq,
                };
                held.unclaim = Some((at, u));
            }
        }
    }

    /// What is held, in the order to apply it: the order each kept report
    /// arrived in, the keys reported complete together in one report as they
    /// were.
    pub fn into_reports(self) -> Vec<Report> {
        let mut reports: Vec<(u64, Report)> = [self.ack, self.capacity, self.routes, self.coord]
            .into_iter()
            .flatten()
            .collect();
        reports.extend(
            self.offers
                .into_values()
                .map(|(seq, b)| (seq, Report::BuildOffer(b))),
        );
        reports.extend(
            self.haves
                .into_values()
                .map(|(seq, b)| (seq, Report::BuildHave(b))),
        );
        let mut complete: BTreeMap<u64, Vec<Vec<u8>>> = BTreeMap::new();
        for (key, seq) in self.complete {
            complete.entry(seq).or_default().push(key);
        }
        reports.extend(
            complete
                .into_iter()
                .map(|(seq, keys)| (seq, Report::Complete(Complete { keys }))),
        );
        for held in self.keys.into_values() {
            reports.extend(held.unclaim.map(|(seq, u)| (seq, Report::Unclaim(u))));
            reports.extend(held.claim.map(|(seq, c)| (seq, Report::Claim(c))));
        }
        reports.sort_by_key(|(seq, _)| *seq);
        reports.into_iter().map(|(_, report)| report).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Artifact, BuildId, Epoch, MemberState, Routes};

    fn capacity(state: MemberState) -> Report {
        Report::Capacity(Capacity {
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

    fn complete(keys: &[&[u8]]) -> Report {
        Report::Complete(Complete {
            keys: keys.iter().map(|k| k.to_vec()).collect(),
        })
    }

    /// **Held reports keep what each kind ended at, however many arrive**: the
    /// latest `CAPACITY`, every key reported complete — together, as they
    /// were reported — and each key's net claim: a release ends a claim held
    /// back before it and is kept itself, for the gateway may hold an earlier
    /// claim of the node's it ends; a later claim follows it.
    #[test]
    fn held_reports_keep_what_each_kind_ended_at() {
        let mut held = Held::default();
        for _ in 0..10_000 {
            held.keep(capacity(MemberState::Loading));
        }
        held.keep(capacity(MemberState::Ready));
        held.keep(complete(&[b"a"]));
        held.keep(complete(&[b"b", b"a"]));
        held.keep(Report::Claim(claim(b"gone", 5)));
        held.keep(Report::Unclaim(unclaim(b"gone", 5)));
        held.keep(Report::Unclaim(unclaim(b"moved", 3)));
        held.keep(Report::Claim(claim(b"moved", 7)));
        assert_eq!(
            held.into_reports(),
            vec![
                capacity(MemberState::Ready),
                complete(&[b"a", b"b"]),
                Report::Unclaim(unclaim(b"gone", 5)),
                Report::Unclaim(unclaim(b"moved", 3)),
                Report::Claim(claim(b"moved", 7)),
            ]
        );
    }

    /// **Held reports are applied in the order they arrived**: a key's
    /// `UNCLAIM` before the `COMPLETE` that followed it — and still before it
    /// when the same release is sent again after — and a `CAPACITY` before
    /// the `ACK` that followed it, whatever their kinds.
    #[test]
    fn held_reports_keep_their_arrival_order() {
        let mut held = Held::default();
        held.keep(Report::Ack(Ack {
            applied: vec![(1, 1)],
        }));
        held.keep(Report::Unclaim(unclaim(b"k", 3)));
        held.keep(capacity(MemberState::Ready));
        held.keep(complete(&[b"k"]));
        held.keep(Report::Unclaim(unclaim(b"k", 3)));
        held.keep(Report::Ack(Ack {
            applied: vec![(1, 9)],
        }));
        assert_eq!(
            held.into_reports(),
            vec![
                Report::Unclaim(unclaim(b"k", 3)),
                capacity(MemberState::Ready),
                complete(&[b"k"]),
                Report::Ack(Ack {
                    applied: vec![(1, 9)],
                }),
            ]
        );
    }

    /// **A build's held reports keep every artifact they named**, merged, as
    /// its holding is announced artifact by artifact.
    #[test]
    fn a_builds_held_artifacts_are_merged() {
        let build = BuildId {
            count: 4,
            hash: "h4".into(),
            local: String::new(),
        };
        let artifact = |kind: &str| Artifact {
            kind: kind.into(),
            platform: "windows".into(),
            size: 1,
            sha256: [0; 32],
        };
        let have = |kinds: &[&str]| {
            Report::BuildHave(BuildArtifacts {
                build: build.clone(),
                artifacts: kinds.iter().map(|k| artifact(k)).collect(),
            })
        };
        let mut held = Held::default();
        held.keep(have(&["motherd"]));
        held.keep(have(&["web", "motherd"]));
        assert_eq!(held.into_reports(), vec![have(&["motherd", "web"])]);
    }

    /// **A frame is sorted, and a report checked and decoded, as it arrives**:
    /// one naming another node's claim and a `ROUTES` that does not decode are
    /// refused `4400`; a claim of its own is a report; a ping is not, and
    /// passes as it came.
    #[test]
    fn a_report_is_checked_as_it_arrives() {
        let other = Frame::Claim(Claim {
            key: b"k".to_vec(),
            epoch: Epoch { ms: 1, node: 2 },
        });
        assert_eq!(check_report(other, 1).err().unwrap().code, 4400);
        let garbled = Frame::Routes(Routes { table: vec![0xff] });
        assert_eq!(check_report(garbled, 1).err().unwrap().code, 4400);
        let Ok(Arrived::Report(own)) = check_report(Frame::Claim(claim(b"k", 1)), 1) else {
            panic!("a report");
        };
        assert_eq!(own, Report::Claim(claim(b"k", 1)));
        assert!(matches!(
            check_report(Frame::Ping([0; 8]), 1),
            Ok(Arrived::Other(Frame::Ping(_)))
        ));
    }
}

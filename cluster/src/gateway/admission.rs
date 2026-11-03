//! Who a gateway admits (§3.3–§3.6), after the proofs: the build floor, the
//! model lock, the clock, and one session per node — in that order, each
//! refusal a close with its code and a reason naming the values involved.
//!
//! All of it is held in memory: a gateway restart forgets every floor and
//! lock, and the first member of each app to reconnect sets them again — the
//! first that is neither a local build (§3.10) nor on a build its `HELLO` says
//! the group rejected (§3.11), for the floor. A member below the floor is
//! admitted to upgrade when a member holds the floor's build (§3.5).
//!
//! The [`SERVICES`](super::SERVICES) have neither: keep releases each
//! machine's on its own, so a member is admitted on whatever build it runs
//! (§3.1, D1).

use std::collections::HashMap;

use super::hub::is_service;
use crate::close::{Close, CloseCode};
use crate::frame::{BuildId, Hello, ModelId};
use crate::handshake::PROTO;

/// How far a member's clock may be from the gateway's (§3.4, 4417).
pub const MAX_SKEW_MS: u64 = 60_000;

/// The apps that carry a model lock. Agents and the services run none (§3.6).
fn locked(app: &str) -> bool {
    app != "agent" && !is_service(app)
}

/// The apps that carry a build floor: every one but the services.
fn floored(app: &str) -> bool {
    !is_service(app)
}

#[derive(Debug, Default)]
struct AppPolicy {
    /// The floor's build, and the node that set it.
    floor: Option<(BuildId, u64, String)>,
    /// The model the group is locked to.
    lock: Option<ModelId>,
    /// The live session of each node.
    sessions: HashMap<u64, u64>,
}

/// A member let in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Admitted {
    pub session: u64,
    /// The session this one replaces, to be closed 4409.
    pub replaces: Option<u64>,
    /// The app's floor as admission compared it; `None` while only local
    /// builds have joined, which never set it.
    pub floor: Option<BuildId>,
    /// Below the floor, admitted to take the floor's build (§3.5).
    pub upgrading: bool,
    /// On an older protocol: its session carries only what crosses protocols.
    pub behind: bool,
    /// This admission set the app's floor, the first to: undone, it leaves
    /// none ([`Admission::undo`]).
    pub set_floor: bool,
}

#[derive(Debug, Default)]
pub struct Admission {
    apps: HashMap<String, AppPolicy>,
    next_session: u64,
}

impl Admission {
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit `hello`, whose proof has passed, at the gateway's `now_ms`.
    /// `held` says whether some member holds a build's artifacts: a member
    /// below the floor is admitted to upgrade when the floor's build is held,
    /// and closed 4410 otherwise (§3.5).
    pub fn decide(
        &mut self,
        hello: &Hello,
        now_ms: u64,
        held: impl Fn(&BuildId) -> bool,
    ) -> Result<Admitted, Close> {
        let app = hello.app.as_str();
        let policy = self.apps.entry(app.to_string()).or_default();

        // An older protocol is a forced upgrade: the group cannot talk to it,
        // so it takes the floor's build at once, with no certification and no
        // grace. The floor is only ever set by a member on this protocol, so
        // the first one to join moves everyone behind it. Until one holds a
        // build, it redials.
        if hello.proto < PROTO {
            let reachable = policy
                .floor
                .as_ref()
                .filter(|(floor, ..)| {
                    floored(app)
                        && !hello.build.is_local()
                        && hello.build.count < floor.count
                        && held(floor)
                })
                .map(|(floor, ..)| floor.clone());
            let Some(floor) = reachable else {
                return Err(Close::new(
                    CloseCode::Protocol,
                    format!(
                        "protocol {} is not supported; this gateway speaks {PROTO} and no {app} \
                         member holds a build to upgrade to",
                        hello.proto
                    ),
                ));
            };
            self.next_session += 1;
            let session = self.next_session;
            let replaces = policy.sessions.insert(hello.node, session);
            return Ok(Admitted {
                session,
                replaces,
                floor: Some(floor),
                upgrading: true,
                behind: true,
                set_floor: false,
            });
        }

        // The build floor, on `count` alone; a local build is never below it
        // (§3.10).
        let mut upgrading = false;
        if let Some((floor, setter, setter_host)) = policy.floor.as_ref().filter(|_| floored(app)) {
            if !hello.build.is_local() && hello.build.count < floor.count {
                if !held(floor) {
                    return Err(Close::new(
                        CloseCode::Superseded,
                        format!(
                            "build {} ({}) is below the {app} floor {} ({}), set by node \
                             {setter:016x} on {setter_host}, and no member holds that build to \
                             upgrade to; rebuild",
                            hello.build.count, hello.build.hash, floor.count, floor.hash
                        ),
                    ));
                }
                upgrading = true;
            }
        }

        // The model lock.
        if locked(app) {
            let Some(model) = &hello.model else {
                return Err(Close::new(
                    CloseCode::Model,
                    format!("a {app} member names no model; its substrate is bound to one"),
                ));
            };
            if let Some(lock) = &policy.lock {
                if !lock.same_model(model) {
                    return Err(Close::new(
                        CloseCode::Model,
                        format!(
                            "the {app} group runs {}; this member runs {}. A new model is moved \
                             to by hand, not by an upgrade",
                            lock.describe, model.describe
                        ),
                    ));
                }
            }
        }

        // The clock.
        let skew = now_ms.abs_diff(hello.now_ms);
        if skew > MAX_SKEW_MS {
            return Err(Close::new(
                CloseCode::Skew,
                format!(
                    "this member's clock is {:.1} s from the gateway's; set its time",
                    skew as f64 / 1000.0
                ),
            ));
        }

        // Admitted: the first member that is neither a local build nor on a
        // build the group rejected sets the floor, the first member the lock,
        // and a node's newer session replaces its older one.
        let set_floor =
            floored(app) && policy.floor.is_none() && !hello.build.is_local() && !hello.rejected;
        if set_floor {
            policy.floor = Some((hello.build.clone(), hello.node, hello.host.clone()));
        }
        if locked(app) && policy.lock.is_none() {
            policy.lock = hello.model.clone();
        }
        self.next_session += 1;
        let session = self.next_session;
        let replaces = policy.sessions.insert(hello.node, session);
        let floor = policy.floor.as_ref().map(|(b, _, _)| b.clone());
        Ok(Admitted {
            session,
            replaces,
            floor,
            upgrading,
            behind: false,
            set_floor,
        })
    }

    /// Set `app`'s floor to `build`, in either direction: a certified build's
    /// rollout, or a forced one (§3.9, §3.11). `by` names what set it.
    pub fn set_floor(&mut self, app: &str, build: &BuildId, node: u64, by: &str) {
        self.apps.entry(app.to_string()).or_default().floor =
            Some((build.clone(), node, by.to_string()));
    }

    /// `admitted`, a session of `node`'s whose handshake failed, is undone, as
    /// though it had never been decided: the session it replaced is the node's
    /// again when `standing`, the session of the node's socket still here, is
    /// that one — so a group of one, its member redialling, keeps its model
    /// lock. With none replaced, or the one replaced gone since, the node has
    /// none. A floor it set is lifted, so a member never admitted holds no
    /// one below it out. Returns whether the model lock cleared, the app's
    /// last session gone, as on [`Self::left`].
    pub fn undo(
        &mut self,
        app: &str,
        node: u64,
        admitted: &Admitted,
        standing: Option<u64>,
    ) -> bool {
        let Some(policy) = self.apps.get_mut(app) else {
            return false;
        };
        if policy.sessions.get(&node) == Some(&admitted.session) {
            match standing.filter(|s| Some(*s) == admitted.replaces) {
                Some(replaced) => policy.sessions.insert(node, replaced),
                None => policy.sessions.remove(&node),
            };
        }
        if admitted.set_floor {
            policy.floor = None;
        }
        let cleared = policy.sessions.is_empty() && policy.lock.is_some();
        if policy.sessions.is_empty() {
            policy.lock = None;
        }
        cleared
    }

    /// `node`'s `session` ended. When it was the app's last, the model lock
    /// clears (§3.6).
    pub fn left(&mut self, app: &str, node: u64, session: u64) {
        let Some(policy) = self.apps.get_mut(app) else {
            return;
        };
        if policy.sessions.get(&node) == Some(&session) {
            policy.sessions.remove(&node);
        }
        if policy.sessions.is_empty() {
            policy.lock = None;
        }
    }

    /// The app's floor, for the status page.
    pub fn floor(&self, app: &str) -> Option<&BuildId> {
        self.apps.get(app)?.floor.as_ref().map(|(b, _, _)| b)
    }

    /// The node and host that set the app's floor.
    pub fn floor_set_by(&self, app: &str) -> Option<(u64, &str)> {
        self.apps
            .get(app)?
            .floor
            .as_ref()
            .map(|(_, node, host)| (*node, host.as_str()))
    }

    /// The app's model lock, for the status page.
    pub fn lock(&self, app: &str) -> Option<&ModelId> {
        self.apps.get(app)?.lock.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(count: u32, local: &str) -> BuildId {
        BuildId {
            count,
            hash: format!("h{count}"),
            local: local.into(),
        }
    }

    fn model(weights: u8) -> ModelId {
        ModelId {
            arch: "Qwen4Exp".into(),
            weights: [weights; 32],
            tokenizer: [1; 32],
            rope: "r".into(),
            describe: format!("model {weights}"),
        }
    }

    fn hello(app: &str, node: u64, build: BuildId, model: Option<ModelId>, now_ms: u64) -> Hello {
        Hello {
            proto: PROTO,
            app: app.into(),
            node,
            host: format!("m{node}"),
            card: node,
            build,
            rejected: false,
            model,
            power: 1,
            repos: vec![],
            now_ms,
            nonce: [0; 32],
        }
    }

    const NOW: u64 = 1_700_000_000_000;

    /// Admit `h` at [`NOW`], no build held anywhere.
    fn admit(a: &mut Admission, h: Hello) -> Result<Admitted, Close> {
        a.decide(&h, NOW, |_| false)
    }

    fn zend(node: u64, b: BuildId) -> Hello {
        hello("zend", node, b, Some(model(1)), NOW)
    }

    /// The first member sets the floor; equal and higher builds are admitted,
    /// a lower one is closed 4410 with both builds and the setter named; a
    /// local build is never below the floor; another app is untouched.
    #[test]
    fn the_floor_is_set_by_the_first_member_and_holds_the_rest() {
        let mut a = Admission::new();
        let first = admit(&mut a, zend(1, build(4815, ""))).unwrap();
        assert_eq!(
            (first.session, first.floor.map(|f| f.count)),
            (1, Some(4815))
        );
        assert!(admit(&mut a, zend(2, build(4815, ""))).is_ok());
        assert!(admit(&mut a, zend(3, build(4820, ""))).is_ok());
        let low = admit(&mut a, zend(4, build(4812, ""))).unwrap_err();
        assert_eq!(low.code, 4410);
        assert_eq!(
            low.reason,
            "build 4812 (h4812) is below the zend floor 4815 (h4815), set by node \
             0000000000000001 on m1, and no member holds that build to upgrade to; rebuild"
        );
        assert!(admit(&mut a, zend(5, build(4000, "abcd"))).is_ok());
        assert!(admit(
            &mut a,
            hello("npcd", 6, build(4000, ""), Some(model(2)), NOW)
        )
        .is_ok());
    }

    /// **An admission undone is as though never decided**: the first member's,
    /// failed, sets no floor and no lock — a lower build is admitted after it,
    /// a member of another model too — and a redial undone while the session it
    /// would replace still stands leaves that one the node's, its lock held.
    #[test]
    fn an_admission_undone_leaves_no_floor_and_keeps_the_standing_lock() {
        let mut a = Admission::new();
        let failed = admit(&mut a, zend(1, build(4815, ""))).unwrap();
        assert!(failed.set_floor);
        assert!(a.undo("zend", 1, &failed, None), "the lock it set clears");
        assert!(admit(&mut a, zend(2, build(4812, ""))).is_ok(), "no floor");

        let mut a = Admission::new();
        let standing = admit(&mut a, zend(1, build(4815, ""))).unwrap();
        let redial = admit(&mut a, zend(1, build(4815, ""))).unwrap();
        assert_eq!(redial.replaces, Some(standing.session));
        assert!(!redial.set_floor);
        assert!(!a.undo("zend", 1, &redial, Some(standing.session)));
        let other = admit(
            &mut a,
            hello("zend", 2, build(4815, ""), Some(model(2)), NOW),
        );
        assert_eq!(other.unwrap_err().code, 4412, "the lock stands");
        a.left("zend", 1, standing.session);
        assert!(
            admit(
                &mut a,
                hello("zend", 3, build(4815, ""), Some(model(2)), NOW)
            )
            .is_ok(),
            "the standing session was the node's: its leaving clears the lock"
        );
    }

    /// **A service has no floor** (D1): keep releases each machine's on its
    /// own, so two machines on different builds are both admitted to serve,
    /// whichever joined first — neither closed 4410 nor sent to upgrade.
    #[test]
    fn a_service_is_admitted_on_any_build() {
        for app in ["keep", "mcp"] {
            let mut a = Admission::new();
            let first = admit(&mut a, hello(app, 1, build(4900, ""), None, NOW)).unwrap();
            assert_eq!(first.floor, None);
            let older = a
                .decide(&hello(app, 2, build(4800, ""), None, NOW), NOW, |_| true)
                .unwrap();
            assert_eq!((older.floor, older.upgrading), (None, false));
            let refused_nowhere = admit(&mut a, hello(app, 3, build(4700, ""), None, NOW));
            assert!(refused_nowhere.is_ok(), "{app}: {refused_nowhere:?}");
            assert!(a.floor(app).is_none());
        }
    }

    /// A member below the floor is admitted to upgrade when some member holds
    /// the floor's build.
    #[test]
    fn below_the_floor_with_the_build_held_is_admitted_to_upgrade() {
        let mut a = Admission::new();
        admit(&mut a, zend(1, build(4815, ""))).unwrap();
        let low = a
            .decide(&zend(4, build(4812, "")), NOW, |b| b.count == 4815)
            .unwrap();
        assert!(low.upgrading);
        assert_eq!(low.floor.map(|f| f.count), Some(4815));
        let level = a.decide(&zend(5, build(4815, "")), NOW, |_| true).unwrap();
        assert!(!level.upgrading);
    }

    /// **A protocol bump upgrades the group by itself.** Behind this gateway's
    /// protocol a member waits (4426, which redials) and never sets the floor;
    /// the first member on this protocol does, and from then on the one behind
    /// is admitted to take its build — certified or not — with no lock taken.
    /// A service, a local build, and a member with nothing to reach still wait.
    #[test]
    fn a_member_behind_the_protocol_takes_the_first_build_on_it() {
        let mut a = Admission::new();
        let behind = |node, b| {
            let mut h = zend(node, b);
            h.proto = PROTO - 1;
            h
        };
        let waits = a.decide(&behind(1, build(4800, "")), NOW, |_| true);
        assert_eq!(waits.unwrap_err().code, 4426);
        assert!(a.floor("zend").is_none(), "it never sets the floor");

        admit(&mut a, zend(2, build(4815, ""))).unwrap();
        let unheld = a.decide(&behind(1, build(4800, "")), NOW, |_| false);
        assert_eq!(unheld.unwrap_err().code, 4426);

        let lane = a
            .decide(&behind(1, build(4800, "")), NOW, |b| b.count == 4815)
            .unwrap();
        assert!(lane.upgrading && lane.behind);
        assert_eq!(lane.floor.map(|f| f.count), Some(4815));

        let local = a.decide(&behind(3, build(4800, "abcd")), NOW, |_| true);
        assert_eq!(local.unwrap_err().code, 4426);
        let level = a.decide(&behind(4, build(4815, "")), NOW, |_| true);
        assert_eq!(level.unwrap_err().code, 4426, "nothing to upgrade to");
        let mut service = hello("mcp", 5, build(4800, ""), None, NOW);
        service.proto = PROTO - 1;
        assert_eq!(a.decide(&service, NOW, |_| true).unwrap_err().code, 4426);
    }

    /// Neither a local build nor a rejected one sets the floor; the first
    /// member that is neither does, and a floor set by a rollout moves either
    /// way.
    #[test]
    fn a_local_or_rejected_build_never_sets_the_floor() {
        let mut a = Admission::new();
        let l = admit(&mut a, zend(1, build(4900, "abcd"))).unwrap();
        assert_eq!(l.floor, None);
        let mut rejected = zend(3, build(4950, ""));
        rejected.rejected = true;
        let r = admit(&mut a, rejected).unwrap();
        assert_eq!(r.floor, None);
        assert!(a.floor("zend").is_none());
        admit(&mut a, zend(2, build(4815, ""))).unwrap();
        assert_eq!(a.floor("zend").map(|f| f.count), Some(4815));
        a.set_floor("zend", &build(4810, ""), 9, "rollout 1 by ops");
        assert_eq!(a.floor("zend").map(|f| f.count), Some(4810));
        assert_eq!(a.floor_set_by("zend"), Some((9, "rollout 1 by ops")));
    }

    /// The first member sets the lock; a mismatch is closed 4412 naming both;
    /// the lock clears when the group empties; agents have none.
    #[test]
    fn the_model_lock_holds_until_the_group_empties() {
        let mut a = Admission::new();
        let s = admit(&mut a, zend(1, build(1, ""))).unwrap();
        let other = admit(&mut a, hello("zend", 2, build(1, ""), Some(model(2)), NOW)).unwrap_err();
        assert_eq!(other.code, 4412);
        assert!(other.reason.contains("model 1") && other.reason.contains("model 2"));
        a.left("zend", 1, s.session);
        assert!(a.lock("zend").is_none());
        assert!(admit(&mut a, hello("zend", 2, build(1, ""), Some(model(2)), NOW)).is_ok());
        assert!(admit(&mut a, hello("agent", 9, build(1, ""), None, NOW)).is_ok());
        assert!(admit(&mut a, hello("keep", 8, build(1, ""), None, NOW)).is_ok());
        assert!(admit(&mut a, hello("mcp", 7, build(1, ""), None, NOW)).is_ok());
        assert_eq!(
            admit(&mut a, hello("npcd", 3, build(1, ""), None, NOW))
                .unwrap_err()
                .code,
            4412
        );
    }

    #[test]
    fn a_clock_more_than_a_minute_off_is_closed_4417() {
        let mut a = Admission::new();
        let ahead = admit(
            &mut a,
            hello("zend", 1, build(1, ""), Some(model(1)), NOW + 61_000),
        )
        .unwrap_err();
        assert_eq!(ahead.code, 4417);
        assert_eq!(
            ahead.reason,
            "this member's clock is 61.0 s from the gateway's; set its time"
        );
        assert!(admit(
            &mut a,
            hello("zend", 1, build(1, ""), Some(model(1)), NOW - 59_000)
        )
        .is_ok());
    }

    /// A node's newer session replaces its older one, which is closed 4409.
    #[test]
    fn a_second_session_of_a_node_replaces_the_first() {
        let mut a = Admission::new();
        let first = admit(&mut a, zend(7, build(1, ""))).unwrap();
        let second = admit(&mut a, zend(7, build(1, ""))).unwrap();
        assert_eq!(second.replaces, Some(first.session));
        // The old session leaving does not remove the new one.
        a.left("zend", 7, first.session);
        assert!(a.lock("zend").is_some());
    }

    /// A refused member changes nothing: no floor, no lock, no session.
    #[test]
    fn a_refusal_records_nothing() {
        let mut a = Admission::new();
        assert!(admit(
            &mut a,
            hello("zend", 1, build(5, ""), Some(model(1)), NOW + 120_000)
        )
        .is_err());
        assert!(a.floor("zend").is_none() && a.lock("zend").is_none());
    }
}

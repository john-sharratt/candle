//! Who a gateway admits (§3.3–§3.6), after the proofs: the build floor, the
//! model lock, the clock, and one session per node — in that order, each
//! refusal a close with its code and a reason naming the values involved.
//!
//! All of it is held in memory: a gateway restart forgets every floor and
//! lock, and the first member of each app to reconnect sets them again.

use std::collections::HashMap;

use crate::close::{Close, CloseCode};
use crate::frame::{BuildId, Hello, ModelId};

/// How far a member's clock may be from the gateway's (§3.4, 4417).
pub const MAX_SKEW_MS: u64 = 60_000;

/// The apps that carry a model lock; agents run none (§3.6).
fn locked(app: &str) -> bool {
    app != "agent"
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
    /// The app's floor as admission compared it.
    pub floor: BuildId,
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
    pub fn decide(&mut self, hello: &Hello, now_ms: u64) -> Result<Admitted, Close> {
        let app = hello.app.as_str();
        let policy = self.apps.entry(app.to_string()).or_default();

        // The build floor, on `count` alone; a local build is never below it
        // (§3.10).
        if let Some((floor, setter, setter_host)) = &policy.floor {
            if !hello.build.is_local() && hello.build.count < floor.count {
                return Err(Close::new(
                    CloseCode::Superseded,
                    format!(
                        "build {} ({}) is below the {app} floor {} ({}), set by node {setter:016x} \
                         on {setter_host}; rebuild",
                        hello.build.count, hello.build.hash, floor.count, floor.hash
                    ),
                ));
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

        // Admitted: the first member sets the floor and the lock, and a node's
        // newer session replaces its older one.
        if policy.floor.is_none() {
            policy.floor = Some((hello.build.clone(), hello.node, hello.host.clone()));
        }
        if locked(app) && policy.lock.is_none() {
            policy.lock = hello.model.clone();
        }
        self.next_session += 1;
        let session = self.next_session;
        let replaces = policy.sessions.insert(hello.node, session);
        let floor = policy.floor.as_ref().expect("set above").0.clone();
        Ok(Admitted {
            session,
            replaces,
            floor,
        })
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
            proto: 1,
            app: app.into(),
            node,
            host: format!("m{node}"),
            build,
            model,
            power: 1,
            repos: vec![],
            now_ms,
            nonce: [0; 32],
        }
    }

    const NOW: u64 = 1_700_000_000_000;

    /// The first member sets the floor; equal and higher builds are admitted,
    /// a lower one is closed 4410 with both builds and the setter named; a
    /// local build is never below the floor; another app is untouched.
    #[test]
    fn the_floor_is_set_by_the_first_member_and_holds_the_rest() {
        let mut a = Admission::new();
        let first = a
            .decide(&hello("zend", 1, build(4815, ""), Some(model(1)), NOW), NOW)
            .unwrap();
        assert_eq!((first.session, first.floor.count), (1, 4815));
        assert!(a
            .decide(&hello("zend", 2, build(4815, ""), Some(model(1)), NOW), NOW)
            .is_ok());
        assert!(a
            .decide(&hello("zend", 3, build(4820, ""), Some(model(1)), NOW), NOW)
            .is_ok());
        let low = a
            .decide(&hello("zend", 4, build(4812, ""), Some(model(1)), NOW), NOW)
            .unwrap_err();
        assert_eq!(low.code, 4410);
        assert_eq!(
            low.reason,
            "build 4812 (h4812) is below the zend floor 4815 (h4815), set by node \
             0000000000000001 on m1; rebuild"
        );
        assert!(a
            .decide(
                &hello("zend", 5, build(4000, "abcd"), Some(model(1)), NOW),
                NOW
            )
            .is_ok());
        assert!(a
            .decide(&hello("npcd", 6, build(4000, ""), Some(model(2)), NOW), NOW)
            .is_ok());
    }

    /// The first member sets the lock; a mismatch is closed 4412 naming both;
    /// the lock clears when the group empties; agents have none.
    #[test]
    fn the_model_lock_holds_until_the_group_empties() {
        let mut a = Admission::new();
        let s = a
            .decide(&hello("zend", 1, build(1, ""), Some(model(1)), NOW), NOW)
            .unwrap();
        let other = a
            .decide(&hello("zend", 2, build(1, ""), Some(model(2)), NOW), NOW)
            .unwrap_err();
        assert_eq!(other.code, 4412);
        assert!(other.reason.contains("model 1") && other.reason.contains("model 2"));
        a.left("zend", 1, s.session);
        assert!(a.lock("zend").is_none());
        assert!(a
            .decide(&hello("zend", 2, build(1, ""), Some(model(2)), NOW), NOW)
            .is_ok());
        assert!(a
            .decide(&hello("agent", 9, build(1, ""), None, NOW), NOW)
            .is_ok());
        assert_eq!(
            a.decide(&hello("npcd", 3, build(1, ""), None, NOW), NOW)
                .unwrap_err()
                .code,
            4412
        );
    }

    #[test]
    fn a_clock_more_than_a_minute_off_is_closed_4417() {
        let mut a = Admission::new();
        let ahead = a
            .decide(
                &hello("zend", 1, build(1, ""), Some(model(1)), NOW + 61_000),
                NOW,
            )
            .unwrap_err();
        assert_eq!(ahead.code, 4417);
        assert_eq!(
            ahead.reason,
            "this member's clock is 61.0 s from the gateway's; set its time"
        );
        assert!(a
            .decide(
                &hello("zend", 1, build(1, ""), Some(model(1)), NOW - 59_000),
                NOW
            )
            .is_ok());
    }

    /// A node's newer session replaces its older one, which is closed 4409.
    #[test]
    fn a_second_session_of_a_node_replaces_the_first() {
        let mut a = Admission::new();
        let first = a
            .decide(&hello("zend", 7, build(1, ""), Some(model(1)), NOW), NOW)
            .unwrap();
        let second = a
            .decide(&hello("zend", 7, build(1, ""), Some(model(1)), NOW), NOW)
            .unwrap();
        assert_eq!(second.replaces, Some(first.session));
        // The old session leaving does not remove the new one.
        a.left("zend", 7, first.session);
        assert!(a.lock("zend").is_some());
    }

    /// A refused member changes nothing: no floor, no lock, no session.
    #[test]
    fn a_refusal_records_nothing() {
        let mut a = Admission::new();
        assert!(a
            .decide(
                &hello("zend", 1, build(5, ""), Some(model(1)), NOW + 120_000),
                NOW
            )
            .is_err());
        assert!(a.floor("zend").is_none() && a.lock("zend").is_none());
    }
}

//! MCP sessions, kept as records so any member answers a client
//! (`docs/claude_zend_gateway.md` §3), and the signal a waiting `task_wait`
//! wakes on.
//!
//! - **Sessions** are `mcp_server`'s ([`mcp_server::session`]): a handshake
//!   client's opened by `initialize`, a 2026-07-28 client's keyed by its
//!   person, its client and — when the aggregator calls on its behalf — the
//!   aggregator's own session. zend keeps them in its substrate
//!   ([`RecordStore`]), so every member reads the same ones, and holds one
//!   thing on each: the thread `thread_attach` bound it to ([`ATTACHED`]), which
//!   `"self"` names. Which client opened it bounds how long a call may wait
//!   (§3.3).
//! - **A change signal**, bumped whenever a worker's state moves on this
//!   member, which `task_wait` wakes on; it also re-reads every few seconds,
//!   since a worker another member runs changes its record without bumping
//!   this one.

use std::sync::Arc;

use mcp_server::session::TOUCH_EVERY_MS;
use mcp_server::{Client, Session, Store};
use tokio::sync::watch;
use web::auth::oidc::random_token;

use super::parts::Parts;
use super::records::Records;

/// The longest any call waits when it names no `wait_s`: short enough that a
/// running worker reports what it has done every 45 seconds, and Claude can
/// stop waiting on what it reads.
pub const DEFAULT_WAIT_S: u64 = 45;

/// What a session holds the thread `thread_attach` bound it to under.
pub const ATTACHED: &str = "attached";

const PREFIX: &str = "mcp/session/";

fn key(id: &str) -> String {
    format!("{PREFIX}{id}")
}

/// Sessions as records in `R`: the substrate in the daemon, a map in tests.
pub struct RecordStore<R>(pub R);

impl<R: Records> RecordStore<R> {
    fn put(&self, id: &str, s: &Session) -> anyhow::Result<()> {
        self.0.put(&key(id), serde_json::to_value(s)?)
    }

    fn read(&self, id: &str) -> Option<Session> {
        serde_json::from_value(self.0.get(&key(id))?).ok()
    }

    /// Cease every stale session.
    fn sweep(&self, now: u64) {
        for (k, v) in self.0.under(PREFIX) {
            // One that does not read as a session is swept with them.
            let stale = match serde_json::from_value::<Session>(v) {
                Ok(s) => s.stale(now),
                Err(_) => true,
            };
            if stale {
                self.0.cease(&k).ok();
            }
        }
    }
}

impl<R: Records> Store for RecordStore<R> {
    fn open(
        &self,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<String> {
        self.sweep(now);
        let id = random_token();
        self.put(&id, &Session::new(owner, client, protocol, now))?;
        Ok(id)
    }

    fn get(&self, id: &str, now: u64) -> Option<Session> {
        let mut s = self.read(id)?;
        if now.saturating_sub(s.used_ms) > TOUCH_EVERY_MS {
            s.used_ms = now;
            self.put(id, &s).ok();
        }
        Some(s)
    }

    /// Two members' first calls at once may each open the session; the
    /// records keep the later write, and either holds nothing yet. Opening
    /// one sweeps the stale, as a handshake does: a server in front opens one
    /// of these per session of its own, and never a handshake.
    fn ensure(
        &self,
        id: &str,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<Session> {
        if let Some(s) = self.get(id, now) {
            return Ok(s);
        }
        self.sweep(now);
        let s = Session::new(owner, client, protocol, now);
        self.put(id, &s)?;
        Ok(s)
    }

    fn hold(&self, id: &str, key: &str, value: &str, now: u64) -> anyhow::Result<()> {
        let Some(mut s) = self.read(id) else {
            anyhow::bail!("the session is gone; start again");
        };
        s.held.insert(key.to_string(), value.to_string());
        s.used_ms = now;
        self.put(id, &s)
    }

    fn close(&self, id: &str) -> anyhow::Result<()> {
        self.0.cease(&key(id))
    }
}

/// What `/v1/mcp` holds in memory: the change signal, and the runner that
/// starts workers' parts.
pub struct McpState {
    changed: watch::Sender<u64>,
    parts: Arc<Parts>,
}

impl Default for McpState {
    fn default() -> Self {
        Self {
            changed: watch::channel(0).0,
            parts: Arc::default(),
        }
    }
}

impl McpState {
    /// The `sub_run` runner every tool context carries.
    pub fn parts(&self) -> &Arc<Parts> {
        &self.parts
    }

    /// Tell every waiter a worker's state moved.
    pub fn bump(&self) {
        self.changed.send_modify(|n| *n = n.wrapping_add(1));
    }

    /// A receiver that wakes on the next [`Self::bump`].
    pub fn watch(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use mcp_server::session::SESSION_IDLE_MS;
    use mcp_server::stateless_key;

    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn a_session_records_its_owner_and_its_attached_thread() {
        let r = RecordStore(MemRecords::default());
        let id = r.open("o", Client::Other, "2025-11-25", 10).unwrap();
        assert_eq!(r.get(&id, 11).unwrap().held(ATTACHED), None);
        r.hold(&id, ATTACHED, "review", 12).unwrap();
        assert_eq!(r.get(&id, 13).unwrap().held(ATTACHED), Some("review"));
        assert_eq!(r.get(&id, 13).unwrap().owner, "o");
        r.close(&id).unwrap();
        assert!(r.get(&id, 14).is_none());
        assert!(r.hold(&id, ATTACHED, "x", 15).is_err());
    }

    /// **Each session of the aggregator in front has its own attached
    /// thread**, though they are one person on one client.
    #[test]
    fn sessions_on_behalf_of_two_aggregator_sessions_attach_apart() {
        let r = RecordStore(MemRecords::default());
        let a = stateless_key("o", Client::Other, Some("s1"));
        let b = stateless_key("o", Client::Other, Some("s2"));
        r.ensure(&a, "o", Client::Other, "2026-07-28", 10).unwrap();
        r.ensure(&b, "o", Client::Other, "2026-07-28", 10).unwrap();
        r.hold(&a, ATTACHED, "review", 11).unwrap();
        r.hold(&b, ATTACHED, "bc-wearables", 11).unwrap();
        let again = r.ensure(&a, "o", Client::Other, "2026-07-28", 12).unwrap();
        assert_eq!(again.held(ATTACHED), Some("review"));
        assert_eq!(r.get(&b, 12).unwrap().held(ATTACHED), Some("bc-wearables"));
    }

    #[test]
    fn a_stateless_session_unused_for_a_month_is_swept_when_another_opens() {
        let r = RecordStore(MemRecords::default());
        let old = stateless_key("o", Client::Other, Some("s1"));
        let new = stateless_key("o", Client::Other, Some("s2"));
        r.ensure(&old, "o", Client::Other, "2026-07-28", 0).unwrap();
        r.ensure(&new, "o", Client::Other, "2026-07-28", SESSION_IDLE_MS + 1)
            .unwrap();
        assert_eq!(r.0.under(PREFIX).len(), 1);
        assert!(r.get(&new, SESSION_IDLE_MS + 2).is_some());
    }

    #[test]
    fn a_session_unused_for_a_month_is_swept_by_the_next_open() {
        let r = RecordStore(MemRecords::default());
        let old = r.open("o", Client::Other, "2025-11-25", 0).unwrap();
        r.open("o", Client::Other, "2025-11-25", SESSION_IDLE_MS + 1)
            .unwrap();
        assert!(r.get(&old, SESSION_IDLE_MS + 2).is_none());
        assert_eq!(r.0.under(PREFIX).len(), 1);
    }

    #[tokio::test]
    async fn a_waiter_wakes_on_a_bump() {
        let m = McpState::default();
        let mut rx = m.watch();
        m.bump();
        rx.changed().await.unwrap();
    }
}

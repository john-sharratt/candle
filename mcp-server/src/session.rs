//! MCP sessions, and where a server keeps them.
//!
//! - **A handshake session** (2025-11-25) is opened by `initialize` and named
//!   by the `Mcp-Session-Id` the server answers.
//! - **A 2026-07-28 client** has no sessions, so what is held for it is keyed
//!   by the person and the client they use ([`stateless_key`]) — the same
//!   across token refreshes — and, when the request names one in its `_meta`
//!   ([`crate::protocol::SESSION_META`]), the session of the server calling on
//!   its client's behalf. That is how an aggregator keeps one backend session
//!   per session of its own.
//!
//! Either records whose it is, which client opened it, and what the server
//! holds on it ([`Session::held`]). A session unused for [`SESSION_IDLE_MS`] is
//! swept.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::client::Client;

/// How long a session may go unused before it is swept.
pub const SESSION_IDLE_MS: u64 = 30 * 24 * 3600 * 1000;
/// How long a session's use is not re-recorded, so reading it is not a write
/// on every call.
pub const TOUCH_EVERY_MS: u64 = 3600 * 1000;

/// One MCP session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// The key of the person who opened it; a request on it from anyone else
    /// is refused.
    pub owner: String,
    pub client: Client,
    pub protocol: String,
    /// What the server keeps on it, by name: zend the thread `zend_attach`
    /// bound it to, an aggregator its sessions on the servers it wraps.
    #[serde(default)]
    pub held: BTreeMap<String, String>,
    /// When it was last used.
    #[serde(default)]
    pub used_ms: u64,
}

impl Session {
    pub fn new(owner: &str, client: Client, protocol: &str, now: u64) -> Self {
        Self {
            owner: owner.to_string(),
            client,
            protocol: protocol.to_string(),
            held: BTreeMap::new(),
            used_ms: now,
        }
    }

    /// What is held under `key`.
    pub fn held(&self, key: &str) -> Option<&str> {
        self.held.get(key).map(String::as_str)
    }

    /// Whether it has gone unused too long to keep at `now`.
    pub fn stale(&self, now: u64) -> bool {
        now.saturating_sub(self.used_ms) > SESSION_IDLE_MS
    }
}

/// What a 2026-07-28 client's session is kept under: its person, the client
/// they use, and the caller's own session when a server calls on a client's
/// behalf — none of which a token refresh changes.
pub fn stateless_key(owner: &str, client: Client, on_behalf: Option<&str>) -> String {
    match on_behalf {
        Some(session) => format!("who:{owner}:{}:{session}", client.slug()),
        None => format!("who:{owner}:{}", client.slug()),
    }
}

/// Where a server keeps its sessions.
pub trait Store: Send + Sync {
    /// Open a handshake session for `owner`; its id. Stale sessions are swept
    /// first.
    fn open(&self, owner: &str, client: Client, protocol: &str, now: u64)
        -> anyhow::Result<String>;

    /// The session under `id`, its use recorded at most every
    /// [`TOUCH_EVERY_MS`].
    fn get(&self, id: &str, now: u64) -> Option<Session>;

    /// The stateless session under `id`, opened for `owner` when there is
    /// none — found or opened at once, so two first calls together share one.
    fn ensure(
        &self,
        id: &str,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<Session>;

    /// Hold `value` under `key` on the session under `id`.
    fn hold(&self, id: &str, key: &str, value: &str, now: u64) -> anyhow::Result<()>;

    /// End the session under `id`.
    fn close(&self, id: &str) -> anyhow::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stateless_key_names_the_person_the_client_and_the_callers_session() {
        assert_eq!(
            stateless_key("o", Client::ClaudeCode, None),
            "who:o:claude_code"
        );
        assert_eq!(
            stateless_key("o", Client::Other, Some("s1")),
            "who:o:other:s1"
        );
    }

    #[test]
    fn a_session_holding_nothing_is_read_from_its_bare_fields_and_goes_stale_after_a_month() {
        let s: Session = serde_json::from_str(
            r#"{"owner":"o","client":"other","protocol":"2025-11-25","used_ms":5}"#,
        )
        .unwrap();
        assert_eq!(s, Session::new("o", Client::Other, "2025-11-25", 5));
        assert!(!s.stale(5 + SESSION_IDLE_MS));
        assert!(s.stale(6 + SESSION_IDLE_MS));
    }
}

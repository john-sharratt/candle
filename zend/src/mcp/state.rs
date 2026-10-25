//! MCP sessions, kept as records so any member answers a client
//! (`docs/claude_zend_gateway.md` §3), and the signal a waiting `zend_wait`
//! wakes on.
//!
//! - **A handshake session** (2025-11-25) is opened by `initialize` and named
//!   by the `Mcp-Session-Id` the server answers; a 2026-07-28 client has no
//!   sessions, so what is held for it is keyed by the person and the client
//!   they use ([`stateless_key`]) — the same across token refreshes. Either
//!   records whose it is, which client opened it — which bounds how long a
//!   call may wait (§3.3) — and the thread `zend_attach` bound it to, which
//!   `"self"` names. A session unused for [`SESSION_IDLE_MS`] is swept.
//! - **A change signal**, bumped whenever a worker's state moves on this
//!   member, which `zend_wait` wakes on; it also re-reads every few seconds,
//!   since a worker another member runs changes its record without bumping
//!   this one.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use web::auth::oidc::random_token;

use super::parts::Parts;
use super::records::Records;

/// Which client opened a session, as its `clientInfo.name` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Client {
    /// Claude Code: a call past two minutes moves to the background, and the
    /// server entry sets a long timeout, so a wait may hold an hour.
    ClaudeCode,
    /// claude.ai, Claude Desktop, or anyone else: a call is cut off at 240 s.
    Other,
}

impl Client {
    pub fn of(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        if name.contains("claude-code") || name.contains("claude code") {
            Self::ClaudeCode
        } else {
            Self::Other
        }
    }

    /// The longest a call from this client may wait, in seconds.
    pub fn wait_ceiling(self) -> u64 {
        match self {
            Self::ClaudeCode => 3_600,
            Self::Other => 200,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude_code",
            Self::Other => "other",
        }
    }
}

/// The longest any call waits when it names no `wait_s`.
pub const DEFAULT_WAIT_S: u64 = 50;

/// How long a session may go unused before it is swept.
pub const SESSION_IDLE_MS: u64 = 30 * 24 * 3600 * 1000;
/// How long a session's use is not re-recorded, so reading it is not a write
/// on every call.
const TOUCH_EVERY_MS: u64 = 3600 * 1000;

const PREFIX: &str = "mcp/session/";

/// One MCP session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// The key of the person who opened it; a request on it from anyone else
    /// is refused.
    pub owner: String,
    pub client: Client,
    pub protocol: String,
    /// The thread `zend_attach` bound it to.
    #[serde(default)]
    pub attached: Option<String>,
    /// When it was last used.
    #[serde(default)]
    pub used_ms: u64,
}

/// What a 2026-07-28 client's session is kept under: its person and the
/// client they use, which a token refresh does not change.
pub fn stateless_key(owner: &str, client: Client) -> String {
    format!("who:{owner}:{}", client.slug())
}

fn key(id: &str) -> String {
    format!("{PREFIX}{id}")
}

fn put(records: &dyn Records, id: &str, s: &Session) -> anyhow::Result<()> {
    records.put(&key(id), serde_json::to_value(s)?)
}

/// Cease every session unused for [`SESSION_IDLE_MS`].
fn sweep(records: &dyn Records, now: u64) {
    for (k, v) in records.under(PREFIX) {
        let used = v.get("used_ms").and_then(|u| u.as_u64()).unwrap_or(0);
        if now.saturating_sub(used) > SESSION_IDLE_MS {
            records.cease(&k).ok();
        }
    }
}

/// Open a handshake session for `owner`; its id.
pub fn open(
    records: &dyn Records,
    owner: &str,
    client: Client,
    protocol: &str,
    now: u64,
) -> anyhow::Result<String> {
    sweep(records, now);
    let id = random_token();
    let s = Session {
        owner: owner.to_string(),
        client,
        protocol: protocol.to_string(),
        attached: None,
        used_ms: now,
    };
    put(records, &id, &s)?;
    Ok(id)
}

/// The session under `id`, its use recorded at most hourly.
pub fn get(records: &dyn Records, id: &str, now: u64) -> Option<Session> {
    let mut s: Session = serde_json::from_value(records.get(&key(id))?).ok()?;
    if now.saturating_sub(s.used_ms) > TOUCH_EVERY_MS {
        s.used_ms = now;
        put(records, id, &s).ok();
    }
    Some(s)
}

/// The stateless session under `id`, opened for `owner` when there is none.
pub fn ensure(
    records: &dyn Records,
    id: &str,
    owner: &str,
    client: Client,
    protocol: &str,
    now: u64,
) -> anyhow::Result<Session> {
    if let Some(s) = get(records, id, now) {
        return Ok(s);
    }
    let s = Session {
        owner: owner.to_string(),
        client,
        protocol: protocol.to_string(),
        attached: None,
        used_ms: now,
    };
    put(records, id, &s)?;
    Ok(s)
}

/// Bind the session under `id` to `thread`.
pub fn attach(records: &dyn Records, id: &str, thread: &str, now: u64) -> anyhow::Result<()> {
    let Some(mut s) = get(records, id, now) else {
        anyhow::bail!("the session is gone; start again");
    };
    s.attached = Some(thread.to_string());
    s.used_ms = now;
    put(records, id, &s)
}

/// End the session under `id`.
pub fn close(records: &dyn Records, id: &str) -> anyhow::Result<()> {
    records.cease(&key(id))
}

/// What `/mcp` holds in memory: the change signal, and the runner that starts
/// workers' parts.
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
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn the_client_bounds_the_wait() {
        assert_eq!(Client::of("claude-code").wait_ceiling(), 3_600);
        assert_eq!(Client::of("Claude Code").wait_ceiling(), 3_600);
        assert_eq!(Client::of("claude-ai").wait_ceiling(), 200);
        assert_eq!(Client::of("anything").wait_ceiling(), 200);
    }

    #[test]
    fn a_session_records_its_owner_and_its_attached_thread() {
        let r = MemRecords::default();
        let id = open(&r, "o", Client::Other, "2025-11-25", 10).unwrap();
        assert_eq!(get(&r, &id, 11).unwrap().attached, None);
        attach(&r, &id, "review", 12).unwrap();
        assert_eq!(
            get(&r, &id, 13).unwrap().attached.as_deref(),
            Some("review")
        );
        close(&r, &id).unwrap();
        assert!(get(&r, &id, 14).is_none());
    }

    #[test]
    fn a_stateless_clients_session_outlives_its_tokens() {
        let r = MemRecords::default();
        let id = stateless_key("o", Client::ClaudeCode);
        assert_eq!(id, "who:o:claude_code");
        ensure(&r, &id, "o", Client::ClaudeCode, "2026-07-28", 10).unwrap();
        attach(&r, &id, "review", 11).unwrap();
        let again = ensure(&r, &id, "o", Client::ClaudeCode, "2026-07-28", 12).unwrap();
        assert_eq!(again.attached.as_deref(), Some("review"));
    }

    #[test]
    fn a_session_unused_for_a_month_is_swept_by_the_next_open() {
        let r = MemRecords::default();
        let old = open(&r, "o", Client::Other, "2025-11-25", 0).unwrap();
        open(&r, "o", Client::Other, "2025-11-25", SESSION_IDLE_MS + 1).unwrap();
        assert!(get(&r, &old, SESSION_IDLE_MS + 2).is_none());
        assert_eq!(r.under(PREFIX).len(), 1);
    }

    #[tokio::test]
    async fn a_waiter_wakes_on_a_bump() {
        let m = McpState::default();
        let mut rx = m.watch();
        m.bump();
        rx.changed().await.unwrap();
    }
}

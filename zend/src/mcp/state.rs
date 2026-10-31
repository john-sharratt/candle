//! MCP sessions, kept as records so any member answers a client
//! (`docs/claude_zend_gateway.md` §3), and the signal a waiting `repo_task_wait`
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
//!   member, which `repo_task_wait` wakes on; it also re-reads every few seconds,
//!   since a worker another member runs changes its record without bumping
//!   this one.

use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};

use mcp_server::session::TOUCH_EVERY_MS;
use mcp_server::{Client, Session, Store};
use tokio::sync::watch;
use web::auth::oidc::random_token;

use super::memory::Memory;
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

/// When each session was last used, as this process has seen it and its
/// record does not say yet: [`RecordStore::tend`] writes them in.
#[derive(Default)]
pub struct Used(Mutex<HashMap<String, u64>>);

impl Used {
    fn note(&self, id: &str, now: u64) {
        let mut used = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let at = used.entry(id.to_string()).or_insert(now);
        *at = (*at).max(now);
    }

    fn forget(&self, id: &str) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
    }

    fn take(&self) -> HashMap<String, u64> {
        mem::take(&mut *self.0.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Sessions as records in `R` — the substrate in the daemon, a map in tests
/// — and their use as noted in memory.
///
/// A request only reads: a stateless session holding nothing has no record
/// (it is its key alone), and a use is noted in memory. A record is written
/// when a session opens by handshake, which any member must then find, and
/// when it holds something; [`RecordStore::tend`], off the request path,
/// writes the noted uses in and sweeps the stale.
pub struct RecordStore<R> {
    pub records: R,
    used: Arc<Used>,
}

impl<R: Records> RecordStore<R> {
    pub fn new(records: R, used: Arc<Used>) -> Self {
        Self { records, used }
    }

    fn put(&self, id: &str, s: &Session) -> anyhow::Result<()> {
        self.records.put(&key(id), serde_json::to_value(s)?)
    }

    fn read(&self, id: &str) -> Option<Session> {
        serde_json::from_value(self.records.get(&key(id))?).ok()
    }

    /// Write each use noted since the last tend into its session's record
    /// when the record is [`TOUCH_EVERY_MS`] behind it, then cease every
    /// stale session; how many uses were written and sessions ceased.
    pub fn tend(&self, now: u64) -> (usize, usize) {
        let mut touched = 0;
        for (id, at) in self.used.take() {
            let Some(mut s) = self.read(&id) else {
                continue;
            };
            if at.saturating_sub(s.used_ms) > TOUCH_EVERY_MS {
                s.used_ms = at;
                if self.put(&id, &s).is_ok() {
                    touched += 1;
                }
            }
        }
        let mut ceased = 0;
        for (k, v) in self.records.under(PREFIX) {
            // One that does not read as a session is swept with them.
            let stale = match serde_json::from_value::<Session>(v) {
                Ok(s) => s.stale(now),
                Err(_) => true,
            };
            if stale && self.records.cease(&k).is_ok() {
                ceased += 1;
            }
        }
        (touched, ceased)
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
        let id = random_token();
        self.put(&id, &Session::new(owner, client, protocol, now))?;
        Ok(id)
    }

    fn get(&self, id: &str, now: u64) -> Option<Session> {
        let mut s = self.read(id)?;
        if now.saturating_sub(s.used_ms) > TOUCH_EVERY_MS {
            self.used.note(id, now);
        }
        s.used_ms = s.used_ms.max(now);
        Some(s)
    }

    /// The session's record, or — when it holds nothing yet — the session
    /// itself, unwritten: two members' first calls at once each find it so,
    /// and the first `hold` writes it.
    fn ensure(
        &self,
        id: &str,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<Session> {
        Ok(self
            .get(id, now)
            .unwrap_or_else(|| Session::new(owner, client, protocol, now)))
    }

    fn hold(
        &self,
        id: &str,
        session: &Session,
        key: &str,
        value: &str,
        now: u64,
    ) -> anyhow::Result<()> {
        let mut s = self.read(id).unwrap_or_else(|| session.clone());
        s.held.insert(key.to_string(), value.to_string());
        s.used_ms = now;
        self.put(id, &s)?;
        self.used.forget(id);
        Ok(())
    }

    fn close(&self, id: &str) -> anyhow::Result<()> {
        self.used.forget(id);
        self.records.cease(&key(id))
    }
}

/// What `/v1/mcp` holds in memory: the change signal, the runner that
/// starts workers' parts, the memory backbone, and the sessions' noted use.
pub struct McpState {
    changed: watch::Sender<u64>,
    parts: Arc<Parts>,
    memory: Arc<Memory>,
    used: Arc<Used>,
}

impl Default for McpState {
    fn default() -> Self {
        Self {
            changed: watch::channel(0).0,
            parts: Arc::default(),
            memory: Arc::default(),
            used: Arc::default(),
        }
    }
}

impl McpState {
    /// The sessions' use as noted in memory, for each [`RecordStore`].
    pub fn used(&self) -> &Arc<Used> {
        &self.used
    }

    /// The memory backbone the `journal_*` tools read and write.
    pub fn memory(&self) -> &Arc<Memory> {
        &self.memory
    }

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

    fn store() -> RecordStore<MemRecords> {
        RecordStore::new(MemRecords::default(), Arc::default())
    }

    #[test]
    fn a_session_records_its_owner_and_its_attached_thread() {
        let r = store();
        let id = r.open("o", Client::Other, "2025-11-25", 10).unwrap();
        let s = r.get(&id, 11).unwrap();
        assert_eq!(s.held(ATTACHED), None);
        r.hold(&id, &s, ATTACHED, "review", 12).unwrap();
        assert_eq!(r.get(&id, 13).unwrap().held(ATTACHED), Some("review"));
        assert_eq!(r.get(&id, 13).unwrap().owner, "o");
        r.close(&id).unwrap();
        assert!(r.get(&id, 14).is_none());
    }

    /// **Each session of the aggregator in front has its own attached
    /// thread**, though they are one person on one client.
    #[test]
    fn sessions_on_behalf_of_two_aggregator_sessions_attach_apart() {
        let r = store();
        let a = stateless_key("o", Client::Other, Some("s1"));
        let b = stateless_key("o", Client::Other, Some("s2"));
        let sa = r.ensure(&a, "o", Client::Other, "2026-07-28", 10).unwrap();
        let sb = r.ensure(&b, "o", Client::Other, "2026-07-28", 10).unwrap();
        r.hold(&a, &sa, ATTACHED, "review", 11).unwrap();
        r.hold(&b, &sb, ATTACHED, "bc-wearables", 11).unwrap();
        let again = r.ensure(&a, "o", Client::Other, "2026-07-28", 12).unwrap();
        assert_eq!(again.held(ATTACHED), Some("review"));
        assert_eq!(r.get(&b, 12).unwrap().held(ATTACHED), Some("bc-wearables"));
    }

    /// **A request only reads**: a stateless session holding nothing is
    /// never written, and a use is noted, not recorded — until a tend.
    #[test]
    fn a_request_writes_no_record() {
        let r = store();
        let id = stateless_key("o", Client::Other, Some("s1"));
        let s = r.ensure(&id, "o", Client::Other, "2026-07-28", 10).unwrap();
        assert_eq!(s, Session::new("o", Client::Other, "2026-07-28", 10));
        assert!(
            r.records.under(PREFIX).is_empty(),
            "nothing held, nothing kept"
        );
        r.hold(&id, &s, ATTACHED, "review", 11).unwrap();
        let later = 11 + TOUCH_EVERY_MS + 1;
        let used = r.get(&id, later).unwrap();
        assert_eq!(used.used_ms, later, "the call sees itself as the last use");
        let recorded: Session = serde_json::from_value(r.records.get(&key(&id)).unwrap()).unwrap();
        assert_eq!(recorded.used_ms, 11, "the record is not written by a read");
        assert_eq!(r.tend(later + 1), (1, 0));
        let recorded: Session = serde_json::from_value(r.records.get(&key(&id)).unwrap()).unwrap();
        assert_eq!(recorded.used_ms, later, "a tend writes the noted use in");
        assert_eq!(r.tend(later + 2), (0, 0), "and only once");
    }

    /// **A use within the hour is not written at all**: the record already
    /// says the session is in use.
    #[test]
    fn a_use_within_the_hour_is_noted_nowhere() {
        let r = store();
        let id = r.open("o", Client::Other, "2025-11-25", 10).unwrap();
        r.get(&id, 10 + TOUCH_EVERY_MS).unwrap();
        assert_eq!(r.tend(10 + TOUCH_EVERY_MS), (0, 0));
    }

    #[test]
    fn a_session_unused_for_a_month_is_swept_by_a_tend() {
        let r = store();
        let old = r.open("o", Client::Other, "2025-11-25", 0).unwrap();
        let new = r
            .open("o", Client::Other, "2025-11-25", SESSION_IDLE_MS + 1)
            .unwrap();
        assert_eq!(r.records.under(PREFIX).len(), 2, "opening sweeps nothing");
        assert_eq!(r.tend(SESSION_IDLE_MS + 2), (0, 1));
        assert!(r.get(&old, SESSION_IDLE_MS + 2).is_none());
        assert!(r.get(&new, SESSION_IDLE_MS + 2).is_some());
    }

    #[tokio::test]
    async fn a_waiter_wakes_on_a_bump() {
        let m = McpState::default();
        let mut rx = m.watch();
        m.bump();
        rx.changed().await.unwrap();
    }
}

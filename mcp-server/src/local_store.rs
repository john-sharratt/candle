//! Sessions kept by one process: in memory, and — for a server that must keep
//! them across a restart — in a file, written whole on each change and
//! replaced by rename, so a write cut off leaves the last one.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::Context;
use web::auth::oidc::random_token;

use crate::client::Client;
use crate::session::{Session, Store, TOUCH_EVERY_MS};

/// Every session, and the file they are kept in when they are kept.
pub struct LocalStore {
    file: Option<PathBuf>,
    held: Mutex<BTreeMap<String, Session>>,
}

impl LocalStore {
    /// Sessions held in memory alone: a restart forgets them, and a client
    /// opens another.
    pub fn in_memory() -> Self {
        Self {
            file: None,
            held: Mutex::new(BTreeMap::new()),
        }
    }

    /// The sessions kept in `dir`, created when there are none.
    pub fn in_dir(dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let file = dir.join("sessions.json");
        let held = match fs::read(&file) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("reading {}", file.display()))?,
            Err(e) if e.kind() == ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", file.display())),
        };
        Ok(Self {
            file: Some(file),
            held: Mutex::new(held),
        })
    }

    fn keep(&self, held: &BTreeMap<String, Session>) -> anyhow::Result<()> {
        let Some(file) = &self.file else {
            return Ok(());
        };
        let fresh = file.with_extension("json.new");
        fs::write(&fresh, serde_json::to_vec(held)?)
            .with_context(|| format!("writing {}", fresh.display()))?;
        fs::rename(&fresh, file).with_context(|| format!("replacing {}", file.display()))
    }

    fn change<T>(&self, f: impl FnOnce(&mut BTreeMap<String, Session>) -> T) -> anyhow::Result<T> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let out = f(&mut held);
        self.keep(&held)?;
        Ok(out)
    }
}

impl Store for LocalStore {
    fn open(
        &self,
        owner: &str,
        client: Client,
        protocol: &str,
        now: u64,
    ) -> anyhow::Result<String> {
        let id = random_token();
        let session = Session::new(owner, client, protocol, now);
        self.change(|held| {
            held.retain(|_, s| !s.stale(now));
            held.insert(id.clone(), session);
        })?;
        Ok(id)
    }

    fn get(&self, id: &str, now: u64) -> Option<Session> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        let s = held.get_mut(id)?;
        if now.saturating_sub(s.used_ms) > TOUCH_EVERY_MS {
            s.used_ms = now;
            let out = s.clone();
            if let Err(e) = self.keep(&held) {
                tracing::warn!("mcp: a session's use could not be kept: {e:#}");
            }
            return Some(out);
        }
        Some(s.clone())
    }

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
        self.change(|held| {
            held.retain(|_, s| !s.stale(now));
            held.entry(id.to_string())
                .or_insert_with(|| Session::new(owner, client, protocol, now))
                .clone()
        })
    }

    fn hold(
        &self,
        id: &str,
        session: &Session,
        key: &str,
        value: &str,
        now: u64,
    ) -> anyhow::Result<()> {
        self.change(|held| {
            let s = held
                .entry(id.to_string())
                .or_insert_with(|| session.clone());
            s.held.insert(key.to_string(), value.to_string());
            s.used_ms = now;
        })
    }

    fn close(&self, id: &str) -> anyhow::Result<()> {
        self.change(|held| {
            held.remove(id);
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use crate::session::{stateless_key, SESSION_IDLE_MS};

    #[test]
    fn a_session_records_its_owner_and_what_it_holds_and_outlives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let s = LocalStore::in_dir(dir.path()).unwrap();
        let id = s.open("o", Client::Other, "2025-11-25", 10).unwrap();
        let session = s.get(&id, 11).unwrap();
        assert_eq!(session.held("attached"), None);
        s.hold(&id, &session, "attached", "review", 12).unwrap();
        let again = LocalStore::in_dir(dir.path()).unwrap();
        assert_eq!(again.get(&id, 13).unwrap().held("attached"), Some("review"));
        again.close(&id).unwrap();
        assert!(again.get(&id, 14).is_none());
    }

    #[test]
    fn a_session_in_memory_is_gone_with_its_store() {
        let s = LocalStore::in_memory();
        let id = s.open("o", Client::Other, "2025-11-25", 10).unwrap();
        assert!(s.get(&id, 11).is_some());
        assert!(LocalStore::in_memory().get(&id, 11).is_none());
    }

    #[test]
    fn a_stateless_clients_session_outlives_its_tokens() {
        let s = LocalStore::in_memory();
        let id = stateless_key("o", Client::ClaudeCode, None);
        let session = s
            .ensure(&id, "o", Client::ClaudeCode, "2026-07-28", 10)
            .unwrap();
        s.hold(&id, &session, "attached", "review", 11).unwrap();
        let again = s
            .ensure(&id, "o", Client::ClaudeCode, "2026-07-28", 12)
            .unwrap();
        assert_eq!(again.held("attached"), Some("review"));
    }

    /// **First calls at once share one session**: whichever opens it, every
    /// other finds it, so what one holds on it is not lost to another.
    #[test]
    fn first_calls_at_once_share_one_session() {
        let s = Arc::new(LocalStore::in_memory());
        let id = stateless_key("o", Client::ClaudeCode, None);
        let racers: Vec<_> = (0..8)
            .map(|_| {
                let (s, id) = (s.clone(), id.clone());
                thread::spawn(move || {
                    s.ensure(&id, "o", Client::ClaudeCode, "2026-07-28", 10)
                        .unwrap();
                })
            })
            .collect();
        let session = s
            .ensure(&id, "o", Client::ClaudeCode, "2026-07-28", 10)
            .unwrap();
        s.hold(&id, &session, "attached", "review", 11).unwrap();
        for r in racers {
            r.join().unwrap();
        }
        assert_eq!(s.get(&id, 12).unwrap().held("attached"), Some("review"));
    }

    #[test]
    fn a_session_unused_for_a_month_is_swept_by_the_next_open() {
        let s = LocalStore::in_memory();
        let old = s.open("o", Client::Other, "2025-11-25", 0).unwrap();
        s.open("o", Client::Other, "2025-11-25", SESSION_IDLE_MS + 1)
            .unwrap();
        assert!(s.get(&old, SESSION_IDLE_MS + 2).is_none());
    }
}

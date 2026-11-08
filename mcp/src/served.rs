//! Which listing each session was served (`docs/mcp_app.md` §3.1).
//!
//! A client keeps the tools it was given, across chats: claude.ai still had
//! `cluster_logs` and no `file_*` or `log_*` tools days after a release
//! replaced them. Every backend names its current listing on every answer
//! (`mcp_server::listing`), so a call answered on a listing other than the one
//! its session was served owes the client `notifications/tools/list_changed`,
//! told on that call's stream before its result.
//!
//! Held in memory alone. A session this process never served — opened before
//! a restart, or served by the other aggregator — is owed the notice on its
//! first call here: what it holds is unknown, and listing again costs a client
//! one request.

use std::collections::HashMap;
use std::mem;
use std::sync::Mutex;

/// The most sessions remembered; past it, the idlest is let go.
const KEPT: usize = 4096;

#[derive(Default)]
struct Entry {
    /// The listing version served for each backend, by index; `None` where
    /// the backend could not be listed.
    versions: Vec<Option<String>>,
    /// Whether the session's next streamed call tells of a changed list.
    owed: bool,
    at_ms: u64,
}

#[derive(Default)]
pub struct Served(Mutex<HashMap<String, Entry>>);

impl Served {
    /// `session` was just served `versions`, one per backend.
    pub fn listed(&self, session: &str, versions: Vec<Option<String>>, now_ms: u64) {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let entry = all.entry(session.to_string()).or_default();
        entry.versions = versions;
        entry.owed = false;
        entry.at_ms = now_ms;
        let_go_idlest(&mut all);
    }

    /// `backend` answered a call of `session`'s on listing `version`.
    pub fn answered(&self, session: &str, backend: usize, version: &str, now_ms: u64) {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let known = all.contains_key(session);
        let entry = all.entry(session.to_string()).or_default();
        entry.at_ms = now_ms;
        if entry.versions.len() <= backend {
            entry.versions.resize(backend + 1, None);
        }
        if entry.versions[backend].as_deref() != Some(version) {
            entry.versions[backend] = Some(version.to_string());
            entry.owed = true;
        }
        if !known {
            entry.owed = true;
        }
        let_go_idlest(&mut all);
    }

    /// Whether `session` is owed the notice; asking pays it.
    pub fn take_owed(&self, session: &str) -> bool {
        let mut all = self.0.lock().unwrap_or_else(|e| e.into_inner());
        all.get_mut(session)
            .is_some_and(|e| mem::replace(&mut e.owed, false))
    }
}

fn let_go_idlest(all: &mut HashMap<String, Entry>) {
    if all.len() <= KEPT {
        return;
    }
    if let Some(idlest) = all
        .iter()
        .min_by_key(|(_, e)| e.at_ms)
        .map(|(k, _)| k.clone())
    {
        all.remove(&idlest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Option<String> {
        Some(s.to_string())
    }

    /// **A call on the listing served owes nothing; one on another owes the
    /// notice, once.**
    #[test]
    fn a_changed_listing_is_owed_once() {
        let s = Served::default();
        s.listed("a", vec![v("z1"), v("n1")], 1);
        s.answered("a", 0, "z1", 2);
        assert!(!s.take_owed("a"));
        s.answered("a", 0, "z2", 3);
        assert!(s.take_owed("a"));
        assert!(!s.take_owed("a"), "paid");
        s.answered("a", 0, "z2", 4);
        assert!(!s.take_owed("a"), "the new listing is now the one it is owed against");
    }

    /// **A session this process never served is owed the notice**, and
    /// listing pays it.
    #[test]
    fn a_session_never_served_here_is_owed_until_it_lists() {
        let s = Served::default();
        s.answered("b", 1, "n1", 1);
        assert!(s.take_owed("b"));
        s.answered("c", 0, "z1", 1);
        s.listed("c", vec![v("z1")], 2);
        assert!(!s.take_owed("c"));
    }

    /// **A backend that could not be listed is owed against once it answers.**
    #[test]
    fn a_backend_not_listed_is_owed_when_it_answers() {
        let s = Served::default();
        s.listed("a", vec![v("z1"), None], 1);
        s.answered("a", 1, "n1", 2);
        assert!(s.take_owed("a"));
        assert!(!s.take_owed("nobody"));
    }

    /// **Past its bound, the idlest session is let go.**
    #[test]
    fn the_idlest_session_is_let_go() {
        let s = Served::default();
        for i in 0..=KEPT {
            s.listed(&format!("s{i}"), vec![v("z")], i as u64 + 1);
        }
        let all = s.0.lock().unwrap();
        assert_eq!(all.len(), KEPT);
        assert!(!all.contains_key("s0"));
        assert!(all.contains_key(&format!("s{KEPT}")));
    }
}

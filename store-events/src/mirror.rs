//! What the substrate holds of a set of stores' files, and what to write so
//! it holds what the stores hold now (`docs/zend_vfs_events.md` §4).
//!
//! The mirror is exactly what the live events say: per repository, its base
//! and, per path, the chain the events make, with the name of every live
//! event behind each. [`Mirror::diff`] compares it with the stores'
//! snapshots and says what to write; [`Mirror::apply`] records what was
//! written, by the same rules replay reads events with ([`absorb`], [`kill`])
//! — so the mirror after a write is the mirror a replay of the log would
//! build.
//!
//! **A path's state event commits it.** Every write to a path ends with its
//! state, and deltas read since the last state are held apart
//! ([`Pending`]) until one arrives. A save cut short — the log keeps every
//! record before a crash and none after — leaves deltas no state followed;
//! [`RepoMirror::settle`] drops them, so the path reads as its last
//! committed chain, and marks it [`torn`](PathMirror::torn) so the next save
//! writes it whole again rather than appending after what was dropped.

use std::collections::BTreeMap;

use candle_conversation::persistence::vfs::{VfsAppend, VfsEventId, VfsKill, VfsWrite};
use zend_vfs::{SavedBase, SavedChain, Snapshot, TimedDelta};

use crate::event::{FileEvent, BASE_KEY};

/// Deltas read since a path's last state, waiting for the state that
/// commits them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pending {
    /// Whether they begin the chain again (`start`) rather than extend it.
    pub restart: bool,
    pub deltas: Vec<TimedDelta>,
}

/// One path as the substrate holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PathMirror {
    /// Every live event of the path — including any a later `start`
    /// superseded, or a torn save left uncommitted, whose tombstone has not
    /// landed.
    pub ids: Vec<VfsEventId>,
    /// The committed chain: what the last state committed.
    pub deltas: Vec<TimedDelta>,
    pub size: Option<usize>,
    pub conflict: bool,
    /// Whether a state has committed a chain for the path.
    pub stated: bool,
    /// Deltas no state has committed yet.
    pub pending: Option<Pending>,
    /// Whether some of `ids` are deltas a torn save left uncommitted. An
    /// append after them would be read as following them, so the path is
    /// written whole, `start`ed, next time.
    pub torn: bool,
}

/// One repository as the substrate holds it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepoMirror {
    /// The latest base event's base.
    pub(crate) base: Option<SavedBase>,
    /// Every live base event.
    pub(crate) base_ids: Vec<VfsEventId>,
    pub(crate) paths: BTreeMap<String, PathMirror>,
}

impl RepoMirror {
    fn is_empty(&self) -> bool {
        self.base_ids.is_empty() && self.paths.is_empty()
    }

    /// End a replay: every path's deltas that no state committed are what a
    /// torn save left, and are dropped — the path keeps its committed chain,
    /// marked [`torn`](PathMirror::torn) while their events stay live.
    pub(crate) fn settle(&mut self) {
        for at in self.paths.values_mut() {
            if at.pending.take().is_some() {
                at.torn = true;
            }
        }
    }
}

/// Read one event into `repo`: a base replaces the last; a delta waits for
/// the state that commits it, extending its path's chain or, with `start`,
/// beginning it again; a state commits what waits and sets the chain's
/// state — or, with `start`, a chain with no delta at all. `Err` for an
/// event under a key it cannot be under.
pub(crate) fn absorb(
    repo: &mut RepoMirror,
    key: &str,
    id: VfsEventId,
    event: FileEvent,
) -> Result<(), String> {
    match (key, event) {
        (BASE_KEY, FileEvent::Base { base }) => {
            repo.base = Some(base);
            repo.base_ids.push(id);
        }
        (BASE_KEY, _) | (_, FileEvent::Base { .. }) => {
            return Err(format!(
                "event {:x}/{} is under the wrong key {key:?}",
                id.origin, id.seq
            ));
        }
        (path, FileEvent::Delta { delta, start }) => {
            let at = path_of(repo, path);
            at.ids.push(id);
            if start {
                // A chain begun again supersedes whatever waited, torn or not.
                at.pending = Some(Pending {
                    restart: true,
                    deltas: vec![delta],
                });
            } else {
                at.pending
                    .get_or_insert_with(|| Pending {
                        restart: false,
                        deltas: Vec::new(),
                    })
                    .deltas
                    .push(delta);
            }
        }
        (
            path,
            FileEvent::State {
                size,
                conflict,
                start,
            },
        ) => {
            let at = path_of(repo, path);
            at.ids.push(id);
            let pending = at.pending.take();
            if start {
                at.deltas.clear();
                at.torn = false;
            } else if let Some(Pending { restart, deltas }) = pending {
                if restart {
                    at.deltas = deltas;
                    at.torn = false;
                } else {
                    at.deltas.extend(deltas);
                }
            }
            at.size = size;
            at.conflict = conflict;
            at.stated = true;
        }
    }
    Ok(())
}

/// Kill `kills` of `key` in `repo`: they are no longer live. A path, or the
/// base, left with no live event is gone.
pub(crate) fn kill(repo: &mut RepoMirror, key: &str, kills: &[VfsEventId]) {
    if key == BASE_KEY {
        repo.base_ids.retain(|s| !kills.contains(s));
        if repo.base_ids.is_empty() {
            repo.base = None;
        }
        return;
    }
    if let Some(at) = repo.paths.get_mut(key) {
        at.ids.retain(|s| !kills.contains(s));
        if at.ids.is_empty() {
            repo.paths.remove(key);
        }
    }
}

fn path_of<'m>(repo: &'m mut RepoMirror, path: &str) -> &'m mut PathMirror {
    repo.paths
        .entry(path.to_string())
        .or_insert_with(|| PathMirror {
            ids: Vec::new(),
            deltas: Vec::new(),
            size: None,
            conflict: false,
            stated: false,
            pending: None,
            torn: false,
        })
}

/// A set of stores' files as the substrate holds them.
///
/// **Unread** until its events have been read back: a mirror that does not
/// know what the substrate holds cannot say what to write — anything it
/// wrote would sit beside events it never tombstoned, and a later replay
/// would read both. So an unread mirror writes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mirror {
    read: bool,
    pub(crate) repos: BTreeMap<String, RepoMirror>,
}

impl Mirror {
    /// A mirror whose events have not been read back yet.
    pub fn unread() -> Self {
        Self {
            read: false,
            repos: BTreeMap::new(),
        }
    }

    /// A mirror of what was read back.
    pub fn of(repos: BTreeMap<String, RepoMirror>) -> Self {
        Self { read: true, repos }
    }

    /// Whether the mirror knows what the substrate holds.
    pub fn is_read(&self) -> bool {
        self.read
    }

    /// What to write so the substrate holds `snapshots` — every repository
    /// with changes, by name. Nothing for an unread mirror.
    pub fn diff(&self, snapshots: &BTreeMap<String, Snapshot>) -> VfsWrite {
        let mut write = VfsWrite::default();
        if !self.read {
            return write;
        }
        let empty = RepoMirror::default();
        let mut names: Vec<&String> = self.repos.keys().chain(snapshots.keys()).collect();
        names.sort();
        names.dedup();
        for repo in names {
            let held = self.repos.get(repo).unwrap_or(&empty);
            match snapshots.get(repo) {
                Some(snapshot) => diff_repo(repo, held, snapshot, &mut write),
                None => forget_repo(repo, held, &mut write),
            }
        }
        write
    }

    /// Record `write` as written — its events having taken `ids` in order,
    /// then its tombstones landing — exactly as a replay would read it.
    pub fn apply(&mut self, write: &VfsWrite, ids: &[VfsEventId]) {
        for (append, &id) in write.events.iter().zip(ids) {
            if let Ok(event) = FileEvent::from_value(&append.body) {
                let repo = self.repos.entry(append.repo.clone()).or_default();
                let _ = absorb(repo, &append.key, id, event);
            }
        }
        for tomb in &write.tombstones {
            if let Some(repo) = self.repos.get_mut(&tomb.repo) {
                kill(repo, &tomb.key, &tomb.kills);
            }
        }
        self.repos.retain(|_, r| !r.is_empty());
    }
}

/// What to write so `repo`, held as `held`, holds `snapshot`.
fn diff_repo(repo: &str, held: &RepoMirror, snapshot: &Snapshot, write: &mut VfsWrite) {
    if snapshot.base() != held.base.as_ref() {
        if let Some(base) = snapshot.base() {
            append(
                write,
                repo,
                BASE_KEY,
                FileEvent::Base { base: base.clone() },
            );
        }
        if !held.base_ids.is_empty() {
            tomb(write, repo, BASE_KEY, held.base_ids.clone());
        }
    }
    for (path, chain) in snapshot.chains() {
        match held.paths.get(path) {
            Some(mirrored)
                if mirrored.stated
                    && !mirrored.torn
                    && chain.deltas.starts_with(&mirrored.deltas) =>
            {
                grow(write, repo, path, mirrored, chain)
            }
            Some(mirrored) => {
                write_whole(write, repo, path, chain);
                tomb(write, repo, path, mirrored.ids.clone());
            }
            None => write_whole(write, repo, path, chain),
        }
    }
    // Each path the store no longer holds: committed, discarded, carried away.
    for (path, mirrored) in &held.paths {
        if !snapshot.chains().contains_key(path) {
            tomb(write, repo, path, mirrored.ids.clone());
        }
    }
}

/// `repo` has no changes any more: everything it held goes.
fn forget_repo(repo: &str, held: &RepoMirror, write: &mut VfsWrite) {
    if !held.base_ids.is_empty() {
        tomb(write, repo, BASE_KEY, held.base_ids.clone());
    }
    for (path, mirrored) in &held.paths {
        tomb(write, repo, path, mirrored.ids.clone());
    }
}

/// A chain that only grew: its new deltas, then its state — which commits
/// them, so it is written after any delta whether or not the state changed.
fn grow(write: &mut VfsWrite, repo: &str, path: &str, held: &PathMirror, chain: &SavedChain) {
    let new = &chain.deltas[held.deltas.len()..];
    for delta in new {
        let event = FileEvent::Delta {
            delta: delta.clone(),
            start: false,
        };
        append(write, repo, path, event);
    }
    if !new.is_empty() || (held.size, held.conflict) != (chain.size, chain.conflict) {
        append(write, repo, path, state(chain, false));
    }
}

/// A chain written from nothing: its first event starts it, then every
/// delta, then its state.
fn write_whole(write: &mut VfsWrite, repo: &str, path: &str, chain: &SavedChain) {
    for (i, delta) in chain.deltas.iter().enumerate() {
        let event = FileEvent::Delta {
            delta: delta.clone(),
            start: i == 0,
        };
        append(write, repo, path, event);
    }
    append(write, repo, path, state(chain, chain.deltas.is_empty()));
}

fn state(chain: &SavedChain, start: bool) -> FileEvent {
    FileEvent::State {
        size: chain.size,
        conflict: chain.conflict,
        start,
    }
}

fn append(write: &mut VfsWrite, repo: &str, key: &str, event: FileEvent) {
    write.events.push(VfsAppend {
        repo: repo.to_string(),
        key: key.to_string(),
        body: event.to_value(),
    });
}

fn tomb(write: &mut VfsWrite, repo: &str, key: &str, kills: Vec<VfsEventId>) {
    write.tombstones.push(VfsKill {
        repo: repo.to_string(),
        key: key.to_string(),
        kills,
    });
}

#[cfg(test)]
mod tests;

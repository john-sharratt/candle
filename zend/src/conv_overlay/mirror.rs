//! What the substrate holds of a conversation's files, and what to write so
//! it holds what the file stores hold now (`docs/zend_vfs_events.md` §4).
//!
//! The mirror is exactly what the conversation's live events say: per
//! repository, its base and, per path, the chain the events make, with the
//! sequence number of every live event behind each. [`Mirror::diff`]
//! compares it with the stores' snapshots and says what to write;
//! [`Mirror::apply`] records what was written, by the same rules replay reads
//! events with ([`absorb`], [`kill`]) — so the mirror after a write is the
//! mirror a replay of the log would build.
//!
//! **A path's state event commits it.** Every write to a path ends with its
//! state, and deltas read since the last state are held apart
//! ([`Pending`]) until one arrives. A save cut short — the log keeps every
//! record before a crash and none after — leaves deltas no state followed;
//! [`RepoMirror::settle`] drops them, so the path reads as its last
//! committed chain, and marks it [`torn`](PathMirror::torn) so the next save
//! writes it whole again rather than appending after what was dropped.

use std::collections::BTreeMap;

use candle_conversation::persistence::vfs::{VfsAppend, VfsKill, VfsWrite};
use zend_vfs::{SavedBase, SavedChain, Snapshot, TimedDelta};

use super::event::{FileEvent, BASE_KEY};

/// Deltas read since a path's last state, waiting for the state that
/// commits them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Pending {
    /// Whether they begin the chain again (`start`) rather than extend it.
    pub restart: bool,
    pub deltas: Vec<TimedDelta>,
}

/// One path as the substrate holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PathMirror {
    /// Every live event of the path — including any a later `start`
    /// superseded, or a torn save left uncommitted, whose tombstone has not
    /// landed.
    pub seqs: Vec<u64>,
    /// The committed chain: what the last state committed.
    pub deltas: Vec<TimedDelta>,
    pub size: Option<usize>,
    pub conflict: bool,
    /// Whether a state has committed a chain for the path.
    pub stated: bool,
    /// Deltas no state has committed yet.
    pub pending: Option<Pending>,
    /// Whether some of `seqs` are deltas a torn save left uncommitted. An
    /// append after them would be read as following them, so the path is
    /// written whole, `start`ed, next time.
    pub torn: bool,
}

/// One repository as the substrate holds it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct RepoMirror {
    /// The latest base event's base.
    pub base: Option<SavedBase>,
    /// Every live base event.
    pub base_seqs: Vec<u64>,
    pub paths: BTreeMap<String, PathMirror>,
}

impl RepoMirror {
    fn is_empty(&self) -> bool {
        self.base_seqs.is_empty() && self.paths.is_empty()
    }

    /// End a replay: every path's deltas that no state committed are what a
    /// torn save left, and are dropped — the path keeps its committed chain,
    /// marked [`torn`](PathMirror::torn) while their events stay live.
    pub(super) fn settle(&mut self) {
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
pub(super) fn absorb(
    repo: &mut RepoMirror,
    key: &str,
    seq: u64,
    event: FileEvent,
) -> Result<(), String> {
    match (key, event) {
        (BASE_KEY, FileEvent::Base { base }) => {
            repo.base = Some(base);
            repo.base_seqs.push(seq);
        }
        (BASE_KEY, _) | (_, FileEvent::Base { .. }) => {
            return Err(format!("event {seq} is under the wrong key {key:?}"));
        }
        (path, FileEvent::Delta { delta, start }) => {
            let at = path_of(repo, path);
            at.seqs.push(seq);
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
            at.seqs.push(seq);
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
pub(super) fn kill(repo: &mut RepoMirror, key: &str, kills: &[u64]) {
    if key == BASE_KEY {
        repo.base_seqs.retain(|s| !kills.contains(s));
        if repo.base_seqs.is_empty() {
            repo.base = None;
        }
        return;
    }
    if let Some(at) = repo.paths.get_mut(key) {
        at.seqs.retain(|s| !kills.contains(s));
        if at.seqs.is_empty() {
            repo.paths.remove(key);
        }
    }
}

fn path_of<'m>(repo: &'m mut RepoMirror, path: &str) -> &'m mut PathMirror {
    repo.paths
        .entry(path.to_string())
        .or_insert_with(|| PathMirror {
            seqs: Vec::new(),
            deltas: Vec::new(),
            size: None,
            conflict: false,
            stated: false,
            pending: None,
            torn: false,
        })
}

/// A conversation's files as the substrate holds them.
///
/// **Unread** until its events have been read back: a mirror that does not
/// know what the substrate holds cannot say what to write — anything it
/// wrote would sit beside events it never tombstoned, and a later replay
/// would read both. So an unread mirror writes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mirror {
    read: bool,
    pub(super) repos: BTreeMap<String, RepoMirror>,
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
    pub(super) fn of(repos: BTreeMap<String, RepoMirror>) -> Self {
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

    /// Record `write` as written — its events having taken `seqs` in order,
    /// then its tombstones landing — exactly as a replay would read it.
    pub fn apply(&mut self, write: &VfsWrite, seqs: &[u64]) {
        for (append, &seq) in write.events.iter().zip(seqs) {
            if let Ok(event) = FileEvent::from_value(&append.body) {
                let repo = self.repos.entry(append.repo.clone()).or_default();
                let _ = absorb(repo, &append.key, seq, event);
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
        if !held.base_seqs.is_empty() {
            tomb(write, repo, BASE_KEY, held.base_seqs.clone());
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
                tomb(write, repo, path, mirrored.seqs.clone());
            }
            None => write_whole(write, repo, path, chain),
        }
    }
    // Each path the store no longer holds: committed, discarded, carried away.
    for (path, mirrored) in &held.paths {
        if !snapshot.chains().contains_key(path) {
            tomb(write, repo, path, mirrored.seqs.clone());
        }
    }
}

/// `repo` has no changes any more: everything it held goes.
fn forget_repo(repo: &str, held: &RepoMirror, write: &mut VfsWrite) {
    if !held.base_seqs.is_empty() {
        tomb(write, repo, BASE_KEY, held.base_seqs.clone());
    }
    for (path, mirrored) in &held.paths {
        tomb(write, repo, path, mirrored.seqs.clone());
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

fn tomb(write: &mut VfsWrite, repo: &str, key: &str, kills: Vec<u64>) {
    write.tombstones.push(VfsKill {
        repo: repo.to_string(),
        key: key.to_string(),
        kills,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zend_vfs::FileDelta;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    fn base(commit: &str) -> SavedBase {
        serde_json::from_value(json!({"tree": B, "parents": [commit]})).unwrap()
    }

    fn replace(at_ns: i64, content: &str) -> TimedDelta {
        TimedDelta {
            at_ns,
            delta: FileDelta::Replace {
                content: content.into(),
            },
        }
    }

    fn chain(deltas: Vec<TimedDelta>, size: Option<usize>, conflict: bool) -> SavedChain {
        SavedChain {
            deltas,
            size,
            conflict,
        }
    }

    fn snapshot(commit: &str, chains: Vec<(&str, SavedChain)>) -> Snapshot {
        Snapshot::new(
            Some(base(commit)),
            chains
                .into_iter()
                .map(|(p, c)| (p.to_string(), c))
                .collect(),
        )
    }

    fn one(repo: &str, s: Snapshot) -> BTreeMap<String, Snapshot> {
        [(repo.to_string(), s)].into()
    }

    type Shape = (Vec<(String, serde_json::Value)>, Vec<(String, Vec<u64>)>);

    /// `(key, body)` of every event, and `(key, kills)` of every tombstone.
    fn shape(write: &VfsWrite) -> Shape {
        (
            write
                .events
                .iter()
                .map(|e| (e.key.clone(), e.body.clone()))
                .collect(),
            write
                .tombstones
                .iter()
                .map(|t| (t.key.clone(), t.kills.clone()))
                .collect(),
        )
    }

    /// Write `write` into `mirror`, its events taking sequence numbers from
    /// `next`.
    fn land(mirror: &mut Mirror, write: &VfsWrite, next: &mut u64) {
        let seqs: Vec<u64> = (0..write.events.len() as u64).map(|i| *next + i).collect();
        *next += write.events.len() as u64 + write.tombstones.len() as u64;
        mirror.apply(write, &seqs);
    }

    fn delta_json(at_ns: i64, content: &str, start: bool) -> serde_json::Value {
        let mut v = json!({"kind": "delta", "delta": {"at_ns": at_ns, "kind": "replace", "content": content}});
        if start {
            v["start"] = json!(true);
        }
        v
    }

    /// **An unread mirror writes nothing**, whatever the stores hold.
    #[test]
    fn an_unread_mirror_writes_nothing() {
        let s = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
            ),
        );
        assert!(Mirror::unread().diff(&s).is_empty());
    }

    /// **A new repository writes its base, and each path a started chain
    /// then its state**; written again, nothing.
    #[test]
    fn a_new_repository_is_written_whole_once() {
        let s = one(
            "candle",
            snapshot(
                A,
                vec![(
                    "a.rs",
                    chain(vec![replace(1, "x"), replace(2, "yz")], Some(2), false),
                )],
            ),
        );
        let mut mirror = Mirror::of(BTreeMap::new());
        let write = mirror.diff(&s);
        let (events, tombstones) = shape(&write);
        assert!(tombstones.is_empty());
        assert_eq!(
            events,
            vec![
                (
                    "".into(),
                    json!({"kind": "base", "base": {"tree": B, "parents": [A]}})
                ),
                ("a.rs".into(), delta_json(1, "x", true)),
                ("a.rs".into(), delta_json(2, "yz", false)),
                (
                    "a.rs".into(),
                    json!({"kind": "state", "size": 2, "conflict": false})
                ),
            ]
        );
        land(&mut mirror, &write, &mut 0);
        assert!(mirror.diff(&s).is_empty(), "nothing changed");
    }

    /// **A chain that only grew writes its new deltas, and its state when
    /// that changed.**
    #[test]
    fn a_grown_chain_appends() {
        let before = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
            ),
        );
        let mut mirror = Mirror::of(BTreeMap::new());
        let first = mirror.diff(&before);
        land(&mut mirror, &first, &mut 0);
        let after = one(
            "candle",
            snapshot(
                A,
                vec![(
                    "a.rs",
                    chain(vec![replace(1, "x"), replace(2, "xy")], Some(2), false),
                )],
            ),
        );
        let (events, tombstones) = shape(&mirror.diff(&after));
        assert!(tombstones.is_empty());
        assert_eq!(
            events,
            vec![
                ("a.rs".into(), delta_json(2, "xy", false)),
                (
                    "a.rs".into(),
                    json!({"kind": "state", "size": 2, "conflict": false})
                ),
            ]
        );
        let flagged = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), true))],
            ),
        );
        let (events, _) = shape(&mirror.diff(&flagged));
        assert_eq!(
            events,
            vec![(
                "a.rs".into(),
                json!({"kind": "state", "size": 1, "conflict": true})
            )],
            "only the flag changed"
        );
    }

    /// **A grown chain always ends with its state**, even one that did not
    /// change: the state is what commits the deltas before it, so a save
    /// torn after them leaves them uncommitted rather than half-applied.
    #[test]
    fn a_grown_chain_always_ends_with_its_state() {
        let before = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
            ),
        );
        let mut mirror = Mirror::of(BTreeMap::new());
        let mut next = 0;
        let first = mirror.diff(&before);
        land(&mut mirror, &first, &mut next);
        let after = one(
            "candle",
            snapshot(
                A,
                vec![(
                    "a.rs",
                    chain(vec![replace(1, "x"), replace(2, "y")], Some(1), false),
                )],
            ),
        );
        let write = mirror.diff(&after);
        let (events, tombstones) = shape(&write);
        assert!(tombstones.is_empty());
        assert_eq!(
            events,
            vec![
                ("a.rs".into(), delta_json(2, "y", false)),
                (
                    "a.rs".into(),
                    json!({"kind": "state", "size": 1, "conflict": false})
                ),
            ]
        );
        land(&mut mirror, &write, &mut next);
        assert!(mirror.diff(&after).is_empty());
    }

    /// **A path a torn save left is written whole next time**, `start`ed,
    /// and every event it had tombstoned — an append would be read as
    /// following the dropped deltas. A torn file no state ever committed is
    /// tombstoned outright.
    #[test]
    fn a_torn_path_is_written_whole_and_its_events_killed() {
        let delta = |at_ns, content, start| FileEvent::Delta {
            delta: replace(at_ns, content),
            start,
        };
        let mut repo = RepoMirror::default();
        absorb(&mut repo, BASE_KEY, 0, FileEvent::Base { base: base(A) }).unwrap();
        absorb(&mut repo, "a.rs", 1, delta(1, "x", true)).unwrap();
        let committed = FileEvent::State {
            size: Some(1),
            conflict: false,
            start: false,
        };
        absorb(&mut repo, "a.rs", 2, committed).unwrap();
        absorb(&mut repo, "a.rs", 3, delta(2, "xy", false)).unwrap();
        absorb(&mut repo, "b.rs", 4, delta(3, "b", true)).unwrap();
        repo.settle();
        assert!(repo.paths["a.rs"].torn && repo.paths["b.rs"].torn);
        let mirror = Mirror::of([("candle".to_string(), repo)].into());

        // The store restored the committed chain and grew it by the same
        // delta the torn save had begun to write.
        let s = one(
            "candle",
            snapshot(
                A,
                vec![(
                    "a.rs",
                    chain(vec![replace(1, "x"), replace(2, "xy")], Some(2), false),
                )],
            ),
        );
        let (events, tombstones) = shape(&mirror.diff(&s));
        assert_eq!(
            events,
            vec![
                ("a.rs".into(), delta_json(1, "x", true)),
                ("a.rs".into(), delta_json(2, "xy", false)),
                (
                    "a.rs".into(),
                    json!({"kind": "state", "size": 2, "conflict": false})
                ),
            ]
        );
        assert_eq!(
            tombstones,
            vec![("a.rs".into(), vec![1, 2, 3]), ("b.rs".into(), vec![4])]
        );
    }

    /// **A rewritten chain is written whole, started, and its old events
    /// tombstoned; a path the store let go of is tombstoned.** Events come
    /// before tombstones, so a batch cut short never kills a chain with
    /// nothing in its place.
    #[test]
    fn a_rewritten_or_dropped_path_is_tombstoned() {
        let before = one(
            "candle",
            snapshot(
                A,
                vec![
                    ("a.rs", chain(vec![replace(1, "x")], Some(1), false)),
                    ("b.rs", chain(vec![replace(2, "b")], Some(1), false)),
                ],
            ),
        );
        // base 0; a.rs delta 1, state 2; b.rs delta 3, state 4.
        let mut mirror = Mirror::of(BTreeMap::new());
        let mut next = 0;
        let first = mirror.diff(&before);
        land(&mut mirror, &first, &mut next);
        let after = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(9, "merged")], Some(6), true))],
            ),
        );
        let write = mirror.diff(&after);
        let (events, tombstones) = shape(&write);
        assert_eq!(
            events,
            vec![
                ("a.rs".into(), delta_json(9, "merged", true)),
                (
                    "a.rs".into(),
                    json!({"kind": "state", "size": 6, "conflict": true})
                ),
            ]
        );
        assert_eq!(
            tombstones,
            vec![("a.rs".into(), vec![1, 2]), ("b.rs".into(), vec![3, 4])]
        );
        land(&mut mirror, &write, &mut next);
        assert!(mirror.diff(&after).is_empty());
        assert_eq!(mirror.repos["candle"].paths["a.rs"].seqs, vec![5, 6]);
    }

    /// **A commit moves the base and lets go of what it took**: the new base
    /// written, the old and the committed path tombstoned.
    #[test]
    fn a_commit_moves_the_base_and_tombstones_what_it_took() {
        let before = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
            ),
        );
        let mut mirror = Mirror::of(BTreeMap::new());
        let mut next = 0;
        let first = mirror.diff(&before);
        land(&mut mirror, &first, &mut next);
        let after = one("candle", snapshot(B, vec![]));
        let write = mirror.diff(&after);
        let (events, tombstones) = shape(&write);
        assert_eq!(
            events,
            vec![(
                "".into(),
                json!({"kind": "base", "base": {"tree": B, "parents": [B]}})
            )]
        );
        assert_eq!(
            tombstones,
            vec![("".into(), vec![0]), ("a.rs".into(), vec![1, 2])]
        );
        land(&mut mirror, &write, &mut next);
        assert!(mirror.diff(&after).is_empty());
        assert_eq!(mirror.repos["candle"].base_seqs, vec![3]);
    }

    /// **A repository with no changes left is forgotten whole.**
    #[test]
    fn a_repository_gone_is_forgotten() {
        let before = one(
            "candle",
            snapshot(
                A,
                vec![("a.rs", chain(vec![replace(1, "x")], Some(1), false))],
            ),
        );
        let mut mirror = Mirror::of(BTreeMap::new());
        let mut next = 0;
        let first = mirror.diff(&before);
        land(&mut mirror, &first, &mut next);
        let write = mirror.diff(&BTreeMap::new());
        let (events, tombstones) = shape(&write);
        assert!(events.is_empty());
        assert_eq!(
            tombstones,
            vec![("".into(), vec![0]), ("a.rs".into(), vec![1, 2])]
        );
        land(&mut mirror, &write, &mut next);
        assert!(mirror.repos.is_empty());
    }

    /// **A conflict kept exactly as the base holds it — no delta at all —
    /// is written as a started state.**
    #[test]
    fn a_chain_with_no_delta_starts_with_its_state() {
        let s = one(
            "candle",
            snapshot(A, vec![("a.rs", chain(vec![], Some(3), true))]),
        );
        let (events, _) = shape(&Mirror::of(BTreeMap::new()).diff(&s));
        assert_eq!(
            events[1],
            (
                "a.rs".into(),
                json!({"kind": "state", "size": 3, "conflict": true, "start": true})
            )
        );
    }
}

//! A conversation's own copy of the workspace, kept with the conversation.
//!
//! While a conversation is open its changes to the workspace's files live in
//! its in-memory state — one overlay store per repository ([`RepoFiles`]),
//! each reading the repository through the branch the conversation works on.
//! That state does not last: a turn that ends without finishing evicts it, a
//! restart loses it, and the next request builds it again from the substrate.
//!
//! So the substrate keeps the same thing as **events** on the conversation's
//! timeline (`docs/zend_vfs_events.md`): after every tool round, [`save`]
//! writes the branch each repository is on and whatever changed in its store
//! — new deltas appended, rewritten and dropped chains tombstoned — and when
//! the conversation's state is built, [`restore`] puts each store on its
//! branch and replays its events back into it. The two mirror each other
//! without either calling the other: the [`Mirror`] beside the stores is what
//! the events say, and every save writes the difference.
//!
//! | Module | Concern |
//! |---|---|
//! | `event` | What one event says |
//! | `mirror` | What the substrate holds, and what to write |
//! | `replay` | Events back into stores |

mod event;
mod mirror;
mod replay;

pub use mirror::Mirror;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use candle_conversation::persistence::vfs::{VfsAppend, VfsEventPayload, VfsKill, VfsWrite};
use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use zend_vfs::RepoFiles;

use self::replay::{by_repo, replay_repo};

/// What a repository name carries when its events could not be replayed into
/// its store: they are moved under `<repo>#unrestored-<n>`, which no
/// repository is called, so they are never restored into anything and never
/// tombstoned by a mirror — kept for good.
const UNRESTORED: &str = "#unrestored";

/// Save `files` as `timeline`'s: the branch each repository is on — a
/// `git_switch` in the round moves it — and, as events, whatever changed in
/// its stores since `mirror` last saw them. Nothing is written when nothing
/// changed, or while `mirror` has not been read.
pub fn save(
    engine: &ConversationEngine,
    timeline: TimelineId,
    files: &RepoFiles,
    mirror: &Mutex<Mirror>,
) {
    engine.set_conversation_branches(timeline, &files.branches());
    let mut mirror = mirror.lock().unwrap_or_else(|e| e.into_inner());
    let write = mirror.diff(&files.snapshots());
    if write.is_empty() {
        return;
    }
    match engine.write_conversation_files(timeline, &write) {
        Ok(seqs) => mirror.apply(&write, &seqs),
        Err(e) => {
            // Some of the batch may have landed. What did is in the log, so
            // the mirror is read back from it — never guessed — and the next
            // save writes whatever is still missing.
            tracing::warn!("a conversation's file changes could not be saved: {e}");
            *mirror = read_back(engine, timeline).unwrap_or_else(|why| {
                tracing::warn!(
                    "and its saved changes could not be read back, so none are saved until \
                     it is built again: {why}"
                );
                Mirror::unread()
            });
        }
    }
}

/// Put `files`, a fresh set, back as `timeline` left it — each repository on
/// the conversation's branch and its events replayed into it — and make
/// `mirror` what the events say. A repository whose events do not replay
/// into its store is not lost: its events are moved under a name of their
/// own ([`UNRESTORED`]) and its store starts fresh. When the events cannot
/// be read at all, `mirror` stays unread and nothing is saved for the
/// conversation, so nothing of it is overwritten.
pub fn restore(
    engine: &ConversationEngine,
    timeline: TimelineId,
    files: &RepoFiles,
    mirror: &Mutex<Mirror>,
) {
    if let Some(state) = engine.conversation_state(timeline) {
        for (repo, why) in files.set_branches(&state.branches) {
            tracing::warn!(
                %repo,
                "a conversation's branch in this repository was not taken up: {why}",
            );
        }
    }
    let events = match engine.conversation_files(timeline) {
        Ok(events) => events,
        Err(e) => {
            tracing::warn!(
                "a conversation's saved file changes could not be read, so none are restored \
                 and none saved: {e}"
            );
            return;
        }
    };
    let grouped = by_repo(events);
    let taken: BTreeSet<String> = grouped.keys().cloned().collect();
    let mut repos = BTreeMap::new();
    let mut moved: Vec<(String, Vec<VfsEventPayload>)> = Vec::new();
    for (repo, events) in grouped {
        if repo.contains(UNRESTORED) {
            continue;
        }
        let restored = replay_repo(&events).and_then(|replayed| {
            match files
                .restore([(repo.clone(), replayed.snapshot)].into())
                .pop()
            {
                Some((_, why)) => Err(why),
                None => Ok(replayed.mirror),
            }
        });
        match restored {
            Ok(repo_mirror) => {
                repos.insert(repo, repo_mirror);
            }
            Err(why) => {
                tracing::warn!(
                    %repo,
                    "a conversation's saved changes to this repository could not be restored; \
                     they are kept under a name of their own: {why}",
                );
                moved.push((repo, events));
            }
        }
    }
    if let Err(e) = keep_unrestored(engine, timeline, moved, &taken) {
        tracing::warn!(
            "saved changes that could not be restored could not be moved aside either, so \
             none are saved until the conversation is built again: {e}"
        );
        return;
    }
    *mirror.lock().unwrap_or_else(|e| e.into_inner()) = Mirror::of(repos);
}

/// The mirror of `timeline`'s events as the log holds them now, for the
/// repositories that replay; one that does not is left out, and so never
/// written over.
fn read_back(engine: &ConversationEngine, timeline: TimelineId) -> Result<Mirror, String> {
    let events = engine
        .conversation_files(timeline)
        .map_err(|e| e.to_string())?;
    let repos = by_repo(events)
        .into_iter()
        .filter(|(repo, _)| !repo.contains(UNRESTORED))
        .filter_map(|(repo, events)| Some((repo, replay_repo(&events).ok()?.mirror)))
        .collect();
    Ok(Mirror::of(repos))
}

/// Move each of `moved` — a repository's events that did not restore —
/// under a name of its own: copies written first, the originals tombstoned
/// after, so a cut short move leaves both and loses neither.
fn keep_unrestored(
    engine: &ConversationEngine,
    timeline: TimelineId,
    moved: Vec<(String, Vec<VfsEventPayload>)>,
    taken: &BTreeSet<String>,
) -> Result<(), String> {
    let mut taken = taken.clone();
    for (repo, events) in moved {
        let kept = kept_name(&repo, &taken);
        taken.insert(kept.clone());
        let copies = VfsWrite {
            tombstones: Vec::new(),
            events: events
                .iter()
                .map(|e| VfsAppend {
                    repo: kept.clone(),
                    key: e.key.clone(),
                    body: e.body.clone(),
                })
                .collect(),
        };
        engine
            .write_conversation_files(timeline, &copies)
            .map_err(|e| e.to_string())?;
        let mut kills: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for e in &events {
            kills.entry(e.key.clone()).or_default().push(e.seq);
        }
        let originals = VfsWrite {
            tombstones: kills
                .into_iter()
                .map(|(key, kills)| VfsKill {
                    repo: repo.clone(),
                    key,
                    kills,
                })
                .collect(),
            events: Vec::new(),
        };
        engine
            .write_conversation_files(timeline, &originals)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The name a repository's unrestorable events are kept under: its own,
/// marked [`UNRESTORED`] and numbered past every name in `taken`.
fn kept_name(repo: &str, taken: &BTreeSet<String>) -> String {
    (0..)
        .map(|n| format!("{repo}{UNRESTORED}-{n}"))
        .find(|k| !taken.contains(k))
        .expect("an unbounded range has a free name")
}

#[cfg(test)]
mod tests {
    use zend_vfs::{RepoSpec, Workspace};

    use super::*;

    fn workspace(dir: &std::path::Path) -> Workspace {
        for repo in ["a", "b"] {
            std::fs::create_dir_all(dir.join(repo)).unwrap();
            std::fs::write(dir.join(repo).join("base.txt"), "base\n").unwrap();
        }
        Workspace::new(dir, vec![RepoSpec::named("a"), RepoSpec::named("b")]).unwrap()
    }

    /// The events a sequence of writes leaves live, as the log would hand
    /// them back: every event, less every one a tombstone killed, in
    /// `(repo, key, seq)` order.
    fn live(writes: &[(VfsWrite, Vec<u64>)]) -> Vec<VfsEventPayload> {
        let mut events = Vec::new();
        let mut killed = BTreeSet::new();
        for (write, seqs) in writes {
            for (append, &seq) in write.events.iter().zip(seqs) {
                events.push(VfsEventPayload {
                    timeline_id: 1,
                    seq,
                    repo: append.repo.clone(),
                    key: append.key.clone(),
                    body: append.body.clone(),
                });
            }
            for tomb in &write.tombstones {
                killed.extend(tomb.kills.iter().copied());
            }
        }
        events.retain(|e| !killed.contains(&e.seq));
        events.sort_by(|a, b| (&a.repo, &a.key, a.seq).cmp(&(&b.repo, &b.key, b.seq)));
        events
    }

    /// **A conversation's changes survive being written as events and
    /// replayed into a fresh set** — written, edited and deleted files alike,
    /// across rounds that grow, rewrite and drop chains — and the mirror a
    /// replay builds writes nothing more.
    #[test]
    fn changes_survive_as_events() {
        let dir = tempfile::tempdir().unwrap();
        let files = RepoFiles::overlay(workspace(dir.path()));
        let mut mirror = Mirror::of(BTreeMap::new());
        let mut next = 0u64;
        let mut writes: Vec<(VfsWrite, Vec<u64>)> = Vec::new();
        let mut round = |files: &RepoFiles, mirror: &mut Mirror| {
            let write = mirror.diff(&files.snapshots());
            let seqs: Vec<u64> = (0..write.events.len() as u64).map(|i| next + i).collect();
            next += (write.events.len() + write.tombstones.len()) as u64;
            mirror.apply(&write, &seqs);
            writes.push((write, seqs));
        };

        let a = files.repo("a").unwrap();
        a.write("new.txt", "new\n".into()).unwrap();
        a.edit("base.txt", "base\nedited\n".into()).unwrap();
        files.repo("b").unwrap().delete("base.txt");
        round(&files, &mut mirror);
        a.edit("base.txt", "base\nedited\nagain\n".into()).unwrap();
        a.write("new.txt", "rewritten\n".into()).unwrap();
        round(&files, &mut mirror);
        a.delete("new.txt");
        round(&files, &mut mirror);

        let fresh = files.fresh();
        let mut repos = BTreeMap::new();
        for (repo, events) in by_repo(live(&writes)) {
            let replayed = replay_repo(&events).unwrap();
            assert!(fresh
                .restore([(repo.clone(), replayed.snapshot)].into())
                .is_empty());
            repos.insert(repo, replayed.mirror);
        }
        let (fa, fb) = (fresh.repo("a").unwrap(), fresh.repo("b").unwrap());
        assert_eq!(fa.read("new.txt").unwrap(), None);
        assert_eq!(
            fa.read("base.txt").unwrap().as_deref(),
            Some("base\nedited\nagain\n")
        );
        assert_eq!(fb.read("base.txt").unwrap(), None);
        assert_eq!(fa.deltas("base.txt"), a.deltas("base.txt"));
        assert_eq!(fresh.snapshots(), files.snapshots());
        assert!(
            Mirror::of(repos).diff(&fresh.snapshots()).is_empty(),
            "the replayed mirror holds exactly what the stores do"
        );
    }

    /// **Unrestorable work is kept under a name of its own**, numbered past
    /// every name taken.
    #[test]
    fn unrestorable_work_is_kept_under_a_name_of_its_own() {
        let taken: BTreeSet<String> = ["a".to_string(), format!("a{UNRESTORED}-0")].into();
        assert_eq!(kept_name("a", &taken), format!("a{UNRESTORED}-1"));
        assert_eq!(kept_name("b", &taken), format!("b{UNRESTORED}-0"));
    }
}

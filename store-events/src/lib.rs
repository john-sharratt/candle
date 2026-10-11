//! A set of file stores kept as file events in a substrate
//! (`docs/zend_vfs_events.md`, `docs/replicated_substrate.md` §10).
//!
//! A set of stores — one [`RepoFiles`], a store per repository — lives in
//! memory while it is used, and does not last. The substrate keeps the same
//! thing as **events** on a timeline: [`save`] writes whatever changed in
//! each store — new deltas appended, rewritten and dropped chains tombstoned —
//! and [`restore`] replays a timeline's events back into a fresh set. The two
//! mirror each other without either calling the other: the [`Mirror`] beside
//! the stores is what the events say, and every save writes the difference.
//! Being substrate records, the events replicate like any other.
//!
//! zend keeps a conversation's files this way, npcd an operation's.
//!
//! | Module | Concern |
//! |---|---|
//! | [`event`] | What one event says |
//! | `mirror` | What the substrate holds, and what to write |
//! | [`replay`] | Events back into stores |

pub mod event;
mod mirror;
pub mod replay;

pub use mirror::{Mirror, RepoMirror};

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use candle_conversation::persistence::vfs::{
    VfsAppend, VfsEventId, VfsEventPayload, VfsKill, VfsWrite,
};
use candle_conversation::projection::TimelineId;
use zend_vfs::RepoFiles;

use self::replay::{by_repo, replay_repo};

/// What a repository name carries when its events could not be replayed into
/// its store: they are moved under `<repo>#unrestored-<n>`, which no
/// repository is called, so they are never restored into anything and never
/// tombstoned by a mirror — kept for good.
pub const UNRESTORED: &str = "#unrestored";

/// Where a timeline's file events are written and read back.
pub trait FileLog {
    /// Write `write`'s events, then its tombstones; the ids the events took,
    /// in order.
    fn write_files(
        &self,
        timeline: TimelineId,
        write: &VfsWrite,
    ) -> Result<Vec<VfsEventId>, String>;

    /// Every live event of `timeline`'s files, in `(repo, key)` order and
    /// within a key in stamp order.
    fn read_files(&self, timeline: TimelineId) -> Result<Vec<VfsEventPayload>, String>;
}

/// Save `files` as `timeline`'s: whatever changed in its stores since
/// `mirror` last saw them, as events. Nothing is written when nothing
/// changed, or while `mirror` has not been read.
pub fn save(log: &dyn FileLog, timeline: TimelineId, files: &RepoFiles, mirror: &Mutex<Mirror>) {
    let mut mirror = mirror.lock().unwrap_or_else(|e| e.into_inner());
    let write = mirror.diff(&files.snapshots());
    if write.is_empty() {
        return;
    }
    match log.write_files(timeline, &write) {
        Ok(ids) => mirror.apply(&write, &ids),
        Err(e) => {
            // Some of the batch may have landed. What did is in the log, so
            // the mirror is read back from it — never guessed — and the next
            // save writes whatever is still missing.
            tracing::warn!("a set of file changes could not be saved: {e}");
            *mirror = read_back(log, timeline).unwrap_or_else(|why| {
                tracing::warn!(
                    "and its saved changes could not be read back, so none are saved until \
                     it is built again: {why}"
                );
                Mirror::unread()
            });
        }
    }
}

/// Put `files`, a fresh set, back as `timeline` left it — each repository's
/// events replayed into its store — and make `mirror` what the events say. A
/// repository whose events do not replay into its store is not lost: its
/// events are moved under a name of their own ([`UNRESTORED`]) and its store
/// starts fresh. When the events cannot be read at all, `mirror` stays
/// unread and nothing is saved for the timeline, so nothing of it is
/// overwritten.
pub fn restore(log: &dyn FileLog, timeline: TimelineId, files: &RepoFiles, mirror: &Mutex<Mirror>) {
    let events = match log.read_files(timeline) {
        Ok(events) => events,
        Err(e) => {
            tracing::warn!(
                "a set of saved file changes could not be read, so none are restored and none \
                 saved: {e}"
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
                    "saved changes to this repository could not be restored; they are kept \
                     under a name of their own: {why}",
                );
                moved.push((repo, events));
            }
        }
    }
    if let Err(e) = keep_unrestored(log, timeline, moved, &taken) {
        tracing::warn!(
            "saved changes that could not be restored could not be moved aside either, so \
             none are saved until the set is built again: {e}"
        );
        return;
    }
    *mirror.lock().unwrap_or_else(|e| e.into_inner()) = Mirror::of(repos);
}

/// The mirror of `timeline`'s events as the log holds them now, for the
/// repositories that replay; one that does not is left out, and so never
/// written over.
fn read_back(log: &dyn FileLog, timeline: TimelineId) -> Result<Mirror, String> {
    let events = log.read_files(timeline)?;
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
    log: &dyn FileLog,
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
        log.write_files(timeline, &copies)?;
        let mut kills: BTreeMap<String, Vec<VfsEventId>> = BTreeMap::new();
        for e in &events {
            kills.entry(e.key.clone()).or_default().push(e.id());
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
        log.write_files(timeline, &originals)?;
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
mod tests;

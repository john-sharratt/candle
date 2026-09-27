//! A conversation's own copy of the workspace, kept with the conversation.
//!
//! While a conversation is open its changes to the workspace's files live in
//! its in-memory state — one overlay store per repository ([`RepoFiles`]),
//! each reading the repository through the branch the conversation works on.
//! That state does not last: a turn that ends without finishing evicts it, a
//! restart loses it, and the next request builds it again from the substrate.
//! So after every tool round the conversation's changes and the branch each
//! repository is on are saved into its persisted state (`ConvState::files`
//! and `ConvState::branches`, one record written whole and last-writer-wins
//! on replay), and a conversation's
//! state, when built, is put back: each store on the conversation's branch,
//! then its changes over it. The substrate keeps the changes as values it
//! never reads; this module is what turns them back into stores.

use std::collections::BTreeMap;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use serde_json::Value;
use zend_vfs::{RepoFiles, Snapshot};

/// What a key of saved work that could not be restored carries after the
/// repository's name: the work is kept under a key of its own, and a key no
/// repository has is never restored into anything, so it is kept for good.
const UNRESTORED: &str = "#unrestored";

/// Save `files` as `timeline`'s: the branch each repository is on — a
/// `git_switch` in the round moves it — and its changes, every repository
/// with any, the whole set replacing the last, with every piece of saved
/// work that could not be restored kept as it was. Nothing is written when
/// nothing changed.
pub fn save(engine: &ConversationEngine, timeline: TimelineId, files: &RepoFiles) {
    engine.set_conversation_branches(timeline, &files.branches());
    let mut saved = to_saved(files.snapshots());
    for (key, raw) in files.unrestored() {
        let value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
        saved.insert(key, value);
    }
    engine.set_conversation_files(timeline, &saved);
}

/// Put `files`, a fresh set, back as `timeline` left it: each repository read
/// through the conversation's branch, and the changes it saved restored over
/// it. Saved work that cannot be taken back is logged and kept as it was
/// ([`RepoFiles::keep_unrestored`]), under a key of its own, so that no later
/// save overwrites it; the rest is restored.
pub fn restore(engine: &ConversationEngine, timeline: TimelineId, files: &RepoFiles) {
    let Some(state) = engine.conversation_state(timeline) else {
        return;
    };
    for (repo, why) in files.set_branches(&state.branches) {
        tracing::warn!(
            %repo,
            "a conversation's branch in this repository was not taken up: {why}",
        );
    }
    let raw: BTreeMap<String, String> = state
        .files
        .iter()
        .map(|(key, value)| (key.clone(), value.to_string()))
        .collect();
    let (kept, restorable) = split_kept(state.files);
    for key in kept.keys() {
        if let Some(value) = raw.get(key) {
            files.keep_unrestored(key.clone(), value.clone());
        }
    }
    let (saved, unreadable) = from_saved(restorable);
    for (key, why) in unreadable.into_iter().chain(files.restore(saved)) {
        tracing::warn!(
            repo = %key,
            "a conversation's saved changes to this repository were not restored, and are \
             kept as saved: {why}",
        );
        if let Some(value) = raw.get(&key) {
            files.keep_unrestored(kept_key(&key, &raw), value.clone());
        }
    }
}

/// Saved work split into what an earlier restore kept — under a key that
/// names no repository, carried on as it was and never restored into a store
/// of that name — and what is to be restored.
fn split_kept(
    saved: BTreeMap<String, Value>,
) -> (BTreeMap<String, Value>, BTreeMap<String, Value>) {
    saved
        .into_iter()
        .partition(|(key, _)| key.contains(UNRESTORED))
}

/// The key saved work under `key` that could not be restored is kept under:
/// its own, when it is kept already; otherwise the repository's name marked
/// [`UNRESTORED`], numbered past every key in `taken`.
fn kept_key(key: &str, taken: &BTreeMap<String, String>) -> String {
    if key.contains(UNRESTORED) {
        return key.to_string();
    }
    (0..)
        .map(|n| format!("{key}{UNRESTORED}-{n}"))
        .find(|k| !taken.contains_key(k))
        .expect("an unbounded range has a free key")
}

/// Snapshots as the values the substrate keeps.
fn to_saved(snapshots: BTreeMap<String, Snapshot>) -> BTreeMap<String, Value> {
    snapshots
        .into_iter()
        .map(|(repo, snapshot)| {
            let value =
                serde_json::to_value(snapshot).expect("a snapshot always serialises to JSON");
            (repo, value)
        })
        .collect()
}

/// Snapshots read back, and each repository whose value is not one, with why.
type Restored = (BTreeMap<String, Snapshot>, Vec<(String, String)>);

/// The substrate's values back as snapshots, with each repository whose value
/// is not one named, and why.
fn from_saved(saved: BTreeMap<String, Value>) -> Restored {
    let mut snapshots = BTreeMap::new();
    let mut unreadable = Vec::new();
    for (repo, value) in saved {
        match serde_json::from_value::<Snapshot>(value) {
            Ok(snapshot) => {
                snapshots.insert(repo, snapshot);
            }
            Err(e) => unreadable.push((repo, e.to_string())),
        }
    }
    (snapshots, unreadable)
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

    /// **A conversation's changes survive being saved and restored into a
    /// fresh set** — through the values the substrate keeps, written, edited
    /// and deleted files alike — and a repository with no changes saves
    /// nothing.
    #[test]
    fn changes_survive_saving_and_restoring() {
        let dir = tempfile::tempdir().unwrap();
        let files = RepoFiles::overlay(workspace(dir.path()));
        let a = files.repo("a").unwrap();
        a.write("new.txt", "new\n".into()).unwrap();
        a.edit("base.txt", "base\nedited\n".into()).unwrap();
        files.repo("b").unwrap().delete("base.txt");

        let saved = to_saved(files.snapshots());
        assert_eq!(saved.keys().collect::<Vec<_>>(), ["a", "b"]);
        let wire = serde_json::to_string(&saved).unwrap();
        let back: BTreeMap<String, Value> = serde_json::from_str(&wire).unwrap();

        let fresh = files.fresh();
        let (snapshots, unreadable) = from_saved(back);
        assert!(unreadable.is_empty());
        assert!(fresh.restore(snapshots).is_empty());
        let (fa, fb) = (fresh.repo("a").unwrap(), fresh.repo("b").unwrap());
        assert_eq!(fa.read("new.txt").unwrap().as_deref(), Some("new\n"));
        assert_eq!(
            fa.read("base.txt").unwrap().as_deref(),
            Some("base\nedited\n")
        );
        assert_eq!(fb.read("base.txt").unwrap(), None);
        assert_eq!(fa.deltas("base.txt"), a.deltas("base.txt"));

        assert!(to_saved(files.fresh().snapshots()).is_empty());
    }

    /// **Saved work that cannot be restored is kept under a key of its own**,
    /// numbered past every key taken, and one already kept keeps its key.
    #[test]
    fn unrestored_work_is_kept_under_its_own_key() {
        let taken: BTreeMap<String, String> = [
            ("a".to_string(), String::new()),
            (format!("a{UNRESTORED}-0"), String::new()),
        ]
        .into();
        assert_eq!(kept_key("a", &taken), format!("a{UNRESTORED}-1"));
        assert_eq!(
            kept_key(&format!("a{UNRESTORED}-0"), &taken),
            format!("a{UNRESTORED}-0")
        );
        let files = RepoFiles::detached();
        files.keep_unrestored("a#unrestored-1".into(), "{\"chains\":{}}".into());
        assert_eq!(
            files.unrestored().keys().collect::<Vec<_>>(),
            ["a#unrestored-1"]
        );
        assert!(
            files.fresh().unrestored().is_empty(),
            "a fresh set keeps none"
        );
    }

    /// **Work kept from an earlier restore is never restored as a
    /// repository** of its key's name.
    #[test]
    fn kept_work_is_carried_not_restored() {
        let saved: BTreeMap<String, Value> = [
            ("a".to_string(), serde_json::json!({})),
            (format!("a{UNRESTORED}-0"), serde_json::json!("kept")),
        ]
        .into();
        let (kept, restorable) = split_kept(saved);
        assert_eq!(
            kept.keys().collect::<Vec<_>>(),
            [&format!("a{UNRESTORED}-0")]
        );
        assert_eq!(restorable.keys().collect::<Vec<_>>(), ["a"]);
    }

    /// A value that is not a snapshot is named, not dropped silently, and
    /// the others still come back.
    #[test]
    fn an_unreadable_value_is_named() {
        let saved: BTreeMap<String, Value> = [
            ("good".to_string(), serde_json::json!({})),
            ("bad".to_string(), serde_json::json!("not a snapshot")),
        ]
        .into();
        let (snapshots, unreadable) = from_saved(saved);
        assert_eq!(snapshots.keys().collect::<Vec<_>>(), ["good"]);
        assert_eq!(
            unreadable
                .iter()
                .map(|(r, _)| r.as_str())
                .collect::<Vec<_>>(),
            ["bad"]
        );
    }
}

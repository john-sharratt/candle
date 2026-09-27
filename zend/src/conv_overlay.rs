//! A conversation's own copy of the workspace, kept with the conversation.
//!
//! While a conversation is open its changes to the workspace's files live in
//! its in-memory state — one overlay store per repository ([`RepoFiles`]).
//! That state does not last: a turn that ends without finishing evicts it, a
//! restart loses it, and the next request builds it again from the substrate.
//! So after every tool round the conversation's changes are saved into its
//! persisted state (`ConvState::files`, one record written whole and
//! last-writer-wins on replay, beside its branches), and a conversation's
//! state, when built, restores them. The substrate keeps them as values it
//! never reads; this module is what turns them back into stores.

use std::collections::BTreeMap;

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use serde_json::Value;
use zend_vfs::{RepoFiles, Snapshot};

/// Save `files`' changes as `timeline`'s — every repository with any, the
/// whole set replacing the last. Nothing is written when nothing changed.
pub fn save(engine: &ConversationEngine, timeline: TimelineId, files: &RepoFiles) {
    engine.set_conversation_files(timeline, &to_saved(files.snapshots()));
}

/// Restore the changes `timeline` saved into `files`, a fresh set. A
/// repository whose changes cannot be read or taken back is logged and left
/// without them; the rest are restored.
pub fn restore(engine: &ConversationEngine, timeline: TimelineId, files: &RepoFiles) {
    let Some(state) = engine.conversation_state(timeline) else {
        return;
    };
    let (saved, unreadable) = from_saved(state.files);
    for (repo, why) in unreadable.into_iter().chain(files.restore(saved)) {
        tracing::warn!(
            %repo,
            "a conversation's saved changes to this repository were not restored: {why}",
        );
    }
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

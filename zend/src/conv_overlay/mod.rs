//! A conversation's own copy of the workspace, kept with the conversation.
//!
//! While a conversation is open its changes to the workspace's files live in
//! its in-memory state — one overlay store per repository ([`RepoFiles`]),
//! each reading the repository through the branch the conversation works on.
//! That state does not last: a turn that ends without finishing evicts it, a
//! restart loses it, and the next request builds it again from the substrate.
//!
//! So the substrate keeps the same thing as **events** on the conversation's
//! timeline (`docs/zend_vfs_events.md`), through `store-events`: after every
//! tool round, [`save`] writes the branch each repository is on and whatever
//! changed in its store, and when the conversation's state is built,
//! [`restore`] puts each store on its branch and replays its events back into
//! it.

use std::sync::Mutex;

use candle_conversation::persistence::vfs::{VfsEventId, VfsEventPayload, VfsWrite};
use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use store_events::FileLog;
use zend_vfs::RepoFiles;

pub use store_events::Mirror;

/// The engine as a conversation's file log.
struct Engine<'a>(&'a ConversationEngine);

impl FileLog for Engine<'_> {
    fn write_files(
        &self,
        timeline: TimelineId,
        write: &VfsWrite,
    ) -> Result<Vec<VfsEventId>, String> {
        self.0
            .write_conversation_files(timeline, write)
            .map_err(|e| e.to_string())
    }

    fn read_files(&self, timeline: TimelineId) -> Result<Vec<VfsEventPayload>, String> {
        self.0
            .conversation_files(timeline)
            .map_err(|e| e.to_string())
    }
}

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
    store_events::save(&Engine(engine), timeline, files, mirror);
}

/// Put `files`, a fresh set, back as `timeline` left it — each repository on
/// the conversation's branch and its events replayed into it — and make
/// `mirror` what the events say (`store_events::restore`).
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
    store_events::restore(&Engine(engine), timeline, files, mirror);
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use store_events::replay::{by_repo, replay_repo};
    use zend_vfs::{RepoSpec, Workspace};

    use super::*;

    /// **An upload survives as events** (§10.2): a binary file stored in the
    /// conversation's uploads store is written as its `ReplaceBinary` delta and
    /// the state it leaves, and replayed into a fresh set reads back byte for
    /// byte.
    #[test]
    fn an_upload_survives_as_events() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("uploads")).unwrap();
        let ws = Workspace::new(dir.path(), vec![RepoSpec::named("uploads")]).unwrap();
        let files = RepoFiles::overlay(ws);
        let png = vec![0x89, 0x50, 0x4e, 0x47, 0xff, 0x00, 0x0d];
        let stored =
            crate::uploads::store(&files.repo("uploads").unwrap(), "logo.png", png.clone())
                .unwrap();
        let mut mirror = Mirror::of(BTreeMap::new());
        let write = mirror.diff(&files.snapshots());
        let kinds: Vec<(&str, &str, &str)> = write
            .events
            .iter()
            .map(|e| {
                (
                    e.repo.as_str(),
                    e.key.as_str(),
                    e.body["kind"].as_str().unwrap_or_default(),
                )
            })
            .collect();
        assert_eq!(
            kinds,
            [
                ("uploads", "logo.png", "delta"),
                ("uploads", "logo.png", "state")
            ]
        );
        assert_eq!(write.events[0].body["delta"]["kind"], "replace_binary");
        let ids: Vec<VfsEventId> = (0..2).map(|seq| VfsEventId { origin: 0xB, seq }).collect();
        mirror.apply(&write, &ids);
        let events: Vec<VfsEventPayload> = write
            .events
            .iter()
            .zip(&ids)
            .map(|(e, id)| VfsEventPayload {
                timeline_id: 1,
                origin: id.origin,
                seq: id.seq,
                repo: e.repo.clone(),
                key: e.key.clone(),
                body: e.body.clone(),
            })
            .collect();
        let fresh = files.fresh();
        for (repo, events) in by_repo(events) {
            let replayed = replay_repo(&events).unwrap();
            assert!(fresh.restore([(repo, replayed.snapshot)].into()).is_empty());
        }
        assert_eq!(
            fresh.repo("uploads").unwrap().read_bytes(&stored).unwrap(),
            Some(png)
        );
    }
}

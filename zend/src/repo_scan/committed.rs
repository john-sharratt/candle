//! The folders the knowledge base holds: its folder records, each one whole
//! the moment it is written, as a file's is (`crate::code_read::committed`).

use super::*;

/// Every folder record, as the plan names a unit.
pub fn committed(engine: &ConversationEngine) -> Vec<Committed> {
    records::all(&engine.conversation())
        .iter()
        .filter(|r| r.unit == FileUnit::Folder)
        .map(committed_of)
        .collect()
}

/// The folder keys whose record stands and whose members this process holds:
/// what a pool skips. A record not sealed here is sealed again, its members
/// restored from their streams (§12 D15).
pub(crate) fn sealed_folder_keys(engine: &ConversationEngine) -> HashSet<String> {
    let conversation = engine.conversation();
    records::all(&conversation)
        .into_iter()
        .filter(|r| r.unit == FileUnit::Folder && records::placeable(&conversation, r))
        .map(|r| r.content_key)
        .collect()
}

/// Retire EVERY folder record — `--wipe-layer repo_map`: the next pass seals
/// every folder's listing again.
pub(crate) fn wipe_layer(engine: &ConversationEngine) {
    let conversation = engine.conversation();
    let mut wiped = 0usize;
    for record in records::all(&conversation) {
        if record.unit != FileUnit::Folder {
            continue;
        }
        match records::retire(&conversation, &record) {
            Ok(()) => wiped += 1,
            Err(err) => tracing::warn!(
                target: "zend::repo_scan",
                dir = %record.path,
                "--wipe-layer repo_map: retiring the record failed: {err:#}",
            ),
        }
    }
    tracing::info!(
        target: "zend::repo_scan",
        wiped,
        "--wipe-layer repo_map: every folder record retired — the next pass seals \
         every listing again",
    );
}

/// Conversation metadata naming the folder a unit of another layer is of:
/// what the self-check reads a folder by.
pub(crate) const DIR_KEY: &str = "dir";

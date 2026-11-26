//! The record of which files the knowledge base holds: its file records
//! (`crate::files::records`), each one whole the moment it is written.
//!
//! A record is written only once every member of its file is sealed, and it
//! replicates as one object, so a record is a file the knowledge base holds —
//! there is no reading that stopped part-way to tell apart. What a pass plans
//! against is every record; what a pool skips is every record whose members
//! this process holds ([`finished_content_keys`]). Uploads are kept out of
//! branch checks, since they live on no branch and their absence from one is
//! not a deletion.

use super::*;

/// Every file record of a repository's branches, as the plan names a unit:
/// its content key and path, and its [`FileKey::scan_id`] where a reading's
/// timeline stood. Uploads are not among them: their endpoint keeps them,
/// never the pass.
pub fn committed(engine: &ConversationEngine) -> Vec<Committed> {
    records::all(&engine.conversation())
        .iter()
        .filter(|r| r.unit == FileUnit::File && !is_upload_path(&r.path))
        .map(committed_of)
        .collect()
}

/// `record` as the plan's unit.
pub(crate) fn committed_of(record: &FileRecord) -> Committed {
    Committed {
        timeline: record.key().scan_id(),
        key: record.content_key.clone(),
        subject: record.path.clone(),
        stamp: Default::default(),
        quant: None,
    }
}

/// Retire the records the plan names by their scan ids.
pub fn retire_planned(engine: &ConversationEngine, layer: &str, ids: &[TimelineId]) {
    if ids.is_empty() {
        return;
    }
    let conversation = engine.conversation();
    for record in records::all(&conversation) {
        if !ids.contains(&record.key().scan_id()) {
            continue;
        }
        if let Err(err) = records::retire(&conversation, &record) {
            tracing::warn!(
                target: "zend::branch_ingest",
                layer,
                file = %record.path,
                "retiring a file no branch holds failed: {err:#}",
            );
        }
    }
}

/// Write each of `held`'s branches where the walk found them moved
/// (`live`: content key → the branches holding it, in walk order). Returns
/// how many records were rewritten.
pub fn retie_records(
    engine: &ConversationEngine,
    layer: &str,
    held: &[Committed],
    live: &HashMap<&str, &[String]>,
) -> usize {
    let conversation = engine.conversation();
    let keys: HashSet<&str> = held.iter().map(|c| c.key.as_str()).collect();
    let mut rewritten = 0usize;
    for mut record in records::all(&conversation) {
        if !keys.contains(record.content_key.as_str()) {
            continue;
        }
        let Some(&want) = live.get(record.content_key.as_str()) else {
            continue;
        };
        if want.is_empty() || record.branches == want {
            continue;
        }
        record.branches = want.to_vec();
        match records::put(&conversation, &record) {
            Ok(()) => rewritten += 1,
            Err(err) => tracing::warn!(
                target: "zend::branch_ingest",
                layer,
                file = %record.path,
                "recording the branches a file is on failed: {err:#}",
            ),
        }
    }
    if rewritten > 0 {
        tracing::info!(
            target: "zend::branch_ingest",
            layer,
            rewritten,
            "recorded the branches that now hold each file",
        );
    }
    rewritten
}

/// Content keys whose record stands and whose members this process holds,
/// uploads included — the snapshot [`ingest_jobs`] probes before sealing a
/// file. A record whose members are not sealed here (one from before a
/// restart, or one another member wrote) is not among them: the pool seals
/// its chain, restoring each member from its stream (§12 D15).
pub(crate) fn sealed_keys(engine: &ConversationEngine) -> HashSet<String> {
    let conversation = engine.conversation();
    records::all(&conversation)
        .into_iter()
        .filter(|r| r.unit == FileUnit::File && records::placeable(&conversation, r))
        .map(|r| r.content_key)
        .collect()
}

/// Whether a workspace-relative `path` (with `/` separators) lives in the
/// daemon's `uploads` repository ([`UPLOADS_REPO`]). Matched on the FIRST
/// segment only, and case-insensitively (the win32 FS is case-insensitive, so
/// an existing `Uploads/` dir still resolves to the daemon's uploads dir) — so
/// an `uploads/` folder inside a repository is NOT matched. Uploads are
/// endpoint-managed and on no branch, so the branch pass must never read their
/// absence from the branches as a deletion.
pub(crate) fn is_upload_path(path: &str) -> bool {
    path.split('/')
        .next()
        .unwrap_or("")
        .eq_ignore_ascii_case(UPLOADS_REPO)
}

/// Retire EVERY file record — `--wipe-layer code_reading`: the next pass
/// finds nothing recorded and seals every file as new, exactly as on a fresh
/// install. Every other layer's content survives untouched.
///
/// **Uploads are exempt.** They live in the endpoint-managed uploads folder,
/// on no branch, so no pass reads them again: retiring one would delete
/// uploaded content with nothing to rebuild it from.
pub(crate) fn wipe_layer(engine: &ConversationEngine) {
    let conversation = engine.conversation();
    let mut wiped = 0usize;
    for record in records::all(&conversation) {
        if record.unit != FileUnit::File || is_upload_path(&record.path) {
            continue;
        }
        match records::retire(&conversation, &record) {
            Ok(()) => wiped += 1,
            Err(err) => tracing::warn!(
                target: "zend::code_read::ingest",
                path = %record.path,
                "--wipe-layer code_reading: retiring the record failed: {err:#}",
            ),
        }
    }
    tracing::info!(
        target: "zend::code_read::ingest",
        wiped,
        "--wipe-layer code_reading: every file record retired (uploads kept) — \
         the next pass seals the whole knowledge base again",
    );
}

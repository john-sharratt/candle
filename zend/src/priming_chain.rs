//! The priming chain: a fixed sequence of foundational documents every
//! conversation in the system — dialogue and ingest alike — ultimately
//! descends from.
//!
//! ```text
//! root ls (repo_map's own "." folder conversation)
//!   -> README.md -> ARCHITECTURE.md -> AGENTS.md -> CLAUDE.md
//!   -> everything else (base_conv, every ingest_bases entry, every
//!      repo_map folder, every code_reading file)
//! ```
//!
//! Each link records its predecessor as its PARENT
//! (`ConversationEngine::set_forked_from`) before its own reading starts, so
//! the projection for every one of its turns already carries the whole chain
//! behind it — every `file_read` round and every summary, in the order they
//! were read (`Substrate::inherited_chain`). By the time CLAUDE.md's
//! conversation answers, it has genuinely read all four documents (plus the
//! root listing) in order, not summarized secondhand. A missing anchor file is
//! skipped; the chain continues from whatever the last present link was. An
//! empty workspace (no top-level files at all) has no root ls to start from,
//! so the whole chain is a no-op.
//!
//! **The link is a durable pointer, not a copy.** Nothing is duplicated onto
//! the child and no ancestor has to stay resident: an ancestor selected into a
//! projection is elevated from whatever tier it is on by the ordinary
//! working-set path. The pointer lives in the timeline's persisted `custom`
//! metadata, so it survives a restart — which the previous design, built on
//! copying turns out of a hot-pinned source, could not.
//!
//! Each link IS its own standalone `repo_map`/`code_reading` entry — same
//! tagging, same resume-cache key as any other file or folder
//! (`repo_scan::ingest_root_unit`, `code_read::ingest_chain_file` reuse the
//! exact same per-unit machinery the worker pools use). A later background
//! pass sees these files as already-ingested — a resume-cache hit — rather
//! than redoing the work.

use std::path::Path;
use std::sync::Mutex;

use candle_conversation::projection::TimelineId;
use candle_conversation::Sequence;

use crate::code_read;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan;

/// Anchor filenames, in chain order after root ls. Matched case-insensitively
/// against the workspace root's own top-level entries — never a subdirectory.
const ANCHOR_FILES: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// Case-insensitive lookup of `name` among `workspace`'s own top-level
/// entries, returning the entry's ACTUAL on-disk name — so the caller reads
/// the real path (`claude.md`, `Claude.MD`, whatever it actually is) rather
/// than assuming `name`'s casing. `None` when absent, or not a plain file.
fn find_anchor(workspace: &Path, name: &str) -> Option<String> {
    let entries = std::fs::read_dir(workspace).ok()?;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.eq_ignore_ascii_case(name) && entry.path().is_file() {
            return Some(file_name.into_owned());
        }
    }
    None
}

/// Build the priming chain (or find it already built via each link's own
/// resume cache) and return its final link's timeline — `None` only when the
/// workspace has no top-level files at all, so there is no root ls to start
/// the chain from.
///
/// Always walks the workspace root at depth 1: root ls is defined as
/// "repo_map on the root folder" specifically, independent of whatever
/// `--max-depth` the daemon's own background repo_map pass otherwise runs
/// at.
pub(crate) fn build(
    ctx: &RefreshContext<'_>,
    workspace: &Path,
    repo_map_base: &Mutex<Sequence>,
    code_reading_base: &Mutex<Sequence>,
    progress: &LoadProgress,
) -> anyhow::Result<Option<TimelineId>> {
    // Resolve which anchors are actually present first, so the step's total is
    // the real number of documents rather than a ceiling it never reaches.
    let anchors: Vec<String> = ANCHOR_FILES
        .iter()
        .filter_map(|name| {
            let found = find_anchor(workspace, name);
            if found.is_none() {
                tracing::debug!(
                    target: "zend::priming_chain",
                    anchor = name,
                    "not present — chain continues from the prior link",
                );
            }
            found
        })
        .collect();
    let total = anchors.len() as u64 + 1; // + root ls
    progress.set_step_progress(0, total);

    let map = repo_scan::walk_workspace(workspace, Some(1));
    let Some(mut current) = repo_scan::ingest_root_unit(ctx, workspace, &map, repo_map_base)?
    else {
        tracing::warn!(
            target: "zend::priming_chain",
            "no top-level files in the workspace — nothing to build root ls from, \
             priming chain skipped",
        );
        return Ok(None);
    };

    progress.set_step_progress(1, total);
    for (done, actual_name) in anchors.iter().enumerate() {
        match code_read::ingest_chain_file(ctx, workspace, actual_name, current, code_reading_base)
        {
            Ok(Some(new_end)) => {
                current = new_end;
            }
            Ok(None) => {
                tracing::warn!(
                    target: "zend::priming_chain",
                    anchor = %actual_name,
                    "present but not a recognised code language, or dropped by the \
                     size guard — chain continues from the prior link",
                );
            }
            Err(e) => {
                return Err(e.context(format!("priming chain: {actual_name}")));
            }
        }
        progress.set_step_progress(done as u64 + 2, total);
    }

    tracing::info!(
        target: "zend::priming_chain",
        end_timeline = current.raw(),
        "priming chain built",
    );
    Ok(Some(current))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_anchor_matches_case_insensitively() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Claude.MD"), "hi").unwrap();
        assert_eq!(
            find_anchor(dir.path(), "CLAUDE.md"),
            Some("Claude.MD".to_string())
        );
    }

    #[test]
    fn find_anchor_is_none_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(find_anchor(dir.path(), "CLAUDE.md"), None);
    }

    #[test]
    fn find_anchor_ignores_a_same_named_subdirectory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("agents.md")).unwrap();
        assert_eq!(
            find_anchor(dir.path(), "AGENTS.md"),
            None,
            "a directory named agents.md is not a file to read",
        );
    }

    #[test]
    fn find_anchor_never_descends_into_subdirectories() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("docs")).unwrap();
        std::fs::write(dir.path().join("docs").join("README.md"), "nested").unwrap();
        assert_eq!(
            find_anchor(dir.path(), "README.md"),
            None,
            "only a top-level README.md counts as the anchor",
        );
    }
}

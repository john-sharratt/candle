//! The priming chain: a fixed sequence of foundational documents every
//! conversation in the system — dialogue and ingest alike — ultimately
//! descends from.
//!
//! ```text
//! workspace ls (repo_map's own "." conversation — the repositories)
//!   -> for each repository, in manifest order:
//!        repo ls (its root folder) -> README.md -> ARCHITECTURE.md
//!          -> AGENTS.md -> CLAUDE.md
//!   -> everything else (base_conv, every ingest_bases entry, every
//!      repo_map folder, every code_reading file)
//! ```
//!
//! Each link records its predecessor as its PARENT
//! (`ConversationEngine::set_forked_from`) before its own reading starts, so
//! the projection for every one of its turns already carries the whole chain
//! behind it — every `file_read` round and every summary, in the order they
//! were read (`Substrate::inherited_chain`). By the time the last repository's
//! CLAUDE.md answers, the chain has genuinely read every repository's listing
//! and anchor documents in order, not summarized secondhand — which is what
//! seeds the base conversation every dialogue forks from with the whole
//! workspace. A missing anchor file, or a repository with no files at its root,
//! is skipped; the chain continues from whatever the last present link was. A
//! workspace whose repositories hold no walked file at their roots has no
//! workspace ls to start from, so the whole chain is a no-op.
//!
//! The `uploads` repository is not part of the chain: it holds the user's
//! files, not the project.
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
use zend_vfs::workspace::Repo;
use zend_vfs::Workspace;

use crate::code_read;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan;

/// Anchor filenames, in chain order after a repository's ls. Matched
/// case-insensitively against the repository root's own entries — never a
/// subdirectory.
const ANCHOR_FILES: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// Case-insensitive lookup of `name` among `dir`'s own entries, returning the
/// entry's ACTUAL on-disk name — so the caller reads the real path
/// (`claude.md`, `Claude.MD`, whatever it actually is) rather than assuming
/// `name`'s casing. `None` when absent, or not a plain file.
fn find_anchor(dir: &Path, name: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if file_name.eq_ignore_ascii_case(name) && entry.path().is_file() {
            return Some(file_name.into_owned());
        }
    }
    None
}

/// Each repository in the chain, in manifest order, with the workspace-relative
/// keys of the anchor documents present at its root (`candle/README.md`).
fn chain_plan(workspace: &Workspace) -> Vec<(&Repo, Vec<String>)> {
    workspace
        .repos()
        .iter()
        .filter(|r| !code_read::is_upload_path(&r.name))
        .map(|repo| {
            let anchors = ANCHOR_FILES
                .iter()
                .filter_map(|name| {
                    let found = find_anchor(&repo.dir, name);
                    if found.is_none() {
                        tracing::debug!(
                            target: "zend::priming_chain",
                            repo = %repo.name,
                            anchor = name,
                            "not present — chain continues from the prior link",
                        );
                    }
                    found.map(|actual| format!("{}/{actual}", repo.name))
                })
                .collect();
            (repo, anchors)
        })
        .collect()
}

/// Build the priming chain (or find it already built via each link's own
/// resume cache) and return its final link's timeline — `None` only when no
/// repository holds a walked file at its root, so there is no workspace ls to
/// start the chain from.
///
/// Always walks each repository at depth 1: a link's ls is defined as
/// "repo_map on that root folder" specifically, independent of whatever
/// `--max-depth` the daemon's own background repo_map pass otherwise runs
/// at.
pub(crate) fn build(
    ctx: &RefreshContext<'_>,
    workspace: &Workspace,
    repo_map_base: &Mutex<Sequence>,
    code_reading_base: &Mutex<Sequence>,
    progress: &LoadProgress,
) -> anyhow::Result<Option<TimelineId>> {
    // Resolve which anchors are actually present first, so the step's total is
    // the real number of documents rather than a ceiling it never reaches.
    let plan = chain_plan(workspace);
    // The workspace ls, then per repository its ls and its anchors.
    let total = 1 + plan
        .iter()
        .map(|(_, anchors)| 1 + anchors.len() as u64)
        .sum::<u64>();
    progress.set_step_progress(0, total);

    let map = repo_scan::walk_workspace(workspace, "", Some(1));
    let Some(mut current) =
        repo_scan::ingest_chain_unit(ctx, workspace, &map, ".", None, repo_map_base)?
    else {
        tracing::warn!(
            target: "zend::priming_chain",
            "no repository holds a file at its root — nothing to build the workspace \
             ls from, priming chain skipped",
        );
        return Ok(None);
    };
    let mut done = 1u64;
    progress.set_step_progress(done, total);

    for (repo, anchors) in &plan {
        let dir = format!("{}/", repo.name);
        match repo_scan::ingest_chain_unit(
            ctx,
            workspace,
            &map,
            &dir,
            Some(current),
            repo_map_base,
        )? {
            Some(repo_ls) => current = repo_ls,
            None => tracing::warn!(
                target: "zend::priming_chain",
                repo = %repo.name,
                "no file at the repository's root — its ls is skipped, the chain \
                 continues from the prior link",
            ),
        }
        done += 1;
        progress.set_step_progress(done, total);

        for key in anchors {
            match code_read::ingest_chain_file(
                ctx,
                workspace.root(),
                key,
                current,
                code_reading_base,
            ) {
                Ok(Some(new_end)) => {
                    current = new_end;
                }
                Ok(None) => {
                    tracing::warn!(
                        target: "zend::priming_chain",
                        anchor = %key,
                        "present but not a recognised code language, or dropped by the \
                         size guard — chain continues from the prior link",
                    );
                }
                Err(e) => {
                    return Err(e.context(format!("priming chain: {key}")));
                }
            }
            done += 1;
            progress.set_step_progress(done, total);
        }
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

    use zend_vfs::RepoSpec;

    /// **The chain walks the repositories in manifest order**, each with the
    /// anchors present at its own root as workspace-relative keys, in anchor
    /// order — and never the uploads repository.
    #[test]
    fn the_plan_takes_each_repository_in_order_with_its_own_anchors() {
        let dir = tempfile::tempdir().unwrap();
        for repo in ["mind", "candle", "uploads"] {
            std::fs::create_dir(dir.path().join(repo)).unwrap();
        }
        std::fs::write(dir.path().join("candle").join("CLAUDE.md"), "c").unwrap();
        std::fs::write(dir.path().join("candle").join("readme.md"), "r").unwrap();
        std::fs::write(dir.path().join("uploads").join("README.md"), "u").unwrap();
        let ws = Workspace::new(
            dir.path(),
            vec![
                RepoSpec::named("mind"),
                RepoSpec::named("candle"),
                RepoSpec::named("uploads"),
            ],
        )
        .unwrap();
        let plan: Vec<(String, Vec<String>)> = chain_plan(&ws)
            .into_iter()
            .map(|(repo, anchors)| (repo.name.clone(), anchors))
            .collect();
        assert_eq!(
            plan,
            vec![
                ("mind".to_string(), vec![]),
                (
                    "candle".to_string(),
                    vec![
                        "candle/readme.md".to_string(),
                        "candle/CLAUDE.md".to_string()
                    ]
                ),
            ]
        );
    }

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

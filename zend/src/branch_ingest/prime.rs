//! The priming chain: the foundational reading every conversation in the
//! system — dialogue and ingest alike — descends from
//! (`docs/zend_workspace_execution.md` §6).
//!
//! ```text
//! for each repository, in the workspace's order:
//!   its root listing -> README -> ARCHITECTURE -> AGENTS -> CLAUDE
//! -> base_conv, and every other repo_map folder and code_reading file
//! ```
//!
//! There is no link for the workspace itself: every file operation works
//! inside one repository, and the repositories are named to the model by
//! every tool's `repo` enum.
//!
//! Each link records the one before it as its PARENT
//! (`ConversationEngine::set_forked_from`) before its own reading starts, so
//! the projection for every one of its turns carries the whole chain behind
//! it (`Substrate::inherited_chain`): by the time the last CLAUDE.md is read,
//! the conversation reading it has genuinely read every listing and document
//! before it, in order. The link is a durable pointer, not a copy — nothing
//! is duplicated onto the child and no ancestor has to stay resident.
//!
//! Every link is an ordinary unit of its layer, keyed by content like any
//! other: the pass that follows finds it committed and does not read it
//! again, and a link already committed on an earlier boot is re-pointed at the
//! link before it rather than read again. Each repository contributes the
//! version its default branch holds — one base conversation serves every
//! branch, and the default branch is the one a conversation starts on. A
//! missing anchor is skipped; the chain continues from the link before it.

use std::collections::HashSet;

use candle_conversation::projection::TimelineId;
use zend_vfs::Workspace;

use super::walk::{walk, FileItem, UnitItem};
use super::{record_branches, BranchIngest, LayerPass};
use crate::code_read::{self, FileJob};
use crate::ingest::IngestMode;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{self, UnitJob};

/// Anchor documents, in chain order after a repository's root listing.
/// Matched case-insensitively against the repository's own root files —
/// never a subfolder's.
pub const ANCHOR_FILES: [&str; 4] = ["README.md", "ARCHITECTURE.md", "AGENTS.md", "CLAUDE.md"];

/// One link of the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link<'a> {
    Folder(&'a UnitItem),
    File(&'a FileItem),
}

/// The chain's links, in order: for each of `repos` — `(name, default
/// branch)`, in the workspace's order — its root folder and its anchors, each
/// the version the default branch holds.
pub fn links<'a>(
    units: &'a [UnitItem],
    files: &'a [FileItem],
    repos: &[(String, String)],
) -> Vec<Link<'a>> {
    let mut out: Vec<Link<'a>> = Vec::new();
    for (repo, default) in repos {
        let on_default = |branches: &[String]| branches.iter().any(|b| b == default);
        let root = format!("{repo}/");
        if let Some(unit) = units
            .iter()
            .find(|u| u.unit.dir == root && on_default(&u.branches))
        {
            out.push(Link::Folder(unit));
        }
        for anchor in ANCHOR_FILES {
            let found = files.iter().find(|f| {
                f.repo == *repo
                    && on_default(&f.branches)
                    && f.file.path.strip_prefix(&root).is_some_and(|name| {
                        !name.contains('/') && name.eq_ignore_ascii_case(anchor)
                    })
            });
            if let Some(file) = found {
                out.push(Link::File(file));
            }
        }
    }
    out
}

impl BranchIngest {
    /// Build the priming chain over `layers` and return its final link —
    /// `None` when there is nothing to chain (no git repository, or neither
    /// layer ingested). Run once per boot, before any pass: the pass then
    /// finds every link committed.
    pub fn prime(
        &self,
        ctx: &RefreshContext<'_>,
        workspace: &Workspace,
        layers: &[LayerPass<'_>],
        progress: &LoadProgress,
    ) -> anyhow::Result<Option<TimelineId>> {
        let (repos, _unreadable) = record_branches(workspace);
        let folders = layers.iter().find(|l| l.mode == IngestMode::Folders);
        let files = layers.iter().find(|l| l.mode == IngestMode::Files);
        let (folder_corpus, file_corpus) = {
            let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
            let mut corpus_of = |layer: Option<&LayerPass<'_>>| {
                layer
                    .map(|l| walk(&repos, &l.scope, &mut trees).0)
                    .unwrap_or_default()
            };
            (corpus_of(folders), corpus_of(files))
        };
        let binary = self.binary.lock().unwrap().clone();
        let file_items: Vec<FileItem> = file_corpus
            .ingest
            .files
            .into_iter()
            .filter(|f| !binary.contains(&f.key))
            .collect();
        let units = folder_corpus.ingest.units;
        let defaults: Vec<(String, String)> = repos
            .iter()
            .filter_map(|r| Some((r.name.clone(), r.tips.first()?.name.as_str().to_string())))
            .collect();
        let chain = links(&units, &file_items, &defaults);
        // What a link's commit may retire is judged at full depth, as the
        // pass judges it.
        let unit_keys: HashSet<String> = folder_corpus
            .retain
            .units
            .iter()
            .map(|u| u.unit.key.clone())
            .collect();
        let file_keys: HashSet<String> = file_corpus
            .retain
            .files
            .iter()
            .filter(|f| !binary.contains(&f.key))
            .map(|f| f.key.clone())
            .collect();

        let total = chain.len() as u64;
        progress.set_step_progress(0, total);
        let mut end: Option<TimelineId> = None;
        for (done, link) in chain.iter().enumerate() {
            if candle_conversation::ingest_cancelled() {
                break;
            }
            let read = match (link, folders, files) {
                (Link::Folder(u), Some(layer), _) => repo_scan::ingest_link(
                    ctx,
                    layer.base,
                    layer.name,
                    &UnitJob {
                        unit: u.unit.clone(),
                        at: u.at.clone(),
                        branches: u.branches.clone(),
                    },
                    &unit_keys,
                    end,
                )?,
                (Link::File(f), _, Some(layer)) => code_read::ingest_link(
                    ctx,
                    layer.base,
                    &FileJob {
                        path: f.file.path.clone(),
                        blob: f.file.blob.clone(),
                        language: f.file.language,
                        at: Some(f.at.clone()),
                        branches: f.branches.clone(),
                    },
                    &file_keys,
                    &self.binary,
                    end,
                )?,
                _ => None,
            };
            if read.is_some() {
                end = read;
            }
            progress.set_step_progress(done as u64 + 1, total);
        }
        if let Some(tl) = end {
            tracing::info!(
                target: "zend::priming_chain",
                links = chain.len(),
                end_timeline = tl.raw(),
                "priming chain built",
            );
        }
        Ok(end)
    }
}

#[cfg(test)]
mod tests {
    use zend_vfs::Oid;

    use super::*;
    use crate::branch_ingest::units::{FolderUnit, TreeFile};
    use crate::repo_scan::types::Language;

    const BLOB: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

    fn unit(dir: &str, branches: &[&str]) -> UnitItem {
        UnitItem {
            unit: FolderUnit {
                repo: dir.split('/').next().unwrap_or("").to_string(),
                dir: dir.to_string(),
                total: 0,
                listed: Vec::new(),
                module_hint: None,
                key: format!("key:{dir}:{}", branches.join(",")),
            },
            at: Oid::parse(BLOB).unwrap(),
            branches: branches.iter().map(|b| b.to_string()).collect(),
        }
    }

    fn file(path: &str, branches: &[&str]) -> FileItem {
        FileItem {
            repo: path.split('/').next().unwrap().to_string(),
            file: TreeFile {
                path: path.to_string(),
                blob: Oid::parse(BLOB).unwrap(),
                size: 1,
                language: Language::Markdown,
            },
            at: Oid::parse(BLOB).unwrap(),
            key: format!("{path}:{}", branches.join(",")),
            branches: branches.iter().map(|b| b.to_string()).collect(),
        }
    }

    fn named(chain: &[Link<'_>]) -> Vec<String> {
        chain
            .iter()
            .map(|l| match l {
                Link::Folder(u) => u.unit.key.clone(),
                Link::File(f) => f.key.clone(),
            })
            .collect()
    }

    /// **Each repository in the workspace's order: its root listing and its
    /// anchors in chain order**, each the version the repository's default
    /// branch holds — whatever order the corpus lists them in, and skipping an
    /// anchor a repository lacks.
    #[test]
    fn the_chain_is_each_repository_in_order() {
        let units = [
            unit("b/", &["main"]),
            unit("a/", &["topic"]),
            unit("a/", &["main", "topic"]),
            unit("a/src/", &["main"]),
        ];
        let files = [
            file("a/CLAUDE.md", &["main"]),
            file("a/README.md", &["topic"]),
            file("a/readme.md", &["main"]),
            file("a/src/README.md", &["main"]),
            file("b/AGENTS.md", &["trunk"]),
            file("a/ARCHITECTURE.md", &["main"]),
        ];
        let repos = [
            ("a".to_string(), "main".to_string()),
            ("b".to_string(), "trunk".to_string()),
        ];
        assert_eq!(
            named(&links(&units, &files, &repos)),
            [
                "key:a/:main,topic",
                "a/readme.md:main",
                "a/ARCHITECTURE.md:main",
                "a/CLAUDE.md:main",
                "b/AGENTS.md:trunk",
            ],
            "b/ is listed on main, not b's default branch trunk, so it is not a link"
        );
    }

    #[test]
    fn nothing_to_chain_is_an_empty_chain() {
        assert!(links(&[], &[], &[("a".to_string(), "main".to_string())]).is_empty());
    }
}

//! Ingesting the repositories' branches — `docs/zend_branch_ingest.md`.
//!
//! One pass, run by the ingest worker at startup and after every fetch that
//! moved a branch: list every record branch's tree ([`walk`]), key every unit
//! by what it shows ([`keys`]), plan each layer against what it already holds
//! ([`plan`]), tombstone what no branch holds at any depth, and ingest what
//! nothing holds yet within the depth bound — each unit read at the commit it
//! was found on.

pub mod filter;
pub mod keys;
pub mod manifest;
pub mod plan;
pub mod prime;
pub mod tie;
pub mod units;
pub mod walk;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use candle_conversation::projection::TimelineId;
use candle_conversation::{ConversationEngine, Sequence};
use zend_vfs::{Repo, Workspace};

use self::filter::IngestScope;
use self::plan::{plan, Committed, Live};
use self::tie::{backfill_commits, retie};
use self::walk::{walk, Corpus, FileItem, RepoBranches, TreeCache};
use crate::code_read::{self, is_upload_path, FileJob};
use crate::ingest::IngestMode;
use crate::loading::LoadProgress;
use crate::refresh_ctx::RefreshContext;
use crate::repo_scan::{self, UnitJob};

/// The tools whose call can move a branch the record holds — by publishing
/// to origin, whose push moves the tracking ref as it lands, or by fetching
/// from it. A round that ran one wakes the ingest worker: no origin probe
/// would see a difference afterwards.
const BRANCH_MOVERS: [&str; 7] = [
    "git_commit",
    "git_merge",
    "git_ref",
    "git_switch",
    "git_reset",
    "git_push",
    "git_fetch",
];

/// Whether a call of the tool `name` can move a branch the record holds.
pub fn moves_branches(name: &str) -> bool {
    BRANCH_MOVERS.contains(&name)
}

/// One layer a pass ingests.
pub struct LayerPass<'a> {
    pub name: &'a str,
    /// [`IngestMode::Folders`] (`repo_map`) or [`IngestMode::Files`]
    /// (`code_reading`); a raw layer is not ingested from branches.
    pub mode: IngestMode,
    pub scope: IngestScope,
    /// The layer's prefilled template its units' conversations fork from.
    pub base: &'a Mutex<Sequence>,
}

/// What the branch ingest keeps between passes.
#[derive(Default)]
pub struct BranchIngest {
    /// The trees of the tips last walked.
    trees: Mutex<TreeCache>,
    /// File keys found to be binary: never read again.
    binary: Mutex<HashSet<String>>,
}

impl BranchIngest {
    /// One pass over `layers`. Returns whether any layer changed.
    pub fn pass(
        &self,
        ctx: &RefreshContext<'_>,
        workspace: &Workspace,
        layers: &[LayerPass<'_>],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<bool> {
        let (repos, unreadable) = record_branches(workspace);
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        let mut corpora: HashMap<IngestScope, (Corpus, Vec<String>)> = HashMap::new();
        let mut changed = false;
        for layer in layers {
            if candle_conversation::ingest_cancelled() {
                break;
            }
            let (corpus, failed) = corpora
                .entry(layer.scope.clone())
                .or_insert_with(|| walk(&repos, &layer.scope, &mut trees));
            // A repository that could not be read this pass has units nobody
            // looked for — they are not gone.
            let held_back: Vec<&str> = failed
                .iter()
                .chain(&unreadable)
                .map(String::as_str)
                .collect();
            changed |= match layer.mode {
                IngestMode::Files => self.files(ctx, layer, corpus, &held_back, progress)?,
                IngestMode::Folders => units(ctx, layer, corpus, &held_back, progress)?,
                IngestMode::Raw => false,
            };
        }
        trees.retain_tips(&repos);
        Ok(changed)
    }

    fn files(
        &self,
        ctx: &RefreshContext<'_>,
        layer: &LayerPass<'_>,
        corpus: &Corpus,
        held_back: &[&str],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<bool> {
        // A file found to be binary is never read, so it is not a unit: its
        // path's older versions must not wait on it as their replacement.
        let binary = self.binary.lock().unwrap().clone();
        let files = readable(&corpus.ingest.files, &binary);
        let retained = readable(&corpus.retain.files, &binary);
        let live: Vec<Live<'_>> = files
            .iter()
            .map(|f| Live {
                key: &f.key,
                subject: &f.file.path,
            })
            .collect();
        let retain: HashSet<&str> = retained.iter().map(|f| f.key.as_str()).collect();
        let committed: Vec<Committed> = code_read::committed(ctx.engine)
            .into_iter()
            .filter(|c| !in_repos(&c.subject, held_back))
            .collect();
        let p = plan(&live, &retain, &committed);
        tombstone(ctx.engine, layer.name, &p.tombstone);
        // A held file past the depth bound keeps its branches and its commit
        // current as well: the full-depth walk found it.
        let branches: HashMap<&str, &[String]> = retained
            .iter()
            .map(|f| (f.key.as_str(), &f.branches[..]))
            .collect();
        let held = still_held(&committed, &p.tombstone);
        retie(ctx.engine, layer.name, &held, &branches);
        let found: HashMap<&str, String> = retained
            .iter()
            .map(|f| (f.key.as_str(), f.at.to_string()))
            .collect();
        backfill_commits(ctx.engine, layer.name, &held, &found);
        let jobs: Vec<FileJob> = p
            .queued
            .iter()
            .map(|&i| files[i])
            .map(|f| FileJob {
                path: f.file.path.clone(),
                blob: f.file.blob.clone(),
                language: f.file.language,
                at: Some(f.at.clone()),
                branches: f.branches.clone(),
            })
            .collect();
        tracing::info!(
            target: "zend::branch_ingest",
            layer = layer.name,
            live = live.len(),
            retained = retain.len(),
            committed = committed.len(),
            queued = jobs.len(),
            tombstoned = p.tombstone.len(),
            "planned the layer against every branch",
        );
        if !jobs.is_empty() {
            let keys: HashSet<String> = retain.iter().map(|k| k.to_string()).collect();
            code_read::ingest_jobs(ctx, &jobs, &keys, progress, layer.base, &self.binary)?;
        }
        Ok(!jobs.is_empty() || !p.tombstone.is_empty())
    }
}

/// `items` less the files found to be binary.
fn readable<'c>(items: &'c [FileItem], binary: &HashSet<String>) -> Vec<&'c FileItem> {
    items.iter().filter(|f| !binary.contains(&f.key)).collect()
}

fn units(
    ctx: &RefreshContext<'_>,
    layer: &LayerPass<'_>,
    corpus: &Corpus,
    held_back: &[&str],
    progress: &Arc<LoadProgress>,
) -> anyhow::Result<bool> {
    let live: Vec<Live<'_>> = corpus
        .ingest
        .units
        .iter()
        .map(|u| Live {
            key: &u.unit.key,
            subject: &u.unit.dir,
        })
        .collect();
    let retain: HashSet<&str> = corpus
        .retain
        .units
        .iter()
        .map(|u| u.unit.key.as_str())
        .collect();
    let committed: Vec<Committed> = repo_scan::committed(ctx.engine)
        .into_iter()
        .filter(|c| !in_repos(&c.subject, held_back))
        .collect();
    let p = plan(&live, &retain, &committed);
    tombstone(ctx.engine, layer.name, &p.tombstone);
    // A held folder past the depth bound keeps its branches current as well.
    let branches: HashMap<&str, &[String]> = corpus
        .retain
        .units
        .iter()
        .map(|u| (u.unit.key.as_str(), &u.branches[..]))
        .collect();
    retie(
        ctx.engine,
        layer.name,
        &still_held(&committed, &p.tombstone),
        &branches,
    );
    let jobs: Vec<UnitJob> = p
        .queued
        .iter()
        .map(|&i| &corpus.ingest.units[i])
        .map(|u| UnitJob {
            unit: u.unit.clone(),
            at: u.at.clone(),
            branches: u.branches.clone(),
        })
        .collect();
    tracing::info!(
        target: "zend::branch_ingest",
        layer = layer.name,
        live = live.len(),
        retained = retain.len(),
        committed = committed.len(),
        queued = jobs.len(),
        tombstoned = p.tombstone.len(),
        "planned the layer against every branch",
    );
    if !jobs.is_empty() {
        let keys: HashSet<String> = retain.iter().map(|k| k.to_string()).collect();
        repo_scan::ingest_units(ctx, &jobs, &keys, progress, layer.name, layer.base)?;
    }
    Ok(!jobs.is_empty() || !p.tombstone.is_empty())
}

/// Every git repository of the workspace with its record branches, and the
/// names of those whose branches could not be read. The uploads folder and
/// any other folder not under git have no branches and are not among either.
///
/// Whether a folder is under git is read from the folder — a `.git` in it —
/// never from a failure to open it: a git that cannot be spawned, a lock held
/// for a moment, an ownership refusal all fail an open, and a repository
/// read as not under git would have every unit it ever had tombstoned.
pub(crate) fn record_branches(workspace: &Workspace) -> (Vec<RepoBranches>, Vec<String>) {
    let mut repos = Vec::new();
    let mut unreadable = Vec::new();
    for spec in workspace.repos() {
        if is_upload_path(&spec.name) || !spec.dir.join(".git").exists() {
            continue;
        }
        let read = Repo::open(&spec.dir).and_then(|repo| RepoBranches::read(&spec.name, repo));
        match read {
            Ok(branches) => repos.push(branches),
            Err(e) => {
                tracing::warn!(
                    target: "zend::branch_ingest",
                    repo = %spec.name,
                    "its branches could not be read; it sits this pass out: {e}",
                );
                unreadable.push(spec.name.clone());
            }
        }
    }
    (repos, unreadable)
}

/// `committed` without the conversations this pass tombstoned.
fn still_held(committed: &[Committed], tombstoned: &[TimelineId]) -> Vec<Committed> {
    committed
        .iter()
        .filter(|c| !tombstoned.contains(&c.timeline))
        .cloned()
        .collect()
}

/// Whether the workspace-relative `subject` lies in one of `repos`.
fn in_repos(subject: &str, repos: &[&str]) -> bool {
    let repo = subject.split('/').next().unwrap_or("");
    repos.contains(&repo)
}

fn tombstone(engine: &Mutex<ConversationEngine>, layer: &str, timelines: &[TimelineId]) {
    if timelines.is_empty() {
        return;
    }
    let e = engine.lock().unwrap();
    for &tl in timelines {
        if let Err(err) = e.tombstone_timeline(tl) {
            tracing::warn!(
                target: "zend::branch_ingest",
                layer,
                timeline = tl.raw(),
                "tombstone of a unit no branch holds failed: {err:#}",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use zend_vfs::RepoSpec;

    use super::*;

    /// Every `git_*` writer can move a branch; the readers and every other
    /// tool cannot. Pinned against the registry so a renamed tool fails here
    /// rather than silently never waking the ingest.
    #[test]
    fn only_the_git_writers_move_branches() {
        for name in BRANCH_MOVERS {
            assert!(
                zend_tools::registry::find(name).is_some(),
                "{name} is not a tool"
            );
            assert!(moves_branches(name));
        }
        for name in [
            "git_status",
            "git_log",
            "git_show",
            "git_grep",
            "git_refs",
            "file_read",
            "write",
        ] {
            assert!(!moves_branches(name), "{name}");
        }
    }

    /// **A folder with a `.git` that cannot be read is held back, never read
    /// as a folder outside git** — the uploads folder and a plain folder are
    /// simply not repositories.
    #[test]
    fn a_repository_that_cannot_be_opened_is_held_back() {
        let root = tempfile::tempdir().unwrap();
        for name in ["broken", "plain", "uploads"] {
            std::fs::create_dir_all(root.path().join(name)).unwrap();
        }
        // A `.git` that is no repository: opening it fails as a real one can.
        std::fs::write(root.path().join("broken/.git"), "gitdir: nowhere\n").unwrap();
        let workspace = Workspace::new(
            root.path(),
            vec![
                RepoSpec::named("broken"),
                RepoSpec::named("plain"),
                RepoSpec::named("uploads"),
            ],
        )
        .unwrap();
        let (repos, unreadable) = record_branches(&workspace);
        assert!(repos.is_empty());
        assert_eq!(unreadable, ["broken"]);
    }

    #[test]
    fn a_subject_is_in_the_repository_its_first_segment_names() {
        assert!(in_repos("candle/zend/src/", &["candle"]));
        assert!(in_repos("candle/a.rs", &["mind", "candle"]));
        assert!(!in_repos("candle-extra/a.rs", &["candle"]));
        assert!(!in_repos(".", &["candle"]));
        assert!(!in_repos("candle/a.rs", &[]));
    }
}

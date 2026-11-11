//! Ingesting the repositories' branches — `docs/zend_branch_ingest.md`.
//!
//! One pass, run by the ingest worker at startup and after every fetch that
//! moved a branch: list the tree of every record branch the whitelist names
//! ([`monitored`], [`walk`]), key every unit
//! by what it shows ([`keys`]), plan each layer against what it already holds
//! ([`plan`]), tombstone what no branch holds at any depth, and ingest what
//! nothing holds yet within the depth bound — each unit read at the commit it
//! was found on. The pass begins and ends by sealing a face for every file and
//! folder unit that has none ([`crate::faces`]).

pub mod filter;
pub mod keys;
pub mod manifest;
pub mod monitored;
pub mod order;
pub mod plan;
pub mod prime;
pub mod reader;
pub mod tie;
pub mod units;
pub mod walk;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::{ConversationEngine, Sequence};
use zend_vfs::Workspace;

use self::filter::IngestScope;
use self::monitored::{keep_monitored, monitored};
use self::plan::{plan, Committed, Live};
use self::tie::{backfill_commits, retie};
use self::walk::{walk, Corpus, FileItem, RepoBranches, TreeCache};
use crate::calibrate::CalibrateSpec;
use crate::code_read::{self, is_upload_path, FileJob};
use crate::faces;
use crate::ingest::IngestMode;
use crate::journal::opening::SeedRead;
use crate::layers::plan::Below;
use crate::layers::{Ground, LayerRun, LayerSpec, Reach};
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
    pub base: &'a Sequence,
    /// How a [`IngestMode::Derived`] layer's units are read.
    pub spec: Option<&'a LayerSpec>,
    /// How a [`IngestMode::Derived`] layer's units are calibrated as they
    /// commit (`crate::calibrate::calibrate_committed`), for a layer that
    /// declares a `calibrate:` block. A file layer's units are not: the branch
    /// pass reads them all first, and `Session::calibrate_layers` calibrates
    /// them after it.
    pub calibrate: Option<&'a CalibrateSpec>,
    /// The scoped groups of the layers of units below a derived layer, by
    /// layer name — what its units may retrieve, on their branches.
    pub sees: &'a [(String, GroupId)],
}

/// What the branch ingest keeps between passes.
#[derive(Default)]
pub struct BranchIngest {
    /// The trees of the tips last walked.
    trees: Mutex<TreeCache>,
    /// File keys found to be binary: never read again.
    binary: Mutex<HashSet<String>>,
    /// The key of every link of the chain [`Self::prime`] last built, in
    /// chain order: what [`Self::chain_moved`] holds the repositories against,
    /// so a link that changes is read into the chain again before the pass.
    primed: Mutex<Vec<String>>,
    /// What that chain read, for the journal's opening
    /// ([`crate::journal::opening`]).
    seed_read: Mutex<SeedRead>,
}

impl BranchIngest {
    /// What the chain [`Self::prime`] last built read: empty before the first
    /// prime, and on a member with no chain.
    pub fn seed_read(&self) -> SeedRead {
        self.seed_read
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// The file keys found to be binary: this process's, and every one a
    /// reading recorded in the substrate (`code_read::binary_mark`) — on this
    /// member before a restart, or on another member, whose verdict arrived
    /// with replication.
    fn known_binary(&self, ctx: &RefreshContext<'_>) -> HashSet<String> {
        let conversation = ctx.engine.conversation();
        let mut binary = self.binary.lock().unwrap();
        binary.extend(code_read::binary_mark::recorded(&conversation));
        binary.clone()
    }

    /// One pass over `layers`. Returns whether any layer changed.
    pub fn pass(
        &self,
        ctx: &RefreshContext<'_>,
        workspace: &Workspace,
        layers: &[LayerPass<'_>],
        progress: &Arc<LoadProgress>,
    ) -> anyhow::Result<bool> {
        // Faces first for what is already read: a pass can queue thousands of
        // readings, and a unit that waited for the pass's end to get its face
        // would be placed with its request for as long as the pass runs.
        faces::fill(ctx, &readings(ctx, layers));
        let (repos, unreadable) = record_branches(workspace);
        let mut trees = self.trees.lock().unwrap_or_else(|e| e.into_inner());
        let mut corpora: HashMap<IngestScope, (Corpus, Vec<String>)> = HashMap::new();
        let mut changed = false;
        // What stands in each layer, carried up to the layers built on it.
        let mut ground = Ground {
            defaults: repos
                .iter()
                .filter_map(|r| {
                    r.tips
                        .first()
                        .map(|t| (r.name.clone(), t.name.as_str().to_string()))
                })
                .collect(),
            ..Ground::default()
        };
        for layer in layers {
            if crate::ingest_cancel::cancelled() {
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
                IngestMode::Files => {
                    let above: Vec<&LayerPass<'_>> = layers
                        .iter()
                        .filter(|l| l.mode == IngestMode::Derived)
                        .collect();
                    let read = Reading {
                        corpus,
                        held_back: &held_back,
                        progress,
                        above: &above,
                    };
                    self.files(ctx, layer, &read, &mut ground)?
                }
                IngestMode::Folders => units(ctx, layer, corpus, &held_back, progress)?,
                IngestMode::Raw => false,
                IngestMode::Derived => match layer.spec {
                    Some(spec) => {
                        let run = LayerRun {
                            spec,
                            base: layer.base,
                            calibrate: layer.calibrate,
                            sees: layer.sees,
                        };
                        crate::layers::pass_layer(ctx, &run, &mut ground, Reach::Default)?
                    }
                    None => false,
                },
            };
        }
        // The default branch has been read all the way up; every other
        // branch's units now, bottom up again, each layer planned afresh
        // against what its layer below holds after this climb.
        for layer in layers.iter().filter(|l| l.mode == IngestMode::Derived) {
            if crate::ingest_cancel::cancelled() {
                break;
            }
            if let Some(spec) = layer.spec {
                let run = LayerRun {
                    spec,
                    base: layer.base,
                    calibrate: layer.calibrate,
                    sees: layer.sees,
                };
                changed |= crate::layers::pass_layer(ctx, &run, &mut ground, Reach::All)?;
            }
        }
        trees.retain_tips(&repos);
        faces::fill(ctx, &readings(ctx, layers));
        Ok(changed)
    }

    fn files(
        &self,
        ctx: &RefreshContext<'_>,
        layer: &LayerPass<'_>,
        read: &Reading<'_>,
        ground: &mut Ground,
    ) -> anyhow::Result<bool> {
        let (corpus, held_back, progress) = (read.corpus, read.held_back, read.progress);
        // A file found to be binary is never read, so it is not a unit: its
        // path's older versions must not wait on it as their replacement.
        let binary = self.known_binary(ctx);
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
        // A file a peer has just read stands as read here while its turns
        // arrive: neither queued again nor tombstoned as a duplicate.
        let planned: Vec<Committed> = committed
            .iter()
            .cloned()
            .chain(code_read::arriving(ctx.engine))
            .filter(|c| !in_repos(&c.subject, held_back))
            .collect();
        let p = plan(&live, &retain, &planned);
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
        let jobs: Vec<FileJob> = order::in_read_order(
            p.queued
                .iter()
                .map(|&i| files[i])
                .map(|f| FileJob {
                    path: f.file.path.clone(),
                    blob: f.file.blob.clone(),
                    language: f.file.language,
                    at: Some(f.at.clone()),
                    branches: f.branches.clone(),
                })
                .collect(),
            |j| j.path.as_str(),
        );
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
        // Stale units above go now, before anything is read: a file being read
        // again has no current reading, and every unit built on its old one —
        // and on anything that depended on it — is tombstoned in this sweep,
        // not once the pass reaches its layer.
        ground_files(
            ctx, layer.name, &files, &retained, held_back, corpus, ground,
        );
        for upper in read.above {
            if let Some(spec) = upper.spec {
                let run = LayerRun {
                    spec,
                    base: upper.base,
                    calibrate: upper.calibrate,
                    sees: upper.sees,
                };
                crate::layers::sweep_layer(ctx, &run, ground);
            }
        }
        if !jobs.is_empty() {
            let keys: HashSet<String> = retain.iter().map(|k| k.to_string()).collect();
            // **Every file is read before any is calibrated.** A unit's 24
            // calibration questions cost many times its reading, and calibrated
            // inline each worker held its next file back for them: a whole
            // tree re-read under a new key took a day to place its first
            // current reading of most files. The layer's calibration pass
            // (`Session::calibrate_layers`) takes the committed units up after
            // this pass, and at the start of every one.
            code_read::ingest_jobs(ctx, &jobs, &keys, progress, layer.base, &self.binary, None)?;
        }
        ground_files(
            ctx, layer.name, &files, &retained, held_back, corpus, ground,
        );
        Ok(!jobs.is_empty() || !p.tombstone.is_empty())
    }
}

/// What the `code_reading` pass reads with: its corpus, the repositories held
/// back this pass, the progress sink, and the layers of units above it, swept
/// before it reads.
struct Reading<'a> {
    corpus: &'a Corpus,
    held_back: &'a [&'a str],
    progress: &'a Arc<LoadProgress>,
    above: &'a [&'a LayerPass<'a>],
}

/// Leave what `code_reading` holds current in `ground`, for the layers built
/// on it: every file version a branch holds, the files within the depth bound,
/// and each committed file read whose version a branch still holds — read
/// again after this pass's pool, so what it just read stands too. A file of a
/// repository held back this pass stands as it was: nothing looked for it.
///
/// Built afresh each time — the sweep before the pool and the pass after it
/// both call it — so a version is never listed twice.
fn ground_files(
    ctx: &RefreshContext<'_>,
    layer: &str,
    files: &[&FileItem],
    retained: &[&FileItem],
    held_back: &[&str],
    corpus: &Corpus,
    ground: &mut Ground,
) {
    ground.live.clear();
    ground.held_back = held_back.iter().map(|r| r.to_string()).collect();
    for f in retained {
        ground
            .live
            .entry(f.file.path.clone())
            .or_default()
            .push((f.key.clone(), f.branches.clone()));
    }
    // Every other file a branch holds is a version an analysis can depend on.
    for (path, versions) in &corpus.retain.unread {
        ground
            .live
            .entry(path.clone())
            .or_default()
            .extend(versions.iter().cloned());
    }
    ground.ingest = files.iter().map(|f| f.key.clone()).collect();
    let on: HashMap<&str, &FileItem> = retained.iter().map(|f| (f.key.as_str(), *f)).collect();
    let standing = code_read::committed(ctx.engine)
        .into_iter()
        .filter_map(|c| {
            let found = on.get(c.key.as_str());
            if found.is_none() && !in_repos(&c.subject, held_back) {
                return None;
            }
            Some(Below {
                file_key: Some(c.key.clone()),
                key: c.key,
                subject: c.subject,
                timeline: Some(c.timeline),
                branches: found.map(|f| f.branches.clone()).unwrap_or_default(),
                at: found.map(|f| f.at.to_string()),
            })
        })
        .collect();
    ground.standing.insert(layer.to_string(), standing);
}

/// Every committed unit of the file and folder layers among `layers` — what a
/// dialogue can place, and so what has a face.
fn readings(
    ctx: &RefreshContext<'_>,
    layers: &[LayerPass<'_>],
) -> Vec<(faces::Reading, Committed)> {
    let has = |mode: IngestMode| layers.iter().any(|l| l.mode == mode);
    let mut out = Vec::new();
    if has(IngestMode::Files) {
        out.extend(
            code_read::committed(ctx.engine)
                .into_iter()
                .map(|c| (faces::Reading::File, c)),
        );
    }
    if has(IngestMode::Folders) {
        out.extend(
            repo_scan::committed(ctx.engine)
                .into_iter()
                .map(|c| (faces::Reading::Folder, c)),
        );
    }
    out
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

/// Every git repository of the workspace with the record branches the ingest
/// reads — those the whitelist names ([`monitored`]) — and the names of those
/// whose branches could not be read. The uploads folder and any other folder
/// not under git have no branches and are not among either.
///
/// A branch off the whitelist is dropped here, before any tree is walked, so
/// no unit is found on it: a reading only such a branch held is planned as
/// gone and tombstoned, and the layers above are built on the listed
/// branches alone.
///
/// Whether a folder is under git is read from the folder — a `.git` in it —
/// never from a failure to open it: a git that cannot be spawned, a lock held
/// for a moment, an ownership refusal all fail an open, and a repository
/// read as not under git would have every unit it ever had tombstoned. A
/// repository read from the substrate is always under git.
pub(crate) fn record_branches(workspace: &Workspace) -> (Vec<RepoBranches>, Vec<String>) {
    let mut repos = Vec::new();
    let mut unreadable = Vec::new();
    for spec in workspace.repos() {
        if is_upload_path(&spec.name) || !spec.is_git() {
            continue;
        }
        let read = spec
            .open()
            .and_then(|repo| RepoBranches::read(&spec.name, repo));
        match read {
            Ok(mut branches) => {
                keep_monitored(&mut branches.tips, monitored());
                repos.push(branches);
            }
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

fn tombstone(engine: &ConversationEngine, layer: &str, timelines: &[TimelineId]) {
    if timelines.is_empty() {
        return;
    }
    let e = engine;
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
    use std::process::Command;

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

    /// **Only the whitelisted branches are walked**: a repository on `main`
    /// with a `topic` branch is read on `main` alone, so what only `topic`
    /// holds is no unit and its readings plan as gone.
    #[test]
    fn a_branch_off_the_whitelist_is_not_walked() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("r");
        std::fs::create_dir_all(&dir).unwrap();
        let git = |args: &[&str]| {
            let out = Command::new("git")
                .arg("-C")
                .arg(&dir)
                .args(["-c", "core.hooksPath=", "-c", "user.name=T"])
                .args(["-c", "user.email=t@x"])
                .args(args)
                .output()
                .expect("git runs");
            assert!(out.status.success(), "git {args:?}");
        };
        git(&["init", "-q"]);
        git(&["symbolic-ref", "HEAD", "refs/heads/main"]);
        std::fs::write(dir.join("a.rs"), "fn a() {}\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "main"]);
        git(&["checkout", "-q", "-b", "topic"]);
        std::fs::write(dir.join("b.rs"), "fn b() {}\n").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "topic"]);
        let workspace = Workspace::new(root.path(), vec![RepoSpec::named("r")]).unwrap();
        let (repos, unreadable) = record_branches(&workspace);
        assert!(unreadable.is_empty());
        let tips: Vec<&str> = repos[0].tips.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(tips, ["main"]);
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

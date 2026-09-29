//! Every branch's tree → the corpus of units the two layers ingest
//! (`docs/zend_branch_ingest.md` §5).
//!
//! Each record branch's tip is listed once — a tree two tips share is listed
//! once, and kept between passes while a tip still names it — and filtered by
//! the layer's [`IngestScope`]. A unit found on several branches is one unit,
//! found first on the repository's default branch, so the commit its bytes
//! are read from is the default branch's wherever it can be — and it records
//! every branch it was found on.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use zend_vfs::vfs::Tree;
use zend_vfs::{GitError, Oid, RecordBranch, Repo, Rev};

use super::filter::IngestScope;
use super::keys::file_key;
use super::manifest::Hints;
use super::units::{folder_units, FolderUnit, TreeFile};

/// The branches a conversation starts on, most preferred first — they lead
/// the walk, so a unit on one of them is read from it.
const PREFERRED: [&str; 2] = ["main", "master"];

/// One git repository's record branches, preferred first.
pub struct RepoBranches {
    pub name: String,
    pub repo: Repo,
    pub tips: Vec<RecordBranch>,
}

impl RepoBranches {
    /// `repo`'s record branches, in walk order: `main`, then `master`, then
    /// the rest by name.
    pub fn read(name: &str, repo: Repo) -> Result<Self, GitError> {
        let mut tips = repo.record_branches()?;
        tips.sort_by_key(|b| {
            let rank = PREFERRED
                .iter()
                .position(|p| *p == b.name.as_str())
                .unwrap_or(PREFERRED.len());
            (rank, b.name.as_str().to_string())
        });
        Ok(Self {
            name: name.to_string(),
            repo,
            tips,
        })
    }
}

/// A file unit, and the commit its bytes are read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileItem {
    pub repo: String,
    pub file: TreeFile,
    pub at: Oid,
    /// [`file_key`].
    pub key: String,
    /// Every branch whose tip holds this version, in walk order.
    pub branches: Vec<String>,
}

/// A folder unit, and the commit its listing and files are read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitItem {
    pub unit: FolderUnit,
    pub at: Oid,
    /// Every branch whose tip lists the folder this way, in walk order.
    pub branches: Vec<String>,
}

/// Every distinct unit on any branch, in walk order.
#[derive(Debug, Default)]
pub struct Corpus {
    pub files: Vec<FileItem>,
    pub units: Vec<UnitItem>,
}

/// The trees of the tips walked, by commit, kept between passes — and each
/// manifest version's hint, which a blob id fixes for good.
#[derive(Default)]
pub struct TreeCache {
    trees: HashMap<Oid, Arc<Tree>>,
    hints: Hints,
}

impl TreeCache {
    fn tree(&mut self, repo: &Repo, commit: &Oid) -> Result<Arc<Tree>, GitError> {
        if let Some(tree) = self.trees.get(commit) {
            return Ok(Arc::clone(tree));
        }
        let (_, tree_id) = repo
            .blobs()
            .commit_of(&Rev::Oid(commit.clone()))?
            .ok_or_else(|| GitError::invalid(format!("the repository holds no commit {commit}")))?;
        let tree = Arc::new(Tree::read(repo, &tree_id)?);
        self.trees.insert(commit.clone(), Arc::clone(&tree));
        Ok(tree)
    }

    /// Let go of every tree no tip of `repos` names — once a pass has walked
    /// every scope it walks.
    pub fn retain_tips(&mut self, repos: &[RepoBranches]) {
        let keep: HashSet<&Oid> = repos
            .iter()
            .flat_map(|r| r.tips.iter().map(|t| &t.tip))
            .collect();
        self.trees.retain(|commit, _| keep.contains(commit));
    }
}

/// The corpus of `repos` under `scope`.
///
/// A repository with a tree that cannot be listed sits the pass out whole:
/// none of its units are in the corpus — a corpus holding some of a
/// repository's branches would read the rest as gone — and its name is
/// returned beside the corpus so the pass that plans from it holds its units
/// back rather than reading them as deleted.
pub fn walk(
    repos: &[RepoBranches],
    scope: &IngestScope,
    cache: &mut TreeCache,
) -> (Corpus, Vec<String>) {
    let mut corpus = Corpus::default();
    let mut failed = Vec::new();
    for repo in repos.iter().filter(|r| scope.reaches(&r.name)) {
        match walk_repo(repo, scope, cache) {
            Ok(of_repo) => {
                corpus.files.extend(of_repo.files);
                corpus.units.extend(of_repo.units);
            }
            Err(e) => {
                tracing::warn!(
                    target: "zend::branch_ingest",
                    repo = %repo.name,
                    "a tree could not be listed; the repository sits this pass out: {e}",
                );
                failed.push(repo.name.clone());
            }
        }
    }
    (corpus, failed)
}

/// One repository's units on every branch, or the error that kept one of its
/// trees from being listed.
fn walk_repo(
    repo: &RepoBranches,
    scope: &IngestScope,
    cache: &mut TreeCache,
) -> Result<Corpus, GitError> {
    let mut corpus = Corpus::default();
    // Key → index into the corpus, so a unit found again on a later branch
    // adds that branch to the one it already is.
    let mut seen_files: HashMap<String, usize> = HashMap::new();
    let mut seen_units: HashMap<String, usize> = HashMap::new();
    let blobs = repo.repo.blobs();
    for tip in &repo.tips {
        let branch = tip.name.as_str().to_string();
        let tree = cache.tree(&repo.repo, &tip.tip)?;
        let files: Vec<TreeFile> = tree
            .files()
            .filter_map(|(path, blob, size)| {
                let language = scope.admits(&repo.name, path, size)?;
                Some(TreeFile {
                    path: format!("{}/{path}", repo.name),
                    blob: blob.clone(),
                    size,
                    language,
                })
            })
            .collect();
        let hints = &mut cache.hints;
        let mut hint_of = |f: &TreeFile| {
            let name = f.path.rsplit('/').next().unwrap_or(&f.path);
            hints.of(name, &f.blob, |blob| blobs.read_blob(blob).ok().flatten())
        };
        for unit in folder_units(&repo.name, &tree, &files, &mut hint_of) {
            match seen_units.get(&unit.key) {
                Some(&at) => corpus.units[at].branches.push(branch.clone()),
                None => {
                    seen_units.insert(unit.key.clone(), corpus.units.len());
                    corpus.units.push(UnitItem {
                        unit,
                        at: tip.tip.clone(),
                        branches: vec![branch.clone()],
                    });
                }
            }
        }
        for file in files {
            let key = file_key(&file.path, &file.blob);
            match seen_files.get(&key) {
                Some(&at) => corpus.files[at].branches.push(branch.clone()),
                None => {
                    seen_files.insert(key.clone(), corpus.files.len());
                    corpus.files.push(FileItem {
                        repo: repo.name.clone(),
                        file,
                        at: tip.tip.clone(),
                        key,
                        branches: vec![branch.clone()],
                    });
                }
            }
        }
    }
    Ok(corpus)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["-c", "core.hooksPath=", "-c", "core.autocrlf=false"])
            .args(["-c", "user.name=T", "-c", "user.email=t@x"])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    fn write(dir: &Path, rel: &str, body: &str) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn commit(dir: &Path, message: &str) {
        git(dir, &["add", "-A"]);
        git(dir, &["commit", "-q", "--allow-empty", "-m", message]);
    }

    /// A repository `r` on `main` holding `a.rs` and `src/b.rs`, with a
    /// `topic` branch that changes `a.rs` and adds `src/c.rs`.
    fn two_branches(root: &Path) -> RepoBranches {
        let dir = root.join("r");
        std::fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q"]);
        git(&dir, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        write(&dir, "a.rs", "fn a() {}\n");
        write(&dir, "src/b.rs", "fn b() {}\n");
        write(&dir, ".github/ci.yml", "on: push\n");
        write(&dir, "LICENSE", "MIT\n");
        commit(&dir, "main");
        git(&dir, &["checkout", "-q", "-b", "topic"]);
        write(&dir, "a.rs", "fn a2() {}\n");
        write(&dir, "src/c.rs", "fn c() {}\n");
        commit(&dir, "topic");
        git(&dir, &["checkout", "-q", "main"]);
        RepoBranches::read("r", Repo::open(&dir).unwrap()).unwrap()
    }

    fn paths(corpus: &Corpus) -> Vec<(&str, &str)> {
        corpus
            .files
            .iter()
            .map(|f| (f.file.path.as_str(), f.file.blob.as_str()))
            .collect()
    }

    /// Walk `repo` alone.
    fn walk_one(
        repo: &RepoBranches,
        scope: &IngestScope,
        cache: &mut TreeCache,
    ) -> (Corpus, Vec<String>) {
        walk(std::slice::from_ref(repo), scope, cache)
    }

    /// **A repository with a tree that cannot be listed sits the pass out
    /// whole** — none of its branches' units, not only the failing one's.
    #[test]
    fn a_repository_that_fails_part_way_sits_the_pass_out_whole() {
        let root = tempfile::tempdir().unwrap();
        let mut repo = two_branches(root.path());
        let good = walk_one(
            &repo,
            &IngestScope::new("", None),
            &mut TreeCache::default(),
        )
        .0;
        assert!(!good.files.is_empty() && !good.units.is_empty());

        repo.tips[1].tip = Oid::parse("1234567890123456789012345678901234567890").unwrap();
        let (corpus, failed) = walk_one(
            &repo,
            &IngestScope::new("", None),
            &mut TreeCache::default(),
        );
        assert_eq!(failed, ["r"]);
        assert!(
            corpus.files.is_empty() && corpus.units.is_empty(),
            "main's units too"
        );
    }

    /// **Every branch is walked, each distinct file once**: a file the same
    /// on both is one unit, found on `main`; a file that differs is one unit
    /// per version; a file only one branch has is found there. Hidden and
    /// unlisted files are not read.
    #[test]
    fn every_branch_is_walked_and_each_version_found_once() {
        let root = tempfile::tempdir().unwrap();
        let repo = two_branches(root.path());
        assert_eq!(
            repo.tips
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["main", "topic"]
        );
        let main = repo.tips[0].tip.clone();
        let topic = repo.tips[1].tip.clone();
        let (corpus, failed) = walk_one(
            &repo,
            &IngestScope::new("", None),
            &mut TreeCache::default(),
        );
        assert!(failed.is_empty());
        let a_main = git(repo.repo.dir(), &["rev-parse", "main:a.rs"]);
        let a_topic = git(repo.repo.dir(), &["rev-parse", "topic:a.rs"]);
        let got = paths(&corpus);
        assert_eq!(
            got.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            ["r/a.rs", "r/src/b.rs", "r/a.rs", "r/src/c.rs"]
        );
        assert_eq!(got[0].1, a_main.trim());
        assert_eq!(got[2].1, a_topic.trim());
        let at: Vec<&Oid> = corpus.files.iter().map(|f| &f.at).collect();
        assert_eq!(at, [&main, &main, &topic, &topic]);
        let branches: Vec<&[String]> = corpus.files.iter().map(|f| &f.branches[..]).collect();
        assert_eq!(
            branches,
            [
                &["main".to_string()][..],
                &["main".to_string(), "topic".to_string()][..],
                &["topic".to_string()][..],
                &["topic".to_string()][..],
            ],
            "src/b.rs is the same on both, so it is one unit that both branches hold",
        );
        assert_eq!(corpus.files[0].key, format!("r/a.rs@{}", a_main.trim()));
    }

    /// **Folders are units once per distinct listing**, every one inside a
    /// repository — there is no unit for the workspace, which no `file_list`
    /// lists; `src/` lists differently on `topic`, so it is two units.
    #[test]
    fn a_folder_is_one_unit_per_distinct_listing() {
        let root = tempfile::tempdir().unwrap();
        let repo = two_branches(root.path());
        let (corpus, _) = walk_one(
            &repo,
            &IngestScope::new("", None),
            &mut TreeCache::default(),
        );
        let dirs: Vec<(&str, &Oid)> = corpus
            .units
            .iter()
            .map(|u| (u.unit.dir.as_str(), &u.at))
            .collect();
        let (main, topic) = (&repo.tips[0].tip, &repo.tips[1].tip);
        assert_eq!(
            dirs,
            [("r/", main), ("r/src/", main), ("r/src/", topic)],
            "r/ lists the same entries on both, so it is one unit",
        );
        let branches: Vec<Vec<&str>> = corpus
            .units
            .iter()
            .map(|u| u.branches.iter().map(String::as_str).collect())
            .collect();
        assert_eq!(
            branches,
            [vec!["main", "topic"], vec!["main"], vec!["topic"]],
            "each unit names every branch that lists the folder its way",
        );
        assert_eq!(
            corpus.units[0].unit.listed,
            ["r/LICENSE", "r/a.rs", "r/src/"],
            "the listing is what file_list shows: LICENSE is not read but is listed"
        );
    }

    /// **The depth bound and the scope folder apply to every branch.**
    #[test]
    fn the_scope_applies_to_every_branch() {
        let root = tempfile::tempdir().unwrap();
        let repo = two_branches(root.path());
        let (shallow, _) = walk_one(
            &repo,
            &IngestScope::new("", Some(1)),
            &mut TreeCache::default(),
        );
        assert_eq!(
            paths(&shallow).iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            ["r/a.rs", "r/a.rs"]
        );
        let (scoped, _) = walk_one(
            &repo,
            &IngestScope::new("r/src", None),
            &mut TreeCache::default(),
        );
        assert_eq!(
            paths(&scoped).iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            ["r/src/b.rs", "r/src/c.rs"]
        );
        let (elsewhere, _) = walk_one(
            &repo,
            &IngestScope::new("other", None),
            &mut TreeCache::default(),
        );
        assert!(elsewhere.files.is_empty() && elsewhere.units.is_empty());
    }

    /// **A tree is listed once and kept while a tip names it**, then let go.
    #[test]
    fn trees_are_kept_while_a_tip_names_them() {
        let root = tempfile::tempdir().unwrap();
        let repo = two_branches(root.path());
        let mut cache = TreeCache::default();
        walk_one(&repo, &IngestScope::new("", None), &mut cache);
        assert_eq!(cache.trees.len(), 2);
        let only_main = RepoBranches {
            name: repo.name.clone(),
            repo: Repo::open(repo.repo.dir()).unwrap(),
            tips: repo.tips[..1].to_vec(),
        };
        cache.retain_tips(std::slice::from_ref(&only_main));
        assert_eq!(cache.trees.keys().collect::<Vec<_>>(), [&repo.tips[0].tip]);
    }
}

//! The file stores of a workspace — one [`VfsStore`] per repository.
//!
//! Every `file_*` call names a repository; [`RepoFiles::repo`] turns the name
//! into that repository's store and refuses a name the workspace does not
//! list. Each store is rooted at its repository's folder, so a path inside it
//! is repository-relative and cannot reach another repository or anything else
//! in the workspace folder.
//!
//! A set belongs to one conversation. Its overlay stores hold that
//! conversation's changes and no one else's, so each conversation is given its
//! own — [`RepoFiles::fresh`] makes one — and a file one conversation writes is
//! never what another one reads.
//!
//! A repository under git is read through a branch ([`VfsStore::on_branch`]):
//! the one checked out when the set was first made, until the conversation's
//! own are given ([`RepoFiles::set_branches`]). The repository's
//! [`GitSource`] is opened once, when the first set is made, and shared by
//! every set made from it — one git process per repository however many
//! conversations read it. A folder that is not a git repository is read as it
//! stands on disk.
//!
//! A [`RepoFiles::detached`] set has no workspace behind it: there is no disk,
//! so there is nothing to scope a name against, and each repository name gets
//! an upper-only store the first time a call uses it. That is the shape a
//! context built without a workspace — a test, a scratch interpreter — has.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use super::vfs::git_source::GitSource;
use super::vfs::{Snapshot, VfsStore};
use super::workspace::{Workspace, ALL_REPOS};
use crate::{BranchName, DiskWriteGrant, Rev};

/// Each git repository's source, by repository name.
type Sources = BTreeMap<String, Arc<GitSource>>;

/// A `repo` argument the workspace does not list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownRepo {
    /// The name the call gave.
    pub name: String,
    /// Every name the workspace lists, in manifest order.
    pub known: Vec<String>,
}

impl fmt::Display for UnknownRepo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "no repository named {:?} — repo must be one of: {}",
            self.name,
            self.known.join(", ")
        )
    }
}

impl std::error::Error for UnknownRepo {}

/// One store per repository, in manifest order.
pub struct RepoFiles {
    workspace: Option<Workspace>,
    sources: Arc<Sources>,
    stores: RwLock<Vec<(String, Arc<VfsStore>)>>,
    direct: bool,
    /// Saved work that could not be restored, by the key it was saved under,
    /// as saved: kept so that saving the set again never overwrites it.
    unrestored: Mutex<BTreeMap<String, String>>,
}

impl RepoFiles {
    /// No workspace: every name gets an upper-only store on first use.
    pub fn detached() -> Self {
        Self {
            workspace: None,
            sources: Arc::default(),
            stores: RwLock::new(Vec::new()),
            direct: false,
            unrestored: Mutex::default(),
        }
    }

    /// An overlay store over each of `workspace`'s repositories: reads fall
    /// through to each one's branch — or, for a folder not under git, to the
    /// folder — and writes stay in memory.
    pub fn overlay(workspace: Workspace) -> Self {
        let sources = workspace
            .repos()
            .iter()
            .filter_map(|r| Some((r.name.clone(), GitSource::open(&r.dir).ok()?)))
            .collect();
        Self::over(workspace, Arc::new(sources))
    }

    /// Overlay stores over `workspace`, reading through `sources`.
    fn over(workspace: Workspace, sources: Arc<Sources>) -> Self {
        let stores = workspace
            .repos()
            .iter()
            .map(|r| {
                let store = match sources.get(&r.name) {
                    Some(source) => {
                        VfsStore::on_branch(Arc::clone(source), source.default_rev().clone())
                    }
                    None => VfsStore::with_root(&r.dir),
                };
                (r.name.clone(), Arc::new(store))
            })
            .collect();
        Self {
            workspace: Some(workspace),
            sources,
            stores: RwLock::new(stores),
            direct: false,
            unrestored: Mutex::default(),
        }
    }

    /// A direct store over each of `workspace`'s repositories: writes and
    /// deletes change the files on disk. Takes the grant [`VfsStore::direct`]
    /// needs.
    pub fn direct(workspace: Workspace, grant: DiskWriteGrant) -> Self {
        let stores = workspace
            .repos()
            .iter()
            .map(|r| (r.name.clone(), Arc::new(VfsStore::direct(&r.dir, &grant))))
            .collect();
        Self {
            workspace: Some(workspace),
            sources: Arc::default(),
            stores: RwLock::new(stores),
            direct: true,
            unrestored: Mutex::default(),
        }
    }

    /// A set of the same kind with none of this one's session changes — what
    /// a new conversation starts from. An overlay set gets fresh overlays over
    /// the same workspace, reading through the same sources, on each
    /// repository's first branch; a detached set a fresh detached one. A
    /// direct set holds no session changes — its writes are on disk — so the
    /// new set shares its stores.
    pub fn fresh(&self) -> Self {
        if self.direct {
            return Self {
                workspace: self.workspace.clone(),
                sources: Arc::clone(&self.sources),
                stores: RwLock::new(self.all()),
                direct: true,
                unrestored: Mutex::default(),
            };
        }
        match &self.workspace {
            Some(workspace) => Self::over(workspace.clone(), Arc::clone(&self.sources)),
            None => Self::detached(),
        }
    }

    /// Read each named repository through the branch `branches` gives it —
    /// the conversation's own. A repository the set does not have, one not
    /// under git, and a name that is not a branch are named in the result
    /// with the reason; the others are set all the same.
    pub fn set_branches(&self, branches: &BTreeMap<String, String>) -> Vec<(String, String)> {
        let mut refused = Vec::new();
        for (name, branch) in branches {
            let set = self
                .repo(name)
                .map_err(|e| e.to_string())
                .and_then(|store| {
                    let branch = BranchName::parse(branch).map_err(|e| e.to_string())?;
                    store
                        .set_branch(branch)
                        .then_some(())
                        .ok_or_else(|| "it is not read through git".to_string())
                });
            if let Err(why) = set {
                refused.push((name.clone(), why));
            }
        }
        refused
    }

    /// The workspace behind these stores, if any.
    pub fn workspace(&self) -> Option<&Workspace> {
        self.workspace.as_ref()
    }

    /// Whether writes and deletes change the files on disk.
    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// The store for the repository called `name`.
    pub fn repo(&self, name: &str) -> Result<Arc<VfsStore>, UnknownRepo> {
        if let Some((_, store)) = self.stores.read().unwrap().iter().find(|(n, _)| n == name) {
            return Ok(Arc::clone(store));
        }
        // `*` means every repository, never one: a tool that reads one
        // repository's store is refused it even with no workspace to check.
        if self.workspace.is_some() || name.is_empty() || name == ALL_REPOS {
            return Err(self.unknown(name));
        }
        let mut stores = self.stores.write().unwrap();
        // Another caller may have created it between the two locks.
        if let Some((_, store)) = stores.iter().find(|(n, _)| n == name) {
            return Ok(Arc::clone(store));
        }
        let store = Arc::new(VfsStore::new());
        stores.push((name.to_string(), Arc::clone(&store)));
        Ok(store)
    }

    /// Every repository's store, in manifest order — for a call that covers
    /// the whole workspace.
    pub fn all(&self) -> Vec<(String, Arc<VfsStore>)> {
        self.stores.read().unwrap().clone()
    }

    /// Every repository name, in manifest order.
    pub fn names(&self) -> Vec<String> {
        self.stores
            .read()
            .unwrap()
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// The branch each repository read through git is on, by name — what a
    /// conversation saves as its branches.
    pub fn branches(&self) -> BTreeMap<String, String> {
        self.all()
            .into_iter()
            .filter_map(|(name, store)| match store.rev()? {
                Rev::Branch(branch) => Some((name, branch.as_str().to_string())),
                _ => None,
            })
            .collect()
    }

    /// Every repository's session changes, by name — those with any, to be
    /// saved with the conversation they belong to.
    pub fn snapshots(&self) -> BTreeMap<String, Snapshot> {
        self.all()
            .into_iter()
            .map(|(name, store)| (name, store.snapshot()))
            .filter(|(_, snapshot)| !snapshot.is_empty())
            .collect()
    }

    /// Put back each repository's saved changes. A repository the set does
    /// not have, or whose store refuses its changes, is named in the result
    /// with the reason; the others are restored all the same.
    pub fn restore(&self, saved: BTreeMap<String, Snapshot>) -> Vec<(String, String)> {
        let mut refused = Vec::new();
        for (name, snapshot) in saved {
            let result = self
                .repo(&name)
                .map_err(|e| e.to_string())
                .and_then(|store| store.restore(snapshot).map_err(|e| e.to_string()));
            if let Err(why) = result {
                refused.push((name, why));
            }
        }
        refused
    }

    /// Keep `saved` — work saved under `key` that could not be restored — so
    /// that it is saved again as it was ([`Self::unrestored`]), never
    /// overwritten by what this set holds.
    pub fn keep_unrestored(&self, key: String, saved: String) {
        self.unrestored
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, saved);
    }

    /// The saved work [`Self::keep_unrestored`] kept, by key.
    pub fn unrestored(&self) -> BTreeMap<String, String> {
        self.unrestored
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Bytes held in every store's upper layer together.
    pub fn total_bytes(&self) -> usize {
        self.stores
            .read()
            .unwrap()
            .iter()
            .map(|(_, s)| s.total_bytes())
            .sum()
    }

    fn unknown(&self, name: &str) -> UnknownRepo {
        UnknownRepo {
            name: name.to_string(),
            known: self.names(),
        }
    }
}

impl Default for RepoFiles {
    fn default() -> Self {
        Self::detached()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::{scratch, TestRepo};
    use crate::workspace::RepoSpec;

    fn two_repos() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        for (repo, file) in [("a", "one.txt"), ("b", "two.txt")] {
            std::fs::create_dir(dir.path().join(repo)).unwrap();
            std::fs::write(dir.path().join(repo).join(file), repo).unwrap();
        }
        let ws =
            Workspace::new(dir.path(), vec![RepoSpec::named("a"), RepoSpec::named("b")]).unwrap();
        (dir, ws)
    }

    /// **Each repository is its own root.** A path resolves inside the named
    /// repository only, and the same path in another repository is a
    /// different file.
    #[test]
    fn each_repository_resolves_against_its_own_folder() {
        let (dir, ws) = two_repos();
        let files = RepoFiles::overlay(ws);
        assert_eq!(
            files.repo("a").unwrap().root(),
            Some(dir.path().join("a").as_path())
        );
        assert_eq!(
            files.repo("a").unwrap().read("one.txt").unwrap().as_deref(),
            Some("a")
        );
        assert_eq!(files.repo("b").unwrap().read("one.txt").unwrap(), None);
        assert_eq!(
            files.repo("b").unwrap().read("two.txt").unwrap().as_deref(),
            Some("b")
        );
        assert_eq!(
            files.repo("a").unwrap().read("../b/two.txt").unwrap(),
            None,
            "`..` stops at the repository's own root"
        );
    }

    /// **A name the workspace does not list is refused**, and the refusal
    /// names every one it does.
    #[test]
    fn an_unlisted_repository_is_refused_with_the_listed_names() {
        let (_dir, ws) = two_repos();
        let files = RepoFiles::overlay(ws);
        let err = files.repo("c").err().unwrap();
        assert_eq!(
            err,
            UnknownRepo {
                name: "c".to_string(),
                known: vec!["a".to_string(), "b".to_string()],
            }
        );
        assert_eq!(
            err.to_string(),
            "no repository named \"c\" — repo must be one of: a, b"
        );
    }

    #[test]
    fn stores_are_kept_in_manifest_order() {
        let (_dir, ws) = two_repos();
        let files = RepoFiles::overlay(ws);
        assert_eq!(files.names(), vec!["a", "b"]);
        assert_eq!(files.all().len(), 2);
        assert!(!files.is_direct());
    }

    /// The same store answers every call for one repository, so a write is
    /// visible to the next read.
    #[test]
    fn one_repository_is_one_store() {
        let (_dir, ws) = two_repos();
        let files = RepoFiles::overlay(ws);
        files
            .repo("a")
            .unwrap()
            .write("new.txt", "x".into())
            .unwrap();
        assert_eq!(
            files.repo("a").unwrap().read("new.txt").unwrap().as_deref(),
            Some("x")
        );
        assert_eq!(files.total_bytes(), 1);
    }

    /// **A fresh set shares no session changes with the one it came from** —
    /// what keeps one conversation's writes out of another's reads — while
    /// still reading the same workspace underneath.
    #[test]
    fn a_fresh_set_starts_without_the_others_changes() {
        let (_dir, ws) = two_repos();
        let first = RepoFiles::overlay(ws);
        first
            .repo("a")
            .unwrap()
            .write("one.txt", "mine".into())
            .unwrap();
        first.repo("b").unwrap().delete("two.txt");

        let second = first.fresh();
        assert!(!second.is_direct());
        assert_eq!(second.names(), first.names());
        assert_eq!(
            second
                .repo("a")
                .unwrap()
                .read("one.txt")
                .unwrap()
                .as_deref(),
            Some("a"),
            "the workspace's copy, not the first set's write"
        );
        assert_eq!(
            second
                .repo("b")
                .unwrap()
                .read("two.txt")
                .unwrap()
                .as_deref(),
            Some("b"),
            "the first set's delete hides nothing here"
        );
        assert_eq!(second.total_bytes(), 0);
        assert_eq!(
            first.repo("a").unwrap().read("one.txt").unwrap().as_deref(),
            Some("mine")
        );
    }

    /// **A set's changes survive into a fresh set** — only the repositories
    /// with changes are saved, and a repository the fresh set does not have
    /// is reported rather than dropped silently.
    #[test]
    fn a_sets_changes_restore_into_a_fresh_set() {
        let (_dir, ws) = two_repos();
        let first = RepoFiles::overlay(ws);
        first
            .repo("a")
            .unwrap()
            .write("one.txt", "mine".into())
            .unwrap();
        let mut saved = first.snapshots();
        assert_eq!(saved.keys().collect::<Vec<_>>(), ["a"], "b has no changes");

        let second = first.fresh();
        saved.insert("gone".to_string(), first.repo("a").unwrap().snapshot());
        let refused = second.restore(saved);
        assert_eq!(
            refused.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["gone"]
        );
        assert_eq!(
            second
                .repo("a")
                .unwrap()
                .read("one.txt")
                .unwrap()
                .as_deref(),
            Some("mine")
        );
    }

    /// **A git repository reads through its branch, never its folder** — the
    /// branch checked out when the set was made, then the conversation's own
    /// — and every fresh set shares one source per repository. A folder not
    /// under git reads from disk and takes no branch.
    #[test]
    fn a_git_repository_reads_through_its_branch_not_its_folder() {
        let t = TestRepo::init();
        t.write("a.txt", b"main\n");
        t.commit_all("main");
        t.git(&["checkout", "-q", "-b", "topic"]);
        t.write("a.txt", b"topic\n");
        t.commit_all("topic");
        t.git(&["checkout", "-q", "main"]);
        t.write("a.txt", b"left on disk by a job\n");
        let plain = tempfile::Builder::new()
            .prefix("plain-")
            .tempdir_in(scratch())
            .unwrap();
        std::fs::write(plain.path().join("b.txt"), b"on disk\n").unwrap();
        let name = |p: &std::path::Path| p.file_name().unwrap().to_string_lossy().into_owned();
        let (repo, folder) = (name(&t.path), name(plain.path()));
        let ws = Workspace::new(
            scratch(),
            vec![RepoSpec::named(&repo), RepoSpec::named(&folder)],
        )
        .unwrap();

        let files = RepoFiles::overlay(ws);
        let read = |files: &RepoFiles, repo: &str, path: &str| {
            files.repo(repo).unwrap().read(path).unwrap()
        };
        assert_eq!(read(&files, &repo, "a.txt").as_deref(), Some("main\n"));
        assert_eq!(read(&files, &folder, "b.txt").as_deref(), Some("on disk\n"));

        let fresh = files.fresh();
        assert!(Arc::ptr_eq(&files.sources, &fresh.sources), "one source");
        let topic = BTreeMap::from([(repo.clone(), "topic".to_string())]);
        assert!(fresh.set_branches(&topic).is_empty());
        assert_eq!(read(&fresh, &repo, "a.txt").as_deref(), Some("topic\n"));
        assert_eq!(fresh.branches(), topic, "the folder has no branch");
        assert_eq!(
            read(&files, &repo, "a.txt").as_deref(),
            Some("main\n"),
            "the first set is on its own branch still"
        );

        let refused = fresh.set_branches(&BTreeMap::from([
            ("nope".to_string(), "main".to_string()),
            (folder.clone(), "main".to_string()),
            (repo.clone(), "bad..name".to_string()),
        ]));
        let names: Vec<&str> = refused.iter().map(|(n, _)| n.as_str()).collect();
        let mut expected = vec![folder.as_str(), "nope", repo.as_str()];
        expected.sort_unstable();
        assert_eq!(names, expected);
        assert_eq!(read(&fresh, &repo, "a.txt").as_deref(), Some("topic\n"));
    }

    #[test]
    fn a_direct_set_writes_each_repository_on_disk() {
        let (dir, ws) = two_repos();
        let files = RepoFiles::direct(ws, DiskWriteGrant::issue());
        assert!(files.is_direct());
        files
            .repo("b")
            .unwrap()
            .write("made.txt", "m".into())
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b").join("made.txt")).unwrap(),
            "m"
        );
    }

    /// **A detached set makes a store per name on first use**, since there is
    /// no workspace to check the name against.
    #[test]
    fn a_detached_set_creates_upper_only_stores_on_demand() {
        let files = RepoFiles::detached();
        assert!(files.names().is_empty());
        files
            .repo("scratch")
            .unwrap()
            .write("a.txt", "x".into())
            .unwrap();
        assert_eq!(files.names(), vec!["scratch"]);
        assert!(files.repo("scratch").unwrap().root().is_none());
        assert_eq!(
            files
                .repo("scratch")
                .unwrap()
                .read("a.txt")
                .unwrap()
                .as_deref(),
            Some("x")
        );
        assert!(
            files.repo("").is_err(),
            "an empty name is never a repository"
        );
        assert!(
            files.repo(ALL_REPOS).is_err(),
            "`*` means every repository, never one"
        );
        assert!(!files.names().contains(&ALL_REPOS.to_string()));
    }
}

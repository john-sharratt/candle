//! Which ingested units a conversation may retrieve — those whose keys its
//! own base holds (`docs/zend_branch_ingest.md` §8.2).
//!
//! The `repo_map` and `code_reading` layers hold every branch's version of a
//! file or folder, keyed by what it shows. A conversation reads each
//! repository at its own base, so the only versions right for it are the
//! ones its base tree holds: every file and folder of that tree is keyed by
//! the same rules the ingest used ([`crate::branch_ingest`]) and looked up in
//! an [`IngestIndex`] of what has been ingested. A path the conversation has
//! changed contributes no key — its own copy is what it reads — and neither
//! does the folder holding it.
//!
//! Working a scope out reads no file and takes no base: a repository the
//! conversation has not read yet is scoped at the base its first read would
//! take ([`VfsStore::peek_base`]). A tree's units are the same for every
//! conversation on it, so they are kept per `(repository, tree, index
//! generation)`, and the whole scope of a conversation that has changed
//! nothing is kept per set of trees and shared by every conversation on
//! them: a turn on an unchanged base costs one lookup.
//!
//! The index is rebuilt lazily. A unit that commits marks it stale
//! ([`RetrievalScope::mark_stale`]) and the next turn rebuilds it — at most
//! once every [`REBUILD_EVERY`], since units commit continuously through an
//! ingest pass. An upload rebuilds it at once ([`RetrievalScope::refresh`]):
//! the very next turn may ask about the file.

mod index;
mod kept;
mod tree_scope;

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::ConversationEngine;
use zend_vfs::{Base, Oid, RepoFiles, VfsStore};

use self::index::IngestIndex;
use self::kept::Kept;
use self::tree_scope::TreeScope;
use crate::branch_ingest::filter::IngestScope;
use crate::branch_ingest::manifest::Hints;
use crate::branch_ingest::units::{dir_of, TreeFile};
use crate::code_read::is_upload_path;

/// The least time between two rebuilds a turn asks for.
const REBUILD_EVERY: Duration = Duration::from_secs(1);

/// How many trees' units are kept: a branch's tree and the one before it,
/// for a handful of repositories and branches in use at once.
const KEPT_TREES: usize = 16;

/// How many whole scopes are kept to share: one per set of bases the
/// conversations that changed nothing are on.
const KEPT_SCOPES: usize = 16;

/// What a whole scope is kept under: the index generation, and each
/// repository's base tree.
type ScopeKey = (u64, Vec<(String, Oid)>);

/// A conversation's scope, per scoped group — shared by every conversation
/// on the same bases that has changed nothing.
#[derive(Debug, Clone)]
struct Scope {
    files: Arc<HashSet<TimelineId>>,
    folders: Arc<HashSet<TimelineId>>,
}

/// One repository a conversation reads at a base.
struct AtBase {
    name: String,
    store: Arc<VfsStore>,
    base: Base,
    tree: Oid,
    changed: Vec<String>,
}

/// The scoped groups, where their scope comes from, and what was worked out.
pub struct RetrievalScope {
    /// The `code_reading` group, when that layer is ingested.
    files_group: Option<GroupId>,
    /// The `repo_map` group and the scope its retained units were walked
    /// under — the ingest scope at full depth.
    folders: Option<(GroupId, IngestScope)>,
    index: RwLock<Arc<IngestIndex>>,
    /// Set when a unit commits: the index no longer holds all that is
    /// committed.
    stale: AtomicBool,
    /// When the index was last rebuilt — held across a rebuild, so rebuilds
    /// publish in the order they read the substrate.
    rebuilt: Mutex<Option<Instant>>,
    trees: Mutex<Kept<(String, Oid, u64), Arc<TreeScope>>>,
    /// Whole scopes, by index generation and every repository's base tree.
    shared: Mutex<Kept<ScopeKey, Scope>>,
    /// Each manifest version's hint — part of a folder's key, as in the walk.
    hints: Mutex<Hints>,
}

impl RetrievalScope {
    pub fn new(files_group: Option<GroupId>, folders: Option<(GroupId, IngestScope)>) -> Self {
        Self {
            files_group,
            folders,
            index: RwLock::new(Arc::default()),
            stale: AtomicBool::new(false),
            rebuilt: Mutex::new(None),
            trees: Mutex::new(Kept::new(KEPT_TREES)),
            shared: Mutex::new(Kept::new(KEPT_SCOPES)),
            hints: Mutex::new(Hints::default()),
        }
    }

    /// A unit committed: the next turn rebuilds the index.
    pub fn mark_stale(&self) {
        self.stale.store(true, Ordering::SeqCst);
    }

    /// Rebuild the index now — at load, and once an upload commits.
    pub fn refresh(&self, engine: &Mutex<ConversationEngine>) {
        let mut rebuilt = self.rebuilt.lock().unwrap();
        self.rebuild(&mut rebuilt, Instant::now(), |generation| {
            IngestIndex::read(&engine.lock().unwrap(), generation)
        });
    }

    /// `target`'s scope from its files, set on the engine for each scoped
    /// group.
    pub fn apply(&self, engine: &Mutex<ConversationEngine>, target: TimelineId, files: &RepoFiles) {
        let index = self.current(Instant::now(), |generation| {
            IngestIndex::read(&engine.lock().unwrap(), generation)
        });
        let scope = self.scope_of(&index, files);
        let e = engine.lock().unwrap();
        if let Some(group) = self.files_group {
            e.set_retrieval_scope(target, group, scope.files);
        }
        if let Some((group, _)) = &self.folders {
            e.set_retrieval_scope(target, *group, scope.folders);
        }
    }

    /// The committed `repo_map` unit for the folder `inner` of `repo` (`""`
    /// for its root), as `files` lists it — what the fast path carries in
    /// place of a `file_list`.
    pub fn folder_unit(
        &self,
        engine: &Mutex<ConversationEngine>,
        files: &RepoFiles,
        repo: &str,
        inner: &str,
    ) -> Option<TimelineId> {
        let index = self.current(Instant::now(), |generation| {
            IngestIndex::read(&engine.lock().unwrap(), generation)
        });
        self.folder_in(&index, files, repo, inner)
    }

    /// [`Self::folder_unit`] under `index`. `None` when the folder layer is not
    /// ingested, the repository has no base, or no unit carries the folder's
    /// key — and whenever the conversation has changed anything at or under
    /// the folder. That is stricter than the scope, which drops only a changed
    /// file's own folder: a scope that keeps a stale folder offers a summary,
    /// while this tells the model the listing is in context, and a new file in
    /// a new subfolder changes the listing above it too.
    fn folder_in(
        &self,
        index: &IngestIndex,
        files: &RepoFiles,
        repo: &str,
        inner: &str,
    ) -> Option<TimelineId> {
        self.folders.as_ref()?;
        let at = at_bases(files).into_iter().find(|r| r.name == repo)?;
        let under = if inner.is_empty() {
            String::new()
        } else {
            format!("{inner}/")
        };
        if at.changed.iter().any(|path| path.starts_with(&under)) {
            return None;
        }
        let dir = format!("{repo}/{under}");
        self.tree_scope(index, &at)?.folders.get(&dir).copied()
    }

    /// The index to scope a turn by — rebuilt first when a unit has
    /// committed since the last rebuild and that one is [`REBUILD_EVERY`]
    /// old.
    fn current(&self, now: Instant, read: impl FnOnce(u64) -> IngestIndex) -> Arc<IngestIndex> {
        if self.stale.load(Ordering::SeqCst) {
            let mut rebuilt = self.rebuilt.lock().unwrap();
            let due = rebuilt.is_none_or(|at| now.saturating_duration_since(at) >= REBUILD_EVERY);
            // A turn that waited on the lock may find the rebuild done.
            if due && self.stale.load(Ordering::SeqCst) {
                self.rebuild(&mut rebuilt, now, read);
            }
        }
        Arc::clone(&self.index.read().unwrap())
    }

    /// Read the index again and publish it under the next generation. The
    /// caller holds `rebuilt`, so no older read is published after it.
    fn rebuild(
        &self,
        rebuilt: &mut Option<Instant>,
        now: Instant,
        read: impl FnOnce(u64) -> IngestIndex,
    ) {
        // Cleared before the read: a unit committing while it runs marks the
        // index stale again rather than being missed.
        self.stale.store(false, Ordering::SeqCst);
        let generation = self.index.read().unwrap().generation() + 1;
        let index = read(generation);
        *self.index.write().unwrap() = Arc::new(index);
        *rebuilt = Some(now);
        self.trees
            .lock()
            .unwrap()
            .retain(|(_, _, g)| *g == generation);
        self.shared
            .lock()
            .unwrap()
            .retain(|(g, _)| *g == generation);
    }

    /// What `files` may retrieve under `index`.
    fn scope_of(&self, index: &IngestIndex, files: &RepoFiles) -> Scope {
        let repos = at_bases(files);
        let unchanged = repos.iter().all(|r| r.changed.is_empty());
        let key: ScopeKey = (
            index.generation(),
            repos
                .iter()
                .map(|r| (r.name.clone(), r.tree.clone()))
                .collect(),
        );
        if unchanged {
            if let Some(kept) = self.shared.lock().unwrap().get(&key) {
                return kept;
            }
        }
        let mut in_files: HashSet<TimelineId> = index.uploads().iter().copied().collect();
        let mut in_folders = HashSet::new();
        for repo in &repos {
            let Some(of_tree) = self.tree_scope(index, repo) else {
                continue;
            };
            let changed: HashSet<&str> = repo.changed.iter().map(String::as_str).collect();
            let changed_dirs: HashSet<String> = repo
                .changed
                .iter()
                .map(|path| dir_of(&format!("{}/{path}", repo.name)))
                .collect();
            in_files.extend(
                of_tree
                    .files
                    .iter()
                    .filter(|(path, _)| !changed.contains(path.as_str()))
                    .map(|(_, tl)| *tl),
            );
            in_folders.extend(
                of_tree
                    .folders
                    .iter()
                    .filter(|(dir, _)| !changed_dirs.contains(*dir))
                    .map(|(_, tl)| *tl),
            );
        }
        let scope = Scope {
            files: Arc::new(in_files),
            folders: Arc::new(in_folders),
        };
        if unchanged {
            self.shared.lock().unwrap().insert(key, scope.clone());
        }
        scope
    }

    /// The ingested units of `repo`'s base tree, kept by tree id: the tree
    /// is listed only when no conversation has been scoped on it under this
    /// index.
    fn tree_scope(&self, index: &IngestIndex, repo: &AtBase) -> Option<Arc<TreeScope>> {
        let at = (repo.name.clone(), repo.tree.clone(), index.generation());
        if let Some(kept) = self.trees.lock().unwrap().get(&at) {
            return Some(kept);
        }
        let tree = match repo.store.tree_at(&repo.base) {
            Ok(Some(tree)) => tree,
            Ok(None) => return None,
            Err(e) => {
                tracing::debug!(repo = %repo.name, "retrieval scope: base tree unreadable: {e}");
                return None;
            }
        };
        let folder_scope = self.folders.as_ref().map(|(_, s)| s);
        let mut hints = self.hints.lock().unwrap();
        let mut hint_of = |f: &TreeFile| {
            let name = f.path.rsplit('/').next().unwrap_or(&f.path);
            hints.of(name, &f.blob, |blob| {
                repo.store.blob_at(blob).ok().flatten()
            })
        };
        let built = Arc::new(TreeScope::of(
            index,
            &repo.name,
            &tree,
            folder_scope,
            &mut hint_of,
        ));
        drop(hints);
        self.trees.lock().unwrap().insert(at, Arc::clone(&built));
        Some(built)
    }
}

/// Every repository of `files` read at a base with a commit, with the paths
/// the conversation changed in it. The uploads folder and a folder not under
/// git have no base.
fn at_bases(files: &RepoFiles) -> Vec<AtBase> {
    let mut out = Vec::new();
    for (name, store) in files.all() {
        if is_upload_path(&name) {
            continue;
        }
        let base = match store.peek_base() {
            Ok(Some(base)) => base,
            Ok(None) => continue,
            Err(e) => {
                tracing::debug!(repo = %name, "retrieval scope: base unreadable: {e}");
                continue;
            }
        };
        let Some(tree) = base.tree.clone() else {
            continue;
        };
        let changed = store.changed_paths();
        out.push(AtBase {
            name,
            store,
            base,
            tree,
            changed,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::process::Command;

    use zend_vfs::vfs::Tree;
    use zend_vfs::{Repo, RepoSpec, Workspace};

    use super::*;
    use crate::branch_ingest::keys::file_key;
    use crate::branch_ingest::units::{folder_units, TreeFile};

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

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn oid(root: &Path, spec: &str) -> Oid {
        Oid::parse(git(&root.join("r"), &["rev-parse", spec]).trim()).unwrap()
    }

    /// A workspace with one repository `r` on `main`: `a.rs`, `src/b.rs`.
    fn workspace() -> (tempfile::TempDir, Workspace) {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("r");
        std::fs::create_dir_all(dir.join("src")).unwrap();
        git(&dir, &["init", "-q"]);
        git(&dir, &["symbolic-ref", "HEAD", "refs/heads/main"]);
        std::fs::write(dir.join("a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(dir.join("src/b.rs"), "fn b() {}\n").unwrap();
        git(&dir, &["add", "-A"]);
        git(&dir, &["commit", "-q", "-m", "first"]);
        let ws = Workspace::new(root.path(), vec![RepoSpec::named("r")]).unwrap();
        (root, ws)
    }

    /// An index holding `a.rs` and `src/b.rs` as `main` has them, a version of
    /// `a.rs` no branch has, the folder units of `main`, and one upload.
    fn index(root: &Path, scope: &IngestScope, generation: u64) -> IngestIndex {
        let (a, b) = (oid(root, "main:a.rs"), oid(root, "main:src/b.rs"));
        let repo = Repo::open(&root.join("r")).unwrap();
        let tree = Tree::read(&repo, &oid(root, "main^{tree}")).unwrap();
        let read: Vec<TreeFile> = tree
            .files()
            .map(|(path, blob, size)| TreeFile {
                path: format!("r/{path}"),
                blob: blob.clone(),
                size,
                language: scope.admits("r", path, size).unwrap(),
            })
            .collect();
        let mut by_key = HashMap::from([
            (file_key("r/a.rs", &a), tl(1)),
            (file_key("r/src/b.rs", &b), tl(2)),
            (
                file_key(
                    "r/a.rs",
                    &Oid::parse("4b825dc642cb6eb9a060e54bf8d69288fbee4904").unwrap(),
                ),
                tl(3),
            ),
        ]);
        for (n, unit) in folder_units("r", &tree, &read, &mut |_: &TreeFile| None)
            .into_iter()
            .enumerate()
        {
            by_key.insert(unit.key, tl(10 + n as u64));
        }
        IngestIndex::of(by_key, vec![tl(30)], generation)
    }

    /// A scope over `r`, both layers ingested, its index already read.
    fn scoped(root: &Path) -> RetrievalScope {
        let folder_scope = IngestScope::new("", None);
        let rs = RetrievalScope::new(
            Some(GroupId::from_raw(2).unwrap()),
            Some((GroupId::from_raw(1).unwrap(), folder_scope.clone())),
        );
        *rs.index.write().unwrap() = Arc::new(index(root, &folder_scope, 1));
        rs
    }

    fn scope_of(rs: &RetrievalScope, files: &RepoFiles) -> Scope {
        let index = Arc::clone(&rs.index.read().unwrap());
        rs.scope_of(&index, files)
    }

    fn set(timelines: &[u64]) -> HashSet<TimelineId> {
        timelines.iter().map(|&n| tl(n)).collect()
    }

    /// **A conversation's scope is what its base holds**: each file's version
    /// on its base, each folder as its base lists it, and every upload —
    /// never a version no branch it is on holds.
    #[test]
    fn the_scope_is_what_the_base_holds() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let scope = scope_of(&rs, &RepoFiles::overlay(ws));
        assert_eq!(*scope.files, set(&[1, 2, 30]));
        assert_eq!(*scope.folders, set(&[10, 11]));
    }

    /// **A path the conversation changed leaves the scope**, and so does the
    /// folder holding it — its own copy is what it reads.
    #[test]
    fn what_the_conversation_changed_leaves_the_scope() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let files = RepoFiles::overlay(ws);
        files
            .repo("r")
            .unwrap()
            .write("src/b.rs", "fn mine() {}\n".into())
            .unwrap();
        let scope = scope_of(&rs, &files);
        assert_eq!(*scope.files, set(&[1, 30]));
        assert_eq!(*scope.folders, set(&[10]), "r/src/ left with its file");
    }

    /// **Conversations on the same bases that changed nothing share one
    /// scope**; one with a change of its own has its own.
    #[test]
    fn conversations_that_changed_nothing_share_a_scope() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let one = RepoFiles::overlay(ws);
        let two = one.fresh();
        let (a, b) = (scope_of(&rs, &one), scope_of(&rs, &two));
        assert!(Arc::ptr_eq(&a.files, &b.files) && Arc::ptr_eq(&a.folders, &b.folders));

        two.repo("r")
            .unwrap()
            .write("a.rs", "fn mine() {}\n".into())
            .unwrap();
        let changed = scope_of(&rs, &two);
        assert!(!Arc::ptr_eq(&a.files, &changed.files));
        assert!(Arc::ptr_eq(&a.files, &scope_of(&rs, &one).files));
    }

    fn folder_in(rs: &RetrievalScope, files: &RepoFiles, repo: &str, inner: &str) -> Option<u64> {
        let index = Arc::clone(&rs.index.read().unwrap());
        rs.folder_in(&index, files, repo, inner)
            .map(TimelineId::raw)
    }

    /// **A folder is found by its repository and path inside it** — the root
    /// as the empty path — at the unit its base lists; a folder with no unit,
    /// or an unlisted repository, is found nowhere.
    #[test]
    fn a_folder_unit_is_the_one_its_base_lists() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let files = RepoFiles::overlay(ws);
        assert_eq!(folder_in(&rs, &files, "r", ""), Some(10));
        assert_eq!(folder_in(&rs, &files, "r", "src"), Some(11));
        assert_eq!(folder_in(&rs, &files, "r", "nope"), None);
        assert_eq!(folder_in(&rs, &files, "other", ""), None);
    }

    /// **Anything changed at or under a folder takes it out**, the folders
    /// above included — a new file in a new subfolder changes their listings
    /// too — while a sibling folder stays.
    #[test]
    fn a_change_under_a_folder_takes_it_and_its_ancestors_out() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let files = RepoFiles::overlay(ws);
        files
            .repo("r")
            .unwrap()
            .write("src/b.rs", "fn mine() {}\n".into())
            .unwrap();
        assert_eq!(folder_in(&rs, &files, "r", "src"), None);
        assert_eq!(folder_in(&rs, &files, "r", ""), None);

        let files = files.fresh();
        files
            .repo("r")
            .unwrap()
            .write("a.rs", "fn mine() {}\n".into())
            .unwrap();
        assert_eq!(folder_in(&rs, &files, "r", "src"), Some(11));
        assert_eq!(folder_in(&rs, &files, "r", ""), None);
    }

    /// With no folder layer ingested, no folder is served.
    #[test]
    fn with_no_folder_layer_no_folder_unit_is_found() {
        let (root, ws) = workspace();
        let rs = RetrievalScope::new(Some(GroupId::from_raw(2).unwrap()), None);
        *rs.index.write().unwrap() = Arc::new(index(root.path(), &IngestScope::new("", None), 1));
        assert_eq!(folder_in(&rs, &RepoFiles::overlay(ws), "r", ""), None);
    }

    /// **Working a scope out takes no base**: a conversation that has not
    /// read the repository yet still reads the branch as it stands when it
    /// does.
    #[test]
    fn working_a_scope_out_takes_no_base() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        let files = RepoFiles::overlay(ws);
        scope_of(&rs, &files);
        let dir = root.path().join("r");
        std::fs::write(dir.join("a.rs"), "fn later() {}\n").unwrap();
        git(&dir, &["commit", "-q", "-am", "later"]);
        let read = files.repo("r").unwrap().read("a.rs").unwrap();
        assert_eq!(read.as_deref(), Some("fn later() {}\n"));
    }

    /// With no folder layer ingested, no folder is in any scope.
    #[test]
    fn with_no_folder_layer_there_are_no_folders() {
        let (root, ws) = workspace();
        let rs = RetrievalScope::new(Some(GroupId::from_raw(2).unwrap()), None);
        *rs.index.write().unwrap() = Arc::new(index(root.path(), &IngestScope::new("", None), 1));
        let scope = scope_of(&rs, &RepoFiles::overlay(ws));
        assert!(scope.folders.is_empty());
        assert_eq!(*scope.files, set(&[1, 2, 30]));
    }

    /// An index whose only committed unit is `key`.
    fn only(key: &str, generation: u64) -> IngestIndex {
        IngestIndex::of(
            HashMap::from([(key.to_string(), tl(1))]),
            Vec::new(),
            generation,
        )
    }

    /// **A turn rebuilds the index only once a unit has committed**, and at
    /// most once every [`REBUILD_EVERY`]; each rebuild publishes the next
    /// generation.
    #[test]
    fn a_turn_rebuilds_a_stale_index_at_most_once_a_second() {
        let rs = RetrievalScope::new(None, None);
        let t0 = Instant::now();
        let current = |at: Instant, key: &str| rs.current(at, |g| only(key, g));
        assert_eq!(current(t0, "a").generation(), 0, "nothing committed");

        rs.mark_stale();
        let first = current(t0, "a");
        assert_eq!((first.generation(), first.get("a")), (1, Some(tl(1))));

        rs.mark_stale();
        let soon = current(t0 + REBUILD_EVERY / 2, "b");
        assert_eq!(soon.generation(), 1, "too soon after the last");
        let later = current(t0 + REBUILD_EVERY, "b");
        assert_eq!((later.generation(), later.get("b")), (2, Some(tl(1))));
        assert_eq!(current(t0 + REBUILD_EVERY * 3, "c").generation(), 2);
    }

    /// **A unit committing during a rebuild is not lost**: the index is
    /// stale again once the rebuild publishes, and the next turn due reads
    /// it.
    #[test]
    fn a_commit_during_a_rebuild_leaves_the_index_stale() {
        let rs = RetrievalScope::new(None, None);
        let t0 = Instant::now();
        rs.mark_stale();
        rs.current(t0, |g| {
            rs.mark_stale();
            only("a", g)
        });
        assert!(rs.stale.load(Ordering::SeqCst));
        let next = rs.current(t0 + REBUILD_EVERY, |g| only("b", g));
        assert_eq!((next.generation(), next.get("b")), (2, Some(tl(1))));
    }

    /// **A rebuild lets go of what was worked out against the old index.**
    #[test]
    fn a_rebuild_lets_go_of_the_old_generation() {
        let (root, ws) = workspace();
        let rs = scoped(root.path());
        scope_of(&rs, &RepoFiles::overlay(ws));
        let tree = oid(root.path(), "main^{tree}");
        let shared_key = (1, vec![("r".to_string(), tree.clone())]);
        let tree_key = ("r".to_string(), tree, 1);
        assert!(rs.shared.lock().unwrap().get(&shared_key).is_some());
        assert!(rs.trees.lock().unwrap().get(&tree_key).is_some());

        rs.mark_stale();
        rs.current(Instant::now(), |g| only("a", g));
        assert!(rs.shared.lock().unwrap().get(&shared_key).is_none());
        assert!(rs.trees.lock().unwrap().get(&tree_key).is_none());
    }
}

//! The repository held in this process.
//!
//! Most of what the layer asks of a repository is a lookup in its own files: what
//! a ref points at, what a branch tracks, whether one commit is an ancestor of
//! another, what a remote's URL is. Asked of the `git` program each is a process
//! start, which costs tens of milliseconds — on Windows most of it the operating
//! system's scanning of the launch, which no setting of git's changes — and a
//! tool call asks dozens of them. libgit2 answers the same questions from the
//! same files in microseconds, so a repository it can hold is opened here when a
//! [`Repo`] is, and the reads that need no `git` run against it.
//!
//! **What stays with the `git` program:** the network (fetch, push, clone, with
//! the user's credentials and a partial clone's promises), the working tree
//! (checkout, merge, the index) and everything that writes. A repository libgit2
//! cannot hold — a `sha256` one — has no library at all, and every question
//! about it goes to `git`.
//!
//! libgit2 reads the files as git leaves them, so a ref moved or a remote added
//! by a `git` process a moment ago is what the next read here sees.

pub(crate) mod config;
pub(crate) mod history;
pub(crate) mod objects;
pub(crate) mod ref_txn;
pub(crate) mod refs;
pub(crate) mod trees;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::error::GitError;
use crate::types::{ObjectFormat, Oid};
use crate::Repo;

/// Open the working tree at `dir` in this process, or `None` when it is not one
/// libgit2 can hold: not a repository, not a top level, a bare repository, or an
/// object format libgit2 does not read. The caller asks `git`, which says why.
pub(crate) fn open_in_process(dir: &Path) -> Option<Repo> {
    let library = git2::Repository::open(dir).ok()?;
    let workdir = library.workdir()?;
    let same = matches!(
        (workdir.canonicalize(), dir.canonicalize()),
        (Ok(a), Ok(b)) if a == b
    );
    if !same {
        return None;
    }
    let git_dir: PathBuf = library.path().components().collect();
    Some(Repo {
        dir: dir.to_path_buf(),
        git_dir,
        format: ObjectFormat::Sha1,
        write: Mutex::new(()),
        library: Some(Mutex::new(library)),
    })
}

impl Repo {
    /// The repository held in this process, when there is one.
    pub(crate) fn library(&self) -> Option<MutexGuard<'_, git2::Repository>> {
        self.library
            .as_ref()
            .map(|held| held.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// This repository as a `git` process alone would see it — the way every
    /// question about a repository libgit2 cannot hold is answered — so a test
    /// can ask the same question both ways.
    #[cfg(test)]
    pub(crate) fn without_library(mut self) -> Self {
        self.library = None;
        self
    }
}

/// The id libgit2 holds, as this layer's.
pub(crate) fn oid_of(id: git2::Oid) -> Result<Oid, GitError> {
    Oid::parse(&id.to_string())
}

/// Whether libgit2 failed because what was asked for is not there — a ref, a
/// revision, an object — and not because anything went wrong.
pub(crate) fn is_absent(e: &git2::Error) -> bool {
    use git2::ErrorCode::{Ambiguous, InvalidSpec, NotFound, UnbornBranch};
    matches!(e.code(), NotFound | InvalidSpec | Ambiguous | UnbornBranch)
}

/// A library failure that is not an absence, as the layer's own error.
pub(crate) fn failed(what: &'static str, e: git2::Error) -> GitError {
    GitError::Unclassified {
        args: vec!["libgit2".to_string(), what.to_string()],
        status: None,
        stderr: e.message().to_string(),
    }
}

/// A revision lookup that failed: an unknown revision when nothing is there,
/// else a failure.
pub(crate) fn revision(e: git2::Error, spec: &str) -> GitError {
    // A revision that names an object which is no commit (or no tree) is as
    // unknown, to the caller, as one that names nothing.
    if is_absent(&e) || e.code() == git2::ErrorCode::Peel {
        GitError::UnknownRevision {
            rev: spec.to_string(),
        }
    } else {
        failed("revparse", e)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::GitError;
    use crate::testing::TestRepo;
    use crate::types::{BranchName, Oid, Rev};
    use crate::write::ref_txn::{RefOp, RefTransaction};
    use crate::Repo;

    #[test]
    fn a_repository_libgit2_can_hold_is_opened_in_process() {
        let t = TestRepo::init();
        let repo = Repo::open(&t.path).unwrap();
        assert!(repo.library().is_some());
        assert!(repo.git_dir().ends_with(".git"), "{:?}", repo.git_dir());
    }

    #[test]
    fn without_a_library_every_question_goes_to_git() {
        let t = TestRepo::init();
        assert!(Repo::open(&t.path)
            .unwrap()
            .without_library()
            .library()
            .is_none());
    }

    /// A symbolic ref whose target is gone is left out of a listing, as
    /// `for-each-ref` leaves out a broken ref — it does not fail the listing.
    #[test]
    fn a_dangling_symbolic_ref_is_left_out_of_a_listing() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        t.commit_all("first");
        t.git(&[
            "symbolic-ref",
            "refs/remotes/origin/HEAD",
            "refs/remotes/origin/main",
        ]);
        let library = t.repo();
        let git = t.repo().without_library();
        assert!(library.refs_under("refs/remotes/").unwrap().is_empty());
        assert_eq!(
            library.refs_under("refs/remotes/").unwrap(),
            git.refs_under("refs/remotes/").unwrap()
        );
    }

    /// A revision naming a tree is no commit: unknown, as `git` reports it.
    #[test]
    fn a_revision_that_names_no_commit_is_unknown() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        t.commit_all("first");
        let tree = Oid::parse(t.git(&["rev-parse", "HEAD^{tree}"]).trim()).unwrap();
        let library = t.repo();
        let git = t.repo().without_library();
        for repo in [&library, &git] {
            assert!(matches!(
                repo.resolve(&Rev::Oid(tree.clone())),
                Err(GitError::UnknownRevision { .. })
            ));
        }
    }

    /// Two ops on one ref are refused, as `update-ref` refuses them, and
    /// nothing is applied.
    #[test]
    fn two_ops_on_one_ref_are_refused() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let first = t.commit_all("first");
        t.write("a.txt", b"two\n");
        let second = t.commit_all("second");
        let repo = t.repo();
        let name = BranchName::parse("twice").unwrap().to_ref();
        let txn = RefTransaction::new()
            .push(RefOp::Create {
                name: name.clone(),
                new: first.clone(),
            })
            .push(RefOp::Create {
                name: name.clone(),
                new: second,
            });
        assert!(repo.update_refs(&txn).is_err());
        assert!(repo.without_library().update_refs(&txn).is_err());
        assert_eq!(t.repo().ref_target(&name).unwrap(), None);
    }

    /// **What a `git` process writes is what the next read here sees**: the
    /// library holds no copy of the files.
    #[test]
    fn a_ref_moved_by_git_after_the_open_is_seen() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let first = t.commit_all("first");
        let repo = t.repo();
        assert_eq!(repo.head().unwrap().oid(), Some(&first));
        t.write("a.txt", b"two\n");
        let second = t.commit_all("second");
        assert_eq!(repo.head().unwrap().oid(), Some(&second));
    }
}

//! A strongly typed interface over the `git` command line, for the
//! repositories a zend workspace lists. Design: `docs/zend_git.md`.
//!
//! Nothing outside this crate spawns `git`, formats a git argument or parses
//! git output. Three rules hold for every operation:
//!
//! - **The user's checkout is never touched.** No operation writes the working
//!   tree, the index or `HEAD`: commits are written as objects
//!   ([`Repo::commit_changes`]) and branches move by reference transaction
//!   ([`Repo::update_refs`]).
//! - **Every ref move is a compare-and-swap.** A local update names the value
//!   it replaces; a push names the value origin must hold ([`Lease`]).
//! - **Values are validated at the type boundary** ([`types`]), so an argument
//!   can never become a flag and a path can never leave the repository.

mod changeset;
mod classify;
mod error;
mod kill_tree;
mod read;
mod redact;
mod remote;
mod runner;
mod setup;
pub mod types;
pub mod version;
mod worktrees;
mod write;

#[cfg(test)]
mod testing;

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

pub use changeset::{Change, ChangeSet};
pub use error::GitError;
pub use read::blame::{BlameLine, LineRange};
pub use read::blob_reader::BlobReader;
pub use read::diff::{DiffEntry, DiffSide, DiffStatus};
pub use read::grep::{GrepHit, GrepQuery};
pub use read::head::Head;
pub use read::log::{CommitInfo, LogRange};
pub use read::patch::{FilePatch, Hunk, LineKind, PatchLine};
pub use read::refs::{Ancestor, Branch, Remote, Upstream};
pub use read::remote_branches::RemoteBranch;
pub use read::status::{StatusCode, StatusEntry, Xy};
pub use read::tags::Tag;
pub use read::tree::{ObjectKind, TreeEntry};
pub use remote::fetch::{FetchFlag, FetchSpec, RefUpdate};
pub use remote::ls_remote::RemoteRefs;
pub use remote::manage::UrlKind;
pub use remote::push::{
    Lease, PushAction, PushOutcome, PushResult, PushSpec, PushTarget, Rejection,
};
pub use setup::clone::CloneOptions;
pub use types::{
    BranchName, FileMode, GitTime, ObjectFormat, Oid, RefName, RemoteName, RemoteUrl, RepoPath,
    Rev, Signature, TagName,
};
pub use worktrees::{Worktree, WorktreeCheckout};
pub use write::apply::ApplyOutcome;
pub use write::merge_tree::MergeOutcome;
pub use write::pick::PickOutcome;
pub use write::ref_txn::{RefOp, RefTransaction};
pub use write::tags::TagAnnotation;

use runner::{utf8, Invocation};

/// One repository: the top level of a git working tree.
///
/// Reads run concurrently. Writes — objects, ref transactions, fetches and
/// pushes — are serialised by one lock per `Repo`, so two writers in this
/// process never race; correctness across processes rests on the
/// compare-and-swap every ref update carries, not on the lock.
pub struct Repo {
    dir: PathBuf,
    /// This working tree's own git folder (`.git`, or `.git/worktrees/<n>`
    /// for a linked worktree): private to the repository's owner, so the
    /// layer's scratch files go here rather than in a shared temp folder.
    git_dir: PathBuf,
    format: ObjectFormat,
    write: Mutex<()>,
}

impl std::fmt::Debug for Repo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Repo")
            .field("dir", &self.dir)
            .field("format", &self.format)
            .finish()
    }
}

impl Repo {
    /// Open the working tree whose top level is `dir`.
    ///
    /// Refuses a directory that is not itself a top level — a subfolder of a
    /// repository, or a plain folder nested inside one — because git would
    /// otherwise walk up and operate on the enclosing repository.
    pub fn open(dir: &Path) -> Result<Self, GitError> {
        version::installed()?;
        let not_a_repo = || GitError::NotARepository {
            dir: dir.to_path_buf(),
        };
        if !dir.is_dir() {
            return Err(not_a_repo());
        }
        let top = Invocation::new(dir, "rev-parse")
            .arg("--show-toplevel")
            .run_ok()
            .map_err(|e| match e {
                GitError::Unclassified { .. } => not_a_repo(),
                other => other,
            })?;
        let top = utf8("rev-parse", top)?;
        let top = PathBuf::from(top.trim_end_matches(['\n', '\r']));
        let same = match (top.canonicalize(), dir.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if !same {
            return Err(not_a_repo());
        }
        // The repository's hash, from its config: absent means SHA-1. Read
        // this way rather than with `rev-parse --show-object-format`, which
        // older releases lack.
        let format = Invocation::new(dir, "config")
            .args(["--get", "extensions.objectformat"])
            .run_accepting(&[0, 1])?;
        let format = match format.status {
            Some(0) => ObjectFormat::parse(utf8("config", format.stdout)?.trim())?,
            _ => ObjectFormat::Sha1,
        };
        let git_dir = Invocation::new(dir, "rev-parse")
            .arg("--absolute-git-dir")
            .run_ok()?;
        let git_dir = PathBuf::from(utf8("rev-parse", git_dir)?.trim_end_matches(['\n', '\r']));
        Ok(Self {
            dir: dir.to_path_buf(),
            git_dir,
            format,
            write: Mutex::new(()),
        })
    }

    /// This working tree's own git folder.
    pub(crate) fn git_dir(&self) -> &Path {
        &self.git_dir
    }

    /// The working tree's top level.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The hash the repository's object ids use.
    pub fn format(&self) -> ObjectFormat {
        self.format
    }

    /// A git invocation in this repository.
    pub(crate) fn git(&self, subcommand: &str) -> Invocation {
        Invocation::new(&self.dir, subcommand)
    }

    /// The write lock. Poisoning is ignored: the lock guards no data, only
    /// the ordering of writes.
    pub(crate) fn write_lock(&self) -> MutexGuard<'_, ()> {
        self.write.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{scratch, TestRepo};

    #[test]
    fn a_repository_top_level_opens() {
        let t = TestRepo::init();
        let repo = Repo::open(&t.path).unwrap();
        assert_eq!(repo.format(), ObjectFormat::Sha1);
    }

    /// **A folder nested inside a repository is refused**, not resolved to the
    /// enclosing repository. The scratch folder sits inside the candle
    /// checkout, so without this check git would operate on candle itself.
    #[test]
    fn a_plain_folder_inside_a_repository_is_refused() {
        let dir = tempfile::Builder::new()
            .prefix("plain-")
            .tempdir_in(scratch())
            .unwrap();
        assert!(matches!(
            Repo::open(dir.path()),
            Err(GitError::NotARepository { .. })
        ));
    }

    #[test]
    fn a_subfolder_of_a_repository_is_refused() {
        let t = TestRepo::init();
        std::fs::create_dir_all(t.path.join("src")).unwrap();
        assert!(matches!(
            Repo::open(&t.path.join("src")),
            Err(GitError::NotARepository { .. })
        ));
    }

    #[test]
    fn a_folder_outside_any_repository_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            Repo::open(dir.path()),
            Err(GitError::NotARepository { .. })
        ));
    }

    #[test]
    fn a_missing_folder_is_refused() {
        let t = TestRepo::init();
        assert!(matches!(
            Repo::open(&t.path.join("nope")),
            Err(GitError::NotARepository { .. })
        ));
    }
}

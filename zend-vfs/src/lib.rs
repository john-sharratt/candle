//! The daemon's view of its workspace repositories: what is in them, what each
//! conversation has changed in them, and the checkout tools run on.
//!
//! Three layers, each built on the one before:
//!
//! - **The git layer** — a strongly typed interface over the `git` command line
//!   ([`Repo`] and everything on it). Design: `docs/zend_git.md`.
//! - **The file layer** — the [`workspace`] manifest; a conversation's changes
//!   as deltas ([`file_delta`], [`file_changes`]); the overlay the `file_*`
//!   tools read and write through ([`vfs`], one store per repository gathered in
//!   [`files`]); and the unified-diff engine behind `file_edit` ([`patch`]).
//! - **The execution checkout** ([`checkout`]) — a conversation's changes
//!   materialised onto a real checkout for a tool to run on, and what the tool
//!   changed captured back as deltas.
//! - **The sandbox** ([`sandbox`]) — one repository's whole command run as a
//!   conversation: lock, set aside what the checkout holds, put it on the
//!   conversation's branch, lay its changes down, check and run the command,
//!   read back and record what it changed, put the checkout's own state back,
//!   unlock — and its server, which runs jobs in the background with their
//!   output in a log file.
//!
//! A branch's record is origin ([`origin`]): a write goes there first and the
//! local branch follows; a read stays local. A conversation's work joins a
//! branch through [`work`]: a commit published whole or not at all, and a
//! merge into the conversation's own copy.
//!
//! Writing the disk takes a [`DiskWriteGrant`], which only the tool layer's
//! capability check issues.
//!
//! Nothing outside this crate spawns `git`, formats a git argument or parses
//! git output. Three rules hold for every git operation:
//!
//! - **The user's checkout is never touched.** No operation writes the working
//!   tree, the index or `HEAD`: commits are written as objects
//!   ([`Repo::commit_changes`]) and branches move by reference transaction
//!   ([`Repo::update_refs`]). The one exception is
//!   [`Repo::force_checkout_branch`] and [`Repo::restore_paths`], for a
//!   checkout the daemon owns and runs tools in, never a developer's.
//! - **Every ref move is a compare-and-swap.** A local update names the value
//!   it replaces; a push names the value origin must hold ([`Lease`]).
//! - **Values are validated at the type boundary** ([`types`]), so an argument
//!   can never become a flag and a path can never leave the repository.

mod changeset;
pub mod checkout;
mod classify;
mod disk_grant;
mod error;
mod execution;
pub mod file_changes;
pub mod file_delta;
pub mod files;
mod kill_tree;
pub mod origin;
pub mod patch;
mod read;
mod redact;
mod remote;
mod runner;
pub mod sandbox;
mod setup;
pub mod types;
pub mod version;
pub mod vfs;
pub mod work;
pub mod workspace;
mod worktrees;
mod write;

#[cfg(test)]
mod testing;

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

pub use changeset::{Change, ChangeSet};
pub use disk_grant::DiskWriteGrant;
pub use error::GitError;
pub use file_changes::FileChanges;
pub use file_delta::{FileDelta, FileTimes, Splice, TimedDelta};
pub use files::{RepoFiles, UnknownRepo};
pub use origin::{Published, Pulled, ORIGIN};
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
pub use read::tree::{ObjectKind, SizedEntry, TreeEntry};
pub use remote::fetch::{FetchFlag, FetchSpec, RefUpdate};
pub use remote::ls_remote::RemoteRefs;
pub use remote::manage::UrlKind;
pub use remote::push::{
    Lease, PushAction, PushOutcome, PushResult, PushSpec, PushTarget, Rejection,
};
pub use sandbox::{
    CommandPolicy, Job, JobHandle, JobId, JobInfo, JobNotFound, JobRequest, JobStatus,
    OutputStream, RunOutcome, Sandbox, SandboxCommand, SandboxError, SandboxServer, StartedJob,
    JOBS_DIR,
};
pub use setup::clone::CloneOptions;
pub use types::{
    BranchName, FileMode, GitTime, ObjectFormat, Oid, RefName, RemoteName, RemoteUrl, RepoPath,
    Rev, Signature, TagName,
};
pub use vfs::git_source::GitSource;
pub use vfs::{has_markers, Base, FileState, Snapshot, VfsError, VfsStore};
pub use work::{merge_into, Committing, Landed, Merged, NotCommitted};
pub use workspace::{RepoSpec, Workspace, WorkspaceError, ALL_REPOS, MANIFEST_FILE};
pub use worktrees::{Worktree, WorktreeCheckout};
pub use write::apply::ApplyOutcome;
pub use write::merge_text::{MergeLabels, MergedText};
pub use write::merge_tree::{MergeOutcome, PartialMerge};
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

impl fmt::Debug for Repo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        // The top level, this working tree's git folder and the repository's
        // hash, in one process.
        let located = Invocation::new(dir, "rev-parse")
            .args([
                "--show-toplevel",
                "--absolute-git-dir",
                "--show-object-format",
            ])
            .run_ok()
            .map_err(|e| match e {
                GitError::Unclassified { .. } => not_a_repo(),
                other => other,
            })?;
        let located = utf8("rev-parse", located)?;
        let mut lines = located.lines();
        let (Some(top), Some(git_dir)) = (lines.next(), lines.next()) else {
            return Err(GitError::malformed("rev-parse", located.clone()));
        };
        let reported_format = lines.next().and_then(|f| ObjectFormat::parse(f).ok());
        let top = PathBuf::from(top);
        let git_dir = PathBuf::from(git_dir);
        let same = match (top.canonicalize(), dir.canonicalize()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if !same {
            return Err(not_a_repo());
        }
        let format = match reported_format {
            Some(format) => format,
            // Releases before 2.25 lack `--show-object-format` and echo it
            // back instead: the hash is then read from the config, where
            // absent means SHA-1.
            None => {
                let format = Invocation::new(dir, "config")
                    .args(["--get", "extensions.objectformat"])
                    .run_accepting(&[0, 1])?;
                match format.status {
                    Some(0) => ObjectFormat::parse(utf8("config", format.stdout)?.trim())?,
                    _ => ObjectFormat::Sha1,
                }
            }
        };
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

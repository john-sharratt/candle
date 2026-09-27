//! Putting a conversation onto a checkout: its branch, and its changes laid
//! over the branch's commit.
//!
//! The result is what `git reset --hard`, a switch to the branch, and then
//! writing every changed file would leave — `HEAD` on the branch, every
//! tracked file at the branch's commit (the *base*) or at the conversation's
//! content, every file a previous run added gone — but reached by touching
//! only files whose bytes are wrong. A file that already holds what it should
//! is left alone, timestamps and all; that is what keeps a build cache over
//! the checkout valid from one run to the next. Three things make that
//! possible:
//!
//! - the checkout is reset onto the branch only when `HEAD` is not on it at its
//!   commit already, or the index has been changed (a tool that ran `git add`,
//!   say);
//! - otherwise `git status` names exactly the files that differ from the base,
//!   and only those — plus the files the previous run wrote, which may be
//!   ignored and so invisible to status — are looked at;
//! - the [`Ledger`] from the previous run vouches for a file's content by its
//!   stamp, so a file this conversation's changes cover is read only when its
//!   stamp says it may have changed.
//!
//! A file that is written is dated by the write. One checkout serves every
//! conversation, and a build cache over it compares a source's modification
//! time with its outputs': a file dated by when its conversation last changed
//! it could be older than outputs another conversation just built, and the
//! cache would count it as built. A file already holding its content is not
//! written, and keeps whatever time it had.
//!
//! Everything a conversation's changes name is resolved and computed before
//! anything on disk changes: a path that could lead outside the checkout, or a
//! chain that does not fit the base commit, fails the call with the checkout as
//! it was.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use super::error::CheckoutError;
use super::ledger::Ledger;
use super::stamp::FileStamp;
use super::target::{self, Found, Target};
use crate::{
    BranchName, DiskWriteGrant, FileChanges, Head, Repo, RepoPath, Rev, StatusCode, StatusEntry,
};

/// What a [`materialize`] did, path by path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Materialized {
    /// Whether the checkout was reset onto the branch first.
    pub reset: bool,
    /// Whether `HEAD` was somewhere else — another branch, or detached — and
    /// was moved onto the branch by that reset.
    pub switched: bool,
    /// Tracked files a previous run left changed, put back to the base.
    pub restored: Vec<String>,
    /// Files a previous run added, removed.
    pub removed: Vec<String>,
    /// Files written, or removed, to hold the conversation's content.
    pub written: Vec<String>,
    /// Files that already held the conversation's content and were not
    /// touched.
    pub kept: Vec<String>,
}

/// Put the checkout `repo` works in on `branch`, with `changes` laid over the
/// branch's commit.
///
/// `previous` is the ledger the last [`materialize`] or
/// [`capture`](super::capture) on this checkout left; `None` when there is
/// none, which costs reads but never correctness. Returns the checkout's new
/// ledger — what [`capture`](super::capture) takes after the tool has run —
/// and what was done.
///
/// **Overwrites the checkout.** Only for one the daemon owns, and only while
/// holding whatever keeps a second run off it.
pub fn materialize(
    repo: &Repo,
    _grant: &DiskWriteGrant,
    branch: &BranchName,
    changes: &FileChanges,
    previous: Option<Ledger>,
) -> Result<(Ledger, Materialized), CheckoutError> {
    let root = repo.dir().to_path_buf();
    let (head, mut status) = repo.status_with_head()?;
    // On the branch, `HEAD`'s commit is the branch's; elsewhere it is read.
    let base = match &head {
        Head::Branch { branch: on, oid } if on == branch => oid.clone(),
        _ => repo.resolve(&Rev::Branch(branch.clone()))?,
    };
    let base_rev = Rev::Oid(base.clone());

    // Everything the conversation's changes name, resolved and computed first
    // — the base's copies read in one pass, and only where a chain opens with
    // an edit: one that opens with a write or a delete replaces the base.
    let targets = changes
        .iter()
        .map(|(path, _)| Ok((path.to_string(), target::resolve(&root, path)?)))
        .collect::<Result<Vec<(String, Target)>, CheckoutError>>()?;
    let edited: Vec<&RepoPath> = targets
        .iter()
        .filter(|(path, _)| {
            changes
                .chain(path)
                .and_then(|chain| chain.first())
                .is_some_and(|first| !first.delta.supersedes())
        })
        .map(|(_, t)| &t.repo_path)
        .collect();
    let mut at_base = repo.read_checked_out_all(&base_rev, &edited)?;
    let mut wanted: BTreeMap<String, (Target, Option<Vec<u8>>)> = BTreeMap::new();
    for (path, target) in targets {
        let base_copy = at_base.remove(target.repo_path.as_str());
        let content = changes
            .replay(&path, base_copy)
            .map_err(|_| CheckoutError::Diverged { path: path.clone() })?;
        wanted.insert(path, (target, content));
    }

    let mut done = Materialized::default();
    let on_branch = head.branch() == Some(branch);
    let at_base = on_branch && head.oid() == Some(&base);
    if !at_base || status.iter().any(changes_the_index) {
        // The branch is read again by the checkout itself; a commit landing on
        // it in between is what the checkout — and so the ledger — is at.
        let landed = repo.force_checkout_branch(branch)?;
        if landed != base {
            return Err(CheckoutError::BranchMoved {
                branch: branch.to_string(),
            });
        }
        status = repo.status()?;
        done.reset = true;
        done.switched = !on_branch;
    }
    let previous_paths: Vec<String> = previous
        .as_ref()
        .map(|l| l.paths().map(str::to_string).collect())
        .unwrap_or_default();
    // What the previous ledger vouches for holds only while the checkout is
    // where it left it.
    let known = previous.filter(|l| !done.reset && l.base() == base.as_str());

    // Tracked files a previous run changed, back to the base; files it added,
    // gone. Both only where the conversation does not want them itself.
    let mut restore: Vec<RepoPath> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for entry in &status {
        let path = entry.path().as_str().to_string();
        seen.insert(path.clone());
        if wanted.contains_key(&path) {
            continue;
        }
        match entry {
            StatusEntry::Untracked { .. } => remove(&root, &path, &mut done.removed)?,
            _ => restore.push(entry.path().clone()),
        }
    }
    if !restore.is_empty() {
        for path in &restore {
            clear_the_way(&root, path.as_str(), &mut done.removed)?;
        }
        let refs: Vec<&RepoPath> = restore.iter().collect();
        repo.restore_paths(&base, &refs)?;
        done.restored = restore.iter().map(|p| p.as_str().to_string()).collect();
    }
    // A file the previous run wrote that git does not report — an ignored one —
    // goes too, unless the base holds it (then status would have named it had
    // it differed). A link standing there goes whatever the base holds.
    let mut leftovers = Vec::new();
    for path in previous_paths {
        if wanted.contains_key(&path) || seen.contains(&path) {
            continue;
        }
        let (target, found) = target::inspect(&root, &path)?;
        let linked = matches!(found, Found::Link | Found::BehindLink { .. });
        leftovers.push((path, target, linked));
    }
    let leftover_paths: Vec<&RepoPath> = leftovers.iter().map(|(_, t, _)| &t.repo_path).collect();
    let held = repo.files_at(&base_rev, &leftover_paths)?;
    for (path, target, linked) in &leftovers {
        if *linked || !held.contains(target.repo_path.as_str()) {
            remove(&root, path, &mut done.removed)?;
        }
    }

    // The conversation's own files: written only where they differ.
    let mut ledger = Ledger::new(base.as_str());
    for (path, (target, content)) in wanted {
        let stamp = stamp(&target.abs, &path)?;
        let holds = match known.as_ref().and_then(|l| l.verified(&path, stamp)) {
            Some(known) => known.map(<[u8]>::to_vec),
            None => read(&target.abs, &path)?,
        };
        if holds == content {
            done.kept.push(path.clone());
            ledger.record(path, content, stamp);
            continue;
        }
        match &content {
            Some(bytes) => write(&target.abs, &path, bytes)?,
            None => {
                make_writable(&target.abs)
                    .and_then(|()| std::fs::remove_file(&target.abs))
                    .map_err(|e| CheckoutError::io(&path, e))?;
            }
        }
        let after = self::stamp(&target.abs, &path)?;
        done.written.push(path.clone());
        ledger.record(path, content, after);
    }
    ledger.seal();
    Ok((ledger, done))
}

/// Whether a status entry reflects a change to the index — something staged, a
/// rename, a conflict — which only a reset takes back.
fn changes_the_index(entry: &StatusEntry) -> bool {
    match entry {
        StatusEntry::Changed { xy, .. } => xy.index != StatusCode::Unmodified,
        StatusEntry::Renamed { .. } | StatusEntry::Unmerged { .. } => true,
        StatusEntry::Untracked { .. } => false,
    }
}

/// `abs`'s stamp; a folder where a file should be is refused.
pub(crate) fn stamp(abs: &Path, path: &str) -> Result<Option<FileStamp>, CheckoutError> {
    if abs.is_dir() {
        return Err(CheckoutError::unsafe_path(
            path,
            "names a folder, not a file",
        ));
    }
    FileStamp::of(abs).map_err(|e| CheckoutError::io(path, e))
}

/// `abs`'s bytes, or `None` when nothing is there.
pub(crate) fn read(abs: &Path, path: &str) -> Result<Option<Vec<u8>>, CheckoutError> {
    match std::fs::read(abs) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CheckoutError::io(path, e)),
    }
}

/// Write `bytes` to `abs` through a sibling temporary file renamed over it, so
/// a write cut short leaves the old file whole rather than truncated. Creates
/// the parent folders; a read-only file a tool left there is replaced all the
/// same.
fn write(abs: &Path, path: &str, bytes: &[u8]) -> Result<(), CheckoutError> {
    let fail = |e| CheckoutError::io(path, e);
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(fail)?;
    }
    let name = abs
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let temp = abs.with_file_name(format!(".{name}.zend-materialize"));
    let written = std::fs::File::create(&temp)
        .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
        .and_then(|()| make_writable(abs))
        .and_then(|()| std::fs::rename(&temp, abs));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&temp);
        return Err(fail(e));
    }
    Ok(())
}

/// Clear a read-only flag a tool left on `abs`, so the checkout can replace or
/// remove the file. Nothing there, or not a file, is left as it is.
fn make_writable(abs: &Path) -> std::io::Result<()> {
    let meta = match std::fs::symlink_metadata(abs) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if !meta.is_file() || !meta.permissions().readonly() {
        return Ok(());
    }
    let mut permissions = meta.permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(permissions.mode() | 0o200);
    }
    // On Windows the read-only attribute is the whole of it; nothing is
    // opened to anyone else.
    #[cfg(windows)]
    #[allow(clippy::permissions_set_readonly_false)]
    permissions.set_readonly(false);
    std::fs::set_permissions(abs, permissions)
}

/// Remove a link at `abs` — the link itself, never what it points at. A
/// junction or a link to a folder is removed as a folder entry, which on
/// every platform takes the link and leaves its target.
fn remove_link(abs: &Path, path: &str) -> Result<(), CheckoutError> {
    match std::fs::remove_file(abs).or_else(|_| std::fs::remove_dir(abs)) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(CheckoutError::io(path, e)),
    }
}

/// Before a tracked file is restored at `path`: a link a tool put there, or
/// on the way to it, is removed so the restore cannot write through it, and
/// an empty folder a tool left in the file's place is removed so the file
/// can go back.
fn clear_the_way(root: &Path, path: &str, removed: &mut Vec<String>) -> Result<(), CheckoutError> {
    let (target, found) = target::inspect(root, path)?;
    match found {
        Found::Link => {
            remove_link(&target.abs, path)?;
            removed.push(path.to_string());
        }
        Found::BehindLink { rel, abs } => {
            remove_link(&abs, &rel)?;
            removed.push(rel);
        }
        Found::Folder => match std::fs::remove_dir(&target.abs) {
            Ok(()) => removed.push(path.to_string()),
            // Something other than the run's own files is in it; git says so.
            Err(e) if e.kind() == std::io::ErrorKind::DirectoryNotEmpty => {}
            Err(e) => return Err(CheckoutError::io(path, e)),
        },
        Found::Absent | Found::File => {}
    }
    Ok(())
}

/// Remove what a previous run added at `path`: a file, or a link — the link
/// itself, or the one on the way to `path`, never what it points at — and
/// then every folder above it the removal left empty, so no trace of the run's
/// folders is left for the next conversation to list. A folder — an untracked
/// repository nested in the checkout — is never removed.
fn remove(root: &Path, path: &str, removed: &mut Vec<String>) -> Result<(), CheckoutError> {
    let (target, found) = target::inspect(root, path)?;
    let gone = match found {
        Found::Absent | Found::Folder => return Ok(()),
        Found::Link => {
            remove_link(&target.abs, path)?;
            removed.push(path.to_string());
            target.abs
        }
        Found::BehindLink { rel, abs } => {
            remove_link(&abs, &rel)?;
            removed.push(rel);
            abs
        }
        Found::File => {
            match make_writable(&target.abs).and_then(|()| std::fs::remove_file(&target.abs)) {
                Ok(()) => removed.push(path.to_string()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(CheckoutError::io(path, e)),
            }
            target.abs
        }
    };
    prune_empty_folders(root, &gone);
    Ok(())
}

/// Remove each folder above `gone`, up to but never including `root`, for as
/// long as it is empty. The first folder that holds anything — or cannot be
/// removed for any reason — ends it: a folder left behind is untidy, never
/// wrong.
fn prune_empty_folders(root: &Path, gone: &Path) {
    let mut folder = gone.parent();
    while let Some(dir) = folder {
        if dir == root || !dir.starts_with(root) || std::fs::remove_dir(dir).is_err() {
            return;
        }
        folder = dir.parent();
    }
}

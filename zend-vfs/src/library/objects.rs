//! Writing and reading objects in this process: blobs stored as they are, trees
//! built from a list of entries, commits made from a tree, and the index's
//! entries.

use std::path::Path;

use git2::build::TreeUpdateBuilder;
use git2::{Commit, FileMode as LibMode, Index, IndexEntry, IndexTime, Repository, Time};

use super::{failed, is_absent, oid_of};
use crate::error::GitError;
use crate::types::{FileMode, Oid, RepoPath, Signature, TagName};

/// The extended index flag marking an entry added with `--intent-to-add`.
const INTENT_TO_ADD: u16 = 1 << 13;
/// The index flag marking an entry git is told to treat as unchanged.
const ASSUME_VALID: u16 = 0x8000;
/// The extended index flag marking an entry git is told to leave out of the
/// working tree.
const SKIP_WORKTREE: u16 = 1 << 14;

fn lib_id(id: &Oid) -> Result<git2::Oid, GitError> {
    git2::Oid::from_str(id.as_str()).map_err(|e| failed("from_str", e))
}

/// `bytes` stored as a blob, unconverted.
pub(crate) fn hash_raw(lib: &Repository, bytes: &[u8]) -> Result<Oid, GitError> {
    oid_of(lib.blob(bytes).map_err(|e| failed("blob", e))?)
}

/// The blob `id`'s bytes; `None` when the repository does not hold it.
pub(crate) fn blob_bytes(lib: &Repository, id: &Oid) -> Result<Option<Vec<u8>>, GitError> {
    match lib.find_blob(lib_id(id)?) {
        Ok(blob) => Ok(Some(blob.content().to_vec())),
        Err(e) if is_absent(&e) => Ok(None),
        Err(e) => Err(failed("find_blob", e)),
    }
}

/// The tree holding exactly `entries`, written to the object store. A private
/// index in memory orders them as a tree is ordered.
pub(crate) fn tree_from_entries(
    lib: &Repository,
    entries: &[(FileMode, Oid, String)],
) -> Result<Oid, GitError> {
    let mut index = Index::new().map_err(|e| failed("index", e))?;
    for (mode, id, path) in entries {
        let stamp = IndexTime::new(0, 0);
        let entry = IndexEntry {
            ctime: stamp,
            mtime: stamp,
            dev: 0,
            ino: 0,
            mode: u32::from_str_radix(mode.as_str(), 8)
                .map_err(|_| GitError::malformed("libgit2", mode.as_str().to_string()))?,
            uid: 0,
            gid: 0,
            file_size: 0,
            id: lib_id(id)?,
            flags: 0,
            flags_extended: 0,
            path: path.clone().into_bytes(),
        };
        index.add(&entry).map_err(|e| failed("index_add", e))?;
    }
    oid_of(
        index
            .write_tree_to(lib)
            .map_err(|e| failed("write_tree", e))?,
    )
}

/// The tree the repository's own index holds, or the reason it cannot be
/// written as one when the index holds unresolved conflicts.
pub(crate) fn index_tree(lib: &Repository) -> Result<Result<Oid, String>, GitError> {
    let mut index = lib.index().map_err(|e| failed("index", e))?;
    index.read(true).map_err(|e| failed("index_read", e))?;
    // An entry added with `--intent-to-add` stands in the index as an empty
    // blob, and is no part of the tree `write-tree` makes.
    if index
        .iter()
        .any(|entry| entry.flags_extended & INTENT_TO_ADD != 0)
    {
        if index.has_conflicts() {
            return Ok(Err("the index holds unresolved conflicts".to_string()));
        }
        let mut without = Index::new().map_err(|e| failed("index", e))?;
        for entry in index
            .iter()
            .filter(|e| e.flags_extended & INTENT_TO_ADD == 0)
        {
            without.add(&entry).map_err(|e| failed("index_add", e))?;
        }
        return Ok(Ok(oid_of(
            without
                .write_tree_to(lib)
                .map_err(|e| failed("write_tree", e))?,
        )?));
    }
    match index.write_tree() {
        Ok(id) => Ok(Ok(oid_of(id)?)),
        Err(e) if e.code() == git2::ErrorCode::Unmerged => Ok(Err(e.message().to_string())),
        Err(e) => Err(failed("write_tree", e)),
    }
}

/// One entry of the repository's index, as `ls-files -v -s` lists it.
pub(crate) struct Indexed {
    pub path: String,
    pub mode: FileMode,
    pub assume_unchanged: bool,
    pub skip_worktree: bool,
}

/// Every entry the repository's index holds, in index order.
pub(crate) fn index_listing(lib: &Repository) -> Result<Vec<Indexed>, GitError> {
    let mut index = lib.index().map_err(|e| failed("index", e))?;
    index.read(true).map_err(|e| failed("index_read", e))?;
    index
        .iter()
        .map(|entry| {
            let path = String::from_utf8(entry.path.clone())
                .map_err(|_| GitError::malformed("libgit2", "an index path that is not UTF-8"))?;
            Ok(Indexed {
                path,
                mode: FileMode::parse(&format!("{:06o}", entry.mode))?,
                assume_unchanged: entry.flags & ASSUME_VALID != 0,
                skip_worktree: entry.flags_extended & SKIP_WORKTREE != 0,
            })
        })
        .collect()
}

/// libgit2's name for `mode`.
fn lib_mode(mode: FileMode) -> LibMode {
    match mode {
        FileMode::Regular => LibMode::Blob,
        FileMode::Executable => LibMode::BlobExecutable,
        FileMode::Symlink => LibMode::Link,
        FileMode::Tree => LibMode::Tree,
        FileMode::Submodule => LibMode::Commit,
    }
}

/// The tree `base` with each change applied: a path set to a mode and blob,
/// or — `None` — removed. A removal of a path `base` does not hold changes
/// nothing, as a fast-import `D` of it does.
pub(crate) fn tree_with(
    lib: &Repository,
    base: &Oid,
    changes: &[(&RepoPath, Option<(FileMode, &Oid)>)],
) -> Result<Oid, GitError> {
    let baseline = lib
        .find_object(lib_id(base)?, None)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| failed("peel_to_tree", e))?;
    let mut update = TreeUpdateBuilder::new();
    for (path, change) in changes {
        match change {
            Some((mode, id)) => {
                update.upsert(path.as_str(), lib_id(id)?, lib_mode(*mode));
            }
            None => {
                if baseline.get_path(Path::new(path.as_str())).is_ok() {
                    update.remove(path.as_str());
                }
            }
        }
    }
    oid_of(
        update
            .create_updated(lib, &baseline)
            .map_err(|e| failed("tree_create_updated", e))?,
    )
}

fn signed(who: &Signature) -> Result<git2::Signature<'static>, GitError> {
    git2::Signature::new(
        who.name(),
        who.email(),
        &Time::new(who.when.seconds, who.when.offset_minutes),
    )
    .map_err(|e| failed("signature", e))
}

/// An annotated tag object naming `target` as `name`, written; its id. No ref
/// is made — the caller's transaction makes it.
pub(crate) fn tag_object(
    lib: &Repository,
    target: &Oid,
    name: &TagName,
    tagger: &Signature,
    message: &str,
) -> Result<Oid, GitError> {
    let object = lib
        .find_object(lib_id(target)?, None)
        .map_err(|e| failed("find_object", e))?;
    oid_of(
        lib.tag_annotation_create(name.as_str(), &object, &signed(tagger)?, message)
            .map_err(|e| failed("tag_annotation_create", e))?,
    )
}

/// A commit of `tree` over `parents`, in order, with `message` as it stands.
/// Moves no ref.
pub(crate) fn commit_tree(
    lib: &Repository,
    tree: &Oid,
    parents: &[&Oid],
    message: &str,
    author: &Signature,
    committer: &Signature,
) -> Result<Oid, GitError> {
    let tree = lib
        .find_tree(lib_id(tree)?)
        .map_err(|e| failed("find_tree", e))?;
    let parents: Vec<Commit<'_>> = parents
        .iter()
        .map(|p| {
            lib.find_commit(lib_id(p)?)
                .map_err(|e| failed("find_commit", e))
        })
        .collect::<Result<_, _>>()?;
    let parent_refs: Vec<&Commit<'_>> = parents.iter().collect();
    let id = lib
        .commit(
            None,
            &signed(author)?,
            &signed(committer)?,
            message,
            &tree,
            &parent_refs,
        )
        .map_err(|e| failed("commit", e))?;
    oid_of(id)
}

#[cfg(test)]
mod tests;

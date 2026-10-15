//! Three-way tree merges from libgit2: `merge_trees` in memory, with no rename
//! detection — a file renamed on one side and edited on the other is a
//! conflict, as the plumbing merge reports it.

use std::collections::BTreeSet;
use std::path::Path;
use std::str;

use git2::{IndexEntry, MergeOptions, Oid as LibOid, Repository, Tree};

use super::{failed, oid_of, revision};
use crate::error::GitError;
use crate::types::{Oid, RepoPath};

/// The stage bits of an index entry's flags.
const STAGE_MASK: u16 = 0x3000;

fn tree<'r>(lib: &'r Repository, oid: &Oid) -> Result<Tree<'r>, GitError> {
    let id = LibOid::from_str(oid.as_str()).map_err(|e| failed("from_str", e))?;
    lib.find_object(id, None)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| revision(e, oid.as_str()))
}

fn path_of(entry: &IndexEntry) -> Result<RepoPath, GitError> {
    let text = str::from_utf8(&entry.path)
        .map_err(|_| GitError::malformed("libgit2", "a merged path that is not UTF-8"))?;
    RepoPath::parse(text)
}

/// Merge `theirs` into `ours` over `base` (each a commit or a tree): the tree,
/// unless paths conflict and `keep_ours` is false — then `None` — and the
/// conflicting paths, in order. Kept as ours, each conflicting path is left as
/// `ours` holds it, absent where `ours` has none.
pub(crate) fn merge_trees(
    lib: &Repository,
    base: &Oid,
    ours: &Oid,
    theirs: &Oid,
    keep_ours: bool,
) -> Result<(Option<Oid>, Vec<RepoPath>), GitError> {
    let (b, o, t) = (tree(lib, base)?, tree(lib, ours)?, tree(lib, theirs)?);
    let mut options = MergeOptions::new();
    options.find_renames(false);
    let mut index = lib
        .merge_trees(&b, &o, &t, Some(&options))
        .map_err(|e| failed("merge_trees", e))?;

    let mut conflicts: BTreeSet<RepoPath> = BTreeSet::new();
    let mut ours_kept: Vec<(RepoPath, Option<IndexEntry>)> = Vec::new();
    for conflict in index.conflicts().map_err(|e| failed("conflicts", e))? {
        let conflict = conflict.map_err(|e| failed("conflicts", e))?;
        let any = conflict
            .our
            .as_ref()
            .or(conflict.their.as_ref())
            .or(conflict.ancestor.as_ref())
            .ok_or_else(|| GitError::malformed("libgit2", "a conflict with no side"))?;
        let path = path_of(any)?;
        if conflicts.insert(path.clone()) {
            ours_kept.push((path, conflict.our));
        }
    }
    let conflicts: Vec<RepoPath> = conflicts.into_iter().collect();
    if !conflicts.is_empty() && !keep_ours {
        return Ok((None, conflicts));
    }
    for (path, our) in ours_kept {
        index
            .conflict_remove(Path::new(path.as_str()))
            .map_err(|e| failed("conflict_remove", e))?;
        if let Some(mut entry) = our {
            entry.flags &= !STAGE_MASK;
            index.add(&entry).map_err(|e| failed("index_add", e))?;
        }
    }
    let tree = index
        .write_tree_to(lib)
        .map_err(|e| failed("write_tree", e))?;
    Ok((Some(oid_of(tree)?), conflicts))
}

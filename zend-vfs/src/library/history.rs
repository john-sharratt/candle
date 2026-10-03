//! How commits stand to one another, from libgit2: first-parent steps, common
//! ancestors, ancestry.

use git2::Repository;

use super::{failed, is_absent, oid_of, revision};
use crate::error::GitError;
use crate::read::refs::Ancestor;
use crate::types::{Oid, Rev};

/// The commit `rev` names, as libgit2 holds it.
fn commit_of(lib: &Repository, rev: &Rev) -> Result<git2::Oid, GitError> {
    let spec = rev.spec();
    let object = lib.revparse_single(&spec).map_err(|e| revision(e, &spec))?;
    let commit = object.peel_to_commit().map_err(|e| revision(e, &spec))?;
    Ok(commit.id())
}

/// The commit `back` steps behind `base` along first parents, or how far back
/// the history does go.
pub(crate) fn first_parent_ancestor(
    lib: &Repository,
    base: &Oid,
    back: u32,
) -> Result<Ancestor, GitError> {
    let start = git2::Oid::from_str(base.as_str()).map_err(|e| failed("from_str", e))?;
    let mut commit = lib
        .find_commit(start)
        .map_err(|e| failed("find_commit", e))?;
    // The first-parent chain, newest first: its length is the history's depth
    // when `back` runs past it.
    let mut steps = 0u32;
    loop {
        if steps == back {
            return Ok(Ancestor::Found(oid_of(commit.id())?));
        }
        match commit.parent(0) {
            Ok(parent) => commit = parent,
            Err(e) if is_absent(&e) => break,
            Err(e) => return Err(failed("parent", e)),
        }
        steps += 1;
    }
    Ok(Ancestor::PastRoot { depth: steps })
}

/// The best common ancestor of `a` and `b`, if they share history.
pub(crate) fn merge_base(lib: &Repository, a: &Rev, b: &Rev) -> Result<Option<Oid>, GitError> {
    let (a, b) = (commit_of(lib, a)?, commit_of(lib, b)?);
    match lib.merge_base(a, b) {
        Ok(base) => Ok(Some(oid_of(base)?)),
        Err(e) if is_absent(&e) => Ok(None),
        Err(e) => Err(failed("merge_base", e)),
    }
}

/// Whether `ancestor` is reachable from `descendant`; a commit is its own
/// ancestor, as `merge-base --is-ancestor` has it.
pub(crate) fn is_ancestor(
    lib: &Repository,
    ancestor: &Rev,
    descendant: &Rev,
) -> Result<bool, GitError> {
    let (ancestor, descendant) = (commit_of(lib, ancestor)?, commit_of(lib, descendant)?);
    if ancestor == descendant {
        return Ok(true);
    }
    lib.graph_descendant_of(descendant, ancestor)
        .map_err(|e| failed("graph_descendant_of", e))
}

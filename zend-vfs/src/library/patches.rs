//! A patch applied to a tree in this process: libgit2 parses it and applies it
//! to the base commit's tree in memory, and the result is written as a tree.

use git2::{Diff, Oid as LibOid, Repository};

use super::patch_text::quote_spaced_paths;
use super::{failed, oid_of, revision};
use crate::error::GitError;
use crate::types::Oid;
use crate::write::apply::ApplyOutcome;

/// `patch` applied to `base`'s tree: the tree, or why it does not apply. Input
/// holding no patch at all is refused, as `git apply` refuses it.
pub(crate) fn apply(lib: &Repository, base: &Oid, patch: &[u8]) -> Result<ApplyOutcome, GitError> {
    let id = LibOid::from_str(base.as_str()).map_err(|e| failed("from_str", e))?;
    let tree = lib
        .find_object(id, None)
        .and_then(|object| object.peel_to_tree())
        .map_err(|e| revision(e, base.as_str()))?;
    let diff = match Diff::from_buffer(&quote_spaced_paths(patch)) {
        Ok(diff) => diff,
        Err(e) => {
            return Ok(ApplyOutcome::Rejected {
                detail: e.message().to_string(),
            })
        }
    };
    if diff.deltas().len() == 0 {
        return Ok(ApplyOutcome::Rejected {
            detail: "no valid patches in input".to_string(),
        });
    }
    let mut index = match lib.apply_to_tree(&tree, &diff, None) {
        Ok(index) => index,
        // libgit2 says which hunk; the paths the patch touches say where.
        Err(e) => {
            let paths: Vec<String> = diff
                .deltas()
                .filter_map(|d| d.new_file().path().or(d.old_file().path()))
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            return Ok(ApplyOutcome::Rejected {
                detail: format!("{}: {}", paths.join(", "), e.message()),
            });
        }
    };
    let tree = index
        .write_tree_to(lib)
        .map_err(|e| failed("write_tree", e))?;
    Ok(ApplyOutcome::Applied {
        tree: oid_of(tree)?,
    })
}

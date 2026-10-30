//! Reading a commit's files — every blob of its tree, by path — as objects,
//! never through the working tree, which jobs on the same checkout own.

use std::collections::BTreeMap;

use zend_vfs::{GitError, ObjectKind, Oid, Repo, Rev};

/// Every file `commit` holds, `path → bytes`.
pub fn files_at(repo: &Repo, commit: &Oid) -> Result<BTreeMap<String, Vec<u8>>, GitError> {
    let blobs = repo.blobs();
    let (_, tree) =
        blobs
            .commit_of(&Rev::Oid(commit.clone()))?
            .ok_or_else(|| GitError::UnknownRevision {
                rev: commit.to_string(),
            })?;
    let mut out = BTreeMap::new();
    for e in repo.ls_tree_all(&tree)? {
        if e.entry.kind != ObjectKind::Blob {
            continue;
        }
        let bytes = blobs
            .read_blob(&e.entry.oid)?
            .ok_or_else(|| GitError::UnknownRevision {
                rev: e.entry.oid.to_string(),
            })?;
        out.insert(e.entry.path.as_str().to_string(), bytes);
    }
    Ok(out)
}
